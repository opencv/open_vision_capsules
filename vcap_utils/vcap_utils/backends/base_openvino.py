import logging
from threading import Event, RLock
from typing import Dict, List, Tuple
from queue import Queue

import numpy as np

from vcap import Resize, BaseBackend, DetectionNode

_SUPPORTED_METRICS = "SUPPORTED_METRICS"
_RANGE_FOR_ASYNC_INFER_REQUESTS = "RANGE_FOR_ASYNC_INFER_REQUESTS"
OV_INPUT_TYPE = Dict[str, np.ndarray]


class BaseOpenVINOBackend(BaseBackend):
    def __init__(self, model_xml: bytes,
                 weights_bin: bytes,
                 device_name: str,
                 ie_core=None):
        """
        :param model_xml: The XML data defining the OpenVINO model architecture
        :param weights_bin: The .bin file data defining the model's weights
        :param ie_core: :
          None (default): The backend will initialize its own IECore to load
          the network with.
          IECore: An initialized openvino.inference_engine.IECore, with any
          settings already applied. This can be used to apply CPU extensions
          or load different plugins to the IECore giving it to the backend.
        """
        # Convert from the vcap device naming format to openvino format
        device_name = "CPU" if device_name[:4] == "CPU:" else device_name

        from openvino.inference_engine import \
            IECore, ExecutableNetwork, IENetwork, StatusCode

        self.ie = ie_core or IECore()

        # If the device is a MULTI device, use the default num_requests
        is_multi_device = device_name.lower().startswith("multi")

        num_requests = 0
        if not is_multi_device:
            # Find the optimal number of InferRequests for this device
            supported_metrics = self.ie.get_metric(
                device_name, _SUPPORTED_METRICS)
            if _RANGE_FOR_ASYNC_INFER_REQUESTS in supported_metrics:
                low, high, _ = self.ie.get_metric(
                    device_name, _RANGE_FOR_ASYNC_INFER_REQUESTS)
                # Cap the num_requests, because setting it too high can crash
                # TODO(Alex): Figure out _why_ hddl crashes when set to 'high'
                num_requests = max(0, min(low * 2, high))

        self.net: IENetwork = self.ie.read_network(
            model=model_xml,
            weights=weights_bin,
            init_from_buffer=True)

        try:
            self.exec_net: ExecutableNetwork = self.ie.load_network(
                network=self.net,
                device_name=device_name,
                num_requests=num_requests)
        except RuntimeError as e:
            if device_name == "CPU":
                # It's unknown why this would happen when loading
                # onto CPU, so we re-raise the error.
                raise

            # This error happens when trying to load onto HDDL or MYRIAD, but
            # somehow the device fails to work. In these cases, we try again
            # but load onto 'CPU', which is a safe choice.
            msg = f"Failed to load {self.__class__} onto device " \
                  f"{device_name}. Error: '{e}'. " \
                  f"Trying again with device_name='CPU'"
            logging.error(msg)
            self.__init__(
                model_xml=model_xml,
                weights_bin=weights_bin,
                device_name="CPU",
                ie_core=ie_core)
            return

        # Pull out a couple useful constants
        self.input_blob_names: List[str] = list(self.net.inputs.keys())
        self.output_blob_names: List[str] = list(self.net.outputs.keys())

        # For running threaded requests to the network
        self._StatusCode = StatusCode
        self._get_free_request_lock = RLock()
        self._request_free_events: List[Event] = [
            Event() for _ in self.exec_net.requests]
        """This list corresponds 1:1 with request ID's from OpenVINO. 
        The Events are used to make sure that the request isn't in the middle
        of running a callback while another thread attempts to start another
        async request. 
        
        The events start as 'set' to indicate they are free to run
        """
        for request_free in self._request_free_events:
            request_free.set()

        # For keeping track of the workload for this backend
        self._total_requests: int = len(self.exec_net.requests)
        self._num_ongoing_requests: int = 0
        self._num_ongoing_requests_lock: RLock = RLock()

    @property
    def workload(self) -> float:
        """Returns the percent saturation of this backend. The backend is
        at max 'efficiency' once the number of ongoing requests is equal to
        or over number of exec_net.requests

        This won't affect much unless a custom DeviceMapper filter is used that
        allows multiple Backends to be loaded, eg, a backend for CPU and a
        backend for HDDL. In those cases, this workload measurement will be used
        heavily to decide which backend is busier.
        """
        return self._num_ongoing_requests / self._total_requests

    def send_to_batch(self, input_data: OV_INPUT_TYPE) -> Queue:
        """Efficiently send the input to be inferenced by the network
        :param input_data: Input to the network
        :returns: A queue that will yield 1 result, the output from the network
        """
        out_queue = Queue(maxsize=1)

        with self._get_free_request_lock:
            # Try to get at least one request
            request_id = self.exec_net.get_idle_request_id()
            if request_id < 0:
                # Since there was no free request, wait for one
                status = self.exec_net.wait(num_requests=1)
                if status != self._StatusCode.OK:
                    raise RuntimeError(
                        f"Wait for idle request failed with code {status}")
                request_id = self.exec_net.get_idle_request_id()
                if request_id < 0:
                    raise RuntimeError(f"Invalid request_id: {request_id}")

            request = self.exec_net.requests[request_id]
            request_free = self._request_free_events[request_id]

            def on_result(*args):
                out_queue.put(request.outputs)
                request_free.set()
                with self._num_ongoing_requests_lock:
                    self._num_ongoing_requests -= 1

            # Make sure that the callback for this request is finished, by
            # calling request_free.wait().
            request_free.wait()
            request_free.clear()
            request.set_completion_callback(on_result)

            with self._num_ongoing_requests_lock:
                self._num_ongoing_requests += 1
            request.async_infer(input_data)
        return out_queue

    def prepare_inputs(self, frame: np.ndarray, frame_input_name: str = None) \
            -> Tuple[OV_INPUT_TYPE, Resize]:
        """A helper method to create an OpenVINO input like {input_name: array}

        This method takes a frame, resizes it to fit the network inputs, then
        returns two things: The input, and the Resize information. The
        Resize information contains all of the operations that were done on
        the frame, allowing users to then map the detections from a resized
        frame to the coordinate space of the original frame.

        :param frame: The image. BGR ordered.
        :param frame_input_name: Set this value to force a certain node to be
            used as the frame input. Useful if you still want to use the
            default implementation from a subclass with network with multiple
            inputs
        :returns: ({input_name: resized_frame}, Resize)
        """

        if not frame_input_name and len(self.net.inputs) > 1:
            raise ValueError("More than one input was expected for model, but "
                             "default prepare_inputs implementation was used.")

        input_blob_name = frame_input_name or self.input_blob_names[0]
        input_blob = self.net.inputs[input_blob_name]

        _, _, h, w = input_blob.shape
        resize = Resize(frame).resize(w, h, Resize.ResizeType.EXACT)

        # Change data layout from HWC to CHW
        in_frame = np.transpose(resize.frame.copy(), (2, 0, 1))

        return {input_blob_name: in_frame}, resize

    def parse_detection_results(
            self, results: np.ndarray,
            resize: Resize,
            label_map: Dict[int, str],
            min_confidence: float = 0.0,
            boxes_output_name: str = None) -> List[DetectionNode]:
        """A helper method to take results from a detection-type network.
        :param results: The inference results from the network
        :param resize: A Resize object that was used to resize the image to
        fit into the network originally.
        :param label_map: A dictionary mapping integers to class_names.
        :param min_confidence: Filter out detections that have a confidence
        less than this number.
        :param boxes_output_name: The name of output that carries the bounding
        box information to be parsed. Default=self.output_blob_names[0]
        :returns: A list of DetectionNodes, in this case representing bounding
        boxes.
        """
        output_blob_name = boxes_output_name or self.output_blob_names[0]
        inference_results = results[output_blob_name]

        _, _, h, w = self.net.inputs[self.input_blob_names[0]].shape

        nodes: List[DetectionNode] = []
        for result in inference_results[0][0]:
            # If the first index == 0, that's the end of real predictions
            # The network always outputs an array of length 200 even if it does
            # not have that many predictions
            if result[0] != 0:
                break

            confidence = float(result[2])
            if confidence <= min_confidence:
                continue

            x_min, y_min, x_max, y_max = result[3:7]
            # x and y in res are in terms of percent of image width/height
            x_min, x_max = x_min * w, x_max * w
            y_min, y_max = y_min * h, y_max * h
            coords = [[x_min, y_min], [x_max, y_min],
                      [x_max, y_max], [x_min, y_max]]

            class_id = round(result[1])
            res = DetectionNode(
                name=label_map[class_id],
                coords=coords,
                extra_data={"detection_confidence": confidence})
            nodes.append(res)

        # Convert the coordinate space of the detections from the
        # resized frame to the
        resize.scale_and_offset_detection_nodes(nodes)
        return nodes

    def close(self):
        # Since there's no way to tell OpenVINO to close sockets to HDDL
        # (or other plugins), dereferencing everything is the safest way
        # to go. Without this, OpenVINO seems to crash the HDDL daemon.
        del self.ie
        del self.net
        del self.exec_net
        self.ie = None
        self.net = None
        self.exec_net = None
