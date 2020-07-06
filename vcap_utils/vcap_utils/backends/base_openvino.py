from threading import Event
from typing import Dict, List, Tuple, Deque
from collections import deque

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
        super().__init__()
        # Convert from the vcap device naming format to openvino format
        device_name = "CPU" if device_name[:4] == "CPU:" else device_name

        from openvino.inference_engine import \
            IECore, ExecutableNetwork, IENetwork, StatusCode

        self.ie = ie_core or IECore()

        # Find the optimal number of InferRequests for this device
        supported_metrics = self.ie.get_metric(
            device_name, _SUPPORTED_METRICS)
        if _RANGE_FOR_ASYNC_INFER_REQUESTS in supported_metrics:
            low, high, _ = self.ie.get_metric(
                device_name, _RANGE_FOR_ASYNC_INFER_REQUESTS)
            # Cap the n_requests, because sometimes high_n crashes the system
            # TODO(Alex): Figure out _why_ hddl crashes when set to 'high'
            n_requests = max(0, min(low * 2, high))
        else:
            # Use the devices default
            n_requests = 0

        self.net: IENetwork = self.ie.read_network(
            model=model_xml,
            weights=weights_bin,
            init_from_buffer=True)

        self.exec_net: ExecutableNetwork = self.ie.load_network(
            network=self.net,
            device_name=device_name,
            num_requests=n_requests)

        # Pull out a couple useful constants
        self.InferRequestStatusCode = StatusCode
        self.input_blob_names: List[str] = list(self.net.inputs.keys())
        self.output_blob_names: List[str] = list(self.net.outputs.keys())

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
            min_confidence: float = 0.0) -> List[DetectionNode]:
        """A helper method to take results from a detection-type network.
        :param results: The inference results from the network
        :param resize: A Resize object that was used to resize the image to
        fit into the network originally.
        :param label_map: A dictionary mapping integers to class_names.
        :param min_confidence: Filter out detections that have a confidence
        less than this number.
        :returns: A list of DetectionNodes, in this case representing bounding
        boxes.
        """
        output_blob_name = self.output_blob_names[0]
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

    def batch_predict(self, inputs: List[OV_INPUT_TYPE]) \
            -> List[object]:
        """Use the network for inference.

        This function will receive a list of inputs and process them as
        efficiently as possible, optimizing for throughput.
        :param inputs: A list of openvino style inputs {input_name: ndarray}
        :returns: A generator of the networks outputs, yielding them in the
        same order as the inputs
        """

        inputs: Deque[Tuple[int, OV_INPUT_TYPE]] = deque(enumerate(inputs))
        """A deque containing tuples of (frame_id, input) for inference"""
        requests_in_progress: Dict['InferRequest', int] = {}  # noqa: F821
        """A dictionary of {InferRequest: frame_id} for ongoing requests"""
        unsent_results: Dict[int: Dict] = {}
        """A dictionary of {frame_id: output}, the results not yet yielded"""
        result_ready: Event = Event()
        """Triggered when any InferRequest finishes processing"""
        next_frame_id: int = 0
        """The next frame_id we are awaiting results to send. This guarantees
        that results are sent in the same order as the inputs."""

        def on_result(request):
            frame_id = requests_in_progress.pop(request)
            unsent_results[frame_id] = request.outputs
            result_ready.set()

        requests = list(self.exec_net.requests)

        # This loop will end when all inputs have been processed and outputs
        # have been yielded
        while inputs or unsent_results or requests_in_progress:
            if len(requests_in_progress):
                # Block until at least one result is ready
                result_ready.wait()
                result_ready.clear()
            while next_frame_id in unsent_results:
                yield unsent_results.pop(next_frame_id)
                next_frame_id += 1

            for request in requests:
                if not len(inputs):
                    break

                # For debugging, verify the request is ready to be used
                request.wait()
                status = request.wait(0)
                assert (status == self.InferRequestStatusCode.INFER_NOT_STARTED
                        or status == self.InferRequestStatusCode.OK)

                # Put another request in the queue, if there are frames
                frame_id, input_dict = inputs.popleft()
                request.async_infer(input_dict)
                requests_in_progress[request] = frame_id
                request.set_completion_callback(
                    lambda *args, request=request: on_result(request))

    def close(self):
        super().close()
        # Since there's no way to tell OpenVINO to close sockets to HDDL
        # (or other plugins), dereferencing everything is the safest way
        # to go. Without this, OpenVINO seems to crash the HDDL daemon.
        self.ie = None
        self.net = None
        self.exec_net = None
