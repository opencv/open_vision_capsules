import logging
from threading import RLock, Condition
from typing import Dict, List, Tuple
from concurrent.futures import Future

import numpy as np
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

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
          IECore: An initialized openvino.runtime.Core, with any
          settings already applied. This can be used to apply CPU extensions
          or load different plugins to the IECore giving it to the backend.
        """
        # Convert from the vcap device naming format to openvino format
        device_name = "CPU" if device_name[:4] == "CPU:" else device_name

        from openvino.runtime import Core, AsyncInferQueue

        self.ie = ie_core or Core()

        self.model = self.ie.read_model(
            model=model_xml,
            weights=weights_bin)

        try:
            self.compiled_model = self.ie.compile_model(
                model=self.model,
                device_name=device_name)
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
        self.input_blob_names: List[str] = [node.get_any_name() for node in self.compiled_model.inputs]
        self.output_blob_names: List[str] = [node.get_any_name() for node in self.compiled_model.outputs]

        # For running threaded requests to the network
        try:
            self._total_requests = self.compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        except RuntimeError:
            self._total_requests = 1  # Default value if the property is not supported

        self.infer_queue = AsyncInferQueue(self.compiled_model, self._total_requests)
        self.infer_queue.set_callback(self.on_result)

        self._get_free_request_lock = RLock()

        # For keeping track of the workload for this backend
        self._num_ongoing_requests: int = 0
        self._cond: Condition = Condition()

    @property
    def workload(self) -> float:
        """Returns the percent saturation of this backend. The backend is
        at max 'efficiency' once the number of ongoing requests is equal to
        or over number of _total_requests

        This won't affect much unless a custom DeviceMapper filter is used that
        allows multiple Backends to be loaded, eg, a backend for CPU and a
        backend for HDDL. In those cases, this workload measurement will be
        used heavily to decide which backend is busier.
        """
        return self._num_ongoing_requests / self._total_requests

    def on_result(self, request, userdata):
        future = userdata
        try:
            future.set_result(request.results)
        except Exception as e:
            future.set_exception(e)
        finally:
            with self._cond:
                self._num_ongoing_requests -= 1
                self._cond.notify()

    def send_to_batch(self, input_data: OV_INPUT_TYPE) -> Future:
        """Efficiently send the input to be inferenced by the network
        :param input_data: Input to the network
        :returns: A Future that will be filled with the output from the network
        """
        future = Future()

        with self._get_free_request_lock:

            with self._cond:
                try:
                    self._num_ongoing_requests += 1

                    self.infer_queue.start_async(input_data, userdata = future)
                    if not self._cond.wait(timeout=30):
                        self._num_ongoing_requests -= 1
                        raise TimeoutError("Inference request timed out!")
                except Exception as e:
                    self._num_ongoing_requests -= 1
                    logging.error(f"Exception during inference or wait: {e}")
                    future.set_exception(e)

        return future

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
        if not frame_input_name and len(self.compiled_model.inputs) > 1:
            raise ValueError("More than one input was expected for model, but "
                             "default prepare_inputs implementation was used.")

        input_blob_name = frame_input_name or self.input_blob_names[0]
        input_node = next((node for node in self.compiled_model.inputs if node.get_any_name() == input_blob_name), None)
        if input_node is None:
            raise ValueError(f"Input blob name '{input_blob_name}' not found in model inputs.")

        _, _, h, w = input_node.get_shape()
        resize = Resize(frame).resize(w, h, Resize.ResizeType.EXACT)

        # Change data layout from HWC to CHW
        in_frame = np.transpose(resize.frame.copy(), (2, 0, 1))
        # Add batch dimension
        in_frame = np.expand_dims(in_frame, axis=0)
        # Preprocess the input if needed
        # in_frame = preprocess_input(in_frame)

        return {input_blob_name: in_frame}, resize

    def parse_detection_results(
            self, results: np.ndarray,
            resize: Resize,
            label_map: Dict[int, str],
            min_confidence: float = 0.0,
            boxes_output_name: str = None,
            frame_input_name: str = None) -> List[DetectionNode]:
        """A helper method to take results from a detection-type network.
        :param results: The inference results from the network
        :param resize: A Resize object that was used to resize the image to
        fit into the network originally.
        :param label_map: A dictionary mapping integers to class_names.
        :param min_confidence: Filter out detections that have a confidence
        less than this number.
        :param boxes_output_name: The name of output that carries the bounding
        box information to be parsed. Default=self.output_blob_names[0]
        :param frame_input_name: The name of the input that took the frame in.

        :returns: A list of DetectionNodes, in this case representing bounding
        boxes.
        """
        try:
            if boxes_output_name and boxes_output_name in results:
                output_blob_name = boxes_output_name
            else:
                output_blob_name = next(iter(results.keys()))

            inference_results = results[output_blob_name]

            input_name = frame_input_name or self.input_blob_names[0]
            input_node = next(node for node in self.compiled_model.inputs if node.get_any_name() == input_name)
            _, _, h, w = input_node.shape

            nodes: List[DetectionNode] = []
        except Exception as e:
            logging.error(f"parse_detection_results: Exception {e}")
            raise

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
                extra_data={"confidence": confidence})
            nodes.append(res)

        # Convert the coordinate space of the detections from the
        # resized frame to the
        resize.scale_and_offset_detection_nodes(nodes)
        return nodes

    def close(self):
        # Since there's no way to tell OpenVINO to close sockets to HDDL
        # (or other plugins), dereferencing everything is the safest way
        # to go. Without this, OpenVINO seems to crash the HDDL daemon.
        del self.infer_queue
        del self.ie
        del self.model
        del self.compiled_model
        self.infer_queue = None
        self.ie = None
        self.model = None
        self.compiled_model = None
 
    def __del__(self):
        if hasattr(self, 'infer_queue') and self.infer_queue is not None:
            self.close()

