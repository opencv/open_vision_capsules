import abc
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from vcap import Resize, BaseBackend, DetectionNode


class BaseOpenVINOBackend(BaseBackend):
    def __init__(self, model_xml: bytes,
                 weights_bin: bytes,
                 device_name: str,
                 cpu_extensions: Optional[List[str]] = None):
        """
        :param model_xml: The XML data defining the OpenVINO model architecture
        :param weights_bin: The .bin file data defining the model's weights
        :param cpu_extensions:
          None (default): Load extensions from the path specified by the
            OPENVINO_EXTENSION_PATH environment variable (should be set
            already to the default path for the extensions)
          List[str]: A list of paths to the .so file to load
          An empty list can be passed to bypass the loading of _any_
            extensions
        """
        super().__init__()
        # Convert from the vcap device naming format openvino format
        device_name = "CPU" if device_name[:4] == "CPU:" else device_name

        from openvino.inference_engine import IECore, StatusCode, WaitMode

        self.ie = IECore()
        if cpu_extensions is None:

            extension_path = os.environ.get("OPENVINO_EXTENSION_PATH",
                                            None)

            if extension_path is not None:
                # TODO: Make this compatible with other OSs (i.e. Windows
                #  will use .dlls)
                cpu_extensions = map(str, Path(extension_path).glob("*.so"))
            else:
                logging.warning("Default OpenVINO extensions were "
                                "requested, but OPENVINO_EXTENSION_PATH "
                                "is not set. No extensions will be "
                                "loaded.")
                cpu_extensions = []
        for cpu_extension in cpu_extensions:
            self.ie.add_extension(cpu_extension, device_name)

        self.net = self.ie.read_network(
            model=model_xml, weights=weights_bin, init_from_buffer=True)

        # device_name = "MYRIAD.3.1-ma2480"
        print("Loaded onto device", device_name)
        # TODO: test num_requests=0, as benchmark.py does
        self.exec_net = self.ie.load_network(
            network=self.net,
            device_name=device_name,
            num_requests=40,
            config={"EXCLUSIVE_ASYNC_REQUESTS": "YES"}
        )
        self.oven.MAX_BATCH_SIZE = 40
        print("nrequests", len(self.exec_net.requests))
        self.StatusCode = StatusCode

        self.output_blob_names: List[str] = list(self.net.outputs.keys())

    @classmethod
    def from_bytes(cls, model_bytes: bytes, weights_bytes: bytes,
                   *args, **kwargs):
        """
        .. deprecated:: 0.1.4
           Use the BaseOpenVINOBackend constructor
        """
        return cls(model_xml=model_bytes, weights_bin=weights_bytes,
                   *args, **kwargs)

    @abc.abstractmethod
    def parse_results(self, results: np.ndarray, resize: Resize) -> object:
        """Handle the results from the network and return them in an
        intelligible manner

        :param results: Result dict returned directly from the network
        :param resize: Resize object used in the prepare_inputs method
        :rtype: object: The parsed results
        """
        ...

    def prepare_inputs(self, frame: np.ndarray, frame_input_name: str = None) \
            -> Tuple[Dict[str, np.ndarray], Resize]:
        """Override me if you want to do something else

        :param frame:
        :param frame_input_name: Set this value to force a certain node to be
            used as the frame input. Useful if you still want to use the
            default implementation from a subclass with network with multiple
            inputs
        """

        if not frame_input_name and len(self.net.inputs) > 1:
            raise ValueError("More than one input was expected for model, but "
                             "default prepare_inputs implementation was used.")

        input_blob_name = frame_input_name or list(self.net.inputs.keys())[0]
        input_blob = self.net.inputs[input_blob_name]

        _, _, h, w = input_blob.shape
        resize = Resize(frame).resize(w, h, Resize.ResizeType.EXACT)

        # Change data layout from HWC to CHW
        in_frame = np.transpose(resize.frame.copy(), (2, 0, 1))

        return {input_blob_name: in_frame}, resize

    def batch_predict(self, input_dicts: List[Dict[str, np.ndarray]]) \
            -> List[object]:
        """Use the network for inference. Main entry point for the capsule."""
        inputs_queue: Deque[Dict] = deque(enumerate(input_dicts))
        # free_request_queue = deque(self.exec_net.requests)
        requests_in_progress: Dict[int, int] = {}
        """A dictionary of {request_id: frame_id}"""
        unsent_results = {}
        """{frame_id: results}"""
        next_frame_id = 0
        """The next frame ID we are awaiting results to send"""

        requests = list(enumerate(self.exec_net.requests))
        print("Batch", len(input_dicts))
        while (len(inputs_queue)
               + len(unsent_results)
               + len(requests_in_progress)):
            self.exec_net.wait(num_requests=1)
            for rid, request in requests:
                status = request.wait(0)
                if status == self.InferRequestStatusCode.INFER_NOT_STARTED:
                    # Put another request in the queue, if there are frames
                    if len(inputs_queue):
                        frame_id, input_dict = inputs_queue.popleft()
                        request.async_infer(input_dict)
                        requests_in_progress[rid] = frame_id

                if status != self.InferRequestStatusCode.OK:
                    continue

                # If this just finished an inference
                if rid in requests_in_progress:
                    frame_id = requests_in_progress.pop(rid)
                    unsent_results[frame_id] = request.outputs

                # Put another request in the queue, if there are frames
                if len(inputs_queue):
                    frame_id, input_dict = inputs_queue.popleft()
                    request.async_infer(input_dict)
                    requests_in_progress[rid] = frame_id

            if next_frame_id in unsent_results:
                yield unsent_results.pop(next_frame_id)
                next_frame_id += 1

    def batch_predict_method_1(self, input_dicts: List[Dict[str, np.ndarray]]) \
            -> List[object]:
        """Use the network for inference. Main entry point for the capsule."""
        input_generator = (f for f in input_dicts)
        print("Batch", len(input_dicts))
        requests = self.exec_net.requests
        while True:
            for request in requests:
                try:
                    input_dict = next(input_generator)
                except StopIteration:
                    if len(requests) == 0:
                        # No more frames to process
                        return
                    # Finish processing the queued request
                    break
                request.async_infer(input_dict)

            for request, resize in requests:
                request.wait()
                inference_result = request.outputs
                yield inference_result

    def parse_detection_results(self, results: np.ndarray,
                                resize: Resize) -> List[DetectionNode]:
        output_blob_name = self.output_blob_names[0]
        inference_results = results[output_blob_name]

        input_blob_name = list(self.net.inputs.keys())[0]
        _, _, h, w = self.net.inputs[input_blob_name].shape

        nodes: List[DetectionNode] = []
        for result in inference_results[0][0]:
            # If the first index == 0, that's the end of real predictions
            # The network always outputs an array of length 200 even if it does
            # not have that many predictions
            if result[0] != 0:
                break

            class_id = round(result[1])

            class_name = self.label_map[class_id]

            x_min, y_min, x_max, y_max = result[3:7]
            # x and y in res are in terms of percent of image width/height
            x_min, x_max = x_min * w, x_max * w
            y_min, y_max = y_min * h, y_max * h
            coords = [[x_min, y_min], [x_max, y_min],
                      [x_max, y_max], [x_min, y_max]]

            confidence = float(result[2])

            res = DetectionNode(name=class_name,
                                coords=coords,
                                extra_data={"confidence": confidence})
            nodes.append(res)

        resize.scale_and_offset_detection_nodes(nodes)
        return nodes

    def close(self):
        """Does nothing"""
        self.exec_net.
