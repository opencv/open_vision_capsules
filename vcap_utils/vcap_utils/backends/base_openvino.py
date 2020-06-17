import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque
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
        # Convert from the vcap device naming format to openvino format
        device_name = "CPU" if device_name[:4] == "CPU:" else device_name

        from openvino.inference_engine import \
            IECore, ExecutableNetwork, IENetwork, StatusCode

        self.ie = IECore()

        if cpu_extensions is None:
            extension_path = os.environ.get("OPENVINO_EXTENSION_PATH", None)

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

        # Find the optimal number of InferRequests for this device
        supported_metrics = self.ie.get_metric(
            device_name, _SUPPORTED_METRICS)
        if _RANGE_FOR_ASYNC_INFER_REQUESTS in supported_metrics:
            _, n_requests, _ = self.ie.get_metric(
                device_name, _RANGE_FOR_ASYNC_INFER_REQUESTS)
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
        """A queue containing tuples of (frame_id, input) for inference"""
        requests_in_progress: Dict[int, int] = {}
        """A dictionary of {request_id: frame_id} for ongoing requests"""
        unsent_results: Dict[int: Dict] = {}
        """A dictionary of {frame_id: output}, the results not yet yielded"""
        next_frame_id: int = 0
        """The next frame_id we are awaiting results to send. This guarantees
        that results are sent in the same order as the inputs."""

        requests = list(enumerate(self.exec_net.requests))

        # This loop will end when all inputs have been processed and outputs
        # have been yielded
        while (len(inputs)
               + len(unsent_results)
               + len(requests_in_progress)):

            # Block until at least one request slot is free
            self.exec_net.wait(num_requests=1)

            for rid, request in requests:
                status = request.wait(0)
                if status == self.InferRequestStatusCode.INFER_NOT_STARTED:
                    # Put another request in the queue if there are frames
                    # not yet sent for processing.
                    if len(inputs):
                        frame_id, input_dict = inputs.popleft()
                        request.async_infer(input_dict)
                        requests_in_progress[rid] = frame_id

                if status != self.InferRequestStatusCode.OK:
                    # This InferRequest is currently working on a job.
                    continue

                # This request just finished an inference.
                if rid in requests_in_progress:
                    frame_id = requests_in_progress.pop(rid)
                    unsent_results[frame_id] = request.outputs

                # Put another request in the queue, if there are frames
                if len(inputs):
                    frame_id, input_dict = inputs.popleft()
                    request.async_infer(input_dict)
                    requests_in_progress[rid] = frame_id

            if next_frame_id in unsent_results:
                yield unsent_results.pop(next_frame_id)
                next_frame_id += 1

    def close(self):
        """Does nothing"""
        pass
