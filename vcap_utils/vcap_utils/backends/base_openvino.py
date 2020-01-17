import abc
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple

import numpy as np

from vcap.modifiers import Resize
from vcap.backend import BaseBackend


class BaseOpenVINOBackend(BaseBackend):
    def __init__(self, model_xml: os.PathLike, weights_bin: os.PathLike,
                 device_name: str = "CPU",
                 cpu_extensions: Optional[List[str]] = None):
        """
        cpu_extension can be one of two options:
          None (default): Load extensions from the path specified by the
            OPENVINO_EXTENSION_PATH environment variable (should be set
            already to the default path for the extensions)
          List[str]: A list of paths to the .so file to load
          An empty list can be passed to bypass the loading of _any_
            extensions
        """
        super().__init__()
        from openvino.inference_engine import IENetwork, IECore

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
        for cpu_extension in cpu_extensions:
            self.ie.add_extension(cpu_extension, device_name)

        self.net = IENetwork(model=str(model_xml), weights=str(weights_bin))

        # (Unused for now)
        batching_enabled = False
        config = {'DYN_BATCH_ENABLED': 'YES'} if batching_enabled else {}

        self.exec_net = self.ie.load_network(network=self.net,
                                             device_name=device_name,
                                             config=config)

        self.output_blob_names: List[str] = list(self.net.outputs.keys())

    @classmethod
    def from_bytes(cls, model_bytes: bytes, weights_bytes: bytes,
                   *args, **kwargs):
        with NamedTemporaryFile() as model_fi, \
                NamedTemporaryFile() as weights_fi:
            model_fi.write(model_bytes)
            weights_fi.write(weights_bytes)
            return cls(model_fi.name, weights_fi.name, *args, **kwargs)

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

    def batch_predict(self, imgs_bgr: List[np.ndarray]) -> List[object]:
        """Use the network for inference. Main entry point for the capsule."""

        for frame in imgs_bgr:
            feed_dict, resize = self.prepare_inputs(frame)
            inference_result = self.exec_net.infer(feed_dict)
            results = self.parse_results(inference_result, resize)

            yield results

    def close(self):
        """Does nothing"""
        pass
