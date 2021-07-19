import cv2
import numpy as np
import tensorflow as tf

from .load_utils import parse_tf_model_bytes
from .predictions import DensityPrediction
from .base_tensorflow import BaseTFBackend


class CrowdDensityCounter(BaseTFBackend):
    """Runs object detection using a given model, meaning that it takes an
    image as input and returns bounding boxes with classifications as output.
    """

    def __init__(self, model_bytes,
                 device: str=None,
                 session_config: tf.compat.v1.ConfigProto=None):
        """
        :param model_bytes: Model file data, likely a loaded *.pb file
        :param device: The device to run the model on
        :param session_config: Model configuration options
        """
        super().__init__()
        self.graph, self.session = parse_tf_model_bytes(model_bytes,
                                                        device,
                                                        session_config)

        self.input = self.graph.get_tensor_by_name("input:0")
        self.output = self.graph.get_tensor_by_name("Threshold_32/Relu:0")

    def batch_predict(self, img_bgr) -> DensityPrediction:
        """Takes a numpy BGR image of the format that OpenCV gives, and returns
        predicted labels in the form of a list of Detection objects
        from predictions.py

        :img_bgr: An OpenCV image in BGR format (the default)
        :return: List of labels
        """
        preprocessed = preprocess(img_bgr)

        # Run the network
        feed = {self.input: preprocessed}
        density_map = self.session.run([self.output], feed)
        density_map = np.squeeze(density_map)

        return DensityPrediction(density_map)


def preprocess(img_bgr):
    # Convert the image to grayscale if it wasn't already
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32, copy=False)
    elif len(img_bgr.shape) == 2:
        gray = img_bgr
    else:
        raise RuntimeError(f"Unexpected image shape: {img_bgr.shape}")

    h = gray.shape[0]
    w = gray.shape[1]
    h_1 = int((h / 4) * 4)
    w_1 = int((w / 4) * 4)
    small_gray = cv2.resize(gray, (w_1, h_1))
    reshaped_small_gray_img = small_gray.reshape(
        (1, 1, small_gray.shape[0], small_gray.shape[1]))
    return reshaped_small_gray_img