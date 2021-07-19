import cv2
import numpy as np
import tensorflow as tf

from .base_tensorflow import BaseTFBackend
from .predictions import DepthPrediction
from .load_utils import parse_tf_model_bytes


class DepthPredictor(BaseTFBackend):
    """
    Loads a model and uses it to run depth prediction, meaning that it takes
    an image as input and returns a matrix the size of the input image where
    the value of each 'pixel' corresponds to the pixel's distance from the
    camera in meters.

    Does not support batch prediction TODO: Is this true?
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

        # Pull all the necessary inputs from the network
        self.input_image = self.graph.get_tensor_by_name("input:0")

        # Pull the necessary attributes that we want to get from the network
        # after running it
        self.segmented_tensor = self.graph.get_tensor_by_name(
            "output_prediction:0")

    def batch_predict(self, img_bgr) -> DepthPrediction:
        """
        Takes numpy BGR images of the format that OpenCV gives, and returns
        predicted labels in the form of a list of DetectionPrediction
        objects from predictions.py

        :param img_bgr: A numpy BGR image from OpenCV
        :return: List of list of DetectionPrediction objects
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        feed = {self.input_image: img_rgb}
        segmented_image = self.session.run(self.segmented_tensor,
                                           feed_dict=feed)

        out = np.squeeze(segmented_image)

        return DepthPrediction(out)
