import numpy as np
import tensorflow as tf

from .base_tensorflow import BaseTFBackend
from .load_utils import parse_dataset_metadata_bytes, parse_tf_model_bytes
from .predictions import SegmentationPrediction


class Segmenter(BaseTFBackend):
    """Loads a model and uses it to run image segmentation, meaning that it
    takes an image as input and returns a matrix the size of the input image
    where the value of each 'pixel' corresponds to an image segment type.

    Does not support batch prediction TODO: Is this true?
    """

    def __init__(self, model_bytes, metadata_bytes,
                 device: str = None,
                 session_config: tf.compat.v1.ConfigProto = None):
        """
        :param model_bytes: Model file data, likely a loaded *.pb file
        :param metadata_bytes: The dataset metadata file data, likely named
            "dataset_metadata.json"
        :param device: The device to run the model on
        :param session_config: Model configuration options
        """
        super().__init__()

        self.graph, self.session = parse_tf_model_bytes(model_bytes,
                                                        device,
                                                        session_config)
        self.label_map = parse_dataset_metadata_bytes(metadata_bytes)

        # Pull all the necessary inputs from the network
        self.input_image = self.graph.get_tensor_by_name("input")

        # Pull the necessary attributes that we want to get from the network
        # after running it
        self.segmented_tensor = self.graph.get_tensor_by_name(
            "output_prediction")

    def batch_predict(self, img_rgb) -> SegmentationPrediction:
        """Takes numpy BGR images of the format that OpenCV gives, and returns
        predicted labels in the form of a list of DetectionPrediction objects
        from predictions.py.

        :imgs: An OpenCV image
        :param bgr: image is bgr format (bgr=True, default) or rgb (bgr=False)
        :return: List of list of DetectionPrediction objects
        """
        # TODO: The story between RGB and BGR here is confusing
        # This method should take a BGR image and convert it
        # Batch predict for the list of images
        feed = {self.input_image: img_rgb}
        segmented_image = self.session.run(self.segmented_tensor,
                                           feed_dict=feed)

        out = np.squeeze(segmented_image)

        return SegmentationPrediction(out, self.label_map)
