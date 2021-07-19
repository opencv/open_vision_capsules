from collections import namedtuple
from typing import List

import cv2
import numpy as np
import tensorflow as tf

from .base_encoder import BaseEncoderBackend
from .base_tensorflow import BaseTFBackend
from .load_utils import parse_tf_model_bytes
from .predictions import EncodingPrediction
from vcap_utils.algorithms import euclidian_distance


class OpenFaceEncoder(BaseEncoderBackend, BaseTFBackend):
    """Encode faces (or whatever the model is capable of encoding)
    using this module. It will easily load and run what is necessary.

    Specifically meant to work with models trained from this repository:
    https://github.com/davidsandberg/facenet
    """

    def __init__(self, model_bytes, model_name,
                 device: str = None,
                 session_config: tf.compat.v1.ConfigProto = None):
        """
        :param model_bytes: Model file bytes, a loaded *.pb file
        :param model_name: The name of the model in order to load correct
        input/output tensor node names
        :param device: The device to run the model on
        :param session_config: Model configuration options
        """
        super().__init__()

        assert model_name in model_to_ops_map.keys(), \
            "The model name must be from here: " + str(model_to_ops_map.keys())

        self.config = model_to_ops_map[model_name]
        self._preprocess = self.config.preprocess
        self.graph, self.session = parse_tf_model_bytes(model_bytes,
                                                        device,
                                                        session_config)

        # Pull all the necessary inputs from the network
        self.image_tensor = self.graph.get_tensor_by_name(
            self.config.input_node)
        self.embedding = self.graph.get_tensor_by_name(
            self.config.embedding)
        self.phase_train_placeholder = self.graph.get_tensor_by_name(
            'phase_train:0')

    def batch_predict(self, imgs_bgr) -> List[EncodingPrediction]:
        imgs_rgb = [self._preprocess(img) for img in imgs_bgr]

        # Batch predict for the list of images
        feed = {self.image_tensor: imgs_rgb,
                self.phase_train_placeholder: False}
        embeddings = self.session.run(self.embedding, feed_dict=feed)
        preds = [EncodingPrediction(e, self.config.distance)
                 for e in embeddings]
        return preds

    def distances(self, encoding_to_compare, encodings):
        return self.config.distance(encoding_to_compare, encodings)


def _preprocess_vggface2_center_loss(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, vggface2_center_loss.img_size)

    # Pre-whiten the image
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
    img = np.multiply(np.subtract(img, mean), 1 / std_adj)

    return img


ModelConfig = namedtuple("ModelConfig",
                         ["input_node", "embedding", "img_size",
                          "preprocess", "distance"])

vggface2_center_loss = ModelConfig(input_node="batch_join:0",
                                   embedding="embeddings:0",
                                   img_size=(160, 160),
                                   preprocess=_preprocess_vggface2_center_loss,
                                   distance=euclidian_distance)
model_to_ops_map = {"vggface2_center_loss": vggface2_center_loss}
