"""This module is made for running image classification inference easily on
various Slim models
"""
import cv2
import numpy as np
import tensorflow as tf
from collections import namedtuple
from typing import List

from . import load_utils
from .base_tensorflow import BaseTFBackend
from .predictions import ClassificationPrediction


class TFImageClassifier(BaseTFBackend):
    def __init__(self, model_bytes, metadata_bytes, model_name,
                 device: str = None,
                 session_config: tf.compat.v1.ConfigProto = None):
        """
        :param model_bytes: Loaded model data, likely from a *.pb file
        :param metadata_bytes: Loaded dataset metadata, likely from a file
            named "dataset_metadata.json"
        :param model_name: Currently supported model names are listed in the
            model_to_ops_map variable.
        :param device: The device to run the model on
        :param session_config: Model configuration options
        """
        super().__init__()
        assert model_name in model_to_ops_map.keys(), \
            "The model name must be from here: " + str(model_to_ops_map.keys())

        self.config = model_to_ops_map[model_name]
        self.graph, self.session = load_utils.parse_tf_model_bytes(
            model_bytes, device, session_config)
        self.label_map = load_utils.parse_dataset_metadata_bytes(
            metadata_bytes)
        self.class_names = tuple(
            class_name for key, class_name in
            sorted(self.label_map.items(), key=lambda i: i[0]))

        # Create the input node to the graph, with preprocessing built-in
        with self.graph.as_default():
            # Create a new input node for images of various sizes
            self.input_node = tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=[None, self.config.img_size, self.config.img_size, 3])

            # Create the preprocessing node
            preprocessed = self.config.preprocess(self.input_node)

            # Connect the new input to the loaded graphs input
            self.output_node, = tf.import_graph_def(
                self.graph.as_graph_def(),
                input_map={self.config.input_node: preprocessed},
                return_elements=[self.config.output_node])

        # Tensorflow models are known to process the first batch of frames
        # slowly due to some lazy initialization logic. Send a fake image to
        # force the model to be fully loaded.
        fake_img = np.zeros((50, 50, 3), dtype=np.uint8)
        self.batch_predict([fake_img])

    def batch_predict(self, imgs_bgr) -> List[ClassificationPrediction]:
        """
        :param imgs_bgr: A list of images [img1, img2] in BGR format
        """

        # Preprocess each image
        imgs_bgr_resized = [_resize_and_pad(img, self.config.img_size)
                            for img in imgs_bgr]

        # Batch predict for the list of images
        feed = {self.input_node: imgs_bgr_resized}
        outs = self.session.run(self.output_node,
                                feed_dict=feed)

        # Create the predictions for return
        predictions = [ClassificationPrediction(
            class_scores=out, class_names=self.class_names)
            for out in outs
        ]

        return predictions


def _preprocess_inception_resnet_v2(img):
    # BGR to RGB conversion
    img = img[..., ::-1]

    # Normalize the image between -1 and 1  900-1000 FPS
    img = (2.0 / 255.0) * img - 1.0
    return img


def _preprocess_resnet_v2_50(img):
    # BGR to RGB conversion
    img = img[..., ::-1]
    return img


def _resize_and_pad(img, desired_size):
    """
    Resize an image to the desired width and height
    :param img:
    :param desired_size:
    :return:
    """
    old_size = img.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size[0] == 0:
        new_size = (new_size[0] + 1, new_size[1])

    if new_size[1] == 0:
        new_size = (new_size[0], new_size[1] + 1)

    # New_size should be in (width, height) format
    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    img = cv2.copyMakeBorder(im, top, bottom, left, right,
                             cv2.BORDER_CONSTANT,
                             value=color)
    return img


ModelConfig = namedtuple("ModelConfig",
                         ["input_node", "output_node", "img_size",
                          "preprocess"])
resnet_v2_50 = \
    ModelConfig(input_node="resnet_v2_50/Pad:0",
                output_node="resnet_v2_50/predictions/Reshape_1:0",
                img_size=230,
                preprocess=_preprocess_resnet_v2_50)
inception_resnet_v2 = \
    ModelConfig(input_node="input:0",
                output_node="InceptionResnetV2/Logits/Predictions:0",
                img_size=299,
                preprocess=_preprocess_inception_resnet_v2)

inception_v4 = \
    ModelConfig(input_node="input:0",
                output_node="InceptionV4/Logits/Predictions:0",
                img_size=299,
                preprocess=_preprocess_inception_resnet_v2)

mobilenet_v1 = \
    ModelConfig(input_node="input:0",
                output_node="MobilenetV1/Predictions/Reshape_1:0",
                img_size=224,
                preprocess=_preprocess_inception_resnet_v2)

model_to_ops_map = {"resnet_v2_50": resnet_v2_50,
                    "inception_resnet_v2": inception_resnet_v2,
                    "mobilenet_v1": mobilenet_v1,
                    "inception_v4": inception_v4}
"""
models_to_ops_map keeps track of information for different model names, so 
that the Brain interface can work without the user needing to remember these
specific node names.
"""
