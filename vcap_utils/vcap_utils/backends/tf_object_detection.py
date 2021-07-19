from itertools import groupby
from typing import List

import numpy as np
import tensorflow as tf

from .base_tensorflow import BaseTFBackend
from .load_utils import parse_dataset_metadata_bytes, parse_tf_model_bytes
from .predictions import DetectionPrediction


class TFObjectDetector(BaseTFBackend):
    """Loads a TensorFlow model and uses it to run object detection, meaning
    that it takes an image as input and returns bounding boxes with
    classifications as output.
    """

    def __init__(self, model_bytes, metadata_bytes,
                 confidence_thresh=0.05,
                 device: str = None,
                 session_config: tf.compat.v1.ConfigProto = None):
        """
        :param model_bytes: Model file data, likely a loaded *.pb file
        :param metadata_bytes: The dataset metadata file data, likely named
            "dataset_metadata.json"
        :param device: The device to run the model on
        :param confidence_thresh: The required confidence threshold to filter
            predictions by. Must be between 0 and 1. There is often an
            additional level of confidence thresholding in the capsule, but
            this is here to avoid creating an unreasonable amount of objects
            for each prediction.
        :param session_config: Model configuration options
        """
        super().__init__()

        assert 0.0 < confidence_thresh < 1.0, \
            "Confidence_thresh must be a number between 0 and 1."

        self.min_confidence = confidence_thresh
        self.graph, self.session = parse_tf_model_bytes(model_bytes,
                                                        device,
                                                        session_config)
        self.label_map = parse_dataset_metadata_bytes(metadata_bytes)

        # Pull all the necessary inputs from the network
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Pull the necessary attributes that we want to get from the network
        # after running it
        self.boxes_tensor = self.graph.get_tensor_by_name('detection_boxes:0')
        self.scores_tensor = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes_tensor = self.graph.get_tensor_by_name(
            'detection_classes:0')

        # Tensorflow models are known to process the first batch of frames
        # slowly due to some lazy initialization logic. Send a fake image to
        # force the model to be fully loaded.
        fake_img = np.zeros((50, 50, 3), dtype=np.uint8)
        self.batch_predict([fake_img])

    def batch_predict(self, imgs_bgr: List[np.ndarray]) -> List[List[DetectionPrediction]]:
        """Takes a list of numpy BGR images of the format that OpenCV gives, and returns
        predicted labels in the form of a list of DetectionPrediction objects
        from predictions.py

        The code in this function is dedictatd to intelligently sorting
        images by their size, and then batching them. Tensorflow can't batch
        inputs of different sizes, so this is something this function takes
        care of.

        The hardest part is making sure that the outputs are re-ordered back
        to their original input order.

        :imgs_bgr: A list of OpenCV images in BGR format (the default)
        :return: List of list of DetectionPrediction objects
        """
        # TODO: This function desperately needs testing, it's easily broken
        # Preprocess each image
        imgs_rgb = [img[..., ::-1] for img in imgs_bgr]

        # Sort images by their size before batching and running inference
        original_indexes = sorted(range(len(imgs_rgb)),
                                  key=lambda index: imgs_rgb[index].shape)
        sorted_imgs = sorted(imgs_rgb, key=lambda img: img.shape)

        # Run batches of images, grouped by their size, and get the results
        results = []
        for index, group in groupby(sorted_imgs, key=lambda img: img.shape):
            # Run inference on each group of equally sized images
            batch_results = self._process_batch(list(group))
            results += batch_results

        # Reorder the results back to the original order of the images
        reordered_results = sorted(enumerate(results),
                                   key=lambda index_img: original_indexes[
                                       index_img[0]])
        reordered_results = [img for _, img in reordered_results]

        return reordered_results

    def _process_batch(self, imgs_rgb: List[np.ndarray]):
        """This function will ONLY WORK if imgs_bgr contains a list of images
        of the same size (resolution, depth). It returns the results of
        imgs_rgb in their original error."""
        assert all([imgs_rgb[0].shape == img.shape for img in imgs_rgb]), \
            "The resolution/depth of all images in this function must be equal!"

        # Batch predict for the list of images
        feed = {self.image_tensor: imgs_rgb}
        all_boxes, all_scores, all_classes = self.session.run(
            [self.boxes_tensor, self.scores_tensor, self.classes_tensor],
            feed_dict=feed)

        # Convert output to DetectionPrediction format
        all_preds = []
        for img_id in range(len(imgs_rgb)):
            # Format the data to a single long array of boxes, scores, and
            # classes.  Each index corresponds with each other.
            boxes = np.squeeze(all_boxes[img_id])
            scores = np.squeeze(all_scores[img_id])
            classes = np.squeeze(all_classes[img_id])

            labels = self._postprocess_output(imgs_rgb[img_id],
                                              boxes=boxes,
                                              scores=scores,
                                              classes=classes)

            all_preds.append(labels)

        return all_preds

    def _postprocess_output(self, image, boxes, scores, classes):
        # Prepare the prediction output
        h, w, _ = image.shape
        labels = []

        for i, score in enumerate(scores):
            if score < self.min_confidence:
                continue

            # Convert the tensorflow format to 'rect' format of
            # [x1, y1, x2, y2] pixels on frame
            rect = [int(round(boxes[i][1] * w, 0)),
                    int(round(boxes[i][0] * h, 0)),
                    int(round(boxes[i][3] * w, 0)),
                    int(round(boxes[i][2] * h, 0))]

            lbl = DetectionPrediction(
                str(self.label_map[int(classes[i])]),
                classes[i],
                rect,
                scores[i])
            labels.append(lbl)
        return labels
