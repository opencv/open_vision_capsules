from typing import Callable, List, Union
import random

import cv2
import numpy as np

from vcap.caching import cache


class DetectionPrediction:
    def __init__(self, class_name: str,
                 class_id: int,
                 rect: List[Union[int, float]],
                 confidence: float):
        """An object detection label. Basically a bounding box classifying
        some portion of an image.

        :param class_name: The class name that was predicted
                           (Price_Tag, Shelf_Empty, Person)
        :param class_id: The class number used by the network
        :param rect: [xmin, ymin, xmax, ymax] format
        :param confidence: The confidence in percent from 0-1, representing
            how confident the network is with the prediction
        """
        self._color = None

        # The full rect, (x1, y1, x2, y2) format
        self.rect = tuple(rect)

        self.name = str(class_name)
        self.confidence = float(confidence)
        self.class_num = int(class_id)

        assert 0. <= self.confidence <= 1, "Confidence must be between 0 and 1!"

    def __str__(self):
        return (str(self.name) + " " +
                str(self.rect) + " " +
                str(round(self.confidence, 2)))

    def __repr__(self):
        return self.__str__()

    @property
    def p1(self):
        """ Top Left of the rect"""
        return tuple(self.rect[:2])

    @property
    def p2(self):
        """ Top Right of the rect"""
        return tuple(self.rect[2:])

    @property
    def color(self):
        if self._color is None:
            # Generate a color based on the class name
            rand_seed = random.Random(self.name)
            self._color = tuple([rand_seed.randint(0, 255) for i in range(3)])

        return self._color


class ClassificationPrediction:
    def __init__(self,
                 class_scores: np.ndarray,  # floats
                 class_names: List[str]):
        """A classification of an entire image.

        :param class_scores: A list of scores for each class
        :param class_names: A list of class names, same length as class scores.
        Each index corresponds to an index in class_scores
        """
        self.class_scores = class_scores
        self.class_names = class_names

    def __iter__(self):
        """Iterate over class_name and class_score"""
        for name, score in zip(self.class_names, self.class_scores):
            # Ensure that no numpy floats get returned (causing bad errors)
            yield str(name), float(score)

    def __str__(self):
        return f"{self.name} {self.confidence:.2f}"

    def __repr__(self):
        return self.__str__()

    @property
    @cache
    def name(self) -> str:
        """The class name that was predicted with the highest confidence"""
        return self.class_names[self.class_num]

    @property
    @cache
    def class_num(self) -> int:
        """The class number used by the network for the class with the highest
        confidence"""
        return int(np.argmax(self.class_scores))

    @property
    @cache
    def confidence(self) -> float:
        """The confidence in percent from 0-1, representing how confident the
        network is with its highest confidence prediction"""
        return float(self.class_scores[self.class_num])


class EncodingPrediction:
    def __init__(self, encoding_vector: np.ndarray,
                 distance_func: Callable):
        """

        :param encoding_vector: The encoding prediction
        :param distance_func: A function with args (encoding, other_encodings)
        returns the distance from one encoding to all the other_encodings in
        """
        self.vector = encoding_vector
        self.distance = distance_func


class DensityPrediction:
    def __init__(self, density_map):
        self.map = density_map
        self.count = np.sum(density_map)

    def __str__(self):
        return str(self.count)

    def __repr__(self):
        return self.__str__()

    def resized_map(self, new_size):
        """
        Returns a resized density map, where the sum of each value is still the
        same count
        :param new_size: (new width, new height)"""

        new_map = cv2.resize(self.map.copy(), new_size)
        cur_count = np.sum(new_map)

        # Avoid dividing by zero
        if cur_count == 0:
            return new_map

        scale = self.count / cur_count
        new_map *= scale
        return new_map


class SegmentationPrediction:
    def __init__(self, segmentation, label_map):
        """
        :param segmentation: 2d segmented image where the value is the label num
        :param label_map: Formatted as {"1" : {"label" : "chair", "color" : [12, 53, 100]}}
        """
        self.segmentation = segmentation
        self.label_map = label_map

    def colored(self):
        """Convert segmented image to RGB  image using label_map"""

        color_img = np.zeros(
            (self.segmentation.shape[0], self.segmentation.shape[1], 3),
            dtype=np.uint8)

        # Replace segment values with color pixels using label_map values
        for label_num, label in self.label_map.items():
            color_img[self.segmentation == int(label_num)] = np.array(
                label["color"], dtype=np.uint8)

        return color_img


class DepthPrediction:
    def __init__(self, depth_prediction):
        """
        :param depth_prediction: 2d image where the value is the depth
        """
        self.depth_prediction = depth_prediction

    def normalized(self):
        """Convert segmented image to RGB  image using label_map"""

        # Scale results to max at 255 for image display
        max_distance = np.max(self.depth_prediction)
        pred = 255 * self.depth_prediction // max_distance

        # Convert results to uint8
        pred = pred.astype(np.uint8, copy=True)

        # Do fancy coloring
        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

        return pred
