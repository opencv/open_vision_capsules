"""Contains utility functions for doing data modification work in a capsule."""
import math
from enum import auto, Enum
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from vcap.detection_node import BoundingBox, DetectionNode


def _clamp(num, min, max):
    if num < min:
        return 0
    if num > max:
        return max - 1
    return num


class Crop:
    """A crop that may be applied to a frame.

    Usage Example:
    >>> det_node = DetectionNode(...)
    >>> frame: np.ndarray
    >>> cropped = Crop.from_detection(det_node).pad_percent(top=10).apply(frame)
    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def pad_percent(self, top=0, bottom=0, left=0, right=0) -> "Crop":
        width = self.x2 - self.x1
        height = self.y2 - self.y1

        self.y1 -= height * (top / 100)
        self.y2 += height * (bottom / 100)
        self.x1 -= width * (left / 100)
        self.x2 += width * (right / 100)

        return self

    def pad_px(self, top=0, bottom=0, left=0, right=0) -> "Crop":
        self.y1 -= top
        self.y2 += bottom
        self.x1 -= left
        self.x2 += right

        return self

    def apply(self, frame: np.ndarray) -> np.ndarray:
        self.y1 = int(round(_clamp(self.y1, 0, frame.shape[0])))
        self.y2 = int(round(_clamp(self.y2, 0, frame.shape[0])))
        self.x1 = int(round(_clamp(self.x1, 0, frame.shape[1])))
        self.x2 = int(round(_clamp(self.x2, 0, frame.shape[1])))
        return frame[self.y1:self.y2, self.x1:self.x2, :]

    @classmethod
    def from_detection(cls, node: DetectionNode) -> "Crop":
        """Uses the given detection area to create a crop."""
        bbox = node.bbox
        return cls(bbox.x1, bbox.y1, bbox.x2, bbox.y2)


class Clamp:
    """Scales down a frame if its size is over a given amount, preserving the
    original aspect ratio. It can then scale up any detections that have been
    run on the smaller frame to fit the original frame size.

    If the frame is under the given amount, nothing is done and the frame is
    kept the same.

    Usage Example:
    >>> clamp = Clamp(frame, 100, 100)
    >>> detection_nodes = some_detection_inference(clamp.apply())
    >>> clamp.scale_detection_nodes(detection_nodes)
    """

    def __init__(self, frame, max_width, max_height):
        """
        :param frame: The frame to scale
        :param max_width: The maximum allowed width of the frame
        :param max_height: The maximum allowed height of the frame
        """
        self.frame = frame
        self.max_width = max_width
        self.max_height = max_height

        height = float(self.frame.shape[0])
        width = float(self.frame.shape[1])

        if height <= self.max_height and width <= self.max_width:
            # Image is already small enough
            self.scale_factor = 1.0
        elif height >= width:
            self.scale_factor = self.max_height / height
        else:
            self.scale_factor = self.max_width / width

    def apply(self) -> np.ndarray:
        """
        :return: A clamped version of the frame
        """
        height = float(self.frame.shape[0])
        width = float(self.frame.shape[1])

        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        return cv2.resize(self.frame, (new_width, new_height))

    def scale_detection_nodes(self, nodes: List[DetectionNode]):
        """Scales up the given DetectionNode objects to fit the original frame
        size.
        """
        node_scale = 1.0 / self.scale_factor
        for node in nodes:
            node.scale(node_scale, node_scale)
        return nodes


class Resize:
    """Resize frames to a specified size

    Allows for maintaining aspect ratio
    Allows for padding
    Allows for cropping

    Usage example:
    >>> # Resize up, then crop right/bottom side to be 480p
    >>> frame: np.ndarray
    >>> frame = (Resize(frame) \
    ...          .resize(1920, 1080, Resize.ResizeType.FIT_BOTH) \
    ...          .crop(640, 480, Resize.CropPadType.RIGHT_BOTTOM) \
    ...          .frame)

    >>> # Resize down, then pad all around to be back to 1080p
    >>> frame: np.ndarray
    >>> frame = (Resize(frame)
    ...          .resize(640, 480, Resize.ResizeType.FIT_BOTH)
    ...          .pad(1920, 1080, Resize.CropPadType.BOTH)
    ...          .frame)
    """

    class ResizeType(Enum):
        WIDTH = auto()
        """Fit width exactly, maintain aspect ratio"""
        HEIGHT = auto()
        """Fit height exactly, maintain aspect ratio"""
        FIT_BOTH = auto()
        """Fit both height and width within bounds, but only one will be exact.
        Will end up smaller than (or same as) requested size.
        Basically the opposite of FIT_ONE in implementation.
        """
        FIT_ONE = auto()
        """Fit one of height or width within size, maintaining aspect ratio.
        Will end up bigger than (or same as) requested size.
        Basically the opposite of FIT_BOTH in implementation.
        Usually followed by a crop.
        """
        EXACT = auto()
        """Resize image exactly, ignoring aspect ratio. crop and pad attributes
        will have no effect"""
        NONE = auto()
        """Don't resize"""

    class CropPadType(Enum):
        RIGHT_BOTTOM = auto()
        """Crop/pad the larger index pixels"""
        LEFT_TOP = auto()
        """Crop/pad the smaller index pixels"""
        ALL = auto()
        """Crop/pad evenly from both sides. Smaller index side will be one
        greater than larger other if odd crop/pad)"""
        CROP_START_POINT = auto()
        """Crop to size with top left of result at top_left. Not available for
        padding"""
        NONE = auto()
        """Don't crop/pad"""

    class _OperationType(Enum):
        SCALE = auto()
        OFFSET = auto()

    def __init__(self, frame: np.ndarray):

        # Used to keep track of operations so that scale_and_offset_detection
        # nodes works right
        self._operations: List[Tuple[Resize._OperationType,
                                     Union[Tuple[int, int],
                                           Tuple[float, float]]]] = []

        self.frame = frame

    def resize(self, resize_width: int, resize_height: int,
               resize_type: ResizeType):

        frame_width = self.frame.shape[1]
        frame_height = self.frame.shape[0]

        new_width = frame_width
        new_height = frame_height

        if resize_type is self.ResizeType.WIDTH:
            # Resize height at same ratio as width to preserve aspect ratio
            new_width = resize_width
            new_height = (resize_width / frame_width) * frame_height

        elif resize_type is self.ResizeType.HEIGHT:
            # Resize width at same ratio as height to preserve aspect ratio
            new_height = resize_height
            new_width = (resize_height / frame_height) * frame_width

        elif resize_type is self.ResizeType.FIT_BOTH:

            if resize_width / resize_height > frame_width / frame_height:
                # Target wider than source. Will be limited by height
                new_height = resize_height
                new_width = (resize_height / frame_height) * frame_width
            else:
                # Target taller than source. Will be limited by width
                new_width = resize_width
                new_height = (resize_width / frame_width) * frame_height

        elif resize_type is self.ResizeType.FIT_ONE:

            if resize_width / resize_height > frame_width / frame_height:
                # Target wider than source. Will be limited by width
                new_width = resize_width
                new_height = (resize_width / frame_width) * frame_height
            else:
                # Target taller than source. Will be limited by height
                new_height = resize_height
                new_width = (resize_height / frame_height) * frame_width

        elif resize_type is self.ResizeType.EXACT:
            new_width = resize_width
            new_height = resize_height

        elif resize_type is self.ResizeType.NONE:
            pass

        else:
            raise ValueError("Invalid resize type")

        new_width = round(new_width)
        new_height = round(new_height)

        # Account for scaling
        scale_width = new_width / frame_width
        scale_height = new_height / frame_height
        self._operations.append(
            (self._OperationType.SCALE, (scale_width, scale_height))
        )

        self.frame = cv2.resize(self.frame, (new_width, new_height))

        return self

    def crop_bbox(self, bbox: BoundingBox):
        return self.crop(
            crop_width=int(bbox.width),
            crop_height=int(bbox.height),
            crop_type=self.CropPadType.CROP_START_POINT,
            top_left=(int(bbox.x1), int(bbox.y1)))

    def crop(self, crop_width: int, crop_height: int, crop_type: CropPadType,
             top_left: Optional[Tuple[int, int]] = None):
        """Crop the frame

        :param crop_width: Resulting width of the frame after cropping
        :param crop_height: Resulting height of the frame after cropping
        :param crop_type: Type of crop to perform
        :param top_left: [Optional] Point in input's coordinate space that will
        serve as the top left point in the output. Only used if
        CropPadType=CROP_START_POINT. Tuple in form (x, y)
        """

        frame_width = self.frame.shape[1]
        frame_height = self.frame.shape[0]

        start_x = 0
        end_x = frame_width
        start_y = 0
        end_y = frame_height

        if crop_type is self.CropPadType.LEFT_TOP:
            start_x = frame_width - crop_width
            start_y = frame_height - crop_height

        elif crop_type is self.CropPadType.RIGHT_BOTTOM:
            end_x = crop_width
            end_y = crop_height

        elif crop_type is self.CropPadType.ALL:
            start_x = (frame_width - crop_width) // 2
            start_y = (frame_height - crop_height) // 2

            end_x = start_x + crop_width
            end_y = start_y + crop_height

        elif crop_type is self.CropPadType.CROP_START_POINT:
            start_x, start_y = top_left
            end_x = start_x + crop_width
            end_y = start_y + crop_height

        elif crop_type is self.CropPadType.NONE:
            pass

        else:
            raise ValueError("Invalid crop type")

        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(frame_width, end_x)  # Not really necessary for splicing
        end_y = min(frame_height, end_y)

        # Account for cropping displacements
        self._operations.append(
            (self._OperationType.OFFSET, (start_x, start_y))
        )

        self.frame = self.frame[start_y:end_y, start_x:end_x]

        return self

    def pad(self, pad_width, pad_height, pad_value,
            pad_type: CropPadType):
        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0

        frame_width = self.frame.shape[1]
        frame_height = self.frame.shape[0]

        if pad_type is self.CropPadType.LEFT_TOP:
            left_pad = pad_width - frame_width
            top_pad = pad_height - frame_height

        elif pad_type is self.CropPadType.RIGHT_BOTTOM:
            right_pad = pad_width - frame_width
            bottom_pad = pad_height - frame_height

        elif pad_type is self.CropPadType.ALL:
            left_pad = (pad_width - frame_width) // 2
            right_pad = (pad_width - frame_width) - left_pad

            top_pad = (pad_height - frame_height) // 2
            bottom_pad = (pad_height - frame_height) - top_pad

        elif pad_type is self.CropPadType.NONE:
            pass

        else:
            raise ValueError("Invalid pad type")

        left_pad = max(0, left_pad)
        right_pad = max(0, right_pad)
        top_pad = max(0, top_pad)
        bottom_pad = max(0, bottom_pad)

        # Account for padding displacements
        self._operations.append(
            (self._OperationType.OFFSET, (-left_pad, -top_pad))
        )

        self.frame = cv2.copyMakeBorder(
            self.frame,
            top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT,
            value=pad_value)

        return self

    def scale_and_offset_detection_nodes(self, nodes: List[DetectionNode]):
        """Scales up the given DetectionNode objects to fit the original frame
        size.
        """
        for node in nodes:
            for operation, (x, y) in self._operations[::-1]:
                if operation is self._OperationType.SCALE:
                    node.scale(1 / x, 1 / y)
                elif operation is self._OperationType.OFFSET:
                    node.offset(-x, -y)
        return nodes


class SizeFilter:
    """Filters detection nodes based on if they're large or small enough.

    Usage example:
    >>> nodes: List[DetectionNode]
    >>> filtered = (SizeFilter(nodes)
    ...             .min_size(10, 10)
    ...             .max_size(1000, 1000)
    ...             .apply())
    """

    def __init__(self, nodes: List[DetectionNode]):
        self.nodes = nodes

        self._min_width = 0
        self._min_height = 0
        self._min_area = 0

        self._max_width = math.inf
        self._max_height = math.inf
        self._max_area = math.inf

    def min_size(self, width, height):
        self._min_width = width
        self._min_height = height
        return self

    def max_size(self, width, height):
        self._max_width = width
        self._max_height = height
        return self

    def min_area(self, area):
        self._min_area = area
        return self

    def max_area(self, area):
        self._max_area = area
        return self

    def apply(self) -> List[DetectionNode]:
        return [node for node in self.nodes if self._within_size(node)]

    def _within_size(self, node: DetectionNode):
        size = node.bbox.size
        return self._min_width <= size[0] <= self._max_width \
               and self._min_height <= size[1] <= self._max_height \
               and self._min_area <= size[0] * size[1] <= self._max_area
