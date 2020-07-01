from typing import List, Dict, Optional, Union
from uuid import UUID

import numpy as np

from .caching import cache


class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def size(self):
        """
        :return: The width and height of the bounding box
        """
        return self.x2 - self.x1, self.y2 - self.y1

    @property
    def rect(self):
        """Get the bounding box in 'rect' format, as (x1, y1, x2, y2)"""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self):
        """Return the bounding box in 'xywh' format, as
        (x1, y1, width, height)"""
        return (self.x1, self.y1,
                self.x2 - self.x1,
                self.y2 - self.y1)

    @property
    def center(self):
        return (self.x1 + (self.x2 - self.x1) / 2,
                self.y1 + (self.y2 - self.y1) / 2)

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    def __eq__(self, other):
        return other.__dict__ == self.__dict__


class DetectionNode:
    """Capsules use DetectionNode objects to communicate results to other
    capsules and the application itself. A DetectionNode contains information
    on a detection in the current frame. Capsules that detect objects in a
    frame create new DetectionNodes. Capsules that discover attributes about
    detections add data to existing DetectionNodes.
    """

    def __init__(self, *, name: str,
                 coords: List[List[Union[int, float]]],
                 attributes: Dict[str, str] = None,
                 children: List['DetectionNode'] = None,
                 encoding: Optional[np.ndarray] = None,
                 track_id: Optional[UUID] = None,
                 extra_data: Dict[str, object] = None):
        """
        :param name: The detection class name. This describes what the
            detection is. A detection of a person would have a name="person".
        :param coords: A list of coordinates defining the detection as a
            polygon in-frame. Comes in the format ``[[x,y], [x,y]...]``.
        :param attributes: A key-value store where the key is the type of
            attribute being described and the value is the attribute's value.
            For instance, a capsule that detects gender might add a "gender"
            key to this dict, with a value of either "masculine" or "feminine".
        :param children: Child DetectionNodes that are a "part" of the parent,
            for instance, a head DetectionNode might be a child of a person
            DetectionNode
        :param encoding: An array of float values that represent an encoding of
            the detection. This can be used to recognize specific instances of
            a class. For instance, given a picture of person’s face, the
            encoding of that face and the encodings of future faces can be
            compared to find that person in the future.
        :param track_id: If this object is tracked, this is the unique
             identifier for this detection node that ties it to other detection
             nodes in future and past frames within the same stream.
        :param extra_data: A dict of miscellaneous data. This data is provided
            directly to clients without modification, so it’s a good way to
            pass extra information from a capsule to other applications.
        """
        self.class_name = name
        self.coords = coords
        self.attributes = attributes if attributes else {}
        self.encoding = encoding
        self.track_id = track_id
        self.children = children if children else []
        self.extra_data = extra_data if extra_data else {}

        self._bbox = None

    def scale(self, scale_amount_x: float, scale_amount_y: float):
        """Scales the detection coordinates of the tree by the given scales.

        :param scale_amount_x: The amount to scale x by
        :param scale_amount_y: The amount to scale y by
        """
        for i, c in enumerate(self.coords):
            self.coords[i] = [round(c[0] * scale_amount_x),
                              round(c[1] * scale_amount_y)]
        self._bbox = None

        if self.children is not None:
            for child in self.children:
                child.scale(scale_amount_x, scale_amount_y)

    def offset(self, offset_x: int, offset_y: int):
        """Offsets the detection coordinates of the tree by the given offsets.

        :param offset_x: The amount to offset x by
        :param offset_y: The amount to offset y by
        """
        for i, c in enumerate(self.coords):
            self.coords[i] = [c[0] - offset_x, c[1] - offset_y]
        self._bbox = None

        if self.children is not None:
            for child in self.children:
                child.offset(offset_x, offset_y)

    @property
    @cache
    def all_attributes(self):
        """Return all attributes including child attributes"""

        def get_attrs(child):
            attributes = child.attributes
            for child in child.children:
                attributes.update(get_attrs(child))
            return attributes

        return get_attrs(self)

    def __repr__(self):
        rep = {
            "class_name": self.class_name,
            "track_id": self.track_id,
            "attributes": self.attributes,
            "extra_data": self.extra_data,
            "coords": [(round(x), round(y)) for x, y in self.coords],
            "encoding": "Encoded" if self.encoding is not None else None
        }

        return str(rep)

    @property
    def bbox(self) -> BoundingBox:
        if self._bbox is None:
            self._bbox = self._make_bbox()
        return self._bbox

    def _make_bbox(self):
        """
        :return: Create a fully containing bounding box of the detection
            polygon
        """
        sorted_x = sorted([c[0] for c in self.coords])
        sorted_y = sorted([c[1] for c in self.coords])
        return BoundingBox(sorted_x[0], sorted_y[0],
                           sorted_x[-1], sorted_y[-1])


def rect_to_coords(rect):
    """Converts a rect in the [x1, y1, x2, y2] format to coordinates."""
    return [[rect[0], rect[1]],
            [rect[2], rect[1]],
            [rect[2], rect[3]],
            [rect[0], rect[3]]]
