from enum import Enum
from typing import List, Dict, Union

from .detection_node import DetectionNode

DETECTION_NODE_TYPE = Union[None, DetectionNode, List[DetectionNode]]


class NodeDescription:
    """Describes required aspects of a DetectionNode. This is used to specify
    what kind of DetectionNodes a capsule takes as input and what kind of
    DetectionNodes the capsule outputs.

    For example, a capsule that encodes people and face detections would use
    NodeDescriptions to define its inputs and outputs like so:

    >>> input_type = NodeDescription(
    ...     detections=["person", "face"])
    >>> output_type = NodeDescription(
    ...     detections=["person", "face"],
    ...     encoded=True)

    A capsule that uses a car's encoding to classify the color of a car would
    look like this.

    >>> input_type = NodeDescription(
    ...     detections=["car"],
    ...     encoded=True)
    >>> output_type = NodeDescription(
    ...     detections=["car"],
    ...     attributes={"color": ["blue", "yellow", "green"]},
    ...     encoded=True)

    A capsule that detects dogs and takes no existing input would look like
    this.

    >>> input_type = NodeDescription(size=NodeDescription.Size.SINGLE)
    >>> output_type = NodeDescription(
    >>>                 size=NodeDescription.Size.ALL,
    >>>                 detections=["dog"])
    """

    class Size(Enum):
        NONE = 1
        """A NodeDescription that does not take in or produce DetectionNodes.
        """
        SINGLE = 2
        """A NodeDescription that takes in or produces a single DetectionNode.
        """
        ALL = 3
        """A NodeDescription that takes in all available DetectionNodes or
        produces any number of DetectionNodes.
        """

    def __init__(self, *,
                 size: Size,
                 detections: List[str] = None,
                 attributes: Dict[str, List[str]] = None,
                 encoded: bool = False,
                 tracked: bool = False,
                 extra_data: List[str] = None):
        """
        :param size: The number of DetectionNodes that this capsule either
            takes as input or provides as output
        :param detections: A list of acceptable detection class names. A node
            that meets this description must have a class name that is present
            in this list
        :param attributes: A dict whose key is the classification type and
            whose value is a list of possible attributes. A node that meets
            this description must have a classification for each classification
            type
        :param encoded: If true, the DetectionNode must be encoded to meet this
            description
        :param tracked: If true, the DetectionNode is being tracked
        :param extra_data: A list of keys in a NodeDescription's extra_data. A
            DetectionNode that meets this description must have extra data for
            each name listed here.
        """
        self.size = size
        self.detections = detections if detections else []
        self.attributes = attributes if attributes else {}
        self.encoded = encoded
        self.tracked = tracked
        self.extra_data = extra_data if extra_data else []

    def difference(self, other: "NodeDescription"):
        """Outputs a NodeDescription that represents the change between this
        NodeDescription and the given one. For instance, if the given
        NodeDescription has been detected for people and this NodeDescription
        has not, the resulting NodeDescription would look like this.

        >>> NodeDescription(size=NodeDescription.Size.ALL,
        >>>                 detections=["person"])

        This can be used to compare a capsule's input and output nodes to see
        what operation the capsule provides.
        """
        # Find the classification difference between this NodeDescription and\
        # the other
        # Found here: https://stackoverflow.com/a/32815681
        new_attributes = {
            k: other.attributes[k]
            for k in set(other.attributes) - set(self.attributes)}

        # Find the difference between this NodeDescription and the other
        new_detections = list(set(other.detections) - set(self.detections))

        # Find the difference between the extra data
        new_extra_data = list(set(other.extra_data) - set(self.extra_data))

        # Set encoded in the diff description to true only if this
        # NodeDescription was not encoded and the other one was
        newly_encoded = not self.encoded and other.encoded

        # Set tracked in diff description to true if this node was not tracked
        # and the other one was
        newly_tracked = not self.tracked and other.tracked

        return NodeDescription(
            size=other.size,
            detections=new_detections,
            attributes=new_attributes,
            encoded=newly_encoded,
            tracked=newly_tracked,
            extra_data=new_extra_data)

    def describes(self, node: DetectionNode):
        """Check if a DetectionNode has the information described in this
        NodeDescription. """

        if len(self.detections) > 0 and node.class_name not in self.detections:
            return False

        if self.encoded and node.encoding is None:
            return False

        if self.tracked and node.track_id is None:
            return False

        if len(self.extra_data):
            all_data_available = all(key in node.extra_data
                                     for key in self.extra_data)
            if not all_data_available:
                return False

        if self.attributes != {}:
            if node.attributes == {}:
                return False

            for category, possible_values in self.attributes.items():
                if category not in node.attributes.keys():
                    return False

                if node.attributes[category] not in possible_values:
                    good_values = self.attributes[category]
                    bad_value = node.attributes[category]
                    raise ValueError(
                        f"Invalid attribute value in DetectionNode! "
                        f"DetectionNode has value '{bad_value}' for attribute "
                        f"category '{category}', but the NodeDescription "
                        f"indicates that only values {good_values} are "
                        f"supported. Consider adding '{bad_value}' to the "
                        f"list of possible values for attribute '{category}'.")

        return True

    def __repr__(self):
        rep = self.__dict__.copy()
        if not self.encoded:
            rep.pop("encoded")
        if not self.extra_data:
            rep.pop("extra_data")
        if not self.detections:
            rep.pop("detections")
        if not self.attributes:
            rep.pop("attributes")
        if not self.tracked:
            rep.pop("tracked")
        return str(rep)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
