from uuid import uuid4

import numpy as np
import pytest

from vcap import NodeDescription, DetectionNode

DIFFERENCE_CASES = [
    # Test the null case
    (NodeDescription(size=NodeDescription.Size.NONE),
     NodeDescription(size=NodeDescription.Size.NONE),
     NodeDescription(size=NodeDescription.Size.NONE),
     NodeDescription(size=NodeDescription.Size.NONE)),

    # Test encoding differences
    (NodeDescription(
        size=NodeDescription.Size.SINGLE,
        encoded=False),
     NodeDescription(
         size=NodeDescription.Size.SINGLE,
         encoded=True),
     NodeDescription(
         size=NodeDescription.Size.SINGLE,
         encoded=True),
     NodeDescription(
         size=NodeDescription.Size.SINGLE)),

    # Test detection differences and Size differences
    (NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["car"]),
     NodeDescription(
         size=NodeDescription.Size.ALL,
         detections=["car", "person"]),
     NodeDescription(
         size=NodeDescription.Size.ALL,
         detections=["person"]),
     NodeDescription(
         size=NodeDescription.Size.SINGLE)),

    # Test attribute differences
    (NodeDescription(
        size=NodeDescription.Size.SINGLE,
        attributes={"Gait": ["walking", "running"]}),
     NodeDescription(
         size=NodeDescription.Size.SINGLE,
         attributes={"Gait": ["walking", "running"],
                     "Speeding": ["yeah", "naw"]}),
     NodeDescription(
         size=NodeDescription.Size.SINGLE,
         attributes={"Speeding": ["yeah", "naw"]}),
     NodeDescription(
         size=NodeDescription.Size.SINGLE)),

    # Test extra_data differences
    (NodeDescription(
        size=NodeDescription.Size.SINGLE,
        extra_data=["behavior_confidence"]),
     NodeDescription(
         size=NodeDescription.Size.SINGLE,
         extra_data=["behavior_confidence", "det_score"]),
     NodeDescription(
         size=NodeDescription.Size.SINGLE,
         extra_data=["det_score"]),
     NodeDescription(
         size=NodeDescription.Size.SINGLE)),

    # Test tracked differences
    (NodeDescription(
        size=NodeDescription.Size.SINGLE,
        tracked=False),
     NodeDescription(
         size=NodeDescription.Size.SINGLE,
         tracked=True),
     NodeDescription(
         size=NodeDescription.Size.SINGLE,
         tracked=True),
     NodeDescription(
         size=NodeDescription.Size.SINGLE))
]


@pytest.mark.parametrize(('desc1', 'desc2', 'diff_1_2', 'diff_2_1'),
                         DIFFERENCE_CASES)
def test_node_description_difference(desc1, desc2, diff_1_2, diff_2_1):
    """Test comparing two node descriptions"""
    assert desc1.difference(desc2) == diff_1_2
    assert desc2.difference(desc1) == diff_2_1


DESCRIPTION_CASES = [
    (DetectionNode(
        name="person",
        coords=[[0, 0]] * 4),
     True, False, False, False, False, False, False),

    (DetectionNode(
        name="person",
        coords=[[0, 0]] * 4,
        encoding=np.array([1])),
     True, False, False, True, False, False, False),

    (DetectionNode(
        name="hair",
        coords=[[0, 0]] * 4,
        attributes={"Gender": "boy"}),
     False, False, True, False, False, False, False),

    (DetectionNode(
        name="cat",
        coords=[[0, 0]] * 4,
        attributes={"Uniform": "Police", "Gender": "girl"},
        encoding=np.ndarray([1, 2, 3, 4, 5])),
     False, False, True, False, True, True, False),

    (DetectionNode(
        name="person",
        coords=[[0, 0]] * 4,
        attributes={"more": "irrelevant"},
        encoding=np.ndarray([1, 2, 3, 4, 5]),
        extra_data={"behavior_confidence": 0.9991999}),
     True, True, False, True, False, False, False),

    (DetectionNode(
        name="person",
        coords=[[0, 0]] * 4,
        track_id=uuid4()),
     True, False, False, False, False, False, True)
]


@pytest.mark.parametrize(('det_node',
                          'nd1', 'nd2', 'nd3', 'nd4', 'nd5', 'nd6', 'nd7'),
                         DESCRIPTION_CASES)
def test_detection_node_descriptions(det_node,
                                     nd1, nd2, nd3, nd4, nd5, nd6, nd7):
    """Test that DetectionNodes can accurately generate a node_description
    for themselves"""

    node_desc_1 = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["person", "dog"])
    node_desc_2 = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        extra_data=["behavior_confidence"])
    node_desc_3 = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        attributes={"Gender": ["boy", "girl"]})
    node_desc_4 = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["person"],
        encoded=True)
    node_desc_5 = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["hair", "cat"],
        attributes={"Uniform": ["Police", "Worker", "Civilian"]},
        encoded=True)
    node_desc_6 = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        attributes={"Uniform": ["Police", "Worker", "Civilian"],
                    "Gender": ["boy", "girl"]})
    node_desc_7 = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        tracked=True)

    assert NodeDescription(size=NodeDescription.Size.SINGLE).describes(det_node)
    assert node_desc_1.describes(det_node) == nd1
    assert node_desc_2.describes(det_node) == nd2
    assert node_desc_3.describes(det_node) == nd3
    assert node_desc_4.describes(det_node) == nd4
    assert node_desc_5.describes(det_node) == nd5
    assert node_desc_6.describes(det_node) == nd6
    assert node_desc_7.describes(det_node) == nd7


def test_describes_error():
    # Test that a ValueError gets raised when a DetectionNode has an attribute
    # with values that are not described by the NodeDescription
    node_desc = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        attributes={"Gender": ["boy", "girl"]})
    det_node = DetectionNode(
        name="irrelevant",
        coords=[[0, 0]] * 4,
        attributes={"Gender": "NOT EXISTENT VALUE"}
    )
    with pytest.raises(ValueError):
        node_desc.describes(det_node)
