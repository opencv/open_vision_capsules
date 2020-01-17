from typing import List, Union, Optional

import numpy as np
from vcap.detection_node import DetectionNode

Num = Union[int, float]


def detection_area(detection: DetectionNode):
    bbox = detection.bbox.xywh
    return bbox_area(bbox)


def detection_intersection(detection: DetectionNode,
                           candidates: List[DetectionNode]) -> np.ndarray:
    candidate_bboxes = [candidate.bbox.xywh for candidate in candidates]
    bbox = detection.bbox.xywh
    return bbox_intersection(bbox, candidate_bboxes)


def detection_iou(detection: DetectionNode,
                  candidates: List[DetectionNode]) -> np.ndarray:
    """Computer intersection over union.

    Parameters
    ----------
    bbox : DetectionNode
    candidates : List of DetectionNodes

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    candidate_bboxes = [candidate.bbox.xywh for candidate in candidates]
    bbox = detection.bbox.xywh
    return bbox_iou(bbox, candidate_bboxes)


def bbox_area(bbox: List[Num]):
    bbox = np.array(bbox)
    area_bbox = bbox[2:].prod()

    return area_bbox


def bbox_intersection(bbox: List[Num], candidate_bboxes: List[List[Num]]) \
        -> np.ndarray:
    candidate_bboxes = np.array(candidate_bboxes)
    bbox = np.array(bbox)

    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidate_bboxes[:, :2]
    candidates_br = candidate_bboxes[:, :2] + candidate_bboxes[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)

    return area_intersection


def bbox_iou(bbox: List[Num], candidate_bboxes: List[List[Num]]) -> np.ndarray:
    """
    :param bbox: Bounding box in the format [x, y, width, height]
    :param candidate_bboxes: A list of bounding boxes in the bbox format
    :return:
    """
    candidate_bboxes = np.array(candidate_bboxes)
    bbox = np.array(bbox)

    area_bbox = bbox_area(bbox)
    area_candidates = candidate_bboxes[:, 2:].prod(axis=1)

    area_intersection = bbox_intersection(bbox, candidate_bboxes)
    area_union = area_bbox + area_candidates - area_intersection

    iou = area_intersection / area_union

    return iou


def iou_cost_matrix(
        detections_a: List[DetectionNode],
        detections_b: List[DetectionNode],
        detections_a_indices: Optional[List[int]] = None,
        detections_b_indices: Optional[List[int]] = None) -> np.ndarray:
    """An intersection over union distance metric.

    Parameters
    ----------
    detections_a : List[vcap.DetectionNode]
        A list of detections.
    detections_b : List[vcap.DetectionNode]
        A list of detections.
    detections_a_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults to
        all `detections`.
    detections_b_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(detections_a_indices), len(detections_b_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """

    if detections_a_indices is None:
        detections_a_indices = np.arange(len(detections_a))
    if detections_b_indices is None:
        detections_b_indices = np.arange(len(detections_b))

    cost_matrix = np.zeros(
        (len(detections_a_indices), len(detections_b_indices)))
    for row, track_idx in enumerate(detections_a_indices):
        detection_from_a = detections_a[track_idx]
        candidate_detections = [detections_b[index]
                                for index in detections_b_indices]
        cost_matrix[row, :] = 1. - detection_iou(detection_from_a,
                                                 candidate_detections)
    return cost_matrix
