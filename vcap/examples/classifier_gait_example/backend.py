import numpy as np
from typing import Dict

from . import config
from vcap import (
    Crop,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils.backends import TFImageClassifier


class Backend(TFImageClassifier):
    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        crop = (Crop
                .from_detection(detection_node)
                .pad_percent(top=10, bottom=10, left=10, right=10)
                .apply(frame))

        prediction = self.send_to_batch(crop).result()

        detection_node.attributes[config.category] = prediction.name
        detection_node.extra_data[config.extra_data] = prediction.confidence
        return detection_node
