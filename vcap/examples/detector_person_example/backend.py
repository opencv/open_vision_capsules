from typing import Dict

import numpy as np

from vcap import (
    Clamp,
    DetectionNode,
    rect_to_coords,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils.backends import TFObjectDetector


class Backend(TFObjectDetector):
    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState):
        if options["scale_frame"]:
            max_frame_side_length = options["scale_frame_max_side_length"]
            clamp = Clamp(frame=frame,
                          max_width=max_frame_side_length,
                          max_height=max_frame_side_length)
            frame = clamp.apply()

        predictions = self.send_to_batch(frame).result()

        results = []

        # Convert all predictions with the required confidence that are people
        # to DetectionNodes
        for pred in predictions:
            if pred.confidence >= options["detection_threshold"] \
                    and pred.name == "person":
                results.append(DetectionNode(
                    name=pred.name,
                    coords=rect_to_coords(pred.rect)))

        # If we scaled the frame down earlier before processing, we need to
        # scale detections back up to match the original frame size
        if options["scale_frame"]:
            clamp.scale_detection_nodes(results)

        return results
