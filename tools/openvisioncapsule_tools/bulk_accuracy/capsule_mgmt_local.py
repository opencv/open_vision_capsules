import cv2
from copy import deepcopy
from pathlib import Path

from vcap import DetectionNode
from vcap.loading.capsule_loading import load_capsule
from vcap.loading.packaging import package_capsule

from tools.bulk_accuracy.capsule_mgmt_basic import BasicCapsuleManagement


class LocalCapsuleManagement(BasicCapsuleManagement):
    def __init__(self, only_classified=False):
        self.only_classified = only_classified
        self.classifier_name = "phoning"
        self.detect_person, self.detect_person_options = load_local_capsule(
            "/home/leefr/brainframe/pharmacy/private/detector_80_objects.cap",
            "/home/leefr/brainframe/pharmacy/private/detector_80_objects"
        )
        self.classified_phoning, self.classified_phoning_options = load_local_capsule(
            "/home/leefr/brainframe/pharmacy/private/classifier_phoning_factory_openvino3.1.cap",
            "/home/leefr/brainframe/pharmacy/private/classifier_phoning_factory_openvino",
            {
                "true threshold": 0.0,
                "false threshold": 0.0,
            }
        )

    def process_image(self, frame):
        if self.only_classified:
            h_orig, w_orig = frame.shape[:-1]
            points = [[0, 0], [w_orig, h_orig]]
            detection = DetectionNode(
                name=self.classifier_name,
                coords=points,
                attributes={},
                extra_data={}
            )

            self.classified_phoning.process_frame(
                frame=frame,
                detection_node=detection,
                options=self.classified_phoning_options,
                state=self.classified_phoning.stream_state(),
            )
            return [detection]
        else:
            detections = self.detect_person.process_frame(
                frame=frame,
                detection_node=None,
                options=self.detect_person_options,
                state=self.detect_person.stream_state(),
            )
            for detection in detections:
                self.classified_phoning.process_frame(
                    frame=frame,
                    detection_node=detection,
                    options=self.classified_phoning_options,
                    state=self.classified_phoning.stream_state(),
                )
            return detections

    def get_positions(self, bbox):
        return bbox.x1, bbox.x2, bbox.y1, bbox.y2


def load_local_capsule(packaged_capsule_path, unpackaged_capsule_path, options=None):
    if not Path(packaged_capsule_path).exists():
        package_capsule(Path(unpackaged_capsule_path), Path(packaged_capsule_path))

    capsule = load_capsule(path=packaged_capsule_path)
    capsule_options = deepcopy(capsule.default_options)

    if options is not None:
        for key in options:
            val = options[key]
            capsule_options[key] = val
    return capsule, capsule_options


class LocalCapsule:
    def __init__(self, packaged_capsule_path, unpackaged_capsule_path, options=None):
        self.packaged_capsule_path = packaged_capsule_path
        self.unpackaged_capsule_path = unpackaged_capsule_path
        self.options = options
        self.capsule = None
        self.capsule_options = None
        self.initial_capsule()

    def initial_capsule(self, ):
        if not Path(self.packaged_capsule_path).exists():
            package_capsule(Path(self.unpackaged_capsule_path), Path(self.packaged_capsule_path))

        self.capsule = load_capsule(path=self.packaged_capsule_path)
        self.capsule_options = deepcopy(self.capsule.default_options)

        if self.options is not None:
            for key in self.options:
                val = self.options[key]
                self.capsule_options[key] = val

    def process_image(self, frame, detections=None):
        if frame is None:
            return None
        if detections is None:
            detections = self.capsule.process_frame(
                frame=frame,
                detection_node=None,
                options=self.capsule_options,
                state=self.capsule.stream_state,
            )
        else:
            for detection in detections:
                self.capsule.process_frame(
                    frame=frame,
                    detection_node=detection,
                    options=self.capsule_options,
                    state=self.capsule.stream_state,
                )
        return detections


if __name__ == "__main__":
    # /home/leefr/capsules-test/capsules/detector_person_openvino.cap
    img = cv2.imread("/home/leefr/Pictures/test.jpg")
    local_capsule_mgmt = LocalCapsuleManagement()
    classified_detections = local_capsule_mgmt.process_image(img)

    print(classified_detections)
