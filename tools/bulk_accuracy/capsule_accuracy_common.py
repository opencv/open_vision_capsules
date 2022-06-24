import glob
from enum import Enum
from pathlib import Path
from threading import Thread

from cv2 import cv2
import json

from tools.bulk_accuracy.capsule_mgmt_basic import BasicCapsuleManagement


class DetectionResult:
    def __init__(
            self, sample_path, offset_time, img_id, classified_type, confidence_value
    ):
        self.sample_path = sample_path
        self.offset_time = offset_time
        self.img_id = img_id
        self.classified_type = classified_type
        self.confidence_value = confidence_value

    @staticmethod
    def from_json(json_obj):
        rst = DetectionResult(
            json_obj["sample_path"],
            json_obj["offset_time"],
            json_obj["img_id"],
            json_obj["classified_type"],
            json_obj["confidence_value"],
        )
        return rst

    @staticmethod
    def from_csv(line: str):
        line = line.strip()
        objs = line.split(",")
        rst = DetectionResult(
            objs[0],
            int(objs[1]),
            int(objs[2]),
            objs[3] == "true",
            float(objs[4]),
        )
        return rst

    def to_json(self):
        json_obj = {
            "sample_path": self.sample_path,
            "offset_time": self.offset_time,
            "img_id": self.img_id,
            "classified_type": self.classified_type,
            "confidence_value": self.confidence_value,
        }
        return json_obj

    def to_json_str(self):
        json_obj = self.to_json()
        return json.dumps(json_obj)


class SampleType(Enum):
    video_rtsp = "video_rtsp"
    video_file = "video_file"
    image = "image"


class AccuracyProcessor(Thread):
    def __init__(
            self,
            capsule_mgmt,
            sample_type,
            sample_path,
            video_file_num,
            save_rendered_image=False,
            show_rendered_image=False,
            output_path="/tmp",
    ):
        super().__init__()
        self.detect_handler = DetectionHandler(self)
        self.capsule_mgmt: BasicCapsuleManagement = capsule_mgmt
        self.sample_type = sample_type
        self.sample_path = str(sample_path)
        self.video_file_num = video_file_num
        self.cv2_window_name = f"UltraAI: {video_file_num}"
        self.save_rendered_image = save_rendered_image
        self.show_rendered_image = show_rendered_image
        self.output_path = output_path
        self.detected_detail_data = []
        self.offset_time = 0.0
        self.img_id = 1
        self.img = None

    def run(self) -> None:
        # cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_AUTOSIZE)
        if (
                self.sample_type == SampleType.video_rtsp
                or self.sample_type == SampleType.video_file
        ):
            self.video_loop()
        elif self.sample_type == SampleType.image:
            self.image_loop()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    def video_loop(self):
        cap = cv2.VideoCapture(str(self.sample_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 25 if fps <= 0 else fps
        fps_interval = 1000 / fps
        while cap.isOpened():
            try:
                self.offset_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                # self.offset_time += fps_interval
                print(f"img_id={self.img_id}, offset_time={self.offset_time}")

                # if fps > 0:
                #     self.offset_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps * 1000
                # else:
                #     self.offset_time = cap.get(cv2.CAP_PROP_POS_MSEC)

                rst, self.img = cap.read()
                if not rst or self.img is None:
                    break
                self.process_image()
                self.show_image()
                self.img_id += 1
            except Exception as e:
                print(f"Failed to process {self.sample_path}: {e}")
                break
        cap.release()

    def image_loop(self):
        image_paths = [
            str(p)
            for p in glob.iglob(str(self.sample_path) + "/**/*.jpg", recursive=True)
        ]
        for image_path in image_paths:
            self.img = cv2.imread(image_path)
            self.process_image()
            self.show_image()
            self.img_id += 1
            print(f"img_id={self.img_id}")

    def append_detection_data(
            self,
            attribute_name,
            attribute_type,
            detection_confidence,
            attribute_name_confidence,
    ):
        result_of_detection = DetectionResult(
            self.sample_path,
            self.offset_time,
            self.img_id,
            attribute_type,
            attribute_name_confidence,
        )
        self.detected_detail_data.append(result_of_detection)

    def show_image(self):
        if not self.show_rendered_image:
            return

        # img = self.img
        # win_name = Path(self.sample_path).name
        # win_name = f"{self.img_id} {self.offset_time}"
        win_name = f"UltraAI:{self.video_file_num}"
        zoom_factor = 0.67
        self.img = cv2.resize(self.img, None, fx=zoom_factor, fy=zoom_factor)
        cv2.waitKey(1)
        cv2.imshow(self.cv2_window_name, self.img)

    def save_image(self):
        if not self.save_rendered_image:
            return
        saved_img_path = str(Path(self.output_path, f"{self.img_id}.jpg"))
        cv2.imwrite(saved_img_path, self.img)

    def process_image(self):
        # zoom_factor = 0.67
        # self.img = cv2.resize(self.img, None, fx=zoom_factor, fy=zoom_factor)

        detections = self.capsule_mgmt.process_image(self.img)
        if detections is None or len(detections) == 0:
            return

        for detection in detections:
            self.detect_handler.process_detection(detection)

        # self.show_image()
        self.save_image()


class DetectionHandler:
    colors = {
        "true": (0, 255, 0),
        "false": (255, 0, 0),
        "unknown": (0, 0, 255),
    }

    def __init__(self, accuracy_processor: AccuracyProcessor):
        self.accuracy_processor = accuracy_processor

    @staticmethod
    def get_attribute_type(detection):
        if detection is None:
            return None, None

        attributes = detection.attributes
        for attribute_name in attributes:
            # The attributes look like: {'phoning': 'false_phoning}
            value: str = attributes[attribute_name]
            attribute_type = value.split("_")[0]
            return attribute_name, attribute_type
        return None, None

    @staticmethod
    def get_confidence(detection, attribute_name):
        if detection is None:
            return None, None

        extra_data = detection.extra_data
        if "detection_confidence" in extra_data:
            detection_confidence = extra_data["detection_confidence"]
        else:
            detection_confidence = 0.0
        if f"{attribute_name}_confidence" in extra_data:
            attribute_name_confidence = extra_data[f"{attribute_name}_confidence"]
        else:
            attribute_name_confidence = 0.0

        return detection_confidence, attribute_name_confidence

    def render_image_with_detections(
            self,
            detection,
            attribute_name,
            attribute_type,
            detection_confidence,
            attribute_name_confidence,
    ):
        bbox = detection.bbox
        x1, x2, y1, y2 = self.accuracy_processor.capsule_mgmt.get_positions(bbox)
        color = DetectionHandler.colors[attribute_type]
        img = self.accuracy_processor.img
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.rectangle(img, (x1, y2 - 60), (x2 + 150, y2), (255, 255, 255), -1)
        cv2.putText(
            img=img,
            text=f"{attribute_name}:{attribute_type}",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            org=(x1, y2 - 40),
            color=color,
        )
        size = f"{abs(x1 - x2)}X{abs(y1 - y2)}"
        cv2.putText(
            img=img,
            text=size,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            org=(x1, y2 - 20),
            color=color,
        )
        cv2.putText(
            img=img,
            text=f"{attribute_name_confidence}",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            org=(x1, y2),
            color=color,
        )

    def process_detection(self, detection):
        attribute_name, attribute_type = self.get_attribute_type(detection)
        if attribute_name is None or attribute_type is None:
            return
        detection_confidence, attribute_name_confidence = self.get_confidence(
            detection, attribute_name
        )
        self.accuracy_processor.append_detection_data(
            attribute_name,
            attribute_type,
            detection_confidence,
            attribute_name_confidence,
        )
        if (
                self.accuracy_processor.save_rendered_image
                or self.accuracy_processor.show_rendered_image
        ):
            self.render_image_with_detections(
                detection,
                attribute_name,
                attribute_type,
                detection_confidence,
                attribute_name_confidence,
            )
