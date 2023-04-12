import base64
from copy import deepcopy

import cv2
from brainframe.api import BrainFrameAPI
from brainframe.api.bf_codecs import image_utils


def get_positions(bbox):
    p1, p2 = bbox[0], bbox[2]
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    return x1, x2, y1, y2


def get_first_item(tup4):
    a, b, c, d = tup4
    return c


class VideoEventLoop:
    def __init__(self, bf_server_url):
        self.exit_flag = False
        self.api: BrainFrameAPI = BrainFrameAPI(bf_server_url)
        print("Connecting to Brainframe Server...")
        self.api.wait_for_server_initialization()
        self.stream_processors = {}
        print("Connected to Brainframe Server...")

    def video_event_loop(self):
        for zone_status_packet in self.api.get_zone_status_stream(timeout=5):
            if self.exit_flag:
                return
            for stream_id, zone_statuses in zone_status_packet.items():
                if stream_id not in self.stream_processors:
                    stream_processor = StreamProcessor(stream_id)
                    self.stream_processors[stream_id] = stream_processor
                else:
                    stream_processor = self.stream_processors[stream_id]
                stream_processor.process_zone_statuses(zone_statuses)
                if stream_processor.latest_rendered_img is not None:
                    # cv2.waitKey(1)
                    # cv2.imshow("demo", stream_processor.latest_rendered_img)
                    pass


class StreamProcessor:
    backend_color = (255, 255, 255)
    frontend_color = (0, 0, 0)

    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.latest_original_img = None
        self.latest_detections = None
        self.latest_rendered_img = None
        self.latest_detection_info = {}
        self.detection_number = 0

    def process_zone_statuses(self, zone_statuses):
        self.detection_number = 0
        if zone_statuses is None:
            return
        for zone_status_name in zone_statuses:
            zone_status = zone_statuses[zone_status_name]
            if zone_status_name == "Screen":
                self.pickup_image_from_detections(zone_status)
            detections = zone_status.within
            self.render_image(zone_status_name, detections)

    def pickup_image_from_detections(self, zone_status):
        attach_original_image_detection = None
        idx = -1
        for detection in zone_status.within:
            idx += 1
            if detection.class_name == "attach_original_image":
                attach_original_image_detection = detection
                break
        if attach_original_image_detection is not None:
            del zone_status.within[idx]
        if attach_original_image_detection is not None and "img" in attach_original_image_detection.extra_data:
            img_encoded = attach_original_image_detection.extra_data["img"]
            img_bytes = base64.b64decode(img_encoded.encode("ascii"))
            self.latest_original_img = image_utils.decode(img_bytes)
            self.latest_rendered_img = deepcopy(self.latest_original_img)

    def render_image(self, zone_status_name, detections):
        self.latest_rendered_img = deepcopy(self.latest_original_img)
        print(f"set to original image")
        detections = sorted(detections, key=lambda e: get_first_item(get_positions(e.bbox)))
        lines = []
        for detection in detections:
            self.detection_number += 1
            bbox = detection.bbox
            x1, x2, y1, y2 = get_positions(bbox)
            size = f"{abs(x1 - x2)}X{abs(y1 - y2)}"

            lines.append("")
            lines.append(f"{self.detection_number}")
            lines.append(f"{detection.class_name}: {size}")
            for key in detection.attributes:
                lines.append(f"{key}: {detection.attributes[key]}")

            for key in detection.extra_data:
                lines.append(f"{key}: {detection.extra_data[key]}")

            print(f"draw detection: {self.detection_number}")
            cv2.rectangle(self.latest_rendered_img, (x1, y1), (x2, y2), StreamProcessor.backend_color, 1)
            # cv2.rectangle(self.latest_rendered_img, (x1, y1), (x1 + 50, y1 + 50), StreamProcessor.backend_color, -1)
            cv2.circle(self.latest_rendered_img, (x1 + 25, y1 + 25), 25, StreamProcessor.backend_color, -1)
            cv2.putText(
                img=self.latest_rendered_img,
                text=f"{self.detection_number}",
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                org=(x1 + 15, y1 + 40),
                color=StreamProcessor.frontend_color,
                thickness=2,
            )
        self.latest_detection_info[zone_status_name] = lines
        if self.latest_rendered_img is not None:
            cv2.waitKey(1)
            cv2.imshow("rendered", self.latest_rendered_img)
            print(f"show image")
            # timestamp = datetime.datetime.now()
            # cv2.imwrite(f"/tmp/{timestamp}.jpg", self.latest_rendered_img)
        if self.latest_original_img is not None:
            cv2.waitKey(1)
            cv2.imshow("original", self.latest_original_img)

    def get_rendered_image(self, height=None, width=None):
        pass


def main():
    video_event_loop = VideoEventLoop("http://localhost")
    video_event_loop.video_event_loop()


if __name__ == "__main__":
    main()
