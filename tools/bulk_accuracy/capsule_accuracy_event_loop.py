import base64
import cv2
import numpy as np
from brainframe.api import BrainFrameAPI
from brainframe.api.bf_codecs import image_utils


def video_event_loop():
    bf_server_url = "http://localhost:80"
    api: BrainFrameAPI = BrainFrameAPI(bf_server_url)
    print("Connecting to Brainframe Server...")

    for zone_status_packet in api.get_zone_status_stream(timeout=5):
        for stream_id, zone_statuses in zone_status_packet.items():
            # print(zone_statuses)
            zone_status = zone_statuses['Screen']
            attach_original_image_detection = None
            idx = -1
            for detection in zone_status.within:
                idx += 1
                if detection.class_name == "attach_original_image":
                    attach_original_image_detection = detection
                    break
            if attach_original_image_detection is not None:
                del zone_status.within[idx]

            img_encoded = attach_original_image_detection.extra_data["img"]
            bbox = attach_original_image_detection.bbox
            img_bytes = base64.b64decode(img_encoded.encode("ascii"))
            img = image_utils.decode(img_bytes)
            cv2.imshow("event_loop", img)
            cv2.waitKey(1)


def main():
    video_event_loop()


if __name__ == "__main__":
    main()
    img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    # _, img_bytes = cv2.imencode('.jpg', img)
    # print(en)
    # img_rgb_arr = img[..., ::-1]
    # image = Image.fromarray(img_rgb_arr)
    # img_bytes = BytesIO()
    # image.save(img_bytes, format="jpeg")
    # img_bytes = img_bytes.getvalue()
    img_bytes = image_utils.encode("jpeg", img)
    print(img_bytes)
    encoded_img_bytes = base64.b64encode(img_bytes).decode("ascii")
    # encoded_img_bytes = img_bytes.decode("utf-8")
    print(encoded_img_bytes)
    decoded_img_bytes = base64.b64decode(encoded_img_bytes.encode("ascii"))
    # img_bytes = decoded_img_bytes.encode("utf-8")
    print(decoded_img_bytes)
    img = image_utils.decode(img_bytes)
    print(img)
    cv2.imshow("demo", img)
    cv2.waitKey(10000)
