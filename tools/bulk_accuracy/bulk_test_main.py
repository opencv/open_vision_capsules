import glob
from io import BytesIO
from pathlib import Path
import base64

import cv2
import numpy as np
from PIL import Image

from tools.bulk_accuracy.capsule_accuracy_common import (
    SampleType,
    AccuracyProcessor,
)
from tools.bulk_accuracy.capsule_accuracy_report import output_detection_result_csv, sum_classified_data
from tools.bulk_accuracy.capsule_mgmt_local import LocalCapsuleManagement, load_local_capsule
from tools.bulk_accuracy.capsule_mgmt_remote import RemoteCapsuleManagement


def test_safety_string_cell_video_file():
    input_path = "/home/leefr/Downloads/shanxi/test.mp4"
    capsule_names = ["detector_driver_and_special_vehicle_fast", "detector_cell", "tracker_vehicle_iou", "encoder_person_openvino", "tracker_person"]
    capsule_mgmt = RemoteCapsuleManagement(capsule_names)
    accuracyProcessor = AccuracyProcessor(
        capsule_mgmt,
        SampleType.video_file,
        input_path,
        1,
        show_rendered_image=True,
    )
    accuracyProcessor.run()
    output_detection_result_csv("/home/leefr/temp/analysis-data-cell", accuracyProcessor.detected_detail_data)


def test_safety_string_cell_images():
    input_path = "/home/leefr/Downloads/shanxi/images"
    capsule_names = ["detector_driver_and_special_vehicle_fast", "detector_cell", "tracker_vehicle_iou", "encoder_person_openvino", "tracker_person"]
    capsule_mgmt = RemoteCapsuleManagement(capsule_names)
    accuracyProcessor = AccuracyProcessor(
        capsule_mgmt,
        SampleType.image,
        input_path,
        1,
        show_rendered_image=True,
    )
    accuracyProcessor.run()
    output_detection_result_csv("/home/leefr/temp/analysis-data-cell", accuracyProcessor.detected_detail_data)


def test_classifier_video_file():
    # capsule_names = ["detector_person_openvino", "classifier_phoning_openvino"]
    # capsule_mgmt = RemoteCapsuleManagement(["classifier_phoning_openvino"])
    # input_path = "/home/leefr/Downloads/phoning"
    input_path = "/home/leefr/Downloads/20220524HBPhoning/test"

    capsule_mgmt = LocalCapsuleManagement()
    analyze_files = [
        Path(p).resolve() for p in glob.iglob(input_path + "/**/*.mp4", recursive=True)
    ]

    detected_data = []
    video_file_num = 1
    for video_file in analyze_files:
        accuracyProcessor = AccuracyProcessor(
            capsule_mgmt,
            SampleType.video_file,
            video_file,
            video_file_num,
            show_rendered_image=True,
            filter_class_names=["person"],
        )
        # accuracyProcessor.start()
        # accuracyProcessor.join()
        accuracyProcessor.run()
        detected_data.extend(accuracyProcessor.detected_detail_data)
        video_file_num += 1
    output_detection_result_csv("/home/leefr/temp/analysis-data", detected_data)


def test_classifier_images():
    # options = {
    #     "true threshold": 0.0,
    #     "false threshold": 0.0,
    # }
    capsule_mgmt = LocalCapsuleManagement(only_classified=True)
    # capsule_names = ["detector_person_administration", "classifier_phoning_openvino"]

    accuracyProcessor = AccuracyProcessor(
        capsule_mgmt,
        SampleType.image,
        "/home/leefr/capsules-test/pictures/test-data/phone",
        1,
        show_rendered_image=False,
        filter_class_names=["person"],
    )
    accuracyProcessor.run()
    detected_data = accuracyProcessor.detected_detail_data
    output_detection_result_csv("/home/leefr/temp/analysis-data-images", detected_data)
    sum_classified_data(detected_data)


def test_package_capsule():
    packaged_capsule_path = "/home/leefr/brainframe/pharmacy/private/attach_original_image1.2.cap"
    unpackaged_capsule_path = "/home/leefr/brainframe/pharmacy/private/attach_original_image"
    capsule_path = Path(packaged_capsule_path)
    if capsule_path.exists():
        capsule_path.unlink()
    load_local_capsule(
        packaged_capsule_path=packaged_capsule_path,
        unpackaged_capsule_path=unpackaged_capsule_path,
    )


def main():
    test_package_capsule()
    # test_classifier_video_file()
    # test_classifier_images()
    # test_safety_string_cell_video_file()
    # test_safety_string_cell_images()


if __name__ == "__main__":
    main()
