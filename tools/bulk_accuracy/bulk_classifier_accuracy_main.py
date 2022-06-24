import glob
from pathlib import Path

from brainframe.api import BrainFrameAPI

from tools.bulk_accuracy.capsule_accuracy_common import (
    SampleType,
    AccuracyProcessor,
)
from tools.bulk_accuracy.capsule_accuracy_report import output_detection_result_csv
from tools.bulk_accuracy.capsule_mgmt_local import LocalCapsuleManagement
from tools.bulk_accuracy.capsule_mgmt_remote import RemoteCapsuleManagement


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
        )
        # accuracyProcessor.start()
        # accuracyProcessor.join()
        accuracyProcessor.run()
        detected_data.extend(accuracyProcessor.detected_detail_data)
        video_file_num += 1
    output_detection_result_csv("/home/leefr/temp/analysis-data", detected_data)


def test_classifier_images():
    capsule_mgmt = LocalCapsuleManagement(only_classified=True)
    # capsule_names = ["detector_person_administration", "classifier_phoning_openvino"]

    accuracyProcessor = AccuracyProcessor(
        capsule_mgmt,
        SampleType.image,
        "/home/leefr/capsules-test/pictures/test-data/phone",
        1,
        show_rendered_image=True,
    )
    accuracyProcessor.run()
    detected_data = accuracyProcessor.detected_detail_data
    output_detection_result_csv("/home/leefr/temp/analysis-data-images", detected_data)


def main():
    test_classifier_video_file()
    # test_classifier_images()


if __name__ == "__main__":
    main()
