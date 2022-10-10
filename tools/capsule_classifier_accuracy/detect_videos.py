import os
import argparse
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
# from PIL import Image
import cv2
import glob

from datetime import datetime
import logging
from vcap import (
    options,
    common_detector_options,
    load_capsule, FloatOption
)
from tools.capsule_infer.capsule_infer import (
    read_options,
    capsule_options_and_key,
    capsule_inference,
    parse_images,
    capsule_infer_add_args,
    parse_capsule_info)

Path("../logs").mkdir(parents=True, exist_ok=True)
appendix = datetime.now().strftime('%Y%d%m%H%M%S')
logging.basicConfig(filename='../logs/brainframe' + appendix + '.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S',
                    level=logging.INFO)

DETECTOR_PERSON_CAPSULE_PATH = "/home/leefr/capsules-test/capsules/detector_person_openvino.cap"
CLASSIFIER_PHONING_CAPSULE_PATH = '/home/leefr/capsules-test/capsules/classifier_phoning_factory_openvino.cap'

capsule_detector = load_capsule(path=DETECTOR_PERSON_CAPSULE_PATH)
options_capsule_detector = deepcopy(capsule_detector.default_options)
options_capsule_detector["threshold"] = 0.5

capsule_classifier = load_capsule(path=CLASSIFIER_PHONING_CAPSULE_PATH)
options_capsule_classifier = deepcopy(capsule_classifier.default_options)
options_capsule_classifier["threshold"] = 0.5

img_id = 1


def split_video(video_file_path, dest_file_path):
    cap = cv2.VideoCapture(str(video_file_path))
    while cap.isOpened():
        _, img = cap.read()
        if img is None:
            print("WAS NONE")
            break

        detect_persons(img, dest_file_path)


def detect_persons(img, dest_file_path):
    all_detections = capsule_detector.process_frame(
        frame=img,
        detection_node=None,
        options=options_capsule_detector.default_options,
        state=options_capsule_detector.stream_state()
    )
    people_detections = [d for d in all_detections
                         if d.class_name == "person"]

    classify_persons(img, people_detections, dest_file_path)


def classify_persons(img_origin, people_detections, dest_file_path):
    global img_id

    for people in people_detections:
        bbox = people.bbox
        x0, y0, x1, y1 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        # img_object = img_origin.crop((x0, y0, x1, y1))
        img_object = img_origin[y0:y1, x0:x1]
        if img_object is None:
            print(f"Failed to crop object: {people}")
            continue

        class_of_object = capsule_classifier.process_frame(
            frame=img_object,
            detection_node=None,
            options=options_capsule_classifier.default_options,
            state=options_capsule_classifier.stream_state()
        )

        img_object_path = os.path.join(dest_file_path, "{:06d}.jpg".format(img_id))
        img_id += 1

        try:
            # img_object.save(img_object_path)
            cv2.imwrite(img_object_path, img_object)
        except Exception as e:
            print(f"Failed to save object: {img_object_path} {e}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="A helpful tool for running inference and generate accuracy benchmarking report on a capsule."
    )

    # args = parser.parse_args()
    #
    # cmdline = read_cmdline()

    # construct the argument parser and parse the arguments
    # parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--capsule", required=True,
                        help="Capsule path")
    parser.add_argument("-v", "--source", required=True,
                        help="Either a video file or a video directory "
                             "to process")
    parser.add_argument("-d", "--dest", required=True,
                        help="A directory to store the processed images")
    parser.add_argument("-w", "--num_workers", type=int, default=10,
                        help="How many videos to parse at a time")
    args = parser.parse_args()
    print("Starting Converted with parameters", args)

    analyze_files = []
    path = Path(args.source).resolve()
    if path.is_dir():
        analyze_files = [Path(p).resolve() for p in glob.iglob(str(path) + "/**/*.*", recursive=True)]
    elif path.is_file():
        analyze_files = [path]

    Path(args.dest).mkdir(parents=True, exist_ok=True)

    # Verify that all the video files exist
    analyze_files = [str(file.resolve()) for file in analyze_files if
                     Path(file).exists()]

    for src_file in analyze_files:
        split_video(src_file, args.dest)
        # print(f"Finished: {src_file}")
