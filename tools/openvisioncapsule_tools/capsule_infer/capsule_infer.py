import json
import os
from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import List, NoReturn, Optional

import cv2
import numpy as np
from vcap import (
    CAPSULE_EXTENSION,
    BaseCapsule,
    DetectionNode,
    NodeDescription,
    load_capsule,
    package_capsule,
)

import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, parent_dir)

from openvisioncapsule_tools.print_module_info import print_module_info

def update_options(default_options, input_options):
    capsule_options = {}
    for option_name, val in default_options.items():
        if input_options is not None and option_name in input_options:
            capsule_options[option_name] = input_options[option_name]
        else:
            capsule_options[option_name] = val.default

    f = open('options.json', "w")
    json.dump(capsule_options, f, indent=4)
    f.close()

    return capsule_options


def capsule_inference(packaged_capsule_path, unpackaged_capsule_path, image_paths, detection_name, input_options=None, capsule_key=None, wait_time=None):
    start_time = time()
    capsule = load_capsule(
        path=packaged_capsule_path, source_path=unpackaged_capsule_path, key=capsule_key
    )
    load_time_ms = (time() - start_time) * 1000
    capsule_file_size = os.path.getsize(packaged_capsule_path)
    print(f"Capsule file {capsule_file_size} bytes, load time {load_time_ms:0.4f}ms")

    detection_required = validate_capsule(capsule)

    classes = capsule.output_type.detections
    print(f"Available detection class_names are {classes}")

    capsule_options = update_options(capsule.options, input_options)
    print(f'Capsule options are {capsule_options}')

    capsule_results = []

    for image_path in image_paths:
        # print(f"Running inference on {image_path}")
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        if detection_required:
            h_orig, w_orig = image.shape[:-1]
            points = [[0, 0], [w_orig, h_orig]]

            detections = DetectionNode(
                name=detection_name,
                coords=points,
                attributes={},
                extra_data={}
            )
            detection_node = detections
        else:
            detection_node = None

        start_time = time()
        results = capsule.process_frame(
            frame=image,
            detection_node=detection_node,
            options=capsule_options,
            state=capsule.stream_state(),
        )
        proc_time_ms = (time() - start_time) * 1000
        print(f"Capsule process frame time {proc_time_ms:0.4f}ms")

        if detection_node:
            valid_description = capsule.output_type.describes(detection_node)

            if not valid_description:
                print(
                    f"Ignoring node from {capsule.name} because it does not match the "
                    f"capsules output type description. "
                    f"\nNode: {detection_node} "
                    f"\nDescription: {capsule.output_type}"
                )

            capsule_results.append(detection_node)

        if not detection_required:
            render_detections(
                classes, image=image, results=results, color=(30, 255, 255), thickness=2
            )

        if wait_time is not None:
            if results is not None:
                [print(f"Results: {result}") for result in results]
            cv2.imshow("Results", image)
            cv2.waitKey(wait_time)

    return capsule_results


def render_detections( classes,
    image: np.ndarray, results: List[DetectionNode], color, thickness
):
    def line(from_pt, to_pt):
        line_color = (30, 255, 255)
        from_pt, to_pt = tuple((int(x), int(y)) for x, y in (from_pt, to_pt))
        cv2.line(image, from_pt, to_pt, color, thickness=thickness, lineType=cv2.LINE_8)

    def text(coords, label, line=0, alpha=0.5, line_spacing=1.2):
        x, y = coords[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (255, 100, 100)
        text_color_bg = color
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size

        x0 = int(x)
        y0 = int(y + text_h * line * line_spacing)
        pt1 = (x0, y0)
        rect_h = int(text_h * line_spacing)
        pt2 = (x0 + text_w, y0 + rect_h)
        overlay = image.copy()
        cv2.rectangle(image, pt1, pt2, text_color_bg, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        pt = (x0, y0 + text_h)
        cv2.putText(image, label, pt, font, font_scale, text_color, font_thickness)

    if results is None:
        return

    for detection in results:
        class_name = detection.class_name
        class_id = classes.index(class_name)

        color = (int(min(class_id * 12.5, 255)), min(class_id * 7, 255), min(class_id * 5, 255))
        if len(detection.coords) > 2:
            for from_pt, to_pt in zip(detection.coords[0:-1], detection.coords[1:]):
                line(from_pt, to_pt)

            # Close the region
            line(detection.coords[0], detection.coords[-1])

        elif len(detection.coords) == 2:
            line(detection.coords[0], detection.coords[1])
        else:
            print(f"Unsupported number of coordinates: {detection.coords}")

        if 'detection_confidence' in detection.extra_data:
            detection_confidence = detection.extra_data['detection_confidence']
        elif 'confidence' in detection.extra_data:
            detection_confidence = detection.extra_data['confidence']
        else:
            detection_confidence = 0
        label = f'{class_name}: {detection_confidence:.04f}'
        text(detection.coords, label)

        if 'text' in detection.extra_data:
            text_result = detection.extra_data['text']
            label = f'{text_result}'
            text(detection.coords, label, 1)
            if 'confidence' in detection.extra_data:
                confidence = detection.extra_data['confidence']
                label = f'confidence: {confidence:.04f}'
                text(detection.coords, label, 2)

def validate_capsule(capsule: BaseCapsule) -> Optional[NoReturn]:
    """Returns errors if the capsule is not compatible with this script"""
    detection_required = False
    if capsule.input_type.size is not NodeDescription.Size.NONE:
        print(
            "This capsule requires detections as input, the whole input image is"
            " to be provided to the capsule as an input of a detection."
        )
        detection_required = True
    return detection_required


def capsule_infer_add_args():
    parser = ArgumentParser(
        description="A helpful tool for running inference on a capsule."
    )

    parser.add_argument(
        "--capsule",
        required=True,
        type=Path,
        help="The path to either an unpackaged or packaged capsule",
    )
    parser.add_argument(
        "--options",
        type=Path,
        default="options.json",
        help="File to store the capsule options",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="Paths to one or more images to run inference on. If the path is a "
        "directory, then *.png or *.jpg images in the directory will be used.",
    )
    parser.add_argument(
        "--detection",
        help="A detection name is required for classifier test.",
    )
    parser.add_argument(
        "--capsule-key",
        type=str,
        required=False,
        help="Capsule key to load an encrypted capsule. Use 'brainframe' as the key "
        "to load capsules signed for BrainFrame"
    )
    return parser


def read_options(option_file):
    if os.path.isfile(option_file):
        with open(option_file) as file:
            options = json.load(file)
    else:
        options = None
    return options


def parse_images(images_input):
    images = []
    for path in images_input:
        if path.is_dir():
            images += list(path.glob("*.png"))
            images += list(path.glob("*.jpg"))
            images += list(path.glob("*.jpeg"))

            if len(images) == 0:
                print(f"No images were found in the directory {images_input}!")
                exit(-1)
        else:
            images.append(path)
    return images


def parse_capsule_info(args):
    capsule_name = args.capsule.with_suffix(CAPSULE_EXTENSION).name
    if args.capsule.is_dir():
        unpackaged_capsule_path = args.capsule
        packaged_capsule_path = unpackaged_capsule_path.parent / capsule_name
        package_capsule(args.capsule, packaged_capsule_path)
    else:
        unpackaged_capsule_path = None
        packaged_capsule_path = args.capsule
    return packaged_capsule_path, unpackaged_capsule_path, capsule_name


def capsule_options_and_key(args):
    if args.capsule_key == 'brainframe':
        capsule_key = None
    else:
        capsule_key = args.capsule_key

    if args.options is not None:
        input_options = read_options(args.options)
    else:
        input_options = None

    return input_options, capsule_key


def capsule_infer(args):

    image_paths = parse_images(args.images)

    packaged_capsule_path, unpackaged_capsule_path, capsule_name = parse_capsule_info(args)

    input_options, capsule_key = capsule_options_and_key(args)

    results = capsule_inference(packaged_capsule_path, unpackaged_capsule_path, image_paths, args.detection, input_options, capsule_key, 0)

    print(results)


def capsule_infer_main():
    parser = capsule_infer_add_args()
    args = parser.parse_args()
    capsule_infer(args)


if __name__ == "__main__":
    print('Module information:')
    print_module_info('vcap')
    print_module_info('vcap_utils')
    print_module_info('openvino')
    capsule_infer_main()
