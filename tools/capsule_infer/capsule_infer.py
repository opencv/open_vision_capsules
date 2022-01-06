from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import List, NoReturn, Optional, Tuple

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


def capsule_inference(packaged_capsule_path, unpackaged_capsule_path, image_paths, detection_name, true_threshold, false_threshold):

    capsule = load_capsule(
        path=packaged_capsule_path, source_path=unpackaged_capsule_path
    )

    detection_required = validate_capsule(capsule)

    classes = capsule.output_type.detections
    print(f"Available detection class_names are {classes}")

    # capsule_options = {'true threshold': true_threshold, 'false threshold': false_threshold}
    capsule_options = {'threshold': 0.5, 'nms_iou_thresh': 0.4}
    capsule_results = []

    for image_path in image_paths:
        print(f"Running inference on {image_path}")
        image = cv2.imread(str(image_path))

        if detection_required:
            h_orig, w_orig = image.shape[:-1]
            points = [[0, 0], [w_orig, h_orig]]

            detections = DetectionNode(
                name=detection_name,
                coords=points,
                attributes=[],
                extra_data=[]
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

        print(f"Inference time {proc_time_ms:0.4f}ms, results: {results}")

        capsule_results.append(detection_node)

        if not detection_required:
            render_detections(
                classes, image=image, results=results, color=(30, 255, 255), thickness=2
            )

        cv2.imshow("Results", image)
        cv2.waitKey()

    return capsule_results


def render_detections( classes,
    image: np.ndarray, results: List[DetectionNode], color, thickness
):
    def line(from_pt, to_pt):
        line_color = (30, 255, 255)
        from_pt, to_pt = tuple((int(x), int(y)) for x, y in (from_pt, to_pt))
        cv2.line(image, from_pt, to_pt, color, thickness=thickness, lineType=cv2.LINE_8)

    def text(coords, label):
        x, y = coords[0]
        pt = (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 100, 100)
        text_color_bg = color # (30, 255, 255)
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(image, pt, (x + text_w, y - text_h), text_color_bg, -1)
        cv2.putText(image, label, pt, font, font_scale, text_color, font_thickness)

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
        else:
            detection_confidence = 0
        label = f'{class_name}: {detection_confidence:.04f}'
        text(detection.coords, label)


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


def capsule_infer_add_args(parser) -> Tuple[Path, Optional[Path], List[Path]]:
    parser.add_argument(
        "--capsule",
        required=True,
        type=Path,
        help="The path to either an unpackaged or packaged capsule",
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
    return


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
    if args.capsule.is_dir():
        capsule_name = args.capsule.with_suffix(CAPSULE_EXTENSION).name
        unpackaged_capsule_path = args.capsule
        packaged_capsule_path = unpackaged_capsule_path.parent / capsule_name
        package_capsule(args.capsule, packaged_capsule_path)
    else:
        capsule_name = None
        unpackaged_capsule_path = None
        packaged_capsule_path = args.capsule
    return packaged_capsule_path, unpackaged_capsule_path, capsule_name


def main():
    parser = ArgumentParser(
        description="A helpful tool for running inference on a capsule."
    )

    capsule_infer_add_args(parser)

    args = parser.parse_args()

    image_paths = parse_images(args.images)

    packaged_capsule_path, unpackaged_capsule_path, capsule_name = parse_capsule_info(args)

    results = capsule_inference(packaged_capsule_path, unpackaged_capsule_path, image_paths, args.detection, 0, 0)

    print(results)


if __name__ == "__main__":
    main()
