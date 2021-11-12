from argparse import ArgumentParser
from pathlib import Path
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

    capsule_options = {'true threshold': true_threshold, 'false threshold': false_threshold}
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

        results = capsule.process_frame(
            frame=image,
            detection_node=detection_node,
            options=capsule_options,
            state=capsule.stream_state(),
        )

        print(f"Inference results: {detection_node}")

        capsule_results.append(detection_node)

        if not detection_required:
            render_detections(
                image=image, results=results, color=(255, 100, 100), thickness=2
            )

        cv2.imshow("Results", image)
        cv2.waitKey(1)

    return capsule_results


def render_detections(
    image: np.ndarray, results: List[DetectionNode], color, thickness
):
    def line(from_pt, to_pt):
        from_pt, to_pt = tuple((int(x), int(y)) for x, y in (from_pt, to_pt))
        cv2.line(image, from_pt, to_pt, color, thickness=thickness, lineType=cv2.LINE_8)

    for detection in results:
        if len(detection.coords) > 2:
            for from_pt, to_pt in zip(detection.coords[0:-1], detection.coords[1:]):
                line(from_pt, to_pt)

            # Close the region
            line(detection.coords[0], detection.coords[-1])

        elif len(detection.coords) == 2:
            line(detection.coords[0], detection.coords[1])
        else:
            print(f"Unsupported number of coordinates: {detection.coords}")


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


if __name__ == "__main__":
    main()
