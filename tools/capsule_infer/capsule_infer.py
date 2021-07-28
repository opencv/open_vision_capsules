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


def main():
    packaged_capsule_path, unpackaged_capsule_path, image_paths = _parse_args()

    capsule = load_capsule(
        path=packaged_capsule_path, source_path=unpackaged_capsule_path
    )
    validate_capsule(capsule)

    for image_path in image_paths:
        print(f"Running inference on {image_path}")
        image = cv2.imread(str(image_path))

        results = capsule.process_frame(
            frame=image,
            detection_node=None,
            options=capsule.default_options,
            state=capsule.stream_state(),
        )

        print(f"Inference results:\n\t{results}")
        print("Press Space to Continue")
        render_detections(
            image=image, results=results, color=(255, 100, 100), thickness=2
        )
        cv2.imshow("Results", image)
        cv2.waitKey(0)


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
    if capsule.input_type.size is not NodeDescription.Size.NONE:
        print(
            "This capsule requires detections as input, which this script does not "
            "currently support. Please try a capusle that has an input_type=None."
        )
        exit(-1)


def _parse_args() -> Tuple[Path, Optional[Path], List[Path]]:
    parser = ArgumentParser(
        description="A helpful tool for running inference on a capsule."
    )
    parser.add_argument(
        "-c",
        "--capsule",
        required=True,
        type=Path,
        help="The path to either an unpackaged or packaged capsule",
    )
    parser.add_argument(
        "-i",
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to one or more images to run inference on. If the path is a "
        "directory, then *.png or *.jpg images in the directory will be used.",
    )
    args = parser.parse_args()

    images = []
    for path in args.images:
        if path.is_dir():
            images += list(path.glob("*.png"))
            images += list(path.glob("*.jpg"))
            images += list(path.glob("*.jpeg"))

            if len(images) == 0:
                print(f"No images were found in the directory {args.images}!")
                exit(-1)
        else:
            images.append(path)

    if args.capsule.is_dir():
        capsule_name = args.capsule.with_suffix(CAPSULE_EXTENSION).name
        unpackaged_capsule_path = args.capsule
        packaged_capsule_path = unpackaged_capsule_path.parent / capsule_name
        package_capsule(args.capsule, packaged_capsule_path)
    else:
        unpackaged_capsule_path = None
        packaged_capsule_path = args.capsule

    return packaged_capsule_path, unpackaged_capsule_path, images


if __name__ == "__main__":
    main()
