import gc
import logging
import random
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Dict, List, Union, Optional
from uuid import uuid4

import cv2
import mock
import numpy as np
import json

from vcap import (
    NodeDescription,
    DetectionNode,
    BaseCapsule,
    OPTION_TYPE,
    FloatOption,
    IntOption,
    BoolOption,
    EnumOption)
from vcap.loading.vcap_packaging import CAPSULE_EXTENSION, package_capsule
from vcap.testing import load_capsule_with_one_device
from vcap.testing.thread_validation import verify_all_threads_closed

NUM_STREAMS = 5
"""Number of random batches of images to send to the capsule"""

NUM_BATCH_CASES = 5
"""Number of batches of fuzzed inputs to perform on different images"""


def make_detection_node(frame_shape,
                        node_description: NodeDescription) -> DetectionNode:
    """Creates a fake detection node that describes the given node description.
    :param frame_shape: The shape of the frame in (height, width, channels)
    :param node_description: The description that the returned node must
        adhere to
    :return: A fake detection node that adheres to this description
    """
    height, width, _ = frame_shape
    attributes = {category: random.choice(possible_values)
                  for category, possible_values in
                  node_description.attributes.items()}
    extra_data = {data_key: 0.5129319283
                  for data_key in node_description.extra_data}
    detection_names = node_description.detections

    # Create random coordinates for this detection
    x1 = random.randint(0, width - 3)
    y1 = random.randint(0, height - 3)
    x2 = x1 + random.randint(0, width - x1 + 1) + 2
    y2 = y1 + random.randint(0, height - y1 + 1) + 2

    return DetectionNode(
        name=random.choice(detection_names) if len(detection_names) else "N/A",
        coords=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        attributes=attributes,
        encoding=np.zeros((128,)) if node_description.encoded else None,
        track_id=uuid4() if node_description.tracked else None,
        extra_data=extra_data)


def make_capsule_options(capsule: BaseCapsule) -> Dict[str, OPTION_TYPE]:
    """Create a random set of options that are valid within the specs that the
    capsule options allow"""
    options = {}
    for name, opt in capsule.options.items():
        possible_vals = [opt.default]
        if isinstance(opt, (FloatOption, IntOption)):
            min_max_vals = [opt.min_val, opt.max_val]
            # If min/max are unrestricted (None) remove them
            possible_vals += [val for val in min_max_vals if val is not None]

        elif isinstance(opt, BoolOption):
            possible_vals += [True, False]
        elif isinstance(opt, EnumOption):
            possible_vals += opt.choices
        else:
            raise ValueError(
                f"This test needs to be updated for the new capsule option "
                f"type: {opt}")
        options[name] = random.choice(list(set(possible_vals)))
    return options


def _run_inference_on_images(images: List[np.ndarray], capsule: BaseCapsule):
    """Run inference on a list of images"""
    # Make multiple parallel requests with different images and different
    # input DetectionNodes to the capsule.
    request_input = []

    with ThreadPoolExecutor(
            thread_name_prefix="InputOutputValidation ") \
            as executor:
        for stream_id, image in enumerate(images):
            if capsule.input_type.size == NodeDescription.Size.NONE:
                input_node = None

            elif capsule.input_type.size == NodeDescription.Size.SINGLE:
                input_node = make_detection_node(image.shape,
                                                 capsule.input_type)

            elif capsule.input_type.size == NodeDescription.Size.ALL:
                input_node = [make_detection_node(image.shape,
                                                  capsule.input_type)
                              for _ in range(random.randint(0, 5))]
            else:
                raise NotImplementedError(
                    "The capsule did not have a NodeDescription.Size that was "
                    "known!")

            options = make_capsule_options(capsule)

            future = executor.submit(
                capsule.process_frame,
                frame=image,
                detection_node=input_node,
                options=options,
                state=capsule.stream_state())

            request_input.append((future, input_node))

        for future, input_node in request_input:
            # Postprocess the results
            try:
                prediction = future.result(timeout=90)
            except futures.TimeoutError as e:
                raise TimeoutError("Wait for capsule process_frame result: TimeoutError!")

            # Verify that the capsule performed correctly

            if isinstance(prediction, DetectionNode):
                # Validate that what was returned by the capsule was valid
                assert (capsule.output_type.size
                        is NodeDescription.Size.SINGLE)
                output_nodes = [prediction]
            elif prediction is None and isinstance(input_node, DetectionNode):
                # If the capsule didn't output something, then it must have
                # modified the input node in-place. Validate the changes.
                assert (capsule.output_type.size
                        is NodeDescription.Size.SINGLE)
                output_nodes = [input_node]
            elif prediction is None and isinstance(input_node, list):
                # If this capsule accepts size ALL as input, then it must
                # have modified the detections within the input list
                assert capsule.input_type.size is NodeDescription.Size.ALL
                assert capsule.output_type.size is NodeDescription.Size.ALL
                output_nodes = input_node
            elif isinstance(prediction, list):
                # Validate that every detection node in the list is correct
                assert capsule.output_type.size is NodeDescription.Size.ALL
                output_nodes = prediction
            else:
                raise RuntimeError(f"Unknown prediction type: {prediction}")

            # Validate every output node against the capsules output_type
            for output_node in output_nodes:
                assert capsule.output_type.describes(output_node), \
                    ("Capsule failed to output a prediction that matches "
                     "the NodeDescription it had for it's output type. "
                     f"Prediction: {prediction}")

                # Assert the nodes "extra_data" attribute can be JSON encoded
                # without errors. Typically this can happen if there's a numpy
                # array carelessly left in the extra_data

                json.loads(json.dumps(output_node.extra_data))

                # If this capsule can encode things, verify that the backend
                # correctly implemented the "distance" function
                if (capsule.capability.encoded
                        and prediction is not None
                        and len(prediction) > 0):
                    # Get one of the predictions
                    pred = prediction[0]

                    # Measure the distance from an encoding to itself
                    # (should be 0)
                    distances = capsule.backends[0].distances(
                        pred.encoding, np.array([pred.encoding]))
                    assert len(distances) == 1
                    assert distances[0] == 0, \
                        ("This assertion can be removed in the case that "
                         "there is some new distance function where two "
                         "encodings that are equal no longer have a distance "
                         "of 0. Until that case exists, keep this assertion.")


def _test_capsule_input_output(capsule, image_paths):
    """Creates necessary resources to load a capsule, then runs the
    necessary batches to test the capsule. """

    # Since we use fuzzing for some tests, we might as well set a seed
    random.seed(capsule.name)

    # Tests multiple cases for multiple images as input
    loaded_images = [cv2.imread(str(path)) for path in image_paths]
    try:
        # Run the test cases
        for _ in range(NUM_BATCH_CASES):
            images = random.choices(loaded_images, k=NUM_STREAMS)
            random.shuffle(images)
            _run_inference_on_images(images, capsule)
    except Exception as e:
        # If an error happens, clean up the threads. Otherwise, we will let
        # the rest of the test (garbage collection) make sure the capsule gets
        # closed properly.
        capsule.close()
        raise e


def perform_capsule_tests(unpackaged_capsule_dir: Union[Path, str],
                          image_paths: List[Union[Path, str]],
                          allowable_threads: Optional[List[str]] = None):
    """This tests several many things:
    1) The capsule can handle batches of images
    2) The capsule can handle different capsule option combinations
    3) The capsule handles the input type that it specifies
    4) The capsule does not keep references to itself, thus causing it not to
       be garbage collected
    """
    unpackaged_capsule_dir = Path(unpackaged_capsule_dir)

    logging.info(f"Testing capsule name: {unpackaged_capsule_dir.name}")
    packaged_capsule_path = (unpackaged_capsule_dir
                             .with_name(unpackaged_capsule_dir.stem)
                             .with_suffix(CAPSULE_EXTENSION))
    package_capsule(unpackaged_capsule_dir, packaged_capsule_path)

    # Start the test with no threads running and garbage collection clean,
    # so that our __del__ mock to BaseCapsule doesn't get called from
    # other objects that are yet to be garbage collected
    verify_all_threads_closed(allowable_threads)
    gc.collect()

    with mock.patch.object(BaseCapsule, '__del__') as patched_del:
        capsule = load_capsule_with_one_device(packaged_capsule_path)
        _test_capsule_input_output(capsule, image_paths)

        # Since the __del__ method was wrapped, mock will not actually call the
        # underlying method when the object gets garbage collected- thus the
        # capsule will never get 'close' called on it, thus the BatchExecutor
        # threads will not close. To fix that, we call 'close' now, and check
        # that __del__ WOULD have been called.
        capsule.close()

        # Check that capsule.close() successfully killed any worker threads
        verify_all_threads_closed(allowable_threads)

        referrers = gc.get_referrers(capsule)

        if sys.version_info >= (3, 7):
            # After Python 3.7, gc.get_referrers no longer includes stack frames in its
            # output. Therefore, the reference held in this function is not counted.
            # See: https://bugs.python.org/issue34608
            expected_referrers = 0
        else:
            expected_referrers = 1

        assert len(referrers) == expected_referrers, \
            "No one else should have a reference to this capsule anymore! " \
            f"There were {len(referrers)} references from: {referrers}. " \
            f"Capsule in question: {capsule}"
        assert patched_del.call_count == 0, \
            "__del__ should only be called when the capsule is garbage " \
            "collected!"

        # Get rid of the last reference to the capsule
        del capsule

        gc.collect()

        assert patched_del.call_count >= 1, \
            "The capsule didn't get garbage collected! The capsule must be " \
            "keeping a reference to itself."

    # Sanity check that the capsules __del__ didn't spawn threads
    verify_all_threads_closed(allowable_threads)
