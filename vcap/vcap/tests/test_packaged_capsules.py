from pathlib import Path

from vcap.testing.input_output_validation import perform_capsule_tests
from vcap.tests.dependencies import ALL_IMAGE_PATHS


def test_classifier_gait_example():
    """Verify that the gait classifier can be packaged and used."""
    perform_capsule_tests(
        Path("examples", "classifier_gait_example"),
        ALL_IMAGE_PATHS)


def test_detector_person_example():
    """Verify that the person detector can be packaged and used."""
    perform_capsule_tests(
        Path("examples", "detector_person_example"),
        ALL_IMAGE_PATHS)
