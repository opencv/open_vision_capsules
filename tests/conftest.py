from pathlib import Path
import urllib.request

import pytest

from vcap.testing import verify_all_threads_closed

_TEST_IMAGE_DIR = Path("tests/test_images")
_S3_BASE_URL = "https://open-vision-capsules.s3-us-west-1.amazonaws.com" \
               "/test-dependencies/{}/{}"


@pytest.fixture(autouse=True, scope="session")
def model_dependencies():
    dependencies = {
        "models": {
            "classification_gait_model.pb":
                Path("vcap/examples/classifier_gait_example/"),
            "ssd_mobilenet_v1_coco.pb":
                Path("vcap/examples/detector_person_example")
        },
        "images": [
            "no_people.jpg",
            "one_person.jpg",
            "two_people.jpg"
        ]
    }

    # Get models
    for model_name, model_path in dependencies["models"].items():
        filepath = model_path / model_name
        if not filepath.exists():
            s3_url = _S3_BASE_URL.format("models", model_name)
            urllib.request.urlretrieve(s3_url, filepath)

    # Get images
    _TEST_IMAGE_DIR.mkdir(exist_ok=True)
    for image_name in dependencies["images"]:
        filepath = _TEST_IMAGE_DIR / image_name
        if not filepath.exists():
            s3_url = _S3_BASE_URL.format("images", image_name)
            urllib.request.urlretrieve(s3_url, filepath)


@pytest.fixture(autouse=True, scope="session")
def all_test_setup_teardown():
    """Runs before and after the entire testing session"""

    yield

    # Any teardown for all tests goes here
    verify_all_threads_closed()


@pytest.fixture(autouse=True)
def every_test_setup_teardown():
    """Runs before and after each test"""
    # Any setup for each test goes here
    yield

    # Any teardown for each tests goes here
    verify_all_threads_closed()
