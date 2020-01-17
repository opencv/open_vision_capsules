import pytest

from vcap.testing import verify_all_threads_closed


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
