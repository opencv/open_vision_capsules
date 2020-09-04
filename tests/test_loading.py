import sys
from pathlib import Path
from zipfile import ZipFile
from typing import Tuple

from vcap import BaseCapsule, package_capsule, CAPSULE_EXTENSION
from vcap.loading.capsule_loading import capsule_module_name
from vcap.testing.capsule_loading import load_capsule_with_one_device


def test_load_capsule_from_memory():
    """Test that a capsule can be loaded from memory, without any source code
    available.
    """
    capsule_path = Path("vcap", "examples", "classifier_gait_example")
    packaged_capsule_path = (capsule_path
                             .with_name(capsule_path.stem)
                             .with_suffix(CAPSULE_EXTENSION))
    package_capsule(capsule_path, packaged_capsule_path)

    capsule: BaseCapsule = load_capsule_with_one_device(
        packaged_capsule_path,
        from_memory=True)

    try:
        assert capsule.name == "classifier_gait_example"
    finally:
        capsule.close()


def test_load_duplicate_capsule():
    """Tests that the capsule's modules are not re-imported when a duplicate
    capsule is loaded.

    This caching behavior isn't necessarily critical for correct behavior, but
    this test ensures our understanding of Python's import system holds.
    """
    capsule_path = Path("vcap", "examples", "classifier_gait_example")
    capsule_copy_1, packaged_capsule_path = _package_and_load_capsule(
        capsule_path)

    # Modify the capsule's module
    module_copy_1 = _get_capsule_module(packaged_capsule_path)
    assert module_copy_1 is not None
    module_copy_1.some_attribute = "Hello"

    # Load the same capsule again
    load_capsule_with_one_device(
        packaged_capsule_path,
        inference_mode=False)

    # Test that the module modification still exists after the re-load
    module_copy_2 = _get_capsule_module(packaged_capsule_path)
    assert module_copy_2 is not None
    assert module_copy_2.some_attribute == "Hello"


def test_load_modified_capsule():
    """Tests that the capsule's modules are re-imported when a modified version
    of a capsule is loaded.

    This ensures that code changes in the revised capsule will be reflected
    when in use.
    """
    capsule_path = Path("vcap", "examples", "classifier_gait_example")
    capsule_revision_1, packaged_capsule_path = _package_and_load_capsule(
        capsule_path)

    # Modify the first revision capsule's module
    module_revision_1 = _get_capsule_module(packaged_capsule_path)
    assert module_revision_1 is not None
    module_revision_1.some_attribute = "Hello"

    # Modify the capsule file
    with ZipFile(packaged_capsule_path, "a") as new_capsule_zip:
        new_capsule_zip.writestr("random_file.txt",
                                 "I'm here to mess up your capsule!")

    # Upload a second revision of the capsule
    load_capsule_with_one_device(
        packaged_capsule_path,
        inference_mode=False)

    # Test that the module modification from the first revision is not
    # reflected in the second revision
    module_revision_2 = _get_capsule_module(packaged_capsule_path)
    assert module_revision_2 is not None
    assert not hasattr(module_revision_2, "some_attribute")


_BLOCK_SIZE = 1024000
"""The block size to read files at. Chosen from this answer:
https://stackoverflow.com/a/3673731
"""


def _get_capsule_module(path: Path):
    module_name = capsule_module_name(path.read_bytes())
    return sys.modules[module_name]


def _package_and_load_capsule(path: Path) -> Tuple[BaseCapsule, Path]:
    packaged_capsule_path = (path
                             .with_name(path.stem)
                             .with_suffix(CAPSULE_EXTENSION))
    package_capsule(path, packaged_capsule_path)

    capsule: BaseCapsule = load_capsule_with_one_device(
        packaged_capsule_path,
        # These tests are just for loading, not running inference
        inference_mode=False)

    return capsule, packaged_capsule_path
