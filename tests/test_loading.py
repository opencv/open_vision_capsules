from pathlib import Path

from vcap import BaseCapsule, package_capsule, CAPSULE_EXTENSION
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

    assert capsule.name == "classifier_gait_example"

    capsule.close()
