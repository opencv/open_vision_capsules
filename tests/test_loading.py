from pathlib import Path

from vcap import BaseCapsule
from vcap.testing.capsule_loading import load_capsule_with_one_device


def test_load_capsule_from_memory():
    """Test that a capsule can be loaded from memory, without any source code
    available.
    """
    capsule: BaseCapsule = load_capsule_with_one_device(
        Path("vcap", "examples", "classifier_gait_example"),
        from_memory=True)

    assert capsule.name == "classifier_gait_example"

    capsule.close()
