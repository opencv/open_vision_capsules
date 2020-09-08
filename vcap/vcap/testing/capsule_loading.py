from pathlib import Path

import mock

from vcap import BaseCapsule
from vcap.loading.capsule_loading import load_capsule_from_bytes, load_capsule


def load_capsule_with_one_device(packaged_capsule_path: Path,
                                 from_memory: bool = False,
                                 inference_mode: bool = True) -> BaseCapsule:
    """
    Load the capsule, but patch out the DeviceMapper so that it never returns
    multiple devices.

    Essentially, this disables capsules from loading backends onto multiple
    devices, for example GPUs. This is for the purpose of speeding up test
    setup and teardown.
    """

    def mock_init(self, filter_func):
        # Don't call the superclass init so that self.filter_func isn't
        # overridden
        self.filter_func = lambda all_devices: [filter_func(all_devices)[0]]

    with mock.patch('vcap.device_mapping.DeviceMapper.__init__',
                    mock_init):
        if from_memory:
            data = packaged_capsule_path.read_bytes()
            capsule: BaseCapsule = load_capsule_from_bytes(
                data=data,
                inference_mode=inference_mode)
        else:
            capsule: BaseCapsule = load_capsule(packaged_capsule_path,
                                                inference_mode=inference_mode)

    return capsule
