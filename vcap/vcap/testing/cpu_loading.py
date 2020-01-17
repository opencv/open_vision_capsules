from pathlib import Path

import mock

from vcap import BaseCapsule
from vcap.loading.capsule_loading import load_capsule


def one_gpu_filter(all_devices):
    """Always return at most 1 GPU but if the machine has no GPUs then load
    the model onto CPU:0. This is nice for testing because loading onto
    multiple GPUs slows down tests significantly due to load time.
    """
    gpus = [d for d in all_devices if d.split(':')[0] == "GPU"]
    if len(gpus) == 0:
        return ['CPU:0']
    return [gpus[0]]


def load_capsule_with_one_gpu(packaged_capsule_path: Path) -> BaseCapsule:
    """
    Load the capsule, but patch out the DeviceMapper so that it ALWAYS returns
    a device. If there are no GPU's found, it will return a CPU.

    Essentially, load any capsule onto a single GPU, or CPU if the machine has
    no GPU.
    """

    def mock_init(self, *_args, **_kwargs):
        # Don't call the superclass init so that self.filter_func isn't
        # overridden
        self.filter_func = one_gpu_filter

    with mock.patch('vcap.device_mapping.DeviceMapper.__init__',
                    mock_init):
        capsule: BaseCapsule = load_capsule(path=packaged_capsule_path)

    return capsule
