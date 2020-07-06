import logging
from threading import RLock
from typing import Callable, List

import tensorflow as tf
from tensorflow.python.client import device_lib

_devices = None
_devices_lock = RLock()


def get_all_devices() -> List[str]:
    """Returns a list of devices in the computer. Example:
    ['CPU:0', 'XLA_GPU:0', 'XLA_CPU:0', 'GPU:0', 'GPU:1', 'GPU:2', 'GPU:3']
    """
    global _devices, _devices_lock

    # Initialize the device list if necessary
    with _devices_lock:
        if _devices is None:
            # We set these config options on the session because otherwise list
            # local_devices allocates all of the available GPU memory to the
            # computer
            #
            # Note that this config has the side effect of being set for all
            # future sessions#
            #
            # TODO: Use tf.config.list_physical_devices in TF 2.1
            # TODO: Remove the config when using TF_FORCE_GPU_ALLOW_GROWTH\
            #  environment variable
            _process_gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(gpu_options=_process_gpu_options)

            with tf.Session(config=config):
                all_devices = device_lib.list_local_devices()

            # Get the device names and remove duplicates, just in case...
            tf_discovered_devices = {
                d.name.replace("/device:", "")
                for d in all_devices
            }

            # Discover devices using OpenVINO
            try:
                from openvino.inference_engine import IECore
                ie = IECore()
                openvino_discovered_devices = {
                    d for d in ie.available_devices
                    if not d.lower().startswith("cpu")
                }
            except ModuleNotFoundError:
                logging.warning("OpenVINO library not found. "
                                "OpenVINO devices will not be discovered. ")
                openvino_discovered_devices = set()
            _devices = list(tf_discovered_devices
                            | openvino_discovered_devices)
    return _devices


class DeviceMapper:
    def __init__(self, filter_func: Callable[[List[str]], List[str]]):
        """The filter will take in a list of devices formatted as
        ["CPU:0", "CPU:1", "GPU:0", "GPU:1"], etc and output a filtered list of
        devices.
        """
        self.filter_func = filter_func

    @staticmethod
    def map_to_all_gpus(cpu_fallback=True) -> 'DeviceMapper':
        def filter_func(devices):
            gpu_devices = [d for d in devices if d.startswith("GPU:")]
            if not gpu_devices and cpu_fallback:
                return ["CPU:0"]
            return gpu_devices

        return DeviceMapper(filter_func=filter_func)

    @staticmethod
    def map_to_single_cpu() -> 'DeviceMapper':
        def filter_func(devices):
            return [next(d for d in devices if d.startswith("CPU:"))]

        return DeviceMapper(filter_func=filter_func)

    @staticmethod
    def map_to_openvino_devices():
        """Intelligently load capsules onto available OpenVINO compatible
        devices.
        Here are the cases:

        ['CPU:0', 'MYRIAD'] => ['CPU:0']
            Because MYRIAD devices don't support multiple capsules,
            loading directly onto them is disabled.
        ['CPU:0', 'MYRIAD', 'HDDL'] =>  ['HDDL']
            Because there is a MYRIAD device available, and HDDL is showing up,
            that means that the HDDL drivers are installed and there is a
            valid MYRIAD device to load onto. Since HDDL supports loading
            multiple capsules, it is thus selected.
        ['CPU:0', 'HDDL'] => ['CPU:0']
            Because there is no MYRIAD device available, loading onto HDDL
            doesn't make any sense.
        ['CPU:0'] => ['CPU:0']
            Always load onto CPU.
        """

        def filter_func(devices):
            myriad_devices = [d for d in devices
                              if d.lower().startswith("myriad")]
            hddl_device = [d for d in devices
                           if d.lower().startswith("hddl")]
            if myriad_devices and hddl_device:
                # Since there are myriad devices available and the HDDL driver
                # is installed, load onto CPU and HDDL
                return ["CPU:0"] + hddl_device
            if not myriad_devices and hddl_device:
                logging.warning("HDDL drivers are correctly configured, but "
                                "no MYRIAD devices were found to load onto. "
                                "Loading onto CPU only.")
            if myriad_devices and not hddl_device:
                logging.warning("Myriad device found, but no HDDL drivers "
                                "are installed on the host computer. "
                                "Loading onto CPU only.")
            return ["CPU:0"]

        return DeviceMapper(filter_func=filter_func)
