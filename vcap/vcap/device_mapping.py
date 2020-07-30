import os
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
        """Intelligently load capsules onto available OpenVINO-compatible
        devices.

        Since support for OpenVINO devices is experimental, there is a
        temporary environment variable being added to whitelist devices
        specifically. This variable will be deprecated and removed after a
        short testing period.

        The device "CPU" is _always_ allowed and always loaded onto and cannot
        be excluded.

        Here are the cases:
        ['CPU:0', 'HDDL', ...] =>  ["MULTI:CPU,HDDL"]

        ['CPU:0'] => ["CPU"]
            Always load onto CPU.
        """

        def filter_func(devices):

            devices_by_priority = os.environ.get(
                "OPENVINO_DEVICE_PRIORITY",
                "CPU,HDDL").split(",")
            load_to_devices = []

            for device in devices_by_priority:
                for existing_device in devices:
                    if existing_device.lower().startswith(device.lower()):
                        load_to_devices.append(device)

            if len(load_to_devices) > 1:
                return ["MULTI:" + ",".join(load_to_devices)]
            else:
                return load_to_devices

        return DeviceMapper(filter_func=filter_func)
