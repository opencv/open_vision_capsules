import logging
import random
from abc import ABC, abstractmethod
from threading import RLock
from typing import Dict, List, Optional

import numpy as np

from .backend import BaseBackend
from .caching import cache
from .device_mapping import DeviceMapper, get_all_devices
from .node_description import DETECTION_NODE_TYPE, NodeDescription
from .options import Option, OPTION_TYPE
from .stream_state import BaseStreamState


class BaseCapsule(ABC):
    """An abstract base class that all capsules must subclass. Defines the
    interface that capsules are expected to implement.

    A capsule is a zipped and encrypted collection of files and configuration
    information. It represents a single algorithm with a specific set
    of functionality, like an inception_resnet_v2 behavior classification
    model.

    A class that subclasses from this class is expected to be defined in a
    capsule.py file in a capsule.
    """

    stream_state = BaseStreamState
    """This attribute is the basic 'stream_state object that is initialized
    for every new stream that a capsule is run on, and de-initialized when that
    stream goes away. It is intended to be overridden by capsules that have 
    stateful operations across a single stream."""

    backends: Optional[List[BaseBackend]] = None
    """A list of the backends the capsule has initialized"""

    backends_lock = RLock()
    """Lock the list of backends whenever it's being accessed"""

    def __init__(self, capsule_files: Dict[str, bytes], inference_mode=True):
        """Load a capsule file.

        :param capsule_files: A dict of {"file_name": FILE_BYTES} of the files
            that were found and loaded in the capsule
        :param inference_mode: If True, the model will be loaded and the
            backends will start for it. If False, the capsule will never be
            able to run inference, but it will still have it's various readable
            attributes.
        """

        if inference_mode:
            # We use type(self) to avoid having capsules need to put
            # staticmethod on each of their backend loaders.
            # self.backend_loader is a function not a method because we usually
            # use the following system for creating subclasses:
            #
            # >>> class CoolCapsule(BaseCapsule):
            # ...     ...
            # ...     backend_loader = lambda: some_func(...)
            #
            # This prevents self from being passed as the first (and wrong)
            # argument. We may want to change this in the future.
            load_backend = lambda device: type(self).backend_loader(
                capsule_files=capsule_files,
                device=device)

            # Initialize the backends on the capsules requested devices
            # for example: GPU:0, GPU:1, CPU:0, MYRIAD, HDDL
            all_devices = get_all_devices()
            devices_to_load_to = self.device_mapper.filter_func(all_devices)

            # Throw an error if the capsule couldn't load
            if not devices_to_load_to:
                message = f"The capsule was not able to find any valid " \
                          f"devices to initialize on! The devices " \
                          f"discovered on this machine are: {all_devices}."
                raise EnvironmentError(message)

            logging.info(f"Loading capsule {self.name} "
                         f"onto devices {', '.join(devices_to_load_to)}")
            self.backends = list(map(load_backend, devices_to_load_to))

        # Keep a dictionary for keeping track of different StreamState objs
        self._stream_states: Dict[int, BaseStreamState] = {}
        """{stream_id: BaseStreamState}"""

        self._stream_state_lock = RLock()

    def __repr__(self):
        rep = f"Capsule(name={self.name}, " \
              f"input={self.input_type}, " \
              f"output={self.output_type})"
        return rep

    def __del__(self):
        self.close()

    def process_frame(self,
                      frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) \
            -> DETECTION_NODE_TYPE:
        """Find the Backend that has an oven with the least amount of images
        in the pipeline, and run with that Backend. In multi-GPU scenarios,
        this is the logic that allows even usage across all GPUs."""
        with self.backends_lock:
            # BaseBackend.oven.total_imgs_in_pipeline only represents the
            # images in a single capsule's pipeline, not all the capsules.
            # This helps distribute the load of multiple capsules.
            #
            # Note the inplace shuffle, but this should be okay, esp. w/ lock
            random.shuffle(self.backends)
            laziest_backend: BaseBackend \
                = min(self.backends,
                      key=lambda backend: backend.workload)

        return laziest_backend.process_frame(
            frame, detection_node, options, state)

    def get_state(self, stream_id: int) -> BaseStreamState:
        """Get or create the StreamState for this stream_id and return it."""
        with self._stream_state_lock:
            # Adds stream state if stream_id is not already a key
            return self._stream_states.setdefault(stream_id,
                                                  self.stream_state())

    def clean_up(self, stream_id: int) -> None:
        """Dereference the StreamState that relates to this stream_id"""
        with self._stream_state_lock:
            if stream_id in self._stream_states:
                del self._stream_states[stream_id]

    def close(self) -> None:
        """This method MUST close the oven first, then de-initialize the
        backend so as to clear up memory.
        """
        if self.backends is not None:
            with self.backends_lock:
                for backend in self.backends:
                    backend.close()
                self.backends = None

    @property
    @cache
    def capability(self) -> NodeDescription:
        """Returns a NodeDescription showing the difference between the input
        and output NodeDescriptions for this capsule. This tells the caller
        what service this capsule provides. For instance, a capsule that
        classifies a person's gait would return the following:

        >>> NodeDescription(
        ...     size=NodeDescription.Size.ALL,
        ...     attributes={"Gait": ["walking", "running"]})

        The fact that the capsule's input is a person is not shown in this
        value because person detection is not something this capsule provides.
        """
        return self.input_type.difference(self.output_type)

    @property
    @cache
    def default_options(self):
        """Return a dict of key->value where the value is the default value
        for this capsules options"""
        return {key: val.default for key, val in self.options.items()}

    # These things are to be filled in by a subclassing capsule
    @property
    def name(self) -> str:
        """A name to uniquely identify the capsule."""
        raise NotImplementedError

    @property
    def version(self) -> int:
        """The version of the capsule.

        When should you bump the version of a capsule? When:
            - You've changed the usage of existing capsule options
            - You've changed the model or algorithm
            - You've changed the input/output node descriptions

        When shouldn't you bump the version of a capsule? When:
            - You only did code restructuring
            - You've updated the capsule to work with a newer API version
            - You've added (but not removed or changed previous) capsule
              options

        In summary, the version is most useful for differentiating a capsule
        from its previous versions.
        """
        raise NotImplementedError

    @property
    def description(self) -> str:
        """A human-readable description of what the capsule does."""
        return ""

    @property
    def device_mapper(self) -> DeviceMapper:
        return DeviceMapper.map_to_all_gpus()

    @staticmethod
    @abstractmethod
    def backend_loader(capsule_files: Dict[str, bytes], device: str) \
            -> "BaseBackend":
        """Initializes the capsule and returns the initialized Backend.

        :param capsule_files: A dict of
            filename: file_bytes of the files within the capsule's root
                      directory.
        :param device: A device, in the format of "GPU:0", "CPU:2"
        The devices passed into the capsule is decided by the
        Capsule.device_mapper
        """
        raise NotImplementedError

    @property
    def input_type(self) -> NodeDescription:
        """Describes the type of node this capsule expects to work with."""
        raise NotImplementedError

    @property
    def output_type(self) -> NodeDescription:
        """Describes the type of node this capsule will output."""
        raise NotImplementedError

    @property
    def options(self) -> Dict[str, Option]:
        """A list of zero or more options that can be configured at runtime."""
        return {}
