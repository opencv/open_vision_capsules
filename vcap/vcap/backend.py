import abc
from queue import Queue
from typing import Any, Dict, List, Union

import numpy as np

from vcap.node_description import DETECTION_NODE_TYPE
from vcap.ovens import Oven
from vcap.options import OPTION_TYPE
from vcap.stream_state import BaseStreamState


class BaseBackend(abc.ABC):
    """An object that provides low-level prediction functionality for batches
    of frames.
    """

    def __init__(self):
        self._oven = Oven(self.batch_predict)

    def send_to_batch(self, input_data: Any) -> Queue:
        """Sends the given object to the batch_predict method for processing.
        This call does not block. Instead, the result will be provided on the
        returned queue. The batch_predict method must be overridden on the
        backend this method is being called on.

        :param input_data: The input object to send to batch_predict
        :return: A queue where results will be stored
        """
        return self._oven.submit(input_data)

    @property
    def workload(self) -> Union[float, int]:
        """Returns a unit representing the amount of 'work' being processed
        This value is comparable only by backends of the same capsule, and
        is intended to give the scheduler the ability to pick the least busy
        backend.
        """
        return self._oven.total_imgs_in_pipeline

    @abc.abstractmethod
    def process_frame(self,
                      frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) \
            -> DETECTION_NODE_TYPE:
        """A method that does the preprocessing, inference, and postprocessing
        work for a frame. It has the ability to call self.send_to_batch(frame)
        to send work for batching (eventually run in batch_predict)."""

    def batch_predict(self, input_data_list: List[Any]) -> List[Any]:
        """Runs prediction on a batch of frames (or objects). This method must
        be overridden for capsules that use send_to_batch.

        :param input_data_list: A list of objects. Whatever the model requires
                                for each frame.
        """
        raise NotImplementedError(
            "Attempt to do batch prediction on a Backend that does not have "
            "the batch_predict method defined. Did you call send_to_batch on "
            "a backend that does not override batch_predict?")

    @abc.abstractmethod
    def close(self) -> None:
        """De-initializes the backend."""
        self._oven.close()
