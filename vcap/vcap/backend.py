import abc
from concurrent.futures import Future
from typing import Any, Dict, List

import numpy as np

from vcap.batch_executor import BatchExecutor
from vcap.node_description import DETECTION_NODE_TYPE
from vcap.options import OPTION_TYPE
from vcap.stream_state import BaseStreamState


class BaseBackend(abc.ABC):
    """An object that provides low-level prediction functionality for batches
    of frames.
    """

    def __init__(self):
        self._batch_executor = BatchExecutor(self.batch_predict)

    def send_to_batch(self, input_data: Any) -> Future:
        """Sends the given object to the batch_predict method for processing.
        This call does not block. Instead, the result will be provided on the
        returned Future. The batch_predict method must be overridden on the
        backend this method is being called on.

        :param input_data: The input object to send to batch_predict
        :return: A Future where results will be stored
        """
        return self._batch_executor.submit(input_data)

    @property
    def workload(self) -> float:
        """Returns a unit representing the amount of 'work' being processed
        This value is comparable only by backends of the same capsule, and
        is intended to give the scheduler the ability to pick the least busy
        backend.
        """
        return self._batch_executor.total_imgs_in_pipeline

    @abc.abstractmethod
    def process_frame(self,
                      frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) \
            -> DETECTION_NODE_TYPE:
        """A method that does the pre-processing, inference, and postprocessing
        work for a frame.

        If the capsule uses an algorithm that benefits from batching,
        this method may call ``self.send_to_batch``, which will asynchronously
        send work out for batching. Doing so requires that the
        ``batch_predict`` method is overridden.

        :param frame: A numpy array representing a frame. It is of shape
            (height, width, num_channels) and the frames come in BGR order.
        :param detection_node: The detection_node type as specified by the
            ``input_type``
        :param options: A dictionary of key (string) value pairs. The key is
            the name of a capsule option, and the value is its configured value
            at the time of processing. Capsule options are specified using the
            ``options`` field in the Capsule class.
        :param state: This will be a StreamState object of the type specified
            by the ``stream_state`` attribute on the Capsule class. If no
            StreamState object was specified, a simple BaseStreamState object
            will be passed in. The StreamState will be the same object for all
            frames in the same video stream.
        """

    def batch_predict(self, input_data_list: List[Any]) -> List[Any]:
        """This method takes in a batch as input and provides a list of result
        objects of any type as output. What the result objects are will depend
        on the algorithm  being defined, but the number of prediction objects
        returned _must_ match the number of video frames provided as input.

        :param input_data_list: A list of objects. Whatever the model requires
            for each frame.
        """
        raise NotImplementedError(
            "Attempt to do batch prediction on a Backend that does not have "
            "the batch_predict method defined. Did you call send_to_batch on "
            "a backend that does not override batch_predict?")

    def close(self) -> None:
        """De-initializes the backend. This is called when the capsule is being
        unloaded. This method should be overridden by any Backend that needs
        to release resources or close other threads.

        The backend will stop receiving frames before this method is
        called, and will not receive frames again.
        """
        self._batch_executor.close()
