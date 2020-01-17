import abc

import numpy as np

from vcap.backend import BaseBackend


class BaseEncoderBackend(BaseBackend):
    # TODO: Remove the concept of Backends needing a 'distances' function
    @abc.abstractmethod
    def distances(self, encoding_to_compare: np.ndarray,
                  encodings: np.ndarray) -> np.ndarray:
        """
        :param encoding_to_compare: An object encoding to compare
        :param encodings: List of object encodings to compare against
        :return: A numpy ndarray with the distance for each face in the same
        order as the 'encodings_to_compare' array
        """
        pass
