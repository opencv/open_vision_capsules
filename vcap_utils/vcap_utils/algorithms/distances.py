import numpy as np


def cosine_distance(encoding_to_compare: np.ndarray, encodings: np.ndarray,
                    is_normalized: bool = False):
    """Return the cosine distance of 'encoding_to_compare' to every encoding
    in 'encodings'"""
    if not is_normalized:
        # normalize arrays to unit length vectors (length 1)
        e_normed = np.linalg.norm(encodings, axis=1, keepdims=True)
        encodings = np.asarray(encodings) / e_normed

        etc_normed = np.linalg.norm(encoding_to_compare, axis=0, keepdims=True)
        encoding_to_compare = np.asarray(encoding_to_compare) / etc_normed
    return 1. - np.dot(encodings, encoding_to_compare.T)


def euclidian_distance(encoding_to_compare: np.ndarray, encodings: np.ndarray):
    """
    Given a list of face encodings, compare them to a known face encoding and
    get a euclidean distance for each comparison face. The distance tells you
    how similar the faces are.

    :param encoding_to_compare: A face encoding to compare against
    :param encodings: List of face encodings to compare
    :return: A numpy ndarray with the distance for each face in the same order
             as the 'faces' array
    """
    if len(encodings) == 0:
        return np.empty(0)
    assert encodings.shape[-1] == encoding_to_compare.shape[-1], \
        "Encoding shapes are incorrect!"
    return np.linalg.norm(encodings - encoding_to_compare, axis=1)
