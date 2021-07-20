import json

import tensorflow as tf


def parse_tf_model_bytes(model_bytes,
                         device: str = None,
                         session_config: tf.compat.v1.ConfigProto = None):
    """

    :param model_bytes: The bytes of the model to load
    :param device_id: The device that this model should be loaded onto
    :param session_config: Configuration options for multiple monitors
    :return:
    """

    # Load the model as a graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # Load a (frozen) Tensorflow model from memory
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(model_bytes)

        with tf.device(device):
            tf.import_graph_def(graph_def,
                                input_map=None,
                                return_elements=None,
                                producer_op_list=None,
                                name='')

    if session_config is None:
        session_config = tf.compat.v1.ConfigProto()

    if device is not None:
        # allow_soft_placement lets us remap GPU only ops to GPU, and doesn't
        # crash for non-gpu only ops (it will place those on CPU, instead)
        session_config.allow_soft_placement = True

    # Create a session for later use
    persistent_sess = tf.compat.v1.Session(graph=detection_graph,
                                           config=session_config)

    return detection_graph, persistent_sess


def parse_dataset_metadata_bytes(dataset_metadata_bytes):
    """ Loads the label_map from the dataset_metadata bytes """
    dataset_metadata = json.loads(dataset_metadata_bytes.decode("utf-8"))

    # Convert the keys of the label map to integer values, instead of strings
    label_map = dataset_metadata["label_map"]
    label_map = {int(k): v for k, v in label_map.items()}
    return label_map
