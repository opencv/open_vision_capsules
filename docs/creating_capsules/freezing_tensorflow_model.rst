###########################
Freezing a TensorFlow Model
###########################

This tutorial will guide you through preparing a TensorFlow object detector
model for encapsulation. This tutorial assumes you have a model trained using
the TensorFlow Object Detection API.

Set up the TF Object Detection API
----------------------------------

First, set up the Tensorflow Object Detection API on your machine by cloning
``github.com/tensorflow/models`` and following the object detection API
installation instructions. Make sure the tests pass before continuing-
otherwise, you might have forgotten to set up certain environment variables!

Freezing the Model
------------------

Now, you can simply freeze the model to get the ``frozen_inference_graph.pb``.
To do that, run ``models/research/object_detection/export_inference_graph.py``
script inside of your downloaded model directory. To do this, you need the
``model.config`` file and a checkpoint file with the weights you want to use.

.. code-block:: bash

   python PATH/TO/export_inference_graph.py \
       --input_type image_tensor \
       --pipeline_config_path model.config \
       --trained_checkpoint_prefix model_weights/model.ckpt \
       --output_directory .

There should now be a frozen_inference_graph.pb in the current directory. This
is the model file that has been optimized for inference, and can be used inside
a capsule.
