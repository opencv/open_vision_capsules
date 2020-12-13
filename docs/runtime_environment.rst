###################
Runtime Environment
###################

Loading
=======

When a capsule is loaded, the ``capsule.py`` file is imported as a module and
an instance of the ``Capsule`` class defined in that module is instantiated.
Then, for each compatible device, an instance of the capsule's ``Backend``
class is created using the provided ``backend_loader`` function.

Importing
=========

Capsules have access to a number of helpful libraries, including:

- The entirety of the Python standard library
- Numpy (``import numpy``)
- OpenCV (``import cv2``)
- Tensorflow (``import tensorflow``)
- Scikit Learn (``import sklearn``)
- OpenVino (``import openvino``)

Applications may provide more libraries in addition to these. Please see that
application's documentation for more information.

Importing From Other Files in the Capsule
-----------------------------------------

In order to allow for more complex capsules that have code reuse within them,
capsules may consist of multiple Python files. These files are made available
through relative imports.

For example, with the following directory structure:

.. code-block::

   capsule_dir/
   ├── capsule.py
   ├── backend.py
   └── utils/
       ├── img_utils.py
       └── ml_utils.py

The ``capsule.py`` file may import the other Python files like so:

.. code-block:: python

   from . import backend
   from .utils import img_utils, ml_utils

Note that non-relative imports to these files will `not` work:

.. code-block:: python

   import backend
   from utils import img_utils, ml_utils

Limiting GPU memory Growth
-----------------------------------------

By default, OpenVisionCapsules maps all available memory of all visible CUDA configured GPUs.
To prevent this, use the following Environment flag while using Tensorflow.

.. code-block:: python

   TF_FORCE_GPU_ALLOW_GROWTH=True

- This Environment variable is only applicable to Tensorflow.

For proper reference, visit Tensorflow: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
