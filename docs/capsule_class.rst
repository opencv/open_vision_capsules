.. _`The Capsule Class`:

#################
The Capsule Class
#################

Introduction
============

The Capsule class provides information to the application about what a capsule
is and how it should be run. Every capsule defines a Capsule class that extends
BaseCapsule in its ``capsule.py`` file.

.. code-block:: python

   from vcap import (
       BaseCapsule,
       NodeDescription,
       options
   )

   class Capsule(BaseCapsule):
       name = "detector_person"
       version = 1
       stream_state = StreamState
       input_type = NodeDescription(size=NodeDescription.Size.NONE),
       output_type = NodeDescription(
           size=NodeDescription.Size.ALL,
           detections=["person"])
       backend_loader = backend_loader_func
       options = {
           "threshold": options.FloatOption(
               default=0.5,
               min_val=0.1,
               max_val=1.0)
       }

.. autoclass:: vcap.BaseCapsule
   :members:
   :exclude-members: backends_lock, backends, get_state, close, clean_up, process_frame
