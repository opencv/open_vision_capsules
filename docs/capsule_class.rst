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

Fields
======

name
----

.. code-block:: python

   name: str

The name of the capsule. This value uniquely defines the capsule and cannot be
shared by other capsules. By convention, the capsule's name is prefixed by some
short description of the role it plays ("detector", "recognizer", etc) followed
by the kind of data it relates to ("person", "face", etc) and, if necessary,
some differentiating factor ("fast", "close_up", "retail", etc).


version
-------

.. code-block:: python

   version: int


The incremental version of the capsule. It's recommended to change the version
on any change of capsule options, algorithms or models.

device_mapper
-------------

.. code-block:: python

   device_mapper: DeviceMapper

A device mapper contains a single field, ``filter_func``, which is a function
that takes in a list of all available device strings and returns a list of
device strings that are compatible with this capsule.

stream_state
------------

.. code-block:: python

   stream_state: Type[BaseStreamState]

(Optional) A class that the application will make instances of for each video
stream. See the :ref:`Stream State` section for more information.

input_type
----------

.. code-block:: python

   input_type: NodeDescription

Describes the types of DetectionNodes that this capsule takes in as input. See
the :ref:`Inputs and Outputs` section for more information.

output_type
-----------

.. code-block:: python

   output_type: NodeDescription

Describes the types of DetectionNodes that this capsule produces as output. See
the :ref:`Inputs and Outputs` section for more information.

backend_loader
--------------

.. code-block:: python

   backend_loader: Callable[[dict, str], BaseBackend]

A function that creates a backend for this capsule. Takes the following as
arguments:

- ``capsule_files``: Provides access to all files in the capsule. The keys are
  file names and the values are ``bytes``.
- ``device``: A string specifying the device that this backend should use. For
  example, the first GPU device is specified as "GPU:0".

The function must return a class that subclasses BaseBackend.
See the :ref:`Backends` section for more information.

options
-------

.. code-block:: python

   options: dict[str, Option]

A dict where the key is an option name and the value is an Option object. See
the :ref:`Options` section for more information.
