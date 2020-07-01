.. _`Backends`:

########
Backends
########

Introduction
============

A backend is what provides the low-level analysis on a video frame. For machine
learning, this is the place where the frame would be fed into the model and the
results would be returned. Every capsule must define a backend class that
subclasses the BaseBackend class. 

The application will create an instance of the backend class for each device
string returned by the capsule's device mapper. 

Required Methods
================

All backends must subclass the BaseBackend abstract base class, meaning that
there are a couple methods that the backend must implement.

process_frame
-------------

.. code-block:: python

   DETECTION_NODE_TYPE = Union[None, DetectionNode, List[DetectionNode]]

   def process_frame(
           self,
           frame: np.ndarray,
           detection_node: DETECTION_NODE_TYPE,
           options: Dict[str, OPTION_TYPE],
           state: BaseStreamState) -> DETECTION_NODE_TYPE:
       ...

A method that does the pre-processing, inference, and postprocessing
work for a frame.

If the capsule uses an algorithm that benefits from batching,
this method may call ``self.send_to_batch``, which will asynchronously send work
out for batching. Doing so requires that the ``batch_predict`` method is
overridden. See the section on batching methods for more information.

Arguments:

- ``frame`` A numpy array representing a frame. It is of shape (height, width,
  channel) and the frames come in BGR order.
- ``detection_node`` The detection_node type as specified by the ``input_type``
- ``options`` A dictionary of key (string) value pairs. The key is the name of
  a capsule option, and the value is its configured value at the time of
  processing. Capsule options are specified using the ``options`` field in the
  Capsule class.
- ``state`` This will be a StreamState object of the type specified by the
  ``stream_state`` attribute on the Capsule class. If no StreamState object was
  specified, a simple BaseStreamState object will be passed in. The StreamState
  will be the same object for all frames in the same video stream.

close
-----

.. code-block:: python

   def close(self):
       ...

The ``close`` method de-initializes the backend. Once this method is called,
the backend will no longer be in use.

Batching Methods
================

Batching refers to the process of collecting more than one video frame into a
"batch" and sending them all out for processing at once. Certain algorithms see
performance improvements when batching is used, because doing so decreases the
amount of round-trips the video frames take between devices.

If you wish to use batching in your capsule, you may call the ``send_to_batch``
method in ``process_frame`` instead of doing analysis in that method directly.
The ``send_to_batch`` method collects one or more input objects into a list and
routinely calls your backend's ``batch_predict`` method with this list. As a
result, users of ``send_to_batch`` must override the ``batch_predict`` method
in addition to the other required methods.

The ``send_to_batch`` method is asynchronous. Instead of immediately returning
analysis results, it returns a ``queue.Queue`` where the result will be provided.
Simple batching capsules may call ``send_to_batch``, then immediately call
``get`` to block for the result.

.. code-block:: python

   result = self.send_to_batch(frame).get()

An argument of any type may be provided to ``send_to_batch``, as the argument
will be passed in a list to ``batch_predict`` without modification. In many
cases only the video frame needs to be provided, but additional metadata may be
included as necessary to fit your algorithm's needs.

batch_predict
-------------

.. code-block:: python

   def batch_predict(self, inputs: List[Any]) -> List[Any]:
       ...

This method takes in a batch as input and provides a list of result objects of
any type as output. What the result objects are will depend on the algorithm 
being defined, but the number of prediction objects returned _must_ match the
number of video frames provided as input.
