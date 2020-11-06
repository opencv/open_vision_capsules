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

.. autoclass:: vcap.BaseBackend
   :members: process_frame, close

Batching Methods
================

Batching refers to the process of collecting more than one video frame into a
"batch" and sending them all out for processing at once. Certain algorithms see
performance improvements when batching is used, because doing so decreases the
amount of round-trips the video frames take between devices.

If you wish to use batching in your capsule, you may call the ``send_to_batch``
method in ``process_frame`` instead of doing analysis in that method directly.
The ``send_to_batch`` method sends the input to a ``BatchExecutor`` which collects
inference requests for this capsule from different streams. Then, the
``BatchExecutor`` routinely calls your backend's ``batch_predict`` method with a
list of the collected inputs. As a result, users of ``send_to_batch`` must
override the ``batch_predict`` method in addition to the other required methods.

The ``send_to_batch`` method is asynchronous. Instead of immediately returning
analysis results, it returns a ``concurrent.futures.Future`` where the result will be provided.
Simple batching capsules may call ``send_to_batch``, then immediately call
``result`` to block for the result.

.. code-block:: python

   result = self.send_to_batch(frame).result()

An argument of any type may be provided to ``send_to_batch``, as the argument
will be passed in a list to ``batch_predict`` without modification. In many
cases only the video frame needs to be provided, but additional metadata may be
included as necessary to fit your algorithm's needs.

.. autoclass:: vcap.BaseBackend
   :members: batch_predict

