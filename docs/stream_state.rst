.. _`Stream State`:

############
Stream State
############


It is sometimes desirable to carry state throughout the lifetime of an entire
video stream, rather than on a frame-by-frame basis. This is where StreamState
comes in.

If the ``stream_state`` field of the capsule's Capsule class is set, the
``process_frame`` method for your capsule's backend will be passed an instance
of the provided class. Any state that should exist for the duration of the
videostream may be saved here.

This is commonly used by capsules that track objects between video frames.
Information on previous detections can be stored in the StreamState object and
read when new detections are found. This can also be useful for result
smoothing, for caching frames (think RNN models), and many other use cases.

A capsule's StreamState class does not need to implement any methods and has
no functional purpose outside of the capsule.
