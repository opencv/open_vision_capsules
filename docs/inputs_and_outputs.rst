.. _`Inputs and Outputs`:

##################
Inputs and Outputs
##################

Introduction
============

Capsules are defined by the data they take as input and the information they
give as output. Applications use this information to connect capsules to each
other and schedule their execution. These inputs and outputs are `defined`
by NodeDescription objects and `realized` by DetectionNode objects.

.. autoclass:: vcap.DetectionNode

.. autoclass:: vcap.NodeDescription

Examples
========

detections
----------

A capsule that can encode cars or trucks would use a NodeDescription like this
as its ``input_type``:

.. code-block:: python

   NodeDescription(detections=["car", "truck"])

A capsule that can detect people and dogs would use a NodeDescription like this
as its ``output_type``:

.. code-block:: python

   NodeDescription(detections=["person", "dog"])

attributes
----------

A capsule that operates on detections that have been classified for gender use
a NodeDescription like this as its ``input_type``:

.. code-block:: python

   NodeDescription(
       attributes={
           "gender": ["male", "female"],
           "color": ["red", "blue", "green"]
       })

A capsule that can classify people’s gender as either male or female would have
the following NodeDescription as its ``output_type``:

.. code-block:: python

   NodeDescription(
       detections=["person"],
       attributes={
           "gender": ["male", "female"]
       })

encoded
-------

A capsule that operates on detections of cars that have been encoded use a
NodeDescription like this as its ``input_type``:

.. code-block:: python

   NodeDescription(
       detections=["car"],
       encoded=True)

A capsule that encodes people would use a NodeDescription like this as its
``output_type``:

.. code-block:: python

   NodeDescription(
       detections=["person"],
       encoded=True)

tracked
-------

A capsule that operates on person detections that have been tracked would use a
NodeDescription like this as its ``input_type``.

.. code-block:: python

   NodeDescription(
       detections=["person"],
       tracked=True)

A capsule that tracks people would use a NodeDescription like this as its
``output_type``:

.. code-block:: python

   NodeDescription(
       detections=["person"],
       tracked=True)

extra_data
----------

A capsule that operates on people detections with a "process_extra_fast"
``extra_data`` field would use a NodeDescription like this as its
``input_type``:

.. code-block:: python

   NodeDescription(
       detections=["person"],
       extra_data=["process_extra_fast"])

A capsule that adds an "is_special" ``extra_data`` field to its person-detected
output would use a NodeDescription like this as its ``output_type``:

.. code-block:: python

   NodeDescription(
       detections=["person"],
       extra_data=["is_special"])
