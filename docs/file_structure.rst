##############
File Structure
##############

All capsules start off their life as unpackaged capsules. An unpackaged capsule
is simply a directory containing all files that will be packaged into the
capsule. This directory must contain, at minimum, a meta.conf file and a
capsule.py file.

.. code-block::

   detector_person
   ├── meta.conf
   └── capsule.py

The meta.conf file is a simple configuration file which specifies the major and
minor version of OpenVisionCapsules that this capsule requires. Applications use
this information to decide if your capsule is compatible with the version of
OpenVisionCapsules the application uses. A capsule with a compatibility version
of 0.1 are expected to be compatible with applications that use
OpenVisionCapsules version 0.1 through 0.x, but not 1.x or 2.x.

.. code-block:: ini

   [about]
   api_compatibility_version = 0.1


The capsule.py file is the meat of the capsule. It contains the actual behavior
of the capsule. We will talk more about the contents of this file in a later
section.

If your capsule uses other files for its operation, like a model file, it should
be included in this directory as well. All files in the capsule's directory
will be included and made accessible once it's packaged.

.. code-block::

  person_detection_capsule
  ├── meta.conf
  ├── capsule.py
  └── frozen_inference_graph.pb
