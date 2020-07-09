###################
Packaging a Capsule
###################

For capsule and application developers, OpenVisionCapsules provides a function
to package up unpackaged capsules. The optional ``key`` field encrypts the
capsule with AES.

.. code-block:: python

   from vcap import package_capsule

   package_capsule(Path("detector_person"),
                   Path("capsules", "detector_person.cap"),
                   key="[AES Key]")
