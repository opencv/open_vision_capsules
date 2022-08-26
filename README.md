# OpenVisionCapsules

[![Documentation Status](https://readthedocs.org/projects/openvisioncapsules/badge/?version=latest)](https://openvisioncapsules.readthedocs.io/en/latest/?badge=latest)

This repository contains the OpenVisionCapsules SDK, a set of Python libraries
for encapsulating machine learning and computer vision algorithms for
intelligent video analytics.

Encapsulating an algorithm allows it to be deployed as a single, self-describing
file that inputs and outputs data in a standard format. This makes deployment
and integration significantly easier than starting with a model file or a
snippet of source code. Capsules are descriptive of their input and output
requirements, allowing OpenVisionCapsules to route data between capsules
automatically.

This project is split into two packages, `vcap` and `vcap-utils`. `vcap`
contains the necessary facilities to create and encapsulate an algorithm.
`vcap-utils` contains a set of utilities that make encapsulating algorithms of
certain types easier.

# Project Status

OpenVisionCapsules is in a developer preview phase. We're looking for developer
feedback before reaching a stable 1.0 release. If you find any bugs or have
suggestions, please open an issue.

# Getting Started

Take a look at the [documentation here][docs].

A couple example capsules are available under `vcap/examples`, demonstrating
how to create classifier and detector capsules from TensorFlow models.

# Installation

To install OpenVisionCapsules locally, clone the repository and run the
following commands to install the `vcap` and `vcap-utils` packages in the
current environment.

<!--
Installing both packages at the same time breaks setuptools_scm. See:
https://github.com/pypa/setuptools_scm/issues/454
-->
```
pip3 install -e ./vcap
pip3 install -e ./vcap_utils
```

# Examples

To make use of the example capsules in the `vcap/examples/` directory, make 
sure to run the tests with pytest (from the root of the repo). The tests
download all the necessary models and images, including the models for the 
example capsules.

A repository of open source capsules can be found [here][capsule_zoo].

[docs]: https://openvisioncapsules.readthedocs.io/en/latest/
[capsule_zoo]: https://github.com/aotuai/capsule_zoo
