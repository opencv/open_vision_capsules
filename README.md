# Introduction

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
contains the necessary facilities to create and encapsulate and algorithm.
`vcap-utils` contains a set of utilities that make encapsulating algorithms of
certain types easier.

# Project Status

OpenVisionCapsules is in a developer preview phase. We're looking for developer
feedback before reaching a stable 1.0 release. If you find any bugs or have
suggestions, please open an issue.

# Getting Started

A couple example capsules are available under `vcap/examples`, demonstrating
how to create classifier and detector capsules from TensorFlow models.

# Installation

To install OpenVisionCapsules locally, clone the repository and run the
following commands to install the `vcap` and `vcap-utils` packages in the
current environment.

```
cd vcap
pip3 install -e .
cd ../vcap_utils
pip3 install -e .
```

## Git LFS

This repository uses Git LFS to store image files for testing and model files
for the example plugins. If you want to run tests or use the example plugins,
you'll need to install Git LFS before cloning to get these files. See the
[Git LFS installation instructions][git lfs install] for more information.

[git lfs install]: https://github.com/git-lfs/git-lfs/wiki/Installation

