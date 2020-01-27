# Introduction

This is a guide on how to encapsulate an algorithm using OpenVisionCapsules.
Capsules are discrete components that define new ways to analyze video streams.

This guide discusses the Open Vision Capsule system generally, not any specific
information on how to write a Capsule of a certain type or with certain
technology. Example capsules are available under `vcap/examples` that show how
to encapsulate models from various popular machine learning frameworks.

# What is a Capsule?

A capsule is a single file with a `.cap` file extension. It contains the code,
metadata, model files, and any other files the capsule needs to operate.

Capsules take a frame and information from other capsules as input, run some
kind of analysis, and provide metadata about the frame as output. For example, a
person detection capsule would take a frame as input and output person
detections in that frame. A gender classifier capsule would take this frame and
each person detection as input, and output a male or female classification for
that detection.

Capsules provide metadata describing these inputs and outputs alongside other
information on the capsule. Applications that are compatible OpenVisionCapsules
use this metadata to know when to run the capsule and with what input.
