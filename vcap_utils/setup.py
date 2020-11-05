#!/usr/bin/env python3
import os
from pathlib import Path

from setuptools import setup, find_namespace_packages

# Get package version/metadata
about = {}
exec(Path("vcap_utils/version.py").read_text(), about)

test_packages = ["pytest", "mock"]

PRE_RELEASE_SUFFIX = os.environ.get("PRE_RELEASE_SUFFIX", "")

setup(
    name='vcap-utils',
    description="Utilities for creating OpenVisionCapsules easily in Python",
    author="Aotu.ai",
    packages=find_namespace_packages(include=["vcap_utils*"]),
    version=about["__version__"] + PRE_RELEASE_SUFFIX,

    install_requires=[
        "vcap",
    ],

    extras_require={
        "tests": test_packages,
    },
    tests_require=test_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
