#!/usr/bin/env python3
from setuptools import setup, find_namespace_packages

test_packages = ["pytest", "mock"]

setup(
    name='vcap-utils',
    version='0.1.3',
    description="Utilities for creating OpenVisionCapsules easily in Python",
    packages=find_namespace_packages(
        include=["vcap_utils*"],
        exclude=["vcap_utils.tests*"]),

    author="Dilili Labs",

    install_requires=[
        "vcap==0.1.3"
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
