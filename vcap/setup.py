#!/usr/bin/env python3
from setuptools import setup, find_namespace_packages

test_packages = ["pytest", "mock"]

setup(
    name='vcap',
    version='0.1.0',
    description="A library for creating OpenVisionCapsules in Python",
    packages=find_namespace_packages(include=["vcap*"]),

    author="Dilili Labs",

    install_requires=[
        "pycryptodomex~=3.0",
        "scipy~=1.0",
        "scikit-learn~=0.20",
        "numpy~=1.18",
        "tensorflow~=1.15"
    ],
    extras_require={
        "tests": test_packages,
        "easy": ["opencv-python-headless~=4.1.2"],
    },
    tests_require=test_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
