#!/usr/bin/env python3
from setuptools import setup, find_namespace_packages

test_packages = ["pytest", "mock"]

setup(
    name='vcap',
    version='0.1.2',
    description="A library for creating OpenVisionCapsules in Python",
    packages=find_namespace_packages(include=["vcap*"]),

    author="Dilili Labs",

    install_requires=[
        "pycryptodomex==3.9.7",
        "scipy==1.4.1",
        "scikit-learn==0.22.2",
        "numpy==1.18.4",
        "tensorflow-gpu==1.15.2",
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
