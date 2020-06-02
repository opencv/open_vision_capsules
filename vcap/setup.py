#!/usr/bin/env python3
from setuptools import setup, find_namespace_packages

test_packages = ["pytest", "mock"]

setup(
    name='vcap',
    description="A library for creating OpenVisionCapsules in Python",
    author="Dilili Labs",
    packages=find_namespace_packages(include=["vcap*"]),

    # Pull the package version from Git tags
    use_scm_version={
        # Helps setuptools_scm find the repository root
        "root": "..",
        "relative_to": __file__,
        # We want to be able to push these releases to PyPI, which doesn't
        # support local versions. Local versions are anything after the "+" in
        # a version string like "0.1.4.dev16+heyguys".
        "local_scheme": "no-local-version",
    },

    setup_requires=[
        "setuptools_scm",
    ],

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
