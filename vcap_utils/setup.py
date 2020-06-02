#!/usr/bin/env python3
from setuptools import setup, find_namespace_packages

test_packages = ["pytest", "mock"]

setup(
    name='vcap-utils',
    description="Utilities for creating OpenVisionCapsules easily in Python",
    author="Dilili Labs",
    packages=find_namespace_packages(
        include=["vcap_utils*"],
        exclude=["vcap_utils.tests*"]),

    # Pull the package version from Git tags
    use_scm_version={
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
