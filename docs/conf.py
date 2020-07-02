# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../vcap'))
sys.path.insert(0, os.path.abspath('../vcap_utils'))


# -- Project information -----------------------------------------------------

project = 'OpenVisionCapsules'
copyright = '2020, OpenCV Foundation and Dilili Labs'
author = 'Aotu'

add_module_names = False


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

# Mock out any third-party imports
autodoc_mock_imports = []
