# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime
import importlib

sys.path.insert(0, os.path.abspath(".."))

project = "CorrAdjust"
orig_year, cur_year = 2024, datetime.now().year
year_str = str(orig_year) if orig_year == cur_year else f"{orig_year}-{cur_year}"
copyright = f"{year_str}, Computational Medicine Center at Thomas Jefferson University"
author = "Stepan Nersisyan"
release = importlib.metadata.version("corradjust")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "numpydoc",
    "nbsphinx"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"CorrAdjust {release} documentation"

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/TJU-CMC-Org/CorrAdjust",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
   ]
}

html_sidebars = {
    "installation": []
}

# Custom options
autodoc_default_options = {
    "member-order": "bysource",
    "private-members": True
}

numpydoc_show_class_members = False
#numpydoc_class_members_toctree = False
numpydoc_validation_checks = {"all", "SA01", "EX01", "SS05", "SS06", "ES01"}

nbsphinx_execute = "never"
