"""Sphinx configuration for Medical Image Standard documentation."""

import os
import sys

# Add the project root to sys.path so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Medical Image Standard"
copyright = "2024, Hamza Gbada"
author = "Hamza Gbada"
release = "0.7.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.md"]

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = True

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = False

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output configuration -----------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_title = "Medical Image Standard"
html_logo = "_static/logo.png"

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#D81B60",
        "color-brand-content": "#D81B60",
        "color-api-name": "#D81B60",
        "color-api-pre-name": "#AD144D",
        "color-link": "#D81B60",
        "color-link--hover": "#AD144D",
        "color-sidebar-link-text--top-level": "#2A1B20",
        "color-sidebar-item-background--current": "#FCE4EC",
        "color-sidebar-item-background--hover": "#FCE4EC",
    },
    "dark_css_variables": {
        "color-brand-primary": "#F06292",
        "color-brand-content": "#F06292",
        "color-api-name": "#F06292",
        "color-api-pre-name": "#F48FB1",
        "color-link": "#F06292",
        "color-link--hover": "#F48FB1",
    },
    "source_repository": "https://github.com/LATIS-medical-Imaging/medical-image-std",
    "source_branch": "master",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/LATIS-medical-Imaging/medical-image-std",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}

# Suppress specific warnings
suppress_warnings = ["autodoc.import_error"]