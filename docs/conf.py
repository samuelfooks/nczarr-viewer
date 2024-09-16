# docs/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = 'Zarr-NetCDF Viewer'
author = 'Samuel Fooks'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']

# Options for PDF output
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Options for EPUB output
epub_show_urls = 'footnote'