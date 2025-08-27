# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'nczarr-viewer'
author = 'Samuel Fooks'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# Build directory configuration
build_dir = 'build'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['static']
html_css_files = ['custom.css']
html_js_files = ['custom.js', 'redirect.js']

# GitHub Pages specific settings - these ensure proper URL handling
html_use_relative_urls = True
html_baseurl = '/nczarr-viewer/'
html_show_sourcelink = False

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'style_nav_header_background': '#667eea',
    'logo_only': False,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
}

# Options for PDF output
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Options for EPUB output
epub_show_urls = 'footnote'