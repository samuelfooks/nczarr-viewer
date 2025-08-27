# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import subprocess
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'nczarr-viewer'
author = 'Samuel Fooks'
release = '0.1.0'

# Get git remote URL dynamically
try:
    git_remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], 
                                       cwd=os.path.dirname(os.path.dirname(__file__)), 
                                       text=True).strip()
    repo_url = git_remote.replace('.git', '')
except:
    repo_url = 'https://github.com/EDITO-Infra/nczarr-viewer'  # fallback

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Add substitutions
rst_epilog = f"""
.. |repo_url| replace:: {repo_url}
"""

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