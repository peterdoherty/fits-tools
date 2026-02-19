# docs/conf.py

"""
Sphinx configuration for fits-defects documentation.
"""

import os
import sys
from datetime import datetime

# Add source to path
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'fits-defects'
copyright = f'{datetime.now().year}, Your Name'
author = 'Your Name'

# The full version, including alpha/beta/rc tags
release = '1.0.0'
version = '1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx.ext.githubpages',
]

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'inherited-members': False,
}

# Autosummary
autosummary_generate = True

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'  # Add your logo

html_theme_options = {
    'canonical_url': 'https://fits-defects.readthedocs.io',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    '