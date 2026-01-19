#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py - DEPRECATED MODULE

This module has been deprecated and its functionality moved to io.py.

.. deprecated:: 0.7.0
    The data module has been deprecated. Use the following replacements:
    
    - fits_extension_count() -> use io.count_hdus() or io.count_image_extensions()

This module will be removed in version 1.0.0.
"""

import warnings

# Show deprecation warning when module is imported
warnings.warn(
    "The 'data' module is deprecated and will be removed in version 1.0.0. "
    "Use 'fits_tools.io' instead. "
    "Replace 'fits_extension_count()' with 'count_hdus()' or 'count_image_extensions()'.",
    DeprecationWarning,
    stacklevel=2
)

# Import from io for backward compatibility
from fits_tools.io import fits_extension_count

__all__ = ['fits_extension_count']