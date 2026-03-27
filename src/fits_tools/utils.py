#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for FITS data processing.

This module provides common utilities used across the fits_tools package,
including:
    - Data type conversion and JSON serialization
    - File path manipulation and validation
    - Array and index validation
    - Logging configuration
    - Common FITS operations

Version History:
    0.1.0 - Initial release with JSON preparation utility
    0.2.0 - Added file utilities, validation, logging, and FITS helpers
            consolidated from other modules
"""

from pathlib import Path
from typing import Any

import numpy as np

from .constants import is_fits_file  
from .types import StrOrPath

# =============================================================================
# Data Type Conversion
# =============================================================================

def prepare_for_json(data: Any) -> Any:
    """
    Recursively convert numpy types to standard Python types for JSON serialization.
    
    Handles dictionaries, lists, numpy arrays, and scalar numpy types.
    
    Args:
        data: Data structure (dict, list, array, or scalar) potentially
              containing numpy types.
    
    Returns:
        Data structure with numpy types converted to Python native types.
    
    Examples:
        >>> import numpy as np
        >>> data = {"mean": np.float32(1.5), "counts": np.int32(100)}
        >>> result = prepare_for_json(data)
        >>> isinstance(result["mean"], float)
        True
        >>> isinstance(result["counts"], int)
        True
        
        >>> arr = np.array([1, 2, 3])
        >>> prepare_for_json(arr)
        [1, 2, 3]
    """
    if isinstance(data, dict):
        return {str(k): prepare_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [prepare_for_json(elem) for elem in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.complexfloating):
        return complex(data)
    else:
        return data


def ensure_native_types(value: Any) -> Any:
    """
    Convert a single value from numpy type to Python native type.
    
    Simpler alternative to prepare_for_json for scalar values.
    
    Args:
        value: Value to convert.
    
    Returns:
        Python native type equivalent.
    
    Examples:
        >>> import numpy as np
        >>> ensure_native_types(np.int32(42))
        42
        >>> ensure_native_types(np.float32(3.14))
        3.14
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.complexfloating):
        return complex(value)
    return value


# =============================================================================
# File Path Utilities
# =============================================================================

def normalize_path(path: StrOrPath) -> Path:
    """
    Normalize a file path to absolute Path object.
    
    Expands user home directory (~) and resolves relative paths.
    
    Args:
        path: Input path (string or Path).
    
    Returns:
        Absolute Path object.
    
    Examples:
        >>> p = normalize_path("~/data/image.fits")
        >>> p.is_absolute()
        True
        >>> p = normalize_path("./image.fits")
        >>> p.is_absolute()
        True
    """
    return Path(path).expanduser().resolve()


def ensure_extension(filepath: StrOrPath, default_ext: str = ".fits") -> Path:
    """
    Ensure a path has a FITS extension.

    If *filepath* already ends with a recognized FITS extension (including
    compressed forms like ``.fits.gz`` and ``.fits.fz``), it is returned
    unchanged. Otherwise, if the path has no suffix, ``default_ext`` is
    appended.

    Parameters
    ----------
    filepath
        Input path.
    default_ext
        Extension to append when no suffix is present. Default is ``".fits"``.

    Returns
    -------
    Path
        Path with a FITS extension.

    Examples
    --------
    >>> ensure_extension("output")
    PosixPath('output.fits')
    >>> ensure_extension("output.txt")
    PosixPath('output.txt')
    >>> ensure_extension("image.fits.gz")
    PosixPath('image.fits.gz')
    """
    path = Path(filepath)

    if is_fits_file(path):
        return path

    # Only append if there is no suffix at all.
    if path.suffix == "":
        return path.with_suffix(default_ext)

    return path

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'prepare_for_json',
    'ensure_native_types',
    'normalize_path',
    'ensure_extension',
]
