"""Small general-purpose helpers for fits_tools.

This module contains lightweight utilities that do not belong to a more
specialized submodule. It owns path/filename helpers and JSON normalization
helpers used throughout the package.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .constants import FITS_FILE_EXTENSIONS
from .types import StrOrPath

logger = logging.getLogger(__name__)

__all__ = [
    "prepare_for_json",
    "ensure_native_types",
    "normalize_path",
    "ensure_extension",
    "is_fits_file",
]


def _complex_to_json(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def prepare_for_json(data: Any) -> Any:
    """Recursively convert values to JSON-safe native Python types.

    Parameters
    ----------
    data : any
        Value or nested structure to convert.

    Returns
    -------
    any
        JSON-safe equivalent of *data*.

    Notes
    -----
    Conversion rules:

    - NumPy integer scalars → ``int``
    - NumPy boolean scalars → ``bool``
    - Finite float / NumPy float scalars → ``float``
    - NaN or infinite floats → ``None``
    - Complex values → ``{"real": ..., "imag": ...}``
    - ``np.ndarray`` → recursively converted ``list``
    - ``tuple`` → ``list``
    - ``dict`` keys → ``str``
    - All other types are returned unchanged.
    """
    if isinstance(data, dict):
        return {str(key): prepare_for_json(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [prepare_for_json(value) for value in data]
    if isinstance(data, np.ndarray):
        return prepare_for_json(data.tolist())
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, (np.floating, float)):
        value = float(data)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, (np.complexfloating, complex)):
        return _complex_to_json(complex(data))
    return data


def ensure_native_types(value: Any) -> Any:
    """Convert a single value to a JSON-friendly native Python type.

    This is an alias for :func:`prepare_for_json` provided for call-site
    clarity when operating on a single scalar rather than a nested structure.

    Parameters
    ----------
    value : any
        Value to convert.

    Returns
    -------
    any
        JSON-safe equivalent of *value*.
    """
    return prepare_for_json(value)


def normalize_path(path: StrOrPath) -> Path:
    """Return *path* as an expanded, resolved :class:`~pathlib.Path`.

    Parameters
    ----------
    path : str or Path
        Path to normalize. ``~`` is expanded and the path is made absolute.

    Returns
    -------
    Path
        Fully resolved path.
    """
    return Path(path).expanduser().resolve()


def is_fits_file(filepath: StrOrPath | None) -> bool:
    """Return True when *filepath* ends with a recognized FITS extension.

    The check is case-insensitive and supports compressed FITS filenames
    such as ``.fits.gz`` and ``.fits.fz``.

    Parameters
    ----------
    filepath : str, Path, or None
        Path to test.

    Returns
    -------
    bool
        True if *filepath* has a recognized FITS extension, False otherwise.
        Returns False when *filepath* is None.
    """
    if filepath is None:
        return False
    lower = str(filepath).lower()
    return any(lower.endswith(ext) for ext in FITS_FILE_EXTENSIONS)


def ensure_extension(filepath: StrOrPath, default_ext: str = ".fits") -> Path:
    """Ensure that *filepath* has a FITS extension.

    Parameters
    ----------
    filepath : str or Path
        Path to check.
    default_ext : str, optional
        Extension to append when *filepath* has no suffix. Default is
        ``".fits"``.

    Returns
    -------
    Path
        - Unchanged if *filepath* already ends with a recognized FITS
          extension.
        - *filepath* with *default_ext* appended if it has no suffix.
        - Unchanged in all other cases (non-FITS suffix is preserved as-is).

    Notes
    -----
    A path with a non-FITS suffix (e.g. ``.txt``) is returned without
    modification and without warning. Callers that need to enforce a FITS
    extension should validate the result with :func:`is_fits_file`.
    """
    path = Path(filepath)
    if is_fits_file(path):
        return path
    if path.suffix == "":
        return path.with_suffix(default_ext)
    logger.debug(
        "ensure_extension: %s has non-FITS suffix %r; returning unchanged.",
        path.name,
        path.suffix,
    )
    return path
