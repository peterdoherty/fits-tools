#!/usr/bin/env python3
"""
Image-type normalization and classification utilities.

This module centralizes logic for interpreting FITS header image-type values
(e.g., ``IMGTYPE`` / ``IMAGETYP``). It provides:

- Normalization of non-standard values to canonical types (via alias mapping)
- Validation of image-type strings
- High-level classification helpers (calibration vs science)
- Human-readable descriptions for canonical types

The canonical type values are defined in :mod:`fits_tools.constants` and this
module provides behavior built on top of those constants. No FITS I/O is
performed here.

Examples
--------
>>> normalize_image_type("zero")
'BIAS'
>>> is_calibration_frame("skyflat")
True
>>> is_science_frame("OBJECT")
True
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Optional, Mapping

from .constants import (
    FITS_VAL_BIAS_IMGTYPE,  
    FITS_VAL_DARK_IMGTYPE,    
    FITS_VAL_FLAT_IMGTYPE,  
    FITS_VAL_LIGHT_IMGTYPE,   
    FITS_VAL_OBJECT_IMGTYPE,  
    FITS_VAL_SCIENCE_IMGTYPE,  
    FITS_VAL_TEST_IMGTYPE,     
    FITS_VAL_SNAPSHOT_IMGTYPE, 
    FITS_VAL_ARC_IMGTYPE,   
    FITS_VAL_COMPARISON_IMGTYPE,
    ALL_IMAGE_TYPES,
    CALIBRATION_IMAGE_TYPES,
    SCIENCE_IMAGE_TYPES,
    )

# =============================================================================
# Image-Type Alias Mapping
# =============================================================================

#: Map from upper-cased alias strings to canonical image type values.
#:
#: Used by :func:`normalize_image_type` to translate non-standard or
#: instrument-specific ``IMGTYPE``/``IMAGETYP`` values into the canonical
#: forms defined by the ``FITS_VAL_*_IMGTYPE`` constants.
_IMAGE_TYPE_ALIASES: dict[str, str] = {
    # --- Bias / Zero ---
    "BIAS":       FITS_VAL_BIAS_IMGTYPE,
    "ZERO":       FITS_VAL_BIAS_IMGTYPE,
    "BIAS FRAME": FITS_VAL_BIAS_IMGTYPE,
    "OFFSET":     FITS_VAL_BIAS_IMGTYPE,  # some CMOS cameras

    # --- Dark ---
    "DARK":       FITS_VAL_DARK_IMGTYPE,
    "DARK FRAME": FITS_VAL_DARK_IMGTYPE,

    # --- Flat ---
    "FLAT":         FITS_VAL_FLAT_IMGTYPE,
    "FLATFIELD":    FITS_VAL_FLAT_IMGTYPE,
    "FLAT FIELD":   FITS_VAL_FLAT_IMGTYPE,
    "FLAT FRAME":   FITS_VAL_FLAT_IMGTYPE,
    "SKYFLAT":      FITS_VAL_FLAT_IMGTYPE,
    "SKY FLAT":     FITS_VAL_FLAT_IMGTYPE,
    "DOMEFLAT":     FITS_VAL_FLAT_IMGTYPE,
    "DOME FLAT":    FITS_VAL_FLAT_IMGTYPE,
    "TWILIGHT":     FITS_VAL_FLAT_IMGTYPE,
    "TWILIGHTFLAT": FITS_VAL_FLAT_IMGTYPE,

    # --- Science / Object ---
    "LIGHT":       FITS_VAL_LIGHT_IMGTYPE,
    "LIGHT FRAME": FITS_VAL_LIGHT_IMGTYPE,
    "OBJECT":      FITS_VAL_OBJECT_IMGTYPE,
    "SCIENCE":     FITS_VAL_SCIENCE_IMGTYPE,
    "SCI":         FITS_VAL_SCIENCE_IMGTYPE,
    "TARGET":      FITS_VAL_SCIENCE_IMGTYPE,
    "OBSERVATION": FITS_VAL_SCIENCE_IMGTYPE,

    # --- Other ---
    "TEST":     FITS_VAL_TEST_IMGTYPE,
    "SNAPSHOT": FITS_VAL_SNAPSHOT_IMGTYPE,
    "SNAP":     FITS_VAL_SNAPSHOT_IMGTYPE,
    "ARC":      FITS_VAL_ARC_IMGTYPE,
    "COMP":     FITS_VAL_COMPARISON_IMGTYPE,
    "COMPARISON": FITS_VAL_COMPARISON_IMGTYPE,
}

IMAGE_TYPE_ALIASES: Mapping[str, str] = MappingProxyType(_IMAGE_TYPE_ALIASES)

#: Frozen set of all valid image-type strings, including both canonical
#: types and their aliases (all upper-cased).
VALID_IMAGE_TYPE_STRINGS: frozenset[str] = frozenset(ALL_IMAGE_TYPES) | frozenset(IMAGE_TYPE_ALIASES.keys())

_IMAGE_TYPE_DESCRIPTIONS: dict[str, str] = {
    FITS_VAL_BIAS_IMGTYPE: (
        "Bias / zero frame — zero-length exposure to characterise "
        "the detector readout offset."
    ),
    FITS_VAL_DARK_IMGTYPE: (
        "Dark frame — shutter-closed exposure to measure thermal dark current."
    ),
    FITS_VAL_FLAT_IMGTYPE: (
        "Flat-field frame — uniformly illuminated exposure for "
        "pixel-to-pixel sensitivity correction."
    ),
    FITS_VAL_LIGHT_IMGTYPE: (
        "Light frame — science exposure of an astronomical target."
    ),
    FITS_VAL_OBJECT_IMGTYPE: (
        "Object frame — science exposure of an astronomical target."
    ),
    FITS_VAL_SCIENCE_IMGTYPE: (
        "Science frame — science exposure of an astronomical target."
    ),
    FITS_VAL_TEST_IMGTYPE: (
        "Test frame — engineering or system-test exposure."
    ),
    FITS_VAL_SNAPSHOT_IMGTYPE: (
        "Snapshot frame — quick preview or focus exposure."
    ),
    FITS_VAL_ARC_IMGTYPE: (
        "Arc frame — wavelength-calibration lamp exposure."
    ),
    FITS_VAL_COMPARISON_IMGTYPE: (
        "Comparison frame — calibration lamp exposure for wavelength reference."
    ),
}

IMAGE_TYPE_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(_IMAGE_TYPE_DESCRIPTIONS)

# =============================================================================
# Helper Functions
# =============================================================================

def _require_str_or_none(name: str, value: object) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be str or None, got {type(value).__name__}"
        )
    stripped = value.strip()
    if stripped == "":
        return None
    return stripped


def _normalize_image_type_str(value: str) -> str:
    """
    Normalize a non-empty image type string (no validation).

    Parameters
    ----------
    value
        Non-empty string (already stripped).

    Returns
    -------
    str
        Canonical type if known; otherwise uppercased input.
    """
    upper = value.upper()
    return IMAGE_TYPE_ALIASES.get(upper, upper)


def normalize_image_type(
    imgtype_value: Optional[str],
    strict: bool = False
) -> Optional[str]:
    """Normalise an image-type string to its canonical form.

    Strips whitespace, converts to upper case, and maps the result through
    :data:`IMAGE_TYPE_ALIASES`.  If the uppercased value is not found in the
    alias table, it is returned as-is (upper-cased).

    Parameters
    ----------
    imgtype_value:
        Raw value from an ``IMGTYPE`` or ``IMAGETYP`` header keyword.
    strict: bool, optional
        If ``True``, raise ``ValueError`` for unrecognised types.
        If ``False`` (default), return the uppercased input as-is.

    Returns
    -------
    str or None
        Canonical image-type string, or the uppercased input when not
        recognised, or ``None`` when *imgtype_value* is ``None`` / empty.

    Raises
    ------
    ValueError
        If *strict* is ``True`` and *imgtype_value* is not recognised.

    Examples
    --------
    >>> normalize_image_type('zero')
    'BIAS'
    >>> normalize_image_type('SkyFlat')
    'FLAT'
    >>> normalize_image_type('UNKNOWN')
    'UNKNOWN'
    >>> normalize_image_type(None) is None
    True
    >>> normalize_image_type('UNKNOWN', strict=True)
    Traceback (most recent call last): 
    ...
    ValueError: Unrecognised image type: 'UNKNOWN'
    """
    s = _require_str_or_none("imgtype_value", imgtype_value)
    if s is None:
        return None

    norm = _normalize_image_type_str(s)

    if strict and norm not in ALL_IMAGE_TYPES:
        raise ValueError(f"Unrecognised image type: {norm!r}")

    return norm


def is_valid_image_type(imgtype_value: Optional[str]) -> bool:
    """Return ``True`` if *imgtype_value* is a recognised image type.

    Accepts both canonical values (e.g. ``'BIAS'``) and documented aliases
    (e.g. ``'ZERO'``, ``'SKYFLAT'``).  Comparison is case-insensitive with
    whitespace trimming.

    Parameters
    ----------
    imgtype_value:
        Value from an ``IMGTYPE`` / ``IMAGETYP`` header keyword.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_valid_image_type('DARK')
    True
    >>> is_valid_image_type('dark')
    True
    >>> is_valid_image_type('SKYFLAT')   # alias
    True
    >>> is_valid_image_type('UNKNOWN')
    False
    >>> is_valid_image_type(None)
    False
    """
    s = _require_str_or_none("imgtype_value", imgtype_value)
    if s is None:
        return False

    return _normalize_image_type_str(s) in ALL_IMAGE_TYPES


def is_calibration_frame(imgtype_value: Optional[str]) -> bool:
    """Return ``True`` if *imgtype_value* represents a calibration frame.

    Accepts both canonical values and aliases.  A calibration frame is one
    of ``BIAS``, ``DARK``, or ``FLAT``.

    Parameters
    ----------
    imgtype_value:
        Value from an ``IMGTYPE`` / ``IMAGETYP`` header keyword.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_calibration_frame('BIAS')
    True
    >>> is_calibration_frame('zero')   # alias
    True
    >>> is_calibration_frame('LIGHT')
    False
    >>> is_calibration_frame(None)
    False
    """
    s = _require_str_or_none("imgtype_value", imgtype_value)
    if s is None:
        return False

    return _normalize_image_type_str(s) in CALIBRATION_IMAGE_TYPES


def is_science_frame(imgtype_value: Optional[str]) -> bool:
    """Return ``True`` if *imgtype_value* represents a science / object frame.

    Accepts both canonical values and aliases.  A science frame is one of
    ``LIGHT``, ``OBJECT``, or ``SCIENCE``.

    Parameters
    ----------
    imgtype_value:
        Value from an ``IMGTYPE`` / ``IMAGETYP`` header keyword.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_science_frame('OBJECT')
    True
    >>> is_science_frame('light')
    True
    >>> is_science_frame('BIAS')
    False
    >>> is_science_frame(None)
    False
    """
    s = _require_str_or_none("imgtype_value", imgtype_value)
    if s is None:
        return False

    return _normalize_image_type_str(s) in SCIENCE_IMAGE_TYPES


def get_image_type_description(imgtype_value: Optional[str]) -> str:
    """Return a human-readable description of an image type.

    Normalises *imgtype_value* via :func:`normalize_image_type` before
    lookup, so aliases are resolved automatically.

    Parameters
    ----------
    imgtype_value:
        Value from an ``IMGTYPE`` / ``IMAGETYP`` header keyword.

    Returns
    -------
    str

    Examples
    --------
    >>> get_image_type_description('BIAS')
    'Bias / zero frame — zero-length exposure to characterise the detector readout offset.'
    >>> get_image_type_description(None)
    'Unknown or missing image type.'
    >>> get_image_type_description('CUSTOM')
    "Unrecognised image type: 'CUSTOM'."
    """
    s = _require_str_or_none("imgtype_value", imgtype_value)
    if s is None:
        return "Unknown or missing image type."

    norm = _normalize_image_type_str(s)
    return IMAGE_TYPE_DESCRIPTIONS.get(norm, f"Unrecognised image type: {norm!r}.")


__all__ = [
    # Data tables
    "IMAGE_TYPE_ALIASES",
    "VALID_IMAGE_TYPE_STRINGS",
    "IMAGE_TYPE_DESCRIPTIONS",
    # Functions
    "normalize_image_type",
    "is_valid_image_type",
    "is_calibration_frame",
    "is_science_frame",
    "get_image_type_description",
]

