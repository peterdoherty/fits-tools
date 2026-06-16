#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Image-type normalization and validation for fits_tools.

This module provides a single authoritative mapping from the many
FITS header strings used to describe image types (``IMGTYPE``,
``IMAGETYP``, ``OBSTYPE``) to the canonical uppercase values defined
in :mod:`fits_tools.constants`.

The canonical values are::

    BIAS, DARK, FLAT, LIGHT, OBJECT, SCIENCE,
    TEST, SNAPSHOT, ARC, COMPARISON

Functions
---------
normalize_image_type
    Map a raw header string to a canonical image-type string.
is_valid_image_type
    Return True if a string is a recognized canonical image type.
"""

from __future__ import annotations

import logging

from .constants import (
    ALL_IMAGE_TYPES,
    FITS_VAL_ARC_IMGTYPE,
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_COMPARISON_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
    FITS_VAL_OBJECT_IMGTYPE,
    FITS_VAL_SCIENCE_IMGTYPE,
    FITS_VAL_SNAPSHOT_IMGTYPE,
    FITS_VAL_TEST_IMGTYPE,
)

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_image_type",
    "is_valid_image_type",
]

# ---------------------------------------------------------------------------
# Normalization map
#
# Keys are lowercase stripped variants found in real FITS files.
# Values are the canonical uppercase constants from fits_tools.constants.
# ---------------------------------------------------------------------------
_NORMALIZATION_MAP: dict[str, str] = {
    # --- BIAS ---
    "bias":           FITS_VAL_BIAS_IMGTYPE,
    "bias frame":     FITS_VAL_BIAS_IMGTYPE,
    "zero":           FITS_VAL_BIAS_IMGTYPE,
    "zero frame":     FITS_VAL_BIAS_IMGTYPE,
    "overscan":       FITS_VAL_BIAS_IMGTYPE,
    # --- DARK ---
    "dark":           FITS_VAL_DARK_IMGTYPE,
    "dark frame":     FITS_VAL_DARK_IMGTYPE,
    "dark current":   FITS_VAL_DARK_IMGTYPE,
    # --- FLAT ---
    "flat":           FITS_VAL_FLAT_IMGTYPE,
    "flat field":     FITS_VAL_FLAT_IMGTYPE,
    "flat frame":     FITS_VAL_FLAT_IMGTYPE,
    "flatfield":      FITS_VAL_FLAT_IMGTYPE,
    "sky flat":       FITS_VAL_FLAT_IMGTYPE,
    "dome flat":      FITS_VAL_FLAT_IMGTYPE,
    "twilight flat":  FITS_VAL_FLAT_IMGTYPE,
    "lamp flat":      FITS_VAL_FLAT_IMGTYPE,
    # --- LIGHT / OBJECT / SCIENCE ---
    "light":          FITS_VAL_LIGHT_IMGTYPE,
    "light frame":    FITS_VAL_LIGHT_IMGTYPE,
    "object":         FITS_VAL_OBJECT_IMGTYPE,
    "object frame":   FITS_VAL_OBJECT_IMGTYPE,
    "science":        FITS_VAL_SCIENCE_IMGTYPE,
    "science frame":  FITS_VAL_SCIENCE_IMGTYPE,
    "expose":         FITS_VAL_LIGHT_IMGTYPE,
    "exposure":       FITS_VAL_LIGHT_IMGTYPE,
    # --- ARC / COMPARISON ---
    "arc":            FITS_VAL_ARC_IMGTYPE,
    "arc frame":      FITS_VAL_ARC_IMGTYPE,
    "arc lamp":       FITS_VAL_ARC_IMGTYPE,
    "comparison":     FITS_VAL_COMPARISON_IMGTYPE,
    "comp":           FITS_VAL_COMPARISON_IMGTYPE,
    "comp lamp":      FITS_VAL_COMPARISON_IMGTYPE,
    "wavelength cal": FITS_VAL_ARC_IMGTYPE,
    # --- TEST / SNAPSHOT ---
    "test":           FITS_VAL_TEST_IMGTYPE,
    "test frame":     FITS_VAL_TEST_IMGTYPE,
    "snapshot":       FITS_VAL_SNAPSHOT_IMGTYPE,
    "snap":           FITS_VAL_SNAPSHOT_IMGTYPE,
}

# Canonical values map to themselves for round-trip safety.
_CANONICAL_SET: frozenset[str] = frozenset(ALL_IMAGE_TYPES)
_NORMALIZATION_MAP.update({v.lower(): v for v in _CANONICAL_SET})


def normalize_image_type(
    raw: str | None,
    strict: bool = False,
) -> str | None:
    """Map a raw FITS image-type string to a canonical value.

    Parameters
    ----------
    raw : str or None
        Raw value of ``IMGTYPE``, ``IMAGETYP``, or ``OBSTYPE``. Leading
        and trailing whitespace is stripped before lookup. ``None`` is
        accepted and returns ``None``.
    strict : bool, optional
        If True, emit a warning when *raw* does not map to a known type.
        The return value is ``None`` regardless of *strict*; this flag
        only controls whether the unrecognized value is logged. Default
        is False.

    Returns
    -------
    str or None
        Canonical image-type string (e.g. ``"BIAS"``, ``"DARK"``), or
        ``None`` if *raw* is ``None``, empty, or unrecognized.

    Examples
    --------
    >>> normalize_image_type("bias frame")
    'BIAS'
    >>> normalize_image_type("FLAT FIELD")
    'FLAT'
    >>> normalize_image_type("unknown") is None
    True
    >>> normalize_image_type(None) is None
    True
    """
    if raw is None:
        return None

    key = str(raw).strip().lower()
    if not key:
        return None

    canonical = _NORMALIZATION_MAP.get(key)
    if canonical is not None:
        return canonical

    # Not found — try the canonical set directly (handles exact-case matches
    # not already covered by the lowercase round-trip entries above).
    upper = key.upper()
    if upper in _CANONICAL_SET:
        return upper

    if strict:
        logger.warning(
            "normalize_image_type: unrecognized image type %r", raw
        )
    return None


def is_valid_image_type(value: str | None) -> bool:
    """Return True if *value* is a recognized canonical image type.

    Parameters
    ----------
    value : str or None
        String to test. Comparison is case-sensitive against the
        canonical uppercase values in :data:`fits_tools.constants.ALL_IMAGE_TYPES`.
        Pass the output of :func:`normalize_image_type` for
        case-insensitive checking.

    Returns
    -------
    bool
        True if *value* is in the canonical image-type set.

    Examples
    --------
    >>> is_valid_image_type("BIAS")
    True
    >>> is_valid_image_type("bias")
    False
    >>> is_valid_image_type(None)
    False
    """
    if value is None:
        return False
    return value in _CANONICAL_SET
