from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .constants import (
    ALL_IMAGE_TYPES,
    CALIBRATION_IMAGE_TYPES,
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
    SCIENCE_IMAGE_TYPES,
)

_IMAGE_TYPE_ALIASES: dict[str, str] = {
    "BIAS": FITS_VAL_BIAS_IMGTYPE,
    "ZERO": FITS_VAL_BIAS_IMGTYPE,
    "BIAS FRAME": FITS_VAL_BIAS_IMGTYPE,
    "OFFSET": FITS_VAL_BIAS_IMGTYPE,
    "DARK": FITS_VAL_DARK_IMGTYPE,
    "DARK FRAME": FITS_VAL_DARK_IMGTYPE,
    "FLAT": FITS_VAL_FLAT_IMGTYPE,
    "FLATFIELD": FITS_VAL_FLAT_IMGTYPE,
    "FLAT FIELD": FITS_VAL_FLAT_IMGTYPE,
    "FLAT FRAME": FITS_VAL_FLAT_IMGTYPE,
    "SKYFLAT": FITS_VAL_FLAT_IMGTYPE,
    "SKY FLAT": FITS_VAL_FLAT_IMGTYPE,
    "DOMEFLAT": FITS_VAL_FLAT_IMGTYPE,
    "DOME FLAT": FITS_VAL_FLAT_IMGTYPE,
    "TWILIGHT": FITS_VAL_FLAT_IMGTYPE,
    "TWILIGHTFLAT": FITS_VAL_FLAT_IMGTYPE,
    "LIGHT": FITS_VAL_LIGHT_IMGTYPE,
    "LIGHT FRAME": FITS_VAL_LIGHT_IMGTYPE,
    "OBJECT": FITS_VAL_OBJECT_IMGTYPE,
    "SCIENCE": FITS_VAL_SCIENCE_IMGTYPE,
    "SCI": FITS_VAL_SCIENCE_IMGTYPE,
    "TARGET": FITS_VAL_SCIENCE_IMGTYPE,
    "OBSERVATION": FITS_VAL_SCIENCE_IMGTYPE,
    "TEST": FITS_VAL_TEST_IMGTYPE,
    "SNAPSHOT": FITS_VAL_SNAPSHOT_IMGTYPE,
    "SNAP": FITS_VAL_SNAPSHOT_IMGTYPE,
    "ARC": FITS_VAL_ARC_IMGTYPE,
    "COMP": FITS_VAL_COMPARISON_IMGTYPE,
    "COMPARISON": FITS_VAL_COMPARISON_IMGTYPE,
}

IMAGE_TYPE_ALIASES: Mapping[str, str] = MappingProxyType(_IMAGE_TYPE_ALIASES)
VALID_IMAGE_TYPE_STRINGS: frozenset[str] = frozenset(ALL_IMAGE_TYPES) | frozenset(IMAGE_TYPE_ALIASES)

_IMAGE_TYPE_DESCRIPTIONS: dict[str, str] = {
    FITS_VAL_BIAS_IMGTYPE: "Bias / zero frame — zero-length exposure to characterize detector readout offset.",
    FITS_VAL_DARK_IMGTYPE: "Dark frame — shutter-closed exposure used to measure thermal dark current.",
    FITS_VAL_FLAT_IMGTYPE: "Flat-field frame — uniformly illuminated exposure for sensitivity correction.",
    FITS_VAL_LIGHT_IMGTYPE: "Light frame — science exposure of a target.",
    FITS_VAL_OBJECT_IMGTYPE: "Object frame — science exposure of a target.",
    FITS_VAL_SCIENCE_IMGTYPE: "Science frame — science exposure of a target.",
    FITS_VAL_TEST_IMGTYPE: "Test frame — engineering or system-test exposure.",
    FITS_VAL_SNAPSHOT_IMGTYPE: "Snapshot frame — quick preview or focus exposure.",
    FITS_VAL_ARC_IMGTYPE: "Arc frame — wavelength-calibration lamp exposure.",
    FITS_VAL_COMPARISON_IMGTYPE: "Comparison frame — calibration lamp exposure for wavelength reference.",
}
IMAGE_TYPE_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(_IMAGE_TYPE_DESCRIPTIONS)


def _require_str_or_none(name: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str or None, got {type(value).__name__}")
    stripped = value.strip()
    return stripped or None


def _normalize_image_type_str(value: str) -> str:
    return IMAGE_TYPE_ALIASES.get(value.upper(), value.upper())


def normalize_image_type(imgtype_value: str | None, strict: bool = False) -> str | None:
    """Normalize an image-type string to its canonical form."""
    value = _require_str_or_none("imgtype_value", imgtype_value)
    if value is None:
        return None
    normalized = _normalize_image_type_str(value)
    if strict and normalized not in ALL_IMAGE_TYPES:
        raise ValueError(f"Unrecognised image type: {normalized!r}")
    return normalized


def is_valid_image_type(imgtype_value: str | None) -> bool:
    """Return True if *imgtype_value* is a recognized image type or alias."""
    value = _require_str_or_none("imgtype_value", imgtype_value)
    return False if value is None else _normalize_image_type_str(value) in ALL_IMAGE_TYPES


def is_calibration_frame(imgtype_value: str | None) -> bool:
    """Return True if *imgtype_value* represents a calibration frame."""
    value = _require_str_or_none("imgtype_value", imgtype_value)
    return False if value is None else _normalize_image_type_str(value) in CALIBRATION_IMAGE_TYPES


def is_science_frame(imgtype_value: str | None) -> bool:
    """Return True if *imgtype_value* represents a science/object frame."""
    value = _require_str_or_none("imgtype_value", imgtype_value)
    return False if value is None else _normalize_image_type_str(value) in SCIENCE_IMAGE_TYPES


def get_image_type_description(imgtype_value: str | None) -> str:
    """Return a human-readable description of an image type."""
    value = _require_str_or_none("imgtype_value", imgtype_value)
    if value is None:
        return "Unknown or missing image type."
    normalized = _normalize_image_type_str(value)
    return IMAGE_TYPE_DESCRIPTIONS.get(normalized, f"Unrecognised image type: {normalized!r}.")


__all__ = [
    "IMAGE_TYPE_ALIASES",
    "VALID_IMAGE_TYPE_STRINGS",
    "IMAGE_TYPE_DESCRIPTIONS",
    "normalize_image_type",
    "is_valid_image_type",
    "is_calibration_frame",
    "is_science_frame",
    "get_image_type_description",
]
