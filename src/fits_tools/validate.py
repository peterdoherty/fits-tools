#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""FITS-aware validation utilities for CCD/CMOS detector test data.

This module provides report-oriented validation of FITS files, headers,
image content, and calibration-frame semantics. It is intentionally layered
above the low-level :mod:`fits_tools.array_validation` module:

- :mod:`array_validation` raises exceptions for invalid scalar/array inputs.
- :mod:`validate` returns :class:`ValidationResult` objects describing
  whether a FITS file or file-set satisfies validation criteria.

The functions here remain FITS-aware and report-oriented rather than
duplicating low-level shape/range checking logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .array_validation import (
    validate_nonnegative,
    validate_numeric,
    validate_range,
    validate_same_ndim,
    validate_shape,
    validate_shapes,
)
from .constants import (
    ALL_IMAGE_TYPES,
    FITS_KEY_EXPTIME,
    FITS_KEY_IMAGETYP,
    FITS_KEY_IMGTYPE,
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
)
from .exceptions import InvalidParameterError
from .hdu import HDUSelector, resolve_hdu_index
from .image_types import is_valid_image_type, normalize_image_type  # see Note below
from .types import StrOrPath

# NOTE: fits_tools.image_types has not yet been reviewed. The functions
# is_valid_image_type and normalize_image_type are imported here and used
# throughout this module. Review image_types.py before finalizing validate.py.

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationResult",
    "validate_fits_file",
    "check_fits_integrity",
    "validate_required_keywords",
    "validate_header_values",
    "validate_keyword_types",
    "validate_data_range",
    "validate_no_nans",
    "validate_image_dimensions",
    "validate_frame_type",
    "validate_bias_frame",
    "validate_dark_frame",
    "validate_flat_frame",
    "validate_exposure_time",
    "validate_frame_set",
    "validate_calibration_set",
    "validate_against_schema",
    "quick_validate",
]


@dataclass
class ValidationResult:
    """Structured result for a validation operation.

    Attributes
    ----------
    is_valid : bool
        Overall success state. Set to False automatically when
        :meth:`add_error` is called.
    errors : list of str
        Critical validation failures.
    warnings : list of str
        Non-fatal validation concerns.
    info : dict
        Supplemental validation metadata.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add a validation error and mark the result invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a non-fatal validation warning."""
        self.warnings.append(message)

    def add_info(self, key: str, value: Any) -> None:
        """Add supplemental metadata to the validation result."""
        self.info[key] = value

    def merge(self, other: ValidationResult, prefix: str = "") -> None:
        """Merge another validation result into this one.

        Parameters
        ----------
        other : ValidationResult
            Result to merge.
        prefix : str, optional
            If non-empty, prepend ``"{prefix}: "`` to all merged messages.
        """
        if not other.is_valid:
            self.is_valid = False

        if prefix:
            self.errors.extend(f"{prefix}: {msg}" for msg in other.errors)
            self.warnings.extend(f"{prefix}: {msg}" for msg in other.warnings)
        else:
            self.errors.extend(other.errors)
            self.warnings.extend(other.warnings)

        self.info.update(other.info)

    def __str__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines = [f"Validation Status: {status}"]

        if self.errors:
            lines.append("")
            lines.append(f"Errors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, start=1):
                lines.append(f"  {i}. {error}")

        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, start=1):
                lines.append(f"  {i}. {warning}")

        if self.info:
            lines.append("")
            lines.append("Additional Info:")
            for key, value in self.info.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            "ValidationResult("
            f"is_valid={self.is_valid}, "
            f"errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )


def _open_selected_hdu(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
    *,
    memmap: bool = True,
) -> tuple[ValidationResult, fits.Header | None, np.ndarray | None]:
    """Open a FITS file and return ``(result, header, data)`` for one HDU.

    Any operational or selector failures are converted into a failed
    :class:`ValidationResult` rather than propagated as exceptions.
    """
    result = ValidationResult()
    path = Path(filepath)

    if not path.exists():
        result.add_error(f"File not found: {path}")
        return result, None, None

    if path.stat().st_size == 0:
        result.add_error(f"File is empty: {path}")
        return result, None, None

    try:
        with fits.open(path, memmap=memmap, mode="readonly") as hdul:
            hdu_index = resolve_hdu_index(
                hdul,
                hdu,
                auto="primary",
                image_like=False,
                require_data=False,
                filepath=path,
            )
            header = hdul[hdu_index].header.copy()
            data = hdul[hdu_index].data
            result.add_info("filepath", path)
            result.add_info("hdu", hdu_index)
            result.add_info("num_hdus", len(hdul))
            return result, header, data
    except Exception as exc:
        result.add_error(f"Failed to open FITS file or resolve HDU {hdu!r}: {exc}")
        return result, None, None


def _get_frame_type_from_header(
    header: fits.Header,
) -> tuple[str | None, str | None]:
    """Return ``(normalized, raw)`` image type from a FITS header."""
    for keyword in (FITS_KEY_IMGTYPE, FITS_KEY_IMAGETYP):
        if keyword in header:
            raw = str(header[keyword]).strip()
            if not raw:
                return None, None
            return normalize_image_type(raw), raw
    return None, None


def validate_fits_file(filepath: StrOrPath) -> ValidationResult:
    """Validate basic FITS file integrity.

    Checks that the file exists, is non-empty, can be opened by Astropy,
    and contains at least one HDU.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()
    path = Path(filepath)

    if not path.exists():
        result.add_error(f"File not found: {path}")
        return result

    result.add_info("filepath", path)
    result.add_info("filesize_kb", path.stat().st_size / 1024.0)

    if path.stat().st_size == 0:
        result.add_error(f"File is empty: {path}")
        return result

    try:
        with path.open("rb"):
            pass
    except PermissionError:
        result.add_error(f"File is not readable: {path}")
        return result
    except OSError as exc:
        result.add_error(f"Could not access file {path}: {exc}")
        return result

    try:
        with fits.open(path, memmap=True, mode="readonly") as hdul:
            if len(hdul) == 0:
                result.add_error("FITS file contains no HDUs")
                return result

            result.add_info("num_hdus", len(hdul))

            try:
                hdul.verify("fix")
            except Exception as exc:
                result.add_warning(f"FITS verification issues: {exc}")

            logger.debug("Basic FITS validation passed for %s", path)
    except Exception as exc:
        result.add_error(f"Failed to open as valid FITS file: {exc}")

    return result


def check_fits_integrity(
    filepath: StrOrPath,
    fix: bool = False,
) -> ValidationResult:
    """Run Astropy FITS verification on a file.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    fix : bool, optional
        If True, attempt to fix minor FITS standard violations in memory.
        Default is False.

    Returns
    -------
    ValidationResult
    """
    path = Path(filepath)
    result = validate_fits_file(path)
    if not result.is_valid:
        return result

    try:
        with fits.open(path, memmap=True, mode="readonly") as hdul:
            verify_output = hdul.verify("fix" if fix else "exception")
            if verify_output:
                for issue in verify_output:
                    result.add_warning(str(issue))
            result.add_info("integrity_check", "passed")
    except Exception as exc:
        result.add_error(f"FITS integrity check failed: {exc}")

    return result


def validate_required_keywords(
    filepath: StrOrPath,
    required_keywords: list[str],
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate that required header keywords are present.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    required_keywords : list of str
        Keywords that must be present in the header.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, header, _ = _open_selected_hdu(filepath, hdu=hdu)
    if header is None:
        return result

    missing = [kw for kw in required_keywords if kw not in header]
    present = [kw for kw in required_keywords if kw in header]

    if missing:
        result.add_error(f"Missing required keywords: {missing}")

    result.add_info("required_keywords", required_keywords)
    result.add_info("present_keywords", present)
    result.add_info("missing_keywords", missing)
    return result


def validate_header_values(
    filepath: StrOrPath,
    keyword_constraints: dict[str, Any],
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate header keyword values against user-provided constraints.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    keyword_constraints : dict
        Mapping of keyword name to constraint. Constraints may be:

        - A callable ``f(value) -> bool``.
        - A 2-element numeric tuple ``(min, max)`` for range checking.
        - A list, tuple, set, or frozenset of allowed values.
        - A scalar value for exact equality checking.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, header, _ = _open_selected_hdu(filepath, hdu=hdu)
    if header is None:
        return result

    for keyword, constraint in keyword_constraints.items():
        if keyword not in header:
            result.add_error(f"Keyword {keyword!r} not found in header")
            continue

        value = header[keyword]

        if callable(constraint):
            try:
                if not constraint(value):
                    result.add_error(
                        f"Keyword {keyword!r} value {value!r} failed validation"
                    )
            except Exception as exc:
                result.add_error(
                    f"Validation function for {keyword!r} raised an error: {exc}"
                )
            continue

        if (
            isinstance(constraint, (list, tuple))
            and len(constraint) == 2
            and all(isinstance(item, (int, float)) for item in constraint)
        ):
            minimum, maximum = constraint
            try:
                numeric_value = validate_numeric(
                    value,
                    label=keyword,
                    minimum=float(minimum),
                    maximum=float(maximum),
                )
            except InvalidParameterError as exc:
                result.add_error(str(exc))
            else:
                result.add_info(f"{keyword}_validated_value", numeric_value)
            continue

        if isinstance(constraint, (list, tuple, set, frozenset)):
            if value not in constraint:
                result.add_error(
                    f"Keyword {keyword!r} value {value!r} not in valid "
                    f"values: {constraint}"
                )
            continue

        if value != constraint:
            result.add_error(
                f"Keyword {keyword!r} value {value!r} does not match "
                f"expected {constraint!r}"
            )

    return result


def validate_keyword_types(
    filepath: StrOrPath,
    keyword_types: dict[str, type],
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate that specified header keywords have the expected Python types.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    keyword_types : dict of str to type
        Mapping of keyword name to expected Python type.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, header, _ = _open_selected_hdu(filepath, hdu=hdu)
    if header is None:
        return result

    for keyword, expected_type in keyword_types.items():
        if keyword not in header:
            result.add_error(f"Keyword {keyword!r} not found")
            continue

        value = header[keyword]
        if not isinstance(value, expected_type):
            result.add_error(
                f"Keyword {keyword!r} has type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )

    return result


def validate_data_range(
    filepath: StrOrPath,
    min_value: float | None = None,
    max_value: float | None = None,
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate that image data values fall within an expected scalar range.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    min_value : float, optional
        Minimum acceptable pixel value.
    max_value : float, optional
        Maximum acceptable pixel value.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, _, data = _open_selected_hdu(filepath, hdu=hdu)
    if data is None:
        if result.is_valid:
            result.add_error(f"HDU {result.info.get('hdu', hdu)!r} contains no data")
        return result

    if data.size == 0:
        result.add_error("Data array is empty")
        return result

    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    data_mean = float(np.nanmean(data))
    data_std = float(np.nanstd(data))

    result.add_info("data_min", data_min)
    result.add_info("data_max", data_max)
    result.add_info("data_mean", data_mean)
    result.add_info("data_std", data_std)

    try:
        if min_value is not None:
            validate_range(data_min, name="data minimum", minimum=min_value)
        if max_value is not None:
            validate_range(data_max, name="data maximum", maximum=max_value)
    except InvalidParameterError as exc:
        result.add_error(str(exc))

    return result


def validate_no_nans(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Check for NaN and Inf values in image data.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, _, data = _open_selected_hdu(filepath, hdu=hdu)
    if data is None:
        if result.is_valid:
            result.add_error(f"HDU {result.info.get('hdu', hdu)!r} contains no data")
        return result

    if np.issubdtype(data.dtype, np.floating):
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
    else:
        nan_count = 0
        inf_count = 0

    result.add_info("nan_count", nan_count)
    result.add_info("inf_count", inf_count)
    result.add_info("total_pixels", int(data.size))

    if nan_count > 0:
        result.add_error(f"Found {nan_count} NaN values in data")
    if inf_count > 0:
        result.add_error(f"Found {inf_count} Inf values in data")

    return result


def validate_image_dimensions(
    filepath: StrOrPath,
    expected_shape: tuple[int, int] | None = None,
    min_shape: tuple[int, int] | None = None,
    max_shape: tuple[int, int] | None = None,
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate image dimensionality and row/column bounds for a FITS HDU.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    expected_shape : tuple of int, optional
        Required ``(rows, cols)`` shape. If None, shape is not checked.
    min_shape : tuple of int, optional
        Minimum ``(rows, cols)`` dimensions.
    max_shape : tuple of int, optional
        Maximum ``(rows, cols)`` dimensions.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, _, data = _open_selected_hdu(filepath, hdu=hdu)
    if data is None:
        if result.is_valid:
            result.add_error(f"HDU {result.info.get('hdu', hdu)!r} contains no data")
        return result

    result.add_info("shape", data.shape)
    result.add_info("dimensions", data.ndim)

    try:
        validate_shape(
            data,
            expected_ndim=2,
            expected_shape=expected_shape,
            operation="validate_image_dimensions",
        )

        rows, cols = data.shape
        if min_shape is not None:
            min_rows, min_cols = min_shape
            validate_range(rows, name="image rows", minimum=min_rows)
            validate_range(cols, name="image cols", minimum=min_cols)

        if max_shape is not None:
            max_rows, max_cols = max_shape
            validate_range(rows, name="image rows", maximum=max_rows)
            validate_range(cols, name="image cols", maximum=max_cols)
    except Exception as exc:
        result.add_error(str(exc))

    return result


def validate_frame_type(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate that ``IMGTYPE``/``IMAGETYP`` contains a recognized frame type.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, header, _ = _open_selected_hdu(filepath, hdu=hdu)
    if header is None:
        return result

    normalized, raw = _get_frame_type_from_header(header)
    if raw is None:
        result.add_error("Required keyword 'IMGTYPE' or 'IMAGETYP' not found")
        return result

    result.add_info("frame_type", normalized)
    result.add_info("frame_type_raw", raw)

    if normalized is None or not is_valid_image_type(normalized):
        result.add_warning(
            f"Frame type {raw!r} not in standard types: {list(ALL_IMAGE_TYPES)}"
        )

    return result


def validate_bias_frame(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate that a frame is consistent with a BIAS exposure.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, header, data = _open_selected_hdu(filepath, hdu=hdu)
    if header is None:
        return result

    imagetype, _ = _get_frame_type_from_header(header)
    if imagetype is None:
        result.add_error("Missing IMGTYPE/IMAGETYP keyword")
    elif imagetype != FITS_VAL_BIAS_IMGTYPE:
        result.add_error(f"Expected IMGTYPE='BIAS', got {imagetype!r}")

    if FITS_KEY_EXPTIME not in header:
        result.add_warning(f"Missing keyword {FITS_KEY_EXPTIME!r}")
    else:
        exptime = header[FITS_KEY_EXPTIME]
        result.add_info("exptime", exptime)
        try:
            validate_nonnegative(exptime, name=FITS_KEY_EXPTIME)
            if float(exptime) > 0.001:
                result.add_error(f"BIAS frame should have EXPTIME≈0, got {exptime}")
        except InvalidParameterError as exc:
            result.add_error(str(exc))

    if data is not None:
        data_max = float(np.nanmax(data))
        if data_max > 10000:
            result.add_warning(
                f"BIAS frame has unusually high maximum value: {data_max}"
            )
        result.add_info("data_mean", float(np.nanmean(data)))
        result.add_info("data_std", float(np.nanstd(data)))

    return result


def validate_dark_frame(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
    min_exptime: float = 0.1,
) -> ValidationResult:
    """Validate that a frame is consistent with a DARK exposure.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.
    min_exptime : float, optional
        Minimum acceptable exposure time in seconds. Default is 0.1.

    Returns
    -------
    ValidationResult
    """
    result, header, data = _open_selected_hdu(filepath, hdu=hdu)
    if header is None:
        return result

    imagetype, _ = _get_frame_type_from_header(header)
    if imagetype is None:
        result.add_error("Missing IMGTYPE/IMAGETYP keyword")
    elif imagetype != FITS_VAL_DARK_IMGTYPE:
        result.add_error(f"Expected IMGTYPE='DARK', got {imagetype!r}")

    if FITS_KEY_EXPTIME not in header:
        result.add_error(f"Missing keyword {FITS_KEY_EXPTIME!r}")
    else:
        exptime = header[FITS_KEY_EXPTIME]
        result.add_info("exptime", exptime)
        try:
            validate_range(exptime, name=FITS_KEY_EXPTIME, minimum=min_exptime)
        except InvalidParameterError as exc:
            result.add_error(str(exc))

    if data is not None:
        result.add_info("data_mean", float(np.nanmean(data)))
        result.add_info("data_std", float(np.nanstd(data)))
        result.add_info("data_max", float(np.nanmax(data)))

    return result


def validate_flat_frame(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
    min_counts: float = 1000,
    max_counts: float = 50000,
) -> ValidationResult:
    """Validate that a frame is consistent with a FLAT exposure.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.
    min_counts : float, optional
        Minimum acceptable mean count level. Default is 1000.
    max_counts : float, optional
        Maximum acceptable mean count level. Default is 50000.

    Returns
    -------
    ValidationResult
    """
    result, header, data = _open_selected_hdu(filepath, hdu=hdu)
    if header is None:
        return result

    imagetype, _ = _get_frame_type_from_header(header)
    if imagetype is None:
        result.add_error("Missing IMGTYPE/IMAGETYP keyword")
    elif imagetype != FITS_VAL_FLAT_IMGTYPE:
        result.add_error(f"Expected IMGTYPE='FLAT', got {imagetype!r}")

    if data is not None:
        try:
            validate_shape(data, expected_ndim=2, operation="validate_flat_frame")
        except Exception as exc:
            result.add_error(str(exc))
            return result

        data_mean = float(np.nanmean(data))
        data_median = float(np.nanmedian(data))
        data_std = float(np.nanstd(data))

        result.add_info("data_mean", data_mean)
        result.add_info("data_median", data_median)
        result.add_info("data_std", data_std)

        try:
            validate_range(data_mean, name="flat mean counts", minimum=min_counts)
            validate_range(data_mean, name="flat mean counts", maximum=max_counts)
        except InvalidParameterError as exc:
            result.add_error(str(exc))

        if data_mean > 0:
            cv = data_std / data_mean
            result.add_info("coefficient_of_variation", float(cv))
            if cv > 0.5:
                result.add_warning(f"FLAT frame has high non-uniformity (CV={cv:.3f})")
    else:
        result.add_error("FLAT frame contains no data")

    return result


def validate_exposure_time(
    filepath: StrOrPath,
    expected_exptime: float | None = None,
    tolerance: float = 0.01,
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate the exposure-time keyword for one FITS HDU.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    expected_exptime : float, optional
        Expected exposure time in seconds. If None, only the presence and
        non-negativity of EXPTIME is checked.
    tolerance : float, optional
        Allowed absolute difference from *expected_exptime* in seconds.
        Default is 0.01.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result, header, _ = _open_selected_hdu(filepath, hdu=hdu)
    if header is None:
        return result

    try:
        validate_nonnegative(tolerance, name="tolerance")
    except InvalidParameterError as exc:
        result.add_error(str(exc))
        return result

    if FITS_KEY_EXPTIME not in header:
        result.add_error(f"Missing keyword {FITS_KEY_EXPTIME!r}")
        return result

    exptime = header[FITS_KEY_EXPTIME]
    result.add_info("exptime", exptime)

    try:
        validate_nonnegative(exptime, name=FITS_KEY_EXPTIME)
        if expected_exptime is not None:
            validate_nonnegative(expected_exptime, name="expected_exptime")
            diff = abs(float(exptime) - float(expected_exptime))
            if diff > tolerance:
                result.add_error(
                    f"Exposure time {exptime} differs from expected "
                    f"{expected_exptime} by {diff} (tolerance: {tolerance})"
                )
    except InvalidParameterError as exc:
        result.add_error(str(exc))

    return result


def validate_frame_set(
    filepaths: list[StrOrPath],
    check_dimensions: bool = True,
    check_frame_type: bool = True,
    expected_frame_type: str | None = None,
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate that a set of FITS frames is internally consistent.

    Parameters
    ----------
    filepaths : list of str or Path
        Paths to the FITS files.
    check_dimensions : bool, optional
        If True, verify all frames have the same shape. Default is True.
    check_frame_type : bool, optional
        If True, verify all frames share the same frame type. Default is
        True.
    expected_frame_type : str, optional
        If provided, verify that the common frame type matches this value.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    if not filepaths:
        result.add_error("Empty file list provided")
        return result

    result.add_info("num_files", len(filepaths))

    shapes: list[tuple[int, ...]] = []
    frame_types: list[str] = []
    exptimes: list[float] = []
    arrays_for_shape_check: list[np.ndarray] = []

    for filepath in filepaths:
        path = Path(filepath)
        hdu_result, header, data = _open_selected_hdu(path, hdu=hdu)
        if header is None:
            result.merge(hdu_result, prefix=str(path))
            continue

        if data is not None:
            shapes.append(data.shape)
            arrays_for_shape_check.append(np.empty(data.shape, dtype=np.uint8))
        else:
            result.add_warning(f"No data in {path}")

        imagetype, _ = _get_frame_type_from_header(header)
        if imagetype is not None:
            frame_types.append(imagetype)

        if FITS_KEY_EXPTIME in header:
            try:
                exptimes.append(
                    float(
                        validate_numeric(
                            header[FITS_KEY_EXPTIME], label=FITS_KEY_EXPTIME
                        )
                    )
                )
            except InvalidParameterError as exc:
                result.add_error(f"{path}: {exc}")

    if check_dimensions and len(arrays_for_shape_check) >= 2:
        try:
            validate_same_ndim(*arrays_for_shape_check, operation="validate_frame_set")
            validate_shapes(*arrays_for_shape_check, operation="validate_frame_set")
            result.add_info("image_shape", shapes[0])
        except Exception as exc:
            result.add_error(f"Inconsistent image dimensions in set: {exc}")

    if check_frame_type and frame_types:
        unique_types = set(frame_types)
        if len(unique_types) > 1:
            result.add_error(f"Inconsistent frame types in set: {unique_types}")
        else:
            result.add_info("frame_type", frame_types[0])

        if expected_frame_type is not None:
            expected = normalize_image_type(expected_frame_type)
            if expected not in unique_types:
                result.add_error(
                    f"Expected frame type {expected!r}, found: {unique_types}"
                )

    if exptimes:
        result.add_info("exptime_min", min(exptimes))
        result.add_info("exptime_max", max(exptimes))
        result.add_info("exptime_mean", float(np.mean(exptimes)))
        if len(set(exptimes)) > 1:
            result.add_warning(f"Exposure times vary in set: {set(exptimes)}")

    return result


def validate_calibration_set(
    bias_files: list[StrOrPath] | None = None,
    dark_files: list[StrOrPath] | None = None,
    flat_files: list[StrOrPath] | None = None,
) -> ValidationResult:
    """Validate a complete bias/dark/flat calibration set.

    Parameters
    ----------
    bias_files : list of str or Path, optional
        Bias frame paths.
    dark_files : list of str or Path, optional
        Dark frame paths.
    flat_files : list of str or Path, optional
        Flat field frame paths.

    Returns
    -------
    ValidationResult
        Merged result from all sub-validations, plus a cross-set shape
        consistency check.
    """
    result = ValidationResult()
    all_shapes: list[tuple[int, ...]] = []

    if bias_files:
        bias_result = validate_frame_set(
            bias_files,
            check_frame_type=True,
            expected_frame_type=FITS_VAL_BIAS_IMGTYPE,
        )
        result.merge(bias_result, prefix="BIAS")
        result.add_info("num_bias_frames", len(bias_files))
        if "image_shape" in bias_result.info:
            all_shapes.append(tuple(bias_result.info["image_shape"]))

    if dark_files:
        dark_result = validate_frame_set(
            dark_files,
            check_frame_type=True,
            expected_frame_type=FITS_VAL_DARK_IMGTYPE,
        )
        result.merge(dark_result, prefix="DARK")
        result.add_info("num_dark_frames", len(dark_files))
        if "image_shape" in dark_result.info:
            all_shapes.append(tuple(dark_result.info["image_shape"]))

    if flat_files:
        flat_result = validate_frame_set(
            flat_files,
            check_frame_type=True,
            expected_frame_type=FITS_VAL_FLAT_IMGTYPE,
        )
        result.merge(flat_result, prefix="FLAT")
        result.add_info("num_flat_frames", len(flat_files))
        if "image_shape" in flat_result.info:
            all_shapes.append(tuple(flat_result.info["image_shape"]))

    if all_shapes:
        unique_shapes = set(all_shapes)
        if len(unique_shapes) > 1:
            result.add_error(
                f"Calibration frames have inconsistent dimensions: {unique_shapes}"
            )
        else:
            result.add_info("calibration_shape", all_shapes[0])

    return result


def validate_against_schema(
    filepath: StrOrPath,
    schema: dict[str, Any],
    hdu: HDUSelector = 0,
) -> ValidationResult:
    """Validate a FITS file against a flexible configuration schema.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    schema : dict
        Validation schema. Recognized keys:

        - ``'required_keywords'`` — list of str.
        - ``'keyword_values'`` — dict of constraints; see
          :func:`validate_header_values`.
        - ``'keyword_types'`` — dict of ``{keyword: type}``.
        - ``'data_range'`` — dict with optional ``'min'`` and ``'max'``.
        - ``'dimensions'`` — dict with optional ``'expected'``, ``'min'``,
          ``'max'`` shape tuples.
        - ``'no_nans'`` — bool; if True, check for NaN/Inf.
    hdu : HDUSelector, optional
        HDU selector. Default is 0.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()
    result.add_info("schema_validation", True)

    if "required_keywords" in schema:
        result.merge(
            validate_required_keywords(filepath, schema["required_keywords"], hdu=hdu)
        )

    if "keyword_values" in schema:
        result.merge(
            validate_header_values(filepath, schema["keyword_values"], hdu=hdu)
        )

    if "keyword_types" in schema:
        result.merge(validate_keyword_types(filepath, schema["keyword_types"], hdu=hdu))

    if "data_range" in schema:
        range_config = schema["data_range"]
        result.merge(
            validate_data_range(
                filepath,
                min_value=range_config.get("min"),
                max_value=range_config.get("max"),
                hdu=hdu,
            )
        )

    if "dimensions" in schema:
        dim_config = schema["dimensions"]
        result.merge(
            validate_image_dimensions(
                filepath,
                expected_shape=dim_config.get("expected"),
                min_shape=dim_config.get("min"),
                max_shape=dim_config.get("max"),
                hdu=hdu,
            )
        )

    if schema.get("no_nans", False):
        result.merge(validate_no_nans(filepath, hdu=hdu))

    return result


def quick_validate(
    filepath: StrOrPath,
    frame_type: str | None = None,
) -> ValidationResult:
    """Perform sensible default validation for CCD/CMOS FITS test data.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    frame_type : str, optional
        Expected frame type (e.g. ``'BIAS'``, ``'DARK'``, ``'FLAT'``).
        If None, the frame type is read from the header.

    Returns
    -------
    ValidationResult
        Combined result of file integrity, frame-type, and NaN checks.
        For bias/dark/flat frames, the appropriate frame-specific
        validation is also included.
    """
    path = Path(filepath)
    result = validate_fits_file(path)
    if not result.is_valid:
        return result

    if frame_type is None:
        type_result = validate_frame_type(path)
        result.merge(type_result)
        frame_type = type_result.info.get("frame_type")

    if frame_type is not None:
        normalized = normalize_image_type(frame_type)
        result.add_info("frame_type", normalized)
        if normalized == FITS_VAL_BIAS_IMGTYPE:
            result.merge(validate_bias_frame(path))
        elif normalized == FITS_VAL_DARK_IMGTYPE:
            result.merge(validate_dark_frame(path))
        elif normalized == FITS_VAL_FLAT_IMGTYPE:
            result.merge(validate_flat_frame(path))

    result.merge(validate_no_nans(path))
    return result
