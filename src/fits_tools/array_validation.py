#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Array- and scalar-level validation utilities for fits_tools.

This module provides FITS-agnostic validation helpers for NumPy arrays and
scalar parameters. Functions here are intentionally low-level and raise
package exceptions immediately on failure. Higher-level FITS-aware
validation and reporting belongs in :mod:`fits_tools.validate`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import DataShapeError, DimensionalityError, InvalidParameterError

Number = int | float | np.number

__all__ = [
    "Number",
    "validate_indices",
    "validate_numeric",
    "validate_positive",
    "validate_nonnegative",
    "validate_range",
    "validate_axis",
    "validate_shape",
    "validate_same_ndim",
    "validate_shapes",
]


# =============================================================================
# Index / scalar validation
# =============================================================================


def validate_indices(
    indices: list[int] | tuple[int, ...] | range,
    max_index: int,
    min_index: int = 0,
    label: str = "index",
) -> None:
    """Validate that all indices lie in ``[min_index, max_index)``.

    Parameters
    ----------
    indices : list, tuple, or range of int
        Sequence of integer indices to validate.
    max_index : int
        Upper bound of the valid range, exclusive.
    min_index : int, optional
        Lower bound of the valid range, inclusive. Default is 0.
    label : str, optional
        Human-readable label used in error messages. Default is
        ``"index"``.

    Raises
    ------
    InvalidParameterError
        If the valid range is empty, if any index is not an integer, or
        if any index is outside ``[min_index, max_index)``.
    """
    if max_index <= min_index:
        raise InvalidParameterError(
            f"{label} validation range is empty: [{min_index}, {max_index})",
            parameter=label,
            value=(min_index, max_index),
            expected="non-empty half-open interval",
        )

    for idx in indices:
        if not isinstance(idx, (int, np.integer)):
            raise InvalidParameterError(
                f"{label} indices must be integers, got {type(idx).__name__}",
                parameter=label,
                value=idx,
                expected="int",
            )
        idx_int = int(idx)
        if idx_int < min_index or idx_int >= max_index:
            raise InvalidParameterError(
                f"{label} {idx_int} out of range [{min_index}, {max_index})",
                parameter=label,
                value=idx_int,
                expected=f"in range [{min_index}, {max_index})",
            )


def validate_numeric(
    value: Any,
    label: str = "value",
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
    allow_zero: bool = True,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> int | float:
    """Validate and normalize a scalar numeric parameter.

    Parameters
    ----------
    value : any
        Scalar value to validate. NumPy scalar types are accepted and
        converted via ``.item()``.
    label : str, optional
        Parameter name used in error messages. Default is ``"value"``.
    minimum : float, optional
        If provided, require ``value >= minimum``.
    maximum : float, optional
        If provided, require ``value <= maximum``.
    integer : bool, optional
        If True, require an integer-valued input and return ``int``.
        Default is False.
    allow_zero : bool, optional
        If False, reject zero. Default is True.
    allow_nan : bool, optional
        If False, reject NaN. Default is False.
    allow_inf : bool, optional
        If False, reject infinite values. Default is False.

    Returns
    -------
    int or float
        Validated scalar value, coerced to ``int`` when ``integer=True``.

    Raises
    ------
    InvalidParameterError
        If *value* is not numeric or violates any constraint.
    """
    try:
        py_value = value.item() if isinstance(value, np.generic) else value
        val = float(py_value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise InvalidParameterError(
            f"{label} must be numeric, got {type(value).__name__}",
            parameter=label,
            value=value,
            expected="numeric (int or float)",
        ) from exc

    if np.isnan(val):
        if not allow_nan:
            raise InvalidParameterError(
                f"{label} must not be NaN",
                parameter=label,
                value=value,
                expected="non-NaN numeric",
            )
        return val

    if np.isinf(val):
        if not allow_inf:
            raise InvalidParameterError(
                f"{label} must not be infinite, got {val}",
                parameter=label,
                value=value,
                expected="finite numeric",
            )
        if integer:
            raise InvalidParameterError(
                f"{label} cannot be infinite when integer=True",
                parameter=label,
                value=value,
                expected="finite integer",
            )
        return val

    if not allow_zero and val == 0:
        raise InvalidParameterError(
            f"{label} must be non-zero, got 0",
            parameter=label,
            value=value,
            expected="!= 0",
        )

    if minimum is not None and val < minimum:
        raise InvalidParameterError(
            f"{label} must be >= {minimum}, got {val}",
            parameter=label,
            value=value,
            expected=f">= {minimum}",
        )

    if maximum is not None and val > maximum:
        raise InvalidParameterError(
            f"{label} must be <= {maximum}, got {val}",
            parameter=label,
            value=value,
            expected=f"<= {maximum}",
        )

    if integer:
        if not val.is_integer():
            raise InvalidParameterError(
                f"{label} must be an integer, got {val}",
                parameter=label,
                value=value,
                expected="integer",
            )
        return int(val)

    return val


def validate_positive(value: int | float, name: str = "value") -> None:
    """Validate that a numeric value is strictly positive (> 0).

    Parameters
    ----------
    value : int or float
        Value to validate.
    name : str, optional
        Parameter name used in error messages. Default is ``"value"``.

    Raises
    ------
    InvalidParameterError
        If *value* is not numeric, is zero, or is negative.
    """
    validate_numeric(value, label=name, minimum=0.0, allow_zero=False)


def validate_nonnegative(value: int | float, name: str = "value") -> None:
    """Validate that a numeric value is non-negative (>= 0).

    Parameters
    ----------
    value : int or float
        Value to validate.
    name : str, optional
        Parameter name used in error messages. Default is ``"value"``.

    Raises
    ------
    InvalidParameterError
        If *value* is not numeric or is negative.
    """
    validate_numeric(value, label=name, minimum=0.0)


def validate_range(
    value: int | float,
    minimum: float | None = None,
    maximum: float | None = None,
    name: str = "value",
    *,
    integer: bool = False,
    allow_zero: bool = True,
) -> int | float:
    """Validate a scalar against optional lower and upper bounds.

    Parameters
    ----------
    value : int or float
        Value to validate.
    minimum : float, optional
        If provided, require ``value >= minimum``.
    maximum : float, optional
        If provided, require ``value <= maximum``.
    name : str, optional
        Parameter name used in error messages. Default is ``"value"``.
    integer : bool, optional
        If True, require an integer-valued input. Default is False.
    allow_zero : bool, optional
        If False, reject zero. Default is True.

    Returns
    -------
    int or float
        Validated value.

    Raises
    ------
    InvalidParameterError
        If *value* violates any constraint.
    """
    return validate_numeric(
        value,
        label=name,
        minimum=minimum,
        maximum=maximum,
        integer=integer,
        allow_zero=allow_zero,
    )


def validate_axis(axis: int, ndim: int, name: str = "axis") -> int:
    """Validate and normalize an axis index.

    Negative axes are converted to their positive equivalents following
    NumPy convention (``-1`` → ``ndim - 1``, etc.).

    Parameters
    ----------
    axis : int
        Axis index to validate. May be negative.
    ndim : int
        Number of dimensions of the target array.
    name : str, optional
        Parameter name used in error messages. Default is ``"axis"``.

    Returns
    -------
    int
        Normalized (non-negative) axis index.

    Raises
    ------
    InvalidParameterError
        If *axis* is not an integer, if *ndim* is not a positive integer,
        or if *axis* is out of the valid range
        ``[-ndim, ndim - 1]``.
    """
    if not isinstance(axis, (int, np.integer)):
        raise InvalidParameterError(
            f"{name} must be an integer, got {type(axis).__name__}",
            parameter=name,
            value=axis,
            expected="int",
        )
    axis_int = int(axis)

    if not isinstance(ndim, (int, np.integer)) or int(ndim) <= 0:
        raise InvalidParameterError(
            f"ndim must be a positive integer, got {ndim!r}",
            parameter="ndim",
            value=ndim,
            expected="positive int",
        )
    ndim_int = int(ndim)

    normalized = axis_int + ndim_int if axis_int < 0 else axis_int
    if normalized < 0 or normalized >= ndim_int:
        raise InvalidParameterError(
            f"{name} {axis_int} out of range for array with "
            f"{ndim_int} dimension(s)",
            parameter=name,
            value=axis_int,
            expected=f"in range [-{ndim_int}, {ndim_int - 1}]",
        )
    return normalized


# =============================================================================
# Array shape / dimensionality validation
# =============================================================================


def validate_shape(
    array: np.ndarray,
    expected_ndim: int | None = None,
    min_ndim: int | None = None,
    max_ndim: int | None = None,
    expected_shape: tuple[int, ...] | None = None,
    operation: str | None = None,
) -> None:
    """Validate array dimensionality and/or exact shape.

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    expected_ndim : int, optional
        Required number of dimensions.
    min_ndim : int, optional
        Minimum acceptable number of dimensions.
    max_ndim : int, optional
        Maximum acceptable number of dimensions.
    expected_shape : tuple of int, optional
        Required exact shape. Checked after dimensionality constraints.
    operation : str, optional
        Name of the calling operation, used in error messages.

    Raises
    ------
    InvalidParameterError
        If *array* is not a :class:`numpy.ndarray`.
    DimensionalityError
        If the number of dimensions violates any constraint.
    DataShapeError
        If the shape does not match *expected_shape*.
    """
    if not isinstance(array, np.ndarray):
        raise InvalidParameterError(
            "array must be a numpy.ndarray",
            parameter="array",
            value=type(array).__name__,
            expected="numpy.ndarray",
        )

    op = operation or "validate_shape"

    if expected_ndim is not None and array.ndim != expected_ndim:
        raise DimensionalityError(
            f"Expected {expected_ndim}D array for {op}, got {array.ndim}D",
            ndim=array.ndim,
            required_ndim=expected_ndim,
            operation=op,
        )

    if min_ndim is not None and array.ndim < min_ndim:
        raise DimensionalityError(
            f"Array has {array.ndim} dimension(s) in {op}, "
            f"minimum {min_ndim} required",
            ndim=array.ndim,
            required_ndim=min_ndim,
            operation=op,
        )

    if max_ndim is not None and array.ndim > max_ndim:
        raise DimensionalityError(
            f"Array has {array.ndim} dimension(s) in {op}, "
            f"maximum {max_ndim} allowed",
            ndim=array.ndim,
            required_ndim=max_ndim,
            operation=op,
        )

    if expected_shape is not None and array.shape != expected_shape:
        raise DataShapeError(
            f"Expected shape {expected_shape} for {op}, got {array.shape}",
            expected_shape=expected_shape,
            actual_shape=array.shape,
            operation=op,
        )


def validate_same_ndim(
    *arrays: np.ndarray,
    operation: str = "operation",
) -> int:
    """Validate that all arrays have the same number of dimensions.

    Parameters
    ----------
    *arrays : np.ndarray
        Arrays to compare. At least one must be provided for a meaningful
        check; zero arrays returns 0 without error.
    operation : str, optional
        Name of the calling operation, used in error messages.
        Default is ``"operation"``.

    Returns
    -------
    int
        Common number of dimensions, or 0 if no arrays are provided.

    Raises
    ------
    DimensionalityError
        If any array has a different number of dimensions from the first.
    """
    if not arrays:
        return 0

    ref_ndim = arrays[0].ndim
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.ndim != ref_ndim:
            raise DimensionalityError(
                f"Dimensionality mismatch in {operation}: "
                f"array 0 has {ref_ndim}D, array {i} has {arr.ndim}D",
                ndim=arr.ndim,
                required_ndim=ref_ndim,
                operation=operation,
            )
    return ref_ndim


def validate_shapes(
    *arrays: np.ndarray,
    operation: str = "operation",
) -> None:
    """Validate that all arrays have identical shapes.

    Parameters
    ----------
    *arrays : np.ndarray
        Arrays to compare. Fewer than two arrays passes without error.
    operation : str, optional
        Name of the calling operation, used in error messages.
        Default is ``"operation"``.

    Raises
    ------
    DataShapeError
        If any array has a different shape from the first.
    """
    if len(arrays) < 2:
        return

    ref_shape = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape != ref_shape:
            raise DataShapeError(
                f"Shape mismatch in {operation}: "
                f"array 0 has shape {ref_shape}, "
                f"array {i} has shape {arr.shape}",
                expected_shape=ref_shape,
                actual_shape=arr.shape,
                operation=operation,
            )
