#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Array- and parameter-level validation utilities for fits_tools.

These helpers are intentionally FITS-agnostic and operate purely on
NumPy arrays and scalar parameters. They are used across the package
to enforce basic shape, dimensionality, and value constraints before
performing image-processing operations.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .exceptions import (
    DataShapeError,
    DimensionalityError,
    InvalidParameterError,
)


# =============================================================================
# Index / Scalar Validation
# =============================================================================

def validate_indices(
    indices: Sequence[int],
    max_index: int,
    min_index: int = 0,
    label: str = "index",
) -> None:
    """Validate that all indices are within a valid range.

    Parameters
    ----------
    indices :
        Sequence of integer indices to validate.
    max_index :
        Maximum valid index (exclusive).
    min_index :
        Minimum valid index (inclusive). Default is 0.
    label :
        Human-readable label for error messages (e.g. ``'row'``, ``'col'``).

    Raises
    ------
    InvalidParameterError
        If any index is outside ``[min_index, max_index)``.
    """
    for idx in indices:
        if not isinstance(idx, int):
            raise InvalidParameterError(
                f"{label} indices must be integers, got {type(idx).__name__}",
                parameter=label,
                value=idx,
                expected="int",
            )
        if idx < min_index or idx >= max_index:
            raise InvalidParameterError(
                f"{label} {idx} out of range [{min_index}, {max_index})",
                parameter=label,
                value=idx,
                expected=f"in range [{min_index}, {max_index})",
            )


from typing import Any, Optional, Union

import numpy as np

from .exceptions import InvalidParameterError

Number = Union[int, float, np.number]


def validate_numeric(
    value: Any,
    label: str = "value",
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    integer: bool = False,
    allow_zero: bool = True,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> Number:
    """
    Validate and normalize a scalar numeric parameter.

    This function converts *value* to a Python number (int or float),
    applies a series of optional constraints, and returns the normalized
    value. It is intended for scalar parameters, not arrays.

    Parameters
    ----------
    value :
        Value to validate (scalar: int, float, numpy scalar, or string).
    label :
        Parameter name used in error messages.
    minimum :
        If not None, require ``value >= minimum``.
    maximum :
        If not None, require ``value <= maximum``.
    integer :
        If True, require that *value* is an integer (no fractional part)
        and return it as ``int``.
    allow_zero :
        If False, reject ``value == 0``.
    allow_nan :
        If False, reject NaN values.
    allow_inf :
        If False, reject +/- infinity.

    Returns
    -------
    int or float
        The validated numeric value, converted to ``int`` if
        ``integer=True``, otherwise to ``float``.

    Raises
    ------
    InvalidParameterError
        If the value is not numeric or violates any of the constraints.
    """
    # Convert to float first
    try:
        # Handle numpy scalars: np.asarray(...).item() gives Python scalar
        if isinstance(value, (np.generic,)):
            py_value = value.item()
        else:
            py_value = value

        val = float(py_value)
    except (TypeError, ValueError):
        raise InvalidParameterError(
            f"{label} must be numeric, got {type(value).__name__}",
            parameter=label,
            value=value,
            expected="numeric (int or float)",
        )

    # NaN / Inf checks
    if np.isnan(val):
        if not allow_nan:
            raise InvalidParameterError(
                f"{label} must not be NaN",
                parameter=label,
                value=value,
                expected="non-NaN numeric",
            )
        # If NaN is allowed, we skip further numeric comparisons
        return val

    if np.isinf(val):
        if not allow_inf:
            raise InvalidParameterError(
                f"{label} must not be infinite, got {val}",
                parameter=label,
                value=value,
                expected="finite numeric",
            )
        # If Inf is allowed, we skip range checks but still allow integer coercion
        if integer:
            raise InvalidParameterError(
                f"{label} cannot be infinite when integer=True",
                parameter=label,
                value=value,
                expected="finite integer",
            )
        return val

    # Zero constraint
    if not allow_zero and val == 0:
        raise InvalidParameterError(
            f"{label} must be non-zero, got 0",
            parameter=label,
            value=value,
            expected="!= 0",
        )

    # Range checks
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

    # Integer constraint
    if integer:
        if not float(val).is_integer():
            raise InvalidParameterError(
                f"{label} must be an integer, got {val}",
                parameter=label,
                value=value,
                expected="integer",
            )
        return int(round(val))

    return val


def validate_positive(value: Union[int, float], name: str = "value") -> None:
    """Validate that a numeric value is strictly positive.

    Parameters
    ----------
    value :
        Value to check.
    name :
        Parameter name used in error messages.

    Raises
    ------
    InvalidParameterError
        If ``value <= 0``.
    """
    if value <= 0:
        raise InvalidParameterError(
            f"{name} must be positive, got {value}",
            parameter=name,
            value=value,
            expected="> 0",
        )


# def validate_nonnegative(value: Union[int, float], name: str = "value") -> None:
#     ...


# def validate_range(
#     value: Union[int, float],
#     min_value: Optional[float] = None,
#     max_value: Optional[float] = None,
#     name: str = "value",
# ) -> None:
#     ...


# =============================================================================
# Array Shape / Dimensionality Validation
# =============================================================================

# def validate_axis(axis: int, ndim: int, name: str = "axis") -> None:
#     validate_indices([axis], max_index=ndim, label=name)


def validate_shape(
    array: np.ndarray,
    expected_ndim: Optional[int] = None,
    min_ndim: Optional[int] = None,
    max_ndim: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    operation: Optional[str] = None,
) -> None:
    """Validate array dimensionality and/or exact shape.

    Parameters
    ----------
    array :
        Array to validate.
    expected_ndim :
        Required number of dimensions. If given and ``array.ndim`` differs,
        a :class:`DimensionalityError` is raised.
    min_ndim :
        Minimum allowed number of dimensions.
    max_ndim :
        Maximum allowed number of dimensions.
    expected_shape :
        Exact required shape. If given and ``array.shape`` differs, a
        :class:`DataShapeError` is raised.
    operation :
        Optional short name of the operation performing the check, used
        in error messages for additional context.

    Raises
    ------
    DimensionalityError
        If ``expected_ndim`` is provided and ``array.ndim`` does not match,
        or if ``array.ndim`` falls outside ``[min_ndim, max_ndim]``.
    DataShapeError
        If ``expected_shape`` is provided and ``array.shape`` does not match.
    """
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
            f"Array has {array.ndim} dimensions in {op}, "
            f"minimum {min_ndim} required",
            ndim=array.ndim,
            required_ndim=min_ndim,
            operation=op,
        )

    if max_ndim is not None and array.ndim > max_ndim:
        raise DimensionalityError(
            f"Array has {array.ndim} dimensions in {op}, "
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


def validate_shapes(*arrays: np.ndarray, operation: str = "operation") -> None:
    """Validate that all arrays have the same shape.

    Parameters
    ----------
    *arrays :
        Arrays to validate. If fewer than two arrays are provided, the
        function returns without error.
    operation :
        Name of the higher-level operation for error messages, e.g.
        ``'combining'``, ``'stacking'``.

    Raises
    ------
    DataShapeError
        If any array's shape differs from that of the first array.

    Examples
    --------
    >>> a = np.zeros((100, 100))
    >>> b = np.zeros((100, 100))
    >>> validate_shapes(a, b, operation="combining")  # OK
    >>> c = np.zeros((50, 50))
    >>> validate_shapes(a, c, operation="combining")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    fits_tools.exceptions.DataShapeError: ...
    """
    if len(arrays) < 2:
        return

    ref_shape = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape != ref_shape:
            raise DataShapeError(
                f"Shape mismatch in {operation}: "
                f"array 0 has shape {ref_shape}, array {i} has shape {arr.shape}",
                expected_shape=ref_shape,
                actual_shape=arr.shape,
                operation=operation,
            )


__all__ = [
    "validate_indices",
    "validate_numeric",
    "validate_positive",
    "validate_shape",
    "validate_shapes",
]