#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import numpy as np
import pytest

from fits_tools.array_validation import (
    validate_axis,
    validate_indices,
    validate_nonnegative,
    validate_numeric,
    validate_positive,
    validate_range,
    validate_same_ndim,
    validate_shape,
    validate_shapes,
)
from fits_tools.exceptions import DataShapeError, DimensionalityError, InvalidParameterError


def test_validate_indices_accepts_numpy_integer_types() -> None:
    validate_indices([np.int64(0), 1, 2], max_index=3)


def test_validate_indices_rejects_out_of_range_index() -> None:
    with pytest.raises(InvalidParameterError, match="out of range"):
        validate_indices([0, 3], max_index=3)


def test_validate_numeric_enforces_integer_and_range() -> None:
    assert validate_numeric(np.int32(4), label="value", integer=True) == 4

    with pytest.raises(InvalidParameterError, match="must be >= 0"):
        validate_numeric(-1, label="gain", minimum=0)


def test_validate_numeric_rejects_non_integer_when_integer_required() -> None:
    with pytest.raises(InvalidParameterError, match="must be an integer"):
        validate_numeric(3.5, label="binning", integer=True)


def test_validate_positive_rejects_zero() -> None:
    with pytest.raises(InvalidParameterError, match="must be non-zero"):
        validate_positive(0, name="exposure")


def test_validate_indices_rejects_empty_range_and_non_integer_inputs() -> None:
    with pytest.raises(InvalidParameterError, match="range is empty"):
        validate_indices([0], max_index=0)

    with pytest.raises(InvalidParameterError, match="must be integers"):
        validate_indices([0, 1.5], max_index=3)


def test_validate_numeric_supports_nan_inf_and_range_wrapper() -> None:
    assert np.isnan(validate_numeric(np.nan, label="gain", allow_nan=True))
    assert validate_numeric(np.inf, label="limit", allow_inf=True) == np.inf
    assert validate_range(3, minimum=1, maximum=5, name="binning", integer=True) == 3

    with pytest.raises(InvalidParameterError, match="must not be NaN"):
        validate_numeric(np.nan, label="gain")

    with pytest.raises(InvalidParameterError, match="must not be infinite"):
        validate_numeric(np.inf, label="gain")

    with pytest.raises(InvalidParameterError, match="cannot be infinite"):
        validate_numeric(np.inf, label="gain", integer=True, allow_inf=True)

    with pytest.raises(InvalidParameterError, match="must be <= 5"):
        validate_range(6, minimum=1, maximum=5, name="binning")


def test_validate_axis_and_nonnegative_cover_normalization_and_errors() -> None:
    assert validate_axis(-1, 3) == 2
    validate_nonnegative(0, name="offset")

    with pytest.raises(InvalidParameterError, match="must be an integer"):
        validate_axis("0", 3)  # type: ignore[arg-type]

    with pytest.raises(InvalidParameterError, match="positive integer"):
        validate_axis(0, 0)

    with pytest.raises(InvalidParameterError, match="out of range"):
        validate_axis(4, 3)

    with pytest.raises(InvalidParameterError, match="must be >= 0"):
        validate_nonnegative(-1, name="offset")


def test_validate_shape_and_dimension_helpers_cover_success_and_failures() -> None:
    arr2d = np.zeros((2, 3), dtype=np.float32)
    arr3d = np.zeros((2, 3, 4), dtype=np.float32)

    validate_shape(arr2d, expected_ndim=2, expected_shape=(2, 3), operation="crop")
    assert validate_same_ndim(arr2d, np.ones((4, 5)), operation="stack") == 2
    validate_shapes(arr2d, np.ones((2, 3)), operation="stack")
    assert validate_same_ndim(operation="empty") == 0
    validate_shapes(arr2d, operation="single")

    with pytest.raises(InvalidParameterError, match="numpy.ndarray"):
        validate_shape([[1, 2], [3, 4]])  # type: ignore[arg-type]

    with pytest.raises(DimensionalityError, match="Expected 2D array"):
        validate_shape(arr3d, expected_ndim=2, operation="crop")

    with pytest.raises(DimensionalityError, match="minimum 3 required"):
        validate_shape(arr2d, min_ndim=3, operation="stack")

    with pytest.raises(DimensionalityError, match="maximum 2 allowed"):
        validate_shape(arr3d, max_ndim=2, operation="stack")

    with pytest.raises(DataShapeError, match="Expected shape"):
        validate_shape(arr2d, expected_shape=(3, 2), operation="crop")

    with pytest.raises(DimensionalityError, match="Dimensionality mismatch"):
        validate_same_ndim(arr2d, arr3d, operation="stack")

    with pytest.raises(DataShapeError, match="Shape mismatch"):
        validate_shapes(arr2d, np.zeros((3, 2)), operation="stack")
