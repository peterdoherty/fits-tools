#!/usr/bin/env python3
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Pixel-wise operations for FITS images.

This module provides comprehensive pixel-wise operations on FITS images,
including unary operations (log, sqrt), scalar arithmetic (add constant),
binary image operations (image + image), comparison operations (masking),
and bitwise operations.

All operations support:
    - In-memory numpy arrays
    - FITS file I/O with header preservation
    - NaN/Inf handling
    - Region-specific operations (via masks)
    - Batch processing (via external orchestration)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..array_validation import validate_shapes
from ..bitmask import (
    BPM_MASK_PERMANENT,
    build_good_pixel_mask,
)
from ..bitmask import (
    DefectType as PixelFlags,
)
from ..exceptions import ImageProcessingError
from ..fits_output import (
    OutputMode,
    ResolvedHDU,
    StructurePolicy,
    _write_operation_result,
    identity_selector,
)
from ..hdu import (
    HDUQuery,
    HDUSelector,
    find_hdu_indices,
    get_hdu_identity,
    resolve_hdu_index,
    resolve_hdu_indices,
)
from ..header_utils import add_history
from ..types import StrOrPath
from .result_types import (
    ImageOperationResult,
    create_failure_result,
    create_success_result,
)

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_FLAGS: int = BPM_MASK_PERMANENT


def _load_resolved_hdu(
    filepath: StrOrPath,
    selector: HDUSelector = "auto",
) -> ResolvedHDU:
    """Load one image HDU using the package-wide selection semantics."""
    path = Path(filepath)
    with fits.open(path, mode="readonly", memmap=False) as hdul:
        index = resolve_hdu_index(
            hdul,
            selector,
            auto="first_sci_image",
            image_like=True,
            require_data=True,
            filepath=path,
        )
        candidate = hdul[index]
        return ResolvedHDU(
            path=path,
            index=index,
            identity=get_hdu_identity(candidate, index),
            data=np.array(candidate.data, copy=True),
            header=candidate.header.copy(),
        )


# ============================================================================
# Operation Registries
# ============================================================================

UNARY_OPS: dict[str, tuple[Callable, str]] = {
    "log": (np.log, "Natural logarithm"),
    "log10": (np.log10, "Base-10 logarithm"),
    "log2": (np.log2, "Base-2 logarithm"),
    "sqrt": (np.sqrt, "Square root"),
    "abs": (np.abs, "Absolute value"),
    "negate": (np.negative, "Negate (multiply by -1)"),
    "neg": (np.negative, "Negate (alias)"),
    "exp": (np.exp, "Exponential (e^x)"),
    "exp2": (np.exp2, "Base-2 exponential (2^x)"),
    "expm1": (np.expm1, "exp(x) - 1"),
    "log1p": (np.log1p, "log(1 + x)"),
    "sin": (np.sin, "Sine"),
    "cos": (np.cos, "Cosine"),
    "tan": (np.tan, "Tangent"),
    "arcsin": (np.arcsin, "Arcsine"),
    "arccos": (np.arccos, "Arccosine"),
    "arctan": (np.arctan, "Arctangent"),
    "sinh": (np.sinh, "Hyperbolic sine"),
    "cosh": (np.cosh, "Hyperbolic cosine"),
    "tanh": (np.tanh, "Hyperbolic tangent"),
    "arcsinh": (np.arcsinh, "Inverse hyperbolic sine"),
    "arccosh": (np.arccosh, "Inverse hyperbolic cosine"),
    "arctanh": (np.arctanh, "Inverse hyperbolic tangent"),
    "round": (np.round, "Round to nearest integer"),
    "floor": (np.floor, "Round down (floor)"),
    "ceil": (np.ceil, "Round up (ceiling)"),
    "trunc": (np.trunc, "Truncate (remove decimal)"),
    "square": (np.square, "Square (x²)"),
    "reciprocal": (np.reciprocal, "Reciprocal (1/x)"),
    "sign": (np.sign, "Sign function (-1, 0, or 1)"),
}

SCALAR_OPS: dict[str, tuple[Callable, str]] = {
    "add": (np.add, "Add scalar to image"),
    "subtract": (np.subtract, "Subtract scalar from image"),
    "multiply": (np.multiply, "Multiply image by scalar"),
    "divide": (np.divide, "Divide image by scalar"),
    "power": (np.power, "Raise image to scalar power"),
    "mod": (np.mod, "Modulo operation"),
    "remainder": (np.remainder, "Remainder operation"),
    "true_divide": (np.true_divide, "True division"),
    "floor_divide": (np.floor_divide, "Floor division"),
}

COMPARISON_OPS: dict[str, tuple[Callable, str]] = {
    "greater": (np.greater, "Greater than (>)"),
    "less": (np.less, "Less than (<)"),
    "greater_equal": (np.greater_equal, "Greater than or equal (>=)"),
    "less_equal": (np.less_equal, "Less than or equal (<=)"),
    "equal": (np.equal, "Equal to (==)"),
    "not_equal": (np.not_equal, "Not equal to (!=)"),
}

BITWISE_OPS: dict[str, tuple[Callable, str]] = {
    "bitwise_and": (np.bitwise_and, "Bitwise AND"),
    "bitwise_or": (np.bitwise_or, "Bitwise OR"),
    "bitwise_xor": (np.bitwise_xor, "Bitwise XOR"),
    "bitwise_not": (np.bitwise_not, "Bitwise NOT"),
    "left_shift": (np.left_shift, "Left bit shift"),
    "right_shift": (np.right_shift, "Right bit shift"),
}

BINARY_IMAGE_OPS: dict[str, tuple[Callable, str]] = {
    "add": (np.add, "Pixelwise addition"),
    "subtract": (np.subtract, "Pixelwise subtraction (a - b)"),
    "multiply": (np.multiply, "Pixelwise multiplication"),
    "divide": (np.divide, "Pixelwise division (a / b)"),
    "maximum": (np.maximum, "Pixelwise maximum"),
    "minimum": (np.minimum, "Pixelwise minimum"),
    "mean": (lambda a, b: 0.5 * (a + b), "Pixelwise mean"),
    "diff": (lambda a, b: np.abs(a - b), "Pixelwise absolute difference"),
    "power": (np.power, "Pixelwise power (a^b)"),
}

CLI_ALIASES: dict[str, str] = {
    # Scalar operations
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "pow": "power",
    # Comparison operations
    "gt": "greater",
    "lt": "less",
    "ge": "greater_equal",
    "le": "less_equal",
    "eq": "equal",
    "ne": "not_equal",
    # Bitwise operations
    "band": "bitwise_and",
    "bor": "bitwise_or",
    "bxor": "bitwise_xor",
    "bnot": "bitwise_not",
    "lshift": "left_shift",
    "rshift": "right_shift",
    # Unary operations
    "ln": "log",
    "neg": "negate",
}

# ============================================================================
# Helper Functions
# ============================================================================

def _safe_divide(a, b, fill_value=0.0, warn=True):
    """Safely divide arrays, handling division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b

    if warn:
        num_invalid = np.sum(~np.isfinite(result))
        if num_invalid > 0:
            logger.warning(
                "Division produced %d invalid values (inf/nan). "
                "Replacing with %s.", num_invalid, fill_value
            )

    return np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)

# ============================================================================
# Core Operation Functions
# ============================================================================

def apply_unary_operation(
    data: np.ndarray,
    operation: str,
    bad_pixel_mask: np.ndarray | None = None,
    exclude_flags: int | PixelFlags = BPM_MASK_PERMANENT,
    exclude_nan: bool = True,
    exclude_inf: bool = False,
    auto_exclude_invalid: bool = True,
    preserve_dtype: bool = False,
    handle_invalid: str = "ignore",
) -> np.ndarray:
    """
    Apply a unary operation to image data with bad pixel masking.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    operation : str
        Operation name (e.g., 'log', 'sqrt', 'abs').
    bad_pixel_mask : np.ndarray, optional
        Bad pixel mask (integer bitmask or boolean array).
        Pixels marked as bad are excluded from operation and retain original values.
    exclude_flags : int or PixelFlags, optional
        Which bad pixel flags to exclude. Default is BPM_MASK_PERMANENT.
    exclude_nan : bool, optional
        Exclude NaN pixels. Default is True.
    exclude_inf : bool, optional
        Exclude Inf pixels. Default is False.
    auto_exclude_invalid : bool, optional
        Automatically exclude pixels that would cause invalid results:
        - For 'log', 'log10', 'log2', 'sqrt': excludes non-positive values
        - For 'arcsin', 'arccos': excludes values outside [-1, 1]
        Default is True.
    preserve_dtype : bool, optional
        Preserve input dtype. Default is False.
    handle_invalid : str, optional
        How to handle invalid operations ('ignore', 'warn', 'raise').

    Returns
    -------
    np.ndarray
        Result array. Masked pixels retain original values.

    Examples
    --------
    >>> # With bitmask - exclude permanent hardware defects
    >>> data = np.array([1.0, 4.0, -1.0, 16.0])
    >>> bpm = np.array([0, 0, 0, BPM_BIT_SATURATED], dtype=np.uint32)
    >>> result = apply_unary_operation(data, "sqrt", bad_pixel_mask=bpm)
    >>> result  # Negative value excluded, saturated pixel unchanged
    array([1., 2., -1., 16.])

    >>> # With boolean mask
    >>> bad_mask = np.array([False, False, False, True])
    >>> result = apply_unary_operation(data, "sqrt", bad_pixel_mask=bad_mask)
    """
    operation = CLI_ALIASES.get(operation, operation)

    if operation not in UNARY_OPS:
        valid_ops = ", ".join(sorted(UNARY_OPS.keys()))
        raise ValueError(
            f"Unknown unary operation '{operation}'. Valid options: {valid_ops}"
        )

    func, _ = UNARY_OPS[operation]
    original_dtype = data.dtype

    # Build additional exclusion conditions based on operation
    extra_kwargs = {}
    if auto_exclude_invalid:
        if operation in ('log', 'log10', 'log2'):
            extra_kwargs['exclude_negative'] = True
            extra_kwargs['exclude_zero'] = True
        elif operation == 'sqrt':
            extra_kwargs['exclude_negative'] = True
        elif operation in ('arcsin', 'arccos'):
            # Exclude values outside valid range [-1, 1]
            extra_kwargs['custom_condition'] = lambda d: (d >= -1) & (d <= 1)

    # Build good pixel mask using centralized function
    good_mask = build_good_pixel_mask(
        data,
        bad_pixel_mask=bad_pixel_mask,
        exclude_flags=exclude_flags,
        exclude_nan=exclude_nan,
        exclude_inf=exclude_inf,
        **extra_kwargs
    )

    try:
        # Create output array (copy of input to preserve masked values)
        result = data.copy()

        # Apply operation only to good pixels
        with np.errstate(
            divide=handle_invalid,
            invalid=handle_invalid,
            over=handle_invalid,
        ):
            # Most numpy ufuncs support 'where' parameter
            if isinstance(func, np.ufunc):
                func(data, out=result, where=good_mask)
            else:
                # Manual masking fallback for non-ufuncs
                result[good_mask] = func(data[good_mask])

        if preserve_dtype and result.dtype != original_dtype:
            result = result.astype(original_dtype)

        return result

    except Exception as e:
        raise ImageProcessingError(
            f"Failed to apply unary operation '{operation}': {e}"
        ) from e


def apply_scalar_operation(
    data: np.ndarray,
    operation: str,
    scalar: float,
    bad_pixel_mask: np.ndarray | None = None,
    exclude_flags: int | PixelFlags = BPM_MASK_PERMANENT,
    exclude_nan: bool = True,
    exclude_inf: bool = False,
    preserve_dtype: bool = False,
    handle_invalid: str = "ignore",
) -> np.ndarray:
    """Apply a scalar arithmetic operation to image data."""
    operation = CLI_ALIASES.get(operation, operation)

    if operation not in SCALAR_OPS:
        valid_ops = ", ".join(sorted(SCALAR_OPS.keys()))
        raise ValueError(
            f"Unknown scalar operation '{operation}'. Valid options: {valid_ops}"
        )

    func, _ = SCALAR_OPS[operation]
    original_dtype = data.dtype

    # Build good pixel mask
    good_mask = build_good_pixel_mask(
        data,
        bad_pixel_mask=bad_pixel_mask,
        exclude_flags=exclude_flags,
        exclude_nan=exclude_nan,
        exclude_inf=exclude_inf,
    )

    try:
        result = data.copy()

        with np.errstate(
            divide=handle_invalid,
            invalid=handle_invalid,
            over=handle_invalid,
        ):
            if isinstance(func, np.ufunc):
                func(data, scalar, out=result, where=good_mask)
            else:
                result[good_mask] = func(data[good_mask], scalar)

        if preserve_dtype and result.dtype != original_dtype:
            result = result.astype(original_dtype)

        return result

    except Exception as e:
        raise ImageProcessingError(
            f"Failed to apply scalar operation '{operation}' with scalar={scalar}: {e}"
        ) from e


def create_mask(
    data: np.ndarray,
    comparison: str,
    threshold: float,
    invert: bool = False,
) -> np.ndarray:
    """Create a boolean mask by comparing image data to a threshold."""

    if not np.isfinite(threshold):
        raise ValueError(f"Threshold must be finite, got {threshold}")

    comparison = CLI_ALIASES.get(comparison, comparison)

    if comparison not in COMPARISON_OPS:
        valid_ops = ", ".join(sorted(COMPARISON_OPS.keys()))
        valid_aliases = ", ".join(
            sorted(k for k, v in CLI_ALIASES.items() if v in COMPARISON_OPS)
        )
        raise ValueError(
            f"Unknown comparison operation '{comparison}'. "
            f"Valid options: {valid_ops}. "
            f"Valid aliases: {valid_aliases}"
        )

    func, _ = COMPARISON_OPS[comparison]
    mask = func(data, threshold)
    return ~mask if invert else mask


def threshold_image(
    data: np.ndarray,
    threshold: float,
    mode: str = "binary",
) -> np.ndarray:
    """Threshold an image using various methods."""
    if mode == "binary":
        return (data >= threshold).astype(np.float32)

    if mode == "zero":
        result = data.copy()
        result[data < threshold] = 0
        return result

    if mode == "truncate":
        result = data.copy()
        result[data > threshold] = threshold
        return result

    if mode == "to_zero_inv":
        result = data.copy()
        result[data >= threshold] = 0
        return result

    raise ValueError(
        f"Unknown threshold mode: '{mode}'. "
        "Valid modes: 'binary', 'zero', 'truncate', 'to_zero_inv'"
    )


def fill_masked(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: int | float,
    invert: bool = False,
    preserve_dtype: bool = False,
) -> np.ndarray:
    """Fill pixels selected by a boolean mask with a constant value."""
    if mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match data shape {data.shape}")

    original_dtype = data.dtype
    work_dtype = np.result_type(original_dtype, type(fill_value))
    result = data.astype(work_dtype, copy=True)

    selector = ~mask if invert else mask
    result[selector] = fill_value

    if preserve_dtype and result.dtype != original_dtype:
        result = result.astype(original_dtype)

    return result


def set_value(
    data: np.ndarray,
    value: int | float,
    preserve_dtype: bool = False,
) -> np.ndarray:
    """Set all pixels in an array to a constant value."""
    original_dtype = data.dtype
    work_dtype = np.result_type(original_dtype, type(value))
    result = np.full(data.shape, value, dtype=work_dtype)

    if preserve_dtype and result.dtype != original_dtype:
        result = result.astype(original_dtype)

    return result


def fill_where(
    data: np.ndarray,
    comparison: str,
    threshold: int | float,
    fill_value: int | float,
    invert: bool = False,
    preserve_dtype: bool = False,
) -> np.ndarray:
    """Fill pixels that satisfy a comparison with a constant value."""
    mask = create_mask(data, comparison=comparison, threshold=threshold, invert=False)
    return fill_masked(
        data,
        mask,
        fill_value,
        invert=invert,
        preserve_dtype=preserve_dtype,
    )


def apply_bitwise_operation(
    data: np.ndarray,
    operation: str,
    operand: int | np.ndarray | None = None,
) -> np.ndarray:
    """Apply a bitwise operation to image data."""
    operation = CLI_ALIASES.get(operation, operation)

    if operation not in BITWISE_OPS:
        valid_ops = ", ".join(sorted(BITWISE_OPS.keys()))
        raise ValueError(
            f"Unknown bitwise operation '{operation}'. Valid options: {valid_ops}"
        )

    func, _ = BITWISE_OPS[operation]

    # Use int64; callers working with bitmasks can cast back as needed.
    data_int = data.astype(np.int64)

    try:
        if operation == "bitwise_not":
            result = func(data_int)
        else:
            if operand is None:
                raise ValueError(f"Operation '{operation}' requires an operand")

            if isinstance(operand, np.ndarray):
                operand = operand.astype(np.int64)
            else:
                operand = int(operand)

            result = func(data_int, operand)

        return result

    except Exception as e:
        raise ImageProcessingError(
            f"Failed to apply bitwise operation '{operation}': {e}"
        ) from e

# ============================================================================
# File-Level Binary Image Operations
# ============================================================================

def _apply_binary_operation_to_files(
    operation_name: str,
    operation_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    input_file1: StrOrPath,
    input_file2: StrOrPath,
    output_file: StrOrPath,
    hdu: HDUSelector = 0,
    operand_hdu: HDUSelector | None = None,
    overwrite: bool = False,
    working_dtype: np.dtype = np.float32,
    preserve_dtype: bool = False,
    output_mode: OutputMode = "sef",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
) -> ImageOperationResult:
    """Generic wrapper to perform binary operations on two FITS files."""
    try:
        left = _load_resolved_hdu(input_file1, hdu)
        right = _load_resolved_hdu(input_file2, operand_hdu or identity_selector(left))
        data1, header1, data2 = left.data, left.header, right.data

        validate_shapes(data1, data2, operation=operation_name)
        original_dtype = data1.dtype

        result_data = operation_func(
            data1.astype(working_dtype),
            data2.astype(working_dtype),
        )

        if preserve_dtype:
            result_data = result_data.astype(original_dtype)

        new_header = header1.copy()
        add_history(new_header, f"{operation_name}: {Path(input_file2).name}")
        new_header["OPERAND1"] = (Path(input_file1).name, "First operand")
        new_header["OPERAND2"] = (Path(input_file2).name, "Second operand")
        new_header["OPERNAME"] = (operation_name.upper(), "Operation")

        _write_operation_result(
            left, output_file, result_data, new_header,
            output_mode=output_mode, destination_hdu=destination_hdu,
            structure_policy=structure_policy, overwrite=overwrite,
        )

        logger.debug("%s complete: %s", operation_name, output_file)

        return create_success_result(
            output_file=output_file,
            operation=operation_name,
            input_files=[input_file1, input_file2],
            shape=result_data.shape,
        )

    except Exception as e:
        logger.error("%s failed: %s", operation_name, e, exc_info=True)
        return create_failure_result(
            error=str(e),
            operation=operation_name,
        )


def subtract_images(
    input_file1: StrOrPath,
    input_file2: StrOrPath,
    output_file: StrOrPath,
    hdu: HDUSelector = 0,
    operand_hdu: HDUSelector | None = None,
    overwrite: bool = False,
    output_mode: OutputMode = "sef",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
) -> ImageOperationResult:
    """
    Subtract the pixel data of one FITS image from another.

    Result = input_file1 - input_file2

    Parameters
    ----------
    input_file1 : str
        First input FITS file (minuend).
    input_file2 : str
        Second input FITS file (subtrahend).
    output_file : str
        Output FITS file path.
    hdu : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.

    Returns
    -------
    ImageOperationResult
        Result of subtraction.

    Examples
    --------
    >>> # Assuming test files exist
    >>> result = subtract_images('img1.fits', 'img2.fits', 'diff.fits', overwrite=True)
    >>> if result.success:
    ...     print(f"Subtraction successful: {result.output_file}")
    """
    return _apply_binary_operation_to_files(
        "Subtract",
        lambda a, b: a - b,
        input_file1,
        input_file2,
        output_file,
        hdu=hdu,
        operand_hdu=operand_hdu,
        overwrite=overwrite,
        output_mode=output_mode, destination_hdu=destination_hdu,
        structure_policy=structure_policy,
    )


def add_images(
    input_file1: StrOrPath,
    input_file2: StrOrPath,
    output_file: StrOrPath,
    hdu: HDUSelector = 0,
    operand_hdu: HDUSelector | None = None,
    overwrite: bool = False,
    output_mode: OutputMode = "sef",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
) -> ImageOperationResult:
    """
    Add the pixel data of two FITS images.

    Result = input_file1 + input_file2

    Parameters
    ----------
    input_file1 : str
        First input FITS file.
    input_file2 : str
        Second input FITS file.
    output_file : str
        Output FITS file path.
    hdu : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.

    Returns
    -------
    ImageOperationResult
        Result of addition.
    """
    return _apply_binary_operation_to_files(
        "Add",
        lambda a, b: a + b,
        input_file1,
        input_file2,
        output_file,
        hdu=hdu,
        operand_hdu=operand_hdu,
        overwrite=overwrite,
        output_mode=output_mode, destination_hdu=destination_hdu,
        structure_policy=structure_policy,
    )


def multiply_images(
    input_file1: StrOrPath,
    input_file2: StrOrPath,
    output_file: StrOrPath,
    hdu: HDUSelector = 0,
    operand_hdu: HDUSelector | None = None,
    overwrite: bool = False,
    output_mode: OutputMode = "sef",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
) -> ImageOperationResult:
    """
    Multiply the pixel data of two FITS images.

    Result = input_file1 * input_file2

    Parameters
    ----------
    input_file1 : str
        First input FITS file (dividend).
    input_file2 : str
        Second input FITS file (divisor).
    output_file : str
        Output FITS file path.
    hdu : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.

    Returns
    -------
    ImageOperationResult
        Result of multiplication.
    """
    return _apply_binary_operation_to_files(
        "Multiply",
        lambda a, b: a * b,
        input_file1,
        input_file2,
        output_file,
        hdu=hdu,
        operand_hdu=operand_hdu,
        overwrite=overwrite,
        output_mode=output_mode, destination_hdu=destination_hdu,
        structure_policy=structure_policy,
    )


def divide_images(
    input_file1: StrOrPath,
    input_file2: StrOrPath,
    output_file: StrOrPath,
    hdu: HDUSelector = 0,
    operand_hdu: HDUSelector | None = None,
    overwrite: bool = False,
    fill_value: float = 0.0,
    output_mode: OutputMode = "sef",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
) -> ImageOperationResult:
    """
    Divide the pixel data of one FITS image by another.

    Result = input_file1 / input_file2

    Handles division by zero by replacing inf/nan with fill_value.

    Parameters
    ----------
    input_file1 : str
        First input FITS file (dividend).
    input_file2 : str
        Second input FITS file (divisor).
    output_file : str
        Output FITS file path.
    hdu : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.
    fill_value : float, optional
        Value to use for division by zero. Defaults to 0.0.

    Returns
    -------
    ImageOperationResult
        Result of division.
    """
    def _divide_with_fill(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = a / b

        # Count and report zeros
        num_zeros = np.sum(b == 0)
        if num_zeros > 0:
            logger.warning(
                "Division by zero in %d pixels. Replacing with %s.",
                num_zeros,
                fill_value,
            )
        # Replace inf/nan with fill_value
        result = np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)
        return result

    return _apply_binary_operation_to_files(
        "Divide",
        _divide_with_fill,
        input_file1,
        input_file2,
        output_file,
        hdu=hdu,
        operand_hdu=operand_hdu,
        overwrite=overwrite,
        output_mode=output_mode, destination_hdu=destination_hdu,
        structure_policy=structure_policy,
    )


def scale_image(
    input_file: StrOrPath,
    scale_factor: float,
    output_file: StrOrPath,
    hdu: HDUSelector = 0,
    overwrite: bool = False,
    working_dtype: np.dtype = np.float32,
    preserve_dtype: bool = False,
    output_mode: OutputMode = "sef",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
) -> ImageOperationResult:
    """Scale a FITS image by a constant factor."""
    try:
        source = _load_resolved_hdu(input_file, hdu)
        data, header = source.data, source.header

        original_dtype = data.dtype
        scaled_data = data.astype(working_dtype) * scale_factor

        if preserve_dtype:
            scaled_data = scaled_data.astype(original_dtype)

        add_history(header, f"Scaled by factor {scale_factor}")
        _write_operation_result(
            source, output_file, scaled_data, header,
            output_mode=output_mode, destination_hdu=destination_hdu,
            structure_policy=structure_policy, overwrite=overwrite,
        )

        logger.debug("Image scaled by %.3f: %s", scale_factor, output_file)

        return create_success_result(
            output_file=output_file,
            operation="scale",
            scale_factor=scale_factor,
            shape=scaled_data.shape,
            # bpm_file=str(bad_pixel_mask_file),
            # bpm_flags=f"0x{int(exclude_flags):08X}",
            # masked_pixels=int(np.count_nonzero(...)),
        )

    except Exception as e:
        logger.error("Scaling failed: %s", e, exc_info=True)
        return create_failure_result(
            error=str(e),
            operation="scale",
        )

# ============================================================================
# High-Level File Operations
# ============================================================================

def apply_operation_to_file(
    input_file: StrOrPath,
    output_file: StrOrPath,
    operation_type: str,
    operation: str,
    hdu: HDUSelector = 0,
    bad_pixel_mask_file: StrOrPath | None = None,
    bad_pixel_mask_hdu: HDUSelector | None = None,
    exclude_flags: int | PixelFlags = DEFAULT_EXCLUDE_FLAGS,
    exclude_nan: bool = True,
    exclude_inf: bool = False,
    scalar: float | None = None,
    threshold: float | None = None,
    operand: int | float | str | Path | np.ndarray | None = None,
    operand_hdu: HDUSelector | None = None,
    overwrite: bool = False,
    preserve_dtype: bool = False,
    output_mode: OutputMode = "sef",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
    invert=False,
) -> ImageOperationResult:
    """
    Apply operation to FITS file with bad pixel mask support.

    Parameters
    ----------
    input_file : str or Path
        Input FITS file path.
    output_file : str or Path
        Output FITS file path.
    operation_type : str
        Type of operation ('unary', 'scalar', 'comparison', etc.).
    operation : str
        Specific operation name.
    hdu : int, optional
        HDU extension. Default is 0.
    bad_pixel_mask_file : str or Path, optional
        FITS file containing bad pixel mask. Can be integer bitmask or boolean.
    bad_pixel_mask_hdu : int, optional
        HDU in bad pixel mask file. Defaults to same as data HDU.
    exclude_flags : int or PixelFlags, optional
        Which bad pixel flags to exclude. Default is PixelFlags.ANY_BAD.
    exclude_nan : bool, optional
        Exclude NaN pixels. Default is True.
    exclude_inf : bool, optional
        Exclude Inf pixels. Default is False.
    scalar : float, optional
        Scalar value for scalar operations.
    threshold : float, optional
        Threshold for comparison operations.
    operand : various, optional
        Secondary operand for binary operations.
    overwrite : bool, optional
        Overwrite output file if exists. Default is False.
    preserve_dtype : bool, optional
        Preserve input data type. Default is False.

    Returns
    -------
    ImageOperationResult
        Result containing success status and metadata.

    Examples
    --------
    >>> # Apply sqrt, excluding saturated pixels
    >>> result = apply_operation_to_file(
    ...     "science.fits", "output.fits",
    ...     operation_type="unary",
    ...     operation="sqrt",
    ...     bad_pixel_mask_file="dq.fits",
    ...     exclude_flags=PixelFlags.SATURATED | PixelFlags.DEAD,
    ...     overwrite=True
    ... )

    >>> # Exclude only saturated pixels (keep other defects in analysis)
    >>> from fits_tools.bitmask import BPM_BIT_SATURATED
    >>> result = apply_operation_to_file(
    ...     "science.fits", "output.fits",
    ...     operation_type="scalar",
    ...     operation="multiply",
    ...     scalar=2.0,
    ...     bad_pixel_mask_file="dq.fits",
    ...     exclude_flags=BPM_BIT_SATURATED,
    ...     overwrite=True
    ... )
    """
    try:
        # Load data
        source = _load_resolved_hdu(input_file, hdu)
        data, header = source.data, source.header

        # Load bad pixel mask if provided
        bad_pixel_mask = None
        if bad_pixel_mask_file is not None:
            mask_hdu = bad_pixel_mask_hdu or identity_selector(source)
            bad_pixel_mask = _load_resolved_hdu(
                bad_pixel_mask_file, mask_hdu
            ).data

            logger.info(
                "Loaded bad pixel mask from %s (HDU %s): dtype=%s",
                bad_pixel_mask_file, mask_hdu, bad_pixel_mask.dtype
            )

        # Apply operation
        if operation_type == "unary":
            result_data = apply_unary_operation(
                data, operation,
                bad_pixel_mask=bad_pixel_mask,
                exclude_flags=exclude_flags,
                exclude_nan=exclude_nan,
                exclude_inf=exclude_inf,
                preserve_dtype=preserve_dtype,
            )
        elif operation_type == "scalar":
            if scalar is None:
                raise ValueError("Scalar operations require 'scalar' parameter")
            result_data = apply_scalar_operation(
                data, operation, scalar,
                bad_pixel_mask=bad_pixel_mask,
                exclude_flags=exclude_flags,
                exclude_nan=exclude_nan,
                exclude_inf=exclude_inf,
                preserve_dtype=preserve_dtype,
            )

        elif operation_type == "comparison":
            if threshold is None:
                raise ValueError("Comparison operations require 'threshold' parameter")
            result_data = create_mask(data, operation, threshold).astype(np.uint8)

        elif operation_type == "bitwise":
            if operation != "bitwise_not" and operand is None:
                raise ValueError(
                    "Bitwise operations (except NOT) require 'operand' parameter"
                )

            if isinstance(operand, (str, Path)) and Path(operand).exists():
                operand = _load_resolved_hdu(
                    Path(operand), operand_hdu or identity_selector(source)
                ).data

            result_data = apply_bitwise_operation(data, operation, operand)

        elif operation_type == "fill":
            if scalar is None:
                raise ValueError("Fill operations require 'scalar' parameter (fill_value)")

            if operation == "mask":
                if operand is None:
                    raise ValueError(
                        "Fill operation 'mask' requires 'operand' as a mask array or mask FITS file path"
                    )

                mask_data = operand
                if isinstance(mask_data, (str, Path)) and Path(mask_data).exists():
                    target_hdu = (
                        operand_hdu
                        or bad_pixel_mask_hdu
                        or identity_selector(source)
                    )
                    mask_data = _load_resolved_hdu(Path(mask_data), target_hdu).data

                mask = np.asarray(mask_data) != 0
                if mask.shape != data.shape:
                    raise ImageProcessingError(
                        f"Mask shape {mask.shape} does not match data shape {data.shape}"
                    )

                result_data = fill_masked(
                    data, mask, scalar, invert=invert, preserve_dtype=preserve_dtype
                )
            else:
                if threshold is None:
                    raise ValueError(
                        "Fill operations based on comparisons require 'threshold' parameter"
                    )
                result_data = fill_where(
                    data,
                    comparison=operation,
                    threshold=threshold,
                    fill_value=scalar,
                    invert=invert,
                    preserve_dtype=preserve_dtype,
                )

        else:
            raise ValueError(f"Unknown operation_type: '{operation_type}'")

        # Update header with operation history
        add_history(header, f"Applied {operation_type} operation: {operation}")
        if bad_pixel_mask_file:
            add_history(
                header,
                f"Used bad pixel mask: {Path(bad_pixel_mask_file).name} "
                f"(flags: 0x{int(exclude_flags):X})"
            )
        if exclude_nan:
            add_history(header, "Excluded NaN pixels from operation")
        if exclude_inf:
            add_history(header, "Excluded Inf pixels from operation")

        # Write output
        _write_operation_result(
            source, output_file, result_data, header,
            output_mode=output_mode, destination_hdu=destination_hdu,
            structure_policy=structure_policy, overwrite=overwrite,
        )

        logger.info("Operation complete: %s", output_file)

        return create_success_result(
            output_file=output_file,
            operation=f"{operation_type}:{operation}",
            shape=result_data.shape,
        )

    except Exception as e:
        logger.error("Operation failed: %s", e, exc_info=True)
        return create_failure_result(
            error=f"{type(e).__name__}: {e}",
            operation=f"{operation_type}:{operation}",
            bpm_file=str(bad_pixel_mask_file) if bad_pixel_mask_file is not None else None,
            bpm_flags=f"0x{int(exclude_flags):08X}",
        )


def apply_operation_to_hdus(
    input_file: StrOrPath,
    output_file: StrOrPath,
    operation_type: str,
    operation: str,
    *,
    selectors: Iterable[HDUSelector] | None = None,
    query: HDUQuery | None = None,
    operand: int | float | str | Path | np.ndarray | None = None,
    overwrite: bool = False,
    **kwargs,
) -> list[ImageOperationResult]:
    """Apply an operation to several MEF extensions, paired by HDU identity."""
    if (selectors is None) == (query is None):
        raise ValueError("Provide exactly one of selectors or query")

    source_path = Path(input_file)
    with fits.open(source_path, mode="readonly", memmap=False) as hdul:
        indices = (
            resolve_hdu_indices(
                hdul, selectors, image_like=True, require_data=True,
                filepath=source_path,
            )
            if selectors is not None
            else find_hdu_indices(hdul, query)
        )
        resolved_selectors = [
            identity_selector(
                ResolvedHDU(
                    source_path,
                    index,
                    get_hdu_identity(hdul[index], index),
                    np.empty(0),
                    fits.Header(),
                )
            )
            for index in indices
        ]

    results: list[ImageOperationResult] = []
    for position, selector in enumerate(resolved_selectors):
        result = apply_operation_to_file(
            input_file,
            output_file,
            operation_type,
            operation,
            hdu=selector,
            operand=operand,
            operand_hdu=selector if isinstance(operand, (str, Path)) else None,
            output_mode="preserve",
            overwrite=overwrite or position > 0,
            **kwargs,
        )
        results.append(result)
        if not result.success:
            break
    return results

# ============================================================================
# Information Functions
# ============================================================================

def list_operations(operation_type: str | None = None) -> dict[str, list[str]]:
    """List available operations by type."""
    operations = {
        "unary": sorted(UNARY_OPS.keys()),
        "scalar": sorted(SCALAR_OPS.keys()),
        "comparison": sorted(COMPARISON_OPS.keys()),
        "bitwise": sorted(BITWISE_OPS.keys()),
        "binary": sorted(BINARY_IMAGE_OPS.keys()),
        "fill": sorted(list(COMPARISON_OPS.keys()) + ["mask"]),
    }

    if operation_type:
        if operation_type not in operations:
            raise ValueError(
                f"Unknown operation_type: '{operation_type}'. "
                f"Valid: {list(operations.keys())}"
            )
        return {operation_type: operations[operation_type]}

    return operations


def get_operation_description(operation: str) -> str:
    """Get a human-readable description of an operation."""
    operation = CLI_ALIASES.get(operation, operation)

    if operation == "mask":
        return "Fill using a boolean mask (operand)"

    for registry in (
        UNARY_OPS,
        SCALAR_OPS,
        COMPARISON_OPS,
        BITWISE_OPS,
        BINARY_IMAGE_OPS,
    ):
        if operation in registry:
            return registry[operation][1]

    raise ValueError(f"Unknown operation: '{operation}'")


__all__ = [
    "apply_unary_operation",
    "apply_scalar_operation",
    "create_mask",
    "threshold_image",
    "apply_bitwise_operation",
    "subtract_images",
    "add_images",
    "multiply_images",
    "divide_images",
    "scale_image",
    "apply_operation_to_file",
    "list_operations",
    "get_operation_description",
    "UNARY_OPS",
    "SCALAR_OPS",
    "COMPARISON_OPS",
    "BITWISE_OPS",
    "BINARY_IMAGE_OPS",
    "CLI_ALIASES",
    "fill_masked",
    "fill_where",
    "set_value",
]
