#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..io import get_data, get_data_and_header, get_header, write_fits
from ..exceptions import ImageProcessingError
from ..utils import add_history
from ..array_utils import validate_shapes
from .result_types import (
    ImageOperationResult,
    create_failure_result,
    create_success_result,
)



logger = logging.getLogger(__name__)
__version__ = "2.0.0"


# ============================================================================
# Operation Registries
# ============================================================================

UNARY_OPS: Dict[str, Tuple[Callable, str]] = {
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

SCALAR_OPS: Dict[str, Tuple[Callable, str]] = {
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

COMPARISON_OPS: Dict[str, Tuple[Callable, str]] = {
    "greater": (np.greater, "Greater than (>)"),
    "less": (np.less, "Less than (<)"),
    "greater_equal": (np.greater_equal, "Greater than or equal (>=)"),
    "less_equal": (np.less_equal, "Less than or equal (<=)"),
    "equal": (np.equal, "Equal to (==)"),
    "not_equal": (np.not_equal, "Not equal to (!=)"),
}

BITWISE_OPS: Dict[str, Tuple[Callable, str]] = {
    "bitwise_and": (np.bitwise_and, "Bitwise AND"),
    "bitwise_or": (np.bitwise_or, "Bitwise OR"),
    "bitwise_xor": (np.bitwise_xor, "Bitwise XOR"),
    "bitwise_not": (np.bitwise_not, "Bitwise NOT"),
    "left_shift": (np.left_shift, "Left bit shift"),
    "right_shift": (np.right_shift, "Right bit shift"),
}

BINARY_IMAGE_OPS: Dict[str, Tuple[Callable, str]] = {
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

CLI_ALIASES: Dict[str, str] = {
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
# Core Operation Functions
# ============================================================================

def apply_unary_operation(
    data: np.ndarray,
    operation: str,
    preserve_dtype: bool = False,
    handle_invalid: str = "ignore",
) -> np.ndarray:
    """Apply a unary operation to image data."""
    operation = CLI_ALIASES.get(operation, operation)

    if operation not in UNARY_OPS:
        valid_ops = ", ".join(sorted(UNARY_OPS.keys()))
        raise ValueError(
            f"Unknown unary operation '{operation}'. Valid options: {valid_ops}"
        )

    func, _ = UNARY_OPS[operation]
    original_dtype = data.dtype

    try:
        with np.errstate(
            divide=handle_invalid,
            invalid=handle_invalid,
            over=handle_invalid,
        ):
            result = func(data)

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

    try:
        with np.errstate(
            divide=handle_invalid,
            invalid=handle_invalid,
            over=handle_invalid,
        ):
            result = func(data, scalar)

        if operation in ("divide", "true_divide", "floor_divide") and scalar == 0:
            logger.warning("Division by zero in scalar operation")
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

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
    fill_value: Union[int, float],
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
    value: Union[int, float],
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
    threshold: Union[int, float],
    fill_value: Union[int, float],
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
    operand: Optional[Union[int, np.ndarray]] = None,
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
    ext: int = 0,
    overwrite: bool = False,
    working_dtype: np.dtype = np.float32,
    preserve_dtype: bool = False,
) -> ImageOperationResult:
    """Generic wrapper to perform binary operations on two FITS files."""
    try:
        data1, header1 = get_data_and_header(input_file1, ext=ext)
        data2 = get_data(input_file2, ext=ext)

        if data1 is None:
            raise ImageProcessingError(
                f"HDU {ext} in {input_file1} contains no image data"
            )
        if data2 is None:
            raise ImageProcessingError(
                f"HDU {ext} in {input_file2} contains no image data"
            )

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
        new_header["OPERATION"] = (operation_name.upper(),"Operation performed")

        write_fits(
            output_file,
            result_data,
            new_header,
            overwrite=overwrite,
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
    ext: int = 0,
    overwrite: bool = False,
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
    ext : int, optional
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
        ext=ext,
        overwrite=overwrite,
    )


def add_images(
    input_file1: StrOrPath,
    input_file2: StrOrPath,
    output_file: StrOrPath,
    ext: int = 0,
    overwrite: bool = False,
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
    ext : int, optional
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
        ext=ext,
        overwrite=overwrite,
    )


def multiply_images(
    input_file1: StrOrPath,
    input_file2: StrOrPath,
    output_file: StrOrPath,
    ext: int = 0,
    overwrite: bool = False,
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
    ext : int, optional
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
        ext=ext,
        overwrite=overwrite,
    )


def divide_images(
    input_file1: StrOrPath,
    input_file2: StrOrPath,
    output_file: StrOrPath,
    ext: int = 0,
    overwrite: bool = False,
    fill_value: float = 0.0,
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
    ext : int, optional
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
        ext=ext,
        overwrite=overwrite,
    )


def scale_image(
    input_file: StrOrPath,
    scale_factor: float,
    output_file: StrOrPath,
    ext: int = 0,
    overwrite: bool = False,
    working_dtype: np.dtype = np.float32,
    preserve_dtype: bool = False,
) -> ImageOperationResult:
    """Scale a FITS image by a constant factor."""
    try:
        data, header = get_data_and_header(input_file, ext=ext)
        if data is None:
            raise ImageProcessingError(
                f"HDU {ext} in {input_file} contains no image data"
            )

        original_dtype = data.dtype
        scaled_data = data.astype(working_dtype) * scale_factor

        if preserve_dtype:
            scaled_data = scaled_data.astype(original_dtype)

        add_history(header, f"Scaled by factor {scale_factor}")
        write_fits(
            output_file,
            scaled_data,
            header,
            overwrite=overwrite,
        )

        logger.debug("Image scaled by %.3f: %s", scale_factor, output_file)

        return create_success_result(
            output_file=output_file,
            operation="scale",
            scale_factor=scale_factor,
            shape=scaled_data.shape,
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
    ext: int = 0,
    mask_ext: Optional[int] = None,
    scalar: Optional[float] = None,
    threshold: Optional[float] = None,
    operand: Optional[Union[int, float, str, Path, np.ndarray]] = None,    
    overwrite: bool = False,
    preserve_dtype: bool = False,
    invert: bool = False,
) -> ImageOperationResult:
    """Apply a unary/scalar/comparison/bitwise/fill operation to a FITS file."""
    try:
        data, header = get_data_and_header(input_file, ext=ext)
        if data is None:
            raise ImageProcessingError(
                f"HDU {ext} in {input_file} contains no image data"
            )

        if operation_type == "unary":
            result_data = apply_unary_operation(
                data, operation, preserve_dtype=preserve_dtype
            )

        elif operation_type == "binary":
            if operand is None:
                raise ValueError("Binary operations require 'operand' parameter (second image path)")
            if not Path(operand).exists():
                raise ValueError(f"Operand file not found: {operand}")
            data2 = get_data(operand, ext=ext)
            if data2 is None:
                raise ImageProcessingError(f"HDU {ext} in {operand} contains no image data")
            operation_resolved = CLI_ALIASES.get(operation, operation)
            if operation_resolved not in BINARY_IMAGE_OPS:
                valid_ops = ", ".join(sorted(BINARY_IMAGE_OPS.keys()))
                raise ValueError(
                    f"Unknown binary operation '{operation}'. Valid options: {valid_ops}"
                )
            func, _ = BINARY_IMAGE_OPS[operation_resolved]
            with np.errstate(divide="ignore", invalid="ignore"):
                result_data = func(
                    data.astype(np.float32),
                    data2.astype(np.float32),
                )
            result_data = np.nan_to_num(result_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        elif operation_type == "scalar":
            if scalar is None:
                raise ValueError("Scalar operations require 'scalar' parameter")
            result_data = apply_scalar_operation(
                data, operation, scalar, preserve_dtype=preserve_dtype
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
                operand = get_data(Path(operand), ext=ext)

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
                    mask_data = get_data(Path(mask_data), ext=ext if mask_ext is None else mask_ext)

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
            raise ValueError(
                f"Unknown operation_type: '{operation_type}'. "
                "Valid: 'unary', 'scalar', 'comparison', 'bitwise', 'binary', 'fill'"
            )

        desc = f"{operation_type}:{operation}"
        params = []
        if scalar is not None:
            params.append(f"scalar={scalar}")
        if threshold is not None:
            params.append(f"threshold={threshold}")
        if invert:
            params.append("invert=True")

        line = f"Applied operation {desc}"
        if params:
            line += " (" + ", ".join(params) + ")"
        add_history(header, line)

        write_fits(
            output_file,
            result_data,
            header,
            overwrite=overwrite,
        )

        logger.debug("Operation complete: %s", output_file)

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
        )

# ============================================================================
# Information Functions
# ============================================================================

def list_operations(operation_type: Optional[str] = None) -> Dict[str, List[str]]:
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