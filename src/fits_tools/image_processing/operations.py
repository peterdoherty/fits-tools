#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pixel-wise operations for FITS images.

This module provides comprehensive pixel-wise operations on FITS images,
including unary operations (log, sqrt), scalar arithmetic (add constant),
binary image operations (image + image), comparison operations (masking),
and bitwise operations.

Operation Categories:
    - Unary: Single-argument functions (log, sqrt, sin, cos, abs, etc.)
    - Scalar: Arithmetic with constants (add 100, multiply by 2, etc.)
    - Binary: Image arithmetic (img1 + img2, img1 - img2, etc.)
    - Comparison: Create masks (greater than, less than, equal, etc.)
    - Bitwise: Logical operations (AND, OR, XOR, NOT)

All operations support:
    - In-memory numpy arrays
    - FITS file I/O with header preservation
    - NaN/Inf handling
    - Region-specific operations
    - Batch processing

Usage Examples:
    Unary operation:
        >>> from fits_tools.image_processing.operations import apply_unary_operation
        >>> result = apply_unary_operation(data, 'log')
    
    Scalar operation:
        >>> from fits_tools.image_processing.operations import apply_scalar_operation
        >>> result = apply_scalar_operation(data, 'add', 100)
    
    Binary image operation:
        >>> from fits_tools.image_processing.operations import subtract_images
        >>> result = subtract_images('img1.fits', 'img2.fits', 'diff.fits')
    
    Masking:
        >>> from fits_tools.image_processing.operations import create_mask
        >>> mask = create_mask(data, 'greater', 1000)

Version: 2.0.0
"""

import os
import logging
from typing import Callable, Optional, Union, Tuple, Dict, List

import numpy as np
# from astropy.io import fits

# Import from fits_tools
from ..io import get_data, get_header, write_fits, get_data_and_header
from ..exceptions import ImageProcessingError, FileHandlingError
from ..utils import add_history, validate_shapes
from .result_types import ImageOperationResult, create_success_result, create_failure_result

logger = logging.getLogger(__name__)
__version__ = "2.0.0"


# ============================================================================
# Operation Registries
# ============================================================================

# Unary operations: f(x) -> y
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

# Scalar operations: f(x, scalar) -> y
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
    "set": (lambda a, s: set_value(a, s, preserve_dtype=False), "Set all pixels to scalar value"),
}

# Comparison operations: f(x, threshold) -> bool
COMPARISON_OPS: Dict[str, Tuple[Callable, str]] = {
    "greater": (np.greater, "Greater than (>)"),
    "less": (np.less, "Less than (<)"),
    "greater_equal": (np.greater_equal, "Greater than or equal (>=)"),
    "less_equal": (np.less_equal, "Less than or equal (<=)"),
    "equal": (np.equal, "Equal to (==)"),
    "not_equal": (np.not_equal, "Not equal to (!=)"),
}

# Bitwise operations: f(x, operand) -> int
BITWISE_OPS: Dict[str, Tuple[Callable, str]] = {
    "bitwise_and": (np.bitwise_and, "Bitwise AND"),
    "bitwise_or": (np.bitwise_or, "Bitwise OR"),
    "bitwise_xor": (np.bitwise_xor, "Bitwise XOR"),
    "bitwise_not": (np.bitwise_not, "Bitwise NOT"),
    "left_shift": (np.left_shift, "Left bit shift"),
    "right_shift": (np.right_shift, "Right bit shift"),
}

# Binary image operations: f(img1, img2) -> result
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

# User-friendly CLI aliases
CLI_ALIASES: Dict[str, str] = {
    # Scalar operations
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "pow": "power",
    "set": "set",
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
    handle_invalid: str = 'ignore',
    **kwargs
) -> np.ndarray:
    """
    Apply a unary operation to image data.
    
    Unary operations take a single input array and return a transformed array.
    Examples include logarithm, square root, trigonometric functions, etc.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    operation : str
        Operation name (e.g., 'log', 'sqrt', 'sin', 'abs').
        See UNARY_OPS for available operations.
    preserve_dtype : bool, optional
        If True, cast output back to input dtype. Defaults to False.
    handle_invalid : str, optional
        How to handle invalid values (see numpy.errstate).
        Options: 'ignore', 'warn', 'raise'. Defaults to 'ignore'.
    **kwargs : dict
        Additional operation-specific parameters (currently unused).
    
    Returns
    -------
    np.ndarray
        Result of applying the operation.
    
    Raises
    ------
    ValueError
        If operation is not recognized.
    ImageProcessingError
        If operation fails.
    
    Examples
    --------
    >>> data = np.array([[1, 4, 9], [16, 25, 36]], dtype=float)
    >>> result = apply_unary_operation(data, 'sqrt')
    >>> result[0, 0], result[1, 2]
    (1.0, 6.0)
    
    >>> data = np.array([[1, 10, 100], [1000, 10000, 100000]], dtype=float)
    >>> result = apply_unary_operation(data, 'log10')
    >>> result[0, 1], result[1, 2]
    (1.0, 5.0)
    """
    # Resolve alias
    operation = CLI_ALIASES.get(operation, operation)
    
    if operation not in UNARY_OPS:
        valid_ops = ', '.join(sorted(UNARY_OPS.keys()))
        raise ValueError(
            f"Unknown unary operation '{operation}'. "
            f"Valid options: {valid_ops}"
        )
    
    func, description = UNARY_OPS[operation]
    original_dtype = data.dtype
    
    try:
        # Apply operation with error handling
        with np.errstate(divide=handle_invalid, invalid=handle_invalid, over=handle_invalid):
            result = func(data)
        
        # Preserve dtype if requested
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
    handle_invalid: str = 'ignore',
    **kwargs
) -> np.ndarray:
    """
    Apply a scalar arithmetic operation to image data.
    
    Scalar operations apply arithmetic between an array and a constant value.
    Examples include adding an offset, multiplying by a gain, etc.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    operation : str
        Operation name (e.g., 'add', 'subtract', 'multiply', 'divide').
        See SCALAR_OPS for available operations.
    scalar : float
        Scalar value to use in operation.
    preserve_dtype : bool, optional
        If True, cast output back to input dtype. Defaults to False.
    handle_invalid : str, optional
        How to handle invalid values. Defaults to 'ignore'.
    **kwargs : dict
        Additional parameters (currently unused).
    
    Returns
    -------
    np.ndarray
        Result of applying the operation.
    
    Raises
    ------
    ValueError
        If operation is not recognized.
    ImageProcessingError
        If operation fails.
    
    Examples
    --------
    >>> data = np.ones((3, 3)) * 100
    >>> result = apply_scalar_operation(data, 'add', 50)
    >>> result[0, 0]
    150.0
    
    >>> data = np.ones((3, 3)) * 100
    >>> result = apply_scalar_operation(data, 'multiply', 2.0)
    >>> result[0, 0]
    200.0
    
    >>> data = np.ones((3, 3)) * 100
    >>> result = apply_scalar_operation(data, 'divide', 2.0)
    >>> result[0, 0]
    50.0
    """
    # Resolve alias
    operation = CLI_ALIASES.get(operation, operation)
    
    if operation not in SCALAR_OPS:
        valid_ops = ', '.join(sorted(SCALAR_OPS.keys()))
        raise ValueError(
            f"Unknown scalar operation '{operation}'. "
            f"Valid options: {valid_ops}"
        )
    
    func, description = SCALAR_OPS[operation]
    original_dtype = data.dtype
    
    try:
        # Apply operation with error handling
        with np.errstate(divide=handle_invalid, invalid=handle_invalid, over=handle_invalid):
            result = func(data, scalar)
        
        # Handle division by zero specifically
        if operation in ('divide', 'true_divide', 'floor_divide') and scalar == 0:
            logger.warning("Division by zero in scalar operation")
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Preserve dtype if requested
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
    **kwargs
) -> np.ndarray:
    """
    Create a boolean mask by comparing image data to a threshold.
    
    Comparison operations return boolean arrays that can be used as masks
    for selecting or filtering pixels.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    comparison : str
        Comparison operation ('greater', 'less', 'equal', etc.).
        Aliases: 'gt', 'lt', 'ge', 'le', 'eq', 'ne'.
        See COMPARISON_OPS for all available operations.
    threshold : float
        Threshold value for comparison.
    invert : bool, optional
        If True, invert the mask. Defaults to False.
    **kwargs : dict
        Additional parameters (currently unused).
    
    Returns
    -------
    np.ndarray
        Boolean mask (True where condition is met).
    
    Raises
    ------
    ValueError
        If comparison is not recognized.
    
    Examples
    --------
    >>> data = np.array([[100, 200, 300], [400, 500, 600]])
    >>> mask = create_mask(data, 'greater', 250)
    >>> mask.astype(int)
    array([[0, 0, 1],
           [1, 1, 1]])
    
    >>> mask = create_mask(data, 'less_equal', 300)
    >>> np.sum(mask)
    3
    
    >>> # Using alias
    >>> mask = create_mask(data, 'gt', 400)
    >>> mask.astype(int)
    array([[0, 0, 0],
           [0, 1, 1]])
    """
    # Resolve alias
    comparison = CLI_ALIASES.get(comparison, comparison)
    
    if comparison not in COMPARISON_OPS:
        valid_ops = ', '.join(sorted(COMPARISON_OPS.keys()))
        valid_aliases = ', '.join(sorted([k for k, v in CLI_ALIASES.items() if v in COMPARISON_OPS]))
        raise ValueError(
            f"Unknown comparison operation '{comparison}'. "
            f"Valid options: {valid_ops}. "
            f"Valid aliases: {valid_aliases}"
        )
    
    func, description = COMPARISON_OPS[comparison]
    mask = func(data, threshold)
    
    if invert:
        mask = ~mask
    
    return mask


def threshold_image(
    data: np.ndarray,
    threshold: float,
    mode: str = 'binary',
    **kwargs
) -> np.ndarray:
    """
    Threshold an image using various methods.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    threshold : float
        Threshold value.
    mode : str, optional
        Thresholding mode:
        - 'binary': Values >= threshold become 1, others become 0
        - 'zero': Values < threshold become 0, others unchanged
        - 'truncate': Values > threshold become threshold, others unchanged
        - 'to_zero_inv': Values >= threshold become 0, others unchanged
    **kwargs : dict
        Additional parameters (currently unused).
    
    Returns
    -------
    np.ndarray
        Thresholded image.
    
    Raises
    ------
    ValueError
        If mode is not recognized.
    
    Examples
    --------
    >>> data = np.array([[50, 100, 150], [200, 250, 300]], dtype=float)
    >>> result = threshold_image(data, 150, mode='binary')
    >>> result
    array([[0., 0., 1.],
           [1., 1., 1.]])
    
    >>> result = threshold_image(data, 150, mode='zero')
    >>> result
    array([[  0.,   0., 150.],
           [200., 250., 300.]])
    
    >>> result = threshold_image(data, 200, mode='truncate')
    >>> result
    array([[ 50., 100., 150.],
           [200., 200., 200.]])
    """
    if mode == 'binary':
        return (data >= threshold).astype(np.float32)
    
    elif mode == 'zero':
        result = data.copy()
        result[data < threshold] = 0
        return result
    
    elif mode == 'truncate':
        result = data.copy()
        result[data > threshold] = threshold
        return result
    
    elif mode == 'to_zero_inv':
        result = data.copy()
        result[data >= threshold] = 0
        return result
    
    else:
        raise ValueError(
            f"Unknown threshold mode: '{mode}'. "
            f"Valid modes: 'binary', 'zero', 'truncate', 'to_zero_inv'"
        )



def fill_masked(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: Union[int, float],
    invert: bool = False,
    preserve_dtype: bool = False
) -> np.ndarray:
    """Fill pixels selected by a boolean mask with a constant."""
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
    preserve_dtype: bool = False
) -> np.ndarray:
    """Set all pixels in an image to a constant value."""
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
    preserve_dtype: bool = False
) -> np.ndarray:
    """Fill pixels meeting a comparison threshold with a constant."""
    mask = create_mask(data, comparison=comparison, threshold=threshold, invert=False)
    return fill_masked(data, mask, fill_value, invert=invert, preserve_dtype=preserve_dtype)


def apply_bitwise_operation(
    data: np.ndarray,
    operation: str,
    operand: Optional[Union[int, np.ndarray]] = None,
    **kwargs
) -> np.ndarray:
    """
    Apply bitwise operation to image data.
    
    Bitwise operations work on integer representations and are useful for
    manipulating bit masks or performing low-level data manipulation.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data (will be converted to int).
    operation : str
        Bitwise operation ('bitwise_and', 'bitwise_or', 'bitwise_xor',
        'bitwise_not', 'left_shift', 'right_shift').
        Aliases: 'band', 'bor', 'bxor', 'bnot', 'lshift', 'rshift'.
    operand : int or np.ndarray, optional
        Operand for binary bitwise operations.
        Not needed for 'bitwise_not'.
    **kwargs : dict
        Additional parameters (currently unused).
    
    Returns
    -------
    np.ndarray
        Result of bitwise operation.
    
    Raises
    ------
    ValueError
        If operation is not recognized or operand is missing.
    
    Examples
    --------
    >>> data = np.array([[255, 128, 64], [32, 16, 8]], dtype=np.uint8)
    >>> result = apply_bitwise_operation(data, 'bitwise_and', 240)
    >>> result[0, 0], result[0, 1]
    (240, 128)
    
    >>> result = apply_bitwise_operation(data, 'bitwise_not')
    >>> result.dtype
    dtype('int64')
    """
    # Resolve alias
    operation = CLI_ALIASES.get(operation, operation)
    
    if operation not in BITWISE_OPS:
        valid_ops = ', '.join(sorted(BITWISE_OPS.keys()))
        raise ValueError(
            f"Unknown bitwise operation '{operation}'. "
            f"Valid options: {valid_ops}"
        )
    
    func, description = BITWISE_OPS[operation]
    
    # Convert to integer type
    data_int = data.astype(np.int64)
    
    try:
        if operation == 'bitwise_not':
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
    input_file1: str,
    input_file2: str,
    output_file: str,
    ext: int = 0,
    overwrite: bool = False,
    preserve_dtype: bool = False,
    verbose: bool = False,
    **kwargs
) -> ImageOperationResult:
    """
    Generic wrapper to perform binary operations on two FITS files.
    
    Parameters
    ----------
    operation_name : str
        Human-readable operation name.
    operation_func : callable
        Function that takes two arrays and returns result.
    input_file1 : str
        First input FITS file.
    input_file2 : str
        Second input FITS file.
    output_file : str
        Output FITS file.
    ext : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.
    preserve_dtype : bool, optional
        Preserve input dtype. Defaults to False.
    verbose : bool, optional
        Enable verbose logging.
    **kwargs : dict
        Additional parameters.
    
    Returns
    -------
    ImageOperationResult
        Result of operation.
    """
    try:
        # Load data
        data1, header1 = get_data_and_header(input_file1, ext=ext, verbose=verbose)
        data2 = get_data(input_file2, ext=ext, verbose=verbose)
        
        # Validate shapes
        validate_shapes(data1, data2, operation=operation_name)
        
        # Get original dtype
        original_dtype = data1.dtype
        
        # Apply operation
        result_data = operation_func(
            data1.astype(np.float32),
            data2.astype(np.float32)
        )
        
        # Preserve dtype if requested
        if preserve_dtype:
            result_data = result_data.astype(original_dtype)
        
        # Update header
        new_header = header1.copy()
        add_history(new_header, f"{operation_name}: {os.path.basename(input_file2)}")
        new_header['OPERAND1'] = (os.path.basename(input_file1), 'First operand')
        new_header['OPERAND2'] = (os.path.basename(input_file2), 'Second operand')
        new_header['OPERATION'] = (operation_name.upper(), 'Operation performed')
        
        # Write output
        write_fits(output_file, result_data, new_header, overwrite=overwrite, verbose=verbose)
        
        if verbose:
            logger.info(f"{operation_name} complete: {output_file}")
        
        return create_success_result(
            output_file=output_file,
            operation=operation_name,
            input_files=[input_file1, input_file2],
            shape=result_data.shape
        )
    
    except Exception as e:
        if verbose:
            logger.error(f"{operation_name} failed: {e}", exc_info=True)
        return create_failure_result(
            error=str(e),
            operation=operation_name
        )


def subtract_images(
    input_file1: str,
    input_file2: str,
    output_file: str,
    ext: int = 0,
    overwrite: bool = False,
    verbose: bool = False
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
    verbose : bool, optional
        Enable verbose logging.
    
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
        verbose=verbose
    )


def add_images(
    input_file1: str,
    input_file2: str,
    output_file: str,
    ext: int = 0,
    overwrite: bool = False,
    verbose: bool = False
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
    verbose : bool, optional
        Enable verbose logging.
    
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
        verbose=verbose
    )


def multiply_images(
    input_file1: str,
    input_file2: str,
    output_file: str,
    ext: int = 0,
    overwrite: bool = False,
    verbose: bool = False
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
    verbose : bool, optional
        Enable verbose logging.
    """
    return _apply_binary_operation_to_files(
        "Multiply",
        lambda a, b: a * b,
        input_file1,
        input_file2,
        output_file,
        ext=ext,
        overwrite=overwrite,
        verbose=verbose
    )


def divide_images(
    input_file1: str,
    input_file2: str,
    output_file: str,
    ext: int = 0,
    overwrite: bool = False,
    fill_value: float = 0.0,
    verbose: bool = False
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
    verbose : bool, optional
        Enable verbose logging.
    
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
        if num_zeros > 0 and verbose:
            logger.warning(f"Division by zero occurred in {num_zeros} pixels. Replacing with {fill_value}.")
        
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
        verbose=verbose
    )


def scale_image(
    input_file: str,
    scale_factor: float,
    output_file: str,
    ext: int = 0,
    overwrite: bool = False,
    preserve_dtype: bool = False,
    verbose: bool = False
) -> ImageOperationResult:
    """
    Scale a FITS image by a constant factor.
    
    Parameters
    ----------
    input_file : str
        Input FITS file.
    scale_factor : float
        Scaling factor.
    output_file : str
        Output FITS file path.
    ext : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.
    preserve_dtype : bool, optional
        Preserve input dtype. Defaults to False.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    ImageOperationResult
        Result of scaling.
    
    Examples
    --------
    >>> result = scale_image('input.fits', 2.0, 'scaled.fits', overwrite=True)
    >>> if result.success:
    ...     print("Scaling successful")
    """
    try:
        # Load data
        data, header = get_data_and_header(input_file, ext=ext, verbose=verbose)
        
        original_dtype = data.dtype
        
        # Scale
        scaled_data = data.astype(np.float32) * scale_factor
        
        # Preserve dtype if requested
        if preserve_dtype:
            scaled_data = scaled_data.astype(original_dtype)
        
        # Update header
        add_history(header, f"Scaled by factor {scale_factor}")
        
        # Write output
        write_fits(output_file, scaled_data, header, overwrite=overwrite, verbose=verbose)
        
        if verbose:
            logger.info(f"Image scaled by {scale_factor}: {output_file}")
        
        return create_success_result(
            output_file=output_file,
            operation="scale",
            scale_factor=scale_factor,
            shape=scaled_data.shape
        )
    
    except Exception as e:
        if verbose:
            logger.error(f"Scaling failed: {e}", exc_info=True)
        return create_failure_result(
            error=str(e),
            operation="scale"
        )


# ============================================================================
# High-Level File Operations
# ============================================================================

def apply_operation_to_file(
    input_file: str,
    output_file: str,
    operation_type: str,
    operation: str,
    ext: int = 0,
    mask_ext: Optional[int] = None,
    scalar: Optional[float] = None,
    threshold: Optional[float] = None,
    operand: Optional[Union[int, str]] = None,
    overwrite: bool = False,
    preserve_dtype: bool = False,
    invert: bool = False,
    verbose: bool = False,
) -> ImageOperationResult:
    """
    Apply any operation to a FITS file.
    
    High-level function that dispatches to appropriate operation function
    based on operation_type.
    
    Parameters
    ----------
    input_file : str
        Input FITS file.
    output_file : str
        Output FITS file.
    operation_type : str
        Type of operation: 'unary', 'scalar', 'comparison', 'bitwise'.
    operation : str
        Specific operation name.
    ext : int, optional
        HDU extension. Defaults to 0.
    mask_ext : int, optional
        HDU extension to read when using a FITS file as a mask (for fill/mask).
        Defaults to `ext` if not provided.
    scalar : float, optional
        Scalar value for scalar operations.
    threshold : float, optional
        Threshold for comparison operations.
    operand : int or str, optional
        Operand for bitwise operations.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.
    preserve_dtype : bool, optional
        Preserve input dtype. Defaults to False.
    invert: bool, optional
        Applie sonly to fill operations
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    ImageOperationResult
        Result of operation.
    
    Examples
    --------
    >>> # Apply log
    >>> result = apply_operation_to_file(
    ...     'input.fits', 'output.fits', 
    ...     operation_type='unary', operation='log',
    ...     overwrite=True
    ... )
    
    >>> # Add constant
    >>> result = apply_operation_to_file(
    ...     'input.fits', 'output.fits',
    ...     operation_type='scalar', operation='add', scalar=100,
    ...     overwrite=True
    ... )
    """
    try:
        # Load data
        data, header = get_data_and_header(input_file, ext=ext, verbose=verbose)
        
        # Apply operation based on type
        if operation_type == 'unary':
            result_data = apply_unary_operation(
                data, operation, preserve_dtype=preserve_dtype
            )
        
        elif operation_type == 'scalar':
            if scalar is None:
                raise ValueError(f"Scalar operations require 'scalar' parameter")
            result_data = apply_scalar_operation(
                data, operation, scalar, preserve_dtype=preserve_dtype
            )
        
        elif operation_type == 'comparison':
            if threshold is None:
                raise ValueError(f"Comparison operations require 'threshold' parameter")
            result_data = create_mask(data, operation, threshold)
            result_data = result_data.astype(np.uint8)
        
        elif operation_type == 'bitwise':
            if operation != 'bitwise_not' and operand is None:
                raise ValueError(f"Bitwise operations (except NOT) require 'operand' parameter")
            
            # Handle file operand
            if isinstance(operand, str) and os.path.exists(operand):
                operand = get_data(operand, ext=ext, verbose=verbose)
            
            result_data = apply_bitwise_operation(data, operation, operand)
        
        elif operation_type == 'fill':
            if scalar is None:
                raise ValueError("Fill operations require 'scalar' parameter (fill_value)")

            if operation == 'mask':
                if operand is None:
                    raise ValueError("Fill operation 'mask' requires 'operand' as a mask array or mask FITS file path")

                mask_data = operand
                if isinstance(mask_data, str) and os.path.exists(mask_data):
                    mask_ext_eff = ext if mask_ext is None else mask_ext
                    mask_data = get_data(mask_data, ext=mask_ext_eff, verbose=verbose)

                mask = (np.asarray(mask_data) != 0)

                if mask.shape != data.shape:
                    raise ImageProcessingError(f"Mask shape {mask.shape} does not match data shape {data.shape}")

                result_data = fill_masked(
                    data, mask, scalar, invert=invert, preserve_dtype=preserve_dtype
                )
            else:
                if threshold is None:
                    raise ValueError("Fill operations based on comparisons require 'threshold' parameter")

                result_data = fill_where(
                    data,
                    comparison=operation,
                    threshold=threshold,
                    fill_value=scalar,
                    invert=invert,
                    preserve_dtype=preserve_dtype
                )


        else:
            raise ValueError(
                f"Unknown operation_type: '{operation_type}'. "
                f"Valid: 'unary', 'scalar', 'comparison', 'bitwise', 'fill'"
            )

        # Update header provenance
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

        
        # Write output
        write_fits(output_file, result_data, header, overwrite=overwrite, verbose=verbose)
        
        if verbose:
            logger.info(f"Operation complete: {output_file}")
        
        return create_success_result(
            output_file=output_file,
            operation=f"{operation_type}:{operation}",
            shape=result_data.shape
        )
    
    except Exception as e:
        if verbose:
            logger.error(f"Operation failed: {e}", exc_info=True)
        return create_failure_result(
            error=f"{type(e).__name__}: {e}",
            operation=f"{operation_type}:{operation}"
        )


# ============================================================================
# Information Functions
# ============================================================================

def list_operations(operation_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available operations.
    
    Parameters
    ----------
    operation_type : str, optional
        If specified, return only operations of this type.
        Options: 'unary', 'scalar', 'comparison', 'bitwise', 'binary', 'fill'.
    
    Returns
    -------
    dict
        Dictionary of operation types and their available operations.
    
    Examples
    --------
    >>> ops = list_operations('unary')
    >>> 'log' in ops['unary']
    True
    
    >>> all_ops = list_operations()
    >>> len(all_ops.keys())
    5
    """
    operations = {
        'unary': sorted(UNARY_OPS.keys()),
        'scalar': sorted(SCALAR_OPS.keys()),
        'comparison': sorted(COMPARISON_OPS.keys()),
        'bitwise': sorted(BITWISE_OPS.keys()),
        'binary': sorted(BINARY_IMAGE_OPS.keys()),
        'fill': sorted(list(COMPARISON_OPS.keys()) + ['mask']),
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
    """
    Get description of an operation.
    
    Parameters
    ----------
    operation : str
        Operation name.
    
    Returns
    -------
    str
        Description of the operation.
    
    Raises
    ------
    ValueError
        If operation is not found.
    
    Examples
    --------
    >>> desc = get_operation_description('log')
    >>> 'logarithm' in desc.lower()
    True
    """
    # Resolve alias
    operation = CLI_ALIASES.get(operation, operation)
    
    if operation == 'mask':
        return "Fill using a boolean mask (operand)"

    # Search all registries
    for registry in [UNARY_OPS, SCALAR_OPS, COMPARISON_OPS, BITWISE_OPS, BINARY_IMAGE_OPS]:
        if operation in registry:
            return registry[operation][1]
    
    raise ValueError(f"Unknown operation: '{operation}'")


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Core operation functions
    'apply_unary_operation',
    'apply_scalar_operation',
    'create_mask',
    'threshold_image',
    'apply_bitwise_operation',
    
    # Binary image operations (file-level)
    'subtract_images',
    'add_images',
    'multiply_images',
    'divide_images',
    'scale_image',
    
    # High-level functions
    'apply_operation_to_file',
    
    # Information functions
    'list_operations',
    'get_operation_description',
    
    # Operation registries
    'UNARY_OPS',
    'SCALAR_OPS',
    'COMPARISON_OPS',
    'BITWISE_OPS',
    'BINARY_IMAGE_OPS',
    'CLI_ALIASES',
    'fill_masked',
    'fill_where',
    'set_value',
]
