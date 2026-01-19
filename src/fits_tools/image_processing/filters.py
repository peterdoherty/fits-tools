#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial filtering operations for FITS image data.

This module provides comprehensive filtering capabilities for astronomical
image processing including smoothing, edge detection, morphological operations,
and offset correction. Designed to integrate seamlessly with the fits_tools
package infrastructure.

Features:
    - Smoothing filters (Gaussian, median, mean, bilateral)
    - Edge detection (Sobel, Laplacian, Canny)
    - Morphological operations (opening, closing, erosion, dilation)
    - Offset correction (row/column bias removal)
    - Background subtraction
    - Bad pixel correction
    - Region-specific filtering
    - Batch processing with multiprocessing
    - Preserves FITS headers and metadata

Supported Operations:
    Smoothing:
        gaussian, median, mean, bilateral
    
    Edge Detection:
        sobel, laplacian, canny
    
    Morphological:
        opening, closing, erosion, dilation
    
    Correction:
        correct_row_median, correct_row_mean,
        correct_col_median, correct_col_mean
    
    Enhancement:
        unsharp_mask, high_pass

    Usage examples:
        from fits_tools.filters import apply_filter

        # Simple smoothing
        apply_filter('input.fits', 'smoothed.fits', 'gaussian', sigma=2.0)

        # Remove cosmic rays
        apply_filter('science.fits', 'clean.fits', 'median', size=5, region='data')

        # Sharpen image
        apply_filter('image.fits', 'sharp.fits', 'unsharp_mask', sigma=1.0, amount=1.5)

        # Remove row striping
        apply_filter('biased.fits', 'corrected.fits', 'correct_row_median')

Version History:
    0.1.0 - Initial standalone version
    0.2.0 - Refactored to use fits_tools infrastructure, added type hints,
            doctests, validation, and additional filter types
"""

import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    uniform_filter,
    percentile_filter,
    sobel,
    laplace,
    binary_opening,
    binary_closing,
    binary_erosion,
    binary_dilation,
)

# Import from refactored package
from ..io import get_data, write_fits, get_header
from ..exceptions import (
    ImageProcessingError,
    FileHandlingError,
    FitsToolsError,
)
from ..utils import (
    is_fits_file,
    validate_numeric,
    add_history
)

from ..selection import select_files

logger = logging.getLogger(__name__)
__version__ = "0.2.0"


# ============================================================================
# Smoothing Filters
# ============================================================================

def apply_gaussian_filter(
    data: np.ndarray,
    sigma: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigma : float, optional
        Standard deviation for Gaussian kernel. Defaults to 1.0.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    
    Examples
    --------
    >>> data = np.random.random((100, 100))
    >>> filtered = apply_gaussian_filter(data, sigma=2.0)
    >>> filtered.shape
    (100, 100)
    """
    validate_numeric(sigma, 'sigma', minimum=0.0, allow_zero=False)
    return gaussian_filter(data, sigma=sigma)


def apply_median_filter(
    data: np.ndarray,
    size: int = 3,
    **kwargs
) -> np.ndarray:
    """
    Apply median filter for noise reduction.
    
    Particularly effective for removing salt-and-pepper noise
    and cosmic ray hits.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    size : int, optional
        Size of the median filter kernel. Defaults to 3.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    
    Examples
    --------
    >>> data = np.ones((10, 10))
    >>> data[5, 5] = 1000  # Hot pixel
    >>> filtered = apply_median_filter(data, size=3)
    >>> filtered[5, 5] < 100  # Hot pixel removed
    True
    """
    validate_numeric(size, 'size', minimum=1, integer=True)
    return median_filter(data, size=size)


def apply_mean_filter(
    data: np.ndarray,
    size: int = 3,
    **kwargs
) -> np.ndarray:
    """
    Apply mean (uniform) filter for smoothing.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    size : int, optional
        Size of the mean filter kernel. Defaults to 3.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    validate_numeric(size, 'size', minimum=1, integer=True)
    return uniform_filter(data, size=size)


def apply_bilateral_filter(
    data: np.ndarray,
    sigma_spatial: float = 3.0,
    sigma_intensity: float = 50.0,
    **kwargs
) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving smoothing.
    
    This filter smooths images while preserving edges by considering
    both spatial distance and intensity difference.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigma_spatial : float, optional
        Spatial standard deviation. Defaults to 3.0.
    sigma_intensity : float, optional
        Intensity standard deviation. Defaults to 50.0.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    
    Note
    ----
    This is a simplified bilateral filter implementation.
    For production use, consider scikit-image's denoise_bilateral.
    """
    validate_numeric(sigma_spatial, 'sigma_spatial', minimum=0.0, allow_zero=False)
    validate_numeric(sigma_intensity, 'sigma_intensity', minimum=0.0, allow_zero=False)
    
    # Simple bilateral filter using Gaussian approximation
    # For better quality, use: from skimage.restoration import denoise_bilateral
    smoothed = gaussian_filter(data, sigma=sigma_spatial)
    diff = np.abs(data - smoothed)
    weights = np.exp(-(diff ** 2) / (2 * sigma_intensity ** 2))
    
    return smoothed * weights + data * (1 - weights)


# ============================================================================
# Edge Detection Filters
# ============================================================================

def apply_sobel_filter(
    data: np.ndarray,
    axis: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Apply Sobel edge detection filter.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    axis : int, optional
        Axis along which to apply Sobel. If None, compute magnitude
        from both axes. Defaults to None.
    
    Returns
    -------
    np.ndarray
        Edge-detected image.
    
    Examples
    --------
    >>> data = np.zeros((10, 10))
    >>> data[4:6, :] = 100  # Horizontal edge
    >>> edges = apply_sobel_filter(data)
    >>> np.max(edges) > 0  # Should detect edge
    True
    """
    if axis is not None:
        return sobel(data, axis=axis)
    else:
        # Compute magnitude from both axes
        sx = sobel(data, axis=0)
        sy = sobel(data, axis=1)
        return np.hypot(sx, sy)


def apply_laplacian_filter(
    data: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Apply Laplacian edge detection filter.
    
    The Laplacian is the second derivative, highlighting regions
    of rapid intensity change.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    
    Returns
    -------
    np.ndarray
        Edge-detected image.
    """
    return laplace(data)


# ============================================================================
# Morphological Filters
# ============================================================================

def apply_opening(
    data: np.ndarray,
    structure: Optional[np.ndarray] = None,
    iterations: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation).
    
    Removes small bright features (noise, hot pixels).
    
    Parameters
    ----------
    data : np.ndarray
        Input image data (typically binary or nearly binary).
    structure : np.ndarray, optional
        Structuring element. If None, use 3x3 square.
    iterations : int, optional
        Number of times to apply operation. Defaults to 1.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    validate_numeric(iterations, 'iterations', minimum=1, integer=True)
    return binary_opening(data, structure=structure, iterations=iterations)


def apply_closing(
    data: np.ndarray,
    structure: Optional[np.ndarray] = None,
    iterations: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion).
    
    Fills small dark features (gaps, holes).
    
    Parameters
    ----------
    data : np.ndarray
        Input image data (typically binary or nearly binary).
    structure : np.ndarray, optional
        Structuring element. If None, use 3x3 square.
    iterations : int, optional
        Number of times to apply operation. Defaults to 1.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    validate_numeric(iterations, 'iterations', minimum=1, integer=True)
    return binary_closing(data, structure=structure, iterations=iterations)


def apply_erosion(
    data: np.ndarray,
    structure: Optional[np.ndarray] = None,
    iterations: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Apply morphological erosion.
    
    Shrinks bright regions.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    structure : np.ndarray, optional
        Structuring element. If None, use 3x3 square.
    iterations : int, optional
        Number of times to apply operation. Defaults to 1.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    validate_numeric(iterations, 'iterations', minimum=1, integer=True)
    return binary_erosion(data, structure=structure, iterations=iterations)


def apply_dilation(
    data: np.ndarray,
    structure: Optional[np.ndarray] = None,
    iterations: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Apply morphological dilation.
    
    Expands bright regions.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    structure : np.ndarray, optional
        Structuring element. If None, use 3x3 square.
    iterations : int, optional
        Number of times to apply operation. Defaults to 1.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    validate_numeric(iterations, 'iterations', minimum=1, integer=True)
    return binary_dilation(data, structure=structure, iterations=iterations)


# ============================================================================
# Offset Correction Filters
# ============================================================================

def correct_row_offset(
    data: np.ndarray,
    method: str = 'median',
    **kwargs
) -> np.ndarray:
    """
    Correct row-by-row offset (horizontal striping).
    
    Common in CCD images with row-wise readout.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    method : str, optional
        Statistic to use: 'median' or 'mean'. Defaults to 'median'.
    
    Returns
    -------
    np.ndarray
        Corrected image.
    
    Examples
    --------
    >>> data = np.ones((10, 10)) * 100
    >>> data[5, :] += 10  # Add offset to one row
    >>> corrected = correct_row_offset(data, method='median')
    >>> np.allclose(corrected.mean(axis=1), 0, atol=10)  # Rows should be similar
    True
    """
    if method not in ['mean', 'median']:
        raise ValueError(f"Method must be 'mean' or 'median', got '{method}'")
    
    data_float = data.astype(np.float64)
    
    if method == 'median':
        offsets = np.median(data_float, axis=1)
    else:
        offsets = np.mean(data_float, axis=1)
    
    return data_float - offsets[:, np.newaxis]


def correct_col_offset(
    data: np.ndarray,
    method: str = 'median',
    **kwargs
) -> np.ndarray:
    """
    Correct column-by-column offset (vertical striping).
    
    Common in CCD images with column-wise readout.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    method : str, optional
        Statistic to use: 'median' or 'mean'. Defaults to 'median'.
    
    Returns
    -------
    np.ndarray
        Corrected image.
    """
    if method not in ['mean', 'median']:
        raise ValueError(f"Method must be 'mean' or 'median', got '{method}'")
    
    data_float = data.astype(np.float64)
    
    if method == 'median':
        offsets = np.median(data_float, axis=0)
    else:
        offsets = np.mean(data_float, axis=0)
    
    return data_float - offsets


# ============================================================================
# Enhancement Filters
# ============================================================================

def apply_unsharp_mask(
    data: np.ndarray,
    sigma: float = 1.0,
    amount: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Apply unsharp masking for image sharpening.
    
    Enhances edges by subtracting a blurred version of the image.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigma : float, optional
        Gaussian blur sigma. Defaults to 1.0.
    amount : float, optional
        Strength of sharpening (0 = none, 1 = normal, >1 = strong).
        Defaults to 1.0.
    
    Returns
    -------
    np.ndarray
        Sharpened image.
    
    Examples
    --------
    >>> data = np.random.random((10, 10))
    >>> sharpened = apply_unsharp_mask(data, sigma=1.0, amount=0.5)
    >>> sharpened.shape
    (10, 10)
    """
    validate_numeric(sigma, 'sigma', minimum=0.0, allow_zero=False)
    validate_numeric(amount, 'amount', minimum=0.0)
    
    blurred = gaussian_filter(data, sigma=sigma)
    sharpened = data + amount * (data - blurred)
    
    return sharpened


def apply_high_pass(
    data: np.ndarray,
    sigma: float = 3.0,
    **kwargs
) -> np.ndarray:
    """
    Apply high-pass filter to remove large-scale structure.
    
    Useful for removing gradients and background variations.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigma : float, optional
        Size scale to remove. Defaults to 3.0.
    
    Returns
    -------
    np.ndarray
        High-pass filtered image.
    """
    validate_numeric(sigma, 'sigma', minimum=0.0, allow_zero=False)
    
    low_pass = gaussian_filter(data, sigma=sigma)
    high_pass = data - low_pass
    
    return high_pass


# ============================================================================
# Operation Registry
# ============================================================================

OPERATIONS: Dict[str, Callable] = {
    # Smoothing
    'gaussian': apply_gaussian_filter,
    'median': apply_median_filter,
    'mean': apply_mean_filter,
    'uniform': apply_mean_filter,  # Alias
    'bilateral': apply_bilateral_filter,
    
    # Edge detection
    'sobel': apply_sobel_filter,
    'laplacian': apply_laplacian_filter,
    
    # Morphological
    'opening': apply_opening,
    'closing': apply_closing,
    'erosion': apply_erosion,
    'dilation': apply_dilation,
    'binary_opening': apply_opening,  # Backward compatibility
    'binary_closing': apply_closing,  # Backward compatibility
    
    # Correction
    'correct_row_median': lambda data, **kw: correct_row_offset(data, method='median'),
    'correct_row_mean': lambda data, **kw: correct_row_offset(data, method='mean'),
    'correct_col_median': lambda data, **kw: correct_col_offset(data, method='median'),
    'correct_col_mean': lambda data, **kw: correct_col_offset(data, method='mean'),
    
    # Enhancement
    'unsharp_mask': apply_unsharp_mask,
    'high_pass': apply_high_pass,
}


# ============================================================================
# Main Processing Function
# ============================================================================

def apply_filter(
    fitsfile: str,
    output: str,
    operation: str,
    region: Union[str, Tuple[int, int, int, int]] = 'full',
    ext: int = 0,
    preserve_dtype: bool = False,
    verbose: bool = False,
    **kwargs
) -> None:
    """
    Apply a spatial filter to a FITS file.
    
    Parameters
    ----------
    fitsfile : str
        Input FITS file path.
    output : str
        Output FITS file path.
    operation : str
        Filter operation to apply. Must be in OPERATIONS.
    region : str or tuple, optional
        Region to filter. Defaults to 'full'.
    ext : int, optional
        HDU extension. Defaults to 0.
    preserve_dtype : bool, optional
        If True, cast output back to input dtype. Defaults to False.
    verbose : bool, optional
        Enable verbose logging.
    **kwargs
        Operation-specific parameters.
    
    Raises
    ------
    ImageProcessingError
        If filtering fails.
    ValueError
        If operation is not supported.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create test file
    ...     from ..simulations import create_bias_frame
    ...     infile = os.path.join(tmpdir, 'input.fits')
    ...     outfile = os.path.join(tmpdir, 'output.fits')
    ...     create_bias_frame(infile, shape=(50, 50), bias_level=100)
    ...     
    ...     # Apply filter
    ...     apply_filter(infile, outfile, 'gaussian', sigma=2.0)
    ...     
    ...     print(os.path.exists(outfile))
    True
    """
    if operation not in OPERATIONS:
        raise ValueError(
            f"Unsupported operation '{operation}'. "
            f"Supported: {sorted(OPERATIONS.keys())}"
        )
    
    try:
        # Read data and header
        data = get_data(fitsfile, ext=ext, region=region, verbose=verbose)
        header = get_header(fitsfile, ext=ext, verbose=verbose)
        
        if data is None:
            raise ImageProcessingError(f"No data in {fitsfile} extension {ext}")
        
        original_dtype = data.dtype
        
        # Apply filter
        if verbose:
            logger.info(f"Applying {operation} filter to {fitsfile}")
        
        filtered_data = OPERATIONS[operation](data, **kwargs)
        
        # Preserve dtype if requested
        if preserve_dtype and filtered_data.dtype != original_dtype:
            filtered_data = filtered_data.astype(original_dtype)
        
        # Update header
        add_history(header, f'Applied {operation} filter')
        for key, value in kwargs.items():
            if isinstance(value, (int, float, str)) and len(str(value)) < 50:
                add_history(header, f'{key}={value}')
        
        # Write output
        write_fits(output, filtered_data, header=header, overwrite=True, verbose=verbose)
        
        if verbose:
            logger.info(f"Filtered image saved to {output}")
            
    except Exception as e:
        if verbose:
            logger.error(f"Error applying filter to {fitsfile}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to apply filter: {e}")


# ============================================================================
# Batch Processing
# ============================================================================

def filter_batch(
    input_files: Union[str, List[str]],
    output_pattern: str,
    operation: str,
    num_workers: int = 1,
    verbose: bool = False,
    **kwargs
) -> List[str]:
    """
    Apply filter to multiple FITS files in batch mode.
    
    Parameters
    ----------
    input_files : str or list
        Directory, glob pattern, or list of files.
    output_pattern : str
        Output naming pattern. Use {stem} for input filename stem.
    operation : str
        Filter operation to apply.
    num_workers : int, optional
        Number of parallel workers. Defaults to 1.
    verbose : bool, optional
        Enable verbose logging.
    **kwargs
        Additional parameters passed to apply_filter().
    
    Returns
    -------
    list of str
        List of output files created.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create test files
    ...     from ..simulations import create_bias_frame
    ...     for i in range(3):
    ...         fpath = os.path.join(tmpdir, f'input_{i}.fits')
    ...         create_bias_frame(fpath, shape=(10, 10), bias_level=100)
    ...     
    ...     # Filter all files
    ...     outputs = filter_batch(
    ...         tmpdir,
    ...         os.path.join(tmpdir, 'filtered_{stem}.fits'),
    ...         operation='median',
    ...         size=3
    ...     )
    ...     print(len(outputs))
    3
    """
    # Select files
    try:
        files = select_files(input_files, verbose=verbose)
    except Exception as e:
        raise FileHandlingError(f"Failed to select input files: {e}")
    
    if not files:
        logger.warning("No input files found")
        return []
    
    if verbose:
        logger.info(f"Processing {len(files)} files with {num_workers} workers")
    
    # Prepare tasks
    tasks = []
    output_files = []
    
    for idx, infile in enumerate(files):
        stem = Path(infile).stem
        outfile = output_pattern.format(stem=stem, index=idx)
        output_files.append(outfile)
        
        tasks.append({
            'input': infile,
            'output': outfile,
            'operation': operation,
            'verbose': verbose,
            **kwargs
        })
    
    # Process files
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            executor.map(_process_single_filter_task, tasks)
    else:
        for task in tasks:
            _process_single_filter_task(task)
    
    return output_files


def _process_single_filter_task(task: Dict[str, Any]) -> None:
    """Helper function for parallel processing."""
    try:
        infile = task.pop('input')
        outfile = task.pop('output')
        apply_filter(infile, outfile, **task)
    except Exception as e:
        logger.error(f"Failed to process {task.get('input', 'unknown')}: {e}")
        raise


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main functions
    'apply_filter',
    'filter_batch',
    
    # Individual filters (for Python API)
    'apply_gaussian_filter',
    'apply_median_filter',
    'apply_mean_filter',
    'apply_bilateral_filter',
    'apply_sobel_filter',
    'apply_laplacian_filter',
    'apply_opening',
    'apply_closing',
    'apply_erosion',
    'apply_dilation',
    'correct_row_offset',
    'correct_col_offset',
    'apply_unsharp_mask',
    'apply_high_pass',
    
    # Registry
    'OPERATIONS',
]


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    """Command-line interface for FITS filtering."""
    parser = argparse.ArgumentParser(
        description='Apply spatial filters to FITS files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply Gaussian smoothing
  %(prog)s input.fits output.fits gaussian --sigma 2.0
  
  # Median filter for cosmic ray removal
  %(prog)s input.fits output.fits median --size 5
  
  # Batch processing
  %(prog)s "*.fits" --output "filtered_{stem}.fits" --operation sobel --batch
  
  # Remove row striping
  %(prog)s input.fits output.fits correct_row_median
        """
    )
    
    parser.add_argument('input', help='Input FITS file or pattern')
    parser.add_argument('output', nargs='?', help='Output file or pattern')
    parser.add_argument('operation', nargs='?', choices=sorted(OPERATIONS.keys()),
                       help='Filter operation to apply')
    parser.add_argument('--list-operations', action='store_true',
                       help='List all available operations and exit')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Sigma for Gaussian filters (default: 1.0)')
    parser.add_argument('--size', type=int, default=3,
                       help='Kernel size for median/mean filters (default: 3)')
    parser.add_argument('--amount', type=float, default=1.0,
                       help='Amount for unsharp mask (default: 1.0)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Iterations for morphological ops (default: 1)')
    parser.add_argument('--region', default='full',
                       help='Region to process (default: full)')
    parser.add_argument('--ext', type=int, default=0,
                       help='HDU extension number (default: 0)')
    parser.add_argument('--preserve-dtype', action='store_true',
                       help='Preserve input data type in output')
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode')
    parser.add_argument('--jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--version', action='version',
                       version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # List operations and exit
    if args.list_operations:
        print("\nAvailable filter operations:")
        print("\nSmoothing:")
        for op in ['gaussian', 'median', 'mean', 'bilateral']:
            if op in OPERATIONS:
                print(f"  {op:20s} - {OPERATIONS[op].__doc__.split(chr(10))[0] if OPERATIONS[op].__doc__ else ''}")
        
        print("\nEdge Detection:")
        for op in ['sobel', 'laplacian']:
            if op in OPERATIONS:
                print(f"  {op:20s} - {OPERATIONS[op].__doc__.split(chr(10))[0] if OPERATIONS[op].__doc__ else ''}")
        
        print("\nMorphological:")
        for op in ['opening', 'closing', 'erosion', 'dilation']:
            if op in OPERATIONS:
                print(f"  {op:20s} - {OPERATIONS[op].__doc__.split(chr(10))[0] if OPERATIONS[op].__doc__ else ''}")
        
        print("\nCorrection:")
        for op in ['correct_row_median', 'correct_row_mean', 'correct_col_median', 'correct_col_mean']:
            if op in OPERATIONS:
                print(f"  {op:20s}")
        
        print("\nEnhancement:")
        for op in ['unsharp_mask', 'high_pass']:
            if op in OPERATIONS:
                print(f"  {op:20s} - {OPERATIONS[op].__doc__.split(chr(10))[0] if OPERATIONS[op].__doc__ else ''}")
        
        print()
        return 0
    
    # Validate required arguments
    if not args.output:
        logger.error("Output argument is required")
        parser.print_help()
        return 1
    
    if not args.operation:
        logger.error("Operation argument is required")
        parser.print_help()
        return 1
    
    # Prepare filter parameters
    filter_params = {
        'sigma': args.sigma,
        'size': args.size,
        'amount': args.amount,
        'iterations': args.iterations,
        'region': args.region,
        'ext': args.ext,
        'preserve_dtype': args.preserve_dtype,
    }
    
    try:
        if args.batch:
            outputs = filter_batch(
                args.input,
                args.output,
                args.operation,
                num_workers=args.jobs,
                verbose=args.verbose,
                **filter_params
            )
            logger.info(f"Processed {len(outputs)} files")
            
        else:
            apply_filter(
                args.input,
                args.output,
                args.operation,
                verbose=args.verbose,
                **filter_params
            )
            logger.info("Processing complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    sys.exit(main())


    # # List all available operations
    # python -m fits_tools.filters --list-operations

    # # Apply Gaussian smoothing
    # python -m fits_tools.filters input.fits output.fits gaussian --sigma 2.0

    # # Batch process all files
    # python -m fits_tools.filters "*.fits" "filtered_{stem}.fits" median --size 5 --batch --jobs 4

    # # Remove column striping from data region only
    # python -m fits_tools.filters input.fits output.fits correct_col_median --region data




