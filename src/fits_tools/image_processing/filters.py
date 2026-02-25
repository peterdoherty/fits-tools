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
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from dataclasses import dataclass
import hashlib
from tqdm import tqdm
import numpy as np

from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    uniform_filter,
    sobel,
    laplace,
    grey_opening,
    grey_closing,
    grey_erosion,
    grey_dilation,
)

# Import from refactored package
from ..io import get_data, write_fits, get_header
from ..exceptions import (
    ImageProcessingError,
    FileHandlingError,
)
from ..utils import add_history
from ..array_utils import validate_numeric


from ..types import StrOrPath
from ..selection import select_files

logger = logging.getLogger(__name__)
__version__ = "2.0.0"

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    input_file: Path
    output_file: Path
    success: bool
    error: Optional[Exception] = None
    checksum: Optional[str] = None
    
    def __repr__(self) -> str:
        if self.success:
            return f"ProcessingResult(success, {self.output_file})"
        return f"ProcessingResult(failed, {self.input_file}: {self.error})"
    
@dataclass
class BatchResult:
    """Complete batch processing results."""
    successful: List[Path]
    failed: List[Tuple[Path, Exception]]
    checksums: Dict[Path, str]
    
    @property
    def success_count(self) -> int:
        """Number of successfully processed files."""
        return len(self.successful)
    
    @property
    def failure_count(self) -> int:
        """Number of failed files."""
        return len(self.failed)
    
    @property
    def total_count(self) -> int:
        """Total number of files processed."""
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
    
    def __repr__(self) -> str:
        return (
            f"BatchResult(successful={self.success_count}, "
            f"failed={self.failure_count}, "
            f"rate={self.success_rate:.1%})"
        )

# ============================================================================
# Helpers
# ============================================================================

def _compute_file_checksum(filepath: Path, algorithm: str = 'sha256') -> str:
    """
    Compute checksum of a file.
    
    Parameters
    ----------
    filepath : Path
        Path to file. Must exist and be readable.
    algorithm : str, optional
        Hash algorithm (sha256, md5, etc.). Defaults to 'sha256'.
    
    Returns
    -------
    str
        Hexadecimal checksum string.
    
    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If algorithm is not supported.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot compute checksum: {filepath} not found")
    
    if not filepath.is_file():
        raise ValueError(f"Cannot compute checksum: {filepath} is not a file")
    
    try:
        hash_obj = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(
            f"Unsupported hash algorithm '{algorithm}'. "
            f"Available: {', '.join(sorted(hashlib.algorithms_available))}"
        ) from e
    
    try:
        with open(filepath, 'rb') as f:
            # Read in chunks for memory efficiency
            for chunk in iter(lambda: f.read(65536), b''):  # 64KB chunks
                hash_obj.update(chunk)
    except IOError as e:
        raise IOError(f"Failed to read {filepath} for checksum: {e}") from e
    
    return hash_obj.hexdigest()


def _process_single_filter_task(task: Dict[str, Any]) -> None:
    """
    Worker function to apply a filter to a single file.

    Designed for use with ProcessPoolExecutor. Extracts input/output
    paths from the task dictionary and calls apply_filter.

    Parameters
    ----------
    task : dict
        Dictionary with keys 'input', 'output', and filter parameters.

    Raises
    ------
    Exception
        Re-raises any exception from apply_filter for proper error tracking.

    Notes
    -----
    This function mutates the task dict by popping 'input' and 'output' keys.
    """
    infile = None
    outfile = None
    
    try:
        infile = task.pop("input")
        outfile = task.pop("output")
        apply_filter(infile, outfile, **task)
    except KeyError as e:
        logger.error("Invalid task structure, missing key: %s", e)
        raise ValueError(f"Task missing required key: {e}") from e
    except Exception as e:
        logger.error(
            "Failed to process %s: %s",
            infile if infile else "unknown file",
            e
        )
        raise


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
    sigma_val = validate_numeric(sigma, 'sigma', minimum=0.0, allow_zero=False)
    return gaussian_filter(data, sigma=sigma_val)


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
    size_val = validate_numeric(size, 'size', minimum=1, integer=True)
    return median_filter(data, size=size_val)


def apply_mean_filter(
    data: np.ndarray,
    size: int = 3,
    **kwargs
) -> np.ndarray:
    """
    Apply a mean (uniform) filter to an image for simple smoothing.

    This function performs local averaging over a square window of side
    length ``size`` using :func:`scipy.ndimage.uniform_filter`. It is a
    simple, fast low-pass filter useful for noise reduction and background
    estimation.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    size : int, optional
        Size of the mean filter kernel (in pixels). Must be >= 1.
        The same size is used in both X and Y directions.
        Defaults to 3.
    **kwargs
        Additional keyword arguments passed through to
        :func:`scipy.ndimage.uniform_filter` (e.g. ``mode``, ``cval``).

    Returns
    -------
    np.ndarray
        Filtered image array of the same shape as ``data``.

    Examples
    --------
    >>> import numpy as np
    >>> from fits_tools.image_processing.operations import apply_mean_filter
    >>>
    >>> img = np.array([[0., 0., 0.],
    ...                 [0., 1., 0.],
    ...                 [0., 0., 0.]], dtype=float)
    >>> smoothed = apply_mean_filter(img, size=3)
    >>> round(smoothed[1, 1], 3)
    0.111

    >>> # Use a larger kernel for stronger smoothing
    >>> smoothed5 = apply_mean_filter(img, size=5)
    """

    size_val = validate_numeric(size, 'size', minimum=1, integer=True)
    return uniform_filter(data, size=size_val, **kwargs)


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
    sigma_spatial_val = validate_numeric(sigma_spatial, 'sigma_spatial', minimum=0.0, allow_zero=False)
    sigma_intensity_val = validate_numeric(sigma_intensity, 'sigma_intensity', minimum=0.0, allow_zero=False)
    
    # Simple bilateral filter using Gaussian approximation
    # For better quality, use: from skimage.restoration import denoise_bilateral
    smoothed = gaussian_filter(data, sigma=sigma_spatial_val)
    diff = np.abs(data - smoothed)
    weights = np.exp(-(diff ** 2) / (2 * sigma_intensity_val ** 2))
    
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
        Input image data: grey-scale (works on binary too)
    structure : np.ndarray, optional
        Structuring element. If None, use 3x3 square.
    iterations : int, optional
        Number of times to apply operation. Defaults to 1.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    iterations_val = validate_numeric(iterations, 'iterations', minimum=1, integer=True)
    out = data
    for _ in range(iterations_val):
        out = grey_opening(out, structure=structure, mode="reflect")
    return out


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
        Input image data: grey-scale (works on binary too)
    structure : np.ndarray, optional
        Structuring element. If None, use 3x3 square.
    iterations : int, optional
        Number of times to apply operation. Defaults to 1.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    iterations_val = validate_numeric(iterations, 'iterations', minimum=1, integer=True)
    out = data
    for _ in range(iterations_val):
        out = grey_closing(out, structure=structure, mode="reflect")
    return out


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
       Input image data: grey-scale (works on binary too)
    structure : np.ndarray, optional
        Structuring element. If None, use 3x3 square.
    iterations : int, optional
        Number of times to apply operation. Defaults to 1.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    iterations_val = validate_numeric(iterations, 'iterations', minimum=1, integer=True)
    out = data
    for _ in range(iterations_val):
        out = grey_erosion(out, structure=structure, mode="reflect")
    return out


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
        Input image data: grey-scale (works on binary too)
    structure : np.ndarray, optional
        Structuring element. If None, use 3x3 square.
    iterations : int, optional
        Number of times to apply operation. Defaults to 1.
    
    Returns
    -------
    np.ndarray
        Filtered image.
    """
    iterations_val = validate_numeric(iterations, 'iterations', minimum=1, integer=True)
    out = data
    for _ in range(iterations_val):
        out = grey_dilation(out, structure=structure, mode="reflect")
    return out


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
    
    data_float = data.astype(np.float32)
    
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
    
    data_float = data.astype(np.float32)
    
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
    sigma_val = validate_numeric(sigma, 'sigma', minimum=0.0, allow_zero=False)
    amount_val = validate_numeric(amount, 'amount', minimum=0.0)
    
    blurred = gaussian_filter(data, sigma=sigma_val)
    sharpened = data + amount_val * (data - blurred)
    
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
    sigma_val = validate_numeric(sigma, 'sigma', minimum=0.0, allow_zero=False)
    
    low_pass = gaussian_filter(data, sigma=sigma_val)
    high_pass = data - low_pass
    
    return high_pass


# ============================================================================
# Operation Registry
# ============================================================================

OPERATIONS: Dict[str, Callable[..., np.ndarray]] = {
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
    input_file: StrOrPath,
    output_file: StrOrPath,
    operation: str,
    region: Union[str, Tuple[int, int, int, int]] = 'full',
    ext: int = 0,
    preserve_dtype: bool = False,
    overwrite: bool = False,
    **kwargs
) -> None:
    """
    Apply a spatial filter to a FITS file.
    
    Parameters
    ----------
    input_file : str
        Input FITS file path.
    output_file : str
        Output FITS file path.
    operation : str
        Filter operation to apply. Must be in OPERATIONS.
    region : str or tuple, optional
        Region to filter. Defaults to 'full'.
    ext : int, optional
        HDU extension. Defaults to 0.
    preserve_dtype : bool, optional
        If True, cast output back to input dtype. Defaults to False.
    overwrite : bool, optional
        If True, allow overwriting output files. Defaults to False.
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
    # Convert to Path objects early
    input_file = Path(input_file)
    output_file = Path(output_file)
    
    # Validate input exists
    if not input_file.exists():
        raise FileHandlingError(
            f"Input file does not exist: {input_file}",
            filepath=str(input_file),
        )
    
    if not input_file.is_file():
        raise FileHandlingError(f"Input path is not a file: {input_file}")
    
    # Validate output directory
    output_dir = output_file.parent
    if output_dir != Path('.') and not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Created output directory: %s", output_dir)
        except OSError as e:
            raise FileHandlingError(
                f"Cannot create output directory {output_dir}: {e}"
            ) from e
    
    # Check overwrite
    if not overwrite and output_file.exists():
        raise FileHandlingError(
            f"Output file exists and overwrite=False: {output_file}"
        )

    if operation not in OPERATIONS:
        raise ValueError(
            f"Unsupported operation '{operation}'. "
            f"Supported: {', '.join(sorted(OPERATIONS.keys()))}"
        )
    
    try:
        # Read data and header
        data = get_data(input_file, ext=ext, region=region)
        header = get_header(input_file, ext=ext)
        
        if data is None:
            raise ImageProcessingError(f"No data in {input_file} extension {ext}")
        
        original_dtype = data.dtype
        
        # Apply filter
        logger.debug("Applying %s filter to %s", operation, input_file.name)
        
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
        write_fits(output_file, filtered_data, header=header, overwrite=overwrite)
        
        logger.debug("Filtered image saved to %s", output_file)
            
    except Exception as e:
        logger.error("Error applying filter to %s: %s", input_file, e)
        raise ImageProcessingError(f"Failed to apply filter: {e}") from e


# ============================================================================
# Batch Processing
# ============================================================================

def filter_batch(
    input_files: Union[StrOrPath, Iterable[StrOrPath]],
    output_pattern: StrOrPath,
    operation: str,
    num_workers: int = 1,
    overwrite: bool = False,
    validate_outputs: bool = False,
    checksum_algorithm: str = 'sha256',
    show_progress: bool = True,
    dry_run: bool = False,
    **kwargs
) -> BatchResult:
    """
    Apply filter to multiple FITS files in batch mode.
    
    Parameters
    ----------
    input_files : str, Path, or iterable of str/Path
        Directory, glob pattern, or list of files.
    output_pattern : str or Path
        Output naming pattern. Use {stem} for input filename stem,
        {index} for enumeration. Must be an absolute path or relative
        to current directory.
    operation : str
        Filter operation to apply.
    num_workers : int, optional
        Number of parallel workers. Defaults to 1.
    overwrite : bool, optional
        If True, allow overwriting output files. Defaults to False.
    validate_outputs : bool, optional
        If True, compute and store checksums of output files. Defaults to False.
    checksum_algorithm : str, optional
        Hash algorithm for validation (sha256, md5, etc.). Defaults to 'sha256'.
    show_progress : bool, optional
        If True, display progress bar. Defaults to True.
    dry_run : bool, optional
        If True, only validate inputs and preview operations without processing.
        Defaults to False.
    **kwargs
        Additional parameters passed to apply_filter().
    
    Returns
    -------
    BatchResult
        Object containing:
        - successful: List of successfully created output file paths
        - failed: List of (input_path, exception) tuples for failed files
        - checksums: Dict mapping output paths to checksums (if validate_outputs=True)
    
    Raises
    ------
    FileHandlingError
        If input file selection fails or output validation fails.
    ValueError
        If output_pattern is invalid or missing required placeholders,
        or if checksum_algorithm is not supported.
    
    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     tmpdir = Path(tmpdir)
    ...     # Create test files
    ...     from ..simulations import create_bias_frame
    ...     for i in range(3):
    ...         fpath = tmpdir / f'input_{i}.fits'
    ...         create_bias_frame(fpath, shape=(10, 10), bias_level=100)
    ...     
    ...     # Filter all files with validation
    ...     result = filter_batch(
    ...         tmpdir,
    ...         tmpdir / 'filtered_{stem}.fits',
    ...         operation='median',
    ...         size=3,
    ...         validate_outputs=True,
    ...         show_progress=True
    ...     )
    ...     print(result)
    ...     print(f"Checksums: {len(result.checksums)}")
    BatchResult(successful=3, failed=0, rate=100.0%)
    Checksums: 3
    
    >>> # Dry run example
    >>> result = filter_batch(
    ...     tmpdir,
    ...     tmpdir / 'filtered_{stem}.fits',
    ...     operation='median',
    ...     size=3,
    ...     dry_run=True
    ... )
    >>> print(f"Would process {result.total_count} files")
    Would process 3 files
    """
    # Validate inputs
    num_workers = validate_numeric(
        num_workers,
        "num_workers",
        minimum=1,
        integer=True,
    )
    
    # Validate checksum algorithm
    if validate_outputs:
        try:
            hashlib.new(checksum_algorithm)
        except ValueError as e:
            raise ValueError(
                f"Unsupported checksum algorithm '{checksum_algorithm}'. "
                f"Available: {', '.join(hashlib.algorithms_available)}"
            ) from e
    
    # Convert output_pattern to string for formatting
    output_pattern_str = str(output_pattern)
    
    # Validate output pattern
    if '{stem}' not in output_pattern_str:
        raise ValueError(
            "output_pattern must contain '{stem}' placeholder for input filename stem"
        )
    
    # Test format string validity
    try:
        output_pattern_str.format(stem='test', index=0)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid output_pattern format string: {e}") from e

    # Select files
    try:
        files = select_files(input_files)
    except Exception as e:
        raise FileHandlingError(f"Failed to select input files: {e}") from e
    
    if not files:
        logger.warning("No input files found matching: %s", input_files)
        return BatchResult(successful=[], failed=[], checksums={})
    
    logger.info(
        "%s %d files with %d workers",
        "Would process" if dry_run else "Processing",
        len(files),
        num_workers
    )
    
    # Prepare tasks and validate outputs
    tasks = []
    output_files: List[Path] = []
    input_paths: List[Path] = []
    
    for idx, infile in enumerate(files):
        # Ensure infile is a Path
        infile_path = Path(infile)
        input_paths.append(infile_path)
        stem = infile_path.stem
        
        # Create output path
        output_str = output_pattern_str.format(stem=stem, index=idx)
        output_path = Path(output_str)
        
        # Check for output directory
        output_dir = output_path.parent
        if not dry_run:
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    logger.debug("Created output directory: %s", output_dir)
                except OSError as e:
                    raise FileHandlingError(
                        f"Cannot create output directory {output_dir}: {e}"
                    ) from e
        else:
            # In dry run, just check if directory would be writable
            if not output_dir.exists():
                logger.info("Would create directory: %s", output_dir)
        
        # Check for existing files if overwrite is False
        if not overwrite and output_path.exists():
            if dry_run:
                logger.warning("Output file exists: %s", output_path)
            else:
                raise FileHandlingError(
                    f"Output file already exists and overwrite=False: {output_path}"
                )
        
        output_files.append(output_path)
        
        tasks.append({
            'input': str(infile_path),  # Convert to str for serialization
            'output': str(output_path),
            'operation': operation,
            'overwrite': overwrite,
            **kwargs
        })
    
    # Dry run: print summary and return
    if dry_run:
        logger.info("=== DRY RUN SUMMARY ===")
        logger.info("Input files: %d", len(input_paths))
        logger.info("Operation: %s", operation)
        logger.info("Parameters: %s", kwargs)
        logger.info("Sample mappings:")
        for i, (inp, out) in enumerate(zip(input_paths[:3], output_files[:3])):
            logger.info("  %s -> %s", inp.name, out.name)
        if len(input_paths) > 3:
            logger.info("  ... and %d more", len(input_paths) - 3)
        logger.info("======================")
        
        # Return empty result for dry run
        return BatchResult(
            successful=[],
            failed=[],
            checksums={}
        )
    
    # Process files with error tracking
    results: List[ProcessingResult] = []
    
    # Progress bar setup
    progress_bar = None
    if show_progress:
        try:
            progress_bar = tqdm(
                total=len(tasks),
                desc="Processing files",
                unit="file",
                dynamic_ncols=True
            )
        except Exception as e:
            logger.warning("Could not create progress bar: %s", e)
            show_progress = False
    
    try:
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(_process_single_filter_task, task): (idx, task)
                    for idx, task in enumerate(tasks)
                }
                
                # Collect results
                for future in as_completed(future_to_task):
                    idx, task = future_to_task[future]
                    input_path = input_paths[idx]
                    output_path = output_files[idx]
                    
                    try:
                        future.result()
                        
                        # Compute checksum if requested
                        checksum = None
                        if validate_outputs and output_path.exists():
                            try:
                                checksum = _compute_file_checksum(
                                    output_path,
                                    checksum_algorithm
                                )
                            except Exception as e:
                                logger.warning(
                                    "Failed to compute checksum for %s: %s",
                                    output_path,
                                    e
                                )
                        
                        results.append(ProcessingResult(
                            input_file=input_path,
                            output_file=output_path,
                            success=True,
                            checksum=checksum
                        ))
                        
                    except Exception as e:
                        logger.error(
                            "Failed to process %s: %s",
                            task['input'],
                            str(e)
                        )
                        results.append(ProcessingResult(
                            input_file=input_path,
                            output_file=output_path,
                            success=False,
                            error=e
                        ))
                    
                    if show_progress and progress_bar:
                        progress_bar.update(1)
        else:
            for idx, task in enumerate(tasks):
                input_path = input_paths[idx]
                output_path = output_files[idx]
                
                try:
                    _process_single_filter_task(task)
                    
                    # Compute checksum if requested
                    checksum = None
                    if validate_outputs and output_path.exists():
                        try:
                            checksum = _compute_file_checksum(
                                output_path,
                                checksum_algorithm
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to compute checksum for %s: %s",
                                output_path,
                                e
                            )
                    
                    results.append(ProcessingResult(
                        input_file=input_path,
                        output_file=output_path,
                        success=True,
                        checksum=checksum
                    ))
                    
                except Exception as e:
                    logger.error(
                        "Failed to process %s: %s",
                        task['input'],
                        str(e)
                    )
                    results.append(ProcessingResult(
                        input_file=input_path,
                        output_file=output_path,
                        success=False,
                        error=e
                    ))
                
                if show_progress and progress_bar:
                    progress_bar.update(1)
    finally:
        if show_progress and progress_bar:
            progress_bar.close()
    
    # Compile results
    successful = [r.output_file for r in results if r.success]
    failed = [(r.input_file, r.error) for r in results if not r.success]
    checksums = {
        r.output_file: r.checksum
        for r in results
        if r.success and r.checksum is not None
    }
    
    # Report summary
    if failed:
        logger.warning(
            "Completed with %d failures out of %d files (%.1f%% success rate)",
            len(failed),
            len(results),
            100 * len(successful) / len(results)
        )
    else:
        logger.info("Successfully processed all %d files", len(results))
    
    if validate_outputs:
        logger.info("Computed checksums for %d output files", len(checksums))
    
    batch_result = BatchResult(
        successful=successful,
        failed=failed,
        checksums=checksums
    )
    
    return batch_result


def save_batch_report(
    result: BatchResult,
    report_path: StrOrPath,
    include_checksums: bool = True
) -> None:
    """
    Save batch processing report to a file.
    
    Parameters
    ----------
    result : BatchResult
        Batch processing result to save.
    report_path : str or Path
        Path to save report file (text or JSON based on extension).
    include_checksums : bool, optional
        Include checksum information in report. Defaults to True.
    
    Examples
    --------
    >>> result = filter_batch(...)
    >>> save_batch_report(result, 'processing_report.txt')
    >>> save_batch_report(result, 'processing_report.json')
    """
    import json
    from datetime import datetime
    
    report_path = Path(report_path)
    timestamp = datetime.now().isoformat()
    
    if report_path.suffix.lower() == '.json':
        # JSON format
        report_data = {
            'timestamp': timestamp,
            'summary': {
                'total': result.total_count,
                'successful': result.success_count,
                'failed': result.failure_count,
                'success_rate': result.success_rate
            },
            'successful_files': [str(p) for p in result.successful],
            'failed_files': [
                {'input': str(inp), 'error': str(err)}
                for inp, err in result.failed
            ]
        }
        
        if include_checksums and result.checksums:
            report_data['checksums'] = {
                str(k): v for k, v in result.checksums.items()
            }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    else:
        # Text format
        with open(report_path, 'w') as f:
            f.write(f"Batch Processing Report\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"{'=' * 60}\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"  Total files: {result.total_count}\n")
            f.write(f"  Successful: {result.success_count}\n")
            f.write(f"  Failed: {result.failure_count}\n")
            f.write(f"  Success rate: {result.success_rate:.1%}\n\n")
            
            if result.successful:
                f.write(f"Successful files ({len(result.successful)}):\n")
                for path in result.successful:
                    f.write(f"  ✓ {path}\n")
                    if include_checksums and path in result.checksums:
                        f.write(f"    {result.checksums[path]}\n")
                f.write("\n")
            
            if result.failed:
                f.write(f"Failed files ({len(result.failed)}):\n")
                for inp, err in result.failed:
                    f.write(f"  ✗ {inp}\n")
                    f.write(f"    Error: {err}\n")
    
    logger.info("Report saved to %s", report_path)


def verify_output_checksums(
    result: BatchResult,
    checksum_algorithm: str = 'sha256'
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Verify output files match their recorded checksums.
    
    Parameters
    ----------
    result : BatchResult
        Result containing checksums to verify.
    checksum_algorithm : str, optional
        Algorithm used for checksums. Defaults to 'sha256'.
    
    Returns
    -------
    tuple of (verified, corrupted)
        verified : List of Path objects that passed verification
        corrupted : List of (Path, error_msg) tuples for failed verifications
    
    Examples
    --------
    >>> result = filter_batch(..., validate_outputs=True)
    >>> verified, corrupted = verify_output_checksums(result)
    >>> if corrupted:
    ...     print(f"Warning: {len(corrupted)} files may be corrupted")
    """
    if not result.checksums:
        logger.warning("No checksums to verify in result")
        return [], []
    
    verified = []
    corrupted = []
    
    for filepath, original_checksum in result.checksums.items():
        if not filepath.exists():
            corrupted.append((filepath, "File no longer exists"))
            continue
        
        try:
            current_checksum = _compute_file_checksum(filepath, checksum_algorithm)
            if current_checksum == original_checksum:
                verified.append(filepath)
            else:
                corrupted.append((
                    filepath,
                    f"Checksum mismatch: expected {original_checksum[:16]}..., "
                    f"got {current_checksum[:16]}..."
                ))
        except Exception as e:
            corrupted.append((filepath, f"Verification failed: {e}"))
    
    return verified, corrupted

# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main functions
    'apply_filter',
    'filter_batch',
    'save_batch_report',
    'verify_output_checksums', 
    
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
  
  # Batch processing with validation
  %(prog)s "*.fits" "filtered_{stem}.fits" median --size 5 --batch --jobs 4 \\
      --validate-outputs --report results.json
  
  # Dry run to preview batch operation
  %(prog)s "*.fits" "filtered_{stem}.fits" sobel --batch --dry-run
  
  # Remove row striping from data region
  %(prog)s input.fits output.fits correct_row_median --region data
        """
    )
    
    # Positional arguments
    parser.add_argument('input', help='Input FITS file or pattern')
    parser.add_argument('output', nargs='?', help='Output file or pattern')
    parser.add_argument('operation', nargs='?', choices=sorted(OPERATIONS.keys()),
                       help='Filter operation to apply')
    
    # Information arguments
    parser.add_argument('--list-operations', action='store_true',
                       help='List all available operations and exit')
    parser.add_argument('--version', action='version',
                       version=f'%(prog)s {__version__}')
    
    # Filter-specific parameters
    filter_group = parser.add_argument_group('filter parameters')
    filter_group.add_argument('--sigma', type=float, default=1.0,
                             help='Sigma for Gaussian filters (default: 1.0)')
    filter_group.add_argument('--size', type=int, default=3,
                             help='Kernel size for median/mean filters (default: 3)')
    filter_group.add_argument('--amount', type=float, default=1.0,
                             help='Amount for unsharp mask (default: 1.0)')
    filter_group.add_argument('--iterations', type=int, default=1,
                             help='Iterations for morphological ops (default: 1)')
    filter_group.add_argument('--sigma-spatial', type=float, default=3.0,
                             help='Spatial sigma for bilateral filter (default: 3.0)')
    filter_group.add_argument('--sigma-intensity', type=float, default=50.0,
                             help='Intensity sigma for bilateral filter (default: 50.0)')
    
    # Processing options
    proc_group = parser.add_argument_group('processing options')
    proc_group.add_argument('--region', default='full',
                           help='Region to process (default: full)')
    proc_group.add_argument('--ext', type=int, default=0,
                           help='HDU extension number (default: 0)')
    proc_group.add_argument('--preserve-dtype', action='store_true',
                           help='Preserve input data type in output')
    proc_group.add_argument('--overwrite', action='store_true',
                           help='Overwrite output files if they exist')
    
    # Batch processing options
    batch_group = parser.add_argument_group('batch processing')
    batch_group.add_argument('--batch', action='store_true',
                            help='Enable batch processing mode')
    batch_group.add_argument('--jobs', type=int, default=1,
                            help='Number of parallel workers (default: 1)')
    batch_group.add_argument('--validate-outputs', action='store_true',
                            help='Compute checksums for output validation')
    batch_group.add_argument('--checksum-algorithm', default='sha256',
                            choices=['md5', 'sha1', 'sha256', 'sha512'],
                            help='Checksum algorithm (default: sha256)')
    batch_group.add_argument('--no-progress', action='store_true',
                            help='Disable progress bar')
    batch_group.add_argument('--dry-run', action='store_true',
                            help='Preview operations without processing')
    batch_group.add_argument('--report', type=str, metavar='PATH',
                            help='Save processing report to file (.txt or .json)')
    
    # Logging options
    log_group = parser.add_argument_group('logging')
    log_group.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging')
    log_group.add_argument('--quiet', action='store_true',
                          help='Suppress non-error output')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s' if not args.verbose else 
               '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # List operations and exit
    if args.list_operations:
        print("\nAvailable filter operations:")
        print("\nSmoothing:")
        for op in ['gaussian', 'median', 'mean', 'bilateral']:
            if op in OPERATIONS:
                doc = OPERATIONS[op].__doc__
                desc = doc.split('\n')[0].strip() if doc else 'No description'
                print(f"  {op:20s} - {desc}")
        
        print("\nEdge Detection:")
        for op in ['sobel', 'laplacian']:
            if op in OPERATIONS:
                doc = OPERATIONS[op].__doc__
                desc = doc.split('\n')[0].strip() if doc else 'No description'
                print(f"  {op:20s} - {desc}")
        
        print("\nMorphological:")
        for op in ['opening', 'closing', 'erosion', 'dilation']:
            if op in OPERATIONS:
                doc = OPERATIONS[op].__doc__
                desc = doc.split('\n')[0].strip() if doc else 'No description'
                print(f"  {op:20s} - {desc}")
        
        print("\nCorrection:")
        for op in ['correct_row_median', 'correct_row_mean', 
                   'correct_col_median', 'correct_col_mean']:
            if op in OPERATIONS:
                print(f"  {op:20s} - Remove row/column offsets")
        
        print("\nEnhancement:")
        for op in ['unsharp_mask', 'high_pass']:
            if op in OPERATIONS:
                doc = OPERATIONS[op].__doc__
                desc = doc.split('\n')[0].strip() if doc else 'No description'
                print(f"  {op:20s} - {desc}")
        
        print()
        return 0
    
    # Validate required arguments
    if not args.output:
        logger.error("Output file/pattern is required")
        parser.print_help()
        return 1
    
    if not args.operation:
        logger.error("Operation is required")
        parser.print_help()
        return 1
    
    # Check for conflicting options
    if args.quiet and args.verbose:
        logger.error("Cannot use --quiet and --verbose together")
        return 1
    
    if args.dry_run and not args.batch:
        logger.error("--dry-run requires --batch mode")
        return 1
    
    # Prepare filter parameters
    filter_params = {
        'sigma': args.sigma,
        'size': args.size,
        'amount': args.amount,
        'iterations': args.iterations,
        'sigma_spatial': args.sigma_spatial,
        'sigma_intensity': args.sigma_intensity,
        'region': args.region,
        'ext': args.ext,
        'preserve_dtype': args.preserve_dtype,
        'overwrite': args.overwrite,
    }
    
    try:
        if args.batch:
            # Batch processing
            result = filter_batch(
                args.input,
                args.output,
                args.operation,
                num_workers=args.jobs,
                validate_outputs=args.validate_outputs,
                checksum_algorithm=args.checksum_algorithm,
                show_progress=not args.no_progress,
                dry_run=args.dry_run,
                **filter_params
            )
            
            # Log results
            if args.dry_run:
                logger.info("Dry run complete - no files were processed")
            else:
                if result.failure_count == 0:
                    logger.info(
                        "Successfully processed all %d files",
                        result.success_count
                    )
                else:
                    logger.warning(
                        "Processed %d files with %d failures (%.1f%% success)",
                        result.total_count,
                        result.failure_count,
                        result.success_rate * 100
                    )
                    
                    # Show first few failures
                    if args.verbose:
                        for inp, err in result.failed[:5]:
                            logger.error("  Failed: %s - %s", inp.name, err)
                        if result.failure_count > 5:
                            logger.error("  ... and %d more failures", 
                                       result.failure_count - 5)
            
            # Save report if requested
            if args.report and not args.dry_run:
                try:
                    save_batch_report(
                        result,
                        args.report,
                        include_checksums=args.validate_outputs
                    )
                    logger.info("Report saved to %s", args.report)
                except Exception as e:
                    logger.error("Failed to save report: %s", e)
                    if args.verbose:
                        logger.exception("Report error details:")
            
            # Return appropriate exit code
            return 0 if result.failure_count == 0 else 1
            
        else:
            # Single file processing
            apply_filter(
                args.input,
                args.output,
                args.operation,
                **filter_params
            )
            logger.info("Processing complete: %s", args.output)
            return 0
        
    except KeyboardInterrupt:
        logger.error("Processing interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except FileHandlingError as e:
        logger.error("File handling error: %s", e)
        if args.verbose:
            logger.exception("Detailed error:")
        return 1
        
    except ImageProcessingError as e:
        logger.error("Image processing error: %s", e)
        if args.verbose:
            logger.exception("Detailed error:")
        return 1
        
    except Exception as e:
        logger.error("Unexpected error: %s", e)
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

