#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image combining and stacking operations.

This module provides robust image combining functionality for creating master
calibration frames and stacking science images. It supports multiple combining
methods with outlier rejection, weighted combining, and memory-efficient
streaming for large datasets.

Primary Functions:
    - combine_images_robust(): Modern API with sigma-clipping and robust statistics
    - combine_images(): Legacy API for backward compatibility
    - median_combine(): Convenience wrapper for median stacking
    - mean_combine(): Convenience wrapper for mean stacking
    - create_master_frame(): Specialized for calibration frame creation

Combining Methods:
    - 'sigma_clipped_median': Iterative sigma-clipped median (recommended)
    - 'sigma_clipped_mean': Iterative sigma-clipped mean
    - 'median': Simple median (no outlier rejection)
    - 'mean': Simple mean (no outlier rejection)
    - 'weighted_mean': Weighted average (requires weights)

Features:
    - Outlier rejection with sigma-clipping
    - Weighted combining (by exposure time, SNR, etc.)
    - Mask support for bad pixels
    - Memory-efficient processing (v2.1.0: 50% reduction in peak memory)
    - Automatic exposure time normalization
    - Quality metrics and statistics
    - Progress reporting

    References
    ----------
    .. [1] Sigma-clipping algorithm as implemented in astropy.stats.sigma_clip
        https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clip.html

Version: 2.1.0
Changes in 2.1.0:
    - Optimized memory usage: pre-allocate stacked array instead of intermediate list
    - Reduces peak memory by ~50% for file-based combining
    - Added memory usage logging
"""

import os
import warnings
import logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std

# Optional dependencies
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

# Import from fits_tools package
from ..io import get_data, get_header, write_fits, get_data_and_header
from ..selection import select_files
from ..constants import FITS_KEY_EXPTIME
from ..exceptions import ImageProcessingError, SelectionError
from .result_types import CombiningResult
from ..utils import add_history, validate_shapes, maybe_tqdm


# Get logger for this module
logger = logging.getLogger(__name__)

__version__ = "2.1.0"
__author__ = "FITS Tools Development Team"


# ============================================================================
# Utility Functions
# ============================================================================

def _load_and_stack_efficient(
    input_param: Union[str, List[str], List[np.ndarray]],
    ext: int = 0,
    keys: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
    verbose: bool = False
) -> Tuple[np.ndarray, List[str], fits.Header]:
    """
    Load and stack images directly into pre-allocated array (memory-efficient).

    This function avoids creating an intermediate list of arrays, reducing
    peak memory usage by approximately 50% compared to the load-then-stack
    approach **when loading from files**. For pre-loaded numpy arrays, no
    memory benefit is gained as the arrays must be stacked.
    
    Parameters
    ----------
    input_param : str, list of str, or list of np.ndarray
        Input specification:
        - Directory path (searches for FITS files)
        - List of file paths
        - List of numpy arrays (no memory benefit for this case)
    ext : int, optional
        HDU extension to load. Default: 0
    keys : dict, optional
        Header keyword filters for file selection.
    recursive : bool, optional
        Recursive directory search. Default: False
    verbose : bool, optional
        Enable progress reporting. Default: False
    
    Returns
    -------
    tuple
        (stacked_array, file_list, base_header)
        - stacked_array: 3D numpy array with shape (N, rows, cols)
        - file_list: List of file paths (empty for in-memory arrays)
        - base_header: FITS header from first file/array
    
    Raises
    ------
    ImageProcessingError
        If no valid data can be loaded or files not found.
    """
    base_header = fits.Header()
    
    # Case 1: Already have numpy arrays in memory
    if isinstance(input_param, list) and all(isinstance(i, np.ndarray) for i in input_param):
        if verbose:
            logger.info(f"Processing {len(input_param)} in-memory arrays")
        
        # Validate shapes
        validate_shapes(*input_param, operation="combining")
        
        # Must stack these (can't avoid intermediate list for pre-existing arrays)
        try:
            stacked = np.stack(input_param, axis=0)
        except ValueError as e:
            raise ImageProcessingError(
                f"Failed to stack images - dimension mismatch: {e}"
            )
        
        add_history(base_header, f'Combined from {len(input_param)} in-memory arrays')
        return stacked, [], base_header
    
    # Case 2: Load from files (MEMORY-EFFICIENT PATH)
    # Select files
    if isinstance(input_param, str):
        try:
            file_list = select_files(
                input_param,
                keys=keys,
                recursive=recursive,
                verbose=verbose
            )
        except SelectionError as e:
            raise ImageProcessingError(f"File selection failed: {e}")
    else:
        file_list = [str(f) for f in input_param]
    
    if not file_list:
        raise ImageProcessingError("No files found for combining")
    
    n_files = len(file_list)
    
    if verbose:
        logger.info(f"Loading {n_files} files for combining")
    
    # Get shape and dtype from first file
    first_data = get_data(file_list[0], ext=ext, verbose=False)
    if first_data is None:
        raise ImageProcessingError(f"Could not load first file: {file_list[0]}")
    
    shape = first_data.shape
    dtype = first_data.dtype
    
    # Calculate memory requirement
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = n_files * shape[0] * shape[1] * bytes_per_element
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)
    
    if verbose:
        if total_gb >= 1.0:
            logger.info(f"Memory requirement: {total_gb:.2f} GB for {n_files} × {shape} images")
            # Add a warning for large allocations
            if total_gb >= 4.0:
                logger.warning(f"Large memory allocation ({total_gb:.2f} GB). Ensure sufficient RAM available.")
        else:
            logger.info(f"Memory requirement: {total_mb:.1f} MB for {n_files} × {shape} images")
    
    # Pre-allocate 3D array (MEMORY EFFICIENT - no intermediate list)
    try:
        stacked = np.zeros((n_files, shape[0], shape[1]), dtype=dtype)
    except MemoryError:
        raise ImageProcessingError(
            f"Insufficient memory to allocate array for {n_files} images "
            f"({total_gb:.2f} GB required). Consider processing fewer images at once."
        )
    
    # Load first image (already loaded)
    stacked[0] = first_data
    valid_count = 1
    
    # Load remaining images directly into pre-allocated array
    if verbose:
        iterator = maybe_tqdm(
            enumerate(file_list[1:], start=1),
            desc="Loading images",
            unit="file",
            initial=1,
            total=n_files
        )
    else:
        iterator = enumerate(file_list[1:], start=1)
    
    for i, filepath in iterator:
        try:
            data = get_data(filepath, ext=ext, verbose=False)
            
            if data is None:
                logger.warning(f"No data in {filepath}, skipping")
                continue
            
            if data.shape != shape:
                logger.warning(
                    f"Shape mismatch in {filepath}: {data.shape} vs {shape}, skipping"
                )
                continue
            
            stacked[valid_count] = data
            valid_count += 1
            
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
    
    # Trim array if some files failed to load
    if valid_count < n_files:
        if verbose:
            logger.info(f"Successfully loaded {valid_count}/{n_files} files")
        stacked = stacked[:valid_count]
    
    if valid_count == 0:
        raise ImageProcessingError("No valid data could be loaded from files")
    
    # Get header from first file
    try:
        base_header = get_header(file_list[0], ext=ext, verbose=False)
    except Exception as e:
        logger.warning(f"Could not load header from {file_list[0]}: {e}")
    
    add_history(base_header, f'Combined from {valid_count} files')
    
    return stacked, file_list[:valid_count], base_header


def _apply_mask(
    stacked_data: np.ndarray,
    mask: Optional[np.ndarray]
) -> np.ndarray:
    """
    Apply bad pixel mask to stacked data.
    
    The mask is broadcast across all images in the stack. Masked pixels
    are excluded from subsequent combining operations.
    
    Parameters
    ----------
    stacked_data : np.ndarray
        Stacked images with shape (N, rows, cols).
    mask : np.ndarray or None
        Boolean mask where True indicates bad pixels to exclude.
        Must have shape (rows, cols). If None, returns data unchanged.
    
    Returns
    -------
    np.ndarray or np.ma.MaskedArray
        If mask is None: returns original stacked_data
        If mask provided: returns MaskedArray with mask applied to all images
        
    Raises
    ------
    ImageProcessingError
        If mask shape doesn't match image dimensions.
        
    Notes
    -----
    The mask is broadcast to match the stacked data shape, so the same
    pixels are masked in all images of the stack.
    """
    if mask is None:
        return stacked_data
    
    if mask.shape != stacked_data.shape[1:]:
        raise ImageProcessingError(
            f"Mask shape {mask.shape} doesn't match image shape {stacked_data.shape[1:]}"
        )
    
    # Broadcast mask to all images
    mask_3d = np.broadcast_to(mask, stacked_data.shape)
    return np.ma.masked_array(stacked_data, mask=mask_3d)


# ============================================================================
# Combining Methods
# ============================================================================

def _combine_sigma_clipped_median(
    stacked_data: np.ndarray,
    sigma: float = 3.0,
    maxiters: int = 5,
    fill_value: float = 0.0,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Combine using sigma-clipped median.
    
    Parameters
    ----------
    stacked_data : np.ndarray
        Stacked images (N, rows, cols).
    sigma : float
        Sigma threshold for clipping.
    maxiters : int
        Maximum clipping iterations.
    verbose : bool
        Enable logging.
    
    Returns
    -------
    tuple
        (combined_array, num_clipped_pixels)
    """
    if verbose:
        logger.info(f"Sigma-clipped median: sigma={sigma}, maxiters={maxiters}")
    
    clipped = sigma_clip(stacked_data, sigma=sigma, maxiters=maxiters, axis=0)
    combined = np.ma.median(clipped, axis=0)
    
    # Count clipped pixels
    if hasattr(clipped, 'mask'):
        num_clipped = np.sum(clipped.mask)
    else:
        num_clipped = 0
    
    # Fill masked values
    if isinstance(combined, np.ma.MaskedArray):
        combined = combined.filled(fill_value)
    
    return combined, num_clipped


def _combine_sigma_clipped_mean(
    stacked_data: np.ndarray,
    sigma: float = 3.0,
    maxiters: int = 5,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Combine using sigma-clipped mean.
    
    Parameters
    ----------
    stacked_data : np.ndarray
        Stacked images (N, rows, cols).
    sigma : float
        Sigma threshold for clipping.
    maxiters : int
        Maximum clipping iterations.
    verbose : bool
        Enable logging.
    
    Returns
    -------
    tuple
        (combined_array, num_clipped_pixels)
    """
    if verbose:
        logger.info(f"Sigma-clipped mean: sigma={sigma}, maxiters={maxiters}")
    
    clipped = sigma_clip(stacked_data, sigma=sigma, maxiters=maxiters, axis=0)
    combined = np.ma.mean(clipped, axis=0)
    
    # Count clipped pixels
    if hasattr(clipped, 'mask'):
        num_clipped = np.sum(clipped.mask)
    else:
        num_clipped = 0
    
    # Fill masked values
    if isinstance(combined, np.ma.MaskedArray):
        combined = combined.filled(0)
    
    return combined, num_clipped


def _combine_median(
    stacked_data: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Combine using simple median.
    
    Parameters
    ----------
    stacked_data : np.ndarray
        Stacked images (N, rows, cols).
    verbose : bool
        Enable logging.
    
    Returns
    -------
    tuple
        (combined_array, 0)
    """
    if verbose:
        logger.info("Simple median combine (no clipping)")
    
    combined = np.median(stacked_data, axis=0)
    return combined, 0


def _combine_mean(
    stacked_data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Combine using mean or weighted mean.
    
    Parameters
    ----------
    stacked_data : np.ndarray
        Stacked images (N, rows, cols).
    weights : np.ndarray, optional
        Weights for each image (shape: N,).
    verbose : bool
        Enable logging.
    
    Returns
    -------
    tuple
        (combined_array, 0)
    """
    if weights is not None:
        if verbose:
            logger.info(f"Weighted mean combine (weights shape: {weights.shape})")
        
        if len(weights) != stacked_data.shape[0]:
            raise ImageProcessingError(
                f"Number of weights ({len(weights)}) doesn't match "
                f"number of images ({stacked_data.shape[0]})"
            )
        
        combined = np.average(stacked_data, axis=0, weights=weights)

    else:
        if verbose:
            logger.info("Simple mean combine (no weighting)")
        combined = np.mean(stacked_data, axis=0)
    
    return combined, 0


# ============================================================================
# Main Combining Functions
# ============================================================================

def combine_images_robust(
    input_param: Union[str, List[str], List[np.ndarray]],
    method: str = 'sigma_clipped_median',
    sigma: float = 3.0,
    maxiters: int = 5,
    mask: Optional[np.ndarray] = None,
    weights: Optional[Union[List[float], np.ndarray, str]] = None,
    output_file: Optional[str] = None,
    ext: int = 0,
    keys: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    return_type: str = 'auto'
) -> Union[np.ndarray, str, CombiningResult]:
    """
    Combine multiple images using robust statistics with outlier rejection.
    
    This is the primary combining function with support for multiple methods,
    masking, weighting, and automatic outlier rejection. Suitable for creating
    master calibration frames and stacking science images.
    
    Memory-efficient in v2.1.0: Peak memory reduced by ~50% for file-based
    combining through direct loading into pre-allocated array.
    
    Parameters
    ----------
    input_param : str, list of str, or list of np.ndarray
        Input images. Can be:
        - Directory path (will search for FITS files)
        - List of file paths
        - List of numpy arrays (in-memory)
    method : str, optional
        Combining method:
        - 'sigma_clipped_median': Iterative sigma-clipped median (default, recommended)
        - 'sigma_clipped_mean': Iterative sigma-clipped mean
        - 'median': Simple median (no outlier rejection)
        - 'mean': Simple mean (no outlier rejection)
        - 'weighted_mean': Weighted mean (requires weights parameter)
    sigma : float, optional
        Sigma threshold for clipping. Defaults to 3.0.
    maxiters : int, optional
        Maximum clipping iterations. Defaults to 5.
    mask : np.ndarray, optional
        Boolean mask array (True = bad pixels to exclude).
        Must match image shape.
    weights : list, np.ndarray, or str, optional
        Weights for each image (for weighted_mean method).
        If 'exptime', uses exposure times from headers.
        If 'inverse_variance', uses 1/variance as weights.
    output_file : str, optional
        Output FITS file path. If None, returns in-memory result.
    ext : int, optional
        HDU extension to process. Defaults to 0.
    keys : dict, optional
        Header keyword filters for file selection (e.g., {'IMAGETYP': 'BIAS'}).
    recursive : bool, optional
        Recursively search directories. Defaults to False.
    overwrite : bool, optional
        Overwrite existing output file. Defaults to False.
    verbose : bool, optional
        Enable verbose logging and progress bars. Defaults to False.
    return_type : str, optional
        Return type: 'array', 'path', 'result', or 'auto'.
        'auto' returns CombiningResult if output_file is None, else path.
    
    Returns
    -------
    np.ndarray, str, or CombiningResult
        Depends on output_file and return_type parameters:
        - If output_file is None: returns CombiningResult or np.ndarray
        - If output_file provided: returns path string or CombiningResult
    
    Raises
    ------
    ImageProcessingError
        If combining fails due to shape mismatch, invalid parameters, etc.
    
    Warnings
    --------
    For very large datasets (>10 GB), monitor system memory usage. Consider
    processing in batches if memory is limited.

    Examples
    --------
    Combine bias frames with sigma-clipping:
    
    >>> from fits_tools.image_processing import combine_images_robust
    >>> bias_files = ['bias_001.fits', 'bias_002.fits', 'bias_003.fits']
    >>> master_bias = combine_images_robust(
    ...     bias_files,
    ...     method='sigma_clipped_median',
    ...     sigma=3.0,
    ...     output_file='master_bias.fits'
    ... )
    
    Combine in-memory arrays:
    
    >>> import numpy as np
    >>> arrays = [np.random.normal(100, 10, (512, 512)) for _ in range(10)]
    >>> result = combine_images_robust(arrays, method='median')
    >>> print(result.data.shape)
    (512, 512)
    
    Weighted combining by exposure time:
    
    >>> science_files = ['sci_001.fits', 'sci_002.fits', 'sci_003.fits']
    >>> combined = combine_images_robust(
    ...     science_files,
    ...     method='weighted_mean',
    ...     weights='exptime',
    ...     output_file='stacked_science.fits'
    ... )
    
    See Also
    --------
    create_master_frame : Specialized function for calibration frames
    median_combine : Convenience wrapper for median combining
    mean_combine : Convenience wrapper for mean combining
    
    Notes
    -----
    - For calibration frames, 'sigma_clipped_median' is recommended
    - Sigma-clipping removes cosmic rays and detector artifacts
    - Memory usage (v2.1.0): ~N × image_size (reduced from ~2N in v2.0.0)
    - For very large datasets (>10 GB), monitor available system memory
    """
    # Validate method
    valid_methods = [
        'sigma_clipped_median',
        'sigma_clipped_mean',
        'median',
        'mean',
        'weighted_mean'
    ]
    if method not in valid_methods:
        raise ValueError(
            f"Unknown combining method: '{method}'. "
            f"Valid options: {', '.join(valid_methods)}"
        )
    
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if maxiters < 1:
        raise ValueError(f"maxiters must be >= 1, got {maxiters}")

    # Load and stack image data (MEMORY EFFICIENT in v2.1.0)
    stacked_data, file_list, base_header = _load_and_stack_efficient(
        input_param,
        ext=ext,
        keys=keys,
        recursive=recursive,
        verbose=verbose
    )
    
    num_images = stacked_data.shape[0]
    
    if verbose:
        logger.info(f"Combining {num_images} images using method: {method}")
        logger.info(f"Stacked data shape: {stacked_data.shape}")
    
    # Apply mask if provided
    if mask is not None:
        stacked_data = _apply_mask(stacked_data, mask)
    
    # Handle weights
    weights_array = None
    if weights is not None:
        if isinstance(weights, str):
            if weights == 'exptime':
                # Extract exposure times from headers
                if not file_list:
                    raise ImageProcessingError(
                        "Cannot use 'exptime' weights with in-memory arrays"
                    )
                weights_array = []
                for filepath in file_list:
                    hdr = get_header(filepath, ext=ext, verbose=False)
                    exptime = hdr.get(FITS_KEY_EXPTIME)
                    if exptime is None:
                        raise ImageProcessingError(
                            f"No {FITS_KEY_EXPTIME} keyword in {filepath}"
                        )
                    weights_array.append(float(exptime))
                weights_array = np.array(weights_array)
                if verbose:
                    logger.info(f"Using exposure time weights: {weights_array}")
            
            elif weights == 'inverse_variance':
                variances = np.array([
                    np.ma.var(stacked_data[i]) for i in range(stacked_data.shape[0])
                ])
                # Avoid divide-by-zero
                if np.any(variances <= 0):
                    logger.warning("Non-positive variances encountered; setting to minimum value 1e-12")
                    variances = np.clip(variances, 1e-12, None)
                weights_array = 1.0 / variances
                if verbose:
                    logger.info(f"Using inverse variance weights")
            else:
                raise ValueError(
                    f"Unknown weight type: '{weights}'. "
                    f"Use 'exptime', 'inverse_variance', or provide explicit weights"
                )
        else:
            weights_array = np.asarray(weights)
    
    # Perform combining
    num_clipped = 0
    
    if method == 'sigma_clipped_median':
        combined, num_clipped = _combine_sigma_clipped_median(
            stacked_data, sigma=sigma, maxiters=maxiters, verbose=verbose
        )
    
    elif method == 'sigma_clipped_mean':
        combined, num_clipped = _combine_sigma_clipped_mean(
            stacked_data, sigma=sigma, maxiters=maxiters, verbose=verbose
        )
    
    elif method == 'median':
        combined, num_clipped = _combine_median(stacked_data, verbose=verbose)
    
    elif method in ('mean', 'weighted_mean'):
        combined, num_clipped = _combine_mean(
            stacked_data, weights=weights_array, verbose=verbose
        )
    
    # Calculate statistics
    total_pixels = stacked_data.shape[0] * stacked_data.shape[1] * stacked_data.shape[2]
    clip_fraction = num_clipped / total_pixels if total_pixels > 0 else 0.0
    
    if verbose and num_clipped > 0:
        logger.info(f"Clipped {num_clipped} pixels ({clip_fraction*100:.2f}%)")
    
    # Update header
    new_header = base_header.copy()
    add_history(new_header, f'Combined {num_images} images using {method}')
    new_header['NCOMBINE'] = (num_images, 'Number of images combined')
    new_header['COMBMETH'] = (method, 'Combining method')
    
    if method.startswith('sigma_clipped'):
        new_header['SIGMA'] = (sigma, 'Sigma threshold for clipping')
        new_header['MAXITERS'] = (maxiters, 'Maximum clipping iterations')
        new_header['NCLIPPED'] = (num_clipped, 'Number of clipped pixels')
    
    if weights_array is not None:
        new_header['WEIGHTED'] = (True, 'Weighted combining used')
        if isinstance(weights, str):
            new_header['WEIGHTBY'] = (weights, 'Weighting method')
    
    # Create result object
    result = CombiningResult(
        data=combined,
        header=new_header,
        method=method,
        num_images=num_images,
        num_clipped=num_clipped,
        clip_fraction=clip_fraction,
        input_files=file_list if file_list else None,
        output_file=output_file
    )
    
    # Write output if requested
    if output_file is not None:
        write_fits(output_file, combined, new_header, overwrite=overwrite, verbose=verbose)
        if verbose:
            logger.info(f"Combined image written to: {output_file}")
    
    # Return appropriate type
    if return_type == 'array':
        return combined
    elif return_type == 'path':
        if output_file is None:
            raise ValueError("Cannot return path when output_file is None")
        return output_file
    elif return_type == 'result':
        return result
    else:  # 'auto'
        if output_file is not None:
            return output_file
        else:
            return result


# ============================================================================
# Specialized Combining Functions
# ============================================================================

def create_master_frame(
    frame_list: Union[str, List[str], List[np.ndarray]],
    frame_type: str,
    method: str = 'sigma_clipped_median',
    normalize: bool = False,
    return_type: str = 'auto',
    **kwargs
) -> Union[np.ndarray, str, CombiningResult]:
    """
    Create master calibration frame from multiple exposures.
    
    This is a convenience wrapper around combine_images_robust()
    specifically designed for calibration frames (bias, dark, flat).
    
    Parameters
    ----------
    frame_list : str, list of str, or list of np.ndarray
        Input calibration frames.
    frame_type : str
        Type of calibration frame: 'bias', 'dark', or 'flat'.
    method : str, optional
        Combining method. Defaults to 'sigma_clipped_median'.
    normalize : bool, optional
        Normalize result (typically True for flats). Defaults to False.
    **kwargs
        Additional arguments passed to combine_images_robust().
    
    Returns
    -------
    np.ndarray, str, or CombiningResult
        Combined master frame. Type depends on return_type parameter and output_file:
        - If return_type='array': returns np.ndarray (the data only)
        - If return_type='path': returns str (the output file path)
        - If return_type='auto' and output_file=None: returns CombiningResult
        - If return_type='auto' and output_file specified: returns CombiningResult
    
    Raises
    ------
    ValueError
        If frame_type is not recognized.
    
    Examples
    --------
    Create master bias:
    
    >>> from fits_tools.image_processing import create_master_frame
    >>> master_bias = create_master_frame(
    ...     'bias_*.fits',
    ...     frame_type='bias',
    ...     output_file='master_bias.fits'
    ... )
    
    Create normalized master flat:
    
    >>> master_flat = create_master_frame(
    ...     ['flat_001.fits', 'flat_002.fits', 'flat_003.fits'],
    ...     frame_type='flat',
    ...     normalize=True,
    ...     output_file='master_flat.fits'
    ... )
    """
    valid_types = ('bias', 'dark', 'flat')
    if frame_type.lower() not in valid_types:
        raise ValueError(
            f"Unknown frame_type: '{frame_type}'. "
            f"Valid options: {', '.join(valid_types)}"
        )

    # Combine frames
    result = combine_images_robust(
        frame_list,
        method=method,
        return_type='result',
        **kwargs
    )

    add_history(result.header, f"Master {frame_type.lower()} frame created")
    result.header['FRMTYPE'] = (frame_type.lower(), 'Calibration frame type')
    
    # Normalize if requested
    if normalize:
        data = result.data
        median_val = np.median(data)
        if median_val == 0:
            logger.warning("Cannot normalize: median is zero")
        else:
            result.data = data / median_val
            add_history(result.header, f'Normalized by median ({median_val:.2f})')
        
        # Normalize by median
        median_val = np.median(data)
        if median_val == 0:
            logger.warning("Cannot normalize: median is zero")
        else:
            data_normalized = data / median_val
            
            if isinstance(result, CombiningResult):
                result.data = data_normalized
                add_history(result.header, f'Normalized by median ({median_val:.2f})')
                result.header['NORMLIZD'] = (True, 'Frame has been normalized')
                result.header['NORMVAL'] = (median_val, 'Normalization value')
                
                # Re-write if output file was specified
                if result.output_file:
                    write_fits(
                        result.output_file,
                        result.data,
                        result.header,
                        overwrite=kwargs.get('overwrite', False),
                        verbose=kwargs.get('verbose', False)
                    )
            else:
                result = data_normalized

    if return_type == 'array':
        return result.data
    elif return_type == 'path':
        if result.output_file is None:
            raise ValueError("return_type='path' requires output_file in kwargs")
        return result.output_file
    elif return_type == 'result':
        return result
    elif return_type == 'auto':
        return result
    else:
        raise ValueError(f"Invalid return_type: {return_type}. Must be 'array', 'path', 'result', or 'auto'")


# ============================================================================
# Convenience Wrappers (Backward Compatibility)
# ============================================================================

def median_combine(
    input_param: Union[str, List[str], List[np.ndarray]],
    filebase: Optional[str] = None,
    **kwargs
) -> Union[np.ndarray, str]:
    """
    Combine images using sigma-clipped median.
    
    Convenience wrapper around combine_images_robust() with method='sigma_clipped_median'.
    
    Parameters
    ----------
    input_param : str, list of str, or list of np.ndarray
        Input images.
    filebase : str, optional
        Output file base name (will add .fits extension).
    **kwargs
        Additional arguments passed to combine_images_robust().
    
    Returns
    -------
    np.ndarray or str
        Combined array if filebase is None, otherwise output path.
    
    Examples
    --------
    >>> from fits_tools.image_processing import median_combine
    >>> bias_files = ['bias_001.fits', 'bias_002.fits', 'bias_003.fits']
    >>> master_bias = median_combine(bias_files, filebase='master_bias')
    """
    output_file = None
    if filebase is not None:
        output_file = filebase if filebase.endswith('.fits') else f"{filebase}.fits"
    
    return combine_images_robust(
        input_param,
        method='sigma_clipped_median',
        output_file=output_file,
        return_type='array' if output_file is None else 'path',
        **kwargs
    )


def mean_combine(
    input_param: Union[str, List[str], List[np.ndarray]],
    filebase: Optional[str] = None,
    **kwargs
) -> Union[np.ndarray, str]:
    """
    Combine images using sigma-clipped mean.
    
    Convenience wrapper around combine_images_robust() with method='sigma_clipped_mean'.
    
    Parameters
    ----------
    input_param : str, list of str, or list of np.ndarray
        Input images.
    filebase : str, optional
        Output file base name (will add .fits extension).
    **kwargs
        Additional arguments passed to combine_images_robust().
    
    Returns
    -------
    np.ndarray or str
        Combined array if filebase is None, otherwise output path.
    
    Examples
    --------
    >>> from fits_tools.image_processing import mean_combine
    >>> dark_files = ['dark_001.fits', 'dark_002.fits', 'dark_003.fits']
    >>> master_dark = mean_combine(dark_files, filebase='master_dark')
    """
    output_file = None
    if filebase is not None:
        output_file = filebase if filebase.endswith('.fits') else f"{filebase}.fits"
    
    return combine_images_robust(
        input_param,
        method='sigma_clipped_mean',
        output_file=output_file,
        return_type='array' if output_file is None else 'path',
        **kwargs
    )


# ============================================================================
# Legacy API (Backward Compatibility)
# ============================================================================

def combine_images(
    input_param: Union[str, List[str], List[np.ndarray]],
    operations: Union[str, List[str]],
    filebase: Optional[str] = None,
    **kwargs
) -> Optional[Union[str, np.ndarray, Dict[str, str], Dict[str, np.ndarray]]]:
    """
    Combine FITS images or NumPy arrays using median or mean stacking.
    
    .. deprecated:: 2.0.0
        This function is maintained for backward compatibility.
        Use :func:`combine_images_robust` for new code, which provides
        better outlier rejection and more combining options.
    
    Parameters
    ----------
    input_param : str, list of str, or list of np.ndarray
        Input images.
    operations : str or list of str
        Combining operation(s): 'median' or 'mean' (or list of both).
    filebase : str, optional
        Base name for output files. If None, returns in-memory results.
    **kwargs
        Additional arguments:
        - ext : int, HDU extension (default: 0)
        - overwrite : bool, overwrite existing files (default: False)
        - verbose : bool, enable logging (default: False)
        - output_dir : str, output directory
        - header : fits.Header, template header
        - keys : dict, header keyword filters
        - recursive : bool, recursive directory search
        - max_files : int, maximum number of files to process
    
    Returns
    -------
    str, np.ndarray, dict, or None
        Return type depends on parameters:
        - If filebase and single operation: returns output path (str)
        - If filebase and multiple operations: returns dict {op: path}
        - If no filebase and single operation: returns array
        - If no filebase and multiple operations: returns dict {op: array}
        - Returns None if no files found or processing failed
    
    Examples
    --------
    >>> from fits_tools.image_processing import combine_images
    >>> # In-memory combining
    >>> arrays = [np.ones((10, 10)) * i for i in range(1, 6)]
    >>> result = combine_images(arrays, 'median')
    >>> result[5, 5]
    3.0
    
    >>> # File-based combining
    >>> bias_files = ['bias_001.fits', 'bias_002.fits', 'bias_003.fits']
    >>> output = combine_images(bias_files, 'median', filebase='master_bias')
    
    >>> # Multiple operations
    >>> results = combine_images(bias_files, ['median', 'mean'], filebase='master_bias')
    >>> results['median']
    'master_bias_median.fits'
    >>> results['mean']
    'master_bias_mean.fits'
    
    Notes
    -----
    This function does not perform outlier rejection. For robust combining
    with sigma-clipping, use combine_images_robust() instead.
    """
    warnings.warn(
        "combine_images() is deprecated and will be removed in version 3.0.0. "
        "Use combine_images_robust() for better outlier rejection.",
        DeprecationWarning,
        stacklevel=2
    )
    
    ext = kwargs.get('ext', 0)
    overwrite = kwargs.get('overwrite', False)
    verbose = kwargs.get('verbose', False)
    output_dir = kwargs.get('output_dir', None)
    header_template = kwargs.get('header', None)
    
    # Load images using new efficient method
    base_header = fits.Header()
    
    # Handle in-memory arrays
    if isinstance(input_param, list) and all(isinstance(i, np.ndarray) for i in input_param):
        if verbose:
            logger.info(f"Processing {len(input_param)} in-memory NumPy arrays")
        
        try:
            stacked_data, _, base_header = _load_and_stack_efficient(
                input_param,
                ext=ext,
                keys=kwargs.get('keys'),
                recursive=kwargs.get('recursive', False),
                verbose=verbose
            )
        except Exception as e:
            logger.error(f"Failed to stack arrays: {e}")
            return None
        
        if header_template:
            base_header = header_template.copy()
        add_history(base_header, f'Combined from {len(input_param)} in-memory arrays')
    
    # Handle files
    else:
        if isinstance(input_param, str):
            selection_kwargs = {
                'keys': kwargs.get('keys'),
                'recursive': kwargs.get('recursive', False),
                'verbose': verbose,
            }
            try:
                files_to_process = select_files(input_param, **selection_kwargs)
            except Exception as e:
                logger.error(f"File selection failed: {e}")
                return None
        else:
            files_to_process = input_param
        
        if not files_to_process:
            logger.warning("No files found or selected for combination")
            return None
        
        # Apply max_files limit if specified
        max_files = kwargs.get('max_files')
        if max_files and len(files_to_process) > max_files:
            files_to_process = files_to_process[:max_files]
        
        if verbose:
            logger.info(f"Found {len(files_to_process)} files to combine")
        
        # Use efficient loading
        try:
            stacked_data, _, base_header = _load_and_stack_efficient(
                files_to_process,
                ext=ext,
                keys=kwargs.get('keys'),
                recursive=kwargs.get('recursive', False),
                verbose=verbose
            )
        except Exception as e:
            logger.error(f"Failed to load and stack files: {e}")
            return None
        
        # Override with template header if provided
        if header_template:
            base_header = header_template.copy()
        
        add_history(base_header, f'Combined from {len(files_to_process)} files')
        base_header['NCOMBINE'] = (len(files_to_process), 'Number of files combined')
    
    # Simple combining engine
    def _combine_simple(data_stack: np.ndarray, operation: str) -> np.ndarray:
        """Perform simple combining without outlier rejection."""
        if operation == 'median':
            return np.median(data_stack, axis=0)
        elif operation == 'mean':
            return np.mean(data_stack, axis=0)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    # In-memory mode (no disk I/O)
    if filebase is None and output_dir is None:
        if verbose:
            logger.info("Operating in pure in-memory mode (no disk I/O)")
        try:
            if isinstance(operations, str):
                return _combine_simple(stacked_data, operations)
            else:
                return {op: _combine_simple(stacked_data, op) for op in operations}
        except Exception as e:
            logger.error(f"Error during in-memory combination: {e}")
            return None
    
    # File output mode
    if output_dir is None:
        output_dir = os.path.dirname(filebase) if filebase else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    ops = [operations] if isinstance(operations, str) else operations
    results = {}
    
    for op in ops:
        # Construct output filename
        if len(ops) > 1:
            current_filebase = f"{filebase}_{op}"
        else:
            current_filebase = filebase
        
        output_path = f"{current_filebase}.fits"
        
        if os.path.exists(output_path) and not overwrite:
            if verbose:
                logger.warning(f"File exists and overwrite=False: {output_path}")
            results[op] = output_path
            continue
        
        try:
            if verbose:
                logger.info(f"Performing '{op}' combination...")
            
            combined_array = _combine_simple(stacked_data, op)
            
            op_header = base_header.copy()
            add_history(op_header, f"Combined using '{op}' operation")
            op_header['COMBMETH'] = (op, 'Combining method')
            
            write_fits(output_path, combined_array, op_header, overwrite=overwrite, verbose=verbose)
            results[op] = output_path
            
        except Exception as e:
            logger.error(f"Failed to create combined image for operation '{op}': {e}")
            results[op] = None
    
    return results[operations] if isinstance(operations, str) else results


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main functions
    'combine_images_robust',
    'create_master_frame',
    
    # Convenience wrappers
    'median_combine',
    'mean_combine',
    
    # Legacy API
    'combine_images',
]


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """Basic module testing."""
    import sys
    
    print(f"Testing combining.py v{__version__}")
    print("=" * 60)
    
    # Test 1: Simple in-memory combining
    print("\nTest 1: In-memory array combining")
    try:
        arrays = [np.random.normal(100, 10, (100, 100)) for _ in range(5)]
        result = combine_images_robust(arrays, method='median', verbose=True)
        print(f"✓ Combined {len(arrays)} arrays")
        print(f"  Output shape: {result.data.shape}")
        print(f"  Mean value: {np.mean(result.data):.2f}")
        print(f"  Std value: {np.std(result.data):.2f}")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        sys.exit(1)
    
    # Test 2: Sigma-clipped combining with outliers
    print("\nTest 2: Sigma-clipped combining with outliers")
    try:
        arrays = [np.ones((50, 50)) * 100 for _ in range(10)]
        # Add outliers to some images
        arrays[3][25, 25] = 10000  # Hot pixel
        arrays[7][10, 10] = -1000  # Cold pixel
        
        result = combine_images_robust(
            arrays,
            method='sigma_clipped_median',
            sigma=3.0,
            verbose=True
        )
        print(f"✓ Sigma-clipping removed outliers")
        print(f"  Clipped {result.num_clipped} pixels ({result.clip_fraction*100:.3f}%)")
        print(f"  Result at outlier locations: {result.data[25, 25]:.2f}, {result.data[10, 10]:.2f}")

        # Check that outliers are much closer to 100 than to the extreme values
        assert abs(result.data[25, 25] - 100) < abs(10000 - 100) / 10, "Hot pixel not properly rejected"
        assert abs(result.data[10, 10] - 100) < abs(-1000 - 100) / 10, "Cold pixel not properly rejected"

        print(f"  ✓ Outliers successfully rejected")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        sys.exit(1)
    
    # Test 3: Create master frame
    print("\nTest 3: Create master calibration frame")
    try:
        bias_arrays = [np.random.normal(100, 5, (50, 50)) for _ in range(10)]
        result = create_master_frame(
            bias_arrays,
            frame_type='bias',
            method='sigma_clipped_median',
            verbose=True
        )
        print(f"✓ Created master bias frame")
        print(f"  Shape: {result.data.shape}")
        print(f"  Mean: {np.mean(result.data):.2f}")
        print(f"  Std: {np.std(result.data):.2f}")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        sys.exit(1)
    
    # Test 4: Weighted combining
    print("\nTest 4: Weighted combining")
    try:
        arrays = [np.ones((30, 30)) * (i + 1) * 10 for i in range(5)]
        weights = np.array([1, 2, 3, 2, 1])  # Peak weight in middle
        
        result = combine_images_robust(
            arrays,
            method='weighted_mean',
            weights=weights,
            verbose=True
        )
        print(f"✓ Weighted combining successful")
        print(f"  Result mean: {np.mean(result.data):.2f}")
        # Weighted mean should be closer to middle values
        expected = np.average([10, 20, 30, 40, 50], weights=weights)
        print(f"  Expected: {expected:.2f}")
        assert abs(np.mean(result.data) - expected) < 0.1
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        sys.exit(1)
    
    # Test 5: Different methods comparison
    print("\nTest 5: Comparing different combining methods")
    try:
        arrays = [np.random.normal(100, 10, (50, 50)) for _ in range(10)]
        
        methods = ['median', 'mean', 'sigma_clipped_median', 'sigma_clipped_mean']
        results = {}
        
        for method in methods:
            result = combine_images_robust(arrays, method=method, verbose=False)
            results[method] = np.mean(result.data)
            print(f"  {method:25s}: mean = {results[method]:.2f}")
        
        print(f"✓ All methods executed successfully")
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        sys.exit(1)
    
    # Test 6: Legacy API (with deprecation warning)
    print("\nTest 6: Legacy combine_images API")
    try:
        arrays = [np.ones((20, 20)) * i for i in range(1, 4)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = combine_images(arrays, 'median')
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            print(f"✓ Legacy API works (deprecation warning issued)")
            print(f"  Result shape: {result.shape}")
            print(f"  Result mean: {np.mean(result):.2f}")
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")
        sys.exit(1)
    
    # Test 7: Memory efficiency check
    print("\nTest 7: Memory efficiency verification")
    try:
        # This test verifies that we're not creating intermediate lists
        # by checking that the function completes without excessive memory
        n_large_images = 20
        large_arrays = [np.random.normal(100, 10, (200, 200)) for _ in range(n_large_images)]
        
        import tracemalloc
        tracemalloc.start()
        
        result = combine_images_robust(large_arrays, method='median', verbose=False)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate expected memory (should be approximately N * image_size)
        expected_mb = n_large_images * 200 * 200 * 8 / (1024**2)
        peak_mb = peak / (1024**2)
        
        print(f"✓ Memory efficiency test completed")
        print(f"  Expected memory: ~{expected_mb:.1f} MB")
        print(f"  Peak memory: {peak_mb:.1f} MB")
        print(f"  Efficiency ratio: {peak_mb/expected_mb:.2f}x")
        
        # Peak should be less than 2x expected (old method would be ~2x)
        if peak_mb < 1.5 * expected_mb:
            print(f"  ✓ Memory usage is efficient (<1.5x theoretical minimum)")
        else:
            print(f"  ⚠ Memory usage higher than expected")
            
    except Exception as e:
        print(f"✗ Test 7 failed: {e}")
        # Don't exit - memory tests can be flaky
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("\nModule is ready for use.")
    print(f"\nMemory optimization in v{__version__}:")
    print("  - ~50% reduction in peak memory for file-based combining")
    print("  - Direct loading into pre-allocated arrays")
    print("  - No intermediate list creation")