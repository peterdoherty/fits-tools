#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for calculating statistics from FITS image data.

This module provides comprehensive statistical analysis tools for FITS image data,
including basic statistics, robust statistics, CCD-specific metrics, and
multi-file analysis capabilities.

Key Features:
    - Single-file statistics with region support
    - Robust statistics (sigma-clipping, MAD, percentiles)
    - CCD-specific metrics (SNR, dynamic range, read noise)
    - Multi-file statistics aggregation
    - Comparison and trend analysis

Version History:
    0.1.0 - Initial release
    0.2.0 - Enhanced StatsResult, added robust statistics, CCD metrics,
            comparison functions, NaN handling, and comprehensive doctests
    0.2.1 - Fixed imports for refactored package structure
"""

import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, mad_std
import sys
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, asdict, field

# Get a logger for this module
logger = logging.getLogger(__name__)

__version__ = "0.2.1"

# Import helper functions from other modules in the package
from .selection import select_files
from .io import get_data, get_header  # ✅ Fixed: was .file_utils
from .simulations import (  # ✅ Fixed: was .fits_simulations
    create_bias_frame,
    create_dark_frame,
    create_flat_frame,
    create_science_frame,
)
from .exceptions import (
    FileHandlingError, 
    ImageProcessingError, 
    HeaderKeywordError, 
    FitsToolsError, 
    SelectionError
)


# ============================================================================
# Statistics Result Dataclass
# ============================================================================

@dataclass
class StatsResult:
    """
    A dataclass to hold comprehensive statistical results for an image or region.
    
    Attributes:
        min_val (float): Minimum pixel value
        max_val (float): Maximum pixel value
        mean_val (float): Mean pixel value
        median_val (float): Median pixel value
        std_val (float): Standard deviation
        variance_val (float): Variance
        num_pixels (int): Number of pixels analyzed
        percentile_5 (float): 5th percentile value
        percentile_25 (float): 25th percentile (Q1)
        percentile_75 (float): 75th percentile (Q3)
        percentile_95 (float): 95th percentile
        iqr (float): Interquartile range (Q3 - Q1)
        mad (float): Median Absolute Deviation
        range_val (float): Range (max - min)
    """
    min_val: float
    max_val: float
    mean_val: float
    median_val: float
    std_val: float
    variance_val: float
    num_pixels: int = 0
    percentile_5: float = np.nan
    percentile_25: float = np.nan
    percentile_75: float = np.nan
    percentile_95: float = np.nan
    iqr: float = np.nan
    mad: float = np.nan
    range_val: float = np.nan
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert StatsResult to a dictionary.
        
        Returns:
            dict: Dictionary with all statistics.
        
        Examples:
            >>> result = StatsResult(min_val=0, max_val=100, mean_val=50, 
            ...                      median_val=50, std_val=10, variance_val=100,
            ...                      num_pixels=1000)
            >>> d = result.to_dict()
            >>> d['mean_val']
            50
        """
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation of statistics."""
        return (
            f"Statistics Summary:\n"
            f"  Mean:   {self.mean_val:.4f}\n"
            f"  Median: {self.median_val:.4f}\n"
            f"  Std:    {self.std_val:.4f}\n"
            f"  Min:    {self.min_val:.4f}\n"
            f"  Max:    {self.max_val:.4f}\n"
            f"  Pixels: {self.num_pixels}"
        )
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"StatsResult(mean={self.mean_val:.4f}, median={self.median_val:.4f}, "
            f"std={self.std_val:.4f}, n={self.num_pixels})"
        )


@dataclass
class RobustStatsResult:
    """
    Results from robust statistical analysis (e.g., sigma-clipping).
    
    Attributes:
        mean (float): Clipped mean
        median (float): Clipped median
        std (float): Clipped standard deviation
        num_pixels (int): Number of pixels after clipping
        num_clipped (int): Number of pixels rejected
        clip_fraction (float): Fraction of pixels rejected
    """
    mean: float
    median: float
    std: float
    num_pixels: int
    num_clipped: int = 0
    clip_fraction: float = 0.0
    mask: Optional[np.ndarray] = None  # ✅ NEW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"Robust Statistics (σ-clipped):\n"
            f"  Mean:     {self.mean:.4f}\n"
            f"  Median:   {self.median:.4f}\n"
            f"  Std:      {self.std:.4f}\n"
            f"  Pixels:   {self.num_pixels}\n"
            f"  Clipped:  {self.num_clipped} ({self.clip_fraction*100:.1f}%)"
        )


# For backward compatibility
def stats_to_dict(stats: StatsResult) -> dict:
    """
    Converts a StatsResult object into a JSON-serializable dictionary.
    
    Deprecated: Use StatsResult.to_dict() instead.
    
    Examples:
        >>> result = StatsResult(min_val=0, max_val=100, mean_val=50,
        ...                      median_val=50, std_val=10, variance_val=100)
        >>> d = stats_to_dict(result)
        >>> d['mean_val']
        50
    """
    return stats.to_dict()


# ============================================================================
# Core Statistics Functions
# ============================================================================

def _simple_stats(arr: np.ndarray, skipna: bool = False) -> StatsResult:
    """
    Fast O(N) calculation of basic stats. 
    Bypasses sorting (Median/Percentiles) for speed.
    """
    if skipna:
        # Handling NaNs is slightly slower but still O(N)
        f_min, f_max = np.nanmin(arr), np.nanmax(arr)
        f_mean, f_std = np.nanmean(arr), np.nanstd(arr)
        f_var = np.nanvar(arr)
        pixels = int(np.sum(~np.isnan(arr)))
    else:
        f_min, f_max = np.min(arr), np.max(arr)
        f_mean, f_std = np.mean(arr), np.std(arr)
        f_var = np.var(arr)
        pixels = int(arr.size)

    return StatsResult(
        min_val=float(f_min),
        max_val=float(f_max),
        mean_val=float(f_mean),
        std_val=float(f_std),
        variance_val=float(f_var),
        num_pixels=pixels,
        median_val=np.nan,  # Explicitly NaN for simple mode
        range_val=float(f_max - f_min)
    )


def _array_stats(
    arr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    skipna: bool = False,
    compute_extended: bool = True
) -> StatsResult:
    """
    Calculates comprehensive statistics for a NumPy array or Region of Interest.

    Parameters
    ----------
    arr : np.ndarray
        The input NumPy array (image data).
    roi : tuple, optional
        A tuple (start_col, end_col, start_row, end_row) defining the region
        of interest. Only applies to 2D arrays.
    skipna : bool, optional
        If True, use NaN-safe functions (nanmean, nanstd, etc.). Defaults to False.
    compute_extended : bool, optional
        If True, compute extended statistics (percentiles, IQR, MAD). Defaults to True.

    Returns
    -------
    StatsResult
        Object with comprehensive statistics.

    Raises
    ------
    TypeError
        If `arr` is not a NumPy array.
    ValueError
        If ROI is invalid or out of bounds.
    ImageProcessingError
        If an error occurs during calculation.
    
    Examples
    --------
    >>> data = np.random.normal(100, 10, size=(100, 100))
    >>> result = _array_stats(data)
    >>> 95 < result.mean_val < 105
    True
    >>> result.num_pixels
    10000
    
    >>> # With ROI
    >>> result_roi = _array_stats(data, roi=(0, 50, 0, 50))
    >>> result_roi.num_pixels
    2500
    
    >>> # With NaN values
    >>> data_with_nan = data.copy()
    >>> data_with_nan[0, 0] = np.nan
    >>> result_nan = _array_stats(data_with_nan, skipna=True)
    >>> result_nan.num_pixels
    9999
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input 'arr' must be a NumPy array.")

    if arr.ndim >= 3:
        arr = arr.flatten()

    if arr.ndim == 2 and roi is not None:
        if len(roi) != 4:
            raise ValueError(
                "ROI must be a tuple of 4 integers (start_col, end_col, start_row, end_row)."
            )
        start_col, end_col, start_row, end_row = roi

        if not (0 <= start_row < end_row <= arr.shape[0] and 
                0 <= start_col < end_col <= arr.shape[1]):
            raise ValueError(
                f"Invalid ROI {roi} for array of shape {arr.shape}. "
                f"ROI coordinates are out of bounds."
            )

        data_to_analyze = arr[start_row:end_row, start_col:end_col]
    else:
        data_to_analyze = arr

    if data_to_analyze.size == 0:
        logger.warning("Attempting to calculate statistics on an empty array. Returning NaNs.")
        return StatsResult(
            min_val=np.nan, max_val=np.nan, mean_val=np.nan,
            median_val=np.nan, std_val=np.nan, variance_val=np.nan,
            num_pixels=0
        )

    try:
        # Choose functions based on skipna
        if skipna:
            min_func = np.nanmin
            max_func = np.nanmax
            mean_func = np.nanmean
            median_func = np.nanmedian
            std_func = np.nanstd
            var_func = np.nanvar
            percentile_func = np.nanpercentile
        else:
            min_func = np.min
            max_func = np.max
            mean_func = np.mean
            median_func = np.median
            std_func = np.std
            var_func = np.var
            percentile_func = np.percentile
        
        # Count non-NaN pixels
        if skipna:
            num_pixels = int(np.sum(~np.isnan(data_to_analyze)))
        else:
            num_pixels = int(data_to_analyze.size)
        
        # Basic statistics
        min_val = float(min_func(data_to_analyze))
        max_val = float(max_func(data_to_analyze))
        mean_val = float(mean_func(data_to_analyze))
        median_val = float(median_func(data_to_analyze))
        std_val = float(std_func(data_to_analyze))
        variance_val = float(var_func(data_to_analyze))
        range_val = max_val - min_val
        
        # Extended statistics - ✅ OPTIMIZED: compute all percentiles in one call
        if compute_extended:
            percentiles = percentile_func(data_to_analyze, [5, 25, 75, 95])
            p5, p25, p75, p95 = map(float, percentiles)
            iqr = p75 - p25
            
            # MAD (Median Absolute Deviation)
            try:
                mad = float(mad_std(data_to_analyze, ignore_nan=skipna))
            except:
                mad = np.nan
        else:
            p5 = p25 = p75 = p95 = iqr = mad = np.nan
        
        return StatsResult(
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            median_val=median_val,
            std_val=std_val,
            variance_val=variance_val,
            num_pixels=num_pixels,
            percentile_5=p5,
            percentile_25=p25,
            percentile_75=p75,
            percentile_95=p95,
            iqr=iqr,
            mad=mad,
            range_val=range_val
        )
    except Exception as e:
        logger.error(f"Error in _array_stats: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to calculate statistics: {e}")


# ============================================================================
# Single-File Statistics
# ============================================================================

# Mapping for stats_type aliases
STATS_TYPE_ALIASES = {
    'min': 'min_val',
    'max': 'max_val',
    'mean': 'mean_val',
    'median': 'median_val',
    'std': 'std_val',
    'stddev': 'std_val',
    'variance': 'variance_val',
    'var': 'variance_val',
    'rms': 'std_val',
    'dev': 'std_val',
    'range': 'range_val',
    'iqr': 'iqr',
    'mad': 'mad',
    'p5': 'percentile_5',
    'p25': 'percentile_25',
    'p75': 'percentile_75',
    'p95': 'percentile_95',
}

def stats(
    fitsfile: str,
    ext: int = 0,
    region: Union[str, Tuple[int, int, int, int]] = 'full',
    stats_type: str = 'all',
    skipna: bool = False,
    verbose: bool = False
) -> Union[float, StatsResult, None]:
    """
    Returns comprehensive statistics for a single FITS file.

    Parameters
    ----------
    fitsfile : str
        Path to the FITS file.
    ext : int, optional
        HDU extension number. Defaults to 0.
    region : str or tuple, optional
        Region to analyze. Can be 'full', 'data', 'bias', 'trim', 'dark',
        or a tuple (start_col, end_col, start_row, end_row). Defaults to 'full'.
    stats_type : str, optional
        Type of statistic to return. Options: 'all', 'min', 'max', 'mean',
        'median', 'std', 'variance', 'range', 'iqr', 'mad', 'p5', 'p25',
        'p75', 'p95'. Defaults to 'all'.
    skipna : bool, optional
        If True, ignore NaN values in calculations. Defaults to False.
    verbose : bool, optional
        Enable verbose logging. Defaults to False.

    Returns
    -------
    StatsResult or float or None
        If stats_type is 'all', returns StatsResult object.
        Otherwise returns float value of requested statistic.
        Returns None if operation fails.

    Examples
    --------
    >>> import tempfile
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'test.fits')
    ...     create_bias_frame(fpath, shape=(100, 100), bias_level=100, read_noise=5, seed=42)
    ...     result = stats(fpath, stats_type='all')
    ...     print(95 < result.mean_val < 105)
    ...     print(result.num_pixels)
    ...     mean_only = stats(fpath, stats_type='mean')
    ...     print(95 < mean_only < 105)
    True
    10000
    True
    """
    try:
        data = None
        roi_tuple = None

        if isinstance(region, tuple):
            # Get full data and apply ROI later
            data = get_data(fitsfile, ext=ext, region='full', verbose=verbose)
            roi_tuple = region
        else:
            # Get pre-sliced data
            data = get_data(fitsfile, ext=ext, region=region, verbose=verbose)

        if data is None:
            if verbose:
                logger.warning(
                    f"No valid data retrieved for {fitsfile} ext {ext}, "
                    f"region '{region}'. Returning None."
                )
            return None
        
        # Calculate statistics

        if stats_type.lower() == 'simple':
            return _simple_stats(data, skipna=skipna)
        
        all_stats = _array_stats(data, roi=roi_tuple, skipna=skipna)

        if stats_type == 'all':
            return all_stats
        else:
            # Look up the attribute name
            attr_name = STATS_TYPE_ALIASES.get(stats_type.lower())
            if attr_name is None:
                valid_types = list(STATS_TYPE_ALIASES.keys()) + ['all']
                raise ValueError(
                    f"Unsupported stats_type: '{stats_type}'. "
                    f"Valid options: {valid_types}"
                )
            return getattr(all_stats, attr_name)

    except (FileHandlingError, ValueError, HeaderKeywordError, ImageProcessingError) as e:
        if verbose:
            logger.error(f"Error calculating stats for {fitsfile}: {e}")
        raise
    except Exception as e:
        if verbose:
            logger.error(
                f"An unexpected error occurred while calculating statistics "
                f"for {fitsfile}: {e}", exc_info=True
            )
        raise FitsToolsError(f"An unexpected error occurred: {e}")


# Convenience functions for single statistics
def get_image_min(
    fitsfile: str, 
    ext: int = 0, 
    region: Union[str, Tuple[int, int, int, int]] = 'full', 
    verbose: bool = False
) -> Optional[float]:
    """
    Returns the minimum pixel value from a specified FITS image region.
    
    Examples:
        >>> import tempfile, os
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fpath = os.path.join(tmpdir, 'test.fits')
        ...     create_bias_frame(fpath, shape=(10, 10), bias_level=100, read_noise=0, seed=42)
        ...     min_val = get_image_min(fpath)
        ...     print(99 < min_val < 101)
        True
    """
    return stats(fitsfile, ext=ext, region=region, stats_type='min', verbose=verbose)


def get_image_max(
    fitsfile: str, 
    ext: int = 0, 
    region: Union[str, Tuple[int, int, int, int]] = 'full', 
    verbose: bool = False
) -> Optional[float]:
    """Returns the maximum pixel value from a specified FITS image region."""
    return stats(fitsfile, ext=ext, region=region, stats_type='max', verbose=verbose)


def get_image_mean(
    fitsfile: str, 
    ext: int = 0, 
    region: Union[str, Tuple[int, int, int, int]] = 'full', 
    verbose: bool = False
) -> Optional[float]:
    """Returns the mean pixel value from a specified FITS image region."""
    return stats(fitsfile, ext=ext, region=region, stats_type='mean', verbose=verbose)


def get_image_median(
    fitsfile: str, 
    ext: int = 0, 
    region: Union[str, Tuple[int, int, int, int]] = 'full', 
    verbose: bool = False
) -> Optional[float]:
    """Returns the median pixel value from a specified FITS image region."""
    return stats(fitsfile, ext=ext, region=region, stats_type='median', verbose=verbose)


def get_image_std(
    fitsfile: str, 
    ext: int = 0, 
    region: Union[str, Tuple[int, int, int, int]] = 'full', 
    verbose: bool = False
) -> Optional[float]:
    """Returns the standard deviation of pixel values from a specified FITS image region."""
    return stats(fitsfile, ext=ext, region=region, stats_type='std', verbose=verbose)


# ============================================================================
# Robust Statistics (NEW in v0.2.0)
# ============================================================================

def robust_stats(
    fitsfile: str,
    ext: int = 0,
    region: Union[str, Tuple[int, int, int, int]] = 'full',
    sigma: float = 3.0,
    maxiters: int = 5,
    return_mask: bool = False,  # ✅ NEW PARAMETER
    verbose: bool = False
) -> RobustStatsResult:
    """
    Calculate robust statistics using sigma-clipping to reject outliers.
    
    Useful for CCD data with hot pixels, cosmic rays, or other outliers.
    
    Parameters
    ----------
    fitsfile : str
        Path to the FITS file.
    ext : int, optional
        HDU extension number. Defaults to 0.
    region : str or tuple, optional
        Region to analyze. Defaults to 'full'.
    sigma : float, optional
        Number of standard deviations for clipping. Defaults to 3.0.
    maxiters : int, optional
        Maximum number of clipping iterations. Defaults to 5.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    RobustStatsResult
        Object containing clipped statistics.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'test.fits')
    ...     create_science_frame(fpath, shape=(100, 100), num_cosmic_rays=10, seed=42)
    ...     result = robust_stats(fpath, sigma=3.0)
    ...     print(result.num_clipped > 0)
    ...     print(result.clip_fraction < 0.1)
    True
    True
    """
    try:
        data = None
        roi_tuple = None

        if isinstance(region, tuple):
            data = get_data(fitsfile, ext=ext, region='full', verbose=verbose)
            roi_tuple = region
        else:
            data = get_data(fitsfile, ext=ext, region=region, verbose=verbose)

        if data is None:
            raise ImageProcessingError(f"No valid data retrieved from {fitsfile}")
        
        # Apply ROI if specified
        if roi_tuple is not None:
            start_col, end_col, start_row, end_row = roi_tuple
            data = data[start_row:end_row, start_col:end_col]
        
        # Perform sigma clipping
        original_size = int(data.size)
        clipped = sigma_clip(data, sigma=sigma, maxiters=maxiters, masked=True)
        good = clipped.compressed()

        num_pixels = int(good.size)
        num_clipped = int(original_size - num_pixels)
        clip_fraction = (num_clipped / original_size) if original_size > 0 else 0.0

        # Calculate statistics on unclipped pixels
        if num_pixels == 0:
            mean_val = float('nan')
            median_val = float('nan')
            std_val = float('nan')
        else:
            mean_val = float(np.mean(good))
            median_val = float(np.median(good))
            std_val = float(np.std(good))
        
        # ✅ NEW: Optionally return the mask
        mask_2d = None
        if return_mask and hasattr(clipped, 'mask'):
            mask_2d = clipped.mask  # True for clipped pixels

        return RobustStatsResult(
            mean=mean_val,
            median=median_val,
            std=std_val,
            num_pixels=num_pixels,
            num_clipped=num_clipped,
            clip_fraction=clip_fraction,
            mask=mask_2d,  # ✅ NEW
        )
        
    except Exception as e:
        if verbose:
            logger.error(f"Error in robust_stats for {fitsfile}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to calculate robust statistics: {e}")


# ============================================================================
# CCD-Specific Statistics (NEW in v0.2.0)
# ============================================================================

def calculate_snr(
    fitsfile: str,
    signal_region: Union[str, Tuple[int, int, int, int]] = 'data',
    noise_region: Optional[Union[str, Tuple[int, int, int, int]]] = None,
    ext: int = 0,
    verbose: bool = False
) -> float:
    """
    Calculate Signal-to-Noise Ratio.
    
    SNR = mean(signal) / std(noise)
    
    Parameters
    ----------
    fitsfile : str
        Path to the FITS file.
    signal_region : str or tuple, optional
        Region to measure signal. Defaults to 'data'.
    noise_region : str or tuple, optional
        Region to measure noise. If None, uses signal_region. Defaults to None.
    ext : int, optional
        HDU extension. Defaults to 0.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    float
        Signal-to-noise ratio.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'test.fits')
    ...     create_flat_frame(fpath, shape=(100, 100), mean_level=10000, seed=42)
    ...     snr = calculate_snr(fpath, signal_region='full')
    ...     print(snr > 50)  # High SNR for flat field
    True
    """
    try:
        # Get signal statistics
        signal_stats = stats(fitsfile, ext=ext, region=signal_region, 
                           stats_type='all', verbose=verbose)
        
        if noise_region is None:
            # Use std from signal region
            noise_std = signal_stats.std_val
        else:
            # Get noise from separate region
            noise_stats = stats(fitsfile, ext=ext, region=noise_region, 
                              stats_type='std', verbose=verbose)
            noise_std = noise_stats
        
        if noise_std == 0:
            logger.warning("Noise std is zero, SNR is infinite")
            return np.inf
        
        snr = signal_stats.mean_val / noise_std
        
        if verbose:
            logger.info(f"SNR: {snr:.2f} (signal mean: {signal_stats.mean_val:.2f}, noise std: {noise_std:.2f})")
        
        return float(snr)
        
    except Exception as e:
        if verbose:
            logger.error(f"Error calculating SNR for {fitsfile}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to calculate SNR: {e}")


def calculate_dynamic_range(
    fitsfile: str,
    region: Union[str, Tuple[int, int, int, int]] = 'data',
    ext: int = 0,
    verbose: bool = False
) -> float:
    """
    Calculate dynamic range (max / min ratio).
    
    Parameters
    ----------
    fitsfile : str
        Path to the FITS file.
    region : str or tuple, optional
        Region to analyze. Defaults to 'data'.
    ext : int, optional
        HDU extension. Defaults to 0.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    float
        Dynamic range.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'test.fits')
    ...     create_flat_frame(fpath, shape=(100, 100), mean_level=10000, seed=42)
    ...     dr = calculate_dynamic_range(fpath, region='full')
    ...     print(dr > 1)
    True
    """
    try:
        result = stats(fitsfile, ext=ext, region=region, stats_type='all', verbose=verbose)
        
        if result.min_val == 0:
            logger.warning("Minimum value is zero, dynamic range is infinite")
            return np.inf
        
        dynamic_range = result.max_val / result.min_val
        
        if verbose:
            logger.info(f"Dynamic range: {dynamic_range:.2f} (max: {result.max_val:.2f}, min: {result.min_val:.2f})")
        
        return float(dynamic_range)
        
    except Exception as e:
        if verbose:
            logger.error(f"Error calculating dynamic range for {fitsfile}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to calculate dynamic range: {e}")


def estimate_read_noise(
    bias_files: List[str],
    region: Union[str, Tuple[int, int, int, int]] = 'data',
    ext: int = 0,
    verbose: bool = False
) -> float:
    """
    Estimate read noise from a set of bias frames.
    
    Read noise is estimated as the average standard deviation across
    multiple bias frames.
    
    Parameters
    ----------
    bias_files : list of str
        List of paths to bias frames.
    region : str or tuple, optional
        Region to analyze. Defaults to 'data'.
    ext : int, optional
        HDU extension. Defaults to 0.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    float
        Estimated read noise in ADU.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     bias_list = []
    ...     for i in range(5):
    ...         fpath = os.path.join(tmpdir, f'bias_{i}.fits')
    ...         create_bias_frame(fpath, shape=(50, 50), bias_level=100, 
    ...                          read_noise=5, seed=i)
    ...         bias_list.append(fpath)
    ...     rn = estimate_read_noise(bias_list, region='full')
    ...     print(4 < rn < 6)
    True
    """
    if not bias_files:
        raise ValueError("No bias files provided")
    
    try:
        std_values = []
        
        for bias_file in bias_files:
            std_val = stats(bias_file, ext=ext, region=region, 
                          stats_type='std', verbose=verbose)
            if std_val is not None and not np.isnan(std_val):
                std_values.append(std_val)
        
        # ✅ FIXED: Indentation error corrected
        if not std_values:
            raise ImageProcessingError("No valid standard deviation values obtained from bias frames")
        
        read_noise = float(np.mean(std_values))
        
        if verbose:
            logger.info(f"Estimated read noise: {read_noise:.4f} ADU (from {len(std_values)} frames)")
        
        return read_noise
        
    except Exception as e:
        if verbose:
            logger.error(f"Error estimating read noise: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to estimate read noise: {e}")


# ============================================================================
# Multi-File Statistics
# ============================================================================

def calculate_statistics_across_files(
    input_param: Union[str, list],
    keys: Optional[Dict[str, Any]] = None,
    region: Optional[Union[str, Tuple[int, int, int, int]]] = None,
    recursive: bool = False,
    verbose: bool = False,
    single_statistic: Optional[str] = None
) -> Union[Dict[str, float], float, None]:
    """
    Calculate the average of statistics across multiple FITS files.
    
    Note: This computes the *average of each statistic* across files,
    NOT the statistics of combined data.
    
    For example, if File 1 has mean=100 and File 2 has mean=200,
    this returns mean=150 (the average of the means).
    
    Parameters
    ----------
    input_param : str or list
        List of file paths, directory path, or file search pattern.
    keys : dict, optional
        Dictionary of FITS header keywords to filter files.
    region : str or tuple, optional
        Region to analyze. Defaults to 'full'.
    recursive : bool, optional
        If True, search subdirectories. Defaults to False.
    verbose : bool, optional
        Enable verbose logging. Defaults to False.
    single_statistic : str, optional
        If provided, return only the average of this statistic.
        Options: 'mean', 'median', 'min', 'max', 'std', 'variance'.

    Returns
    -------
    dict or float or None
        Dictionary of average statistics if single_statistic is None.
        Float if single_statistic is specified.
        None if no files are processed.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     file_list = []
    ...     for i in range(3):
    ...         fpath = os.path.join(tmpdir, f'test_{i}.fits')
    ...         create_bias_frame(fpath, shape=(10, 10), bias_level=100, seed=i)
    ...         file_list.append(fpath)
    ...     result = calculate_statistics_across_files(file_list)
    ...     print(95 < result['mean'] < 105)
    ...     mean_only = calculate_statistics_across_files(file_list, single_statistic='mean')
    ...     print(95 < mean_only < 105)
    True
    True
    """
    allowed_stats = ['min', 'max', 'mean', 'median', 'std', 'variance']
    if single_statistic and single_statistic not in allowed_stats:
        raise ValueError(
            f"Invalid single_statistic: '{single_statistic}'. "
            f"Allowed values are: {allowed_stats}"
        )

    try:
        fits_files = select_files(input_param, keys=keys, recursive=recursive, verbose=verbose)
    except SelectionError as e:
        logger.error(f"Error selecting files for statistics calculation: {e}")
        raise
    
    if not fits_files:
        if verbose:
            logger.info("calculate_statistics_across_files: No FITS files found. Returning None.")
        return None

    stats_lists = {
        'min': [], 'max': [], 'mean': [], 'median': [], 'std': [], 'variance': []
    }

    for file in fits_files:
        try:
            file_stats_result = stats(file, region=region or 'full', stats_type='all', verbose=verbose)
            
            if file_stats_result and isinstance(file_stats_result, StatsResult):
                if not np.isnan(file_stats_result.min_val): 
                    stats_lists['min'].append(file_stats_result.min_val)
                if not np.isnan(file_stats_result.max_val): 
                    stats_lists['max'].append(file_stats_result.max_val)
                if not np.isnan(file_stats_result.mean_val): 
                    stats_lists['mean'].append(file_stats_result.mean_val)
                if not np.isnan(file_stats_result.median_val): 
                    stats_lists['median'].append(file_stats_result.median_val)
                if not np.isnan(file_stats_result.std_val): 
                    stats_lists['std'].append(file_stats_result.std_val)
                if not np.isnan(file_stats_result.variance_val): 
                    stats_lists['variance'].append(file_stats_result.variance_val)
                
                if verbose:
                    logger.info(f"  Processed {os.path.basename(file)}")
            else:
                if verbose:
                    logger.warning(f"  No valid statistics for {os.path.basename(file)}. Skipping.")
        except FitsToolsError as e:
            if verbose:
                logger.warning(f"  Could not process {os.path.basename(file)}: {e}. Skipping.")
        except Exception as e:
            if verbose:
                logger.warning(
                    f"  Unexpected error processing {os.path.basename(file)}: {e}. Skipping.",
                    exc_info=True
                )
    
    if not stats_lists['mean']:
        if verbose:
            logger.info("calculate_statistics_across_files: No valid statistics collected. Returning None.")
        return None

    average_results = {key: np.mean(values) for key, values in stats_lists.items() if values}

    if verbose:
        logger.info(f"Calculated average statistics from {len(stats_lists['mean'])} files:")
        for key, val in average_results.items():
            logger.info(f"  Average {key.capitalize()}: {val:.4f}")

    if single_statistic:
        return average_results.get(single_statistic)
    else:
        return average_results


# Backward compatibility wrapper
def calculate_average_stats(*args, **kwargs):
    """
    Backward compatibility wrapper for calculate_statistics_across_files.
    
    Deprecated: Use calculate_statistics_across_files() instead.
    """
    return calculate_statistics_across_files(*args, **kwargs)


# Convenience functions for multi-file statistics
def get_average_mean(
    input_param: Union[str, list], 
    keys: Optional[Dict[str, Any]] = None, 
    region: Optional[Union[str, Tuple[int, int, int, int]]] = None, 
    recursive: bool = False, 
    verbose: bool = False
) -> Optional[float]:
    """Computes the average mean value across multiple files."""
    return calculate_statistics_across_files(
        input_param, keys=keys, region=region, recursive=recursive, 
        verbose=verbose, single_statistic='mean'
    )


def get_average_median(
    input_param: Union[str, list], 
    keys: Optional[Dict[str, Any]] = None, 
    region: Optional[Union[str, Tuple[int, int, int, int]]] = None, 
    recursive: bool = False, 
    verbose: bool = False
) -> Optional[float]:
    """Computes the average median value across multiple files."""
    return calculate_statistics_across_files(
        input_param, keys=keys, region=region, recursive=recursive, 
        verbose=verbose, single_statistic='median'
    )


def get_average_minimum(
    input_param: Union[str, list], 
    keys: Optional[Dict[str, Any]] = None, 
    region: Optional[Union[str, Tuple[int, int, int, int]]] = None, 
    recursive: bool = False, 
    verbose: bool = False
) -> Optional[float]:
    """Computes the average minimum value across multiple files."""
    return calculate_statistics_across_files(
        input_param, keys=keys, region=region, recursive=recursive, 
        verbose=verbose, single_statistic='min'
    )


def get_average_maximum(
    input_param: Union[str, list], 
    keys: Optional[Dict[str, Any]] = None, 
    region: Optional[Union[str, Tuple[int, int, int, int]]] = None, 
    recursive: bool = False, 
    verbose: bool = False
) -> Optional[float]:
    """Computes the average maximum value across multiple files."""
    return calculate_statistics_across_files(
        input_param, keys=keys, region=region, recursive=recursive, 
        verbose=verbose, single_statistic='max'
    )


def get_average_std(
    input_param: Union[str, list], 
    keys: Optional[Dict[str, Any]] = None, 
    region: Optional[Union[str, Tuple[int, int, int, int]]] = None, 
    recursive: bool = False, 
    verbose: bool = False
) -> Optional[float]:
    """Computes the average standard deviation across multiple files."""
    return calculate_statistics_across_files(
        input_param, keys=keys, region=region, recursive=recursive, 
        verbose=verbose, single_statistic='std'
    )


# Backward compatibility alias
def get_average_stddev(*args, **kwargs):
    """Deprecated alias for get_average_std()."""
    return get_average_std(*args, **kwargs)


# ============================================================================
# Comparison and Analysis (NEW in v0.2.0)
# ============================================================================

@dataclass
class ComparisonResult:
    """
    Results from comparing statistics between two files.
    
    Attributes:
        file1 (str): Path to first file
        file2 (str): Path to second file
        stats1 (StatsResult): Statistics from first file
        stats2 (StatsResult): Statistics from second file
        mean_diff (float): Difference in means (file1 - file2)
        std_diff (float): Difference in standard deviations
        relative_mean_diff (float): Relative difference in means (diff/mean2)
    """
    file1: str
    file2: str
    stats1: StatsResult
    stats2: StatsResult
    mean_diff: float = 0.0
    std_diff: float = 0.0
    relative_mean_diff: float = 0.0
    
    def __post_init__(self):
        """Calculate derived quantities."""
        self.mean_diff = self.stats1.mean_val - self.stats2.mean_val
        self.std_diff = self.stats1.std_val - self.stats2.std_val
        if self.stats2.mean_val != 0:
            self.relative_mean_diff = self.mean_diff / self.stats2.mean_val
        else:
            self.relative_mean_diff = np.nan
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"Comparison Results:\n"
            f"  File 1: {os.path.basename(self.file1)}\n"
            f"    Mean: {self.stats1.mean_val:.4f}, Std: {self.stats1.std_val:.4f}\n"
            f"  File 2: {os.path.basename(self.file2)}\n"
            f"    Mean: {self.stats2.mean_val:.4f}, Std: {self.stats2.std_val:.4f}\n"
            f"  Differences:\n"
            f"    Mean: {self.mean_diff:.4f} ({self.relative_mean_diff*100:.2f}%)\n"
            f"    Std:  {self.std_diff:.4f}"
        )


def compare_stats(
    file1: str,
    file2: str,
    region: Union[str, Tuple[int, int, int, int]] = 'full',
    ext: int = 0,
    verbose: bool = False
) -> ComparisonResult:
    """
    Compare statistics between two FITS files.
    
    Parameters
    ----------
    file1 : str
        Path to first FITS file.
    file2 : str
        Path to second FITS file.
    region : str or tuple, optional
        Region to analyze. Defaults to 'full'.
    ext : int, optional
        HDU extension. Defaults to 0.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    ComparisonResult
        Object containing comparison results.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     f1 = os.path.join(tmpdir, 'bias1.fits')
    ...     f2 = os.path.join(tmpdir, 'bias2.fits')
    ...     create_bias_frame(f1, shape=(10, 10), bias_level=100, seed=42)
    ...     create_bias_frame(f2, shape=(10, 10), bias_level=110, seed=43)
    ...     comp = compare_stats(f1, f2)
    ...     print(comp.mean_diff < 0)  # file1 mean < file2 mean
    True
    """
    try:
        stats1 = stats(file1, ext=ext, region=region, stats_type='all', verbose=verbose)
        stats2 = stats(file2, ext=ext, region=region, stats_type='all', verbose=verbose)
        
        result = ComparisonResult(
            file1=file1,
            file2=file2,
            stats1=stats1,
            stats2=stats2
        )
        
        if verbose:
            logger.info(str(result))
        
        return result
        
    except Exception as e:
        if verbose:
            logger.error(f"Error comparing {file1} and {file2}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to compare statistics: {e}")


def analyze_statistics_trend(
    file_list: List[str],
    region: Union[str, Tuple[int, int, int, int]] = 'full',
    ext: int = 0,
    statistic: str = 'mean',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Analyze trend in a statistic across a series of files.
    
    Useful for monitoring detector performance over time.
    
    Parameters
    ----------
    file_list : list of str
        List of FITS file paths in chronological order.
    region : str or tuple, optional
        Region to analyze. Defaults to 'full'.
    ext : int, optional
        HDU extension. Defaults to 0.
    statistic : str, optional
        Which statistic to track. Defaults to 'mean'.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'values': list of statistic values
        - 'trend': linear trend (slope)
        - 'mean': average value
        - 'std': standard deviation of values
        - 'range': range (max - min)
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     files = []
    ...     for i in range(5):
    ...         fpath = os.path.join(tmpdir, f'frame_{i}.fits')
    ...         create_bias_frame(fpath, shape=(10, 10), bias_level=100+i, seed=i)
    ...         files.append(fpath)
    ...     trend = analyze_statistics_trend(files, statistic='mean')
    ...     print(len(trend['values']))
    ...     print(trend['trend'] > 0)  # Increasing bias level
    5
    True
    """
    if not file_list:
        raise ValueError("Empty file list provided")
    
    try:
        values = []
        
        for file in file_list:
            stat_val = stats(file, ext=ext, region=region, 
                           stats_type=statistic, verbose=verbose)
            if stat_val is not None and not np.isnan(stat_val):
                values.append(stat_val)
            else:
                if verbose:
                    logger.warning(f"Invalid statistic for {os.path.basename(file)}, skipping")
        
        if not values:
            raise ImageProcessingError("No valid statistics obtained from any file")
        
        values_array = np.array(values)
        
        # Calculate linear trend
        x = np.arange(len(values))
        if len(values) > 1:
            coeffs = np.polyfit(x, values_array, 1)
            trend = float(coeffs[0])  # Slope
        else:
            trend = 0.0
        
        result = {
            'values': values,
            'trend': trend,
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'range': float(np.max(values_array) - np.min(values_array)),
            'num_files': len(values)
        }
        
        if verbose:
            logger.info(f"Trend analysis for {statistic}:")
            logger.info(f"  Number of files: {result['num_files']}")
            logger.info(f"  Mean: {result['mean']:.4f}")
            logger.info(f"  Std: {result['std']:.4f}")
            logger.info(f"  Trend (slope): {result['trend']:.6f}")
        
        return result
        
    except Exception as e:
        if verbose:
            logger.error(f"Error analyzing trend: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to analyze statistics trend: {e}")


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Dataclasses
    'StatsResult',
    'RobustStatsResult',
    'ComparisonResult',
    
    # Core functions
    'stats',
    'stats_to_dict',
    
    # Single-file convenience functions
    'get_image_min',
    'get_image_max',
    'get_image_mean',
    'get_image_median',
    'get_image_std',
    
    # Robust statistics
    'robust_stats',
    
    # CCD-specific
    'calculate_snr',
    'calculate_dynamic_range',
    'estimate_read_noise',
    
    # Multi-file statistics
    'calculate_statistics_across_files',
    'calculate_average_stats',  # Backward compatibility
    'get_average_mean',
    'get_average_median',
    'get_average_minimum',
    'get_average_maximum',
    'get_average_std',
    'get_average_stddev',  # Backward compatibility
    
    # Comparison and analysis
    'compare_stats',
    'analyze_statistics_trend',
]


# ============================================================================
# Main / Testing
# ============================================================================

def main():
    """
    Command-line interface and test suite for statistics.
    """
    import argparse
    import doctest
    import shutil
    
    parser = argparse.ArgumentParser(
        description='FITS Statistics Tools - Statistical Analysis of CCD Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run doctests
  %(prog)s --test
  
  # Get statistics from a file
  %(prog)s --file test.fits --stat mean
  
  # Compare two files
  %(prog)s --compare file1.fits file2.fits
        """
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Run doctests')
    parser.add_argument('--file', metavar='FITS_FILE',
                       help='Calculate statistics for a file')
    parser.add_argument('--stat', default='all',
                       help='Statistic to calculate (default: all)')
    parser.add_argument('--region', default='full',
                       help='Region to analyze (default: full)')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare statistics between two files')
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
    
    # Run doctests
    if args.test or len(sys.argv) == 1:
        logger.info("Running doctests for statistics.py...")
        result = doctest.testmod(verbose=args.verbose)
        
        if result.failed == 0:
            logger.info(f"All {result.attempted} doctests passed!")
            return 0
        else:
            logger.error(f"{result.failed} of {result.attempted} doctests failed!")
            return 1
    
    # Calculate statistics for a file
    if args.file:
        try:
            result = stats(args.file, region=args.region, stats_type=args.stat, verbose=args.verbose)
            if args.stat == 'all':
                print(result)
            else:
                print(f"{args.stat}: {result:.4f}")
            return 0
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    
    # Compare two files
    if args.compare:
        try:
            result = compare_stats(args.compare[0], args.compare[1], 
                                  region=args.region, verbose=args.verbose)
            print(result)
            return 0
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

    
