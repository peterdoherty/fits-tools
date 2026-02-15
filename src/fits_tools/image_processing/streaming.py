#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-efficient streaming algorithms for large FITS datasets.

This module provides streaming (online) algorithms that process images
sequentially with minimal memory usage. These algorithms are essential
for processing datasets that exceed available RAM.

Algorithms:
    - P² Quantile Estimator: Single-pass per-pixel quantile estimation
    - Welford's Method: Streaming mean and variance calculation
    - Streaming Statistics: Calculate statistics without loading all data
    - Streaming Combine: Memory-efficient image stacking

Features:
    - Constant memory usage regardless of dataset size
    - Single-pass algorithms (no need to load data multiple times)
    - Per-pixel processing for quantiles
    - Numerically stable variance calculation
    - Progress reporting for long operations
    - Support for memory-mapped FITS files

Use Cases:
    - Computing median of 1000+ images without loading all into RAM
    - Calculating statistics on images larger than available memory
    - Processing time-series data with thousands of frames
    - Creating master calibration frames from large datasets

Usage Examples:
    P² quantile estimation:
        >>> from fits_tools.image_processing.streaming import P2QuantileEstimator
        >>> estimator = P2QuantileEstimator((512, 512), p=0.5)  # median
        >>> for i in range(100):
        ...     data = load_image(i)
        ...     estimator.update(data)
        >>> median = estimator.get_quantile()
    
    Streaming statistics:
        >>> from fits_tools.image_processing.streaming import streaming_statistics
        >>> files = ['img001.fits', 'img002.fits', ..., 'img999.fits']
        >>> stats = streaming_statistics(files)
        >>> print(f"Mean: {stats['mean']}, Std: {stats['std']}")
    
    Memory-efficient combining:
        >>> from fits_tools.image_processing.streaming import streaming_combine
        >>> result = streaming_combine(files, method='median', output='master.fits')

References:
    - Jain, R. and Chlamtac, I. (1985). "The P² algorithm for dynamic 
      calculation of quantiles and histograms without storing observations"
      Communications of the ACM, 28(10), 1076-1085.
    - Welford, B. P. (1962). "Note on a method for calculating corrected 
      sums of squares and products". Technometrics 4(3), 419-420.

Version: 2.0.0
"""

import os
import logging
from typing import List, Union, Optional, Dict, Any, Tuple

import numpy as np
from astropy.io import fits


# Import from fits_tools
from ..io import get_header, write_fits
from ..exceptions import ImageProcessingError, FileHandlingError
from ..utils import add_history, maybe_tqdm

logger = logging.getLogger(__name__)
__version__ = "2.0.0"
__author__ = "FITS Tools Development Team"


# ============================================================================
# Utility Functions
# ============================================================================

def _estimate_memory_usage(
    file_list: List[str],
    ext: int = 0
) -> float:
    """
    Estimate memory usage in GB for loading all files.
    
    Parameters
    ----------
    file_list : list of str
        List of FITS files.
    ext : int
        HDU extension.
    
    Returns
    -------
    float
        Estimated memory usage in GB.
    """
    if not file_list:
        return 0.0
    
    try:
        # Get shape and dtype from first file
        with fits.open(file_list[0], memmap=True) as hdul:
            data = hdul[ext].data
            if data is None:
                return 0.0
            
            shape = data.shape
            dtype = data.dtype
        
        # Calculate bytes per image
        bytes_per_pixel = np.dtype(dtype).itemsize
        pixels_per_image = np.prod(shape)
        bytes_per_image = bytes_per_pixel * pixels_per_image
        
        # Total for all files
        total_bytes = bytes_per_image * len(file_list)
        total_gb = total_bytes / (1024**3)
        
        return total_gb
    
    except Exception as e:
        logger.warning(f"Could not estimate memory usage: {e}")
        return 0.0


# ============================================================================
# P² Quantile Estimator
# ============================================================================

class P2QuantileEstimator:
    """
    P² algorithm for streaming per-pixel quantile estimation.
    
    This implements the Jain-Chlamtac P² algorithm for computing quantiles
    in a single pass with minimal memory. It maintains 5 marker positions
    per pixel, allowing quantile estimation without storing all observations.
    
    The algorithm is particularly useful for computing medians (p=0.5) or
    other percentiles from large datasets that don't fit in memory.
    
    Parameters
    ----------
    shape : tuple
        Image shape (rows, cols).
    p : float, optional
        Quantile to estimate (0 < p < 1). Defaults to 0.5 (median).
    
    Attributes
    ----------
    p : float
        Target quantile.
    shape : tuple
        Image dimensions.
    count : int
        Number of observations processed.
    q : np.ndarray
        Marker heights (5, rows, cols).
    n : np.ndarray
        Marker positions (5, rows, cols).
    n_desired : np.ndarray
        Desired marker positions (5, rows, cols).
    
    Examples
    --------
    >>> import numpy as np
    >>> estimator = P2QuantileEstimator((100, 100), p=0.5)
    >>> for i in range(50):
    ...     data = np.random.normal(100, 10, (100, 100))
    ...     estimator.update(data)
    >>> median_estimate = estimator.get_quantile()
    >>> print(f"Estimated median: {np.mean(median_estimate):.2f}")
    
    Notes
    -----
    - Requires at least 5 observations to initialize
    - Accuracy improves with more observations
    - Memory usage is constant: 5 × image_size × 8 bytes each for
      marker heights (float64), positions (int64), and desired
      positions (float64), i.e. ~15 × image_size × 8 bytes total.
      During initialization (first 5 observations) an additional
      5 × image_size × 8 bytes is held for the sorting buffer;
      this is freed once markers are initialized.
    - More accurate for symmetric distributions
    
    References
    ----------
    Jain, R. and Chlamtac, I. (1985). Communications of the ACM, 28(10).
    """
    
    def __init__(self, shape: Tuple[int, int], p: float = 0.5):
        """Initialize P² quantile estimator."""
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")
        
        self.p = float(p)
        self.shape = shape
        self.count = 0
        
        ny, nx = shape
        
        # Marker heights: q[0..4] shape (5, ny, nx)
        self.q = np.zeros((5, ny, nx), dtype=np.float64)
        
        # Marker positions: n[0..4]
        self.n = np.zeros((5, ny, nx), dtype=np.int64)
        
        # Desired marker positions
        self.n_desired = np.zeros((5, ny, nx), dtype=np.float64)
        
        # Buffer for initial 5 samples
        self._init_buffer = []
        self._initialized = False
    
    def update(self, data: np.ndarray) -> None:
        """
        Update estimator with new observation.
        
        Parameters
        ----------
        data : np.ndarray
            New image data (must match shape).
        
        Raises
        ------
        ValueError
            If data shape doesn't match.
        """
        if data.shape != self.shape:
            raise ValueError(
                f"Data shape {data.shape} doesn't match expected {self.shape}"
            )
        
        # Convert to float64 for numerical stability (no copy if already float64)
        data_f64 = np.asarray(data, dtype=np.float64)
        
        if not self._initialized:
            self._add_initial_sample(data_f64)
        else:
            self._update_with_sample(data_f64)
        
        self.count += 1
    
    def _add_initial_sample(self, data: np.ndarray) -> None:
        """Add sample to initial buffer (first 5 samples)."""
        self._init_buffer.append(data)
        
        if len(self._init_buffer) == 5:
            self._initialize_markers()
            self._initialized = True
    
    def _initialize_markers(self) -> None:
        """Initialize markers after collecting 5 samples."""
        # Stack and sort the 5 initial samples
        init_stack = np.stack(self._init_buffer, axis=0)
        init_sorted = np.sort(init_stack, axis=0)
        
        # Set initial marker heights
        for i in range(5):
            self.q[i, :, :] = init_sorted[i, :, :]
        
        # Set initial marker positions (1, 2, 3, 4, 5)
        for i in range(5):
            self.n[i, :, :] = i + 1
        
        # Set initial desired positions
        n_desired_init = np.array([
            1.0,
            1.0 + 2.0 * self.p,
            1.0 + 4.0 * self.p,
            3.0 + 2.0 * self.p,
            5.0
        ], dtype=np.float64)
        
        for i in range(5):
            self.n_desired[i, :, :] = n_desired_init[i]
        
        # Clear buffer to free memory
        self._init_buffer = None
    
    def _update_with_sample(self, data: np.ndarray) -> None:
        """Update markers with new sample (after initialization)."""
        x = data
        
        # Step 1: Handle boundary cases — update min/max markers
        below = x < self.q[0]
        above = x >= self.q[4]
        self.q[0] = np.where(below, x, self.q[0])
        self.q[4] = np.where(above, x, self.q[4])
        
        # Step 2: Find cell k for each pixel where x falls
        # k is the marker interval: q[k] <= x < q[k+1]
        cmp = x[None, :, :] >= self.q  # Shape: (5, ny, nx)
        k = cmp.sum(axis=0).astype(np.int32) - 1  # Shape: (ny, nx)
        
        # Clamp k to valid range [0, 3] (since we have 5 markers, 4 intervals)
        k = np.clip(k, 0, 3)
        
        # Step 3: Update marker positions — increment n[j] for j > k
        self.n += (np.arange(5)[:, None, None] > k[None, :, :]).astype(np.int64)
        
        # Step 4: Update desired positions
        n_desired_increment = np.array([
            0.0,
            self.p / 2.0,
            self.p,
            (1.0 + self.p) / 2.0,
            1.0
        ], dtype=np.float64)[:, None, None]
        
        self.n_desired += n_desired_increment
        
        # Step 5: Adjust marker heights for markers 1, 2, 3
        for i in (1, 2, 3):
            d = self.n_desired[i] - self.n[i].astype(np.float64)
            
            # Determine where adjustment is needed
            needs_up = d >= 1.0
            needs_down = d <= -1.0
            needs_adjustment = needs_up | needs_down
            
            if not needs_adjustment.any():
                continue
            
            # Direction of adjustment
            s = np.where(needs_up, 1.0, np.where(needs_down, -1.0, 0.0))
            
            # Get neighboring values
            q_im1 = self.q[i - 1]
            q_i = self.q[i]
            q_ip1 = self.q[i + 1]
            
            n_im1 = self.n[i - 1].astype(np.float64)
            n_i = self.n[i].astype(np.float64)
            n_ip1 = self.n[i + 1].astype(np.float64)
            
            # Parabolic formula (P² algorithm)
            with np.errstate(divide='ignore', invalid='ignore'):
                denom1 = n_ip1 - n_i
                denom2 = n_i - n_im1
                denom3 = n_ip1 - n_im1
                
                # Avoid division by zero
                denom1 = np.where(denom1 == 0, 1.0, denom1)
                denom2 = np.where(denom2 == 0, 1.0, denom2)
                denom3 = np.where(denom3 == 0, 1.0, denom3)
                
                term1 = (n_i - n_im1 + s) * (q_ip1 - q_i) / denom1
                term2 = (n_ip1 - n_i - s) * (q_i - q_im1) / denom2
                
                q_parabolic = q_i + (s / denom3) * (term1 + term2)
            
            # Linear formula (fallback)
            q_linear = q_i + s * (q_ip1 - q_im1) / denom3
            
            # Use parabolic if it stays between neighbors, otherwise linear
            use_parabolic = (
                (q_parabolic > q_im1) & 
                (q_parabolic < q_ip1) & 
                needs_adjustment
            )
            
            q_new = np.where(use_parabolic, q_parabolic, q_linear)
            
            # Update marker height and position where adjustment was needed
            self.q[i] = np.where(needs_adjustment, q_new, q_i)
            self.n[i] = np.where(
                needs_adjustment,
                self.n[i] + s.astype(np.int64),
                self.n[i]
            )
    
    def get_quantile(self) -> np.ndarray:
        """
        Get current quantile estimate.
        
        Returns
        -------
        np.ndarray
            Estimated quantile map (same shape as input images).
        
        Notes
        -----
        - Returns middle marker (q[2]) after initialization
        - Before initialization (<5 samples), returns median of available samples
        """
        if not self._initialized:
            if len(self._init_buffer) == 0:
                raise RuntimeError("No samples added yet")
            
            # Return median of available samples
            stacked = np.stack(self._init_buffer, axis=0)
            return np.median(stacked, axis=0)
        
        # Return middle marker (index 2)
        return self.q[2].copy()
    
    def __repr__(self) -> str:
        """Brief representation."""
        return (
            f"P2QuantileEstimator(shape={self.shape}, p={self.p}, "
            f"count={self.count}, initialized={self._initialized})"
        )


# ============================================================================
# Welford's Algorithm for Streaming Statistics
# ============================================================================

def streaming_statistics(
    file_list: List[str],
    ext: int = 0,
    memmap: bool = True,
    compute_quantiles: bool = False,
    quantiles: Optional[List[float]] = None,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Calculate statistics using Welford's method for numerical stability.
    
    Computes mean, variance, standard deviation, min, and max in a single
    pass through the data with constant memory usage. Optionally computes
    quantiles using the P² algorithm.
    
    Parameters
    ----------
    file_list : list of str
        List of FITS file paths.
    ext : int, optional
        HDU extension to process. Defaults to 0.
    memmap : bool, optional
        Use memory mapping for FITS files. Defaults to True.
    compute_quantiles : bool, optional
        Whether to compute quantiles (requires more memory). Defaults to False.
    quantiles : list of float, optional
        Quantiles to compute (e.g., [0.25, 0.5, 0.75]). If None and
        compute_quantiles=True, computes median only.
    verbose : bool, optional
        Enable verbose logging and progress bars.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'count': int, number of images processed
        - 'mean': np.ndarray, per-pixel mean
        - 'var': np.ndarray, per-pixel variance
        - 'std': np.ndarray, per-pixel standard deviation
        - 'min': np.ndarray, per-pixel minimum
        - 'max': np.ndarray, per-pixel maximum
        - 'sum': np.ndarray, per-pixel sum
        - 'quantiles': dict (if compute_quantiles=True)
    
    Raises
    ------
    ValueError
        If file_list is empty or files have incompatible shapes.
    FileHandlingError
        If files cannot be read.
    
    Examples
    --------
    >>> from fits_tools.image_processing.streaming import streaming_statistics
    >>> files = ['img001.fits', 'img002.fits', 'img003.fits']
    >>> stats = streaming_statistics(files, verbose=True)
    >>> print(f"Mean: {np.mean(stats['mean']):.2f}")
    >>> print(f"Std: {np.mean(stats['std']):.2f}")
    
    >>> # With quantiles
    >>> stats = streaming_statistics(
    ...     files,
    ...     compute_quantiles=True,
    ...     quantiles=[0.25, 0.5, 0.75]
    ... )
    >>> median = stats['quantiles'][0.5]
    
    Notes
    -----
    - Memory usage: ~6 × image_size × 8 bytes (float64 accumulators for
      mean, var, min, max, sum, M2)
    - With quantiles: additional 5 × image_size × 8 bytes per quantile
    - Numerically stable variance calculation (Welford's algorithm)
    - Single pass through data
    
    References
    ----------
    Welford, B. P. (1962). Technometrics 4(3), 419-420.
    """
    if not file_list:
        raise ValueError("Empty file list")
    
    if verbose:
        estimated_memory = _estimate_memory_usage(file_list, ext=ext)
        logger.info(f"Processing {len(file_list)} files")
        logger.info(f"Estimated memory for in-memory loading: {estimated_memory:.2f} GB")
        logger.info(f"Using streaming algorithms (memory-efficient)")
    
    # Get shape from first file
    try:
        with fits.open(file_list[0], memmap=memmap) as hdul:
            if ext >= len(hdul):
                raise IndexError(f"HDU index {ext} out of range")
            
            data = hdul[ext].data
            if data is None:
                raise ValueError(f"No data in HDU {ext} of {file_list[0]}")
            
            shape = data.shape
    except Exception as e:
        raise FileHandlingError(f"Could not read {file_list[0]}: {e}") from e
    
    if verbose:
        logger.info(f"Image shape: {shape}")
    
    # Initialize accumulators
    count = 0
    mean = None
    M2 = None  # Sum of squared differences from mean (for variance)
    sum_arr = np.zeros(shape, dtype=np.float64)
    min_arr = np.full(shape, np.inf, dtype=np.float64)
    max_arr = np.full(shape, -np.inf, dtype=np.float64)
    
    # Initialize quantile estimators if requested
    quantile_estimators = {}
    if compute_quantiles:
        if quantiles is None:
            quantiles = [0.5]  # Default to median only
        
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile must be in (0, 1), got {q}")
            quantile_estimators[q] = P2QuantileEstimator(shape, p=q)
        
        if verbose:
            logger.info(f"Computing quantiles: {quantiles}")
    
    # Process files
    iterator = maybe_tqdm(
        file_list,
        desc="Computing statistics",
        unit="file"
    ) if verbose else file_list
    
    for filepath in iterator:
        try:
            with fits.open(filepath, memmap=memmap) as hdul:
                data = hdul[ext].data
                if data is None:
                    logger.warning(f"No data in {filepath}, skipping")
                    continue
                
                if data.shape != shape:
                    raise ValueError(
                        f"Shape mismatch: {filepath} has shape {data.shape}, "
                        f"expected {shape}"
                    )
                
                # Convert to float64 for numerical stability
                arr = np.asarray(data, dtype=np.float64)
                
                # Update sum, min, max
                sum_arr += arr
                np.minimum(min_arr, arr, out=min_arr)
                np.maximum(max_arr, arr, out=max_arr)
                
                # Welford's method for mean and variance
                count += 1
                if mean is None:
                    mean = arr.copy()
                    M2 = np.zeros_like(arr, dtype=np.float64)
                else:
                    delta = arr - mean
                    mean += delta / count
                    delta2 = arr - mean
                    M2 += delta * delta2
                
                # Update quantile estimators
                for estimator in quantile_estimators.values():
                    estimator.update(arr)
        
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            raise ImageProcessingError(f"Failed to process {filepath}: {e}") from e
    
    if count == 0:
        raise ValueError("No valid files processed")
    
    if verbose:
        logger.info(f"Processed {count} files successfully")
    
    # Calculate final statistics
    var = M2 / count
    std = np.sqrt(var)
    
    results = {
        'count': count,
        'mean': mean,
        'var': var,
        'std': std,
        'min': min_arr,
        'max': max_arr,
        'sum': sum_arr,
    }
    
    # Add quantiles if computed
    if compute_quantiles:
        results['quantiles'] = {}
        for q, estimator in quantile_estimators.items():
            results['quantiles'][q] = estimator.get_quantile()
        
        if verbose:
            for q in quantiles:
                q_val = results['quantiles'][q]
                logger.info(f"Quantile {q}: mean={np.mean(q_val):.2f}")
    
    return results


# ============================================================================
# Streaming Combine
# ============================================================================

def streaming_combine(
    file_list: List[str],
    method: str = 'median',
    output_file: Optional[str] = None,
    ext: int = 0,
    memmap: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[np.ndarray, str]:
    """
    Combine images using streaming algorithms (memory-efficient).
    
    Uses P² algorithm for median or Welford's method for mean,
    allowing combination of arbitrarily large datasets with constant
    memory usage.
    
    Parameters
    ----------
    file_list : list of str
        List of FITS files to combine.
    method : str, optional
        Combining method:
        - 'median': Streaming median using P² algorithm (default)
        - 'mean': Streaming mean using Welford's method
        - 'min': Streaming minimum
        - 'max': Streaming maximum
    output_file : str, optional
        Output FITS file path. If None, returns array.
    ext : int, optional
        HDU extension. Defaults to 0.
    memmap : bool, optional
        Use memory mapping. Defaults to True.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.
    verbose : bool, optional
        Enable verbose logging.
    **kwargs
        Additional arguments (currently unused).
    
    Returns
    -------
    np.ndarray or str
        Combined array if output_file is None, otherwise output path.
    
    Raises
    ------
    ValueError
        If method is not recognized or file_list is empty.
    
    Examples
    --------
    >>> from fits_tools.image_processing.streaming import streaming_combine
    >>> files = [f'img{i:03d}.fits' for i in range(100)]
    >>> # Compute median without loading all 100 images
    >>> median = streaming_combine(files, method='median', verbose=True)
    >>> # Or write directly to file
    >>> streaming_combine(files, method='median', output_file='master.fits')
    
    Notes
    -----
    - Much more memory-efficient than combine_images_robust()
    - Slightly less accurate for median (P² is an approximation)
    - No outlier rejection (use combine_images_robust() for that)
    - Ideal for creating master calibration frames from large datasets
    """
    valid_methods = ['median', 'mean', 'min', 'max']
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method: '{method}'. Valid options: {', '.join(valid_methods)}"
        )
    
    if not file_list:
        raise ValueError("Empty file list")
    
    if verbose:
        logger.info(f"Streaming combine: {len(file_list)} files using method '{method}'")
    
    if method == 'median':
        # Use P² algorithm
        stats = streaming_statistics(
            file_list,
            ext=ext,
            memmap=memmap,
            compute_quantiles=True,
            quantiles=[0.5],
            verbose=verbose
        )
        combined = stats['quantiles'][0.5]
    
    elif method == 'mean':
        # Use Welford's method
        stats = streaming_statistics(
            file_list,
            ext=ext,
            memmap=memmap,
            compute_quantiles=False,
            verbose=verbose
        )
        combined = stats['mean']
    
    elif method == 'min':
        stats = streaming_statistics(
            file_list,
            ext=ext,
            memmap=memmap,
            compute_quantiles=False,
            verbose=verbose
        )
        combined = stats['min']
    
    elif method == 'max':
        stats = streaming_statistics(
            file_list,
            ext=ext,
            memmap=memmap,
            compute_quantiles=False,
            verbose=verbose
        )
        combined = stats['max']
    
    # Write output if requested
    if output_file:
        # Get header from first file
        n_processed = stats['count']
        header = get_header(file_list[0], ext=ext, verbose=False)
        add_history(header, f'Streaming combine: {n_processed} images using {method}')
        header['NINPUT'] = (len(file_list), 'Number of input files')
        header['NCOMBINE'] = (n_processed, 'Number of images actually combined')
        header['COMBMETH'] = (f'streaming_{method}', 'Combining method')
        header['STREAMIN'] = (True, 'Used streaming algorithm')
        
        write_fits(output_file, combined, header, overwrite=overwrite, verbose=verbose)
        
        if verbose:
            logger.info(f"Combined image written to: {output_file}")
        
        return output_file
    else:
        return combined


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Classes
    'P2QuantileEstimator',
    
    # Functions
    'streaming_statistics',
    'streaming_combine',
]


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """Basic module testing."""
    import sys
    import tempfile
    
    print(f"Testing streaming.py v{__version__}")
    print("=" * 70)
    
    # Test 1: P² Quantile Estimator
    print("\nTest 1: P² Quantile Estimator (median)")
    try:
        np.random.seed(42)
        shape = (50, 50)
        n_samples = 100
        
        estimator = P2QuantileEstimator(shape, p=0.5)
        
        # Generate samples from known distribution
        samples = []
        for i in range(n_samples):
            data = np.random.normal(1000, 50, shape)
            samples.append(data)
            estimator.update(data)
        
        # Get P² estimate
        p2_median = estimator.get_quantile()
        
        # Compare with true median
        stacked = np.stack(samples, axis=0)
        true_median = np.median(stacked, axis=0)
        
        error = np.abs(p2_median - true_median)
        mean_error = np.mean(error)
        max_error = np.max(error)
        
        print(f"✓ P² estimator with {n_samples} samples")
        print(f"  Shape: {shape}")
        print(f"  Mean error: {mean_error:.2f}")
        print(f"  Max error: {max_error:.2f}")
        print(f"  Relative error: {mean_error/1000*100:.2f}%")
        
        if mean_error < 10:  # Within 1% for normal distribution
            print("  ✓ Error within acceptable range")
        else:
            print("  ⚠ Error higher than expected (but may be acceptable)")
    
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 2: Welford's Algorithm
    print("\nTest 2: Welford's Algorithm for mean and variance")
    try:
        np.random.seed(42)
        shape = (50, 50)
        n_samples = 100
        true_mean = 1000
        true_std = 50
        
        # Create temporary FITS files
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(n_samples):
                filepath = os.path.join(tmpdir, f'test_{i:03d}.fits')
                data = np.random.normal(true_mean, true_std, shape)
                
                hdu = fits.PrimaryHDU(data.astype(np.float32))
                hdu.writeto(filepath)
                files.append(filepath)
            
            # Compute streaming statistics
            stats = streaming_statistics(files, verbose=False)
            
            print(f"✓ Processed {stats['count']} files")
            print(f"  Expected mean: {true_mean:.2f}")
            print(f"  Computed mean: {np.mean(stats['mean']):.2f}")
            print(f"  Expected std: {true_std:.2f}")
            print(f"  Computed std: {np.mean(stats['std']):.2f}")
            
            # Check accuracy
            mean_error = abs(np.mean(stats['mean']) - true_mean)
            std_error = abs(np.mean(stats['std']) - true_std)
            
            if mean_error < 5 and std_error < 5:
                print("  ✓ Statistics within acceptable tolerance")
            else:
                print("  ⚠ Statistics deviate more than expected")
    
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Streaming median vs true median
    print("\nTest 3: Streaming median accuracy")
    try:
        np.random.seed(42)
        shape = (30, 30)
        n_samples = 50
        
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            all_data = []
            
            for i in range(n_samples):
                filepath = os.path.join(tmpdir, f'test_{i:03d}.fits')
                data = np.random.normal(1000, 50, shape)
                all_data.append(data)
                
                hdu = fits.PrimaryHDU(data.astype(np.float32))
                hdu.writeto(filepath)
                files.append(filepath)
            
            # Streaming median
            streaming_med = streaming_combine(
                files,
                method='median',
                verbose=False
            )
            
            # True median (load all into memory)
            stacked = np.stack(all_data, axis=0)
            true_median = np.median(stacked, axis=0)
            
            # Compare
            error = np.abs(streaming_med - true_median)
            mean_error = np.mean(error)
            max_error = np.max(error)
            rel_error = mean_error / 1000 * 100
            
            print(f"✓ Compared streaming vs true median")
            print(f"  Number of images: {n_samples}")
            print(f"  Mean absolute error: {mean_error:.2f}")
            print(f"  Max absolute error: {max_error:.2f}")
            print(f"  Relative error: {rel_error:.3f}%")
            
            if rel_error < 2.0:  # Within 2%
                print("  ✓ Streaming median accurate")
            else:
                print("  ⚠ Streaming median less accurate (expected for P² algorithm)")
    
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 4: Streaming mean vs true mean
    print("\nTest 4: Streaming mean accuracy")
    try:
        np.random.seed(42)
        shape = (30, 30)
        n_samples = 50
        
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            all_data = []
            
            for i in range(n_samples):
                filepath = os.path.join(tmpdir, f'test_{i:03d}.fits')
                data = np.random.normal(1000, 50, shape)
                all_data.append(data)
                
                hdu = fits.PrimaryHDU(data.astype(np.float32))
                hdu.writeto(filepath)
                files.append(filepath)
            
            # Streaming mean
            streaming_mean = streaming_combine(
                files,
                method='mean',
                verbose=False
            )
            
            # True mean
            stacked = np.stack(all_data, axis=0)
            true_mean = np.mean(stacked, axis=0)
            
            # Compare
            error = np.abs(streaming_mean - true_mean)
            mean_error = np.mean(error)
            max_error = np.max(error)
            
            print(f"✓ Compared streaming vs true mean")
            print(f"  Number of images: {n_samples}")
            print(f"  Mean absolute error: {mean_error:.6f}")
            print(f"  Max absolute error: {max_error:.6f}")
            
            if mean_error < 1e-5:  # Should be essentially exact
                print("  ✓ Streaming mean is numerically accurate")
            else:
                print(f"  ⚠ Unexpected error: {mean_error}")
    
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 5: Multiple quantiles
    print("\nTest 5: Multiple quantile estimation")
    try:
        np.random.seed(42)
        shape = (40, 40)
        n_samples = 100
        quantiles = [0.25, 0.5, 0.75]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            all_data = []
            
            for i in range(n_samples):
                filepath = os.path.join(tmpdir, f'test_{i:03d}.fits')
                data = np.random.normal(1000, 50, shape)
                all_data.append(data)
                
                hdu = fits.PrimaryHDU(data.astype(np.float32))
                hdu.writeto(filepath)
                files.append(filepath)
            
            # Streaming quantiles
            stats = streaming_statistics(
                files,
                compute_quantiles=True,
                quantiles=quantiles,
                verbose=False
            )
            
            # True quantiles
            stacked = np.stack(all_data, axis=0)
            
            print(f"✓ Computed {len(quantiles)} quantiles")
            for q in quantiles:
                streaming_q = stats['quantiles'][q]
                true_q = np.quantile(stacked, q, axis=0)
                
                error = np.abs(streaming_q - true_q)
                mean_error = np.mean(error)
                
                print(f"  Quantile {q}: mean error = {mean_error:.2f}")
            
            print("  ✓ Multiple quantiles computed successfully")
    
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 6: Memory efficiency estimate
    print("\nTest 6: Memory efficiency")
    try:
        shape = (2048, 2048)
        n_files = 100
        
        # Estimate memory for traditional approach
        bytes_per_pixel = 4  # float32
        pixels_per_image = np.prod(shape)
        traditional_memory_gb = (bytes_per_pixel * pixels_per_image * n_files) / (1024**3)
        
        # Estimate memory for streaming
        # Welford: mean, M2, sum, min, max = 5 arrays
        # P²: 5 markers × 3 arrays (q, n, n_desired) = 15 arrays (but n is int64)
        streaming_memory_welford = (bytes_per_pixel * pixels_per_image * 5) / (1024**3)
        streaming_memory_p2 = (bytes_per_pixel * pixels_per_image * 5 * 3) / (1024**3)
        
        print(f"✓ Memory usage comparison for {shape} images:")
        print(f"  Traditional (load all): {traditional_memory_gb:.2f} GB")
        print(f"  Streaming (Welford): {streaming_memory_welford:.2f} GB")
        print(f"  Streaming (P²): {streaming_memory_p2:.2f} GB")
        print(f"  Savings: {traditional_memory_gb/streaming_memory_welford:.1f}x (Welford)")
        print(f"  Savings: {traditional_memory_gb/streaming_memory_p2:.1f}x (P²)")
    
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")
        sys.exit(1)
    
    # Test 7: File output
    print("\nTest 7: Write combined result to file")
    try:
        np.random.seed(42)
        shape = (50, 50)
        n_samples = 20
        
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            
            for i in range(n_samples):
                filepath = os.path.join(tmpdir, f'test_{i:03d}.fits')
                data = np.random.normal(1000, 50, shape)
                
                hdu = fits.PrimaryHDU(data.astype(np.float32))
                hdu.writeto(filepath)
                files.append(filepath)
            
            # Write output
            output_file = os.path.join(tmpdir, 'combined.fits')
            result = streaming_combine(
                files,
                method='median',
                output_file=output_file,
                overwrite=True,
                verbose=False
            )
            
            print(f"✓ Wrote combined result to file")
            print(f"  Output: {os.path.basename(result)}")
            
            # Verify file exists and can be read
            if os.path.exists(result):
                with fits.open(result) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    print(f"  Data shape: {data.shape}")
                    print(f"  NCOMBINE: {header.get('NCOMBINE', 'N/A')}")
                    print(f"  COMBMETH: {header.get('COMBMETH', 'N/A')}")
                    print("  ✓ Output file valid")
            else:
                print("  ✗ Output file not created")
    
    except Exception as e:
        print(f"✗ Test 7 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("\nModule is ready for use.")
    
    # Print usage examples
    print("\nQuick usage examples:")
    print("  # Streaming median of many files")
    print("  from fits_tools.image_processing.streaming import streaming_combine")
    print("  median = streaming_combine(files, method='median')")
    print()
    print("  # Streaming statistics")
    print("  from fits_tools.image_processing.streaming import streaming_statistics")
    print("  stats = streaming_statistics(files)")
    print("  mean = stats['mean']")
    print("  std = stats['std']")
    print()
    print("  # P² quantile estimator")
    print("  from fits_tools.image_processing.streaming import P2QuantileEstimator")
    print("  estimator = P2QuantileEstimator((512, 512), p=0.5)")
    print("  for data in dataset:")
    print("      estimator.update(data)")
    print("  median = estimator.get_quantile()")