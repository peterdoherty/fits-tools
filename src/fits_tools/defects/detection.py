#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defect detection algorithms for bad pixel masks.

This module provides unified, configurable algorithms for identifying various
defect types in CCD/CMOS sensor calibration data. All detection methods are
designed to work on full-frame data while optionally using ROI-based statistics
for robust threshold calculation.

Detection Philosophy
--------------------
1. Calculate statistics from clean regions (ROI or sigma-clipped)
2. Apply thresholds to FULL FRAME to find all defects
3. Return defect maps that can be merged cumulatively
4. Each stage excludes previously identified defects from statistics

Pipeline Workflow
-----------------
::

    Bias Frame          Dark Frame          Flat Frame
        │                   │                   │
        ▼                   ▼                   ▼
 [Bias Detection]    [Dark Detection]    [Flat Detection]
  (no exclusions)    (exclude bias)      (exclude bias+dark)
        │                   │                   │
        └───────────┬───────┴───────────────────┘
                    ▼
          [Accumulated BPM]
                    │
                    ▼
       [Row/Column Detection]
                    │
                    ▼
        [Final BPM + Stats]

Each stage:
    1. Calculates statistics from clean pixels (using existing_bpm mask)
    2. Applies thresholds to FULL FRAME to detect defects
    3. Returns stage-specific BPM with appropriate bit flags
    4. Pipeline merges stages with bitwise OR

This ensures:
    - Early-stage defects don't bias later-stage statistics
    - All defects are detected across the full sensor area
    - Multiple defect types can coexist on a single pixel

Detection Stages
----------------
Bias Stage
    Identifies pixels with anomalous offset levels that deviate from median.
    Default: ±100 ADU threshold (configurable to sigma-based).
    Flags: ``BPM_BIT_PIX_BRIGHT``

Dark Stage
    Flags pixels with excess dark current (hot pixels only, high-side tail).
    Default: 5σ above median. Cold pixels are deferred to flat stage.
    Flags: ``BPM_BIT_PIX_HOT``
    
    **Why only hot pixels?** Dark frames measure thermal noise, which primarily
    manifests as elevated signal. Pixels with abnormally *low* dark current
    provide little diagnostic value; true cold/dead pixels are better identified
    from flat fields, which directly probe quantum efficiency.

Flat Stage
    Detects pixels with abnormal quantum efficiency (both hot and cold).
    Default: ±20% from median response.
    Flags: ``BPM_BIT_PIX_QE_HIGH``, ``BPM_BIT_PIX_QE_LOW``
    
    This is where dead pixels are definitively identified, as they show
    near-zero response to uniform illumination.

Rows/Columns Stage
    Identifies structural defects (entire rows/columns with high defect density).
    Default: >10% bad pixels in a row/column.
    Flags: ``BPM_BIT_ROW_BAD``, ``BPM_BIT_COL_BAD``
    
    Indicates readout path issues, amplifier problems, or damaged register sections.

Temporal Stage (optional)
    Detects pixels with high temporal variance (RTS/telegraph noise).
    Requires multiple frames of the same type.
    Flags: ``BPM_BIT_PIX_RTS``

Performance Characteristics
---------------------------
Typical processing times for 4K × 4K (16 megapixel) sensor:

=============================  ==========  ===============
Operation                      Time        Complexity
=============================  ==========  ===============
``sigma_clip_stats()``         5-20 ms     O(N × iters)
``detect_bias_defects()``      10-30 ms    O(N)
``detect_dark_hot_pixels()``   10-30 ms    O(N)
``detect_flat_defects()``      10-30 ms    O(N)
``detect_bad_rows_cols()``     5-10 ms     O(N)
``create_calibration_bpm()``   40-120 ms   O(N)
``detect_temporal_defects()``  50-200 ms   O(N_frames × N)
=============================  ==========  ===============

Where N = number of pixels (ny × nx).

Memory footprint
    - Calibration frames: 64 MB per 4K frame (float32) or 256 MB (float64)
    - BPM: 64 MB (uint32) or 32 MB (uint16)
    - Peak usage: ~3× largest frame (during sigma clipping)

Optimization notes
    - All operations are vectorized via NumPy
    - Sigma clipping dominates runtime (iterative algorithm)
    - Temporal detection scales linearly with number of frames
    - Consider multiprocessing for batch processing multiple sensors
    - Memory-map large frame stacks to reduce RAM usage

Basic Usage
-----------
Quick start with default parameters::

    >>> from fits_tools.defects import create_calibration_bpm
    >>> 
    >>> # Detect defects from master calibration frames
    >>> bpm, stats = create_calibration_bpm(
    ...     master_bias,
    ...     master_dark,
    ...     master_flat
    ... )
    >>> 
    >>> print(f"Detected {stats['total_defects']} defects "
    ...       f"({stats['total_fraction']*100:.2f}%)")

With custom parameters and ROI::

    >>> config = get_default_detection_config()
    >>> config['bias']['threshold_adu'] = 150.0  # Less sensitive
    >>> config['dark']['threshold_sigma'] = 7.0   # More conservative
    >>> 
    >>> bpm, stats = create_calibration_bpm(
    ...     master_bias,
    ...     master_dark,
    ...     master_flat,
    ...     config=config,
    ...     stats_roi=(100, 1900, 100, 1900)  # Use clean central region
    ... )

Individual stage detection::

    >>> # Detect only dark hot pixels
    >>> dark_bpm, dark_stats = detect_dark_hot_pixels(
    ...     master_dark,
    ...     threshold_sigma=5.0,
    ...     stats_roi=(100, 1900, 100, 1900)
    ... )
    >>> 
    >>> # Apply to existing BPM
    >>> from fits_tools.defects import apply_bpm
    >>> science_masked = apply_bpm(science_data, dark_bpm, fill_value=np.nan)

Temporal defect detection::

    >>> # Requires multiple frames (>=3)
    >>> dark_stack = np.array([dark1, dark2, dark3, ...])
    >>> 
    >>> temporal_bpm, temporal_stats = detect_temporal_defects(
    ...     dark_stack,
    ...     threshold_sigma=5.0
    ... )
    >>> 
    >>> print(f"RTS noise pixels: {temporal_stats['num_defects']}")

Batch processing with frame combination::

    >>> # Load frame stacks
    >>> bias_stack = np.array([bias1, bias2, ..., bias10])
    >>> dark_stack = np.array([dark1, dark2, ..., dark10])
    >>> flat_stack = np.array([flat1, flat2, ..., flat10])
    >>> 
    >>> # Combine and detect in one step
    >>> bpm, masters, stats = create_calibration_bpm_batch(
    ...     bias_stack, dark_stack, flat_stack,
    ...     combine_method='sigma_clip'
    ... )
    >>> 
    >>> master_bias, master_dark, master_flat = masters

Saving and loading::

    >>> from fits_tools.defects.io import save_bpm_to_fits, load_bpm_from_fits
    >>> 
    >>> # Save with metadata
    >>> save_bpm_to_fits(bpm, 'master_bpm.fits', stats=stats)
    >>> 
    >>> # Load and validate
    >>> bpm_loaded, metadata = load_bpm_from_fits('master_bpm.fits')
    >>> print(metadata['total_bad'])

Visualization::

    >>> from fits_tools.defects import create_detection_report
    >>> 
    >>> # Generate comprehensive PDF report
    >>> create_detection_report(
    ...     bpm, stats,
    ...     output_path='bpm_report.pdf',
    ...     show_plots=False
    ... )

Best Practices
--------------
1. **Use high SNR master frames**: Combine ≥10 frames per calibration type
   to reduce statistical noise and improve defect detection accuracy.

2. **Exclude edge regions**: Use ``stats_roi`` to compute statistics from
   the central detector region, avoiding vignetting, overscan, and edge effects:
   
   >>> stats_roi=(200, 1800, 200, 1800)  # For 2048×2048 sensor

3. **Validate results visually**: Always generate and review the detection
   report to catch algorithm failures or inappropriate thresholds:
   
   >>> create_detection_report(bpm, stats, output_path='review.pdf')

4. **Start conservative**: Use higher thresholds initially, then tighten
   based on false positive rate in science data.

5. **Track detector health**: Save dated BPMs and compare over time to
   monitor degradation:
   
   >>> plot_defect_comparison([bpm_2023, bpm_2024], ['2023', '2024'])

6. **Re-generate periodically**: Detectors degrade over time (radiation damage,
   aging). Regenerate BPMs every 6-12 months or after major temperature cycles.

7. **Document provenance**: Always save detection parameters with the BPM
   using ``save_bpm_to_fits()`` to ensure reproducibility.

See Also
--------
fits_tools.defects.bitmask : Bit flag definitions and utilities
fits_tools.defects.io : FITS I/O for bad pixel masks
fits_tools.calibration : Calibration frame creation and combination

Notes
-----
All detection functions return two values:
    1. ``bpm`` (np.ndarray): Bad pixel mask with appropriate bit flags set
    2. ``stats`` (dict): Detection statistics including thresholds, counts, etc.

The stats dictionary always includes:
    - Computed thresholds (method-dependent)
    - Median and standard deviation of clean region
    - Number of defects detected (``num_defects``)
    - Fraction of detector affected (``fraction_defects``)

References
----------
.. [1] Janesick, J. R. (2001). "Scientific Charge-Coupled Devices"
       SPIE Press. Chapter 5: Defects and Blemishes.
.. [2] Howell, S. B. (2006). "Handbook of CCD Astronomy" (2nd ed.)
       Cambridge University Press. Section 3.2: Bad Pixels.

Version: 1.0.0
"""

import logging
from typing import Tuple, Optional, Dict, List
import numpy as np
from ..bitmask import (
    BPM_BIT_PIX_BRIGHT, BPM_BIT_PIX_HOT, BPM_BIT_PIX_QE_HIGH, BPM_BIT_PIX_QE_LOW,
    BPM_BIT_ROW_BAD, BPM_BIT_COL_BAD,
    create_bpm
)

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

def _validate_roi(shape: Tuple[int, int], roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Validate and normalize an ROI tuple.

    Parameters
    ----------
    shape : tuple
        (ny, nx) of the full-frame array.
    roi : tuple
        ROI as (x1, x2, y1, y2), half-open bounds.

    Returns
    -------
    tuple
        Normalized ROI (x1, x2, y1, y2) as ints.

    Raises
    ------
    ValueError
        If ROI is out of bounds or degenerate.
    """
    if roi is None:
        raise ValueError("roi must not be None")
    if len(roi) != 4:
        raise ValueError(f"roi must have 4 elements (x1, x2, y1, y2); got {roi!r}")

    x1, x2, y1, y2 = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    ny, nx = int(shape[0]), int(shape[1])

    if not (0 <= x1 <= nx and 0 <= x2 <= nx and 0 <= y1 <= ny and 0 <= y2 <= ny):
        raise ValueError(f"roi {roi!r} is out of bounds for shape {shape}")

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"roi {roi!r} is degenerate (requires x2>x1 and y2>y1)")

    return x1, x2, y1, y2

# =============================================================================
# Statistical Utilities
# =============================================================================

def sigma_clip_stats(
    data: np.ndarray,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    max_iters: int = 5,
    mask: Optional[np.ndarray] = None,
    scale: str = 'std'
) -> Tuple[float, float]:
    """
    Calculate robust statistics using iterative sigma clipping.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    sigma_low : float
        Lower clipping threshold in sigma.
    sigma_high : float
        Upper clipping threshold in sigma.
    max_iters : int
        Maximum iterations for convergence.
    mask : np.ndarray, optional
        Boolean mask of pixels to exclude (True = exclude).
    scale : str
        Scale estimator used for clipping thresholds: 'std' (default) or 'mad'.

    Returns
    -------
    median : float
        Clipped median value.
    std : float
        Clipped scale estimate (std or MAD-derived sigma).

    Notes
    -----
    Returns NaN statistics if all samples are excluded/invalid.
    """

    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be ndarray; got {type(data)}")
    
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"data must have numeric dtype; got {data.dtype}")
    
    if sigma_low < 0 or sigma_high < 0:
        raise ValueError(
            f"sigma_low and sigma_high must be non-negative; "
            f"got {sigma_low}, {sigma_high}"
        )
    
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1; got {max_iters}")
    
    work = np.asarray(data).ravel()

    # Apply mask and finite filter
    if mask is not None:
        m = np.asarray(mask).ravel()
        if m.shape != work.shape:
            raise ValueError(f"mask shape {mask.shape} does not match data shape {data.shape}")
        work = work[~m]

    if work.size == 0:
        return float('nan'), float('nan')

    finite = np.isfinite(work)
    if not np.all(finite):
        work = work[finite]
    if work.size == 0:
        return float('nan'), float('nan')

    scale_mode = str(scale).lower()
    if scale_mode not in {'std', 'mad'}:
        raise ValueError(f"scale must be 'std' or 'mad'; got {scale!r}")

    for _ in range(int(max_iters)):
        median = np.nanmedian(work)

        if scale_mode == 'std':
            s = np.nanstd(work)
        else:  # 'mad'
            mad = np.nanmedian(np.abs(work - median))
            s = 1.4826 * mad

        if not np.isfinite(s) or s == 0:
            # Scale is degenerate (constant data or all-identical values).
            # Break early; the post-loop recompute below will return s=0,
            # which callers must handle (e.g. detect_dark_hot_pixels guards on s>0).
            break

        lower = median - float(sigma_low) * s
        upper = median + float(sigma_high) * s

        clipped = work[(work >= lower) & (work <= upper)]
        if clipped.size == work.size:
            break  # converged

        work = clipped
        if work.size == 0:
            return float('nan'), float('nan')

    # Return stats for the final clipped sample
    median = np.nanmedian(work)
    if scale_mode == 'std':
        s = np.nanstd(work)
    else:
        mad = np.nanmedian(np.abs(work - median))
        s = 1.4826 * mad

    # At the end, addperform validation:
    if work.size < 2:
        logger.debug(
            f"sigma_clip_stats: only {work.size} valid samples remain after clipping; "
            f"scale estimate is unreliable."
        )
    
    return float(median), float(s)

def _sigma_clip_combine(frames: np.ndarray, sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Combine frames using sigma-clipped mean."""
    n_frames = frames.shape[0]
    
    # Compute per-pixel median and MAD across frames
    median = np.median(frames, axis=0)
    mad = np.median(np.abs(frames - median[np.newaxis, :, :]), axis=0)
    sigma_est = 1.4826 * mad
    
    # Mask outliers
    lower = median - sigma * sigma_est
    upper = median + sigma * sigma_est
    
    # Create mask: True where pixel should be excluded
    mask = (frames < lower[np.newaxis, :, :]) | (frames > upper[np.newaxis, :, :])
    
    # Compute masked mean
    masked_frames = np.ma.masked_array(frames, mask=mask)
    combined = np.ma.mean(masked_frames, axis=0).filled(median)
    n_rejected = np.sum(mask, axis=0)
    
    return combined, n_rejected

def roi_stats(
    data: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    sigma_clip: bool = True,
    mask: Optional[np.ndarray] = None,
    scale: str = 'std',
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    max_iters: int = 5,
) -> Tuple[float, float]:
    """
    Calculate statistics from a region of interest.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    roi : tuple, optional
        ROI as (x1, x2, y1, y2). If None, uses full frame.
    sigma_clip : bool
        If True, use sigma clipping for robust statistics.
    mask : np.ndarray, optional
        Boolean mask of pixels to exclude (True = exclude). If ROI is provided,
        the mask is sliced identically.
    scale : str
        Scale estimator for sigma clipping thresholds ('std' or 'mad').
    sigma_low : float
        Lower sigma bound for clipping. Default: 3.0.
    sigma_high : float
        Upper sigma bound for clipping. Default: 3.0.
    max_iters : int
        Maximum iterations for sigma clipping. Default: 5.

    Returns
    -------
    median : float
        Median (or clipped median) of ROI.
    std : float
        Standard deviation or robust sigma estimate of ROI.
    """
    if roi is not None:
        x1, x2, y1, y2 = _validate_roi(data.shape, roi)
        roi_data = data[y1:y2, x1:x2]
        roi_mask = None if mask is None else mask[y1:y2, x1:x2]
    else:
        roi_data = data
        roi_mask = mask

    if sigma_clip:
        return sigma_clip_stats(
            roi_data, mask=roi_mask, scale=scale,
            sigma_low=sigma_low, sigma_high=sigma_high, max_iters=max_iters,
        )

    if roi_mask is not None:
        vals = np.asarray(roi_data)[~roi_mask]
        if vals.size == 0:
            return float('nan'), float('nan')
        return float(np.nanmedian(vals)), float(np.nanstd(vals))

    return float(np.nanmedian(roi_data)), float(np.nanstd(roi_data))

def detect_bias_defects(
    bias_data: np.ndarray,
    threshold_adu: float = 100.0,
    method: str = 'absolute',
    sigma: float = 5.0,
    stats_roi: Optional[Tuple[int, int, int, int]] = None,
    existing_bpm: Optional[np.ndarray] = None,
    mode: str = 'high',
    stats_scale: str = 'std',
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    max_iters: int = 5,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect bias-frame defects.

    Identifies pixels with anomalous offset levels that deviate from the
    median bias level. Uses an absolute ADU offset by default (simple and
    interpretable), with optional sigma-based thresholding.

    Parameters
    ----------
    bias_data : np.ndarray
        Master bias frame (ADU).
    threshold_adu : float
        Absolute threshold *offset* from median (ADU). Default: 100.0.
    method : str
        Detection method: 'absolute' or 'sigma'. Default: 'absolute'.
    sigma : float
        Sigma threshold when method='sigma'. Default: 5.0.
    stats_roi : tuple, optional
        ROI for statistics as (x1, x2, y1, y2). If None, uses full frame.
    existing_bpm : np.ndarray, optional
        Existing BPM to exclude from statistics (pixels where existing_bpm>0).
    mode : str
        Which side(s) to flag relative to the median: 'high' (default), 'low', or 'both'.
    stats_scale : str
        Scale estimator for sigma clipping ('std' or 'mad'). Default: 'std'.
    sigma_low : float
        Lower sigma bound for the internal sigma-clipping statistics pass.
        Default: 3.0.
    sigma_high : float
        Upper sigma bound for the internal sigma-clipping statistics pass.
        Default: 3.0.
    max_iters : int
        Maximum iterations for the internal sigma-clipping statistics pass.
        Default: 5.

    Returns
    -------
    bpm : np.ndarray
        Stage BPM with BPM_BIT_PIX_BRIGHT set for detected defects.
    stats : dict
        Detection statistics and thresholds used.
    """
    bpm = create_bpm(bias_data.shape, dtype=bpm_dtype)

    mask = (existing_bpm > 0) if existing_bpm is not None else None

    method = str(method).lower()
    mode = str(mode).lower()
    stats_scale = str(stats_scale)

    if method not in {'absolute', 'sigma'}:
        raise ValueError(f"method must be 'absolute' or 'sigma'; got {method!r}")
    if mode not in {'high', 'low', 'both'}:
        raise ValueError(f"mode must be 'high', 'low', or 'both'; got {mode!r}")

    # Compute robust stats from clean region
    if stats_roi is not None:
        median, std = roi_stats(
            bias_data, roi=stats_roi, sigma_clip=True, mask=mask,
            scale=stats_scale,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            max_iters=max_iters,
        )
    else:
        median, std = sigma_clip_stats(
            bias_data, mask=mask, scale=stats_scale,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            max_iters=max_iters,
        )

    stats: Dict[str, object] = {
        'method': method,
        'mode': mode,
        'stats_scale': stats_scale,
        'sigma_low': float(sigma_low),
        'sigma_high': float(sigma_high),
        'max_iters': int(max_iters),
        'median': float(median),
        'std': float(std),
    }

    if not np.isfinite(median):
        logger.warning(
            "detect_bias_defects: degenerate median (%s); no defects will be flagged. "
            "Check that bias_data contains valid signal.",
            median,
        )
        stats.update({
            'threshold_mode': method,
            'threshold_hi_adu': float('nan'),
            'threshold_lo_adu': float('nan'),
            'num_defects': 0,
            'fraction_defects': 0.0,
            # Keep schema consistent with the normal return path
            'threshold_offset_adu': float(threshold_adu) if method == 'absolute' else float('nan'),
            'threshold_sigma': float(sigma) if method == 'sigma' else float('nan'),
        })
        return bpm, stats

    if method == 'absolute':
        thr_hi = median + float(threshold_adu)
        thr_lo = median - float(threshold_adu)

        if mode == 'high':
            defect_mask = bias_data > thr_hi
        elif mode == 'low':
            defect_mask = bias_data < thr_lo
        else:
            defect_mask = (bias_data > thr_hi) | (bias_data < thr_lo)

        stats.update({
            'threshold_mode': 'absolute_offset',
            'threshold_offset_adu': float(threshold_adu),
            'threshold_hi_adu': float(thr_hi),
            'threshold_lo_adu': float(thr_lo),
        })

    else:  # method == 'sigma'
        if not np.isfinite(std) or std <= 0:
            defect_mask = np.zeros_like(bias_data, dtype=bool)
            thr_hi = float('nan')
            thr_lo = float('nan')
        else:
            thr_hi = median + float(sigma) * std
            thr_lo = median - float(sigma) * std

            if mode == 'high':
                defect_mask = bias_data > thr_hi
            elif mode == 'low':
                defect_mask = bias_data < thr_lo
            else:
                defect_mask = (bias_data > thr_hi) | (bias_data < thr_lo)

        stats.update({
            'threshold_mode': 'sigma',
            'threshold_sigma': float(sigma),
            'threshold_hi_adu': float(thr_hi),
            'threshold_lo_adu': float(thr_lo),
        })

    bpm[defect_mask] |= BPM_BIT_PIX_BRIGHT

    stats['num_defects'] = int(np.sum(defect_mask))
    stats['fraction_defects'] = stats['num_defects'] / bias_data.size

    return bpm, stats

def detect_dark_hot_pixels(
    dark_data: np.ndarray,
    threshold_sigma: float = 5.0,
    stats_roi: Optional[Tuple[int, int, int, int]] = None,
    existing_bpm: Optional[np.ndarray] = None,
    stats_scale: str = 'std',
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect dark-frame hot pixels.

    Flags pixels whose dark current exceeds the robust median by more than
    ``threshold_sigma`` sigma. Only the high-side tail is examined: a pixel
    with anomalously *low* dark current is not flagged here because the master
    dark is a poor discriminator for that condition. Cold/dead pixels are
    better identified from the flat field, which directly probes quantum
    efficiency and charge collection (see :func:`detect_flat_defects`).

    Parameters
    ----------
    dark_data : np.ndarray
        Master dark frame (bias-subtracted, ADU or e-).
    threshold_sigma : float
        Sigma threshold above the robust median to flag hot pixels.
        Default: 5.0.
    stats_roi : tuple, optional
        ROI for statistics as (x1, x2, y1, y2). If None, uses full frame.
    existing_bpm : np.ndarray, optional
        Existing BPM to exclude from statistics (pixels where existing_bpm>0).
    stats_scale : str
        Scale estimator for sigma clipping ('std' or 'mad'). Default: 'std'.
    bpm_dtype : np.dtype, optional
        Dtype of the output BPM array. Default: np.uint32.

    Returns
    -------
    bpm : np.ndarray
        Stage BPM with BPM_BIT_PIX_HOT set for detected hot pixels.
    stats : dict
        Detection statistics and thresholds used.
    """
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError(f"threshold_sigma must be positive; got {threshold_sigma!r}")

    bpm = create_bpm(dark_data.shape, dtype=bpm_dtype)
    mask = (existing_bpm > 0) if existing_bpm is not None else None
    stats_scale = str(stats_scale)

    # Robust stats from clean region
    if stats_roi is not None:
        median, std = roi_stats(dark_data, roi=stats_roi, sigma_clip=True, mask=mask, scale=stats_scale)
    else:
        median, std = sigma_clip_stats(dark_data, mask=mask, scale=stats_scale)

    # Flag only the high-side tail
    if not np.isfinite(std) or std <= 0:
        hot_mask = np.zeros_like(dark_data, dtype=bool)
        thr_hi = float('nan')
    else:
        thr_hi = median + float(threshold_sigma) * std
        hot_mask = dark_data > thr_hi

    bpm[hot_mask] |= BPM_BIT_PIX_HOT

    stats = {
        'stats_scale': stats_scale,
        'median': float(median),
        'std': float(std),
        'threshold_sigma': float(threshold_sigma),
        'threshold_hi': float(thr_hi),
        'num_defects': int(np.sum(hot_mask)),
        'fraction_defects': float(np.sum(hot_mask)) / dark_data.size,
    }

    return bpm, stats

def detect_flat_defects(
    flat_data: np.ndarray,
    hot_threshold_percent: float = 20.0,
    cold_threshold_percent: float = 20.0,
    normalize: bool = True,
    stats_roi: Optional[Tuple[int, int, int, int]] = None,
    existing_bpm: Optional[np.ndarray] = None,
    stats_scale: str = 'std',
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect flat-field response defects (hot and cold/dead pixels).

    Parameters
    ----------
    flat_data : np.ndarray
        Master flat frame (bias/dark-corrected).
    hot_threshold_percent : float
        Percent above median response to flag as flat hot.
    cold_threshold_percent : float
        Percent below median response to flag as flat cold/dead.
    normalize : bool
        If True, normalize flat by the robust median before thresholding.
    stats_roi : tuple, optional
        ROI for statistics as (x1, x2, y1, y2). If None, uses full frame.
    existing_bpm : np.ndarray, optional
        Existing BPM to exclude from statistics (pixels where existing_bpm>0).
    stats_scale : str
        Scale estimator for sigma clipping ('std' or 'mad'). Default: 'std'.

    Returns
    -------
    bpm : np.ndarray
        Stage BPM with BPM_BIT_FLAT_HOT and/or BPM_BIT_FLAT_COLD set.
    stats : dict
        Detection statistics and thresholds used.

    Notes
    -----
    - When normalize=False, pixels with flat_data < cold_threshold are flagged.
      This includes pixels with zero response (dead pixels).
    - When normalize=True, response is relative to median (e.g., < 0.8 × median).
    """

    if not np.isfinite(hot_threshold_percent) or hot_threshold_percent < 0:
        raise ValueError(f"hot_threshold_percent must be >= 0; got {hot_threshold_percent!r}")
    if not np.isfinite(cold_threshold_percent) or cold_threshold_percent < 0:
        raise ValueError(f"cold_threshold_percent must be >= 0; got {cold_threshold_percent!r}")

    bpm = create_bpm(flat_data.shape, dtype=bpm_dtype)
    mask = (existing_bpm > 0) if existing_bpm is not None else None
    stats_scale = str(stats_scale)

    if stats_roi is not None:
        median, std = roi_stats(flat_data, roi=stats_roi, sigma_clip=True, mask=mask, scale=stats_scale)
    else:
        median, std = sigma_clip_stats(flat_data, mask=mask, scale=stats_scale)

    if not np.isfinite(median) or median == 0:
        logger.warning(
            "detect_flat_defects: degenerate median (%s); no defects will be flagged. "
            "Check that flat_data contains valid, non-zero signal.",
            median,
        )
        return bpm, {
            'stats_scale': stats_scale,
            'median': float(median),
            'std': float(std),
            'normalize': bool(normalize),
            'hot_threshold_percent': float(hot_threshold_percent),
            'cold_threshold_percent': float(cold_threshold_percent),
            'hot_threshold': float('nan'),
            'cold_threshold': float('nan'),
            'num_hot': 0,
            'num_cold': 0,
            'num_both': 0,
            'num_defects': 0,
            'fraction_defects': 0.0,
        }

    if normalize:
        flat_norm = flat_data / float(median)
        norm_median = 1.0
    else:
        flat_norm = flat_data
        norm_median = float(median)

    hot_thr = norm_median * (1.0 + float(hot_threshold_percent) / 100.0)
    cold_thr = norm_median * (1.0 - float(cold_threshold_percent) / 100.0)

    hot_mask = flat_norm > hot_thr
    cold_mask = flat_norm < cold_thr

    bpm[hot_mask] |= BPM_BIT_PIX_QE_HIGH
    bpm[cold_mask] |= BPM_BIT_PIX_QE_LOW

    stats = {
        'stats_scale': stats_scale,
        'median': float(median),
        'std': float(std),
        'normalize': bool(normalize),
        'hot_threshold_percent': float(hot_threshold_percent),
        'cold_threshold_percent': float(cold_threshold_percent),
        'hot_threshold': float(hot_thr),
        'cold_threshold': float(cold_thr),
        'num_hot': int(np.sum(hot_mask)),
        'num_cold': int(np.sum(cold_mask)),
        'num_defects': int(np.sum(hot_mask | cold_mask)),  # Union
        'fraction_defects': float(np.sum(hot_mask | cold_mask)) / flat_data.size,
    }

    return bpm, stats

def detect_bad_rows_cols(
    defect_map: np.ndarray,
    row_threshold: float = 0.1,
    col_threshold: float = 0.1,
    active_region: Optional[Tuple[slice, slice]] = None,
    defect_mask: Optional[int] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect entire rows and columns with high defect density.

    Analyzes an existing defect map to identify rows/columns where
    the fraction of bad pixels exceeds a threshold. These are flagged
    as structural defects (bad readout paths, amplifier issues, etc.).

    Parameters
    ----------
    defect_map : np.ndarray
        Existing BPM with pixel-level defects already identified.
    row_threshold : float
        Fraction of defective pixels to flag entire row. Default: 0.1 (10%).
    col_threshold : float
        Fraction of defective pixels to flag entire column. Default: 0.1 (10%).
    active_region : tuple of slices, optional
        (row_slice, col_slice) defining active region to analyze.
        If None, uses entire array.
    defect_mask : int, optional
        If provided, only these defect bits contribute to density (bitwise OR mask).
        If None, any nonzero pixel contributes.

    Returns
    -------
    bpm : np.ndarray
        BPM with BPM_BIT_ROW_BAD and BPM_BIT_COL_BAD flags added.
    stats : dict
        Statistics about bad rows and columns.
    """
    bpm = create_bpm(defect_map.shape, dtype=bpm_dtype)

    if active_region is None:
        active_region = (slice(None), slice(None))

    rows, cols = active_region
    active_defects = defect_map[rows, cols]

    # Calculate defect density per row/column within the active region
    if defect_mask is None:
        bad_pixel_mask = active_defects > 0
    else:
        bad_pixel_mask = (active_defects & int(defect_mask)) != 0

    # Row analysis
    row_bad_fractions = np.mean(bad_pixel_mask, axis=1)
    rel_bad_rows = np.where(row_bad_fractions > row_threshold)[0]

    # Column analysis
    col_bad_fractions = np.mean(bad_pixel_mask, axis=0)
    rel_bad_cols = np.where(col_bad_fractions > col_threshold)[0]

    # FIXED: Robust slice-to-index conversion using np.arange + slicing
    # This handles negative indices, None bounds, and step values correctly
    ny, nx = defect_map.shape
    
    # Convert slice to actual indices
    row_indices = np.arange(ny)[rows]
    col_indices = np.arange(nx)[cols]
    
    # Validate that we got the expected number of indices
    if len(row_indices) != active_defects.shape[0]:
        raise RuntimeError(
            f"Internal error: row slice produced {len(row_indices)} indices "
            f"but active_defects has {active_defects.shape[0]} rows"
        )
    if len(col_indices) != active_defects.shape[1]:
        raise RuntimeError(
            f"Internal error: col slice produced {len(col_indices)} indices "
            f"but active_defects has {active_defects.shape[1]} columns"
        )
    
    # Map relative indices back to full frame
    bad_rows = row_indices[rel_bad_rows]
    bad_cols = col_indices[rel_bad_cols]

    # Flag entire rows/columns in the FULL FRAME
    if bad_rows.size > 0:
        bpm[bad_rows, :] |= BPM_BIT_ROW_BAD

    if bad_cols.size > 0:
        bpm[:, bad_cols] |= BPM_BIT_COL_BAD

    stats = {
        'num_bad_rows': int(bad_rows.size),
        'num_bad_cols': int(bad_cols.size),
        'bad_rows': bad_rows.tolist(),
        'bad_cols': bad_cols.tolist(),
        'row_threshold': float(row_threshold),
        'col_threshold': float(col_threshold),
        'active_region': str(active_region),
        'defect_mask': None if defect_mask is None else int(defect_mask),
    }

    return bpm, stats

def detect_temporal_defects(
    frame_stack: np.ndarray,
    threshold_sigma: float = 5.0,
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect pixels with high temporal variance (RTS/telegraph noise).

    Parameters
    ----------
    frame_stack : np.ndarray
        Stack of frames with shape (n_frames, ny, nx).  All frames should
        be of the same calibration type (e.g. all darks) and processed
        identically (bias-subtracted, same exposure time).
    threshold_sigma : float
        Flag pixels where temporal std exceeds
        ``median_std + threshold_sigma × robust_std`` of the
        distribution of per-pixel temporal stds.  Default: 5.0.
    existing_bpm : np.ndarray, optional
        Master BPM of permanently known-bad pixels.  These are excluded
        from both the per-pixel std computation and the threshold
        statistics, preventing hot pixels from inflating the noise floor
        and generating spurious telegraph detections.
    bpm_dtype : np.dtype, optional
        Dtype of the output BPM array.  Default: np.uint32.

    Returns
    -------
    bpm : np.ndarray
        BPM with BPM_BIT_PIX_RTS set for temporally unstable pixels.
    stats : dict
        Detection statistics.

    Notes
    -----
    Requires at least 3 frames; more frames give a more reliable std
    estimate.  10+ frames are recommended for production use.
    """
    from ..bitmask import BPM_BIT_PIX_RTS

    if frame_stack.ndim != 3:
        raise ValueError(
            f"frame_stack must be 3D (n_frames, ny, nx); got shape {frame_stack.shape}"
        )

    n_frames, ny, nx = frame_stack.shape
    if n_frames < 3:
        raise ValueError(
            f"Need at least 3 frames for temporal analysis; got {n_frames}"
        )

    bpm = create_bpm((ny, nx), dtype=bpm_dtype)

    # Build a float working copy with known-bad pixels set to NaN so they
    # do not contribute to the per-pixel temporal std.  Without this, a
    # permanently hot pixel always has a large temporal std and will fire
    # every time, inflating the telegraph count.
    if existing_bpm is not None:
        work_stack = frame_stack.astype(float, copy=True)
        work_stack[:, existing_bpm > 0] = np.nan
        temporal_std = np.nanstd(work_stack, axis=0)
    else:
        temporal_std = np.std(frame_stack, axis=0)

    # Build an exclusion mask for the threshold statistics: pixels that are
    # already known-bad, or whose temporal std is NaN (all frames were NaN),
    # are excluded from the distribution used to set the threshold.
    stat_mask = ~np.isfinite(temporal_std)
    if existing_bpm is not None:
        stat_mask |= (existing_bpm > 0)

    median_std, robust_std = sigma_clip_stats(temporal_std, mask=stat_mask)

    if not np.isfinite(median_std) or not np.isfinite(robust_std) or robust_std <= 0:
        logger.warning(
            "detect_temporal_defects: degenerate statistics "
            "(median_std=%s, robust_std=%s); no pixels will be flagged.",
            median_std, robust_std,
        )
        return bpm, {
            'n_frames': n_frames,
            'median_temporal_std': float(median_std),
            'robust_std': float(robust_std),
            'threshold': float('nan'),
            'num_defects': 0,
            'fraction_defects': 0.0,
        }

    threshold = median_std + threshold_sigma * robust_std
    telegraph_mask = np.isfinite(temporal_std) & (temporal_std > threshold)

    bpm[telegraph_mask] |= BPM_BIT_PIX_RTS

    stats = {
        'n_frames':            n_frames,
        'median_temporal_std': float(median_std),
        'robust_std':          float(robust_std),
        'threshold_sigma':     float(threshold_sigma),
        'threshold':           float(threshold),
        'num_defects':         int(np.sum(telegraph_mask)),
        'fraction_defects':    float(np.sum(telegraph_mask)) / (ny * nx),
    }

    return bpm, stats

def detect_defect_clusters(
    bpm: np.ndarray,
    min_cluster_size: int = 5,
    defect_mask: Optional[int] = None,
) -> Dict:
    """
    Identify clusters of adjacent defective pixels.
    
    Useful for detecting localized sensor damage, particle impacts, etc.
    
    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask.
    min_cluster_size : int
        Minimum cluster size to report.
    defect_mask : int, optional
        Which defect types to analyze.
    
    Returns
    -------
    dict
        Cluster statistics with keys:
        - 'num_clusters': Number of clusters found
        - 'cluster_sizes': List of cluster sizes
        - 'largest_cluster': Size of largest cluster
        - 'mean_cluster_size': Mean cluster size
    """
    from scipy import ndimage
    
    if defect_mask is None:
        bad_pixels = bpm > 0
    else:
        bad_pixels = (bpm & defect_mask) != 0
    
    # 8-connectivity: all 8 neighbours (including diagonals) are considered
    # connected.  This is appropriate for image sensors where a charge
    # injection or particle track can spread diagonally between pixels.
    structure_8 = ndimage.generate_binary_structure(2, 2)  # 3×3 all-True
    labeled, num_clusters = ndimage.label(bad_pixels, structure=structure_8)
    
    if num_clusters == 0:
        return {
            'num_clusters': 0,
            'cluster_sizes': [],
            'largest_cluster': 0,
            'mean_cluster_size': 0.0,
        }
    
    # Get cluster sizes
    cluster_sizes = np.bincount(labeled.ravel())[1:]  # Exclude background (label 0)
    large_clusters = cluster_sizes[cluster_sizes >= min_cluster_size]
    
    return {
        'num_clusters': int(num_clusters),
        'num_large_clusters': int(len(large_clusters)),
        'cluster_sizes': large_clusters.tolist(),
        'largest_cluster': int(np.max(cluster_sizes)),
        'mean_cluster_size': float(np.mean(cluster_sizes)),
        'median_cluster_size': float(np.median(cluster_sizes)),
    }

def detect_saturated_pixels(
    flat_data: np.ndarray,
    saturation_threshold: float = 0.95,
    bit_depth: int = 16,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect pixels that are at or above the saturation level in a given frame.

    Parameters
    ----------
    flat_data : np.ndarray
        Image frame to check (ADU).
    saturation_threshold : float
        Fraction of full-well capacity to treat as saturated (0.0–1.0).
        Default: 0.95.
    bit_depth : int
        ADC bit depth determining the maximum ADU value. Default: 16 (→ 65535).

    Returns
    -------
    bpm : np.ndarray
        BPM with BPM_BIT_SATURATED set wherever the pixel is saturated.
    stats : dict
        Detection statistics.

    Notes
    -----
    This function operates on a **single frame** and the resulting BPM describes
    saturation *in that frame only*.  It is not appropriate for merging into a
    master bad-pixel map.  See usage guidance below.
    """
    from ..bitmask import BPM_BIT_SATURATED

    if not (0.0 < saturation_threshold <= 1.0):
        raise ValueError(
            f"saturation_threshold must be in (0, 1]; got {saturation_threshold!r}"
        )

    bpm = create_bpm(flat_data.shape, dtype=bpm_dtype)
    max_value  = (2 ** bit_depth) - 1
    sat_level  = max_value * saturation_threshold
    sat_mask   = flat_data >= sat_level
    bpm[sat_mask] |= BPM_BIT_SATURATED

    stats = {
        'bit_depth':            bit_depth,
        'max_value':            max_value,
        'saturation_threshold': float(saturation_threshold),
        'saturation_level':     float(sat_level),
        'num_saturated':        int(np.sum(sat_mask)),
        'fraction_saturated':   float(np.sum(sat_mask)) / flat_data.size,
    }

    return bpm, stats


def create_defect_density_map(
    defect_map: np.ndarray,
    tile_size: int = 64,
    defect_mask: Optional[int] = None
) -> Dict:
    """
    Create a downsampled defect density map.
    
    Divides the detector into tiles and counts defects per tile.
    Useful for visualizing spatial distribution of defects.
    
    Parameters
    ----------
    defect_map : np.ndarray
        BPM with defects flagged.
    tile_size : int
        Size of square tiles in pixels. Default: 64.
    
    Returns
    -------
    dict
        Dictionary with density map and metadata:
        - 'tile_size': tile size in pixels
        - 'ny_tiles': number of tiles in y
        - 'nx_tiles': number of tiles in x
        - 'counts': 2D array of defect counts per tile
    
    Examples
    --------
    >>> density_map = create_defect_density_map(bpm, tile_size=64)
    >>> print(f"Tile with most defects: {np.max(density_map['counts'])}")
    """
    ny, nx = defect_map.shape
    ny_tiles = ny // tile_size
    nx_tiles = nx // tile_size
    
    if ny_tiles == 0 or nx_tiles == 0:
        logger.warning(
            f"Image too small for defect density tiling "
            f"(tile_size={tile_size}, shape={defect_map.shape})"
        )
        return {
            'tile_size': tile_size,
            'ny_tiles': 0,
            'nx_tiles': 0,
            'counts': None,
        }

    # Trim to exact tile coverage, then reshape into (ny_tiles, tile_size, nx_tiles, tile_size)
    # and sum over the two tile axes — ~100× faster than a Python loop for large arrays.
    trimmed = defect_map[:ny_tiles * tile_size, :nx_tiles * tile_size]
    if defect_mask is None:
        bad = (trimmed > 0).astype(np.uint8)
    else:
        bad = ((trimmed & int(defect_mask)) != 0).view(np.uint8)

    density = (
        bad.reshape(ny_tiles, tile_size, nx_tiles, tile_size)
           .sum(axis=(1, 3))
    )

    return {
        'tile_size': tile_size,
        'ny_tiles': int(ny_tiles),
        'nx_tiles': int(nx_tiles),
        'counts': density.tolist(),
    }

def create_detection_report(
    bpm: np.ndarray,
    stats: Dict,
    output_path: Optional[str] = None,
    show_plots: bool = True
) -> None:
    """
    Create comprehensive detection report with plots.
    
    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask from detection pipeline.
    stats : dict
        Statistics dict from create_calibration_bpm().
    output_path : str, optional
        If provided, save report to this path (PDF or HTML).
    show_plots : bool
        If True, display plots interactively.
    
    Examples
    --------
    >>> bpm, stats = create_calibration_bpm(bias, dark, flat)
    >>> create_detection_report(bpm, stats, output_path='bpm_report.pdf')
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    if output_path:
        pdf = PdfPages(output_path)
    
    # Figure 1: Defect map overview
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Full BPM
    ax = axes[0, 0]
    im = ax.imshow(bpm > 0, cmap='gray', origin='lower')
    ax.set_title(f'Bad Pixel Map\n({stats["total_defects"]} defects, '
                 f'{stats["total_fraction"]*100:.2f}%)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Bad (white)')
    
    # Defect density map
    ax = axes[0, 1]
    density = create_defect_density_map(bpm, tile_size=64)
    if density['counts'] is not None:
        im = ax.imshow(density['counts'], cmap='hot', origin='lower')
        ax.set_title('Defect Density Map\n(64×64 pixel tiles)')
        plt.colorbar(im, ax=ax, label='Defects per tile')
    
    # Row profile
    ax = axes[1, 0]
    row_counts = np.sum(bpm > 0, axis=1)
    ax.plot(row_counts, 'b-', linewidth=0.5)
    ax.set_title('Row Defect Profile')
    ax.set_xlabel('Row')
    ax.set_ylabel('Number of bad pixels')
    ax.grid(True, alpha=0.3)
    
    # Column profile
    ax = axes[1, 1]
    col_counts = np.sum(bpm > 0, axis=0)
    ax.plot(col_counts, 'r-', linewidth=0.5)
    ax.set_title('Column Defect Profile')
    ax.set_xlabel('Column')
    ax.set_ylabel('Number of bad pixels')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        pdf.savefig(fig)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    # Figure 2: Defect type breakdown
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    defect_types = [
        (BPM_BIT_PIX_BRIGHT, 'Bright', 'Blues'),
        (BPM_BIT_PIX_HOT, 'Hot', 'Reds'),
        (BPM_BIT_PIX_QE_HIGH, 'Flat Hot', 'Oranges'),
        (BPM_BIT_PIX_QE_LOW, 'Flat Cold', 'Purples'),
        (BPM_BIT_ROW_BAD, 'Bad Rows', 'Greens'),
        (BPM_BIT_COL_BAD, 'Bad Cols', 'YlOrBr'),
    ]
    
    for idx, (bit, name, cmap) in enumerate(defect_types):
        ax = axes.flat[idx]
        mask = (bpm & bit) != 0
        im = ax.imshow(mask, cmap=cmap, origin='lower', vmin=0, vmax=1)
        count = np.sum(mask)
        frac = count / bpm.size
        ax.set_title(f'{name}\n({count} pixels, {frac*100:.3f}%)')
        ax.axis('off')
    
    plt.tight_layout()
    if output_path:
        pdf.savefig(fig)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    # Figure 3: Statistics summary
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Create statistics table
    summary_text = f"""
DETECTION STATISTICS SUMMARY
{'=' * 60}

BIAS DEFECTS
  Method: {stats['bias']['method']}
  Median: {stats['bias']['median']:.2f} ADU
  Std: {stats['bias']['std']:.2f} ADU
  Defects: {stats['bias']['num_defects']} ({stats['bias']['fraction_defects']*100:.3f}%)

DARK HOT PIXELS
  Threshold: {stats['dark']['threshold_sigma']:.1f}σ
  Median: {stats['dark']['median']:.2f} ADU
  Std: {stats['dark']['std']:.2f} ADU
  Defects: {stats['dark']['num_defects']} ({stats['dark']['fraction_defects']*100:.3f}%)

FLAT FIELD DEFECTS
  Hot threshold: {stats['flat']['hot_threshold_percent']:.1f}%
  Cold threshold: {stats['flat']['cold_threshold_percent']:.1f}%
  Median: {stats['flat']['median']:.2f} ADU
  Hot pixels: {stats['flat']['num_hot']}
  Cold pixels: {stats['flat']['num_cold']}
  Total: {stats['flat']['num_defects']} ({stats['flat']['fraction_defects']*100:.3f}%)

STRUCTURAL DEFECTS
  Bad rows: {stats['rows_cols']['num_bad_rows']}
  Bad columns: {stats['rows_cols']['num_bad_cols']}

OVERALL SUMMARY
  Total defects: {stats['total_defects']} ({stats['total_fraction']*100:.3f}%)
  Good pixels: {bpm.size - stats['total_defects']} ({(1-stats['total_fraction'])*100:.3f}%)
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    if output_path:
        pdf.savefig(fig)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    if output_path:
        pdf.close()
        logger.info(f"Detection report saved to {output_path}")


def plot_defect_comparison(
    bpm_list: List[np.ndarray],
    labels: List[str],
    title: str = "Defect Map Comparison"
) -> None:
    """
    Compare multiple BPMs side-by-side.
    
    Useful for comparing different detection parameters
    or tracking detector degradation over time.
    
    Parameters
    ----------
    bpm_list : list of np.ndarray
        List of BPMs to compare.
    labels : list of str
        Label for each BPM.
    title : str
        Plot title.
    
    Examples
    --------
    >>> bpm1, _ = create_calibration_bpm(bias1, dark1, flat1)
    >>> bpm2, _ = create_calibration_bpm(bias2, dark2, flat2)
    >>> plot_defect_comparison(
    ...     [bpm1, bpm2],
    ...     ['2023-01-01', '2023-06-01'],
    ...     title='Detector Degradation'
    ... )
    """
    import matplotlib.pyplot as plt
    
    n = len(bpm_list)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, bpm, label in zip(axes, bpm_list, labels):
        im = ax.imshow(bpm > 0, cmap='gray', origin='lower')
        count = np.sum(bpm > 0)
        frac = count / bpm.size
        ax.set_title(f'{label}\n{count} defects ({frac*100:.2f}%)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# High-Level Pipeline Functions
# =============================================================================

# detection.py

def create_calibration_bpm_batch(
    bias_frames: np.ndarray,
    dark_frames: np.ndarray,
    flat_frames: np.ndarray,
    config: Optional[Dict] = None,
    stats_roi: Optional[Tuple[int, int, int, int]] = None,
    combine_method: str = 'median',
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create BPM from multiple calibration frames.
    
    Combines frames, detects defects, and returns both the master
    calibration frames and the BPM.
    
    Parameters
    ----------
    bias_frames : np.ndarray
        Stack of bias frames, shape (N, ny, nx).
    dark_frames : np.ndarray
        Stack of dark frames, shape (N, ny, nx).
    flat_frames : np.ndarray
        Stack of flat frames, shape (N, ny, nx).
    combine_method : str
        How to combine frames: 'median', 'mean', or 'sigma_clip'.
    
    Returns
    -------
    bpm : np.ndarray
        Bad pixel mask.
    masters : np.ndarray
        Combined master frames, shape (3, ny, nx) [bias, dark, flat].
    stats : dict
        Detection statistics.
    
    Examples
    --------
    >>> bias_stack = np.array([bias1, bias2, bias3, ...])  # (10, 2048, 2048)
    >>> dark_stack = np.array([dark1, dark2, dark3, ...])
    >>> flat_stack = np.array([flat1, flat2, flat3, ...])
    >>> bpm, masters, stats = create_calibration_bpm_batch(
    ...     bias_stack, dark_stack, flat_stack
    ... )
    >>> master_bias, master_dark, master_flat = masters
    """
    if combine_method == 'median':
        master_bias = np.median(bias_frames, axis=0)
        master_dark = np.median(dark_frames, axis=0)
        master_flat = np.median(flat_frames, axis=0)
    elif combine_method == 'mean':
        master_bias = np.mean(bias_frames, axis=0)
        master_dark = np.mean(dark_frames, axis=0)
        master_flat = np.mean(flat_frames, axis=0)
    elif combine_method == 'sigma_clip':
        # Sigma-clipped mean for better outlier rejection
        master_bias, _ = _sigma_clip_combine(bias_frames)
        master_dark, _ = _sigma_clip_combine(dark_frames)
        master_flat, _ = _sigma_clip_combine(flat_frames)
    else:
        raise ValueError(
            f"combine_method must be 'median', 'mean', or 'sigma_clip'; "
            f"got {combine_method!r}"
        )
    
    # Detect temporal defects (RTS noise) before combining
    temporal_stats = {}
    if bias_frames.shape[0] >= 3:
        temporal_bpm_bias, temporal_stats['bias'] = detect_temporal_defects(
            bias_frames, threshold_sigma=5.0
        )
    else:
        temporal_bpm_bias = create_bpm(bias_frames.shape[1:], dtype=bpm_dtype)
        temporal_stats['bias'] = {}
    
    if dark_frames.shape[0] >= 3:
        temporal_bpm_dark, temporal_stats['dark'] = detect_temporal_defects(
            dark_frames, threshold_sigma=5.0, existing_bpm=temporal_bpm_bias
        )
    else:
        temporal_bpm_dark = create_bpm(dark_frames.shape[1:], dtype=bpm_dtype)
        temporal_stats['dark'] = {}
    
    # Now detect spatial defects from master frames
    bpm, spatial_stats = create_calibration_bpm(
        master_bias, master_dark, master_flat,
        config=config,
        stats_roi=stats_roi,
        bpm_dtype=bpm_dtype
    )
    
    # Merge temporal defects
    bpm |= temporal_bpm_bias
    bpm |= temporal_bpm_dark
    
    # Stack master frames for output
    masters = np.stack([master_bias, master_dark, master_flat], axis=0)
    
    # Combine statistics
    all_stats = {
        'spatial': spatial_stats,
        'temporal': temporal_stats,
        'combine_method': combine_method,
        'n_bias_frames': int(bias_frames.shape[0]),
        'n_dark_frames': int(dark_frames.shape[0]),
        'n_flat_frames': int(flat_frames.shape[0]),
    }
    
    return bpm, masters, all_stats

def create_calibration_bpm(
    bias_frame: np.ndarray,
    dark_frame: np.ndarray,
    flat_frame: np.ndarray,
    config: Optional[Dict] = None,
    stats_roi: Optional[Tuple[int, int, int, int]] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Create comprehensive BPM from calibration frames.
    
    This high-level function runs all detection algorithms in the proper
    order, accumulating defects from each stage while excluding previously
    identified defects from statistics.

    Detection stages
    ----------------
    1. **Bias** — flags pixels with anomalous offset levels (``BPM_BIT_PIX_BRIGHT``).
    2. **Dark** — flags hot pixels with excess dark current, high-side only
       (``BPM_BIT_PIX_HOT``). Cold/dead pixels are intentionally deferred to
       the flat stage, where quantum efficiency is measured directly.
    3. **Flat** — flags pixels with high (``BPM_BIT_PIX_QE_HIGH``) or low
       (``BPM_BIT_PIX_QE_LOW``) response, including cold and dead pixels.
    4. **Rows/columns** — flags structural defects accumulated across all
       prior stages (``BPM_BIT_ROW_BAD``, ``BPM_BIT_COL_BAD``).

    Parameters
    ----------
    bias_frame : np.ndarray
        Master bias frame (ADU).
    dark_frame : np.ndarray
        Master dark frame, bias-subtracted (ADU or e-).
    flat_frame : np.ndarray
        Master flat frame, bias and dark subtracted (ADU or e-).
    config : dict, optional
        Detection configuration. If None, uses defaults.
    stats_roi : tuple, optional
        ROI for statistics as (x1, x2, y1, y2).
    
    Returns
    -------
    bpm : np.ndarray
        Comprehensive bad pixel mask with all defect types.
    stats : dict
        Detection statistics from all stages.
    
    Examples
    --------
    >>> bpm, stats = create_calibration_bpm(
    ...     master_bias,
    ...     master_dark,
    ...     master_flat,
    ...     stats_roi=(100, 1900, 100, 1900)
    ... )
    >>> print(f"Total defects: {stats['total_defects']}")
    """
    if config is None:
        config = get_default_detection_config()

    # Validate that all input frames share the same shape before running any
    # detection stage — a mismatch here would otherwise produce a cryptic
    # NumPy broadcast error deep inside one of the detection functions.
    if not (bias_frame.shape == dark_frame.shape == flat_frame.shape):
        raise ValueError(
            f"All calibration frames must have the same shape. "
            f"Got bias={bias_frame.shape}, dark={dark_frame.shape}, flat={flat_frame.shape}."
        )

    # Validate dtypes are numeric
    for name, frame in [('bias', bias_frame), ('dark', dark_frame), ('flat', flat_frame)]:
        if not np.issubdtype(frame.dtype, np.number):
            raise TypeError(
                f"{name}_frame must have numeric dtype; got {frame.dtype}"
            )
    
    # Validate ROI if provided
    if stats_roi is not None:
        _validate_roi(bias_frame.shape, stats_roi)
    
    all_stats = {}
    
    # Stage 1: Bias defects (no exclusions)
    bias_bpm, bias_stats = detect_bias_defects(
        bias_frame,
        stats_roi=stats_roi,
        bpm_dtype=bpm_dtype,
        **config['bias']
    )
    all_stats['bias'] = bias_stats
    cumulative_bpm = bias_bpm.copy()

    # Stage 2: Dark hot pixels (exclude bias defects from stats)
    dark_bpm, dark_stats = detect_dark_hot_pixels(
        dark_frame,
        stats_roi=stats_roi,
        existing_bpm=cumulative_bpm,
        bpm_dtype=bpm_dtype,
        **config['dark']
    )
    all_stats['dark'] = dark_stats
    cumulative_bpm |= dark_bpm

    # Stage 3: Flat field defects (exclude bias + dark from stats)
    flat_bpm, flat_stats = detect_flat_defects(
        flat_frame,
        stats_roi=stats_roi,
        existing_bpm=cumulative_bpm,
        bpm_dtype=bpm_dtype,
        **config['flat']
    )
    all_stats['flat'] = flat_stats
    cumulative_bpm |= flat_bpm

    # Stage 4: Bad rows/columns (from accumulated defects)
    rowcol_bpm, rowcol_stats = detect_bad_rows_cols(
        cumulative_bpm,
        bpm_dtype=bpm_dtype,
        **config['rows_cols']
    )
    all_stats['rows_cols'] = rowcol_stats
    cumulative_bpm |= rowcol_bpm
    
    # Overall statistics
    all_stats['total_defects'] = int(np.sum(cumulative_bpm > 0))
    all_stats['total_fraction'] = all_stats['total_defects'] / cumulative_bpm.size
    
    return cumulative_bpm, all_stats

def get_default_detection_config() -> Dict:
    """
    Get default detection configuration.
    
    Returns
    -------
    dict
        Default parameters for all detection functions.
    
    Examples
    --------
    >>> config = get_default_detection_config()
    >>> config['bias']['threshold_adu'] = 150.0  # Customize
    >>> bpm, stats = create_calibration_bpm(bias, dark, flat, config=config)
    """
    return {
        'bias': {
            'threshold_adu': 100.0,
            'method': 'absolute',
            'sigma': 5.0,
            'sigma_low': 3.0,
            'sigma_high': 3.0,
            'max_iters': 5,
        },
        'dark': {
            'threshold_sigma': 5.0,
        },
        'flat': {
            'hot_threshold_percent': 20.0,
            'cold_threshold_percent': 20.0,
            'normalize': True,
        },
        'rows_cols': {
            'row_threshold': 0.1,
            'col_threshold': 0.1,
        },
    }

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Statistical utilities
    'sigma_clip_stats',
    'roi_stats',
    
    # Detection functions
    'detect_bias_defects',
    'detect_dark_hot_pixels',
    'detect_flat_defects',
    'detect_bad_rows_cols',
    
    # Utilities
    'create_defect_density_map',
    
    # High-level pipeline
    'create_calibration_bpm',
    'get_default_detection_config',
]
