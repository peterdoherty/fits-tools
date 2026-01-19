#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality control and metrics for CCD/CMOS calibrated images.

This module provides comprehensive quality assessment by integrating validation
(validate.py) with statistical analysis (statistics.py) to evaluate the quality
of calibrated images and calibration workflows.

Features:
    - Comprehensive quality metrics combining validation and statistics
    - Quality grading system (EXCELLENT, GOOD, ACCEPTABLE, POOR, FAIL)
    - Before/after calibration comparison
    - Automated quality assessment with configurable thresholds
    - Batch quality assessment for multiple images
    - Quality report generation
    - Integration with image_processing calibration workflow

Quality Assessment Criteria:
    - File integrity (from validate.py)
    - Frame type validation (from validate.py)
    - Statistical properties (from statistics.py)
    - Signal-to-noise ratio (from statistics.py)
    - Dynamic range (from statistics.py)
    - Outlier detection (sigma-clipping)
    - Saturation detection
    - Negative pixel detection
    - Read noise estimation

Usage Examples:
    Basic quality assessment:
        >>> from fits_tools.quality_control import assess_quality
        >>> qc = assess_quality('calibrated.fits', frame_type='LIGHT')
        >>> print(qc.quality_grade)
        >>> if qc.passed:
        >>>     print("Image passed quality control")
    
    Compare raw vs calibrated:
        >>> from fits_tools.quality_control import assess_calibration_quality
        >>> comparison = assess_calibration_quality(
        ...     calibrated_file='calibrated.fits',
        ...     raw_file='raw.fits'
        ... )
        >>> print(f"SNR improvement: {comparison['snr_improvement']:.2f}x")
    
    Batch assessment:
        >>> from fits_tools.quality_control import batch_quality_assessment
        >>> results = batch_quality_assessment(['img1.fits', 'img2.fits', 'img3.fits'])
        >>> print(f"Pass rate: {results['pass_rate']*100:.1f}%")

Version History:
    1.0.0 - Initial release integrating validate.py and statistics.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

import numpy as np
from astropy.io import fits

# Import from fits_tools modules
from .validate import (
    ValidationResult,
    validate_fits_file,
    validate_frame_type,
    validate_bias_frame,
    validate_dark_frame,
    validate_flat_frame,
    validate_no_nans,
    quick_validate,
)

from .statistics import (
    stats,
    robust_stats,
    calculate_snr,
    calculate_dynamic_range,
    estimate_read_noise,
    StatsResult,
    RobustStatsResult,
)

from .io import get_data, get_header
from .constants import (
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
    normalize_image_type,
)
from .exceptions import ImageProcessingError, FileHandlingError

logger = logging.getLogger(__name__)
__version__ = "1.0.0"
__author__ = "FITS Tools Development Team"


# ============================================================================
# Quality Metrics Dataclass
# ============================================================================

@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for calibrated images.
    
    Combines validation results with statistical analysis to provide
    a complete quality assessment suitable for automated processing pipelines.
    
    Attributes
    ----------
    filepath : str
        Path to the FITS file assessed.
    frame_type : str
        Detected or specified frame type (BIAS, DARK, FLAT, LIGHT).
    
    Basic Statistics:
        mean : float
            Mean pixel value.
        median : float
            Median pixel value.
        std : float
            Standard deviation.
        mad : float
            Median Absolute Deviation (robust std estimate).
    
    Range Metrics:
        min_val : float
            Minimum pixel value.
        max_val : float
            Maximum pixel value.
        dynamic_range : float, optional
            Dynamic range (max/min ratio).
    
    Signal Quality:
        snr : float, optional
            Signal-to-noise ratio.
        read_noise : float, optional
            Estimated read noise (for bias frames).
    
    Outlier Detection:
        num_outliers : int
            Number of outlier pixels detected (sigma-clipping).
        outlier_fraction : float
            Fraction of outlier pixels.
        num_negative_pixels : int
            Number of negative pixels (unphysical for calibrated data).
        num_saturated_pixels : int
            Number of saturated pixels.
        saturation_fraction : float
            Fraction of saturated pixels.
    
    Quality Assessment:
        validation : ValidationResult
            Detailed validation results from validate.py.
        passed : bool
            Overall pass/fail status.
        quality_grade : str
            Quality grade: EXCELLENT, GOOD, ACCEPTABLE, POOR, or FAIL.
        quality_score : float
            Numerical quality score (0-100).
        warnings : list of str
            List of warning messages.
        errors : list of str
            List of error messages.
    
    Metadata:
        timestamp : str
            Timestamp of assessment.
        image_shape : tuple
            Image dimensions (rows, cols).
        total_pixels : int
            Total number of pixels.
    
    Examples
    --------
    >>> qc = QualityMetrics(
    ...     filepath='test.fits',
    ...     frame_type='LIGHT',
    ...     mean=1000.0,
    ...     median=998.0,
    ...     std=50.0,
    ...     mad=48.0,
    ...     min_val=0.0,
    ...     max_val=65000.0
    ... )
    >>> print(qc.quality_grade)
    >>> print(qc)
    """
    # Identification
    filepath: str
    frame_type: str = "UNKNOWN"
    
    # Basic statistics
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    mad: float = 0.0
    
    # Range metrics
    min_val: float = 0.0
    max_val: float = 0.0
    dynamic_range: Optional[float] = None
    
    # Signal quality
    snr: Optional[float] = None
    read_noise: Optional[float] = None
    
    # Outlier detection
    num_outliers: int = 0
    outlier_fraction: float = 0.0
    num_negative_pixels: int = 0
    num_saturated_pixels: int = 0
    saturation_fraction: float = 0.0
    
    # Quality assessment
    validation: Optional[ValidationResult] = None
    passed: bool = True
    quality_grade: str = "UNKNOWN"
    quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    image_shape: Optional[Tuple[int, int]] = None
    total_pixels: int = 0
    
    def __post_init__(self):
        """Calculate derived quantities."""
        if self.total_pixels > 0:
            self.outlier_fraction = self.num_outliers / self.total_pixels
            self.saturation_fraction = self.num_saturated_pixels / self.total_pixels
    
    def __str__(self) -> str:
        """Human-readable quality report."""
        status_symbol = "✓" if self.passed else "✗"
        
        lines = [
            "="*70,
            f"Quality Assessment Report: {status_symbol} {self.quality_grade}",
            "="*70,
            f"File: {os.path.basename(self.filepath)}",
            f"Type: {self.frame_type}",
            f"Score: {self.quality_score:.1f}/100",
            f"Timestamp: {self.timestamp}",
            "",
        ]
        
        if self.image_shape:
            lines.append(f"Image: {self.image_shape[0]}×{self.image_shape[1]} ({self.total_pixels:,} pixels)")
            lines.append("")
        
        lines.extend([
            "Statistics:",
            f"  Mean:   {self.mean:>10.2f}",
            f"  Median: {self.median:>10.2f}",
            f"  Std:    {self.std:>10.2f}",
            f"  MAD:    {self.mad:>10.2f}",
            "",
            "Range:",
            f"  Min:    {self.min_val:>10.2f}",
            f"  Max:    {self.max_val:>10.2f}",
        ])
        
        if self.dynamic_range is not None:
            lines.append(f"  Dynamic Range: {self.dynamic_range:>6.1f}")
        
        if self.snr is not None or self.read_noise is not None:
            lines.append("")
            lines.append("Signal Quality:")
            if self.snr is not None:
                lines.append(f"  SNR:        {self.snr:>10.2f}")
            if self.read_noise is not None:
                lines.append(f"  Read Noise: {self.read_noise:>10.2f} e⁻")
        
        # Outliers section
        has_issues = (self.num_outliers > 0 or 
                     self.num_negative_pixels > 0 or 
                     self.num_saturated_pixels > 0)
        
        if has_issues:
            lines.append("")
            lines.append("Outliers & Issues:")
            if self.num_outliers > 0:
                lines.append(f"  Outliers:   {self.num_outliers:>6d} ({self.outlier_fraction*100:>5.2f}%)")
            if self.num_negative_pixels > 0:
                lines.append(f"  Negative:   {self.num_negative_pixels:>6d}")
            if self.num_saturated_pixels > 0:
                lines.append(f"  Saturated:  {self.num_saturated_pixels:>6d} ({self.saturation_fraction*100:>5.2f}%)")
        
        if self.errors:
            lines.append("")
            lines.append(f"Errors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
        
        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
        
        lines.append("="*70)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Brief representation."""
        return (
            f"QualityMetrics(file='{os.path.basename(self.filepath)}', "
            f"grade='{self.quality_grade}', score={self.quality_score:.1f}, "
            f"passed={self.passed})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert ValidationResult to dict if present
        if self.validation:
            data['validation'] = {
                'is_valid': self.validation.is_valid,
                'errors': self.validation.errors,
                'warnings': self.validation.warnings,
                'info': self.validation.info,
            }
        return data
    
    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """
        Export to JSON format.
        
        Parameters
        ----------
        filepath : str, optional
            If provided, write to file. Otherwise return JSON string.
        indent : int, optional
            JSON indentation. Defaults to 2.
        
        Returns
        -------
        str
            JSON string representation.
        """
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Quality metrics exported to: {filepath}")
        
        return json_str


# ============================================================================
# Quality Assessment Configuration
# ============================================================================

@dataclass
class QualityThresholds:
    """
    Configurable thresholds for quality assessment.
    
    Allows customization of quality criteria for different detectors
    and observing conditions.
    
    Attributes
    ----------
    min_snr : float
        Minimum acceptable SNR. Defaults to 5.0.
    max_outlier_fraction : float
        Maximum acceptable fraction of outliers. Defaults to 0.01 (1%).
    max_saturation_fraction : float
        Maximum acceptable fraction of saturated pixels. Defaults to 0.05 (5%).
    saturation_threshold : float
        Pixel value above which is considered saturated. Defaults to 60000.
    max_negative_fraction : float
        Maximum acceptable fraction of negative pixels. Defaults to 0.001 (0.1%).
    min_dynamic_range : float
        Minimum acceptable dynamic range. Defaults to 10.0.
    
    Quality Score Weights:
        weight_validation : float
            Weight for validation status (0-1). Defaults to 0.3.
        weight_snr : float
            Weight for SNR (0-1). Defaults to 0.2.
        weight_outliers : float
            Weight for outlier fraction (0-1). Defaults to 0.2.
        weight_statistics : float
            Weight for statistical properties (0-1). Defaults to 0.3.
    
    Examples
    --------
    >>> # Custom thresholds for high-quality data
    >>> thresholds = QualityThresholds(
    ...     min_snr=10.0,
    ...     max_outlier_fraction=0.005,
    ...     saturation_threshold=50000
    ... )
    """
    # Thresholds
    min_snr: float = 5.0
    max_outlier_fraction: float = 0.01
    max_saturation_fraction: float = 0.05
    saturation_threshold: float = 60000.0
    max_negative_fraction: float = 0.001
    min_dynamic_range: float = 10.0
    
    # Quality score weights (should sum to 1.0)
    weight_validation: float = 0.3
    weight_snr: float = 0.2
    weight_outliers: float = 0.2
    weight_statistics: float = 0.3
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total_weight = (self.weight_validation + self.weight_snr + 
                       self.weight_outliers + self.weight_statistics)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Quality score weights sum to {total_weight:.3f}, not 1.0. "
                f"Normalizing weights."
            )
            # Normalize
            self.weight_validation /= total_weight
            self.weight_snr /= total_weight
            self.weight_outliers /= total_weight
            self.weight_statistics /= total_weight


# Default thresholds
DEFAULT_THRESHOLDS = QualityThresholds()


# ============================================================================
# Core Quality Assessment Functions
# ============================================================================

def assess_quality(
    filepath: str,
    ext: int = 0,
    frame_type: Optional[str] = None,
    thresholds: Optional[QualityThresholds] = None,
    region: str = 'full',
    compute_snr: bool = True,
    verbose: bool = False
) -> QualityMetrics:
    """
    Perform comprehensive quality assessment of a FITS image.
    
    Integrates validation from validate.py with statistical analysis from
    statistics.py to provide a complete quality evaluation.
    
    Parameters
    ----------
    filepath : str
        Path to FITS file.
    ext : int, optional
        HDU extension. Defaults to 0.
    frame_type : str, optional
        Expected frame type ('BIAS', 'DARK', 'FLAT', 'LIGHT').
        If None, will attempt to detect from header.
    thresholds : QualityThresholds, optional
        Custom quality thresholds. If None, uses defaults.
    region : str, optional
        Region to analyze ('full', 'data', 'bias', 'trim'). Defaults to 'full'.
    compute_snr : bool, optional
        Whether to compute SNR (may be slow for large images). Defaults to True.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    QualityMetrics
        Comprehensive quality assessment.
    
    Raises
    ------
    FileHandlingError
        If file cannot be opened or read.
    ImageProcessingError
        If quality assessment fails.
    
    Examples
    --------
    >>> from fits_tools.quality_control import assess_quality
    >>> qc = assess_quality('calibrated.fits', frame_type='LIGHT', verbose=True)
    >>> if qc.passed:
    ...     print(f"Quality grade: {qc.quality_grade}")
    ...     print(f"SNR: {qc.snr:.2f}")
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    if verbose:
        logger.info(f"Assessing quality of: {filepath}")
    
    # Initialize metrics
    metrics = QualityMetrics(filepath=filepath)
    
    try:
        # Step 1: Basic file validation
        validation = quick_validate(filepath, frame_type=frame_type, verbose=verbose)
        metrics.validation = validation
        
        # Extract frame type from validation
        if 'frame_type' in validation.info:
            metrics.frame_type = validation.info['frame_type']
        elif frame_type:
            metrics.frame_type = normalize_image_type(frame_type)
        
        # Collect errors and warnings
        metrics.errors.extend(validation.errors)
        metrics.warnings.extend(validation.warnings)
        
        if not validation.is_valid:
            metrics.passed = False
            if verbose:
                logger.warning(f"Validation failed with {len(validation.errors)} errors")
        
        # Step 2: Get image dimensions
        data = get_data(filepath, ext=ext, region=region, verbose=False)
        if data is None:
            raise ImageProcessingError(f"Could not load data from {filepath}")
        
        metrics.image_shape = data.shape
        metrics.total_pixels = int(data.size)
        
        # Step 3: Basic statistics
        if verbose:
            logger.debug("Computing basic statistics...")
        
        basic_stats = stats(
            filepath, 
            ext=ext, 
            region=region, 
            stats_type='all', 
            skipna=True,
            verbose=False
        )
        
        metrics.mean = basic_stats.mean_val
        metrics.median = basic_stats.median_val
        metrics.std = basic_stats.std_val
        metrics.min_val = basic_stats.min_val
        metrics.max_val = basic_stats.max_val
        metrics.mad = basic_stats.mad if not np.isnan(basic_stats.mad) else basic_stats.std_val
        
        # Step 4: Robust statistics (for outlier detection)
        if verbose:
            logger.debug("Computing robust statistics...")
        
        try:
            robust = robust_stats(
                filepath,
                ext=ext,
                region=region,
                sigma=3.0,
                maxiters=5,
                verbose=False
            )
            
            metrics.num_outliers = robust.num_clipped
            metrics.outlier_fraction = robust.clip_fraction
            
            # Update MAD from robust stats if available
            if hasattr(robust, 'std'):
                metrics.mad = robust.std
        
        except Exception as e:
            if verbose:
                logger.warning(f"Could not compute robust statistics: {e}")
            metrics.warnings.append(f"Robust statistics unavailable: {e}")
        
        # Step 5: Signal quality metrics
        if compute_snr:
            if verbose:
                logger.debug("Computing signal quality metrics...")
            
            try:
                # SNR
                if metrics.frame_type == FITS_VAL_LIGHT_IMGTYPE:
                    metrics.snr = calculate_snr(
                        filepath,
                        signal_region=region,
                        ext=ext,
                        verbose=False
                    )
                
                # Dynamic range
                metrics.dynamic_range = calculate_dynamic_range(
                    filepath,
                    region=region,
                    ext=ext,
                    verbose=False
                )
            
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not compute signal quality: {e}")
                metrics.warnings.append(f"Signal quality metrics unavailable: {e}")
        
        # Step 6: Read noise (for bias frames)
        if metrics.frame_type == FITS_VAL_BIAS_IMGTYPE:
            try:
                # For single frame, read noise ≈ std
                metrics.read_noise = metrics.std
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not estimate read noise: {e}")
        
        # Step 7: Count negative and saturated pixels
        if verbose:
            logger.debug("Checking for outliers and saturation...")
        
        metrics.num_negative_pixels = int(np.sum(data < 0))
        metrics.num_saturated_pixels = int(np.sum(data > thresholds.saturation_threshold))
        
        # Update fractions
        if metrics.total_pixels > 0:
            metrics.outlier_fraction = metrics.num_outliers / metrics.total_pixels
            metrics.saturation_fraction = metrics.num_saturated_pixels / metrics.total_pixels
        
        # Step 8: Apply quality thresholds
        if verbose:
            logger.debug("Applying quality thresholds...")
        
        # Check SNR
        if metrics.snr is not None and metrics.snr < thresholds.min_snr:
            metrics.warnings.append(
                f"SNR ({metrics.snr:.2f}) below minimum ({thresholds.min_snr})"
            )
            metrics.passed = False
        
        # Check outliers
        if metrics.outlier_fraction > thresholds.max_outlier_fraction:
            metrics.warnings.append(
                f"Outlier fraction ({metrics.outlier_fraction*100:.2f}%) "
                f"exceeds maximum ({thresholds.max_outlier_fraction*100:.2f}%)"
            )
            metrics.passed = False
        
        # Check saturation
        if metrics.saturation_fraction > thresholds.max_saturation_fraction:
            metrics.warnings.append(
                f"Saturation fraction ({metrics.saturation_fraction*100:.2f}%) "
                f"exceeds maximum ({thresholds.max_saturation_fraction*100:.2f}%)"
            )
            metrics.passed = False
        
        # Check negative pixels
        negative_fraction = metrics.num_negative_pixels / metrics.total_pixels if metrics.total_pixels > 0 else 0
        if negative_fraction > thresholds.max_negative_fraction:
            metrics.warnings.append(
                f"Negative pixel fraction ({negative_fraction*100:.3f}%) "
                f"exceeds maximum ({thresholds.max_negative_fraction*100:.3f}%)"
            )
            metrics.passed = False
        
        # Check dynamic range
        if metrics.dynamic_range is not None and metrics.dynamic_range < thresholds.min_dynamic_range:
            metrics.warnings.append(
                f"Dynamic range ({metrics.dynamic_range:.1f}) "
                f"below minimum ({thresholds.min_dynamic_range})"
            )
        
        # Step 9: Calculate quality score
        metrics.quality_score = _calculate_quality_score(metrics, thresholds)
        
        # Step 10: Assign quality grade
        metrics.quality_grade = _assign_quality_grade(metrics)
        
        if verbose:
            logger.info(f"Quality assessment complete: {metrics.quality_grade} (score: {metrics.quality_score:.1f})")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Quality assessment failed for {filepath}: {e}", exc_info=verbose)
        raise ImageProcessingError(f"Quality assessment failed: {e}")

def assess_quality_array(
    data: np.ndarray,
    header: fits.Header,
    filepath: str = "in-memory",
    frame_type: Optional[str] = None,
    thresholds: Optional[QualityThresholds] = None,
    compute_snr: bool = True,
    verbose: bool = False
) -> QualityMetrics:
    """
    Perform quality assessment on in-memory array.
    
    Similar to assess_quality() but works with numpy arrays instead of files.
    Useful for assessing calibrated data before writing to disk.
    
    Parameters
    ----------
    data : np.ndarray
        Image data array.
    header : fits.Header
        FITS header.
    filepath : str, optional
        Identifier for the data (e.g., filename or "in-memory"). Defaults to "in-memory".
    frame_type : str, optional
        Frame type ('BIAS', 'DARK', 'FLAT', 'LIGHT').
    thresholds : QualityThresholds, optional
        Custom quality thresholds.
    compute_snr : bool, optional
        Whether to compute SNR. Defaults to True.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    QualityMetrics
        Quality assessment results.
    
    Examples
    --------
    >>> data = np.random.normal(1000, 50, (512, 512))
    >>> header = fits.Header({'IMAGETYP': 'LIGHT', 'EXPTIME': 60.0})
    >>> qc = assess_quality_array(data, header)
    >>> print(qc.quality_grade)
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    if verbose:
        logger.info(f"Assessing quality of array: {filepath}")
    
    # Initialize metrics
    metrics = QualityMetrics(filepath=filepath)
    
    # Get frame type
    if frame_type:
        metrics.frame_type = normalize_image_type(frame_type)
    elif 'IMAGETYP' in header:
        metrics.frame_type = normalize_image_type(header['IMAGETYP'])
    
    # Image dimensions
    metrics.image_shape = data.shape
    metrics.total_pixels = int(data.size)
    
    # Basic statistics
    if verbose:
        logger.debug("Computing basic statistics...")
    
    metrics.mean = float(np.mean(data))
    metrics.median = float(np.median(data))
    metrics.std = float(np.std(data))
    metrics.min_val = float(np.min(data))
    metrics.max_val = float(np.max(data))
    
    # MAD (robust std estimate)
    from astropy.stats import mad_std
    metrics.mad = float(mad_std(data, ignore_nan=True))
    
    # Robust statistics (outlier detection)
    if verbose:
        logger.debug("Computing robust statistics...")
    
    try:
        from astropy.stats import sigma_clip
        clipped = sigma_clip(data, sigma=3.0, maxiters=5)
        if hasattr(clipped, 'mask'):
            metrics.num_outliers = int(np.sum(clipped.mask))
            metrics.outlier_fraction = metrics.num_outliers / metrics.total_pixels
    except Exception as e:
        if verbose:
            logger.warning(f"Could not compute robust statistics: {e}")
    
    # Signal quality metrics
    if compute_snr:
        if verbose:
            logger.debug("Computing signal quality metrics...")
        
        try:
            # SNR for LIGHT frames
            if metrics.frame_type == FITS_VAL_LIGHT_IMGTYPE:
                # Simple SNR estimate: mean / std
                if metrics.std > 0:
                    metrics.snr = metrics.mean / metrics.std
            
            # Dynamic range
            if metrics.min_val > 0:
                metrics.dynamic_range = metrics.max_val / metrics.min_val
            
            # Read noise for BIAS frames
            if metrics.frame_type == FITS_VAL_BIAS_IMGTYPE:
                metrics.read_noise = metrics.std
        
        except Exception as e:
            if verbose:
                logger.warning(f"Could not compute signal quality: {e}")
    
    # Count negative and saturated pixels
    if verbose:
        logger.debug("Checking for outliers and saturation...")
    
    metrics.num_negative_pixels = int(np.sum(data < 0))
    metrics.num_saturated_pixels = int(np.sum(data > thresholds.saturation_threshold))
    
    # Update fractions
    metrics.outlier_fraction = metrics.num_outliers / metrics.total_pixels
    metrics.saturation_fraction = metrics.num_saturated_pixels / metrics.total_pixels
    
    # Apply quality thresholds
    metrics.passed = True
    
    # Check SNR
    if metrics.snr is not None and metrics.snr < thresholds.min_snr:
        metrics.warnings.append(
            f"SNR ({metrics.snr:.2f}) below minimum ({thresholds.min_snr})"
        )
        metrics.passed = False
    
    # Check outliers
    if metrics.outlier_fraction > thresholds.max_outlier_fraction:
        metrics.warnings.append(
            f"Outlier fraction ({metrics.outlier_fraction*100:.2f}%) "
            f"exceeds maximum ({thresholds.max_outlier_fraction*100:.2f}%)"
        )
        metrics.passed = False
    
    # Check saturation
    if metrics.saturation_fraction > thresholds.max_saturation_fraction:
        metrics.warnings.append(
            f"Saturation fraction ({metrics.saturation_fraction*100:.2f}%) "
            f"exceeds maximum ({thresholds.max_saturation_fraction*100:.2f}%)"
        )
        metrics.passed = False
    
    # Check negative pixels
    negative_fraction = metrics.num_negative_pixels / metrics.total_pixels
    if negative_fraction > thresholds.max_negative_fraction:
        metrics.warnings.append(
            f"Negative pixel fraction ({negative_fraction*100:.3f}%) "
            f"exceeds maximum ({thresholds.max_negative_fraction*100:.3f}%)"
        )
        metrics.passed = False
    
    # Calculate quality score and grade
    metrics.quality_score = _calculate_quality_score(metrics, thresholds)
    metrics.quality_grade = _assign_quality_grade(metrics)
    
    if verbose:
        logger.info(
            f"Quality assessment complete: {metrics.quality_grade} "
            f"(score: {metrics.quality_score:.1f})"
        )
    
    return metrics


def _calculate_quality_score(
    metrics: QualityMetrics,
    thresholds: QualityThresholds
) -> float:
    """
    Calculate numerical quality score (0-100).
    
    Uses weighted combination of validation, SNR, outliers, and statistics.
    """
    score = 0.0
    
    # Validation score (0 or 100)
    validation_score = 100.0 if (metrics.validation and metrics.validation.is_valid) else 0.0
    score += validation_score * thresholds.weight_validation
    
    # SNR score (0-100, capped at 2x minimum)
    if metrics.snr is not None and thresholds.min_snr > 0:
        snr_score = min(100.0, (metrics.snr / thresholds.min_snr) * 50.0)
        score += snr_score * thresholds.weight_snr
    else:
        # If SNR not available, assume perfect
        score += 100.0 * thresholds.weight_snr
    
    # Outlier score (100 - penalty for outliers)
    if thresholds.max_outlier_fraction > 0:
        outlier_penalty = (metrics.outlier_fraction / thresholds.max_outlier_fraction) * 100.0
        outlier_score = max(0.0, 100.0 - outlier_penalty)
    else:
        outlier_score = 100.0
    score += outlier_score * thresholds.weight_outliers
    
    # Statistics score (based on MAD/median ratio, ideally < 0.1 for good data)
    if metrics.median > 0:
        cv = metrics.mad / metrics.median  # Coefficient of variation
        stats_score = max(0.0, 100.0 - (cv * 100.0))  # Penalize high CV
    else:
        stats_score = 50.0  # Neutral score if median is zero
    score += stats_score * thresholds.weight_statistics
    
    return min(100.0, max(0.0, score))


def _assign_quality_grade(metrics: QualityMetrics) -> str:
    """Assign quality grade based on score and pass/fail status."""
    if not metrics.passed:
        return "FAIL"
    
    if metrics.quality_score >= 90:
        return "EXCELLENT"
    elif metrics.quality_score >= 75:
        return "GOOD"
    elif metrics.quality_score >= 60:
        return "ACCEPTABLE"
    elif metrics.quality_score >= 40:
        return "POOR"
    else:
        return "FAIL"


# ============================================================================
# Calibration Quality Assessment
# ============================================================================

def assess_calibration_quality(
    calibrated_file: str,
    raw_file: Optional[str] = None,
    ext: int = 0,
    thresholds: Optional[QualityThresholds] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Assess quality improvement after calibration.
    
    Compares raw vs. calibrated image statistics to quantify the
    effectiveness of the calibration process.
    
    Parameters
    ----------
    calibrated_file : str
        Path to calibrated image.
    raw_file : str, optional
        Path to raw (uncalibrated) image for comparison.
        If None, only assesses calibrated image.
    ext : int, optional
        HDU extension. Defaults to 0.
    thresholds : QualityThresholds, optional
        Custom quality thresholds.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    dict
        Quality comparison with keys:
        - 'calibrated': QualityMetrics for calibrated image
        - 'passed': bool, overall pass status
        - 'quality_grade': str, quality grade
        - 'quality_score': float, quality score
        - 'raw_stats': dict (if raw_file provided)
        - 'improvement': dict (if raw_file provided)
    
    Examples
    --------
    >>> from fits_tools.quality_control import assess_calibration_quality
    >>> comparison = assess_calibration_quality(
    ...     calibrated_file='calibrated.fits',
    ...     raw_file='raw.fits',
    ...     verbose=True
    ... )
    >>> if comparison['improvement']['snr_improvement'] > 2.0:
    ...     print("Significant SNR improvement after calibration")
    """
    if verbose:
        logger.info("Assessing calibration quality...")
    
    # Assess calibrated image
    cal_metrics = assess_quality(
        calibrated_file,
        ext=ext,
        thresholds=thresholds,
        compute_snr=True,
        verbose=verbose
    )
    
    result = {
        'calibrated': cal_metrics,
        'passed': cal_metrics.passed,
        'quality_grade': cal_metrics.quality_grade,
        'quality_score': cal_metrics.quality_score,
    }
    
    # Compare with raw if provided
    if raw_file:
        if verbose:
            logger.info(f"Comparing with raw image: {raw_file}")
        
        try:
            raw_stats = stats(
                raw_file,
                ext=ext,
                region='full',
                stats_type='all',
                skipna=True,
                verbose=False
            )
            
            raw_robust = robust_stats(
                raw_file,
                ext=ext,
                region='full',
                sigma=3.0,
                verbose=False
            )
            
            result['raw_stats'] = {
                'mean': raw_stats.mean_val,
                'median': raw_stats.median_val,
                'std': raw_stats.std_val,
                'mad': raw_robust.std if hasattr(raw_robust, 'std') else raw_stats.std_val,
                'min': raw_stats.min_val,
                'max': raw_stats.max_val,
            }
            
            # Calculate improvements
            result['improvement'] = {}
            
            # Noise reduction (lower is better)
            if cal_metrics.std > 0 and raw_stats.std_val > 0:
                result['improvement']['noise_reduction'] = raw_stats.std_val / cal_metrics.std
            
            # SNR improvement (if available)
            try:
                raw_snr = calculate_snr(raw_file, ext=ext, verbose=False)
                if cal_metrics.snr and raw_snr:
                    result['improvement']['snr_improvement'] = cal_metrics.snr / raw_snr
            except Exception as e:
                if verbose:
                    logger.debug(f"Could not calculate SNR improvement: {e}")
            
            # Mean level change
            result['improvement']['mean_change'] = cal_metrics.mean - raw_stats.mean_val
            result['improvement']['mean_change_fraction'] = (
                (cal_metrics.mean - raw_stats.mean_val) / raw_stats.mean_val 
                if raw_stats.mean_val != 0 else 0
            )
            
            # Outlier reduction
            if raw_robust.num_clipped > 0:
                result['improvement']['outlier_reduction'] = (
                    raw_robust.num_clipped - cal_metrics.num_outliers
                )
            
            if verbose:
                logger.info("Calibration improvements:")
                for key, value in result['improvement'].items():
                    logger.info(f"  {key}: {value:.3f}")
        
        except Exception as e:
            logger.warning(f"Could not compare with raw image: {e}")
            result['raw_comparison_error'] = str(e)
    
    return result


# ============================================================================
# Batch Quality Assessment
# ============================================================================

def batch_quality_assessment(
    file_list: List[str],
    ext: int = 0,
    frame_type: Optional[str] = None,
    thresholds: Optional[QualityThresholds] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform quality assessment on multiple FITS files.
    
    Useful for assessing a complete observation set or calibration library.
    
    Parameters
    ----------
    file_list : list of str
        List of FITS file paths to assess.
    ext : int, optional
        HDU extension. Defaults to 0.
    frame_type : str, optional
        Expected frame type for all files.
    thresholds : QualityThresholds, optional
        Custom quality thresholds.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    dict
        Batch assessment results with keys:
        - 'total_files': int, number of files assessed
        - 'passed': int, number that passed
        - 'failed': int, number that failed
        - 'pass_rate': float, fraction that passed
        - 'results': list of QualityMetrics
        - 'summary': dict with aggregate statistics
        - 'worst': QualityMetrics, lowest scoring file
        - 'best': QualityMetrics, highest scoring file
    
    Examples
    --------
    >>> from fits_tools.quality_control import batch_quality_assessment
    >>> files = ['img1.fits', 'img2.fits', 'img3.fits']
    >>> results = batch_quality_assessment(files, frame_type='LIGHT', verbose=True)
    >>> print(f"Pass rate: {results['pass_rate']*100:.1f}%")
    >>> print(f"Best quality: {results['best'].quality_grade}")
    """
    if not file_list:
        raise ValueError("Empty file list provided")
    
    if verbose:
        logger.info(f"Performing batch quality assessment on {len(file_list)} files...")
    
    results = []
    passed = 0
    failed = 0
    
    for i, filepath in enumerate(file_list, 1):
        if verbose:
            logger.info(f"[{i}/{len(file_list)}] Assessing: {os.path.basename(filepath)}")
        
        try:
            metrics = assess_quality(
                filepath,
                ext=ext,
                frame_type=frame_type,
                thresholds=thresholds,
                verbose=False
            )
            
            results.append(metrics)
            
            if metrics.passed:
                passed += 1
            else:
                failed += 1
                if verbose:
                    logger.warning(f"  Failed: {len(metrics.errors)} errors, {len(metrics.warnings)} warnings")
        
        except Exception as e:
            logger.error(f"Failed to assess {filepath}: {e}")
            failed += 1
    
    # Calculate aggregate statistics
    if results:
        quality_scores = [r.quality_score for r in results]
        snr_values = [r.snr for r in results if r.snr is not None]
        
        summary = {
            'mean_quality_score': float(np.mean(quality_scores)),
            'median_quality_score': float(np.median(quality_scores)),
            'std_quality_score': float(np.std(quality_scores)),
            'min_quality_score': float(np.min(quality_scores)),
            'max_quality_score': float(np.max(quality_scores)),
        }
        
        if snr_values:
            summary['mean_snr'] = float(np.mean(snr_values))
            summary['median_snr'] = float(np.median(snr_values))
            summary['min_snr'] = float(np.min(snr_values))
            summary['max_snr'] = float(np.max(snr_values))
        
        # Find best and worst
        best = max(results, key=lambda r: r.quality_score)
        worst = min(results, key=lambda r: r.quality_score)
    else:
        summary = {}
        best = None
        worst = None
    
    batch_results = {
        'total_files': len(file_list),
        'passed': passed,
        'failed': failed,
        'pass_rate': passed / len(file_list) if file_list else 0.0,
        'results': results,
        'summary': summary,
        'best': best,
        'worst': worst,
    }
    
    if verbose:
        logger.info("\nBatch Assessment Summary:")
        logger.info(f"  Total files: {batch_results['total_files']}")
        logger.info(f"  Passed: {passed} ({batch_results['pass_rate']*100:.1f}%)")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Mean quality score: {summary.get('mean_quality_score', 0):.1f}")
        if best:
            logger.info(f"  Best: {os.path.basename(best.filepath)} (score: {best.quality_score:.1f})")
        if worst:
            logger.info(f"  Worst: {os.path.basename(worst.filepath)} (score: {worst.quality_score:.1f})")
    
    return batch_results


def generate_quality_report(
    batch_results: Dict[str, Any],
    output_file: Optional[str] = None,
    format: str = 'text'
) -> str:
    """
    Generate a formatted quality report from batch assessment results.
    
    Parameters
    ----------
    batch_results : dict
        Results from batch_quality_assessment().
    output_file : str, optional
        If provided, write report to file.
    format : str, optional
        Report format: 'text', 'json', 'csv'. Defaults to 'text'.
    
    Returns
    -------
    str
        Formatted report.
    
    Examples
    --------
    >>> from fits_tools.quality_control import batch_quality_assessment, generate_quality_report
    >>> batch = batch_quality_assessment(['img1.fits', 'img2.fits'])
    >>> report = generate_quality_report(batch, output_file='report.txt')
    """
    if format == 'text':
        report = _generate_text_report(batch_results)
    elif format == 'json':
        report = _generate_json_report(batch_results)
    elif format == 'csv':
        report = _generate_csv_report(batch_results)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'text', 'json', or 'csv'.")
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Quality report written to: {output_file}")
    
    return report


def _generate_text_report(batch_results: Dict[str, Any]) -> str:
    """Generate text format report."""
    lines = [
        "="*80,
        "FITS QUALITY CONTROL BATCH REPORT",
        "="*80,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Summary:",
        f"  Total files assessed: {batch_results['total_files']}",
        f"  Passed: {batch_results['passed']} ({batch_results['pass_rate']*100:.1f}%)",
        f"  Failed: {batch_results['failed']}",
        "",
    ]
    
    if batch_results['summary']:
        lines.append("Aggregate Statistics:")
        for key, value in batch_results['summary'].items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        lines.append("")
    
    if batch_results['best']:
        lines.append("Best Quality Image:")
        lines.append(f"  File: {os.path.basename(batch_results['best'].filepath)}")
        lines.append(f"  Score: {batch_results['best'].quality_score:.1f}")
        lines.append(f"  Grade: {batch_results['best'].quality_grade}")
        lines.append("")
    
    if batch_results['worst']:
        lines.append("Lowest Quality Image:")
        lines.append(f"  File: {os.path.basename(batch_results['worst'].filepath)}")
        lines.append(f"  Score: {batch_results['worst'].quality_score:.1f}")
        lines.append(f"  Grade: {batch_results['worst'].quality_grade}")
        lines.append("")
    
    lines.append("="*80)
    lines.append("Individual Results:")
    lines.append("="*80)
    
    for i, result in enumerate(batch_results['results'], 1):
        status = "✓" if result.passed else "✗"
        lines.append(
            f"{i:3d}. {status} {os.path.basename(result.filepath):40s} "
            f"Grade: {result.quality_grade:10s} Score: {result.quality_score:5.1f}"
        )
    
    lines.append("="*80)
    
    return "\n".join(lines)


def _generate_json_report(batch_results: Dict[str, Any]) -> str:
    """Generate JSON format report."""
    # Convert QualityMetrics objects to dicts
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_files': batch_results['total_files'],
        'passed': batch_results['passed'],
        'failed': batch_results['failed'],
        'pass_rate': batch_results['pass_rate'],
        'summary': batch_results['summary'],
        'best': batch_results['best'].to_dict() if batch_results['best'] else None,
        'worst': batch_results['worst'].to_dict() if batch_results['worst'] else None,
        'results': [r.to_dict() for r in batch_results['results']],
    }
    
    return json.dumps(report_data, indent=2, default=str)


def _generate_csv_report(batch_results: Dict[str, Any]) -> str:
    """Generate CSV format report."""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Filename', 'Frame Type', 'Passed', 'Quality Grade', 'Quality Score',
        'Mean', 'Median', 'Std', 'MAD', 'Min', 'Max',
        'SNR', 'Dynamic Range', 'Outliers', 'Negative Pixels', 'Saturated Pixels',
        'Warnings', 'Errors'
    ])
    
    # Data rows
    for result in batch_results['results']:
        writer.writerow([
            os.path.basename(result.filepath),
            result.frame_type,
            result.passed,
            result.quality_grade,
            f"{result.quality_score:.2f}",
            f"{result.mean:.2f}",
            f"{result.median:.2f}",
            f"{result.std:.2f}",
            f"{result.mad:.2f}",
            f"{result.min_val:.2f}",
            f"{result.max_val:.2f}",
            f"{result.snr:.2f}" if result.snr else "",
            f"{result.dynamic_range:.2f}" if result.dynamic_range else "",
            result.num_outliers,
            result.num_negative_pixels,
            result.num_saturated_pixels,
            len(result.warnings),
            len(result.errors),
        ])
    
    return output.getvalue()


# ============================================================================
# Calibration Set Quality Assessment
# ============================================================================

def assess_calibration_set_quality(
    bias_files: Optional[List[str]] = None,
    dark_files: Optional[List[str]] = None,
    flat_files: Optional[List[str]] = None,
    ext: int = 0,
    thresholds: Optional[QualityThresholds] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Assess quality of a complete calibration frame set.
    
    Evaluates bias, dark, and flat frame sets for quality and consistency.
    
    Parameters
    ----------
    bias_files : list of str, optional
        Bias frame file paths.
    dark_files : list of str, optional
        Dark frame file paths.
    flat_files : list of str, optional
        Flat frame file paths.
    ext : int, optional
        HDU extension. Defaults to 0.
    thresholds : QualityThresholds, optional
        Custom quality thresholds.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    dict
        Calibration set quality results with keys for each frame type.
    
    Examples
    --------
    >>> from fits_tools.quality_control import assess_calibration_set_quality
    >>> results = assess_calibration_set_quality(
    ...     bias_files=['bias_001.fits', 'bias_002.fits'],
    ...     dark_files=['dark_001.fits', 'dark_002.fits'],
    ...     flat_files=['flat_001.fits', 'flat_002.fits']
    ... )
    >>> print(f"Overall pass: {results['overall_passed']}")
    """
    if verbose:
        logger.info("Assessing calibration set quality...")
    
    results = {
        'overall_passed': True,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Assess bias frames
    if bias_files:
        if verbose:
            logger.info(f"Assessing {len(bias_files)} bias frames...")
        
        bias_results = batch_quality_assessment(
            bias_files,
            ext=ext,
            frame_type='BIAS',
            thresholds=thresholds,
            verbose=verbose
        )
        
        results['bias'] = bias_results
        
        if bias_results['pass_rate'] < 1.0:
            results['overall_passed'] = False
        
        # Estimate read noise from bias set
        if len(bias_files) > 1:
            try:
                read_noise = estimate_read_noise(
                    bias_files,
                    region='full',
                    ext=ext,
                    verbose=verbose
                )
                results['estimated_read_noise'] = read_noise
                if verbose:
                    logger.info(f"Estimated read noise: {read_noise:.2f} e⁻")
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not estimate read noise: {e}")
    
    # Assess dark frames
    if dark_files:
        if verbose:
            logger.info(f"Assessing {len(dark_files)} dark frames...")
        
        dark_results = batch_quality_assessment(
            dark_files,
            ext=ext,
            frame_type='DARK',
            thresholds=thresholds,
            verbose=verbose
        )
        
        results['dark'] = dark_results
        
        if dark_results['pass_rate'] < 1.0:
            results['overall_passed'] = False
    
    # Assess flat frames
    if flat_files:
        if verbose:
            logger.info(f"Assessing {len(flat_files)} flat frames...")
        
        flat_results = batch_quality_assessment(
            flat_files,
            ext=ext,
            frame_type='FLAT',
            thresholds=thresholds,
            verbose=verbose
        )
        
        results['flat'] = flat_results
        
        if flat_results['pass_rate'] < 1.0:
            results['overall_passed'] = False
    
    if verbose:
        logger.info(f"Calibration set assessment complete: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    
    return results


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Data classes
    'QualityMetrics',
    'QualityThresholds',
    
    # Core functions
    'assess_quality',
    'assess_quality_array',  
    'assess_calibration_quality',
    
    # Batch processing
    'batch_quality_assessment',
    'generate_quality_report',
    'assess_calibration_set_quality',
    
    # Defaults
    'DEFAULT_THRESHOLDS',
]


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Command-line interface for quality control."""
    parser = argparse.ArgumentParser(
        description='FITS Quality Control and Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Assess single file
  %(prog)s --file calibrated.fits --type LIGHT
  
  # Assess multiple files
  %(prog)s --files *.fits --type LIGHT --report report.txt
  
  # Assess calibration set
  %(prog)s --bias bias_*.fits --dark dark_*.fits --flat flat_*.fits
  
  # Custom thresholds
  %(prog)s --file image.fits --min-snr 10.0 --max-outliers 0.005
        """
    )
    
    parser.add_argument('--file', help='Single FITS file to assess')
    parser.add_argument('--files', nargs='+', help='Multiple FITS files')
    parser.add_argument('--type', choices=['BIAS', 'DARK', 'FLAT', 'LIGHT'],
                       help='Expected frame type')
    parser.add_argument('--bias', nargs='+', help='Bias frames for set assessment')
    parser.add_argument('--dark', nargs='+', help='Dark frames for set assessment')
    parser.add_argument('--flat', nargs='+', help='Flat frames for set assessment')
    parser.add_argument('--report', help='Output report file')
    parser.add_argument('--format', choices=['text', 'json', 'csv'],
                       default='text', help='Report format (default: text)')
    parser.add_argument('--min-snr', type=float, help='Minimum SNR threshold')
    parser.add_argument('--max-outliers', type=float, help='Maximum outlier fraction')
    parser.add_argument('--saturation', type=float, help='Saturation threshold')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # Create custom thresholds if specified
    thresholds = QualityThresholds()
    if args.min_snr:
        thresholds.min_snr = args.min_snr
    if args.max_outliers:
        thresholds.max_outlier_fraction = args.max_outliers
    if args.saturation:
        thresholds.saturation_threshold = args.saturation
    
    try:
        # Single file assessment
        if args.file:
            metrics = assess_quality(
                args.file,
                frame_type=args.type,
                thresholds=thresholds,
                verbose=args.verbose
            )
            print(metrics)
            
            if args.report:
                metrics.to_json(args.report)
            
            return 0 if metrics.passed else 1
        
        # Batch assessment
        if args.files:
            results = batch_quality_assessment(
                args.files,
                frame_type=args.type,
                thresholds=thresholds,
                verbose=args.verbose
            )
            
            report = generate_quality_report(results, format=args.format)
            print(report)
            
            if args.report:
                with open(args.report, 'w') as f:
                    f.write(report)
            
            return 0 if results['pass_rate'] == 1.0 else 1
        
        # Calibration set assessment
        if args.bias or args.dark or args.flat:
            results = assess_calibration_set_quality(
                bias_files=args.bias,
                dark_files=args.dark,
                flat_files=args.flat,
                thresholds=thresholds,
                verbose=args.verbose
            )
            
            print("\nCalibration Set Quality Assessment")
            print("="*70)
            print(f"Overall: {'✓ PASSED' if results['overall_passed'] else '✗ FAILED'}")
            print()
            
            for frame_type in ['bias', 'dark', 'flat']:
                if frame_type in results:
                    data = results[frame_type]
                    print(f"{frame_type.upper()}:")
                    print(f"  Files: {data['total_files']}")
                    print(f"  Passed: {data['passed']} ({data['pass_rate']*100:.1f}%)")
                    print(f"  Mean score: {data['summary'].get('mean_quality_score', 0):.1f}")
                    print()
            
            if 'estimated_read_noise' in results:
                print(f"Estimated read noise: {results['estimated_read_noise']:.2f} e⁻")
            
            if args.report:
                with open(args.report, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            return 0 if results['overall_passed'] else 1
        
        # No arguments
        parser.print_help()
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# CLI Support:
# Single file
# python -m fits_tools.quality_control --file image.fits --type LIGHT
#
# Batch with report
# python -m fits_tools.quality_control --files *.fits --report qc_report.json --format json
#
# Calibration set
# python -m fits_tools.quality_control --bias bias_*.fits --dark dark_*.fits --flat flat_*.fits


