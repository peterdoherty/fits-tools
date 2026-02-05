#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image calibration workflows for CCD/CMOS data.

This module provides comprehensive calibration workflows for bias, dark, and
flat field correction with proper exposure time scaling, validation, and
quality control integration.

Calibration Pipeline:
    1. Bias correction: raw - master_bias
    2. Dark correction: (raw - bias) - (master_dark * exptime_scale)
    3. Flat correction: (raw - bias - dark) / normalized_flat

Features:
    - Proper exposure time scaling for darks
    - Shape validation between frames
    - Optional input validation
    - Optional quality metrics computation
    - Header updates with calibration history
    - Support for in-memory arrays or file paths
    - Batch calibration with multiprocessing

Usage Examples:
    Basic calibration:
        >>> from fits_tools.image_processing.calibration import calibrate_image
        >>> result = calibrate_image('raw.fits', master_bias='bias.fits', 
        ...                          output_file='calibrated.fits')
    
    Full calibration:
        >>> result = calibrate_image('raw.fits',
        ...                          master_bias='bias.fits',
        ...                          master_dark='dark.fits',
        ...                          master_flat='flat.fits',
        ...                          validate_inputs=True,
        ...                          compute_quality=True)
    
    In-memory calibration:
        >>> result = calibrate_image('raw.fits', master_bias=bias_array)
        >>> calibrated_data = result.data

Version: 2.0.0
"""

import os
import logging
from typing import Optional, Union, List, Dict
from datetime import datetime

import numpy as np
from astropy.io import fits

# Import from fits_tools
from ..io import get_data, write_fits, get_data_and_header
from ..constants import FITS_KEY_EXPTIME
from ..exceptions import ImageProcessingError, FileHandlingError
from ..utils import add_history, validate_shapes
from ..validate import (
    validate_fits_file,
    validate_bias_frame,
    validate_dark_frame,
    validate_flat_frame,
    ValidationResult,
)
from ..quality_control import (
    assess_quality,
    assess_quality_array, 
    assess_calibration_quality,
    QualityMetrics,
    QualityThresholds,
    DEFAULT_THRESHOLDS,
)
from .result_types import CalibrationResult, CalibrationBatchResult

logger = logging.getLogger(__name__)

__version__ = "2.0.0"

# ============================================================================
# Helper Function
# ============================================================================

def _validate_and_normalize_region(
    region: tuple,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Validate a (x1, x2, y1, y2) region against an image shape (ny, nx).

    Returns the region as integers if valid, otherwise raises ImageProcessingError.
    """
    if region is None:
        return None  # type: ignore[return-value]

    try:
        x1, x2, y1, y2 = map(int, region)
    except Exception as e:
        raise ImageProcessingError(f"Invalid region {region!r}: {e}")

    h, w = shape
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ImageProcessingError(
            f"Region {region} is out of bounds for image shape {shape}"
        )
    return x1, x2, y1, y2

# ============================================================================
# Core Calibration Function
# ============================================================================

def apply_calibration(
    raw_data: np.ndarray,
    raw_header: fits.Header,
    master_bias_data: Optional[np.ndarray] = None,
    master_dark_data: Optional[np.ndarray] = None,
    master_dark_header: Optional[fits.Header] = None,
    master_flat_data: Optional[np.ndarray] = None,
    verbose: bool = False
) -> CalibrationResult:
    """
    Apply bias, dark, and flat calibration to a raw image array.
    
    Calibration order:
        1. raw - bias
        2. result - (dark * exptime_scale)
        3. result / flat
    
    Parameters
    ----------
    raw_data : np.ndarray
        Raw image data.
    raw_header : fits.Header
        Header from raw image (for exposure time).
    master_bias_data : np.ndarray, optional
        Master bias frame.
    master_dark_data : np.ndarray, optional
        Master dark frame.
    master_dark_header : fits.Header, optional
        Header from master dark (for exposure time scaling).
        Required if master_dark_data is provided.
    master_flat_data : np.ndarray, optional
        Normalized master flat frame.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    CalibrationResult
        Calibration result with data, header, and metadata.
    
    Raises
    ------
    ImageProcessingError
        If shapes don't match or required headers are missing.
    
    Examples
    --------
    >>> raw = np.ones((100, 100)) * 1000
    >>> bias = np.ones((100, 100)) * 100
    >>> header = fits.Header()
    >>> header['EXPTIME'] = 1.0
    >>> result = apply_calibration(raw, header, master_bias_data=bias)
    >>> np.mean(result.data)
    900.0
    >>> result.bias_corrected
    True
    """
    # Convert to float for processing
    calibrated_data = raw_data.astype(np.float32)
    corrections = {
        'bias': False,
        'dark': False,
        'flat': False
    }
    
    # Step 1: Bias subtraction
    if master_bias_data is not None:
        validate_shapes(calibrated_data, master_bias_data, operation="bias correction")
        calibrated_data -= master_bias_data.astype(np.float32)
        corrections['bias'] = True
        if verbose:
            logger.debug("  - Subtracted master bias")
    
    # Step 2: Dark subtraction (scaled by exposure time)
    if master_dark_data is not None:
        if master_dark_header is None:
            raise ImageProcessingError(
                "master_dark_header is required for dark scaling. "
                "Cannot determine dark exposure time."
            )
        
        # Get exposure times
        raw_exptime = raw_header.get(FITS_KEY_EXPTIME)
        dark_exptime = master_dark_header.get(FITS_KEY_EXPTIME)
        
        if raw_exptime is None:
            logger.warning(
                f"Raw image missing {FITS_KEY_EXPTIME} keyword. "
                f"Skipping dark correction."
            )
        elif dark_exptime is None:
            logger.warning(
                f"Dark frame missing {FITS_KEY_EXPTIME} keyword. "
                f"Skipping dark correction."
            )
        elif float(dark_exptime) == 0:
            logger.warning(
                f"Dark frame has EXPTIME=0. Skipping dark correction."
            )
        else:
            validate_shapes(calibrated_data, master_dark_data, operation="dark correction")
            
            # Calculate scaling factor
            scale_factor = float(raw_exptime) / float(dark_exptime)
            
            # Subtract scaled dark
            calibrated_data -= (master_dark_data.astype(np.float32) * scale_factor)
            corrections['dark'] = True
            
            if verbose:
                logger.debug(
                    f"  - Subtracted scaled master dark "
                    f"(raw_exp={raw_exptime:.2f}s, dark_exp={dark_exptime:.2f}s, "
                    f"scale={scale_factor:.3f})"
                )
    
    # Step 3: Flat field correction
    if master_flat_data is not None:
        validate_shapes(calibrated_data, master_flat_data, operation="flat correction")
        
        # Check for problematic flat values
        flat_min = float(np.min(master_flat_data))
        flat_max = float(np.max(master_flat_data))
        
        if flat_min <= 0:
            logger.warning(
                f"Master flat contains zero or negative values (min={flat_min:.4f}). "
                f"This may cause division issues."
            )
        
        # Divide by flat (should already be normalized to ~1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            calibrated_data /= master_flat_data.astype(np.float32)
        
        # Replace inf/nan with zero
        num_invalid = np.sum(~np.isfinite(calibrated_data))
        if num_invalid > 0:
            if verbose:
                logger.warning(
                    f"Flat correction produced {num_invalid} invalid pixels. "
                    f"Replacing with 0."
                )
            calibrated_data = np.nan_to_num(calibrated_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        corrections['flat'] = True
        
        if verbose:
            logger.debug(
                f"  - Divided by master flat "
                f"(flat range: [{flat_min:.4f}, {flat_max:.4f}])"
            )
    
    # Create updated header
    new_header = raw_header.copy()
    if corrections['bias']:
        add_history(new_header, 'Bias corrected')
    if corrections['dark']:
        add_history(new_header, 'Dark corrected')
    if corrections['flat']:
        add_history(new_header, 'Flat corrected')
    
    # Create result
    result = CalibrationResult(
        data=calibrated_data,
        header=new_header,
        bias_corrected=corrections['bias'],
        dark_corrected=corrections['dark'],
        flat_corrected=corrections['flat']
    )
    
    return result


# ============================================================================
# High-Level Calibration Function
# ============================================================================

def calibrate_image(
    raw_file: str,
    output_file: Optional[str] = None,
    master_bias: Optional[Union[str, np.ndarray]] = None,
    master_dark: Optional[Union[str, np.ndarray]] = None,
    master_flat: Optional[Union[str, np.ndarray]] = None,
    ext: int = 0,
    overwrite: bool = False,
    validate_inputs: bool = True,
    compute_quality: bool = False,
    compare_with_raw: bool = False,
    verbose: bool = False
) -> CalibrationResult:
    """
    Calibrate a FITS image with bias, dark, and flat correction.
    
    High-level function that handles file I/O, validation, and quality
    assessment. Supports both file paths and in-memory arrays for
    calibration frames.
    
    Parameters
    ----------
    raw_file : str
        Path to raw FITS file.
    output_file : str, optional
        Output path. If None, no file is written.
    master_bias : str or np.ndarray, optional
        Master bias (file path or array).
    master_dark : str or np.ndarray, optional
        Master dark (file path or array).
    master_flat : str or np.ndarray, optional
        Master flat (file path or array).
    ext : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.
    validate_inputs : bool, optional
        Validate input files before calibration. Defaults to True.
    compute_quality : bool, optional
        Compute quality metrics after calibration. Defaults to False.
    compare_with_raw : bool, optional
        Include comparison metrics with raw data. Defaults to False.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    CalibrationResult
        Calibration result containing:
        - data: Calibrated image array
        - header: Updated FITS header
        - input_file: Path to input file
        - output_file: Path to output file (if written)
        - bias_corrected, dark_corrected, flat_corrected: Correction flags
        - quality_metrics: Quality assessment (if compute_quality=True)
        - validation: Validation results (if validate_inputs=True)
    
    Raises
    ------
    FileHandlingError
        If input files cannot be read or output exists and overwrite=False.
    ImageProcessingError
        If calibration fails.
    
    Examples
    --------
    >>> # Basic bias correction (in-memory)
    >>> result = calibrate_image('raw.fits', master_bias='bias.fits')
    >>> print(result.bias_corrected)
    True
    >>> print(result.data.shape)
    (2048, 2048)
    
    >>> # Full calibration with output file
    >>> result = calibrate_image(
    ...     'raw.fits',
    ...     output_file='calibrated.fits',
    ...     master_bias='bias.fits',
    ...     master_dark='dark.fits',
    ...     master_flat='flat.fits',
    ...     overwrite=True
    ... )
    >>> print(result.output_file)
    calibrated.fits
    
    >>> # With validation and quality metrics
    >>> result = calibrate_image(
    ...     'raw.fits',
    ...     master_bias='bias.fits',
    ...     validate_inputs=True,
    ...     compute_quality=True,
    ...     verbose=True
    ... )
    >>> print(result.quality_metrics)
    """
    # Check output file before processing
    if output_file is not None and os.path.exists(output_file) and not overwrite:
        raise FileHandlingError(
            f"Output file exists and overwrite=False: {output_file}. "
            f"Use overwrite=True to overwrite."
        )
    
    if verbose:
        logger.info(f"Calibrating: {raw_file}")
    
    validation_results = ValidationResult()
    
    try:
        # Step 1: Validate inputs if requested
        if validate_inputs:
            if verbose:
                logger.info("Validating input files...")
            
            # Validate raw file
            raw_validation = validate_fits_file(raw_file, verbose=verbose)
            if not raw_validation.is_valid:
                raise ImageProcessingError(
                    f"Raw file validation failed: {raw_validation.errors}"
                )
            validation_results.merge(raw_validation, prefix="raw")
            
            # Validate master bias
            if isinstance(master_bias, str):
                bias_validation = validate_bias_frame(master_bias, ext=ext, verbose=verbose)
                if not bias_validation.is_valid:
                    logger.warning(f"Master bias validation issues: {bias_validation.errors}")
                validation_results.merge(bias_validation, prefix="bias")
            
            # Validate master dark
            if isinstance(master_dark, str):
                dark_validation = validate_dark_frame(master_dark, ext=ext, verbose=verbose)
                if not dark_validation.is_valid:
                    logger.warning(f"Master dark validation issues: {dark_validation.errors}")
                validation_results.merge(dark_validation, prefix="dark")
            
            # Validate master flat
            if isinstance(master_flat, str):
                flat_validation = validate_flat_frame(master_flat, ext=ext, verbose=verbose)
                if not flat_validation.is_valid:
                    logger.warning(f"Master flat validation issues: {flat_validation.errors}")
                validation_results.merge(flat_validation, prefix="flat")
        
        # Step 2: Load raw data
        if verbose:
            logger.info("Loading raw data...")
        raw_data, raw_header = get_data_and_header(raw_file, ext=ext, verbose=verbose)
        
        # Step 3: Load calibration frames
        bias_data = None
        dark_data = None
        dark_header = None
        flat_data = None
        
        if master_bias is not None:
            if isinstance(master_bias, str):
                if verbose:
                    logger.info(f"Loading master bias: {master_bias}")
                bias_data = get_data(master_bias, ext=ext, verbose=verbose)
            else:
                if verbose:
                    logger.info("Using in-memory master bias")
                bias_data = master_bias
        
        if master_dark is not None:
            if isinstance(master_dark, str):
                if verbose:
                    logger.info(f"Loading master dark: {master_dark}")
                dark_data, dark_header = get_data_and_header(master_dark, ext=ext, verbose=verbose)
            else:
                if verbose:
                    logger.info("Using in-memory master dark")
                dark_data = master_dark
                dark_header = fits.Header()
                # If using array, need exposure time from somewhere
                logger.warning(
                    "Master dark provided as array without header. "
                    "Cannot perform exposure time scaling. "
                    "Assuming same exposure as raw."
                )
                if FITS_KEY_EXPTIME in raw_header:
                    dark_header[FITS_KEY_EXPTIME] = raw_header[FITS_KEY_EXPTIME]
        
        if master_flat is not None:
            if isinstance(master_flat, str):
                if verbose:
                    logger.info(f"Loading master flat: {master_flat}")
                flat_data = get_data(master_flat, ext=ext, verbose=verbose)
            else:
                if verbose:
                    logger.info("Using in-memory master flat")
                flat_data = master_flat
        
        # Step 4: Apply calibration
        if verbose:
            logger.info("Applying calibration...")
        
        result = apply_calibration(
            raw_data,
            raw_header,
            master_bias_data=bias_data,
            master_dark_data=dark_data,
            master_dark_header=dark_header,
            master_flat_data=flat_data,
            verbose=verbose
        )
        
        # Step 5: Add file information
        result.input_file = raw_file
        result.output_file = output_file  # Will be None if not provided
        
        # Step 6: Compute quality metrics if requested
        if compute_quality:
            if verbose:
                logger.info("Computing quality metrics...")
            
            try:
                # Use quality_control module for comprehensive assessment
                quality_assessment = assess_quality_array(
                    data=result.data,
                    header=result.header,
                    filepath=raw_file,  # Use original filename for reference
                    frame_type=result.header.get('IMAGETYP'),
                    thresholds=DEFAULT_THRESHOLDS,
                    compute_snr=True,
                    verbose=verbose
                )
                
                # Store full quality metrics in result
                result.quality_metrics = quality_assessment.to_dict()
                
                # Add quality information to header
                result.header['QCGRADE'] = (
                    quality_assessment.quality_grade,
                    'Quality control grade'
                )
                result.header['QCSCORE'] = (
                    quality_assessment.quality_score,
                    'Quality control score (0-100)'
                )
                result.header['QCPASS'] = (
                    quality_assessment.passed,
                    'Passed quality control'
                )
                
                # Add specific metrics
                if quality_assessment.snr:
                    result.header['QCSNR'] = (quality_assessment.snr, 'QC SNR estimate')
                if quality_assessment.num_negative_pixels > 0:
                    result.header['QCNEGNEG'] = (
                        quality_assessment.num_negative_pixels,
                        'Number of negative pixels'
                    )
                if quality_assessment.num_saturated_pixels > 0:
                    result.header['QCNSAT'] = (
                        quality_assessment.num_saturated_pixels,
                        'Number of saturated pixels'
                    )
                
                if not quality_assessment.passed:
                    add_history(result.header, f'Quality control: {quality_assessment.quality_grade} (FAILED)')
                    if verbose:
                        logger.warning(f"Quality control FAILED: {quality_assessment.quality_grade}")
                        for warning in quality_assessment.warnings[:3]:  # Show first 3
                            logger.warning(f"  - {warning}")
                else:
                    add_history(result.header, f'Quality control: {quality_assessment.quality_grade}')
                
                if verbose:
                    logger.info(f"  Quality grade: {quality_assessment.quality_grade}")
                    logger.info(f"  Quality score: {quality_assessment.quality_score:.1f}/100")
                    if quality_assessment.snr:
                        logger.info(f"  SNR: {quality_assessment.snr:.2f}")
                    if quality_assessment.num_negative_pixels > 0:
                        logger.warning(
                            f"  Negative pixels: {quality_assessment.num_negative_pixels} "
                            f"({quality_assessment.num_negative_pixels/quality_assessment.total_pixels*100:.2f}%)"
                        )
            
            except Exception as e:
                logger.warning(f"Could not compute quality metrics: {e}")
                if verbose:
                    logger.exception("Quality assessment error:")
                result.quality_metrics = {
                    'error': str(e),
                    'exception_type': type(e).__name__,
                }
        
        # Step 7: Attach validation results
        result.validation = validation_results
        
        # Step 8: Update header with processing metadata
        add_history(result.header, f'Calibrated with fits_tools v{__version__}')
        result.header['PROCDATE'] = (datetime.now().isoformat(), 'Processing date')
        
        # Add calibration file references
        if isinstance(master_bias, str):
            result.header['BIASFILE'] = (os.path.basename(master_bias), 'Master bias file')
        if isinstance(master_dark, str):
            result.header['DARKFILE'] = (os.path.basename(master_dark), 'Master dark file')
        if isinstance(master_flat, str):
            result.header['FLATFILE'] = (os.path.basename(master_flat), 'Master flat file')
        
        # Add validation warning if needed
        if validation_results and not validation_results.is_valid:
            result.header['VALWARN'] = (True, 'Validation warnings present')
        else:
            result.header['VALWARN'] = (False, 'No validation warnings')

        # Step 9: Write output if requested
        if output_file is not None:
            if verbose:
                logger.info(f"Writing calibrated image: {output_file}")
            
            write_fits(
                output_file,
                result.data,
                result.header,
                overwrite=overwrite,
                verbose=verbose
            )
            
            if verbose:
                logger.info(f"Calibration complete: {output_file}")
        else:
            if verbose:
                logger.info("Calibration complete (in-memory)")
        
        return result
    
    except Exception as e:
        if verbose:
            logger.error(f"Calibration failed for {raw_file}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to calibrate image: {e}")
    
def preview_calibration(
    raw_file: str,
    master_bias: Optional[Union[str, np.ndarray]] = None,
    master_dark: Optional[Union[str, np.ndarray]] = None,
    master_flat: Optional[Union[str, np.ndarray]] = None,
    ext: int = 0,
    region: Optional[tuple] = None,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Preview calibration without writing output.
    
    Useful for checking calibration effect on a region before processing
    the entire dataset.
    
    Parameters
    ----------
    raw_file : str
        Path to raw FITS file.
    master_bias : str or np.ndarray, optional
        Master bias frame.
    master_dark : str or np.ndarray, optional
        Master dark frame.
    master_flat : str or np.ndarray, optional
        Master flat frame.
    ext : int, optional
        HDU extension. Defaults to 0.
    region : tuple, optional
        Region to preview (x1, x2, y1, y2). If None, uses full image.
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    dict
        Dictionary with keys: 'raw', 'bias_subtracted', 'dark_subtracted', 'flat_corrected'
        showing calibration stages.
    
    Examples
    --------
    >>> preview = preview_calibration(
    ...     'raw.fits',
    ...     master_bias='bias.fits',
    ...     region=(500, 600, 500, 600)
    ... )
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2)
    >>> axes[0].imshow(preview['raw'])
    >>> axes[1].imshow(preview['flat_corrected'])
    """
    # Load raw data
    raw_data, raw_header = get_data_and_header(raw_file, ext=ext, verbose=verbose)
    
    # Crop to region if specified
    x1 = x2 = y1 = y2 = None
    if region is not None:
        x1, x2, y1, y2 = _validate_and_normalize_region(region, raw_data.shape)
        raw_data = raw_data[y1:y2, x1:x2]
    
    stages = {'raw': raw_data.copy()}
    
    # Track calibration through stages
    working_data = raw_data.astype(np.float32)
    
    # Load and apply bias
    if master_bias is not None:
        if isinstance(master_bias, str):
            bias_data = get_data(master_bias, ext=ext, verbose=verbose)
        else:
            bias_data = master_bias
        
        if region is not None:
            bias_data = bias_data[y1:y2, x1:x2]
        
        working_data -= bias_data.astype(np.float32)
        stages['bias_subtracted'] = working_data.copy()
    
    # Load and apply dark
    if master_dark is not None:
        if isinstance(master_dark, str):
            dark_data, dark_header = get_data_and_header(master_dark, ext=ext, verbose=verbose)
        else:
            dark_data = master_dark
            dark_header = fits.Header()
            dark_header[FITS_KEY_EXPTIME] = raw_header.get(FITS_KEY_EXPTIME, 1.0)
        
        if region is not None:
            dark_data = dark_data[y1:y2, x1:x2]
        
        raw_exptime = raw_header.get(FITS_KEY_EXPTIME, 1.0)
        dark_exptime = dark_header.get(FITS_KEY_EXPTIME, 1.0)
        scale = float(raw_exptime) / float(dark_exptime)
        
        working_data -= dark_data.astype(np.float32) * scale
        stages['dark_subtracted'] = working_data.copy()
    
    # Load and apply flat
    if master_flat is not None:
        if isinstance(master_flat, str):
            flat_data = get_data(master_flat, ext=ext, verbose=verbose)
        else:
            flat_data = master_flat
        
        if region is not None:
            flat_data = flat_data[y1:y2, x1:x2]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            working_data /= flat_data.astype(np.float32)
        working_data = np.nan_to_num(working_data, nan=0.0, posinf=0.0, neginf=0.0)
        stages['flat_corrected'] = working_data.copy()
    
    return stages

# ============================================================================
# Batch Calibration
# ============================================================================

def calibrate_batch(
    raw_files: List[str],
    output_dir: str,
    master_bias: Optional[Union[str, np.ndarray]] = None,
    master_dark: Optional[Union[str, np.ndarray]] = None,
    master_flat: Optional[Union[str, np.ndarray]] = None,
    ext: int = 0,
    overwrite: bool = False,
    validate_inputs: bool = True,
    compute_quality: bool = False,
    num_workers: int = 1,
    verbose: bool = False
) -> List[CalibrationBatchResult]:
    """
    Calibrate multiple images in batch mode.
    
    Parameters
    ----------
    raw_files : list of str
        List of raw FITS file paths.
    output_dir : str
        Output directory for calibrated images.
    master_bias : str or np.ndarray, optional
        Master bias (file path or array).
    master_dark : str or np.ndarray, optional
        Master dark (file path or array).
    master_flat : str or np.ndarray, optional
        Master flat (file path or array).
    ext : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing outputs. Defaults to False.
    validate_inputs : bool, optional
        Validate input files. Defaults to True.
    compute_quality : bool, optional
        Compute quality metrics. Defaults to False.
    num_workers : int, optional
        Number of parallel workers. Defaults to 1 (serial).
    verbose : bool, optional
        Enable verbose logging.
    
    Returns
    -------
    CalibrationBatchResult
        Batch calibration results and summary statistics.
    
    Examples
    --------
    >>> raw_files = ['raw1.fits', 'raw2.fits', 'raw3.fits']
    >>> batch = calibrate_batch(raw_files, output_dir='calibrated/', master_bias='bias.fits')
    ... print(f"{batch.success_count}/{batch.total_count} calibrated")
    ... for r in batch.results:
    ... print(r.input_file, "->", r.output_file)
    """
    
    if not raw_files:
        raise ValueError("Empty raw_files list")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        logger.info(f"Calibrating {len(raw_files)} images...")
        logger.info(f"Output directory: {output_dir}")
    
    bias_data = None
    dark_ref = None   # can be str path or np.ndarray
    flat_data = None

    # Pre-load bias if given
    if master_bias is not None:
        if isinstance(master_bias, str):
            if verbose:
                logger.info(f"Loading master bias: {master_bias}")
            bias_data = get_data(master_bias, ext=ext, verbose=False)
        else:
            bias_data = master_bias

    # For dark: if string, keep the *filename*; if array, keep the array
    if master_dark is not None:
        if isinstance(master_dark, str):
            # Do NOT pre-load; let calibrate_image load data + header
            dark_ref = master_dark
        else:
            # In-memory array
            dark_ref = master_dark

    # Pre-load flat if given
    if master_flat is not None:
        if isinstance(master_flat, str):
            if verbose:
                logger.info(f"Loading master flat: {master_flat}")
            flat_data = get_data(master_flat, ext=ext, verbose=False)
        else:
            flat_data = master_flat
    
    batch_result = CalibrationBatchResult()
    
    # Optionally record master frames in metadata for later inspection
    batch_result.metadata.update({
        'master_bias': master_bias if isinstance(master_bias, str) else None,
        'master_dark': master_dark if isinstance(master_dark, str) else None,
        'master_flat': master_flat if isinstance(master_flat, str) else None,
    })

    # TODO: Implement parallel processing with ProcessPoolExecutor if num_workers > 1
    # For now, serial processing
    for i, raw_file in enumerate(raw_files, 1):
        if verbose:
            logger.info(f"[{i}/{len(raw_files)}] Processing: {os.path.basename(raw_file)}")
        
        try:
            # Generate output filename
            basename = os.path.basename(raw_file)
            name, ext_str = os.path.splitext(basename)
            output_file = os.path.join(output_dir, f"{name}_cal{ext_str}")
            
            # Calibrate using pre-loaded arrays
            result = calibrate_image(
                raw_file,
                output_file=output_file,
                master_bias=bias_data,
                master_dark=dark_ref if master_dark is not None else None,
                master_flat=flat_data,
                ext=ext,
                overwrite=overwrite,
                validate_inputs=validate_inputs,
                compute_quality=compute_quality,
                verbose=False  # keep batch logs only
            )
            
            batch_result.add_success(result)

            if verbose:
                logger.info(f"  ✓ Success: {os.path.basename(output_file)}")
        
        except Exception as e:
            logger.error(f"  ✗ Failed: {raw_file}: {e}")
            batch_result.add_failure(raw_file, e)
    
    if verbose:
        logger.info(
            f"Batch calibration complete: "
            f"{batch_result.success_count}/{batch_result.total_count} successful "
            f"({batch_result.success_rate*100:.1f}%)"
        )

    return batch_result


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'apply_calibration',
    'calibrate_image',
    'calibrate_batch',
]



# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """Basic module testing."""
    import sys
    
    print(f"Testing calibration.py v{__version__}")
    print("=" * 60)
    
    # Test 1: In-memory calibration
    print("\nTest 1: In-memory bias calibration")
    try:
        raw_data = np.random.normal(1000, 50, (100, 100))
        bias_data = np.random.normal(100, 5, (100, 100))
        
        header = fits.Header()
        header['EXPTIME'] = 1.0
        
        result = apply_calibration(raw_data, header, master_bias_data=bias_data, verbose=True)
        
        print(f"✓ Calibration successful")
        print(f"  Input mean: {np.mean(raw_data):.2f}")
        print(f"  Bias mean: {np.mean(bias_data):.2f}")
        print(f"  Output mean: {np.mean(result.data):.2f}")
        print(f"  Bias corrected: {result.bias_corrected}")
        
        expected_mean = np.mean(raw_data) - np.mean(bias_data)
        assert abs(np.mean(result.data) - expected_mean) < 1.0
        
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        sys.exit(1)
    
    # Test 2: Full calibration with dark and flat
    print("\nTest 2: Full calibration (bias + dark + flat)")
    try:
        raw_data = np.random.normal(10000, 100, (100, 100))
        bias_data = np.random.normal(100, 5, (100, 100))
        dark_data = np.random.normal(10, 2, (100, 100))
        flat_data = np.random.normal(1.0, 0.05, (100, 100))
        
        raw_header = fits.Header({'EXPTIME': 60.0})
        dark_header = fits.Header({'EXPTIME': 60.0})
        
        result = apply_calibration(
            raw_data,
            raw_header,
            master_bias_data=bias_data,
            master_dark_data=dark_data,
            master_dark_header=dark_header,
            master_flat_data=flat_data,
            verbose=True
        )
        
        print(f"✓ Full calibration successful")
        print(f"  Bias corrected: {result.bias_corrected}")
        print(f"  Dark corrected: {result.dark_corrected}")
        print(f"  Flat corrected: {result.flat_corrected}")
        print(f"  Output shape: {result.data.shape}")
        
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        sys.exit(1)
    
    # Test 3: Dark scaling
    print("\nTest 3: Dark exposure time scaling")
    try:
        raw_data = np.ones((50, 50)) * 1000
        dark_data = np.ones((50, 50)) * 10
        
        raw_header = fits.Header({'EXPTIME': 120.0})
        dark_header = fits.Header({'EXPTIME': 60.0})
        
        result = apply_calibration(
            raw_data,
            raw_header,
            master_dark_data=dark_data,
            master_dark_header=dark_header,
            verbose=True
        )
        
        # Dark should be scaled by 2x (120/60)
        expected = 1000 - (10 * 2)
        print(f"✓ Dark scaling correct")
        print(f"  Expected: {expected:.2f}")
        print(f"  Got: {np.mean(result.data):.2f}")
        
        assert abs(np.mean(result.data) - expected) < 0.1
        
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("\nModule is ready for use.")

