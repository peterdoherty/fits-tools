#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FITS file validation utilities for CCD/CMOS imager test data.

This module provides comprehensive validation functions for ensuring data quality
and integrity of FITS files used in CCD/CMOS detector characterization and testing.

Features:
    - Basic FITS file integrity validation
    - Header keyword validation (presence, types, values)
    - Data validation (ranges, NaN checking, dimensions)
    - CCD-specific frame validation (BIAS, DARK, FLAT)
    - Multi-file set validation for calibration frames
    - Schema-based validation with flexible configuration
    - Comprehensive ValidationResult reporting

Version History:
    1.0.0 - Initial implementation with basic, header, data, and CCD-specific validation.
    1.1.0 - Refactored to use fits_tools modules, fixed indentation bug,
            integrated with constants.py image types, improved type hints
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from fits_tools.constants import (
    FITS_KEY_IMGTYPE,
    FITS_KEY_IMAGETYP,
    FITS_KEY_EXPTIME,
    ALL_IMAGE_TYPES,
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    is_valid_image_type,
    normalize_image_type,
)

from .exceptions import ValidationError as FitsValidationError

from .types import StrOrPath

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result Container
# =============================================================================

@dataclass
class ValidationResult:
    """
    Container for validation results.
    
    Provides structured storage of validation outcomes with errors, warnings,
    and additional information.
    
    Attributes:
        is_valid: Overall validation status.
        errors: List of error messages for critical issues.
        warnings: List of warning messages for non-critical issues.
        info: Additional information about the validation.
    
    Examples:
        >>> result = ValidationResult()
        >>> result.add_error("Missing required keyword")
        >>> result.is_valid
        False
        >>> result.add_warning("Non-standard value")
        >>> len(result.warnings)
        1
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add an error message and mark validation as failed."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message (does not affect is_valid status)."""
        self.warnings.append(message)
    
    def add_info(self, key: str, value: Any) -> None:
        """Add informational data to the results."""
        self.info[key] = value
    
    def merge(self, other: 'ValidationResult', prefix: str = "") -> None:
        """
        Merge another ValidationResult into this one.
        
        Args:
            other: Another ValidationResult to merge.
            prefix: Optional prefix for error/warning messages.
        """
        if not other.is_valid:
            self.is_valid = False
        
        if prefix:
            self.errors.extend([f"{prefix}: {e}" for e in other.errors])
            self.warnings.extend([f"{prefix}: {w}" for w in other.warnings])
        else:
            self.errors.extend(other.errors)
            self.warnings.extend(other.warnings)
        
        self.info.update(other.info)
    
    def __str__(self) -> str:
        """String representation of validation results."""
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines = [f"Validation Status: {status}"]
        
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
        
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
        
        if self.info:
            lines.append(f"\nAdditional Info:")
            for key, value in self.info.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self.is_valid}, errors={len(self.errors)}, warnings={len(self.warnings)})"


# =============================================================================
# Basic File Integrity Validation
# =============================================================================

def validate_fits_file(filepath: StrOrPath) -> ValidationResult:
    """
    Validate basic FITS file integrity.
    
    Checks:
        - File exists
        - File is readable
        - File is valid FITS format
        - File can be opened by astropy
        - File contains at least one HDU
    
    Args:
        filepath: Path to the FITS file.
    
    Returns:
        ValidationResult with status and messages.
    
    Examples:
        >>> # result = validate_fits_file('test.fits')
        >>> # if result.is_valid:
        >>> #     print("File is valid")
    """
    result = ValidationResult()
    
    filepath = Path(filepath)
    # Check file exists
    if not filepath.exists():
        result.add_error(f"File not found: {filepath}")
        return result
    
    result.add_info('filepath', filepath)
    result.add_info('filesize_kb', filepath.stat().st_size / 1024)
    
    # Check file is readable
    try:
        with filepath.open("rb"):
            pass
    except PermissionError:
        result.add_error(f"File is not readable: {filepath}")
        return result
    except OSError as e:
        result.add_error(f"Could not access file {filepath}: {e}")
        return result
    
    # Check file size
    if filepath.stat().st_size == 0:
        result.add_error(f"File is empty: {filepath}")
        return result
    
    # Try to open as FITS
    try:
        with fits.open(filepath, memmap=True) as hdul:
            # Check that file has at least one HDU
            if len(hdul) == 0:
                result.add_error("FITS file contains no HDUs")
                return result
            
            result.add_info('num_hdus', len(hdul))
            
            # Verify FITS structure
            try:
                hdul.verify('fix')
            except Exception as e:
                result.add_warning(f"FITS verification issues: {e}")
            
            logger.debug("Successfully validated FITS file: %s", filepath)
    
    except Exception as e:
        result.add_error(f"Failed to open as valid FITS file: {e}")
        return result
    
    return result


def check_fits_integrity(
    filepath: StrOrPath, 
    fix: bool = False, 
) -> ValidationResult:
    """
    Perform comprehensive FITS integrity check using astropy's verify.
    
    Args:
        filepath: Path to the FITS file.
        fix: Attempt to fix fixable issues. Defaults to False.
    
    Returns:
        Detailed validation results.
    
    Examples:
        >>> # result = check_fits_integrity('test.fits')
        >>> # print(result)
    """
    filepath = Path(filepath)

    result = validate_fits_file(filepath)
    if not result.is_valid:
        return result
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            # Run astropy's verification
            verify_output = hdul.verify('fix' if fix else 'exception')
            
            if verify_output:
                for issue in verify_output:
                    result.add_warning(str(issue))
            
            result.add_info('integrity_check', 'passed')
            
    except Exception as e:
        result.add_error(f"FITS integrity check failed: {e}")
    
    return result


# =============================================================================
# Header Validation
# =============================================================================

def validate_required_keywords(
    filepath: StrOrPath,
    required_keywords: List[str],
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate that required header keywords are present.
    
    Args:
        filepath: Path to the FITS file.
        required_keywords: List of required keyword names.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult with keyword presence information.
    
    Examples:
        >>> # result = validate_required_keywords('test.fits', ['EXPTIME', 'IMAGETYP'])
        >>> # if not result.is_valid:
        >>> #     print("Missing required keywords")
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist (file has {len(hdul)} HDUs)")
                return result
            
            header = hdul[hdu].header
            missing_keywords = []
            present_keywords = []
            
            for keyword in required_keywords:
                if keyword not in header:
                    missing_keywords.append(keyword)
                else:
                    present_keywords.append(keyword)
            
            if missing_keywords:
                result.add_error(f"Missing required keywords: {missing_keywords}")
            
            result.add_info('required_keywords', required_keywords)
            result.add_info('present_keywords', present_keywords)
            result.add_info('missing_keywords', missing_keywords)
            
            if result.is_valid:
                logger.debug("All required keywords present in %s", filepath)
    
    except Exception as e:
        result.add_error(f"Failed to validate keywords: {e}")
    
    return result


def validate_header_values(
    filepath: StrOrPath,
    keyword_constraints: Dict[str, Any],
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate header keyword values against constraints.
    
    Args:
        filepath: Path to the FITS file.
        keyword_constraints: Dictionary mapping keywords to expected values
            or validation functions. Value can be:
            - A specific value (equality check)
            - A tuple of (min, max) for numeric range
            - A list of valid values
            - A callable that returns bool
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult with constraint checking results.
    
    Examples:
        >>> # constraints = {
        >>> #     'EXPTIME': (0, 3600),  # Range
        >>> #     'IMAGETYP': ['BIAS', 'DARK', 'FLAT'],  # Valid values
        >>> #     'NAXIS': 2  # Exact value
        >>> # }
        >>> # result = validate_header_values('test.fits', constraints)
    """
    filepath = Path(filepath)

    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            header = hdul[hdu].header
            
            for keyword, constraint in keyword_constraints.items():
                if keyword not in header:
                    result.add_error(f"Keyword '{keyword}' not found in header")
                    continue
                
                value = header[keyword]
                
                # Check constraint type
                if callable(constraint):
                    # Custom validation function
                    try:
                        if not constraint(value):
                            result.add_error(
                                f"Keyword '{keyword}' value '{value}' failed validation"
                            )
                    except Exception as e:
                        result.add_error(
                            f"Validation function for '{keyword}' raised error: {e}"
                        )
                
                elif isinstance(constraint, (list, set, tuple)) and len(constraint) == 2 and \
                     isinstance(constraint[0], (int, float)) and isinstance(constraint[1], (int, float)):
                    # Numeric range (min, max)
                    min_val, max_val = constraint
                    if not (min_val <= value <= max_val):
                        result.add_error(
                            f"Keyword '{keyword}' value {value} outside valid range [{min_val}, {max_val}]"
                        )
                
                elif isinstance(constraint, (list, set, tuple)):
                    # List of valid values
                    if value not in constraint:
                        result.add_error(
                            f"Keyword '{keyword}' value '{value}' not in valid values: {constraint}"
                        )
                
                else:
                    # Exact value match
                    if value != constraint:
                        result.add_error(
                            f"Keyword '{keyword}' value '{value}' does not match expected '{constraint}'"
                        )
            
            if result.is_valid:
                logger.debug("All header value constraints satisfied for %s", filepath)
    
    except Exception as e:
        result.add_error(f"Failed to validate header values: {e}")
    
    return result


def validate_keyword_types(
    filepath: StrOrPath,
    keyword_types: Dict[str, type],
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate that header keywords have the expected data types.
    
    Args:
        filepath: Path to the FITS file.
        keyword_types: Dictionary mapping keywords to expected Python types.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult with type checking results.
    
    Examples:
        >>> # types = {'EXPTIME': float, 'NAXIS': int, 'OBSERVER': str}
        >>> # result = validate_keyword_types('test.fits', types)
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            header = hdul[hdu].header
            
            for keyword, expected_type in keyword_types.items():
                if keyword not in header:
                    result.add_error(f"Keyword '{keyword}' not found")
                    continue
                
                value = header[keyword]
                
                if not isinstance(value, expected_type):
                    result.add_error(
                        f"Keyword '{keyword}' has type {type(value).__name__}, "
                        f"expected {expected_type.__name__}"
                    )
            
            if result.is_valid:
                logger.debug("All keyword types correct for %s", filepath)
    
    except Exception as e:
        result.add_error(f"Failed to validate keyword types: {e}")
    
    return result


# =============================================================================
# Data Validation
# =============================================================================

def validate_data_range(
    filepath: StrOrPath,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate that image data values are within expected range.
    
    Args:
        filepath: Path to the FITS file.
        min_value: Minimum allowed value. None for no minimum.
        max_value: Maximum allowed value. None for no maximum.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult including actual data range statistics.
    
    Examples:
        >>> # result = validate_data_range('bias.fits', min_value=0, max_value=65535)
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            data = hdul[hdu].data
            
            if data is None:
                result.add_error(f"HDU {hdu} contains no data")
                return result
            
            data_min = np.min(data)
            data_max = np.max(data)
            
            result.add_info('data_min', float(data_min))
            result.add_info('data_max', float(data_max))
            result.add_info('data_mean', float(np.mean(data)))
            result.add_info('data_std', float(np.std(data)))
            
            if min_value is not None and data_min < min_value:
                result.add_error(
                    f"Data minimum {data_min} is below allowed minimum {min_value}"
                )
            
            if max_value is not None and data_max > max_value:
                result.add_error(
                    f"Data maximum {data_max} exceeds allowed maximum {max_value}"
                )
            
            if result.is_valid:
                logger.debug("Data range valid for %s: [%s, %s]", filepath, data_min, data_max)
    
    except Exception as e:
        result.add_error(f"Failed to validate data range: {e}")
    
    return result


def validate_no_nans(
    filepath: StrOrPath, 
    hdu: int = 0, 
) -> ValidationResult:
    """
    Check for NaN and Inf values in image data.
    
    Args:
        filepath: Path to the FITS file.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult including NaN/Inf counts.
    
    Examples:
        >>> # result = validate_no_nans('test.fits')
        >>> # if result.is_valid:
        >>> #     print("No NaN values found")
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            data = hdul[hdu].data
            
            if data is None:
                result.add_error(f"HDU {hdu} contains no data")
                return result
            
            if np.issubdtype(data.dtype, np.floating):
                nan_count = int(np.sum(np.isnan(data)))
                inf_count = int(np.sum(np.isinf(data)))
            else:
                nan_count = 0
                inf_count = 0
            
            result.add_info('nan_count', int(nan_count))
            result.add_info('inf_count', int(inf_count))
            result.add_info('total_pixels', int(data.size))
            
            if nan_count > 0:
                result.add_error(f"Found {nan_count} NaN values in data")
            
            if inf_count > 0:
                result.add_error(f"Found {inf_count} Inf values in data")
            
            if result.is_valid:
                logger.debug("No NaN/Inf values in %s", filepath)
    
    except Exception as e:
        result.add_error(f"Failed to check for NaN values: {e}")
    
    return result


def validate_image_dimensions(
    filepath: StrOrPath,
    expected_shape: Optional[Tuple[int, int]] = None,
    min_shape: Optional[Tuple[int, int]] = None,
    max_shape: Optional[Tuple[int, int]] = None,
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate image dimensions.
    
    Args:
        filepath: Path to the FITS file.
        expected_shape: Expected (rows, cols) shape.
        min_shape: Minimum allowed (rows, cols).
        max_shape: Maximum allowed (rows, cols).
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult including actual shape information.
    
    Examples:
        >>> # result = validate_image_dimensions('test.fits', expected_shape=(2048, 2048))
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            data = hdul[hdu].data
            
            if data is None:
                result.add_error(f"HDU {hdu} contains no data")
                return result
            
            shape = data.shape
            result.add_info('shape', shape)
            result.add_info('dimensions', len(shape))
            
            # Check for 2D image
            if len(shape) != 2:
                result.add_error(f"Expected 2D image, got {len(shape)}D array with shape {shape}")
                return result
            
            rows, cols = shape
            
            # Check exact shape
            if expected_shape is not None:
                if shape != expected_shape:
                    result.add_error(
                        f"Image shape {shape} does not match expected {expected_shape}"
                    )
            
            # Check minimum dimensions
            if min_shape is not None:
                min_rows, min_cols = min_shape
                if rows < min_rows or cols < min_cols:
                    result.add_error(
                        f"Image shape {shape} is smaller than minimum {min_shape}"
                    )
            
            # Check maximum dimensions
            if max_shape is not None:
                max_rows, max_cols = max_shape
                if rows > max_rows or cols > max_cols:
                    result.add_error(
                        f"Image shape {shape} exceeds maximum {max_shape}"
                    )
            
            if result.is_valid:
                logger.debug("Image dimensions valid for %s: %s", filepath, shape)
    
    except Exception as e:
        result.add_error(f"Failed to validate image dimensions: {e}")
    
    return result


# =============================================================================
# CCD-Specific Validation
# =============================================================================

def validate_frame_type(
    filepath: StrOrPath,
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate that IMGTYPE/IMAGETYP keyword contains a recognized frame type.
    
    Args:
        filepath: Path to the FITS file.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult including detected frame type.
    
    Examples:
        >>> # result = validate_frame_type('bias.fits')
        >>> # print(result.info['frame_type'])
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            header = hdul[hdu].header
            
            # Try both IMGTYPE and IMAGETYP
            imagetyp = None
            for keyword in [FITS_KEY_IMGTYPE, FITS_KEY_IMAGETYP]:
                if keyword in header:
                    imagetyp = header[keyword].strip().upper()
                    break
            
            if imagetyp is None:
                result.add_error(f"Required keyword 'IMGTYPE' or 'IMAGETYP' not found")
                return result
            
            # Normalize the image type
            normalized = normalize_image_type(imagetyp)
            result.add_info('frame_type', normalized)
            result.add_info('frame_type_raw', imagetyp)
            
            # Check if it's a recognized type
            if not is_valid_image_type(normalized):
                result.add_warning(
                    f"Frame type '{imagetyp}' not in standard types: {list(ALL_IMAGE_TYPES)}"
                )
            
            logger.debug("Frame type for %s: %s", filepath, normalized)

    except Exception as e:
        result.add_error(f"Failed to validate frame type: {e}")
    
    return result


def validate_bias_frame(
    filepath: StrOrPath,
    hdu: int = 0, 
) -> ValidationResult:
    """
    Validate a BIAS frame meets expected criteria.
    
    Checks:
        - IMGTYPE/IMAGETYP = 'BIAS'
        - EXPTIME = 0 (or very close to 0)
        - Reasonable pixel value range
    
    Args:
        filepath: Path to the FITS file.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult for BIAS frame.
    
    Examples:
        >>> # result = validate_bias_frame('bias_001.fits')
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            header = hdul[hdu].header
            
            # Check IMGTYPE/IMAGETYP
            imagetyp = None
            for keyword in [FITS_KEY_IMGTYPE, FITS_KEY_IMAGETYP]:
                if keyword in header:
                    imagetyp = normalize_image_type(header[keyword].strip())
                    break
            
            if imagetyp is None:
                result.add_error("Missing IMGTYPE/IMAGETYP keyword")
            elif imagetyp != FITS_VAL_BIAS_IMGTYPE:
                result.add_error(
                    f"Expected IMGTYPE='BIAS', got '{imagetyp}'"
                )
            
            # Check EXPTIME
            if FITS_KEY_EXPTIME not in header:
                result.add_warning(f"Missing keyword '{FITS_KEY_EXPTIME}'")
            else:
                exptime = header[FITS_KEY_EXPTIME]
                if exptime > 0.001:  # Allow tiny values for electronic overhead
                    result.add_error(
                        f"BIAS frame should have EXPTIME≈0, got {exptime}"
                    )
                result.add_info('exptime', exptime)
            
            # Check data range (bias should be relatively low values)
            data = hdul[hdu].data
            if data is not None:
                data_max = np.max(data)
                if data_max > 10000:  # Assuming 16-bit detector
                    result.add_warning(
                        f"BIAS frame has unusually high maximum value: {data_max}"
                    )
                result.add_info('data_mean', float(np.mean(data)))
                result.add_info('data_std', float(np.std(data)))
            
            if result.is_valid:
                logger.debug("BIAS frame validation passed for %s", filepath)

    
    except Exception as e:
        result.add_error(f"Failed to validate BIAS frame: {e}")
    
    return result


def validate_dark_frame(
    filepath: StrOrPath,
    hdu: int = 0,
    min_exptime: float = 0.1,
) -> ValidationResult:
    """
    Validate a DARK frame meets expected criteria.
    
    Checks:
        - IMGTYPE/IMAGETYP = 'DARK'
        - EXPTIME > 0 and reasonable
        - No saturation (unless expected for long darks)
    
    Args:
        filepath: Path to the FITS file.
        hdu: HDU extension to check. Defaults to 0.
        min_exptime: Minimum expected exposure time. Defaults to 0.1.
    
    Returns:
        ValidationResult for DARK frame.
    
    Examples:
        >>> # result = validate_dark_frame('dark_300s.fits', min_exptime=1.0)
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            header = hdul[hdu].header
            
            # Check IMGTYPE/IMAGETYP
            imagetyp = None
            for keyword in [FITS_KEY_IMGTYPE, FITS_KEY_IMAGETYP]:
                if keyword in header:
                    imagetyp = normalize_image_type(header[keyword].strip())
                    break
            
            if imagetyp is None:
                result.add_error("Missing IMGTYPE/IMAGETYP keyword")
            elif imagetyp != FITS_VAL_DARK_IMGTYPE:
                result.add_error(
                    f"Expected IMGTYPE='DARK', got '{imagetyp}'"
                )
            
            # Check EXPTIME
            if FITS_KEY_EXPTIME not in header:
                result.add_error(f"Missing keyword '{FITS_KEY_EXPTIME}'")
            else:
                exptime = header[FITS_KEY_EXPTIME]
                if exptime < min_exptime:
                    result.add_error(
                        f"DARK frame EXPTIME ({exptime}) below minimum ({min_exptime})"
                    )
                result.add_info('exptime', exptime)
            
            # Check data
            data = hdul[hdu].data
            if data is not None:
                result.add_info('data_mean', float(np.mean(data)))
                result.add_info('data_std', float(np.std(data)))
                result.add_info('data_max', float(np.max(data)))
            
            if result.is_valid:
                logger.debug("DARK frame validation passed for %s", filepath)
    
    except Exception as e:
        result.add_error(f"Failed to validate DARK frame: {e}")
    
    return result


def validate_flat_frame(
    filepath: StrOrPath,
    hdu: int = 0,
    min_counts: float = 1000,
    max_counts: float = 50000,
) -> ValidationResult:
    """
    Validate a FLAT frame meets expected criteria.
    
    Checks:
        - IMGTYPE/IMAGETYP = 'FLAT'
        - Mean counts in reasonable range (not too dim, not saturated)
        - Reasonable uniformity
    
    Args:
        filepath: Path to the FITS file.
        hdu: HDU extension to check. Defaults to 0.
        min_counts: Minimum acceptable mean counts. Defaults to 1000.
        max_counts: Maximum acceptable mean counts. Defaults to 50000.
    
    Returns:
        ValidationResult for FLAT frame.
    
    Examples:
        >>> # result = validate_flat_frame('flat_V.fits', min_counts=5000, max_counts=40000)
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            header = hdul[hdu].header
            
            # Check IMGTYPE/IMAGETYP
            imagetyp = None
            for keyword in [FITS_KEY_IMGTYPE, FITS_KEY_IMAGETYP]:
                if keyword in header:
                    imagetyp = normalize_image_type(header[keyword].strip())
                    break
            
            if imagetyp is None:
                result.add_error("Missing IMGTYPE/IMAGETYP keyword")
            elif imagetyp != FITS_VAL_FLAT_IMGTYPE:
                result.add_error(
                    f"Expected IMGTYPE='FLAT', got '{imagetyp}'"
                )
            
            # Check data counts
            data = hdul[hdu].data
            if data is not None:
                data_mean = np.mean(data)
                data_median = np.median(data)
                data_std = np.std(data)
                
                result.add_info('data_mean', float(data_mean))
                result.add_info('data_median', float(data_median))
                result.add_info('data_std', float(data_std))
                
                # FIXED INDENTATION BUG HERE!
                if data_mean < min_counts:
                    result.add_error(
                        f"FLAT frame mean counts ({data_mean:.1f}) below minimum ({min_counts})"
                    )
                
                if data_mean > max_counts:
                    result.add_error(
                        f"FLAT frame mean counts ({data_mean:.1f}) exceeds maximum ({max_counts})"
                    )
                
                # Check for reasonable uniformity (CV = std/mean)
                if data_mean > 0:
                    coefficient_of_variation = data_std / data_mean
                    result.add_info('coefficient_of_variation', float(coefficient_of_variation))
                    
                    if coefficient_of_variation > 0.5:
                        result.add_warning(
                            f"FLAT frame has high non-uniformity (CV={coefficient_of_variation:.3f})"
                        )
            
            if result.is_valid:
                logger.debug("FLAT frame validation passed for %s", filepath)
    
    except Exception as e:
        result.add_error(f"Failed to validate FLAT frame: {e}")
    
    return result


def validate_exposure_time(
    filepath: StrOrPath,
    expected_exptime: Optional[float] = None,
    tolerance: float = 0.01,
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate exposure time keyword.
    
    Args:
        filepath: Path to the FITS file.
        expected_exptime: Expected exposure time in seconds.
        tolerance: Allowed deviation from expected value. Defaults to 0.01.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult for exposure time.
    
    Examples:
        >>> # result = validate_exposure_time('dark.fits', expected_exptime=300.0)
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu >= len(hdul):
                result.add_error(f"HDU {hdu} does not exist")
                return result
            
            header = hdul[hdu].header
            
            if FITS_KEY_EXPTIME not in header:
                result.add_error(f"Missing keyword '{FITS_KEY_EXPTIME}'")
                return result
            
            exptime = header[FITS_KEY_EXPTIME]
            result.add_info('exptime', exptime)
            
            # Check for negative exposure time
            if exptime < 0:
                result.add_error(f"Exposure time cannot be negative: {exptime}")
            
            # Check against expected value if provided
            if expected_exptime is not None:
                diff = abs(exptime - expected_exptime)
                if diff > tolerance:
                    result.add_error(
                        f"Exposure time {exptime} differs from expected {expected_exptime} "
                        f"by {diff} (tolerance: {tolerance})"
                    )
            
            if result.is_valid:
                logger.debug("Exposure time validation passed: %ss", exptime)
    
    except Exception as e:
        result.add_error(f"Failed to validate exposure time: {e}")
    
    return result


# =============================================================================
# Multi-File Validation
# =============================================================================

def validate_frame_set(
    filepaths: List[StrOrPath],
    check_dimensions: bool = True,
    check_frame_type: bool = True,
    expected_frame_type: Optional[str] = None,
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate that a set of frames has consistent properties.
    
    Useful for ensuring calibration frame sets are homogeneous.
    
    Args:
        filepaths: List of FITS file paths.
        check_dimensions: Verify all frames have same dimensions.
        check_frame_type: Verify all frames have same IMGTYPE.
        expected_frame_type: Expected IMGTYPE value for all frames.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        ValidationResult for the set.
    
    Examples:
        >>> # files = ['bias_001.fits', 'bias_002.fits', 'bias_003.fits']
        >>> # result = validate_frame_set(files, expected_frame_type='BIAS')
    """
    result = ValidationResult()
    
    if not filepaths:
        result.add_error("Empty file list provided")
        return result
    
    result.add_info('num_files', len(filepaths))
    
    # Collect properties from all files
    shapes = []
    frame_types = []
    exptimes = []
    
    for filepath in filepaths:
        filepath = Path(filepath)
        if not filepath.exists():
            result.add_error(f"File not found: {filepath}")
            continue
        
        try:
            with fits.open(filepath, memmap=True) as hdul:
                if hdu >= len(hdul):
                    result.add_error(f"HDU {hdu} does not exist in {filepath}")
                    continue
                
                header = hdul[hdu].header
                data = hdul[hdu].data
                
                # Collect shape
                if data is not None:
                    shapes.append(data.shape)
                else:
                    result.add_warning(f"No data in {filepath}")
                
                # Collect frame type
                for keyword in [FITS_KEY_IMGTYPE, FITS_KEY_IMAGETYP]:
                    if keyword in header:
                        frame_types.append(normalize_image_type(header[keyword].strip()))
                        break
                
                # Collect exposure time
                if FITS_KEY_EXPTIME in header:
                    exptimes.append(header[FITS_KEY_EXPTIME])
        
        except Exception as e:
            result.add_error(f"Failed to read {filepath}: {e}")
    
    # Check dimensions consistency
    if check_dimensions and shapes:
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            result.add_error(f"Inconsistent image dimensions in set: {unique_shapes}")
        else:
            result.add_info('image_shape', list(unique_shapes)[0])
    
    # Check frame type consistency
    if check_frame_type and frame_types:
        unique_types = set(frame_types)
        if len(unique_types) > 1:
            result.add_error(f"Inconsistent frame types in set: {unique_types}")
        else:
            result.add_info('frame_type', list(unique_types)[0])
        
        # Check against expected type
        if expected_frame_type is not None:
            expected = normalize_image_type(expected_frame_type)
            if expected not in unique_types:
                result.add_error(
                    f"Expected frame type '{expected}', found: {unique_types}"
                )
    
    # Report exposure time statistics
    if exptimes:
        result.add_info('exptime_min', min(exptimes))
        result.add_info('exptime_max', max(exptimes))
        result.add_info('exptime_mean', float(np.mean(exptimes)))
        
        # Warn if exposure times vary significantly
        if len(set(exptimes)) > 1:
            result.add_warning(f"Exposure times vary in set: {set(exptimes)}")
    
    if result.is_valid:
        logger.debug("Frame set validation passed for %d files", len(filepaths))
    
    return result


def validate_calibration_set(
    bias_files: Optional[List[str]] = None,
    dark_files: Optional[List[str]] = None,
    flat_files: Optional[List[str]] = None,
) -> ValidationResult:
    """
    Validate a complete calibration frame set.
    
    Checks that bias, dark, and flat frame sets are internally consistent
    and compatible with each other.
    
    Args:
        bias_files: List of bias frame paths.
        dark_files: List of dark frame paths.
        flat_files: List of flat frame paths.
    
    Returns:
        Comprehensive validation results.
    
    Examples:
        >>> # result = validate_calibration_set(
        >>> #     bias_files=['bias_001.fits', 'bias_002.fits'],
        >>> #     dark_files=['dark_001.fits', 'dark_002.fits'],
        >>> #     flat_files=['flat_V_001.fits', 'flat_V_002.fits']
        >>> # )
    """
    result = ValidationResult()
    
    all_shapes = []
    
    # Validate bias frames
    if bias_files:
        bias_result = validate_frame_set(
            bias_files,
            check_frame_type=True,
            expected_frame_type='BIAS',
        )
        
        result.merge(bias_result, prefix="BIAS")

        result.add_info('num_bias_frames', len(bias_files))
        if 'image_shape' in bias_result.info:
            all_shapes.append(bias_result.info['image_shape'])
    
    # Validate dark frames
    if dark_files:
        dark_result = validate_frame_set(
            dark_files,
            check_frame_type=True,
            expected_frame_type='DARK',
        )
        
        result.merge(dark_result, prefix="DARK")
        
        result.add_info('num_dark_frames', len(dark_files))
        if 'image_shape' in dark_result.info:
            all_shapes.append(dark_result.info['image_shape'])
    
    # Validate flat frames
    if flat_files:
        flat_result = validate_frame_set(
            flat_files,
            check_frame_type=True,
            expected_frame_type='FLAT',
        )
        
        result.merge(flat_result, prefix="FLAT")
        
        result.add_info('num_flat_frames', len(flat_files))
        if 'image_shape' in flat_result.info:
            all_shapes.append(flat_result.info['image_shape'])
    
    # Check that all calibration frames have consistent dimensions
    if all_shapes:
        unique_shapes = set(tuple(s) for s in all_shapes)
        if len(unique_shapes) > 1:
            result.add_error(
                f"Calibration frames have inconsistent dimensions: {unique_shapes}"
            )
        else:
            result.add_info('calibration_shape', all_shapes[0])
    
    if result.is_valid:
        logger.debug("Calibration set validation passed")
    
    return result

# =============================================================================
# Configuration-Based Validation
# =============================================================================

def validate_against_schema(
    filepath: StrOrPath,
    schema: Dict[str, Any],
    hdu: int = 0,
) -> ValidationResult:
    """
    Validate a FITS file against a flexible configuration schema.
    
    Schema format:
    {
        'required_keywords': ['EXPTIME', 'IMAGETYP'],
        'keyword_values': {'IMAGETYP': ['BIAS', 'DARK']},
        'keyword_types': {'EXPTIME': float},
        'data_range': {'min': 0, 'max': 65535},
        'dimensions': {'min': (100, 100), 'max': (10000, 10000)},
        'no_nans': True
    }
    
    Args:
        filepath: Path to the FITS file.
        schema: Validation schema configuration.
        hdu: HDU extension to check. Defaults to 0.
    
    Returns:
        Comprehensive validation results.
    
    Examples:
        >>> # schema = {
        >>> #     'required_keywords': ['EXPTIME', 'IMAGETYP'],
        >>> #     'keyword_values': {'IMAGETYP': ['BIAS']},
        >>> #     'data_range': {'min': 0, 'max': 65535}
        >>> # }
        >>> # result = validate_against_schema('bias.fits', schema)
    """
    filepath = Path(filepath)
    result = ValidationResult()
    result.add_info('schema_validation', True)
    
    # Required keywords
    if 'required_keywords' in schema:
        kw_result = validate_required_keywords(
            filepath, schema['required_keywords'], hdu=hdu
        )
        if not kw_result.is_valid:
            result.merge(kw_result)
    
    # Keyword values
    if 'keyword_values' in schema:
        val_result = validate_header_values(
            filepath, schema['keyword_values'], hdu=hdu
        )
        if not val_result.is_valid:
            result.merge(val_result)
    
    # Keyword types
    if 'keyword_types' in schema:
        type_result = validate_keyword_types(
            filepath, schema['keyword_types'], hdu=hdu
        )
        if not type_result.is_valid:
            result.merge(type_result)
    
    # Data range
    if 'data_range' in schema:
        range_config = schema['data_range']
        range_result = validate_data_range(
            filepath,
            min_value=range_config.get('min'),
            max_value=range_config.get('max'),
            hdu=hdu,
        )
        result.merge(range_result)
    
    # Dimensions
    if 'dimensions' in schema:
        dim_config = schema['dimensions']
        dim_result = validate_image_dimensions(
            filepath,
            expected_shape=dim_config.get('expected'),
            min_shape=dim_config.get('min'),
            max_shape=dim_config.get('max'),
            hdu=hdu,
        )
        result.merge(dim_result)
    
    # NaN check
    if schema.get('no_nans', False):
        nan_result = validate_no_nans(filepath, hdu=hdu)
        result.merge(nan_result)
    
    if result.is_valid:
        logger.debug("Schema validation passed for %s", filepath)
    
    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_validate(
    filepath: StrOrPath, 
    frame_type: Optional[str] = None, 
) -> ValidationResult:
    """
    Quick validation with sensible defaults for CCD test data.
    
    Args:
        filepath: Path to the FITS file.
        frame_type: Expected frame type ('BIAS', 'DARK', 'FLAT').
            If None, will auto-detect from header.
    
    Returns:
        ValidationResult with comprehensive checks.
    
    Examples:
        >>> # result = quick_validate('bias_001.fits', frame_type='BIAS')
        >>> # print(result)
    """
    filepath = Path(filepath)
    result = ValidationResult()
    
    # Basic integrity
    integrity_result = validate_fits_file(filepath)
    if not integrity_result.is_valid:
        return integrity_result
    
    result.merge(integrity_result)
    
    # Detect frame type if not provided
    if frame_type is None:
        type_result = validate_frame_type(filepath)
        if type_result.is_valid and 'frame_type' in type_result.info:
            frame_type = type_result.info['frame_type']
    
    # Frame-specific validation
    if frame_type:
        frame_type = normalize_image_type(frame_type)
        
        if frame_type == FITS_VAL_BIAS_IMGTYPE:
            frame_result = validate_bias_frame(filepath)
        elif frame_type == FITS_VAL_DARK_IMGTYPE:
            frame_result = validate_dark_frame(filepath)
        elif frame_type == FITS_VAL_FLAT_IMGTYPE:
            frame_result = validate_flat_frame(filepath)
        else:
            # Generic validation
            frame_result = ValidationResult()
            frame_result.add_info('frame_type', frame_type)
        
        result.merge(frame_result)
    
    # Check for NaNs
    nan_result = validate_no_nans(filepath)
    result.merge(nan_result)
    
    return result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Result container
    'ValidationResult',
    
    # Basic validation
    'validate_fits_file',
    'check_fits_integrity',
    
    # Header validation
    'validate_required_keywords',
    'validate_header_values',
    'validate_keyword_types',
    
    # Data validation
    'validate_data_range',
    'validate_no_nans',
    'validate_image_dimensions',
    
    # CCD-specific validation
    'validate_frame_type',
    'validate_bias_frame',
    'validate_dark_frame',
    'validate_flat_frame',
    'validate_exposure_time',
    
    # Multi-file validation
    'validate_frame_set',
    'validate_calibration_set',
    
    # Schema-based validation
    'validate_against_schema',
    
    # Convenience
    'quick_validate',
]

