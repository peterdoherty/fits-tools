#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result types for image processing operations.

This module defines dataclasses for structured returns from various image
processing operations throughout the fits_tools.image_processing package.
Using dataclasses provides type safety, clear interfaces, and easy serialization.

Result Types:
    - CalibrationResult: Output from calibration workflows
    - CombiningResult: Output from image stacking/combining
    - ImageOperationResult: Output from single image operations
    - ProcessingResult: Output from batch processing
    - TransformationResult: Output from geometric transformations

All result types include:
    - Structured data with type hints
    - String representations for debugging
    - Dictionary/JSON serialization
    - Success/failure status tracking
    - Error and warning collection

Version: 2.0.0
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import numpy as np
from astropy.io import fits

# Import from fits_tools
from ..validate import ValidationResult
from ..exceptions import ImageProcessingError

__version__ = "2.0.0"
__author__ = "FITS Tools Development Team"


# ============================================================================
# Base Result Class (Optional - for shared functionality)
# ============================================================================

class BaseResult:
    """
    Base class providing common functionality for result types.
    
    Provides serialization, string representation, and success checking.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation with all fields.
        """
        result = asdict(self)
        
        # Handle special types that don't serialize directly
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                # Store shape and dtype info, not actual data
                result[key] = {
                    '_type': 'ndarray',
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                }
            elif isinstance(value, fits.Header):
                result[key] = dict(value)
            elif isinstance(value, ValidationResult):
                result[key] = {
                    'is_valid': value.is_valid,
                    'errors': value.errors,
                    'warnings': value.warnings,
                    'info': value.info,
                }
        
        return result
    
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
        
        return json_str
    
    def __bool__(self) -> bool:
        """Allow result objects to be used in boolean context."""
        return getattr(self, 'success', True)


# ============================================================================
# CalibrationResult
# ============================================================================

@dataclass
class CalibrationResult(BaseResult):
    """
    Result from image calibration workflow.
    
    Contains calibrated data, updated header, and metadata about what
    corrections were applied. Optionally includes validation results
    and quality metrics for quality control.
    
    Attributes
    ----------
    data : np.ndarray
        Calibrated image data.
    header : fits.Header
        Updated FITS header with calibration history.
    bias_corrected : bool, optional
        Whether bias correction was applied. Defaults to False.
    dark_corrected : bool, optional
        Whether dark correction was applied. Defaults to False.
    flat_corrected : bool, optional
        Whether flat field correction was applied. Defaults to False.
    validation : ValidationResult, optional
        Validation results from input checking.
    quality_metrics : dict, optional
        Quality metrics (SNR, dynamic range, etc.).
    input_file : str, optional
        Path to input raw file.
    output_file : str, optional
        Path to output calibrated file (if written).
    timestamp : str
        ISO format timestamp of processing.
    
    Examples
    --------
    >>> from fits_tools.image_processing import calibrate_image
    >>> result = calibrate_image('raw.fits', master_bias='bias.fits')
    >>> print(result.bias_corrected)
    True
    >>> print(result)
    Calibration Result:
      Shape: (2048, 2048)
      Corrections: bias
      Quality: SNR=25.3
    """
    data: np.ndarray
    header: fits.Header
    bias_corrected: bool = False
    dark_corrected: bool = False
    flat_corrected: bool = False
    validation: Optional[ValidationResult] = None
    quality_metrics: Optional[Dict[str, float]] = None
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate result after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if not isinstance(self.header, fits.Header):
            raise TypeError("header must be a fits.Header")
    
    @property
    def corrections_applied(self) -> List[str]:
        """List of corrections that were applied."""
        corrections = []
        if self.bias_corrected:
            corrections.append("bias")
        if self.dark_corrected:
            corrections.append("dark")
        if self.flat_corrected:
            corrections.append("flat")
        return corrections
    
    @property
    def success(self) -> bool:
        """Whether calibration was successful."""
        # Consider successful if validation passed or no validation was done
        if self.validation:
            return self.validation.is_valid
        return True
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        corrections = self.corrections_applied
        
        lines = [
            "Calibration Result:",
            f"  Shape: {self.data.shape}",
            f"  Corrections: {', '.join(corrections) if corrections else 'none'}",
        ]
        
        if self.input_file:
            lines.append(f"  Input: {os.path.basename(self.input_file)}")
        
        if self.output_file:
            lines.append(f"  Output: {os.path.basename(self.output_file)}")
        
        if self.quality_metrics:
            lines.append(f"  Quality Metrics:")
            for key, val in self.quality_metrics.items():
                lines.append(f"    {key}: {val:.2f}")
        
        if self.validation:
            status = "✓ VALID" if self.validation.is_valid else "✗ INVALID"
            lines.append(f"  Validation: {status}")
            if self.validation.errors:
                lines.append(f"    Errors: {len(self.validation.errors)}")
            if self.validation.warnings:
                lines.append(f"    Warnings: {len(self.validation.warnings)}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Brief representation."""
        corrections = "+".join(self.corrections_applied) or "none"
        return (
            f"CalibrationResult(shape={self.data.shape}, "
            f"corrections={corrections}, success={self.success})"
        )


# ============================================================================
# CombiningResult
# ============================================================================

@dataclass
class CombiningResult(BaseResult):
    """
    Result from image combining/stacking operations.
    
    Contains combined data and metadata about the combining process,
    including method used, number of images, and outlier statistics.
    
    Attributes
    ----------
    data : np.ndarray
        Combined image data.
    header : fits.Header
        Combined/updated FITS header.
    method : str
        Combining method used ('median', 'mean', 'sigma_clipped_median', etc.).
    num_images : int
        Number of images combined.
    num_clipped : int, optional
        Number of pixels clipped (if using sigma-clipping).
    clip_fraction : float, optional
        Fraction of pixels clipped.
    input_files : list of str, optional
        List of input file paths.
    output_file : str, optional
        Output file path (if written).
    timestamp : str
        ISO format timestamp of processing.
    
    Examples
    --------
    >>> from fits_tools.image_processing import combine_images_robust
    >>> files = ['img1.fits', 'img2.fits', 'img3.fits']
    >>> result = combine_images_robust(files, method='sigma_clipped_median')
    >>> print(f"Combined {result.num_images} images using {result.method}")
    """
    data: np.ndarray
    header: fits.Header
    method: str
    num_images: int
    num_clipped: int = 0
    clip_fraction: float = 0.0
    input_files: Optional[List[str]] = None
    output_file: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate result after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if self.num_images < 1:
            raise ValueError("num_images must be >= 1")
    
    @property
    def success(self) -> bool:
        """Whether combining was successful."""
        return self.num_images > 0 and self.data is not None
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            "Combining Result:",
            f"  Method: {self.method}",
            f"  Images combined: {self.num_images}",
            f"  Output shape: {self.data.shape}",
        ]
        
        if self.num_clipped > 0:
            lines.append(f"  Pixels clipped: {self.num_clipped} ({self.clip_fraction*100:.2f}%)")
        
        if self.output_file:
            lines.append(f"  Output: {os.path.basename(self.output_file)}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Brief representation."""
        return (
            f"CombiningResult(method={self.method}, num_images={self.num_images}, "
            f"shape={self.data.shape})"
        )


# ============================================================================
# ImageOperationResult
# ============================================================================

@dataclass
class ImageOperationResult(BaseResult):
    """
    Result from a single image operation.
    
    Used for operations like subtract, add, multiply, divide, unary operations,
    and transformations on individual images.
    
    Attributes
    ----------
    success : bool
        Whether operation succeeded.
    output_file : str, optional
        Output file path (if written to disk).
    data : np.ndarray, optional
        Result data (if kept in memory).
    header : fits.Header, optional
        Updated FITS header.
    operation : str, optional
        Name of operation performed.
    error : str, optional
        Error message if operation failed.
    metadata : dict, optional
        Additional operation-specific metadata.
    timestamp : str
        ISO format timestamp of processing.
    
    Examples
    --------
    >>> from fits_tools.image_processing import subtract_images
    >>> result = subtract_images('img1.fits', 'img2.fits', 'diff.fits')
    >>> if result.success:
    ...     print(f"Subtraction successful: {result.output_file}")
    """
    success: bool
    output_file: Optional[str] = None
    data: Optional[np.ndarray] = None
    header: Optional[fits.Header] = None
    operation: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.success:
            lines = [
                f"✓ Operation succeeded: {self.operation or 'unknown'}",
            ]
            if self.output_file:
                lines.append(f"  Output: {self.output_file}")
            if self.data is not None:
                lines.append(f"  Data shape: {self.data.shape}")
            if self.metadata:
                lines.append(f"  Metadata: {self.metadata}")
        else:
            lines = [
                f"✗ Operation failed: {self.operation or 'unknown'}",
                f"  Error: {self.error}",
            ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Brief representation."""
        status = "success" if self.success else "failed"
        return f"ImageOperationResult({status}, operation={self.operation})"


# ============================================================================
# ProcessingResult
# ============================================================================

@dataclass
class ProcessingResult(BaseResult):
    """
    Result from batch processing operations.
    
    Aggregates results from processing multiple files, tracking successes,
    failures, and output files.
    
    Attributes
    ----------
    success_count : int
        Number of successfully processed files.
    failure_count : int
        Number of failed files.
    output_files : list of str
        List of output file paths created.
    errors : list of str
        List of error messages for failed files.
    warnings : list of str
        List of warning messages.
    metadata : dict
        Additional batch processing metadata.
    timestamp : str
        ISO format timestamp of processing.
    
    Examples
    --------
    >>> from fits_tools.image_processing import process_batch
    >>> files = ['img1.fits', 'img2.fits', 'img3.fits']
    >>> result = process_batch(files, operations=['sqrt'], output_dir='processed/')
    >>> print(f"Success rate: {result.success_rate*100:.1f}%")
    """
    success_count: int = 0
    failure_count: int = 0
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def total_count(self) -> int:
        """Total number of files processed."""
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        """Fraction of files successfully processed."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
    
    @property
    def success(self) -> bool:
        """Whether all files were processed successfully."""
        return self.failure_count == 0 and self.success_count > 0
    
    def add_success(self, output_file: str) -> None:
        """Record a successful file processing."""
        self.success_count += 1
        self.output_files.append(output_file)
    
    def add_failure(self, error_message: str) -> None:
        """Record a failed file processing."""
        self.failure_count += 1
        self.errors.append(error_message)
    
    def add_warning(self, warning_message: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning_message)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            "Processing Result:",
            f"  Total files: {self.total_count}",
            f"  Successful: {self.success_count} ({self.success_rate*100:.1f}%)",
            f"  Failed: {self.failure_count}",
            f"  Output files: {len(self.output_files)}",
        ]
        
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:5], 1):  # Show first 5
                lines.append(f"  {i}. {error}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")
        
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:5], 1):
                lines.append(f"  {i}. {warning}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Brief representation."""
        return (
            f"ProcessingResult(total={self.total_count}, "
            f"success={self.success_count}, failed={self.failure_count})"
        )


# ============================================================================
# TransformationResult
# ============================================================================

@dataclass
class TransformationResult(BaseResult):
    """
    Result from geometric or intensity transformation.
    
    Used for operations like crop, bin, rotate, normalize that transform
    the image geometry or intensity distribution.
    
    Attributes
    ----------
    data : np.ndarray
        Transformed image data.
    header : fits.Header
        Updated FITS header.
    transformation : str
        Type of transformation applied.
    input_shape : tuple
        Original image shape.
    output_shape : tuple
        Transformed image shape.
    parameters : dict
        Transformation parameters used.
    output_file : str, optional
        Output file path (if written).
    timestamp : str
        ISO format timestamp of processing.
    
    Examples
    --------
    >>> from fits_tools.image_processing import crop_image
    >>> data = np.random.random((1000, 1000))
    >>> result = crop_image(data, region=(100, 900, 100, 900))
    >>> print(f"Cropped from {result.input_shape} to {result.output_shape}")
    """
    data: np.ndarray
    header: fits.Header
    transformation: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_file: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success(self) -> bool:
        """Whether transformation was successful."""
        return self.data is not None and self.data.size > 0
    
    @property
    def shape_changed(self) -> bool:
        """Whether transformation changed the image shape."""
        return self.input_shape != self.output_shape
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Transformation Result: {self.transformation}",
            f"  Input shape: {self.input_shape}",
            f"  Output shape: {self.output_shape}",
        ]
        
        if self.parameters:
            lines.append(f"  Parameters:")
            for key, val in self.parameters.items():
                lines.append(f"    {key}: {val}")
        
        if self.output_file:
            lines.append(f"  Output: {os.path.basename(self.output_file)}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Brief representation."""
        return (
            f"TransformationResult(transformation={self.transformation}, "
            f"shape={self.input_shape}->{self.output_shape})"
        )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'BaseResult',
    'CalibrationResult',
    'CombiningResult',
    'ImageOperationResult',
    'ProcessingResult',
    'TransformationResult',
]


# ============================================================================
# Utility Functions
# ============================================================================

def create_success_result(
    output_file: str,
    operation: str,
    **metadata
) -> ImageOperationResult:
    """
    Convenience function to create a success result.
    
    Parameters
    ----------
    output_file : str
        Path to output file.
    operation : str
        Name of operation.
    **metadata
        Additional metadata.
    
    Returns
    -------
    ImageOperationResult
        Success result.
    """
    return ImageOperationResult(
        success=True,
        output_file=output_file,
        operation=operation,
        metadata=metadata
    )


def create_failure_result(
    error: str,
    operation: str,
    **metadata
) -> ImageOperationResult:
    """
    Convenience function to create a failure result.
    
    Parameters
    ----------
    error : str
        Error message.
    operation : str
        Name of operation.
    **metadata
        Additional metadata.
    
    Returns
    -------
    ImageOperationResult
        Failure result.
    """
    return ImageOperationResult(
        success=False,
        error=error,
        operation=operation,
        metadata=metadata
    )