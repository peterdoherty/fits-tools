"""Result types for image processing operations.

This module defines dataclasses for structured returns from image processing
operations throughout the :mod:`fits_tools.image_processing` package.
Dataclasses provide type safety, clear interfaces, and straightforward
serialization.

Result Types
------------
- :class:`CalibrationResult` — single-image calibration workflow.
- :class:`CalibrationBatchResult` — batch calibration of multiple files.
- :class:`CombiningResult` — image stacking/combining.
- :class:`ImageOperationResult` — single image operation (subtract, add,
  transform, etc.).
- :class:`ProcessingResult` — generic batch processing.
- :class:`TransformationResult` — geometric or intensity transformation.

All result types provide:

- Structured fields with type hints.
- ``__str__`` and ``__repr__`` for debugging.
- ``to_dict()`` / ``to_json()`` serialization.
- ``success`` property for boolean status checking.
- Error and warning collection where applicable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from ..types import StrOrPath
from ..validate import ValidationResult

__all__ = [
    "BaseResult",
    "CalibrationResult",
    "CalibrationBatchResult",
    "CombiningResult",
    "ImageOperationResult",
    "ProcessingResult",
    "TransformationResult",
    "create_success_result",
    "create_failure_result",
]


# =============================================================================
# Base Result
# =============================================================================


class BaseResult:
    """Base class providing common serialization for dataclass result types.

    Subclasses must be dataclasses for :meth:`to_dict` and :meth:`to_json`
    to work correctly, since both rely on :func:`dataclasses.asdict`.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a plain dictionary.

        Returns
        -------
        dict
            All dataclass fields serialized. ``np.ndarray`` values are
            replaced by a shape/dtype summary dict. ``fits.Header`` values
            are converted to plain dicts. :class:`ValidationResult` values
            are expanded to their component fields.
        """
        result = asdict(self)  # type: ignore[call-overload]

        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = {
                    "_type": "ndarray",
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                }
            elif isinstance(value, fits.Header):
                result[key] = dict(value)
            elif isinstance(value, ValidationResult):
                result[key] = {
                    "is_valid": value.is_valid,
                    "errors":   value.errors,
                    "warnings": value.warnings,
                    "info":     value.info,
                }

        return result

    def to_json(
        self,
        filepath: StrOrPath | None = None,
        indent: int = 2,
    ) -> str:
        """Serialize result to JSON, optionally writing to a file.

        Parameters
        ----------
        filepath : str or Path, optional
            If provided, the JSON string is written to this path.
        indent : int, optional
            JSON indentation level. Default is 2.

        Returns
        -------
        str
            JSON-encoded representation of the result.
        """
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)
        if filepath is not None:
            Path(filepath).write_text(json_str, encoding="utf-8")
        return json_str

    def __bool__(self) -> bool:
        """Return the ``success`` attribute when used in a boolean context."""
        return getattr(self, "success", True)


# =============================================================================
# CalibrationResult
# =============================================================================


@dataclass
class CalibrationResult(BaseResult):
    """Result from a single-image calibration workflow.

    Contains calibrated data, an updated header, and metadata describing
    which corrections were applied. Optionally includes validation results
    and quality metrics.

    Parameters
    ----------
    data : np.ndarray
        Calibrated image data.
    header : fits.Header
        Updated FITS header with calibration history.
    bias_corrected : bool, optional
        Whether bias correction was applied. Default is False.
    dark_corrected : bool, optional
        Whether dark correction was applied. Default is False.
    flat_corrected : bool, optional
        Whether flat field correction was applied. Default is False.
    bpm : np.ndarray, optional
        Bad pixel mask applied during calibration.
    bpm_file : Path, optional
        Path to the BPM file used.
    validation : ValidationResult, optional
        Validation results from input checking.
    quality_metrics : dict, optional
        Quality metrics (SNR, dynamic range, etc.).
    input_file : Path, optional
        Path to the raw input file.
    output_file : Path, optional
        Path to the calibrated output file (if written).
    timestamp : str
        ISO 8601 UTC timestamp of processing.

    Examples
    --------
    >>> result = calibrate_image("raw.fits", master_bias="bias.fits")
    >>> print(result.bias_corrected)
    True
    """

    data: np.ndarray
    header: fits.Header
    bias_corrected: bool = False
    dark_corrected: bool = False
    flat_corrected: bool = False
    bpm: np.ndarray | None = None
    bpm_file: Path | None = None
    validation: ValidationResult | None = None
    quality_metrics: dict[str, Any] | None = None
    input_file: Path | None = None
    output_file: Path | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy array.")
        if not isinstance(self.header, fits.Header):
            raise TypeError("header must be a fits.Header.")

    @property
    def corrections_applied(self) -> list[str]:
        """Corrections that were applied, in bias → dark → flat order."""
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
        """True if validation passed or no validation was performed."""
        if self.validation is not None:
            return self.validation.is_valid
        return True

    def __str__(self) -> str:
        corrections = self.corrections_applied
        lines = [
            "Calibration Result:",
            f"  Shape:       {self.data.shape}",
            f"  Corrections: {', '.join(corrections) if corrections else 'none'}",
        ]
        if self.input_file:
            lines.append(f"  Input:  {Path(self.input_file).name}")
        if self.output_file:
            lines.append(f"  Output: {Path(self.output_file).name}")
        if self.quality_metrics:
            lines.append("  Quality Metrics:")
            for key, val in self.quality_metrics.items():
                if isinstance(val, float):
                    lines.append(f"    {key}: {val:.2f}")
                else:
                    lines.append(f"    {key}: {val}")
        if self.validation is not None:
            status = "✓ VALID" if self.validation.is_valid else "✗ INVALID"
            lines.append(f"  Validation: {status}")
            if self.validation.errors:
                lines.append(f"    Errors:   {len(self.validation.errors)}")
            if self.validation.warnings:
                lines.append(f"    Warnings: {len(self.validation.warnings)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        corrections = "+".join(self.corrections_applied) or "none"
        return (
            f"CalibrationResult(shape={self.data.shape}, "
            f"corrections={corrections}, success={self.success})"
        )


# =============================================================================
# CalibrationBatchResult
# =============================================================================


@dataclass
class CalibrationBatchResult(BaseResult):
    """Result from a batch calibration of multiple raw files.

    Aggregates per-file :class:`CalibrationResult` objects alongside
    failure messages and batch-level warnings.

    Parameters
    ----------
    results : list of CalibrationResult, optional
        Successfully calibrated files. Default is empty list.
    errors : list of str, optional
        Error messages for failed files, typically
        ``"<raw_file>: <error message>"``. Default is empty list.
    warnings : list of str, optional
        Batch-level warning messages. Default is empty list.
    metadata : dict, optional
        Additional batch processing metadata. Default is empty dict.
    timestamp : str
        ISO 8601 UTC timestamp of batch processing.

    Properties
    ----------
    total_count : int
        Total files attempted (success + failure).
    success_count : int
        Number of successfully calibrated files.
    failure_count : int
        Number of files that failed.
    success_rate : float
        Fraction successfully calibrated (0.0–1.0).
    success : bool
        True if all files succeeded and at least one was processed.
    """

    results:   list[CalibrationResult] = field(default_factory=list)
    errors:    list[str]               = field(default_factory=list)
    warnings:  list[str]               = field(default_factory=list)
    metadata:  dict[str, Any]          = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_count(self) -> int:
        """Total number of files processed (success + failure)."""
        return self.success_count + self.failure_count

    @property
    def success_count(self) -> int:
        """Number of successfully calibrated files."""
        return len(self.results)

    @property
    def failure_count(self) -> int:
        """Number of files that failed calibration."""
        return len(self.errors)

    @property
    def success_rate(self) -> float:
        """Fraction of files successfully calibrated."""
        return self.success_count / self.total_count if self.total_count else 0.0

    @property
    def success(self) -> bool:
        """True if all files were calibrated successfully."""
        return self.failure_count == 0 and self.success_count > 0

    def add_success(self, result: CalibrationResult) -> None:
        """Record a successful calibration result.

        Parameters
        ----------
        result : CalibrationResult
            Result to append to :attr:`results`.
        """
        self.results.append(result)

    def add_failure(
        self,
        raw_file: StrOrPath,
        error: Exception | str,
    ) -> None:
        """Record a failed calibration.

        Parameters
        ----------
        raw_file : str or Path
            Path to the file that failed.
        error : Exception or str
            The error that occurred.
        """
        self.errors.append(f"{raw_file}: {error}")

    def add_warning(self, message: str) -> None:
        """Append a batch-level warning message.

        Parameters
        ----------
        message : str
            Warning text to record.
        """
        self.warnings.append(message)

    def __str__(self) -> str:
        lines = [
            "Calibration Batch Result:",
            f"  Total:      {self.total_count}",
            f"  Successful: {self.success_count} ({self.success_rate * 100:.1f}%)",
            f"  Failed:     {self.failure_count}",
        ]
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for i, e in enumerate(self.errors[:5], 1):
                lines.append(f"  {i}. {e}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for i, w in enumerate(self.warnings[:5], 1):
                lines.append(f"  {i}. {w}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CalibrationBatchResult(total={self.total_count}, "
            f"success={self.success_count}, failed={self.failure_count})"
        )


# =============================================================================
# CombiningResult
# =============================================================================


@dataclass
class CombiningResult(BaseResult):
    """Result from an image combining/stacking operation.

    Parameters
    ----------
    data : np.ndarray
        Combined image data.
    header : fits.Header
        Combined or updated FITS header.
    method : str
        Combining method used (e.g. ``"median"``, ``"sigma_clipped_median"``).
    num_images : int
        Number of images combined.
    num_clipped : int, optional
        Number of pixels clipped (sigma-clipping only). Default is 0.
    clip_fraction : float, optional
        Fraction of pixels clipped. Default is 0.0.
    input_files : list of str or Path, optional
        Paths to the input files.
    output_file : str or Path, optional
        Output file path (if written to disk).
    timestamp : str
        ISO 8601 UTC timestamp of processing.
    """

    data: np.ndarray
    header: fits.Header
    method: str
    num_images: int
    num_clipped: int = 0
    clip_fraction: float = 0.0
    input_files: list[StrOrPath] | None = None
    output_file: StrOrPath | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy array.")
        if self.num_images < 1:
            raise ValueError("num_images must be >= 1.")

    @property
    def success(self) -> bool:
        """True if combining produced a non-empty result."""
        return self.num_images > 0 and self.data is not None

    def __str__(self) -> str:
        lines = [
            "Combining Result:",
            f"  Method:  {self.method}",
            f"  Images:  {self.num_images}",
            f"  Shape:   {self.data.shape}",
        ]
        if self.num_clipped > 0:
            lines.append(
                f"  Clipped: {self.num_clipped} "
                f"({self.clip_fraction * 100:.2f}%)"
            )
        if self.output_file:
            lines.append(f"  Output: {Path(self.output_file).name}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CombiningResult(method={self.method!r}, "
            f"num_images={self.num_images}, shape={self.data.shape})"
        )


# =============================================================================
# ImageOperationResult
# =============================================================================


@dataclass
class ImageOperationResult(BaseResult):
    """Result from a single image operation.

    Used for operations such as arithmetic (subtract, add, multiply,
    divide), unary operations, and transformations on individual images.

    Parameters
    ----------
    success : bool
        Whether the operation succeeded.
    output_file : str or Path, optional
        Output file path (if written to disk).
    data : np.ndarray, optional
        Result data (if retained in memory).
    header : fits.Header, optional
        Updated FITS header.
    operation : str, optional
        Name of the operation performed.
    error : str, optional
        Error message if the operation failed.
    metadata : dict, optional
        Additional operation-specific metadata.
    timestamp : str
        ISO 8601 UTC timestamp of processing.
    """

    success: bool
    output_file: StrOrPath | None = None
    data: np.ndarray | None = None
    header: fits.Header | None = None
    operation: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __bool__(self) -> bool:
        return self.success

    def __str__(self) -> str:
        op = self.operation or "unknown"
        if self.success:
            lines = [f"✓ Operation succeeded: {op}"]
            if self.output_file:
                lines.append(f"  Output: {self.output_file}")
            if self.data is not None:
                lines.append(f"  Shape:  {self.data.shape}")
            if self.metadata:
                lines.append(f"  Metadata: {self.metadata}")
        else:
            lines = [
                f"✗ Operation failed: {op}",
                f"  Error: {self.error}",
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "success" if self.success else "failed"
        return f"ImageOperationResult({status}, operation={self.operation!r})"


# =============================================================================
# ProcessingResult
# =============================================================================


@dataclass
class ProcessingResult(BaseResult):
    """Result from a generic batch processing operation.

    Aggregates per-file outcomes, tracking successes, failures, output
    paths, and diagnostic messages.

    Parameters
    ----------
    success_count : int, optional
        Number of successfully processed files. Default is 0.
    failure_count : int, optional
        Number of files that failed. Default is 0.
    output_files : list of Path, optional
        Paths to output files created. Default is empty list.
    errors : list of str, optional
        Error messages for failed files. Default is empty list.
    warnings : list of str, optional
        Warning messages. Default is empty list.
    metadata : dict, optional
        Additional batch processing metadata. Default is empty dict.
    timestamp : str
        ISO 8601 UTC timestamp of processing.
    """

    success_count: int = 0
    failure_count: int = 0
    output_files: list[Path] = field(default_factory=list)
    errors:   list[str]      = field(default_factory=list)
    warnings: list[str]      = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_count(self) -> int:
        """Total number of files processed."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Fraction of files successfully processed."""
        return self.success_count / self.total_count if self.total_count else 0.0

    @property
    def success(self) -> bool:
        """True if all files were processed successfully."""
        return self.failure_count == 0 and self.success_count > 0

    def add_success(self, output_file: StrOrPath) -> None:
        """Record a successful file processing.

        Parameters
        ----------
        output_file : str or Path
            Path to the output file produced.
        """
        self.success_count += 1
        self.output_files.append(Path(output_file))

    def add_failure(self, error_message: str) -> None:
        """Record a failed file processing.

        Parameters
        ----------
        error_message : str
            Description of the failure.
        """
        self.failure_count += 1
        self.errors.append(error_message)

    def add_warning(self, warning_message: str) -> None:
        """Append a warning message.

        Parameters
        ----------
        warning_message : str
            Warning text to record.
        """
        self.warnings.append(warning_message)

    def __str__(self) -> str:
        lines = [
            "Processing Result:",
            f"  Total:        {self.total_count}",
            f"  Successful:   {self.success_count} "
            f"({self.success_rate * 100:.1f}%)",
            f"  Failed:       {self.failure_count}",
            f"  Output files: {len(self.output_files)}",
        ]
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for i, e in enumerate(self.errors[:5], 1):
                lines.append(f"  {i}. {e}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for i, w in enumerate(self.warnings[:5], 1):
                lines.append(f"  {i}. {w}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ProcessingResult(total={self.total_count}, "
            f"success={self.success_count}, failed={self.failure_count})"
        )


# =============================================================================
# TransformationResult
# =============================================================================


@dataclass
class TransformationResult(BaseResult):
    """Result from a geometric or intensity transformation.

    Used for operations such as crop, bin, rotate, and normalize that
    change image geometry or intensity distribution.

    Parameters
    ----------
    data : np.ndarray
        Transformed image data.
    header : fits.Header
        Updated FITS header.
    transformation : str
        Name of the transformation applied (e.g. ``"crop"``, ``"bin"``).
    input_shape : tuple of int
        Original image shape.
    output_shape : tuple of int
        Transformed image shape.
    parameters : dict, optional
        Transformation parameters used. Default is empty dict.
    output_file : str or Path, optional
        Output file path (if written to disk).
    timestamp : str
        ISO 8601 UTC timestamp of processing.
    """

    data: np.ndarray
    header: fits.Header
    transformation: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    output_file: StrOrPath | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def success(self) -> bool:
        """True if the transformation produced a non-empty result."""
        return self.data is not None and self.data.size > 0

    @property
    def shape_changed(self) -> bool:
        """True if the transformation changed the image shape."""
        return self.input_shape != self.output_shape

    def __str__(self) -> str:
        lines = [
            f"Transformation Result: {self.transformation}",
            f"  Input shape:  {self.input_shape}",
            f"  Output shape: {self.output_shape}",
        ]
        if self.parameters:
            lines.append("  Parameters:")
            for key, val in self.parameters.items():
                lines.append(f"    {key}: {val}")
        if self.output_file:
            lines.append(f"  Output: {Path(self.output_file).name}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TransformationResult(transformation={self.transformation!r}, "
            f"shape={self.input_shape}->{self.output_shape})"
        )


# =============================================================================
# Convenience constructors
# =============================================================================


def create_success_result(
    output_file: StrOrPath,
    operation: str,
    **metadata: Any,
) -> ImageOperationResult:
    """Create a successful :class:`ImageOperationResult`.

    Parameters
    ----------
    output_file : str or Path
        Path to the output file produced.
    operation : str
        Name of the operation performed.
    **metadata
        Additional key/value metadata to include.

    Returns
    -------
    ImageOperationResult
        Result with ``success=True``.
    """
    return ImageOperationResult(
        success=True,
        output_file=output_file,
        operation=operation,
        metadata=dict(metadata),
    )


def create_failure_result(
    error: str,
    operation: str,
    **metadata: Any,
) -> ImageOperationResult:
    """Create a failed :class:`ImageOperationResult`.

    Parameters
    ----------
    error : str
        Error message describing the failure.
    operation : str
        Name of the operation that failed.
    **metadata
        Additional key/value metadata to include.

    Returns
    -------
    ImageOperationResult
        Result with ``success=False``.
    """
    return ImageOperationResult(
        success=False,
        error=error,
        operation=operation,
        metadata=dict(metadata),
    )
