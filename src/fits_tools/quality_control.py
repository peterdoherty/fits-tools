#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Quality control and metrics for CCD/CMOS calibrated images.

This module provides quality assessment tools that integrate validation
(:mod:`fits_tools.validate`) with statistical analysis
(:mod:`fits_tools.statistics`) to evaluate the quality of calibrated
images and calibration workflows.

Key Features
------------
- Per-image quality metrics via :func:`assess_quality` (file) and
  :func:`assess_quality_array` (in-memory array)
- Configurable quality thresholds via :class:`QualityThresholds`
- Calibration improvement measurement via :func:`assess_calibration_quality`
- Batch assessment and reporting via :func:`batch_quality_assessment` and
  :func:`generate_quality_report`
- Full calibration-set assessment via :func:`assess_calibration_set_quality`
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip

from .constants import (
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
)
from .exceptions import ImageProcessingError
from .hdu import HDUSelector
from .image_types import normalize_image_type
from .io import get_data, get_data_and_header
from .statistics import (
    calculate_snr,
    estimate_read_noise,
    robust_stats,
    stats,
)
from .types import StrOrPath
from .validate import ValidationResult, quick_validate

logger = logging.getLogger(__name__)

__all__ = [
    "QualityMetrics",
    "QualityThresholds",
    "DEFAULT_THRESHOLDS",
    "assess_quality",
    "assess_quality_array",
    "assess_calibration_quality",
    "batch_quality_assessment",
    "generate_quality_report",
    "assess_calibration_set_quality",
]


# =============================================================================
# Quality Metrics Dataclass
# =============================================================================


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a single image.

    Attributes
    ----------
    filepath : Path
        Path to the assessed file (or ``Path("in-memory")`` for arrays).
    frame_type : str
        Normalized image type, e.g. ``"BIAS"``, ``"FLAT"``.
    mean, median, std, mad : float
        Basic statistics of the good-pixel population.
    min_val, max_val : float
        Pixel value extremes.
    dynamic_range : float or None
        max / min; None if min is zero or not computed.
    snr : float or None
        Signal-to-noise ratio; None if not applicable or not computed.
    read_noise : float or None
        Estimated read noise in ADU; populated for BIAS frames only.
    num_outliers : int
        Number of sigma-clipped outlier pixels.
    outlier_fraction : float
        Fraction of total pixels flagged as outliers.
    num_negative_pixels : int
        Number of pixels with negative values.
    num_saturated_pixels : int
        Number of pixels above the saturation threshold.
    saturation_fraction : float
        Fraction of total pixels above the saturation threshold.
    validation : ValidationResult or None
        File-level validation result (None for in-memory assessments).
    passed : bool
        True if all quality thresholds are satisfied.
    quality_grade : str
        Letter grade: ``EXCELLENT``, ``GOOD``, ``ACCEPTABLE``, ``POOR``,
        or ``FAIL``.
    quality_score : float
        Numerical score in [0, 100].
    warnings, errors : list of str
        Non-fatal and fatal quality messages.
    timestamp : str
        ISO 8601 UTC timestamp of assessment.
    image_shape : tuple of int or None
        ``(rows, cols)`` of the assessed image.
    total_pixels : int
        Number of good (non-masked, finite) pixels analysed.
    """

    # Identification
    filepath: Path
    frame_type: str = "UNKNOWN"

    # Basic statistics
    mean:    float = 0.0
    median:  float = 0.0
    std:     float = 0.0
    mad:     float = 0.0

    # Range metrics
    min_val:       float = 0.0
    max_val:       float = 0.0
    dynamic_range: float | None = None

    # Signal quality
    snr:        float | None = None
    read_noise: float | None = None

    # Outlier detection
    num_outliers:        int   = 0
    outlier_fraction:    float = 0.0
    num_negative_pixels: int   = 0
    num_saturated_pixels: int  = 0
    saturation_fraction: float = 0.0

    # Quality assessment
    validation:    ValidationResult | None = None
    passed:        bool  = True
    quality_grade: str   = "UNKNOWN"
    quality_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors:   list[str] = field(default_factory=list)

    # Metadata
    timestamp:    str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    image_shape:  tuple[int, int] | None = None
    total_pixels: int = 0

    def __post_init__(self) -> None:
        if self.total_pixels > 0:
            self.outlier_fraction = self.num_outliers / self.total_pixels
            self.saturation_fraction = (
                self.num_saturated_pixels / self.total_pixels
            )

    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        lines = [
            "=" * 70,
            f"Quality Assessment Report: {status} {self.quality_grade}",
            "=" * 70,
            f"File:      {self.filepath.name}",
            f"Type:      {self.frame_type}",
            f"Score:     {self.quality_score:.1f}/100",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        if self.image_shape:
            lines.append(
                f"Image: {self.image_shape[0]}×{self.image_shape[1]} "
                f"({self.total_pixels:,} pixels)"
            )
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

        has_issues = (
            self.num_outliers > 0
            or self.num_negative_pixels > 0
            or self.num_saturated_pixels > 0
        )
        if has_issues:
            lines.append("")
            lines.append("Outliers & Issues:")
            if self.num_outliers > 0:
                lines.append(
                    f"  Outliers:  {self.num_outliers:>6d} "
                    f"({self.outlier_fraction * 100:>5.2f}%)"
                )
            if self.num_negative_pixels > 0:
                lines.append(
                    f"  Negative:  {self.num_negative_pixels:>6d}"
                )
            if self.num_saturated_pixels > 0:
                lines.append(
                    f"  Saturated: {self.num_saturated_pixels:>6d} "
                    f"({self.saturation_fraction * 100:>5.2f}%)"
                )

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

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QualityMetrics(file={self.filepath.name!r}, "
            f"grade={self.quality_grade!r}, "
            f"score={self.quality_score:.1f}, "
            f"passed={self.passed})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a plain dictionary.

        Returns
        -------
        dict
            All fields serialized. The ``validation`` field is expanded
            to a nested dict rather than kept as a :class:`ValidationResult`.
        """
        data = asdict(self)
        if self.validation is not None:
            data["validation"] = {
                "is_valid": self.validation.is_valid,
                "errors":   self.validation.errors,
                "warnings": self.validation.warnings,
                "info":     self.validation.info,
            }
        return data

    def to_json(
        self,
        filepath: StrOrPath | None = None,
        indent: int = 2,
    ) -> str:
        """Serialize metrics to a JSON string, optionally writing to a file.

        Parameters
        ----------
        filepath : str or Path, optional
            If provided, the JSON string is written to this path.
        indent : int, optional
            JSON indentation level. Default is 2.

        Returns
        -------
        str
            JSON-encoded metrics.
        """
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)
        if filepath is not None:
            Path(filepath).write_text(json_str, encoding="utf-8")
            logger.info("Quality metrics exported to: %s", filepath)
        return json_str


# =============================================================================
# Quality Thresholds
# =============================================================================


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds and weights for quality scoring.

    Parameters
    ----------
    min_snr : float
        Minimum acceptable SNR. Default is 5.0.
    max_outlier_fraction : float
        Maximum acceptable fraction of outlier pixels. Default is 0.01.
    max_saturation_fraction : float
        Maximum acceptable fraction of saturated pixels. Default is 0.05.
    saturation_threshold : float
        Pixel value above which a pixel is considered saturated.
        Default is 60000.0.
    max_negative_fraction : float
        Maximum acceptable fraction of negative pixels. Default is 0.001.
    min_dynamic_range : float
        Minimum acceptable dynamic range (max / min). Default is 10.0.
    weight_validation : float
        Score weight for file-level validation. Default is 0.3.
    weight_snr : float
        Score weight for SNR. Default is 0.2.
    weight_outliers : float
        Score weight for outlier fraction. Default is 0.2.
    weight_statistics : float
        Score weight for statistical uniformity. Default is 0.3.

    Notes
    -----
    Weights are automatically normalized if they do not sum to 1.0
    (within a tolerance of 0.01). A warning is logged when normalization
    occurs.
    """

    min_snr:                  float = 5.0
    max_outlier_fraction:     float = 0.01
    max_saturation_fraction:  float = 0.05
    saturation_threshold:     float = 60000.0
    max_negative_fraction:    float = 0.001
    min_dynamic_range:        float = 10.0

    weight_validation: float = 0.3
    weight_snr:        float = 0.2
    weight_outliers:   float = 0.2
    weight_statistics: float = 0.3

    def __post_init__(self) -> None:
        if self.min_snr < 0:
            raise ValueError("min_snr must be >= 0.")
        if not 0.0 <= self.max_outlier_fraction <= 1.0:
            raise ValueError("max_outlier_fraction must be in [0, 1].")
        if not 0.0 <= self.max_saturation_fraction <= 1.0:
            raise ValueError("max_saturation_fraction must be in [0, 1].")
        if self.saturation_threshold <= 0:
            raise ValueError("saturation_threshold must be > 0.")
        if not 0.0 <= self.max_negative_fraction <= 1.0:
            raise ValueError("max_negative_fraction must be in [0, 1].")
        if self.min_dynamic_range < 0:
            raise ValueError("min_dynamic_range must be >= 0.")

        weights = (
            self.weight_validation,
            self.weight_snr,
            self.weight_outliers,
            self.weight_statistics,
        )
        if any(w < 0 for w in weights):
            raise ValueError("All quality score weights must be >= 0.")

        total = sum(weights)
        if total <= 0:
            raise ValueError("At least one quality score weight must be > 0.")

        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Quality score weights sum to %.3f, not 1.0. Normalizing.",
                total,
            )
            # object.__setattr__ is required to mutate a frozen dataclass.
            object.__setattr__(
                self, "weight_validation", self.weight_validation / total
            )
            object.__setattr__(self, "weight_snr", self.weight_snr / total)
            object.__setattr__(
                self, "weight_outliers", self.weight_outliers / total
            )
            object.__setattr__(
                self, "weight_statistics", self.weight_statistics / total
            )


DEFAULT_THRESHOLDS = QualityThresholds()


# =============================================================================
# Core Quality Assessment
# =============================================================================


def assess_quality(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
    frame_type: str | None = None,
    thresholds: QualityThresholds | None = None,
    region: str = "full",
    compute_snr: bool = True,
    bpm: np.ndarray | None = None,
) -> QualityMetrics:
    """Perform comprehensive quality assessment of a FITS image file.

    Combines file-level validation (:func:`~fits_tools.validate.quick_validate`)
    with array-level analysis (:func:`assess_quality_array`).

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file to assess.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0.
    frame_type : str, optional
        Expected image type (e.g. ``"BIAS"``). If None, the type is read
        from the header.
    thresholds : QualityThresholds, optional
        Quality thresholds and scoring weights. Defaults to
        :data:`DEFAULT_THRESHOLDS`.
    region : str, optional
        Image region to analyse. Default is ``"full"``.
    compute_snr : bool, optional
        If True, compute SNR and dynamic range metrics. Default is True.
    bpm : np.ndarray, optional
        Bad pixel mask (0 = good, non-zero = bad). If provided, masked
        pixels are excluded from all statistics.

    Returns
    -------
    QualityMetrics
        Comprehensive quality metrics including file validation results.

    Raises
    ------
    ImageProcessingError
        If the file cannot be loaded or assessment fails.
    """
    filepath = Path(filepath)

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    logger.debug("Assessing quality of: %s", filepath)

    try:
        validation = quick_validate(filepath, frame_type=frame_type)

        resolved_frame_type = frame_type
        if "frame_type" in validation.info:
            resolved_frame_type = validation.info["frame_type"]

        data, header = get_data_and_header(filepath, hdu=hdu)
        if data is None:
            raise ImageProcessingError(
                f"Could not load data from {filepath}"
            )

        if region != "full":
            data = get_data(filepath, hdu=hdu, region=region)
            if data is None:
                raise ImageProcessingError(
                    f"Could not load region {region!r} data from {filepath}"
                )

        metrics = assess_quality_array(
            data=data,
            header=header,
            filepath=filepath,
            frame_type=resolved_frame_type,
            thresholds=thresholds,
            compute_snr=compute_snr,
            bpm=bpm,
        )

        # Merge file-level validation — assess_quality_array has no file access.
        metrics.validation = validation
        metrics.errors.extend(validation.errors)
        metrics.warnings.extend(validation.warnings)
        if not validation.is_valid:
            metrics.passed = False

        # Re-score now that validation is populated.
        metrics.quality_score = _calculate_quality_score(metrics, thresholds)
        metrics.quality_grade = _assign_quality_grade(metrics)

        logger.debug(
            "Quality assessment complete: %s (score: %.1f)",
            metrics.quality_grade,
            metrics.quality_score,
        )
        return metrics

    except ImageProcessingError:
        raise
    except Exception as e:
        logger.error(
            "Quality assessment failed for %s: %s", filepath, e, exc_info=True
        )
        raise ImageProcessingError(
            f"Quality assessment failed: {e}"
        ) from e


def assess_quality_array(
    data: np.ndarray,
    header: fits.Header,
    filepath: StrOrPath = "in-memory",
    frame_type: str | None = None,
    thresholds: QualityThresholds | None = None,
    compute_snr: bool = True,
    bpm: np.ndarray | None = None,
) -> QualityMetrics:
    """Perform quality assessment on an in-memory image array.

    Parameters
    ----------
    data : np.ndarray
        Image data array.
    header : fits.Header
        FITS header associated with *data*. Used to resolve frame type
        when *frame_type* is None.
    filepath : str or Path, optional
        Identifier for the source of *data*. Used only in log messages
        and the returned :class:`QualityMetrics`. Default is
        ``"in-memory"``.
    frame_type : str, optional
        Image type string (e.g. ``"BIAS"``). If None, resolved from
        *header*.
    thresholds : QualityThresholds, optional
        Quality thresholds and scoring weights. Defaults to
        :data:`DEFAULT_THRESHOLDS`.
    compute_snr : bool, optional
        If True, compute SNR and dynamic range metrics. Default is True.
    bpm : np.ndarray, optional
        Bad pixel mask (0 = good, non-zero = bad). Shape must match
        *data*. Masked pixels are excluded from all statistics.

    Returns
    -------
    QualityMetrics
        Quality metrics. The ``validation`` field is always ``None``
        (file-level validation requires a file path; use
        :func:`assess_quality` for that).

    Raises
    ------
    ImageProcessingError
        If all pixels are masked or non-finite, or if assessment fails.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    filepath_path = Path(filepath)
    logger.debug("Assessing quality of array: %s", filepath_path)

    finite_mask = np.isfinite(data)
    if bpm is not None:
        good_mask = bpm == 0
        finite_good_mask = np.logical_and(good_mask, finite_mask)
        good_pixels = data[finite_good_mask]
        if good_pixels.size == 0:
            raise ImageProcessingError(
                "BPM masks all pixels — no good data to assess."
            )
        ignored_nonfinite = int(np.sum(np.logical_and(good_mask, ~finite_mask)))
    else:
        finite_good_mask = finite_mask
        good_pixels = data[finite_good_mask]
        if good_pixels.size == 0:
            raise ImageProcessingError(
                "No finite pixels available for quality assessment."
            )
        ignored_nonfinite = int(np.sum(~finite_mask))

    metrics = QualityMetrics(filepath=filepath_path)

    if frame_type:
        metrics.frame_type = normalize_image_type(frame_type) or frame_type
    elif "IMAGETYP" in header:
        metrics.frame_type = (
            normalize_image_type(header["IMAGETYP"]) or "UNKNOWN"
        )

    metrics.image_shape  = data.shape
    metrics.total_pixels = int(good_pixels.size)

    if ignored_nonfinite > 0:
        metrics.warnings.append(
            f"Ignored {ignored_nonfinite} non-finite pixel(s) "
            "(NaN/Inf) during quality assessment."
        )

    logger.debug("Computing basic statistics...")
    metrics.mean    = float(np.mean(good_pixels))
    metrics.median  = float(np.median(good_pixels))
    metrics.std     = float(np.std(good_pixels))
    metrics.min_val = float(np.min(good_pixels))
    metrics.max_val = float(np.max(good_pixels))
    metrics.mad     = float(mad_std(good_pixels, ignore_nan=True))

    logger.debug("Computing robust statistics (sigma-clip)...")
    try:
        data_masked = np.ma.array(
            data, mask=np.logical_not(finite_good_mask)
        )
        clipped = sigma_clip(data_masked, sigma=3.0, maxiters=5)
        clipped_only = np.logical_and(
            np.ma.getmaskarray(clipped), finite_good_mask
        )
        metrics.num_outliers = int(np.sum(clipped_only))
        if metrics.total_pixels > 0:
            metrics.outlier_fraction = (
                metrics.num_outliers / metrics.total_pixels
            )
    except Exception as e:
        logger.warning("Could not compute robust statistics: %s", e)

    if compute_snr:
        logger.debug("Computing signal quality metrics...")
        try:
            if metrics.frame_type == FITS_VAL_LIGHT_IMGTYPE and metrics.std > 0:
                metrics.snr = metrics.mean / metrics.std
            if metrics.min_val > 0:
                metrics.dynamic_range = metrics.max_val / metrics.min_val
            if metrics.frame_type == FITS_VAL_BIAS_IMGTYPE:
                metrics.read_noise = metrics.std
        except Exception as e:
            logger.warning("Could not compute signal quality: %s", e)

    logger.debug("Checking for negative and saturated pixels...")
    metrics.num_negative_pixels  = int(np.sum(good_pixels < 0))
    metrics.num_saturated_pixels = int(
        np.sum(good_pixels > thresholds.saturation_threshold)
    )
    if metrics.total_pixels > 0:
        metrics.outlier_fraction = metrics.num_outliers / metrics.total_pixels
        metrics.saturation_fraction = (
            metrics.num_saturated_pixels / metrics.total_pixels
        )

    # --- Threshold checks ---
    metrics.passed = True

    if metrics.snr is not None and metrics.snr < thresholds.min_snr:
        metrics.warnings.append(
            f"SNR ({metrics.snr:.2f}) below minimum ({thresholds.min_snr})"
        )
        metrics.passed = False

    if metrics.outlier_fraction > thresholds.max_outlier_fraction:
        metrics.warnings.append(
            f"Outlier fraction ({metrics.outlier_fraction * 100:.2f}%) "
            f"exceeds maximum ({thresholds.max_outlier_fraction * 100:.2f}%)"
        )
        metrics.passed = False

    if metrics.saturation_fraction > thresholds.max_saturation_fraction:
        metrics.warnings.append(
            f"Saturation fraction ({metrics.saturation_fraction * 100:.2f}%) "
            f"exceeds maximum "
            f"({thresholds.max_saturation_fraction * 100:.2f}%)"
        )
        metrics.passed = False

    neg_frac = (
        metrics.num_negative_pixels / metrics.total_pixels
        if metrics.total_pixels > 0
        else 0.0
    )
    if neg_frac > thresholds.max_negative_fraction:
        metrics.warnings.append(
            f"Negative pixel fraction ({neg_frac * 100:.3f}%) "
            f"exceeds maximum "
            f"({thresholds.max_negative_fraction * 100:.3f}%)"
        )
        metrics.passed = False

    metrics.quality_score = _calculate_quality_score(metrics, thresholds)
    metrics.quality_grade = _assign_quality_grade(metrics)

    logger.debug(
        "Array quality assessment complete: %s (score: %.1f)",
        metrics.quality_grade,
        metrics.quality_score,
    )
    return metrics


def _calculate_quality_score(
    metrics: QualityMetrics,
    thresholds: QualityThresholds,
) -> float:
    """Calculate a numerical quality score in [0, 100]."""
    score = 0.0

    validation_score = (
        100.0
        if (metrics.validation is not None and metrics.validation.is_valid)
        else 0.0
    )
    score += validation_score * thresholds.weight_validation

    if metrics.snr is not None and thresholds.min_snr > 0:
        snr_score = min(100.0, (metrics.snr / thresholds.min_snr) * 50.0)
        score += snr_score * thresholds.weight_snr
    # SNR not applicable — omit term rather than awarding free points.

    if thresholds.max_outlier_fraction > 0:
        outlier_penalty = (
            metrics.outlier_fraction / thresholds.max_outlier_fraction
        ) * 100.0
        outlier_score = max(0.0, 100.0 - outlier_penalty)
    else:
        outlier_score = 100.0
    score += outlier_score * thresholds.weight_outliers

    if metrics.median > 0:
        cv = metrics.mad / metrics.median
        stats_score = max(0.0, 100.0 - (cv * 100.0))
    else:
        stats_score = 50.0
    score += stats_score * thresholds.weight_statistics

    return float(min(100.0, max(0.0, score)))


def _assign_quality_grade(metrics: QualityMetrics) -> str:
    """Assign a quality grade based on score and pass/fail state."""
    if not metrics.passed:
        return "FAIL"
    if metrics.quality_score >= 90:
        return "EXCELLENT"
    if metrics.quality_score >= 75:
        return "GOOD"
    if metrics.quality_score >= 60:
        return "ACCEPTABLE"
    if metrics.quality_score >= 40:
        return "POOR"
    return "FAIL"


# =============================================================================
# Calibration Quality & Batch Assessment
# =============================================================================


def assess_calibration_quality(
    calibrated_file: StrOrPath,
    raw_file: StrOrPath | None = None,
    hdu: HDUSelector = 0,
    thresholds: QualityThresholds | None = None,
) -> dict[str, Any]:
    """Assess quality improvement after calibration.

    Parameters
    ----------
    calibrated_file : str or Path
        Path to the calibrated FITS file.
    raw_file : str or Path, optional
        Path to the corresponding raw (uncalibrated) file. If provided,
        improvement metrics (noise reduction, SNR gain) are computed.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0.
    thresholds : QualityThresholds, optional
        Quality thresholds. Defaults to :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    dict
        Always contains ``'calibrated'`` (:class:`QualityMetrics`),
        ``'passed'``, ``'quality_grade'``, and ``'quality_score'``.
        When *raw_file* is provided, also contains ``'raw_stats'`` and
        ``'improvement'`` dicts.
    """
    logger.debug("Assessing calibration quality...")

    raw_path = Path(raw_file) if raw_file is not None else None

    cal_metrics = assess_quality(
        calibrated_file,
        hdu=hdu,
        thresholds=thresholds,
        compute_snr=True,
    )

    result: dict[str, Any] = {
        "calibrated":    cal_metrics,
        "passed":        cal_metrics.passed,
        "quality_grade": cal_metrics.quality_grade,
        "quality_score": cal_metrics.quality_score,
    }

    if raw_path is not None:
        logger.debug("Comparing with raw image: %s", raw_path)
        try:
            raw_stat = stats(
                raw_path, hdu=hdu, region="full", stats_type="all", skipna=True
            )
            raw_robust = robust_stats(raw_path, hdu=hdu, region="full", sigma=3.0)

            result["raw_stats"] = {
                "mean":   raw_stat.mean_val,
                "median": raw_stat.median_val,
                "std":    raw_stat.std_val,
                "mad":    raw_robust.std,
                "min":    raw_stat.min_val,
                "max":    raw_stat.max_val,
            }

            improvement: dict[str, Any] = {}

            if cal_metrics.std > 0 and raw_stat.std_val > 0:
                improvement["noise_reduction"] = (
                    raw_stat.std_val / cal_metrics.std
                )

            try:
                raw_snr = calculate_snr(raw_path, hdu=hdu)
                if cal_metrics.snr is not None and raw_snr:
                    improvement["snr_improvement"] = (
                        cal_metrics.snr / raw_snr
                    )
            except Exception as e:
                logger.debug("Could not calculate SNR improvement: %s", e)

            improvement["mean_change"] = (
                cal_metrics.mean - raw_stat.mean_val
            )
            improvement["mean_change_fraction"] = (
                (cal_metrics.mean - raw_stat.mean_val) / raw_stat.mean_val
                if raw_stat.mean_val != 0
                else 0.0
            )

            if raw_robust.num_clipped > 0:
                improvement["outlier_reduction"] = (
                    raw_robust.num_clipped - cal_metrics.num_outliers
                )

            result["improvement"] = improvement

            for key, value in improvement.items():
                logger.debug("  %s: %.3f", key, value)

        except Exception as e:
            logger.warning("Could not compare with raw image: %s", e)
            result["raw_comparison_error"] = str(e)

    return result


def batch_quality_assessment(
    file_list: list[StrOrPath],
    hdu: HDUSelector = 0,
    frame_type: str | None = None,
    thresholds: QualityThresholds | None = None,
) -> dict[str, Any]:
    """Perform quality assessment on multiple FITS files.

    Parameters
    ----------
    file_list : list of str or Path
        Paths to the FITS files to assess.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0.
    frame_type : str, optional
        Expected image type for all files. If None, read from each header.
    thresholds : QualityThresholds, optional
        Quality thresholds. Defaults to :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    dict
        Keys: ``'total_files'``, ``'passed'``, ``'failed'``,
        ``'pass_rate'``, ``'results'`` (list of :class:`QualityMetrics`),
        ``'summary'`` (aggregate statistics), ``'best'``, ``'worst'``.

    Raises
    ------
    ValueError
        If *file_list* is empty.
    """
    if not file_list:
        raise ValueError("Empty file list provided.")

    logger.debug("Batch quality assessment on %d files...", len(file_list))

    results: list[QualityMetrics] = []
    passed = 0
    failed = 0

    for i, filepath in enumerate(file_list, 1):
        fp = Path(filepath)
        logger.debug("[%d/%d] Assessing: %s", i, len(file_list), fp.name)
        try:
            metrics = assess_quality(
                fp,
                hdu=hdu,
                frame_type=frame_type,
                thresholds=thresholds,
            )
            results.append(metrics)
            if metrics.passed:
                passed += 1
            else:
                failed += 1
                logger.warning(
                    "  Failed: %d error(s), %d warning(s)",
                    len(metrics.errors),
                    len(metrics.warnings),
                )
        except Exception as e:
            logger.error("Failed to assess %s: %s", fp.name, e)
            failed += 1

    summary: dict[str, Any] = {}
    best: QualityMetrics | None = None
    worst: QualityMetrics | None = None

    if results:
        scores = [r.quality_score for r in results]
        snr_values = [r.snr for r in results if r.snr is not None]

        summary = {
            "mean_quality_score":   float(np.mean(scores)),
            "median_quality_score": float(np.median(scores)),
            "std_quality_score":    float(np.std(scores)),
            "min_quality_score":    float(np.min(scores)),
            "max_quality_score":    float(np.max(scores)),
        }
        if snr_values:
            summary.update({
                "mean_snr":   float(np.mean(snr_values)),
                "median_snr": float(np.median(snr_values)),
                "min_snr":    float(np.min(snr_values)),
                "max_snr":    float(np.max(snr_values)),
            })

        best  = max(results, key=lambda r: r.quality_score)
        worst = min(results, key=lambda r: r.quality_score)

    batch: dict[str, Any] = {
        "total_files": len(file_list),
        "passed":      passed,
        "failed":      failed,
        "pass_rate":   passed / len(file_list),
        "results":     results,
        "summary":     summary,
        "best":        best,
        "worst":       worst,
    }

    logger.debug("Batch Assessment Summary:")
    logger.debug("  Total: %d  Passed: %d  Failed: %d", len(file_list), passed, failed)
    if summary:
        logger.debug(
            "  Mean quality score: %.1f", summary.get("mean_quality_score", 0.0)
        )
    if best:
        logger.debug("  Best:  %s (%.1f)", best.filepath.name, best.quality_score)
    if worst:
        logger.debug("  Worst: %s (%.1f)", worst.filepath.name, worst.quality_score)

    return batch


def generate_quality_report(
    batch_results: dict[str, Any],
    output_file: StrOrPath | None = None,
    fmt: str = "text",
    **kwargs: Any,
) -> str:
    """Generate a formatted quality report from batch assessment results.

    Parameters
    ----------
    batch_results : dict
        Output of :func:`batch_quality_assessment`.
    output_file : str or Path, optional
        If provided, the report is written to this path.
    fmt : str, optional
        Output format: ``"text"``, ``"json"``, or ``"csv"``.
        Default is ``"text"``.
    **kwargs
        No additional keyword arguments are accepted. Passing unknown
        keys raises :exc:`TypeError`.

    Returns
    -------
    str
        Formatted report string.

    Raises
    ------
    ValueError
        If *fmt* is not ``"text"``, ``"json"``, or ``"csv"``.
    TypeError
        If unexpected keyword arguments are passed.
    """
    if "format" in kwargs:
        fmt = kwargs.pop("format")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if fmt == "text":
        report = _generate_text_report(batch_results)
    elif fmt == "json":
        report = _generate_json_report(batch_results)
    elif fmt == "csv":
        report = _generate_csv_report(batch_results)
    else:
        raise ValueError(
            f"Unknown format: {fmt!r}. Use 'text', 'json', or 'csv'."
        )

    if output_file is not None:
        Path(output_file).write_text(report, encoding="utf-8")
        logger.info("Quality report written to: %s", output_file)

    return report


def _generate_text_report(batch_results: dict[str, Any]) -> str:
    """Generate a plain-text batch quality report."""
    lines = [
        "=" * 80,
        "FITS QUALITY CONTROL BATCH REPORT",
        "=" * 80,
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "Summary:",
        f"  Total files assessed: {batch_results['total_files']}",
        f"  Passed: {batch_results['passed']} "
        f"({batch_results['pass_rate'] * 100:.1f}%)",
        f"  Failed: {batch_results['failed']}",
        "",
    ]

    if batch_results["summary"]:
        lines.append("Aggregate Statistics:")
        for key, value in batch_results["summary"].items():
            label = key.replace("_", " ").title()
            if isinstance(value, float) and value.is_integer():
                lines.append(f"  {label}: {int(value)}")
            elif isinstance(value, float):
                lines.append(f"  {label}: {value:.2f}")
            else:
                lines.append(f"  {label}: {value}")
        lines.append("")

    if batch_results["best"]:
        best = batch_results["best"]
        lines += [
            "Best Quality Image:",
            f"  File:  {best.filepath.name}",
            f"  Score: {best.quality_score:.1f}",
            f"  Grade: {best.quality_grade}",
            "",
        ]

    if batch_results["worst"]:
        worst = batch_results["worst"]
        lines += [
            "Lowest Quality Image:",
            f"  File:  {worst.filepath.name}",
            f"  Score: {worst.quality_score:.1f}",
            f"  Grade: {worst.quality_grade}",
            "",
        ]

    lines += ["=" * 80, "Individual Results:", "=" * 80]
    for i, result in enumerate(batch_results["results"], 1):
        status = "✓" if result.passed else "✗"
        lines.append(
            f"{i:3d}. {status} {result.filepath.name:40s} "
            f"Grade: {result.quality_grade:10s} "
            f"Score: {result.quality_score:5.1f}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def _generate_json_report(batch_results: dict[str, Any]) -> str:
    """Generate a JSON batch quality report."""
    report_data = {
        "timestamp":   datetime.now(UTC).isoformat(),
        "total_files": batch_results["total_files"],
        "passed":      batch_results["passed"],
        "failed":      batch_results["failed"],
        "pass_rate":   batch_results["pass_rate"],
        "summary":     batch_results["summary"],
        "best":  batch_results["best"].to_dict() if batch_results["best"] else None,
        "worst": batch_results["worst"].to_dict() if batch_results["worst"] else None,
        "results": [r.to_dict() for r in batch_results["results"]],
    }
    return json.dumps(report_data, indent=2, default=str)


def _generate_csv_report(batch_results: dict[str, Any]) -> str:
    """Generate a CSV batch quality report."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Filename", "Frame Type", "Passed", "Quality Grade", "Quality Score",
        "Mean", "Median", "Std", "MAD", "Min", "Max",
        "SNR", "Dynamic Range",
        "Outliers", "Negative Pixels", "Saturated Pixels",
        "Warnings", "Errors",
    ])
    for r in batch_results["results"]:
        writer.writerow([
            r.filepath.name,
            r.frame_type,
            r.passed,
            r.quality_grade,
            f"{r.quality_score:.2f}",
            f"{r.mean:.2f}",
            f"{r.median:.2f}",
            f"{r.std:.2f}",
            f"{r.mad:.2f}",
            f"{r.min_val:.2f}",
            f"{r.max_val:.2f}",
            f"{r.snr:.2f}"           if r.snr            is not None else "",
            f"{r.dynamic_range:.2f}" if r.dynamic_range  is not None else "",
            r.num_outliers,
            r.num_negative_pixels,
            r.num_saturated_pixels,
            len(r.warnings),
            len(r.errors),
        ])
    return output.getvalue()


def assess_calibration_set_quality(
    bias_files: list[StrOrPath] | None = None,
    dark_files: list[StrOrPath] | None = None,
    flat_files: list[StrOrPath] | None = None,
    hdu: HDUSelector = 0,
    thresholds: QualityThresholds | None = None,
) -> dict[str, Any]:
    """Assess quality of a complete calibration frame set.

    Parameters
    ----------
    bias_files : list of str or Path, optional
        Bias frame paths.
    dark_files : list of str or Path, optional
        Dark frame paths.
    flat_files : list of str or Path, optional
        Flat field frame paths.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0.
    thresholds : QualityThresholds, optional
        Quality thresholds. Defaults to :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    dict
        Keys: ``'overall_passed'``, ``'timestamp'``, and one sub-dict
        per frame type provided (``'bias'``, ``'dark'``, ``'flat'``).
        When multiple bias files are provided, also includes
        ``'estimated_read_noise'``.
    """
    logger.debug("Assessing calibration set quality...")

    results: dict[str, Any] = {
        "overall_passed": True,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if bias_files:
        logger.debug("Assessing %d bias frame(s)...", len(bias_files))
        bias_results = batch_quality_assessment(
            bias_files, hdu=hdu, frame_type="BIAS", thresholds=thresholds
        )
        results["bias"] = bias_results
        if bias_results["pass_rate"] < 1.0:
            results["overall_passed"] = False
        if len(bias_files) > 1:
            try:
                read_noise = estimate_read_noise(
                    bias_files, region="full", hdu=hdu
                )
                results["estimated_read_noise"] = read_noise
                logger.debug("Estimated read noise: %.2f e⁻", read_noise)
            except Exception as e:
                logger.warning("Could not estimate read noise: %s", e)

    if dark_files:
        logger.debug("Assessing %d dark frame(s)...", len(dark_files))
        dark_results = batch_quality_assessment(
            dark_files, hdu=hdu, frame_type="DARK", thresholds=thresholds
        )
        results["dark"] = dark_results
        if dark_results["pass_rate"] < 1.0:
            results["overall_passed"] = False

    if flat_files:
        logger.debug("Assessing %d flat frame(s)...", len(flat_files))
        flat_results = batch_quality_assessment(
            flat_files, hdu=hdu, frame_type="FLAT", thresholds=thresholds
        )
        results["flat"] = flat_results
        if flat_results["pass_rate"] < 1.0:
            results["overall_passed"] = False

    logger.info(
        "Calibration set assessment complete: %s",
        "PASSED" if results["overall_passed"] else "FAILED",
    )
    return results
