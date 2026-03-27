#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality control and metrics for CCD/CMOS calibrated images.

This module provides quality assessment tools that integrate validation
(:mod:`fits_tools.validate`) with statistical analysis
(:mod:`fits_tools.statistics`) to evaluate the quality of calibrated
images and calibration workflows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from .validate import (
    ValidationResult,
    quick_validate,
)
from .statistics import (
    stats,
    robust_stats,
    calculate_snr,
    estimate_read_noise,
)
from .io import get_data, get_data_and_header
from .constants import (
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
    normalize_image_type,
)
from .exceptions import ImageProcessingError
from .types import StrOrPath

logger = logging.getLogger(__name__)


# ============================================================================
# Quality Metrics Dataclass
# ============================================================================

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for an image."""

    # Identification
    filepath: Path
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
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    image_shape: Optional[Tuple[int, int]] = None
    total_pixels: int = 0

    def __post_init__(self):
        if self.total_pixels > 0:
            self.outlier_fraction = self.num_outliers / self.total_pixels
            self.saturation_fraction = self.num_saturated_pixels / self.total_pixels

    def __str__(self) -> str:
        status_symbol = "✓" if self.passed else "✗"
        lines = [
            "=" * 70,
            f"Quality Assessment Report: {status_symbol} {self.quality_grade}",
            "=" * 70,
            f"File: {self.filepath.name}",
            f"Type: {self.frame_type}",
            f"Score: {self.quality_score:.1f}/100",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        if self.image_shape:
            lines.append(
                f"Image: {self.image_shape[0]}×{self.image_shape[1]} "
                f"({self.total_pixels:,} pixels)"
            )
            lines.append("")

        lines.extend(
            [
                "Statistics:",
                f"  Mean:   {self.mean:>10.2f}",
                f"  Median: {self.median:>10.2f}",
                f"  Std:    {self.std:>10.2f}",
                f"  MAD:    {self.mad:>10.2f}",
                "",
                "Range:",
                f"  Min:    {self.min_val:>10.2f}",
                f"  Max:    {self.max_val:>10.2f}",
            ]
        )

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
                    f"  Outliers:   {self.num_outliers:>6d} "
                    f"({self.outlier_fraction*100:>5.2f}%)"
                )
            if self.num_negative_pixels > 0:
                lines.append(f"  Negative:   {self.num_negative_pixels:>6d}")
            if self.num_saturated_pixels > 0:
                lines.append(
                    f"  Saturated:  {self.num_saturated_pixels:>6d} "
                    f"({self.saturation_fraction*100:>5.2f}%)"
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
            f"QualityMetrics(file='{self.filepath.name}', "
            f"grade='{self.quality_grade}', score={self.quality_score:.1f}, "
            f"passed={self.passed})"
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.validation:
            data["validation"] = {
                "is_valid": self.validation.is_valid,
                "errors": self.validation.errors,
                "warnings": self.validation.warnings,
                "info": self.validation.info,
            }
        return data

    def to_json(self, filepath: Optional[StrOrPath] = None, indent: int = 2) -> str:
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            logger.info("Quality metrics exported to: %s", filepath)
        return json_str


# ============================================================================
# Quality Thresholds
# ============================================================================

@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds and weights for quality assessment."""

    min_snr: float = 5.0
    max_outlier_fraction: float = 0.01
    max_saturation_fraction: float = 0.05
    saturation_threshold: float = 60000.0
    max_negative_fraction: float = 0.001
    min_dynamic_range: float = 10.0

    weight_validation: float = 0.3
    weight_snr: float = 0.2
    weight_outliers: float = 0.2
    weight_statistics: float = 0.3

    def __post_init__(self):
        if self.min_snr < 0:
            raise ValueError("min_snr must be >= 0.")
        if not (0.0 <= self.max_outlier_fraction <= 1.0):
            raise ValueError("max_outlier_fraction must be in [0, 1].")
        if not (0.0 <= self.max_saturation_fraction <= 1.0):
            raise ValueError("max_saturation_fraction must be in [0, 1].")
        if self.saturation_threshold <= 0:
            raise ValueError("saturation_threshold must be > 0.")
        if not (0.0 <= self.max_negative_fraction <= 1.0):
            raise ValueError("max_negative_fraction must be in [0, 1].")
        if self.min_dynamic_range < 0:
            raise ValueError("min_dynamic_range must be >= 0.")
        if (
            self.weight_validation < 0
            or self.weight_snr < 0
            or self.weight_outliers < 0
            or self.weight_statistics < 0
        ):
            raise ValueError("All quality score weights must be >= 0.")

        total_weight = (
            self.weight_validation
            + self.weight_snr
            + self.weight_outliers
            + self.weight_statistics
        )
        if total_weight <= 0:
            raise ValueError("At least one quality score weight must be > 0.")
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                "Quality score weights sum to %.3f, not 1.0. Normalizing.",
                total_weight,
            )
            object.__setattr__(
                self, "weight_validation", self.weight_validation / total_weight
            )
            object.__setattr__(self, "weight_snr", self.weight_snr / total_weight)
            object.__setattr__(
                self, "weight_outliers", self.weight_outliers / total_weight
            )
            object.__setattr__(
                self, "weight_statistics", self.weight_statistics / total_weight
            )


DEFAULT_THRESHOLDS = QualityThresholds()


# ============================================================================
# Core Quality Assessment
# ============================================================================

def assess_quality(
    filepath: StrOrPath,
    hdu: int = 0,
    frame_type: Optional[str] = None,
    thresholds: Optional[QualityThresholds] = None,
    region: str = "full",
    compute_snr: bool = True,
    bpm: Optional[np.ndarray] = None,
) -> QualityMetrics:
    """Perform comprehensive quality assessment of a FITS image."""
    filepath = Path(filepath)

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    logger.debug("Assessing quality of: %s", filepath)

    try:
        # 1. Validation — file-specific, must happen before data load
        validation = quick_validate(filepath, frame_type=frame_type)

        # Resolve frame type from validation info or caller argument
        resolved_frame_type = frame_type
        if "frame_type" in validation.info:
            resolved_frame_type = validation.info["frame_type"]

        # 2. Load data and header
        data, header = get_data_and_header(filepath, hdu=hdu)
        if data is None:
            raise ImageProcessingError(f"Could not load data from {filepath}")

        # Apply region slicing if needed
        if region != "full":
            data = get_data(filepath, hdu=hdu, region=region)
            if data is None:
                raise ImageProcessingError(
                    "Could not load region data from %s (region=%s)" % (filepath, region)
                )

        # 3. Delegate all statistics to assess_quality_array
        metrics = assess_quality_array(
            data=data,
            header=header,
            filepath=filepath,
            frame_type=resolved_frame_type,
            thresholds=thresholds,
            compute_snr=compute_snr,
            bpm=bpm,
        )

        # 4. Patch in the validation results — assess_quality_array
        #    has no access to the file-level validation
        metrics.validation = validation
        metrics.errors.extend(validation.errors)
        metrics.warnings.extend(validation.warnings)
        if not validation.is_valid:
            metrics.passed = False

        # Re-score now that validation is populated
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
        logger.error("Quality assessment failed for %s: %s", filepath, e, exc_info=True)
        raise ImageProcessingError(f"Quality assessment failed: {e}") from e
    

def assess_quality_array(
    data: np.ndarray,
    header: fits.Header,
    filepath: StrOrPath = "in-memory",
    frame_type: Optional[str] = None,
    thresholds: Optional[QualityThresholds] = None,
    compute_snr: bool = True,
    bpm: Optional[np.ndarray] = None,  
) -> QualityMetrics:
    """Quality assessment for an in-memory image array."""
    from astropy.stats import mad_std, sigma_clip

    finite_mask = np.isfinite(data)
    if bpm is not None:
        good_mask = bpm == 0
        finite_good_mask = np.logical_and(good_mask, finite_mask)
        good_pixels = data[finite_good_mask]
        if good_pixels.size == 0:
            raise ImageProcessingError("BPM masks all pixels — no good data to assess.")
        ignored_nonfinite = int(np.sum(np.logical_and(good_mask, ~finite_mask)))
    else:
        finite_good_mask = finite_mask
        good_pixels = data[finite_good_mask]
        if good_pixels.size == 0:
            raise ImageProcessingError("No finite pixels available for quality assessment.")
        ignored_nonfinite = int(np.sum(~finite_mask))

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    logger.debug("Assessing quality of array: %s", filepath)

    filepath_path = Path(filepath)
    metrics = QualityMetrics(filepath=filepath_path)

    if frame_type:
        metrics.frame_type = normalize_image_type(frame_type)
    elif "IMAGETYP" in header:
        metrics.frame_type = normalize_image_type(header["IMAGETYP"])

    metrics.image_shape = data.shape
    metrics.total_pixels = int(good_pixels.size)
    if ignored_nonfinite > 0:
        metrics.warnings.append(
            "Ignored %d non-finite pixels (NaN/Inf) during quality assessment."
            % ignored_nonfinite
        )

    logger.debug("Computing basic statistics...")
    metrics.mean   = float(np.mean(good_pixels))
    metrics.median = float(np.median(good_pixels))
    metrics.std    = float(np.std(good_pixels))
    metrics.min_val = float(np.min(good_pixels))
    metrics.max_val = float(np.max(good_pixels))
    metrics.mad    = float(mad_std(good_pixels, ignore_nan=True))

    logger.debug("Computing robust statistics...")
    try:
        data_for_clipping = np.ma.array(data, mask=np.logical_not(finite_good_mask))
        clipped = sigma_clip(data_for_clipping, sigma=3.0, maxiters=5)
        combined_mask = np.ma.getmaskarray(clipped)
        clipped_only = np.logical_and(combined_mask, finite_good_mask)
        metrics.num_outliers = int(np.sum(clipped_only))

        if hasattr(clipped, "mask"):
            metrics.outlier_fraction = (
                metrics.num_outliers / metrics.total_pixels if metrics.total_pixels else 0.0
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
    metrics.num_saturated_pixels = int(np.sum(good_pixels > thresholds.saturation_threshold))
    if metrics.total_pixels > 0:
        metrics.outlier_fraction = metrics.num_outliers / metrics.total_pixels
        metrics.saturation_fraction = (
            metrics.num_saturated_pixels / metrics.total_pixels
        )

    metrics.passed = True

    if metrics.snr is not None and metrics.snr < thresholds.min_snr:
        metrics.warnings.append(
            f"SNR ({metrics.snr:.2f}) below minimum ({thresholds.min_snr})"
        )
        metrics.passed = False

    if metrics.outlier_fraction > thresholds.max_outlier_fraction:
        metrics.warnings.append(
            f"Outlier fraction ({metrics.outlier_fraction*100:.2f}%) "
            f"exceeds maximum ({thresholds.max_outlier_fraction*100:.2f}%)"
        )
        metrics.passed = False

    if metrics.saturation_fraction > thresholds.max_saturation_fraction:
        metrics.warnings.append(
            f"Saturation fraction ({metrics.saturation_fraction*100:.2f}%) "
            f"exceeds maximum ({thresholds.max_saturation_fraction*100:.2f}%)"
        )
        metrics.passed = False

    neg_frac = (
        metrics.num_negative_pixels / metrics.total_pixels
        if metrics.total_pixels > 0
        else 0.0
    )
    if neg_frac > thresholds.max_negative_fraction:
        metrics.warnings.append(
            f"Negative pixel fraction ({neg_frac*100:.3f}%) "
            f"exceeds maximum ({thresholds.max_negative_fraction*100:.3f}%)"
        )
        metrics.passed = False

    metrics.quality_score = _calculate_quality_score(metrics, thresholds)
    metrics.quality_grade = _assign_quality_grade(metrics)

    logger.debug(
        "Quality assessment complete: %s (score: %.1f)",
        metrics.quality_grade,
        metrics.quality_score,
    )

    return metrics


def _calculate_quality_score(
    metrics: QualityMetrics,
    thresholds: QualityThresholds,
) -> float:
    """Calculate numerical quality score (0–100)."""
    score = 0.0

    validation_score = (
        100.0 if (metrics.validation and metrics.validation.is_valid) else 0.0
    )
    score += validation_score * thresholds.weight_validation

    if metrics.snr is not None and thresholds.min_snr > 0:
        snr_score = min(100.0, (metrics.snr / thresholds.min_snr) * 50.0)
        score += snr_score * thresholds.weight_snr
    # else: SNR not applicable for this frame type — omit term, don't award free points

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
    """Assign a quality grade based on score and pass/fail."""
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


# ============================================================================
# Calibration Quality & Batch Assessment
# ============================================================================

def assess_calibration_quality(
    calibrated_file: StrOrPath,
    raw_file: Optional[StrOrPath] = None,
    hdu: int = 0,
    thresholds: Optional[QualityThresholds] = None,
) -> Dict[str, Any]:
    """Assess quality improvement after calibration."""
    logger.debug("Assessing calibration quality...")

    raw_file = Path(raw_file) if raw_file else None

    cal_metrics = assess_quality(
        calibrated_file,
        hdu=hdu,
        thresholds=thresholds,
        compute_snr=True,
    )

    result: Dict[str, Any] = {
        "calibrated": cal_metrics,
        "passed": cal_metrics.passed,
        "quality_grade": cal_metrics.quality_grade,
        "quality_score": cal_metrics.quality_score,
    }

    if raw_file:
        logger.debug("Comparing with raw image: %s", raw_file)
        try:
            raw_stats = stats(
                raw_file,
                hdu=hdu,
                region="full",
                stats_type="all",
                skipna=True,
            )
            raw_robust = robust_stats(
                raw_file,
                hdu=hdu,
                region="full",
                sigma=3.0,
            )
            result["raw_stats"] = {
                "mean": raw_stats.mean_val,
                "median": raw_stats.median_val,
                "std": raw_stats.std_val,
                "mad": getattr(raw_robust, "std", raw_stats.std_val),
                "min": raw_stats.min_val,
                "max": raw_stats.max_val,
            }

            improvement: Dict[str, Any] = {}
            if cal_metrics.std > 0 and raw_stats.std_val > 0:
                improvement["noise_reduction"] = raw_stats.std_val / cal_metrics.std

            try:
                raw_snr = calculate_snr(raw_file, hdu=hdu)
                if cal_metrics.snr and raw_snr:
                    improvement["snr_improvement"] = cal_metrics.snr / raw_snr
            except Exception as e:
                logger.debug("Could not calculate SNR improvement: %s", e)

            improvement["mean_change"] = cal_metrics.mean - raw_stats.mean_val
            improvement["mean_change_fraction"] = (
                (cal_metrics.mean - raw_stats.mean_val) / raw_stats.mean_val
                if raw_stats.mean_val != 0
                else 0
            )

            if raw_robust.num_clipped > 0:
                improvement["outlier_reduction"] = (
                    raw_robust.num_clipped - cal_metrics.num_outliers
                )

            result["improvement"] = improvement

            logger.debug("Calibration improvements:")
            for key, value in improvement.items():
                logger.debug("  %s: %.3f", key, value)

        except Exception as e:
            logger.warning("Could not compare with raw image: %s", e)
            result["raw_comparison_error"] = str(e)

    return result


def batch_quality_assessment(
    file_list: List[StrOrPath],
    hdu: int = 0,
    frame_type: Optional[str] = None,
    thresholds: Optional[QualityThresholds] = None,
) -> Dict[str, Any]:
    """Perform quality assessment on multiple FITS files."""
    if not file_list:
        raise ValueError("Empty file list provided")

    logger.debug("Batch quality assessment on %d files...", len(file_list))

    results: List[QualityMetrics] = []
    passed = 0
    failed = 0

    for i, filepath in enumerate(file_list, 1):
        filepath = Path(filepath)
        logger.debug("[%d/%d] Assessing: %s", i, len(file_list), filepath.name)
        try:
            metrics = assess_quality(
                filepath,
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
                    "  Failed: %d errors, %d warnings",
                    len(metrics.errors),
                    len(metrics.warnings),
                )
        except Exception as e:
            logger.error("Failed to assess %s: %s", filepath.name, e)
            failed += 1

    if results:
        quality_scores = [r.quality_score for r in results]
        snr_values = [r.snr for r in results if r.snr is not None]

        summary: Dict[str, Any] = {
            "mean_quality_score": float(np.mean(quality_scores)),
            "median_quality_score": float(np.median(quality_scores)),
            "std_quality_score": float(np.std(quality_scores)),
            "min_quality_score": float(np.min(quality_scores)),
            "max_quality_score": float(np.max(quality_scores)),
        }
        if snr_values:
            summary.update(
                {
                    "mean_snr": float(np.mean(snr_values)),
                    "median_snr": float(np.median(snr_values)),
                    "min_snr": float(np.min(snr_values)),
                    "max_snr": float(np.max(snr_values)),
                }
            )
        best = max(results, key=lambda r: r.quality_score)
        worst = min(results, key=lambda r: r.quality_score)
    else:
        summary = {}
        best = None
        worst = None

    batch_results: Dict[str, Any] = {
        "total_files": len(file_list),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(file_list) if file_list else 0.0,
        "results": results,
        "summary": summary,
        "best": best,
        "worst": worst,
    }

    logger.debug("Batch Assessment Summary:")
    logger.debug("  Total files: %d", batch_results["total_files"])
    logger.debug("  Passed: %d (%.1f%%)", passed, batch_results["pass_rate"] * 100)
    logger.debug("  Failed: %d", failed)
    logger.debug(
        "  Mean quality score: %.1f",
        summary.get("mean_quality_score", 0.0),
    )
    if best:
        logger.debug(
            "  Best: %s (score: %.1f)",
            best.filepath.name,
            best.quality_score,
        )
    if worst:
        logger.debug(
            "  Worst: %s (score: %.1f)",
            worst.filepath.name,
            worst.quality_score,
        )

    return batch_results


def generate_quality_report(
    batch_results: Dict[str, Any],
    output_file: Optional[StrOrPath] = None,
    fmt: str = "text",
    **kwargs: Any,
) -> str:
    """Generate a formatted quality report from batch assessment results."""
    if "format" in kwargs:
        fmt = kwargs.pop("format")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError("Unexpected keyword argument(s): %s" % unexpected)

    if fmt == "text":
        report = _generate_text_report(batch_results)
    elif fmt == "json":
        report = _generate_json_report(batch_results)
    elif fmt == "csv":
        report = _generate_csv_report(batch_results)
    else:
        raise ValueError("Unknown format: %r. Use 'text', 'json', or 'csv'." % fmt)

    if output_file:
        Path(output_file).write_text(report, encoding="utf-8")
        logger.info("Quality report written to: %s", output_file)

    return report


def _generate_text_report(batch_results: Dict[str, Any]) -> str:
    """Generate text format report."""
    lines = [
        "=" * 80,
        "FITS QUALITY CONTROL BATCH REPORT",
        "=" * 80,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Summary:",
        f"  Total files assessed: {batch_results['total_files']}",
        f"  Passed: {batch_results['passed']} ({batch_results['pass_rate']*100:.1f}%)",
        f"  Failed: {batch_results['failed']}",
        "",
    ]

    if batch_results["summary"]:
        lines.append("Aggregate Statistics:")
        for key, value in batch_results["summary"].items():
            label = key.replace("_", " ").title()
            # Format integers without decimals; floats with two decimals.
            if isinstance(value, int) or (
                isinstance(value, float) and value.is_integer()
            ):
                lines.append(f"  {label}: {int(value)}")
            else:
                lines.append(f"  {label}: {value:.2f}")
        lines.append("")

    if batch_results["best"]:
        best = batch_results["best"]
        lines.append("Best Quality Image:")
        lines.append(f"  File: {best.filepath.name}")
        lines.append(f"  Score: {best.quality_score:.1f}")
        lines.append(f"  Grade: {best.quality_grade}")
        lines.append("")

    if batch_results["worst"]:
        worst = batch_results["worst"]
        lines.append("Lowest Quality Image:")
        lines.append(f"  File: {worst.filepath.name}")
        lines.append(f"  Score: {worst.quality_score:.1f}")
        lines.append(f"  Grade: {worst.quality_grade}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("Individual Results:")
    lines.append("=" * 80)

    for i, result in enumerate(batch_results["results"], 1):
        status = "✓" if result.passed else "✗"
        lines.append(
            f"{i:3d}. {status} {result.filepath.name:40s} "
            f"Grade: {result.quality_grade:10s} Score: {result.quality_score:5.1f}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def _generate_json_report(batch_results: Dict[str, Any]) -> str:
    """Generate JSON format report."""
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_files": batch_results["total_files"],
        "passed": batch_results["passed"],
        "failed": batch_results["failed"],
        "pass_rate": batch_results["pass_rate"],
        "summary": batch_results["summary"],
        "best": batch_results["best"].to_dict()
        if batch_results["best"]
        else None,
        "worst": batch_results["worst"].to_dict()
        if batch_results["worst"]
        else None,
        "results": [r.to_dict() for r in batch_results["results"]],
    }
    return json.dumps(report_data, indent=2, default=str)


def _generate_csv_report(batch_results: Dict[str, Any]) -> str:
    """Generate CSV format report."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "Filename",
            "Frame Type",
            "Passed",
            "Quality Grade",
            "Quality Score",
            "Mean",
            "Median",
            "Std",
            "MAD",
            "Min",
            "Max",
            "SNR",
            "Dynamic Range",
            "Outliers",
            "Negative Pixels",
            "Saturated Pixels",
            "Warnings",
            "Errors",
        ]
    )

    for result in batch_results["results"]:
        writer.writerow(
            [
                result.filepath.name,
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
                f"{result.snr:.2f}" if result.snr is not None else "",
                f"{result.dynamic_range:.2f}"
                if result.dynamic_range is not None
                else "",
                result.num_outliers,
                result.num_negative_pixels,
                result.num_saturated_pixels,
                len(result.warnings),
                len(result.errors),
            ]
        )

    return output.getvalue()


def assess_calibration_set_quality(
    bias_files: Optional[List[StrOrPath]] = None,
    dark_files: Optional[List[StrOrPath]] = None,
    flat_files: Optional[List[StrOrPath]] = None,
    hdu: int = 0,
    thresholds: Optional[QualityThresholds] = None,
) -> Dict[str, Any]:
    """Assess quality of a complete calibration frame set."""
    logger.debug("Assessing calibration set quality...")

    results: Dict[str, Any] = {
        "overall_passed": True,
        "timestamp": datetime.now().isoformat(),
    }

    if bias_files:
        logger.debug("Assessing %d bias frames...", len(bias_files))
        bias_results = batch_quality_assessment(
            bias_files,
            hdu=hdu,
            frame_type="BIAS",
            thresholds=thresholds,
        )
        results["bias"] = bias_results
        if bias_results["pass_rate"] < 1.0:
            results["overall_passed"] = False
        if len(bias_files) > 1:
            try:
                read_noise = estimate_read_noise(
                    bias_files,
                    region="full",
                    hdu=hdu,
                )
                results["estimated_read_noise"] = read_noise
                logger.debug("Estimated read noise: %.2f e⁻", read_noise)
            except Exception as e:
                logger.warning("Could not estimate read noise: %s", e)

    if dark_files:
        logger.debug("Assessing %d dark frames...", len(dark_files))
        dark_results = batch_quality_assessment(
            dark_files,
            hdu=hdu,
            frame_type="DARK",
            thresholds=thresholds,
        )
        results["dark"] = dark_results
        if dark_results["pass_rate"] < 1.0:
            results["overall_passed"] = False

    if flat_files:
        logger.debug("Assessing %d flat frames...", len(flat_files))
        flat_results = batch_quality_assessment(
            flat_files,
            hdu=hdu,
            frame_type="FLAT",
            thresholds=thresholds,
        )
        results["flat"] = flat_results
        if flat_results["pass_rate"] < 1.0:
            results["overall_passed"] = False

    logger.info(
        "Calibration set assessment complete: %s",
        "PASSED" if results["overall_passed"] else "FAILED",
    )

    return results


__all__ = [
    "QualityMetrics",
    "QualityThresholds",
    "assess_quality",
    "assess_quality_array",
    "assess_calibration_quality",
    "batch_quality_assessment",
    "generate_quality_report",
    "assess_calibration_set_quality",
    "DEFAULT_THRESHOLDS",
]
