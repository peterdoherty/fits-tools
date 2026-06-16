#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import ImageProcessingError
from fits_tools.quality_control import (
    QualityMetrics,
    QualityThresholds,
    assess_calibration_quality,
    assess_calibration_set_quality,
    assess_quality,
    assess_quality_array,
    batch_quality_assessment,
    generate_quality_report,
)
from fits_tools.validate import ValidationResult


def test_quality_thresholds_are_frozen_and_normalize_weights() -> None:
    thresholds = QualityThresholds(
        weight_validation=2.0,
        weight_snr=1.0,
        weight_outliers=1.0,
        weight_statistics=0.0,
    )

    assert thresholds.weight_validation == pytest.approx(0.5)
    with pytest.raises(FrozenInstanceError):
        thresholds.min_snr = 7.0


def test_assess_quality_array_ignores_nonfinite_pixels() -> None:
    data = np.array([[1.0, 2.0, np.nan], [3.0, np.inf, 5.0]], dtype=float)

    metrics = assess_quality_array(data=data, header=fits.Header(), compute_snr=False)

    assert metrics.total_pixels == 4
    assert metrics.mean == pytest.approx(2.75)
    assert any("Ignored 2 non-finite pixel(s)" in warning for warning in metrics.warnings)


def test_assess_quality_array_uses_bpm_to_exclude_bad_pixels() -> None:
    data = np.array([[1.0, 999.0], [2.0, 3.0]], dtype=float)
    bpm = np.array([[0, 1], [0, 0]], dtype=np.uint32)

    metrics = assess_quality_array(
        data=data,
        header=fits.Header({"IMAGETYP": "LIGHT"}),
        bpm=bpm,
        compute_snr=True,
    )

    assert metrics.total_pixels == 3
    assert metrics.mean == pytest.approx(2.0)
    assert metrics.snr == pytest.approx(metrics.mean / metrics.std)


def test_assess_quality_raises_clear_error_when_file_loader_returns_no_data(monkeypatch) -> None:
    monkeypatch.setattr(
        "fits_tools.quality_control.quick_validate",
        lambda filepath, frame_type=None: ValidationResult(is_valid=True),
    )
    monkeypatch.setattr(
        "fits_tools.quality_control.get_data_and_header",
        lambda filepath, hdu=0: (None, fits.Header()),
    )

    with pytest.raises(ImageProcessingError, match="Could not load data from"):
        assess_quality("dummy.fits")


def test_assess_quality_reads_file_and_merges_validation(sample_fits_file) -> None:
    metrics = assess_quality(sample_fits_file, compute_snr=True)

    assert metrics.validation is not None
    assert metrics.frame_type == "LIGHT"
    assert metrics.total_pixels == 20
    assert metrics.snr == pytest.approx(metrics.mean / metrics.std)


def test_quality_metrics_string_dict_and_json_exports(tmp_path: Path) -> None:
    metrics = QualityMetrics(
        filepath=Path("demo.fits"),
        frame_type="LIGHT",
        mean=10.0,
        median=9.0,
        std=2.0,
        mad=1.5,
        min_val=1.0,
        max_val=20.0,
        snr=5.0,
        dynamic_range=20.0,
        num_outliers=2,
        num_negative_pixels=1,
        num_saturated_pixels=3,
        total_pixels=10,
        passed=False,
        quality_grade="POOR",
        quality_score=42.0,
        warnings=["warn"],
        errors=["err"],
        validation=ValidationResult(is_valid=True, warnings=["vwarn"]),
        image_shape=(2, 5),
    )

    as_dict = metrics.to_dict()
    json_path = tmp_path / "metrics.json"
    json_text = metrics.to_json(json_path)

    assert "Quality Assessment Report" in str(metrics)
    assert "QualityMetrics" in repr(metrics)
    assert as_dict["validation"]["is_valid"] is True
    assert metrics.outlier_fraction == pytest.approx(0.2)
    assert metrics.saturation_fraction == pytest.approx(0.3)
    assert json.loads(json_text)["quality_grade"] == "POOR"
    assert json.loads(json_path.read_text(encoding="utf-8"))["frame_type"] == "LIGHT"


def test_quality_thresholds_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="min_snr"):
        QualityThresholds(min_snr=-1.0)

    with pytest.raises(ValueError, match="max_outlier_fraction"):
        QualityThresholds(max_outlier_fraction=1.5)

    with pytest.raises(ValueError, match="saturation_threshold"):
        QualityThresholds(saturation_threshold=0.0)

    with pytest.raises(ValueError, match="max_negative_fraction"):
        QualityThresholds(max_negative_fraction=2.0)

    with pytest.raises(ValueError, match="All quality score weights must be >= 0"):
        QualityThresholds(weight_snr=-1.0)

    with pytest.raises(ValueError, match="At least one quality score weight must be > 0"):
        QualityThresholds(
            weight_validation=0.0,
            weight_snr=0.0,
            weight_outliers=0.0,
            weight_statistics=0.0,
        )


def test_assess_quality_array_detects_failures_and_bias_read_noise() -> None:
    thresholds = QualityThresholds(
        min_snr=100.0,
        max_outlier_fraction=0.0,
        max_saturation_fraction=0.0,
        max_negative_fraction=0.0,
        saturation_threshold=5.0,
    )
    data = np.array([[1.0, -1.0], [100.0, 2.0]], dtype=float)

    metrics = assess_quality_array(
        data=data,
        header=fits.Header({"IMAGETYP": "BIAS"}),
        thresholds=thresholds,
        compute_snr=True,
    )

    assert metrics.frame_type == "BIAS"
    assert metrics.read_noise == pytest.approx(metrics.std)
    assert metrics.dynamic_range is None
    assert metrics.num_negative_pixels == 1
    assert metrics.num_saturated_pixels == 1
    assert metrics.passed is False
    assert metrics.quality_grade == "FAIL"
    assert any("Negative pixel fraction" in warning for warning in metrics.warnings)
    assert any("Saturation fraction" in warning for warning in metrics.warnings)


def test_assess_quality_array_handles_no_finite_or_all_masked_pixels() -> None:
    with pytest.raises(ImageProcessingError, match="No finite pixels available"):
        assess_quality_array(
            data=np.array([[np.nan, np.inf]], dtype=float),
            header=fits.Header(),
        )

    with pytest.raises(ImageProcessingError, match="BPM masks all pixels"):
        assess_quality_array(
            data=np.ones((2, 2), dtype=float),
            header=fits.Header(),
            bpm=np.ones((2, 2), dtype=np.uint32),
        )


def test_assess_quality_uses_region_loader_and_wraps_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fits_tools.quality_control.quick_validate",
        lambda filepath, frame_type=None: ValidationResult(
            is_valid=False,
            errors=["invalid"],
            warnings=["warn"],
            info={"frame_type": "LIGHT"},
        ),
    )
    monkeypatch.setattr(
        "fits_tools.quality_control.get_data_and_header",
        lambda filepath, hdu=0: (np.ones((4, 4), dtype=float), fits.Header({"IMAGETYP": "LIGHT"})),
    )
    monkeypatch.setattr(
        "fits_tools.quality_control.get_data",
        lambda filepath, hdu=0, region="full": np.ones((2, 2), dtype=float),
    )

    metrics = assess_quality("demo.fits", region="data", compute_snr=True)

    assert metrics.validation is not None
    assert metrics.passed is False
    assert "invalid" in metrics.errors
    assert "warn" in metrics.warnings

    monkeypatch.setattr(
        "fits_tools.quality_control.quick_validate",
        lambda filepath, frame_type=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(ImageProcessingError, match="Quality assessment failed"):
        assess_quality("demo.fits")


def test_assess_calibration_quality_compares_raw_and_handles_comparison_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrated = QualityMetrics(
        filepath=Path("cal.fits"),
        frame_type="LIGHT",
        mean=8.0,
        median=8.0,
        std=2.0,
        mad=1.0,
        min_val=1.0,
        max_val=16.0,
        snr=4.0,
        passed=True,
        quality_grade="GOOD",
        quality_score=80.0,
    )
    raw_stats = type(
        "StatsObj",
        (),
        {"mean_val": 10.0, "median_val": 9.0, "std_val": 4.0, "min_val": 1.0, "max_val": 20.0},
    )()
    raw_robust = type("RobustObj", (), {"std": 3.0, "num_clipped": 5})()

    monkeypatch.setattr("fits_tools.quality_control.assess_quality", lambda *args, **kwargs: calibrated)
    monkeypatch.setattr("fits_tools.quality_control.stats", lambda *args, **kwargs: raw_stats)
    monkeypatch.setattr("fits_tools.quality_control.robust_stats", lambda *args, **kwargs: raw_robust)
    monkeypatch.setattr("fits_tools.quality_control.calculate_snr", lambda *args, **kwargs: 2.0)

    result = assess_calibration_quality("cal.fits", raw_file="raw.fits")

    assert result["quality_grade"] == "GOOD"
    assert result["raw_stats"]["std"] == pytest.approx(4.0)
    assert result["improvement"]["noise_reduction"] == pytest.approx(2.0)
    assert result["improvement"]["snr_improvement"] == pytest.approx(2.0)
    assert result["improvement"]["mean_change"] == pytest.approx(-2.0)
    assert result["improvement"]["outlier_reduction"] == 5

    monkeypatch.setattr(
        "fits_tools.quality_control.stats",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("raw failed")),
    )
    failed = assess_calibration_quality("cal.fits", raw_file="raw.fits")
    assert failed["raw_comparison_error"] == "raw failed"


def test_batch_quality_assessment_and_reports_cover_summary_formats_and_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    good = QualityMetrics(
        filepath=Path("good.fits"),
        frame_type="LIGHT",
        mean=1.0,
        median=1.0,
        std=1.0,
        mad=1.0,
        min_val=1.0,
        max_val=2.0,
        snr=5.0,
        passed=True,
        quality_grade="GOOD",
        quality_score=80.0,
    )
    bad = QualityMetrics(
        filepath=Path("bad.fits"),
        frame_type="LIGHT",
        mean=1.0,
        median=1.0,
        std=2.0,
        mad=1.0,
        min_val=0.5,
        max_val=2.0,
        passed=False,
        quality_grade="FAIL",
        quality_score=20.0,
        warnings=["warn"],
        errors=["err"],
    )

    def fake_assess(filepath, *args, **kwargs):
        name = Path(filepath).name
        if name == "explode.fits":
            raise RuntimeError("boom")
        return good if name == "good.fits" else bad

    monkeypatch.setattr("fits_tools.quality_control.assess_quality", fake_assess)

    batch = batch_quality_assessment(["good.fits", "bad.fits", "explode.fits"])

    assert batch["total_files"] == 3
    assert batch["passed"] == 1
    assert batch["failed"] == 2
    assert batch["best"].filepath.name == "good.fits"
    assert batch["worst"].filepath.name == "bad.fits"
    assert batch["summary"]["mean_quality_score"] == pytest.approx(50.0)
    assert batch["summary"]["mean_snr"] == pytest.approx(5.0)

    text_report = generate_quality_report(batch, fmt="text")
    json_report = generate_quality_report(batch, fmt="json")
    csv_report = generate_quality_report(batch, fmt="csv")
    output_path = tmp_path / "report.txt"
    generate_quality_report(batch, output_file=output_path, format="text")

    assert "FITS QUALITY CONTROL BATCH REPORT" in text_report
    assert json.loads(json_report)["passed"] == 1
    assert "Filename,Frame Type,Passed,Quality Grade,Quality Score" in csv_report
    assert output_path.read_text(encoding="utf-8").startswith("=")

    with pytest.raises(ValueError, match="Unknown format"):
        generate_quality_report(batch, fmt="yaml")

    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        generate_quality_report(batch, fmt="text", extra=True)

    with pytest.raises(ValueError, match="Empty file list provided"):
        batch_quality_assessment([])


def test_assess_calibration_set_quality_aggregates_frame_groups_and_read_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_batch(file_list, hdu=0, frame_type=None, thresholds=None):
        return {
            "total_files": len(file_list),
            "passed": len(file_list) if frame_type != "DARK" else 0,
            "failed": 0 if frame_type != "DARK" else len(file_list),
            "pass_rate": 1.0 if frame_type != "DARK" else 0.0,
            "results": [],
            "summary": {},
            "best": None,
            "worst": None,
        }

    monkeypatch.setattr("fits_tools.quality_control.batch_quality_assessment", fake_batch)
    monkeypatch.setattr("fits_tools.quality_control.estimate_read_noise", lambda *args, **kwargs: 3.5)

    result = assess_calibration_set_quality(
        bias_files=["b1.fits", "b2.fits"],
        dark_files=["d1.fits"],
        flat_files=["f1.fits"],
    )

    assert result["overall_passed"] is False
    assert result["bias"]["total_files"] == 2
    assert result["dark"]["pass_rate"] == 0.0
    assert result["flat"]["pass_rate"] == 1.0
    assert result["estimated_read_noise"] == pytest.approx(3.5)

    monkeypatch.setattr(
        "fits_tools.quality_control.estimate_read_noise",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("skip read noise")),
    )
    no_noise = assess_calibration_set_quality(bias_files=["b1.fits", "b2.fits"])
    assert "estimated_read_noise" not in no_noise
