from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from fits_tools.image_processing.result_types import (
    CalibrationBatchResult,
    CalibrationResult,
    CombiningResult,
    ProcessingResult,
    TransformationResult,
    create_failure_result,
    create_success_result,
)
from fits_tools.validate import ValidationResult


def test_calibration_result_exposes_corrections_and_validation_state() -> None:
    result = CalibrationResult(
        data=np.ones((2, 2), dtype=np.float32),
        header=fits.Header(),
        bias_corrected=True,
        dark_corrected=True,
        validation=ValidationResult(is_valid=False, errors=["bad"]),
    )

    assert result.corrections_applied == ["bias", "dark"]
    assert result.success is False
    assert "bias, dark" in str(result)


def test_calibration_batch_result_tracks_success_failure_and_rate() -> None:
    batch = CalibrationBatchResult()
    batch.add_success(
        CalibrationResult(data=np.ones((2, 2)), header=fits.Header())
    )
    batch.add_failure("raw2.fits", "boom")
    batch.add_warning("careful")

    assert batch.total_count == 2
    assert batch.success_count == 1
    assert batch.failure_count == 1
    assert batch.success_rate == 0.5
    assert batch.success is False
    assert "raw2.fits" in batch.errors[0]


def test_combining_result_to_dict_summarizes_arrays_and_headers() -> None:
    result = CombiningResult(
        data=np.ones((2, 2), dtype=np.float32),
        header=fits.Header({"NCOMBINE": 2}),
        method="mean",
        num_images=2,
        output_file=Path("combined.fits"),
    )

    as_dict = result.to_dict()

    assert as_dict["data"]["_type"] == "ndarray"
    assert as_dict["data"]["shape"] == (2, 2)
    assert as_dict["header"]["NCOMBINE"] == 2
    assert result.success is True


def test_processing_result_accumulates_outputs_and_errors() -> None:
    result = ProcessingResult()
    result.add_success("a.fits")
    result.add_success("b.fits")
    result.add_failure("failed c")

    assert result.total_count == 3
    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.output_files == [Path("a.fits"), Path("b.fits")]
    assert result.success is False


def test_transformation_result_reports_shape_change() -> None:
    result = TransformationResult(
        data=np.ones((3, 2), dtype=np.float32),
        header=fits.Header(),
        transformation="crop",
        input_shape=(4, 4),
        output_shape=(3, 2),
        parameters={"region": (1, 3, 1, 4)},
        output_file="cropped.fits",
    )

    assert result.success is True
    assert result.shape_changed is True
    assert "crop" in str(result)


def test_success_and_failure_convenience_constructors() -> None:
    ok = create_success_result("out.fits", "subtract", gain=2.0)
    bad = create_failure_result("oops", "subtract", gain=2.0)

    assert ok.success is True
    assert ok.output_file == "out.fits"
    assert ok.metadata["gain"] == 2.0
    assert bad.success is False
    assert bad.error == "oops"
