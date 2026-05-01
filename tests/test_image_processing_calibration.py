from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.bitmask import BPM_BIT_OUTLIER, BPM_BIT_SATURATED
from fits_tools.exceptions import FileHandlingError, ImageProcessingError
from fits_tools.image_processing.calibration import (
    _validate_and_normalize_region,
    apply_calibration,
    calibrate_batch,
    calibrate_image,
    preview_calibration,
)


@pytest.fixture
def calibration_raw_header() -> fits.Header:
    return fits.Header({"EXPTIME": 20.0, "IMAGETYP": "LIGHT"})


def test_validate_and_normalize_region_accepts_valid_region() -> None:
    assert _validate_and_normalize_region((1, 4, 2, 5), (6, 7)) == (1, 4, 2, 5)


def test_validate_and_normalize_region_rejects_out_of_bounds_region() -> None:
    with pytest.raises(ImageProcessingError, match="out of bounds"):
        _validate_and_normalize_region((0, 10, 0, 2), (4, 5))


def test_apply_calibration_applies_bias_dark_flat_and_merges_dq(
    calibration_raw_header: fits.Header,
) -> None:
    raw = np.array([[100.0, 110.0], [80.0, 70.0]], dtype=np.float32)
    bias = np.full((2, 2), 10.0, dtype=np.float32)
    dark = np.full((2, 2), 5.0, dtype=np.float32)
    dark_header = fits.Header({"EXPTIME": 10.0})
    flat = np.array([[2.0, 1.0], [0.0, 4.0]], dtype=np.float32)
    input_bpm = np.zeros((2, 2), dtype=np.uint32)
    input_bpm[0, 0] = 1

    result = apply_calibration(
        raw,
        calibration_raw_header,
        master_bias_data=bias,
        master_dark_data=dark,
        master_dark_header=dark_header,
        master_flat_data=flat,
        input_bpm=input_bpm,
        saturation_level=85.0,
        normalize_flat=False,
    )

    expected = np.array([[40.0, 90.0], [0.0, 12.5]], dtype=np.float32)
    assert np.allclose(result.data, expected)
    assert result.bias_corrected is True
    assert result.dark_corrected is True
    assert result.flat_corrected is True
    assert result.bpm is not None
    assert result.bpm[0, 0] & 1
    assert result.bpm[0, 1] & BPM_BIT_SATURATED
    assert result.bpm[1, 0] & BPM_BIT_OUTLIER
    history = result.header.get("HISTORY", [])
    history_text = "\n".join(history) if isinstance(history, list) else str(history)
    assert "Bias corrected" in history_text
    assert "Dark corrected" in history_text
    assert "Flat corrected" in history_text


def test_apply_calibration_requires_dark_header_for_dark_scaling(
    calibration_raw_header: fits.Header,
) -> None:
    raw = np.ones((2, 2), dtype=np.float32)
    dark = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ImageProcessingError, match="master_dark_header is required"):
        apply_calibration(
            raw,
            calibration_raw_header,
            master_dark_data=dark,
        )


def test_calibrate_image_writes_output_and_dq_sidecar(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.fits"
    bias_path = tmp_path / "bias.fits"
    dark_path = tmp_path / "dark.fits"
    flat_path = tmp_path / "flat.fits"
    bpm_path = tmp_path / "bpm.fits"
    output_path = tmp_path / "raw_cal.fits"

    raw_header = fits.Header({"EXPTIME": 20.0, "IMAGETYP": "LIGHT"})
    fits.PrimaryHDU(
        data=np.array([[100.0, 90.0], [80.0, 70.0]], dtype=np.float32),
        header=raw_header,
    ).writeto(raw_path)
    fits.PrimaryHDU(
        data=np.full((2, 2), 10.0, dtype=np.float32),
        header=fits.Header({"IMAGETYP": "BIAS", "EXPTIME": 0.0}),
    ).writeto(bias_path)
    fits.PrimaryHDU(
        data=np.full((2, 2), 5.0, dtype=np.float32),
        header=fits.Header({"IMAGETYP": "DARK", "EXPTIME": 10.0}),
    ).writeto(dark_path)
    fits.PrimaryHDU(
        data=np.array([[2.0, 1.0], [1.0, 4.0]], dtype=np.float32),
        header=fits.Header({"IMAGETYP": "FLAT"}),
    ).writeto(flat_path)
    fits.PrimaryHDU(
        data=np.zeros((2, 2), dtype=np.uint32),
        header=fits.Header(),
    ).writeto(bpm_path)

    result = calibrate_image(
        raw_path,
        output_file=output_path,
        master_bias=bias_path,
        master_dark=dark_path,
        master_flat=flat_path,
        master_bpm=bpm_path,
        saturation_level=85.0,
        validate_inputs=True,
        compute_quality=True,
        overwrite=True,
    )

    dq_path = tmp_path / "raw_cal_dq.fits"
    assert output_path.exists()
    assert dq_path.exists()
    assert result.output_file == output_path
    assert result.bpm_file == dq_path
    assert result.validation is not None
    assert result.quality_metrics is not None
    assert result.header["BIASFILE"] == "bias.fits"
    assert result.header["DARKFILE"] == "dark.fits"
    assert result.header["FLATFILE"] == "flat.fits"
    assert result.header["BPMFILE"] == "bpm.fits"
    assert result.header["DQFILE"] == "raw_cal_dq.fits"
    assert "PROCDATE" in result.header
    written_data = fits.getdata(output_path)
    assert written_data.shape == (2, 2)


def test_calibrate_image_rejects_existing_output_without_overwrite(
    tmp_path: Path,
    sample_fits_file: Path,
) -> None:
    output_path = tmp_path / "exists.fits"
    output_path.write_text("already here", encoding="utf-8")

    with pytest.raises(FileHandlingError, match="overwrite=False"):
        calibrate_image(sample_fits_file, output_file=output_path, overwrite=False)


def test_preview_calibration_returns_intermediate_region_stages(tmp_path: Path) -> None:
    raw_path = tmp_path / "preview_raw.fits"
    raw_header = fits.Header({"EXPTIME": 20.0})
    raw_data = np.arange(16, dtype=np.float32).reshape(4, 4) + 100.0
    fits.PrimaryHDU(data=raw_data, header=raw_header).writeto(raw_path)

    bias = np.ones((4, 4), dtype=np.float32) * 10.0
    dark = np.ones((4, 4), dtype=np.float32) * 5.0
    flat = np.ones((4, 4), dtype=np.float32) * 2.0

    stages = preview_calibration(
        raw_path,
        master_bias=bias,
        master_dark=dark,
        master_flat=flat,
        region=(1, 3, 0, 2),
    )

    assert set(stages) == {"raw", "bias_subtracted", "dark_subtracted", "flat_corrected"}
    assert stages["raw"].shape == (2, 2)
    expected_raw = raw_data[0:2, 1:3]
    assert np.array_equal(stages["raw"], expected_raw)
    assert np.allclose(stages["bias_subtracted"], expected_raw - 10.0)
    assert np.allclose(stages["dark_subtracted"], expected_raw - 15.0)
    assert np.allclose(stages["flat_corrected"], (expected_raw - 15.0) / 2.0)


def test_calibrate_batch_collects_successes_and_failures(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    good_raw = tmp_path / "good.fits"
    missing_raw = tmp_path / "missing.fits"
    bias = np.ones((2, 2), dtype=np.float32)

    fits.PrimaryHDU(
        data=np.ones((2, 2), dtype=np.float32) * 100.0,
        header=fits.Header({"EXPTIME": 5.0, "IMAGETYP": "LIGHT"}),
    ).writeto(good_raw)

    result = calibrate_batch(
        [good_raw, missing_raw],
        output_dir=output_dir,
        master_bias=bias,
        overwrite=True,
        validate_inputs=False,
        write_bpm=False,
    )

    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.total_count == 2
    assert result.success_rate == pytest.approx(0.5)
    assert any("missing.fits" in error for error in result.errors)
    assert (output_dir / "good_cal.fits").exists()
