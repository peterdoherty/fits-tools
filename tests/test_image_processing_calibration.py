#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.bitmask import BPM_BIT_OUTLIER, BPM_BIT_SATURATED
from fits_tools.exceptions import (
    FileHandlingError,
    ImageProcessingError,
    InvalidParameterError,
)
from fits_tools.hdu import HDUQuery
from fits_tools.image_processing.calibration import (
    _validate_and_normalize_region,
    apply_calibration,
    calibrate_batch,
    calibrate_hdus,
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


def test_calibrate_image_preserves_mef_and_pairs_reordered_master_by_identity(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_mef.fits"
    bias_path = tmp_path / "bias_mef.fits"
    output_path = tmp_path / "calibrated_mef.fits"

    raw_sci1 = fits.ImageHDU(
        data=np.full((2, 2), 100.0, dtype=np.float32), name="SCI"
    )
    raw_sci2 = fits.ImageHDU(
        data=np.full((3, 2), 200.0, dtype=np.float32), name="SCI"
    )
    raw_sci1.header["EXTVER"] = 1
    raw_sci2.header["EXTVER"] = 2
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="VALUE", format="J", array=np.array([7]))],
        name="META",
    )
    fits.HDUList(
        [fits.PrimaryHDU(), raw_sci1, table, raw_sci2]
    ).writeto(raw_path)

    bias_sci2 = fits.ImageHDU(
        data=np.full((3, 2), 20.0, dtype=np.float32), name="SCI"
    )
    bias_sci1 = fits.ImageHDU(
        data=np.full((2, 2), 10.0, dtype=np.float32), name="SCI"
    )
    bias_sci2.header["EXTVER"] = 2
    bias_sci1.header["EXTVER"] = 1
    fits.HDUList(
        [fits.PrimaryHDU(), bias_sci2, bias_sci1]
    ).writeto(bias_path)

    result = calibrate_image(
        raw_path,
        output_file=output_path,
        master_bias=bias_path,
        hdu=("SCI", 2),
        output_mode="preserve",
        validate_inputs=False,
        write_bpm=False,
    )

    assert result.output_file == output_path
    with fits.open(output_path) as hdul:
        assert len(hdul) == 4
        assert np.array_equal(hdul[1].data, raw_sci1.data)
        assert hdul[2].name == "META"
        assert hdul[2].data["VALUE"][0] == 7
        assert np.array_equal(
            hdul[("SCI", 2)].data,
            np.full((3, 2), 180.0, dtype=np.float32),
        )


def test_calibrate_image_updates_selected_hdu_inplace(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_inplace.fits"
    sci = fits.ImageHDU(
        data=np.full((2, 2), 50.0, dtype=np.float32), name="SCI"
    )
    sci.header["EXTVER"] = 1
    unchanged = fits.ImageHDU(
        data=np.full((2, 2), 9, dtype=np.int16), name="AUX"
    )
    fits.HDUList([fits.PrimaryHDU(), sci, unchanged]).writeto(raw_path)

    result = calibrate_image(
        raw_path,
        master_bias=np.full((2, 2), 5.0, dtype=np.float32),
        hdu=("SCI", 1),
        output_mode="inplace",
        validate_inputs=False,
        write_bpm=False,
    )

    assert result.output_file == raw_path
    with fits.open(raw_path) as hdul:
        assert np.array_equal(
            hdul[("SCI", 1)].data,
            np.full((2, 2), 45.0, dtype=np.float32),
        )
        assert np.array_equal(hdul["AUX"].data, unchanged.data)


def test_calibrate_image_honors_explicit_master_selector(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_selector.fits"
    bias_path = tmp_path / "bias_selector.fits"
    sci = fits.ImageHDU(
        data=np.full((2, 2), 20.0, dtype=np.float32), name="SCI"
    )
    sci.header["EXTVER"] = 1
    fits.HDUList([fits.PrimaryHDU(), sci]).writeto(raw_path)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.full((2, 2), 3.0, dtype=np.float32),
                name="CALIBRATION",
            ),
        ]
    ).writeto(bias_path)

    result = calibrate_image(
        raw_path,
        master_bias=bias_path,
        hdu=("SCI", 1),
        bias_hdu="CALIBRATION",
        validate_inputs=False,
        write_bpm=False,
    )

    assert np.array_equal(
        result.data, np.full((2, 2), 17.0, dtype=np.float32)
    )


def test_calibrate_hdus_pairs_reordered_masters_and_commits_once(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "plural_raw.fits"
    bias_path = tmp_path / "plural_bias.fits"
    output_path = tmp_path / "plural_calibrated.fits"

    def sci(value: float, version: int, shape: tuple[int, int]) -> fits.ImageHDU:
        hdu = fits.ImageHDU(
            data=np.full(shape, value, dtype=np.float32), name="SCI"
        )
        hdu.header["EXTVER"] = version
        return hdu

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            sci(100, 1, (2, 2)),
            fits.BinTableHDU.from_columns([], name="KEEP"),
            sci(200, 2, (3, 2)),
        ]
    ).writeto(raw_path)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            sci(20, 2, (3, 2)),
            sci(10, 1, (2, 2)),
        ]
    ).writeto(bias_path)

    results = calibrate_hdus(
        raw_path,
        output_path,
        query=HDUQuery(extnames=frozenset({"SCI"})),
        master_bias=bias_path,
        validate_inputs=False,
    )

    assert len(results) == 2
    with fits.open(output_path) as hdul:
        assert np.all(hdul[("SCI", 1)].data == 90)
        assert hdul[2].name == "KEEP"
        assert np.all(hdul[("SCI", 2)].data == 180)


def test_calibrate_hdus_late_missing_master_rolls_back(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "plural_rollback_raw.fits"
    bias_path = tmp_path / "plural_rollback_bias.fits"
    destination = tmp_path / "plural_rollback_destination.fits"
    sci1 = fits.ImageHDU(data=np.full((2, 2), 10.0), name="SCI", ver=1)
    sci2 = fits.ImageHDU(data=np.full((2, 2), 20.0), name="SCI", ver=2)
    fits.HDUList([fits.PrimaryHDU(), sci1, sci2]).writeto(raw_path)
    fits.HDUList([fits.PrimaryHDU(), sci1.copy()]).writeto(bias_path)
    fits.HDUList([fits.PrimaryHDU(), sci1.copy(), sci2.copy()]).writeto(
        destination
    )
    before = destination.read_bytes()

    with pytest.raises(ImageProcessingError, match="No HDU matches selector"):
        calibrate_hdus(
            raw_path,
            destination,
            query=HDUQuery(extnames=frozenset({"SCI"})),
            master_bias=bias_path,
            validate_inputs=False,
        )

    assert destination.read_bytes() == before


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


def test_preview_calibration_skips_dark_like_apply_calibration(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "preview_no_exptime.fits"
    dark_path = tmp_path / "preview_dark_no_exptime.fits"
    raw_data = np.full((2, 2), 50.0, dtype=np.float32)
    fits.PrimaryHDU(data=raw_data).writeto(raw_path)
    fits.PrimaryHDU(
        data=np.full((2, 2), 5.0, dtype=np.float32)
    ).writeto(dark_path)

    # Raw frame has no EXPTIME: dark correction must be skipped, exactly as
    # apply_calibration() would skip it.
    stages = preview_calibration(raw_path, master_dark=dark_path)
    assert "dark_subtracted" not in stages

    # Dark frame with EXPTIME=0 must also be skipped instead of scaling.
    raw_with_exptime = tmp_path / "preview_with_exptime.fits"
    zero_dark = tmp_path / "preview_zero_dark.fits"
    fits.PrimaryHDU(
        data=raw_data, header=fits.Header({"EXPTIME": 10.0})
    ).writeto(raw_with_exptime)
    fits.PrimaryHDU(
        data=np.full((2, 2), 5.0, dtype=np.float32),
        header=fits.Header({"EXPTIME": 0.0}),
    ).writeto(zero_dark)
    stages = preview_calibration(raw_with_exptime, master_dark=zero_dark)
    assert "dark_subtracted" not in stages


def test_preview_calibration_normalizes_flat_when_requested(tmp_path: Path) -> None:
    raw_path = tmp_path / "preview_normalized.fits"
    raw_data = np.full((2, 2), 10.0, dtype=np.float32)
    fits.PrimaryHDU(data=raw_data).writeto(raw_path)

    stages = preview_calibration(
        raw_path,
        master_flat=np.full((2, 2), 2.0, dtype=np.float32),
        normalize_flat=True,
    )
    assert np.array_equal(stages["flat_corrected"], raw_data)

    with pytest.raises(ImageProcessingError, match="median of zero"):
        preview_calibration(
            raw_path,
            master_flat=np.zeros((2, 2), dtype=np.float32),
            normalize_flat=True,
        )


def test_preview_calibration_pairs_reordered_mef_master_by_identity(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "preview_mef.fits"
    bias_path = tmp_path / "preview_bias_mef.fits"

    raw_sci1 = fits.ImageHDU(
        data=np.full((2, 2), 40.0, dtype=np.float32), name="SCI"
    )
    raw_sci2 = fits.ImageHDU(
        data=np.full((3, 2), 80.0, dtype=np.float32), name="SCI"
    )
    raw_sci1.header["EXTVER"] = 1
    raw_sci2.header["EXTVER"] = 2
    fits.HDUList([fits.PrimaryHDU(), raw_sci1, raw_sci2]).writeto(raw_path)

    bias_sci2 = fits.ImageHDU(
        data=np.full((3, 2), 8.0, dtype=np.float32), name="SCI"
    )
    bias_sci1 = fits.ImageHDU(
        data=np.full((2, 2), 4.0, dtype=np.float32), name="SCI"
    )
    bias_sci2.header["EXTVER"] = 2
    bias_sci1.header["EXTVER"] = 1
    fits.HDUList([fits.PrimaryHDU(), bias_sci2, bias_sci1]).writeto(
        bias_path
    )

    stages = preview_calibration(
        raw_path,
        master_bias=bias_path,
        hdu=("SCI", 2),
    )

    assert np.array_equal(
        stages["bias_subtracted"],
        np.full((3, 2), 72.0, dtype=np.float32),
    )


def test_preview_calibration_honors_explicit_master_selector(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "preview_explicit.fits"
    bias_path = tmp_path / "preview_explicit_bias.fits"
    raw_sci = fits.ImageHDU(
        data=np.full((2, 2), 10.0, dtype=np.float32), name="SCI"
    )
    raw_sci.header["EXTVER"] = 1
    fits.HDUList([fits.PrimaryHDU(), raw_sci]).writeto(raw_path)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.full((2, 2), 3.0, dtype=np.float32),
                name="CAL",
            ),
        ]
    ).writeto(bias_path)

    stages = preview_calibration(
        raw_path,
        master_bias=bias_path,
        hdu=("SCI", 1),
        bias_hdu="CAL",
    )

    assert np.array_equal(
        stages["bias_subtracted"],
        np.full((2, 2), 7.0, dtype=np.float32),
    )


def test_preview_calibration_rejects_ambiguous_bare_name(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "preview_ambiguous.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI")
    sci2 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI")
    sci1.header["EXTVER"] = 1
    sci2.header["EXTVER"] = 2
    fits.HDUList([fits.PrimaryHDU(), sci1, sci2]).writeto(raw_path)

    with pytest.raises(InvalidParameterError, match="Ambiguous EXTNAME"):
        preview_calibration(raw_path, hdu="SCI")


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


def test_calibrate_batch_records_master_provenance_in_outputs(
    tmp_path: Path,
) -> None:
    import hashlib

    output_dir = tmp_path / "out"
    raw_path = tmp_path / "raw.fits"
    bias_path = tmp_path / "bias.fits"
    dark_path = tmp_path / "dark.fits"

    fits.PrimaryHDU(
        data=np.full((2, 2), 100.0, dtype=np.float32),
        header=fits.Header({"EXPTIME": 10.0}),
    ).writeto(raw_path)
    fits.PrimaryHDU(
        data=np.ones((2, 2), dtype=np.float32)
    ).writeto(bias_path)
    fits.PrimaryHDU(
        data=np.full((2, 2), 2.0, dtype=np.float32),
        header=fits.Header({"EXPTIME": 10.0}),
    ).writeto(dark_path)

    result = calibrate_batch(
        [raw_path],
        output_dir=output_dir,
        master_bias=bias_path,
        master_dark=dark_path,
        validate_inputs=False,
        write_bpm=False,
    )

    assert result.success_count == 1
    with fits.open(output_dir / "raw_cal.fits") as hdul:
        header = hdul[0].header
        assert header["BIASFILE"] == "bias.fits"
        assert header["DARKFILE"] == "dark.fits"
        assert header["BIASSUM"] == hashlib.sha256(
            bias_path.read_bytes()
        ).hexdigest()
        assert header["DARKSUM"] == hashlib.sha256(
            dark_path.read_bytes()
        ).hexdigest()


def test_calibrate_batch_pairs_reordered_mefs_and_caches_master(
    tmp_path: Path,
) -> None:
    raw1 = tmp_path / "raw1.fits"
    raw2 = tmp_path / "raw2.fits"
    bias_path = tmp_path / "bias.fits"
    output_dir = tmp_path / "batch"

    def science_hdu(value: float, extver: int) -> fits.ImageHDU:
        hdu = fits.ImageHDU(
            data=np.full((2, 2), value, dtype=np.float32), name="SCI"
        )
        hdu.header["EXTVER"] = extver
        return hdu

    fits.HDUList(
        [fits.PrimaryHDU(), science_hdu(100, 1), science_hdu(200, 2)]
    ).writeto(raw1)
    fits.HDUList(
        [fits.PrimaryHDU(), science_hdu(220, 2), science_hdu(120, 1)]
    ).writeto(raw2)
    fits.HDUList(
        [fits.PrimaryHDU(), science_hdu(20, 2), science_hdu(10, 1)]
    ).writeto(bias_path)

    result = calibrate_batch(
        [raw1, raw2],
        output_dir,
        master_bias=bias_path,
        hdu=("SCI", 2),
        validate_inputs=False,
        write_bpm=False,
    )

    assert result.success_count == 2
    assert np.array_equal(
        fits.getdata(output_dir / "raw1_cal.fits"),
        np.full((2, 2), 180.0, dtype=np.float32),
    )
    assert np.array_equal(
        fits.getdata(output_dir / "raw2_cal.fits"),
        np.full((2, 2), 200.0, dtype=np.float32),
    )


def test_calibrate_batch_rejects_changed_selected_identity_shape(
    tmp_path: Path,
) -> None:
    raw1 = tmp_path / "same.fits"
    raw2 = tmp_path / "different.fits"
    output_dir = tmp_path / "batch_mismatch"

    sci1 = fits.ImageHDU(
        data=np.full((2, 2), 10.0, dtype=np.float32), name="SCI"
    )
    sci2 = fits.ImageHDU(
        data=np.full((3, 2), 10.0, dtype=np.float32), name="SCI"
    )
    sci1.header["EXTVER"] = 1
    sci2.header["EXTVER"] = 1
    fits.HDUList([fits.PrimaryHDU(), sci1]).writeto(raw1)
    fits.HDUList([fits.PrimaryHDU(), sci2]).writeto(raw2)

    result = calibrate_batch(
        [raw1, raw2],
        output_dir,
        master_bias=np.ones((2, 2), dtype=np.float32),
        hdu=("SCI", 1),
        validate_inputs=False,
        write_bpm=False,
    )

    assert result.success_count == 1
    assert result.failure_count == 1
    assert any("identity/shape" in error for error in result.errors)
