#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.bitmask import BPM_BIT_PIX_BRIGHT, BPM_BIT_PIX_HOT
from fits_tools.defects.io import (
    analyze_bpm_statistics,
    load_bpm_from_fits,
    merge_bad_pixel_maps,
    save_bpm_to_fits,
    save_calibration_bpm,
    validate_bpm_fits,
)


def test_save_and_load_bpm_roundtrip_preserves_metadata_and_defs_extension(tmp_path: Path) -> None:
    bpm = np.zeros((4, 4), dtype=np.uint32)
    bpm[0, 0] = BPM_BIT_PIX_BRIGHT
    bpm[1, 1] = BPM_BIT_PIX_HOT
    stats = {
        "bias": {"threshold_offset_adu": 50.0, "median": 1000.0, "std": 3.0},
        "dark": {"threshold_sigma": 5.0, "median": 10.0, "std": 1.5},
        "flat": {"hot_threshold_percent": 20.0, "cold_threshold_percent": 20.0, "median": 10000.0},
    }
    output = tmp_path / "bpm.fits"

    save_bpm_to_fits(
        bpm,
        output,
        stats=stats,
        header_cards={"OBSERVER": ("Tester", "Synthetic test file")},
    )

    loaded, metadata = load_bpm_from_fits(output)

    assert np.array_equal(loaded, bpm)
    assert metadata["total_bad"] == 2
    assert metadata["defect_counts"]["PIX_BRIGHT"] >= 1
    assert metadata["detection_params"]["BIASTHR"] == pytest.approx(50.0)
    with fits.open(output) as hdul:
        assert "DQ_DEF" in [hdu.name for hdu in hdul]
        assert hdul[0].header["OBSERVER"] == "Tester"


def test_validate_bpm_fits_rejects_signed_dtype_and_non_2d() -> None:
    with pytest.raises(ValueError, match="unsigned integer dtype"):
        validate_bpm_fits(np.zeros((2, 2), dtype=np.int16), {"total_bad": 0})

    with pytest.raises(ValueError, match="must be 2-D"):
        validate_bpm_fits(np.zeros((2, 2, 2), dtype=np.uint32), {"total_bad": 0})


def test_merge_bad_pixel_maps_supports_fresh_and_incremental_merge(tmp_path: Path) -> None:
    new1 = np.zeros((4, 4), dtype=np.uint32)
    new1[0, 0] = BPM_BIT_PIX_BRIGHT
    new2 = np.zeros((4, 4), dtype=np.uint32)
    new2[1, 1] = BPM_BIT_PIX_HOT

    new1_path = tmp_path / "new1.fits"
    new2_path = tmp_path / "new2.fits"
    merged_path = tmp_path / "merged.fits"
    save_bpm_to_fits(new1, new1_path)
    save_bpm_to_fits(new2, new2_path)

    first_stats = merge_bad_pixel_maps(None, new1_path, merged_path, stage="BIAS")
    second_stats = merge_bad_pixel_maps(merged_path, new2_path, merged_path, stage="DARK")

    assert first_stats["PIX_BRIGHT"]["count"] == 1
    assert second_stats["PIX_BRIGHT"]["count"] == 1
    assert second_stats["PIX_HOT"]["count"] == 1
    with fits.open(merged_path) as hdul:
        assert hdul[0].header["BPMTYPE"] == "CUMULATIVE"
        assert hdul[0].header["BPMSTAGE"] == "DARK"


def test_save_calibration_bpm_writes_linked_master_frames(tmp_path: Path) -> None:
    bpm = np.zeros((3, 3), dtype=np.uint32)
    bpm[0, 0] = BPM_BIT_PIX_BRIGHT
    master_bias = np.ones((3, 3), dtype=np.float32)
    master_dark = np.ones((3, 3), dtype=np.float32) * 2
    master_flat = np.ones((3, 3), dtype=np.float32) * 3
    stats = {"bias": {}, "dark": {}, "flat": {}}

    files = save_calibration_bpm(
        bpm,
        master_bias,
        master_dark,
        master_flat,
        stats,
        output_prefix=tmp_path / "calset",
    )

    assert set(files) == {"bpm", "bias", "dark", "flat"}
    with fits.open(files["dark"]) as hdul:
        assert hdul[0].header["IMAGETYP"] == "MASTER_DARK"
        assert hdul[0].header["BPMFILE"].endswith("_bpm.fits")


def test_analyze_bpm_statistics_returns_counts_for_known_bits() -> None:
    bpm = np.zeros((4, 4), dtype=np.uint32)
    bpm[0, 0] = BPM_BIT_PIX_BRIGHT
    bpm[1, 1] = BPM_BIT_PIX_HOT
    bpm[2, 2] = BPM_BIT_PIX_HOT

    stats = analyze_bpm_statistics(bpm)

    assert stats["PIX_BRIGHT"]["count"] == 1
    assert stats["PIX_HOT"]["count"] == 2


def test_load_bpm_from_fits_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="BPM file not found"):
        load_bpm_from_fits(tmp_path / "missing.fits")
