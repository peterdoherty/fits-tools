#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.validate import (
    ValidationResult,
    check_fits_integrity,
    quick_validate,
    validate_against_schema,
    validate_bias_frame,
    validate_calibration_set,
    validate_dark_frame,
    validate_data_range,
    validate_exposure_time,
    validate_fits_file,
    validate_flat_frame,
    validate_frame_set,
    validate_frame_type,
    validate_header_values,
    validate_image_dimensions,
    validate_keyword_types,
    validate_no_nans,
    validate_required_keywords,
)


def test_validation_result_merge_and_text_output() -> None:
    left = ValidationResult()
    left.add_warning("careful")
    left.add_info("left", 1)

    right = ValidationResult()
    right.add_error("broken")
    right.add_info("right", 2)

    left.merge(right, prefix="child")

    assert not left.is_valid
    assert left.errors == ["child: broken"]
    assert "Validation Status: ✗ INVALID" in str(left)
    assert "errors=1" in repr(left)


def test_validate_fits_file_reports_missing_empty_and_invalid_files(tmp_path) -> None:
    missing = validate_fits_file(tmp_path / "missing.fits")
    assert not missing.is_valid

    empty_path = tmp_path / "empty.fits"
    empty_path.write_bytes(b"")
    empty = validate_fits_file(empty_path)
    assert not empty.is_valid

    bad_path = tmp_path / "bad.fits"
    bad_path.write_text("not a fits file")
    invalid = validate_fits_file(bad_path)
    assert not invalid.is_valid


def test_check_fits_integrity_passes_for_valid_file(sample_fits_file) -> None:
    result = check_fits_integrity(sample_fits_file)

    assert result.is_valid
    assert result.info["integrity_check"] == "passed"


def test_validate_required_keywords_reports_missing_key(sample_fits_file) -> None:
    result = validate_required_keywords(sample_fits_file, ["IMAGETYP", "FILTER"])

    assert not result.is_valid
    assert "FILTER" in result.info["missing_keywords"]


def test_validate_header_values_supports_multiple_constraint_types(
    sample_fits_file,
) -> None:
    ok = validate_header_values(
        sample_fits_file,
        {
            "EXPTIME": (10.0, 40.0),
            "IMAGETYP": {"LIGHT", "BIAS"},
            "DATASEC": lambda value: str(value).startswith("["),
        },
    )
    bad = validate_header_values(
        sample_fits_file,
        {
            "MISSING": 1,
            "EXPTIME": (40.0, 50.0),
            "IMAGETYP": {"BIAS"},
            "DATASEC": lambda _value: (_ for _ in ()).throw(RuntimeError("boom")),
        },
    )

    assert ok.is_valid
    assert ok.info["EXPTIME_validated_value"] == pytest.approx(30.0)
    assert not bad.is_valid
    assert any(
        "Validation function for 'DATASEC' raised an error" in err for err in bad.errors
    )


def test_validate_keyword_types_and_frame_type_warning_paths(tmp_path) -> None:
    path = tmp_path / "odd_type.fits"
    header = fits.Header()
    header["IMAGETYP"] = "mystery"
    header["EXPTIME"] = "30"
    fits.PrimaryHDU(data=np.ones((4, 4)), header=header).writeto(path)

    type_result = validate_keyword_types(path, {"IMAGETYP": str, "EXPTIME": float})
    frame_result = validate_frame_type(path)

    assert not type_result.is_valid
    assert any("expected float" in err for err in type_result.errors)
    assert frame_result.is_valid
    assert frame_result.warnings


def test_validate_data_range_passes_within_limits(sample_fits_file) -> None:
    result = validate_data_range(sample_fits_file, min_value=0, max_value=19)

    assert result.is_valid
    assert result.info["data_mean"] == 9.5


def test_validate_data_range_reports_missing_and_empty_data(tmp_path) -> None:
    no_data = tmp_path / "no_data.fits"
    fits.PrimaryHDU(header=fits.Header({"IMAGETYP": "LIGHT"})).writeto(no_data)

    no_data_result = validate_data_range(no_data)
    assert not no_data_result.is_valid

    empty_data = tmp_path / "empty_data.fits"
    fits.PrimaryHDU(data=np.array([], dtype=float)).writeto(empty_data)
    empty_result = validate_data_range(empty_data)
    assert not empty_result.is_valid


def test_validate_no_nans_flags_nonfinite_values(tmp_path) -> None:
    path = tmp_path / "nonfinite.fits"
    fits.PrimaryHDU(data=np.array([[1.0, np.nan], [np.inf, 2.0]])).writeto(path)

    result = validate_no_nans(path)

    assert not result.is_valid
    assert result.info["nan_count"] == 1
    assert result.info["inf_count"] == 1


def test_validate_image_dimensions_covers_expected_and_invalid_rank(
    sample_fits_file, tmp_path
) -> None:
    ok = validate_image_dimensions(
        sample_fits_file,
        expected_shape=(4, 5),
        min_shape=(2, 2),
        max_shape=(10, 10),
    )
    assert ok.is_valid

    cube = tmp_path / "cube.fits"
    fits.PrimaryHDU(data=np.ones((2, 3, 4))).writeto(cube)
    bad = validate_image_dimensions(cube)
    assert not bad.is_valid


def test_validate_exposure_time_checks_tolerance(sample_fits_file) -> None:
    result = validate_exposure_time(
        sample_fits_file, expected_exptime=30.005, tolerance=0.01
    )

    assert result.is_valid


def test_validate_exposure_time_handles_missing_mismatch_and_invalid_tolerance(
    sample_fits_file,
    tmp_path,
) -> None:
    path = tmp_path / "missing_exptime.fits"
    fits.PrimaryHDU(
        data=np.ones((2, 2)), header=fits.Header({"IMAGETYP": "LIGHT"})
    ).writeto(path)

    missing = validate_exposure_time(path)
    mismatch = validate_exposure_time(
        sample_fits_file, expected_exptime=31.0, tolerance=0.1
    )
    invalid_tolerance = validate_exposure_time(sample_fits_file, tolerance=-1.0)

    assert not missing.is_valid
    assert not mismatch.is_valid
    assert any("differs from expected" in err for err in mismatch.errors)
    assert not invalid_tolerance.is_valid


def test_validate_bias_dark_and_flat_frame_specific_rules(tmp_path) -> None:
    bias_path = tmp_path / "bad_bias.fits"
    bias_header = fits.Header({"IMAGETYP": "BIAS", "EXPTIME": 0.01})
    fits.PrimaryHDU(data=np.full((4, 4), 20001.0), header=bias_header).writeto(
        bias_path
    )
    bias = validate_bias_frame(bias_path)
    assert not bias.is_valid
    assert bias.warnings

    dark_path = tmp_path / "bad_dark.fits"
    dark_header = fits.Header({"IMAGETYP": "LIGHT", "EXPTIME": 0.0})
    fits.PrimaryHDU(data=np.ones((4, 4)), header=dark_header).writeto(dark_path)
    dark = validate_dark_frame(dark_path, min_exptime=1.0)
    assert not dark.is_valid

    flat_path = tmp_path / "bad_flat.fits"
    flat_header = fits.Header({"IMAGETYP": "FLAT"})
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=float), header=flat_header).writeto(
        flat_path
    )
    flat = validate_flat_frame(flat_path, min_counts=1.0)
    assert not flat.is_valid


def test_validate_frame_set_detects_varying_exposure_times(
    sample_fits_file, bias_fits_file
) -> None:
    result = validate_frame_set(
        [sample_fits_file, bias_fits_file], check_frame_type=False
    )

    assert result.is_valid
    assert any("Exposure times vary in set" in warning for warning in result.warnings)


def test_validate_frame_set_detects_inconsistent_shape_type_and_empty_list(
    sample_fits_file,
    tmp_path,
) -> None:
    empty = validate_frame_set([])
    assert not empty.is_valid

    other = tmp_path / "other.fits"
    header = fits.Header({"IMAGETYP": "DARK", "EXPTIME": 30.0})
    fits.PrimaryHDU(data=np.ones((3, 3)), header=header).writeto(other)

    result = validate_frame_set([sample_fits_file, other], expected_frame_type="LIGHT")
    assert not result.is_valid
    assert any("Inconsistent" in err for err in result.errors)


def test_validate_calibration_set_and_schema_merge_results(
    bias_fits_file,
    sample_fits_file,
    tmp_path,
) -> None:
    flat_path = tmp_path / "flat.fits"
    flat_header = fits.Header({"IMAGETYP": "FLAT", "EXPTIME": 5.0})
    fits.PrimaryHDU(data=np.full((4, 5), 5000.0), header=flat_header).writeto(flat_path)

    cal = validate_calibration_set(
        bias_files=[bias_fits_file],
        dark_files=[sample_fits_file],
        flat_files=[flat_path],
    )
    assert not cal.is_valid
    assert cal.info["num_bias_frames"] == 1

    schema = validate_against_schema(
        sample_fits_file,
        {
            "required_keywords": ["IMAGETYP", "EXPTIME"],
            "keyword_values": {"IMAGETYP": "LIGHT"},
            "keyword_types": {"EXPTIME": float},
            "data_range": {"min": 0.0, "max": 19.0},
            "dimensions": {"expected": (4, 5)},
            "no_nans": True,
        },
    )
    assert schema.is_valid
    assert schema.info["schema_validation"] is True


def test_quick_validate_accepts_well_formed_bias_frame(bias_fits_file) -> None:
    result = quick_validate(bias_fits_file, frame_type="BIAS")

    assert result.is_valid
    assert result.info["frame_type"] == "BIAS"


def test_quick_validate_infers_type_and_flags_nonfinite_data(tmp_path) -> None:
    path = tmp_path / "auto_flat.fits"
    header = fits.Header({"IMAGETYP": "FLAT", "EXPTIME": 3.0})
    data = np.array([[np.nan, 1000.0], [1000.0, 1000.0]])
    fits.PrimaryHDU(data=data, header=header).writeto(path)

    result = quick_validate(path)

    assert not result.is_valid
    assert result.info["frame_type"] == "FLAT"
