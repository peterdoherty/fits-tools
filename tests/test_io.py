#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import (
    FileHandlingError,
    InvalidHDUIndexError,
    InvalidParameterError,
)
from fits_tools.io import (
    count_hdus,
    count_image_extensions,
    fits_extension_count,
    get_all_data,
    get_data,
    get_data_and_header,
    get_hdu_count,
    get_header,
    get_header_value,
    get_image_hdus,
    get_image_shape,
    inspect_fits,
    open_all_data_memmap,
    region_to_slices,
    safe_write,
    write_fits,
    write_hdulist,
    write_simple_fits,
)


def test_count_hdus_and_image_hdus_for_mef_file(mef_fits_file) -> None:
    assert count_hdus(mef_fits_file) == 3
    assert get_image_hdus(mef_fits_file) == [0, 1, 2]
    assert get_image_hdus(mef_fits_file, extname="SCI") == [1, 2]


def test_get_data_and_header_return_independent_objects(sample_fits_file) -> None:
    data, header = get_data_and_header(sample_fits_file)

    assert data.shape == (4, 5)
    assert header["IMAGETYP"] == "LIGHT"
    header["IMAGETYP"] = "CHANGED"

    fresh_header = get_header(sample_fits_file)
    assert fresh_header["IMAGETYP"] == "LIGHT"


def test_get_data_supports_named_regions(sample_fits_file) -> None:
    region = get_data(sample_fits_file, region="data")

    assert region.shape == (3, 4)
    assert np.array_equal(
        region, np.arange(20, dtype=np.float32).reshape(4, 5)[1:4, 1:5]
    )


def test_get_data_returns_copy_when_extracting_region_from_memmap(
    sample_fits_file,
) -> None:
    region = get_data(sample_fits_file, region="data", memmap=True)

    assert isinstance(region, np.ndarray)
    assert not isinstance(region, np.memmap)


def test_get_all_data_return_dict_uses_unique_extnames(mef_fits_file) -> None:
    data = get_all_data(mef_fits_file, return_dict=True)

    assert set(data) == {"HDU_0", "SCI", "SCI_1"}
    assert data["SCI"].shape == (3, 3)


def test_open_all_data_memmap_supports_dict_output(mef_fits_file) -> None:
    with open_all_data_memmap(mef_fits_file, return_dict=True) as arrays:
        assert set(arrays) == {"HDU_0", "SCI", "SCI_1"}
        assert arrays["SCI"].shape == (3, 3)


def test_write_simple_fits_round_trips_compressed_data(tmp_path) -> None:
    path = tmp_path / "compressed.fits"
    data = np.arange(9, dtype=np.int16).reshape(3, 3)
    header = fits.Header({"IMAGETYP": "BIAS"})

    write_simple_fits(path, data, header=header, compress=True)

    assert get_image_shape(path, hdu=1) == (3, 3)
    restored, restored_header = get_data_and_header(path, hdu=1)
    assert np.array_equal(restored, data)
    assert restored_header["IMAGETYP"] == "BIAS"


def test_get_header_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileHandlingError, match="File not found"):
        get_header(tmp_path / "missing.fits")


def test_get_header_value_uses_primary_then_selected_hdu_fallback(
    tmp_path,
) -> None:
    path = tmp_path / "metadata_fallback.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(
                header=fits.Header(
                    {
                        "EXPTIME": 12.5,
                        "IMAGETYP": "FLAT",
                        "SETPOINT": -80.0,
                    }
                )
            ),
            fits.ImageHDU(
                data=np.ones((2, 3), dtype=np.float32),
                header=fits.Header(
                    {
                        "EXPTIME": 99.0,
                        "CCDTEMP": -79.8,
                    }
                ),
                name="SCI",
            ),
        ]
    ).writeto(path)

    assert get_header_value(path, "exptime", hdu=1) == 12.5
    assert get_header_value(path, "IMAGETYP", hdu="SCI") == "FLAT"
    assert get_header_value(path, "SETPOINT", hdu=1) == -80.0
    assert get_header_value(path, "CCDTEMP", hdu=1) == -79.8
    assert get_header_value(path, "MISSING", hdu=1, default="unknown") == "unknown"

    with pytest.raises(ValueError, match="non-empty"):
        get_header_value(path, " ", hdu=1)


def test_get_data_and_header_raises_for_bad_hdu(sample_fits_file) -> None:
    with pytest.raises(InvalidHDUIndexError):
        get_data_and_header(sample_fits_file, hdu=2)


def test_singular_io_readers_support_named_and_extver_selectors(tmp_path) -> None:
    path = tmp_path / "selector_io.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 3)), name="SCI", ver=1),
            fits.ImageHDU(data=np.full((4, 5), 2.0), name="SCI", ver=2),
        ]
    ).writeto(path)

    data, header = get_data_and_header(path, hdu=("SCI", 2))

    assert data is not None
    assert data.shape == (4, 5)
    assert header["EXTVER"] == 2
    assert get_image_shape(path, hdu=("SCI", 1)) == (2, 3)
    assert get_header(path, hdu=("SCI", 2))["EXTVER"] == 2
    with pytest.raises(InvalidParameterError, match="Ambiguous EXTNAME"):
        get_data(path, hdu="SCI")


def test_inspect_fits_returns_basic_summary(sample_fits_file) -> None:
    info = inspect_fits(sample_fits_file, detailed=False)

    assert info.filename == "sample.fits"
    assert info.num_hdus == 1
    assert info.hdus[0].shape == (4, 5)
    assert info.hdus[0].dtype == "float32"
    assert info.hdus[0].data_min is None
    assert "FITS File: sample.fits" in str(info)


def test_io_count_helpers_and_region_errors_cover_deprecated_and_edge_paths(
    sample_fits_file, sample_header
) -> None:
    with pytest.deprecated_call():
        assert get_hdu_count(sample_fits_file) == 1
    with pytest.deprecated_call():
        assert fits_extension_count(sample_fits_file) == 0

    assert count_image_extensions(sample_fits_file) == 1

    with pytest.raises(TypeError, match="Unsupported region string"):
        region_to_slices((4, 5), sample_header, "unknown")

    with pytest.raises(TypeError, match="must be a string, tuple, Section, or None"):
        region_to_slices((4, 5), sample_header, 123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="element\\(s\\)"):
        region_to_slices((4, 5), sample_header, (0, 1, 2))  # type: ignore[arg-type]


def test_io_helpers_cover_missing_data_hdus_and_extname_filters(tmp_path) -> None:
    table_only = tmp_path / "table_only.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([])]).writeto(
        table_only
    )

    assert get_image_shape(table_only, hdu=0) is None
    data, header = get_data_and_header(table_only, hdu=1)
    assert data is not None
    assert data.shape == (0,)
    assert header["XTENSION"] == "BINTABLE"
    assert get_data(table_only, hdu=1) is not None
    assert get_all_data(table_only) == []
    assert get_all_data(table_only, return_dict=True) == {}

    with open_all_data_memmap(table_only, extname_filter="SCI") as arrays:
        assert arrays == []


def test_io_write_helpers_cover_atomic_writes_and_detailed_inspection(
    tmp_path, sample_fits_file
) -> None:
    output = tmp_path / "written.fits"
    hdul = fits.HDUList([fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32))])

    safe_write(hdul, output, overwrite=False, dry_run=False, operation="unit-test")
    assert output.exists()
    history = get_header(output)["HISTORY"]
    history_lines = [history] if isinstance(history, str) else list(history)
    assert any("fits_tools: unit-test" in line for line in history_lines)

    dry_run = tmp_path / "dry_run.fits"
    safe_write(hdul, dry_run, overwrite=False, dry_run=True, operation="unit-test")
    assert dry_run.exists() is False

    output_hdul = tmp_path / "out_hdul.fits"
    write_hdulist(output_hdul, hdul, overwrite=False)
    assert count_hdus(output_hdul) == 1

    with pytest.deprecated_call():
        write_fits(
            tmp_path / "deprecated.fits",
            np.zeros((2, 2), dtype=np.float32),
            overwrite=True,
        )

    info = inspect_fits(sample_fits_file, detailed=True)
    text = str(info)
    assert "--- HDU 0 ---" in text
    assert "Data shape: (4, 5)" in text
    assert info.hdus[0].data_min is not None
    assert info.hdus[0].header_cards.get("EXPTIME") == 30.0


def test_io_helpers_raise_for_invalid_hdu_and_missing_inspection_target(
    tmp_path, sample_fits_file
) -> None:
    with pytest.raises(InvalidHDUIndexError):
        get_header(sample_fits_file, hdu=9)

    with pytest.raises(InvalidHDUIndexError):
        get_image_shape(sample_fits_file, hdu=9)

    with pytest.raises(FileHandlingError, match="File not found"):
        inspect_fits(tmp_path / "missing.fits")
