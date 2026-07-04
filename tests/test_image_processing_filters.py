#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import fits_tools.image_processing.filters as filter_module
from fits_tools.exceptions import ImageProcessingError, InvalidParameterError
from fits_tools.hdu import HDUQuery
from fits_tools.image_processing.filters import (
    _compute_file_checksum,
    _correct_col_offset,
    _correct_row_offset,
    _safe_cast_to_dtype,
    _validated_call,
    filter_hdu,
    filter_hdu_batch,
    filter_image_hdus,
    get_operations_by_category,
    list_operations,
    save_batch_report,
    verify_output_checksums,
)


def test_safe_cast_to_dtype_clips_rounds_and_replaces_nonfinite_values() -> None:
    data = np.array([-5.2, np.inf, 3.6, 500.0], dtype=np.float32)

    cast = _safe_cast_to_dtype(data, np.dtype(np.uint8))

    assert np.array_equal(cast, np.array([0, 0, 4, 255], dtype=np.uint8))


def test_filter_hdu_applies_region_and_preserves_dtype(tmp_path: Path) -> None:
    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output.fits"
    fits.PrimaryHDU(data=np.arange(25, dtype=np.uint16).reshape(5, 5)).writeto(input_path)

    filter_hdu(
        input_path,
        output_path,
        operation="mean",
        region=(1, 4, 1, 4),
        preserve_dtype=True,
        size=3,
        overwrite=True,
    )

    with fits.open(output_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    assert data.dtype == np.uint16
    assert np.array_equal(data[0, :], np.arange(5, dtype=np.uint16))
    assert not np.array_equal(data[1:4, 1:4], np.arange(25, dtype=np.uint16).reshape(5, 5)[1:4, 1:4])
    history = "\n".join(str(x) for x in header.get("HISTORY", []))
    assert "Applied filter: mean" in history
    assert "preserve_dtype=True" in history


def test_filter_image_hdus_filters_only_selected_image_extensions(tmp_path: Path) -> None:
    input_path = tmp_path / "mef.fits"
    output_path = tmp_path / "mef_out.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=np.arange(9, dtype=np.float32).reshape(3, 3)),
            fits.ImageHDU(data=np.ones((3, 3), dtype=np.float32) * 5, name="SCI"),
            fits.ImageHDU(data=np.ones((3, 3), dtype=np.float32) * 9, name="DQ"),
        ]
    )
    hdul.writeto(input_path)

    processed = filter_image_hdus(
        input_path,
        output_path,
        operation="high_pass",
        include_extnames=("SCI",),
        sigma=1.0,
    )

    assert processed == [1]
    with fits.open(output_path) as hdul_out:
        assert np.array_equal(hdul_out[2].data, np.ones((3, 3), dtype=np.float32) * 9)


def test_filter_hdu_uses_extver_and_rejects_ambiguous_extname(tmp_path: Path) -> None:
    input_path = tmp_path / "duplicate_sci.fits"
    output_path = tmp_path / "selected_sci.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.arange(9, dtype=np.float32).reshape(3, 3),
                name="SCI",
                ver=1,
            ),
            fits.ImageHDU(
                data=np.arange(9, dtype=np.float32).reshape(3, 3) + 20,
                name="SCI",
                ver=2,
            ),
        ]
    )
    hdul.writeto(input_path)

    with pytest.raises(InvalidParameterError, match="Ambiguous EXTNAME"):
        filter_hdu(
            input_path,
            output_path,
            operation="high_pass",
            hdu="SCI",
            sigma=1.0,
        )

    filter_hdu(
        input_path,
        output_path,
        operation="high_pass",
        hdu=("SCI", 2),
        sigma=1.0,
        overwrite=True,
    )

    with fits.open(input_path) as original, fits.open(output_path) as filtered:
        assert np.array_equal(filtered[1].data, original[1].data)
        assert not np.array_equal(filtered[2].data, original[2].data)


def test_filter_image_hdus_supports_exact_selectors_and_queries(tmp_path: Path) -> None:
    input_path = tmp_path / "query_mef.fits"
    exact_output = tmp_path / "exact_mef.fits"
    query_output = tmp_path / "queried_mef.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=np.arange(9, dtype=np.float32).reshape(3, 3)),
            fits.ImageHDU(
                data=np.arange(9, dtype=np.float32).reshape(3, 3) + 10,
                name="SCI",
                ver=1,
            ),
            fits.ImageHDU(
                data=np.arange(9, dtype=np.float32).reshape(3, 3) + 20,
                name="SCI",
                ver=2,
            ),
            fits.ImageHDU(
                data=np.ones((3, 3), dtype=np.float32),
                name="DQ",
                ver=2,
            ),
        ]
    )
    hdul.writeto(input_path)

    exact_indices = filter_image_hdus(
        input_path,
        exact_output,
        operation="high_pass",
        selectors=[("SCI", 2), 0, ("SCI", 2)],
        sigma=1.0,
    )
    query_indices = filter_image_hdus(
        input_path,
        query_output,
        operation="high_pass",
        query=HDUQuery(
            extnames=frozenset({"sci"}),
            extvers=frozenset({1}),
        ),
        sigma=1.0,
    )

    assert exact_indices == [2, 0]
    assert query_indices == [1]


def test_filter_image_hdus_rejects_empty_or_conflicting_selection(tmp_path: Path) -> None:
    input_path = tmp_path / "selection_mef.fits"
    output_path = tmp_path / "unused.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((3, 3), dtype=np.float32), name="SCI"),
        ]
    ).writeto(input_path)

    with pytest.raises(ValueError, match="No image HDUs found"):
        filter_image_hdus(
            input_path,
            output_path,
            operation="gaussian",
            include_extnames=(),
            sigma=1.0,
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        filter_image_hdus(
            input_path,
            output_path,
            operation="gaussian",
            selectors=[1],
            query=HDUQuery(),
            sigma=1.0,
        )


def test_filter_output_disposition_supports_sef_destination_and_inplace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "filter_modes_source.fits"
    sef = tmp_path / "filter_modes_sef.fits"
    destination = tmp_path / "filter_modes_destination.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.arange(9, dtype=np.float32).reshape(3, 3),
                name="SCI",
                ver=1,
            ),
            fits.BinTableHDU.from_columns([], name="CAT"),
        ]
    ).writeto(source)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns([], name="KEEP"),
            fits.ImageHDU(data=np.zeros((3, 3)), name="MODEL", ver=1),
        ]
    ).writeto(destination)

    filter_hdu(
        source,
        sef,
        operation="high_pass",
        hdu=("SCI", 1),
        output_mode="sef",
        sigma=1.0,
    )
    with fits.open(sef) as hdul:
        assert len(hdul) == 1

    filter_hdu(
        source,
        destination,
        operation="high_pass",
        hdu=("SCI", 1),
        output_mode="preserve",
        destination_hdu=("MODEL", 1),
        structure_policy="target",
        sigma=1.0,
    )
    with fits.open(destination) as hdul:
        assert hdul[1].name == "KEEP"
        assert hdul[2].name == "MODEL"

    before = fits.getdata(source, ("SCI", 1)).copy()
    filter_hdu(
        source,
        source,
        operation="high_pass",
        hdu=("SCI", 1),
        output_mode="inplace",
        sigma=1.0,
    )
    assert not np.array_equal(fits.getdata(source, ("SCI", 1)), before)


def test_filter_hdu_batch_generates_checksums_and_reports(tmp_path: Path) -> None:
    input1 = tmp_path / "a.fits"
    input2 = tmp_path / "b.fits"
    for idx, path in enumerate((input1, input2), start=1):
        fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32) * idx).writeto(path)

    result = filter_hdu_batch(
        [input1, input2],
        tmp_path / "{stem}_filtered.fits",
        operation="gaussian",
        sigma=1.0,
        overwrite=True,
        validate_outputs=True,
        show_progress=False,
    )

    assert result.success_count == 2
    assert result.failure_count == 0
    assert len(result.checksums) == 2

    verified, corrupted = verify_output_checksums(result)
    assert len(verified) == 2
    assert corrupted == []

    json_report = tmp_path / "report.json"
    text_report = tmp_path / "report.txt"
    save_batch_report(result, json_report)
    save_batch_report(result, text_report)

    report_data = json.loads(json_report.read_text(encoding="utf-8"))
    assert report_data["summary"]["successful"] == 2
    assert "Batch Processing Report" in text_report.read_text(encoding="utf-8")


def test_verify_output_checksums_reports_missing_or_modified_output(tmp_path: Path) -> None:
    input_path = tmp_path / "source.fits"
    fits.PrimaryHDU(data=np.ones((3, 3), dtype=np.float32)).writeto(input_path)

    result = filter_hdu_batch(
        [input_path],
        tmp_path / "{stem}_out.fits",
        operation="gaussian",
        sigma=1.0,
        overwrite=True,
        validate_outputs=True,
        show_progress=False,
    )

    written = result.successful[0]
    written.write_bytes(b"corrupted")
    verified, corrupted = verify_output_checksums(result)

    assert verified == []
    assert corrupted


def test_filter_hdu_batch_dry_run_returns_empty_result(tmp_path: Path) -> None:
    input_path = tmp_path / "dry.fits"
    fits.PrimaryHDU(data=np.ones((3, 3), dtype=np.float32)).writeto(input_path)

    result = filter_hdu_batch(
        [input_path],
        tmp_path / "{stem}_dry.fits",
        operation="median",
        dry_run=True,
        show_progress=False,
    )

    assert result.total_count == 0
    assert not (tmp_path / "dry_dry.fits").exists()


def test_compute_file_checksum_rejects_invalid_algorithm(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")

    with pytest.raises(ValueError, match="Unsupported hash algorithm"):
        _compute_file_checksum(path, "definitely_not_real")


def test_row_and_col_offset_helpers_validate_mode_and_center_data() -> None:
    data = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)

    row_corrected = _correct_row_offset(data, method="mean")
    col_corrected = _correct_col_offset(data, method="median")

    assert np.allclose(np.mean(row_corrected, axis=1), 0.0)
    assert np.allclose(np.median(col_corrected, axis=0), 0.0)

    with pytest.raises(ValueError, match="method must be"):
        _correct_row_offset(data, method="bad")


def test_filter_hdu_rejects_same_input_and_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "same.fits"
    fits.PrimaryHDU(data=np.ones((3, 3), dtype=np.float32)).writeto(input_path)

    with pytest.raises(ValueError, match="must be different paths"):
        filter_hdu(input_path, input_path, operation="gaussian", sigma=1.0)


def test_filter_hdu_batch_rejects_missing_stem_placeholder(tmp_path: Path) -> None:
    input_path = tmp_path / "input.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(input_path)

    with pytest.raises(ValueError, match="must contain '\\{stem\\}'"):
        filter_hdu_batch(
            [input_path],
            tmp_path / "fixed_name.fits",
            operation="gaussian",
            show_progress=False,
        )


def test_filter_metadata_helpers_expose_registered_operations() -> None:
    operations = list_operations()
    categories = get_operations_by_category()

    assert "gaussian" in operations
    assert "high_pass" in categories["enhancement"]
    assert "sobel" in categories["edge"]


def test_validated_call_and_filter_param_errors_surface_cleanly() -> None:
    def sample(a: int, *, scale: int = 1) -> int:
        return a * scale

    assert _validated_call(sample, 3, scale=2) == 6

    with pytest.raises(ValueError, match="Invalid filter parameters"):
        _validated_call(sample, 3, unknown=1)

    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    with pytest.raises(ValueError, match="Invalid filter parameters"):
        filter_hdu.__globals__["OPERATIONS"]["gaussian"](data, sigma=1.0, invalid_kw=1)


def test_filter_hdu_supports_sobel_axis_and_reports_invalid_axis(tmp_path: Path) -> None:
    input_path = tmp_path / "sobel_in.fits"
    axis_out = tmp_path / "sobel_axis.fits"
    bad_out = tmp_path / "sobel_bad.fits"
    fits.PrimaryHDU(data=np.arange(9, dtype=np.float32).reshape(3, 3)).writeto(input_path)

    filter_hdu(input_path, axis_out, operation="sobel", axis=0, overwrite=True)
    assert fits.getdata(axis_out).shape == (3, 3)

    with pytest.raises(ValueError, match="axis out of range"):
        filter_hdu(input_path, bad_out, operation="sobel", axis=3, overwrite=True)


def test_filter_image_hdus_errors_when_requested_extensions_are_absent(tmp_path: Path) -> None:
    input_path = tmp_path / "mef_missing.fits"
    output_path = tmp_path / "unused.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32), name="DQ"),
        ]
    ).writeto(input_path)

    with pytest.raises(ValueError, match="No image HDUs found"):
        filter_image_hdus(
            input_path,
            output_path,
            operation="gaussian",
            include_extnames=("SCI",),
            sigma=1.0,
        )


def test_filter_image_hdus_late_validation_failure_is_atomic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "filter_atomic_source.fits"
    destination = tmp_path / "filter_atomic_destination.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1),
            fits.ImageHDU(data=np.ones((3, 3)), name="SCI", ver=2),
        ]
    ).writeto(source)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((2, 2), 7), name="SCI", ver=1),
            fits.ImageHDU(data=np.full((2, 2), 8), name="SCI", ver=2),
        ]
    ).writeto(destination)
    before = destination.read_bytes()

    with pytest.raises(ImageProcessingError, match="shape"):
        filter_image_hdus(
            source,
            destination,
            operation="gaussian",
            query=HDUQuery(extnames=frozenset({"SCI"})),
            sigma=1.0,
        )

    assert destination.read_bytes() == before


def test_filter_hdu_batch_records_failures_and_report_can_omit_checksums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = tmp_path / "good.fits"
    bad = tmp_path / "bad_keep.fits"
    fits.PrimaryHDU(data=np.ones((3, 3), dtype=np.float32)).writeto(good)
    fits.PrimaryHDU(data=np.ones((3, 3), dtype=np.float32) * 2).writeto(bad)

    real_filter_hdu = filter_module.filter_hdu

    def flaky_filter_hdu(input_file: Path, output_file: Path, *args: object, **kwargs: object) -> None:
        if Path(input_file).name == "bad_keep.fits":
            raise RuntimeError("simulated batch failure")
        real_filter_hdu(input_file, output_file, *args, **kwargs)

    monkeypatch.setattr(filter_module, "filter_hdu", flaky_filter_hdu)

    result = filter_hdu_batch(
        [good, bad],
        tmp_path / "{stem}_batch.fits",
        operation="gaussian",
        sigma=1.0,
        overwrite=True,
        validate_outputs=True,
        show_progress=False,
    )

    assert result.success_count == 1
    assert result.failure_count == 1
    assert bad in [path for path, _ in result.failed]

    report_path = tmp_path / "report_no_checksums.json"
    save_batch_report(result, report_path, include_checksums=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "checksums" not in report
