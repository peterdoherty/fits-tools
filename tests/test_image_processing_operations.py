#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import ImageProcessingError
from fits_tools.hdu import HDUQuery
from fits_tools.image_processing.operations import (
    _safe_divide,
    add_images,
    apply_bitwise_operation,
    apply_operation_to_file,
    apply_operation_to_hdus,
    apply_scalar_operation,
    apply_unary_operation,
    create_mask,
    divide_images,
    fill_masked,
    fill_where,
    get_operation_description,
    list_operations,
    multiply_images,
    scale_image,
    set_value,
    threshold_image,
)


def test_apply_unary_operation_respects_bad_pixels_and_invalid_domain() -> None:
    data = np.array([1.0, 4.0, -1.0, 16.0], dtype=np.float32)
    bad_mask = np.array([False, False, False, True])

    result = apply_unary_operation(data, "sqrt", bad_pixel_mask=bad_mask)

    assert np.allclose(result[:2], np.array([1.0, 2.0], dtype=np.float32))
    assert result[2] == -1.0
    assert result[3] == 16.0


def test_apply_scalar_operation_preserves_dtype_when_requested() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    result = apply_scalar_operation(data, "mul", 2, preserve_dtype=True)

    assert result.dtype == np.uint16
    assert np.array_equal(result, np.array([[2, 4], [6, 8]], dtype=np.uint16))


def test_safe_divide_and_create_mask_cover_error_branches() -> None:
    divided = _safe_divide(np.array([1.0, 2.0]), np.array([1.0, 0.0]), fill_value=-9.0, warn=False)
    inverted = create_mask(np.array([1.0, 5.0]), "greater", 2.0, invert=True)

    assert np.array_equal(divided, np.array([1.0, -9.0]))
    assert np.array_equal(inverted, np.array([True, False]))

    with pytest.raises(ValueError, match="Threshold must be finite"):
        create_mask(np.array([1.0]), "greater", np.inf)


def test_mask_threshold_and_fill_helpers_cover_common_modes() -> None:
    data = np.array([[1.0, 5.0], [7.0, 2.0]], dtype=np.float32)

    assert np.array_equal(create_mask(data, "gt", 4.0), np.array([[False, True], [True, False]]))
    assert np.array_equal(threshold_image(data, 4.0, mode="binary"), np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    assert np.array_equal(threshold_image(data, 4.0, mode="zero"), np.array([[0.0, 5.0], [7.0, 0.0]], dtype=np.float32))
    assert np.array_equal(threshold_image(data, 4.0, mode="truncate"), np.array([[1.0, 4.0], [4.0, 2.0]], dtype=np.float32))
    assert np.array_equal(threshold_image(data, 4.0, mode="to_zero_inv"), np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    assert np.array_equal(fill_masked(data, data > 4.0, -1.0), np.array([[1.0, -1.0], [-1.0, 2.0]], dtype=np.float32))
    assert np.array_equal(fill_masked(data, data > 4.0, 8.0, invert=True), np.array([[8.0, 5.0], [7.0, 8.0]], dtype=np.float32))
    assert np.array_equal(fill_where(data, "less_equal", 2.0, 9.0), np.array([[9.0, 5.0], [7.0, 9.0]], dtype=np.float32))
    assert np.array_equal(set_value(data, 3.0), np.full((2, 2), 3.0, dtype=np.float32))
    assert np.array_equal(set_value(data.astype(np.uint16), 2.9, preserve_dtype=True), np.full((2, 2), 2, dtype=np.uint16))


def test_apply_bitwise_operation_supports_not_and_requires_operand() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    inverted = apply_bitwise_operation(data, "bnot")
    assert inverted.dtype == np.int64
    assert np.array_equal(inverted, np.bitwise_not(data.astype(np.int64)))
    assert np.array_equal(
        apply_bitwise_operation(data, "band", 1),
        np.bitwise_and(data.astype(np.int64), 1),
    )

    with pytest.raises(ImageProcessingError, match="requires an operand"):
        apply_bitwise_operation(data, "bitwise_and")

    with pytest.raises(ValueError, match="Unknown bitwise operation"):
        apply_bitwise_operation(data, "definitely_not_real")


def test_threshold_and_fill_helpers_reject_invalid_inputs() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    with pytest.raises(ValueError, match="Unknown threshold mode"):
        threshold_image(data, 2.0, mode="bad")

    with pytest.raises(ValueError, match="Mask shape"):
        fill_masked(data, np.ones((3, 3), dtype=bool), 0)


def test_add_images_and_apply_operation_to_file_bitwise_and_fill_where(tmp_path: Path) -> None:
    left = tmp_path / "left.fits"
    right = tmp_path / "right.fits"
    added_out = tmp_path / "added.fits"
    bitwise_out = tmp_path / "bitwise.fits"
    fillcmp_out = tmp_path / "fillcmp.fits"

    fits.PrimaryHDU(data=np.array([[1, 2], [3, 4]], dtype=np.uint16)).writeto(left)
    fits.PrimaryHDU(data=np.array([[4, 3], [2, 1]], dtype=np.uint16)).writeto(right)

    add_result = add_images(left, right, added_out, overwrite=True)
    bitwise_result = apply_operation_to_file(
        left,
        bitwise_out,
        "bitwise",
        "bitwise_or",
        operand=right,
        overwrite=True,
    )
    fillcmp_result = apply_operation_to_file(
        left,
        fillcmp_out,
        "fill",
        "greater_equal",
        scalar=99.0,
        threshold=3.0,
        overwrite=True,
    )

    assert add_result.success is True
    assert np.array_equal(fits.getdata(added_out), np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32))
    assert bitwise_result.success is True
    assert np.array_equal(fits.getdata(bitwise_out), np.array([[5, 3], [3, 5]], dtype=np.int64))
    assert fillcmp_result.success is True
    assert np.array_equal(fits.getdata(fillcmp_out), np.array([[1, 2], [99, 99]], dtype=np.uint16))


def test_divide_images_replaces_division_by_zero_with_fill_value(tmp_path: Path) -> None:
    left = tmp_path / "left.fits"
    right = tmp_path / "right.fits"
    output = tmp_path / "divided.fits"
    fits.PrimaryHDU(data=np.array([[4.0, 6.0]], dtype=np.float32)).writeto(left)
    fits.PrimaryHDU(data=np.array([[2.0, 0.0]], dtype=np.float32)).writeto(right)

    result = divide_images(left, right, output, overwrite=True, fill_value=-5.0)

    assert result.success is True
    assert np.array_equal(fits.getdata(output), np.array([[2.0, -5.0]], dtype=np.float32))


def test_scale_image_scales_file_and_can_preserve_dtype(tmp_path: Path) -> None:
    input_path = tmp_path / "scale_in.fits"
    output_path = tmp_path / "scale_out.fits"
    fits.PrimaryHDU(data=np.array([[1, 2], [3, 4]], dtype=np.uint16)).writeto(input_path)

    result = scale_image(input_path, 1.5, output_path, overwrite=True, preserve_dtype=True)

    assert result.success is True
    assert fits.getdata(output_path).dtype == np.uint16
    assert np.array_equal(fits.getdata(output_path), np.array([[1, 3], [4, 6]], dtype=np.uint16))


def test_apply_operation_to_file_handles_multiple_operation_types(tmp_path: Path) -> None:
    input_path = tmp_path / "input.fits"
    mask_path = tmp_path / "mask.fits"
    unary_out = tmp_path / "sqrt.fits"
    cmp_out = tmp_path / "cmp.fits"
    fill_out = tmp_path / "fill.fits"
    fail_out = tmp_path / "fail.fits"

    fits.PrimaryHDU(data=np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)).writeto(input_path)
    fits.PrimaryHDU(data=np.array([[0, 1], [1, 0]], dtype=np.uint8)).writeto(mask_path)

    unary = apply_operation_to_file(input_path, unary_out, "unary", "sqrt", overwrite=True)
    comparison = apply_operation_to_file(input_path, cmp_out, "comparison", "greater", threshold=8.0, overwrite=True)
    fill = apply_operation_to_file(
        input_path,
        fill_out,
        "fill",
        "mask",
        scalar=99.0,
        operand=mask_path,
        overwrite=True,
    )
    failure = apply_operation_to_file(input_path, fail_out, "scalar", "add", overwrite=True)

    assert unary.success is True
    assert np.array_equal(fits.getdata(unary_out), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert comparison.success is True
    assert np.array_equal(fits.getdata(cmp_out), np.array([[0, 0], [1, 1]], dtype=np.uint8))
    assert fill.success is True
    assert np.array_equal(fits.getdata(fill_out), np.array([[1.0, 99.0], [99.0, 16.0]], dtype=np.float32))
    assert failure.success is False
    assert "Scalar operations require 'scalar' parameter" in str(failure.error)


def test_binary_wrappers_and_operation_list_error_paths(tmp_path: Path) -> None:
    left = tmp_path / "left_mul.fits"
    right = tmp_path / "right_mul.fits"
    out = tmp_path / "mul_out.fits"
    bad = tmp_path / "bad_shape.fits"
    fits.PrimaryHDU(data=np.array([[2, 3]], dtype=np.uint16)).writeto(left)
    fits.PrimaryHDU(data=np.array([[4, 5]], dtype=np.uint16)).writeto(right)
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.uint16)).writeto(bad)

    mult = multiply_images(left, right, out, overwrite=True)
    assert mult.success is True
    assert np.array_equal(fits.getdata(out), np.array([[8.0, 15.0]], dtype=np.float32))
    header = fits.getheader(out)
    history_cards = header.get("HISTORY", [])
    history_text = "\n".join(history_cards if isinstance(history_cards, list) else [str(history_cards)])
    assert "Multiply" in history_text
    assert header["OPERNAME"] == "MULTIPLY"

    failed = add_images(left, bad, tmp_path / "failed.fits", overwrite=True)
    assert failed.success is False
    assert "Shape mismatch in Add" in str(failed.error)

    assert "comparison" in list_operations()
    assert "mask" in list_operations("fill")["fill"]
    with pytest.raises(ValueError, match="Unknown operation_type"):
        list_operations("nope")
    with pytest.raises(ValueError, match="Unknown operation"):
        get_operation_description("nope")


def test_apply_operation_to_file_bitwise_fill_and_unknown_type_failures(tmp_path: Path) -> None:
    input_path = tmp_path / "input_ops.fits"
    mask_path = tmp_path / "mask_ops.fits"
    fits.PrimaryHDU(data=np.array([[1, 2], [3, 4]], dtype=np.uint16)).writeto(input_path)
    fits.PrimaryHDU(data=np.array([[1, 0], [0, 1]], dtype=np.uint8)).writeto(mask_path)

    bitwise = apply_operation_to_file(
        input_path,
        tmp_path / "xor_out.fits",
        "bitwise",
        "bitwise_xor",
        operand=1,
        overwrite=True,
    )
    assert bitwise.success is True
    assert np.array_equal(fits.getdata(tmp_path / "xor_out.fits"), np.bitwise_xor(np.array([[1, 2], [3, 4]], dtype=np.int64), 1))

    fill_invert = apply_operation_to_file(
        input_path,
        tmp_path / "fill_invert.fits",
        "fill",
        "mask",
        scalar=8.0,
        operand=mask_path,
        invert=True,
        overwrite=True,
    )
    assert fill_invert.success is True
    assert np.array_equal(fits.getdata(tmp_path / "fill_invert.fits"), np.array([[1, 8], [8, 4]], dtype=np.uint16))

    unknown = apply_operation_to_file(
        input_path,
        tmp_path / "unknown.fits",
        "mystery",
        "noop",
        overwrite=True,
    )
    assert unknown.success is False
    assert "Unknown operation_type" in str(unknown.error)

    bad_mask = apply_operation_to_file(
        input_path,
        tmp_path / "bad_mask_out.fits",
        "fill",
        "mask",
        scalar=3.0,
        operand=np.ones((3, 3), dtype=np.uint8),
        overwrite=True,
    )
    assert bad_mask.success is False
    assert "Mask shape" in str(bad_mask.error)


def test_operation_metadata_helpers_report_registered_operations() -> None:
    operations = list_operations()

    assert "unary" in operations
    assert "sqrt" in operations["unary"]
    assert get_operation_description("mask") == "Fill using a boolean mask (operand)"
    assert get_operation_description("mul") == "Multiply image by scalar"


def test_mef_selection_masks_and_existing_destination_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source_mef.fits"
    mask = tmp_path / "mask_mef.fits"
    destination = tmp_path / "destination_mef.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.full((2, 2), 4.0), name="SCI", ver=1),
        fits.ImageHDU(data=np.full((2, 2), 9.0), name="SCI", ver=2),
        fits.BinTableHDU.from_columns([], name="CAT"),
    ]).writeto(source)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.array([[0, 1], [0, 1]]), name="DQ", ver=2),
    ]).writeto(mask)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns([], name="KEEP"),
        fits.ImageHDU(data=np.zeros((2, 2)), name="SCI", ver=2),
    ]).writeto(destination)

    result = apply_operation_to_file(
        source, destination, "scalar", "multiply",
        hdu=("SCI", 2), scalar=2.0,
        bad_pixel_mask_file=mask, bad_pixel_mask_hdu=("DQ", 2),
        output_mode="preserve",
    )

    assert result.success
    with fits.open(destination) as hdul:
        assert hdul[1].name == "KEEP"
        assert np.array_equal(hdul[2].data, np.array([[18.0, 9.0], [18.0, 9.0]]))


def test_plural_binary_pairs_reordered_extensions_and_reports_missing_partner(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left_mef.fits"
    right = tmp_path / "right_mef.fits"
    output = tmp_path / "plural_out.fits"
    missing = tmp_path / "missing_mef.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.full((2, 2), 1.0), name="SCI", ver=1),
        fits.ImageHDU(data=np.full((2, 2), 2.0), name="SCI", ver=2),
    ]).writeto(left)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.full((2, 2), 20.0), name="SCI", ver=2),
        fits.ImageHDU(data=np.full((2, 2), 10.0), name="SCI", ver=1),
    ]).writeto(right)

    results = apply_operation_to_hdus(
        left, output, "bitwise", "bitwise_or",
        query=HDUQuery(extnames=frozenset({"SCI"})),
        operand=right,
    )
    assert all(result.success for result in results)
    with fits.open(output) as hdul:
        assert np.all(hdul[1].data == 11)
        assert np.all(hdul[2].data == 22)

    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1),
    ]).writeto(missing)
    failed = apply_operation_to_hdus(
        left, tmp_path / "missing_out.fits", "bitwise", "bitwise_or",
        selectors=[("SCI", 2)], operand=missing,
    )
    assert failed[0].success is False
    assert "No HDU matches selector" in str(failed[0].error)


def test_plural_operation_late_compute_failure_leaves_destination_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "atomic_source.fits"
    operand = tmp_path / "atomic_operand.fits"
    destination = tmp_path / "atomic_destination.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1),
            fits.ImageHDU(data=np.full((2, 2), 2), name="SCI", ver=2),
        ]
    ).writeto(source)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((2, 2), 10), name="SCI", ver=1),
        ]
    ).writeto(operand)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((2, 2), 7), name="SCI", ver=1),
            fits.ImageHDU(data=np.full((2, 2), 8), name="SCI", ver=2),
        ]
    ).writeto(destination)
    before = destination.read_bytes()

    results = apply_operation_to_hdus(
        source,
        destination,
        "bitwise",
        "bitwise_or",
        query=HDUQuery(extnames=frozenset({"SCI"})),
        operand=operand,
    )

    assert all(result.success is False for result in results)
    assert all("no selected HDUs were committed" in str(result.error) for result in results)
    assert destination.read_bytes() == before


def test_compressed_hdu_selector_is_supported(tmp_path: Path) -> None:
    source = tmp_path / "compressed.fits"
    output = tmp_path / "compressed_result.fits"
    compressed = fits.CompImageHDU(data=np.full((3, 3), 4.0), name="SCI")
    compressed.header["EXTVER"] = 3
    fits.HDUList([fits.PrimaryHDU(), compressed]).writeto(source)

    result = scale_image(
        source, 2.0, output, hdu=("SCI", 3), output_mode="preserve",
    )

    assert result.success
    with fits.open(output) as hdul:
        assert isinstance(hdul[1], fits.CompImageHDU)
        assert np.all(hdul[1].data == 8)


def test_output_modes_destination_selector_and_inplace_commit(tmp_path: Path) -> None:
    source = tmp_path / "modes_source.fits"
    destination = tmp_path / "modes_destination.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.full((2, 2), 3.0), name="SCI", ver=1),
    ]).writeto(source)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns([], name="KEEP"),
        fits.ImageHDU(data=np.zeros((2, 2)), name="MODEL", ver=1),
    ]).writeto(destination)

    preserved = scale_image(
        source, 2.0, destination, hdu=("SCI", 1),
        output_mode="preserve", destination_hdu=("MODEL", 1),
        structure_policy="target",
    )
    assert preserved.success
    with fits.open(destination) as hdul:
        assert hdul[1].name == "KEEP"
        assert hdul[2].name == "MODEL"
        assert np.all(hdul[2].data == 6)

    inplace = scale_image(
        source, 4.0, source, hdu=("SCI", 1), output_mode="inplace",
    )
    assert inplace.success
    assert np.all(fits.getdata(source, ("SCI", 1)) == 12)


def test_structure_failures_leave_existing_destination_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "rollback_source.fits"
    destination = tmp_path / "rollback_destination.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1),
    ]).writeto(source)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.full((3, 3), 7.0), name="SCI", ver=1),
        fits.BinTableHDU.from_columns([], name="KEEP"),
    ]).writeto(destination)
    before = destination.read_bytes()

    incompatible = scale_image(
        source, 2.0, destination, hdu=("SCI", 1),
        output_mode="preserve", structure_policy="compatible",
    )
    assert incompatible.success is False
    assert "shape" in str(incompatible.error)
    assert destination.read_bytes() == before

    missing = scale_image(
        source, 2.0, destination, hdu=("SCI", 1),
        output_mode="preserve", destination_hdu=("SCI", 9),
    )
    assert missing.success is False
    assert destination.read_bytes() == before
