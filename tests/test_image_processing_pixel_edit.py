#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import DataShapeError, InvalidParameterError, RegionError
from fits_tools.image_processing import pixel_edit


def test_pixel_edit_module_imports_and_exports_public_ops() -> None:
    assert pixel_edit.__all__ == [
        "ThreshOp",
        "FillValue",
        "op_crop",
        "op_mask",
        "op_set_value",
    ]


def test_apply_threshold_supports_scalar_and_range_modes() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert np.array_equal(
        pixel_edit._apply_threshold(arr, ">", 2.5),
        np.array([[False, False], [True, True]]),
    )
    assert np.array_equal(
        pixel_edit._apply_threshold(arr, "in", (2.0, 3.0)),
        np.array([[False, True], [True, False]]),
    )


def test_apply_threshold_rejects_invalid_range_usage() -> None:
    arr = np.array([[1.0]])

    with pytest.raises(InvalidParameterError, match="Use thresh_op='in'"):
        pixel_edit._apply_threshold(arr, ">", (0.0, 1.0))

    with pytest.raises(InvalidParameterError, match="lo .* > hi"):
        pixel_edit._apply_threshold(arr, "in", (2.0, 1.0))

    with pytest.raises(InvalidParameterError, match="Unsupported"):
        pixel_edit._apply_threshold(arr, "approximately", 1.0)


def test_geometry_and_mask_loading_helpers(tmp_path) -> None:
    assert pixel_edit._slice_bounds_from_xywh(2, 3, 4, 5) == (
        slice(3, 8),
        slice(2, 6),
    )
    assert pixel_edit._xywh_from_center(5, 6, 2, 3) == (3, 3, 4, 6)

    mask_path = tmp_path / "mask.fits"
    fits.PrimaryHDU(
        data=np.array([[0, 1], [1, 0]], dtype=np.uint8)
    ).writeto(mask_path)
    mask = pixel_edit._load_mask_from_fits(mask_path, (3, 2, 2))
    assert mask.dtype == bool
    assert mask.shape == (2, 2)

    with pytest.raises(DataShapeError, match="incompatible"):
        pixel_edit._load_mask_from_fits(mask_path, (3, 3))


def test_build_spatial_mask_combines_cube_planes_and_external_mask(
    tmp_path,
) -> None:
    cube = np.array(
        [
            [[0.0, 5.0], [0.0, 0.0]],
            [[0.0, 0.0], [7.0, 0.0]],
        ]
    )
    mask_path = tmp_path / "mask.fits"
    fits.PrimaryHDU(
        data=np.array([[1, 0], [0, 0]], dtype=np.uint8)
    ).writeto(mask_path)

    mask = pixel_edit._build_spatial_mask(cube, mask_path, ">", 4.0)
    assert np.array_equal(mask, np.array([[True, True], [True, False]]))


def test_fill_masked_nan_promotes_integer_arrays_to_float32() -> None:
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    mask = np.array([[False, True], [False, False]])

    filled = pixel_edit._fill_masked(arr, mask, "nan")

    assert filled.dtype == np.float32
    assert np.isnan(filled[0, 1])
    assert filled[1, 1] == pytest.approx(4.0)

    numeric = pixel_edit._fill_masked(arr.astype(np.float32), mask, "12.5")
    assert numeric[0, 1] == pytest.approx(12.5)

    with pytest.raises(InvalidParameterError, match="Unrecognized fill"):
        pixel_edit._fill_masked(arr, mask, "not-a-number")


def test_op_crop_crops_image_hdus_and_can_cast_dtype() -> None:
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=np.arange(16, dtype=np.int16).reshape(4, 4)),
            fits.ImageHDU(data=np.arange(18, dtype=np.int16).reshape(2, 3, 3), name="CUBE"),
            fits.BinTableHDU(),
        ]
    )

    cropped = pixel_edit.op_crop(hdul, x0=1, y0=1, width=2, height=2, adjust_wcs=False, dtype="float32")

    assert cropped[0].data.dtype == np.float32
    assert np.array_equal(cropped[0].data, np.array([[5, 6], [9, 10]], dtype=np.float32))
    assert cropped[1].data.shape == (2, 2, 2)
    assert isinstance(cropped[2], fits.BinTableHDU)


def test_op_crop_rejects_out_of_bounds_region() -> None:
    hdul = fits.HDUList([fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))])

    with pytest.raises(RegionError, match="out of bounds"):
        pixel_edit.op_crop(hdul, x0=3, y0=0, width=2, height=2, adjust_wcs=False, dtype=None)


def test_op_mask_combines_threshold_and_fill_value() -> None:
    hdul = fits.HDUList([fits.PrimaryHDU(data=np.array([[1.0, 5.0], [7.0, 2.0]], dtype=np.float32))])

    masked = pixel_edit.op_mask(hdul, thresh_op=">", thresh=4.0, mask_file=None, fill=-1.0, dtype=None)

    assert np.array_equal(masked[0].data, np.array([[1.0, -1.0], [-1.0, 2.0]], dtype=np.float32))


def test_op_mask_broadcasts_spatial_mask_across_cube_planes() -> None:
    cube = np.array(
        [
            [[1.0, 9.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    hdul = fits.HDUList([fits.PrimaryHDU(data=cube)])

    masked = pixel_edit.op_mask(hdul, thresh_op=">", thresh=8.5, mask_file=None, fill="zero", dtype=None)

    expected = cube.copy()
    expected[:, 0, 1] = 0.0
    assert np.array_equal(masked[0].data, expected)


def test_op_set_value_sets_all_pixels_when_no_masking_requested() -> None:
    hdul = fits.HDUList([fits.PrimaryHDU(data=np.arange(4, dtype=np.float32).reshape(2, 2))])

    updated = pixel_edit.op_set_value(hdul, value=42.0, thresh_op=None, thresh=None, mask_file=None, dtype=None)

    assert np.array_equal(updated[0].data, np.full((2, 2), 42.0, dtype=np.float32))


def test_op_set_value_applies_threshold_selectively() -> None:
    hdul = fits.HDUList([fits.PrimaryHDU(data=np.array([[1.0, 5.0], [7.0, 2.0]], dtype=np.float32))])

    updated = pixel_edit.op_set_value(hdul, value=99.0, thresh_op="out", thresh=(2.0, 6.0), mask_file=None, dtype=None)

    assert np.array_equal(updated[0].data, np.array([[99.0, 5.0], [99.0, 2.0]], dtype=np.float32))
