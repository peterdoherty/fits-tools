#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import ImageProcessingError
from fits_tools.image_processing.result_types import TransformationResult
from fits_tools.image_processing.transformations import (
    _validate_region,
    bin_image,
    crop_image,
    flip_image,
    invert_image,
    normalize_image,
    rescale_intensity,
    rotate_image,
    transform_image_file,
    transpose_image,
)


def test_validate_region_rejects_out_of_bounds() -> None:
    with pytest.raises(ImageProcessingError, match="exceeds image bounds"):
        _validate_region((0, 5, 0, 5), (4, 4))


def test_crop_image_returns_expected_subarray() -> None:
    data = np.arange(25).reshape(5, 5)

    cropped = crop_image(data, (1, 4, 2, 5))

    assert np.array_equal(cropped, data[2:5, 1:4])


def test_bin_image_supports_mean_sum_and_median() -> None:
    data = np.arange(16, dtype=float).reshape(4, 4)

    assert np.array_equal(
        bin_image(data, bin_factor=2, method="mean"),
        np.array([[2.5, 4.5], [10.5, 12.5]]),
    )
    assert np.array_equal(
        bin_image(data, bin_factor=2, method="sum"),
        np.array([[10.0, 18.0], [42.0, 50.0]]),
    )
    assert np.array_equal(
        bin_image(data, bin_factor=2, method="median"),
        np.array([[2.5, 4.5], [10.5, 12.5]]),
    )


def test_bin_image_rejects_unknown_method() -> None:
    with pytest.raises(ImageProcessingError, match="Unknown binning method"):
        bin_image(np.ones((4, 4)), bin_factor=2, method="bad")


def test_rotate_flip_and_transpose_transform_geometry() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=float)

    rotated = rotate_image(data, 90, reshape=False, order=0)
    assert np.array_equal(rotated, np.array([[2.0, 4.0], [1.0, 3.0]]))

    assert np.array_equal(flip_image(data, "horizontal"), np.array([[2, 1], [4, 3]]))
    assert np.array_equal(flip_image(data, "vertical"), np.array([[3, 4], [1, 2]]))
    assert np.array_equal(transpose_image(data), np.array([[1, 3], [2, 4]]))


def test_flip_image_rejects_unknown_axis() -> None:
    with pytest.raises(ImageProcessingError, match="Unknown flip axis"):
        flip_image(np.ones((2, 2)), "diagonal")


def test_normalize_image_supports_multiple_modes() -> None:
    data = np.array([[0.0, 50.0], [100.0, 150.0]])

    assert np.mean(normalize_image(np.ones((2, 2)) * 4.0, method="mean")) == pytest.approx(1.0)
    assert np.max(normalize_image(np.array([[1.0, 2.0]]), method="max", target_value=2.0)) == pytest.approx(2.0)
    minmax = normalize_image(data, method="minmax", target_value=1.0)
    assert minmax.min() == pytest.approx(0.0)
    assert minmax.max() == pytest.approx(1.0)


def test_normalize_image_returns_zeros_for_zero_factor() -> None:
    normalized = normalize_image(np.zeros((2, 2)), method="median")
    assert np.array_equal(normalized, np.zeros((2, 2), dtype=np.float32))


def test_rescale_intensity_and_invert_image_behave_as_expected() -> None:
    data = np.array([[0.0, 50.0], [100.0, 150.0]])

    rescaled = rescale_intensity(data, out_range=(0.0, 255.0))
    assert rescaled.min() == pytest.approx(0.0)
    assert rescaled.max() == pytest.approx(255.0)

    inverted = invert_image(data, max_value=200.0)
    assert np.array_equal(inverted, np.array([[200.0, 150.0], [100.0, 50.0]]))


def test_transform_image_file_writes_output_and_updates_header(tmp_path: Path) -> None:
    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output.fits"
    header = fits.Header({"CRPIX1": 10.0, "CRPIX2": 20.0})
    fits.PrimaryHDU(data=np.arange(16, dtype=np.float32).reshape(4, 4), header=header).writeto(input_path)

    result = transform_image_file(
        input_path,
        output_path,
        transformation="crop",
        region=(1, 3, 1, 4),
        adjust_wcs=True,
        overwrite=True,
    )

    assert isinstance(result, TransformationResult)
    assert output_path.exists()
    assert result.input_shape == (4, 4)
    assert result.output_shape == (3, 2)
    assert fits.getdata(output_path).shape == (3, 2)
    assert result.header["CRPIX1"] == pytest.approx(9.0)
    assert result.header["CRPIX2"] == pytest.approx(19.0)


def test_transform_image_file_rejects_unknown_transformation(tmp_path: Path) -> None:
    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(input_path)

    with pytest.raises(ImageProcessingError, match="Failed to apply transformation"):
        transform_image_file(
            input_path,
            output_path,
            transformation="unknown",
            overwrite=True,
        )
