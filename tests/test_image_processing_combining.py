from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import ImageProcessingError
from fits_tools.image_processing.combining import (
    _apply_mask,
    combine_images,
    combine_images_robust,
    create_master_frame,
    mean_combine,
    median_combine,
)
from fits_tools.image_processing.result_types import CombiningResult


def test_apply_mask_rejects_shape_mismatch() -> None:
    stack = np.zeros((2, 3, 3), dtype=np.float64)
    mask = np.zeros((2, 2), dtype=bool)

    with pytest.raises(ImageProcessingError, match="Mask shape"):
        _apply_mask(stack, mask)


def test_combine_images_robust_sigma_clipped_median_clips_outlier() -> None:
    arrays = [
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32) * 2.0,
        np.array([[100.0, 2.0], [2.0, 2.0]], dtype=np.float32),
    ]

    result = combine_images_robust(
        arrays,
        method="sigma_clipped_median",
        sigma=1.0,
        maxiters=2,
        return_type="result",
    )

    assert isinstance(result, CombiningResult)
    assert result.data.shape == (2, 2)
    assert result.data[0, 0] == pytest.approx(1.5)
    assert result.num_clipped >= 1
    assert result.header["NCOMBINE"] == 3


def test_combine_images_robust_weighted_mean_uses_explicit_weights() -> None:
    arrays = [
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32) * 3.0,
    ]

    combined = combine_images_robust(
        arrays,
        method="weighted_mean",
        weights=[1.0, 3.0],
        return_type="array",
    )

    assert np.allclose(combined, np.full((2, 2), 2.5))


def test_combine_images_robust_rejects_exptime_weights_for_in_memory_arrays() -> None:
    arrays = [np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]

    with pytest.raises(ImageProcessingError, match="Cannot use 'exptime' weights"):
        combine_images_robust(arrays, method="weighted_mean", weights="exptime")


def test_combine_images_robust_uses_exptime_weights_for_file_inputs(tmp_path: Path) -> None:
    file1 = tmp_path / "a.fits"
    file2 = tmp_path / "b.fits"

    fits.PrimaryHDU(
        data=np.ones((2, 2), dtype=np.float32),
        header=fits.Header({"EXPTIME": 1.0}),
    ).writeto(file1)
    fits.PrimaryHDU(
        data=np.ones((2, 2), dtype=np.float32) * 3.0,
        header=fits.Header({"EXPTIME": 3.0}),
    ).writeto(file2)

    result = combine_images_robust(
        [file1, file2],
        method="weighted_mean",
        weights="exptime",
        return_type="result",
    )

    assert isinstance(result, CombiningResult)
    assert np.allclose(result.data, np.full((2, 2), 2.5))
    assert result.header["WEIGHTED"] is True
    assert result.header["WEIGHTBY"] == "exptime"


def test_combine_images_robust_writes_output_and_can_return_path(tmp_path: Path) -> None:
    arrays = [np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32) * 2.0]
    output = tmp_path / "combined.fits"

    returned = combine_images_robust(
        arrays,
        method="mean",
        output_file=output,
        overwrite=True,
        return_type="path",
    )

    assert returned == output
    assert output.exists()
    assert np.allclose(fits.getdata(output), np.full((2, 2), 1.5))


def test_create_master_frame_normalizes_flat_and_updates_header(tmp_path: Path) -> None:
    arrays = [
        np.ones((2, 2), dtype=np.float32) * 10.0,
        np.ones((2, 2), dtype=np.float32) * 20.0,
    ]
    output = tmp_path / "master_flat.fits"

    result = create_master_frame(
        arrays,
        frame_type="flat",
        method="mean",
        normalize=True,
        output_file=output,
        overwrite=True,
        return_type="result",
    )

    assert isinstance(result, CombiningResult)
    assert np.allclose(result.data, np.ones((2, 2)))
    assert result.header["FRMTYPE"] == "flat"
    assert result.header["NORMLIZD"] is True
    assert output.exists()
    assert np.allclose(fits.getdata(output), np.ones((2, 2)))


def test_median_and_mean_combine_wrappers_return_expected_outputs(tmp_path: Path) -> None:
    arrays = [
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32) * 3.0,
    ]

    median_array = median_combine(arrays)
    assert np.allclose(median_array, np.full((2, 2), 2.0))

    output_base = tmp_path / "mean_output"
    mean_path = mean_combine(arrays, filebase=output_base, overwrite=True)
    assert mean_path == output_base.with_suffix(".fits")
    assert mean_path.exists()
    assert np.allclose(fits.getdata(mean_path), np.full((2, 2), 2.0))


def test_combine_images_legacy_api_returns_result_for_requested_operation(tmp_path: Path) -> None:
    arrays = [
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32) * 3.0,
    ]

    with pytest.deprecated_call():
        result = combine_images(arrays, operations="median")

    assert np.allclose(result, np.full((2, 2), 2.0))
