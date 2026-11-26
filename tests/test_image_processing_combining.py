#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import FileOverwriteError, ImageProcessingError
from fits_tools.hdu import HDUQuery
from fits_tools.image_processing.combining import (
    _apply_mask,
    combine_image_hdus,
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


def test_combine_images_robust_sigma_mean_and_mask_paths() -> None:
    arrays = [
        np.array([[1.0, 10.0], [3.0, 4.0]]),
        np.array([[3.0, 20.0], [5.0, 6.0]]),
        np.array([[5.0, 30.0], [7.0, 8.0]]),
    ]
    mask = np.array([[False, True], [False, False]])

    result = combine_images_robust(
        arrays,
        method="sigma_clipped_mean",
        mask=mask,
        sigma=3.0,
        return_type="result",
    )

    assert isinstance(result, CombiningResult)
    assert result.data[0, 0] == pytest.approx(3.0)
    # Masked in every frame -> no valid data -> NaN (not a fake 0.0 value).
    assert np.isnan(result.data[0, 1])
    assert result.num_clipped == 0


def test_combine_images_robust_validates_method_weights_and_return_type() -> None:
    arrays = [np.ones((2, 2)), np.ones((2, 2))]

    with pytest.raises(ValueError, match="Unknown combining method"):
        combine_images_robust(arrays, method="mystery")

    with pytest.raises(ImageProcessingError, match="Number of weights"):
        combine_images_robust(
            arrays,
            method="weighted_mean",
            weights=[1.0],
        )

    with pytest.raises(ValueError, match="return_type"):
        combine_images_robust(arrays, return_type="mystery")


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


def test_combine_images_robust_accepts_single_path_inputs(tmp_path: Path) -> None:
    file1 = tmp_path / "path_a.fits"
    file2 = tmp_path / "path_b.fits"

    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(file1)
    fits.PrimaryHDU(data=np.full((2, 2), 3.0, dtype=np.float32)).writeto(file2)

    from_file = combine_images_robust(
        file1,
        method="mean",
        return_type="array",
    )
    from_directory = combine_images_robust(
        tmp_path,
        method="mean",
        return_type="array",
    )

    assert np.allclose(from_file, np.ones((2, 2)))
    assert np.allclose(from_directory, np.full((2, 2), 2.0))


def test_combine_images_robust_pairs_named_hdu_across_reordered_mefs(
    tmp_path: Path,
) -> None:
    file1 = tmp_path / "ordered.fits"
    file2 = tmp_path / "reordered.fits"

    def science_hdu(value: float, extver: int) -> fits.ImageHDU:
        hdu = fits.ImageHDU(
            data=np.full((2, 2), value, dtype=np.float32), name="SCI"
        )
        hdu.header["EXTVER"] = extver
        return hdu

    fits.HDUList(
        [fits.PrimaryHDU(), science_hdu(10, 1), science_hdu(20, 2)]
    ).writeto(file1)
    fits.HDUList(
        [fits.PrimaryHDU(), science_hdu(30, 2), science_hdu(40, 1)]
    ).writeto(file2)

    combined = combine_images_robust(
        [file1, file2],
        method="mean",
        hdu=("SCI", 2),
        return_type="array",
    )

    assert np.array_equal(combined, np.full((2, 2), 25.0))


def test_combine_images_robust_rejects_numeric_identity_mismatch(
    tmp_path: Path,
) -> None:
    file1 = tmp_path / "sci1.fits"
    file2 = tmp_path / "sci2.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI")
    sci2 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI")
    sci1.header["EXTVER"] = 1
    sci2.header["EXTVER"] = 2
    fits.HDUList([fits.PrimaryHDU(), sci1]).writeto(file1)
    fits.HDUList([fits.PrimaryHDU(), sci2]).writeto(file2)

    with pytest.raises(ImageProcessingError, match="identity mismatch"):
        combine_images_robust([file1, file2], hdu=1)


def test_combine_images_robust_rejects_missing_named_partner(
    tmp_path: Path,
) -> None:
    file1 = tmp_path / "has_sci2.fits"
    file2 = tmp_path / "missing_sci2.fits"
    sci2 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI")
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI")
    sci2.header["EXTVER"] = 2
    sci1.header["EXTVER"] = 1
    fits.HDUList([fits.PrimaryHDU(), sci2]).writeto(file1)
    fits.HDUList([fits.PrimaryHDU(), sci1]).writeto(file2)

    with pytest.raises(ImageProcessingError, match="missing_sci2.*No HDU"):
        combine_images_robust([file1, file2], hdu=("SCI", 2))


def test_combine_images_robust_rejects_ambiguous_bare_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI")
    sci2 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI")
    sci1.header["EXTVER"] = 1
    sci2.header["EXTVER"] = 2
    fits.HDUList([fits.PrimaryHDU(), sci1, sci2]).writeto(path)

    with pytest.raises(ImageProcessingError, match="Ambiguous EXTNAME"):
        combine_images_robust([path], hdu="SCI")


def test_combine_image_hdus_pairs_reordered_mefs_and_preserves_structure(
    tmp_path: Path,
) -> None:
    file1 = tmp_path / "plural_first.fits"
    file2 = tmp_path / "plural_second.fits"
    output = tmp_path / "plural_combined.fits"

    def sci(value: float, version: int, shape: tuple[int, int]) -> fits.ImageHDU:
        hdu = fits.ImageHDU(
            data=np.full(shape, value, dtype=np.float32), name="SCI"
        )
        hdu.header["EXTVER"] = version
        return hdu

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            sci(10, 1, (2, 2)),
            fits.BinTableHDU.from_columns([], name="KEEP"),
            sci(20, 2, (3, 2)),
        ]
    ).writeto(file1)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            sci(40, 2, (3, 2)),
            sci(30, 1, (2, 2)),
        ]
    ).writeto(file2)

    results = combine_image_hdus(
        [file1, file2],
        output,
        query=HDUQuery(extnames=frozenset({"SCI"})),
        method="mean",
    )

    assert len(results) == 2
    with fits.open(output) as hdul:
        assert np.all(hdul[("SCI", 1)].data == 20)
        assert hdul[2].name == "KEEP"
        assert np.all(hdul[("SCI", 2)].data == 30)


def test_combine_image_hdus_missing_late_partner_rolls_back(
    tmp_path: Path,
) -> None:
    file1 = tmp_path / "plural_complete.fits"
    file2 = tmp_path / "plural_missing.fits"
    destination = tmp_path / "plural_destination.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1)
    sci2 = fits.ImageHDU(data=np.full((2, 2), 2), name="SCI", ver=2)
    fits.HDUList([fits.PrimaryHDU(), sci1, sci2]).writeto(file1)
    fits.HDUList([fits.PrimaryHDU(), sci1.copy()]).writeto(file2)
    fits.HDUList([fits.PrimaryHDU(), sci1.copy(), sci2.copy()]).writeto(
        destination
    )
    before = destination.read_bytes()

    with pytest.raises(ImageProcessingError, match="No HDU matches selector"):
        combine_image_hdus(
            [file1, file2],
            destination,
            query=HDUQuery(extnames=frozenset({"SCI"})),
            method="mean",
        )

    assert destination.read_bytes() == before


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


# ---------------------------------------------------------------------------
# Regression tests: weights, masks, and clip accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["median", "sigma_clipped_median"])
def test_combine_images_robust_rejects_weights_for_median_methods(method: str) -> None:
    # Previously weights were silently ignored while WEIGHTED=True was written.
    arrays = [np.ones((2, 2)), np.ones((2, 2)) * 2.0, np.ones((2, 2)) * 3.0]

    with pytest.raises(ValueError, match="weights are not supported"):
        combine_images_robust(arrays, method=method, weights=[1.0, 1.0, 1.0])


def test_combine_images_robust_sigma_clipped_mean_applies_weights() -> None:
    arrays = [np.full((3, 3), 1.0), np.full((3, 3), 2.0), np.full((3, 3), 4.0)]

    result = combine_images_robust(
        arrays, method="sigma_clipped_mean", weights=[1.0, 1.0, 2.0]
    )

    assert isinstance(result, CombiningResult)
    assert np.allclose(result.data, (1.0 + 2.0 + 8.0) / 4.0)
    assert result.header["WEIGHTED"] is True


@pytest.mark.parametrize(
    "method", ["median", "mean", "weighted_mean", "sigma_clipped_mean", "sigma_clipped_median"]
)
def test_combine_images_robust_masked_output_is_plain_ndarray(method: str) -> None:
    arrays = [np.full((3, 3), v) for v in (1.0, 2.0, 3.0)]
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    result = combine_images_robust(arrays, method=method, mask=mask)

    assert isinstance(result, CombiningResult)
    assert type(result.data) is np.ndarray
    assert np.isnan(result.data[1, 1])
    assert np.allclose(np.delete(result.data.ravel(), 4), 2.0)


def test_combine_images_robust_fill_value_is_configurable() -> None:
    arrays = [np.full((2, 2), v) for v in (1.0, 2.0, 3.0)]
    mask = np.array([[True, False], [False, False]])

    result = combine_images_robust(
        arrays, method="sigma_clipped_median", mask=mask, fill_value=-1.0
    )

    assert isinstance(result, CombiningResult)
    assert result.data[0, 0] == -1.0


def test_combine_images_robust_nclipped_excludes_premasked_pixels() -> None:
    arrays = [np.full((5, 5), 100.0) for _ in range(9)]
    arrays[4][2, 2] = 1000.0  # one genuine outlier (~3.2 sigma by std)
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, :] = True  # 5 pixels x 9 frames pre-masked

    result = combine_images_robust(
        arrays, method="sigma_clipped_median", mask=mask, sigma=2.0
    )

    assert isinstance(result, CombiningResult)
    assert result.num_clipped == 1
    assert result.header["NCLIPPED"] == 1


def test_create_master_frame_writes_final_header_and_normalization(tmp_path: Path) -> None:
    # Previously the file was written before FRMTYPE/HISTORY were added, so
    # the saved master lacked them unless normalize=True forced a rewrite.
    frames = [np.full((4, 4), v) for v in (100.0, 102.0, 104.0)]
    output = tmp_path / "master_bias.fits"

    create_master_frame(frames, "bias", method="median", output_file=output)

    header = fits.getheader(output)
    assert header["FRMTYPE"] == "bias"
    assert "Master bias frame created" in str(header["HISTORY"])
    with pytest.raises(FileOverwriteError):
        create_master_frame(frames, "bias", method="median", output_file=output)
    create_master_frame(frames, "bias", method="median", output_file=output, overwrite=True)


def test_create_master_frame_normalizes_flat_with_masked_pixels(tmp_path: Path) -> None:
    # Pixels masked in every frame are NaN; normalization must ignore them.
    frames = [np.full((4, 4), v) for v in (19000.0, 20000.0, 21000.0)]
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    output = tmp_path / "master_flat.fits"

    result = create_master_frame(
        frames, "flat", method="median", normalize=True, mask=mask, output_file=output
    )

    assert isinstance(result, CombiningResult)
    assert np.isnan(result.data[0, 0])
    assert np.allclose(result.data[1:, :], 1.0)
    saved = fits.getdata(output)
    assert fits.getheader(output)["NORMVAL"] == pytest.approx(20000.0)
    assert np.allclose(saved[1:, :], 1.0)
