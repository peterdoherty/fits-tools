#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import (
    FileHandlingError,
    ImageProcessingError,
    InvalidHDUIndexError,
)
from fits_tools.image_processing.streaming import (
    P2QuantileEstimator,
    _estimate_memory_usage,
    _is_fits_path,
    _load_frame_for_streaming,
    compute_stack_reference,
    streaming_calibrated_combine,
    streaming_combine,
    streaming_statistics,
)


def test_estimate_memory_usage_and_fits_path_detection(tmp_path: Path) -> None:
    fits_path = tmp_path / "frame.fits"
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32)).writeto(fits_path)

    traditional = _estimate_memory_usage([fits_path])
    mean_stream = _estimate_memory_usage([fits_path], method="mean")

    assert traditional > 0
    assert mean_stream > 0
    assert _is_fits_path(fits_path) is True
    assert _is_fits_path(tmp_path / "frame.npy") is False


def test_p2_quantile_estimator_handles_bootstrap_and_updates() -> None:
    estimator = P2QuantileEstimator((2, 2), p=0.5)

    for value in [1, 2, 3, 4, 5, 6]:
        estimator.update(np.full((2, 2), value, dtype=np.float32))

    median = estimator.get_quantile()
    assert estimator.count == 6
    assert median.shape == (2, 2)
    assert np.all((median >= 2.0) & (median <= 5.0))


def test_p2_quantile_estimator_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="p must be in"):
        P2QuantileEstimator((2, 2), p=1.0)

    with pytest.raises(ValueError, match="shape must be a 2-tuple"):
        P2QuantileEstimator((2, 2, 2), p=0.5)  # type: ignore[arg-type]


def test_streaming_statistics_computes_mean_and_quantiles(tmp_path: Path) -> None:
    paths = []
    for idx, value in enumerate((1.0, 3.0, 5.0), start=1):
        path = tmp_path / f"stat_{idx}.fits"
        fits.PrimaryHDU(data=np.full((2, 2), value, dtype=np.float32)).writeto(path)
        paths.append(path)

    stats = streaming_statistics(
        paths, compute_quantiles=True, quantiles=[0.5], show_progress=False
    )

    assert stats["count"] == 3
    assert np.allclose(stats["mean"], np.full((2, 2), 3.0))
    assert np.allclose(stats["min"], np.full((2, 2), 1.0))
    assert np.allclose(stats["max"], np.full((2, 2), 5.0))
    assert np.allclose(stats["quantiles"][0.5], np.full((2, 2), 3.0), atol=1.0)


def test_streaming_statistics_rejects_shape_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(a)
    fits.PrimaryHDU(data=np.ones((3, 3), dtype=np.float32)).writeto(b)

    with pytest.raises(ImageProcessingError, match="Shape mismatch"):
        streaming_statistics([a, b], show_progress=False)


def test_streaming_statistics_rejects_invalid_hdu(tmp_path: Path) -> None:
    path = tmp_path / "single_hdu.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(path)

    with pytest.raises(InvalidHDUIndexError):
        streaming_statistics([path], hdu=1, show_progress=False)


def test_streaming_combine_returns_array_and_can_write_output(tmp_path: Path) -> None:
    paths = []
    for idx, value in enumerate((2.0, 4.0), start=1):
        path = tmp_path / f"combine_{idx}.fits"
        fits.PrimaryHDU(data=np.full((2, 2), value, dtype=np.float32)).writeto(path)
        paths.append(path)

    median = streaming_combine(paths, method="median", show_progress=False)
    output_path = tmp_path / "combined_mean.fits"
    written = streaming_combine(
        paths,
        method="mean",
        output_file=output_path,
        overwrite=True,
        show_progress=False,
    )

    assert np.array_equal(median, np.full((2, 2), 3.0))
    assert written == str(output_path)
    assert np.allclose(fits.getdata(output_path), np.full((2, 2), 3.0))


def test_streaming_apis_pair_named_hdu_across_reordered_mefs(
    tmp_path: Path,
) -> None:
    file1 = tmp_path / "stream_ordered.fits"
    file2 = tmp_path / "stream_reordered.fits"

    def sci(value: float, version: int) -> fits.ImageHDU:
        return fits.ImageHDU(
            data=np.full((2, 2), value, dtype=np.float32),
            name="SCI",
            ver=version,
        )

    fits.HDUList([fits.PrimaryHDU(), sci(2, 1), sci(10, 2)]).writeto(file1)
    fits.HDUList([fits.PrimaryHDU(), sci(30, 2), sci(4, 1)]).writeto(file2)

    stats = streaming_statistics([file1, file2], hdu=("SCI", 2))
    combined = streaming_combine([file1, file2], method="mean", hdu=("SCI", 2))

    assert np.all(stats["mean"] == 20)
    assert np.all(combined == 20)


def test_streaming_statistics_rejects_missing_and_ambiguous_named_hdus(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "stream_complete.fits"
    missing = tmp_path / "stream_missing.fits"
    ambiguous = tmp_path / "stream_ambiguous.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1)
    sci2 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=2)
    fits.HDUList([fits.PrimaryHDU(), sci1, sci2]).writeto(complete)
    fits.HDUList([fits.PrimaryHDU(), sci1.copy()]).writeto(missing)
    fits.HDUList([fits.PrimaryHDU(), sci1.copy(), sci2.copy()]).writeto(ambiguous)

    with pytest.raises(ImageProcessingError, match="No HDU matches selector"):
        streaming_statistics([complete, missing], hdu=("SCI", 2))

    with pytest.raises(FileHandlingError, match="Ambiguous EXTNAME"):
        streaming_statistics([ambiguous], hdu="SCI")


def test_streaming_statistics_rejects_numeric_identity_drift(
    tmp_path: Path,
) -> None:
    file1 = tmp_path / "stream_identity1.fits"
    file2 = tmp_path / "stream_identity2.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1),
        ]
    ).writeto(file1)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=2),
        ]
    ).writeto(file2)

    with pytest.raises(ImageProcessingError, match="identity mismatch"):
        streaming_statistics([file1, file2], hdu=1)


def test_compute_stack_reference_supports_chunked_and_streaming_methods(
    tmp_path: Path,
) -> None:
    paths = []
    for idx, value in enumerate((1.0, 2.0, 100.0), start=1):
        path = tmp_path / f"ref_{idx}.fits"
        fits.PrimaryHDU(data=np.full((2, 2), value, dtype=np.float32)).writeto(path)
        paths.append(path)

    bpm = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    chunked_median, chunked_sigma = compute_stack_reference(
        paths, existing_bpm=bpm, method="chunked", show_progress=False
    )
    streaming_median, streaming_sigma = compute_stack_reference(
        paths, method="streaming", show_progress=False
    )

    assert chunked_median.shape == (2, 2)
    assert chunked_sigma.shape == (2, 2)
    assert streaming_median.shape == (2, 2)
    assert streaming_sigma.shape == (2, 2)


def test_compute_stack_reference_resolves_named_hdu_in_each_file(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index, reverse in enumerate((False, True, False), start=1):
        sci1 = fits.ImageHDU(
            data=np.full((2, 2), index, dtype=np.float32),
            name="SCI",
            ver=1,
        )
        sci2 = fits.ImageHDU(
            data=np.full((2, 2), index * 10, dtype=np.float32),
            name="SCI",
            ver=2,
        )
        path = tmp_path / f"stream_reference_{index}.fits"
        extensions = [sci2, sci1] if reverse else [sci1, sci2]
        fits.HDUList([fits.PrimaryHDU(), *extensions]).writeto(path)
        paths.append(path)

    median, sigma = compute_stack_reference(
        paths,
        hdu=("SCI", 2),
        method="chunked",
    )

    assert np.all(median == 20)
    assert median.shape == sigma.shape == (2, 2)


def test_load_frame_for_streaming_supports_numpy_formats_and_rejects_unknown_files(
    tmp_path: Path,
) -> None:
    npy_path = tmp_path / "frame.npy"
    npz_path = tmp_path / "frame.npz"
    txt_path = tmp_path / "frame.txt"
    np.save(npy_path, np.array([[1.0, 2.0]], dtype=np.float32))
    np.savez(npz_path, arr=np.array([[3.0, 4.0]], dtype=np.float32))
    txt_path.write_text("not an array", encoding="utf-8")

    assert np.array_equal(
        _load_frame_for_streaming(npy_path), np.array([[1.0, 2.0]], dtype=np.float64)
    )
    assert np.array_equal(
        _load_frame_for_streaming(npz_path), np.array([[3.0, 4.0]], dtype=np.float64)
    )

    with pytest.raises(FileHandlingError, match="Unsupported file format"):
        _load_frame_for_streaming(txt_path)


# ---------------------------------------------------------------------------
# Regression tests: P² correctness
# ---------------------------------------------------------------------------


def test_p2_median_converges_to_exact_median() -> None:
    # Previously, missing neighbour-spacing guards let markers collide and the
    # worst-pixel error stayed above 1 sigma even at N=100.
    rng = np.random.default_rng(12345)
    frames = rng.normal(0.0, 1.0, size=(100, 50, 50))
    estimator = P2QuantileEstimator((50, 50), p=0.5)
    for frame in frames:
        estimator.update(frame)

    error = estimator.get_quantile() - np.median(frames, axis=0)
    assert np.sqrt(np.mean(error**2)) < 0.08
    assert np.max(np.abs(error)) < 0.6


def test_p2_marker_positions_stay_strictly_increasing() -> None:
    rng = np.random.default_rng(7)
    estimator = P2QuantileEstimator((20, 20), p=0.5)
    for _ in range(60):
        estimator.update(rng.exponential(1.0, size=(20, 20)))

    assert np.all(np.diff(estimator.n, axis=0) >= 1)
    assert np.all(np.diff(estimator.q, axis=0) >= 0)


@pytest.mark.parametrize("p", [0.1, 0.25, 0.9])
def test_p2_bootstrap_returns_requested_quantile(p: float) -> None:
    # With fewer than five samples the exact quantile is returned; previously
    # the median was returned regardless of p.
    samples = [np.full((2, 2), v, dtype=np.float64) for v in (1.0, 2.0, 4.0, 8.0)]
    estimator = P2QuantileEstimator((2, 2), p=p)
    for sample in samples:
        estimator.update(sample)

    expected = np.quantile(np.stack(samples), p, axis=0)
    assert np.allclose(estimator.get_quantile(), expected)


# ---------------------------------------------------------------------------
# Regression tests: unsigned / scaled (BZERO) input
# ---------------------------------------------------------------------------


def _write_uint16_frames(tmp_path: Path, n: int = 3) -> list[Path]:
    paths = []
    for i in range(n):
        path = tmp_path / f"u16_{i}.fits"
        fits.PrimaryHDU(data=np.full((4, 4), 1000 + i, dtype=np.uint16)).writeto(path)
        paths.append(path)
    return paths


def test_compute_stack_reference_default_handles_uint16(tmp_path: Path) -> None:
    paths = _write_uint16_frames(tmp_path)

    median, sigma = compute_stack_reference(paths)

    assert np.allclose(median, 1001.0)
    assert sigma.shape == (4, 4)


def test_estimate_memory_usage_reads_uint16_header_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = _write_uint16_frames(tmp_path, n=2)

    with caplog.at_level("WARNING"):
        estimate = _estimate_memory_usage(paths)

    assert estimate == pytest.approx(2 * 16 * 2 / 1024**3)
    assert "Could not estimate" not in caplog.text


# ---------------------------------------------------------------------------
# streaming_calibrated_combine
# ---------------------------------------------------------------------------


def _write_raw_frames(
    tmp_path: Path,
    values: list[float],
    *,
    exptime: float | None = 2.0,
    hot_pixel: float | None = None,
) -> list[Path]:
    paths = []
    for i, value in enumerate(values):
        data = np.full((4, 4), value, dtype=np.float32)
        if hot_pixel is not None and i == 0:
            data[0, 0] = hot_pixel
        header = fits.Header()
        if exptime is not None:
            header["EXPTIME"] = exptime
        path = tmp_path / f"raw_{i}.fits"
        fits.PrimaryHDU(data=data, header=header).writeto(path)
        paths.append(path)
    return paths


def test_streaming_calibrated_combine_subtracts_bias_and_scaled_dark(
    tmp_path: Path,
) -> None:
    paths = _write_raw_frames(tmp_path, [1120.0, 1130.0, 1140.0])

    combined = streaming_calibrated_combine(
        paths,
        master_bias_data=np.full((4, 4), 100.0),
        master_dark_data=np.full((4, 4), 10.0),
        master_dark_exptime=1.0,
        method="mean",
    )

    # (raw - 100) - 10 * (2.0 / 1.0)
    assert isinstance(combined, np.ndarray)
    assert np.allclose(combined, 1010.0)


def test_streaming_calibrated_combine_rejects_frame_without_exptime(
    tmp_path: Path,
) -> None:
    # Previously the dark was silently skipped for such frames.
    paths = _write_raw_frames(tmp_path, [1000.0, 1000.0], exptime=None)

    with pytest.raises(ImageProcessingError, match="no EXPTIME"):
        streaming_calibrated_combine(
            paths,
            master_dark_data=np.full((4, 4), 10.0),
            master_dark_exptime=1.0,
            method="mean",
        )


def test_streaming_calibrated_combine_validates_dark_exptime() -> None:
    with pytest.raises(ValueError, match="master_dark_exptime must be positive"):
        streaming_calibrated_combine(["unused.fits"], master_dark_data=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="max_saturated_pixels"):
        streaming_calibrated_combine(["unused.fits"], max_saturated_pixels=-1)


def test_streaming_calibrated_combine_reports_saturation_rejections(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = _write_raw_frames(tmp_path, [100.0, 200.0, 300.0], hot_pixel=65535.0)
    output = tmp_path / "combined.fits"

    with caplog.at_level("WARNING"):
        result = streaming_calibrated_combine(
            paths,
            saturation_threshold_adu=65000.0,
            method="mean",
            output_file=output,
        )

    assert result == str(output)
    data, header = fits.getdata(output, header=True)
    assert np.allclose(data, 250.0)
    assert header["NREJSAT"] == 1
    assert header["NCOMBINE"] == 2
    assert header["NINPUT"] == 3
    assert "raw_0.fits" in caplog.text


def test_streaming_calibrated_combine_tolerates_saturated_pixel_limit(
    tmp_path: Path,
) -> None:
    paths = _write_raw_frames(tmp_path, [100.0, 200.0], hot_pixel=65535.0)

    combined = streaming_calibrated_combine(
        paths,
        saturation_threshold_adu=65000.0,
        max_saturated_pixels=1,
        method="max",
    )

    assert isinstance(combined, np.ndarray)
    assert combined[0, 0] == pytest.approx(65535.0)


def test_streaming_calibrated_combine_all_frames_saturated(tmp_path: Path) -> None:
    paths = _write_raw_frames(tmp_path, [100.0, 200.0])

    with pytest.raises(ValueError, match="All 2 frames were rejected"):
        streaming_calibrated_combine(paths, saturation_threshold_adu=0.0)


def test_streaming_calibrated_combine_applies_bad_pixel_mask(tmp_path: Path) -> None:
    # Previously the mask was only shape-checked and never applied.
    paths = _write_raw_frames(tmp_path, [100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    bpm = np.zeros((4, 4), dtype=np.uint8)
    bpm[1, 2] = 4

    combined = streaming_calibrated_combine(paths, bad_pixel_mask=bpm, method="median")

    assert isinstance(combined, np.ndarray)
    assert np.isnan(combined[1, 2])
    assert np.count_nonzero(np.isnan(combined)) == 1


def test_streaming_calibrated_combine_rejects_mask_shape_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_raw_frames(tmp_path, [100.0])

    with pytest.raises(ImageProcessingError, match="Bad-pixel mask shape"):
        streaming_calibrated_combine(paths, bad_pixel_mask=np.zeros((2, 2), dtype=bool))


# ---------------------------------------------------------------------------
# Exact (banded) streaming median
# ---------------------------------------------------------------------------


def _write_random_frames(
    tmp_path: Path, n: int, shape: tuple[int, int] = (9, 7), seed: int = 0
) -> tuple[list[Path], np.ndarray]:
    rng = np.random.default_rng(seed)
    stack = rng.normal(1000.0, 30.0, size=(n, *shape)).astype(np.float32)
    paths = []
    for i, frame in enumerate(stack):
        path = tmp_path / f"frame_{n}_{i}.fits"
        fits.PrimaryHDU(data=frame, header=fits.Header({"EXPTIME": 2.0})).writeto(path)
        paths.append(path)
    return paths, stack.astype(np.float64)


@pytest.mark.parametrize("n", [1, 2, 6, 7])
@pytest.mark.parametrize("max_memory_mb", [1024.0, 1e-6])
def test_streaming_combine_median_is_exact(
    tmp_path: Path, n: int, max_memory_mb: float
) -> None:
    # 1e-6 MB forces one-row bands; the result must still be exact.
    paths, stack = _write_random_frames(tmp_path, n)

    combined = streaming_combine(paths, method="median", max_memory_mb=max_memory_mb)

    assert np.array_equal(combined, np.median(stack, axis=0))


def test_streaming_combine_exact_median_reads_scaled_uint16(tmp_path: Path) -> None:
    values = [60000, 65535, 61000]
    paths = []
    for i, value in enumerate(values):
        path = tmp_path / f"u16_{i}.fits"
        fits.PrimaryHDU(data=np.full((4, 3), value, dtype=np.uint16)).writeto(path)
        paths.append(path)

    combined = streaming_combine(paths, method="median", max_memory_mb=1e-6)

    assert np.array_equal(combined, np.full((4, 3), 61000.0))


def test_streaming_combine_exact_median_selects_named_mef_hdu(tmp_path: Path) -> None:
    paths = []
    for i, value in enumerate((1.0, 5.0, 3.0)):
        path = tmp_path / f"mef_{i}.fits"
        hdus = [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((3, 3), -value), name="SCI", ver=1),
            fits.ImageHDU(data=np.full((3, 3), value), name="SCI", ver=2),
        ]
        if i == 1:  # reorder extensions in one file
            hdus = [hdus[0], hdus[2], hdus[1]]
        fits.HDUList(hdus).writeto(path)
        paths.append(path)

    combined = streaming_combine(paths, method="median", hdu=("SCI", 2))

    assert np.array_equal(combined, np.full((3, 3), 3.0))


def test_streaming_combine_p2_option_and_validation(tmp_path: Path) -> None:
    paths, stack = _write_random_frames(tmp_path, 20, shape=(20, 20))

    p2 = streaming_combine(paths, method="median", median_algorithm="p2")

    assert isinstance(p2, np.ndarray)
    assert not np.array_equal(p2, np.median(stack, axis=0))
    with pytest.raises(ValueError, match="median_algorithm"):
        streaming_combine(paths, method="median", median_algorithm="approx")
    with pytest.raises(ValueError, match="max_memory_mb"):
        streaming_combine(paths, method="median", max_memory_mb=0)


def test_streaming_combine_exact_median_applies_transform(tmp_path: Path) -> None:
    paths, stack = _write_random_frames(tmp_path, 6)
    skip_value = float(stack[0, 0, 0])

    def transform(data: np.ndarray, header: fits.Header) -> np.ndarray | None:
        if data[0, 0] == skip_value:
            return None  # drop the first frame
        return data - 100.0

    combined = streaming_combine(
        paths, method="median", transform=transform, max_memory_mb=1e-6
    )

    assert np.allclose(combined, np.median(stack[1:] - 100.0, axis=0))


def test_streaming_combine_exact_median_rejects_nondeterministic_transform(
    tmp_path: Path,
) -> None:
    paths, _ = _write_random_frames(tmp_path, 4)
    calls = 0

    def flaky(data: np.ndarray, header: fits.Header) -> np.ndarray | None:
        nonlocal calls
        calls += 1
        return None if calls == 5 else data  # skips a frame only in band 2

    with pytest.raises(ImageProcessingError, match="must be deterministic"):
        streaming_combine(paths, method="median", transform=flaky, max_memory_mb=1e-6)


def test_streaming_combine_records_median_algorithm(tmp_path: Path) -> None:
    paths, _ = _write_random_frames(tmp_path, 3)
    output = tmp_path / "median.fits"

    streaming_combine(paths, method="median", output_file=output)

    assert fits.getheader(output)["MEDALG"] == "exact"


def test_streaming_calibrated_combine_exact_median(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths, stack = _write_random_frames(tmp_path, 7)
    with fits.open(paths[3], mode="update") as hdul:
        hdul[0].data[2, 2] = 70000.0  # saturate one frame
    bias = np.full(stack.shape[1:], 100.0)
    dark = np.full(stack.shape[1:], 10.0)
    output = tmp_path / "cal_median.fits"

    with caplog.at_level("WARNING"):
        streaming_calibrated_combine(
            paths,
            master_bias_data=bias,
            master_dark_data=dark,
            master_dark_exptime=1.0,
            saturation_threshold_adu=65000.0,
            max_memory_mb=1e-6,
            output_file=output,
        )

    kept = np.delete(stack, 3, axis=0)
    expected = np.median(
        kept.astype(np.float32) - np.float32(100.0) - np.float32(20.0), axis=0
    )
    data, header = fits.getdata(output, header=True)
    assert np.allclose(data, expected, rtol=0, atol=1e-3)
    assert header["NREJSAT"] == 1
    assert header["NCOMBINE"] == 6
    assert header["MEDALG"] == "exact"
    assert "frame_7_3.fits" in caplog.text


# ---------------------------------------------------------------------------
# compute_stack_reference: exact, bounded-memory median and MAD sigma
# ---------------------------------------------------------------------------


def _numpy_reference(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(stack, axis=0)
    return median, 1.4826 * np.median(np.abs(stack - median), axis=0)


@pytest.mark.parametrize("n", [1, 4, 7])
@pytest.mark.parametrize("max_memory_mb", [1024.0, 1e-6])
def test_compute_stack_reference_is_exact(
    tmp_path: Path, n: int, max_memory_mb: float
) -> None:
    # 1e-6 MB forces one-row bands; results must match NumPy exactly.
    paths, stack = _write_random_frames(tmp_path, n)
    expected_median, expected_sigma = _numpy_reference(stack)

    median, sigma = compute_stack_reference(paths, max_memory_mb=max_memory_mb)

    assert np.array_equal(median, expected_median)
    assert np.allclose(sigma, expected_sigma, rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", ["auto", "chunked", "streaming"])
def test_compute_stack_reference_legacy_method_names_are_exact(
    tmp_path: Path, method: str
) -> None:
    paths, stack = _write_random_frames(tmp_path, 60, shape=(4, 3))

    median, _ = compute_stack_reference(paths, method=method)

    assert np.array_equal(median, np.median(stack, axis=0))


def test_compute_stack_reference_masks_bpm_pixels_as_nan(tmp_path: Path) -> None:
    # Previously the in-memory path filled masked pixels with 0, giving sigma=0.
    paths, stack = _write_random_frames(tmp_path, 5)
    bpm = np.zeros(stack.shape[1:], dtype=np.uint32)
    bpm[2, 3] = 4
    expected_median, _ = _numpy_reference(stack)

    median, sigma = compute_stack_reference(paths, existing_bpm=bpm, max_memory_mb=1e-6)

    assert np.isnan(median[2, 3]) and np.isnan(sigma[2, 3])
    good = bpm == 0
    assert np.array_equal(median[good], expected_median[good])
    with pytest.raises(ValueError, match="existing_bpm shape"):
        compute_stack_reference(paths, existing_bpm=np.zeros((2, 2)))


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_compute_stack_reference_reads_numpy_frames(
    tmp_path: Path, suffix: str
) -> None:
    rng = np.random.default_rng(5)
    stack = rng.normal(100.0, 3.0, size=(5, 6, 4))
    paths = []
    for i, frame in enumerate(stack):
        path = tmp_path / f"frame_{i}{suffix}"
        if suffix == ".npy":
            np.save(path, frame)
        else:
            np.savez(path, data=frame)
        paths.append(path)
    expected_median, expected_sigma = _numpy_reference(stack)

    median, sigma = compute_stack_reference(paths, max_memory_mb=1e-6)

    assert np.array_equal(median, expected_median)
    assert np.allclose(sigma, expected_sigma, rtol=0, atol=1e-12)


def test_compute_stack_reference_p2_option_and_validation(tmp_path: Path) -> None:
    paths, stack = _write_random_frames(tmp_path, 20, shape=(20, 20))

    p2_median, _ = compute_stack_reference(paths, method="p2")

    assert not np.array_equal(p2_median, np.median(stack, axis=0))
    with pytest.raises(ValueError, match="method must be one of"):
        compute_stack_reference(paths, method="approximate")
    with pytest.raises(ValueError, match="max_memory_mb"):
        compute_stack_reference(paths, max_memory_mb=0)


def test_compute_stack_reference_rejects_mismatched_numpy_frame(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.zeros((4, 4)))
    np.save(tmp_path / "b.npy", np.zeros((5, 4)))

    with pytest.raises(ImageProcessingError, match="Shape mismatch"):
        compute_stack_reference([tmp_path / "a.npy", tmp_path / "b.npy"])


# ---------------------------------------------------------------------------
# streaming_statistics: exact quantiles with bounded memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7])
@pytest.mark.parametrize("max_memory_mb", [1024.0, 1e-6])
def test_streaming_statistics_quantiles_are_exact(
    tmp_path: Path, n: int, max_memory_mb: float
) -> None:
    paths, stack = _write_random_frames(tmp_path, n)
    qs = [0.1, 0.5, 0.9]

    stats = streaming_statistics(
        paths, compute_quantiles=True, quantiles=qs, max_memory_mb=max_memory_mb
    )

    expected = np.quantile(stack, qs, axis=0)
    for i, q in enumerate(qs):
        assert np.array_equal(stats["quantiles"][q], expected[i])
    assert stats["count"] == n
    assert np.allclose(stats["mean"], stack.mean(axis=0), rtol=0, atol=1e-9)
    assert np.allclose(stats["sum"], stack.sum(axis=0), rtol=0, atol=1e-6)
    assert np.array_equal(stats["min"], stack.min(axis=0))
    assert np.array_equal(stats["max"], stack.max(axis=0))
    expected_var = stack.var(axis=0, ddof=1) if n > 1 else np.zeros(stack.shape[1:])
    assert np.allclose(stats["var"], expected_var, rtol=1e-12, atol=1e-12)
    assert np.allclose(stats["std"], np.sqrt(expected_var), rtol=1e-12, atol=1e-12)


def test_streaming_statistics_exact_matches_welford_statistics(tmp_path: Path) -> None:
    paths, _ = _write_random_frames(tmp_path, 9)

    single_pass = streaming_statistics(paths)
    banded = streaming_statistics(paths, compute_quantiles=True, max_memory_mb=1e-6)

    for key in ("mean", "var", "std", "min", "max", "sum"):
        assert np.allclose(banded[key], single_pass[key], rtol=1e-10, atol=1e-9), key
    assert list(banded["quantiles"]) == [0.5]


def test_streaming_statistics_exact_quantiles_apply_transform(tmp_path: Path) -> None:
    paths, stack = _write_random_frames(tmp_path, 5)
    skip_value = float(stack[2, 0, 0])

    def transform(data: np.ndarray, header: fits.Header) -> np.ndarray | None:
        return None if data[0, 0] == skip_value else data * 2.0

    stats = streaming_statistics(
        paths, compute_quantiles=True, transform=transform, max_memory_mb=1e-6
    )

    kept = np.delete(stack, 2, axis=0) * 2.0
    assert stats["count"] == 4
    assert np.allclose(stats["quantiles"][0.5], np.median(kept, axis=0))


def test_streaming_statistics_p2_option_and_validation(tmp_path: Path) -> None:
    paths, stack = _write_random_frames(tmp_path, 20, shape=(20, 20))

    p2 = streaming_statistics(paths, compute_quantiles=True, quantile_algorithm="p2")

    assert not np.array_equal(p2["quantiles"][0.5], np.median(stack, axis=0))
    with pytest.raises(ValueError, match="quantile_algorithm"):
        streaming_statistics(paths, compute_quantiles=True, quantile_algorithm="approx")
    with pytest.raises(ValueError, match="max_memory_mb"):
        streaming_statistics(paths, max_memory_mb=0)
    with pytest.raises(ValueError, match=r"Quantile must be in \(0, 1\)"):
        streaming_statistics(paths, compute_quantiles=True, quantiles=[1.5])
