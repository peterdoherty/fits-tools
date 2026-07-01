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

    stats = streaming_statistics(paths, compute_quantiles=True, quantiles=[0.5], show_progress=False)

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
    written = streaming_combine(paths, method="mean", output_file=output_path, overwrite=True, show_progress=False)

    assert np.allclose(median, np.full((2, 2), 3.0), atol=1.0)
    assert written == str(output_path)
    assert np.allclose(fits.getdata(output_path), np.full((2, 2), 3.0))


def test_compute_stack_reference_supports_chunked_and_streaming_methods(tmp_path: Path) -> None:
    paths = []
    for idx, value in enumerate((1.0, 2.0, 100.0), start=1):
        path = tmp_path / f"ref_{idx}.fits"
        fits.PrimaryHDU(data=np.full((2, 2), value, dtype=np.float32)).writeto(path)
        paths.append(path)

    bpm = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    chunked_median, chunked_sigma = compute_stack_reference(paths, existing_bpm=bpm, method="chunked", show_progress=False)
    streaming_median, streaming_sigma = compute_stack_reference(paths, method="streaming", show_progress=False)

    assert chunked_median.shape == (2, 2)
    assert chunked_sigma.shape == (2, 2)
    assert streaming_median.shape == (2, 2)
    assert streaming_sigma.shape == (2, 2)


def test_load_frame_for_streaming_supports_numpy_formats_and_rejects_unknown_files(tmp_path: Path) -> None:
    npy_path = tmp_path / "frame.npy"
    npz_path = tmp_path / "frame.npz"
    txt_path = tmp_path / "frame.txt"
    np.save(npy_path, np.array([[1.0, 2.0]], dtype=np.float32))
    np.savez(npz_path, arr=np.array([[3.0, 4.0]], dtype=np.float32))
    txt_path.write_text("not an array", encoding="utf-8")

    assert np.array_equal(_load_frame_for_streaming(npy_path), np.array([[1.0, 2.0]], dtype=np.float64))
    assert np.array_equal(_load_frame_for_streaming(npz_path), np.array([[3.0, 4.0]], dtype=np.float64))

    with pytest.raises(FileHandlingError, match="Unsupported file format"):
        _load_frame_for_streaming(txt_path)
