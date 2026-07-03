#!/usr/bin/env python3
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Memory-efficient streaming algorithms for large FITS datasets.

This module provides streaming (online) algorithms that process images
sequentially with minimal memory usage. These algorithms are essential
for processing datasets that exceed available RAM.

Features
--------
- Constant memory usage regardless of dataset size
- Single-pass algorithms (no need to load data multiple times)
- Per-pixel processing for quantiles (P² estimator)
- Numerically stable variance calculation (Welford)
- Optional progress reporting for long operations
- Support for memory-mapped FITS files

Notes
-----
This module is FITS-aware primarily through file reading and header handling.
Array-level algorithms are general and operate on NumPy arrays.

Version
-------
2.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from ..cli_utils import maybe_tqdm
from ..constants import FITS_FILE_EXTENSIONS, MAD_TO_STD_FACTOR
from ..exceptions import (
    FileHandlingError,
    ImageProcessingError,
    InvalidHDUIndexError,
)
from ..hdu import (
    HDUIdentity,
    HDUSelector,
    get_hdu_identity,
    resolve_hdu_index,
)
from ..header_utils import add_history
from ..io import write_simple_fits
from ..types import StrOrPath

logger = logging.getLogger(__name__)


def _streaming_identity(
    identity: HDUIdentity,
) -> tuple[str | None, int | None, str, tuple[int, ...] | None]:
    """Return cross-file identity fields relevant to streaming."""
    return (
        identity.extname,
        identity.extver,
        identity.hdu_type,
        identity.shape,
    )


def _resolve_streaming_hdu(
    hdul: fits.HDUList,
    path: Path,
    hdu: HDUSelector,
    *,
    expected_identity: tuple[
        str | None, int | None, str, tuple[int, ...] | None
    ]
    | None = None,
) -> tuple[int, tuple[str | None, int | None, str, tuple[int, ...] | None]]:
    """Resolve and validate one HDU without loading its pixel array."""
    index = resolve_hdu_index(
        hdul,
        hdu,
        auto="first_sci_image",
        image_like=True,
        require_data=False,
        filepath=path,
    )
    identity = _streaming_identity(get_hdu_identity(hdul[index], index))
    if (
        expected_identity is not None
        and identity[3] != expected_identity[3]
    ):
        raise ImageProcessingError(
            f"Shape mismatch in {path}: selected HDU has shape "
            f"{identity[3]}, expected {expected_identity[3]}"
        )
    if expected_identity is not None and identity[:3] != expected_identity[:3]:
        raise ImageProcessingError(
            f"Selected HDU identity mismatch in {path}: {identity!r} does "
            f"not match reference {expected_identity!r}"
        )
    return index, identity


# ============================================================================
# Utility Functions
# ============================================================================


def _estimate_memory_usage(
    file_list: list[StrOrPath],
    hdu: HDUSelector = 0,
    method: str | None = None,
) -> float:
    """Estimate memory usage in GB for various strategies.

    Parameters
    ----------
    file_list
        List of FITS files.
    hdu
        HDU selector.
    method
        If provided, estimate memory for a streaming method:

        - ``None``   : traditional (all images in memory)
        - ``"mean"`` : Welford accumulators
        - ``"median"`` : P² estimator buffers/markers

    Returns
    -------
    float
        Estimated memory usage in GB.
    """
    if not file_list:
        return 0.0

    first = Path(file_list[0])
    try:
        with fits.open(first, memmap=True) as hdul:
            index, _ = _resolve_streaming_hdu(hdul, first, hdu)
            data = hdul[index].data
            if data is None:
                return 0.0
            shape = data.shape
            dtype = data.dtype
    except Exception as e:  # pragma: no cover
        logger.warning("Could not estimate memory usage from %s: %s", first, e)
        return 0.0

    bytes_per_pixel = int(np.dtype(dtype).itemsize)
    pixels_per_image = int(np.prod(shape))

    if method is None:
        total_bytes = bytes_per_pixel * pixels_per_image * len(file_list)
    elif method == "mean":
        # mean + M2 + sum + min + max (5 float64 arrays)
        total_bytes = 8 * pixels_per_image * 5
    elif method == "median":
        # P²: q (5 float64) + n (5 int64) + n_desired (5 float64) + init buffer (~5 images)
        total_bytes = 8 * pixels_per_image * (5 + 5 + 5 + 5)
    else:
        total_bytes = bytes_per_pixel * pixels_per_image * len(file_list)

    return total_bytes / (1024**3)


def _is_fits_path(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(sfx) for sfx in FITS_FILE_EXTENSIONS)


def _get_streaming_file_identity(
    path: StrOrPath,
    hdu: HDUSelector,
    *,
    memmap: bool,
) -> tuple[str | None, int | None, str, tuple[int, ...] | None] | None:
    """Return the selected FITS identity, or None for NumPy array files."""
    resolved_path = Path(path)
    if not _is_fits_path(resolved_path):
        return None
    try:
        with fits.open(resolved_path, memmap=memmap) as hdul:
            _, identity = _resolve_streaming_hdu(
                hdul, resolved_path, hdu
            )
            return identity
    except Exception as exc:
        raise FileHandlingError(
            f"Failed to resolve selected HDU in {resolved_path}: {exc}",
            filepath=resolved_path,
        ) from exc


# ============================================================================
# P² Quantile Estimator
# ============================================================================


class P2QuantileEstimator:
    """
    P² algorithm for streaming per-pixel quantile estimation.

    Parameters
    ----------
    shape
        Image shape (rows, cols).
    p
        Quantile to estimate (0 < p < 1). Default is 0.5 (median).
    """

    def __init__(self, shape: tuple[int, int], p: float = 0.5) -> None:
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")

        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError(f"shape must be a 2-tuple, got {shape!r}")

        if not all(isinstance(s, int) and s > 0 for s in shape):
            raise ValueError(f"shape must contain positive integers, got {shape!r}")

        self.p = float(p)
        self.shape = shape
        self.count = 0

        ny, nx = shape

        # Marker heights
        self.q = np.zeros((5, ny, nx), dtype=np.float64)
        # Marker positions
        self.n = np.zeros((5, ny, nx), dtype=np.int64)
        # Desired marker positions
        self.n_desired = np.zeros((5, ny, nx), dtype=np.float64)

        # Buffer for first 5 samples
        self._init_buffer: list[np.ndarray] = []
        self._initialized = False

        # Precompute desired-position increments per update
        self._n_desired_increment = np.array(
            [0.0, self.p / 2.0, self.p, (1.0 + self.p) / 2.0, 1.0],
            dtype=np.float64,
        )[:, None, None]

    def update(self, data: np.ndarray) -> None:
        """Update estimator with a new observation."""
        if data.shape != self.shape:
            raise ValueError(f"Data shape {data.shape} doesn't match expected {self.shape}")

        data_f64 = np.asarray(data, dtype=np.float64)

        if not self._initialized:
            self._add_initial_sample(data_f64)
        else:
            self._update_with_sample(data_f64)

        self.count += 1

    def _add_initial_sample(self, data: np.ndarray) -> None:
        self._init_buffer.append(data)
        if len(self._init_buffer) == 5:
            self._initialize_markers()
            self._initialized = True
            # Free memory
            self._init_buffer.clear()

    def _initialize_markers(self) -> None:
        init_stack = np.stack(self._init_buffer, axis=0)
        init_sorted = np.sort(init_stack, axis=0)

        for i in range(5):
            self.q[i, :, :] = init_sorted[i, :, :]
            self.n[i, :, :] = i + 1

        n_desired_init = np.array(
            [1.0, 1.0 + 2.0 * self.p, 1.0 + 4.0 * self.p, 3.0 + 2.0 * self.p, 5.0],
            dtype=np.float64,
        )
        for i in range(5):
            self.n_desired[i, :, :] = n_desired_init[i]

    def _update_with_sample(self, data: np.ndarray) -> None:
        x = data

        below = x < self.q[0]
        above = x >= self.q[4]
        self.q[0] = np.where(below, x, self.q[0])
        self.q[4] = np.where(above, x, self.q[4])

        # Find interval k where q[k] <= x < q[k+1]
        cmp = x[None, :, :] >= self.q
        k = cmp.sum(axis=0).astype(np.int32) - 1
        k = np.clip(k, 0, 3)

        # Increment positions for markers above k
        self.n += (np.arange(5)[:, None, None] > k[None, :, :]).astype(np.int64)

        # Desired positions increment
        self.n_desired += self._n_desired_increment

        # Adjust heights for markers 1..3
        for i in (1, 2, 3):
            d = self.n_desired[i] - self.n[i].astype(np.float64)

            needs_up = d >= 1.0
            needs_down = d <= -1.0
            needs_adjustment = needs_up | needs_down
            if not needs_adjustment.any():
                continue

            s = np.where(needs_up, 1.0, np.where(needs_down, -1.0, 0.0))

            q_im1 = self.q[i - 1]
            q_i = self.q[i]
            q_ip1 = self.q[i + 1]

            n_im1 = self.n[i - 1].astype(np.float64)
            n_i = self.n[i].astype(np.float64)
            n_ip1 = self.n[i + 1].astype(np.float64)

            with np.errstate(divide="ignore", invalid="ignore"):
                denom1 = n_ip1 - n_i
                denom2 = n_i - n_im1
                denom3 = n_ip1 - n_im1

                eps = 1e-10
                denom1 = np.where(np.abs(denom1) < eps, eps, denom1)
                denom2 = np.where(np.abs(denom2) < eps, eps, denom2)
                denom3 = np.where(np.abs(denom3) < eps, eps, denom3)

                term1 = (n_i - n_im1 + s) * (q_ip1 - q_i) / denom1
                term2 = (n_ip1 - n_i - s) * (q_i - q_im1) / denom2
                q_parabolic = q_i + (s / denom3) * (term1 + term2)

                q_linear = q_i + s * (q_ip1 - q_im1) / denom3

            use_parabolic = (q_parabolic > q_im1) & (q_parabolic < q_ip1) & needs_adjustment
            q_new = np.where(use_parabolic, q_parabolic, q_linear)

            self.q[i] = np.where(needs_adjustment, q_new, q_i)
            self.n[i] = np.where(needs_adjustment, self.n[i] + s.astype(np.int64), self.n[i])

    def get_quantile(self) -> np.ndarray:
        """Return the current quantile estimate."""
        if not self._initialized:
            if len(self._init_buffer) == 0:
                raise RuntimeError("No samples added yet.")
            stacked = np.stack(self._init_buffer, axis=0)
            return np.median(stacked, axis=0)

        return self.q[2].copy()

    def __repr__(self) -> str:
        return (
            f"P2QuantileEstimator(shape={self.shape}, p={self.p}, "
            f"count={self.count}, initialized={self._initialized})"
        )


# ============================================================================
# Streaming Statistics (Welford)
# ============================================================================


def streaming_statistics(
    file_list: list[StrOrPath],
    hdu: HDUSelector = 0,
    memmap: bool = True,
    compute_quantiles: bool = False,
    quantiles: list[float] | None = None,
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    """
    Compute per-pixel statistics in a single pass (Welford), optionally quantiles (P²).

    Parameters
    ----------
    file_list
        List of FITS file paths.
    hdu
        HDU selector resolved independently in every file.
    memmap
        Whether to use FITS memmap.
    compute_quantiles
        If True, compute quantiles using P² estimators.
    quantiles
        Quantiles to compute (0 < q < 1). If None and compute_quantiles=True,
        computes median only.
    show_progress
        If True, show a progress bar (requires tqdm).

    Returns
    -------
    dict
        Keys include: ``count``, ``mean``, ``var``, ``std``, ``min``, ``max``, ``sum``,
        and optionally ``quantiles`` (a dict mapping quantile -> array).

    Raises
    ------
    ValueError
        If file_list is empty or frames have inconsistent shapes.
    FileHandlingError
        If files cannot be read.
    ImageProcessingError
        For processing failures during iteration.
    """
    if not file_list:
        raise ValueError("Empty file_list.")

    est_mem = _estimate_memory_usage(file_list, hdu=hdu)
    logger.debug("Processing %d files (traditional in-memory estimate: %.2f GB)", len(file_list), est_mem)

    first_path = Path(file_list[0])
    try:
        with fits.open(first_path, memmap=memmap) as hdul:
            _, reference_identity = _resolve_streaming_hdu(
                hdul, first_path, hdu
            )
            shape = reference_identity[3]
            if shape is None:
                raise ValueError(
                    f"Selected HDU {hdu!r} in {first_path} has no image shape"
                )
    except InvalidHDUIndexError:
        raise
    except Exception as e:
        raise FileHandlingError(f"Could not read {first_path}: {e}", filepath=first_path) from e

    count = 0
    mean: np.ndarray | None = None
    M2: np.ndarray | None = None
    sum_arr = np.zeros(shape, dtype=np.float64)
    min_arr = np.full(shape, np.inf, dtype=np.float64)
    max_arr = np.full(shape, -np.inf, dtype=np.float64)

    quantile_estimators: dict[float, P2QuantileEstimator] = {}
    if compute_quantiles:
        qs = [0.5] if quantiles is None else list(quantiles)
        for q in qs:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile must be in (0, 1), got {q}")
            quantile_estimators[q] = P2QuantileEstimator(shape, p=q)

    iterator = maybe_tqdm(file_list, desc="Computing statistics", unit="file", disable=not show_progress)
    for i, filepath in enumerate(iterator, start=1):
        path = Path(filepath)
        try:
            with fits.open(path, memmap=memmap) as hdul:
                index, _ = _resolve_streaming_hdu(
                    hdul,
                    path,
                    hdu,
                    expected_identity=reference_identity,
                )
                data = hdul[index].data
                if data is None:
                    raise ValueError(
                        f"No data in selected HDU {hdu!r} of {path}"
                    )
                if data.shape != shape:
                    raise ValueError(f"Shape mismatch: {path} has shape {data.shape}, expected {shape}")

                arr = np.asarray(data, dtype=np.float64)

                sum_arr += arr
                np.minimum(min_arr, arr, out=min_arr)
                np.maximum(max_arr, arr, out=max_arr)

                count += 1
                if mean is None:
                    mean = arr.copy()
                    M2 = np.zeros_like(arr, dtype=np.float64)
                else:
                    delta = arr - mean
                    mean += delta / count
                    delta2 = arr - mean
                    M2 += delta * delta2  # type: ignore[operator]

                for estimator in quantile_estimators.values():
                    estimator.update(arr)

        except Exception as e:
            msg = f"Failed processing file {i}/{len(file_list)} ({path}): {e}"
            logger.error(msg)
            raise ImageProcessingError(msg, operation="streaming_statistics") from e

    if count == 0:
        raise ValueError("No valid files processed.")

    assert mean is not None and M2 is not None  # for type checkers
    var = M2 / (count - 1) if count > 1 else M2
    std = np.sqrt(var)

    results: dict[str, Any] = {
        "count": count,
        "mean": mean,
        "var": var,
        "std": std,
        "min": min_arr,
        "max": max_arr,
        "sum": sum_arr,
    }

    if compute_quantiles:
        results["quantiles"] = {q: est.get_quantile() for q, est in quantile_estimators.items()}

    return results


# ============================================================================
# Streaming Combine
# ============================================================================


def streaming_combine(
    file_list: list[StrOrPath],
    method: str = "median",
    output_file: StrOrPath | None = None,
    hdu: HDUSelector = 0,
    memmap: bool = True,
    overwrite: bool = False,
    *,
    show_progress: bool = False,
) -> np.ndarray | str:
    """Combine images using streaming algorithms (memory-efficient)."""
    valid_methods = {"median", "mean", "min", "max"}
    method = method.lower().strip()
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Valid options: {sorted(valid_methods)}")
    if not file_list:
        raise ValueError("Empty file_list.")

    if method == "median":
        stats = streaming_statistics(
            file_list,
            hdu=hdu,
            memmap=memmap,
            compute_quantiles=True,
            quantiles=[0.5],
            show_progress=show_progress,
        )
        combined = stats["quantiles"][0.5]
    else:
        stats = streaming_statistics(
            file_list,
            hdu=hdu,
            memmap=memmap,
            compute_quantiles=False,
            show_progress=show_progress,
        )
        combined = stats[method]

    if output_file is None:
        return combined

    first_path = Path(file_list[0])
    with fits.open(first_path, memmap=memmap) as hdul:
        index, _ = _resolve_streaming_hdu(hdul, first_path, hdu)
        header = hdul[index].header.copy()
    n_processed = int(stats["count"])
    add_history(header, f"Streaming combine: {n_processed} images using {method}")
    header["NINPUT"] = (len(file_list), "Number of input files")
    header["NCOMBINE"] = (n_processed, "Number of images actually combined")
    header["COMBMETH"] = (f"streaming_{method}", "Combining method")
    header["STREAMIN"] = (True, "Used streaming algorithm")

    write_simple_fits(output_file, combined, header=header, overwrite=overwrite)
    return str(output_file)


# ============================================================================
# Stack Reference Computation
# ============================================================================


def compute_stack_reference(
    frame_paths: list[StrOrPath],
    hdu: HDUSelector = 0,
    existing_bpm: np.ndarray | None = None,
    memmap: bool = True,
    method: str = "auto",
    chunk_size: int = 10,
    *,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel reference median and robust sigma from a frame stack."""
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")

    n_frames = len(frame_paths)
    method = method.lower().strip()
    if method not in {"auto", "streaming", "chunked"}:
        raise ValueError(f"method must be 'auto', 'streaming', or 'chunked'; got {method!r}")

    if method == "auto":
        if existing_bpm is not None:
            method = "chunked"
        elif n_frames > 50:
            method = "streaming"
        else:
            method = "chunked"

    if method == "streaming":
        return _compute_reference_streaming(frame_paths, hdu=hdu, memmap=memmap, show_progress=show_progress)

    return _compute_reference_chunked(
        frame_paths,
        hdu=hdu,
        existing_bpm=existing_bpm,
        chunk_size=chunk_size,
        memmap=memmap,
        show_progress=show_progress,
    )


def _compute_reference_streaming(
    frame_paths: list[StrOrPath],
    hdu: HDUSelector = 0,
    memmap: bool = True,
    *,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference using streaming P² algorithm (constant memory)."""
    reference_identity = _get_streaming_file_identity(
        frame_paths[0], hdu, memmap=memmap
    )
    first_frame = _load_frame_for_streaming(
        frame_paths[0],
        hdu=hdu,
        memmap=memmap,
        expected_identity=reference_identity,
    )
    shape = first_frame.shape

    median_estimator = P2QuantileEstimator(shape, p=0.5)
    iterator = maybe_tqdm(frame_paths, desc="Pass 1: Median", unit="frame", disable=not show_progress)
    for path in iterator:
        median_estimator.update(
            _load_frame_for_streaming(
                path,
                hdu=hdu,
                memmap=memmap,
                expected_identity=reference_identity,
            )
        )
    reference_median = median_estimator.get_quantile()

    mad_estimator = P2QuantileEstimator(shape, p=0.5)
    iterator = maybe_tqdm(frame_paths, desc="Pass 2: MAD", unit="frame", disable=not show_progress)
    for path in iterator:
        data = _load_frame_for_streaming(
            path,
            hdu=hdu,
            memmap=memmap,
            expected_identity=reference_identity,
        )
        mad_estimator.update(np.abs(data - reference_median))

    mad = mad_estimator.get_quantile()
    reference_sigma = MAD_TO_STD_FACTOR * mad
    return reference_median, reference_sigma


def _compute_reference_chunked(
    frame_paths: list[StrOrPath],
    hdu: HDUSelector = 0,
    existing_bpm: np.ndarray | None = None,
    chunk_size: int = 10,
    memmap: bool = True,
    *,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference using chunked loading (exact statistics, supports masking)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    reference_identity = _get_streaming_file_identity(
        frame_paths[0], hdu, memmap=memmap
    )
    first_frame = _load_frame_for_streaming(
        frame_paths[0],
        hdu=hdu,
        memmap=memmap,
        expected_identity=reference_identity,
    )
    shape = first_frame.shape

    bad_pixel_mask: np.ndarray | None = None
    if existing_bpm is not None:
        if existing_bpm.shape != shape:
            raise ValueError(f"existing_bpm shape {existing_bpm.shape} does not match frame shape {shape}")
        bad_pixel_mask = existing_bpm > 0

    all_frames_list: list[np.ndarray] = []

    idx_iter = range(0, len(frame_paths), chunk_size)
    idx_iter = maybe_tqdm(
        idx_iter,
        desc="Loading chunks",
        unit="chunk",
        total=(len(frame_paths) + chunk_size - 1) // chunk_size,
        disable=not show_progress,
    )

    for i in idx_iter:
        chunk_paths = frame_paths[i : i + chunk_size]
        chunk_frames = [
            _load_frame_for_streaming(
                path,
                hdu=hdu,
                memmap=memmap,
                expected_identity=reference_identity,
            )
            for path in chunk_paths
        ]
        all_frames_list.append(np.array(chunk_frames, dtype=np.float64))

    all_frames = np.concatenate(all_frames_list, axis=0)
    del all_frames_list

    if bad_pixel_mask is not None:
        mask_3d = np.broadcast_to(bad_pixel_mask, all_frames.shape)
        masked_frames = np.ma.masked_array(all_frames, mask=mask_3d)
        reference_median = np.ma.median(masked_frames, axis=0).filled(0)
    else:
        reference_median = np.median(all_frames, axis=0)

    residuals = np.abs(all_frames - reference_median[np.newaxis, :, :])

    if bad_pixel_mask is not None:
        masked_residuals = np.ma.masked_array(residuals, mask=mask_3d)
        mad = np.ma.median(masked_residuals, axis=0).filled(0)
    else:
        mad = np.median(residuals, axis=0)

    reference_sigma = MAD_TO_STD_FACTOR * mad
    return reference_median, reference_sigma


def _load_frame_for_streaming(
    path: StrOrPath,
    hdu: HDUSelector = 0,
    memmap: bool = True,
    *,
    expected_identity: tuple[
        str | None, int | None, str, tuple[int, ...] | None
    ]
    | None = None,
) -> np.ndarray:
    """Load a single frame as float64 for streaming processing."""
    p = Path(path)
    try:
        if _is_fits_path(p):
            with fits.open(p, memmap=memmap) as hdul:
                index, _ = _resolve_streaming_hdu(
                    hdul,
                    p,
                    hdu,
                    expected_identity=expected_identity,
                )
                data = hdul[index].data
                if data is None:
                    raise ValueError(f"No data in HDU {hdu} for {p}")
                return np.asarray(data, dtype=np.float64)

        if p.suffix.lower() == ".npy":
            return np.asarray(np.load(p), dtype=np.float64)

        if p.suffix.lower() == ".npz":
            with np.load(p) as npz:
                key = sorted(npz.files)[0]
                return np.asarray(npz[key], dtype=np.float64)

        raise ValueError(f"Unsupported file format: {p}")

    except Exception as e:
        raise FileHandlingError(f"Failed to load frame from {p}: {e}", filepath=p) from e


__all__ = [
    "P2QuantileEstimator",
    "streaming_statistics",
    "streaming_combine",
    "compute_stack_reference",
]
