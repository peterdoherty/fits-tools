#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Memory-efficient streaming algorithms for large FITS datasets.

This module computes per-pixel statistics of image stacks too large to hold
in memory. Memory use is bounded by a budget, not by the number of frames,
and each dataset is read once.

Features
--------
- Exact medians, quantiles, and MAD: the stack is processed in bands of rows,
  reading only those rows from each file (``max_memory_mb`` sets the budget)
- Single-pass mean, variance (Welford), min, and max when no quantiles are
  needed
- Optional approximate single-pass quantiles (P² estimator,
  ``median_algorithm="p2"`` / ``quantile_algorithm="p2"`` / ``method="p2"``)
- Works with unsigned/scaled (BZERO) camera frames and MEF HDU selection
- Optional progress reporting for long operations

Notes
-----
This module is FITS-aware primarily through file reading and header handling.
Array-level algorithms are general and operate on NumPy arrays.

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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
        # Header-only inspection: never touches pixel data, so it works for
        # scaled/unsigned images (BZERO/BSCALE) that cannot be memory-mapped.
        with fits.open(first, memmap=False, lazy_load_hdus=True) as hdul:
            index, identity = _resolve_streaming_hdu(hdul, first, hdu)
            shape = identity[3]
            bitpix = hdul[index].header.get("BITPIX")
            if not shape or bitpix is None:
                return 0.0
    except Exception as e:  # pragma: no cover
        logger.warning("Could not estimate memory usage from %s: %s", first, e)
        return 0.0

    bytes_per_pixel = abs(int(bitpix)) // 8
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
        # Jain & Chlamtac (1985), vectorized per pixel.
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

        # Adjust heights for markers 1..3. A marker moves only when it is at
        # least one position away from its desired position AND its
        # neighbour in that direction is more than one position away, so
        # marker positions stay strictly increasing.
        for i in (1, 2, 3):
            n_im1 = self.n[i - 1].astype(np.float64)
            n_i = self.n[i].astype(np.float64)
            n_ip1 = self.n[i + 1].astype(np.float64)
            d = self.n_desired[i] - n_i

            needs_up = (d >= 1.0) & (n_ip1 - n_i > 1.0)
            needs_down = (d <= -1.0) & (n_im1 - n_i < -1.0)
            needs_adjustment = needs_up | needs_down
            if not needs_adjustment.any():
                continue

            s = np.where(needs_up, 1.0, -1.0)

            q_im1 = self.q[i - 1]
            q_i = self.q[i]
            q_ip1 = self.q[i + 1]

            # Neighbour spacings are >= 1 wherever an adjustment happens;
            # elsewhere the results are discarded, so suppress warnings.
            with np.errstate(divide="ignore", invalid="ignore"):
                term1 = (n_i - n_im1 + s) * (q_ip1 - q_i) / (n_ip1 - n_i)
                term2 = (n_ip1 - n_i - s) * (q_i - q_im1) / (n_i - n_im1)
                q_parabolic = q_i + (s / (n_ip1 - n_im1)) * (term1 + term2)

                # Linear fallback toward the neighbour in direction s.
                q_nb = np.where(needs_up, q_ip1, q_im1)
                n_nb = np.where(needs_up, n_ip1, n_im1)
                q_linear = q_i + s * (q_nb - q_i) / (n_nb - n_i)

            use_parabolic = (q_parabolic > q_im1) & (q_parabolic < q_ip1)
            q_new = np.where(use_parabolic, q_parabolic, q_linear)

            self.q[i] = np.where(needs_adjustment, q_new, q_i)
            self.n[i] = np.where(
                needs_adjustment, self.n[i] + s.astype(np.int64), self.n[i]
            )

    def get_quantile(self) -> np.ndarray:
        """Return the current quantile estimate."""
        if not self._initialized:
            if len(self._init_buffer) == 0:
                raise RuntimeError("No samples added yet.")
            # Fewer than five samples: return the exact quantile.
            stacked = np.stack(self._init_buffer, axis=0)
            return np.asarray(np.quantile(stacked, self.p, axis=0))

        return np.asarray(self.q[2].copy())

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
    memmap: bool = False,
    compute_quantiles: bool = False,
    quantiles: list[float] | None = None,
    *,
    show_progress: bool = False,
    transform: Callable[[np.ndarray, fits.Header], np.ndarray | None] | None = None,
    quantile_algorithm: str = "exact",
    max_memory_mb: float = 1024.0,
) -> dict[str, Any]:
    """
    Compute per-pixel statistics of a stack without holding it in memory.

    Without quantiles, mean/variance/min/max/sum are accumulated in a single
    pass (Welford). With ``compute_quantiles=True`` and the default
    ``quantile_algorithm="exact"``, everything is computed exactly in bands of
    rows (see *max_memory_mb*), still reading the dataset once.

    Parameters
    ----------
    file_list
        List of FITS file paths.
    hdu
        HDU selector resolved independently in every file.
    memmap
        Whether to use FITS memmap.
    compute_quantiles
        If True, also compute per-pixel quantiles.
    quantiles
        Quantiles to compute (0 < q < 1). If None and compute_quantiles=True,
        computes median only.
    show_progress
        If True, show a progress bar (requires tqdm).
    transform
        Optional per-frame callback. It receives the image and header and
        returns the array to accumulate. Returning ``None`` skips the frame.
        With exact quantiles it runs once per frame per band, so it must be
        deterministic.
    quantile_algorithm
        ``"exact"`` (default): exact quantiles (NumPy's linear interpolation)
        computed band by band with memory bounded by *max_memory_mb*.
        ``"p2"``: the single-pass P² estimator, which is approximate.
    max_memory_mb
        Approximate memory budget for the exact quantiles' row buffer. The
        results (one image-sized float64 array per statistic) are in addition.

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
    if quantile_algorithm not in _MEDIAN_ALGORITHMS:
        raise ValueError(
            f"quantile_algorithm must be one of {_MEDIAN_ALGORITHMS}, got {quantile_algorithm!r}"
        )
    if max_memory_mb <= 0:
        raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")
    if compute_quantiles and quantile_algorithm == "exact":
        qs = [0.5] if quantiles is None else [float(q) for q in quantiles]
        for q in qs:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile must be in (0, 1), got {q}")
        return _exact_statistics_banded(
            file_list, hdu, memmap, qs, max_memory_mb,
            transform=transform, show_progress=show_progress,
        )

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
        if len(shape) != 2:
            raise ValueError(
                f"Quantile estimation requires 2-D images; got shape {shape}"
            )
        shape_2d = (int(shape[0]), int(shape[1]))
        qs = [0.5] if quantiles is None else list(quantiles)
        for q in qs:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile must be in (0, 1), got {q}")
            quantile_estimators[q] = P2QuantileEstimator(shape_2d, p=q)

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
                if transform is not None:
                    transformed = transform(arr, hdul[index].header)
                    if transformed is None:
                        continue
                    arr = np.asarray(transformed, dtype=np.float64)
                    if arr.shape != shape:
                        raise ValueError(
                            f"Transform returned shape {arr.shape}, expected {shape}"
                        )

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


_FrameTransform = Callable[[np.ndarray, fits.Header], np.ndarray | None]

# Reads rows [r0, r1) of the selected HDU of *path*; returns None to skip the
# frame. Must make the same skip decision for every band.
_BandReader = Callable[[Any, Path, int, int], np.ndarray | None]

_MEDIAN_ALGORITHMS = ("exact", "p2")


def _reference_identity(
    path: Path, hdu: HDUSelector, memmap: bool
) -> tuple[str | None, int | None, str, tuple[int, ...] | None]:
    """Resolve the selected HDU of the first file (header only)."""
    try:
        with fits.open(path, memmap=memmap) as hdul:
            _, identity = _resolve_streaming_hdu(hdul, path, hdu)
    except InvalidHDUIndexError:
        raise
    except Exception as e:
        raise FileHandlingError(f"Could not read {path}: {e}", filepath=path) from e
    if not identity[3]:
        raise ValueError(f"Selected HDU {hdu!r} in {path} has no image shape")
    return identity


def _read_rows(hdu_obj: Any, r0: int, r1: int) -> np.ndarray:
    """Read rows [r0, r1) as float64 without loading the whole image.

    ``HDU.section`` reads only the requested bytes and applies BZERO/BSCALE,
    so it also works for unsigned/scaled images that cannot be memory-mapped.
    """
    section = getattr(hdu_obj, "section", None)
    if section is not None:
        return np.asarray(section[r0:r1], dtype=np.float64)
    return np.asarray(hdu_obj.data[r0:r1], dtype=np.float64)  # pragma: no cover


# Reads rows [r0, r1) of one input (by position and path); None skips it.
_RowReader = Callable[[int, Path, int, int], np.ndarray | None]
# Reduces a (n_frames, n_rows, ...) float64 band in place to output bands.
_BandReduce = Callable[[np.ndarray], tuple[np.ndarray, ...]]


def _banded_reduce(
    paths: list[Path],
    shape: tuple[int, ...],
    read_rows: _RowReader,
    reduce: _BandReduce,
    n_outputs: int,
    max_memory_mb: float,
    *,
    operation: str,
    show_progress: bool = False,
) -> tuple[list[np.ndarray], int]:
    """Apply a per-pixel reduction across a stack, one band of rows at a time.

    For each band the same rows are read from every input into a float64
    buffer of at most ~*max_memory_mb*, and *reduce* turns it into one band
    of each output. Memory is bounded by the budget, not the stack size;
    total I/O is about one read of the dataset when *read_rows* reads only
    the requested rows.
    """
    n_rows = shape[0]
    row_shape = shape[1:]
    bytes_per_row = 8 * len(paths) * int(np.prod(row_shape, dtype=np.int64))
    band_rows = int(max_memory_mb * 1024**2 // max(bytes_per_row, 1))
    band_rows = max(1, min(n_rows, band_rows))
    logger.debug(
        "%s: %d inputs, shape %s, %d rows per band",
        operation, len(paths), shape, band_rows,
    )

    outputs = [np.empty(shape, dtype=np.float64) for _ in range(n_outputs)]
    buffer = np.empty((len(paths), band_rows, *row_shape), dtype=np.float64)
    n_used: int | None = None

    bands = range(0, n_rows, band_rows)
    iterator = maybe_tqdm(
        bands, desc=operation, unit="band", total=len(bands), disable=not show_progress,
    )
    for r0 in iterator:
        r1 = min(r0 + band_rows, n_rows)
        expected = (r1 - r0, *row_shape)
        k = 0
        for i, path in enumerate(paths):
            try:
                band = read_rows(i, path, r0, r1)
                if band is not None and band.shape != expected:
                    raise ValueError(f"Band has shape {band.shape}, expected {expected}")
            except Exception as e:
                msg = f"Failed processing file {i + 1}/{len(paths)} ({path}): {e}"
                logger.error(msg)
                raise ImageProcessingError(msg, operation=operation) from e
            if band is None:
                continue
            buffer[k, : r1 - r0] = band
            k += 1

        if n_used is None:
            n_used = k
        elif k != n_used:
            raise ImageProcessingError(
                "Frame selection changed between row bands; the per-frame "
                "transform must be deterministic.",
                operation=operation,
            )
        if k == 0:
            raise ValueError("No valid files processed.")
        for output, band_result in zip(outputs, reduce(buffer[:k, : r1 - r0]), strict=True):
            output[r0:r1] = band_result

    assert n_used is not None
    return outputs, n_used


def _band_median(stack: np.ndarray) -> tuple[np.ndarray]:
    return (np.median(stack, axis=0, overwrite_input=True),)


def _exact_median_banded(
    file_list: list[StrOrPath],
    hdu: HDUSelector,
    memmap: bool,
    max_memory_mb: float,
    *,
    read_band: _BandReader | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, int]:
    """Exact per-pixel median of FITS frames with bounded memory."""
    paths = [Path(p) for p in file_list]
    reference = _reference_identity(paths[0], hdu, memmap)
    shape = tuple(int(s) for s in reference[3] or ())

    def read_rows(_: int, path: Path, r0: int, r1: int) -> np.ndarray | None:
        with fits.open(path, memmap=memmap) as hdul:
            index, _id = _resolve_streaming_hdu(hdul, path, hdu, expected_identity=reference)
            hdu_obj = hdul[index]
            if read_band is None:
                return _read_rows(hdu_obj, r0, r1)
            return read_band(hdu_obj, path, r0, r1)

    (median,), n_used = _banded_reduce(
        paths, shape, read_rows, _band_median, 1, max_memory_mb,
        operation="streaming_median", show_progress=show_progress,
    )
    return median, n_used


def _exact_statistics_banded(
    file_list: list[StrOrPath],
    hdu: HDUSelector,
    memmap: bool,
    quantiles: list[float],
    max_memory_mb: float,
    *,
    transform: _FrameTransform | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """streaming_statistics with exact quantiles, computed band by band."""
    paths = [Path(p) for p in file_list]
    reference = _reference_identity(paths[0], hdu, memmap)
    shape = tuple(int(s) for s in reference[3] or ())

    def read_rows(_: int, path: Path, r0: int, r1: int) -> np.ndarray | None:
        with fits.open(path, memmap=memmap) as hdul:
            index, _id = _resolve_streaming_hdu(hdul, path, hdu, expected_identity=reference)
            hdu_obj = hdul[index]
            if transform is None:
                return _read_rows(hdu_obj, r0, r1)
            full = np.asarray(hdu_obj.data, dtype=np.float64)
            out = transform(full, hdu_obj.header)
            if out is None:
                return None
            out = np.asarray(out, dtype=np.float64)
            if out.shape != full.shape:
                raise ValueError(f"Transform returned shape {out.shape}, expected {full.shape}")
            return out[r0:r1]

    def reduce(stack: np.ndarray) -> tuple[np.ndarray, ...]:
        # No buffer-sized temporaries: reductions first, then quantiles
        # (which only reorder values within each pixel), then the variance
        # computed in place.
        n = stack.shape[0]
        total = stack.sum(axis=0)
        mean = total / n
        lo = stack.min(axis=0)
        hi = stack.max(axis=0)
        qvals = np.quantile(stack, quantiles, axis=0, overwrite_input=True)
        if n > 1:
            np.subtract(stack, mean, out=stack)
            np.square(stack, out=stack)
            var = stack.sum(axis=0) / (n - 1)
        else:
            var = np.zeros_like(mean)
        return (total, mean, var, lo, hi, *qvals)

    outputs, count = _banded_reduce(
        paths, shape, read_rows, reduce, 5 + len(quantiles), max_memory_mb,
        operation="streaming_statistics", show_progress=show_progress,
    )
    total, mean, var, lo, hi = outputs[:5]
    return {
        "count": count,
        "mean": mean,
        "var": var,
        "std": np.sqrt(var),
        "min": lo,
        "max": hi,
        "sum": total,
        "quantiles": dict(zip(quantiles, outputs[5:], strict=True)),
    }


def _streaming_combine_array(
    file_list: list[StrOrPath],
    method: str,
    hdu: HDUSelector,
    memmap: bool,
    show_progress: bool,
    transform: _FrameTransform | None,
    *,
    median_algorithm: str = "exact",
    max_memory_mb: float = 1024.0,
    read_band: _BandReader | None = None,
) -> tuple[np.ndarray, int]:
    """Return the combined array and the number of frames used.

    For the exact median, *read_band* (rows-at-a-time) takes precedence over
    *transform* (whole frame); a whole-frame *transform* is applied to the
    full frame once per band.
    """
    valid_methods = {"median", "mean", "min", "max"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Valid options: {sorted(valid_methods)}")
    if median_algorithm not in _MEDIAN_ALGORITHMS:
        raise ValueError(
            f"median_algorithm must be one of {_MEDIAN_ALGORITHMS}, got {median_algorithm!r}"
        )
    if max_memory_mb <= 0:
        raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")
    if not file_list:
        raise ValueError("Empty file_list.")

    if method == "median" and median_algorithm == "exact":
        if read_band is None and transform is not None:
            frame_transform = transform

            def read_band(hdu_obj: Any, path: Path, r0: int, r1: int) -> np.ndarray | None:
                full = np.asarray(hdu_obj.data, dtype=np.float64)
                out = frame_transform(full, hdu_obj.header)
                if out is None:
                    return None
                out = np.asarray(out, dtype=np.float64)
                if out.shape != full.shape:
                    raise ValueError(
                        f"Transform returned shape {out.shape}, expected {full.shape}"
                    )
                return out[r0:r1]

        return _exact_median_banded(
            file_list, hdu, memmap, max_memory_mb,
            read_band=read_band, show_progress=show_progress,
        )

    stats = streaming_statistics(
        file_list,
        hdu=hdu,
        memmap=memmap,
        compute_quantiles=method == "median",
        quantiles=[0.5] if method == "median" else None,
        show_progress=show_progress,
        transform=transform,
        quantile_algorithm="p2",  # only reached for median_algorithm="p2"
    )
    if method == "median":
        combined = np.asarray(stats["quantiles"][0.5])
    else:
        combined = np.asarray(stats[method])
    return combined, int(stats["count"])


def _write_streaming_output(
    file_list: list[StrOrPath],
    combined: np.ndarray,
    n_processed: int,
    method: str,
    output_file: StrOrPath,
    hdu: HDUSelector,
    memmap: bool,
    overwrite: bool,
    extra_cards: dict[str, tuple[Any, str]] | None = None,
    history: list[str] | None = None,
) -> str:
    first_path = Path(file_list[0])
    with fits.open(first_path, memmap=memmap) as hdul:
        index, _ = _resolve_streaming_hdu(hdul, first_path, hdu)
        header = hdul[index].header.copy()
    add_history(header, f"Streaming combine: {n_processed} images using {method}")
    for line in history or []:
        add_history(header, line)
    header["NINPUT"] = (len(file_list), "Number of input files")
    header["NCOMBINE"] = (n_processed, "Number of images actually combined")
    header["COMBMETH"] = (f"streaming_{method}", "Combining method")
    header["STREAMIN"] = (True, "Used streaming algorithm")
    for key, card in (extra_cards or {}).items():
        header[key] = card

    write_simple_fits(output_file, combined, header=header, overwrite=overwrite)
    return str(output_file)


def _median_cards(method: str, median_algorithm: str) -> dict[str, tuple[Any, str]]:
    if method != "median":
        return {}
    return {"MEDALG": (median_algorithm, "Median algorithm (exact or p2)")}


def streaming_combine(
    file_list: list[StrOrPath],
    method: str = "median",
    output_file: StrOrPath | None = None,
    hdu: HDUSelector = 0,
    memmap: bool = False,
    overwrite: bool = False,
    *,
    show_progress: bool = False,
    transform: _FrameTransform | None = None,
    median_algorithm: str = "exact",
    max_memory_mb: float = 1024.0,
) -> np.ndarray | str:
    """Combine images without holding the whole stack in memory.

    Parameters
    ----------
    file_list
        FITS files to combine.
    method
        ``"median"``, ``"mean"``, ``"min"``, or ``"max"``.
    output_file
        If given, write the result here and return the path as a string.
    hdu
        HDU selector resolved independently in every file.
    memmap
        Whether to memory-map FITS files.
    overwrite
        Overwrite *output_file* if it exists.
    show_progress
        Show a progress bar (requires tqdm).
    transform
        Optional per-frame callback ``(data, header) -> array | None``;
        returning ``None`` skips the frame. With the exact median the
        callback runs once per frame per row band, so it must be
        deterministic, and it is cheaper to raise *max_memory_mb*.
    median_algorithm
        ``"exact"`` (default) computes the exact per-pixel median in bands
        of rows, reading only the needed rows of each file; total I/O is
        about one read of the dataset. ``"p2"`` uses the single-pass P²
        estimator, which is approximate (its error is comparable to the
        median's own sampling noise for stacks of tens of frames).
    max_memory_mb
        Approximate memory budget for the exact median's row buffer.
        Smaller budgets mean more, thinner bands.

    Returns
    -------
    numpy.ndarray or str
        The combined image, or *output_file* as a string.
    """
    method = method.lower().strip()
    combined, n_processed = _streaming_combine_array(
        file_list, method, hdu, memmap, show_progress, transform,
        median_algorithm=median_algorithm, max_memory_mb=max_memory_mb,
    )
    if output_file is None:
        return combined
    return _write_streaming_output(
        file_list, combined, n_processed, method, output_file, hdu, memmap,
        overwrite, extra_cards=_median_cards(method, median_algorithm),
    )


def streaming_calibrated_combine(
    file_list: list[StrOrPath],
    *,
    master_bias_data: np.ndarray | None = None,
    master_dark_data: np.ndarray | None = None,
    master_dark_exptime: float | None = None,
    bad_pixel_mask: np.ndarray | None = None,
    saturation_threshold_adu: float | None = None,
    max_saturated_pixels: int = 0,
    method: str = "median",
    output_file: StrOrPath | None = None,
    hdu: HDUSelector = 0,
    # Astropy cannot memmap the unsigned/scaled FITS files commonly produced
    # by cameras such as the Tucsen. False still keeps memory bounded because
    # only one image (or one band of rows per file) is loaded at a time.
    memmap: bool = False,
    overwrite: bool = False,
    show_progress: bool = False,
    median_algorithm: str = "exact",
    max_memory_mb: float = 1024.0,
) -> np.ndarray | str:
    """Combine calibrated FITS images without materializing a frame stack.

    If *saturation_threshold_adu* is set, a first pass reads each raw frame
    and rejects those with too many saturated pixels. The accepted frames
    are then bias- and dark-subtracted and combined. With the default exact
    median, calibration is applied band by band, so only the needed rows of
    each file are read.

    Parameters
    ----------
    file_list
        Raw FITS frames.
    master_bias_data
        Master bias subtracted from every frame.
    master_dark_data
        Bias-subtracted master dark, scaled by ``EXPTIME / master_dark_exptime``.
        Every frame must then carry an ``EXPTIME`` keyword.
    master_dark_exptime
        Exposure time of *master_dark_data*; required (and positive) when a
        dark is supplied.
    bad_pixel_mask
        Boolean or integer mask (non-zero = bad). Masked pixels are set to
        ``NaN`` in the combined result.
    saturation_threshold_adu
        Raw ADU at or above which a pixel counts as saturated.
    max_saturated_pixels
        A frame is rejected when more than this many pixels are saturated.
        Default 0 rejects any frame containing a saturated pixel. Rejected
        frames are logged by filename and counted in the ``NREJSAT`` header
        card of *output_file*.
    method, output_file, hdu, memmap, overwrite, show_progress, median_algorithm, max_memory_mb
        As for :func:`streaming_combine`.

    Raises
    ------
    ValueError
        For invalid arguments, or if every frame is rejected for saturation.
    ImageProcessingError
        If a master or mask does not match the frame shape, or a frame cannot
        be calibrated (e.g. missing ``EXPTIME`` when a dark is supplied).
    """
    if not file_list:
        raise ValueError("Empty file_list.")
    if max_saturated_pixels < 0:
        raise ValueError(f"max_saturated_pixels must be >= 0, got {max_saturated_pixels}")
    dark_exptime: float | None = None
    if master_dark_data is not None:
        if master_dark_exptime is None or master_dark_exptime <= 0:
            raise ValueError(
                "master_dark_exptime must be positive when master_dark_data is supplied"
            )
        dark_exptime = float(master_dark_exptime)
    method = method.lower().strip()

    bias = None if master_bias_data is None else np.asarray(master_bias_data, dtype=np.float32)
    dark = None if master_dark_data is None else np.asarray(master_dark_data, dtype=np.float32)
    bpm = None if bad_pixel_mask is None else np.asarray(bad_pixel_mask) != 0

    paths = [Path(p) for p in file_list]
    reference = _reference_identity(paths[0], hdu, memmap)
    frame_shape = reference[3]
    for name, arr in (("Master bias", bias), ("Master dark", dark), ("Bad-pixel mask", bpm)):
        if arr is not None and arr.shape != frame_shape:
            raise ImageProcessingError(
                f"{name} shape {arr.shape} does not match frame {frame_shape}",
                operation="streaming_calibrated_combine",
            )

    # Pass 1 (only with a threshold): decide which frames are accepted.
    accepted: list[StrOrPath] = list(file_list)
    rejected: list[str] = []
    if saturation_threshold_adu is not None:
        accepted = []
        for i, path in enumerate(paths, start=1):
            try:
                with fits.open(path, memmap=memmap) as hdul:
                    index, _ = _resolve_streaming_hdu(
                        hdul, path, hdu, expected_identity=reference
                    )
                    n_sat = int(
                        np.count_nonzero(hdul[index].data >= saturation_threshold_adu)
                    )
            except Exception as e:
                msg = f"Failed processing file {i}/{len(paths)} ({path}): {e}"
                logger.error(msg)
                raise ImageProcessingError(
                    msg, operation="streaming_calibrated_combine"
                ) from e
            if n_sat > max_saturated_pixels:
                logger.warning(
                    "Rejecting %s: %d pixels >= %.1f ADU (limit %d)",
                    path, n_sat, saturation_threshold_adu, max_saturated_pixels,
                )
                rejected.append(str(path))
            else:
                accepted.append(file_list[i - 1])
        if not accepted:
            raise ValueError(
                f"All {len(file_list)} frames were rejected for saturation"
            )
        if rejected:
            logger.warning(
                "Rejected %d of %d frames for saturation", len(rejected), len(file_list)
            )

    def calibrate_rows(
        data: np.ndarray, header: fits.Header, path: Path | str, rows: slice
    ) -> np.ndarray:
        result = np.asarray(data, dtype=np.float32).copy()
        if bias is not None:
            result -= bias[rows]
        if dark is not None and dark_exptime is not None:
            exptime = header.get("EXPTIME")
            if exptime is None:
                raise ValueError(
                    f"{path} has no EXPTIME keyword; cannot scale the master dark"
                )
            result -= dark[rows] * (float(exptime) / dark_exptime)
        return result

    def calibrate_frame(data: np.ndarray, header: fits.Header) -> np.ndarray:
        return calibrate_rows(data, header, "frame", slice(None))

    def calibrate_band(hdu_obj: Any, path: Path, r0: int, r1: int) -> np.ndarray:
        rows = _read_rows(hdu_obj, r0, r1)
        return calibrate_rows(rows, hdu_obj.header, path, slice(r0, r1))

    combined, n_processed = _streaming_combine_array(
        accepted, method, hdu, memmap, show_progress, calibrate_frame,
        median_algorithm=median_algorithm, max_memory_mb=max_memory_mb,
        read_band=calibrate_band,
    )

    if bpm is not None:
        combined = combined.astype(np.float64, copy=True)
        combined[bpm] = np.nan

    if output_file is None:
        return combined

    history = []
    if bias is not None:
        history.append("Bias corrected per frame")
    if dark is not None:
        history.append("Dark corrected per frame (EXPTIME scaled)")
    if bpm is not None:
        history.append("Bad-pixel mask applied (masked pixels set to NaN)")
    extra = _median_cards(method, median_algorithm)
    if saturation_threshold_adu is not None:
        extra["SATLEVEL"] = (float(saturation_threshold_adu), "Raw ADU saturation threshold")
        extra["NREJSAT"] = (len(rejected), "Frames rejected for saturation")
    return _write_streaming_output(
        file_list, combined, n_processed, method, output_file, hdu, memmap,
        overwrite, extra_cards=extra, history=history,
    )


# ============================================================================
# Stack Reference Computation
# ============================================================================


_REFERENCE_METHODS = {"exact": "exact", "p2": "p2",
                      # Legacy names from when "chunked" (exact, in memory) and
                      # "streaming" (P², two passes) were separate code paths.
                      "auto": "exact", "chunked": "exact", "streaming": "exact"}


def compute_stack_reference(
    frame_paths: list[StrOrPath],
    hdu: HDUSelector = 0,
    existing_bpm: np.ndarray | None = None,
    memmap: bool = False,
    method: str = "exact",
    chunk_size: int = 10,
    *,
    show_progress: bool = False,
    max_memory_mb: float = 1024.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel reference median and robust sigma (1.4826 x MAD) of a stack.

    Parameters
    ----------
    frame_paths
        FITS, ``.npy``, or ``.npz`` frames of identical shape.
    hdu
        HDU selector for FITS inputs, resolved independently in every file.
    existing_bpm
        Optional bad-pixel mask (non-zero = bad). Masked pixels have no valid
        data in any frame and are ``NaN`` in both outputs.
    memmap
        Whether to memory-map FITS files.
    method
        ``"exact"`` (default) computes the exact median and MAD in bands of
        rows with memory bounded by *max_memory_mb*, reading the dataset once.
        ``"p2"`` uses two passes of the approximate P² estimator.
        ``"auto"``, ``"chunked"``, and ``"streaming"`` are accepted as
        aliases for ``"exact"``.
    chunk_size
        Ignored; kept so existing calls keep working.
    show_progress
        Show a progress bar (requires tqdm).
    max_memory_mb
        Approximate memory budget for the exact method's row buffer.

    Returns
    -------
    (median, sigma)
        Float64 arrays with the frame shape.
    """
    del chunk_size  # memory is governed by max_memory_mb
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")
    key = method.lower().strip()
    if key not in _REFERENCE_METHODS:
        raise ValueError(f"method must be one of {sorted(_REFERENCE_METHODS)}; got {method!r}")
    if max_memory_mb <= 0:
        raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")

    if _REFERENCE_METHODS[key] == "p2":
        median, sigma = _compute_reference_p2(
            frame_paths, hdu=hdu, memmap=memmap, show_progress=show_progress
        )
    else:
        median, sigma = _compute_reference_exact(
            frame_paths, hdu=hdu, memmap=memmap,
            max_memory_mb=max_memory_mb, show_progress=show_progress,
        )

    if existing_bpm is not None:
        if existing_bpm.shape != median.shape:
            raise ValueError(
                f"existing_bpm shape {existing_bpm.shape} does not match frame shape {median.shape}"
            )
        bad = np.asarray(existing_bpm) != 0
        median[bad] = np.nan
        sigma[bad] = np.nan
    return median, sigma


def _frame_shape(path: StrOrPath, hdu: HDUSelector, memmap: bool) -> tuple[int, ...]:
    """Shape of one input frame, read without loading pixel data."""
    p = Path(path)
    if _is_fits_path(p):
        identity = _reference_identity(p, hdu, memmap)
        return tuple(int(s) for s in identity[3] or ())
    return tuple(int(s) for s in _open_numpy_frame(p).shape)


def _open_numpy_frame(path: Path) -> np.ndarray:
    """A .npy frame memory-mapped (rows read on demand), or an .npz frame."""
    if path.suffix.lower() == ".npy":
        return cast(np.ndarray, np.load(path, mmap_mode="r"))
    if path.suffix.lower() == ".npz":
        with np.load(path) as npz:
            return np.asarray(npz[sorted(npz.files)[0]])
    raise ValueError(f"Unsupported file format: {path}")


def _band_median_and_sigma(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact median and 1.4826 x MAD along axis 0, reusing *stack* in place."""
    median = np.median(stack, axis=0, overwrite_input=True)
    np.subtract(stack, median, out=stack)
    np.abs(stack, out=stack)
    mad = np.median(stack, axis=0, overwrite_input=True)
    return median, MAD_TO_STD_FACTOR * mad


def _compute_reference_exact(
    frame_paths: list[StrOrPath],
    hdu: HDUSelector,
    memmap: bool,
    max_memory_mb: float,
    *,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact median and MAD-based sigma in one banded pass over the stack."""
    paths = [Path(p) for p in frame_paths]
    reference = (
        _get_streaming_file_identity(paths[0], hdu, memmap=memmap)
        if _is_fits_path(paths[0]) else None
    )
    shape = _frame_shape(paths[0], hdu, memmap)

    def read_rows(_: int, path: Path, r0: int, r1: int) -> np.ndarray:
        if _is_fits_path(path):
            with fits.open(path, memmap=memmap) as hdul:
                index, _id = _resolve_streaming_hdu(hdul, path, hdu, expected_identity=reference)
                return _read_rows(hdul[index], r0, r1)
        frame = _open_numpy_frame(path)
        if frame.shape != shape:
            raise ValueError(f"Shape mismatch: {frame.shape}, expected {shape}")
        return np.asarray(frame[r0:r1], dtype=np.float64)

    (median, sigma), _n = _banded_reduce(
        paths, shape, read_rows, _band_median_and_sigma, 2, max_memory_mb,
        operation="stack_reference", show_progress=show_progress,
    )
    return median, sigma


def _compute_reference_p2(
    frame_paths: list[StrOrPath],
    hdu: HDUSelector = 0,
    memmap: bool = False,
    *,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate reference with two passes of the P² estimator."""
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
                path, hdu=hdu, memmap=memmap, expected_identity=reference_identity,
            )
        )
    reference_median = median_estimator.get_quantile()

    mad_estimator = P2QuantileEstimator(shape, p=0.5)
    iterator = maybe_tqdm(frame_paths, desc="Pass 2: MAD", unit="frame", disable=not show_progress)
    for path in iterator:
        data = _load_frame_for_streaming(
            path, hdu=hdu, memmap=memmap, expected_identity=reference_identity,
        )
        mad_estimator.update(np.abs(data - reference_median))

    return reference_median, MAD_TO_STD_FACTOR * mad_estimator.get_quantile()


def _load_frame_for_streaming(
    path: StrOrPath,
    hdu: HDUSelector = 0,
    memmap: bool = False,
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
    "streaming_calibrated_combine",
    "compute_stack_reference",
]
