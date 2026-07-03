#!/usr/bin/env python3
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Spatial filtering operations for FITS image data.

This module provides filtering capabilities for astronomical image processing,
including smoothing, edge detection, morphological operations, and offset
correction.

Notes
-----
- This module does not currently perform mask-aware filtering. Bad pixels/columns
  should be corrected or removed prior to filtering (e.g., via preprocessing).
- Filtering is computed in floating point to avoid integer overflow/underflow.
"""

from __future__ import annotations

# Standard library
import hashlib
import inspect
import json
import logging
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Third-party
import numpy as np
from astropy.io import fits
from scipy.ndimage import (
    gaussian_filter,
    grey_closing,
    grey_dilation,
    grey_erosion,
    grey_opening,
    laplace,
    median_filter,
    sobel,
    uniform_filter,
)

# Package imports
from ..array_validation import validate_numeric
from ..cli_utils import maybe_tqdm
from ..exceptions import FileHandlingError, FileOverwriteError, ImageProcessingError
from ..fits_output import (
    HDUResult,
    OutputMode,
    ResolvedHDU,
    StructurePolicy,
    commit_hdu_result,
    commit_hdu_results,
)
from ..hdu import (
    HDUQuery,
    HDUSelector,
    find_hdu_indices,
    get_hdu_identity,
    resolve_hdu_index,
    resolve_hdu_indices,
)
from ..header_utils import add_history
from ..io import region_to_slices
from ..selection import select_files
from ..types import StrOrPath

logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())


# ============================================================================
# Constants
# ============================================================================

DEFAULT_CHECKSUM_ALGORITHM = "sha256"
DEFAULT_CHUNK_SIZE = 65536  # 64KB


# ============================================================================
# Types
# ============================================================================

Region = str | tuple[int, int, int, int]
FilterParams = dict[str, Any]
WorkDtype = Literal["auto", "float32", "float64"]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(slots=True)
class ProcessingResult:
    """Result of processing a single file."""

    input_file: Path
    output_file: Path
    success: bool
    error: Exception | None = None
    checksum: str | None = None

    def __repr__(self) -> str:
        if self.success:
            return f"ProcessingResult(success, {self.output_file})"
        return f"ProcessingResult(failed, {self.input_file}: {self.error})"


@dataclass(slots=True)
class BatchResult:
    """Complete batch processing results."""

    successful: list[Path]
    failed: list[tuple[Path, Exception]]
    checksums: dict[Path, str]

    @property
    def success_count(self) -> int:
        """Number of successfully processed files."""
        return len(self.successful)

    @property
    def failure_count(self) -> int:
        """Number of failed files."""
        return len(self.failed)

    @property
    def total_count(self) -> int:
        """Total number of files processed."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        return 0.0 if self.total_count == 0 else (self.success_count / self.total_count)

    def __repr__(self) -> str:
        return (
            f"BatchResult(successful={self.success_count}, "
            f"failed={self.failure_count}, "
            f"rate={self.success_rate:.1%})"
        )


# ============================================================================
# Helper functions
# ============================================================================

def _validated_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Validate call parameters against ``func`` signature, then invoke it."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    try:
        sig.bind_partial(*args, **kwargs)
    except TypeError as e:
        raise ValueError(f"Invalid filter parameters for {func.__name__}: {e}") from e

    return func(*args, **kwargs)


def _compute_file_checksum(filepath: Path, algorithm: str = DEFAULT_CHECKSUM_ALGORITHM) -> str:
    """Compute checksum of a file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot compute checksum: {filepath} not found")
    if not filepath.is_file():
        raise ValueError(f"Cannot compute checksum: {filepath} is not a file")

    try:
        hash_obj = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(
            f"Unsupported hash algorithm {algorithm!r}. "
            f"Available: {', '.join(sorted(hashlib.algorithms_available))}"
        ) from e

    try:
        with filepath.open("rb") as f:
            for chunk in iter(lambda: f.read(DEFAULT_CHUNK_SIZE), b""):
                hash_obj.update(chunk)
    except OSError as e:
        raise OSError(f"Failed to read {filepath} for checksum: {e}") from e

    return hash_obj.hexdigest()


def _process_single_filter_task(task: FilterParams) -> None:
    """Worker function for :class:`~concurrent.futures.ProcessPoolExecutor`.

    Expects a dict with keys "input" and "output" plus filter parameters.
    """
    infile = task.get("input")
    outfile = task.get("output")
    if infile is None or outfile is None:
        raise ValueError("Task missing required 'input' or 'output' key")

    filter_params = {k: v for k, v in task.items() if k not in ("input", "output")}
    filter_hdu(infile, outfile, **filter_params)


def _safe_cast_to_dtype(data: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Safely cast data to target dtype with clipping/rounding for integers."""
    if np.issubdtype(target_dtype, np.integer):
        work = np.array(data, copy=True)
        work[~np.isfinite(work)] = 0
        work = np.rint(work)
        info = np.iinfo(target_dtype)
        work = np.clip(work, info.min, info.max)
        return work.astype(target_dtype)
    return data.astype(target_dtype)


def _require_2d(data: np.ndarray, op_name: str) -> None:
    if data.ndim != 2:
        raise ValueError(f"{op_name} requires 2D data, got shape={data.shape}")


# ============================================================================
# Smoothing Filters
# ============================================================================

def _apply_gaussian_filter(data: np.ndarray, sigma: float = 1.0, **kwargs: Any) -> np.ndarray:
    sigma_val = validate_numeric(sigma, "sigma", minimum=0.0, allow_zero=False)
    return _validated_call(gaussian_filter, data, sigma=sigma_val, **kwargs)


def _apply_median_filter(data: np.ndarray, size: int = 3, **kwargs: Any) -> np.ndarray:
    size_val = validate_numeric(size, "size", minimum=1, integer=True)
    return _validated_call(median_filter, data, size=size_val, **kwargs)


def _apply_mean_filter(data: np.ndarray, size: int = 3, **kwargs: Any) -> np.ndarray:
    size_val = validate_numeric(size, "size", minimum=1, integer=True)
    return _validated_call(uniform_filter, data, size=size_val, **kwargs)


def _apply_bilateral_filter(
    data: np.ndarray,
    sigma_spatial: float = 3.0,
    sigma_intensity: float = 50.0,
    **kwargs: Any,
) -> np.ndarray:
    sigma_spatial_val = validate_numeric(sigma_spatial, "sigma_spatial", minimum=0.0, allow_zero=False)
    sigma_intensity_val = validate_numeric(sigma_intensity, "sigma_intensity", minimum=0.0, allow_zero=False)

    work = np.asarray(data, dtype=np.float32)
    smoothed = _validated_call(gaussian_filter, work, sigma=sigma_spatial_val, **kwargs)
    diff = np.abs(work - smoothed)
    weights = np.exp(-((diff**2) / (2 * sigma_intensity_val**2)))
    return smoothed * weights + work * (1 - weights)


# ============================================================================
# Edge Detection Filters
# ============================================================================

def _apply_sobel_filter(data: np.ndarray, axis: int | None = None, **kwargs: Any) -> np.ndarray:
    work = np.asarray(data, dtype=np.float32)

    if axis is not None:
        if not (-work.ndim <= axis < work.ndim):
            raise ValueError(f"axis out of range for ndim={work.ndim}: {axis}")
        return _validated_call(sobel, work, axis=axis, **kwargs)

    if work.ndim < 2:
        raise ValueError(f"sobel magnitude requires ndim>=2, got shape={work.shape}")

    sx = _validated_call(sobel, work, axis=-2, **kwargs)
    sy = _validated_call(sobel, work, axis=-1, **kwargs)
    return np.hypot(sx, sy)


def _apply_laplacian_filter(data: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _validated_call(laplace, data, **kwargs)


# ============================================================================
# Morphological Filters
# ============================================================================

def _apply_opening(
    data: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
    **kwargs: Any,
) -> np.ndarray:
    iterations_val = validate_numeric(iterations, "iterations", minimum=1, integer=True)
    out = data
    scipy_kwargs = {"mode": "reflect", **kwargs}
    for _ in range(iterations_val):
        out = _validated_call(grey_opening, out, structure=structure, **scipy_kwargs)
    return out


def _apply_closing(
    data: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
    **kwargs: Any,
) -> np.ndarray:
    iterations_val = validate_numeric(iterations, "iterations", minimum=1, integer=True)
    out = data
    scipy_kwargs = {"mode": "reflect", **kwargs}
    for _ in range(iterations_val):
        out = _validated_call(grey_closing, out, structure=structure, **scipy_kwargs)
    return out


def _apply_erosion(
    data: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
    **kwargs: Any,
) -> np.ndarray:
    iterations_val = validate_numeric(iterations, "iterations", minimum=1, integer=True)
    out = data
    scipy_kwargs = {"mode": "reflect", **kwargs}
    for _ in range(iterations_val):
        out = _validated_call(grey_erosion, out, structure=structure, **scipy_kwargs)
    return out


def _apply_dilation(
    data: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
    **kwargs: Any,
) -> np.ndarray:
    iterations_val = validate_numeric(iterations, "iterations", minimum=1, integer=True)
    out = data
    scipy_kwargs = {"mode": "reflect", **kwargs}
    for _ in range(iterations_val):
        out = _validated_call(grey_dilation, out, structure=structure, **scipy_kwargs)
    return out


# ============================================================================
# Offset Correction Filters
# ============================================================================

def _correct_row_offset(data: np.ndarray, method: str = "median") -> np.ndarray:
    _require_2d(data, "correct_row_offset")
    if method not in ("mean", "median"):
        raise ValueError(f"method must be 'mean' or 'median', got {method!r}")

    data_float = data.astype(np.float32)
    offsets = np.median(data_float, axis=1) if method == "median" else np.mean(data_float, axis=1)
    return data_float - offsets[:, np.newaxis]


def _correct_col_offset(data: np.ndarray, method: str = "median") -> np.ndarray:
    _require_2d(data, "correct_col_offset")
    if method not in ("mean", "median"):
        raise ValueError(f"method must be 'mean' or 'median', got {method!r}")

    data_float = data.astype(np.float32)
    offsets = np.median(data_float, axis=0) if method == "median" else np.mean(data_float, axis=0)
    return data_float - offsets


def _apply_correct_row_median(data: np.ndarray) -> np.ndarray:
    return _correct_row_offset(data, method="median")


def _apply_correct_row_mean(data: np.ndarray) -> np.ndarray:
    return _correct_row_offset(data, method="mean")


def _apply_correct_col_median(data: np.ndarray) -> np.ndarray:
    return _correct_col_offset(data, method="median")


def _apply_correct_col_mean(data: np.ndarray) -> np.ndarray:
    return _correct_col_offset(data, method="mean")


# ============================================================================
# Enhancement Filters
# ============================================================================

def _apply_unsharp_mask(data: np.ndarray, sigma: float = 1.0, amount: float = 1.0, **kwargs: Any) -> np.ndarray:
    sigma_val = validate_numeric(sigma, "sigma", minimum=0.0, allow_zero=False)
    amount_val = validate_numeric(amount, "amount", minimum=0.0)

    work = np.asarray(data, dtype=np.float32)
    blurred = _validated_call(gaussian_filter, work, sigma=sigma_val, **kwargs)
    return work + amount_val * (work - blurred)


def _apply_high_pass(data: np.ndarray, sigma: float = 3.0, **kwargs: Any) -> np.ndarray:
    sigma_val = validate_numeric(sigma, "sigma", minimum=0.0, allow_zero=False)

    work = np.asarray(data, dtype=np.float32)
    low_pass = _validated_call(gaussian_filter, work, sigma=sigma_val, **kwargs)
    return work - low_pass


# ============================================================================
# Operation Registry
# ============================================================================

OPERATIONS: dict[str, Any] = {
    # Smoothing
    "gaussian": _apply_gaussian_filter,
    "median": _apply_median_filter,
    "mean": _apply_mean_filter,
    "uniform": _apply_mean_filter,
    "bilateral": _apply_bilateral_filter,
    # Edge detection
    "sobel": _apply_sobel_filter,
    "laplacian": _apply_laplacian_filter,
    # Morphological
    "opening": _apply_opening,
    "closing": _apply_closing,
    "erosion": _apply_erosion,
    "dilation": _apply_dilation,
    # Backward-compat aliases (keep for now)
    "binary_opening": _apply_opening,
    "binary_closing": _apply_closing,
    # Correction
    "correct_row_median": _apply_correct_row_median,
    "correct_row_mean": _apply_correct_row_mean,
    "correct_col_median": _apply_correct_col_median,
    "correct_col_mean": _apply_correct_col_mean,
    # Enhancement
    "unsharp_mask": _apply_unsharp_mask,
    "high_pass": _apply_high_pass,
}


# ============================================================================
# Worker helpers
# ============================================================================

def _select_work_dtype(orig_dtype: np.dtype, work_dtype: WorkDtype) -> np.dtype:
    if work_dtype == "float32":
        return np.dtype(np.float32)
    if work_dtype == "float64":
        return np.dtype(np.float64)

    if np.issubdtype(orig_dtype, np.floating) and orig_dtype == np.float64:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _apply_filter_hdu(
    hdul_out: fits.HDUList,
    hdu_index: int,
    operation: str,
    region: Region,
    preserve_dtype: bool,
    work_dtype: WorkDtype = "auto",
    **kwargs: Any,
) -> None:
    hdu_obj = hdul_out[hdu_index]

    if not isinstance(hdu_obj, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        raise ValueError(
            f"hdu_index={hdu_index} is not an image HDU (type={type(hdu_obj).__name__})"
        )
    if hdu_obj.data is None:
        raise ImageProcessingError(f"HDU {hdu_index} contains no image data")

    orig_full = np.asarray(hdu_obj.data)
    if orig_full.ndim != 2:
        raise ValueError(f"{operation} expects 2D image data; got shape={orig_full.shape}")

    orig_dtype = orig_full.dtype
    if not np.issubdtype(orig_dtype, np.number):
        raise ValueError(f"Unsupported data dtype for filtering: {orig_dtype}")

    work_dtype_np = _select_work_dtype(orig_dtype, work_dtype)
    work_full = orig_full.astype(work_dtype_np, copy=False)

    # Call region_to_slices without filepath kwarg to avoid signature coupling.
    row_slice, col_slice = region_to_slices(work_full.shape, hdu_obj.header, region)
    view = work_full[row_slice, col_slice]

    filtered = OPERATIONS[operation](view, **kwargs)
    filtered = np.asarray(filtered, dtype=work_dtype_np)

    if filtered.shape != view.shape:
        raise ImageProcessingError(
            f"Filter {operation!r} returned shape {filtered.shape}, expected {view.shape}"
        )

    if preserve_dtype:
        filtered_cast = filtered if filtered.dtype == orig_dtype else _safe_cast_to_dtype(filtered, orig_dtype)
        if region in (None, "full"):
            out_data = filtered_cast
        else:
            out_full = np.array(orig_full, copy=True)
            out_full[row_slice, col_slice] = filtered_cast
            out_data = out_full
    else:
        if region in (None, "full"):
            out_data = filtered
        else:
            out_full = np.array(work_full, copy=True)
            out_full[row_slice, col_slice] = filtered
            out_data = out_full

    hdu_obj.data = out_data

    add_history(hdu_obj.header, f"Applied filter: {operation}")
    add_history(hdu_obj.header, f"Region={region!r}")
    add_history(hdu_obj.header, f"Work dtype={work_dtype_np.name}, preserve_dtype={preserve_dtype}")
    if np.asarray(out_data).dtype != orig_dtype:
        add_history(hdu_obj.header, f"Data dtype: {orig_dtype} -> {np.asarray(out_data).dtype}")


# ============================================================================
# Main Processing Functions
# ============================================================================

def filter_hdu(
    input_file: StrOrPath,
    output_file: StrOrPath,
    operation: str,
    region: Region = "full",
    hdu: HDUSelector = "auto",
    *,
    output_mode: OutputMode = "preserve",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
    preserve_dtype: bool = False,
    work_dtype: WorkDtype = "auto",
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Apply a spatial filter to one selected 2D image HDU.

    ``hdu`` accepts an index, an ``EXTNAME``, an ``(EXTNAME, EXTVER)``
    pair, or ``"auto"``. Ambiguous ``EXTNAME`` values are rejected.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if output_mode != "inplace" and input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different paths")

    operation = operation.strip().lower()
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation!r}")

    try:
        hdul = fits.open(input_path, mode="readonly", memmap=False)
    except Exception as e:
        raise FileHandlingError(f"FITS I/O error opening file: {e}", filepath=input_path) from e

    with hdul:
        hdul_out = fits.HDUList([h.copy() for h in hdul])
        hdu_index = resolve_hdu_index(
            hdul_out,
            hdu,
            auto="first_sci_image",
            image_like=True,
            require_data=True,
            filepath=input_path,
        )

        _apply_filter_hdu(
            hdul_out=hdul_out,
            hdu_index=hdu_index,
            operation=operation,
            region=region,
            preserve_dtype=preserve_dtype,
            work_dtype=work_dtype,
            **kwargs,
        )
        source_hdu = hdul[hdu_index]
        result_hdu = hdul_out[hdu_index]
        source = ResolvedHDU(
            input_path,
            hdu_index,
            get_hdu_identity(source_hdu, hdu_index),
            np.array(source_hdu.data, copy=True),
            source_hdu.header.copy(),
        )
        hdul.close()
        commit_hdu_result(
            source,
            output_path,
            np.asarray(result_hdu.data),
            result_hdu.header,
            output_mode=output_mode,
            destination_hdu=destination_hdu,
            structure_policy=structure_policy,
            overwrite=overwrite,
        )
        logger.debug("Filtered image saved to %s", output_path)


def filter_image_hdus(
    input_file: StrOrPath,
    output_file: StrOrPath,
    operation: str,
    region: Region = "full",
    include_extnames: tuple[str, ...] = ("SCI", "IMAGE"),
    *,
    selectors: Iterable[HDUSelector] | None = None,
    query: HDUQuery | None = None,
    output_mode: OutputMode = "preserve",
    structure_policy: StructurePolicy = "compatible",
    preserve_dtype: bool = False,
    work_dtype: WorkDtype = "auto",
    overwrite: bool = False,
    **kwargs: Any,
) -> list[int]:
    """Apply a spatial filter to selected 2D image HDUs in a MEF file.

    Use ``selectors`` for exact, caller-ordered selection or ``query`` for
    discovery in file order. These modes are mutually exclusive. If neither
    is supplied, ``include_extnames`` provides the backward-compatible
    EXTNAME query.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if output_mode == "sef":
        raise ValueError("filter_image_hdus does not support output_mode='sef'")
    if output_mode != "inplace" and input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different paths")

    operation = operation.strip().lower()
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation!r}")
    if selectors is not None and query is not None:
        raise ValueError("selectors and query are mutually exclusive")

    try:
        hdul = fits.open(input_path, mode="readonly", memmap=False)
    except Exception as e:
        raise FileHandlingError(f"FITS I/O error opening file: {e}", filepath=input_path) from e

    with hdul:
        hdul_out = fits.HDUList([h.copy() for h in hdul])

        if selectors is not None:
            hdu_indices = resolve_hdu_indices(
                hdul_out,
                selectors,
                image_like=True,
                require_data=True,
                filepath=input_path,
            )
        else:
            effective_query = (
                HDUQuery(
                    extnames=query.extnames,
                    extvers=query.extvers,
                    include_primary=query.include_primary,
                    image_like=True,
                    require_data=True,
                )
                if query is not None
                else HDUQuery(extnames=frozenset(include_extnames))
            )
            hdu_indices = find_hdu_indices(
                hdul_out,
                effective_query,
                require_match=False,
            )
        if not hdu_indices:
            selection = query if query is not None else include_extnames
            raise ValueError(f"No image HDUs found for selection {selection!r}")

        sources: list[ResolvedHDU] = []
        for hdu_index in hdu_indices:
            source_hdu = hdul[hdu_index]
            sources.append(
                ResolvedHDU(
                    input_path,
                    hdu_index,
                    get_hdu_identity(source_hdu, hdu_index),
                    np.array(source_hdu.data, copy=True),
                    source_hdu.header.copy(),
                )
            )
            _apply_filter_hdu(
                hdul_out=hdul_out,
                hdu_index=hdu_index,
                operation=operation,
                region=region,
                preserve_dtype=preserve_dtype,
                work_dtype=work_dtype,
                **kwargs,
            )

        payloads = [
            HDUResult(
                source,
                np.asarray(hdul_out[hdu_index].data),
                hdul_out[hdu_index].header,
            )
            for source, hdu_index in zip(
                sources, hdu_indices, strict=True
            )
        ]
        hdul.close()
        commit_hdu_results(
            payloads,
            output_path,
            output_mode=output_mode,
            structure_policy=structure_policy,
            overwrite=overwrite,
        )
        logger.debug("Filtered MEF saved to %s", output_path)

    return hdu_indices


def filter_hdu_batch(
    input_files: StrOrPath | Any,
    output_pattern: StrOrPath,
    operation: str,
    *,
    region: Region = "full",
    hdu: HDUSelector = "auto",
    preserve_dtype: bool = False,
    work_dtype: WorkDtype = "auto",
    overwrite: bool = False,
    num_workers: int = 1,
    validate_outputs: bool = False,
    checksum_algorithm: str = DEFAULT_CHECKSUM_ALGORITHM,
    show_progress: bool = True,
    dry_run: bool = False,
    **kwargs: Any,
) -> BatchResult:
    """Apply a filter to multiple FITS files in batch mode."""
    reserved = {
        "input", "output", "operation", "hdu", "region",
        "preserve_dtype", "work_dtype", "overwrite",
        "num_workers", "validate_outputs", "checksum_algorithm", "show_progress", "dry_run",
    }
    conflicts = reserved.intersection(kwargs)
    if conflicts:
        raise ValueError(f"filter_hdu_batch received reserved kwargs: {sorted(conflicts)}")

    num_workers = int(validate_numeric(num_workers, "num_workers", minimum=1, integer=True))

    if validate_outputs:
        hashlib.new(checksum_algorithm)

    output_pattern_str = str(output_pattern)
    if "{stem}" not in output_pattern_str:
        raise ValueError("output_pattern must contain '{stem}' placeholder")

    try:
        output_pattern_str.format(stem="test", index=0)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid output_pattern format string: {e}") from e

    try:
        files = select_files(input_files)
    except Exception as e:
        raise FileHandlingError(f"Failed to select input files: {e}") from e

    if not files:
        logger.warning("No input files found matching: %s", input_files)
        return BatchResult(successful=[], failed=[], checksums={})

    tasks: list[dict[str, Any]] = []
    output_files: list[Path] = []
    input_paths: list[Path] = []

    for idx, infile in enumerate(files):
        infile_path = Path(infile)
        input_paths.append(infile_path)

        # Better "stem" for compressed FITS: Path.stem of image.fits.gz -> "image.fits"
        # We want "image" in most cases.
        name = infile_path.name
        for ext in (".fits.gz", ".fits.fz", ".fit.fz", ".fts.fz", ".fits", ".fit", ".fts"):
            if name.lower().endswith(ext):
                stem = name[: -len(ext)]
                break
        else:
            stem = infile_path.stem

        output_str = output_pattern_str.format(stem=stem, index=idx)
        output_path = Path(output_str)

        if not overwrite and output_path.exists():
            if dry_run:
                logger.warning("Output file exists: %s", output_path)
            else:
                raise FileOverwriteError(output_path)

        output_files.append(output_path)

        tasks.append(
            {
                "input": str(infile_path),
                "output": str(output_path),
                "operation": operation,
                "region": region,
                "hdu": hdu,
                "preserve_dtype": preserve_dtype,
                "work_dtype": work_dtype,
                "overwrite": overwrite,
                **kwargs,
            }
        )

    if dry_run:
        logger.info("=== DRY RUN SUMMARY ===")
        logger.info("Input files: %d", len(input_paths))
        logger.info("Operation: %s", operation)
        logger.info("Parameters: %s", kwargs)
        logger.info("======================")
        return BatchResult(successful=[], failed=[], checksums={})

    results: list[ProcessingResult] = []

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_process_single_filter_task, task): i for i, task in enumerate(tasks)
            }

            futures_iter = as_completed(future_to_idx)
            if show_progress:
                futures_iter = maybe_tqdm(
                    futures_iter,
                    desc="Processing files",
                    total=len(future_to_idx),
                    unit="file",
                    dynamic_ncols=True,
                )

            for future in futures_iter:
                idx = future_to_idx[future]
                input_path = input_paths[idx]
                output_path = output_files[idx]

                try:
                    future.result()
                    checksum = None
                    if validate_outputs and output_path.exists():
                        checksum = _compute_file_checksum(output_path, checksum_algorithm)
                    results.append(
                        ProcessingResult(
                            input_file=input_path,
                            output_file=output_path,
                            success=True,
                            checksum=checksum,
                        )
                    )
                except Exception as e:
                    results.append(
                        ProcessingResult(
                            input_file=input_path,
                            output_file=output_path,
                            success=False,
                            error=e,
                        )
                    )
    else:
        task_iter = enumerate(tasks)
        if show_progress:
            task_iter = maybe_tqdm(task_iter, desc="Processing files", total=len(tasks), unit="file", dynamic_ncols=True)

        for idx, task in task_iter:
            input_path = input_paths[idx]
            output_path = output_files[idx]
            try:
                _process_single_filter_task(task)
                checksum = None
                if validate_outputs and output_path.exists():
                    checksum = _compute_file_checksum(output_path, checksum_algorithm)
                results.append(
                    ProcessingResult(
                        input_file=input_path,
                        output_file=output_path,
                        success=True,
                        checksum=checksum,
                    )
                )
            except Exception as e:
                results.append(
                    ProcessingResult(
                        input_file=input_path,
                        output_file=output_path,
                        success=False,
                        error=e,
                    )
                )

    successful = [r.output_file for r in results if r.success]
    failed = [(r.input_file, r.error) for r in results if not r.success and r.error is not None]
    checksums = {r.output_file: r.checksum for r in results if r.success and r.checksum is not None}

    if failed:
        logger.warning(
            "Completed with %d failures out of %d files (%.1f%% success rate)",
            len(failed),
            len(results),
            100.0 * len(successful) / len(results),
        )
    else:
        logger.info("Successfully processed all %d files", len(results))

    return BatchResult(successful=successful, failed=failed, checksums=checksums)


def save_batch_report(result: BatchResult, report_path: StrOrPath, include_checksums: bool = True) -> None:
    """Save a batch processing report (text or JSON based on suffix)."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    if report_path.suffix.lower() == ".json":
        report_data: dict[str, Any] = {
            "timestamp": timestamp,
            "summary": {
                "total": result.total_count,
                "successful": result.success_count,
                "failed": result.failure_count,
                "success_rate": result.success_rate,
            },
            "successful_files": [str(p) for p in result.successful],
            "failed_files": [{"input": str(inp), "error": str(err)} for inp, err in result.failed],
        }
        if include_checksums and result.checksums:
            report_data["checksums"] = {str(k): v for k, v in result.checksums.items()}

        report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    else:
        lines: list[str] = []
        lines.append("Batch Processing Report")
        lines.append(f"Generated: {timestamp}")
        lines.append("=" * 60)
        lines.append("")
        lines.append("Summary:")
        lines.append(f"  Total files:   {result.total_count}")
        lines.append(f"  Successful:    {result.success_count}")
        lines.append(f"  Failed:        {result.failure_count}")
        lines.append(f"  Success rate:  {result.success_rate:.1%}")
        lines.append("")

        if result.successful:
            lines.append(f"Successful files ({len(result.successful)}):")
            for path in result.successful:
                lines.append(f"  OK   {path}")
                if include_checksums and path in result.checksums:
                    lines.append(f"       {result.checksums[path]}")
            lines.append("")

        if result.failed:
            lines.append(f"Failed files ({len(result.failed)}):")
            for inp, err in result.failed:
                lines.append(f"  FAIL {inp}")
                lines.append(f"       {err}")

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Report saved to %s", report_path)


def verify_output_checksums(
    result: BatchResult,
    checksum_algorithm: str = DEFAULT_CHECKSUM_ALGORITHM,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Verify output files match their recorded checksums."""
    if not result.checksums:
        logger.warning("No checksums to verify in result")
        return [], []

    verified: list[Path] = []
    corrupted: list[tuple[Path, str]] = []

    for filepath, original_checksum in result.checksums.items():
        if not filepath.exists():
            corrupted.append((filepath, "File no longer exists"))
            continue

        try:
            current_checksum = _compute_file_checksum(filepath, checksum_algorithm)
            if current_checksum == original_checksum:
                verified.append(filepath)
            else:
                corrupted.append(
                    (filepath, f"Checksum mismatch: expected {original_checksum}, got {current_checksum}")
                )
        except Exception as e:
            corrupted.append((filepath, f"Verification failed: {e}"))

    return verified, corrupted


def list_operations() -> dict[str, str]:
    """List all available operations with brief descriptions."""
    return {
        "gaussian": "Gaussian smoothing filter (sigma: blur strength)",
        "median": "Median filter (size: kernel size)",
        "mean": "Mean/uniform smoothing filter (size: kernel size)",
        "uniform": "Alias for mean filter",
        "bilateral": "Edge-preserving smoothing (sigma_spatial, sigma_intensity)",
        "sobel": "Sobel edge detection (axis: 0/1 or None for magnitude)",
        "laplacian": "Laplacian edge detection",
        "opening": "Morphological opening",
        "closing": "Morphological closing",
        "erosion": "Morphological erosion",
        "dilation": "Morphological dilation",
        "binary_opening": "Alias for opening (legacy)",
        "binary_closing": "Alias for closing (legacy)",
        "correct_row_median": "Remove row-wise median offset",
        "correct_row_mean": "Remove row-wise mean offset",
        "correct_col_median": "Remove column-wise median offset",
        "correct_col_mean": "Remove column-wise mean offset",
        "unsharp_mask": "Unsharp mask (sigma, amount)",
        "high_pass": "High-pass filter (sigma)",
    }


def print_operations(category: str | None = None) -> None:
    """Print available operations in a formatted table."""
    ops = list_operations()
    print("\nAVAILABLE FILTER OPERATIONS:")
    print("=" * 80)
    for name in sorted(ops):
        print(f"  {name:20s} {ops[name]}")
    print()


def get_operations_by_category() -> dict[str, list[str]]:
    """Return operation names grouped by category."""
    return {
        "smoothing": ["gaussian", "median", "mean", "uniform", "bilateral"],
        "edge": ["sobel", "laplacian"],
        "morphological": ["opening", "closing", "erosion", "dilation", "binary_opening", "binary_closing"],
        "correction": ["correct_row_median", "correct_row_mean", "correct_col_median", "correct_col_mean"],
        "enhancement": ["unsharp_mask", "high_pass"],
    }


# ============================================================================
# API aliases (helps resolve stale docs/old names without breaking)
# ============================================================================

apply_filter = filter_hdu
apply_filter_mef = filter_image_hdus
filter_batch = filter_hdu_batch


__all__ = [
    # Data classes
    "ProcessingResult",
    "BatchResult",
    # Core functions
    "filter_hdu",
    "filter_image_hdus",
    "filter_hdu_batch",
    "save_batch_report",
    "verify_output_checksums",
    # Operation registry + introspection
    "OPERATIONS",
    "list_operations",
    "print_operations",
    "get_operations_by_category",
    # Aliases (legacy/stability)
    "apply_filter",
    "apply_filter_mef",
    "filter_batch",
]
