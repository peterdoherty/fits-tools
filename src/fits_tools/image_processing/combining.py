#!/usr/bin/env python3
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Image combining and stacking operations.

This module provides robust image combining functionality for creating master
calibration frames and stacking science images. It supports multiple combining
methods with outlier rejection, weighted combining, and memory-efficient
loading for large datasets.

Public API
----------
combine_images_robust
    Modern combining API with sigma clipping, masking, and weighting.

create_master_frame
    Convenience wrapper for calibration frames (bias/dark/flat), with optional
    normalization for flats.

median_combine / mean_combine
    Convenience wrappers around `combine_images_robust`.

combine_images
    Deprecated legacy API.

Notes
-----
- For file-based inputs, images are loaded into a pre-allocated stack
  (N, rows, cols) to reduce peak memory versus list-then-stack approaches.
- For in-memory array inputs, stacking is unavoidable and no memory benefit is
  expected.
"""

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

from ..array_validation import validate_shapes
from ..constants import FITS_KEY_EXPTIME
from ..exceptions import ImageProcessingError, SelectionError
from ..fits_output import (
    HDUResult,
    OutputMode,
    StructurePolicy,
    commit_hdu_results,
)
from ..hdu import (
    HDUIdentity,
    HDUQuery,
    HDUSelector,
    find_hdu_indices,
    resolve_hdu_indices,
)
from ..header_utils import add_history
from ..io import get_header, write_simple_fits
from ..resolved_hdu import identity_selector, load_resolved_hdu
from ..selection import select_files
from ..types import StrOrPath
from .result_types import CombiningResult

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================


def _combining_identity(
    identity: HDUIdentity,
) -> tuple[str | None, int | None, str, tuple[int, ...] | None]:
    """Return the cross-file structural identity used for singular stacks."""
    return (
        identity.extname,
        identity.extver,
        identity.hdu_type,
        identity.shape,
    )


def _load_and_stack_efficient(
    input_param: str | list[StrOrPath] | list[np.ndarray],
    hdu: HDUSelector = 0,
    keys: dict[str, Any] | None = None,
    working_dtype: np.dtype = np.float64,
    recursive: bool = False,
) -> tuple[np.ndarray, list[Path], fits.Header]:
    """Load and stack images into a single 3D array.

    Parameters
    ----------
    input_param
        Input specification:

        - Directory path (FITS files selected via :func:`fits_tools.selection.select_files`)
        - List of FITS file paths
        - List of in-memory NumPy arrays
    hdu
        HDU selector resolved independently in every file input.
    keys
        Optional primary-header keyword filters used only when *input_param* is a directory.
    working_dtype
        Dtype for the stacked output array (and downstream arithmetic).
    recursive
        If True, recursively search a directory input.

    Returns
    -------
    stacked_data, input_files, base_header
        stacked_data
            Array of shape ``(N, rows, cols)`` with dtype ``working_dtype``.
        input_files
            List of files actually loaded into the stack (empty for in-memory arrays).
        base_header
            Header from the first successfully loaded file (or empty header for in-memory arrays).

    Raises
    ------
    ImageProcessingError
        If no valid data can be loaded, selection fails, or the stack cannot be allocated.

    Notes
    -----
    File inputs are all-or-nothing. A missing, ambiguous, or structurally
    mismatched selected HDU rejects the complete stack.
    """
    base_header = fits.Header()

    # Case 1: In-memory arrays
    if isinstance(input_param, list) and input_param and all(isinstance(i, np.ndarray) for i in input_param):
        logger.debug("Processing %d in-memory arrays", len(input_param))
        validate_shapes(*input_param, operation="combining")

        try:
            stacked = np.stack(input_param, axis=0).astype(working_dtype, copy=False)
        except ValueError as e:
            raise ImageProcessingError(f"Failed to stack images — dimension mismatch: {e}") from e

        add_history(base_header, f"Combined from {len(input_param)} in-memory arrays")
        return stacked, [], base_header

    # Case 2: Load from files
    if isinstance(input_param, str):
        try:
            candidates = select_files(input_param, keys=keys, recursive=recursive)
        except SelectionError as e:
            raise ImageProcessingError(f"File selection failed: {e}") from e
        file_list = [Path(p) for p in candidates]
    else:
        file_list = [Path(f) for f in input_param]

    if not file_list:
        raise ImageProcessingError("No files found for combining")

    try:
        first = load_resolved_hdu(file_list[0], hdu)
    except Exception as exc:
        raise ImageProcessingError(
            f"Failed to resolve/load selected HDU {hdu!r} from "
            f"{file_list[0]}: {exc}"
        ) from exc

    first_data = first.data
    shape = first_data.shape
    reference_identity = _combining_identity(first.identity)

    # Pre-allocate stack
    bytes_per_element = int(np.dtype(working_dtype).itemsize)
    total_bytes = len(file_list) * int(shape[0]) * int(shape[1]) * bytes_per_element
    total_gb = total_bytes / (1024**3)

    if total_gb >= 4.0:
        logger.warning("Large memory allocation (%.2f GB). Ensure sufficient RAM.", total_gb)
    else:
        logger.debug("Estimated stack allocation: %.2f GB", total_gb)

    try:
        stacked = np.zeros((len(file_list), shape[0], shape[1]), dtype=working_dtype)
    except MemoryError as exc:
        raise ImageProcessingError(
            f"Insufficient memory to allocate array for {len(file_list)} images "
            f"({total_gb:.2f} GB required)."
        ) from exc

    stacked[0] = np.asarray(first_data, dtype=working_dtype)

    for position, filepath in enumerate(file_list[1:], start=1):
        try:
            resolved = load_resolved_hdu(filepath, hdu)
        except Exception as exc:
            raise ImageProcessingError(
                f"Failed to resolve/load selected HDU {hdu!r} from "
                f"{filepath}: {exc}"
            ) from exc

        identity = _combining_identity(resolved.identity)
        if identity != reference_identity:
            raise ImageProcessingError(
                f"Selected HDU identity mismatch in {filepath}: "
                f"{identity!r} does not match reference "
                f"{reference_identity!r} from {file_list[0]}"
            )
        stacked[position] = np.asarray(resolved.data, dtype=working_dtype)

    base_header = first.header.copy()
    add_history(base_header, f"Combined from {len(file_list)} file(s)")
    return stacked, file_list, base_header


def _apply_mask(stacked_data: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Apply a 2D bad pixel mask to a stacked cube."""
    if mask is None:
        return stacked_data

    if mask.shape != stacked_data.shape[1:]:
        raise ImageProcessingError(
            f"Mask shape {mask.shape} does not match image shape {stacked_data.shape[1:]}"
        )

    mask_3d = np.broadcast_to(mask, stacked_data.shape)
    return np.ma.masked_array(stacked_data, mask=mask_3d)


def _combine_sigma_clipped_median(
    stacked_data: np.ndarray,
    sigma: float = 3.0,
    maxiters: int = 5,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Sigma-clipped median combine."""
    clipped = sigma_clip(stacked_data, sigma=sigma, maxiters=maxiters, axis=0)
    combined = np.ma.median(clipped, axis=0)

    num_clipped = int(np.sum(clipped.mask)) if hasattr(clipped, "mask") else 0
    if isinstance(combined, np.ma.MaskedArray):
        combined = combined.filled(fill_value)

    return combined, num_clipped


def _combine_sigma_clipped_mean(
    stacked_data: np.ndarray,
    sigma: float = 3.0,
    maxiters: int = 5,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Sigma-clipped mean combine."""
    clipped = sigma_clip(stacked_data, sigma=sigma, maxiters=maxiters, axis=0)
    combined = np.ma.mean(clipped, axis=0)

    num_clipped = int(np.sum(clipped.mask)) if hasattr(clipped, "mask") else 0
    if isinstance(combined, np.ma.MaskedArray):
        combined = combined.filled(fill_value)

    return combined, num_clipped


def _combine_median(stacked_data: np.ndarray) -> tuple[np.ndarray, int]:
    """Simple median combine (no outlier rejection)."""
    return np.median(stacked_data, axis=0), 0


def _combine_mean(stacked_data: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Mean combine, optionally weighted."""
    if weights is not None:
        weights = np.asarray(weights)
        if len(weights) != stacked_data.shape[0]:
            raise ImageProcessingError(
                f"Number of weights ({len(weights)}) does not match number of images ({stacked_data.shape[0]})"
            )
        combined = np.average(stacked_data, axis=0, weights=weights)
    else:
        combined = np.mean(stacked_data, axis=0)
    return combined, 0


# ============================================================================
# Main Combining Functions
# ============================================================================


def combine_images_robust(
    input_param: str | list[StrOrPath] | list[np.ndarray],
    method: str = "sigma_clipped_median",
    sigma: float = 3.0,
    maxiters: int = 5,
    mask: np.ndarray | None = None,
    weights: list[float] | np.ndarray | str | None = None,
    output_file: StrOrPath | None = None,
    hdu: HDUSelector = 0,
    keys: dict[str, Any] | None = None,
    working_dtype: np.dtype = np.float64,
    recursive: bool = False,
    overwrite: bool = False,
    return_type: str = "auto",
) -> np.ndarray | Path | CombiningResult:
    """Combine multiple images using robust statistics with outlier rejection.

    Parameters
    ----------
    input_param
        Directory path, list of file paths, or list of NumPy arrays.
    method
        Combining method. Supported values:

        - ``"sigma_clipped_median"`` (recommended for calibration frames)
        - ``"sigma_clipped_mean"``
        - ``"median"``
        - ``"mean"``
        - ``"weighted_mean"``
    sigma
        Sigma threshold for sigma clipping (sigma-clipped methods only).
    maxiters
        Maximum clipping iterations (sigma-clipped methods only).
    mask
        Optional boolean bad-pixel mask (True = exclude). Must match image shape.
    weights
        Weights for ``"weighted_mean"``. May be:

        - array-like of shape (N,)
        - ``"exptime"`` to weight by FITS ``EXPTIME`` (file inputs only)
        - ``"inverse_variance"`` to weight by 1/variance of each frame
    output_file
        If provided, write the combined image to this FITS path.
    hdu
        HDU selector resolved independently in every input file. The selected
        extension identity, HDU type, and shape must agree across the stack.
    keys
        Primary-header selection filters used only when *input_param* is a directory.
    working_dtype
        Dtype used for stacking and arithmetic.
    recursive
        Recursive directory search when *input_param* is a directory.
    overwrite
        Overwrite *output_file* if it exists.
    return_type
        Controls the return value:

        - ``"array"``  -> return ``np.ndarray``
        - ``"path"``   -> return ``Path`` (requires *output_file*)
        - ``"result"`` -> return :class:`~fits_tools.image_processing.result_types.CombiningResult`
        - ``"auto"``   -> currently the same as ``"result"``

        **Note:** ``return_type="auto"`` returns a `CombiningResult` even when
        *output_file* is provided. The output path is available as
        ``result.output_file``.

    Returns
    -------
    np.ndarray or Path or CombiningResult
        Return type depends on *return_type*.

    Raises
    ------
    ValueError
        For invalid method/parameters.
    ImageProcessingError
        For data loading, shape mismatch, masking, or combining failures.
    """
    valid_methods = {
        "sigma_clipped_median",
        "sigma_clipped_mean",
        "median",
        "mean",
        "weighted_mean",
    }
    if method not in valid_methods:
        raise ValueError(f"Unknown combining method {method!r}. Valid options: {sorted(valid_methods)}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if maxiters < 1:
        raise ValueError(f"maxiters must be >= 1, got {maxiters}")

    stacked_data, file_list, base_header = _load_and_stack_efficient(
        input_param,
        hdu=hdu,
        keys=keys,
        working_dtype=working_dtype,
        recursive=recursive,
    )

    num_images = int(stacked_data.shape[0])

    if mask is not None:
        stacked_data = _apply_mask(stacked_data, mask)

    weights_array: np.ndarray | None = None
    if weights is not None:
        if isinstance(weights, str):
            if weights == "exptime":
                if not file_list:
                    raise ImageProcessingError("Cannot use 'exptime' weights with in-memory arrays")
                exp_weights: list[float] = []
                for filepath in file_list:
                    hdr = get_header(filepath, hdu=hdu)
                    exptime = hdr.get(FITS_KEY_EXPTIME)
                    if exptime is None:
                        raise ImageProcessingError(f"No {FITS_KEY_EXPTIME} keyword in {filepath}")
                    exp_weights.append(float(exptime))
                weights_array = np.array(exp_weights, dtype=np.float64)

            elif weights == "inverse_variance":
                variances = np.array(
                    [np.ma.var(stacked_data[i]) for i in range(stacked_data.shape[0])],
                    dtype=np.float64,
                )
                variances = np.clip(variances, 1e-12, None)
                weights_array = 1.0 / variances

            else:
                raise ValueError(
                    f"Unknown weight type {weights!r}. Use 'exptime', 'inverse_variance', or provide explicit weights."
                )
        else:
            weights_array = np.asarray(weights, dtype=np.float64)

    if method == "sigma_clipped_median":
        combined, num_clipped = _combine_sigma_clipped_median(stacked_data, sigma=sigma, maxiters=maxiters)
    elif method == "sigma_clipped_mean":
        combined, num_clipped = _combine_sigma_clipped_mean(stacked_data, sigma=sigma, maxiters=maxiters)
    elif method == "median":
        combined, num_clipped = _combine_median(stacked_data)
    elif method in ("mean", "weighted_mean"):
        combined, num_clipped = _combine_mean(stacked_data, weights=weights_array)
    else:  # pragma: no cover
        raise ValueError(f"Unhandled method: {method!r}")

    total_pixels = int(np.prod(stacked_data.shape))
    clip_fraction = (int(num_clipped) / total_pixels) if total_pixels else 0.0

    new_header = base_header.copy()
    add_history(new_header, f"Combined {num_images} images using {method}")
    new_header["NCOMBINE"] = (num_images, "Number of images combined")
    new_header["COMBMETH"] = (method, "Combining method")

    if method.startswith("sigma_clipped"):
        new_header["SIGMA"] = (float(sigma), "Sigma threshold for clipping")
        new_header["MAXITERS"] = (int(maxiters), "Maximum clipping iterations")
        new_header["NCLIPPED"] = (int(num_clipped), "Number of clipped pixels")

    if weights_array is not None:
        new_header["WEIGHTED"] = (True, "Weighted combining used")
        if isinstance(weights, str):
            new_header["WEIGHTBY"] = (weights, "Weighting method")

    output_path: Path | None = None
    if output_file is not None:
        output_path = Path(output_file)
        write_simple_fits(output_path, combined, header=new_header, overwrite=overwrite)
        logger.debug("Combined image written to: %s", output_path)

    result = CombiningResult(
        data=combined,
        header=new_header,
        method=method,
        num_images=num_images,
        num_clipped=int(num_clipped),
        clip_fraction=float(clip_fraction),
        input_files=file_list if file_list else None,
        output_file=output_path,
    )

    if return_type == "array":
        return combined
    if return_type == "path":
        if output_path is None:
            raise ValueError("return_type='path' requires output_file to be specified")
        return output_path
    if return_type in ("result", "auto"):
        return result

    raise ValueError("return_type must be 'array', 'path', 'result', or 'auto'")


def combine_image_hdus(
    input_files: list[StrOrPath],
    output_file: StrOrPath | None,
    *,
    selectors: Iterable[HDUSelector] | None = None,
    query: HDUQuery | None = None,
    method: str = "sigma_clipped_median",
    output_mode: OutputMode = "preserve",
    structure_policy: StructurePolicy = "compatible",
    overwrite: bool = False,
    **kwargs: Any,
) -> list[CombiningResult]:
    """Combine several logical HDUs and commit all products atomically.

    Selected identities are discovered from the first input and then resolved
    independently in every file by :func:`combine_images_robust`. Query-result
    positions are never used to associate extensions across files.
    """
    if (selectors is None) == (query is None):
        raise ValueError("Provide exactly one of selectors or query")
    if output_mode == "sef":
        raise ValueError("combine_image_hdus does not support SEF output")
    if not input_files:
        raise ValueError("input_files must not be empty")
    reserved = {"hdu", "output_file", "return_type", "overwrite", "method"}
    conflicts = reserved.intersection(kwargs)
    if conflicts:
        raise ValueError(
            f"Reserved combine arguments must use explicit parameters: "
            f"{sorted(conflicts)}"
        )

    paths = [Path(path) for path in input_files]
    first_path = paths[0]
    with fits.open(first_path, mode="readonly", memmap=False) as hdul:
        indices = (
            resolve_hdu_indices(
                hdul,
                selectors,
                image_like=True,
                require_data=True,
                filepath=first_path,
            )
            if selectors is not None
            else find_hdu_indices(hdul, query)
        )

    sources = [load_resolved_hdu(first_path, index) for index in indices]
    resolved_selectors = [identity_selector(source) for source in sources]
    results: list[CombiningResult] = []
    payloads: list[HDUResult] = []

    for source, selector in zip(
        sources, resolved_selectors, strict=True
    ):
        result = combine_images_robust(
            paths,
            method=method,
            output_file=None,
            hdu=selector,
            overwrite=overwrite,
            return_type="result",
            **kwargs,
        )
        assert isinstance(result, CombiningResult)
        results.append(result)
        payloads.append(HDUResult(source, result.data, result.header))

    commit_hdu_results(
        payloads,
        output_file,
        output_mode=output_mode,
        structure_policy=structure_policy,
        overwrite=overwrite,
    )
    output_path = first_path if output_mode == "inplace" else Path(output_file)
    for result in results:
        result.output_file = output_path
    return results


def create_master_frame(
    frame_list: str | list[StrOrPath] | list[np.ndarray],
    frame_type: str,
    method: str = "sigma_clipped_median",
    normalize: bool = False,
    working_dtype: np.dtype = np.float64,
    return_type: str = "auto",
    **kwargs: Any,
) -> np.ndarray | Path | CombiningResult:
    """Create a master calibration frame (bias/dark/flat) from multiple exposures."""
    valid_types = {"bias", "dark", "flat"}
    if frame_type.lower() not in valid_types:
        raise ValueError(f"Unknown frame_type: {frame_type!r}. Valid: {sorted(valid_types)}")

    result = combine_images_robust(
        frame_list,
        method=method,
        working_dtype=working_dtype,
        return_type="result",
        **kwargs,
    )
    assert isinstance(result, CombiningResult)

    add_history(result.header, f"Master {frame_type.lower()} frame created")
    result.header["FRMTYPE"] = (frame_type.lower(), "Calibration frame type")

    if normalize:
        median_val = float(np.median(result.data))
        if median_val == 0.0:
            logger.warning("Cannot normalize %s frame: median is zero.", frame_type)
        else:
            result.data = result.data / median_val
            add_history(result.header, f"Normalized by median ({median_val:.6g})")
            result.header["NORMLIZD"] = (True, "Frame has been normalized")
            result.header["NORMVAL"] = (median_val, "Normalization value (median)")

            if result.output_file is not None:
                write_simple_fits(result.output_file, result.data, header=result.header, overwrite=True)

    if return_type == "array":
        return result.data
    if return_type == "path":
        if result.output_file is None:
            raise ValueError("return_type='path' requires output_file to be specified in kwargs")
        return result.output_file
    if return_type in ("result", "auto"):
        return result

    raise ValueError("return_type must be 'array', 'path', 'result', or 'auto'")


def median_combine(
    input_param: str | list[StrOrPath] | list[np.ndarray],
    filebase: StrOrPath | None = None,
    **kwargs: Any,
) -> np.ndarray | Path:
    """Convenience wrapper: sigma-clipped median combine."""
    output_file: Path | None = None
    if filebase is not None:
        fb = Path(filebase)
        output_file = fb if fb.suffix.lower() == ".fits" else fb.with_suffix(".fits")

    # Pop header before forwarding kwargs — combine_images_robust does not
    # accept a header parameter. If provided, it is written to the output
    # file after combination.
    header = kwargs.pop("header", None)

    out = combine_images_robust(
        input_param,
        method="sigma_clipped_median",
        output_file=output_file,
        return_type="array" if output_file is None else "path",
        **kwargs,
    )

    # If a header was supplied and an output file was written, update it.
    if header is not None and output_file is not None and Path(output_file).exists():
        from astropy.io import fits as _fits
        with _fits.open(output_file, mode="update") as hdul:
            hdul[0].header.update(header)

    return out


def mean_combine(
    input_param: str | list[StrOrPath] | list[np.ndarray],
    filebase: StrOrPath | None = None,
    **kwargs: Any,
) -> np.ndarray | Path:
    """Convenience wrapper: sigma-clipped mean combine."""
    output_file: Path | None = None
    if filebase is not None:
        fb = Path(filebase)
        output_file = fb if fb.suffix.lower() == ".fits" else fb.with_suffix(".fits")

    out = combine_images_robust(
        input_param,
        method="sigma_clipped_mean",
        output_file=output_file,
        return_type="array" if output_file is None else "path",
        **kwargs,
    )
    return out  # type: ignore[return-value]


def combine_images(
    input_param: str | list[StrOrPath] | list[np.ndarray],
    operations: str | list[str],
    filebase: StrOrPath | None = None,
    working_dtype: np.dtype = np.float64,
    **kwargs: Any,
) -> Any:
    """Deprecated legacy combining API.

    .. deprecated:: 2.0.0
       Use :func:`combine_images_robust` for new code.
    """
    warnings.warn(
        "combine_images() is deprecated and will be removed in version 3.0.0. "
        "Use combine_images_robust() for better outlier rejection.",
        DeprecationWarning,
        stacklevel=2,
    )

    hdu = kwargs.get("hdu", kwargs.get("ext", 0))
    overwrite = kwargs.get("overwrite", False)
    output_dir = kwargs.get("output_dir", None)
    header_template = kwargs.get("header", None)

    base_header = fits.Header()

    if isinstance(input_param, list) and input_param and all(isinstance(i, np.ndarray) for i in input_param):
        try:
            stacked_data, _, base_header = _load_and_stack_efficient(
                input_param,
                hdu=hdu,
                keys=kwargs.get("keys"),
                working_dtype=working_dtype,
                recursive=kwargs.get("recursive", False),
            )
        except Exception as e:
            logger.error("Failed to stack arrays: %s", e)
            return None

        if header_template:
            base_header = header_template.copy()
        add_history(base_header, f"Combined from {len(input_param)} in-memory arrays")

    else:
        if isinstance(input_param, str):
            try:
                files_to_process = select_files(
                    input_param,
                    keys=kwargs.get("keys"),
                    recursive=kwargs.get("recursive", False),
                )
            except Exception as e:
                logger.error("File selection failed: %s", e)
                return None
        else:
            files_to_process = input_param

        if not files_to_process:
            logger.warning("No files found or selected for combination")
            return None

        max_files = kwargs.get("max_files")
        if max_files and len(files_to_process) > max_files:
            files_to_process = files_to_process[:max_files]

        try:
            stacked_data, _, base_header = _load_and_stack_efficient(
                files_to_process,
                hdu=hdu,
                keys=kwargs.get("keys"),
                working_dtype=working_dtype,
                recursive=kwargs.get("recursive", False),
            )
        except Exception as e:
            logger.error("Failed to load and stack files: %s", e)
            return None

        if header_template:
            base_header = header_template.copy()

        add_history(base_header, f"Combined from {len(files_to_process)} file(s)")
        base_header["NCOMBINE"] = (len(files_to_process), "Number of files combined")

    def _combine_simple(data_stack: np.ndarray, operation: str) -> np.ndarray:
        if operation == "median":
            return np.median(data_stack, axis=0)
        if operation == "mean":
            return np.mean(data_stack, axis=0)
        raise ValueError(f"Unsupported operation: {operation}")

    if filebase is None and output_dir is None:
        try:
            if isinstance(operations, str):
                return _combine_simple(stacked_data, operations)
            return {op: _combine_simple(stacked_data, op) for op in operations}
        except Exception as e:
            logger.error("Error during in-memory combination: %s", e)
            return None

    if output_dir is None:
        output_dir = Path(filebase).parent if filebase else Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ops = [operations] if isinstance(operations, str) else list(operations)
    results: dict[str, Any] = {}

    for op in ops:
        current_filebase = f"{filebase}_{op}" if len(ops) > 1 else filebase
        output_path = output_dir / f"{current_filebase}.fits"

        if output_path.exists() and not overwrite:
            logger.warning("File exists and overwrite=False: %s", output_path)
            results[op] = output_path
            continue

        try:
            combined_array = _combine_simple(stacked_data, op)

            op_header = base_header.copy()
            add_history(op_header, f"Combined using '{op}' operation")
            op_header["COMBMETH"] = (op, "Combining method")

            write_simple_fits(output_path, combined_array, header=op_header, overwrite=overwrite)
            results[op] = output_path

        except Exception as e:
            logger.error("Failed to create combined image for operation %r: %s", op, e)
            results[op] = None

    return results[operations] if isinstance(operations, str) else results


__all__ = [
    "combine_image_hdus",
    "combine_images_robust",
    "create_master_frame",
    "median_combine",
    "mean_combine",
    "combine_images",
]
