#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fits_tools.image_processing.pixel_edit

Pixel-domain FITS editing utilities.

This module contains operations that modify image *pixel arrays* (crop/ROI,
masking/thresholding, setting values), with optional WCS CRPIX adjustment on
crop when ``astropy.wcs`` is available.

It is factored out of the original ``fits_tools.edit`` implementation to keep
pixel edits separate from HDU structure manipulation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from astropy.io import fits

# Optional WCS support for CRPIX shift on crop
try:
    from astropy.wcs import WCS
except Exception:
    WCS = None

# Import from fits_tools package
from fits_tools.constants import (
    FITS_VAL_PRIMARY,
    FITS_FILE_EXTENSIONS,
    FITS_KEY_EXTNAME,
    FITS_KEY_HISTORY,
    FITS_KEY_XTENSION,
    FITS_KEY_EXTVER,
    FITS_KEY_EXTLEVEL,
    FITS_KEY_PCOUNT,
    FITS_KEY_GCOUNT,
)

from fits_tools.exceptions import (
    InvalidParameterError,
    FileOverwriteError,
    RegionError,
    DimensionalityError,
    HDUStructureError,
    InvalidHDUIndexError,
    DataShapeError,
    FileHandlingError,
    fits_operation_context,
)

from fits_tools.utils import (
    setup_logger,
    validate_indices,
    validate_positive,
    make_output_path,
)

from ..utils import add_history

# ---------------------------------------------------------------------
# Module Constants
# ---------------------------------------------------------------------

__version__ = "1.0.0"

# Type aliases for better clarity
ThreshOp = Literal[">", ">=", "<", "<=", "==", "!=", "in", "out"]
FillValue = Union[str, float]


# ---------------------------------------------------------------------
# Utilities & helpers
# ---------------------------------------------------------------------

def _ensure_single_or_dir(inputs: List[str], output_path: Optional[str], output_dir: Optional[str]) -> None:
    """Validate output path specification for single or multiple inputs."""
    if len(inputs) == 1 and output_path:
        return
    if output_dir:
        return
    raise InvalidParameterError(
        "For multiple inputs, provide --output-dir; for a single input, provide --output or --output-dir.",
        parameter="output_path/output_dir"
    )

def _derive_out_path(in_path: str, output_path: Optional[str], output_dir: Optional[str], suffix: str) -> str:
    """Derive output path from input path, output specification, and suffix."""
    if output_dir:
        return str(make_output_path(in_path, output_dir=output_dir, suffix=suffix))
    
    assert output_path is not None
    if os.path.isdir(output_path):
        return os.path.join(output_path, os.path.basename(in_path))
    return output_path

def _safe_write(hdul: fits.HDUList, path: str, overwrite: bool, dry_run: bool, operation: str = "edit") -> None:
    """Safely write HDUList with metadata and validation.

    Notes
    -----
    This function is intentionally conservative:
    - Adds an operation HISTORY line to the primary header (when possible).
    - Writes to a temporary file in the destination directory and atomically replaces
      the destination via :func:`os.replace` to avoid corrupting an existing file if
      the write fails partway through.
    """
    if dry_run:
        logging.info(f"[dry-run] Would write: {path}")
        return

    if os.path.exists(path) and not overwrite:
        raise FileOverwriteError(path)

    # Add HISTORY card to document the operation
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    history_msg = f"fits_tools: {operation} at {timestamp} UTC"

    try:
        hdr = hdul[0].header  # will raise if hdul empty, etc.
        add_history(hdr, history_msg)
    except Exception as e:
        except Exception:
            logging.debug(f"Could not add HISTORY card: {e}")

    # Atomic write: write to a temp file in the destination directory, then replace.
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=f".{base}.tmp.", dir=dest_dir)
        os.close(fd)
        # Always overwrite temp path
        hdul.writeto(tmp_path, overwrite=True, checksum=True)
        os.replace(tmp_path, path)
    except Exception as e:
        # Clean up temp file if present; do NOT delete existing destination.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise FileHandlingError(f"Failed to write {path}: {e}", filepath=path) from e

def _slice_bounds_from_xywh(x0: int, y0: int, width: int, height: int) -> Tuple[slice, slice]:
    """Convert (x0, y0, width, height) to numpy slice objects."""
    validate_positive(width, "width")
    validate_positive(height, "height")
    
    xs = slice(int(x0), int(x0 + width))
    ys = slice(int(y0), int(y0 + height))
    return ys, xs

def _xywh_from_center(xc: int, yc: int, half_width: int, half_height: int) -> Tuple[int, int, int, int]:
    """Convert center coordinates and half-dimensions to (x0, y0, width, height)."""
    validate_positive(half_width, "half_width")
    validate_positive(half_height, "half_height")
    
    x0 = int(xc - half_width)
    y0 = int(yc - half_height)
    width = int(2 * half_width)
    height = int(2 * half_height)
    return x0, y0, width, height

def _adjust_wcs_header_on_crop(hdu: fits.ImageHDU, x0: int, y0: int, slices: Tuple[slice, slice]) -> None:
    """
    Adjust WCS on crop. Attempts proper slicing if possible, falls back to CRPIX shift.
    Non-fatal on failure.
    """
    if WCS is None or hdu.header is None:
        return
    
    try:
        wcs = WCS(hdu.header)
        if wcs.naxis < 2:
            logging.debug("WCS has fewer than 2 axes; cannot adjust for crop.")
            return
        
        # Attempt proper WCS slicing (handles distortions, etc.)
        try:
            ys, xs = slices
            # WCS.slice expects (slice1, slice2, ...) for each axis
            cropped_wcs = wcs[ys, xs]
            hdu.header.update(cropped_wcs.to_header())
            logging.debug("WCS adjusted using proper slicing.")
            return
        except Exception as e:
            logging.debug(f"WCS.slice() failed: {e}, falling back to CRPIX shift.")
        
        # Fallback: simple CRPIX shift (doesn't handle distortions properly)
        if "CRPIX1" in hdu.header:
            hdu.header["CRPIX1"] = float(hdu.header["CRPIX1"]) - float(x0)
        if "CRPIX2" in hdu.header:
            hdu.header["CRPIX2"] = float(hdu.header["CRPIX2"]) - float(y0)
        logging.debug("WCS adjusted using CRPIX shift (approximate).")
        
    except Exception as e:
        logging.debug(f"WCS adjust skipped for this HDU: {e}")

def _load_mask_from_fits(mask_path: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Load mask from FITS file, validating shape matches target."""
    try:
        with fits.open(mask_path, memmap=True) as mh:
            m = mh[0].data
            if m is None:
                raise FileHandlingError(
                    "Mask FITS has no data in primary HDU.",
                    filepath=mask_path
                )
            
            # Allow broadcasting for compatible shapes
            if m.shape != shape:
                # Check if last 2 dimensions match (for 3D+ arrays)
                if len(shape) >= 2 and len(m.shape) >= 2:
                    if m.shape[-2:] != shape[-2:]:
                        raise DataShapeError(
                            f"Mask shape {m.shape} incompatible with image shape {shape}.",
                            expected_shape=shape,
                            actual_shape=m.shape,
                            operation="mask loading"
                        )
                else:
                    raise DataShapeError(
                        f"Mask shape {m.shape} does not match image shape {shape}.",
                        expected_shape=shape,
                        actual_shape=m.shape,
                        operation="mask loading"
                    )
            
            return (m != 0).astype(bool)
    except Exception as e:
        if isinstance(e, (FileHandlingError, DataShapeError)):
            raise
        raise FileHandlingError(f"Failed to load mask from {mask_path}: {e}", filepath=mask_path) from e

def _apply_threshold(arr: np.ndarray, op: str, thresh: Union[float, Tuple[float, float]]) -> np.ndarray:
    """Apply threshold operation to array, returning boolean mask."""
    if isinstance(thresh, (list, tuple)) and len(thresh) == 2:
        lo, hi = float(thresh[0]), float(thresh[1])
        if lo > hi:
            raise InvalidParameterError(
                f"Invalid threshold range: lo ({lo}) > hi ({hi}). Use [lo, hi] order.",
                parameter="thresh",
                value=thresh,
                expected="[low, high] with low <= high"
            )
        if op not in ("in", "out"):
            raise InvalidParameterError(
                "For two-value thresholds, use --thresh-op 'in' (inside [lo,hi]) or 'out' (outside).",
                parameter="thresh_op",
                value=op,
                expected="'in' or 'out'"
            )
        return ((arr >= lo) & (arr <= hi)) if op == "in" else ((arr < lo) | (arr > hi))
    else:
        t = float(thresh)
        if op == ">":   return arr > t
        if op == ">=":  return arr >= t
        if op == "<":   return arr < t
        if op == "<=":  return arr <= t
        if op == "==":  return arr == t
        if op == "!=":  return arr != t
        raise InvalidParameterError(f"Unsupported thresh-op: {op}", parameter="thresh_op", value=op)

def _fill_masked(arr: np.ndarray, mask: np.ndarray, fill: FillValue) -> np.ndarray:
    """Fill masked pixels with specified value. Always returns a copy."""
    out = arr.copy()
    
    if fill.lower() == "nan":
        # Convert to float if not already
        if not np.issubdtype(out.dtype, np.floating):
            out = out.astype(np.float64)
        out[mask] = np.nan
    elif fill.lower() == "zero":
        out[mask] = 0
    else:
        try:
            val = float(fill)
        except Exception as e:
            raise InvalidParameterError(
                f"Unrecognized fill value: {fill}",
                parameter="fill",
                value=fill,
                expected="'nan', 'zero', or numeric value"
            ) from e
        out[mask] = val
    
    return out

def _validate_array_dimensionality(arr: np.ndarray, hdu_idx: int, min_dims: int = 2) -> bool:
    """
    Validate array has minimum dimensions required for operation.
    Returns True if valid, False if should skip (with warning logged).
    """
    if arr.ndim < min_dims:
        logging.warning(f"HDU {hdu_idx}: Cannot process {arr.ndim}D array (requires {min_dims}D), skipping.")
        return False
    
    if arr.ndim > 2:
        logging.info(f"HDU {hdu_idx}: Processing {arr.ndim}D array (operating on last 2 dimensions).")
    
    return True

def _op_crop(hdul: fits.HDUList, x0: int, y0: int, width: int, height: int,
             adjust_wcs: bool, dtype: Optional[str]) -> fits.HDUList:
    """
    Crop operation - builds output HDUList incrementally to minimize memory usage.
    """
    ys, xs = _slice_bounds_from_xywh(x0, y0, width, height)
    
    # Build output incrementally instead of copying everything
    out = fits.HDUList()
    
    for i, h in enumerate(hdul):
        if h.data is None:
            # No data - just copy header
            out.append(h.copy())
            continue
        
        arr = h.data
        
        # Validate dimensionality
        if not _validate_array_dimensionality(arr, i, min_dims=2):
            out.append(h.copy())
            continue
        
        # Get dimensions of last 2 axes (height, width)
        H, W = arr.shape[-2], arr.shape[-1]
        
        # Validate crop bounds
        if xs.start < 0 or ys.start < 0 or xs.stop > W or ys.stop > H:
            raise RegionError(
                f"Crop window out of bounds for HDU {i}: "
                f"image {W}×{H}, ROI x[{xs.start}:{xs.stop}] y[{ys.start}:{ys.stop}]",
                region=(x0, y0, width, height),
                image_shape=(H, W),
                operation="crop"
            )
        
        # Perform crop (on last 2 dimensions)
        if arr.ndim == 2:
            cropped = arr[ys, xs].copy()
        else:
            # For 3D+, crop last two dimensions
            cropped = arr[..., ys, xs].copy()
        
        # Apply dtype conversion if requested
        if dtype:
            cropped = cropped.astype(dtype, copy=False)
        
        # Create new HDU with cropped data
        if isinstance(h, fits.PrimaryHDU):
            new_hdu = fits.PrimaryHDU(data=cropped, header=h.header.copy())
        elif isinstance(h, fits.ImageHDU):
            new_hdu = fits.ImageHDU(data=cropped, header=h.header.copy())
        elif isinstance(h, fits.CompImageHDU):
            new_hdu = fits.CompImageHDU(data=cropped, header=h.header.copy())
        else:
            new_hdu = h.copy()
            new_hdu.data = cropped
        
        # Adjust WCS if requested
        if adjust_wcs:
            _adjust_wcs_header_on_crop(new_hdu, x0, y0, (ys, xs))
        
        out.append(new_hdu)
    
    return out

def _op_mask(hdul: fits.HDUList,
             thresh_op: Optional[str],
             thresh: Optional[Union[float, Tuple[float, float]]],
             mask_file: Optional[str],
             fill: FillValue,
             dtype: Optional[str]) -> fits.HDUList:
    """Mask operation - builds output incrementally."""
    out = fits.HDUList()
    
    for i, h in enumerate(hdul):
        if h.data is None:
            out.append(h.copy())
            continue
        
        arr = h.data
        
        # Validate dimensionality
        if not _validate_array_dimensionality(arr, i, min_dims=2):
            out.append(h.copy())
            continue
        
        # Make a copy to avoid modifying original
        arr = arr.copy()
        H, W = arr.shape[-2], arr.shape[-1]
        
        # Build mask
        mask = np.zeros((H, W), dtype=bool)
        
        if mask_file:
            mask |= _load_mask_from_fits(mask_file, arr.shape)
        
        if thresh_op and thresh is not None:
            # For multi-dimensional arrays, apply threshold to last 2 dimensions
            if arr.ndim == 2:
                mask |= _apply_threshold(arr, thresh_op, thresh)
            else:
                # Apply to each plane
                for idx in np.ndindex(arr.shape[:-2]):
                    plane_mask = _apply_threshold(arr[idx], thresh_op, thresh)
                    mask |= plane_mask
        
        # Apply mask if any pixels are masked
        if mask.any():
            filled = _fill_masked(arr, mask, fill)
        else:
            filled = arr
        
        # Apply dtype conversion if requested
        if dtype:
            filled = filled.astype(dtype, copy=False)
        
        # Create new HDU
        if isinstance(h, fits.PrimaryHDU):
            new_hdu = fits.PrimaryHDU(data=filled, header=h.header.copy())
        elif isinstance(h, fits.ImageHDU):
            new_hdu = fits.ImageHDU(data=filled, header=h.header.copy())
        elif isinstance(h, fits.CompImageHDU):
            new_hdu = fits.CompImageHDU(data=filled, header=h.header.copy())
        else:
            new_hdu = h.copy()
            new_hdu.data = filled
        
        out.append(new_hdu)
    
    return out

def _op_set_value(hdul: fits.HDUList, value: float,
                  thresh_op: Optional[str],
                  thresh: Optional[Union[float, Tuple[float, float]]],
                  mask_file: Optional[str],
                  dtype: Optional[str]) -> fits.HDUList:
    """
    Set pixels to a constant. If no mask/threshold provided, sets the entire image.
    If mask/threshold provided, only sets those pixels.
    """
    out = fits.HDUList()
    
    for i, h in enumerate(hdul):
        if h.data is None:
            out.append(h.copy())
            continue
        
        arr = h.data
        
        # Validate dimensionality
        if not _validate_array_dimensionality(arr, i, min_dims=2):
            out.append(h.copy())
            continue
        
        H, W = arr.shape[-2], arr.shape[-1]
        
        if mask_file or (thresh_op and thresh is not None):
            # Selective setting with mask
            arr = arr.copy()
            mask = np.zeros((H, W), dtype=bool)
            
            if mask_file:
                mask |= _load_mask_from_fits(mask_file, arr.shape)
            
            if thresh_op and thresh is not None:
                if arr.ndim == 2:
                    mask |= _apply_threshold(arr, thresh_op, thresh)
                else:
                    for idx in np.ndindex(arr.shape[:-2]):
                        plane_mask = _apply_threshold(arr[idx], thresh_op, thresh)
                        mask |= plane_mask
            
            arr[mask] = float(value)
        else:
            # Set entire array
            arr = np.full_like(arr, float(value))
        
        # Apply dtype conversion if requested
        if dtype:
            arr = arr.astype(dtype, copy=False)
        
        # Create new HDU
        if isinstance(h, fits.PrimaryHDU):
            new_hdu = fits.PrimaryHDU(data=arr, header=h.header.copy())
        elif isinstance(h, fits.ImageHDU):
            new_hdu = fits.ImageHDU(data=arr, header=h.header.copy())
        elif isinstance(h, fits.CompImageHDU):
            new_hdu = fits.CompImageHDU(data=arr, header=h.header.copy())
        else:
            new_hdu = h.copy()
            new_hdu.data = arr
        
        out.append(new_hdu)
    
    return out

