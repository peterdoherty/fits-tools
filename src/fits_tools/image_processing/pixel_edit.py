#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pixel-domain FITS editing utilities.

This module contains operations that modify image *pixel arrays* (crop/ROI,
masking/thresholding, setting values), with optional WCS CRPIX adjustment on
crop when ``astropy.wcs`` is available.

The functions here are primarily intended to be driven by higher-level
front-ends (CLI or library wrappers) and operate on entire HDULists in a
conservative, batch-friendly way:

- Work HDU-by-HDU to avoid loading everything into memory at once
- Preserve non-image HDUs where possible
- Optionally adjust WCS information when cropping
- Use atomic writes to avoid corrupting existing files

Quick examples
--------------

Basic crop
^^^^^^^^^^

Crop a rectangular region of interest (ROI) from all image HDUs and write
to a new file (using a higher-level wrapper around ``_op_crop`` and
``_safe_write``):

    >>> from astropy.io import fits
    >>> from fits_tools.image_processing.pixel_edit import _op_crop
    >>> from fits_tools.utils import safe_write
    >>> hdul = fits.open("raw.fits", memmap=True)
    >>> # Crop a 200x200 box starting at (x0, y0) = (100, 100)
    >>> cropped_hdul = _op_crop(
    ...     hdul,
    ...     x0=100, y0=100,
    ...     width=200, height=200,
    ...     adjust_wcs=True,
    ...     dtype="float32",
    ... )
    >>> _safe_write(cropped_hdul, "raw_crop.fits", overwrite=True, dry_run=False, operation="crop")

Threshold + external mask
^^^^^^^^^^^^^^^^^^^^^^^^^

Combine a numerical threshold and an external mask file to fill selected
pixels (e.g. saturated or bad columns) with zero:

    >>> from fits_tools.image_processing.pixel_edit import _op_mask
    >>> from fits_tools.utils import safe_write
    >>> hdul = fits.open("image.fits", memmap=True)
    >>> # Fill pixels where data > 50000 or where the external mask is non-zero
    >>> masked_hdul = _op_mask(
    ...     hdul,
    ...     thresh_op=">",             # threshold operator
    ...     thresh=50000.0,            # threshold value
    ...     mask_file="badmask.fits",  # optional external mask
    ...     fill="zero",               # set masked pixels to 0
    ...     dtype=None,                # preserve original dtype
    ... )
    >>> _safe_write(masked_hdul, "image_masked.fits", overwrite=True, dry_run=False, operation="mask")

Set bad pixels to NaN
^^^^^^^^^^^^^^^^^^^^^

Use a range threshold to mark out-of-range pixels and replace them with
NaN, promoting the data type to floating point if necessary:

    >>> from fits_tools.image_processing.pixel_edit import _op_mask
    >>> from fits_tools.utils import safe_write
    >>> hdul = fits.open("image.fits", memmap=True)
    >>> # Mark values outside [0, 30000] as NaN
    >>> masked_hdul = _op_mask(
    ...     hdul,
    ...     thresh_op="out",           # outside the [lo, hi] range
    ...     thresh=(0.0, 30000.0),     # inclusive range
    ...     mask_file=None,
    ...     fill="nan",                # fill with NaN
    ...     dtype="float32",           # cast to float32 for NaN support
    ... )
    >>> _safe_write(masked_hdul, "image_nan.fits", overwrite=True, dry_run=False, operation="nan_mask")

In normal usage you will typically not call these low-level helpers
directly, but instead use higher-level functions or command-line tools
that wrap and parameterize them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

# Optional WCS support for CRPIX shift on crop
try:
    from astropy.wcs import WCS
except Exception:
    WCS = None

from ..exceptions import (
    InvalidParameterError,
    RegionError,
    DataShapeError,
    FileHandlingError,
)

from ..utils import validate_positive

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Module Constants
# ---------------------------------------------------------------------

__version__ = "2.0.0"

# Type aliases for better clarity
ThreshOp = Literal[">", ">=", "<", "<=", "==", "!=", "in", "out"]
FillValue = Union[str, float]


# ---------------------------------------------------------------------
# Utilities & helpers
# ---------------------------------------------------------------------

def _slice_bounds_from_xywh(x0: int, y0: int, width: int, height: int) -> Tuple[slice, slice]:
    """
    Convert an (x0, y0, width, height) ROI into NumPy slice objects.

    The returned slices are in (y, x) order, suitable for indexing
    image arrays as ``arr[ys, xs]``.

    Parameters
    ----------
    x0 : int
        X-coordinate of the left edge (0-based pixel index).
    y0 : int
        Y-coordinate of the top edge (0-based pixel index).
    width : int
        Width of the region in pixels (must be > 0).
    height : int
        Height of the region in pixels (must be > 0).

    Returns
    -------
    (slice, slice)
        Tuple ``(ys, xs)`` where ``ys`` is the vertical slice and
        ``xs`` is the horizontal slice.

    Raises
    ------
    InvalidParameterError
        If ``width`` or ``height`` is not positive.
    """
    validate_positive(width, "width")
    validate_positive(height, "height")
    
    xs = slice(int(x0), int(x0 + width))
    ys = slice(int(y0), int(y0 + height))
    return ys, xs

def _xywh_from_center(xc: int, yc: int, half_width: int, half_height: int) -> Tuple[int, int, int, int]:
    """
    Convert center coordinates and half-sizes into an (x0, y0, width, height) ROI.

    This is a convenience for specifying a crop region by its center and
    half-width/half-height in pixels.

    Parameters
    ----------
    xc : int
        X-coordinate of the ROI center (0-based pixel index).
    yc : int
        Y-coordinate of the ROI center (0-based pixel index).
    half_width : int
        Half of the ROI width in pixels (must be > 0).
    half_height : int
        Half of the ROI height in pixels (must be > 0).

    Returns
    -------
    (int, int, int, int)
        Tuple ``(x0, y0, width, height)`` suitable for passing to
        :func:`_slice_bounds_from_xywh`.
    """
    validate_positive(half_width, "half_width")
    validate_positive(half_height, "half_height")
    
    x0 = int(xc - half_width)
    y0 = int(yc - half_height)
    width = int(2 * half_width)
    height = int(2 * half_height)
    return x0, y0, width, height

def _adjust_wcs_header_on_crop(hdu: fits.ImageHDU, x0: int, y0: int, slices: Tuple[slice, slice]) -> None:
    """
    Adjust an image HDU's WCS to account for a crop operation.

    When ``astropy.wcs`` is available, this function tries to update the
    WCS in the cropped HDU in one of two ways:

    1. **Preferred**: use ``WCS.__getitem__`` / slicing, which handles
       distortions and higher-order effects.
    2. **Fallback**: if slicing fails, shift CRPIX1/CRPIX2 by the crop
       offset (``x0``, ``y0``). This is an approximation that works for
       simple, undistorted WCS but may be inaccurate for complex WCS.

    All failures are treated as non-fatal; in such cases the WCS is left
    unchanged and a debug message is logged.

    Parameters
    ----------
    hdu : fits.ImageHDU
        HDU containing the cropped image data and header to update.
    x0 : int
        X-origin of the crop in the original image (0-based).
    y0 : int
        Y-origin of the crop in the original image (0-based).
    slices : tuple of slice
        Tuple ``(ys, xs)`` used to crop the original image. Only the
        slices' start/stop values are used for WCS slicing.

    Notes
    -----
    - This function is intentionally best-effort. It never raises on WCS
      issues and will simply log debug information when WCS adjustment
      cannot be performed.
    """
    if WCS is None or hdu.header is None:
        return
    
    try:
        wcs = WCS(hdu.header)
        if wcs.naxis < 2:
            logger.debug("WCS has fewer than 2 axes; cannot adjust for crop.")
            return
        
        # Attempt proper WCS slicing (handles distortions, etc.)
        try:
            ys, xs = slices
            # WCS.slice expects (slice1, slice2, ...) for each axis
            cropped_wcs = wcs[ys, xs]
            hdu.header.update(cropped_wcs.to_header())
            logger.debug("WCS adjusted using proper slicing.")
            return
        except Exception as e:
            logger.debug("WCS.slice() failed: %s, falling back to CRPIX shift.", e)
        
        # Fallback: simple CRPIX shift (doesn't handle distortions properly)
        if "CRPIX1" in hdu.header:
            hdu.header["CRPIX1"] = float(hdu.header["CRPIX1"]) - float(x0)
        if "CRPIX2" in hdu.header:
            hdu.header["CRPIX2"] = float(hdu.header["CRPIX2"]) - float(y0)
        logger.debug("WCS adjusted using CRPIX shift (approximate).")
        
    except Exception as e:
        logger.debug("WCS adjust skipped for this HDU: %s", e)


def _load_mask_from_fits(mask_path: StrOrPath, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Load a boolean mask from a FITS file, validating compatibility with a target shape.

    The mask is read from the primary HDU of ``mask_path``. The result
    is a boolean array where non-zero values are interpreted as ``True``.

    Shape compatibility rules:

    - If the mask shape exactly matches ``shape``, it is accepted.
    - For N-dimensional images (N >= 2), a mask whose *last two axes*
      match the image's last two axes is also accepted. This allows,
      for example, using a 2D mask with a 3D data cube.
    - Any other shape mismatch results in a :class:`DataShapeError`.

    Parameters
    ----------
    mask_path : str or Path
        Path to the FITS file containing the mask data in its primary HDU.
    shape : tuple of int
        Shape of the target image array that the mask will be applied to.

    Returns
    -------
    np.ndarray
        Boolean mask array with a shape compatible with ``shape``.

    Raises
    ------
    FileHandlingError
        If the FITS file cannot be read or has no data in the primary HDU.
    DataShapeError
        If the mask array shape is incompatible with the requested image
        shape.
    """
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
    """
    Apply a threshold operation to an array and return a boolean mask.

    Two usage modes are supported:

    1. **Single threshold** (scalar ``thresh``):
       - ``'>'``, ``'>='``, ``'<'``, ``'<='``, ``'=='``, ``'!='``.

    2. **Range threshold** (two-element sequence ``[lo, hi]`` or ``(lo, hi)``):
       - ``'in'``  : select values in the inclusive range ``[lo, hi]``.
       - ``'out'`` : select values outside the inclusive range ``[lo, hi]``.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    op : str
        Threshold operator. One of ``'>', '>=', '<', '<=', '==', '!=', 'in', 'out'``.
    thresh : float or (float, float)
        Threshold value(s). For range thresholds, must be given as
        ``(low, high)`` with ``low <= high``.

    Returns
    -------
    np.ndarray
        Boolean mask where ``True`` indicates pixels that satisfy the
        threshold condition.

    Raises
    ------
    InvalidParameterError
        If ``op`` is unsupported or inconsistent with the number of
        threshold values, or if the provided range is invalid.
    """
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
    """
    Fill masked pixels in an array with a special or numeric value.

    The input array is never modified in-place; a copy is always returned.

    Supported ``fill`` values:

    - ``"nan"``  : fill masked pixels with NaN (array is cast to float
      if necessary).
    - ``"zero"`` : fill masked pixels with 0 (preserves dtype).
    - Any other string is interpreted as a numeric literal and converted
      to ``float``. A failure to parse raises :class:`InvalidParameterError`.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    mask : np.ndarray
        Boolean mask with the same shape as ``arr``. For N-D arrays
        the caller must broadcast a 2-D spatial mask to the full
        array shape before passing it here. Pixels where
        ``mask == True`` will be replaced.
    fill : {"nan", "zero"}, str, or float
        Fill specification. Strings are case-insensitive. Numeric
        ``float`` values are used directly.

    Returns
    -------
    np.ndarray
        New array with masked pixels replaced according to ``fill``.

    Raises
    ------
    InvalidParameterError
        If ``fill`` is not ``"nan"``, ``"zero"`` or a parseable numeric value.
    """
    out = arr.copy()
    
    # Normalise fill to a string so the branches below can use .lower()
    if isinstance(fill, (int, float)):
        out[mask] = float(fill)
        return out
    
    if fill.lower() == "nan":
        # Convert to float if not already
        if not np.issubdtype(out.dtype, np.floating):
            out = out.astype(np.float32)
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
    Check that an array has at least the minimum dimensionality required.

    Used by pixel-edit operations to skip HDUs that do not contain 2D
    (or higher) image data. Logs a warning when skipping arrays that
    are too low dimensional, and an informational message when
    operating on higher-dimensional cubes.

    Parameters
    ----------
    arr : np.ndarray
        Array from the current HDU.
    hdu_idx : int
        Index of the HDU in the HDUList (for logging).
    min_dims : int, optional
        Minimum number of dimensions required. Defaults to 2.

    Returns
    -------
    bool
        True if the array meets the dimensionality requirement and
        should be processed; False if it should be skipped.
    """
    if arr.ndim < min_dims:
        logger.warning("HDU %d: Cannot process %dD array (requires %dD), skipping.", hdu_idx, arr.ndim, min_dims)
        return False
    
    if arr.ndim > 2:
        logger.debug("HDU %d: Processing %dD array (operating on last 2 dimensions).", hdu_idx, arr.ndim)
    
    return True


def _clone_hdu_with_data(hdu: fits.ImageHDU, data: np.ndarray) -> fits.ImageHDU:
    """
    Create a new HDU of the same type with replacement data and a copied header.

    Parameters
    ----------
    hdu : fits.ImageHDU
        Original HDU whose type and header are preserved.
    data : np.ndarray
        New pixel data for the cloned HDU.

    Returns
    -------
    fits.ImageHDU
        New HDU with the given data and a copy of the original header.
    """
    header = hdu.header.copy()
    if isinstance(hdu, fits.PrimaryHDU):
        return fits.PrimaryHDU(data=data, header=header)
    if isinstance(hdu, fits.CompImageHDU):
        return fits.CompImageHDU(data=data, header=header)
    if isinstance(hdu, fits.ImageHDU):
        return fits.ImageHDU(data=data, header=header)
    # Fallback for unexpected HDU types (BinTableHDU, etc.)
    new_hdu = hdu.copy()
    new_hdu.data = data
    return new_hdu


def _op_crop(hdul: fits.HDUList, x0: int, y0: int, width: int, height: int,
             adjust_wcs: bool, dtype: Optional[str]) -> fits.HDUList:
    """
    Crop all image HDUs in a FITS file to a rectangular region of interest.

    The crop is applied to each HDU that contains image data with at least
    two dimensions; non-image HDUs are copied verbatim. For arrays with
    more than two dimensions, the crop is applied to the last two axes
    (treated as ``(y, x)``).

    WCS handling
    ------------
    If ``adjust_wcs`` is True and ``astropy.wcs`` is available, the WCS of
    each cropped image HDU is adjusted to reflect the new region, either via
    proper WCS slicing or, as a fallback, via a CRPIX shift.

    Parameters
    ----------
    hdul : fits.HDUList
        Input HDU list to crop.
    x0 : int
        X-origin of the ROI (left edge, 0-based).
    y0 : int
        Y-origin of the ROI (top edge, 0-based).
    width : int
        Width of the ROI in pixels.
    height : int
        Height of the ROI in pixels.
    adjust_wcs : bool
        If True, attempt to adjust WCS information in each cropped image HDU.
    dtype : str or None
        Optional NumPy dtype string to cast cropped arrays to (e.g. ``'float32'``);
        if None, the original dtype is preserved.

    Returns
    -------
    fits.HDUList
        New HDU list containing cropped HDUs.

    Raises
    ------
    RegionError
        If the requested crop region is out of bounds for any HDU's image
        dimensions.
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
        new_hdu = _clone_hdu_with_data(h, cropped)
        
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
    """
    Apply masking and filling to image pixels across all HDUs in an HDUList.

    A mask is constructed for each image HDU based on:

    - An optional external mask FITS file (``mask_file``).
    - An optional threshold operator and value(s) (``thresh_op``, ``thresh``)
      applied to the image data itself.

    Pixels selected by the combined mask are replaced according to ``fill``
    using :func:`_fill_masked`. Non-image HDUs are copied unchanged.

    For N-dimensional arrays (N > 2), thresholding is applied plane-by-plane
    over the last two axes; the resulting per-plane masks are OR-ed into a
    2D mask for those axes.

    Parameters
    ----------
    hdul : fits.HDUList
        Input HDU list to process.
    thresh_op : str or None
        Threshold operator for generating a mask from the image data.
        See :func:`_apply_threshold` for supported values. If None, no
        threshold-based mask is applied.
    thresh : float or (float, float) or None
        Threshold value(s) used with ``thresh_op``. Ignored if ``thresh_op``
        is None.
    mask_file : str or None
        Path to an external mask FITS file. If provided, it is loaded once
        per HDU and combined (OR) with any threshold-based mask.
    fill : FillValue
        Fill specification passed to :func:`_fill_masked` (e.g. ``"nan"``,
        ``"zero"``, or a numeric value).
    dtype : str or None
        Optional dtype string to cast the output arrays to.

    Returns
    -------
    fits.HDUList
        New HDU list with masked pixels filled accordingly.

    Raises
    ------
    FileHandlingError
        If ``mask_file`` cannot be read.
    DataShapeError
        If the mask file's array shape is incompatible with an HDU's image.
    InvalidParameterError
        If ``thresh_op`` / ``thresh`` are inconsistent or invalid.
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
        
        # Broadcast 2D mask to match array shape for 3D+ arrays
        if arr.ndim > 2:
            mask = np.broadcast_to(mask, arr.shape).copy()
        
        # Apply mask if any pixels are masked
        if mask.any():
            filled = _fill_masked(arr, mask, fill)
        else:
            filled = arr
        
        # Apply dtype conversion if requested
        if dtype:
            filled = filled.astype(dtype, copy=False)
        
        # Create new HDU
        out.append(_clone_hdu_with_data(h, filled))
    
    return out

def _op_set_value(hdul: fits.HDUList, value: float,
                  thresh_op: Optional[str],
                  thresh: Optional[Union[float, Tuple[float, float]]],
                  mask_file: Optional[str],
                  dtype: Optional[str]) -> fits.HDUList:
    """
    Set pixels to a constant value, optionally restricted by a mask/threshold.

    Two modes of operation:

    1. **Global set** (no mask / threshold):
       - Every pixel in each image HDU is set to ``value``.

    2. **Selective set** (mask and/or threshold provided):
       - A boolean mask is constructed from:
         * An optional external mask FITS file (``mask_file``).
         * An optional threshold operator and value(s) (``thresh_op``, ``thresh``)
           applied to the image data.
       - Only pixels where the combined mask is True are set to ``value``.

    For N-dimensional arrays (N > 2), thresholding is applied per plane on
    the last two axes, and plane masks are OR-ed into a shared 2D mask.

    Non-image HDUs are copied unchanged.

    Parameters
    ----------
    hdul : fits.HDUList
        Input HDU list to process.
    value : float
        Constant value to assign to selected pixels (or all pixels when
        no mask/threshold is provided).
    thresh_op : str or None
        Threshold operator for generating a mask from the image data, as in
        :func:`_apply_threshold`. If None, no threshold-based mask is used.
    thresh : float or (float, float) or None
        Threshold value(s) used with ``thresh_op``.
    mask_file : str or None
        Path to an external mask FITS file. If provided, combined with
        threshold-based masks via logical OR.
    dtype : str or None
        Optional dtype string to cast the output arrays to.

    Returns
    -------
    fits.HDUList
        New HDU list with pixels set to the requested value.

    Raises
    ------
    FileHandlingError
        If the mask FITS file cannot be read.
    DataShapeError
        If the mask file's shape is incompatible with an HDU's image.
    InvalidParameterError
        If ``thresh_op`` / ``thresh`` are inconsistent or invalid.
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
            
            # Broadcast 2D mask to match array shape for 3D+ arrays
            if arr.ndim > 2:
                mask = np.broadcast_to(mask, arr.shape).copy()
            
            arr[mask] = float(value)
        else:
            # Set entire array
            arr = np.full_like(arr, float(value))
        
        # Apply dtype conversion if requested
        if dtype:
            arr = arr.astype(dtype, copy=False)
        
        # Create new HDU
        out.append(_clone_hdu_with_data(h, arr))
    
    return out

__all__ = [
    'ThreshOp',
    'FillValue',
    # NOTE: These use underscore prefixes by convention (internal helpers driven
    # by higher-level front-ends) but are exported here for discoverability.
    '_op_crop',
    '_op_mask',
    '_op_set_value',
]