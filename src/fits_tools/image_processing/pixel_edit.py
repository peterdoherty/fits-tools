#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Pixel-domain FITS editing utilities.

This module contains operations that modify image pixel arrays — crop/ROI,
masking/thresholding, and setting values — with optional WCS CRPIX
adjustment on crop when :mod:`astropy.wcs` is available.

The operations here are primarily intended to be driven by higher-level
front-ends (CLI or library wrappers) and operate on entire HDULists in a
conservative, batch-friendly way:

- Work HDU-by-HDU to avoid loading everything into memory at once.
- Preserve non-image HDUs verbatim.
- Optionally adjust WCS information when cropping.
- Compatible with atomic writes via :func:`fits_tools.io.safe_write`.

Examples
--------
Crop a rectangular ROI from all image HDUs::

    from astropy.io import fits
    from fits_tools.image_processing.pixel_edit import op_crop
    from fits_tools.io import safe_write

    with fits.open("raw.fits", memmap=True) as hdul:
        cropped = op_crop(hdul, x0=100, y0=100, width=200, height=200,
                          adjust_wcs=True, dtype="float32")
    safe_write(cropped, "raw_crop.fits", overwrite=True,
               dry_run=False, operation="crop")

Threshold and mask pixels::

    from fits_tools.image_processing.pixel_edit import op_mask

    with fits.open("image.fits", memmap=True) as hdul:
        masked = op_mask(hdul, thresh_op=">", thresh=50000.0,
                         mask_file="badmask.fits", fill="zero", dtype=None)
    safe_write(masked, "image_masked.fits", overwrite=True,
               dry_run=False, operation="mask")

Mark out-of-range pixels as NaN::

    with fits.open("image.fits", memmap=True) as hdul:
        result = op_mask(hdul, thresh_op="out", thresh=(0.0, 30000.0),
                         mask_file=None, fill="nan", dtype="float32")
    safe_write(result, "image_nan.fits", overwrite=True,
               dry_run=False, operation="nan_mask")
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from astropy.io import fits

try:
    from astropy.wcs import WCS
except ImportError:
    WCS = None  # type: ignore[assignment,misc]

from ..array_validation import validate_positive
from ..exceptions import (
    DataShapeError,
    FileHandlingError,
    InvalidParameterError,
    RegionError,
)
from ..types import StrOrPath

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ThreshOp = Literal[">", ">=", "<", "<=", "==", "!=", "in", "out"]
FillValue = str | float

__all__ = [
    "ThreshOp",
    "FillValue",
    "op_crop",
    "op_mask",
    "op_set_value",
]


# ---------------------------------------------------------------------------
# Private geometry helpers
# ---------------------------------------------------------------------------


def _slice_bounds_from_xywh(
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> tuple[slice, slice]:
    """Convert an (x0, y0, width, height) ROI into NumPy slice objects.

    Parameters
    ----------
    x0 : int
        X-coordinate of the left edge (0-based).
    y0 : int
        Y-coordinate of the top edge (0-based).
    width : int
        ROI width in pixels (must be > 0).
    height : int
        ROI height in pixels (must be > 0).

    Returns
    -------
    tuple of slice
        ``(ys, xs)`` in (row, column) order for ``arr[ys, xs]``.

    Raises
    ------
    InvalidParameterError
        If *width* or *height* is not positive.
    """
    validate_positive(width, "width")
    validate_positive(height, "height")
    xs = slice(int(x0), int(x0 + width))
    ys = slice(int(y0), int(y0 + height))
    return ys, xs


def _xywh_from_center(
    xc: int,
    yc: int,
    half_width: int,
    half_height: int,
) -> tuple[int, int, int, int]:
    """Convert center coordinates and half-sizes to (x0, y0, width, height).

    Parameters
    ----------
    xc : int
        X-coordinate of the ROI center (0-based).
    yc : int
        Y-coordinate of the ROI center (0-based).
    half_width : int
        Half of the ROI width in pixels (must be > 0).
    half_height : int
        Half of the ROI height in pixels (must be > 0).

    Returns
    -------
    tuple of int
        ``(x0, y0, width, height)`` suitable for :func:`_slice_bounds_from_xywh`.

    Raises
    ------
    InvalidParameterError
        If *half_width* or *half_height* is not positive.
    """
    validate_positive(half_width, "half_width")
    validate_positive(half_height, "half_height")
    return (
        int(xc - half_width),
        int(yc - half_height),
        int(2 * half_width),
        int(2 * half_height),
    )


def _adjust_wcs_header_on_crop(
    hdu: fits.ImageHDU,
    x0: int,
    y0: int,
    slices: tuple[slice, slice],
) -> None:
    """Adjust an image HDU's WCS to account for a crop operation.

    When :mod:`astropy.wcs` is available, attempts to update the WCS
    in one of two ways:

    1. **Preferred** — ``WCS.__getitem__`` slicing, which handles
       distortions and higher-order effects.
    2. **Fallback** — shift CRPIX1/CRPIX2 by the crop origin. This
       is an approximation valid only for simple, undistorted WCS.

    All failures are non-fatal; the WCS is left unchanged and a debug
    message is logged.

    Parameters
    ----------
    hdu : fits.ImageHDU
        HDU containing the cropped data and header to update in place.
    x0 : int
        X-origin of the crop in the original image (0-based).
    y0 : int
        Y-origin of the crop in the original image (0-based).
    slices : tuple of slice
        ``(ys, xs)`` used to crop the original image.
    """
    if WCS is None or hdu.header is None:
        return

    try:
        wcs = WCS(hdu.header)
        if wcs.naxis < 2:
            logger.debug("WCS has fewer than 2 axes; skipping crop adjustment.")
            return

        try:
            ys, xs = slices
            cropped_wcs = wcs[ys, xs]
            hdu.header.update(cropped_wcs.to_header())
            logger.debug("WCS adjusted via slicing.")
            return
        except Exception as e:
            logger.debug("WCS.slice() failed (%s); falling back to CRPIX shift.", e)

        # Fallback: simple CRPIX shift (approximate; ignores distortions)
        if "CRPIX1" in hdu.header:
            hdu.header["CRPIX1"] = float(hdu.header["CRPIX1"]) - float(x0)
        if "CRPIX2" in hdu.header:
            hdu.header["CRPIX2"] = float(hdu.header["CRPIX2"]) - float(y0)
        logger.debug("WCS adjusted via CRPIX shift (approximate).")

    except Exception as e:
        logger.debug("WCS adjustment skipped for this HDU: %s", e)


def _load_mask_from_fits(
    mask_path: StrOrPath,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Load a boolean mask from a FITS file.

    The mask is read from the primary HDU of *mask_path*. Non-zero values
    are treated as True (bad pixel).

    Shape compatibility:

    - Exact match to *shape* is always accepted.
    - For N-D images (N ≥ 2), a mask whose last two axes match those of
      *shape* is also accepted, enabling a 2-D mask to be used with a
      3-D data cube.

    Parameters
    ----------
    mask_path : str or Path
        Path to the FITS mask file.
    shape : tuple of int
        Shape of the target image array.

    Returns
    -------
    np.ndarray of bool
        Boolean mask compatible with *shape*.

    Raises
    ------
    FileHandlingError
        If the file cannot be read or its primary HDU has no data.
    DataShapeError
        If the mask shape is incompatible with *shape*.
    """
    try:
        with fits.open(mask_path, memmap=True) as mh:
            m = mh[0].data
            if m is None:
                raise FileHandlingError(
                    "Mask FITS has no data in primary HDU.",
                    filepath=mask_path,
                )

            if m.shape != shape:
                if len(shape) >= 2 and len(m.shape) >= 2:
                    if m.shape[-2:] != shape[-2:]:
                        raise DataShapeError(
                            f"Mask shape {m.shape} incompatible with "
                            f"image shape {shape}.",
                            expected_shape=shape,
                            actual_shape=m.shape,
                            operation="mask loading",
                        )
                else:
                    raise DataShapeError(
                        f"Mask shape {m.shape} does not match "
                        f"image shape {shape}.",
                        expected_shape=shape,
                        actual_shape=m.shape,
                        operation="mask loading",
                    )

            return (m != 0).astype(bool)

    except (FileHandlingError, DataShapeError):
        raise
    except Exception as e:
        raise FileHandlingError(
            f"Failed to load mask from {mask_path}: {e}",
            filepath=mask_path,
        ) from e


def _apply_threshold(
    arr: np.ndarray,
    op: str,
    thresh: float | tuple[float, float],
) -> np.ndarray:
    """Apply a threshold operation and return a boolean mask.

    Two modes are supported:

    **Single threshold** (scalar *thresh*):
    ``">"``, ``">="``, ``"<"``, ``"<="``, ``"=="``, ``"!="``.

    **Range threshold** (2-element *thresh* = ``(lo, hi)``):
    ``"in"``  — values in the inclusive range ``[lo, hi]``.
    ``"out"`` — values outside the inclusive range ``[lo, hi]``.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    op : str
        Threshold operator.
    thresh : float or tuple of float
        Threshold value(s). For range operators, provide ``(low, high)``
        with ``low <= high``.

    Returns
    -------
    np.ndarray of bool
        True where pixels satisfy the threshold condition.

    Raises
    ------
    InvalidParameterError
        If *op* is unsupported, or inconsistent with the number of
        threshold values, or if a range is invalid (lo > hi).
    """
    if isinstance(thresh, (list, tuple)) and len(thresh) == 2:
        lo, hi = float(thresh[0]), float(thresh[1])
        if lo > hi:
            raise InvalidParameterError(
                f"Invalid threshold range: lo ({lo}) > hi ({hi}). "
                "Provide [lo, hi] with lo <= hi.",
                parameter="thresh",
                value=thresh,
                expected="[low, high] with low <= high",
            )
        if op not in ("in", "out"):
            raise InvalidParameterError(
                "Use thresh_op='in' (inside [lo,hi]) or 'out' (outside) "
                "for two-value thresholds.",
                parameter="thresh_op",
                value=op,
                expected="'in' or 'out'",
            )
        return (
            ((arr >= lo) & (arr <= hi))
            if op == "in"
            else ((arr < lo) | (arr > hi))
        )

    t = float(thresh)
    if op == ">":
        return arr > t
    if op == ">=":
        return arr >= t
    if op == "<":
        return arr < t
    if op == "<=":
        return arr <= t
    if op == "==":
        return arr == t
    if op == "!=":
        return arr != t
    raise InvalidParameterError(
        f"Unsupported thresh_op: {op!r}.",
        parameter="thresh_op",
        value=op,
    )


def _build_spatial_mask(
    arr: np.ndarray,
    mask_file: StrOrPath | None,
    thresh_op: str | None,
    thresh: float | tuple[float, float] | None,
) -> np.ndarray:
    """Build a 2-D boolean spatial mask for the last two axes of *arr*.

    Combines an optional external mask file and an optional threshold
    applied to the image data via logical OR. Returns an all-False mask
    when neither source is provided.

    Parameters
    ----------
    arr : np.ndarray
        Image array (at least 2-D). Threshold is applied per plane on
        the last two axes for N > 2.
    mask_file : str, Path, or None
        Path to an external FITS mask file. None disables file masking.
    thresh_op : str or None
        Threshold operator; see :func:`_apply_threshold`. None disables
        threshold masking.
    thresh : float, tuple of float, or None
        Threshold value(s). Ignored when *thresh_op* is None.

    Returns
    -------
    np.ndarray of bool
        2-D boolean mask with shape ``arr.shape[-2:]``. True = pixel
        selected for masking or replacement. Callers are responsible for
        broadcasting to the full array shape when needed.

    Raises
    ------
    FileHandlingError
        If *mask_file* cannot be read.
    DataShapeError
        If the mask file shape is incompatible with *arr*.
    InvalidParameterError
        If *thresh_op* / *thresh* are inconsistent or invalid.
    """
    height, width = arr.shape[-2], arr.shape[-1]
    mask = np.zeros((height, width), dtype=bool)

    if mask_file is not None:
        mask |= _load_mask_from_fits(mask_file, arr.shape)

    if thresh_op is not None and thresh is not None:
        if arr.ndim == 2:
            mask |= _apply_threshold(arr, thresh_op, thresh)
        else:
            for idx in np.ndindex(arr.shape[:-2]):
                mask |= _apply_threshold(arr[idx], thresh_op, thresh)

    return mask


def _fill_masked(
    arr: np.ndarray,
    mask: np.ndarray,
    fill: FillValue,
) -> np.ndarray:
    """Fill masked pixels with a special or numeric value.

    A copy of *arr* is always returned; the input is never modified.

    Supported *fill* values:

    - ``"nan"``  — NaN (array cast to float32 if necessary).
    - ``"zero"`` — 0 (dtype preserved).
    - Any other string is parsed as a numeric literal.
    - A numeric value is used directly.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    mask : np.ndarray of bool
        Same shape as *arr*. True = pixel to replace.
    fill : str or float
        Fill specification (case-insensitive for string values).

    Returns
    -------
    np.ndarray
        Copy of *arr* with masked pixels replaced.

    Raises
    ------
    InvalidParameterError
        If *fill* is a string that cannot be parsed as a number and is
        not ``"nan"`` or ``"zero"``.
    """
    out = arr.copy()

    if isinstance(fill, (int, float)):
        out[mask] = float(fill)
        return out

    fill_lower = fill.lower()
    if fill_lower == "nan":
        if not np.issubdtype(out.dtype, np.floating):
            out = out.astype(np.float32)
        out[mask] = np.nan
    elif fill_lower == "zero":
        out[mask] = 0
    else:
        try:
            out[mask] = float(fill)
        except (ValueError, TypeError) as e:
            raise InvalidParameterError(
                f"Unrecognized fill value: {fill!r}.",
                parameter="fill",
                value=fill,
                expected="'nan', 'zero', or a numeric value",
            ) from e

    return out


def _validate_array_dimensionality(
    arr: np.ndarray,
    hdu_idx: int,
    min_dims: int = 2,
) -> bool:
    """Return True if *arr* has at least *min_dims* dimensions.

    Parameters
    ----------
    arr : np.ndarray
        Array from the current HDU.
    hdu_idx : int
        HDU index, used in log messages.
    min_dims : int, optional
        Minimum required dimensions. Default is 2.

    Returns
    -------
    bool
        True if the array should be processed; False if it should be
        skipped.
    """
    if arr.ndim < min_dims:
        logger.warning(
            "HDU %d: skipping %dD array (requires >= %dD).",
            hdu_idx, arr.ndim, min_dims,
        )
        return False
    if arr.ndim > 2:
        logger.debug(
            "HDU %d: processing %dD array on last 2 dimensions.",
            hdu_idx, arr.ndim,
        )
    return True


def _clone_hdu_with_data(
    hdu: fits.hdu.base._BaseHDU,
    data: np.ndarray,
) -> fits.hdu.base._BaseHDU:
    """Create a new HDU of the same type with replacement data.

    Parameters
    ----------
    hdu : fits HDU
        Original HDU whose type and header are preserved.
    data : np.ndarray
        Replacement pixel data.

    Returns
    -------
    fits HDU
        New HDU with *data* and a copy of the original header.

    Raises
    ------
    TypeError
        If *hdu* is not a supported image HDU type.
    """
    header = hdu.header.copy()
    if isinstance(hdu, fits.PrimaryHDU):
        return fits.PrimaryHDU(data=data, header=header)
    if isinstance(hdu, fits.CompImageHDU):
        return fits.CompImageHDU(data=data, header=header)
    if isinstance(hdu, fits.ImageHDU):
        return fits.ImageHDU(data=data, header=header)
    raise TypeError(
        f"Cannot clone HDU of type {type(hdu).__name__!r} with replacement "
        "data. Only PrimaryHDU, ImageHDU, and CompImageHDU are supported."
    )


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------


def op_crop(
    hdul: fits.HDUList,
    x0: int,
    y0: int,
    width: int,
    height: int,
    adjust_wcs: bool,
    dtype: str | None,
) -> fits.HDUList:
    """Crop all image HDUs in an HDUList to a rectangular ROI.

    The crop is applied to each HDU containing image data with at least
    two dimensions. Non-image HDUs are copied verbatim. For arrays with
    more than two dimensions, the crop is applied to the last two axes
    (treated as row, column).

    Parameters
    ----------
    hdul : fits.HDUList
        Input HDU list.
    x0 : int
        X-origin of the ROI (left edge, 0-based).
    y0 : int
        Y-origin of the ROI (top edge, 0-based).
    width : int
        ROI width in pixels.
    height : int
        ROI height in pixels.
    adjust_wcs : bool
        If True, attempt to adjust WCS in each cropped image HDU.
    dtype : str or None
        NumPy dtype string to cast cropped arrays to (e.g. ``"float32"``).
        If None, the original dtype is preserved.

    Returns
    -------
    fits.HDUList
        New HDU list with cropped image data.

    Raises
    ------
    RegionError
        If the requested ROI is out of bounds for any image HDU.
    InvalidParameterError
        If *width* or *height* is not positive.
    """
    ys, xs = _slice_bounds_from_xywh(x0, y0, width, height)
    out = fits.HDUList()

    for i, h in enumerate(hdul):
        if h.data is None:
            out.append(h.copy())
            continue

        arr = h.data
        if not _validate_array_dimensionality(arr, i, min_dims=2):
            out.append(h.copy())
            continue

        H, W = arr.shape[-2], arr.shape[-1]

        if xs.start < 0 or ys.start < 0 or xs.stop > W or ys.stop > H:
            raise RegionError(
                f"Crop window out of bounds for HDU {i}: "
                f"image {W}×{H}, "
                f"ROI x[{xs.start}:{xs.stop}] y[{ys.start}:{ys.stop}].",
                region=(x0, y0, width, height),
                image_shape=(H, W),
                operation="crop",
            )

        cropped = arr[..., ys, xs].copy() if arr.ndim > 2 else arr[ys, xs].copy()

        if dtype:
            cropped = cropped.astype(dtype, copy=False)

        new_hdu = _clone_hdu_with_data(h, cropped)

        if adjust_wcs:
            _adjust_wcs_header_on_crop(new_hdu, x0, y0, (ys, xs))

        out.append(new_hdu)

    return out


def op_mask(
    hdul: fits.HDUList,
    thresh_op: str | None,
    thresh: float | tuple[float, float] | None,
    mask_file: StrOrPath | None,
    fill: FillValue,
    dtype: str | None,
) -> fits.HDUList:
    """Apply masking and pixel filling across all image HDUs.

    A boolean mask is built for each image HDU from:

    - An optional external mask FITS file (*mask_file*).
    - An optional threshold applied to the image data (*thresh_op*,
      *thresh*).

    The combined mask (logical OR) is passed to :func:`_fill_masked`.
    Non-image HDUs are copied unchanged.

    For N-D arrays (N > 2), thresholding is applied per plane on the last
    two axes; per-plane masks are OR-ed into a single 2-D spatial mask.

    Parameters
    ----------
    hdul : fits.HDUList
        Input HDU list.
    thresh_op : str or None
        Threshold operator; see :func:`_apply_threshold`. None disables
        threshold masking.
    thresh : float, tuple of float, or None
        Threshold value(s). Ignored when *thresh_op* is None.
    mask_file : str, Path, or None
        Path to an external mask FITS file.
    fill : str or float
        Fill specification; see :func:`_fill_masked`.
    dtype : str or None
        NumPy dtype string for output arrays. None preserves original dtype.

    Returns
    -------
    fits.HDUList
        New HDU list with masked pixels filled.

    Raises
    ------
    FileHandlingError
        If *mask_file* cannot be read.
    DataShapeError
        If the mask file shape is incompatible with an HDU's image.
    InvalidParameterError
        If *thresh_op* / *thresh* are inconsistent or invalid.
    """
    out = fits.HDUList()

    for i, h in enumerate(hdul):
        if h.data is None:
            out.append(h.copy())
            continue

        arr = h.data
        if not _validate_array_dimensionality(arr, i, min_dims=2):
            out.append(h.copy())
            continue

        arr = arr.copy()
        mask = _build_spatial_mask(arr, mask_file, thresh_op, thresh)

        if arr.ndim > 2:
            mask = np.broadcast_to(mask, arr.shape).copy()

        filled = _fill_masked(arr, mask, fill) if mask.any() else arr

        if dtype:
            filled = filled.astype(dtype, copy=False)

        out.append(_clone_hdu_with_data(h, filled))

    return out


def op_set_value(
    hdul: fits.HDUList,
    value: float,
    thresh_op: str | None,
    thresh: float | tuple[float, float] | None,
    mask_file: StrOrPath | None,
    dtype: str | None,
) -> fits.HDUList:
    """Set pixels to a constant value, optionally restricted by a mask.

    Two modes of operation:

    **Global set** (no mask or threshold):
        Every pixel in each image HDU is set to *value*.

    **Selective set** (mask and/or threshold provided):
        A boolean mask is built from *mask_file* and/or *thresh_op* /
        *thresh*; only masked pixels are set to *value*.

    Non-image HDUs are copied unchanged.

    Parameters
    ----------
    hdul : fits.HDUList
        Input HDU list.
    value : float
        Constant value to assign to selected (or all) pixels.
    thresh_op : str or None
        Threshold operator; see :func:`_apply_threshold`. None disables
        threshold masking.
    thresh : float, tuple of float, or None
        Threshold value(s). Ignored when *thresh_op* is None.
    mask_file : str, Path, or None
        Path to an external mask FITS file.
    dtype : str or None
        NumPy dtype string for output arrays. None preserves original dtype.

    Returns
    -------
    fits.HDUList
        New HDU list with selected pixels set to *value*.

    Raises
    ------
    FileHandlingError
        If the mask FITS file cannot be read.
    DataShapeError
        If the mask file shape is incompatible with an HDU's image.
    InvalidParameterError
        If *thresh_op* / *thresh* are inconsistent or invalid.
    """
    out = fits.HDUList()

    for i, h in enumerate(hdul):
        if h.data is None:
            out.append(h.copy())
            continue

        arr = h.data
        if not _validate_array_dimensionality(arr, i, min_dims=2):
            out.append(h.copy())
            continue

        if mask_file is not None or (thresh_op is not None and thresh is not None):
            arr = arr.copy()
            mask = _build_spatial_mask(arr, mask_file, thresh_op, thresh)

            if arr.ndim > 2:
                mask = np.broadcast_to(mask, arr.shape).copy()

            arr[mask] = float(value)
        else:
            arr = np.full_like(arr, float(value))

        if dtype:
            arr = arr.astype(dtype, copy=False)

        out.append(_clone_hdu_with_data(h, arr))

    return out
