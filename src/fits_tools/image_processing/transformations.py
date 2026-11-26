#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Image transformations for FITS data.

This module provides geometric and intensity transformations for FITS images,
including cropping, binning, normalization, rotation, and flipping.

Transformation Categories:
    - Geometric: crop, bin, rotate, flip, transpose
    - Intensity: normalize, rescale, invert
    - Resampling: resize, zoom

All transformations:
    - Preserve FITS headers with updates
    - Support both in-memory arrays and file I/O
    - Include proper validation
    - Return TransformationResult objects

Usage Examples:
    Crop image:
        >>> from fits_tools.image_processing.transformations import crop_image
        >>> cropped = crop_image(data, region=(100, 900, 100, 900))

    Bin image:
        >>> from fits_tools.image_processing.transformations import bin_image
        >>> binned = bin_image(data, bin_factor=2, method='mean')

    Normalize:
        >>> from fits_tools.image_processing.transformations import normalize_image
        >>> normalized = normalize_image(data, method='median', target_value=1.0)

Version: 2.0.0
"""

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage

from ..array_validation import validate_numeric
from ..exceptions import ImageProcessingError
from ..fits_output import (
    OutputMode,
    StructurePolicy,
    commit_hdu_result,
)
from ..hdu import HDUSelector
from ..header_utils import add_history

# Import from fits_tools
from ..resolved_hdu import load_resolved_hdu
from ..types import StrOrPath
from .result_types import TransformationResult

logger = logging.getLogger(__name__)

# ============================================================================
# Validation Helpers
# ============================================================================


def _validate_region(
    region: tuple[int, int, int, int], image_shape: tuple[int, int]
) -> None:
    """
    Validate that a region is within image bounds.

    Parameters
    ----------
    region : tuple
        (start_col, end_col, start_row, end_row) in 0-indexed coordinates.
    image_shape : tuple
        (rows, cols) shape of the image.

    Raises
    ------
    ImageProcessingError
        If region is invalid.
    """
    start_col, end_col, start_row, end_row = region
    rows, cols = image_shape

    if start_col < 0 or start_row < 0:
        raise ImageProcessingError(f"Region coordinates cannot be negative: {region}")

    if start_col >= end_col or start_row >= end_row:
        raise ImageProcessingError(f"Region start must be less than end: {region}")

    if end_row > rows or end_col > cols:
        raise ImageProcessingError(
            f"Region {region} exceeds image bounds {image_shape}"
        )


def _validate_array_not_empty(arr: np.ndarray, name: str = "array") -> None:
    """Validate array is not empty."""
    if arr.size == 0:
        raise ImageProcessingError(f"{name} is empty (size=0)")


# ============================================================================
# Geometric Transformations
# ============================================================================


def crop_image(data: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop an image to a specified region.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    region : tuple
        Region as (start_col, end_col, start_row, end_row) in 0-indexed coordinates.
        FITS convention: columns are x-axis, rows are y-axis.

    Returns
    -------
    np.ndarray
        Cropped image.

    Raises
    ------
    ImageProcessingError
        If region is invalid or out of bounds.

    Examples
    --------
    >>> data = np.arange(100).reshape(10, 10)
    >>> cropped = crop_image(data, (2, 7, 3, 8))
    >>> cropped.shape
    (5, 5)
    >>> cropped[0, 0] == data[3, 2]
    True
    """
    _validate_array_not_empty(data, "input image")
    _validate_region(region, data.shape)

    start_col, end_col, start_row, end_row = region
    return data[start_row:end_row, start_col:end_col].copy()


def _adjust_crpix_for_crop(header, start_col: int, start_row: int) -> None:
    """Adjust CRPIX* keywords after a crop (CRPIX-only).

    Updates CRPIX1/CRPIX2 and any alternate WCS variants (e.g., CRPIX1A/CRPIX2A).
    No full WCS recomputation is attempted.
    """
    try:
        for axis, delta in [(1, start_col), (2, start_row)]:
            for key in list(header.keys()):
                if re.fullmatch(rf"CRPIX{axis}[A-Z]?", str(key)):
                    try:
                        header[key] = float(header[key]) - float(delta)
                    except Exception:
                        pass
    except Exception:
        return


def bin_image(
    data: np.ndarray,
    bin_factor: int = 2,
    method: Literal["mean", "sum", "median"] = "mean",
) -> np.ndarray:
    """
    Bin an image by combining adjacent pixels.

    Reduces image dimensions by combining pixels in blocks. Useful for
    increasing signal-to-noise ratio at the cost of spatial resolution.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    bin_factor : int, optional
        Binning factor (e.g., 2 for 2x2 binning). Defaults to 2.
    method : {'mean', 'sum', 'median'}, optional
        Binning method. Defaults to 'mean'.
        - 'mean': Average pixels in each bin
        - 'sum': Sum pixels in each bin
        - 'median': Median of pixels in each bin

    Returns
    -------
    np.ndarray
        Binned image with shape reduced by bin_factor.

    Raises
    ------
    ImageProcessingError
        If bin_factor is invalid or method is unknown.

    Examples
    --------
    >>> data = np.arange(100).reshape(10, 10).astype(float)
    >>> binned = bin_image(data, bin_factor=2, method='mean')
    >>> binned.shape
    (5, 5)
    >>> binned[0, 0]  # Mean of [0, 1, 10, 11]
    5.5

    >>> # Sum method
    >>> binned_sum = bin_image(data, bin_factor=2, method='sum')
    >>> binned_sum[0, 0]  # Sum of [0, 1, 10, 11]
    22.0
    """
    validate_numeric(bin_factor, "bin_factor", minimum=1, integer=True)
    _validate_array_not_empty(data, "input image")

    if bin_factor == 1:
        return data.copy()

    if method not in ("mean", "sum", "median"):
        raise ImageProcessingError(
            f"Unknown binning method: '{method}'. Use 'mean', 'sum', or 'median'."
        )

    ny, nx = data.shape

    # Trim to multiple of bin_factor
    ny_trim = (ny // bin_factor) * bin_factor
    nx_trim = (nx // bin_factor) * bin_factor

    if ny_trim < ny or nx_trim < nx:
        logger.warning(
            f"Image dimensions ({ny}, {nx}) not multiple of bin factor {bin_factor}. "
            f"Trimming to ({ny_trim}, {nx_trim})"
        )
        data = data[:ny_trim, :nx_trim]

    # Reshape and combine
    new_shape = (ny_trim // bin_factor, bin_factor, nx_trim // bin_factor, bin_factor)
    reshaped = data.reshape(new_shape)

    if method == "mean":
        binned = reshaped.mean(axis=(1, 3))
    elif method == "sum":
        binned = reshaped.sum(axis=(1, 3))
    elif method == "median":
        # Reshape to (new_rows, new_cols, bin_factor*bin_factor)
        temp = reshaped.transpose(0, 2, 1, 3).reshape(
            ny_trim // bin_factor, nx_trim // bin_factor, bin_factor * bin_factor
        )
        binned = np.median(temp, axis=2)

    return binned


def rotate_image(
    data: np.ndarray,
    angle: float,
    reshape: bool = True,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Rotate an image by a specified angle.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    angle : float
        Rotation angle in degrees (counter-clockwise).
    reshape : bool, optional
        If True, expand output to fit entire rotated image.
        If False, maintain input dimensions. Defaults to True.
    order : int, optional
        Interpolation order (0=nearest, 1=linear, 3=cubic). Defaults to 1.
    mode : str, optional
        How to handle points outside boundaries. Defaults to 'constant'.
    cval : float, optional
        Fill value for mode='constant'. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        Rotated image.

    Examples
    --------
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    >>> rotated = rotate_image(data, 90)
    >>> rotated.shape
    (3, 3)
    """
    _validate_array_not_empty(data, "input image")

    return ndimage.rotate(
        data, angle, reshape=reshape, order=order, mode=mode, cval=cval
    )


def flip_image(
    data: np.ndarray, axis: Literal["horizontal", "vertical", "both"] = "horizontal"
) -> np.ndarray:
    """
    Flip an image along specified axis.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    axis : {'horizontal', 'vertical', 'both'}, optional
        Flip axis. Defaults to 'horizontal'.
        - 'horizontal': Flip left-right (mirror around vertical axis)
        - 'vertical': Flip top-bottom (mirror around horizontal axis)
        - 'both': Flip both directions (180° rotation)

    Returns
    -------
    np.ndarray
        Flipped image.

    Examples
    --------
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> flipped = flip_image(data, 'horizontal')
    >>> flipped
    array([[3, 2, 1],
           [6, 5, 4],
           [9, 8, 7]])

    >>> flipped = flip_image(data, 'vertical')
    >>> flipped
    array([[7, 8, 9],
           [4, 5, 6],
           [1, 2, 3]])
    """
    _validate_array_not_empty(data, "input image")

    if axis == "horizontal":
        return np.fliplr(data)
    elif axis == "vertical":
        return np.flipud(data)
    elif axis == "both":
        return np.flipud(np.fliplr(data))
    else:
        raise ImageProcessingError(
            f"Unknown flip axis: '{axis}'. Use 'horizontal', 'vertical', or 'both'."
        )


def transpose_image(data: np.ndarray) -> np.ndarray:
    """
    Transpose an image (swap rows and columns).

    Parameters
    ----------
    data : np.ndarray
        Input image data.

    Returns
    -------
    np.ndarray
        Transposed image.

    Examples
    --------
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> transposed = transpose_image(data)
    >>> transposed.shape
    (3, 2)
    >>> transposed
    array([[1, 4],
           [2, 5],
           [3, 6]])
    """
    _validate_array_not_empty(data, "input image")
    return data.T.copy()


# ============================================================================
# Intensity Transformations
# ============================================================================


def normalize_image(
    data: np.ndarray,
    method: Literal["median", "mean", "max", "minmax"] = "median",
    target_value: float = 1.0,
) -> np.ndarray:
    """
    Normalize an image using various methods.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    method : {'median', 'mean', 'max', 'minmax'}, optional
        Normalization method. Defaults to 'median'.
        - 'median': Normalize by median value
        - 'mean': Normalize by mean value
        - 'max': Normalize by maximum value
        - 'minmax': Scale to [0, target_value] range
    target_value : float, optional
        Target value for normalization. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        Normalized image.

    Raises
    ------
    ImageProcessingError
        If method is unknown or normalization factor is zero.

    Examples
    --------
    >>> data = np.ones((10, 10)) * 1000
    >>> normalized = normalize_image(data, method='mean', target_value=1.0)
    >>> np.mean(normalized)
    1.0

    >>> data = np.array([[0, 50, 100], [150, 200, 250]], dtype=float)
    >>> normalized = normalize_image(data, method='minmax', target_value=1.0)
    >>> normalized.min(), normalized.max()
    (0.0, 1.0)
    """
    _validate_array_not_empty(data, "input image")

    if method == "median":
        norm_factor = np.median(data)
    elif method == "mean":
        norm_factor = np.mean(data)
    elif method == "max":
        norm_factor = np.max(data)
    elif method == "minmax":
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max <= data_min:
            logger.warning("Min equals max, returning zeros")
            return np.zeros_like(data, dtype=np.float32)
        return ((data - data_min) / (data_max - data_min)) * target_value
    else:
        raise ImageProcessingError(
            f"Unknown normalization method: '{method}'. "
            f"Use 'median', 'mean', 'max', or 'minmax'."
        )

    if norm_factor == 0:
        logger.warning("Normalization factor is zero. Returning zeros.")
        return np.zeros_like(data, dtype=np.float32)

    return (data / norm_factor) * target_value


def rescale_intensity(
    data: np.ndarray,
    in_range: tuple[float, float] | None = None,
    out_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = True,
) -> np.ndarray:
    """
    Rescale image intensity to a new range.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    in_range : tuple, optional
        (min, max) input intensity range. If None, uses data min/max.
    out_range : tuple, optional
        (min, max) output intensity range. Defaults to (0.0, 1.0).
    clip : bool, optional
        If True, clip values to out_range. Defaults to True.

    Returns
    -------
    np.ndarray
        Rescaled image.

    Examples
    --------
    >>> data = np.array([[0, 50, 100], [150, 200, 250]], dtype=float)
    >>> rescaled = rescale_intensity(data, out_range=(0, 255))
    >>> rescaled.min(), rescaled.max()
    (0.0, 255.0)
    """
    _validate_array_not_empty(data, "input image")

    if in_range is None:
        in_min, in_max = float(np.min(data)), float(np.max(data))
    else:
        in_min, in_max = float(in_range[0]), float(in_range[1])

    out_min, out_max = float(out_range[0]), float(out_range[1])

    if in_max <= in_min:
        logger.warning("Input range invalid, returning constant array")
        return np.full_like(data, out_min, dtype=np.float32)

    # Scale to 0-1
    scaled = (data - in_min) / (in_max - in_min)

    # Scale to output range
    result = scaled * (out_max - out_min) + out_min

    if clip:
        result = np.clip(result, out_min, out_max)

    return result


def invert_image(data: np.ndarray, max_value: float | None = None) -> np.ndarray:
    """
    Invert image intensities.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    max_value : float, optional
        Maximum value for inversion. If None, uses data maximum.

    Returns
    -------
    np.ndarray
        Inverted image where each pixel = max_value - pixel.

    Examples
    --------
    >>> data = np.array([[0, 50, 100], [150, 200, 250]], dtype=float)
    >>> inverted = invert_image(data)
    >>> inverted
    array([[250., 200., 150.],
           [100.,  50.,   0.]])
    """
    _validate_array_not_empty(data, "input image")

    if max_value is None:
        max_value = np.max(data)

    return max_value - data


# ============================================================================
# File-Level Transformation Wrapper
# ============================================================================


def transform_image_file(
    input_file: StrOrPath,
    output_file: StrOrPath,
    transformation: str,
    hdu: HDUSelector = 0,
    overwrite: bool = False,
    output_mode: OutputMode = "sef",
    destination_hdu: HDUSelector | None = None,
    structure_policy: StructurePolicy = "compatible",
    **transform_params,
) -> TransformationResult:
    """
    Apply a transformation to a FITS file.

    Parameters
    ----------
    input_file : str
        Input FITS file.
    output_file : str
        Output FITS file.
    transformation : str
        Transformation name: 'crop', 'bin', 'normalize', 'rotate', 'flip', etc.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.
    **transform_params
        Parameters specific to the transformation.

    Returns
    -------
    TransformationResult
        Result of transformation.

    Examples
    --------
    >>> result = transform_image_file(
    ...     'input.fits', 'output.fits',
    ...     transformation='bin',
    ...     bin_factor=2, method='mean',
    ...     overwrite=True
    ... )
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    try:
        # Load data
        source = load_resolved_hdu(input_file, hdu)
        data, header = source.data, source.header
        input_shape = data.shape

        # Apply transformation
        if transformation == "crop":
            # Optional WCS CRPIX-only adjustment after cropping
            adjust_wcs = bool(transform_params.get("adjust_wcs", False))
            crop_params = dict(transform_params)
            crop_params.pop("adjust_wcs", None)

            result_data = crop_image(data, **crop_params)

            if adjust_wcs and "region" in crop_params:
                start_col, end_col, start_row, end_row = crop_params["region"]
                _adjust_crpix_for_crop(header, start_col=start_col, start_row=start_row)
        elif transformation == "bin":
            result_data = bin_image(data, **transform_params)
        elif transformation == "normalize":
            result_data = normalize_image(data, **transform_params)
        elif transformation == "rotate":
            result_data = rotate_image(data, **transform_params)
        elif transformation == "flip":
            result_data = flip_image(data, **transform_params)
        elif transformation == "transpose":
            result_data = transpose_image(data)
        elif transformation == "rescale":
            result_data = rescale_intensity(data, **transform_params)
        elif transformation == "invert":
            result_data = invert_image(data, **transform_params)
        else:
            raise ImageProcessingError(f"Unknown transformation: '{transformation}'")

        output_shape = result_data.shape

        # Update header
        add_history(header, f"Applied transformation: {transformation}")

        for key, val in transform_params.items():
            if isinstance(val, (int, float, str)) and len(str(val)) < 50:
                add_history(header, f"  {key}={val}")

        # Update dimension keywords if shape changed
        if output_shape != input_shape:
            header["NAXIS1"] = output_shape[1]
            header["NAXIS2"] = output_shape[0]

        # Write output
        commit_hdu_result(
            source,
            output_file,
            result_data,
            header,
            output_mode=output_mode,
            destination_hdu=destination_hdu,
            structure_policy=structure_policy,
            overwrite=overwrite,
            allow_shape_change=True,
        )

        logger.debug("Transformation complete: %s", output_file)

        return TransformationResult(
            data=result_data,
            header=header,
            transformation=transformation,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters=transform_params,
            output_file=output_file,
        )

    except Exception as e:
        logger.error("Transformation failed: %s", e, exc_info=True)
        raise ImageProcessingError(f"Failed to apply transformation: {e}") from e


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Geometric transformations
    "crop_image",
    "bin_image",
    "rotate_image",
    "flip_image",
    "transpose_image",
    # Intensity transformations
    "normalize_image",
    "rescale_intensity",
    "invert_image",
    # File-level wrapper
    "transform_image_file",
]
