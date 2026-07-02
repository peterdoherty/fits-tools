#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Functions for calculating statistics from FITS image data.

This module provides comprehensive statistical analysis tools for FITS image
data, including basic statistics, robust statistics, CCD-specific metrics,
and multi-file analysis capabilities.

Key Features
------------
- Array-based statistics via :func:`array_stats` (no I/O)
- Single-file statistics with region support via :func:`stats`
- Robust statistics using sigma-clipping via :func:`robust_stats`
- CCD-specific metrics (SNR, dynamic range, read noise)
- Multi-file statistics aggregation
- Comparison and trend analysis

ROI Tuple Convention
--------------------
Throughout this module, regions of interest are specified as a 4-tuple::

    (start_col, end_col, start_row, end_row)   # (x1, x2, y1, y2)

applied to an array as ``arr[start_row:end_row, start_col:end_col]``.

.. note::
    This is **column-first** ordering, consistent with FITS pixel-coordinate
    conventions (FITS uses (column, row) = (x, y) order). It is the
    **opposite** of NumPy's native (row, col) shape convention. Always
    read the tuple as (x1, x2, y1, y2) and apply with
    ``arr[y1:y2, x1:x2]`` to avoid off-axis confusion.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.stats import mad_std, sigma_clip

from .exceptions import (
    FileHandlingError,
    FitsToolsError,
    HeaderKeywordError,
    ImageProcessingError,
    SelectionError,
)
from .hdu import HDUSelector
from .io import get_data
from .selection import select_files
from .types import StrOrPath

logger = logging.getLogger(__name__)

__all__ = [
    # Dataclasses
    "StatsResult",
    "RobustStatsResult",
    "ComparisonResult",
    # Core array-based function
    "array_stats",
    # Single-file functions
    "stats",
    "get_image_min",
    "get_image_max",
    "get_image_mean",
    "get_image_median",
    "get_image_std",
    # Robust statistics
    "robust_stats",
    # CCD-specific
    "calculate_snr",
    "calculate_dynamic_range",
    "estimate_read_noise",
    # Multi-file statistics
    "calculate_statistics_across_files",
    "get_average_mean",
    "get_average_median",
    "get_average_minimum",
    "get_average_maximum",
    "get_average_std",
    # Comparison and trend
    "compare_stats",
    "analyze_statistics_trend",
    # Deprecated
    "stats_to_dict",
    "calculate_average_stats",
    "get_average_stddev",
    # Alias constant
    "STATS_TYPE_ALIASES",
]


# =============================================================================
# Statistics Result Dataclasses
# =============================================================================


@dataclass
class StatsResult:
    """Comprehensive statistical results for an image or region.

    Attributes
    ----------
    min_val, max_val : float
        Minimum and maximum pixel values.
    mean_val : float
        Mean pixel value.
    median_val : float
        Median pixel value. ``nan`` when median was not computed (e.g.
        when :func:`stats` is called with ``stats_type='simple'``).
    std_val : float
        Standard deviation.
    variance_val : float
        Variance.
    num_pixels : int
        Number of pixels analysed (excludes NaNs when ``skipna=True``).
    percentile_5, percentile_25, percentile_75, percentile_95 : float
        Percentile values. ``nan`` when extended statistics were not
        computed.
    iqr : float
        Interquartile range (Q3 − Q1). ``nan`` if not computed.
    mad : float
        Median Absolute Deviation (scaled to σ). ``nan`` if not computed.
    range_val : float
        Range (max − min).

    Notes
    -----
    Fields deliberately skipped (rather than computed as NaN from data)
    will also be ``nan`` in the returned object. Check the calling
    function's documentation to know which fields are populated.
    """

    min_val:       float
    max_val:       float
    mean_val:      float
    median_val:    float
    std_val:       float
    variance_val:  float
    num_pixels:    int   = 0
    percentile_5:  float = np.nan
    percentile_25: float = np.nan
    percentile_75: float = np.nan
    percentile_95: float = np.nan
    iqr:           float = np.nan
    mad:           float = np.nan
    range_val:     float = np.nan

    def to_dict(self) -> dict[str, Any]:
        """Return all statistics as a plain dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Statistics Summary:\n"
            f"  Mean:   {self.mean_val:.4f}\n"
            f"  Median: {self.median_val:.4f}\n"
            f"  Std:    {self.std_val:.4f}\n"
            f"  Min:    {self.min_val:.4f}\n"
            f"  Max:    {self.max_val:.4f}\n"
            f"  Pixels: {self.num_pixels}"
        )

    def __repr__(self) -> str:
        return (
            f"StatsResult(mean={self.mean_val:.4f}, "
            f"median={self.median_val:.4f}, "
            f"std={self.std_val:.4f}, n={self.num_pixels})"
        )


@dataclass
class RobustStatsResult:
    """Results from robust (sigma-clipped) statistical analysis.

    Attributes
    ----------
    mean, median, std, variance : float
        Statistics computed on the surviving (un-clipped) pixels.
    num_pixels : int
        Number of pixels after clipping.
    num_clipped : int
        Number of pixels rejected by sigma-clipping.
    clip_fraction : float
        Fraction of pixels rejected.
    mask : np.ndarray or None
        Boolean mask of clipped pixels (True = clipped), if
        ``return_mask=True`` was passed; otherwise None.
    """

    mean:          float
    median:        float
    std:           float
    variance:      float
    num_pixels:    int
    num_clipped:   int   = 0
    clip_fraction: float = 0.0
    mask: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return statistics as a plain dictionary (mask excluded)."""
        d = asdict(self)
        d.pop("mask", None)
        return d

    def __str__(self) -> str:
        return (
            f"Robust Statistics (σ-clipped):\n"
            f"  Mean:     {self.mean:.4f}\n"
            f"  Median:   {self.median:.4f}\n"
            f"  Std:      {self.std:.4f}\n"
            f"  Pixels:   {self.num_pixels}\n"
            f"  Clipped:  {self.num_clipped} ({self.clip_fraction*100:.1f}%)"
        )


@dataclass
class ComparisonResult:
    """Statistics comparison between two FITS files.

    Attributes
    ----------
    file1, file2 : Path
        Paths to the compared files (stored as Path regardless of input
        type).
    stats1, stats2 : StatsResult
        Statistics for each file.
    mean_diff : float
        Difference of means (file1 − file2), computed in
        ``__post_init__``.
    std_diff : float
        Difference of standard deviations, computed in ``__post_init__``.
    relative_mean_diff : float
        ``mean_diff / stats2.mean_val``; ``nan`` if ``stats2.mean_val``
        is zero.
    """

    file1:  Path
    file2:  Path
    stats1: StatsResult
    stats2: StatsResult
    mean_diff:          float = field(init=False)
    std_diff:           float = field(init=False)
    relative_mean_diff: float = field(init=False)

    def __post_init__(self) -> None:
        self.file1 = Path(self.file1)
        self.file2 = Path(self.file2)
        self.mean_diff = self.stats1.mean_val - self.stats2.mean_val
        self.std_diff  = self.stats1.std_val  - self.stats2.std_val
        self.relative_mean_diff = (
            self.mean_diff / self.stats2.mean_val
            if self.stats2.mean_val != 0
            else float("nan")
        )

    def __str__(self) -> str:
        return (
            f"Comparison Results:\n"
            f"  File 1: {self.file1.name}\n"
            f"    Mean: {self.stats1.mean_val:.4f},  "
            f"Std: {self.stats1.std_val:.4f}\n"
            f"  File 2: {self.file2.name}\n"
            f"    Mean: {self.stats2.mean_val:.4f},  "
            f"Std: {self.stats2.std_val:.4f}\n"
            f"  Differences:\n"
            f"    Mean: {self.mean_diff:.4f} "
            f"({self.relative_mean_diff*100:.2f}%)\n"
            f"    Std:  {self.std_diff:.4f}"
        )


# =============================================================================
# Deprecated backward-compatibility wrapper
# =============================================================================


def stats_to_dict(stats_result: StatsResult) -> dict[str, Any]:
    """Convert a :class:`StatsResult` to a dictionary.

    .. deprecated::
        Use :meth:`StatsResult.to_dict` instead.
    """
    warnings.warn(
        "stats_to_dict() is deprecated; use StatsResult.to_dict() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return stats_result.to_dict()


# =============================================================================
# Core Statistics Functions
# =============================================================================


def _simple_stats(arr: np.ndarray, skipna: bool = False) -> StatsResult:
    """Compute fast O(N) basic statistics, skipping median and percentiles.

    Returns a :class:`StatsResult` with ``median_val``, all percentile
    fields, ``iqr``, and ``mad`` set to ``nan``. Median requires a sort
    (O(N log N)) which this function deliberately avoids. Callers that
    need the median should use :func:`array_stats`.
    """
    if skipna:
        f_min  = np.nanmin(arr)
        f_max  = np.nanmax(arr)
        f_mean = np.nanmean(arr)
        f_std  = np.nanstd(arr)
        f_var  = np.nanvar(arr)
        pixels = int(np.sum(~np.isnan(arr)))
    else:
        f_min  = np.min(arr)
        f_max  = np.max(arr)
        f_mean = np.mean(arr)
        f_std  = np.std(arr)
        f_var  = np.var(arr)
        pixels = int(arr.size)

    return StatsResult(
        min_val=float(f_min),
        max_val=float(f_max),
        mean_val=float(f_mean),
        median_val=np.nan,  # deliberately skipped — see docstring
        std_val=float(f_std),
        variance_val=float(f_var),
        num_pixels=pixels,
        range_val=float(f_max - f_min),
    )


# Mapping for stats_type aliases used by :func:`stats`.
STATS_TYPE_ALIASES: dict[str, str] = {
    "min":      "min_val",
    "max":      "max_val",
    "mean":     "mean_val",
    "median":   "median_val",
    "std":      "std_val",
    "stddev":   "std_val",
    "variance": "variance_val",
    "var":      "variance_val",
    "rms":      "std_val",
    "dev":      "std_val",
    "range":    "range_val",
    "iqr":      "iqr",
    "mad":      "mad",
    "p5":       "percentile_5",
    "p25":      "percentile_25",
    "p75":      "percentile_75",
    "p95":      "percentile_95",
}


def array_stats(
    arr: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    skipna: bool = False,
    compute_extended: bool = True,
) -> StatsResult:
    """Calculate comprehensive statistics for a NumPy array or sub-region.

    Parameters
    ----------
    arr : np.ndarray
        Input array (image data). Must be numeric. Arrays with three or
        more dimensions are flattened before analysis.
    roi : tuple of int, optional
        Region of interest as ``(start_col, end_col, start_row, end_row)``
        — i.e. **(x1, x2, y1, y2)** — applied as
        ``arr[start_row:end_row, start_col:end_col]``. Only valid for 2-D
        arrays. See the module-level *ROI Tuple Convention* note.
        If None, the full array is analysed.
    skipna : bool, optional
        If True, use NaN-safe functions and count only finite pixels
        toward ``num_pixels``. Default is False.
    compute_extended : bool, optional
        If True (default), compute percentiles, IQR, and MAD. Set to
        False for a faster run when only mean/median/std are needed.

    Returns
    -------
    StatsResult
        Comprehensive statistics object. ``percentile_*``, ``iqr``, and
        ``mad`` are ``nan`` when ``compute_extended=False``.

    Raises
    ------
    TypeError
        If *arr* is not a NumPy array.
    ValueError
        If *roi* is invalid or out of bounds.
    ImageProcessingError
        If an error occurs during computation.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.default_rng(0).normal(100, 10, size=(100, 100))
    >>> result = array_stats(data)
    >>> 95 < result.mean_val < 105
    True
    >>> result.num_pixels
    10000

    With ROI (column-first ordering):

    >>> result_roi = array_stats(data, roi=(0, 50, 0, 50))
    >>> result_roi.num_pixels
    2500
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input 'arr' must be a NumPy array.")

    if roi is not None and arr.ndim != 2:
        raise ValueError("ROI is only supported for 2-D arrays.")

    if arr.ndim >= 3:
        arr = arr.flatten()

    if arr.ndim == 2 and roi is not None:
        if len(roi) != 4:
            raise ValueError(
                "roi must be a tuple of 4 integers "
                "(start_col, end_col, start_row, end_row)."
            )
        start_col, end_col, start_row, end_row = roi
        if not (
            0 <= start_row < end_row <= arr.shape[0]
            and 0 <= start_col < end_col <= arr.shape[1]
        ):
            raise ValueError(
                f"Invalid roi {roi} for array of shape {arr.shape}."
            )
        data_to_analyze = arr[start_row:end_row, start_col:end_col]
    else:
        data_to_analyze = arr

    if data_to_analyze.size == 0:
        logger.warning("array_stats: empty array — returning all-NaN StatsResult.")
        return StatsResult(
            min_val=np.nan, max_val=np.nan, mean_val=np.nan,
            median_val=np.nan, std_val=np.nan, variance_val=np.nan,
            num_pixels=0,
        )

    try:
        if skipna:
            finite_mask = np.isfinite(data_to_analyze)
            finite_values = np.asarray(data_to_analyze[finite_mask])
            num_pixels = int(finite_values.size)

            if num_pixels == 0:
                logger.warning(
                    "array_stats: no finite pixels after skipna filtering."
                )
                return StatsResult(
                    min_val=np.nan,
                    max_val=np.nan,
                    mean_val=np.nan,
                    median_val=np.nan,
                    std_val=np.nan,
                    variance_val=np.nan,
                    num_pixels=0,
                )

            data_for_stats = finite_values
            min_func = np.min
            max_func = np.max
            mean_func = np.mean
            median_func = np.median
            std_func = np.std
            var_func = np.var
            percentile_func = np.percentile
        else:
            data_for_stats = data_to_analyze
            min_func        = np.min
            max_func        = np.max
            mean_func       = np.mean
            median_func     = np.median
            std_func        = np.std
            var_func        = np.var
            percentile_func = np.percentile
            num_pixels = int(data_to_analyze.size)

        min_val      = float(min_func(data_for_stats))
        max_val      = float(max_func(data_for_stats))
        mean_val     = float(mean_func(data_for_stats))
        median_val   = float(median_func(data_for_stats))
        std_val      = float(std_func(data_for_stats))
        variance_val = float(var_func(data_for_stats))
        range_val    = max_val - min_val

        if compute_extended:
            p5, p25, p75, p95 = map(
                float, percentile_func(data_for_stats, [5, 25, 75, 95])
            )
            iqr = p75 - p25
            try:
                mad = float(mad_std(data_for_stats, ignore_nan=False))
            except Exception:
                mad = np.nan
        else:
            p5 = p25 = p75 = p95 = iqr = mad = np.nan

        return StatsResult(
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            median_val=median_val,
            std_val=std_val,
            variance_val=variance_val,
            num_pixels=num_pixels,
            percentile_5=p5,
            percentile_25=p25,
            percentile_75=p75,
            percentile_95=p95,
            iqr=iqr,
            mad=mad,
            range_val=range_val,
        )

    except Exception as e:
        logger.error("array_stats: computation failed: %s", e, exc_info=True)
        raise ImageProcessingError(
            f"Failed to calculate statistics: {e}"
        ) from e


def _array_stats(
    arr: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    skipna: bool = False,
    compute_extended: bool = True,
) -> StatsResult:
    """Deprecated alias for :func:`array_stats`.

    .. deprecated::
        Use :func:`array_stats` instead.
    """
    warnings.warn(
        "_array_stats() is a private name and will be removed in a future "
        "version. Use array_stats() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return array_stats(arr, roi=roi, skipna=skipna, compute_extended=compute_extended)


# =============================================================================
# Single-File Statistics
# =============================================================================


def stats(
    input_file: StrOrPath,
    hdu: HDUSelector = 0,
    region: str | tuple[int, int, int, int] = "full",
    stats_type: str = "all",
    skipna: bool = False,
) -> float | StatsResult | None:
    """Return statistics for a single FITS file.

    Parameters
    ----------
    input_file : str or Path
        Path to the FITS file.
    hdu : int, optional
        HDU extension index. Default is 0.
    region : str or tuple, optional
        Region to analyse. May be ``'full'``, ``'data'``, ``'bias'``,
        ``'trim'``, ``'dark'``, or a tuple
        ``(start_col, end_col, start_row, end_row)``. Default is
        ``'full'``.
    stats_type : str, optional
        Which statistic to return. Options:

        - ``'all'`` (default) — returns a :class:`StatsResult`.
        - ``'simple'`` — returns a :class:`StatsResult` with only
          min/max/mean/std populated; median and all percentile fields
          are ``nan``. Faster than ``'all'`` for large arrays.
        - Any key from :data:`STATS_TYPE_ALIASES` (``'mean'``,
          ``'median'``, ``'std'``, etc.) — returns a single ``float``.
    skipna : bool, optional
        If True, ignore NaN values. Default is False.

    Returns
    -------
    StatsResult, float, or None
        :class:`StatsResult` for ``stats_type='all'`` or ``'simple'``;
        ``float`` for a named scalar statistic; ``None`` if data cannot
        be loaded.

    Raises
    ------
    ValueError
        If *stats_type* is not recognised.
    FileHandlingError, ImageProcessingError
        On I/O or computation failure.
    """
    input_file = Path(input_file)
    try:
        roi_tuple = None

        if isinstance(region, tuple):
            data = get_data(input_file, hdu=hdu, region="full")
            roi_tuple = region
        else:
            data = get_data(input_file, hdu=hdu, region=region)

        if data is None:
            logger.warning(
                "No valid data retrieved for %s hdu=%s region=%r. "
                "Returning None.",
                input_file, hdu, region,
            )
            return None

        if stats_type.lower() == "simple":
            return _simple_stats(data, skipna=skipna)

        all_stats = array_stats(data, roi=roi_tuple, skipna=skipna)

        if stats_type == "all":
            return all_stats

        attr_name = STATS_TYPE_ALIASES.get(stats_type.lower())
        if attr_name is None:
            valid_types = sorted(STATS_TYPE_ALIASES.keys()) + ["all", "simple"]
            raise ValueError(
                f"Unsupported stats_type: {stats_type!r}. "
                f"Valid options: {valid_types}"
            )
        return getattr(all_stats, attr_name)

    except (FileHandlingError, ValueError, HeaderKeywordError, ImageProcessingError):
        raise
    except Exception as e:
        logger.error(
            "Unexpected error calculating stats for %s: %s",
            input_file, e, exc_info=True,
        )
        raise FitsToolsError(f"An unexpected error occurred: {e}") from e


# =============================================================================
# Convenience wrappers — single statistic from a single file
# =============================================================================


def get_image_min(
    input_file: StrOrPath,
    hdu: HDUSelector = 0,
    region: str | tuple[int, int, int, int] = "full",
) -> float | None:
    """Return the minimum pixel value from a FITS image region."""
    return stats(input_file, hdu=hdu, region=region, stats_type="min")


def get_image_max(
    input_file: StrOrPath,
    hdu: HDUSelector = 0,
    region: str | tuple[int, int, int, int] = "full",
) -> float | None:
    """Return the maximum pixel value from a FITS image region."""
    return stats(input_file, hdu=hdu, region=region, stats_type="max")


def get_image_mean(
    input_file: StrOrPath,
    hdu: HDUSelector = 0,
    region: str | tuple[int, int, int, int] = "full",
) -> float | None:
    """Return the mean pixel value from a FITS image region."""
    return stats(input_file, hdu=hdu, region=region, stats_type="mean")


def get_image_median(
    input_file: StrOrPath,
    hdu: HDUSelector = 0,
    region: str | tuple[int, int, int, int] = "full",
) -> float | None:
    """Return the median pixel value from a FITS image region."""
    return stats(input_file, hdu=hdu, region=region, stats_type="median")


def get_image_std(
    input_file: StrOrPath,
    hdu: HDUSelector = 0,
    region: str | tuple[int, int, int, int] = "full",
) -> float | None:
    """Return the standard deviation of pixel values from a FITS image region."""
    return stats(input_file, hdu=hdu, region=region, stats_type="std")


# =============================================================================
# Robust Statistics
# =============================================================================


def robust_stats(
    data_or_file: str | Path | np.ndarray,
    hdu: HDUSelector = 0,
    region: str | tuple[int, int, int, int] = "full",
    sigma: float = 3.0,
    maxiters: int = 5,
    return_mask: bool = False,
) -> RobustStatsResult:
    """Calculate robust statistics using iterative sigma-clipping.

    Parameters
    ----------
    data_or_file : str, Path, or np.ndarray
        FITS file path or pre-loaded array (including masked arrays).
    hdu : int, optional
        HDU extension index (ignored for array input). Default is 0.
    region : str or tuple, optional
        Region to analyse (ignored for array input). Default is ``'full'``.
    sigma : float, optional
        Clipping threshold in units of standard deviation. Default is 3.0.
    maxiters : int, optional
        Maximum number of clipping iterations. Default is 5.
    return_mask : bool, optional
        If True, attach the clipping mask to the result. Default is False.

    Returns
    -------
    RobustStatsResult
        Statistics computed on the surviving (un-clipped) pixels.

    Raises
    ------
    FileHandlingError
        If the file cannot be opened (propagated without wrapping).
    ImageProcessingError
        On computation failure.
    """
    try:
        roi_tuple = None

        if isinstance(data_or_file, np.ndarray):
            data = data_or_file
            if isinstance(region, tuple):
                roi_tuple = region
        elif isinstance(data_or_file, (str, Path)):
            if isinstance(region, tuple):
                data = get_data(data_or_file, hdu=hdu, region="full")
                roi_tuple = region
            else:
                data = get_data(data_or_file, hdu=hdu, region=region)
        else:
            raise TypeError(
                f"Expected file path (str/Path) or numpy array, "
                f"got {type(data_or_file).__name__}"
            )

        if data is None:
            raise ImageProcessingError("No valid data provided")

        if roi_tuple is not None:
            start_col, end_col, start_row, end_row = roi_tuple
            data = data[start_row:end_row, start_col:end_col]

        original_size = int(data.size)
        clipped = sigma_clip(data, sigma=sigma, maxiters=maxiters, masked=True)
        good = clipped.compressed()

        num_pixels    = int(good.size)
        num_clipped   = int(original_size - num_pixels)
        clip_fraction = num_clipped / original_size if original_size > 0 else 0.0

        if num_pixels == 0:
            mean_val = median_val = std_val = var_val = float("nan")
        else:
            mean_val   = float(np.mean(good))
            median_val = float(np.median(good))
            std_val    = float(np.std(good))
            var_val    = float(np.var(good))

        mask_2d = (
            clipped.mask
            if (return_mask and hasattr(clipped, "mask"))
            else None
        )

        return RobustStatsResult(
            mean=mean_val,
            median=median_val,
            std=std_val,
            variance=var_val,
            num_pixels=num_pixels,
            num_clipped=num_clipped,
            clip_fraction=clip_fraction,
            mask=mask_2d,
        )

    except (FileHandlingError, HeaderKeywordError, TypeError):
        raise
    except Exception as e:
        input_desc = (
            str(data_or_file)
            if isinstance(data_or_file, (str, Path))
            else f"array{getattr(data_or_file, 'shape', '')}"
        )
        logger.error(
            "robust_stats failed for %s: %s", input_desc, e, exc_info=True
        )
        raise ImageProcessingError(
            f"Failed to calculate robust statistics: {e}"
        ) from e


# =============================================================================
# CCD-Specific Statistics
# =============================================================================


def calculate_snr(
    input_file: StrOrPath,
    signal_region: str | tuple[int, int, int, int] = "data",
    noise_region: str | tuple[int, int, int, int] | None = None,
    hdu: HDUSelector = 0,
) -> float:
    """Calculate Signal-to-Noise Ratio (SNR = mean / std).

    Parameters
    ----------
    input_file : str or Path
        Path to the FITS file.
    signal_region : str or tuple, optional
        Region from which signal and (optionally) noise are measured.
        Default is ``'data'``.
    noise_region : str, tuple, or None, optional
        Separate region for noise measurement. If None, the std of
        *signal_region* is used. Default is None.
    hdu : int, optional
        HDU extension index. Default is 0.

    Returns
    -------
    float
        Signal-to-noise ratio. Returns ``inf`` if noise std is zero.

    Raises
    ------
    ImageProcessingError
        On computation failure.
    """
    try:
        signal_result = stats(
            input_file, hdu=hdu, region=signal_region, stats_type="all"
        )

        if noise_region is None:
            noise_std = signal_result.std_val
        else:
            noise_std = stats(
                input_file, hdu=hdu, region=noise_region, stats_type="std"
            )

        if noise_std == 0:
            logger.warning("Noise std is zero; SNR is infinite.")
            return float("inf")

        snr = signal_result.mean_val / noise_std
        logger.debug(
            "SNR: %.2f (signal mean: %.2f, noise std: %.2f)",
            snr, signal_result.mean_val, noise_std,
        )
        return float(snr)

    except Exception as e:
        logger.error(
            "Error calculating SNR for %s: %s", input_file, e, exc_info=True
        )
        raise ImageProcessingError(f"Failed to calculate SNR: {e}") from e


def calculate_dynamic_range(
    input_file: StrOrPath,
    region: str | tuple[int, int, int, int] = "data",
    hdu: HDUSelector = 0,
) -> float:
    """Calculate dynamic range (max / min).

    Parameters
    ----------
    input_file : str or Path
        Path to the FITS file.
    region : str or tuple, optional
        Region to analyse. Default is ``'data'``.
    hdu : int, optional
        HDU extension index. Default is 0.

    Returns
    -------
    float
        Dynamic range. Returns ``inf`` if the minimum is zero.

    Raises
    ------
    ImageProcessingError
        On computation failure.
    """
    try:
        result = stats(input_file, hdu=hdu, region=region, stats_type="all")
        if result.min_val == 0:
            logger.warning("Minimum value is zero; dynamic range is infinite.")
            return float("inf")
        dr = result.max_val / result.min_val
        logger.debug(
            "Dynamic range: %.2f (max: %.2f, min: %.2f)",
            dr, result.max_val, result.min_val,
        )
        return float(dr)

    except Exception as e:
        logger.error(
            "Error calculating dynamic range for %s: %s",
            input_file, e, exc_info=True,
        )
        raise ImageProcessingError(
            f"Failed to calculate dynamic range: {e}"
        ) from e


def estimate_read_noise(
    bias_files: list[StrOrPath],
    region: str | tuple[int, int, int, int] = "data",
    hdu: HDUSelector = 0,
) -> float:
    """Estimate read noise as the mean standard deviation across bias frames.

    Parameters
    ----------
    bias_files : list of str or Path
        Paths to bias frames.
    region : str or tuple, optional
        Region to analyse. Default is ``'data'``.
    hdu : int, optional
        HDU extension index. Default is 0.

    Returns
    -------
    float
        Estimated read noise in ADU.

    Raises
    ------
    ValueError
        If *bias_files* is empty.
    ImageProcessingError
        If no valid standard deviations can be obtained from any file.
    """
    if not bias_files:
        raise ValueError("No bias files provided.")

    std_values: list[float] = []
    for bias_file in bias_files:
        try:
            std_val = stats(Path(bias_file), hdu=hdu, region=region, stats_type="std")
            if std_val is not None and np.isfinite(std_val):
                std_values.append(std_val)
        except (FileHandlingError, HeaderKeywordError) as e:
            logger.warning(
                "Could not read %s for read noise estimation: %s. Skipping.",
                bias_file, e,
            )
        except Exception as e:
            logger.warning(
                "Unexpected error processing %s: %s. Skipping.",
                bias_file, e, exc_info=True,
            )

    if not std_values:
        raise ImageProcessingError(
            "No valid standard deviation values obtained from bias frames."
        )

    read_noise = float(np.mean(std_values))
    logger.debug(
        "Estimated read noise: %.4f ADU (from %d frames)",
        read_noise, len(std_values),
    )
    return read_noise


# =============================================================================
# Multi-File Statistics
# =============================================================================


def calculate_statistics_across_files(
    files: StrOrPath | list[StrOrPath],
    keys: dict[str, Any] | None = None,
    region: str | tuple[int, int, int, int] | None = None,
    recursive: bool = False,
    single_statistic: str | None = None,
) -> dict[str, float] | float | None:
    """Calculate the average of each statistic across multiple FITS files.

    .. note::
        This computes the **average of each per-file statistic**, not the
        statistics of the combined data. For example, if file 1 has
        mean=100 and file 2 has mean=200, the result is ``{'mean': 150.0}``.

    Parameters
    ----------
    files : str, Path, or list thereof
        File paths, a directory path, or a glob pattern.
    keys : dict, optional
        FITS header keyword/value pairs used to filter files.
    region : str or tuple, optional
        Region to analyse per file. Default is ``'full'``.
    recursive : bool, optional
        Search subdirectories when *files* is a directory. Default is False.
    single_statistic : str or None, optional
        If provided, return only the average of this statistic. Must be
        one of ``'mean'``, ``'median'``, ``'min'``, ``'max'``,
        ``'std'``, ``'variance'``.

    Returns
    -------
    dict of str to float, float, or None
        Dict of averages when *single_statistic* is None; ``float``
        otherwise. None if no files are processed successfully.

    Raises
    ------
    ValueError
        If *single_statistic* is not a recognised key.
    SelectionError
        If file selection fails.
    """
    allowed_stats = ["min", "max", "mean", "median", "std", "variance"]
    if single_statistic and single_statistic not in allowed_stats:
        raise ValueError(
            f"Invalid single_statistic: {single_statistic!r}. "
            f"Allowed values: {allowed_stats}"
        )

    try:
        fits_files = select_files(files, keys=keys, recursive=recursive)
    except SelectionError:
        raise

    if not fits_files:
        logger.debug(
            "calculate_statistics_across_files: no FITS files found; "
            "returning None."
        )
        return None

    stats_lists: dict[str, list[float]] = {k: [] for k in allowed_stats}

    for file_path in fits_files:
        try:
            file_stats = stats(
                file_path,
                region=region if region is not None else "full",
                stats_type="all",
            )
            if not isinstance(file_stats, StatsResult):
                logger.warning(
                    "No valid statistics for %s. Skipping.", file_path
                )
                continue
            for key in allowed_stats:
                attr = STATS_TYPE_ALIASES.get(key, key)
                val = getattr(file_stats, attr, np.nan)
                if np.isfinite(val):
                    stats_lists[key].append(val)
            logger.debug("Processed %s", Path(file_path).name)
        except FitsToolsError as e:
            logger.warning(
                "Could not process %s: %s. Skipping.", file_path, e
            )
        except Exception as e:
            logger.warning(
                "Unexpected error processing %s: %s. Skipping.",
                file_path, e, exc_info=True,
            )

    if not stats_lists["mean"]:
        logger.debug(
            "calculate_statistics_across_files: no valid statistics "
            "collected; returning None."
        )
        return None

    average_results = {
        key: float(np.mean(values))
        for key, values in stats_lists.items()
        if values
    }

    logger.debug(
        "Average statistics from %d files:", len(stats_lists["mean"])
    )
    for key, val in average_results.items():
        logger.debug("  %s: %.4f", key.capitalize(), val)

    if single_statistic:
        return average_results.get(single_statistic)
    return average_results


def calculate_average_stats(*args: Any, **kwargs: Any) -> Any:
    """Deprecated alias for :func:`calculate_statistics_across_files`.

    .. deprecated::
        Use :func:`calculate_statistics_across_files` instead.
    """
    warnings.warn(
        "calculate_average_stats() is deprecated; "
        "use calculate_statistics_across_files() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calculate_statistics_across_files(*args, **kwargs)


def get_average_mean(
    files: StrOrPath | list[StrOrPath],
    keys: dict[str, Any] | None = None,
    region: str | tuple[int, int, int, int] | None = None,
    recursive: bool = False,
) -> float | None:
    """Return the mean of per-file mean values."""
    return calculate_statistics_across_files(
        files, keys=keys, region=region, recursive=recursive,
        single_statistic="mean",
    )


def get_average_median(
    files: StrOrPath | list[StrOrPath],
    keys: dict[str, Any] | None = None,
    region: str | tuple[int, int, int, int] | None = None,
    recursive: bool = False,
) -> float | None:
    """Return the mean of per-file median values."""
    return calculate_statistics_across_files(
        files, keys=keys, region=region, recursive=recursive,
        single_statistic="median",
    )


def get_average_minimum(
    files: StrOrPath | list[StrOrPath],
    keys: dict[str, Any] | None = None,
    region: str | tuple[int, int, int, int] | None = None,
    recursive: bool = False,
) -> float | None:
    """Return the mean of per-file minimum values."""
    return calculate_statistics_across_files(
        files, keys=keys, region=region, recursive=recursive,
        single_statistic="min",
    )


def get_average_maximum(
    files: StrOrPath | list[StrOrPath],
    keys: dict[str, Any] | None = None,
    region: str | tuple[int, int, int, int] | None = None,
    recursive: bool = False,
) -> float | None:
    """Return the mean of per-file maximum values."""
    return calculate_statistics_across_files(
        files, keys=keys, region=region, recursive=recursive,
        single_statistic="max",
    )


def get_average_std(
    files: StrOrPath | list[StrOrPath],
    keys: dict[str, Any] | None = None,
    region: str | tuple[int, int, int, int] | None = None,
    recursive: bool = False,
) -> float | None:
    """Return the mean of per-file standard deviation values."""
    return calculate_statistics_across_files(
        files, keys=keys, region=region, recursive=recursive,
        single_statistic="std",
    )


def get_average_stddev(*args: Any, **kwargs: Any) -> float | None:
    """Deprecated alias for :func:`get_average_std`.

    .. deprecated::
        Use :func:`get_average_std` instead.
    """
    warnings.warn(
        "get_average_stddev() is deprecated; use get_average_std() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_average_std(*args, **kwargs)


# =============================================================================
# Comparison and Trend Analysis
# =============================================================================


def compare_stats(
    file1: StrOrPath,
    file2: StrOrPath,
    region: str | tuple[int, int, int, int] = "full",
    hdu: HDUSelector = 0,
) -> ComparisonResult:
    """Compare statistics between two FITS files.

    Parameters
    ----------
    file1 : str or Path
        Path to the first FITS file.
    file2 : str or Path
        Path to the second FITS file.
    region : str or tuple, optional
        Region to analyse. Default is ``'full'``.
    hdu : int, optional
        HDU extension index. Default is 0.

    Returns
    -------
    ComparisonResult
        Statistics for both files and their differences.

    Raises
    ------
    ImageProcessingError
        On I/O or computation failure.
    """
    file1 = Path(file1)
    file2 = Path(file2)
    try:
        stats1 = stats(file1, hdu=hdu, region=region, stats_type="all")
        stats2 = stats(file2, hdu=hdu, region=region, stats_type="all")
        result = ComparisonResult(
            file1=file1, file2=file2, stats1=stats1, stats2=stats2
        )
        logger.debug("%s", result)
        return result

    except Exception as e:
        logger.error(
            "Error comparing %s and %s: %s",
            file1.name, file2.name, e, exc_info=True,
        )
        raise ImageProcessingError(
            f"Failed to compare statistics: {e}"
        ) from e


def analyze_statistics_trend(
    file_list: list[StrOrPath],
    region: str | tuple[int, int, int, int] = "full",
    hdu: HDUSelector = 0,
    statistic: str = "mean",
) -> dict[str, Any]:
    """Analyse a trend in one statistic across an ordered series of files.

    Useful for monitoring detector performance over time.

    Parameters
    ----------
    file_list : list of str or Path
        Files in chronological (or natural) order.
    region : str or tuple, optional
        Region to analyse. Default is ``'full'``.
    hdu : int, optional
        HDU extension index. Default is 0.
    statistic : str, optional
        Statistic to track. Must be a key in :data:`STATS_TYPE_ALIASES`.
        Default is ``'mean'``.

    Returns
    -------
    dict
        Keys:

        - ``'values'``    — per-file statistic values.
        - ``'trend'``     — linear slope (statistic per file-index).
        - ``'mean'``      — mean of values.
        - ``'std'``       — standard deviation of values.
        - ``'range'``     — max − min of values.
        - ``'num_files'`` — number of files contributing.

    Raises
    ------
    ValueError
        If *file_list* is empty.
    ImageProcessingError
        If no valid statistics can be obtained from any file.
    """
    if not file_list:
        raise ValueError("Empty file list provided.")

    values: list[float] = []
    for file_path in file_list:
        try:
            val = stats(
                Path(file_path), hdu=hdu, region=region, stats_type=statistic
            )
            if val is not None and np.isfinite(val):
                values.append(float(val))
            else:
                logger.warning(
                    "Invalid statistic for %s. Skipping.", file_path
                )
        except Exception as e:
            logger.warning(
                "Could not process %s: %s. Skipping.", file_path, e
            )

    if not values:
        raise ImageProcessingError(
            "No valid statistics obtained from any file in the list."
        )

    values_arr = np.asarray(values)
    trend = (
        float(np.polyfit(np.arange(len(values)), values_arr, 1)[0])
        if len(values) > 1
        else 0.0
    )

    result: dict[str, Any] = {
        "values":    values,
        "trend":     trend,
        "mean":      float(np.mean(values_arr)),
        "std":       float(np.std(values_arr)),
        "range":     float(np.max(values_arr) - np.min(values_arr)),
        "num_files": len(values),
    }

    logger.debug("Trend analysis for %r:", statistic)
    logger.debug("  Files:  %d", result["num_files"])
    logger.debug("  Mean:   %.4f", result["mean"])
    logger.debug("  Std:    %.4f", result["std"])
    logger.debug("  Trend:  %.6f per file", result["trend"])

    return result


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> int:
    """Command-line interface for statistics."""
    import argparse
    import doctest
    import sys

    parser = argparse.ArgumentParser(
        description="FITS Statistics Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test
  %(prog)s --file test.fits --stat mean
  %(prog)s --compare file1.fits file2.fits
        """,
    )
    parser.add_argument("--test", action="store_true", help="Run doctests")
    parser.add_argument("--file", metavar="FITS_FILE",
                        help="Calculate statistics for a file")
    parser.add_argument("--stat", default="all",
                        help="Statistic to calculate (default: all)")
    parser.add_argument("--region", default="full",
                        help="Region to analyze (default: full)")
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE1", "FILE2"),
        help="Compare statistics between two files",
    )
    # --verbose is appropriate here as a CLI flag; it controls log level
    # for the entry point only and is not propagated into library functions.
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.test or len(sys.argv) == 1:
        logger.info("Running doctests for statistics.py...")
        result = doctest.testmod(verbose=args.verbose)
        if result.failed == 0:
            logger.info("All %d doctests passed.", result.attempted)
            return 0
        logger.error(
            "%d of %d doctests failed.", result.failed, result.attempted
        )
        return 1

    if args.file:
        try:
            result = stats(args.file, region=args.region, stats_type=args.stat)
            print(
                result if args.stat == "all"
                else f"{args.stat}: {result:.4f}"
            )
            return 0
        except Exception as e:
            logger.error("Error: %s", e)
            return 1

    if args.compare:
        try:
            result = compare_stats(
                args.compare[0], args.compare[1], region=args.region
            )
            print(result)
            return 0
        except Exception as e:
            logger.error("Error: %s", e)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
