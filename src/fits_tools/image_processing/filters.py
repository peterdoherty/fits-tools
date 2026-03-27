#!/usr/bin/env python3
"""
Spatial filtering operations for FITS image data.

This module provides comprehensive filtering capabilities for astronomical
image processing including smoothing, edge detection, morphological operations,
and offset correction.

This module does not currently perform mask-aware filtering. Bad pixels/columns 
should be corrected or removed prior to filtering (e.g., via preprocessing).

Features:
    - Smoothing filters (Gaussian, median, mean, bilateral)
    - Edge detection (Sobel, Laplacian)
    - Morphological operations (opening, closing, erosion, dilation)
    - Offset correction (row/column bias removal)
    - Region-specific filtering
    - Batch processing with multiprocessing
    - Preserves FITS headers and metadata

Supported Operations:
    Smoothing:
        gaussian, median, mean, bilateral

    Edge Detection:
        sobel, laplacian

    Morphological:
        opening, closing, erosion, dilation

    Correction:
        correct_row_median, correct_row_mean,
        correct_col_median, correct_col_mean

    Enhancement:
        unsharp_mask, high_pass

    Usage examples:
        >>> from fits_tools.filters import apply_filter

        >>> # Simple smoothing
        >>> apply_filter("input.fits",
        ...     "smoothed.fits", "gaussian", sigma=2.0)  # doctest: +SKIP

        >>> # Sharpen image
        >>> apply_filter(
        ...     "image.fits", "sharp.fits", "unsharp_mask",
        ...     sigma=1.0, amount=1.5
        ... )  # doctest: +SKIP

        >>> # Remove row striping
        >>> apply_filter("biased.fits", "corrected.fits",
        ...              "correct_row_median")  # doctest: +SKIP

Version History:
    1.0.0 - Initial release
    2.0.0 - Production release
        - Added batch processing with multiprocessing
        - Added checksum validation
        - Fixed integer overflow in arithmetic filters
        - Improved exception handling semantics
        - Added safe dtype preservation with clipping
        - Enhanced logging for library use
"""

from __future__ import annotations

# Standard library
import hashlib
import inspect
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

# Third-party
from astropy.io import fits
import numpy as np
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
from tqdm import tqdm

# Import from package
from ..array_utils import validate_numeric
from ..exceptions import FileHandlingError, ImageProcessingError
from ..io import write_hdulist, region_to_slices, iter_image_hdus
from ..selection import select_files
from ..types import StrOrPath
from ..utils import add_history

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# Constants
# ============================================================================
DEFAULT_CHECKSUM_ALGORITHM = "sha256"
DEFAULT_CHUNK_SIZE = 65536  # 64KB

# ============================================================================
# Types
# ============================================================================
# StrOrPath = Union[str, Path]
Region = Union[str, tuple[int, int, int, int]]
FilterParams = Dict[str, Any]

# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class ProcessingResult:
    """Result of processing a single file."""

    input_file: Path
    output_file: Path
    success: bool
    error: Optional[Exception] = None
    checksum: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return f"ProcessingResult(success, {self.output_file})"
        return f"ProcessingResult(failed, {self.input_file}: {self.error})"


@dataclass
class BatchResult:
    """Complete batch processing results."""

    successful: List[Path]
    failed: List[Tuple[Path, Exception]]
    checksums: Dict[Path, str]

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
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    def __repr__(self) -> str:
        return (
            f"BatchResult(successful={self.success_count}, "
            f"failed={self.failure_count}, "
            f"rate={self.success_rate:.1%})"
        )


# ============================================================================
# Helper functions
# ============================================================================

def _validated_call(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Validate call parameters against ``func`` signature, then invoke it."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    try:
        sig.bind_partial(*args, **kwargs)
    except TypeError as e:
        raise ValueError(
            f"Invalid filter parameters for {func.__name__}: {e}"
        ) from e

    return func(*args, **kwargs)


def _compute_file_checksum(
    filepath: Path, algorithm: str = "sha256"
) -> str:
    """
    Compute checksum of a file.

    Parameters
    ----------
    filepath : Path
        Path to file. Must exist and be readable.
    algorithm : str, optional
        Hash algorithm (sha256, md5, etc.). Defaults to 'sha256'.

    Returns
    -------
    str
        Hexadecimal checksum string.

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If algorithm is not supported.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot compute checksum: {filepath} not found")

    if not filepath.is_file():
        raise ValueError(f"Cannot compute checksum: {filepath} is not a file")

    try:
        hash_obj = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(
            f"Unsupported hash algorithm '{algorithm}'. "
            f"Available: {', '.join(sorted(hashlib.algorithms_available))}"
        ) from e

    try:
        with open(filepath, "rb") as f:
            # Read in chunks for memory efficiency
            for chunk in iter(lambda: f.read(DEFAULT_CHUNK_SIZE), b""):
                hash_obj.update(chunk)
    except OSError as e:
        raise OSError(f"Failed to read {filepath} for checksum: {e}") from e

    return hash_obj.hexdigest()


def _process_single_filter_task(task: FilterParams) -> None:
    """
    Worker function to apply a filter to a single file.

    Designed for use with ProcessPoolExecutor. Extracts input/output
    paths from the task dictionary and calls apply_filter.

    Parameters
    ----------
    task : dict
        Dictionary with keys "input", "output", and filter parameters.

    Raises
    ------
    ValueError
        If required keys are missing from task.
    Exception
        Re-raises any exception from apply_filter for proper error tracking.
    """
    infile = task.get("input")
    outfile = task.get("output")

    if infile is None or outfile is None:
        raise ValueError("Task missing required 'input' or 'output' key")

    # Extract filter parameters without mutating original dict
    filter_params = {
        k: v for k, v in task.items() if k not in ("input", "output")
    }

    filter_hdu(infile, outfile, **filter_params)


def _safe_cast_to_dtype(data: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """
    Safely cast data to target dtype with proper clipping and rounding.

    Parameters
    ----------
    data : np.ndarray
        Input data (typically float32 from filter operations).
    target_dtype : np.dtype
        Target dtype to cast to.

    Returns
    -------
    np.ndarray
        Data cast to target dtype with proper handling.

    Notes
    -----
    For integer dtypes:
    - NaN/Inf are replaced with 0
    - Values are rounded to nearest integer
    - Values are clipped to dtype range to prevent overflow

    For float dtypes:
    - Direct cast (no clipping needed)
    """
    if np.issubdtype(target_dtype, np.integer):
        # Replace NaN/Inf with 0 for integer targets
        work = np.copy(data)
        work[~np.isfinite(work)] = 0

        # Round to nearest integer
        work = np.rint(work)

        # Clip to dtype range
        info = np.iinfo(target_dtype)
        work = np.clip(work, info.min, info.max)

        return work.astype(target_dtype)
    else:
        # Float to float: direct cast is safe
        return data.astype(target_dtype)
    

def _is_image_hdu(hdu: "fits.hdu.base.ExtensionHDU") -> bool:
    return isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))


def _hduname(hdu: "fits.hdu.base.ExtensionHDU") -> str:
    return str(hdu.header.get("EXTNAME", "")).strip().upper()


def _resolve_single_hdu(hdul: "fits.HDUList", hdu: Union[int, str, Literal["auto"]]) -> int:
    if isinstance(hdu, int):
        if not (0 <= hdu < len(hdul)):
            raise ValueError(f"Invalid hdu={hdu}; file has {len(hdul)} HDUs")
        return hdu

    if isinstance(hdu, str) and hdu != "auto":
        target = hdu.strip().upper()
        for i, hdu in enumerate(hdul):
            if _hduname(hdu) == target:
                return i
        raise ValueError(f"No HDU with EXTNAME={hdu!r}")

    # hdu == "auto": prefer SCI/IMAGE image HDUs with data; else first image HDU with data
    preferred = {"SCI", "IMAGE"}
    for i, hdu in enumerate(hdul):
        if _is_image_hdu(hdu) and hdu.data is not None and _hduname(hdu) in preferred:
            return i

    for i, hdu in enumerate(hdul):
        if _is_image_hdu(hdu) and hdu.data is not None:
            return i

    raise ValueError("No image HDU with data found (cannot auto-select hdu)")


def _require_2d(data: np.ndarray, op_name: str) -> None:
    if data.ndim != 2:
        raise ValueError(f"{op_name} requires 2D data, got shape={data.shape}")
    
# ============================================================================
# Smoothing Filters
# ============================================================================

def _apply_gaussian_filter(
    data: np.ndarray, sigma: float = 1.0, **kwargs
) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.gaussian_filter`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    """
    sigma_val = validate_numeric(sigma, "sigma", minimum=0.0, allow_zero=False)
    return _validated_call(gaussian_filter, data, sigma=sigma_val, **kwargs)


def _apply_median_filter(
    data: np.ndarray, size: int = 3, **kwargs
) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.median_filter`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
    """
    size_val = validate_numeric(size, "size", minimum=1, integer=True)
    return _validated_call(median_filter, data, size=size_val, **kwargs)


def _apply_mean_filter(data: np.ndarray, size: int = 3, **kwargs) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.uniform_filter`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html
    """

    size_val = validate_numeric(size, "size", minimum=1, integer=True)
    return _validated_call(uniform_filter, data, size=size_val, **kwargs)


def _apply_bilateral_filter(
    data: np.ndarray,
    sigma_spatial: float = 3.0,
    sigma_intensity: float = 50.0,
    **kwargs,
) -> np.ndarray:
    """
    Apply simplified bilateral filter for edge-preserving smoothing.

    This is an approximation using Gaussian blur with intensity-based
    weighting. For production quality bilateral filtering, consider using
    skimage.restoration.denoise_bilateral.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigma_spatial : float
        Spatial standard deviation for Gaussian blur. Default is 3.0.
    sigma_intensity : float
        Intensity range standard deviation for weighting. Default is 50.0.
    **kwargs
        Additional arguments passed to gaussian_filter.

    Returns
    -------
    np.ndarray
        Filtered image in float32.

    Notes
    -----
    This implementation:
    1. Applies Gaussian blur to get smoothed version
    2. Computes intensity differences
    3. Blends original and smoothed based on local intensity variation

    This is an approximation. For better quality, use:
        from skimage.restoration import denoise_bilateral

    See also
    --------
    scipy.ndimage.gaussian_filter : Used for spatial smoothing
    """
    sigma_spatial_val = validate_numeric(
        sigma_spatial, "sigma_spatial", minimum=0.0, allow_zero=False
    )
    sigma_intensity_val = validate_numeric(
        sigma_intensity, "sigma_intensity", minimum=0.0, allow_zero=False
    )

    # Work in float to avoid integer overflow
    work = np.asarray(data, dtype=np.float32)
    smoothed = _validated_call(
        gaussian_filter, work, sigma=sigma_spatial_val, **kwargs
    )
    diff = np.abs(work - smoothed)
    weights = np.exp(-((diff**2) / (2 * sigma_intensity_val**2)))

    return smoothed * weights + work * (1 - weights)


# ============================================================================
# Edge Detection Filters
# ============================================================================

def _apply_sobel_filter(
    data: np.ndarray, axis: Optional[int] = None, **kwargs
) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.sobel`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html
    """
    work = np.asarray(data, dtype=np.float32)

    if axis is not None:
        # allow negative axes too
        if not (-work.ndim <= axis < work.ndim):
            raise ValueError(f"axis out of range for ndim={work.ndim}: {axis}")
        return _validated_call(sobel, work, axis=axis, **kwargs)

    if work.ndim < 2:
        raise ValueError(f"sobel magnitude requires ndim>=2, got shape={work.shape}")

    sx = _validated_call(sobel, work, axis=-2, **kwargs)
    sy = _validated_call(sobel, work, axis=-1, **kwargs)
    return np.hypot(sx, sy)

def _apply_laplacian_filter(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.laplace`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html
    """
    return _validated_call(laplace, data, **kwargs)

# ============================================================================
# Morphological Filters
# ============================================================================

def _apply_opening(
    data: np.ndarray,
    structure: Optional[np.ndarray] = None,
    iterations: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.grey_opening`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_opening.html
    """
    iterations_val = validate_numeric(iterations, "iterations", minimum=1, integer=True)
    out = data
    scipy_kwargs = {"mode": "reflect", **kwargs}
    for _ in range(iterations_val):
        out = _validated_call(grey_opening, out, structure=structure, **scipy_kwargs)
    return out


def _apply_closing(
    data: np.ndarray,
    structure: Optional[np.ndarray] = None,
    iterations: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.grey_closing`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_closing.html
    """
    iterations_val = validate_numeric(iterations, "iterations", minimum=1, integer=True)
    out = data
    scipy_kwargs = {"mode": "reflect", **kwargs}
    for _ in range(iterations_val):
        out = _validated_call(grey_closing, out, structure=structure, **scipy_kwargs)
    return out


def _apply_erosion(
    data: np.ndarray,
    structure: Optional[np.ndarray] = None,
    iterations: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.grey_erosion`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_erosion.html
    """
    iterations_val = validate_numeric(iterations, "iterations", minimum=1, integer=True)
    out = data
    scipy_kwargs = {"mode": "reflect", **kwargs}
    for _ in range(iterations_val):
        out = _validated_call(grey_erosion, out, structure=structure, **scipy_kwargs)
    return out


def _apply_dilation(
    data: np.ndarray,
    structure: Optional[np.ndarray] = None,
    iterations: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Thin wrapper over `scipy.ndimage.grey_dilation`.

    SciPy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html
    """
    iterations_val = validate_numeric(iterations, "iterations", minimum=1, integer=True)
    out = data
    scipy_kwargs = {"mode": "reflect", **kwargs}
    for _ in range(iterations_val):
        out = _validated_call(grey_dilation, out, structure=structure, **scipy_kwargs)
    return out


# ============================================================================
# Offset Correction Filters
# ============================================================================


def _correct_row_offset(
    data: np.ndarray, method: str = "median"
) -> np.ndarray:
    """
    Correct row-by-row offset (horizontal striping).

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    method : str, optional
        Statistic to use: 'median' or 'mean'. Defaults to 'median'.

    Returns
    -------
    np.ndarray
        Corrected image in float32 (to avoid integer overflow during subtraction).
        Use preserve_dtype=True in apply_filter() to cast back to original dtype.

    Notes
    -----
    Computation is performed in float32 regardless of input dtype to prevent
    integer underflow when subtracting offsets.
    """
    _require_2d(data, "correct_row_offset")

    if method not in ("mean", "median"):
        raise ValueError(f"Method must be 'mean' or 'median', got '{method}'")

    data_float = data.astype(np.float32)

    if method == "median":
        offsets = np.median(data_float, axis=1)
    else:
        offsets = np.mean(data_float, axis=1)

    return data_float - offsets[:, np.newaxis]


def _correct_col_offset(
    data: np.ndarray, method: str = "median"
) -> np.ndarray:
    """
    Correct column-by-column offset (vertical striping).

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    method : str, optional
        Statistic to use: 'median' or 'mean'. Defaults to 'median'.

    Returns
    -------
    np.ndarray
        Corrected image in float32 (to avoid integer overflow during subtraction).
        Use preserve_dtype=True in apply_filter() to cast back to original dtype.

    Notes
    -----
    Computation is performed in float32 regardless of input dtype to prevent
    integer underflow when subtracting offsets.
    """
    _require_2d(data, "correct_col_offset")

    if method not in ("mean", "median"):
        raise ValueError(f"Method must be 'mean' or 'median', got '{method}'")

    data_float = data.astype(np.float32)

    if method == "median":
        offsets = np.median(data_float, axis=0)
    else:
        offsets = np.mean(data_float, axis=0)

    return data_float - offsets


def _apply_correct_row_median(data: np.ndarray) -> np.ndarray:
    """Apply row median offset correction."""
    return _correct_row_offset(data, method="median")


def _apply_correct_row_mean(data: np.ndarray) -> np.ndarray:
    """Apply row mean offset correction."""
    return _correct_row_offset(data, method="mean")


def _apply_correct_col_median(data: np.ndarray) -> np.ndarray:
    """Apply column median offset correction."""
    return _correct_col_offset(data, method="median")


def _apply_correct_col_mean(data: np.ndarray) -> np.ndarray:
    """Apply column mean offset correction."""
    return _correct_col_offset(data, method="mean")

# ============================================================================
# Enhancement Filters
# ============================================================================


def _apply_unsharp_mask(
    data: np.ndarray, sigma: float = 1.0, amount: float = 1.0, **kwargs
) -> np.ndarray:
    """
    Apply unsharp masking for image sharpening.

    Enhances edges by subtracting a blurred version of the image.
    Computation is performed in float32 to avoid integer overflow.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigma : float
        Standard deviation for Gaussian blur. Default is 1.0.
    amount : float
        Strength of sharpening effect. Default is 1.0.
    **kwargs
        Additional arguments passed to gaussian_filter.

    Returns
    -------
    np.ndarray
        Sharpened image in float32 (or original dtype if preserve_dtype=True
        is used in apply_filter).

    Notes
    -----
    Uses scipy.ndimage.gaussian_filter for the blur step.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    """
    sigma_val = validate_numeric(sigma, "sigma", minimum=0.0, allow_zero=False)
    amount_val = validate_numeric(amount, "amount", minimum=0.0)

    # Work in float to avoid integer overflow
    work = np.asarray(data, dtype=np.float32)
    blurred = _validated_call(gaussian_filter, work, sigma=sigma_val, **kwargs)
    sharpened = work + amount_val * (work - blurred)

    return sharpened


def _apply_high_pass(
    data: np.ndarray, sigma: float = 3.0, **kwargs
) -> np.ndarray:
    """
    Apply high-pass filter to remove large-scale structure.

    Computation is performed in float32 to avoid integer overflow.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigma : float
        Standard deviation for low-pass Gaussian filter. Default is 3.0.
    **kwargs
        Additional arguments passed to gaussian_filter.

    Returns
    -------
    np.ndarray
        High-pass filtered image in float32.

    Notes
    -----
    Uses scipy.ndimage.gaussian_filter for the low-pass component.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    """
    sigma_val = validate_numeric(sigma, "sigma", minimum=0.0, allow_zero=False)

    # Work in float to avoid integer overflow
    work = np.asarray(data, dtype=np.float32)
    low_pass = _validated_call(gaussian_filter, work, sigma=sigma_val, **kwargs)
    high_pass = work - low_pass

    return high_pass


# ============================================================================
# Operation Registry
# ============================================================================

OPERATIONS: Dict[str, Callable[..., np.ndarray]] = {
    # Smoothing
    "gaussian": _apply_gaussian_filter,
    "median": _apply_median_filter,
    "mean": _apply_mean_filter,
    "uniform": _apply_mean_filter,  # Alias
    "bilateral": _apply_bilateral_filter,
    # Edge detection
    "sobel": _apply_sobel_filter,
    "laplacian": _apply_laplacian_filter,
    # Morphological
    "opening": _apply_opening,
    "closing": _apply_closing,
    "erosion": _apply_erosion,
    "dilation": _apply_dilation,
    "binary_opening": _apply_opening,  # Backward compatibility
    "binary_closing": _apply_closing,  # Backward compatibility
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
# Private Worker Functions
# ============================================================================

WorkDtype = Literal["auto", "float32", "float64"]


def _select_work_dtype(orig_dtype: np.dtype, work_dtype: WorkDtype) -> np.dtype:
    """
    Select the floating-point dtype used for filter computations.

    Parameters
    ----------
    orig_dtype
        Original numpy dtype of the image data.
    work_dtype
        Working dtype policy:
        - "float32": always compute in float32
        - "float64": always compute in float64
        - "auto": compute in float64 only if input is float64; otherwise float32

    Returns
    -------
    numpy.dtype
        The dtype to use for intermediate computations.

    Notes
    -----
    This function never returns an integer dtype; it is intended to avoid
    integer overflow/underflow and to provide stable results for operations
    involving subtraction (e.g. high-pass, unsharp mask, bias removal).
    """
    if work_dtype == "float32":
        return np.dtype(np.float32)
    if work_dtype == "float64":
        return np.dtype(np.float64)

    # auto: keep float64 if provided; otherwise compute in float32
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
    """
    Apply a registered filter operation to a single image HDU in-place.

    This function mutates ``hdul_out`` by replacing ``hdul_out[hdu_index].data``
    with the filtered data and appending HISTORY cards to that HDU's header.

    Parameters
    ----------
    hdul_out
        Output HDUList (typically a clone/copy of the input HDUList).
    hdu_index
        Index of the image HDU to process.
    operation
        Name of the operation in the ``OPERATIONS`` registry.
    region
        Region specification (same semantics as ``fits_tools.io.region_to_slices``).
    preserve_dtype
        If True, the output image is cast back to the original dtype (with safe
        clipping/rounding for integers). If False, output dtype is the working
        float dtype selected by ``work_dtype``.
    work_dtype
        Working dtype policy for intermediate computations (see
        ``_select_work_dtype``).
    **kwargs
        Operation-specific keyword arguments passed to the operation function.

    Raises
    ------
    ImageProcessingError
        If the HDU has no data or region merge-back fails.
    ValueError
        If the HDU data is not 2D or is a non-numeric dtype.
    """
    hdu = hdul_out[hdu_index]

    if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        raise ValueError(f"hdu_index={hdu_index} is not an image HDU (type={type(hdu).__name__})")
    
    if hdu.data is None:
        raise ImageProcessingError(f"HDU {hdu_index} contains no image data")

    orig_full = np.asarray(hdu.data)
    if orig_full.ndim != 2:
        raise ValueError(
            f"{operation} expects 2D image data; got shape={orig_full.shape}"
        )

    orig_dtype = orig_full.dtype
    if not np.issubdtype(orig_dtype, np.number):
        raise ValueError(f"Unsupported data dtype for filtering: {orig_dtype}")

    # Compute in float32/float64 according to policy
    work_dtype_np = _select_work_dtype(orig_dtype, work_dtype)
    work_full = orig_full.astype(work_dtype_np, copy=False)

    row_slice, col_slice = region_to_slices(
        work_full.shape, hdu.header, region, filepath="<in-memory>"
    )
    view = work_full[row_slice, col_slice]

    # Apply operation and enforce compute dtype
    filtered = OPERATIONS[operation](view, **kwargs)
    filtered = np.asarray(filtered, dtype=work_dtype_np)

    # Safety: ensure operation returned the expected shape
    if filtered.shape != view.shape:
        raise ImageProcessingError(
            f"Filter {operation!r} returned shape {filtered.shape}, expected {view.shape}"
        )

    if preserve_dtype:
        # Cast filtered region back to original dtype
        filtered_cast = (
            filtered
            if filtered.dtype == orig_dtype
            else _safe_cast_to_dtype(filtered, orig_dtype)
        )

        if region in (None, "full"):
            out_data = filtered_cast
        else:
            out_full = np.array(orig_full, copy=True)  # keep unfiltered pixels in original dtype
            out_full[row_slice, col_slice] = filtered_cast
            out_data = out_full
    else:
        # Output uses working dtype (float32 or float64)
        if region in (None, "full"):
            out_data = filtered
        else:
            out_full = np.array(work_full, copy=True)  # keep unfiltered pixels in working float dtype
            out_full[row_slice, col_slice] = filtered
            out_data = out_full

    hdu.data = out_data

    add_history(hdu.header, f"Applied {operation} filter")
    add_history(hdu.header, f"Region={region!r}")
    add_history(hdu.header, f"Work dtype={work_dtype_np.name}, preserve_dtype={preserve_dtype}")
    if np.asarray(out_data).dtype != orig_dtype:
        add_history(hdu.header, f"Data dtype: {orig_dtype} -> {np.asarray(out_data).dtype}")

# ============================================================================
# Main Processing Functions
# ============================================================================

def filter_hdu(
    input_file: StrOrPath,
    output_file: StrOrPath,
    operation: str,
    region: Region = "full",
    hdu: Union[int, str, Literal["auto"]] = "auto",
    *,
    preserve_dtype: bool = False,
    work_dtype: WorkDtype = "auto",
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """
    Apply a spatial filter to a FITS file.

    Parameters
    ----------
    input_file : str or Path
        Input FITS file path.
    output_file : str or Path
        Output FITS file path. Can be absolute or relative to current
        working directory. Parent directories will be created if they
        don't exist (requires appropriate permissions).
    operation : str
        Filter operation to apply. Must be in OPERATIONS.
    region : str or tuple, optional
        Region to filter. May be:
        - 'full' (default)
        - 'data'/'bias'/'trim'/'dark' (section keywords)
        - (start_col, end_col, start_row, end_row) using 0-based Python slicing coords
    hdu : int or str or 'auto', optional
        HDU extension index, EXTNAME, or 'auto' to select the first SCI/IMAGE image HDU
        (fallback: first image HDU with data). Default is 'auto'.
    preserve_dtype : bool, optional
        If True, cast output back to input dtype with proper clipping and
        rounding. For integer dtypes, NaN/Inf values are replaced with 0,
        values are rounded and clipped to the valid range. Defaults to False.
    work_dtype : {'auto','float32','float64'}, optional
        Floating-point dtype used for intermediate computations.
        'auto' preserves float64 inputs; otherwise uses float32.
    overwrite : bool, optional
        If True, allow overwriting output files. Defaults to False.
    **kwargs
        Operation-specific parameters.

    Raises
    ------
    ImageProcessingError
        If filtering fails.
    ValueError
        If operation is not supported or parameters are invalid.
    FileHandlingError
        If input/output file operations fail.

    Notes
    -----
    **Data Type Handling:**
    Filtering is computed in float32 for integer/float32 inputs, and float64 
    for float64 inputs (unless overridden). Output dtype matches compute dtype 
    unless preserve_dtype=True
    
    When preserve_dtype=True:
    - Integer outputs: values are rounded to nearest integer and clipped to
      dtype range; NaN/Inf are replaced with 0
    - Float outputs: direct cast (no clipping needed)

    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different paths")

    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")

    # Optional: you can omit this since write_hdulist() creates parents too.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hdul = fits.open(input_path, mode="readonly", memmap=True)
    except Exception as e:
        raise FileHandlingError(
            f"FITS I/O error opening file: {e}",
            filepath=input_path,
        ) from e

    with hdul:
        hdul_out = fits.HDUList([h.copy() for h in hdul])
        hdu_index = _resolve_single_hdu(hdul_out, hdu)

        try:
            _apply_filter_hdu(
                hdul_out=hdul_out,
                hdu_index=hdu_index,
                operation=operation,
                region=region,
                preserve_dtype=preserve_dtype,
                work_dtype=work_dtype,
                **kwargs,
            )
            write_hdulist(output_path, hdul_out, overwrite=overwrite)
            logger.debug("Filtered image saved to %s", output_path)

        except (ValueError, ImageProcessingError, FileHandlingError):
            raise
        except Exception as e:
            raise ImageProcessingError(f"Filtering failed: {e}") from e


def filter_image_hdus(
    input_file: StrOrPath,
    output_file: StrOrPath,
    operation: str,
    region: Region = "full",
    include_hdunames: Sequence[str] = ("SCI", "IMAGE"),
    *,
    preserve_dtype: bool = False,
    work_dtype: WorkDtype = "auto",
    overwrite: bool = False,
    **kwargs: Any,
) -> list[int]:
    """
    Apply a spatial filter to all selected image extensions in a multi-extension FITS (MEF) file.

    Parameters
    ----------
    input_file : str or Path
        Input FITS file path.
    output_file : str or Path
        Output FITS file path. Can be absolute or relative to current
        working directory. Parent directories will be created if they
        don't exist (requires appropriate permissions).
    operation : str
        Filter operation to apply. Must be in OPERATIONS.
    region : str or tuple, optional
        Region to filter. May be:
        - 'full' (default)
        - 'data'/'bias'/'trim'/'dark' (section keywords)
        - (start_col, end_col, start_row, end_row) using 0-based Python slicing coords
    include_hdunames : sequence of str, optional
        EXTNAME values to select for processing (e.g. ('SCI', 'IMAGE')).
    preserve_dtype : bool, optional
        If True, cast output back to input dtype with proper clipping and
        rounding. For integer dtypes, NaN/Inf values are replaced with 0,
        values are rounded and clipped to the valid range. Defaults to False.
    work_dtype : {'auto','float32','float64'}, optional
        Floating-point dtype used for intermediate computations.
        'auto' preserves float64 inputs; otherwise uses float32.
    overwrite : bool, optional
        If True, allow overwriting output files. Defaults to False.
    **kwargs
        Operation-specific parameters.

    Raises
    ------
    ImageProcessingError
        If filtering fails.
    ValueError
        If operation is not supported or parameters are invalid.
    FileHandlingError
        If input/output file operations fail.

    Notes
    -----
    **Data Type Handling:**
    Filtering is computed in float32 for integer/float32 inputs, and float64 
    for float64 inputs (unless overridden). Output dtype matches compute dtype 
    unless preserve_dtype=True
    
    When preserve_dtype=True:
    - Integer outputs: values are rounded to nearest integer and clipped to
      dtype range; NaN/Inf are replaced with 0
    - Float outputs: direct cast (no clipping needed)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different paths")

    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hdul = fits.open(input_path, mode="readonly", memmap=True)
    except Exception as e:
        raise FileHandlingError(
            f"FITS I/O error opening file: {e}",
            filepath=input_path,
        ) from e

    with hdul:
        hdul_out = fits.HDUList([h.copy() for h in hdul])

        hdu_indices = [
            i for i, _ in iter_image_hdus(hdul_out, hduname=include_hdunames)
        ]
        if not hdu_indices:
            raise ValueError(f"No image HDUs found with EXTNAME in {include_hdunames}")

        try:
            for hdu_index in hdu_indices:
                _apply_filter_hdu(
                    hdul_out=hdul_out,
                    hdu_index=hdu_index,
                    operation=operation,
                    region=region,
                    preserve_dtype=preserve_dtype,
                    work_dtype=work_dtype,
                    **kwargs,
                )

            write_hdulist(output_path, hdul_out, overwrite=overwrite)
            logger.debug("Filtered MEF saved to %s", output_path)

        except (ValueError, ImageProcessingError, FileHandlingError):
            raise
        except Exception as e:
            raise ImageProcessingError(f"Filtering failed: {e}") from e

    return hdu_indices

# ============================================================================
# Batch Processing
# ============================================================================


def filter_hdu_batch(
    input_files: Union[StrOrPath, Iterable[StrOrPath]],
    output_pattern: StrOrPath,
    operation: str,
    *,
    region: Region = "full",
    hdu: Union[int, str, Literal["auto"]] = "auto",
    preserve_dtype: bool = False,
    work_dtype: WorkDtype = "auto",
    overwrite: bool = False,
    # batch specific parameters
    num_workers: int = 1,
    validate_outputs: bool = False,
    checksum_algorithm: str = DEFAULT_CHECKSUM_ALGORITHM,
    show_progress: bool = True,
    dry_run: bool = False,
    **kwargs: Any,
) -> BatchResult:
    """
    Apply filter to multiple FITS files in batch mode.

    Parameters
    ----------
    input_files : str, Path, or iterable of str/Path
        Directory, glob pattern, or list of files.
    output_pattern : str or Path
        Output naming pattern. Use {stem} for input filename stem,
        {index} for enumeration. Must be an absolute path or relative
        to current directory.
    operation : str
        Filter operation to apply.
    num_workers : int, optional
        Number of parallel workers. Defaults to 1.
    overwrite : bool, optional
        If True, allow overwriting output files. Defaults to False.
    validate_outputs : bool, optional
        If True, compute and store checksums of output files. Defaults to False.
    checksum_algorithm : str, optional
        Hash algorithm for validation (sha256, md5, etc.). Defaults to 'sha256'.
    show_progress : bool, optional
        If True, display progress bar. Defaults to True.
    dry_run : bool, optional
        If True, only validate inputs and preview operations without processing.
        Defaults to False.
    **kwargs :
        Operation-specific parameters passed to the selected filter 
        operation (e.g., sigma, size, amount).

    Returns
    -------
    BatchResult
        Object containing:
        - successful: List of successfully created output file paths
        - failed: List of (input_path, exception) tuples for failed files
        - checksums: Dict mapping output paths to checksums (if validate_outputs=True)

    Raises
    ------
    FileHandlingError
        If input file selection fails or output validation fails.
    ValueError
        If output_pattern is invalid or missing required placeholders,
        or if checksum_algorithm is not supported.

    Notes
    -----
    - For I/O-bound operations, num_workers=4-8 is typically optimal
    - For CPU-bound operations, num_workers=cpu_count() is recommended
    - Memory usage scales with num_workers * image_size
    - Progress bar adds ~1-2% overhead
    """
    reserved = {
        "input", "output", "operation", "hdu", "region", 
        "preserve_dtype", "work_dtype", "overwrite",
        "num_workers", "validate_outputs", "checksum_algorithm", "show_progress", "dry_run"
    }
    conflicts = reserved.intersection(kwargs)
    if conflicts:
        raise ValueError(f"filter_batch received reserved kwargs: {sorted(conflicts)}")

    # Validate inputs
    num_workers = validate_numeric(num_workers, "num_workers", minimum=1, integer=True)

    # Validate checksum algorithm
    if validate_outputs:
        try:
            hashlib.new(checksum_algorithm)
        except ValueError as e:
            raise ValueError(
                f"Unsupported checksum algorithm '{checksum_algorithm}'. "
                f"Available: {', '.join(hashlib.algorithms_available)}"
            ) from e

    # Convert output_pattern to string for formatting
    output_pattern_str = str(output_pattern)

    # Validate output pattern
    if "{stem}" not in output_pattern_str:
        raise ValueError(
            "output_pattern must contain '{stem}' placeholder for input filename stem"
        )

    # Test format string validity
    try:
        output_pattern_str.format(stem="test", index=0)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid output_pattern format string: {e}") from e

    # Select files
    try:
        files = select_files(input_files)
    except Exception as e:
        raise FileHandlingError(f"Failed to select input files: {e}") from e

    if not files:
        logger.warning("No input files found matching: %s", input_files)
        return BatchResult(successful=[], failed=[], checksums={})

    logger.info(
        "%s %d files with %d workers",
        "Would process" if dry_run else "Processing",
        len(files),
        num_workers,
    )

    # Prepare tasks and validate outputs
    tasks = []
    output_files: List[Path] = []
    input_paths: List[Path] = []

    for idx, infile in enumerate(files):
        # Ensure infile is a Path
        infile_path = Path(infile)
        input_paths.append(infile_path)
        stem = infile_path.stem

        # Create output path
        output_str = output_pattern_str.format(stem=stem, index=idx)
        output_path = Path(output_str)

        # Check for output directory
        output_dir = output_path.parent
        if not dry_run:
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    logger.debug("Created output directory: %s", output_dir)
                except OSError as e:
                    raise FileHandlingError(
                        f"Cannot create output directory {output_dir}: {e}"
                    ) from e
        else:
            # In dry run, just check if directory would be writable
            if not output_dir.exists():
                logger.info("Would create directory: %s", output_dir)

        # Check for existing files if overwrite is False
        if not overwrite and output_path.exists():
            if dry_run:
                logger.warning("Output file exists: %s", output_path)
            else:
                raise FileHandlingError(
                    f"Output file already exists and overwrite=False: {output_path}"
                )

        output_files.append(output_path)

        tasks.append(
            {
                "input": str(infile_path),  # Convert to str for serialization
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

    # Dry run: print summary and return
    if dry_run:
        logger.info("=== DRY RUN SUMMARY ===")
        logger.info("Input files: %d", len(input_paths))
        logger.info("Operation: %s", operation)
        logger.info("Parameters: %s", kwargs)
        logger.info("Sample mappings:")
        for inp, out in zip(input_paths[:3], output_files[:3]):
            logger.info("  %s -> %s", inp.name, out.name)
        if len(input_paths) > 3:
            logger.info("  ... and %d more", len(input_paths) - 3)
        logger.info("======================")

        # Return empty result for dry run
        return BatchResult(successful=[], failed=[], checksums={})

    # Process files with error tracking
    results: List[ProcessingResult] = []

    # Progress bar setup
    progress_bar = None
    if show_progress:
        try:
            progress_bar = tqdm(
                total=len(tasks),
                desc="Processing files",
                unit="file",
                dynamic_ncols=True,
            )
        except Exception as e:
            logger.warning("Could not create progress bar: %s", e)
            show_progress = False

    try:
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(_process_single_filter_task, task): (idx, task)
                    for idx, task in enumerate(tasks)
                }

                # Collect results
                for future in as_completed(future_to_task):
                    idx, task = future_to_task[future]
                    input_path = input_paths[idx]
                    output_path = output_files[idx]

                    try:
                        future.result()

                        # Compute checksum if requested
                        checksum = None
                        if validate_outputs and output_path.exists():
                            try:
                                checksum = _compute_file_checksum(
                                    output_path, checksum_algorithm
                                )
                            except Exception as e:
                                logger.warning(
                                    "Failed to compute checksum for %s: %s",
                                    output_path,
                                    e,
                                )

                        results.append(
                            ProcessingResult(
                                input_file=input_path,
                                output_file=output_path,
                                success=True,
                                checksum=checksum,
                            )
                        )

                    except Exception as e:
                        logger.error(
                            "Failed to process %s -> %s (operation=%s): %s",
                            task["input"],
                            task["output"],
                            task["operation"],
                            str(e),
                        )
                        results.append(
                            ProcessingResult(
                                input_file=input_path,
                                output_file=output_path,
                                success=False,
                                error=e,
                            )
                        )

                    if show_progress and progress_bar:
                        progress_bar.update(1)
        else:
            for idx, task in enumerate(tasks):
                input_path = input_paths[idx]
                output_path = output_files[idx]

                try:
                    _process_single_filter_task(task)

                    # Compute checksum if requested
                    checksum = None
                    if validate_outputs and output_path.exists():
                        try:
                            checksum = _compute_file_checksum(
                                output_path, checksum_algorithm
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to compute checksum for %s: %s",
                                output_path,
                                e,
                            )

                    results.append(
                        ProcessingResult(
                            input_file=input_path,
                            output_file=output_path,
                            success=True,
                            checksum=checksum,
                        )
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process %s -> %s (operation=%s): %s",
                        task["input"],
                        task["output"],
                        task["operation"],
                        str(e),
                    )
                    results.append(
                        ProcessingResult(
                            input_file=input_path,
                            output_file=output_path,
                            success=False,
                            error=e,
                        )
                    )

                if show_progress and progress_bar:
                    progress_bar.update(1)
    finally:
        if show_progress and progress_bar:
            progress_bar.close()

    # Compile results
    successful = [r.output_file for r in results if r.success]
    failed = [(r.input_file, r.error) for r in results if not r.success]
    checksums = {
        r.output_file: r.checksum
        for r in results
        if r.success and r.checksum is not None
    }

    # Report summary
    if failed:
        logger.warning(
            "Completed with %d failures out of %d files (%.1f%% success rate)",
            len(failed),
            len(results),
            100 * len(successful) / len(results),
        )
    else:
        logger.info("Successfully processed all %d files", len(results))

    if validate_outputs:
        logger.info("Computed checksums for %d output files", len(checksums))

    batch_result = BatchResult(
        successful=successful, failed=failed, checksums=checksums
    )

    return batch_result


def save_batch_report(
    result: BatchResult, report_path: StrOrPath, include_checksums: bool = True
) -> None:
    """
    Save batch processing report to a file.

    Parameters
    ----------
    result : BatchResult
        Batch processing result to save.
    report_path : str or Path
        Path to save report file (text or JSON based on extension).
    include_checksums : bool, optional
        Include checksum information in report. Defaults to True.

    Examples
    --------
    >>> result = filter_batch(...)  # doctest: +SKIP
    >>> save_batch_report(result, 'processing_report.txt')  # doctest: +SKIP
    >>> save_batch_report(result, 'processing_report.json')  # doctest: +SKIP
    """
    import json
    from datetime import datetime

    report_path = Path(report_path)

    # Ensure parent directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    if report_path.suffix.lower() == ".json":
        # JSON format
        report_data = {
            "timestamp": timestamp,
            "summary": {
                "total": result.total_count,
                "successful": result.success_count,
                "failed": result.failure_count,
                "success_rate": result.success_rate,
            },
            "successful_files": [str(p) for p in result.successful],
            "failed_files": [
                {"input": str(inp), "error": str(err)} for inp, err in result.failed
            ],
        }

        if include_checksums and result.checksums:
            report_data["checksums"] = {
                str(k): v for k, v in result.checksums.items()
            }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
    else:
        # Text format
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Batch Processing Report\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"{'=' * 60}\n\n")

            f.write("Summary:\n")
            f.write(f"  Total files: {result.total_count}\n")
            f.write(f"  Successful: {result.success_count}\n")
            f.write(f"  Failed: {result.failure_count}\n")
            f.write(f"  Success rate: {result.success_rate:.1%}\n\n")

            if result.successful:
                f.write(f"Successful files ({len(result.successful)}):\n")
                for path in result.successful:
                    f.write(f"  ✓ {path}\n")
                    if include_checksums and path in result.checksums:
                        f.write(f"    {result.checksums[path]}\n")
                f.write("\n")

            if result.failed:
                f.write(f"Failed files ({len(result.failed)}):\n")
                for inp, err in result.failed:
                    f.write(f"  ✗ {inp}\n")
                    f.write(f"    Error: {err}\n")

    logger.info("Report saved to %s", report_path)


def verify_output_checksums(
    result: BatchResult, checksum_algorithm: str = DEFAULT_CHECKSUM_ALGORITHM
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Verify output files match their recorded checksums.

    Parameters
    ----------
    result : BatchResult
        Result containing checksums to verify.
    checksum_algorithm : str, optional
        Algorithm used for checksums. Defaults to 'sha256'.

    Returns
    -------
    tuple of (verified, corrupted)
        verified : List of Path objects that passed verification
        corrupted : List of (Path, error_msg) tuples for failed verifications

    Examples
    --------
    >>> result = filter_batch(..., validate_outputs=True)  # doctest: +SKIP
    >>> verified, corrupted = verify_output_checksums(result)  # doctest: +SKIP
    >>> if corrupted:  # doctest: +SKIP
    ...     print(f"Warning: {len(corrupted)} files may be corrupted")
    """
    if not result.checksums:
        logger.warning("No checksums to verify in result")
        return [], []

    verified = []
    corrupted = []

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
                    (
                        filepath,
                        f"Checksum mismatch: expected {original_checksum[:16]}..., "
                        f"got {current_checksum[:16]}...",
                    )
                )
        except Exception as e:
            corrupted.append((filepath, f"Verification failed: {e}"))

    return verified, corrupted

# ============================================================================
# Operations List Access
# ============================================================================

def list_operations() -> Dict[str, str]:
    """
    List all available filter operations with brief descriptions.

    Returns
    -------
    dict
        Mapping of operation name to one-line description.
        Keys match those in the OPERATIONS registry.

    Examples
    --------
    >>> from fits_tools.filters import list_operations
    >>> ops = list_operations()
    >>> print(ops['gaussian'])
    Gaussian smoothing filter (sigma: blur strength)

    >>> # List all operations by category
    >>> for name, desc in sorted(ops.items()):
    ...     print(f"{name:20s} - {desc}")  # doctest: +SKIP

    >>> # Check if an operation exists
    >>> if 'median' in list_operations():
    ...     print("Median filter is available")  # doctest: +SKIP
    """
    return {
        # Smoothing filters
        "gaussian": "Gaussian smoothing filter (sigma: blur strength)",
        "median": "Median filter for noise reduction (size: kernel size)",
        "mean": "Mean/uniform smoothing filter (size: kernel size)",
        "uniform": "Alias for mean filter",
        "bilateral": "Edge-preserving smoothing (sigma_spatial, sigma_intensity)",
        # Edge detection
        "sobel": "Sobel edge detection (axis: 0=horiz, 1=vert, None=magnitude)",
        "laplacian": "Laplacian edge detection (second derivative)",
        # Morphological operations
        "opening": "Morphological opening - removes small bright features",
        "closing": "Morphological closing - removes small dark features",
        "erosion": "Morphological erosion - shrinks bright regions",
        "dilation": "Morphological dilation - expands bright regions",
        "binary_opening": "Alias for opening (backward compatibility)",
        "binary_closing": "Alias for closing (backward compatibility)",
        # Offset correction
        "correct_row_median": "Remove row-wise median offset (horizontal striping)",
        "correct_row_mean": "Remove row-wise mean offset (horizontal striping)",
        "correct_col_median": "Remove column-wise median offset (vertical striping)",
        "correct_col_mean": "Remove column-wise mean offset (vertical striping)",
        # Enhancement
        "unsharp_mask": "Image sharpening via unsharp masking (sigma, amount)",
        "high_pass": "High-pass filter - removes large-scale structure (sigma)",
    }


def print_operations(category: Optional[str] = None) -> None:
    """
    Print available operations in a formatted table.

    Parameters
    ----------
    category : str, optional
        Filter by category: 'smoothing', 'edge', 'morphological',
        'correction', 'enhancement'. If None, shows all.

    Examples
    --------
    >>> print_operations()  # doctest: +SKIP
    >>> print_operations('smoothing')  # doctest: +SKIP
    """
    ops = list_operations()

    categories = {
        "smoothing": ["gaussian", "median", "mean", "uniform", "bilateral"],
        "edge": ["sobel", "laplacian"],
        "morphological": [
            "opening",
            "closing",
            "erosion",
            "dilation",
            "binary_opening",
            "binary_closing",
        ],
        "correction": [
            "correct_row_median",
            "correct_row_mean",
            "correct_col_median",
            "correct_col_mean",
        ],
        "enhancement": ["unsharp_mask", "high_pass"],
    }

    if category:
        if category not in categories:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Available: {', '.join(categories.keys())}"
            )
        names = categories[category]
        print(f"\n{category.upper()} OPERATIONS:")
    else:
        print("\nAVAILABLE FILTER OPERATIONS:")
        names = sorted(ops.keys())

    print("=" * 80)
    for name in names:
        print(f"  {name:20s} {ops[name]}")
    print()


def get_operations_by_category() -> Dict[str, List[str]]:
    """
    Get available operations grouped by category.

    Returns
    -------
    dict
        Mapping of category name to list of operation names.

    Examples
    --------
    >>> from fits_tools.filters import get_operations_by_category
    >>> cats = get_operations_by_category()
    >>> print(cats['smoothing'])
    ['gaussian', 'median', 'mean', 'uniform', 'bilateral']
    """
    return {
        "smoothing": ["gaussian", "median", "mean", "uniform", "bilateral"],
        "edge": ["sobel", "laplacian"],
        "morphological": [
            "opening",
            "closing",
            "erosion",
            "dilation",
            "binary_opening",
            "binary_closing",
        ],
        "correction": [
            "correct_row_median",
            "correct_row_mean",
            "correct_col_median",
            "correct_col_mean",
        ],
        "enhancement": ["unsharp_mask", "high_pass"],
    }

# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Data classes (useful for type hints)
    "ProcessingResult",
    "BatchResult",
    # Main functions
    "configure_logging",
    "list_operations",
    "print_operations",
    "get_operations_by_category",
    "apply_filter",
    "apply_filter_mef",
    "filter_batch",
    "save_batch_report",
    "verify_output_checksums",
    # Registry
    "OPERATIONS",
]
