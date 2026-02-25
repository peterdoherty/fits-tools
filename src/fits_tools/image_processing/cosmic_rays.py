#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosmic ray detection and rejection for CCD/CMOS images.

This module provides algorithms for detecting and removing cosmic ray hits
from astronomical images.

Detection methods:
- Sigma-clipping: statistical outlier detection with median filtering
- L.A.Cosmic: Laplacian edge detection algorithm (via astroscrappy)
- Median filtering: simple median-based outlier detection
"""

from __future__ import annotations

from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

# Optional dependencies
try:
    from scipy.ndimage import median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    median_filter = None
    uniform_filter = None

try:
    import astroscrappy
    LACOSMIC_AVAILABLE = True
except ImportError:
    LACOSMIC_AVAILABLE = False
    astroscrappy = None

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from ..io import get_data, get_data_and_header, write_fits
from ..exceptions import ImageProcessingError
from ..utils import add_history, maybe_tqdm, is_tqdm_available
from ..types import StrOrPath

MAD_TO_STD_FACTOR = 1.4826  # For normal distribution

logger = logging.getLogger(__name__)
__version__ = "2.0.0"
__author__ = "FITS Tools Development Team"


# ============================================================================
# Defaults for LACosmic
# ============================================================================

@dataclass
class LACosmicConfig:
    sigclip: float = 4.5
    sigfrac: float = 0.3
    objlim: float = 5.0
    gain: float = 1.0
    readnoise: float = 6.5
    satlevel: float = 65535.0

# ============================================================================
# Result Types
# ============================================================================

@dataclass
class CosmicRayResult:
    """Result from cosmic ray detection/rejection."""
    cleaned_data: np.ndarray
    mask: np.ndarray
    num_cosmic_rays: int
    cosmic_ray_fraction: float
    method: str
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            "Cosmic Ray Result:\n"
            f"  Method: {self.method}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Cosmic rays detected: {self.num_cosmic_rays} "
            f"({self.cosmic_ray_fraction*100:.3f}%)\n"
            f"  Cleaned data shape: {self.cleaned_data.shape}"
        )

    def __repr__(self) -> str:
        return (
            f"CosmicRayResult(method='{self.method}', "
            f"num_cosmic_rays={self.num_cosmic_rays}, "
            f"fraction={self.cosmic_ray_fraction:.4f})"
        )


# ============================================================================
# Utility
# ============================================================================

def _calculate_mad(data: np.ndarray) -> float:
    """Robust estimator of standard deviation via MAD."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return MAD_TO_STD_FACTOR * mad


# ============================================================================
# Detection Methods
# ============================================================================

def _detect_cosmic_rays_sigma_clip(
    data: np.ndarray,
    sigma: float = 5.0,
    box_size: int = 5,
    iterations: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect cosmic rays using sigma-clipping on median-filtered residuals."""
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "Sigma-clipping cosmic ray detection requires scipy. "
            "Install with: pip install scipy"
        )

    if box_size % 2 == 0:
        box_size += 1
        logger.debug("Adjusted box_size to %d (must be odd)", box_size)

    cosmic_ray_mask = np.zeros(data.shape, dtype=bool)
    data_copy = data.astype(np.float32, copy=True)
    threshold = None  # track last threshold for metadata

    for iteration in range(iterations):
        median_filtered = median_filter(data_copy, size=box_size)
        residuals = data_copy - median_filtered

        median_residual = np.median(residuals)
        mad = _calculate_mad(residuals)

        if mad == 0:
            logger.warning("Iteration %d: MAD is zero, stopping", iteration + 1)
            break

        threshold = median_residual + sigma * mad
        new_cosmic_rays = residuals > threshold
        num_new = int(np.sum(new_cosmic_rays))

        logger.debug(
            "Iteration %d: detected %d cosmic ray pixels",
            iteration + 1,
            num_new,
        )

        if num_new == 0:
            logger.debug(
                "No new cosmic rays detected, stopping after %d iterations",
                iteration + 1,
            )
            break

        cosmic_ray_mask |= new_cosmic_rays
        data_copy[new_cosmic_rays] = median_filtered[new_cosmic_rays]

    metadata = {
        "sigma": sigma,
        "box_size": box_size,
        "iterations_performed": iteration + 1,
        "threshold_value": float(threshold) if threshold is not None else None,
    }
    return data_copy, cosmic_ray_mask, metadata


def _detect_cosmic_rays_lacosmic(
    data: np.ndarray,
    sigclip: float = 4.5,
    sigfrac: float = 0.3,
    objlim: float = 5.0,
    gain: float = 1.0,
    readnoise: float = 6.5,
    satlevel: float = 65535.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Detect cosmic rays using the L.A.Cosmic algorithm via astroscrappy."""
    if not LACOSMIC_AVAILABLE:
        raise ImportError(
            "L.A.Cosmic detection requires astroscrappy. "
            "Install with: pip install astroscrappy"
        )

    logger.debug(
        "Running L.A.Cosmic with sigclip=%.2f, objlim=%.2f", sigclip, objlim
    )

    mask, cleaned = astroscrappy.detect_cosmics(
        data,
        sigclip=sigclip,
        sigfrac=sigfrac,
        objlim=objlim,
        gain=gain,
        readnoise=readnoise,
        satlevel=satlevel,
    )

    metadata = {
        "sigclip": sigclip,
        "sigfrac": sigfrac,
        "objlim": objlim,
        "gain": gain,
        "readnoise": readnoise,
        "satlevel": satlevel,
    }
    return cleaned, mask, metadata


def _detect_cosmic_rays_median(
    data: np.ndarray,
    threshold: float = 100.0,
    box_size: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simple cosmic ray detection using median filtering."""
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "Median filtering requires scipy. Install with: pip install scipy"
        )

    if box_size % 2 == 0:
        box_size += 1

    median_filtered = median_filter(data, size=box_size)
    residuals = data - median_filtered
    cosmic_ray_mask = residuals > threshold

    metadata = {
        "threshold": threshold,
        "box_size": box_size,
    }

    logger.debug("Detected %d cosmic ray pixels", int(np.sum(cosmic_ray_mask)))

    return median_filtered, cosmic_ray_mask, metadata


# ============================================================================
# Replacement Methods
# ============================================================================

def _replace_with_median(
    data: np.ndarray,
    mask: np.ndarray,
    box_size: int = 5,
) -> np.ndarray:
    """Replace cosmic ray pixels with local median."""
    if not SCIPY_AVAILABLE:
        raise ImportError("Median replacement requires scipy")

    if box_size % 2 == 0:
        box_size += 1

    cleaned = data.copy()
    median_filtered = median_filter(data, size=box_size)
    cleaned[mask] = median_filtered[mask]
    return cleaned


def _replace_with_interpolation(
    data: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Replace cosmic ray pixels using interpolation from neighbors."""

    if data.ndim != 2:
        raise ImageProcessingError(
            "Interpolation replacement requires a 2D array.",
            operation="cosmic ray interpolation"
        )

    try:
        from scipy.interpolate import griddata
    except ImportError as e:
        raise ImportError(
            "Interpolation requires scipy (scipy.interpolate.griddata). "
            "Install with: pip install scipy"
        ) from e

    cleaned = data.copy()
    ny, nx = data.shape
    y, x = np.mgrid[0:ny, 0:nx]

    good_mask = ~mask
    points = np.column_stack([y[good_mask], x[good_mask]])
    values = data[good_mask]

    cr_points = np.column_stack([y[mask], x[mask]])

    interpolated = griddata(points, values, cr_points, method="linear")

    nan_mask = np.isnan(interpolated)
    if nan_mask.any():
        interpolated[nan_mask] = griddata(
            points, values, cr_points[nan_mask], method="nearest"
        )

    cleaned[mask] = interpolated
    return cleaned


# ============================================================================
# Public Detection / Rejection
# ============================================================================

def detect_cosmic_rays(
    data: np.ndarray,
    method: str = "sigma_clip",
    sigma: float = 5.0,
    iterations: int = 1,
    box_size: int = 5,
    lacosmic_config: Optional[LACosmicConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Detect cosmic rays in image data (detection only, no replacement).
    
    """

    if method == "sigma_clip":
        return _detect_cosmic_rays_sigma_clip(
            data,
            sigma=sigma,
            box_size=box_size,
            iterations=iterations,
        )
    if method == "lacosmic":
        cfg = lacosmic_config or LACosmicConfig()
        return _detect_cosmic_rays_lacosmic(
            data,
            sigclip=cfg.sigclip,
            sigfrac=cfg.sigfrac,
            objlim=cfg.objlim,
            gain=cfg.gain,
            readnoise=cfg.readnoise,
            satlevel=cfg.satlevel,
        )

    if method == "median":
        return _detect_cosmic_rays_median(data, box_size=box_size)

    raise ValueError(
        f"Unknown cosmic ray detection method: '{method}'. "
        "Valid options: 'sigma_clip', 'lacosmic', 'median'"
    )


def reject_cosmic_rays(
    data: np.ndarray,
    method: str = "sigma_clip",
    sigma: float = 5.0,
    iterations: int = 1,
    box_size: int = 5,
    replace_with: str = "median",
    lacosmic_config: Optional[LACosmicConfig] = None,
) -> CosmicRayResult:
    """Detect and remove cosmic rays from image data."""
    logger.debug("Detecting cosmic rays using method: %s", method)

    cleaned, mask, metadata = detect_cosmic_rays(
        data,
        method=method,
        sigma=sigma,
        iterations=iterations,
        box_size=box_size,
        lacosmic_config=lacosmic_config,
    )

    num_cosmic_rays = int(np.sum(mask))
    cosmic_ray_fraction = num_cosmic_rays / data.size

    if num_cosmic_rays > 0:
        if method == "lacosmic":
            cleaned_data = cleaned   # use astroscrappy's own output
        elif replace_with == "median":
            cleaned_data = _replace_with_median(data, mask, box_size=box_size)
        elif replace_with == "interpolation":
            cleaned_data = _replace_with_interpolation(data, mask)
        else:
            raise ValueError
    
    logger.debug(
        "Detected %d cosmic ray pixels (%.3f%%)",
        num_cosmic_rays,
        cosmic_ray_fraction * 100.0,
    )

    if num_cosmic_rays > 0:
        logger.debug("Replacing cosmic rays using method: %s", replace_with)

        if replace_with == "median":
            cleaned_data = _replace_with_median(data, mask, box_size=box_size)
        elif replace_with == "interpolation":
            cleaned_data = _replace_with_interpolation(data, mask)
        else:
            raise ValueError(
                f"Unknown replacement method: '{replace_with}'. "
                "Valid options: 'median', 'interpolation'"
            )
    else:
        cleaned_data = data.copy()
        logger.debug("No cosmic rays detected, returning original data")

    return CosmicRayResult(
        cleaned_data=cleaned_data,
        mask=mask,
        num_cosmic_rays=num_cosmic_rays,
        cosmic_ray_fraction=cosmic_ray_fraction,
        method=method,
        iterations=metadata.get("iterations_performed", iterations),
        metadata=metadata,
    )


# ============================================================================
# File-level operations
# ============================================================================

def clean_cosmic_rays_file(
    input_file: StrOrPath,
    output_file: Optional[StrOrPath] = None,
    method: str = "sigma_clip",
    sigma: float = 5.0,
    iterations: int = 1,
    ext: int = 0,
    overwrite: bool = False,
    save_mask: bool = False,
    lacosmic_config: Optional[LACosmicConfig] = None,
) -> CosmicRayResult:
    """
    Clean cosmic rays from a FITS file.

    Always returns a CosmicRayResult. If output_file is provided,
    the cleaned data is also written to disk.
    """
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file is not None else None

    logger.debug("Cleaning cosmic rays from: %s", input_path.name)

    if output_path is not None and output_path.exists() and not overwrite:
        raise ImageProcessingError(
            f"Output file exists and overwrite=False: {output_path}"
        )

    try:
        data, header = get_data_and_header(input_path, ext=ext)
        if data is None:
            raise ImageProcessingError(
                f"HDU {ext} in {input_path.name} contains no image data"
            )

        result = reject_cosmic_rays(
            data,
            method=method,
            sigma=sigma,
            iterations=iterations,
            lacosmic_config=lacosmic_config,
        )

        add_history(header, f"Cosmic rays removed using {method} method")
        header["CRMETH"] = (method, "Cosmic ray detection method")
        header["CRNUM"] = (result.num_cosmic_rays, "Number of cosmic ray pixels")
        header["CRFRAC"] = (result.cosmic_ray_fraction, "Cosmic ray fraction")
        header["CRITER"] = (result.iterations, "CR detection iterations")

        if method == "sigma_clip":
            header["CRSIGMA"] = (sigma, "CR detection sigma threshold")

        if output_path is not None:
            if save_mask:
                primary_hdu = fits.PrimaryHDU(result.cleaned_data, header=header)
                mask_hdu = fits.ImageHDU(
                    result.mask.astype(np.uint8),
                    name="CR_MASK",
                )
                hdul = fits.HDUList([primary_hdu, mask_hdu])
                hdul.writeto(output_path, overwrite=overwrite)
                logger.debug("Wrote cleaned data and mask to: %s", output_path.name)
            else:
                write_fits(output_path, result.cleaned_data, header, overwrite=overwrite)

        return result

    except Exception as e:
        logger.error("Failed to clean cosmic rays from %s: %s", input_path.name, e)
        raise ImageProcessingError(f"Cosmic ray cleaning failed: {e}") from e


# ============================================================================
# Batch Processing
# ============================================================================

def batch_clean_cosmic_rays(
    file_list: List[StrOrPath],
    output_dir: StrOrPath,
    method: str = "sigma_clip",
    sigma: float = 5.0,
    iterations: int = 1,
    ext: int = 0,
    overwrite: bool = False,
    save_masks: bool = False,
    lacosmic_config: Optional[LACosmicConfig] = None,
) -> Dict[str, Any]:
    """
    Clean cosmic rays from multiple files.

    Returns a dict with keys:
      'total_files', 'successful', 'failed', 'total_cosmic_rays',
      'output_files', 'errors'.
    """
    if not file_list:
        raise ValueError("Empty file list")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Batch cleaning %d files...", len(file_list))

    results: Dict[str, Any] = {
        "total_files": len(file_list),
        "successful": 0,
        "failed": 0,
        "total_cosmic_rays": 0,
        "output_files": [],
        "errors": [],
    }

    for i, filepath in maybe_tqdm(enumerate(file_list, 1), total=len(file_list)):
        filepath = Path(filepath)
        try:
            output_file = output_dir / f"{filepath.stem}_clean{filepath.suffix}"

            result = clean_cosmic_rays_file(
                filepath,
                method=method,
                sigma=sigma,
                iterations=iterations,
                ext=ext,
                lacosmic_config=lacosmic_config,
            )

            write_fits(output_file, result.cleaned_data, overwrite=overwrite)

            if save_masks:
                mask_file = output_dir / f"{filepath.stem}_crmask{filepath.suffix}"
                write_fits(mask_file, result.mask.astype(np.uint8), overwrite=overwrite)

            results["successful"] += 1
            results["total_cosmic_rays"] += result.num_cosmic_rays
            results["output_files"].append(output_file)

            if not is_tqdm_available():
                logger.debug("[%d/%d] ✓ %s", i, len(file_list), filepath.name)

        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{filepath.name}: {e}")

            if not is_tqdm_available():
                logger.error("[%d/%d] ✗ %s: %s", i, len(file_list), filepath.name, e)

    logger.info("Batch cleaning complete:")
    logger.info("  Successful: %d/%d", results["successful"], results["total_files"])
    logger.info("  Failed: %d", results["failed"])
    if results["total_cosmic_rays"] > 0:
        logger.info("  Total cosmic rays removed: %d", results["total_cosmic_rays"])

    return results

# ============================================================================
# Convenience / Introspection
# ============================================================================

def get_available_methods() -> Dict[str, Dict[str, Any]]:
    """Return information about available cosmic ray detection methods."""
    return {
        "sigma_clip": {
            "name": "Sigma-clipping",
            "description": "Statistical outlier detection with median filtering",
            "available": SCIPY_AVAILABLE,
            "requires": "scipy",
            "recommended_for": "General use, single exposures",
        },
        "lacosmic": {
            "name": "L.A.Cosmic",
            "description": "Laplacian edge detection algorithm",
            "available": LACOSMIC_AVAILABLE,
            "requires": "astroscrappy",
            "recommended_for": "High-quality cosmic ray removal",
            "reference": "van Dokkum 2001, PASP, 113, 1420",
        },
        "median": {
            "name": "Median filtering",
            "description": "Simple median-based detection",
            "available": SCIPY_AVAILABLE,
            "requires": "scipy",
            "recommended_for": "Quick processing, isolated cosmic rays",
        },
    }


def check_dependencies() -> Dict[str, bool]:
    """Check availability of optional dependencies."""
    return {
        "scipy": SCIPY_AVAILABLE,
        "astroscrappy": LACOSMIC_AVAILABLE,
        "tqdm": _tqdm is not None,
    }


__all__ = [
    "CosmicRayResult",
    "detect_cosmic_rays",
    "reject_cosmic_rays",
    "clean_cosmic_rays_file",
    "batch_clean_cosmic_rays",
    "get_available_methods",
    "check_dependencies",
    "SCIPY_AVAILABLE",
    "LACOSMIC_AVAILABLE",
]