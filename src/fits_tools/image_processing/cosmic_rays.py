#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Cosmic ray detection and rejection for CCD/CMOS images.

This module provides algorithms for detecting and removing cosmic ray hits
from astronomical images.

Detection methods
-----------------
- ``sigma_clip`` : sigma-clipping on residuals after median filtering (SciPy)
- ``lacosmic``   : L.A.Cosmic via astroscrappy
- ``median``     : median-filter residual thresholding (SciPy)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from ..cli_utils import is_tqdm_available, make_output_path, maybe_tqdm
from ..exceptions import FileOverwriteError, ImageProcessingError
from ..header_utils import add_history
from ..io import get_data_and_header, write_hdulist, write_simple_fits
from ..types import StrOrPath

# Optional dependencies
try:
    from scipy.ndimage import median_filter

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False
    median_filter = None  # type: ignore[assignment]

try:
    import astroscrappy

    LACOSMIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    LACOSMIC_AVAILABLE = False
    astroscrappy = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MAD_TO_STD_FACTOR = 1.4826  # for a normal distribution


# ============================================================================
# Defaults for LACosmic
# ============================================================================


@dataclass(frozen=True, slots=True)
class LACosmicConfig:
    """Configuration for the L.A.Cosmic (astroscrappy) algorithm."""

    sigclip: float = 4.5
    sigfrac: float = 0.3
    objlim: float = 5.0
    gain: float = 1.0
    readnoise: float = 6.5
    satlevel: float = 65535.0


# ============================================================================
# Result Types
# ============================================================================


@dataclass(slots=True)
class CosmicRayResult:
    """Result from cosmic ray detection/rejection."""

    cleaned_data: np.ndarray
    mask: np.ndarray
    num_cosmic_rays: int
    cosmic_ray_fraction: float
    method: str
    iterations: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            "Cosmic Ray Result:\n"
            f"  Method: {self.method}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Cosmic rays detected: {self.num_cosmic_rays} "
            f"({self.cosmic_ray_fraction * 100:.3f}%)\n"
            f"  Cleaned data shape: {self.cleaned_data.shape}"
        )


# ============================================================================
# Utility
# ============================================================================


def _calculate_mad(data: np.ndarray) -> float:
    """Robust estimator of standard deviation via MAD."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return float(MAD_TO_STD_FACTOR * mad)


# ============================================================================
# Detection Methods
# ============================================================================


def _detect_cosmic_rays_sigma_clip(
    data: np.ndarray,
    sigma: float = 5.0,
    box_size: int = 5,
    iterations: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Detect cosmic rays using sigma-clipping on median-filtered residuals."""
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "Sigma-clipping cosmic ray detection requires scipy. "
            "Install with: pip install scipy"
        )

    if data.ndim != 2:
        raise ImageProcessingError(
            f"sigma_clip requires 2D input, got {data.ndim}D",
            operation="cosmic ray detection (sigma_clip)",
        )

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if box_size % 2 == 0:
        box_size += 1
        logger.debug("Adjusted box_size to %d (must be odd)", box_size)

    cosmic_ray_mask = np.zeros(data.shape, dtype=bool)
    data_work = data.astype(np.float32, copy=True)
    threshold_value: float | None = None

    iterations_performed = 0
    for iteration in range(iterations):
        iterations_performed = iteration + 1

        med = median_filter(data_work, size=box_size)
        residuals = data_work - med

        median_residual = float(np.median(residuals))
        mad = _calculate_mad(residuals)
        if mad == 0:
            logger.warning("Iteration %d: MAD is zero, stopping", iteration + 1)
            break

        threshold_value = median_residual + sigma * mad
        new_cosmic_rays = residuals > threshold_value
        num_new = int(np.sum(new_cosmic_rays))

        logger.debug("Iteration %d: detected %d cosmic ray pixels", iteration + 1, num_new)

        if num_new == 0:
            break

        cosmic_ray_mask |= new_cosmic_rays
        data_work[new_cosmic_rays] = med[new_cosmic_rays]

    metadata = {
        "sigma": sigma,
        "box_size": box_size,
        "iterations_requested": iterations,
        "iterations_performed": iterations_performed,
        "threshold_value": threshold_value,
    }
    return data_work, cosmic_ray_mask, metadata


def _detect_cosmic_rays_lacosmic(
    data: np.ndarray,
    sigclip: float = 4.5,
    sigfrac: float = 0.3,
    objlim: float = 5.0,
    gain: float = 1.0,
    readnoise: float = 6.5,
    satlevel: float = 65535.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Detect cosmic rays using the L.A.Cosmic algorithm via astroscrappy."""
    if not LACOSMIC_AVAILABLE:
        raise ImportError(
            "L.A.Cosmic detection requires astroscrappy. "
            "Install with: pip install astroscrappy"
        )

    if data.ndim != 2:
        raise ImageProcessingError(
            f"lacosmic requires 2D input, got {data.ndim}D",
            operation="cosmic ray detection (lacosmic)",
        )

    mask, cleaned = astroscrappy.detect_cosmics(  # type: ignore[union-attr]
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Detect cosmic rays using median filtering + residual threshold."""
    if not SCIPY_AVAILABLE:
        raise ImportError("Median filtering requires scipy. Install with: pip install scipy")

    if data.ndim != 2:
        raise ImageProcessingError(
            f"median method requires 2D input, got {data.ndim}D",
            operation="cosmic ray detection (median)",
        )

    if box_size % 2 == 0:
        box_size += 1

    med = median_filter(data, size=box_size)
    residuals = data - med
    mask = residuals > threshold

    cleaned = np.array(data, copy=True)
    cleaned[mask] = med[mask]

    metadata = {"threshold": threshold, "box_size": box_size}
    logger.debug("Detected %d cosmic ray pixels", int(np.sum(mask)))
    return cleaned, mask, metadata


# ============================================================================
# Replacement Methods
# ============================================================================


def _replace_with_interpolation(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace cosmic ray pixels using interpolation from neighbors."""
    if data.ndim != 2:
        raise ImageProcessingError(
            "Interpolation replacement requires a 2D array.",
            operation="cosmic ray replacement (interpolation)",
        )

    try:
        from scipy.interpolate import griddata
    except ImportError as e:
        raise ImportError(
            "Interpolation requires scipy (scipy.interpolate.griddata). "
            "Install with: pip install scipy"
        ) from e

    cleaned = np.array(data, copy=True)
    ny, nx = data.shape
    y, x = np.mgrid[0:ny, 0:nx]

    good_mask = ~mask
    points = np.column_stack([y[good_mask], x[good_mask]])
    values = data[good_mask]
    cr_points = np.column_stack([y[mask], x[mask]])

    interpolated = griddata(points, values, cr_points, method="linear")
    nan_mask = np.isnan(interpolated)
    if nan_mask.any():
        interpolated[nan_mask] = griddata(points, values, cr_points[nan_mask], method="nearest")

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
    lacosmic_config: LACosmicConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Detect cosmic rays in image data.

    Returns
    -------
    cleaned, mask, metadata
        ``cleaned`` is a cleaned image produced by the chosen method.
        ``mask`` is a boolean cosmic-ray mask (True where cosmic rays detected).
        ``metadata`` contains method-specific diagnostics.
    """
    method = method.lower().strip()

    if method == "sigma_clip":
        return _detect_cosmic_rays_sigma_clip(data, sigma=sigma, box_size=box_size, iterations=iterations)

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
        f"Unknown cosmic ray detection method: {method!r}. "
        "Valid options: 'sigma_clip', 'lacosmic', 'median'."
    )


def reject_cosmic_rays(
    data: np.ndarray,
    method: str = "sigma_clip",
    sigma: float = 5.0,
    iterations: int = 1,
    box_size: int = 5,
    replace_with: str = "median",
    lacosmic_config: LACosmicConfig | None = None,
) -> CosmicRayResult:
    """Detect and remove cosmic rays from image data."""
    if data.size == 0:
        raise ImageProcessingError("Input data array is empty.", operation="cosmic ray rejection")

    replace_with = replace_with.lower().strip()
    cleaned_detect, mask, metadata = detect_cosmic_rays(
        data,
        method=method,
        sigma=sigma,
        iterations=iterations,
        box_size=box_size,
        lacosmic_config=lacosmic_config,
    )

    num_cosmic_rays = int(np.sum(mask))
    cosmic_ray_fraction = num_cosmic_rays / float(data.size)

    if num_cosmic_rays == 0:
        cleaned_final = np.array(data, copy=True)
    else:
        if method.lower().strip() == "lacosmic":
            cleaned_final = cleaned_detect
        elif replace_with == "median":
            cleaned_final = cleaned_detect
        elif replace_with == "interpolation":
            cleaned_final = _replace_with_interpolation(data, mask)
        else:
            raise ValueError(
                f"Unknown replacement method: {replace_with!r}. "
                "Valid options: 'median', 'interpolation'."
            )

    return CosmicRayResult(
        cleaned_data=cleaned_final,
        mask=mask,
        num_cosmic_rays=num_cosmic_rays,
        cosmic_ray_fraction=cosmic_ray_fraction,
        method=method,
        iterations=int(metadata.get("iterations_performed", iterations)),
        metadata=metadata,
    )


# ============================================================================
# File-level operations
# ============================================================================


def clean_cosmic_rays_file(
    input_file: StrOrPath,
    output_file: StrOrPath | None = None,
    method: str = "sigma_clip",
    sigma: float = 5.0,
    iterations: int = 1,
    hdu: int = 0,
    overwrite: bool = False,
    save_mask: bool = False,
    lacosmic_config: LACosmicConfig | None = None,
    box_size: int = 5,
    replace_with: str = "median",
) -> CosmicRayResult:
    """Clean cosmic rays from a FITS file."""
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file is not None else None

    logger.debug("Cleaning cosmic rays from: %s", input_path)

    data, header = get_data_and_header(input_path, hdu=hdu)
    if data is None:
        raise ImageProcessingError(
            f"HDU {hdu} in {input_path.name} contains no image data",
            operation="cosmic ray cleaning",
        )

    result = reject_cosmic_rays(
        data,
        method=method,
        sigma=sigma,
        iterations=iterations,
        box_size=box_size,
        replace_with=replace_with,
        lacosmic_config=lacosmic_config,
    )

    add_history(header, f"Cosmic rays removed (method={method})")
    header["CRMETH"] = (str(method), "Cosmic ray detection method")
    header["CRNUM"] = (int(result.num_cosmic_rays), "Number of cosmic ray pixels")
    header["CRFRAC"] = (float(result.cosmic_ray_fraction), "Cosmic ray fraction")
    header["CRITER"] = (int(result.iterations), "CR detection iterations")
    header["CRREPL"] = (str(replace_with), "CR replacement method")
    header["CRBOXSZ"] = (int(box_size), "Median filter box size")
    if method.lower().strip() == "sigma_clip":
        header["CRSIGMA"] = (float(sigma), "CR detection sigma threshold")

    if output_path is None:
        return result

    if output_path.exists() and not overwrite:
        raise FileOverwriteError(output_path)

    if save_mask:
        primary_hdu = fits.PrimaryHDU(result.cleaned_data, header=header)
        mask_hdu = fits.ImageHDU(result.mask.astype(np.uint8), name="CR_MASK")
        hdul = fits.HDUList([primary_hdu, mask_hdu])
        write_hdulist(output_path, hdul, overwrite=overwrite)
    else:
        write_simple_fits(output_path, result.cleaned_data, header=header, overwrite=overwrite)

    return result


# ============================================================================
# Batch Processing
# ============================================================================


def batch_clean_cosmic_rays(
    file_list: list[StrOrPath],
    output_dir: StrOrPath,
    method: str = "sigma_clip",
    sigma: float = 5.0,
    iterations: int = 1,
    hdu: int = 0,
    overwrite: bool = False,
    save_masks: bool = False,
    lacosmic_config: LACosmicConfig | None = None,
    box_size: int = 5,
    replace_with: str = "median",
) -> dict[str, Any]:
    """Clean cosmic rays from multiple files."""
    if not file_list:
        raise ValueError("Empty file_list")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "total_files": len(file_list),
        "successful": 0,
        "failed": 0,
        "total_cosmic_rays": 0,
        "output_files": [],
        "errors": [],
    }

    iterator = maybe_tqdm(enumerate(file_list, 1), total=len(file_list), desc="Cosmic ray cleaning")
    for i, filepath in iterator:
        filepath = Path(filepath)
        try:
            output_file = make_output_path(filepath, output_dir=output_dir, suffix="_clean", ensure_fits_ext=True)

            result = clean_cosmic_rays_file(
                filepath,
                output_file=output_file,
                method=method,
                sigma=sigma,
                iterations=iterations,
                hdu=hdu,
                overwrite=overwrite,
                save_mask=False,
                lacosmic_config=lacosmic_config,
                box_size=box_size,
                replace_with=replace_with,
            )

            if save_masks:
                mask_file = make_output_path(filepath, output_dir=output_dir, suffix="_crmask", ensure_fits_ext=True)
                write_simple_fits(mask_file, result.mask.astype(np.uint8), overwrite=overwrite)

            results["successful"] += 1
            results["total_cosmic_rays"] += result.num_cosmic_rays
            results["output_files"].append(str(output_file))

            if not is_tqdm_available():
                logger.info("[%d/%d] OK  %s", i, len(file_list), filepath.name)

        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{filepath.name}: {e}")

            if not is_tqdm_available():
                logger.error("[%d/%d] FAIL %s: %s", i, len(file_list), filepath.name, e)

    logger.info(
        "Batch cleaning complete: %d/%d successful (%d failed).",
        results["successful"],
        results["total_files"],
        results["failed"],
    )
    if results["total_cosmic_rays"] > 0:
        logger.info("Total cosmic rays removed: %d", results["total_cosmic_rays"])

    return results


# ============================================================================
# Convenience / Introspection
# ============================================================================


def get_available_methods() -> dict[str, dict[str, Any]]:
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


def check_dependencies() -> dict[str, bool]:
    """Check availability of optional dependencies."""
    return {
        "scipy": SCIPY_AVAILABLE,
        "astroscrappy": LACOSMIC_AVAILABLE,
        "tqdm": is_tqdm_available(),
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
