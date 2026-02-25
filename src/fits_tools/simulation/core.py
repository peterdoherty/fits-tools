#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic FITS data generation for CCD/CMOS testing and simulation.

This module provides utilities to generate realistic synthetic FITS files
for testing calibration pipelines, validation routines, and image processing
algorithms. It includes convenience functions for creating standard CCD/CMOS
calibration frames (bias, dark, flat) with realistic noise characteristics
and detector artifacts.

Features:
    - Standard calibration frames (bias, dark, flat, science)
    - Realistic noise models (read noise, photon noise, dark current)
    - Detector artifacts (hot pixels, cosmic rays, CTE trails, tree rings)
    - Multi-extension FITS (MEF) support for multi-amplifier detectors
    - Time-series generation (drift, thermal cycling)
    - Proper overscan region simulation
    - Integration with fits_tools infrastructure

Version History:
    0.1.0 - Initial release with basic dummy FITS creation
    0.2.0 - Added convenience functions, improved noise models, realistic
            CCD features, integration with fits_constants, and doctests
    0.3.0 - Complete refactoring: validation, MEF support, CTE trails,
            tree rings, brighter-fatter, time-series, overscan fixes
"""

from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy import ndimage
import logging
from typing import List, Dict, Any, Optional, Tuple


logger = logging.getLogger(__name__)

__version__ = "0.3.0"
__author__ = "fits_tools team"

# Import constants for proper FITS keywords
from ..constants import (
    FITS_KEY_IMGTYPE,
    FITS_KEY_EXPTIME,
    FITS_KEY_OBSERVER,
    # FITS_KEY_DATE_OBS,
    FITS_KEY_GAIN,
    FITS_KEY_RDNOISE,
    FITS_KEY_DATASEC,
    FITS_KEY_BIASSEC,
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
)

from ..array_utils import validate_numeric
from ..exceptions import ImageProcessingError
from ..types import StrOrPath


# ============================================================================
# Core Function (Backward Compatible)
# ============================================================================

def _create_dummy_fits(
    filepath: str,
    nx: int = 10,
    ny: int = 10,
    primary_data_value: Any = 0.0,
    noise_level: float = 0.0,
    primary_keywords: Optional[Dict[str, Any]] = None,
    extensions_info: Optional[List[Dict[str, Any]]] = None,
    clip_negative: bool = True
) -> None:
    """
    Creates a dummy FITS file with a primary HDU and optional extensions for testing.

    This function is designed to generate synthetic FITS files with customizable
    primary HDU data, noise, keywords, and additional image or table extensions.
    It's useful for creating test data for various FITS processing and analysis
    routines.

    Parameters
    ----------
    filepath : str
        The path where the dummy FITS file will be created.
    nx : int, optional
        Width of the primary HDU data (and default for extensions if not specified).
        Defaults to 10.
    ny : int, optional
        Height of the primary HDU data (and default for extensions if not specified).
        Defaults to 10.
    primary_data_value : Any, optional
        Base value to fill the primary HDU data array with. If None, the primary
        HDU will be created without data. If a scalar (int, float), the array
        will be filled with this value. If a NumPy array, it will be used directly.
        Defaults to 0.0.
    noise_level : float, optional
        Standard deviation of Gaussian random noise to add to the primary HDU data.
        Only applied if `primary_data_value` results in a data array. Defaults to 0.0.
    primary_keywords : dict, optional
        Dictionary of keyword-value pairs to add to the primary HDU header. Defaults to None.
    extensions_info : list of dict, optional
        A list where each dictionary describes an extension to add. Each dict can contain:
        - 'XTENSION': str (e.g., 'IMAGE', 'BINTABLE', 'ASCIITABLE'). Defaults to 'IMAGE'.
        - 'data_value': Any (value to fill the extension data array). If None, data is zeros
          for IMAGE extensions, or an empty table for table extensions.
        - 'keywords': dict (keyword-value pairs for the extension header).
        - 'nx', 'ny': int (dimensions for this specific extension). Defaults to `nx`, `ny`.
        Defaults to None (no extensions added).
    clip_negative : bool, optional
        If True, clip negative values to zero after adding noise. Defaults to True.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `primary_data_value` is a NumPy array but its dimensions don't match `(ny, nx)`.
        If `ext_info['data_value']` is a NumPy array but its dimensions don't match `(ext_ny, ext_nx)`.
    
    Examples
    --------
    >>> import tempfile
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'test.fits')
    ...     _create_dummy_fits(fpath, nx=10, ny=10, primary_data_value=100.0)
    ...     with fits.open(fpath) as hdul:
    ...         print(hdul[0].data.shape)
    (10, 10)
    """
    hdul = fits.HDUList()

    # --- Create Primary HDU ---
    primary_hdr = fits.Header()
    if primary_keywords:
        for k, v in primary_keywords.items():
            primary_hdr[k] = v

    primary_data_array = None
    if primary_data_value is not None:
        if isinstance(primary_data_value, np.ndarray):
            if primary_data_value.shape != (ny, nx):
                raise TypeError(
                    f"Provided primary_data_value array shape {primary_data_value.shape} "
                    f"does not match (ny, nx) = ({ny}, {nx})."
                )
            primary_data_array = primary_data_value.astype(np.float32)
        elif isinstance(primary_data_value, (int, float)):
            primary_data_array = np.full((ny, nx), primary_data_value, dtype=np.float32)
        
        if primary_data_array is not None and noise_level > 0:
            primary_data_array += np.random.normal(0, noise_level, size=(ny, nx)).astype(np.float32)
            if clip_negative:
                primary_data_array = np.maximum(0, primary_data_array)

    hdul.append(fits.PrimaryHDU(data=primary_data_array, header=primary_hdr))

    # --- Add Extensions ---
    if extensions_info:
        for ext_info in extensions_info:
            ext_type = ext_info.get('XTENSION', 'IMAGE').strip().upper()
            ext_data_value = ext_info.get('data_value')
            ext_keywords = ext_info.get('keywords', {})

            ext_hdr = fits.Header()
            # Add EXTNAME first if present, then other keywords
            if 'EXTNAME' in ext_keywords:
                ext_hdr['EXTNAME'] = ext_keywords['EXTNAME']
                ext_keywords = {k: v for k, v in ext_keywords.items() if k != 'EXTNAME'}
            for key, val in ext_keywords.items():
                ext_hdr[key] = val
            
            ext_nx = ext_info.get('nx', nx)
            ext_ny = ext_info.get('ny', ny)

            ext_data_array = None
            if ext_data_value is not None:
                if isinstance(ext_data_value, np.ndarray):
                    if ext_data_value.shape != (ext_ny, ext_nx):
                        raise TypeError(
                            f"Provided extension data_value array shape {ext_data_value.shape} "
                            f"does not match (ext_ny, ext_nx) = ({ext_ny}, {ext_nx})."
                        )
                    ext_data_array = ext_data_value.astype(np.float32)
                elif isinstance(ext_data_value, (int, float)):
                    ext_data_array = np.full((ext_ny, ext_nx), ext_data_value, dtype=np.float32)
            
            # For IMAGE extensions, ensure data array exists even if filled with zeros
            if ext_type == 'IMAGE' and ext_data_array is None:
                ext_data_array = np.zeros((ext_ny, ext_nx), dtype=np.float32)
            
            # Append HDU based on type
            if ext_type == 'IMAGE':
                hdul.append(fits.ImageHDU(data=ext_data_array, header=ext_hdr))
            elif ext_type == 'BINTABLE':
                if ext_data_array is not None and isinstance(ext_data_array, np.ndarray) and ext_data_array.dtype.fields:
                    hdul.append(fits.BinTableHDU(data=ext_data_array, header=ext_hdr))
                else:
                    hdul.append(fits.BinTableHDU(header=ext_hdr))
            elif ext_type == 'ASCIITABLE':
                if ext_data_array is not None and isinstance(ext_data_array, np.ndarray) and ext_data_array.dtype.fields:
                    hdul.append(fits.TableHDU(data=ext_data_array, header=ext_hdr))
                else:
                    hdul.append(fits.TableHDU(header=ext_hdr))
            else:
                logger.warning(f"Unsupported XTENSION type '{ext_type}' for extension. Defaulting to ImageHDU.")
                if ext_data_array is None:
                    ext_data_array = np.zeros((ext_ny, ext_nx), dtype=np.float32)
                hdul.append(fits.ImageHDU(data=ext_data_array, header=ext_hdr))

    hdul.writeto(filepath, overwrite=True)
    hdul.close()
    
    logger.debug(f"Created dummy FITS file: {filepath}")


# ============================================================================
# Basic Noise and Artifact Generation
# ============================================================================

def add_read_noise(
    data: np.ndarray,
    read_noise: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian read noise to image data.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    read_noise : float
        Read noise standard deviation in electrons (or ADU if not converted).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Data with added read noise.
    
    Examples
    --------
    >>> data = np.ones((10, 10)) * 100
    >>> noisy = add_read_noise(data, read_noise=5.0, seed=42)
    >>> noisy.shape
    (10, 10)
    >>> 90 < np.mean(noisy) < 110  # Should be close to 100
    True
    """
    validate_numeric(read_noise, 'read_noise', minimum=0.0)
    
    rng = np.random.default_rng(seed)
    
    noise = rng.normal(0, read_noise, size=data.shape)
    return data + noise.astype(data.dtype)


def add_photon_noise(
    data: np.ndarray,
    gain: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Poisson photon noise to image data.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data in ADU.
    gain : float, optional
        Gain in electrons per ADU. Defaults to 1.0.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Data with added photon noise.
    
    Examples
    --------
    >>> data = np.ones((10, 10)) * 1000
    >>> noisy = add_photon_noise(data, gain=1.0, seed=42)
    >>> 950 < np.mean(noisy) < 1050  # Should be close to 1000
    True
    """
    validate_numeric(gain, 'gain', minimum=0.0, allow_zero=False)
    
    rng = np.random.default_rng(seed)
    
    # Convert to electrons, add Poisson noise, convert back to ADU
    electrons = data * gain
    # Clip negative values before Poisson (Poisson requires non-negative)
    electrons = np.maximum(0, electrons)
    noisy_electrons = rng.poisson(electrons)
    return (noisy_electrons / gain).astype(data.dtype)


def add_hot_pixels(
    data: np.ndarray,
    num_hot_pixels: int = 0,
    hot_value_range: Tuple[float, float] = (1000, 10000),
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add hot pixels to image data.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    num_hot_pixels : int, optional
        Number of hot pixels to add. Defaults to 0.
    hot_value_range : tuple, optional
        (min, max) values for hot pixels. Defaults to (1000, 10000).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Data with hot pixels added.
    
    Examples
    --------
    >>> data = np.ones((100, 100)) * 100
    >>> with_hot = add_hot_pixels(data, num_hot_pixels=5, seed=42)
    >>> np.max(with_hot) > 100  # Should have some hot pixels
    True
    """
    validate_numeric(num_hot_pixels, 'num_hot_pixels', minimum=0, integer=True)
    
    if num_hot_pixels <= 0:
        return data
    
    rng = np.random.default_rng(seed)
    
    result = data.copy()
    ny, nx = data.shape
    
    # Random positions for hot pixels
    hot_rows = rng.integers(0, ny, size=num_hot_pixels)
    hot_cols = rng.integers(0, nx, size=num_hot_pixels)
    
    # Random values for hot pixels
    hot_values = rng.uniform(hot_value_range[0], hot_value_range[1], size=num_hot_pixels)
    
    result[hot_rows, hot_cols] = hot_values
    
    return result


def add_cosmic_rays(
    data: np.ndarray,
    num_cosmic_rays: int = 0,
    energy_range: Tuple[float, float] = (1000, 50000),
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add cosmic ray hits to image data.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    num_cosmic_rays : int, optional
        Number of cosmic ray events to add. Defaults to 0.
    energy_range : tuple, optional
        (min, max) energy for cosmic rays. Defaults to (1000, 50000).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Data with cosmic rays added.
    
    Examples
    --------
    >>> data = np.ones((100, 100)) * 100
    >>> with_cr = add_cosmic_rays(data, num_cosmic_rays=3, seed=42)
    >>> np.max(with_cr) > 1000  # Cosmic rays should be bright
    True
    """
    validate_numeric(num_cosmic_rays, 'num_cosmic_rays', minimum=0, integer=True)
    
    if num_cosmic_rays <= 0:
        return data
    
    rng = np.random.default_rng(seed)
    
    result = data.copy()
    ny, nx = data.shape
    
    for _ in range(num_cosmic_rays):
        # Random position
        row = rng.integers(1, ny - 1)
        col = rng.integers(1, nx - 1)
        
        # Random energy
        energy = rng.uniform(energy_range[0], energy_range[1])
        
        # Add cosmic ray with some spread (3x3 or cross pattern)
        pattern = rng.choice(['point', 'cross', 'blob'])
        
        if pattern == 'point':
            result[row, col] += energy
        elif pattern == 'cross':
            result[row, col] += energy
            result[row-1, col] += energy * 0.3
            result[row+1, col] += energy * 0.3
            result[row, col-1] += energy * 0.3
            result[row, col+1] += energy * 0.3
        else:  # blob
            result[row, col] += energy
            result[row-1:row+2, col-1:col+2] += energy * 0.2
    
    return result


def create_vignetting_pattern(
    shape: Tuple[int, int],
    vignetting_factor: float = 0.2
) -> np.ndarray:
    """
    Create a vignetting (illumination fall-off) pattern.
    
    Parameters
    ----------
    shape : tuple
        (ny, nx) shape of the pattern.
    vignetting_factor : float, optional
        Strength of vignetting (0 = none, 1 = severe). Defaults to 0.2.
    
    Returns
    -------
    np.ndarray
        Vignetting pattern (1.0 at center, falls off toward edges).
    
    Examples
    --------
    >>> pattern = create_vignetting_pattern((100, 100), vignetting_factor=0.3)
    >>> pattern.shape
    (100, 100)
    >>> pattern[50, 50] > pattern[0, 0]  # Center brighter than corner
    True
    """
    validate_numeric(vignetting_factor, 'vignetting_factor', minimum=0.0, maximum=1.0)
    
    ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    
    # Distance from center, normalized
    cy, cx = ny / 2, nx / 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_max = np.sqrt(cx**2 + cy**2)
    r_norm = r / r_max
    
    # Vignetting: 1 at center, falls off to (1 - vignetting_factor) at edges
    vignetting = 1.0 - vignetting_factor * r_norm**2
    
    return vignetting


# ============================================================================
# Advanced Detector Artifacts (NEW in v0.3.0)
# ============================================================================

def add_cte_trails(
    data: np.ndarray,
    cte_efficiency: float = 0.999999,
    direction: str = 'vertical',
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Charge Transfer Efficiency (CTE) trails.
    
    CTE < 1 causes bright sources to leave faint trails in the
    readout direction.
    
    Parameters
    ----------
    data : np.ndarray
        Input image.
    cte_efficiency : float, optional
        CTE per pixel transfer (0-1). Defaults to 0.999999.
    direction : str, optional
        'vertical' or 'horizontal'. Defaults to 'vertical'.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    np.ndarray
        Image with CTE trails.
    
    Examples
    --------
    >>> data = np.zeros((100, 100))
    >>> data[50, 50] = 10000  # Bright source
    >>> with_cte = add_cte_trails(data, cte_efficiency=0.9999, direction='vertical')
    >>> np.sum(with_cte[51:60, 50]) > 0  # Should have trail below source
    True
    """
    validate_numeric(cte_efficiency, 'cte_efficiency', minimum=0.0, maximum=1.0)
    
    if direction not in ['vertical', 'horizontal']:
        raise ValueError(f"Direction must be 'vertical' or 'horizontal', got '{direction}'")
    
    
    result = data.copy().astype(np.float32)
    inefficiency = 1.0 - cte_efficiency
    
    if direction == 'vertical':
        # Vertical CTE trails (typical for row readout)
        for row in range(1, data.shape[0]):
            # Some charge left behind from previous row
            trail = result[row-1, :] * inefficiency
            result[row, :] += trail
            result[row-1, :] -= trail
    else:
        # Horizontal CTE trails
        for col in range(1, data.shape[1]):
            trail = result[:, col-1] * inefficiency
            result[:, col] += trail
            result[:, col-1] -= trail
    
    return result.astype(data.dtype)


def add_tree_rings(
    data: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    amplitude: float = 0.02,
    num_rings: int = 5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add tree ring pattern (silicon crystal defects).
    
    Tree rings are concentric patterns of slightly higher/lower
    sensitivity caused by dopant concentration variations during
    silicon crystal growth.
    
    Parameters
    ----------
    data : np.ndarray
        Input image.
    center : tuple, optional
        Center of tree ring pattern. If None, use image center.
    amplitude : float, optional
        Fractional amplitude of pattern. Defaults to 0.02 (2%).
    num_rings : int, optional
        Number of rings. Defaults to 5.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    np.ndarray
        Image with tree ring pattern applied.
    
    Examples
    --------
    >>> data = np.ones((100, 100)) * 1000
    >>> with_rings = add_tree_rings(data, amplitude=0.05, num_rings=3, seed=42)
    >>> np.std(with_rings) > np.std(data)  # Should have more variation
    True
    """
    validate_numeric(amplitude, 'amplitude', minimum=0.0, maximum=1.0)
    validate_numeric(num_rings, 'num_rings', minimum=1, integer=True)
    
    rng = np.random.default_rng(seed)
    
    ny, nx = data.shape
    
    if center is None:
        center = (ny // 2, nx // 2)
    
    cy, cx = center
    
    # Create radial distance map
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create ring pattern with some randomness
    pattern = np.zeros_like(r)
    for i in range(num_rings):
        ring_radius = (i + 1) * min(nx, ny) / (2 * num_rings)
        ring_width = ring_radius * 0.3
        
        # Gaussian ring with random amplitude
        ring_amp = amplitude * rng.uniform(0.5, 1.5)
        ring = ring_amp * np.exp(-((r - ring_radius) ** 2) / (2 * ring_width ** 2))
        pattern += ring
    
    # Apply multiplicative pattern
    return data * (1.0 + pattern)


def add_brighter_fatter_effect(
    data: np.ndarray,
    strength: float = 1e-6,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add brighter-fatter effect.
    
    Bright pixels appear slightly larger due to electric field
    distortion in the silicon. This causes charge to spread to
    neighboring pixels.
    
    Parameters
    ----------
    data : np.ndarray
        Input image.
    strength : float, optional
        Strength of effect. Defaults to 1e-6.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    np.ndarray
        Image with brighter-fatter effect.
    
    Examples
    --------
    >>> data = np.zeros((100, 100))
    >>> data[50, 50] = 50000  # Very bright pixel
    >>> with_bf = add_brighter_fatter_effect(data, strength=1e-5)
    >>> with_bf[50, 51] > 0  # Should spread to neighbors
    True
    """
    validate_numeric(strength, 'strength', minimum=0.0)
    
    
    # Simplified model: convolve bright regions with small kernel
    # Real implementation would be much more complex
    kernel = np.array([
        [0, strength, 0],
        [strength, 1-4*strength, strength],
        [0, strength, 0]
    ])
    
    # Only apply to bright pixels
    bright_threshold = np.percentile(data, 90)
    bright_mask = data > bright_threshold
    
    result = data.copy()
    if np.any(bright_mask):
        result[bright_mask] = ndimage.convolve(data, kernel, mode='constant')[bright_mask]
    
    return result


# ============================================================================
# Calibration Frame Creation Functions
# ============================================================================

def create_bias_frame(
    filepath: str,
    shape: Tuple[int, int] = (512, 512),
    bias_level: float = 100.0,
    read_noise: float = 5.0,
    num_hot_pixels: int = 0,
    add_sections: bool = False,
    overscan_width: int = 32,
    seed: Optional[int] = None,
    extra_keywords: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a realistic synthetic bias frame.
    
    Bias frames have zero exposure time and measure the electronic bias level
    and read noise of the detector.
    
    Parameters
    ----------
    filepath : str
        Output FITS file path.
    shape : tuple, optional
        (rows, cols) dimensions of the image. Defaults to (512, 512).
    bias_level : float, optional
        Mean bias level in ADU. Defaults to 100.0.
    read_noise : float, optional
        Read noise standard deviation in ADU. Defaults to 5.0.
    num_hot_pixels : int, optional
        Number of hot pixels to include. Defaults to 0.
    add_sections : bool, optional
        If True, add DATASEC and BIASSEC keywords and create proper overscan.
        Defaults to False.
    overscan_width : int, optional
        Width of overscan region if add_sections=True. Defaults to 32.
    seed : int, optional
        Random seed for reproducibility.
    extra_keywords : dict, optional
        Additional header keywords to include.
    
    Examples
    --------
    >>> import tempfile
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'bias.fits')
    ...     create_bias_frame(fpath, shape=(100, 100), bias_level=100, seed=42)
    ...     with fits.open(fpath) as hdul:
    ...         print(hdul[0].header[FITS_KEY_IMGTYPE])
    ...         print(hdul[0].data.shape)
    BIAS
    (100, 100)
    """
    # Validate parameters
    validate_numeric(shape[0], 'rows', minimum=1, integer=True)
    validate_numeric(shape[1], 'cols', minimum=1, integer=True)
    validate_numeric(bias_level, 'bias_level', minimum=0.0)
    validate_numeric(read_noise, 'read_noise', minimum=0.0)
    validate_numeric(num_hot_pixels, 'num_hot_pixels', minimum=0, integer=True)
    validate_numeric(overscan_width, 'overscan_width', minimum=0, integer=True)
    
    
    ny, nx = shape
    
    # Build header
    keywords = {
        FITS_KEY_IMGTYPE: FITS_VAL_BIAS_IMGTYPE,
        FITS_KEY_EXPTIME: 0.0,
        FITS_KEY_RDNOISE: read_noise,
        'BIASLVL': bias_level,
    }
    
    if add_sections and overscan_width > 0:
        # Create frame with overscan region
        data_cols = nx - overscan_width
        
        if data_cols <= 0:
            raise ValueError(f"Overscan width {overscan_width} >= image width {nx}")
        
        # Create full array
        data = np.full((ny, nx), bias_level, dtype=np.float32)
        
        # Data region: add read noise and hot pixels
        data_region = data[:, :data_cols]
        data_region = add_read_noise(data_region, read_noise, seed=seed)
        if num_hot_pixels > 0:
            data_region = add_hot_pixels(data_region, num_hot_pixels, seed=seed)
        data[:, :data_cols] = data_region
        
        # Overscan region: only bias + read noise (no hot pixels, no signal)
        overscan_region = data[:, data_cols:]
        overscan_region = add_read_noise(overscan_region, read_noise, 
                                        seed=seed+1000 if seed is not None else None)
        data[:, data_cols:] = overscan_region
        
# Add section keywords (FITS 1-indexed)
        keywords[FITS_KEY_DATASEC] = f'[1:{data_cols},1:{ny}]'
        keywords[FITS_KEY_BIASSEC] = f'[{data_cols+1}:{nx},1:{ny}]'
    else:
        # No sections - create full array
        data = np.full((ny, nx), bias_level, dtype=np.float32)
        data = add_read_noise(data, read_noise, seed=seed)
        if num_hot_pixels > 0:
            data = add_hot_pixels(data, num_hot_pixels, seed=seed)
    
    if extra_keywords:
        keywords.update(extra_keywords)
    
    _create_dummy_fits(
        filepath,
        nx=nx, ny=ny,
        primary_data_value=data,
        primary_keywords=keywords,
        clip_negative=False  # Bias can have small negative variations
    )
    
    logger.info(f"Created bias frame: {filepath}")


def create_dark_frame(
    filepath: str,
    shape: Tuple[int, int] = (512, 512),
    exposure_time: float = 300.0,
    bias_level: float = 100.0,
    dark_current: float = 0.1,
    dark_variation: float = 0.1,
    read_noise: float = 5.0,
    gain: float = 1.0,
    num_hot_pixels: int = 5,
    add_sections: bool = False,
    overscan_width: int = 32,
    seed: Optional[int] = None,
    extra_keywords: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a realistic synthetic dark frame.
    
    Dark frames measure dark current (thermal electrons) accumulated during
    an exposure with no illumination.
    
    Parameters
    ----------
    filepath : str
        Output FITS file path.
    shape : tuple, optional
        (rows, cols) dimensions of the image. Defaults to (512, 512).
    exposure_time : float, optional
        Exposure time in seconds. Defaults to 300.0.
    bias_level : float, optional
        Bias level in ADU. Defaults to 100.0.
    dark_current : float, optional
        Mean dark current in electrons/pixel/second. Defaults to 0.1.
    dark_variation : float, optional
        Fractional pixel-to-pixel dark current variation (0-1). Defaults to 0.1.
    read_noise : float, optional
        Read noise in ADU. Defaults to 5.0.
    gain : float, optional
        Gain in electrons/ADU. Defaults to 1.0.
    num_hot_pixels : int, optional
        Number of hot pixels. Defaults to 5.
    add_sections : bool, optional
        Add DATASEC/BIASSEC keywords. Defaults to False.
    overscan_width : int, optional
        Overscan width if add_sections=True. Defaults to 32.
    seed : int, optional
        Random seed for reproducibility.
    extra_keywords : dict, optional
        Additional header keywords.
    
    Examples
    --------
    >>> import tempfile
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'dark.fits')
    ...     create_dark_frame(fpath, shape=(100, 100), exposure_time=100, seed=42)
    ...     with fits.open(fpath) as hdul:
    ...         print(hdul[0].header[FITS_KEY_IMGTYPE])
    ...         print(hdul[0].header[FITS_KEY_EXPTIME])
    DARK
    100.0
    """
    # Validate parameters
    validate_numeric(shape[0], 'rows', minimum=1, integer=True)
    validate_numeric(shape[1], 'cols', minimum=1, integer=True)
    validate_numeric(exposure_time, 'exposure_time', minimum=0.0)
    validate_numeric(bias_level, 'bias_level', minimum=0.0)
    validate_numeric(dark_current, 'dark_current', minimum=0.0)
    validate_numeric(dark_variation, 'dark_variation', minimum=0.0, maximum=1.0)
    validate_numeric(read_noise, 'read_noise', minimum=0.0)
    validate_numeric(gain, 'gain', minimum=0.0, allow_zero=False)
    validate_numeric(num_hot_pixels, 'num_hot_pixels', minimum=0, integer=True)
    
    rng = np.random.default_rng(seed)
    
    ny, nx = shape
    
    # Build header
    keywords = {
        FITS_KEY_IMGTYPE: FITS_VAL_DARK_IMGTYPE,
        FITS_KEY_EXPTIME: exposure_time,
        FITS_KEY_RDNOISE: read_noise,
        FITS_KEY_GAIN: gain,
        'BIASLVL': bias_level,
        'DARKCUR': dark_current,
    }
    
    if add_sections and overscan_width > 0:
        data_cols = nx - overscan_width
        
        if data_cols <= 0:
            raise ValueError(f"Overscan width {overscan_width} >= image width {nx}")
        
        # Create full array with bias
        data = np.full((ny, nx), bias_level, dtype=np.float32)
        
        # Data region: add dark current, photon noise, read noise, hot pixels
        data_region = data[:, :data_cols].copy()
        
        # Add dark current with spatial variation
        dark_mean = dark_current * exposure_time / gain
        if dark_variation > 0:
            dark_map = rng.normal(1.0, dark_variation, size=(ny, data_cols))
            dark_signal = dark_mean * dark_map
        else:
            dark_signal = dark_mean
        
        data_region += dark_signal
        
        # Add photon noise from dark signal
        data_region = add_photon_noise(data_region, gain=gain, seed=seed)
        
        # Add read noise
        data_region = add_read_noise(data_region, read_noise, seed=seed+1 if seed else None)
        
        # Add hot pixels
        if num_hot_pixels > 0:
            data_region = add_hot_pixels(
                data_region, num_hot_pixels,
                hot_value_range=(bias_level + 100, bias_level + 1000),
                seed=seed+2 if seed else None
            )
        
        data[:, :data_cols] = data_region
        
        # Overscan region: only bias + read noise
        overscan_region = data[:, data_cols:]
        overscan_region = add_read_noise(overscan_region, read_noise, 
                                        seed=seed+1000 if seed else None)
        data[:, data_cols:] = overscan_region
        
        # Add section keywords
        keywords[FITS_KEY_DATASEC] = f'[1:{data_cols},1:{ny}]'
        keywords[FITS_KEY_BIASSEC] = f'[{data_cols+1}:{nx},1:{ny}]'
    else:
        # No sections - create full array
        data = np.full((ny, nx), bias_level, dtype=np.float32)
        
        # Add dark current with spatial variation
        dark_mean = dark_current * exposure_time / gain
        if dark_variation > 0:
            dark_map = rng.normal(1.0, dark_variation, size=shape)
            dark_signal = dark_mean * dark_map
        else:
            dark_signal = dark_mean
        
        data += dark_signal
        
        # Add photon noise from dark signal
        data = add_photon_noise(data, gain=gain, seed=seed)
        
        # Add read noise
        data = add_read_noise(data, read_noise, seed=seed+1 if seed else None)
        
        # Add hot pixels
        if num_hot_pixels > 0:
            data = add_hot_pixels(
                data, num_hot_pixels,
                hot_value_range=(bias_level + 100, bias_level + 1000),
                seed=seed+2 if seed else None
            )
    
    if extra_keywords:
        keywords.update(extra_keywords)
    
    _create_dummy_fits(
        filepath,
        nx=nx, ny=ny,
        primary_data_value=data,
        primary_keywords=keywords,
        clip_negative=False
    )
    
    logger.info(f"Created dark frame: {filepath} ({exposure_time}s)")


def create_flat_frame(
    filepath: str,
    shape: Tuple[int, int] = (512, 512),
    exposure_time: float = 10.0,
    mean_level: float = 20000.0,
    bias_level: float = 100.0,
    read_noise: float = 5.0,
    gain: float = 1.0,
    vignetting: float = 0.2,
    dust_specs: int = 0,
    add_sections: bool = False,
    overscan_width: int = 32,
    seed: Optional[int] = None,
    extra_keywords: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a realistic synthetic flat field frame.
    
    Flat fields measure pixel-to-pixel sensitivity variations and are used
    to correct for non-uniform illumination.
    
    Parameters
    ----------
    filepath : str
        Output FITS file path.
    shape : tuple, optional
        (rows, cols) dimensions. Defaults to (512, 512).
    exposure_time : float, optional
        Exposure time in seconds. Defaults to 10.0.
    mean_level : float, optional
        Mean signal level in ADU. Defaults to 20000.0.
    bias_level : float, optional
        Bias level in ADU. Defaults to 100.0.
    read_noise : float, optional
        Read noise in ADU. Defaults to 5.0.
    gain : float, optional
        Gain in electrons/ADU. Defaults to 1.0.
    vignetting : float, optional
        Vignetting strength (0-1). Defaults to 0.2.
    dust_specs : int, optional
        Number of dust donuts/shadows. Defaults to 0.
    add_sections : bool, optional
        Add DATASEC/BIASSEC keywords. Defaults to False.
    overscan_width : int, optional
        Overscan width. Defaults to 32.
    seed : int, optional
        Random seed.
    extra_keywords : dict, optional
        Additional keywords.
    
    Examples
    --------
    >>> import tempfile
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'flat.fits')
    ...     create_flat_frame(fpath, shape=(100, 100), mean_level=10000, seed=42)
    ...     with fits.open(fpath) as hdul:
    ...         print(hdul[0].header[FITS_KEY_IMGTYPE])
    ...         data_mean = np.mean(hdul[0].data)
    ...         print(9000 < data_mean < 11000)  # Should be near 10000
    FLAT
    True
    """
    # Validate parameters
    validate_numeric(shape[0], 'rows', minimum=1, integer=True)
    validate_numeric(shape[1], 'cols', minimum=1, integer=True)
    validate_numeric(exposure_time, 'exposure_time', minimum=0.0)
    validate_numeric(mean_level, 'mean_level', minimum=0.0)
    validate_numeric(bias_level, 'bias_level', minimum=0.0)
    validate_numeric(read_noise, 'read_noise', minimum=0.0)
    validate_numeric(gain, 'gain', minimum=0.0, allow_zero=False)
    validate_numeric(vignetting, 'vignetting', minimum=0.0, maximum=1.0)
    validate_numeric(dust_specs, 'dust_specs', minimum=0, integer=True)
    
    rng = np.random.default_rng(seed)
    
    ny, nx = shape
    
    # Build header
    keywords = {
        FITS_KEY_IMGTYPE: FITS_VAL_FLAT_IMGTYPE,
        FITS_KEY_EXPTIME: exposure_time,
        FITS_KEY_RDNOISE: read_noise,
        FITS_KEY_GAIN: gain,
        'BIASLVL': bias_level,
    }
    
    if add_sections and overscan_width > 0:
        data_cols = nx - overscan_width
        
        if data_cols <= 0:
            raise ValueError(f"Overscan width {overscan_width} >= image width {nx}")
        
        # Create full array with bias
        data = np.full((ny, nx), bias_level, dtype=np.float32)
        
        # Data region: add flat field signal
        data_region = data[:, :data_cols].copy()
        
        # Add flat field signal with vignetting
        vignetting_pattern = create_vignetting_pattern((ny, data_cols), vignetting)
        flat_signal = mean_level * vignetting_pattern
        data_region += flat_signal
        
        # Add photon noise
        data_region = add_photon_noise(data_region, gain=gain, seed=seed)
        
        # Add read noise
        data_region = add_read_noise(data_region, read_noise, seed=seed+1 if seed else None)
        
        # Add dust specs (dark spots)
        if dust_specs > 0:
            for i in range(dust_specs):
                cy = rng.integers(ny//4, 3*ny//4)
                cx = rng.integers(data_cols//4, 3*data_cols//4)
                radius = rng.integers(3, 10)
                
                y, x = np.ogrid[:ny, :data_cols]
                mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                data_region[mask] *= rng.uniform(0.5, 0.8)  # Darken region
        
        data[:, :data_cols] = data_region
        
        # Overscan region: only bias + read noise
        overscan_region = data[:, data_cols:]
        overscan_region = add_read_noise(overscan_region, read_noise, 
                                        seed=seed+1000 if seed else None)
        data[:, data_cols:] = overscan_region
        
        # Add section keywords
        keywords[FITS_KEY_DATASEC] = f'[1:{data_cols},1:{ny}]'
        keywords[FITS_KEY_BIASSEC] = f'[{data_cols+1}:{nx},1:{ny}]'
    else:
        # No sections - create full array
        data = np.full((ny, nx), bias_level, dtype=np.float32)
        
        # Add flat field signal with vignetting
        vignetting_pattern = create_vignetting_pattern(shape, vignetting)
        flat_signal = mean_level * vignetting_pattern
        data += flat_signal
        
        # Add photon noise
        data = add_photon_noise(data, gain=gain, seed=seed)
        
        # Add read noise
        data = add_read_noise(data, read_noise, seed=seed+1 if seed else None)
        
        # Add dust specs (dark spots)
        if dust_specs > 0:
            for i in range(dust_specs):
                cy = rng.integers(ny//4, 3*ny//4)
                cx = rng.integers(nx//4, 3*nx//4)
                radius = rng.integers(3, 10)
                
                y, x = np.ogrid[:ny, :nx]
                mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                data[mask] *= rng.uniform(0.5, 0.8)  # Darken region
    
    if extra_keywords:
        keywords.update(extra_keywords)
    
    _create_dummy_fits(
        filepath,
        nx=nx, ny=ny,
        primary_data_value=data,
        primary_keywords=keywords,
        clip_negative=True
    )
    
    logger.info(f"Created flat frame: {filepath} (mean level: {mean_level:.0f} ADU)")


def create_science_frame(
    filepath: str,
    shape: Tuple[int, int] = (512, 512),
    exposure_time: float = 60.0,
    sky_level: float = 1000.0,
    bias_level: float = 100.0,
    read_noise: float = 5.0,
    gain: float = 1.0,
    num_stars: int = 10,
    num_cosmic_rays: int = 3,
    add_sections: bool = False,
    overscan_width: int = 32,
    seed: Optional[int] = None,
    extra_keywords: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a realistic synthetic science/light frame.
    
    Science frames contain astronomical targets with sky background, stars,
    and cosmic ray hits.
    
    Parameters
    ----------
    filepath : str
        Output FITS file path.
    shape : tuple, optional
        (rows, cols) dimensions. Defaults to (512, 512).
    exposure_time : float, optional
        Exposure time in seconds. Defaults to 60.0.
    sky_level : float, optional
        Sky background level in ADU. Defaults to 1000.0.
    bias_level : float, optional
        Bias level in ADU. Defaults to 100.0.
    read_noise : float, optional
        Read noise in ADU. Defaults to 5.0.
    gain : float, optional
        Gain in electrons/ADU. Defaults to 1.0.
    num_stars : int, optional
        Number of point sources to add. Defaults to 10.
    num_cosmic_rays : int, optional
        Number of cosmic ray hits. Defaults to 3.
    add_sections : bool, optional
        Add DATASEC/BIASSEC keywords. Defaults to False.
    overscan_width : int, optional
        Overscan width. Defaults to 32.
    seed : int, optional
        Random seed.
    extra_keywords : dict, optional
        Additional keywords.
    
    Examples
    --------
    >>> import tempfile
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'science.fits')
    ...     create_science_frame(fpath, shape=(100, 100), num_stars=5, seed=42)
    ...     with fits.open(fpath) as hdul:
    ...         print(hdul[0].header[FITS_KEY_IMGTYPE])
    ...         print(hdul[0].header[FITS_KEY_EXPTIME])
    LIGHT
    60.0
    """
    # Validate parameters
    validate_numeric(shape[0], 'rows', minimum=1, integer=True)
    validate_numeric(shape[1], 'cols', minimum=1, integer=True)
    validate_numeric(exposure_time, 'exposure_time', minimum=0.0)
    validate_numeric(sky_level, 'sky_level', minimum=0.0)
    validate_numeric(bias_level, 'bias_level', minimum=0.0)
    validate_numeric(read_noise, 'read_noise', minimum=0.0)
    validate_numeric(gain, 'gain', minimum=0.0, allow_zero=False)
    validate_numeric(num_stars, 'num_stars', minimum=0, integer=True)
    validate_numeric(num_cosmic_rays, 'num_cosmic_rays', minimum=0, integer=True)
    
    rng = np.random.default_rng(seed)
    
    ny, nx = shape
    
    # Build header
    keywords = {
        FITS_KEY_IMGTYPE: FITS_VAL_LIGHT_IMGTYPE,
        FITS_KEY_EXPTIME: exposure_time,
        FITS_KEY_RDNOISE: read_noise,
        FITS_KEY_GAIN: gain,
        'BIASLVL': bias_level,
        'SKYLEVEL': sky_level,
    }
    
    if add_sections and overscan_width > 0:
        data_cols = nx - overscan_width
        
        if data_cols <= 0:
            raise ValueError(f"Overscan width {overscan_width} >= image width {nx}")
        
        # Create full array with bias + sky
        data = np.full((ny, nx), bias_level + sky_level, dtype=np.float32)
        
        # Data region: add stars, photon noise, read noise, cosmic rays
        data_region = data[:, :data_cols].copy()
        
        # Add stars (simple Gaussian PSF)
        if num_stars > 0:
            for _ in range(num_stars):
                # Random position (avoid edges)
                star_y = rng.integers(20, ny - 20)
                star_x = rng.integers(20, data_cols - 20)
                
                # Random brightness
                star_flux = rng.uniform(5000, 50000)
                
                # Gaussian PSF (FWHM ~ 3-5 pixels)
                sigma = rng.uniform(1.5, 2.5)
                
                # Create small region around star
                y, x = np.ogrid[max(0, star_y-15):min(ny, star_y+15),
                               max(0, star_x-15):min(data_cols, star_x+15)]
                
                psf = star_flux * np.exp(-((x - star_x)**2 + (y - star_y)**2) / (2 * sigma**2))
                
                data_region[max(0, star_y-15):min(ny, star_y+15),
                           max(0, star_x-15):min(data_cols, star_x+15)] += psf
        
        # Add photon noise
        data_region = add_photon_noise(data_region, gain=gain, seed=seed)
        
        # Add read noise
        data_region = add_read_noise(data_region, read_noise, seed=seed+1 if seed else None)
        
        # Add cosmic rays
        if num_cosmic_rays > 0:
            data_region = add_cosmic_rays(data_region, num_cosmic_rays, seed=seed+2 if seed else None)
        
        data[:, :data_cols] = data_region
        
        # Overscan region: only bias + read noise
        overscan_region = data[:, data_cols:]
        overscan_region[:] = bias_level
        overscan_region = add_read_noise(overscan_region, read_noise, 
                                        seed=seed+1000 if seed else None)
        data[:, data_cols:] = overscan_region
        
        # Add section keywords
        keywords[FITS_KEY_DATASEC] = f'[1:{data_cols},1:{ny}]'
        keywords[FITS_KEY_BIASSEC] = f'[{data_cols+1}:{nx},1:{ny}]'
    else:
        # No sections - create full array
        data = np.full((ny, nx), bias_level + sky_level, dtype=np.float32)
        
        # Add stars (simple Gaussian PSF)
        if num_stars > 0:
            for _ in range(num_stars):
                # Random position (avoid edges)
                star_y = rng.integers(20, ny - 20)
                star_x = rng.integers(20, nx - 20)
                
                # Random brightness
                star_flux = rng.uniform(5000, 50000)
                
                # Gaussian PSF (FWHM ~ 3-5 pixels)
                sigma = rng.uniform(1.5, 2.5)
                
                # Create small region around star
                y, x = np.ogrid[max(0, star_y-15):min(ny, star_y+15),
                               max(0, star_x-15):min(nx, star_x+15)]
                
                psf = star_flux * np.exp(-((x - star_x)**2 + (y - star_y)**2) / (2 * sigma**2))
                
                data[max(0, star_y-15):min(ny, star_y+15),
                     max(0, star_x-15):min(nx, star_x+15)] += psf
        
        # Add photon noise
        data = add_photon_noise(data, gain=gain, seed=seed)
        
        # Add read noise
        data = add_read_noise(data, read_noise, seed=seed+1 if seed else None)
        
        # Add cosmic rays
        if num_cosmic_rays > 0:
            data = add_cosmic_rays(data, num_cosmic_rays, seed=seed+2 if seed else None)
    
    if extra_keywords:
        keywords.update(extra_keywords)
    
    _create_dummy_fits(
        filepath,
        nx=nx, ny=ny,
        primary_data_value=data,
        primary_keywords=keywords,
        clip_negative=True
    )
    
    logger.info(f"Created science frame: {filepath} ({num_stars} stars, {num_cosmic_rays} cosmic rays)")


def create_calibration_set(
    output_dir: str,
    num_bias: int = 10,
    num_dark: int = 10,
    num_flat: int = 10,
    shape: Tuple[int, int] = (512, 512),
    dark_exposure_time: float = 300.0,
    flat_exposure_time: float = 10.0,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Create a complete set of calibration frames.
    
    Parameters
    ----------
    output_dir : str
        Directory to save frames.
    num_bias : int, optional
        Number of bias frames. Defaults to 10.
    num_dark : int, optional
        Number of dark frames. Defaults to 10.
    num_flat : int, optional
        Number of flat frames. Defaults to 10.
    shape : tuple, optional
        Image dimensions. Defaults to (512, 512).
    dark_exposure_time : float, optional
        Dark frame exposure time. Defaults to 300.0.
    flat_exposure_time : float, optional
        Flat frame exposure time. Defaults to 10.0.
    **kwargs
        Additional parameters passed to frame creation functions.
    
    Returns
    -------
    dict
        Dictionary with 'bias', 'dark', 'flat' keys containing lists of file paths.
    
    Examples
    --------
    >>> import tempfile
    >>> import shutil
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     files = create_calibration_set(tmpdir, num_bias=2, num_dark=2, num_flat=2,
    ...                                     shape=(50, 50))
    ...     print(len(files['bias']))
    ...     print(len(files['dark']))
    ...     print(len(files['flat']))
    2
    2
    2
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    created_files = {'bias': [], 'dark': [], 'flat': []}
    
    # Create bias frames
    for i in range(num_bias):
        filepath = str(output_dir_path / f'bias_{i+1:03d}.fits')
        create_bias_frame(filepath, shape=shape, seed=i, **kwargs)
        created_files['bias'].append(filepath)
    
    # Create dark frames
    for i in range(num_dark):
        filepath = str(output_dir_path / f'dark_{i+1:03d}.fits')
        create_dark_frame(filepath, shape=shape, exposure_time=dark_exposure_time,
                         seed=i+1000, **kwargs)
        created_files['dark'].append(filepath)
    
    # Create flat frames
    for i in range(num_flat):
        filepath = str(output_dir_path / f'flat_{i+1:03d}.fits')
        create_flat_frame(filepath, shape=shape, exposure_time=flat_exposure_time,
                         seed=i+2000, **kwargs)
        created_files['flat'].append(filepath)
    
    logger.info(f"Created calibration set in {output_dir}: "
               f"{num_bias} bias, {num_dark} dark, {num_flat} flat")
    
    return created_files


# ============================================================================
# Multi-Extension FITS (MEF) Support (NEW in v0.3.0)
# ============================================================================

def create_mef_bias_frame(
    filepath: str,
    num_amps: int = 4,
    amp_shape: Tuple[int, int] = (512, 512),
    bias_levels: Optional[List[float]] = None,
    read_noises: Optional[List[float]] = None,
    seed: Optional[int] = None,
    **kwargs
) -> None:
    """
    Create a Multi-Extension FITS bias frame with multiple amplifiers.
    
    Parameters
    ----------
    filepath : str
        Output file path.
    num_amps : int, optional
        Number of amplifiers/extensions. Defaults to 4.
    amp_shape : tuple, optional
        Shape of each amplifier region. Defaults to (512, 512).
    bias_levels : list of float, optional
        Bias level for each amp. If None, use random variation.
    read_noises : list of float, optional
        Read noise for each amp. If None, use random variation.
    seed : int, optional
        Random seed.
    **kwargs
        Additional parameters passed to bias frame creation.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     fpath = os.path.join(tmpdir, 'mef_bias.fits')
    ...     create_mef_bias_frame(fpath, num_amps=4, amp_shape=(50, 50), seed=42)
    ...     with fits.open(fpath) as hdul:
    ...         print(len(hdul))  # PRIMARY + 4 extensions
    5
    """
    validate_numeric(num_amps, 'num_amps', minimum=1, integer=True)
    validate_numeric(amp_shape[0], 'amp_rows', minimum=1, integer=True)
    validate_numeric(amp_shape[1], 'amp_cols', minimum=1, integer=True)
    
    rng = np.random.default_rng(seed)
    
    # Generate bias levels and read noises if not provided
    if bias_levels is None:
        # Realistic: each amp has slightly different bias
        base_bias = kwargs.get('bias_level', 100.0)
        bias_levels = [base_bias + rng.uniform(-5, 5) for _ in range(num_amps)]
    
    if read_noises is None:
        # Realistic: each amp has slightly different read noise
        base_rn = kwargs.get('read_noise', 5.0)
        read_noises = [base_rn * rng.uniform(0.9, 1.1) for _ in range(num_amps)]
    
    # Create extensions
    extensions_info = []
    for i in range(num_amps):
        # Create data for this amp
        amp_data = np.full(amp_shape, bias_levels[i], dtype=np.float32)
        amp_data = add_read_noise(amp_data, read_noises[i], seed=seed+i if seed else None)
        
        ext_info = {
            'XTENSION': 'IMAGE',
            'data_value': amp_data,
            'keywords': {
                'EXTNAME': f'AMP{i+1}',
                FITS_KEY_IMGTYPE: FITS_VAL_BIAS_IMGTYPE,
                FITS_KEY_EXPTIME: 0.0,
                FITS_KEY_RDNOISE: read_noises[i],
                'BIASLVL': bias_levels[i],
                'AMPID': i+1,
            },
            'nx': amp_shape[1],
            'ny': amp_shape[0],
        }
        extensions_info.append(ext_info)
    
    # Create MEF file with empty primary
    primary_keywords = {
        FITS_KEY_IMGTYPE: FITS_VAL_BIAS_IMGTYPE,
        'NEXTEND': num_amps,
        'COMMENT': 'Multi-extension FITS with simulated amplifier data'
    }
    
    _create_dummy_fits(
        filepath,
        nx=1, ny=1,
        primary_data_value=None,  # Empty primary
        primary_keywords=primary_keywords,
        extensions_info=extensions_info,
    )
    
    logger.info(f"Created MEF bias frame with {num_amps} amplifiers: {filepath}")


# ============================================================================
# Time-Series Generation (NEW in v0.3.0)
# ============================================================================

def create_bias_time_series(
    output_dir: str,
    num_frames: int = 100,
    shape: Tuple[int, int] = (512, 512),
    bias_level_start: float = 100.0,
    bias_drift_rate: float = 0.1,  # ADU per frame
    thermal_cycle_period: Optional[int] = None,
    thermal_amplitude: float = 2.0,
    **kwargs
) -> List[str]:
    """
    Create a time series of bias frames with drift and/or thermal cycling.
    
    Useful for testing long-term stability analysis.
    
    Parameters
    ----------
    output_dir : str
        Output directory for frames.
    num_frames : int, optional
        Number of frames to generate. Defaults to 100.
    shape : tuple, optional
        Image shape. Defaults to (512, 512).
    bias_level_start : float, optional
        Starting bias level. Defaults to 100.0.
    bias_drift_rate : float, optional
        Linear drift rate in ADU/frame. Defaults to 0.1.
    thermal_cycle_period : int, optional
        Period of thermal cycle in frames. If None, no cycling.
    thermal_amplitude : float, optional
        Amplitude of thermal oscillation in ADU. Defaults to 2.0.
    **kwargs
        Additional parameters passed to create_bias_frame.
    
    Returns
    -------
    list of str
        List of created file paths.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     files = create_bias_time_series(
    ...         tmpdir,
    ...         num_frames=10,
    ...         shape=(50, 50),
    ...         bias_drift_rate=0.5,
    ...         seed=42
    ...     )
    ...     print(len(files))
    10
    """
    validate_numeric(num_frames, 'num_frames', minimum=1, integer=True)
    validate_numeric(bias_level_start, 'bias_level_start', minimum=0.0)
    validate_numeric(bias_drift_rate, 'bias_drift_rate')
    validate_numeric(thermal_amplitude, 'thermal_amplitude', minimum=0.0)
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    for i in range(num_frames):
        # Calculate bias level for this frame
        bias_level = bias_level_start + bias_drift_rate * i
        
        # Add thermal cycle if requested
        if thermal_cycle_period is not None:
            phase = 2 * np.pi * i / thermal_cycle_period
            thermal_offset = thermal_amplitude * np.sin(phase)
            bias_level += thermal_offset
        
        # Create frame
        filepath = str(output_dir_path / f'bias_{i:04d}.fits')
        
        # Handle seed parameter
        frame_kwargs = kwargs.copy()
        if 'seed' in kwargs:
            frame_kwargs['seed'] = kwargs['seed'] + i
        
        create_bias_frame(
            filepath,
            shape=shape,
            bias_level=bias_level,
            **frame_kwargs
        )
        
        created_files.append(filepath)
    
    logger.info(f"Created time series of {num_frames} bias frames in {output_dir}")
    
    return created_files


def create_bias_pair_for_noise_estimation(
    output_dir: str,
    num_pairs: int = 10,
    shape: Tuple[int, int] = (512, 512),
    bias_level: float = 100.0,
    read_noise: float = 5.0,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Create pairs of bias frames for accurate read noise estimation.
    
    The two-frame difference method is more accurate than single-frame
    statistics because it removes fixed-pattern noise.
    
    Read noise = std(frame1 - frame2) / sqrt(2)
    
    Parameters
    ----------
    output_dir : str
        Output directory.
    num_pairs : int, optional
        Number of frame pairs. Defaults to 10.
    shape : tuple, optional
        Image shape. Defaults to (512, 512).
    bias_level : float, optional
        Bias level. Defaults to 100.0.
    read_noise : float, optional
        True read noise value. Defaults to 5.0.
    **kwargs
        Additional parameters.
    
    Returns
    -------
    dict
        Dictionary with 'frame1' and 'frame2' lists of file paths.
    
    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     pairs = create_bias_pair_for_noise_estimation(
    ...         tmpdir,
    ...         num_pairs=5,
    ...         shape=(100, 100),
    ...         read_noise=5.0,
    ...         seed=42
    ...     )
    ...     print(len(pairs['frame1']), len(pairs['frame2']))
    5 5
    """
    validate_numeric(num_pairs, 'num_pairs', minimum=1, integer=True)
    validate_numeric(bias_level, 'bias_level', minimum=0.0)
    validate_numeric(read_noise, 'read_noise', minimum=0.0)
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    frame1_files = []
    frame2_files = []
    
    seed_base = kwargs.get('seed', 0)
    
    for i in range(num_pairs):
        # Create pair with different noise realizations
        file1 = str(output_dir_path / f'bias_pair{i:03d}_A.fits')
        file2 = str(output_dir_path / f'bias_pair{i:03d}_B.fits')
        
        # Prepare kwargs without seed
        frame_kwargs = {k: v for k, v in kwargs.items() if k != 'seed'}
        
        create_bias_frame(file1, shape=shape, bias_level=bias_level,
                         read_noise=read_noise, seed=seed_base + i*2, **frame_kwargs)
        create_bias_frame(file2, shape=shape, bias_level=bias_level,
                         read_noise=read_noise, seed=seed_base + i*2 + 1, **frame_kwargs)
        
        frame1_files.append(file1)
        frame2_files.append(file2)
    
    logger.info(f"Created {num_pairs} bias frame pairs in {output_dir}")
    
    return {'frame1': frame1_files, 'frame2': frame2_files}


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Core function (backward compatible)
    '_create_dummy_fits',
    
    # Basic noise and artifacts
    'add_read_noise',
    'add_photon_noise',
    'add_hot_pixels',
    'add_cosmic_rays',
    'create_vignetting_pattern',
    
    # Advanced detector artifacts (NEW in v0.3.0)
    'add_cte_trails',
    'add_tree_rings',
    'add_brighter_fatter_effect',
    
    # Standard calibration frames
    'create_bias_frame',
    'create_dark_frame',
    'create_flat_frame',
    'create_science_frame',
    'create_calibration_set',
    
    # MEF support (NEW in v0.3.0)
    'create_mef_bias_frame',
    
    # Time-series generation (NEW in v0.3.0)
    'create_bias_time_series',
    'create_bias_pair_for_noise_estimation',
]


# ============================================================================
# Main / Testing
# ============================================================================

def main():
    """
    Command-line interface and test suite for fits_simulations.
    """
    import argparse
    import shutil
    import doctest
    
    parser = argparse.ArgumentParser(
        description='FITS Simulation Tools - Generate Synthetic CCD Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tests
  %(prog)s --test
  
  # Create sample calibration set
  %(prog)s --create-cal-set ./test_data --num-bias 5 --num-dark 5 --num-flat 5
  
  # Create MEF bias frame
  %(prog)s --create-mef-bias mef_bias.fits --num-amps 4
  
  # Create time series
  %(prog)s --create-time-series ./time_series --num-frames 100
  
  # Run doctests
  %(prog)s --doctest
        """
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Run basic functionality tests')
    parser.add_argument('--doctest', action='store_true',
                       help='Run doctests')
    parser.add_argument('--create-cal-set', metavar='DIR',
                       help='Create a calibration set in specified directory')
    parser.add_argument('--create-mef-bias', metavar='FILE',
                       help='Create a MEF bias frame')
    parser.add_argument('--create-time-series', metavar='DIR',
                       help='Create bias time series')
    parser.add_argument('--num-bias', type=int, default=10,
                       help='Number of bias frames (default: 10)')
    parser.add_argument('--num-dark', type=int, default=10,
                       help='Number of dark frames (default: 10)')
    parser.add_argument('--num-flat', type=int, default=10,
                       help='Number of flat frames (default: 10)')
    parser.add_argument('--num-amps', type=int, default=4,
                       help='Number of amplifiers for MEF (default: 4)')
    parser.add_argument('--num-frames', type=int, default=100,
                       help='Number of frames for time series (default: 100)')
    parser.add_argument('--shape', nargs=2, type=int, default=[512, 512],
                       metavar=('ROWS', 'COLS'),
                       help='Image dimensions (default: 512 512)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--version', action='version',
                       version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run doctests
    if args.doctest:
        logger.info("Running doctests...")
        result = doctest.testmod(verbose=args.verbose)
        if result.failed == 0:
            logger.info(f"All {result.attempted} doctests passed!")
            return 0
        else:
            logger.error(f"{result.failed} of {result.attempted} doctests failed!")
            return 1
    
    # Create MEF bias frame
    if args.create_mef_bias:
        shape = tuple(args.shape)
        logger.info(f"Creating MEF bias frame with {args.num_amps} amplifiers")
        create_mef_bias_frame(
            args.create_mef_bias,
            num_amps=args.num_amps,
            amp_shape=shape,
            seed=42
        )
        logger.info(f"Created {args.create_mef_bias}")
        return 0
    
    # Create time series
    if args.create_time_series:
        shape = tuple(args.shape)
        logger.info(f"Creating time series with {args.num_frames} frames")
        files = create_bias_time_series(
            args.create_time_series,
            num_frames=args.num_frames,
            shape=shape,
            bias_drift_rate=0.1,
            seed=42
        )
        logger.info(f"Created {len(files)} files in {args.create_time_series}")
        return 0
    
    # Create calibration set
    if args.create_cal_set:
        shape = tuple(args.shape)
        logger.info(f"Creating calibration set in {args.create_cal_set}")
        logger.info(f"Shape: {shape}, Bias: {args.num_bias}, "
                   f"Dark: {args.num_dark}, Flat: {args.num_flat}")
        
        files = create_calibration_set(
            args.create_cal_set,
            num_bias=args.num_bias,
            num_dark=args.num_dark,
            num_flat=args.num_flat,
            shape=shape
        )
        
        logger.info(f"Successfully created {sum(len(v) for v in files.values())} files")
        return 0
    
    # Run basic tests
    if args.test or len(sys.argv) == 1:
        logger.info("Running basic tests for fits_simulations.py...")
        
        temp_dir = Path("temp_fits_sim_test")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Test 1: Basic dummy FITS file with primary HDU data and keywords
            file1_path = str(temp_dir / "test_sim_1.fits")
            _create_dummy_fits(
                file1_path,
                nx=50, ny=50,
                primary_data_value=100.0,
                noise_level=5.0,
                primary_keywords={FITS_KEY_OBSERVER: "Sim Tester", FITS_KEY_EXPTIME: 10.0}
            )
            with fits.open(file1_path) as hdul:
                assert hdul[0].data is not None
                assert hdul[0].data.shape == (50, 50)
                assert hdul[0].header[FITS_KEY_OBSERVER] == 'Sim Tester'
                assert hdul[0].header[FITS_KEY_EXPTIME] == 10.0
            logger.info("✓ Test 1: Basic dummy FITS - PASSED")
            
            # Test 2: Bias frame
            file2_path = str(temp_dir / "test_bias.fits")
            create_bias_frame(file2_path, shape=(100, 100), bias_level=100, seed=42)
            with fits.open(file2_path) as hdul:
                assert hdul[0].header[FITS_KEY_IMGTYPE] == FITS_VAL_BIAS_IMGTYPE
                assert hdul[0].header[FITS_KEY_EXPTIME] == 0.0
                assert hdul[0].data.shape == (100, 100)
            logger.info("✓ Test 2: Bias frame - PASSED")
            
            # Test 3: Bias frame with overscan
            file3_path = str(temp_dir / "test_bias_overscan.fits")
            create_bias_frame(file3_path, shape=(100, 132), bias_level=100, 
                            add_sections=True, overscan_width=32, seed=42)
            with fits.open(file3_path) as hdul:
                assert FITS_KEY_DATASEC in hdul[0].header
                assert FITS_KEY_BIASSEC in hdul[0].header
                # Check overscan region has lower std (no hot pixels)
                data = hdul[0].data
                data_std = np.std(data[:, :100])
                overscan_std = np.std(data[:, 100:])
                # Overscan should have similar or slightly lower std (no hot pixels)
            logger.info("✓ Test 3: Bias frame with overscan - PASSED")
            
            # Test 4: Dark frame
            file4_path = str(temp_dir / "test_dark.fits")
            create_dark_frame(file4_path, shape=(100, 100), exposure_time=300, seed=42)
            with fits.open(file4_path) as hdul:
                assert hdul[0].header[FITS_KEY_IMGTYPE] == FITS_VAL_DARK_IMGTYPE
                assert hdul[0].header[FITS_KEY_EXPTIME] == 300.0
            logger.info("✓ Test 4: Dark frame - PASSED")
            
            # Test 5: Flat frame
            file5_path = str(temp_dir / "test_flat.fits")
            create_flat_frame(file5_path, shape=(100, 100), mean_level=10000, seed=42)
            with fits.open(file5_path) as hdul:
                assert hdul[0].header[FITS_KEY_IMGTYPE] == FITS_VAL_FLAT_IMGTYPE
                assert 8000 < np.mean(hdul[0].data) < 12000  # Should be near 10000
            logger.info("✓ Test 5: Flat frame - PASSED")
            
            # Test 6: Science frame
            file6_path = str(temp_dir / "test_science.fits")
            create_science_frame(file6_path, shape=(100, 100), num_stars=5, seed=42)
            with fits.open(file6_path) as hdul:
                assert hdul[0].header[FITS_KEY_IMGTYPE] == FITS_VAL_LIGHT_IMGTYPE
                assert np.max(hdul[0].data) > np.mean(hdul[0].data) * 2  # Stars should be bright
            logger.info("✓ Test 6: Science frame - PASSED")
            
            # Test 7: Calibration set
            cal_dir = str(temp_dir / "cal_set")
            files = create_calibration_set(cal_dir, num_bias=3, num_dark=3, num_flat=3,
                                          shape=(50, 50))
            assert len(files['bias']) == 3
            assert len(files['dark']) == 3
            assert len(files['flat']) == 3
            logger.info("✓ Test 7: Calibration set - PASSED")
            
            # Test 8: MEF bias frame
            file8_path = str(temp_dir / "test_mef_bias.fits")
            create_mef_bias_frame(file8_path, num_amps=4, amp_shape=(50, 50), seed=42)
            with fits.open(file8_path) as hdul:
                assert len(hdul) == 5  # PRIMARY + 4 extensions
                assert hdul[1].header['EXTNAME'] == 'AMP1'
            logger.info("✓ Test 8: MEF bias frame - PASSED")
            
            # Test 9: CTE trails
            data = np.zeros((100, 100))
            data[50, 50] = 10000
            with_cte = add_cte_trails(data, cte_efficiency=0.9999, seed=42)
            assert np.sum(with_cte[51:60, 50]) > 0  # Should have trail
            logger.info("✓ Test 9: CTE trails - PASSED")
            
            # Test 10: Tree rings
            data = np.ones((100, 100)) * 1000
            with_rings = add_tree_rings(data, amplitude=0.05, num_rings=3, seed=42)
            assert np.std(with_rings) > np.std(data)  # Should have more variation
            logger.info("✓ Test 10: Tree rings - PASSED")
            
            logger.info("\n" + "="*70)
            logger.info("All tests PASSED! ✓")
            logger.info("="*70)
            
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up {temp_dir}")
        
        return 0
    
    # No arguments
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
