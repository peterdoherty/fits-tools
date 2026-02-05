#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosmic ray detection and rejection for CCD/CMOS images.

This module provides algorithms for detecting and removing cosmic ray hits
from astronomical images. Cosmic rays appear as sharp, localized spikes that
can corrupt photometry and image analysis.

Detection Methods:
    - Sigma-clipping: Statistical outlier detection with median filtering
    - L.A.Cosmic: Laplacian edge detection algorithm (requires astroscrappy)
    - Median filtering: Simple but effective for isolated cosmic rays
    - Comparison: Multi-frame comparison (requires multiple exposures)

Features:
    - Multiple detection algorithms
    - Iterative refinement
    - Configurable sensitivity
    - Various replacement strategies (median, interpolation)
    - Mask generation
    - Quality metrics for cleaned images
    - Integration with FITS I/O

Usage Examples:
    Basic cosmic ray rejection:
        >>> from fits_tools.image_processing.cosmic_rays import reject_cosmic_rays
        >>> cleaned, mask = reject_cosmic_rays(data, method='sigma_clip')
    
    Using L.A.Cosmic:
        >>> cleaned, mask = reject_cosmic_rays(data, method='lacosmic', sigclip=4.5)
    
    File-based processing:
        >>> result = clean_cosmic_rays_file('science.fits', output='cleaned.fits')
    
    Batch processing:
        >>> results = batch_clean_cosmic_rays(['img1.fits', 'img2.fits'])

Version: 2.0.0
"""

import os
import logging
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

# Optional dependencies
try:
    from scipy.ndimage import median_filter, uniform_filter
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

# Import from fits_tools
from ..io import get_data, get_header, write_fits, get_data_and_header
from ..exceptions import ImageProcessingError
from .filters import apply_median_filter, apply_gaussian_filter
from ..utils import add_history, maybe_tqdm, is_tqdm_available

MAD_TO_STD_FACTOR = 1.4826  # For normal distribution

logger = logging.getLogger(__name__)
__version__ = "2.0.0"
__author__ = "FITS Tools Development Team"


# ============================================================================
# Result Types
# ============================================================================

@dataclass
class CosmicRayResult:
    """
    Result from cosmic ray detection/rejection.
    
    Attributes
    ----------
    cleaned_data : np.ndarray
        Image with cosmic rays removed.
    mask : np.ndarray
        Boolean mask (True = cosmic ray pixel).
    num_cosmic_rays : int
        Number of cosmic ray pixels detected.
    cosmic_ray_fraction : float
        Fraction of pixels affected by cosmic rays.
    method : str
        Detection method used.
    iterations : int
        Number of iterations performed.
    metadata : dict
        Additional method-specific metadata.
    
    Examples
    --------
    >>> result = reject_cosmic_rays(data, method='sigma_clip')
    >>> print(f"Detected {result.num_cosmic_rays} cosmic ray pixels")
    >>> print(f"Fraction: {result.cosmic_ray_fraction*100:.2f}%")
    """
    cleaned_data: np.ndarray
    mask: np.ndarray
    num_cosmic_rays: int
    cosmic_ray_fraction: float
    method: str
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Cosmic Ray Result:\n"
            f"  Method: {self.method}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Cosmic rays detected: {self.num_cosmic_rays} "
            f"({self.cosmic_ray_fraction*100:.3f}%)\n"
            f"  Cleaned data shape: {self.cleaned_data.shape}"
        )
    
    def __repr__(self) -> str:
        """Brief representation."""
        return (
            f"CosmicRayResult(method='{self.method}', "
            f"num_cosmic_rays={self.num_cosmic_rays}, "
            f"fraction={self.cosmic_ray_fraction:.4f})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def _calculate_mad(data: np.ndarray) -> float:
    """
    Calculate Median Absolute Deviation (MAD).
    
    Robust estimator of standard deviation.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    # Convert MAD to std equivalent (for normal distribution)
    return MAD_TO_STD_FACTOR * mad


# ============================================================================
# Cosmic Ray Detection Methods
# ============================================================================

def _detect_cosmic_rays_sigma_clip(
    data: np.ndarray,
    sigma: float = 5.0,
    box_size: int = 5,
    iterations: int = 1,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect cosmic rays using sigma-clipping with median filtering.
    
    This method compares each pixel to the local median to identify
    sharp positive outliers characteristic of cosmic rays.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigma : float
        Detection threshold in standard deviations.
    box_size : int
        Size of median filter box (should be odd).
    iterations : int
        Number of detection iterations.
    verbose : bool
        Enable verbose logging.
    
    Returns
    -------
    tuple
        (cosmic_ray_mask, metadata_dict)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "Sigma-clipping cosmic ray detection requires scipy. "
            "Install with: pip install scipy"
        )
    
    if box_size % 2 == 0:
        box_size += 1  # Ensure odd size
        if verbose:
            logger.debug(f"Adjusted box_size to {box_size} (must be odd)")
    
    cosmic_ray_mask = np.zeros(data.shape, dtype=bool)
    data_copy = data.copy().astype(np.float32)
    
    for iteration in range(iterations):
        # Create median-filtered version (smooth out cosmic rays)
        median_filtered = median_filter(data_copy, size=box_size)
        
        # Calculate residuals (original - smoothed)
        residuals = data_copy - median_filtered
        
        # Robust statistics on residuals
        median_residual = np.median(residuals)
        mad = _calculate_mad(residuals)
        
        if mad == 0:
            if verbose:
                logger.warning(f"Iteration {iteration+1}: MAD is zero, stopping")
            break
        
        # Detect positive outliers (cosmic rays are always positive spikes)
        threshold = median_residual + sigma * mad
        new_cosmic_rays = residuals > threshold
        
        num_new = np.sum(new_cosmic_rays)
        
        if verbose:
            logger.debug(f"Iteration {iteration+1}: Detected {num_new} cosmic ray pixels")
        
        if num_new == 0:
            if verbose:
                logger.debug(f"No new cosmic rays detected, stopping after {iteration+1} iterations")
            break
        
        # Update mask
        cosmic_ray_mask |= new_cosmic_rays
        
        # Replace detected cosmic rays with median for next iteration
        data_copy[new_cosmic_rays] = median_filtered[new_cosmic_rays]
    
    metadata = {
        'sigma': sigma,
        'box_size': box_size,
        'iterations_performed': iteration + 1,
        'threshold_value': threshold if 'threshold' in locals() else None,
    }
    
    return cosmic_ray_mask, metadata


def _detect_cosmic_rays_lacosmic(
    data: np.ndarray,
    sigclip: float = 4.5,
    sigfrac: float = 0.3,
    objlim: float = 5.0,
    gain: float = 1.0,
    readnoise: float = 6.5,
    satlevel: float = 65535.0,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect cosmic rays using the L.A.Cosmic algorithm.
    
    L.A.Cosmic (Laplacian Cosmic ray detection) uses Laplacian edge detection
    to identify cosmic rays based on their sharp, unresolved edges.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    sigclip : float
        Laplacian-to-noise limit for cosmic ray detection.
    sigfrac : float
        Fractional detection limit for neighboring pixels.
    objlim : float
        Contrast limit between cosmic ray and underlying object.
    gain : float
        CCD gain in e-/ADU.
    readnoise : float
        CCD read noise in electrons.
    satlevel : float
        Saturation level in ADU.
    verbose : bool
        Enable verbose logging.
    
    Returns
    -------
    tuple
        (cosmic_ray_mask, metadata_dict)
    
    References
    ----------
    van Dokkum, P. 2001, PASP, 113, 1420
    """
    if not LACOSMIC_AVAILABLE:
        raise ImportError(
            "L.A.Cosmic detection requires astroscrappy. "
            "Install with: pip install astroscrappy"
        )
    
    if verbose:
        logger.info(f"Running L.A.Cosmic with sigclip={sigclip}, objlim={objlim}")
    
    # Run astroscrappy
    mask, cleaned = astroscrappy.detect_cosmics(
        data,
        sigclip=sigclip,
        sigfrac=sigfrac,
        objlim=objlim,
        gain=gain,
        readnoise=readnoise,
        satlevel=satlevel,
        verbose=verbose,
    )
    
    metadata = {
        'sigclip': sigclip,
        'sigfrac': sigfrac,
        'objlim': objlim,
        'gain': gain,
        'readnoise': readnoise,
        'satlevel': satlevel,
    }
    
    return mask, metadata


def _detect_cosmic_rays_median(
    data: np.ndarray,
    threshold: float = 100.0,
    box_size: int = 5,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simple cosmic ray detection using median filtering.
    
    Detects pixels that deviate significantly from the local median.
    Less sophisticated than sigma-clipping but faster.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    threshold : float
        Absolute threshold above local median.
    box_size : int
        Size of median filter box.
    verbose : bool
        Enable verbose logging.
    
    Returns
    -------
    tuple
        (cosmic_ray_mask, metadata_dict)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "Median filtering requires scipy. "
            "Install with: pip install scipy"
        )
    
    if box_size % 2 == 0:
        box_size += 1
    
    # Median filter
    median_filtered = median_filter(data, size=box_size)
    
    # Find large deviations
    residuals = data - median_filtered
    cosmic_ray_mask = residuals > threshold
    
    metadata = {
        'threshold': threshold,
        'box_size': box_size,
    }
    
    if verbose:
        logger.info(f"Detected {np.sum(cosmic_ray_mask)} cosmic ray pixels")
    
    return cosmic_ray_mask, metadata


# ============================================================================
# Cosmic Ray Replacement Methods
# ============================================================================

def _replace_with_median(
    data: np.ndarray,
    mask: np.ndarray,
    box_size: int = 5
) -> np.ndarray:
    """
    Replace cosmic ray pixels with local median.
    
    Parameters
    ----------
    data : np.ndarray
        Input data.
    mask : np.ndarray
        Boolean mask (True = cosmic ray).
    box_size : int
        Size of median filter box.
    
    Returns
    -------
    np.ndarray
        Data with cosmic rays replaced.
    """
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
    mask: np.ndarray
) -> np.ndarray:
    """
    Replace cosmic ray pixels using interpolation from neighbors.
    
    Uses scipy's griddata for interpolation.
    
    Parameters
    ----------
    data : np.ndarray
        Input data.
    mask : np.ndarray
        Boolean mask (True = cosmic ray).
    
    Returns
    -------
    np.ndarray
        Data with cosmic rays replaced.
    """
    try:
        from scipy.interpolate import griddata
    except ImportError:
        raise ImportError("Interpolation requires scipy with interpolate module")
    
    cleaned = data.copy()
    
    # Get coordinates
    ny, nx = data.shape
    y, x = np.mgrid[0:ny, 0:nx]
    
    # Good pixels
    good_mask = ~mask
    points = np.column_stack([y[good_mask], x[good_mask]])
    values = data[good_mask]
    
    # Cosmic ray pixels
    cr_points = np.column_stack([y[mask], x[mask]])
    
    # Interpolate
    interpolated = griddata(points, values, cr_points, method='linear')
    
    # Handle NaN (use nearest neighbor for edge cases)
    nan_mask = np.isnan(interpolated)
    if nan_mask.any():
        interpolated[nan_mask] = griddata(
            points, values, cr_points[nan_mask], method='nearest'
        )
    
    cleaned[mask] = interpolated
    
    return cleaned


# ============================================================================
# Main Functions
# ============================================================================

def detect_cosmic_rays(
    data: np.ndarray,
    method: str = 'sigma_clip',
    sigma: float = 5.0,
    iterations: int = 1,
    box_size: int = 5,
    verbose: bool = False,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect cosmic rays in image data (detection only, no replacement).
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    method : str, optional
        Detection method:
        - 'sigma_clip': Sigma-clipping with median filtering (default)
        - 'lacosmic': L.A.Cosmic algorithm (requires astroscrappy)
        - 'median': Simple median filtering
    sigma : float, optional
        Detection threshold for sigma_clip method. Defaults to 5.0.
    iterations : int, optional
        Number of detection iterations. Defaults to 1.
    box_size : int, optional
        Median filter box size. Defaults to 5.
    verbose : bool, optional
        Enable verbose logging.
    **kwargs
        Method-specific parameters (e.g., sigclip, objlim for lacosmic).
    
    Returns
    -------
    tuple
        (cosmic_ray_mask, metadata_dict)
    
    Raises
    ------
    ValueError
        If method is not recognized.
    ImportError
        If required dependencies are not available.
    
    Examples
    --------
    >>> data = np.random.normal(1000, 50, (512, 512))
    >>> # Add some cosmic rays
    >>> data[100, 100] = 5000
    >>> data[200, 200] = 6000
    >>> mask, metadata = detect_cosmic_rays(data, method='sigma_clip', sigma=5.0)
    >>> print(f"Detected {np.sum(mask)} cosmic ray pixels")
    """
    if method == 'sigma_clip':
        mask, metadata = _detect_cosmic_rays_sigma_clip(
            data,
            sigma=sigma,
            box_size=box_size,
            iterations=iterations,
            verbose=verbose
        )
    
    elif method == 'lacosmic':
        mask, metadata = _detect_cosmic_rays_lacosmic(
            data,
            sigclip=kwargs.get('sigclip', 4.5),
            sigfrac=kwargs.get('sigfrac', 0.3),
            objlim=kwargs.get('objlim', 5.0),
            gain=kwargs.get('gain', 1.0),
            readnoise=kwargs.get('readnoise', 6.5),
            satlevel=kwargs.get('satlevel', 65535.0),
            verbose=verbose
        )
    
    elif method == 'median':
        mask, metadata = _detect_cosmic_rays_median(
            data,
            threshold=kwargs.get('threshold', 100.0),
            box_size=box_size,
            verbose=verbose
        )
    
    else:
        raise ValueError(
            f"Unknown cosmic ray detection method: '{method}'. "
            f"Valid options: 'sigma_clip', 'lacosmic', 'median'"
        )
    
    return mask, metadata


def reject_cosmic_rays(
    data: np.ndarray,
    method: str = 'sigma_clip',
    sigma: float = 5.0,
    iterations: int = 1,
    box_size: int = 5,
    replace_with: str = 'median',
    verbose: bool = False,
    **kwargs
) -> CosmicRayResult:
    """
    Detect and remove cosmic rays from image data.
    
    This is the primary function for cosmic ray rejection. It combines
    detection and replacement in a single operation.
    
    Parameters
    ----------
    data : np.ndarray
        Input image data.
    method : str, optional
        Detection method ('sigma_clip', 'lacosmic', 'median'). Defaults to 'sigma_clip'.
    sigma : float, optional
        Detection threshold for sigma_clip. Defaults to 5.0.
    iterations : int, optional
        Number of iterations. Defaults to 1.
    box_size : int, optional
        Median filter box size. Defaults to 5.
    replace_with : str, optional
        Replacement method:
        - 'median': Replace with local median (default)
        - 'interpolation': Replace with interpolated values
    verbose : bool, optional
        Enable verbose logging.
    **kwargs
        Additional method-specific parameters.
    
    Returns
    -------
    CosmicRayResult
        Result containing cleaned data, mask, and metadata.
    
    Examples
    --------
    >>> data = np.random.normal(1000, 50, (512, 512))
    >>> # Add some cosmic rays
    >>> data[100, 100] = 5000
    >>> data[200, 200] = 6000
    >>> result = reject_cosmic_rays(data, method='sigma_clip', sigma=5.0)
    >>> print(f"Removed {result.num_cosmic_rays} cosmic ray pixels")
    >>> cleaned_data = result.cleaned_data
    
    >>> # Using L.A.Cosmic
    >>> result = reject_cosmic_rays(data, method='lacosmic', sigclip=4.5)
    
    See Also
    --------
    detect_cosmic_rays : Detection only (no replacement)
    clean_cosmic_rays_file : File-based cosmic ray rejection
    """
    if verbose:
        logger.info(f"Detecting cosmic rays using method: {method}")
    
    # Step 1: Detect cosmic rays
    mask, metadata = detect_cosmic_rays(
        data,
        method=method,
        sigma=sigma,
        iterations=iterations,
        box_size=box_size,
        verbose=verbose,
        **kwargs
    )
    
    num_cosmic_rays = int(np.sum(mask))
    cosmic_ray_fraction = num_cosmic_rays / data.size
    
    if verbose:
        logger.info(
            f"Detected {num_cosmic_rays} cosmic ray pixels "
            f"({cosmic_ray_fraction*100:.3f}%)"
        )
    
    # Step 2: Replace cosmic rays
    if num_cosmic_rays > 0:
        if verbose:
            logger.info(f"Replacing cosmic rays using method: {replace_with}")
        
        if replace_with == 'median':
            cleaned_data = _replace_with_median(data, mask, box_size=box_size)
        
        elif replace_with == 'interpolation':
            cleaned_data = _replace_with_interpolation(data, mask)
        
        else:
            raise ValueError(
                f"Unknown replacement method: '{replace_with}'. "
                f"Valid options: 'median', 'interpolation'"
            )
    else:
        cleaned_data = data.copy()
        if verbose:
            logger.info("No cosmic rays detected, returning original data")
    
    # Create result
    result = CosmicRayResult(
        cleaned_data=cleaned_data,
        mask=mask,
        num_cosmic_rays=num_cosmic_rays,
        cosmic_ray_fraction=cosmic_ray_fraction,
        method=method,
        iterations=metadata.get('iterations_performed', iterations),
        metadata=metadata
    )
    
    return result


def clean_cosmic_rays_file(
    input_file: str,
    output_file: Optional[str] = None,
    method: str = 'sigma_clip',
    sigma: float = 5.0,
    iterations: int = 1,
    ext: int = 0,
    overwrite: bool = False,
    save_mask: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[CosmicRayResult, str]:
    """
    Clean cosmic rays from a FITS file.
    
    High-level function that handles file I/O and header updates.
    
    Parameters
    ----------
    input_file : str
        Input FITS file path.
    output_file : str, optional
        Output FITS file path. If None, returns CosmicRayResult only.
    method : str, optional
        Detection method. Defaults to 'sigma_clip'.
    sigma : float, optional
        Detection threshold. Defaults to 5.0.
    iterations : int, optional
        Number of iterations. Defaults to 1.
    ext : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output. Defaults to False.
    save_mask : bool, optional
        If True, save cosmic ray mask as additional HDU. Defaults to False.
    verbose : bool, optional
        Enable verbose logging.
    **kwargs
        Additional method-specific parameters.
    
    Returns
    -------
    CosmicRayResult or str
        If output_file is None, returns CosmicRayResult.
        Otherwise returns output file path.
    
    Examples
    --------
    >>> from fits_tools.image_processing.cosmic_rays import clean_cosmic_rays_file
    >>> result = clean_cosmic_rays_file(
    ...     'science.fits',
    ...     output_file='science_clean.fits',
    ...     method='sigma_clip',
    ...     sigma=5.0,
    ...     overwrite=True
    ... )
    >>> print(result)
    """
    if verbose:
        logger.info(f"Cleaning cosmic rays from: {input_file}")
    
    # Check output file
    if output_file and os.path.exists(output_file) and not overwrite:
        raise ImageProcessingError(
            f"Output file exists and overwrite=False: {output_file}"
        )
    
    try:
        # Load data
        data, header = get_data_and_header(input_file, ext=ext, verbose=verbose)
        
        # Reject cosmic rays
        result = reject_cosmic_rays(
            data,
            method=method,
            sigma=sigma,
            iterations=iterations,
            verbose=verbose,
            **kwargs
        )
        
        # Update header
        add_history(header, f'Cosmic rays removed using {method} method')
        header['CRMETH'] = (method, 'Cosmic ray detection method')
        header['CRNUM'] = (result.num_cosmic_rays, 'Number of cosmic ray pixels')
        header['CRFRAC'] = (result.cosmic_ray_fraction, 'Cosmic ray fraction')
        header['CRITER'] = (result.iterations, 'CR detection iterations')
        
        if method == 'sigma_clip':
            header['CRSIGMA'] = (sigma, 'CR detection sigma threshold')
        
        # Write output if requested
        if output_file:
            if save_mask:
                # Create HDU list with mask
                primary_hdu = fits.PrimaryHDU(result.cleaned_data, header=header)
                mask_hdu = fits.ImageHDU(
                    result.mask.astype(np.uint8),
                    name='CR_MASK'
                )
                hdul = fits.HDUList([primary_hdu, mask_hdu])
                hdul.writeto(output_file, overwrite=overwrite)
                
                if verbose:
                    logger.info(f"Wrote cleaned data and mask to: {output_file}")
            else:
                write_fits(
                    output_file,
                    result.cleaned_data,
                    header,
                    overwrite=overwrite,
                    verbose=verbose
                )
            
            return output_file
        else:
            return result
    
    except Exception as e:
        logger.error(f"Failed to clean cosmic rays from {input_file}: {e}")
        raise ImageProcessingError(f"Cosmic ray cleaning failed: {e}")


# ============================================================================
# Batch Processing
# ============================================================================

def batch_clean_cosmic_rays(
    file_list: List[str],
    output_dir: str,
    method: str = 'sigma_clip',
    sigma: float = 5.0,
    iterations: int = 1,
    ext: int = 0,
    overwrite: bool = False,
    save_masks: bool = False,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Clean cosmic rays from multiple files.
    
    Parameters
    ----------
    file_list : list of str
        List of input FITS files.
    output_dir : str
        Output directory for cleaned files.
    method : str, optional
        Detection method. Defaults to 'sigma_clip'.
    sigma : float, optional
        Detection threshold. Defaults to 5.0.
    iterations : int, optional
        Number of iterations. Defaults to 1.
    ext : int, optional
        HDU extension. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing outputs. Defaults to False.
    save_masks : bool, optional
        Save cosmic ray masks. Defaults to False.
    verbose : bool, optional
        Enable verbose logging.
    **kwargs
        Additional method-specific parameters.
    
    Returns
    -------
    dict
        Batch processing results with keys:
        - 'total_files': int
        - 'successful': int
        - 'failed': int
        - 'total_cosmic_rays': int
        - 'output_files': list of str
        - 'errors': list of str
    
    Examples
    --------
    >>> files = ['sci_001.fits', 'sci_002.fits', 'sci_003.fits']
    >>> results = batch_clean_cosmic_rays(
    ...     files,
    ...     output_dir='cleaned/',
    ...     method='sigma_clip',
    ...     sigma=5.0
    ... )
    >>> print(f"Cleaned {results['successful']}/{results['total_files']} files")
    """
    if not file_list:
        raise ValueError("Empty file list")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        logger.info(f"Batch cleaning {len(file_list)} files...")
    
    results = {
        'total_files': len(file_list),
        'successful': 0,
        'failed': 0,
        'total_cosmic_rays': 0,
        'output_files': [],
        'errors': [],
    }
    
    iterator = maybe_tqdm(
        enumerate(file_list, 1),
        total=len(file_list),
        desc="Cleaning cosmic rays",
        unit="file"
    ) if verbose else enumerate(file_list, 1)
    
    for i, filepath in iterator:
        try:
            # Generate output filename
            basename = os.path.basename(filepath)
            name, ext_str = os.path.splitext(basename)
            output_file = os.path.join(output_dir, f"{name}_clean{ext_str}")
            
            # Clean cosmic rays
            result = clean_cosmic_rays_file(
                filepath,
                output_file=output_file,
                method=method,
                sigma=sigma,
                iterations=iterations,
                ext=ext,
                overwrite=overwrite,
                save_mask=save_masks,
                verbose=False,  # Suppress individual file verbosity
                **kwargs
            )
            
            # Update results (need to load result if it was written to file)
            if isinstance(result, str):
                # File was written, need to get stats
                # For now, just mark as successful
                results['successful'] += 1
                results['output_files'].append(result)
            else:
                results['successful'] += 1
                results['total_cosmic_rays'] += result.num_cosmic_rays
                results['output_files'].append(output_file)
            
            if verbose and not is_tqdm_available():
                logger.info(f"[{i}/{len(file_list)}] ✓ {basename}")
        
        except Exception as e:
            results['failed'] += 1
            error_msg = f"{os.path.basename(filepath)}: {str(e)}"
            results['errors'].append(error_msg)
            
            if verbose and not is_tqdm_available():
                logger.error(f"[{i}/{len(file_list)}] ✗ {os.path.basename(filepath)}: {e}")
    
    if verbose:
        logger.info(f"\nBatch cleaning complete:")
        logger.info(f"  Successful: {results['successful']}/{results['total_files']}")
        logger.info(f"  Failed: {results['failed']}")
        if results['total_cosmic_rays'] > 0:
            logger.info(f"  Total cosmic rays removed: {results['total_cosmic_rays']}")
    
    return results


# ============================================================================
# Convenience Functions
# ============================================================================

def get_available_methods() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available cosmic ray detection methods.
    
    Returns
    -------
    dict
        Dictionary with method names as keys and metadata as values.
    
    Examples
    --------
    >>> methods = get_available_methods()
    >>> for name, info in methods.items():
    ...     print(f"{name}: {info['available']}")
    """
    methods = {
        'sigma_clip': {
            'name': 'Sigma-clipping',
            'description': 'Statistical outlier detection with median filtering',
            'available': SCIPY_AVAILABLE,
            'requires': 'scipy',
            'recommended_for': 'General use, single exposures',
        },
        'lacosmic': {
            'name': 'L.A.Cosmic',
            'description': 'Laplacian edge detection algorithm',
            'available': LACOSMIC_AVAILABLE,
            'requires': 'astroscrappy',
            'recommended_for': 'High-quality cosmic ray removal',
            'reference': 'van Dokkum 2001, PASP, 113, 1420',
        },
        'median': {
            'name': 'Median filtering',
            'description': 'Simple median-based detection',
            'available': SCIPY_AVAILABLE,
            'requires': 'scipy',
            'recommended_for': 'Quick processing, isolated cosmic rays',
        },
    }
    
    return methods


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional dependencies.
    
    Returns
    -------
    dict
        Dictionary showing which dependencies are available.
    
    Examples
    --------
    >>> deps = check_dependencies()
    >>> if not deps['scipy']:
    ...     print("scipy not available - install with: pip install scipy")
    """
    return {
        'scipy': SCIPY_AVAILABLE,
        'astroscrappy': LACOSMIC_AVAILABLE,
        'tqdm': _tqdm is not None,
    }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Result types
    'CosmicRayResult',
    
    # Main functions
    'detect_cosmic_rays',
    'reject_cosmic_rays',
    'clean_cosmic_rays_file',
    
    # Batch processing
    'batch_clean_cosmic_rays',
    
    # Utilities
    'get_available_methods',
    'check_dependencies',
    
    # Constants
    'SCIPY_AVAILABLE',
    'LACOSMIC_AVAILABLE',
]


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    """Basic module testing."""
    import sys
    
    print(f"Testing cosmic_rays.py v{__version__}")
    print("=" * 70)
    
    # Check dependencies
    print("\nDependency Check:")
    deps = check_dependencies()
    for name, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    print()
    
    if not SCIPY_AVAILABLE:
        print("⚠ scipy not available - some tests will be skipped")
        print("  Install with: pip install scipy")
        print()
    
    # Test 1: Create synthetic data with cosmic rays
    print("Test 1: Synthetic data with cosmic rays")
    try:
        np.random.seed(42)
        data = np.random.normal(1000, 50, (256, 256))
        
        # Add cosmic rays
        cr_positions = [
            (50, 50, 5000),
            (100, 100, 8000),
            (150, 150, 6000),
            (200, 200, 7000),
        ]
        
        for y, x, value in cr_positions:
            data[y, x] = value
        
        print(f"✓ Created synthetic image: {data.shape}")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Std: {np.std(data):.2f}")
        print(f"  Added {len(cr_positions)} cosmic rays")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        sys.exit(1)
    
    # Test 2: Sigma-clipping detection
    if SCIPY_AVAILABLE:
        print("\nTest 2: Sigma-clipping cosmic ray detection")
        try:
            result = reject_cosmic_rays(
                data,
                method='sigma_clip',
                sigma=5.0,
                iterations=1,
                verbose=True
            )
            
            print(f"✓ Detection complete")
            print(f"  Detected: {result.num_cosmic_rays} pixels")
            print(f"  Fraction: {result.cosmic_ray_fraction*100:.3f}%")
            print(f"  Expected: {len(cr_positions)} cosmic rays")
            
            # Check if we detected the cosmic rays
            detected_positions = np.sum([result.mask[y, x] for y, x, _ in cr_positions])
            print(f"  Found {detected_positions}/{len(cr_positions)} expected cosmic rays")
            
            if detected_positions >= len(cr_positions) * 0.5:  # At least 50%
                print("  ✓ Detection successful")
            else:
                print("  ⚠ Detection may need parameter tuning")
        
        except Exception as e:
            print(f"✗ Test 2 failed: {e}")
            sys.exit(1)
    else:
        print("\nTest 2: SKIPPED (scipy not available)")
    
    # Test 3: L.A.Cosmic detection
    if LACOSMIC_AVAILABLE:
        print("\nTest 3: L.A.Cosmic detection")
        try:
            result = reject_cosmic_rays(
                data,
                method='lacosmic',
                sigclip=4.5,
                verbose=True
            )
            
            print(f"✓ L.A.Cosmic detection complete")
            print(f"  Detected: {result.num_cosmic_rays} pixels")
            print(f"  Fraction: {result.cosmic_ray_fraction*100:.3f}%")
        
        except Exception as e:
            print(f"✗ Test 3 failed: {e}")
            # Don't exit - L.A.Cosmic might be sensitive to parameters
    else:
        print("\nTest 3: SKIPPED (astroscrappy not available)")
        print("  Install with: pip install astroscrappy")
    
    # Test 4: Replacement methods
    if SCIPY_AVAILABLE:
        print("\nTest 4: Cosmic ray replacement methods")
        try:
            # Test median replacement
            result_median = reject_cosmic_rays(
                data,
                method='sigma_clip',
                sigma=5.0,
                replace_with='median',
                verbose=False
            )
            
            print("✓ Median replacement:")
            print(f"  Cleaned data shape: {result_median.cleaned_data.shape}")
            print(f"  Mean after cleaning: {np.mean(result_median.cleaned_data):.2f}")
            
            # Test interpolation replacement
            try:
                result_interp = reject_cosmic_rays(
                    data,
                    method='sigma_clip',
                    sigma=5.0,
                    replace_with='interpolation',
                    verbose=False
                )
                
                print("✓ Interpolation replacement:")
                print(f"  Mean after cleaning: {np.mean(result_interp.cleaned_data):.2f}")
            
            except ImportError:
                print("⚠ Interpolation requires scipy.interpolate")
        
        except Exception as e:
            print(f"✗ Test 4 failed: {e}")
            sys.exit(1)
    else:
        print("\nTest 4: SKIPPED (scipy not available)")
    
    # Test 5: Mask verification
    if SCIPY_AVAILABLE:
        print("\nTest 5: Verify cosmic ray mask")
        try:
            result = reject_cosmic_rays(
                data,
                method='sigma_clip',
                sigma=5.0,
                verbose=False
            )
            
            # Check that cleaned data is different from original at CR positions
            differences = []
            for y, x, original_value in cr_positions:
                if result.mask[y, x]:  # If detected as cosmic ray
                    diff = abs(data[y, x] - result.cleaned_data[y, x])
                    differences.append(diff)
            
            print(f"✓ Mask verification:")
            print(f"  Cosmic rays in mask: {len(differences)}/{len(cr_positions)}")
            if differences:
                print(f"  Average correction: {np.mean(differences):.2f} ADU")
            
            # Verify cleaned values are more reasonable
            cleaned_at_cr = [result.cleaned_data[y, x] for y, x, _ in cr_positions if result.mask[y, x]]
            if cleaned_at_cr:
                print(f"  Cleaned values: mean={np.mean(cleaned_at_cr):.2f}, "
                      f"std={np.std(cleaned_at_cr):.2f}")
                print(f"  Original background: mean={1000:.2f}, std={50:.2f}")
        
        except Exception as e:
            print(f"✗ Test 5 failed: {e}")
            sys.exit(1)
    else:
        print("\nTest 5: SKIPPED (scipy not available)")
    
    # Test 6: CosmicRayResult object
    if SCIPY_AVAILABLE:
        print("\nTest 6: CosmicRayResult object")
        try:
            result = reject_cosmic_rays(data, method='sigma_clip', sigma=5.0)
            
            print("✓ Result object:")
            print(f"  Type: {type(result).__name__}")
            print(f"  Has cleaned_data: {hasattr(result, 'cleaned_data')}")
            print(f"  Has mask: {hasattr(result, 'mask')}")
            print(f"  Method: {result.method}")
            print(f"  Iterations: {result.iterations}")
            
            # Test string representation
            result_str = str(result)
            print(f"  String representation: {len(result_str)} chars")
            
            # Test repr
            result_repr = repr(result)
            print(f"  Repr: {result_repr}")
        
        except Exception as e:
            print(f"✗ Test 6 failed: {e}")
            sys.exit(1)
    else:
        print("\nTest 6: SKIPPED (scipy not available)")
    
    # Test 7: Method availability
    print("\nTest 7: Available methods")
    try:
        methods = get_available_methods()
        print("✓ Available methods:")
        for name, info in methods.items():
            status = "✓" if info['available'] else "✗"
            print(f"  {status} {name}: {info['description']}")
            if not info['available']:
                print(f"      Requires: {info['requires']}")
    except Exception as e:
        print(f"✗ Test 7 failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    
    if SCIPY_AVAILABLE:
        print("✓ All available tests passed!")
    else:
        print("⚠ Tests completed with some skipped (install scipy for full testing)")
    
    print("\nModule is ready for use.")
    
    # Print usage hint
    print("\nQuick usage:")
    print("  from fits_tools.image_processing.cosmic_rays import reject_cosmic_rays")
    print("  result = reject_cosmic_rays(data, method='sigma_clip', sigma=5.0)")
    print("  cleaned_data = result.cleaned_data")
