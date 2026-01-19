#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for FITS file handling.

This module provides comprehensive I/O utilities for FITS files including:
    - Reading data and headers from single or multi-extension FITS files
    - Writing FITS files with optional compression
    - Region extraction (DATASEC, BIASSEC, TRIMSEC, DARKSEC)
    - Memory-mapped file support for large datasets
    - FITS file inspection and validation

Version History:
    0.5.0 - Added memmap support for large files and get_all_data() for 
            multi-extension FITS files.
    0.6.0 - Refactored to use fits_tools modules, integrated Section class,
            better error handling, improved type hints
"""

import argparse
import doctest
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from .constants import (
    FITS_KEY_IMGTYPE, 
    FITS_KEY_EXPTIME, 
    FITS_KEY_OBSERVER, 
    FITS_KEY_BIASSEC, 
    FITS_KEY_DATASET, 
    FITS_KEY_TRIMSEC, 
    FITS_KEY_DARKSEC,
    FITS_KEY_DATASEC,
    RegionType, 
)

from .exceptions import (
    FileHandlingError,
    HeaderKeywordError,
    InvalidHDUIndexError,
    FitsToolsError,
    RegionError,
)

from .sections import (
    Section,
    get_section_by_keyword,
)

from .utils import (
    is_fits_file,
)

logger = logging.getLogger(__name__)
__version__ = "0.6.0"


# =============================================================================
# HDU Discovery Functions
# =============================================================================

def get_image_hdus(
    fitsfile: str, 
    extname: Optional[str] = None, 
    verbose: bool = False
) -> List[int]:
    """
    Return a list of HDU indices that contain image data.
    
    Correctly includes the primary HDU (index 0) if it contains data.
    Supports compressed FITS files (.fits.gz, .fits.fz, .fz).
    
    Args:
        fitsfile: Path to the FITS file (compressed or uncompressed).
        extname: Filter by extension name (case-insensitive).
        verbose: Enable verbose logging.
    
    Returns:
        List of HDU indices containing image data.
    
    Raises:
        FileHandlingError: If the file is not found or cannot be read.
    
    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> primary = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> ext1 = fits.ImageHDU(data=np.ones((5, 5)), name='SCI')
        >>> hdul = fits.HDUList([primary, ext1])
        >>> hdul.writeto(test_file, overwrite=True)
        >>> hdus = get_image_hdus(test_file)
        >>> len(hdus)
        2
        >>> hdus_filtered = get_image_hdus(test_file, extname='SCI')
        >>> len(hdus_filtered)
        1
        >>> os.unlink(test_file)
    """
    if not os.path.exists(fitsfile):
        raise FileHandlingError(f"File not found: {fitsfile}", filepath=fitsfile)
    
    image_hdu_indices = []
    try:
        # astropy.io.fits automatically handles compressed FITS files
        with fits.open(fitsfile, memmap=True) as hdul:
            for i, hdu in enumerate(hdul):
                # Include CompImageHDU for compressed FITS support
                is_image = isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                has_data = hdu.data is not None
                
                if is_image and has_data:
                    if extname:
                        hdu_extname = hdu.header.get('EXTNAME', '').strip().upper()
                        if hdu_extname == extname.strip().upper():
                            image_hdu_indices.append(i)
                    else:
                        image_hdu_indices.append(i)
        
        if verbose:
            logger.info(f"Found image HDUs {image_hdu_indices} in {os.path.basename(fitsfile)}")
        
        return image_hdu_indices
        
    except Exception as e:
        if verbose:
            logger.error(f"Could not read HDUs from {fitsfile}: {e}", exc_info=True)
        raise FileHandlingError(f"Failed to get image HDUs from {fitsfile}: {e}", filepath=fitsfile) from e


def count_image_extensions(fitsfile: str, verbose: bool = False) -> int:
    """
    Count the number of HDUs with image data in a FITS file.
    
    Args:
        fitsfile: Path to the FITS file.
        verbose: Enable verbose logging.
    
    Returns:
        Number of HDUs containing image data.
    
    Raises:
        FileHandlingError: If the file is not found or cannot be read.
    
    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> primary = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> ext1 = fits.ImageHDU(data=np.ones((5, 5)))
        >>> hdul = fits.HDUList([primary, ext1])
        >>> hdul.writeto(test_file, overwrite=True)
        >>> count_image_extensions(test_file)
        2
        >>> os.unlink(test_file)
    """
    return len(get_image_hdus(fitsfile, verbose=verbose))

def count_hdus(filepath: str, verbose: bool = False) -> int:
    """
    Count the total number of HDUs in a FITS file.
    
    This counts ALL HDUs including the primary HDU.
    For counting only image extensions, use count_image_extensions().
    
    Args:
        filepath: Path to the FITS file.
        verbose: Enable verbose logging.
    
    Returns:
        Total number of HDUs in the file (including primary).
    
    Raises:
        FileHandlingError: If the file is not found or cannot be read.
    
    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> primary = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> ext1 = fits.ImageHDU(data=np.ones((5, 5)))
        >>> hdul = fits.HDUList([primary, ext1])
        >>> hdul.writeto(test_file, overwrite=True)
        >>> count_hdus(test_file)
        2
        >>> os.unlink(test_file)
    """
    if not os.path.exists(filepath):
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            count = len(hdul)
            if verbose:
                logger.info(f"File {filepath} has {count} HDUs")
            return count
    except Exception as e:
        if verbose:
            logger.error(f"Error reading {filepath}: {e}", exc_info=True)
        raise FileHandlingError(
            f"Failed to count HDUs in {filepath}: {e}",
            filepath=filepath
        ) from e


def fits_extension_count(fits_file_path: str, verbose: bool = False) -> int:
    """
    Count the number of extensions in a FITS file.
    
    .. deprecated:: 0.7.0
        Use :func:`count_hdus` instead. This function is provided for
        backward compatibility and will be removed in version 1.0.0.
    
    Note:
        Returns the number of extensions (total HDUs minus 1), not including
        the primary HDU. For total HDU count, use count_hdus().
    
    Args:
        fits_file_path: Path to the FITS file.
        verbose: Enable verbose logging.
    
    Returns:
        Number of extensions (len(HDUList) - 1).
    
    Raises:
        FileHandlingError: If the file is not found or cannot be read.
    """
    import warnings
    warnings.warn(
        "fits_extension_count() is deprecated and will be removed in version 1.0.0. "
        "Use count_hdus() for total HDU count or count_image_extensions() for image HDUs.",
        DeprecationWarning,
        stacklevel=2
    )
    
    total_hdus = count_hdus(fits_file_path, verbose=verbose)
    return total_hdus - 1  # Subtract primary HDU


# =============================================================================
# Header Access Functions
# =============================================================================

def get_header(
    filepath: str, 
    ext: int = 0, 
    verbose: bool = False
) -> fits.Header:
    """
    Return the header from a specified FITS HDU.
    
    Supports compressed FITS files.
    
    Args:
        filepath: Path to the FITS file (compressed or uncompressed).
        ext: The HDU extension number. Defaults to 0.
        verbose: Enable verbose logging.
    
    Returns:
        The FITS header object (a copy that persists after file closes).
    
    Raises:
        FileHandlingError: If the file is not found.
        InvalidHDUIndexError: If the extension index is out of range.
    
    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['TESTKEY'] = 'TESTVAL'
        >>> hdu.writeto(test_file, overwrite=True)
        >>> header = get_header(test_file)
        >>> header['TESTKEY']
        'TESTVAL'
        >>> os.unlink(test_file)
    """
    if not os.path.exists(filepath):
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))
            return hdul[ext].header.copy()  # Return a copy
            
    except InvalidHDUIndexError:
        raise
    except Exception as e:
        if verbose:
            logger.error(f"Error reading header from {filepath}, ext {ext}: {e}", exc_info=True)
        raise FileHandlingError(f"Failed to read header: {e}", filepath=filepath) from e


# =============================================================================
# Region Validation
# =============================================================================

def _validate_region_bounds(
    data_shape: Tuple[int, int], 
    section: Section,
    filepath: str = ""
) -> None:
    """
    Validate that region bounds are within data array dimensions.
    
    Args:
        data_shape: Shape of the data array (rows, cols).
        section: Section object with bounds to validate.
        filepath: File path for error messages.
    
    Raises:
        RegionError: If region bounds are invalid or out of range.
    
    Examples:
        >>> from fits_tools.sections import Section
        >>> _validate_region_bounds((100, 100), Section(0, 50, 0, 50))  # Valid
        
        >>> _validate_region_bounds((100, 100), Section(0, 150, 0, 50))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        fits_tools.exceptions.RegionError: ...
    """
    rows, cols = data_shape
    
    # Check for negative coordinates
    if section.start_col < 0 or section.start_row < 0:
        raise RegionError(
            f"Section has negative coordinates: {section}",
            region=section.to_tuple(),
            image_shape=data_shape,
            operation="validation"
        )
    
    # Check bounds are within array dimensions
    if section.end_col > cols or section.end_row > rows:
        raise RegionError(
            f"Region {section} exceeds data dimensions {data_shape}",
            region=section.to_tuple(),
            image_shape=data_shape,
            operation="validation"
        )
    
    # Check start < end (Section should enforce this, but double-check)
    if section.start_col >= section.end_col or section.start_row >= section.end_row:
        raise RegionError(
            f"Invalid region: start must be less than end. Got {section}",
            region=section.to_tuple(),
            image_shape=data_shape,
            operation="validation"
        )


# =============================================================================
# Data Access Functions
# =============================================================================

def get_data_and_header(
    filepath: str, 
    ext: int = 0, 
    memmap: bool = False, 
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], fits.Header]:
    """
    Return both the data and header from a specified FITS HDU.
    
    Supports compressed FITS files and memory mapping.
    
    Args:
        filepath: Path to the FITS file (compressed or uncompressed).
        ext: The HDU extension number. Defaults to 0.
        memmap: If True, use memory mapping. This allows efficient access
            to large files by loading data on-demand rather than reading
            the entire file into RAM. Defaults to False.
        verbose: Enable verbose logging.
    
    Returns:
        Tuple containing (data, header) where:
            - data: The image data array, memmap array if memmap=True,
              or None if HDU is empty
            - header: The FITS header object
    
    Raises:
        FileHandlingError: If the file is not found.
        InvalidHDUIndexError: If the extension index is out of range.
    
    Note:
        When using memmap=True, the returned data array is a memory-mapped
        array that references the file on disk. This is efficient for large
        files but requires the file to remain accessible.
    
    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> hdu = fits.PrimaryHDU(data=np.ones((5, 5)))
        >>> hdu.header['OBSERVER'] = 'TEST'
        >>> hdu.writeto(test_file, overwrite=True)
        >>> data, header = get_data_and_header(test_file)
        >>> data.shape
        (5, 5)
        >>> header['OBSERVER']
        'TEST'
        >>> os.unlink(test_file)
    """
    if not os.path.exists(filepath):
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)
    
    try:
        # Open with memmap if requested
        with fits.open(filepath, memmap=memmap, mode='readonly') as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))
            
            # Get data and header
            data = hdul[ext].data
            header = hdul[ext].header.copy()
            
            if data is None:
                if verbose:
                    logger.info(f"HDU {ext} in {filepath} contains no image data")
                return None, header
            
            # If memmap is False, ensure we have a copy
            if not memmap:
                data = np.array(data)
            else:
                if verbose:
                    logger.debug(f"Returning memory-mapped array for {filepath} HDU {ext}")
            
            return data, header
            
    except InvalidHDUIndexError:
        raise
    except Exception as e:
        if verbose:
            logger.error(f"Error reading data and header from {filepath}, ext {ext}: {e}", exc_info=True)
        raise FileHandlingError(f"Failed to read data and header: {e}", filepath=filepath) from e


def get_data(
    filepath: str,
    ext: int = 0,
    region: Optional[Union[str, Tuple[int, int, int, int], Section]] = 'full',
    memmap: bool = False,
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Return the data from a specified FITS HDU, with optional section slicing.
    
    Supports compressed FITS files (.fits.gz, .fits.fz, .fz) and memory mapping
    for efficient handling of large files.

    Args:
        filepath: Path to the FITS file (compressed or uncompressed).
        ext: The HDU extension number. Defaults to 0.
        region: The region to extract. Can be:
            - 'full': entire array (default)
            - 'data': DATASEC region from header
            - 'bias': BIASSEC region from header
            - 'trim': TRIMSEC region from header
            - 'dark': DARKSEC region from header
            - tuple (start_col, end_col, start_row, end_row): custom region
              using 0-indexed Python slice coordinates
            - Section: Section object
        memmap: If True, use memory mapping for large files. This allows
            working with files larger than RAM by loading data on-demand.
            Note: memmap arrays are read-only. Defaults to False.
        verbose: Enable verbose logging.

    Returns:
        The requested data array (or memmap array if memmap=True),
        or None if HDU contains no data.
    
    Raises:
        FileHandlingError: If the file is not found.
        InvalidHDUIndexError: If the extension index is out of range.
        HeaderKeywordError: If a region keyword is not found in the header.
        RegionError: If the region specification is invalid or out of bounds.
        TypeError: If the region parameter type is invalid.
    
    Note:
        Memory mapping is most effective for large files where you only need
        to access portions of the data. For small files or when you need the
        entire array, regular loading (memmap=False) may be faster.
    
    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> test_data = np.arange(100).reshape(10, 10)
        >>> hdu = fits.PrimaryHDU(data=test_data)
        >>> hdu.writeto(test_file, overwrite=True)
        >>> 
        >>> # Test full data
        >>> data = get_data(test_file, region='full')
        >>> data.shape
        (10, 10)
        >>> 
        >>> # Test region extraction
        >>> region_data = get_data(test_file, region=(0, 5, 0, 5))
        >>> region_data.shape
        (5, 5)
        >>> 
        >>> os.unlink(test_file)
    """
    try:
        data, header = get_data_and_header(filepath, ext=ext, memmap=memmap, verbose=verbose)
        
        # Return None if HDU contains no data
        if data is None:
            if verbose:
                logger.warning(f"HDU {ext} in {filepath} contains no image data")
            return None

        # Handle 'full' or None region - return entire array
        if region is None or region == RegionType.FULL.value or region == 'full':
            return data

        # Convert region to Section object
        section = None
        
        if isinstance(region, Section):
            section = region
        elif isinstance(region, tuple):
            if len(region) != 4:
                raise TypeError(
                    f"Region tuple must have 4 elements (start_col, end_col, start_row, end_row), "
                    f"got {len(region)}"
                )
            section = Section(*region)
        elif isinstance(region, str):
            # Map region string to FITS keyword
            keyword_map = {
                RegionType.DATA.value: FITS_KEY_DATASEC,
                RegionType.BIAS.value: FITS_KEY_BIASSEC,
                RegionType.TRIM.value: FITS_KEY_TRIMSEC,
                RegionType.DARK.value: FITS_KEY_DARKSEC
            }
            keyword = keyword_map.get(region.lower())
            if keyword is None:
                valid_regions = [r.value for r in RegionType]
                raise TypeError(
                    f"Unsupported region string: '{region}'. "
                    f"Valid options are: {valid_regions}"
                )
            section = get_section_by_keyword(header, keyword)
        else:
            raise TypeError(
                f"Region must be a string, tuple, Section, or None. "
                f"Got {type(region).__name__}"
            )

        # Extract the region if specified
        if section:
            _validate_region_bounds(data.shape, section, filepath)
            row_slice, col_slice = section.to_slices()
            sliced_data = data[row_slice, col_slice]
            
            # If using memmap and slicing, make a copy to ensure data persists
            if memmap and isinstance(data, np.memmap):
                return np.array(sliced_data)
            return sliced_data
        
        return data

    except (InvalidHDUIndexError, HeaderKeywordError, RegionError, TypeError) as e:
        if verbose:
            logger.error(f"Error getting data region '{region}' from {filepath}: {e}", exc_info=True)
        raise
    except Exception as e:
        if verbose:
            logger.error(f"An unexpected error occurred in get_data for {filepath}: {e}", exc_info=True)
        raise FitsToolsError(f"Failed to read data: {e}") from e


def get_all_data(
    filepath: str,
    extname_filter: Optional[str] = None,
    memmap: bool = False,
    return_dict: bool = False,
    verbose: bool = False
) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
    """
    Return data from all image HDUs in a FITS file.
    
    This is particularly useful for multi-extension FITS (MEF) files common
    in CCD imaging, where each extension may represent a different amplifier,
    CCD, or detector segment.
    
    Args:
        filepath: Path to the FITS file.
        extname_filter: If provided, only return HDUs with this EXTNAME.
            Useful for filtering specific types of extensions.
        memmap: If True, use memory mapping for efficient large file handling.
            Defaults to False.
        return_dict: If True, return a dictionary with EXTNAME as keys.
            If False, return a list. Defaults to False.
        verbose: Enable verbose logging.
    
    Returns:
        If return_dict=False: List of numpy arrays, one per image HDU
        If return_dict=True: Dictionary with EXTNAME (or 'HDU_{i}') as 
            keys and numpy arrays as values
    
    Raises:
        FileHandlingError: If the file is not found or cannot be read.
    
    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> primary = fits.PrimaryHDU()
        >>> ext1 = fits.ImageHDU(data=np.ones((10, 10)), name='AMP1')
        >>> ext2 = fits.ImageHDU(data=np.ones((10, 10)) * 2, name='AMP2')
        >>> hdul = fits.HDUList([primary, ext1, ext2])
        >>> hdul.writeto(test_file, overwrite=True)
        >>> 
        >>> # Get all data as list
        >>> data_list = get_all_data(test_file)
        >>> len(data_list)
        2
        >>> 
        >>> # Get all data as dict
        >>> data_dict = get_all_data(test_file, return_dict=True)
        >>> 'AMP1' in data_dict and 'AMP2' in data_dict
        True
        >>> 
        >>> os.unlink(test_file)
    """
    if not os.path.exists(filepath):
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)
    
    try:
        # Get all image HDU indices
        image_hdus = get_image_hdus(filepath, extname=extname_filter, verbose=verbose)
        
        if not image_hdus:
            logger.warning(f"No image HDUs found in {filepath}")
            return {} if return_dict else []
        
        if verbose:
            logger.info(f"Reading {len(image_hdus)} image HDU(s) from {filepath}")
        
        # Read data from each HDU
        with fits.open(filepath, memmap=memmap, mode='readonly') as hdul:
            if return_dict:
                data_dict = {}
                for idx in image_hdus:
                    hdu = hdul[idx]
                    # Get extension name, or create a default
                    extname = hdu.header.get('EXTNAME', f'HDU_{idx}').strip()
                    
                    # Handle duplicate extension names
                    original_extname = extname
                    counter = 1
                    while extname in data_dict:
                        extname = f"{original_extname}_{counter}"
                        counter += 1
                    
                    if hdu.data is not None:
                        if memmap:
                            data_dict[extname] = hdu.data
                        else:
                            data_dict[extname] = np.array(hdu.data)
                    else:
                        if verbose:
                            logger.warning(f"HDU {idx} ({extname}) contains no data")
                
                return data_dict
            else:
                data_list = []
                for idx in image_hdus:
                    hdu = hdul[idx]
                    if hdu.data is not None:
                        if memmap:
                            data_list.append(hdu.data)
                        else:
                            data_list.append(np.array(hdu.data))
                    else:
                        if verbose:
                            extname = hdu.header.get('EXTNAME', f'HDU_{idx}')
                            logger.warning(f"HDU {idx} ({extname}) contains no data")
                
                return data_list
            
    except Exception as e:
        if verbose:
            logger.error(f"Error reading all data from {filepath}: {e}", exc_info=True)
        raise FileHandlingError(f"Failed to read all data: {e}", filepath=filepath) from e


# =============================================================================
# Writing Functions
# =============================================================================

def write_fits(
    filepath: str, 
    data: np.ndarray, 
    header: Optional[fits.Header] = None, 
    verbose: bool = False, 
    overwrite: bool = False,
    compress: bool = False
) -> None:
    """
    Write a FITS file with the given data and header.
    
    Args:
        filepath: Path where the FITS file will be written.
        data: The image data array to write.
        header: The FITS header. If None, a minimal header will be created.
        verbose: Enable verbose logging.
        overwrite: If True, overwrite existing files. If False and file exists,
            skip writing. Defaults to False.
        compress: If True, use compression (Rice for integer, GZIP for float).
            Useful for large files. Defaults to False.
    
    Returns:
        None
    
    Raises:
        FitsToolsError: If an error occurs during writing.
    
    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> data = np.zeros((100, 100))
        >>> write_fits(test_file, data, overwrite=True)
        >>> os.path.exists(test_file)
        True
        >>> os.unlink(test_file)
    """
    try:
        if os.path.exists(filepath) and not overwrite:
            if verbose:
                logger.warning(f"File '{filepath}' exists and overwrite is False. Skipping write.")
            return

        if compress:
            # Use Rice compression for integer data, GZIP for float
            if np.issubdtype(data.dtype, np.integer):
                hdu = fits.CompImageHDU(data=data, header=header, compression_type='RICE_1')
            else:
                hdu = fits.CompImageHDU(data=data, header=header, compression_type='GZIP_1')
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        else:
            hdu = fits.PrimaryHDU(data=data, header=header)
            hdul = fits.HDUList([hdu])
            
        hdul.writeto(filepath, overwrite=overwrite, checksum=True)
        
        if verbose:
            compress_str = " (compressed)" if compress else ""
            logger.info(f"FITS file{compress_str} written to: {filepath}")
            
    except Exception as e:
        if verbose:
            logger.error(f"Error writing FITS file {filepath}: {e}", exc_info=True)
        raise FitsToolsError(f"Failed to write FITS file: {e}") from e


# =============================================================================
# Inspection Functions
# =============================================================================

def inspect_fits(filepath: str, detailed: bool = False) -> None:
    """
    Print information about a FITS file structure.
    
    Args:
        filepath: Path to the FITS file.
        detailed: If True, show detailed header info for each HDU.
    
    Examples:
        >>> # inspect_fits('test.fits')  # doctest: +SKIP
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    try:
        with fits.open(filepath) as hdul:
            print(f"\n{'='*70}")
            print(f"FITS File: {os.path.basename(filepath)}")
            print(f"Path: {filepath}")
            print(f"Size: {os.path.getsize(filepath) / 1024:.2f} KB")
            print(f"{'='*70}\n")
            
            hdul.info()
            print()
            
            if detailed:
                for i, hdu in enumerate(hdul):
                    print(f"\n--- HDU {i} ---")
                    if hdu.data is not None:
                        print(f"Data shape: {hdu.data.shape}")
                        print(f"Data type: {hdu.data.dtype}")
                        print(f"Data range: [{np.min(hdu.data):.3f}, {np.max(hdu.data):.3f}]")
                    else:
                        print("No data")
                    
                    print("\nKey header cards:")
                    for key in ['EXTNAME', 'NAXIS', 'NAXIS1', 'NAXIS2', 'BITPIX', 
                               'OBSERVER', 'DATE-OBS', 'EXPTIME', 'IMAGETYP', 'IMGTYPE']:
                        if key in hdu.header:
                            print(f"  {key}: {hdu.header[key]}")
                            
    except Exception as e:
        print(f"Error reading FITS file: {e}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'RegionType',
    
    # HDU discovery
    'get_image_hdus',
    'count_image_extensions',
    'count_hdus', 
    'fits_extension_count',  
    
    # Reading
    'get_header',
    'get_data',
    'get_data_and_header',
    'get_all_data',
    
    # Writing
    'write_fits',
    
    # Inspection
    'inspect_fits',
]


# =============================================================================
# Main / Testing
# =============================================================================

def main():
    """
    Command-line interface for FITS file utilities.
    
    Provides inspection tools and runs doctests.
    """
    parser = argparse.ArgumentParser(
        description='FITS File I/O Utilities - Inspection and Testing Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --inspect myfile.fits
  %(prog)s --inspect myfile.fits --detailed
  %(prog)s --test
  %(prog)s --test --verbose
        """
    )
    
    parser.add_argument('--inspect', metavar='FITS_FILE', 
                       help='Inspect a FITS file structure')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed information when inspecting')
    parser.add_argument('--test', action='store_true',
                       help='Run doctests')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--version', action='version', 
                       version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
# Handle inspect mode
    if args.inspect:
        inspect_fits(args.inspect, detailed=args.detailed)
        return 0
    
    # Handle test mode
    if args.test:
        logger.info("Running doctests for io.py...")
        result = doctest.testmod(verbose=args.verbose)
        
        if result.failed == 0:
            logger.info(f"All {result.attempted} doctests passed!")
            return 0
        else:
            logger.error(f"{result.failed} of {result.attempted} doctests failed!")
            return 1
    
    # No arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
