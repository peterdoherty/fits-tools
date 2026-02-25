#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FITS I/O utilities for fits_tools.

This module provides higher-level utilities for working with FITS files:

- Discovering HDUs with image data
- Reading data and headers (with optional memmap)
- Extracting standard regions (DATASEC, BIASSEC, TRIMSEC, DARKSEC)
- Reading data from all image HDUs in MEF files
- Writing simple FITS files (optionally compressed)
- Lightweight inspection of FITS structure

These functions sit on top of :mod:`astropy.io.fits` and the fits_tools
exception / section / constants infrastructure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from .constants import (
    FITS_KEY_BIASSEC,
    FITS_KEY_DATASEC,
    FITS_KEY_DARKSEC,
    FITS_KEY_TRIMSEC,
    RegionType,
)
from .exceptions import (
    FileHandlingError,
    HeaderKeywordError,
    InvalidHDUIndexError,
    FitsToolsError,
    RegionError,
)
from .sections import Section, get_section_by_keyword
from .types import StrOrPath

logger = logging.getLogger(__name__)
__version__ = "0.6.0"


# =============================================================================
# HDU Discovery Functions
# =============================================================================

def get_image_hdus(
    input_file: StrOrPath,
    extname: Optional[str] = None,
) -> List[int]:
    """Return a list of HDU indices that contain image data.

    Includes the primary HDU (index 0) if it contains data. Supports
    compressed FITS files (.fits.gz, .fits.fz, .fz).

    Parameters
    ----------
    input_file :
        Path to the FITS file (compressed or uncompressed).
    extname :
        Optional extension name to filter on (case-insensitive). When
        provided, only HDUs whose ``EXTNAME`` matches are returned.

    Returns
    -------
    list[int]
        HDU indices containing image data.

    Raises
    ------
    FileHandlingError
        If the file is not found or cannot be read.
    """

    input_file = Path(input_file)
    if not input_file.exists():
        raise FileHandlingError(f"File not found: {input_file}", filepath=input_file)

    image_hdu_indices: List[int] = []
    try:
        with fits.open(input_file, memmap=True) as hdul:
            for i, hdu in enumerate(hdul):
                is_image = isinstance(
                    hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)
                )
                has_data = hdu.data is not None

                if not (is_image and has_data):
                    continue

                if extname:
                    hdu_extname = hdu.header.get("EXTNAME", "").strip().upper()
                    if hdu_extname == extname.strip().upper():
                        image_hdu_indices.append(i)
                else:
                    image_hdu_indices.append(i)

        logger.debug(
            "Found image HDUs %s in %s",
            image_hdu_indices,
            Path(input_file).name,
        )

        return image_hdu_indices

    except Exception as e:  # pragma: no cover - exercised via tests
        logger.error("Could not read HDUs from %s: %s", input_file, e, exc_info=True)
        raise FileHandlingError(
            f"Failed to get image HDUs from {input_file}: {e}", filepath=input_file.name) from e


def count_image_extensions(input_file: StrOrPath) -> int:
    """Count the number of HDUs with image data in a FITS file.

    This is a thin wrapper around :func:`get_image_hdus`.

    Parameters
    ----------
    input_file :
        Path to the FITS file.

    Returns
    -------
    int
        Number of HDUs containing image data.

    Raises
    ------
    FileHandlingError
        If the file is not found or cannot be read.
    """
    return len(get_image_hdus(input_file))


def count_hdus(filepath: StrOrPath) -> int:
    """Count the total number of HDUs in a FITS file.

    This counts *all* HDUs including the primary HDU. To count only image
    extensions, use :func:`count_image_extensions`.

    Parameters
    ----------
    filepath :
        Path to the FITS file.

    Returns
    -------
    int
        Total number of HDUs in the file (including primary).

    Raises
    ------
    FileHandlingError
        If the file is not found or cannot be read.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=True) as hdul:
            count = len(hdul)
            logger.debug("File %s has %d HDUs", filepath, count)
            return count
    except Exception as e:  # pragma: no cover - error path
        logger.error("Error reading %s: %s", filepath, e, exc_info=True)
        raise FileHandlingError(
            f"Failed to count HDUs in {filepath}: {e}", filepath=filepath
        ) from e


def fits_extension_count(fits_file_path: StrOrPath) -> int:
    """Count the number of extensions in a FITS file.

    .. deprecated:: 0.7.0
       Use :func:`count_hdus` instead. This function will be removed in 1.0.0.

    Returns the number of extensions (total HDUs minus 1), not including the
    primary HDU.

    Parameters
    ----------
    fits_file_path :
        Path to the FITS file.

    Returns
    -------
    int
        Number of extensions (``len(HDUList) - 1``).

    Raises
    ------
    FileHandlingError
        If the file is not found or cannot be read.
    """
    import warnings

    warnings.warn(
        "fits_extension_count() is deprecated and will be removed in version 1.0.0. "
        "Use count_hdus() for total HDU count or count_image_extensions() for image HDUs.",
        DeprecationWarning,
        stacklevel=2,
    )

    total_hdus = count_hdus(fits_file_path)
    return total_hdus - 1


# =============================================================================
# Header Access
# =============================================================================

def get_header(
    filepath: StrOrPath,
    ext: int = 0,
) -> fits.Header:
    """Return the header from a specified FITS HDU.

    Parameters
    ----------
    filepath :
        Path to the FITS file (compressed or uncompressed).
    ext :
        HDU extension index (0-based). Default is 0 (primary HDU).

    Returns
    -------
    astropy.io.fits.Header
        A copy of the header for the requested HDU.

    Raises
    ------
    FileHandlingError
        If the file is not found or cannot be opened.
    InvalidHDUIndexError
        If *ext* is out of range for the file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=True) as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))
            return hdul[ext].header.copy()
    except InvalidHDUIndexError:
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Error reading header from %s, ext %d: %s", filepath, ext, e, exc_info=True
        )
        raise FileHandlingError(
            f"Failed to read header: {e}", filepath=filepath
        ) from e


# =============================================================================
# Region Validation
# =============================================================================

def _validate_region_bounds(
    data_shape: Tuple[int, int],
    section: Section,
    filepath: StrOrPath = "",
) -> None:
    """Validate that region bounds are within data array dimensions.

    Parameters
    ----------
    data_shape :
        Shape of the data array as ``(rows, cols)``.
    section :
        :class:`Section` instance describing the region to validate.
    filepath :
        File path for error message context (optional).

    Raises
    ------
    RegionError
        If region bounds are invalid or out of range.
    """
    rows, cols = data_shape

    if section.start_col < 0 or section.start_row < 0:
        raise RegionError(
            f"Section has negative coordinates: {section}",
            region=section.to_tuple(),
            image_shape=data_shape,
            operation="validation",
        )

    if section.end_col > cols or section.end_row > rows:
        raise RegionError(
            f"Region {section} exceeds data dimensions {data_shape}",
            region=section.to_tuple(),
            image_shape=data_shape,
            operation="validation",
        )

    if section.start_col >= section.end_col or section.start_row >= section.end_row:
        raise RegionError(
            f"Invalid region: start must be less than end. Got {section}",
            region=section.to_tuple(),
            image_shape=data_shape,
            operation="validation",
        )


# =============================================================================
# Data Access
# =============================================================================

def get_data_and_header(
    filepath: StrOrPath,
    ext: int = 0,
    memmap: bool = False,
) -> Tuple[Optional[np.ndarray], fits.Header]:
    """Return both the data and header from a specified FITS HDU.

    Parameters
    ----------
    filepath :
        Path to the FITS file (compressed or uncompressed).
    ext :
        HDU extension index. Default is 0.
    memmap :
        If True, use memory-mapped arrays. The returned array will then
        reference the file on disk and is read-only. Default is False.

    Returns
    -------
    (data, header) :
        * data : ndarray or memmap or None (if no data in HDU)
        * header : fits.Header copy

    Raises
    ------
    FileHandlingError
        If the file is not found or cannot be opened.
    InvalidHDUIndexError
        If *ext* is out of range.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=memmap, mode="readonly") as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))

            hdu = hdul[ext]
            data = hdu.data
            header = hdu.header.copy()

            if data is None:
                logger.debug("HDU %d in %s contains no image data", ext, filepath)
                return None, header

            if not memmap:
                data = np.array(data)
            else:
                logger.debug("Returning memmap array for %s HDU %d", filepath, ext)

            return data, header

    except InvalidHDUIndexError:
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Error reading data and header from %s, ext %d: %s",
            filepath,
            ext,
            e,
            exc_info=True,
        )
        raise FileHandlingError(
            f"Failed to read data and header: {e}", filepath=filepath
        ) from e


def get_data(
    filepath: StrOrPath,
    ext: int = 0,
    region: Optional[Union[str, Tuple[int, int, int, int], Section]] = None,
    memmap: bool = False,
) -> Optional[np.ndarray]:
    """Return data from a FITS HDU, with optional region extraction.

    Parameters
    ----------
    filepath :
        Path to the FITS file.
    ext :
        HDU extension index. Default is 0.
    region :
        Region to extract. May be:

        - ``'full'`` or :attr:`RegionType.FULL.value`: full array
        - ``'data'``, ``'bias'``, ``'trim'``, ``'dark'``: use corresponding
          FITS section keywords (DATASEC, BIASSEC, TRIMSEC, DARKSEC)
        - ``(start_col, end_col, start_row, end_row)``: 0-based Python
          slice coordinates
        - :class:`Section` instance
        - ``None``: equivalent to ``'full'``

    memmap :
        If True, return a memory-mapped array where possible. If a region
        slice is taken from a memmap, a regular NumPy copy is returned so
        the data remains valid after the file is closed.

    Returns
    -------
    ndarray or None
        The requested data array, or ``None`` if the HDU contains no data.

    Raises
    ------
    FileHandlingError
    InvalidHDUIndexError
    HeaderKeywordError
    RegionError
    TypeError
    """
    try:
        data, header = get_data_and_header(filepath, ext=ext, memmap=memmap)

        if data is None:
            logger.warning("HDU %d in %s contains no image data", ext, filepath)
            return None

        # Full array
        if region is None or region in (RegionType.FULL.value, "full"):
            return data

        # Normalise to Section
        if isinstance(region, Section):
            section = region
        elif isinstance(region, tuple):
            if len(region) != 4:
                raise TypeError(
                    "Region tuple must be (start_col, end_col, start_row, end_row); "
                    f"got {len(region)} element(s)"
                )
            section = Section(*region)
        elif isinstance(region, str):
            keyword_map = {
                RegionType.DATA.value: FITS_KEY_DATASEC,
                RegionType.BIAS.value: FITS_KEY_BIASSEC,
                RegionType.TRIM.value: FITS_KEY_TRIMSEC,
                RegionType.DARK.value: FITS_KEY_DARKSEC,
            }
            key = keyword_map.get(region.lower())
            if key is None:
                valid_regions = [r.value for r in RegionType]
                raise TypeError(
                    f"Unsupported region string: {region!r}. "
                    f"Valid options are: {valid_regions}"
                )
            section = get_section_by_keyword(header, key)
        else:
            raise TypeError(
                "region must be a string, tuple, Section, or None; "
                f"got {type(region).__name__}"
            )

        _validate_region_bounds(data.shape, section, filepath)
        row_slice, col_slice = section.to_slices()
        sliced = data[row_slice, col_slice]

        if memmap and isinstance(data, np.memmap):
            return np.array(sliced)

        return sliced

    except (InvalidHDUIndexError, HeaderKeywordError, RegionError, TypeError):
        logger.error(
            "Error getting data region %r from %s", region, filepath, exc_info=True
        )
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Unexpected error in get_data for %s: %s", filepath, e, exc_info=True
        )
        raise FitsToolsError(f"Failed to read data: {e}") from e


def get_all_data(
    filepath: StrOrPath,
    extname_filter: Optional[str] = None,
    memmap: bool = False,
    return_dict: bool = False,
) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
    """Return data arrays from all image HDUs in a FITS file.

    Parameters
    ----------
    filepath :
        Path to the FITS file.
    extname_filter :
        If provided, only HDUs whose ``EXTNAME`` matches (case-insensitive)
        are included.
    memmap :
        If True, return memory-mapped arrays where possible.
    return_dict :
        If True, return a dict mapping EXTNAME (or ``'HDU_{i}'``) to arrays.
        If False, return a list of arrays.

    Returns
    -------
    list[ndarray] or dict[str, ndarray]
        Image data from each selected HDU. Empty list/dict if no image HDUs
        are found.

    Raises
    ------
    FileHandlingError
        If the file is not found or cannot be read.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        image_hdus = get_image_hdus(filepath, extname=extname_filter)

        if not image_hdus:
            logger.warning("No image HDUs found in %s", filepath)
            return {} if return_dict else []

        logger.debug("Reading %d image HDU(s) from %s", len(image_hdus), filepath)

        with fits.open(filepath, memmap=memmap, mode="readonly") as hdul:
            if return_dict:
                data_dict: Dict[str, np.ndarray] = {}
                for idx in image_hdus:
                    hdu = hdul[idx]
                    extname = hdu.header.get("EXTNAME", f"HDU_{idx}").strip()

                    # Avoid overwriting keys for duplicate EXTNAMEs
                    base = extname
                    counter = 1
                    while extname in data_dict:
                        extname = f"{base}_{counter}"
                        counter += 1

                    if hdu.data is not None:
                        data_dict[extname] = hdu.data if memmap else np.array(hdu.data)
                    else:
                        logger.warning(
                            "HDU %d (%s) contains no data", idx, extname or f"HDU_{idx}"
                        )
                return data_dict

            else:
                data_list: List[np.ndarray] = []
                for idx in image_hdus:
                    hdu = hdul[idx]
                    if hdu.data is not None:
                        data_list.append(hdu.data if memmap else np.array(hdu.data))
                    else:
                        extname = hdu.header.get("EXTNAME", f"HDU_{idx}")
                        logger.warning(
                            "HDU %d (%s) contains no data", idx, extname or f"HDU_{idx}"
                        )
                return data_list

    except Exception as e:  # pragma: no cover - error path
        logger.error("Error reading all data from %s: %s", filepath, e, exc_info=True)
        raise FileHandlingError(
            f"Failed to read all data: {e}", filepath=filepath
        ) from e


# =============================================================================
# Writing
# =============================================================================

def write_fits(
    filepath: StrOrPath,
    data: np.ndarray,
    header: Optional[fits.Header] = None,
    overwrite: bool = False,
    compress: bool = False,
) -> None:
    """Write a simple FITS file with the given data and header.

    This is a convenience wrapper around :mod:`astropy.io.fits` writing
    routines. For CLI tools that must avoid any risk of file corruption,
    prefer :func:`fits_tools.utils.safe_write`.

    Parameters
    ----------
    filepath :
        Destination path.
    data :
        Image data array to write.
    header :
        Optional FITS header for the primary HDU.
    overwrite :
        Whether to overwrite an existing file. Default is False.
    compress :
        If True, write a compressed FITS file: Rice compression for integer
        data, GZIP for floating-point data.
    """
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileHandlingError(
            f"File '{filepath}' exists and overwrite=False.",
            filepath=filepath,
        )
    try:
        if compress:
            if np.issubdtype(data.dtype, np.integer):
                hdu = fits.CompImageHDU(
                    data=data, header=header, compression_type="RICE_1"
                )
            else:
                hdu = fits.CompImageHDU(
                    data=data, header=header, compression_type="GZIP_1"
                )
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        else:
            hdu = fits.PrimaryHDU(data=data, header=header)
            hdul = fits.HDUList([hdu])

        hdul.writeto(filepath, overwrite=overwrite, checksum=True)

        compress_str = " (compressed)" if compress else ""
        logger.debug("FITS file%s written to: %s", compress_str, filepath)

    except Exception as e:  # pragma: no cover - error path
        logger.error("Error writing FITS file %s: %s", filepath, e, exc_info=True)
        raise FitsToolsError(f"Failed to write FITS file: {e}") from e


# =============================================================================
# Inspection
# =============================================================================

def inspect_fits(filepath: StrOrPath, 
                 detailed: bool = False) -> None:
    """Print basic structural information about a FITS file.

    This is a lightweight helper intended for interactive use or debugging.
    For programmatic inspection, prefer :mod:`astropy.io.fits` directly.

    Parameters
    ----------
    filepath :
        Path to the FITS file.
    detailed :
        If True, print per-HDU data ranges and selected header cards.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return

    try:
        with fits.open(filepath) as hdul:
            print(f"\n{'=' * 70}")
            print(f"FITS File: {filepath.name}")
            print(f"Path: {filepath}")
            print(f"Size: {filepath.stat().st_size / 1024:.2f} KB")
            print(f"{'=' * 70}\n")

            hdul.info()
            print()

            if detailed:
                for i, hdu in enumerate(hdul):
                    print(f"\n--- HDU {i} ---")
                    if hdu.data is not None:
                        print(f"Data shape: {hdu.data.shape}")
                        print(f"Data type:  {hdu.data.dtype}")
                        print(
                            "Data range: [{:.3f}, {:.3f}]".format(
                                float(np.min(hdu.data)), float(np.max(hdu.data))
                            )
                        )
                    else:
                        print("No data")

                    print("\nKey header cards:")
                    for key in [
                        "EXTNAME",
                        "NAXIS",
                        "NAXIS1",
                        "NAXIS2",
                        "BITPIX",
                        "OBSERVER",
                        "DATE-OBS",
                        "EXPTIME",
                        "IMAGETYP",
                        "IMGTYPE",
                    ]:
                        if key in hdu.header:
                            print(f"  {key}: {hdu.header[key]}")

    except Exception as e:  # pragma: no cover - diagnostic helper
        print(f"Error reading FITS file: {e}")


__all__ = [
    "RegionType",
    "get_image_hdus",
    "count_image_extensions",
    "count_hdus",
    "fits_extension_count",
    "get_header",
    "get_data",
    "get_data_and_header",
    "get_all_data",
    "write_fits",
    "inspect_fits",
]