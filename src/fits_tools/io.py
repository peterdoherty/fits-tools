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
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import (
    Any, Dict, List, Iterable, Iterator, Optional, Set, Sequence, Tuple, Union
)

import numpy as np
from astropy.io import fits

from .exceptions import (
    FileOverwriteError,
    FileHandlingError,
)
from .constants import FITS_FILE_EXTENSIONS 
from .types import StrOrPath

# Optional dependencies
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None



import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.base import ExtensionHDU

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

from .hdu import resolve_hdu_index, resolve_hdu_index_in_file, HDUSelector

# =============================================================================
# FITS Utilities
# =============================================================================

def add_history(header, msg: str) -> None:
    """Add HISTORY card to FITS header with fallback."""
    try:
        header.add_history(msg)
    except Exception:
        header["HISTORY"] = msg

def add_comment(header, msg: str) -> None:
    """Add COMMENT card to FITS header with fallback."""
    try:
        header.add_comment(msg)
    except Exception:
        header["COMMENT"] = msg


def get_hdu_count(filepath: StrOrPath) -> int:
    """
    Get the number of HDUs in a FITS file without loading data.
    
    Args:
        filepath: Path to FITS file.
    
    Returns:
        Number of HDUs.
    
    Examples:
        >>> # count = get_hdu_count("image.fits")
        >>> # print(f"File has {count} HDUs")
    """
    with fits.open(filepath, memmap=True) as hdul:
        return len(hdul)


def get_image_shape(
    filepath: StrOrPath,
    hdu_index: int = 0
) -> Optional[Tuple[int, ...]]:
    """
    Get the shape of an image HDU without loading the data.
    
    Args:
        filepath: Path to FITS file.
        hdu_index: HDU index to check (default 0).
    
    Returns:
        Shape tuple or None if HDU has no data.
    
    Examples:
        >>> # shape = get_image_shape("image.fits")
        >>> # print(f"Image shape: {shape}")
    """
    with fits.open(filepath, memmap=True) as hdul:
        if hdul[hdu_index].data is None:
            return None
        return hdul[hdu_index].data.shape


def copy_header(source: fits.Header, exclude_keys: Optional[List[str]] = None) -> fits.Header:
    """
    Create a copy of a FITS header, optionally excluding specific keywords.
    
    Args:
        source: Source header to copy.
        exclude_keys: Keywords to exclude from copy.
    
    Returns:
        Copied header.
    
    Examples:
        >>> from astropy.io import fits
        >>> hdr = fits.Header()
        >>> hdr['EXPTIME'] = 10.0
        >>> hdr['OBJECT'] = 'M31'
        >>> new_hdr = copy_header(hdr, exclude_keys=['OBJECT'])
        >>> 'EXPTIME' in new_hdr
        True
        >>> 'OBJECT' in new_hdr
        False
    """
    new_header = source.copy()
    if exclude_keys:
        for key in exclude_keys:
            new_header.remove(key, ignore_missing=True)
    return new_header


def safe_write(hdul: fits.HDUList, 
               path: StrOrPath,
               overwrite: bool, 
               dry_run: bool, 
               operation: str = "edit") -> None:
    """
    Safely write an HDUList to disk with atomic replace and HISTORY update.

    This is a conservative writer intended for CLI-style tools where failures
    must never corrupt an existing file:

    - If ``dry_run`` is True, nothing is written; a log message is emitted.
    - If ``overwrite`` is False and ``path`` exists, a :class:`FileOverwriteError`
      is raised.
    - A HISTORY line documenting the operation and UTC timestamp is added to
      the primary header when possible.
    - The file is first written to a temporary file in the destination
      directory, then atomically moved into place using :func:`os.replace`.
      If writing fails, the temporary file is removed and the original target
      file is left untouched.

    Parameters
    ----------
    hdul : fits.HDUList
        HDU list to write.
    path : str
        Destination file path.
    overwrite : bool
        Whether to allow overwriting an existing file.
    dry_run : bool
        If True, do not actually write anything to disk.
    operation : str, optional
        Short operation name recorded in the HISTORY card, e.g. ``'crop'``,
        ``'mask'``. Defaults to ``'edit'``.

    Raises
    ------
    FileOverwriteError
        If the destination exists and ``overwrite`` is False.
    FileHandlingError
        If writing the file (or renaming the temporary file) fails.
    """
    if dry_run:
        logger.info(f"[dry-run] Would write: {path}")
        return

    target_path = Path(path)

    if target_path.exists() and not overwrite:
        raise FileOverwriteError(path)

    # Add HISTORY card to document the operation
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    history_msg = f"fits_tools: {operation} at {timestamp} UTC"

    try:
        hdr = hdul[0].header  # will raise if hdul empty, etc.
        add_history(hdr, history_msg)
    except Exception as e:
        logger.debug(f"Could not add HISTORY card: {e}")

    # Atomic write: write to a temp file in the destination directory, then replace.
    dest_dir = target_path.resolve().parent
    base = target_path.name
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{base}.tmp.",
            dir=dest_dir,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        # Always overwrite temp path
        hdul.writeto(tmp_path, overwrite=True, checksum=True)
        tmp_path.replace(target_path)
    except Exception as e:
        # Clean up temp file if present; do NOT delete existing destination.
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise FileHandlingError(f"Failed to write {path}: {e}", filepath=path) from e
    

# =============================================================================
# HDU Discovery Functions
# =============================================================================
_ExtNameFilter = Optional[Union[str, Iterable[str]]]

def _normalize_extname_filter(extname: _ExtNameFilter) -> Optional[Set[str]]:
    if extname is None:
        return None

    if isinstance(extname, str):
        raw = [extname]
    else:
        raw = list(extname)

    normalized = {str(n).strip().upper() for n in raw if str(n).strip()}
    return normalized or None


def _unique_key(existing: Dict[str, Any], desired: str) -> str:
    """Make `desired` unique within `existing` by appending _{n} if needed."""
    if desired not in existing:
        return desired
    counter = 1
    while True:
        candidate = f"{desired}_{counter}"
        if candidate not in existing:
            return candidate
        counter += 1


def iter_image_hdus(
    hdul: fits.HDUList,
    extname: _ExtNameFilter = None,
) -> Iterator[tuple[int, ExtensionHDU]]:
    """
    Internal iterator over (index, hdu) for HDUs that contain image data.

    - Works on an already-open HDUList (so no extra file open/close).
    - If extname is provided, filters by EXTNAME (case-insensitive).
      extname may be a string or an iterable of strings.
    """
    allowed = _normalize_extname_filter(extname)

    for i, hdu in enumerate(hdul):
        is_image = isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
        if not is_image:
            continue

        if hdu.data is None:
            continue

        if allowed is not None:
            hdu_extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            if hdu_extname not in allowed:
                continue

        yield i, hdu


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

    try:
        with fits.open(input_file, memmap=True, mode="readonly") as hdul:
            indices = [i for i, _ in iter_image_hdus(hdul, extname=extname)]
        logger.debug("Found image HDUs %s in %s", indices, input_file.name)
        return indices
    except Exception as e:  # pragma: no cover
        logger.error("Could not read HDUs from %s: %s", input_file, e, exc_info=True)
        raise FileHandlingError(f"Failed to get image HDUs from {input_file}: {e}", filepath=input_file) from e


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
    hdu: int = 0,
) -> fits.Header:
    """Return the header from a specified FITS HDU.

    Parameters
    ----------
    filepath :
        Path to the FITS file (compressed or uncompressed).
    hdu :
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
        If *hdu* is out of range for the file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=True) as hdul:
            if hdu < 0 or hdu >= len(hdul):
                raise InvalidHDUIndexError(filepath, hdu, len(hdul))
            return hdul[hdu].header.copy()
    except InvalidHDUIndexError:
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Error reading header from %s, hdu %d: %s", filepath, hdu, e, exc_info=True
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



def region_to_slices(
    data_shape: Tuple[int, int],
    header: fits.Header,
    region: Optional[Union[str, Tuple[int, int, int, int], Section]] = None,
    filepath: StrOrPath = "",
) -> Tuple[slice, slice]:
    """
    Resolve a region specification into (row_slice, col_slice).

    Region may be:
      - None/'full' -> full image
      - strings like 'data','bias','trim','dark' -> via section keywords
      - tuple (start_col, end_col, start_row, end_row) in 0-based Python coords
      - Section instance

    Returns
    -------
    (row_slice, col_slice)
        Slices suitable for indexing a 2D image array as data[row_slice, col_slice].
    """
    if len(data_shape) != 2:
        raise RegionError(
            f"region_to_slices expects a 2D image shape (rows, cols), got {data_shape}",
            region=None,
            image_shape=data_shape,
            operation="validation",
        )
    
    if region is None or region in (RegionType.FULL.value, "full"):
        return slice(None), slice(None)

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
                f"Unsupported region string: {region!r}. Valid options are: {valid_regions}"
            )
        section = get_section_by_keyword(header, key)
    else:
        raise TypeError(
            "region must be a string, tuple, Section, or None; "
            f"got {type(region).__name__}"
        )

    _validate_region_bounds(data_shape, section, filepath)
    return section.to_slices()

# =============================================================================
# Data Access
# =============================================================================

def get_data_and_header(
    filepath: StrOrPath,
    hdu: int = 0,
    memmap: bool = False,
) -> Tuple[Optional[np.ndarray], fits.Header]:
    """Return both the data and header from a specified FITS HDU.

    Parameters
    ----------
    filepath :
        Path to the FITS file (compressed or uncompressed).
    hdu :
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
        If *hdu* is out of range.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=memmap, mode="readonly") as hdul:
            if hdu < 0 or hdu >= len(hdul):
                raise InvalidHDUIndexError(filepath, hdu, len(hdul))

            hdu = hdul[hdu]
            data = hdu.data
            header = hdu.header.copy()

            if data is None:
                logger.debug("HDU %d in %s contains no image data", hdu, filepath)
                return None, header

            if not memmap:
                data = np.array(data)
            else:
                logger.debug("Returning memmap array for %s HDU %d", filepath, hdu)

            return data, header

    except InvalidHDUIndexError:
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Error reading data and header from %s, hdu %d: %s",
            filepath,
            hdu,
            e,
            exc_info=True,
        )
        raise FileHandlingError(
            f"Failed to read data and header: {e}", filepath=filepath
        ) from e
    

@contextmanager
def open_all_data_memmap(
    filepath: StrOrPath,
    extname_filter: Optional[Union[str, Sequence[str]]] = None,
    return_dict: bool = False,
) -> Iterator[Union[List[np.ndarray], Dict[str, np.ndarray]]]:
    """
    Context manager yielding memmapped image arrays.

    Arrays are valid only while the context is open.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    with fits.open(filepath, memmap=True, mode="readonly") as hdul:
        pairs = list(iter_image_hdus(hdul, extname=extname_filter))

        if return_dict:
            out: Dict[str, np.ndarray] = {}
            for idx, hdu in pairs:
                key = str(hdu.header.get("EXTNAME", "")).strip() or f"HDU_{idx}"
                key = _unique_key(out, key)
                out[key] = hdu.data  # memmap-backed
            yield out
        else:
            yield [hdu.data for _, hdu in pairs]  # memmap-backed


def get_data(
    filepath: StrOrPath,
    hdu: int = 0,
    region: Optional[Union[str, Tuple[int, int, int, int], Section]] = None,
    memmap: bool = False,
) -> Optional[np.ndarray]:
    """Return data from a FITS HDU, with optional region extraction.

    Parameters
    ----------
    filepath :
        Path to the FITS file.
    hdu :
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
        data, header = get_data_and_header(filepath, hdu=hdu, memmap=memmap)

        if data is None:
            logger.warning("HDU %d in %s contains no image data", hdu, filepath)
            return None

        # Full array
        if region is None or region in (RegionType.FULL.value, "full"):
            return data

        row_slice, col_slice = region_to_slices(data.shape, header, region, filepath)
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
        with fits.open(filepath, memmap=False, mode="readonly") as hdul:
            pairs = list(iter_image_hdus(hdul, extname=extname_filter))

            if not pairs:
                logger.warning("No image HDUs found in %s", filepath)
                return {} if return_dict else []

            logger.debug("Reading %d image HDU(s) from %s", len(pairs), filepath)

            if return_dict:
                out: Dict[str, np.ndarray] = {}
                for idx, hdu in pairs:
                    key = str(hdu.header.get("EXTNAME", "")).strip() or f"HDU_{idx}"
                    key = _unique_key(out, key)
                    out[key] = np.array(hdu.data)
                return out

            return [np.array(hdu.data) for _, hdu in pairs]

    except FileHandlingError:
        raise
    except Exception as e:  # pragma: no cover
        logger.error("Error reading all data from %s: %s", filepath, e, exc_info=True)
        raise FileHandlingError(f"Failed to read all data: {e}", filepath=filepath) from e


# =============================================================================
# Writing
# =============================================================================

def write_hdulist(filepath: StrOrPath, hdul: fits.HDUList, overwrite: bool = False) -> None:
    """
    Write an Astropy HDUList to disk (MEF-preserving).

    This function writes the entire FITS structure represented by ``hdul``.
    It is intended for multi-extension FITS (MEF) workflows where preserving
    all HDUs and their headers is required.

    Parameters
    ----------
    filepath
        Destination path for the output FITS file.
    hdul
        The `astropy.io.fits.HDUList` to write.
    overwrite
        If False (default), raise an error if ``filepath`` already exists.
        If True, overwrite any existing file.

    Raises
    ------
    FileHandlingError
        If the output file exists and ``overwrite=False``, the output directory
        cannot be created, or the FITS file cannot be written.

    Notes
    -----
    - Parent directories are created automatically if needed.
    - FITS checksums are enabled (``checksum=True``).
    """
    filepath = Path(filepath)

    if filepath.exists() and not overwrite:
        raise FileHandlingError(
            f"File '{filepath}' exists and overwrite=False.",
            filepath=filepath,
        )

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(filepath, overwrite=overwrite, checksum=True)
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to write FITS HDUList: {e}",
            filepath=filepath,
        ) from e
    

def write_simple_fits(
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

def write_fits(*args, **kwargs):
    import warnings
    warnings.warn(
        "write_fits() is deprecated; use write_simple_fits() for single-HDU output "
        "or write_hdulist() for MEF preservation.",
        DeprecationWarning,
        stacklevel=2,
    )
    return write_simple_fits(*args, **kwargs)

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
    "region_to_slices",
    "get_image_hdus",
    "count_image_extensions",
    "count_hdus",
    "fits_extension_count",
    "iter_image_hdus",
    "get_header",
    "get_data",
    "get_data_and_header",
    "get_all_data",
    "write_hdulist",
    "write_simple_fits",
    "write_fits",
    "inspect_fits",
]