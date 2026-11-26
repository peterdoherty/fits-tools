#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""FITS I/O utilities for fits_tools.

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
import os
import tempfile
import warnings
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from astropy.io import fits

from .constants import (
    FITS_KEY_BIASSEC,
    FITS_KEY_DARKSEC,
    FITS_KEY_DATASEC,
    FITS_KEY_TRIMSEC,
)
from .exceptions import (
    FileHandlingError,
    FileOverwriteError,
    FitsToolsError,
    HeaderKeywordError,
    RegionError,
)
from .hdu import HDUSelector, resolve_hdu_index
from .header_utils import add_history
from .sections import Section, get_section_by_keyword
from .types import RegionType, StrOrPath

logger = logging.getLogger(__name__)

__all__ = [
    # Region utilities
    "RegionType",
    "region_to_slices",
    # HDU discovery
    "iter_image_hdus",
    "get_image_hdus",
    "count_image_extensions",
    "count_hdus",
    "fits_extension_count",   # deprecated
    # Header and shape access
    "get_header",
    "get_header_value",
    "get_image_shape",
    # Data access
    "get_data",
    "get_data_and_header",
    "get_all_data",
    "open_all_data_memmap",
    # Writing
    "write_hdulist",
    "transactional_write_hdulist",
    "write_simple_fits",
    "write_fits",             # deprecated
    "safe_write",
    # Inspection
    "HDUInfo",
    "FitsInfo",
    "inspect_fits",
]


# =============================================================================
# Internal type aliases
# =============================================================================

_ExtNameFilter: TypeAlias = str | Iterable[str] | None
_ImageHDU: TypeAlias = fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU


# =============================================================================
# FITS Utilities
# =============================================================================


def get_hdu_count(filepath: StrOrPath) -> int:
    """Return the total number of HDUs in a FITS file.

    .. deprecated::
        Use :func:`count_hdus` instead, which adds existence checking
        and proper error handling.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.

    Returns
    -------
    int
        Total number of HDUs.
    """
    warnings.warn(
        "get_hdu_count() is deprecated. Use count_hdus() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return count_hdus(filepath)


def get_image_shape(
    filepath: StrOrPath, hdu: HDUSelector = 0
) -> tuple[int, ...] | None:
    """Return the shape of an image HDU without loading the data.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0.

    Returns
    -------
    tuple of int or None
        Shape of the image array, or None if the HDU contains no data.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *hdu* is out of range.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=False, mode="readonly") as hdul:
            index = resolve_hdu_index(hdul, hdu, filepath=filepath)
            data = hdul[index].data
            return None if data is None else tuple(data.shape)
    except FitsToolsError:
        raise
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to read image shape: {e}", filepath=filepath
        ) from e


def safe_write(
    hdul: fits.HDUList,
    path: StrOrPath,
    overwrite: bool,
    dry_run: bool,
    operation: str = "edit",
) -> None:
    """Write an HDUList to disk atomically with HISTORY update.

    Writes to a temporary file in the same directory as *path*, then
    atomically replaces the target via :func:`os.replace`. This ensures
    that the target is never left in a partially-written state.

    Parameters
    ----------
    hdul : fits.HDUList
        HDU list to write.
    path : str or Path
        Destination file path.
    overwrite : bool
        If True, allow overwriting an existing file. If False and the
        file exists, raises :class:`~fits_tools.exceptions.FileOverwriteError`.
    dry_run : bool
        If True, log the intended write but do not write anything.
    operation : str, optional
        Short label added to the HISTORY card written to the primary HDU
        header. Default is ``"edit"``.

    Raises
    ------
    FileOverwriteError
        If *path* exists and ``overwrite=False``.
    FileHandlingError
        If the write or atomic replace fails.
    """
    if dry_run:
        logger.info("[dry-run] Would write: %s", path)
        return

    target_path = Path(path)

    if target_path.exists() and not overwrite:
        raise FileOverwriteError(target_path)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    history_msg = f"fits_tools: {operation} at {timestamp}"

    try:
        add_history(hdul[0].header, history_msg)
    except Exception as e:  # pragma: no cover
        logger.debug("Could not add HISTORY card: %s", e)

    dest_dir = target_path.resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = target_path.name
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{base}.tmp.",
            dir=dest_dir,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        hdul.writeto(tmp_path, overwrite=True, checksum=True)
        tmp_path.replace(target_path)

    except Exception as e:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:  # pragma: no cover
                pass
        raise FileHandlingError(
            f"Failed to write {path}: {e}", filepath=target_path
        ) from e


# =============================================================================
# HDU Discovery Functions
# =============================================================================


def _normalize_extname_filter(extname: _ExtNameFilter) -> set[str] | None:
    """Normalize an EXTNAME filter to a set of uppercase strings, or None."""
    if extname is None:
        return None
    raw = [extname] if isinstance(extname, str) else list(extname)
    normalized = {str(n).strip().upper() for n in raw if str(n).strip()}
    return normalized or None


def _unique_key(existing: dict[str, Any], desired: str) -> str:
    """Return *desired* made unique within *existing* by appending ``_{n}``."""
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
) -> Iterator[tuple[int, _ImageHDU]]:
    """Iterate over ``(index, hdu)`` for HDUs that contain image data.

    Operates on an already-open HDUList to avoid extra file open/close
    overhead.

    Parameters
    ----------
    hdul : fits.HDUList
        Open HDU list to iterate over.
    extname : str, iterable of str, or None, optional
        If provided, only HDUs whose EXTNAME matches (case-insensitive)
        are yielded. If None, all image HDUs are yielded.

    Yields
    ------
    tuple of (int, HDU)
        Zero-based index and HDU object for each qualifying HDU.
    """
    allowed = _normalize_extname_filter(extname)

    for i, hdu in enumerate(hdul):
        if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
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
    extname: str | None = None,
) -> list[int]:
    """Return a list of HDU indices that contain image data.

    Parameters
    ----------
    input_file : str or Path
        Path to the FITS file.
    extname : str, optional
        If provided, restrict results to HDUs with this EXTNAME
        (case-insensitive).

    Returns
    -------
    list of int
        Zero-based indices of HDUs containing image data.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileHandlingError(f"File not found: {input_file}", filepath=input_file)

    try:
        with fits.open(input_file, memmap=False, mode="readonly") as hdul:
            indices = [i for i, _ in iter_image_hdus(hdul, extname=extname)]
        logger.debug("Found image HDUs %s in %s", indices, input_file.name)
        return indices
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to get image HDUs from {input_file}: {e}",
            filepath=input_file,
        ) from e


def count_image_extensions(input_file: StrOrPath) -> int:
    """Return the number of HDUs containing image data in a FITS file.

    Parameters
    ----------
    input_file : str or Path
        Path to the FITS file.

    Returns
    -------
    int
        Number of image-bearing HDUs.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    """
    return len(get_image_hdus(input_file))


def count_hdus(filepath: StrOrPath) -> int:
    """Return the total number of HDUs in a FITS file.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.

    Returns
    -------
    int
        Total HDU count, including the primary HDU.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=False, mode="readonly") as hdul:
            return len(hdul)
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to count HDUs in {filepath}: {e}", filepath=filepath
        ) from e


def fits_extension_count(fits_file_path: StrOrPath) -> int:
    """Return the number of extensions (HDUs excluding the primary) in a file.

    .. deprecated::
        Use :func:`count_hdus` for the total HDU count or
        :func:`count_image_extensions` for image HDUs only.

    Parameters
    ----------
    fits_file_path : str or Path
        Path to the FITS file.

    Returns
    -------
    int
        Number of extension HDUs (total HDUs minus 1).
    """
    warnings.warn(
        "fits_extension_count() is deprecated. "
        "Use count_hdus() for total HDU count or "
        "count_image_extensions() for image HDUs.",
        DeprecationWarning,
        stacklevel=2,
    )
    return count_hdus(fits_file_path) - 1


# =============================================================================
# Header Access
# =============================================================================


def get_header(filepath: StrOrPath, hdu: HDUSelector = 0) -> fits.Header:
    """Return a copy of the header from a specified FITS HDU.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0 (primary HDU).

    Returns
    -------
    fits.Header
        Independent copy of the HDU header.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *hdu* is out of range.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=False, mode="readonly") as hdul:
            index = resolve_hdu_index(hdul, hdu, filepath=filepath)
            return hdul[index].header.copy()
    except FitsToolsError:
        raise
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to read header: {e}", filepath=filepath
        ) from e


def get_header_value(
    filepath: StrOrPath,
    keyword: str,
    hdu: HDUSelector = 0,
    default: Any = None,
) -> Any:
    """Return a keyword value using primary-then-selected-HDU fallback.

    The primary header is checked first. If the keyword is absent there and
    *hdu* resolves to a non-primary extension, that extension header is
    checked next. This supports MEF files that keep observation metadata in
    either the primary header or the image extension.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    keyword : str
        FITS header keyword to look up. Matching is case-insensitive.
    hdu : HDUSelector, optional
        Selected image HDU used as the fallback header. Default is 0.
    default : Any, optional
        Value returned when the keyword is absent from both headers.

    Returns
    -------
    Any
        The first matching header value, or *default* when absent.

    Raises
    ------
    ValueError
        If *keyword* is empty.
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *hdu* cannot be resolved.
    """
    normalized_keyword = str(keyword).strip().upper()
    if not normalized_keyword:
        raise ValueError("keyword must be a non-empty string")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=False, mode="readonly") as hdul:
            selected_index = resolve_hdu_index(hdul, hdu, filepath=filepath)
            primary_header = hdul[0].header
            if normalized_keyword in primary_header:
                return primary_header[normalized_keyword]

            if selected_index != 0:
                selected_header = hdul[selected_index].header
                if normalized_keyword in selected_header:
                    return selected_header[normalized_keyword]

            return default
    except FitsToolsError:
        raise
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to read header keyword {normalized_keyword}: {e}",
            filepath=filepath,
        ) from e


# =============================================================================
# Region Validation
# =============================================================================


def _validate_region_bounds(data_shape: tuple[int, int], section: Section) -> None:
    """Raise RegionError if *section* is invalid or outside *data_shape*."""
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
    data_shape: tuple[int, int],
    header: fits.Header,
    region: str | tuple[int, int, int, int] | Section | None = None,
) -> tuple[slice, slice]:
    """Resolve a region specification into ``(row_slice, col_slice)``.

    Parameters
    ----------
    data_shape : tuple of int
        Image shape as ``(rows, cols)``. Must be 2-D.
    header : fits.Header
        FITS header used to look up section keywords when *region* is a
        string keyword name.
    region : str, tuple, Section, or None, optional
        Region to resolve. Accepted forms:

        - ``None`` or ``"full"`` — the entire image.
        - ``"data"``, ``"bias"``, ``"trim"``, ``"dark"`` — read the
          corresponding FITS section keyword from *header*.
        - 4-tuple ``(start_col, end_col, start_row, end_row)`` — explicit
          0-indexed, exclusive-end coordinates.
        - :class:`~fits_tools.sections.Section` instance.

    Returns
    -------
    tuple of slice
        ``(row_slice, col_slice)`` suitable for indexing ``data[row, col]``.

    Raises
    ------
    RegionError
        If the resolved section is out of bounds for *data_shape*.
    HeaderKeywordError
        If a string region keyword is not present in *header*.
    TypeError
        If *region* is an unrecognized type or unsupported string value.
    """
    if len(data_shape) != 2:
        raise RegionError(
            "region_to_slices expects a 2D image shape (rows, cols).",
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
                f"got {len(region)} element(s)."
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
            valid = [r.value for r in RegionType]
            raise TypeError(
                f"Unsupported region string: {region!r}. "
                f"Valid options are: {valid}"
            )
        resolved_section = get_section_by_keyword(header, key)
        if resolved_section is None:
            raise HeaderKeywordError(
                f"Missing required keyword {key!r}", keyword=key
            )
        section = resolved_section
    else:
        raise TypeError(
            f"region must be a string, tuple, Section, or None; "
            f"got {type(region).__name__}"
        )

    _validate_region_bounds(data_shape, section)
    return section.to_slices()


# =============================================================================
# Data Access
# =============================================================================


def get_data_and_header(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
    memmap: bool = False,
) -> tuple[np.ndarray | None, fits.Header]:
    """Return both the data array and header from a specified FITS HDU.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0.
    memmap : bool, optional
        If True, use memory mapping. The returned array will be a
        :class:`numpy.memmap` and the file must remain accessible for its
        lifetime. Default is False.

    Returns
    -------
    data : np.ndarray or None
        Image data array, or None if the HDU contains no data.
    header : fits.Header
        Independent copy of the HDU header.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *hdu* is out of range.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=memmap, mode="readonly") as hdul:
            index = resolve_hdu_index(hdul, hdu, filepath=filepath)
            hdu_obj = hdul[index]
            header = hdu_obj.header.copy()

            if hdu_obj.data is None:
                logger.debug(
                    "HDU %s in %s contains no image data", hdu, filepath
                )
                return None, header

            if memmap:
                return hdu_obj.data, header

            return np.array(hdu_obj.data), header

    except FitsToolsError:
        raise
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to read data and header: {e}", filepath=filepath
        ) from e


@contextmanager
def open_all_data_memmap(
    filepath: StrOrPath,
    extname_filter: str | Sequence[str] | None = None,
    return_dict: bool = False,
) -> Iterator[list[np.ndarray] | dict[str, np.ndarray]]:
    """Context manager yielding memory-mapped arrays for all image HDUs.

    The file remains open for the lifetime of the context. Arrays must not
    be used after the context exits.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    extname_filter : str, sequence of str, or None, optional
        If provided, only HDUs with matching EXTNAME values are included.
    return_dict : bool, optional
        If True, yield a ``dict`` keyed by EXTNAME (or ``HDU_{i}`` for
        unnamed HDUs). If False, yield a ``list``. Default is False.

    Yields
    ------
    list of np.ndarray or dict of str to np.ndarray
        Memory-mapped image arrays, open for the duration of the context.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    with fits.open(filepath, memmap=True, mode="readonly") as hdul:
        pairs = list(iter_image_hdus(hdul, extname=extname_filter))

        if return_dict:
            out: dict[str, np.ndarray] = {}
            for idx, hdu_obj in pairs:
                key = (
                    str(hdu_obj.header.get("EXTNAME", "")).strip()
                    or f"HDU_{idx}"
                )
                key = _unique_key(out, key)
                out[key] = hdu_obj.data
            yield out
        else:
            yield [hdu_obj.data for _, hdu_obj in pairs]


def get_data(
    filepath: StrOrPath,
    hdu: HDUSelector = 0,
    region: str | tuple[int, int, int, int] | Section | None = None,
    memmap: bool = False,
) -> np.ndarray | None:
    """Return data from a FITS HDU, optionally restricted to a region.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector: index, ``EXTNAME``, or ``(EXTNAME, EXTVER)``. Default is 0.
    region : str, tuple, Section, or None, optional
        Region to extract; see :func:`region_to_slices` for accepted forms.
        Default is None (full image).
    memmap : bool, optional
        If True, use memory mapping. If a region is extracted from a
        memmap array, the result is copied into a regular array before
        returning. Default is False.

    Returns
    -------
    np.ndarray or None
        Image data, or None if the HDU contains no data.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *hdu* is out of range.
    HeaderKeywordError
        If a string region references a missing header keyword.
    RegionError
        If the region is invalid or out of bounds.
    """
    try:
        data, header = get_data_and_header(filepath, hdu=hdu, memmap=memmap)

        if data is None:
            logger.warning(
                "HDU %s in %s contains no image data", hdu, filepath
            )
            return None

        if region is None or region in (RegionType.FULL.value, "full"):
            return data

        row_slice, col_slice = region_to_slices(data.shape, header, region)
        sliced = data[row_slice, col_slice]

        if memmap and isinstance(data, np.memmap):
            return np.array(sliced)

        return sliced

    except (FitsToolsError, TypeError):
        raise
    except Exception as e:  # pragma: no cover
        raise FitsToolsError(f"Failed to read data: {e}") from e


def get_all_data(
    filepath: StrOrPath,
    extname_filter: str | None = None,
    return_dict: bool = False,
) -> list[np.ndarray] | dict[str, np.ndarray]:
    """Return data arrays from all image HDUs in a FITS file.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    extname_filter : str, optional
        If provided, restrict to HDUs with this EXTNAME (case-insensitive).
    return_dict : bool, optional
        If True, return a ``dict`` keyed by EXTNAME (or ``HDU_{i}`` for
        unnamed HDUs). If False, return a ``list``. Default is False.

    Returns
    -------
    list of np.ndarray or dict of str to np.ndarray
        Image data arrays loaded into memory.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
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

            if return_dict:
                out: dict[str, np.ndarray] = {}
                for idx, hdu_obj in pairs:
                    key = (
                        str(hdu_obj.header.get("EXTNAME", "")).strip()
                        or f"HDU_{idx}"
                    )
                    key = _unique_key(out, key)
                    out[key] = np.array(hdu_obj.data)
                return out

            return [np.array(hdu_obj.data) for _, hdu_obj in pairs]

    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to read all data: {e}", filepath=filepath
        ) from e


# =============================================================================
# Writing
# =============================================================================


def write_hdulist(
    filepath: StrOrPath,
    hdul: fits.HDUList,
    overwrite: bool = False,
) -> None:
    """Write an :class:`astropy.io.fits.HDUList` to disk.

    Preserves the full MEF structure of *hdul*.

    Parameters
    ----------
    filepath : str or Path
        Destination file path.
    hdul : fits.HDUList
        HDU list to write.
    overwrite : bool, optional
        If True, overwrite an existing file. Default is False.

    Raises
    ------
    FileOverwriteError
        If *filepath* exists and ``overwrite=False``.
    FileHandlingError
        If the write fails.
    """
    filepath = Path(filepath)

    if filepath.exists() and not overwrite:
        raise FileOverwriteError(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(filepath, overwrite=overwrite, checksum=True)
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to write FITS HDUList: {e}", filepath=filepath
        ) from e


def transactional_write_hdulist(
    filepath: StrOrPath,
    hdul: fits.HDUList,
) -> None:
    """Atomically replace *filepath* with a completely written HDU list.

    The temporary file is created beside the destination so the final
    :func:`os.replace` remains on the same filesystem.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{filepath.name}.",
            suffix=".tmp",
            dir=filepath.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        hdul.writeto(temp_path, overwrite=True, checksum=True)
        os.replace(temp_path, filepath)
    except Exception as exc:
        raise FileHandlingError(
            f"Failed to transactionally write FITS HDUList: {exc}",
            filepath=filepath,
        ) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_simple_fits(
    filepath: StrOrPath,
    data: np.ndarray,
    header: fits.Header | None = None,
    overwrite: bool = False,
    compress: bool = False,
) -> None:
    """Write a single-image FITS file.

    Parameters
    ----------
    filepath : str or Path
        Destination file path.
    data : np.ndarray
        Image data to write.
    header : fits.Header, optional
        Header to attach to the image HDU.
    overwrite : bool, optional
        If True, overwrite an existing file. Default is False.
    compress : bool, optional
        If True, use tile compression (RICE_1 for integer data, GZIP_1
        for floating-point). The compressed image is placed in a
        ``CompImageHDU`` with an empty primary HDU. Default is False.

    Raises
    ------
    FileOverwriteError
        If *filepath* exists and ``overwrite=False``.
    FileHandlingError
        If the write fails.
    """
    filepath = Path(filepath)

    if filepath.exists() and not overwrite:
        raise FileOverwriteError(filepath)

    try:
        if compress:
            compression_type = (
                "RICE_1" if np.issubdtype(data.dtype, np.integer) else "GZIP_1"
            )
            hdu = fits.CompImageHDU(
                data=data,
                header=header,
                compression_type=compression_type,
            )
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        else:
            hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])

        hdul.writeto(filepath, overwrite=overwrite, checksum=True)
        logger.debug("FITS file written to: %s", filepath)

    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to write FITS file: {e}", filepath=filepath
        ) from e


def write_fits(*args: Any, **kwargs: Any) -> None:
    """Write a single-image FITS file.

    .. deprecated::
        Use :func:`write_simple_fits` for single-HDU output or
        :func:`write_hdulist` for MEF preservation.
    """
    warnings.warn(
        "write_fits() is deprecated; use write_simple_fits() for single-HDU "
        "output or write_hdulist() for MEF preservation.",
        DeprecationWarning,
        stacklevel=2,
    )
    write_simple_fits(*args, **kwargs)


# =============================================================================
# Inspection
# =============================================================================


_INSPECT_HEADER_KEYS = (
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
)


@dataclass
class HDUInfo:
    """Structural metadata for a single HDU.

    Attributes
    ----------
    index : int
        Zero-based HDU index.
    hdu_type : str
        HDU class name (e.g. ``'PrimaryHDU'``, ``'ImageHDU'``).
    shape : tuple of int or None
        Data shape, or None if the HDU has no data.
    dtype : str or None
        Data dtype string, or None if the HDU has no data.
    data_min : float or None
        Minimum pixel value, populated only when *detailed* is True.
    data_max : float or None
        Maximum pixel value, populated only when *detailed* is True.
    header_cards : dict of str to Any
        Selected header keyword/value pairs (populated when *detailed*
        is True).
    """

    index: int
    hdu_type: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    data_min: float | None = None
    data_max: float | None = None
    header_cards: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"--- HDU {self.index} ---"]
        if self.shape is not None:
            lines.append(f"Data shape: {self.shape}")
            lines.append(f"Data type:  {self.dtype}")
            if self.data_min is not None:
                lines.append(
                    f"Data range: [{self.data_min:.3f}, {self.data_max:.3f}]"
                )
        else:
            lines.append("No data")
        if self.header_cards:
            lines.append("")
            lines.append("Key header cards:")
            for key, value in self.header_cards.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class FitsInfo:
    """Structured inspection result for a FITS file.

    Attributes
    ----------
    filename : str
        Base name of the file.
    filepath : Path
        Full path to the file.
    size_kb : float
        File size in kilobytes.
    num_hdus : int
        Total number of HDUs.
    hdus : list of HDUInfo
        Per-HDU metadata. Always populated with type and shape; data
        range and header cards are included only when the inspection was
        run with ``detailed=True``.
    """

    filename: str
    filepath: Path
    size_kb: float
    num_hdus: int
    hdus: list[HDUInfo] = field(default_factory=list)

    def __str__(self) -> str:
        sep = "=" * 70
        lines = [
            "",
            sep,
            f"FITS File: {self.filename}",
            f"Path: {self.filepath}",
            f"Size: {self.size_kb:.2f} KB",
            sep,
            "",
        ]
        for hdu_info in self.hdus:
            lines.append("")
            lines.append(str(hdu_info))
        return "\n".join(lines)


def inspect_fits(
    filepath: StrOrPath,
    detailed: bool = False,
) -> FitsInfo:
    """Inspect the structure of a FITS file.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    detailed : bool, optional
        If True, include per-HDU data range and selected header
        keywords. Default is False.

    Returns
    -------
    FitsInfo
        Structured inspection result. Use ``print(result)`` for
        human-readable output.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(
            f"File not found: {filepath}", filepath=filepath
        )

    try:
        with fits.open(filepath, memmap=False, mode="readonly") as hdul:
            info = FitsInfo(
                filename=filepath.name,
                filepath=filepath,
                size_kb=filepath.stat().st_size / 1024,
                num_hdus=len(hdul),
            )

            for i, hdu_obj in enumerate(hdul):
                hdu_info = HDUInfo(
                    index=i,
                    hdu_type=type(hdu_obj).__name__,
                )

                if hdu_obj.data is not None:
                    hdu_info.shape = tuple(hdu_obj.data.shape)
                    # Report the canonical scalar type without FITS' on-disk
                    # byte-order marker (for example, ``float32`` not ``>f4``).
                    hdu_info.dtype = hdu_obj.data.dtype.name

                    if detailed:
                        hdu_info.data_min = float(np.min(hdu_obj.data))
                        hdu_info.data_max = float(np.max(hdu_obj.data))

                if detailed:
                    for key in _INSPECT_HEADER_KEYS:
                        if key in hdu_obj.header:
                            hdu_info.header_cards[key] = hdu_obj.header[key]

                info.hdus.append(hdu_info)

            return info

    except FileHandlingError:
        raise
    except Exception as e:  # pragma: no cover
        raise FileHandlingError(
            f"Failed to inspect FITS file: {e}", filepath=filepath
        ) from e
