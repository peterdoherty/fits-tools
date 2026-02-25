#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for managing FITS header keywords.

This module provides functions to check for, get, set, and delete FITS
header keywords in individual FITS files, as well as batch operations
across multiple files.

Features
--------
- Check for keyword existence
- Get/set/delete individual keywords
- Batch operations across multiple files
- Support for filtering files by header criteria (via selection.select_files)
- Consistent error handling using fits_tools exceptions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU

from .exceptions import (
    FileHandlingError,
    HeaderKeywordError,
    InvalidHDUIndexError,
    SelectionError,
)
from .types import StrOrPath

logger = logging.getLogger(__name__)
__version__ = "0.2.0"


def def _is_image_like_hdu(hdu: _BaseHDU) -> bool:
    """Return True if the HDU looks like an image HDU, without reading data.

    This avoids materializing large arrays or triggering decompression when
    performing header-only operations (e.g., setting or deleting keywords).
    """
    if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        return False
    try:
        naxis = int(hdu.header.get("NAXIS", 0) or 0)
        if naxis < 2:
            return False
        naxis1 = int(hdu.header.get("NAXIS1", 0) or 0)
        naxis2 = int(hdu.header.get("NAXIS2", 0) or 0)
        return (naxis1 > 0) and (naxis2 > 0)
    except Exception:
        # If header is malformed, be conservative and treat it as non-image.
        return False


# =============================================================================
# Single-File Keyword Operations
# =============================================================================

def check_keyword(
    filepath: StrOrPath,
    keyword: str,
    ext: int = 0,
) -> bool:
    """Check if a FITS keyword exists in a specified HDU.

    Parameters
    ----------
    filepath :
        Path to the FITS file.
    keyword :
        FITS keyword to check for (case-insensitive).
    ext :
        HDU extension index to check (0-based). Default is 0 (primary).

    Returns
    -------
    bool
        True if the keyword exists in the specified HDU, False otherwise.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *ext* is out of bounds.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=True) as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))
            return keyword.upper() in hdul[ext].header

    except InvalidHDUIndexError:
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Error checking keyword %r in %s (ext %d): %s",
            keyword,
            filepath,
            ext,
            e,
            exc_info=True,
        )
        raise FileHandlingError(
            f"Error opening or reading FITS file {filepath}: {e}",
            filepath=filepath,
        ) from e


def get_key(
    filepath: StrOrPath,
    keyword: str,
    ext: int = 0,
) -> Any:
    """Retrieve the value of a FITS keyword from a specified HDU.

    Parameters
    ----------
    filepath :
        Path to the FITS file.
    keyword :
        FITS keyword to retrieve (case-insensitive).
    ext :
        HDU extension index (0-based). Default is 0 (primary).

    Returns
    -------
    Any
        Value of the specified keyword.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *ext* is out of bounds.
    HeaderKeywordError
        If the keyword is not present in the specified HDU.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=True) as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))

            header = hdul[ext].header
            key = keyword.upper()

            if key in header:
                return header[key]

            raise HeaderKeywordError(
                f"Keyword {keyword!r} not found in HDU {ext} of file "
                f"{filepath.name!r}.",
                keyword=keyword,
                filepath=filepath,
            )

    except (InvalidHDUIndexError, HeaderKeywordError):
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Error getting keyword %r from %s (ext %d): %s",
            keyword,
            filepath,
            ext,
            e,
            exc_info=True,
        )
        raise FileHandlingError(
            f"Error opening or reading FITS file {filepath}: {e}",
            filepath=filepath,
        ) from e


def set_key(
    filepath: StrOrPath,
    keyword: str,
    value: Any,
    comment: Optional[str] = None,
    ext: int = 0,
) -> bool:
    """Set or update a FITS header keyword.

    Parameters
    ----------
    filepath :
        Path to the FITS file.
    keyword :
        FITS keyword to set or update (case-insensitive).
    value :
        Value to assign.
    comment :
        Optional comment for the header card.
    ext :
        HDU extension index to modify. Default is 0 (primary).

    Returns
    -------
    bool
        True if the keyword was set/updated successfully.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened/written.
    InvalidHDUIndexError
        If *ext* is out of bounds.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, mode="update") as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))

            key = keyword.upper()
            if comment:
                hdul[ext].header[key] = (value, comment)
            else:
                hdul[ext].header[key] = value

            hdul.flush()

            logger.debug(
                "Set keyword %r to %r in %s (ext %d).",
                keyword,
                value,
                filepath,
                ext,
            )
            return True

    except InvalidHDUIndexError:
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Error setting keyword %r in %s (ext %d): %s",
            keyword,
            filepath,
            ext,
            e,
            exc_info=True,
        )
        raise FileHandlingError(
            f"Error opening or writing to FITS file {filepath}: {e}",
            filepath=filepath,
        ) from e


def delete_key(
    filepath: StrOrPath,
    keyword: str,
    ext: int = 0,
) -> bool:
    """Delete a FITS header keyword from a specified HDU.

    Parameters
    ----------
    filepath :
        Path to the FITS file.
    keyword :
        FITS keyword to delete (case-insensitive).
    ext :
        HDU extension index to modify. Default is 0 (primary).

    Returns
    -------
    bool
        True if the keyword was deleted.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened/written.
    InvalidHDUIndexError
        If *ext* is out of bounds.
    HeaderKeywordError
        If the keyword does not exist in the specified HDU.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, mode="update") as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))

            header = hdul[ext].header
            key = keyword.upper()

            if key in header:
                del header[key]
                hdul.flush()
                logger.debug(
                    "Deleted keyword %r from %s (ext %d).",
                    keyword,
                    filepath.name,
                    ext,
                )
                return True

            raise HeaderKeywordError(
                f"Keyword {keyword!r} not found in HDU {ext} of file "
                f"{filepath.name!r}.",
                keyword=keyword,
                filepath=filepath,
            )

    except (InvalidHDUIndexError, HeaderKeywordError):
        raise
    except Exception as e:  # pragma: no cover - error path
        logger.error(
            "Error deleting keyword %r from %s (ext %d): %s",
            keyword,
            filepath.name,
            ext,
            e,
            exc_info=True,
        )
        raise FileHandlingError(
            f"Error opening or writing to FITS file {filepath}: {e}",
            filepath=filepath,
        ) from e


# =============================================================================
# Batch Keyword Operations
# =============================================================================

def batch_set_key(
    files: Union[StrOrPath, List[StrOrPath]],
    keyword: str,
    value: Any,
    comment: Optional[str] = None,
    ext: Optional[int] = None,
    keys: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
) -> int:
    """Set or update a FITS keyword in multiple FITS files.

    Parameters
    ----------
    files :
        Directory path, glob pattern, or list of FITS file paths.
    keyword :
        FITS keyword to set or update (case-insensitive).
    value :
        Value to assign.
    comment :
        Optional comment for the header card.
    ext :
        HDU extension index to modify. If None, modifies all image-like
        HDUs (including a primary HDU that contains image data).
    keys :
        Dictionary of header keyword/value pairs used to filter files
        (passed to :func:`fits_tools.selection.select_files`).
    recursive :
        If True and *files* is a directory or pattern, search
        subdirectories recursively.

    Returns
    -------
    int
        Number of FITS files where the keyword was successfully set/updated.

    Raises
    ------
    SelectionError
        If file selection fails.
    """
    from .selection import select_files  # local import to avoid cycles

    try:
        files_to_process = select_files(
        files,
        keys=keys,
        recursive=recursive,
    )
    except SelectionError as e:
        logger.error("Error selecting files for batch keyword setting: %s", e)
        raise

    if not files_to_process:
        logger.warning(
            "No FITS files found matching the criteria for batch setting keyword."
        )
        return 0

    updated_files_count = 0
    key = keyword.upper()

    for file_path in files_to_process:
        file_path = Path(file_path)
        try:
            with fits.open(file_path, mode="update") as hdul:
                if ext is not None:
                    if ext < 0 or ext >= len(hdul):
                        logger.warning(
                            "Skipping %s: extension %d is out of bounds "
                            "(file has %d HDUs).",
                            file_path.name,
                            ext,
                            len(hdul),
                        )
                        continue
                    exts_to_modify = [ext]
                else:
                    exts_to_modify = [
                        i for i, hdu in enumerate(hdul) if _is_image_like_hdu(hdu)
                    ]
                    if not exts_to_modify:
                        logger.warning(
                            "Skipping %s: no image extensions found to modify.",
                            file_path.name,
                        )
                        continue

                for current_ext in exts_to_modify:
                    if comment:
                        hdul[current_ext].header[key] = (value, comment)
                    else:
                        hdul[current_ext].header[key] = value

                    logger.debug(
                        "  Set keyword %r to %r in %s (ext %d).",
                        keyword,
                        value,
                        file_path.name,
                        current_ext,
                    )

                hdul.flush()
                updated_files_count += 1

        except FileNotFoundError:
            logger.warning(
                "  File not found during batch processing: %s. Skipping.", file_path
            )
        except OSError as e:
            logger.warning(
                "  Error opening or writing FITS file %s: %s. Skipping.",
                file_path,
                e,
            )
        except Exception as e:  # pragma: no cover - error path
            logger.warning(
                "  Unexpected error while processing %s: %s. Skipping.",
                file_path,
                e,
                exc_info=True,
            )

    logger.info(
        "Total FITS files where keyword %r was set/updated: %d",
        keyword,
        updated_files_count,
    )

    return updated_files_count


def batch_delete_key(
    files: Union[StrOrPath, List[StrOrPath]],
    keyword: str,
    ext: Optional[int] = None,
    keys: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
) -> int:
    """Delete a FITS keyword from multiple FITS files.

    Parameters
    ----------
    files :
        Directory path, glob pattern, or list of FITS file paths.
    keyword :
        FITS keyword to delete (case-insensitive).
    ext :
        HDU extension index to modify. If None, deletes from all image-like
        HDUs (including a primary HDU that contains image data).
    keys :
        Dictionary of header keyword/value pairs used to filter files
        (passed to :func:`fits_tools.selection.select_files`).
    recursive :
        If True and *files* is a directory or pattern, search
        subdirectories recursively.

    Returns
    -------
    int
        Number of FITS files where the keyword was deleted.

    Raises
    ------
    SelectionError
        If file selection fails.
    """
    from fits_tools.selection import select_files  # local import to avoid cycles

    try:
        files_to_process = select_files(
            files, keys=keys, recursive=recursive
        )
    except SelectionError as e:
        logger.error("Error selecting files for batch keyword deletion: %s", e)
        raise

    if not files_to_process:
        logger.warning(
            "No FITS files found matching the criteria for batch deleting keyword."
        )
        return 0

    updated_files_count = 0
    key = keyword.upper()

    for file_path in files_to_process:
        file_path = Path(file_path)
        try:
            with fits.open(file_path, mode="update") as hdul:
                if ext is not None:
                    if ext < 0 or ext >= len(hdul):
                        logger.warning(
                            "Skipping %s: extension %d is out of bounds "
                            "(file has %d HDUs).",
                            file_path.name,
                            ext,
                            len(hdul),
                        )
                        continue
                    exts_to_modify = [ext]
                else:
                    exts_to_modify = [
                        i for i, hdu in enumerate(hdul) if _is_image_like_hdu(hdu)
                    ]
                    if not exts_to_modify:
                        logger.warning(
                            "Skipping %s: no image extensions found to modify.",
                            file_path.name,
                        )
                        continue

                changes_made = False
                for current_ext in exts_to_modify:
                    header = hdul[current_ext].header
                    if key in header:
                        del header[key]
                        changes_made = True
                        logger.debug(
                            "  Deleted keyword %r from %s (ext %d).",
                            keyword,
                            file_path.name,
                            current_ext,
                        )
                    else:
                        logger.debug(
                            "  Keyword %r not found in %s (ext %d).",
                            keyword,
                            file_path.name,
                            current_ext,
                        )

                if changes_made:
                    hdul.flush()
                    updated_files_count += 1

        except FileNotFoundError:
            logger.warning(
                "  File not found during batch processing: %s. Skipping.", file_path
            )
        except OSError as e:
            logger.warning(
                "  Error opening or reading FITS file %s: %s. Skipping.",
                file_path,
                e,
            )
        except Exception as e:  # pragma: no cover - error path
            logger.warning(
                "  Unexpected error while processing %s: %s. Skipping.",
                file_path,
                e,
                exc_info=True,
            )

    logger.debug(
        "Total FITS files where keyword %r was deleted: %d",
        keyword,
        updated_files_count,
    )

    return updated_files_count

__all__ = [
    # Single-file operations
    "check_keyword",
    "get_key",
    "set_key",
    "delete_key",
    # Batch operations
    "batch_set_key",
    "batch_delete_key",
]
