#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Utilities for managing FITS header keywords.

This module provides functions to check for, get, set, and delete FITS
header keywords in individual FITS files, as well as batch operations
across multiple files. Batch operations support directory/glob patterns
and can filter files by header criteria.

Features
--------
**Single-file operations:**

- :func:`check_keyword` -- Check if a keyword exists
- :func:`get_key` -- Retrieve a keyword value
- :func:`set_key` -- Set or update a keyword
- :func:`delete_key` -- Remove a keyword

**Batch operations:**

- :func:`batch_set_key` -- Set keywords across multiple files
- :func:`batch_delete_key` -- Delete keywords from multiple files

All functions support flexible HDU selection and provide comprehensive
error handling through the :mod:`fits_tools.exceptions` module.

HDU Selection
-------------
Most functions accept an *hdu* parameter supporting multiple selection methods:

- **int** -- Zero-based HDU index (``0`` = primary HDU)
- **str** -- Extension name (matches EXTNAME header value)
- **(str, int)** -- Tuple of (EXTNAME, EXTVER)
- **"auto"** -- Automatic selection (default: primary HDU)
- **None** -- Batch operations only: operate on all image-like HDUs

See :class:`fits_tools.hdu.HDUSelector` for details.

Examples
--------
Check and retrieve keyword values::

    from fits_tools.keywords import check_keyword, get_key

    if check_keyword('image.fits', 'OBJECT'):
        target = get_key('image.fits', 'OBJECT')
        exptime = get_key('image.fits', 'EXPTIME', hdu=('SCI', 1))

Set and delete keywords::

    from fits_tools.keywords import set_key, delete_key

    set_key('image.fits', 'PROCESSED', True, comment='Pipeline v2.0')
    delete_key('image.fits', 'COMMENT')

Batch operations on multiple files::

    from fits_tools.keywords import batch_set_key

    # Process all FITS files in directory
    count = batch_set_key('data/', 'PIPELINE', 'v2.1.0', recursive=True)

    # Process only files matching header criteria
    count = batch_set_key(
        'data/*.fits',
        'CALIBRATED',
        True,
        keys={'IMAGETYP': 'LIGHT'},
    )

See Also
--------
fits_tools.selection : File selection and filtering
fits_tools.hdu : HDU selection utilities
fits_tools.exceptions : Custom exception classes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from astropy.io import fits

from .exceptions import (
    FileHandlingError,
    HeaderKeywordError,
    InvalidHDUIndexError,
    SelectionError,
)
from .hdu import HDUSelector, is_image_like_hdu, resolve_hdu_index
from .types import StrOrPath

logger = logging.getLogger(__name__)

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


# =============================================================================
# Single-File Keyword Operations
# =============================================================================


def check_keyword(
    filepath: StrOrPath,
    keyword: str,
    *,
    hdu: HDUSelector = "auto",
) -> bool:
    """Check if a FITS keyword exists in a specified HDU.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    keyword : str
        FITS keyword to check for (case-insensitive).
    hdu : HDUSelector, optional
        HDU selector: integer index (0-based), extension name, (EXTNAME,
        EXTVER) tuple, or ``"auto"`` for the primary HDU. Default is
        ``"auto"``.

    Returns
    -------
    bool
        True if the keyword exists in the specified HDU, False otherwise.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *hdu* is out of bounds.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=True, mode="readonly") as hdul:
            hdu_index = resolve_hdu_index(
                hdul,
                hdu,
                auto="primary",
                image_like=False,
                require_data=False,
                filepath=filepath,
            )
            return keyword.upper() in hdul[hdu_index].header

    except InvalidHDUIndexError:
        raise
    except ValueError as e:
        raise ValueError(f"Invalid HDU selector {hdu!r}: {e}") from e
    except Exception as e:  # pragma: no cover
        logger.error(
            "Error checking keyword %r in %s (hdu=%r): %s",
            keyword,
            filepath,
            hdu,
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
    *,
    hdu: HDUSelector = "auto",
) -> Any:
    """Retrieve the value of a FITS keyword from a specified HDU.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    keyword : str
        FITS keyword to retrieve (case-insensitive).
    hdu : HDUSelector, optional
        HDU selector: integer index (0-based), extension name, (EXTNAME,
        EXTVER) tuple, or ``"auto"`` for the primary HDU. Default is
        ``"auto"``.

    Returns
    -------
    Any
        Value of the specified keyword.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If *hdu* is out of bounds.
    HeaderKeywordError
        If the keyword is not present in the specified HDU.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=True, mode="readonly") as hdul:
            hdu_index = resolve_hdu_index(
                hdul,
                hdu,
                auto="primary",
                image_like=False,
                require_data=False,
                filepath=filepath,
            )

            header = hdul[hdu_index].header
            key = keyword.upper()

            if key in header:
                return header[key]

            raise HeaderKeywordError(
                f"Keyword {keyword!r} not found in HDU {hdu_index} of "
                f"file {filepath.name!r}.",
                keyword=keyword,
                filepath=filepath,
            )

    except (InvalidHDUIndexError, HeaderKeywordError):
        raise
    except ValueError as e:
        raise ValueError(f"Invalid HDU selector {hdu!r}: {e}") from e
    except Exception as e:  # pragma: no cover
        logger.error(
            "Error getting keyword %r from %s (hdu=%r): %s",
            keyword,
            filepath,
            hdu,
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
    comment: str | None = None,
    *,
    hdu: HDUSelector = "auto",
) -> bool:
    """Set or update a FITS header keyword.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    keyword : str
        FITS keyword to set or update (case-insensitive).
    value : Any
        Value to assign.
    comment : str, optional
        Comment for the header card.
    hdu : HDUSelector, optional
        HDU selector: integer index (0-based), extension name, (EXTNAME,
        EXTVER) tuple, or ``"auto"`` for the primary HDU. Default is
        ``"auto"``.

    Returns
    -------
    bool
        True if the keyword was set or updated successfully.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened or written.
    InvalidHDUIndexError
        If *hdu* is out of bounds.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, mode="update", memmap=True) as hdul:
            hdu_index = resolve_hdu_index(
                hdul,
                hdu,
                auto="primary",
                image_like=False,
                require_data=False,
                filepath=filepath,
            )

            key = keyword.upper()
            if comment is not None:
                hdul[hdu_index].header[key] = (value, comment)
            else:
                hdul[hdu_index].header[key] = value

            hdul.flush()

            logger.debug(
                "Set keyword %r to %r in %s (hdu=%r -> hdu_index=%d).",
                keyword,
                value,
                filepath.name,
                hdu,
                hdu_index,
            )
            return True

    except InvalidHDUIndexError:
        raise
    except ValueError as e:
        raise ValueError(f"Invalid HDU selector {hdu!r}: {e}") from e
    except Exception as e:  # pragma: no cover
        logger.error(
            "Error setting keyword %r in %s (hdu=%r): %s",
            keyword,
            filepath,
            hdu,
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
    *,
    hdu: HDUSelector = "auto",
) -> bool:
    """Delete a FITS header keyword from a specified HDU.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    keyword : str
        FITS keyword to delete (case-insensitive).
    hdu : HDUSelector, optional
        HDU selector: integer index (0-based), extension name, (EXTNAME,
        EXTVER) tuple, or ``"auto"`` for the primary HDU. Default is
        ``"auto"``.

    Returns
    -------
    bool
        True if the keyword was deleted.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened or written.
    InvalidHDUIndexError
        If *hdu* is out of bounds.
    HeaderKeywordError
        If the keyword does not exist in the specified HDU.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, mode="update", memmap=True) as hdul:
            hdu_index = resolve_hdu_index(
                hdul,
                hdu,
                auto="primary",
                image_like=False,
                require_data=False,
                filepath=filepath,
            )

            header = hdul[hdu_index].header
            key = keyword.upper()

            if key in header:
                del header[key]
                hdul.flush()
                logger.debug(
                    "Deleted keyword %r from %s (hdu=%r -> hdu_index=%d).",
                    keyword,
                    filepath.name,
                    hdu,
                    hdu_index,
                )
                return True

            raise HeaderKeywordError(
                f"Keyword {keyword!r} not found in HDU {hdu_index} of "
                f"file {filepath.name!r}.",
                keyword=keyword,
                filepath=filepath,
            )

    except (InvalidHDUIndexError, HeaderKeywordError):
        raise
    except ValueError as e:
        raise ValueError(f"Invalid HDU selector {hdu!r}: {e}") from e
    except Exception as e:  # pragma: no cover
        logger.error(
            "Error deleting keyword %r from %s (hdu=%r): %s",
            keyword,
            filepath.name,
            hdu,
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
    files: StrOrPath | list[StrOrPath],
    keyword: str,
    value: Any,
    *,
    comment: str | None = None,
    hdu: HDUSelector | None = "auto",
    keys: dict[str, Any] | None = None,
    recursive: bool = False,
) -> int:
    """Set or update a FITS keyword in multiple FITS files.

    Parameters
    ----------
    files : str, Path, or list of str or Path
        Directory path, glob pattern, or list of FITS file paths.
    keyword : str
        FITS keyword to set or update (case-insensitive).
    value : Any
        Value to assign.
    comment : str, optional
        Comment for the header card.
    hdu : HDUSelector or None, optional
        HDU selector. If ``None``, modifies all image-like HDUs in each
        file. If ``"auto"`` (default), modifies the primary HDU only.
        Can also be an integer index, extension name, or (EXTNAME, EXTVER)
        tuple.
    keys : dict, optional
        Header keyword/value pairs used to filter files (passed to
        :func:`fits_tools.selection.select_files`).
    recursive : bool, optional
        If True and *files* is a directory or pattern, search
        subdirectories recursively. Default is False.

    Returns
    -------
    int
        Number of FITS files where the keyword was successfully set or
        updated.

    Raises
    ------
    SelectionError
        If file selection fails.
    """
    from .selection import select_files  # local import to avoid cycles

    try:
        files_to_process = select_files(files, keys=keys, recursive=recursive)
    except SelectionError as e:
        logger.error("Error selecting files for batch keyword setting: %s", e)
        raise

    if not files_to_process:
        logger.warning(
            "No FITS files found matching the criteria for batch setting keyword."
        )
        return 0

    updated_count = 0
    key = keyword.upper()

    for file_path in files_to_process:
        file_path = Path(file_path)

        try:
            with fits.open(file_path, mode="update", memmap=True) as hdul:
                if hdu is None:
                    indices = [i for i, h in enumerate(hdul) if is_image_like_hdu(h)]
                    if not indices:
                        logger.warning(
                            "Skipping %s: no image-like HDUs found.",
                            file_path.name,
                        )
                        continue
                else:
                    idx = resolve_hdu_index(
                        hdul,
                        hdu,
                        auto="primary",
                        image_like=False,
                        require_data=False,
                        filepath=file_path,
                    )
                    indices = [idx]

                for idx in indices:
                    if comment is not None:
                        hdul[idx].header[key] = (value, comment)
                    else:
                        hdul[idx].header[key] = value

                    logger.debug(
                        "Set keyword %r to %r in %s (hdu=%r -> idx=%d).",
                        keyword,
                        value,
                        file_path.name,
                        hdu,
                        idx,
                    )

                hdul.flush()
                updated_count += 1

        except FileNotFoundError:
            logger.warning(
                "File not found during batch processing: %s. Skipping.", file_path
            )
        except OSError as e:
            logger.warning(
                "Error opening/writing FITS file %s: %s. Skipping.", file_path, e
            )
        except Exception as e:  # pragma: no cover
            logger.warning(
                "Unexpected error processing %s: %s. Skipping.",
                file_path,
                e,
                exc_info=True,
            )

    logger.info("Keyword %r set/updated in %d file(s).", keyword, updated_count)
    return updated_count


def batch_delete_key(
    files: StrOrPath | list[StrOrPath],
    keyword: str,
    *,
    hdu: HDUSelector | None = "auto",
    keys: dict[str, Any] | None = None,
    recursive: bool = False,
) -> int:
    """Delete a FITS keyword from multiple FITS files.

    Parameters
    ----------
    files : str, Path, or list of str or Path
        Directory path, glob pattern, or list of FITS file paths.
    keyword : str
        FITS keyword to delete (case-insensitive).
    hdu : HDUSelector or None, optional
        HDU selector. If ``None``, applies to all image-like HDUs.
        If ``"auto"`` (default), applies to the primary HDU only.
        Can also be an integer index, extension name, or (EXTNAME, EXTVER)
        tuple.
    keys : dict, optional
        Header keyword/value pairs used to filter files (passed to
        :func:`fits_tools.selection.select_files`).
    recursive : bool, optional
        If True and *files* is a directory or pattern, search
        subdirectories recursively. Default is False.

    Returns
    -------
    int
        Number of FITS files where the keyword was deleted.

    Raises
    ------
    SelectionError
        If file selection fails.
    """
    from .selection import select_files  # local import to avoid cycles

    try:
        files_to_process = select_files(files, keys=keys, recursive=recursive)
    except SelectionError as e:
        logger.error("Error selecting files for batch keyword deletion: %s", e)
        raise

    if not files_to_process:
        logger.warning(
            "No FITS files found matching the criteria for batch deleting keyword."
        )
        return 0

    updated_count = 0
    key = keyword.upper()

    for file_path in files_to_process:
        file_path = Path(file_path)

        try:
            with fits.open(file_path, mode="update", memmap=True) as hdul:
                if hdu is None:
                    indices = [i for i, h in enumerate(hdul) if is_image_like_hdu(h)]
                    if not indices:
                        logger.warning(
                            "Skipping %s: no image-like HDUs found.",
                            file_path.name,
                        )
                        continue
                else:
                    idx = resolve_hdu_index(
                        hdul,
                        hdu,
                        auto="primary",
                        image_like=False,
                        require_data=False,
                        filepath=file_path,
                    )
                    indices = [idx]

                changes_made = False
                for idx in indices:
                    header = hdul[idx].header
                    if key in header:
                        del header[key]
                        changes_made = True
                        logger.debug(
                            "Deleted keyword %r from %s (hdu=%r -> idx=%d).",
                            keyword,
                            file_path.name,
                            hdu,
                            idx,
                        )
                    else:
                        logger.debug(
                            "Keyword %r not found in %s (hdu=%r -> idx=%d).",
                            keyword,
                            file_path.name,
                            hdu,
                            idx,
                        )

                if changes_made:
                    hdul.flush()
                    updated_count += 1

        except FileNotFoundError:
            logger.warning(
                "File not found during batch processing: %s. Skipping.", file_path
            )
        except OSError as e:
            logger.warning(
                "Error opening/reading FITS file %s: %s. Skipping.", file_path, e
            )
        except Exception as e:  # pragma: no cover
            logger.warning(
                "Unexpected error processing %s: %s. Skipping.",
                file_path,
                e,
                exc_info=True,
            )

    logger.info("Keyword %r deleted from %d file(s).", keyword, updated_count)
    return updated_count
