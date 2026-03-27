#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for selecting and filtering FITS files.

This module provides utilities to select FITS files based on various criteria,
including directory scanning, glob patterns, and header keyword filtering.
It is particularly useful for batch processing of CCD/CMOS calibration frames.

Features
--------
- Select files from directories, glob patterns, or explicit lists
- Filter by FITS header keywords
- Recursive directory scanning
- Convenience functions for bias, dark, flat, and light frames
- Optional maximum file limits for large datasets
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from astropy.io import fits

from .constants import (
    FITS_KEY_IMGTYPE,
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
)
from .exceptions import SelectionError
from .utils import is_fits_file
from .types import StrOrPath

logger = logging.getLogger(__name__)


def _normalize_for_compare(value: Any) -> Any:
    """Normalize values for keyword comparison.

    Strings are stripped and uppercased for case-insensitive matching.
    Non-strings are returned unchanged.
    """
    if isinstance(value, str):
        return value.strip().upper()
    return value


# =============================================================================
# Main Selection Function
# =============================================================================

def select_files(
    files: Union[StrOrPath, List[StrOrPath]],
    keys: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
    max_files: Optional[int] = None,
) -> List[Path]:
    """Select FITS files based on path and optional header keyword filters.

    Parameters
    ----------
    files :
        One of:
        - list of FITS file paths,
        - directory path,
        - glob pattern (e.g. ``'*.fits'``).
    keys :
        Optional dictionary of FITS header keyword/value pairs to filter on.
        Values may be:
        - Single value (str, int, float, bool) for exact match.
        - List of values to accept any of the listed values.
        Comparison is case-insensitive for string values.
    recursive :
        If True, search subdirectories when *files* is a directory or
        glob pattern.
    max_files :
        If provided, cap the number of returned files at this value.

    Returns
    -------
    list of paths
        Full paths to FITS files that match the criteria, up to *max_files*.
        Returns an empty list if no files are found.

    Raises
    ------
    ValueError
        If *max_files* is provided but is not a positive integer.
    SelectionError
        If *files* (when a single string) is not a valid FITS file,
        directory, or glob pattern.
    """
    matching_files: List[Path] = []

    if max_files is not None:
        if not isinstance(max_files, int) or max_files <= 0:
            raise ValueError(
                "The 'max_files' parameter must be a positive integer if provided."
            )

    # -------------------------------------------------------------------------
    # Step 1: Build initial iterator of potential files
    # -------------------------------------------------------------------------
    if isinstance(files, list):
        potential_files = files
        logger.debug("Processing explicit list of %d files", len(potential_files))

    elif isinstance(files, (str, Path)):
        path = Path(files)
        files_str = str(files)
        if path.is_dir():
            logger.debug(
                "Processing directory: %s%s",
                files,
                " (recursive)" if recursive else "",
            )
            if recursive:
                potential_files = (
                    path
                    for path in path.rglob("*")
                    if path.is_file() and is_fits_file(path.name)
                )
            else:
                potential_files = (
                    path
                    for path in path.iterdir()
                    if path.is_file() and is_fits_file(path.name)
                )

        elif any(ch in files_str for ch in "*?[]"):
            logger.debug("Processing glob pattern: %s", files_str)
            if recursive:
                potential_files = glob.iglob(files_str, recursive=True)
            else:
                potential_files = glob.iglob(files_str)

        else:
            # Single file path
            if path.is_file() and is_fits_file(str(path)):
                potential_files = [str(path)]
                logger.debug("Processing single file: %s", files)
            else:
                raise SelectionError(
                    f"Input {files!r} is not a valid FITS file, "
                    "directory, or glob pattern."
                )
    else:
        raise SelectionError(
            "Invalid files type. Must be a string or list of strings."
        )

    # -------------------------------------------------------------------------
    # Step 2: Filter by header keywords (if provided) and apply max_files limit
    # -------------------------------------------------------------------------
    files_processed_count = 0

    for file_path in potential_files:
        file_path = Path(file_path)
        # Enforce max_files early
        if max_files is not None and len(matching_files) >= max_files:
            logger.debug(
                "Reached max_files limit (%d). Stopping file processing.", max_files
            )
            break

        if not is_fits_file(file_path):
            logger.debug("Skipping non-FITS file: %s", file_path)
            continue

        try:
            with fits.open(file_path, memmap=True) as hdul:
                header = hdul[0].header  # Primary HDU for filtering

                if keys is None:
                    matching_files.append(file_path)
                    logger.debug("Match (no keyword filter): %s", file_path)
                else:
                    all_keys_match = True

                    for key, value in keys.items():
                        key_upper = key.upper()
                        if key_upper not in header:
                            all_keys_match = False
                            logger.debug(
                                "  No match for %s: keyword %r not found.",
                                file_path,
                                key,
                            )
                            break

                        header_value = _normalize_for_compare(header[key_upper])

                        if isinstance(value, list):
                            expected_list = [_normalize_for_compare(v) for v in value]
                            if header_value not in expected_list:
                                all_keys_match = False
                                logger.debug(
                                    "  No match for %s: value %r for %r "
                                    "not in expected list %r.",
                                    file_path,
                                    header_value,
                                    key,
                                    value,
                                )
                                break
                        else:
                            expected_value = _normalize_for_compare(value)
                            if header_value != expected_value:
                                all_keys_match = False
                                logger.debug(
                                    "  No match for %s: value %r for %r "
                                    "does not match expected %r.",
                                    file_path,
                                    header_value,
                                    key,
                                    value,
                                )
                                break

                    if all_keys_match:
                        matching_files.append(Path(file_path))
                        logger.debug("Match: %s", file_path)
                    else:
                        logger.debug("No match for %s based on keywords.", file_path)

            files_processed_count += 1

        except FileNotFoundError:
            logger.warning(
                "File not found during processing: %s. Skipping.", file_path
            )
            files_processed_count += 1
            continue

        except (OSError, fits.verify.VerifyError) as e:
            logger.warning(
                "Error processing %s: not a valid FITS file or I/O issue: %s. "
                "Skipping.",
                file_path,
                e,
            )
            files_processed_count += 1
            continue

        except Exception as e:
            logger.warning(
                "Unexpected error while processing %s: %s. Skipping.",
                file_path,
                e,
                exc_info=True,
            )
            files_processed_count += 1
            continue

    logger.debug(
        "Found %d matching files out of %d processed.",
        len(matching_files),
        files_processed_count,
    )

    return matching_files


# =============================================================================
# Convenience Functions for Common Frame Types
# =============================================================================

def select_bias_files(
    files: Union[StrOrPath, List[StrOrPath]],
    recursive: bool = False,
    max_files: Optional[int] = None,
) -> List[Path]:
    """Select FITS bias frames (``IMGTYPE='BIAS'``).

    This is a convenience wrapper around :func:`select_files`.
    """
    logger.debug(
        "select_bias_files: calling select_files with keys={%r: %r}",
        FITS_KEY_IMGTYPE,
        FITS_VAL_BIAS_IMGTYPE,
    )
    return select_files(
        files=files,
        keys={FITS_KEY_IMGTYPE: FITS_VAL_BIAS_IMGTYPE},
        recursive=recursive,
        max_files=max_files,
    )


def select_dark_files(
    files: Union[StrOrPath, List[StrOrPath]],
    recursive: bool = False,
    max_files: Optional[int] = None,
) -> List[Path]:
    """Select FITS dark frames (``IMGTYPE='DARK'``)."""
    logger.debug(
        "select_dark_files: calling select_files with keys={%r: %r}",
        FITS_KEY_IMGTYPE,
        FITS_VAL_DARK_IMGTYPE,
    )
    return select_files(
        files=files,
        keys={FITS_KEY_IMGTYPE: FITS_VAL_DARK_IMGTYPE},
        recursive=recursive,
        max_files=max_files,
    )


def select_flat_files(
    files: Union[StrOrPath, List[StrOrPath]],
    recursive: bool = False,
    max_files: Optional[int] = None,
) -> List[Path]:
    """Select FITS flat frames (``IMGTYPE='FLAT'``)."""
    logger.debug(
        "select_flat_files: calling select_files with keys={%r: %r}",
        FITS_KEY_IMGTYPE,
        FITS_VAL_FLAT_IMGTYPE,
    )
    return select_files(
        files=files,
        keys={FITS_KEY_IMGTYPE: FITS_VAL_FLAT_IMGTYPE},
        recursive=recursive,
        max_files=max_files,
    )


def select_light_files(
    files: Union[StrOrPath, List[StrOrPath]],
    recursive: bool = False,
    max_files: Optional[int] = None,
) -> List[Path]:
    """Select FITS light/science frames (``IMGTYPE='LIGHT'``)."""
    logger.debug(
        "select_light_files: calling select_files with keys={%r: %r}",
        FITS_KEY_IMGTYPE,
        FITS_VAL_LIGHT_IMGTYPE,
    )
    return select_files(
        files=files,
        keys={FITS_KEY_IMGTYPE: FITS_VAL_LIGHT_IMGTYPE},
        recursive=recursive,
        max_files=max_files,
    )


__all__ = [
    "select_files",
    "select_bias_files",
    "select_dark_files",
    "select_flat_files",
    "select_light_files",
]
