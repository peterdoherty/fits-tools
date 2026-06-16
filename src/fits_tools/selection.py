#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""FITS file selection utilities.

This module provides helpers for selecting FITS files from:

- a single FITS file path
- a directory (optionally recursive)
- a glob pattern
- an explicit list of paths

Files can be filtered by primary-header keyword/value pairs. Convenience
functions are provided for selecting bias, dark, flat, and light frames
based on normalized image type metadata.
"""

from __future__ import annotations

import glob
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from astropy.io import fits

from .constants import (
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
)
from .exceptions import SelectionError
from .image_types import normalize_image_type
from .types import StrOrPath
from .utils import is_fits_file

logger = logging.getLogger(__name__)

__all__ = [
    "select_files",
    "select_bias_files",
    "select_dark_files",
    "select_flat_files",
    "select_light_files",
]


def _normalize_for_compare(value: Any) -> Any:
    """Normalize a header or query value for comparison.

    Strings are uppercased and stripped. Non-strings are returned as-is.
    """
    if isinstance(value, str):
        return value.strip().upper()
    return value


def _iter_candidate_paths(
    files: StrOrPath | list[StrOrPath],
    *,
    recursive: bool,
) -> Iterator[Path]:
    """Yield candidate file paths from a path-like input.

    Parameters
    ----------
    files : str, Path, or list of str or Path
        A FITS file path, directory, glob pattern, or explicit list of
        file paths.
    recursive : bool
        If True, search subdirectories when *files* is a directory or
        glob pattern.

    Yields
    ------
    Path
        Candidate file paths. Existence and FITS validity are not
        guaranteed; callers should filter further.

    Raises
    ------
    SelectionError
        If *files* is not a valid file, directory, glob pattern, or list.
    """
    if isinstance(files, list):
        logger.debug("Processing explicit list of %d path(s).", len(files))
        for path in files:
            p = Path(path)
            if p.is_file():
                yield p
        return

    if not isinstance(files, (str, Path)):
        raise SelectionError(
            "files must be a path, glob pattern, or list of paths."
        )

    path = Path(files)
    files_str = str(files)

    if path.is_dir():
        logger.debug(
            "Processing directory %s%s.",
            path,
            " recursively" if recursive else "",
        )
        iterator = path.rglob("*") if recursive else path.iterdir()
        for candidate in iterator:
            if candidate.is_file():
                yield candidate
        return

    if any(ch in files_str for ch in "*?[]"):
        logger.debug("Processing glob pattern: %s", files_str)
        for candidate in glob.iglob(files_str, recursive=recursive):
            p = Path(candidate)
            if p.is_file():
                yield p
        return

    if path.is_file():
        yield path
        return

    raise SelectionError(
        f"Input {files!r} is not a valid file, directory, or glob pattern."
    )


def _header_matches(header: fits.Header, keys: dict[str, Any]) -> bool:
    """Return True if *header* satisfies all keyword/value filters in *keys*.

    String comparisons are case-insensitive. A filter value may be a
    scalar (exact match) or a collection (membership test).
    """
    for key, expected in keys.items():
        header_key = str(key).upper()
        if header_key not in header:
            return False

        header_value = _normalize_for_compare(header[header_key])

        if isinstance(expected, (list, tuple, set, frozenset)):
            expected_values = {_normalize_for_compare(v) for v in expected}
            if header_value not in expected_values:
                return False
        else:
            if header_value != _normalize_for_compare(expected):
                return False

    return True


def _is_matching_frame_type(
    header: fits.Header,
    expected_type: str,
) -> bool:
    """Return True if the header's image type normalizes to *expected_type*."""
    for key in ("IMGTYPE", "IMAGETYP", "OBSTYPE"):
        raw_value = header.get(key)
        if raw_value is None:
            continue
        if normalize_image_type(raw_value, strict=False) == expected_type:
            return True
    return False


def select_files(
    files: StrOrPath | list[StrOrPath],
    keys: dict[str, Any] | None = None,
    recursive: bool = False,
    max_files: int | None = None,
) -> list[Path]:
    """Select FITS files based on path and optional primary-header filters.

    Parameters
    ----------
    files : str, Path, or list of str or Path
        A FITS file path, directory, glob pattern, or explicit list of
        file paths to search.
    keys : dict, optional
        Mapping of primary-header keyword names to expected values. Only
        files whose primary header satisfies all constraints are returned.
        String comparisons are case-insensitive. A value may be a scalar
        (exact match) or a list/set (membership test).
    recursive : bool, optional
        If True, search subdirectories when *files* is a directory or
        glob pattern. Default is False.
    max_files : int, optional
        Maximum number of matching files to return. If None, all matches
        are returned.

    Returns
    -------
    list of Path
        Matching FITS file paths in the order they were discovered.

    Raises
    ------
    SelectionError
        If *files* is an invalid input type, or if *max_files* is not a
        positive integer.
    """
    if max_files is not None and (
        not isinstance(max_files, int) or max_files <= 0
    ):
        raise SelectionError(
            "max_files must be a positive integer when provided."
        )

    matching: list[Path] = []
    processed = 0

    for file_path in _iter_candidate_paths(files, recursive=recursive):
        processed += 1

        if max_files is not None and len(matching) >= max_files:
            logger.debug("Reached max_files limit (%d).", max_files)
            break

        if not is_fits_file(file_path):
            logger.debug("Skipping non-FITS file: %s", file_path)
            continue

        try:
            with fits.open(file_path, memmap=True, mode="readonly") as hdul:
                header = hdul[0].header
                if keys is None or _header_matches(header, keys):
                    matching.append(file_path)
                    logger.debug("Selected: %s", file_path)
        except FileNotFoundError:
            logger.warning(
                "File disappeared during selection: %s", file_path
            )
        except (OSError, fits.verify.VerifyError) as exc:
            logger.warning(
                "Skipping invalid FITS file %s: %s", file_path, exc
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Unexpected error processing %s: %s",
                file_path,
                exc,
                exc_info=True,
            )

    logger.debug(
        "Selected %d file(s) from %d candidate(s).",
        len(matching),
        processed,
    )
    return matching


def _select_by_frame_type(
    files: StrOrPath | list[StrOrPath],
    expected_type: str,
    *,
    recursive: bool = False,
    max_files: int | None = None,
) -> list[Path]:
    """Select files whose image-type metadata normalizes to *expected_type*."""
    matches: list[Path] = []
    for file_path in _iter_candidate_paths(files, recursive=recursive):
        if max_files is not None and len(matches) >= max_files:
            break
        if not is_fits_file(file_path):
            continue
        try:
            with fits.open(file_path, memmap=True, mode="readonly") as hdul:
                if _is_matching_frame_type(hdul[0].header, expected_type):
                    matches.append(file_path)
        except Exception as exc:
            logger.warning(
                "Skipping %s during frame-type selection: %s",
                file_path,
                exc,
            )
    return matches


def select_bias_files(
    files: StrOrPath | list[StrOrPath],
    recursive: bool = False,
    max_files: int | None = None,
) -> list[Path]:
    """Select FITS bias frames.

    Parameters
    ----------
    files : str, Path, or list of str or Path
        Source path(s) to search.
    recursive : bool, optional
        Search subdirectories. Default is False.
    max_files : int, optional
        Maximum number of files to return.

    Returns
    -------
    list of Path
        Paths to files whose image type normalizes to ``"BIAS"``.
    """
    return _select_by_frame_type(
        files,
        FITS_VAL_BIAS_IMGTYPE,
        recursive=recursive,
        max_files=max_files,
    )


def select_dark_files(
    files: StrOrPath | list[StrOrPath],
    recursive: bool = False,
    max_files: int | None = None,
) -> list[Path]:
    """Select FITS dark frames.

    Parameters
    ----------
    files : str, Path, or list of str or Path
        Source path(s) to search.
    recursive : bool, optional
        Search subdirectories. Default is False.
    max_files : int, optional
        Maximum number of files to return.

    Returns
    -------
    list of Path
        Paths to files whose image type normalizes to ``"DARK"``.
    """
    return _select_by_frame_type(
        files,
        FITS_VAL_DARK_IMGTYPE,
        recursive=recursive,
        max_files=max_files,
    )


def select_flat_files(
    files: StrOrPath | list[StrOrPath],
    recursive: bool = False,
    max_files: int | None = None,
) -> list[Path]:
    """Select FITS flat-field frames.

    Parameters
    ----------
    files : str, Path, or list of str or Path
        Source path(s) to search.
    recursive : bool, optional
        Search subdirectories. Default is False.
    max_files : int, optional
        Maximum number of files to return.

    Returns
    -------
    list of Path
        Paths to files whose image type normalizes to ``"FLAT"``.
    """
    return _select_by_frame_type(
        files,
        FITS_VAL_FLAT_IMGTYPE,
        recursive=recursive,
        max_files=max_files,
    )


def select_light_files(
    files: StrOrPath | list[StrOrPath],
    recursive: bool = False,
    max_files: int | None = None,
) -> list[Path]:
    """Select FITS light/science frames.

    Parameters
    ----------
    files : str, Path, or list of str or Path
        Source path(s) to search.
    recursive : bool, optional
        Search subdirectories. Default is False.
    max_files : int, optional
        Maximum number of files to return.

    Returns
    -------
    list of Path
        Paths to files whose image type normalizes to ``"LIGHT"``.
    """
    return _select_by_frame_type(
        files,
        FITS_VAL_LIGHT_IMGTYPE,
        recursive=recursive,
        max_files=max_files,
    )
