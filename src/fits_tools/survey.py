#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""FITS directory survey and CSV inventory utilities.

This module scans one or more directories for FITS files and writes
per-file metadata and basic statistics to CSV inventory files. It is
intended as a lightweight discovery tool — a quick way to understand
the contents of a data directory without loading full image arrays.

The primary entry point is :func:`survey_fits`, which supports both
single-file and per-directory output modes.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from collections.abc import Iterator

from .constants import FITS_KEY_EXPTIME, FITS_KEY_IMAGETYP
from .io import get_header
from .statistics import stats
from .types import StrOrPath
from .utils import is_fits_file

logger = logging.getLogger(__name__)

__all__ = ["survey_fits"]


def _iter_dirs(root: Path, recursive: bool) -> Iterator[Path]:
    """Yield directories to scan under *root*.

    Parameters
    ----------
    root : Path
        Starting directory.
    recursive : bool
        If True, yield all subdirectories recursively in addition to
        *root*.

    Yields
    ------
    Path
        Directories to scan, starting with *root*.
    """
    yield root
    if not recursive:
        return
    for path in root.rglob("*"):
        if path.is_dir():
            yield path


def _is_under_dir(path: Path, parent: Path) -> bool:
    """Return True if *path* is inside *parent* (or is *parent* itself)."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _format_float(value: object) -> str:
    """Format a numeric value for CSV output, handling NaN, Inf, and None.

    Parameters
    ----------
    value : object
        Value to format.

    Returns
    -------
    str
        Two-decimal float string, or ``"N/A"`` for non-finite or
        non-numeric values.
    """
    try:
        if value is None:
            return "N/A"
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return "N/A"
        return f"{numeric:.2f}"
    except Exception:
        return "N/A"


def _iter_fits_files_in_dir(directory: Path) -> list[Path]:
    """Return sorted FITS files directly inside *directory* (non-recursive).

    Parameters
    ----------
    directory : Path
        Directory to list.

    Returns
    -------
    list of Path
        Sorted FITS file paths.
    """
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and is_fits_file(p)
    )


def _make_row(
    file_path: Path,
    root_dir: Path,
    *,
    hdu: int,
    keywords: list[str],
    relative: bool,
) -> dict[str, object] | None:
    """Build a CSV inventory row for a single FITS file.

    Parameters
    ----------
    file_path : Path
        Path to the FITS file.
    root_dir : Path
        Root directory used to compute relative paths.
    hdu : int
        HDU index from which statistics and header values are read.
    keywords : list of str
        Additional header keywords to include as columns.
    relative : bool
        If True, record the path relative to *root_dir*; otherwise
        record only the filename.

    Returns
    -------
    dict or None
        Row dict suitable for :class:`csv.DictWriter`, or None if the
        file cannot be read.
    """
    result = stats(str(file_path), hdu=hdu, stats_type="simple")
    if result is None:
        return None

    header = get_header(file_path, hdu=hdu)
    row: dict[str, object] = {
        "filename": (
            str(file_path.relative_to(root_dir)) if relative else file_path.name
        ),
        "mean":   _format_float(result.mean_val),
        "std":    _format_float(result.std_val),
        "min":    _format_float(result.min_val),
        "max":    _format_float(result.max_val),
        "pixels": result.num_pixels,
    }
    for keyword in keywords:
        row[keyword] = header.get(keyword, "N/A")
    return row


def survey_fits(
    root_dir: StrOrPath,
    output_path: StrOrPath | None = None,
    output_dir: StrOrPath | None = None,
    keywords: list[str] | None = None,
    recursive: bool = True,
    hdu: int = 0,
    split_by_dir: bool = False,
    overwrite: bool = False,
) -> None:
    """Scan directories under *root_dir* and write FITS metadata to CSV.

    Parameters
    ----------
    root_dir : str or Path
        Root directory to scan for FITS files.
    output_path : str or Path, optional
        Destination path for the master CSV when ``split_by_dir=False``.
        Defaults to ``<output_dir>/master_inventory.csv``.
    output_dir : str or Path, optional
        Base directory for all report output. If None, a ``reports/``
        subdirectory of *root_dir* is used when it exists, otherwise
        *root_dir* itself.
    keywords : list of str, optional
        Additional FITS header keywords to include as CSV columns.
        Defaults to ``[IMAGETYP, EXPTIME, FILTER, OBJECT]``.
    recursive : bool, optional
        If True, scan subdirectories of *root_dir* recursively.
        Default is True.
    hdu : int, optional
        HDU index from which statistics and header values are read.
        Default is 0.
    split_by_dir : bool, optional
        If True, write one ``inventory.csv`` per scanned directory
        rather than a single master file. Default is False.
    overwrite : bool, optional
        If False, skip any inventory file that already exists.
        Default is False.
    """
    if keywords is None:
        keywords = [FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME, "FILTER", "OBJECT"]

    fieldnames = ["filename", "mean", "std", "min", "max", "pixels", *keywords]
    root_dir_path = Path(root_dir).resolve()

    if output_dir is not None:
        base_report_dir = Path(output_dir).resolve()
    else:
        reports_candidate = root_dir_path / "reports"
        base_report_dir = (
            reports_candidate if reports_candidate.is_dir() else root_dir_path
        )

    base_report_dir.mkdir(parents=True, exist_ok=True)
    skip_output_tree = (
        base_report_dir != root_dir_path
        and _is_under_dir(base_report_dir, root_dir_path)
    )

    def _scan_dirs() -> Iterator[Path]:
        for directory in _iter_dirs(root_dir_path, recursive=recursive):
            if skip_output_tree and _is_under_dir(directory, base_report_dir):
                continue
            yield directory

    if not split_by_dir:
        target = (
            Path(output_path).resolve()
            if output_path is not None
            else base_report_dir / "master_inventory.csv"
        )
        if target.exists() and not overwrite:
            logger.info("Skipping existing inventory: %s", target)
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for directory in _scan_dirs():
                for file_path in _iter_fits_files_in_dir(directory):
                    try:
                        row = _make_row(
                            file_path,
                            root_dir_path,
                            hdu=hdu,
                            keywords=keywords,
                            relative=True,
                        )
                        if row is not None:
                            writer.writerow(row)
                    except Exception as exc:
                        logger.error(
                            "Error processing %s: %s", file_path, exc
                        )
        logger.info("Inventory saved: %s", target)
        return

    # split_by_dir: one inventory.csv per directory
    for directory in _scan_dirs():
        fits_files = _iter_fits_files_in_dir(directory)
        if not fits_files:
            continue

        rel_dir = directory.relative_to(root_dir_path)
        out_dir = (
            base_report_dir
            if rel_dir == Path(".")
            else base_report_dir / rel_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "inventory.csv"

        if target.exists() and not overwrite:
            logger.info("Skipping existing inventory: %s", target)
            continue

        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for file_path in fits_files:
                try:
                    row = _make_row(
                        file_path,
                        root_dir_path,
                        hdu=hdu,
                        keywords=keywords,
                        relative=False,
                    )
                    if row is not None:
                        writer.writerow(row)
                except Exception as exc:
                    logger.error(
                        "Error processing %s: %s", file_path, exc
                    )
        logger.info("Inventory saved: %s", target)
