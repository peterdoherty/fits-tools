#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FITS survey and inventory utilities.

This module scans a directory tree for FITS files and produces CSV
inventories summarizing basic statistics and selected header keywords.

Two modes are supported:

- Single master inventory:
  One CSV file listing all FITS files under ``root_dir``.
- Per-directory inventories:
  One ``inventory.csv`` per directory containing FITS files, mirroring
  the directory tree under a chosen output directory.
"""

import csv
import logging
import math
from pathlib import Path
from typing import List, Optional, Union

from .constants import FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME
from .statistics import stats
from .io import get_header
from .utils import is_fits_file

from .types import StrOrPath

logger = logging.getLogger(__name__)

__all__ = ["survey_fits"]


def _iter_dirs(root: Path, recursive: bool):
    """Yield directories to scan, optionally recursively."""
    yield root
    if not recursive:
        return
    for path in root.rglob("*"):
        if path.is_dir():
            yield path


def _is_under_dir(path: str, parent: str) -> bool:
    """Return True if *path* is inside *parent* (or equal)."""
    try:
        path_abs = Path(path).resolve()
        parent_abs = Path(parent).resolve()
        path_abs.relative_to(parent_abs)
        return True
    except Exception:
        return False


def _format_float(val) -> str:
    """Format a numeric value for CSV output, handling NaN/Inf/None."""
    try:
        if val is None:
            return "N/A"
        x = float(val)
        if math.isnan(x) or math.isinf(x):
            return "N/A"
        return f"{x:.2f}"
    except Exception:
        return "N/A"


def survey_fits(
    root_dir: StrOrPath,
    output_path: Optional[StrOrPath] = None,
    output_dir: Optional[StrOrPath] = None,
    keywords: Optional[List[str]] = None,
    recursive: bool = True,
    extension: int = 0,
    split_by_dir: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Scan directories under *root_dir* and write FITS metadata to CSV.

    Parameters
    ----------
    root_dir :
        Root directory to scan.
    output_path :
        Path for the master CSV when ``split_by_dir=False``. If None, a
        default name (``master_inventory.csv``) is used under *output_dir*
        or *root_dir*.
    output_dir :
        Base directory where reports are written. If None, reports are
        written under ``root_dir/reports`` (if it exists) or directly
        under *root_dir*. When provided, all output is written there,
        and the scan will skip that subtree to avoid self-processing.
    keywords :
        List of header keywords to include as columns. Defaults to
        ``[IMAGETYP, EXPTIME, 'FILTER', 'OBJECT']``.
    recursive :
        If True, scan subdirectories recursively. If False, only scan
        *root_dir*.
    extension :
        HDU index to read statistics and headers from. Default is 0.
    split_by_dir :
        If False, write a single combined CSV (master inventory).
        If True, write one ``inventory.csv`` per directory containing
        FITS files, mirroring the directory tree under *output_dir*.
    overwrite :
        If False and an output CSV already exists, skip generation for
        that inventory instead of overwriting.

    Notes
    -----
    - Uses :func:`fits_tools.statistics.stats` with ``stats_type='simple'``
      to avoid computing percentiles.
    - Uses :func:`fits_tools.io.get_header` for header access.
    """
    if keywords is None:
        keywords = [FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME, "FILTER", "OBJECT"]

    fieldnames = ["filename", "mean", "std", "min", "max", "pixels"] + keywords

    root_dir_path = Path(root_dir).resolve()

    # Determine where outputs go
    if output_dir is not None:
        base_report_dir = Path(output_dir).resolve()
    else:
        reports_dir = root_dir_path / "reports"
        base_report_dir = reports_dir if reports_dir.is_dir() else root_dir_path

    base_report_dir.mkdir(parents=True, exist_ok=True)

    # If output dir is inside scan tree, skip it while walking to avoid self-processing
    skip_output_tree = _is_under_dir(str(base_report_dir), str(root_dir_path))

    # ------------------------------------------------------------------
    # Combined (master) inventory mode
    # ------------------------------------------------------------------
    if not split_by_dir:
        current_output = (
            Path(output_path)
            if output_path
            else base_report_dir / "master_inventory.csv"
        )

        if current_output.exists() and not overwrite:
            logger.info("Skipping; inventory already exists: %s", current_output)
            return

        current_output.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(current_output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for root in _iter_dirs(root_dir_path, recursive=recursive):
                    if skip_output_tree and _is_under_dir(str(root), str(base_report_dir)):
                        continue

                    fits_files = sorted(
                        path.name
                        for path in root.iterdir()
                        if path.is_file() and is_fits_file(path.name)
                    )
                    if not fits_files:
                        continue

                    for filename in fits_files:
                        file_path = root / filename
                        try:
                            res = stats(
                                str(file_path),
                                hdu=extension,
                                stats_type="simple",
                            )
                            if res is None:
                                continue

                            header = get_header(str(file_path), hdu=extension)
                            rel = file_path.relative_to(root_dir_path)

                            row = {
                                "filename": str(rel),
                                "mean": _format_float(res.mean_val),
                                "std": _format_float(res.std_val),
                                "min": _format_float(res.min_val),
                                "max": _format_float(res.max_val),
                                "pixels": res.num_pixels,
                            }
                            for kw in keywords:
                                row[kw] = header.get(kw, "N/A")

                            writer.writerow(row)

                        except Exception as e:
                            logger.error("Error processing %s: %s", file_path, e)

                f.flush()
                logger.info("Inventory saved: %s", current_output)

        except IOError as e:
            logger.error("File I/O Error for %s: %s", current_output, e)

        return

    # ------------------------------------------------------------------
    # Per-directory inventory mode (mirrored tree)
    # ------------------------------------------------------------------
    for root in _iter_dirs(root_dir_path, recursive=recursive):
        if skip_output_tree and _is_under_dir(str(root), str(base_report_dir)):
            continue

        fits_files = sorted(
            path.name
            for path in root.iterdir()
            if path.is_file() and is_fits_file(path.name)
        )
        if not fits_files:
            continue

        rel_dir = root.relative_to(root_dir_path)
        out_dir = (
            base_report_dir if str(rel_dir) == "." else base_report_dir / rel_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        current_output = out_dir / "inventory.csv"

        if current_output.exists() and not overwrite:
            logger.info(
                "Skipping directory; inventory already exists: %s", current_output
            )
            continue

        try:
            with open(current_output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for filename in fits_files:
                    file_path = root / filename
                    try:
                        res = stats(
                            str(file_path),
                            hdu=extension,
                            stats_type="simple",
                        )
                        if res is None:
                            continue

                        header = get_header(str(file_path), hdu=extension)

                        row = {
                            "filename": filename,
                            "mean": _format_float(res.mean_val),
                            "std": _format_float(res.std_val),
                            "min": _format_float(res.min_val),
                            "max": _format_float(res.max_val),
                            "pixels": res.num_pixels,
                        }
                        for kw in keywords:
                            row[kw] = header.get(kw, "N/A")

                        writer.writerow(row)

                    except Exception as e:
                        logger.error("Error processing %s: %s", file_path, e)

                f.flush()
                logger.info("Inventory saved: %s", current_output)

        except IOError as e:
            logger.error("File I/O Error for %s: %s", current_output, e)
