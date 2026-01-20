import os
import csv
import math
import logging
from typing import List, Optional

from .constants import FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME
from .statistics import stats
from .io import get_header

logger = logging.getLogger(__name__)


def _is_under_dir(path: str, parent: str) -> bool:
    """True if path is inside parent (or equal)."""
    try:
        path_abs = os.path.abspath(path)
        parent_abs = os.path.abspath(parent)
        return os.path.commonpath([path_abs, parent_abs]) == parent_abs
    except Exception:
        return False


def _format_float(val) -> str:
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
    root_dir: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,  # NEW: where reports are written
    keywords: Optional[List[str]] = None,
    recursive: bool = True,
    extension: int = 0,
    split_by_dir: bool = False,
    overwrite: bool = False,
    
) -> None:
    """
    Scan directories under root_dir and write FITS metadata to CSV.

    - If split_by_dir=False: writes a single combined CSV (master inventory).
    - If split_by_dir=True: writes one inventory.csv per directory containing FITS files,
      mirroring the directory tree under output_dir.

    If output_dir is provided, all output is written there (and never inside root_dir
    unless you explicitly set output_dir under root_dir).
    """
    if keywords is None:
        keywords = [FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME, "FILTER", "OBJECT"]

    fieldnames = ["filename", "mean", "std", "min", "max", "pixels"] + keywords

    root_dir = os.path.abspath(root_dir)

    # Determine where outputs go
    if output_dir is not None:
        base_report_dir = os.path.abspath(output_dir)
    else:
        reports_dir = os.path.join(root_dir, "reports")
        base_report_dir = reports_dir if os.path.isdir(reports_dir) else root_dir

    os.makedirs(base_report_dir, exist_ok=True)

    # If output dir is inside scan tree, skip it while walking to avoid self-processing
    skip_output_tree = _is_under_dir(base_report_dir, root_dir)

    # --- Combined (master) inventory mode ---
    if not split_by_dir:
        current_output = output_path if output_path else os.path.join(base_report_dir, "master_inventory.csv")

        if os.path.exists(current_output) and not overwrite:
            logger.info(f"Skipping; inventory already exists: {current_output}")
            return

        os.makedirs(os.path.dirname(current_output), exist_ok=True)

        try:
            with open(current_output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for root, dirs, files in os.walk(root_dir):
                    if not recursive:
                        dirs[:] = []

                    if skip_output_tree and _is_under_dir(root, base_report_dir):
                        continue

                    fits_files = sorted([fn for fn in files if fn.lower().endswith((".fits", ".fit", ".fts"))])
                    if not fits_files:
                        continue

                    for filename in fits_files:
                        file_path = os.path.join(root, filename)
                        try:
                            res = stats(file_path, ext=extension, stats_type="simple")
                            if res is None:
                                continue

                            header = get_header(file_path, ext=extension)

                            rel = os.path.relpath(file_path, root_dir)  # unique within master inventory

                            row = {
                                "filename": rel,
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
                            logger.error(f"Error processing {file_path}: {e}")

                f.flush()
                logger.info(f"Inventory saved: {current_output}")

        except IOError as e:
            logger.error(f"File I/O Error for {current_output}: {e}")

        return

    # --- Per-directory inventory mode (mirrored tree) ---
    for root, dirs, files in os.walk(root_dir):
        if not recursive:
            dirs[:] = []

        if skip_output_tree and _is_under_dir(root, base_report_dir):
            continue

        fits_files = sorted([fn for fn in files if fn.lower().endswith((".fits", ".fit", ".fts"))])
        if not fits_files:
            continue

        # Mirror the directory structure under base_report_dir
        rel_dir = os.path.relpath(root, root_dir)  # "." or "bias" or "linearity/..."
        out_dir = base_report_dir if rel_dir == "." else os.path.join(base_report_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        current_output = os.path.join(out_dir, "inventory.csv")

        if os.path.exists(current_output) and not overwrite:
            logger.info(f"Skipping directory; inventory already exists: {current_output}")
            continue

        try:
            with open(current_output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for filename in fits_files:
                    file_path = os.path.join(root, filename)
                    try:
                        res = stats(file_path, ext=extension, stats_type="simple")
                        if res is None:
                            continue

                        header = get_header(file_path, ext=extension)

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
                        logger.error(f"Error processing {file_path}: {e}")

                f.flush()
                logger.info(f"Inventory saved: {current_output}")

        except IOError as e:
            logger.error(f"File I/O Error for {current_output}: {e}")
