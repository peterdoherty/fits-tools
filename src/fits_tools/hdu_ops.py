#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fits_tools.hdu_ops

Structural FITS editing utilities (HDUList-level operations).

This module contains operations that modify the *structure* of a FITS file:
adding/removing/reordering HDUs, selecting HDUs, and enforcing FITS invariants
(e.g., PRIMARY at index 0).

It is factored out of the original ``fits_tools.edit`` implementation to keep
HDU-level operations separate from pixel-domain edits.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from astropy.io import fits

# Optional WCS support for CRPIX shift on crop
try:
    from astropy.wcs import WCS
except Exception:
    WCS = None

# Import from fits_tools package
from fits_tools.constants import (
    FITS_VAL_PRIMARY,
    FITS_FILE_EXTENSIONS,
    FITS_KEY_EXTNAME,
    FITS_KEY_HISTORY,
    FITS_KEY_XTENSION,
    FITS_KEY_EXTVER,
    FITS_KEY_EXTLEVEL,
    FITS_KEY_PCOUNT,
    FITS_KEY_GCOUNT,
)

from fits_tools.exceptions import (
    InvalidParameterError,
    FileOverwriteError,
    RegionError,
    DimensionalityError,
    HDUStructureError,
    InvalidHDUIndexError,
    DataShapeError,
    FileHandlingError,
    fits_operation_context,
)

from fits_tools.utils import (
    setup_logger,
    validate_indices,
    validate_positive,
    make_output_path,
)


# ---------------------------------------------------------------------
# Module Constants
# ---------------------------------------------------------------------

__version__ = "1.0.0"

# Type aliases for better clarity
ThreshOp = Literal[">", ">=", "<", "<=", "==", "!=", "in", "out"]
FillValue = Union[str, float]


# ---------------------------------------------------------------------
# Utilities & helpers
# ---------------------------------------------------------------------

def _ensure_single_or_dir(inputs: List[str], output_path: Optional[str], output_dir: Optional[str]) -> None:
    """Validate output path specification for single or multiple inputs."""
    if len(inputs) == 1 and output_path:
        return
    if output_dir:
        return
    raise InvalidParameterError(
        "For multiple inputs, provide --output-dir; for a single input, provide --output or --output-dir.",
        parameter="output_path/output_dir"
    )

def _derive_out_path(in_path: str, output_path: Optional[str], output_dir: Optional[str], suffix: str) -> str:
    """Derive output path from input path, output specification, and suffix."""
    if output_dir:
        return str(make_output_path(in_path, output_dir=output_dir, suffix=suffix))
    
    assert output_path is not None
    if os.path.isdir(output_path):
        return os.path.join(output_path, os.path.basename(in_path))
    return output_path

def _safe_write(hdul: fits.HDUList, path: str, overwrite: bool, dry_run: bool, operation: str = "edit") -> None:
    """Safely write HDUList with metadata and validation.

    Notes
    -----
    This function is intentionally conservative:
    - Adds an operation HISTORY line to the primary header (when possible).
    - Writes to a temporary file in the destination directory and atomically replaces
      the destination via :func:`os.replace` to avoid corrupting an existing file if
      the write fails partway through.
    """
    if dry_run:
        logging.info(f"[dry-run] Would write: {path}")
        return

    if os.path.exists(path) and not overwrite:
        raise FileOverwriteError(path)

    # Add HISTORY card to document the operation
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    history_msg = f"fits_tools.edit: {operation} at {timestamp} UTC"
    try:
        if len(hdul) > 0 and getattr(hdul[0], "header", None) is not None:
            hdr = hdul[0].header
            # Prefer explicit append semantics when available
            if hasattr(hdr, "add_history"):
                hdr.add_history(history_msg)
            else:
                hdr[FITS_KEY_HISTORY] = history_msg
    except Exception as e:
        logging.debug(f"Could not add HISTORY card: {e}")

    # Atomic write: write to a temp file in the destination directory, then replace.
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=f".{base}.tmp.", dir=dest_dir)
        os.close(fd)
        # Always overwrite temp path
        hdul.writeto(tmp_path, overwrite=True, checksum=True)
        os.replace(tmp_path, path)
    except Exception as e:
        # Clean up temp file if present; do NOT delete existing destination.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise FileHandlingError(f"Failed to write {path}: {e}", filepath=path) from e

def _ensure_primary_first(hdul: fits.HDUList) -> fits.HDUList:
    """
    Ensure the first HDU is a PrimaryHDU, converting if necessary.
    This maintains FITS standard compliance.
    """
    if len(hdul) == 0:
        raise HDUStructureError("HDUList is empty; cannot create valid FITS file.", issue="empty HDUList")
    
    first_hdu = hdul[0]
    
    # If already PrimaryHDU, we're good
    if isinstance(first_hdu, fits.PrimaryHDU):
        return hdul
    
    # Need to convert first HDU to PrimaryHDU
    logging.warning(
        f"First HDU is {type(first_hdu).__name__}, converting to PrimaryHDU "
        "to maintain FITS standard compliance."
    )
    
    # Convert ImageHDU/CompImageHDU to PrimaryHDU
    if isinstance(first_hdu, (fits.ImageHDU, fits.CompImageHDU)):
        new_primary = fits.PrimaryHDU(data=first_hdu.data, header=first_hdu.header.copy())
        # Remove XTENSION-related keywords that are invalid for PRIMARY
        for kw in [FITS_KEY_XTENSION, FITS_KEY_EXTNAME, FITS_KEY_EXTVER, 
                   FITS_KEY_EXTLEVEL, FITS_KEY_PCOUNT, FITS_KEY_GCOUNT]:
            new_primary.header.remove(kw, ignore_missing=True)
        hdul[0] = new_primary
    else:
        # For other types (BinTableHDU, etc.), create minimal PRIMARY and shift
        logging.warning(
            f"Cannot convert {type(first_hdu).__name__} to PrimaryHDU; "
            "creating minimal PRIMARY HDU at position 0."
        )
        minimal_primary = fits.PrimaryHDU()
        hdul.insert(0, minimal_primary)
    
    return hdul

def _select_hdus(hdul: fits.HDUList,
                 idx: Optional[Sequence[int]] = None,
                 names: Optional[Sequence[str]] = None) -> List[fits.hdu.base.ExtensionHDU]:
    """
    Select HDUs by index or name. 
    Note: Does NOT select PRIMARY by default to avoid duplication.
    """
    selected = []
    n = len(hdul)
    
    if idx:
        validate_indices(idx, n)
        for i in idx:
            selected.append(hdul[i])
    
    if names:
        names_l = {nm.lower() for nm in names}
        for i, h in enumerate(hdul):
            # Skip PRIMARY unless explicitly named
            if i == 0 and FITS_VAL_PRIMARY.lower() not in names_l:
                continue
            nm = h.header.get(FITS_KEY_EXTNAME, "")
            if nm and nm.lower() in names_l:
                selected.append(h)
    
    # If nothing specified, select all EXCEPT primary (to avoid duplication in add-hdus)
    if not idx and not names:
        selected = [h for i, h in enumerate(hdul) if i > 0]
        if not selected:
            logging.warning("No extension HDUs found to select (only PRIMARY exists).")
    
    return selected

def _op_remove_hdus(hdul: fits.HDUList, remove_indices: Sequence[int], remove_names: Sequence[str]) -> fits.HDUList:
    """Remove specified HDUs, ensuring PRIMARY is preserved."""
    n = len(hdul)
    keep = [True] * n
    
    if remove_indices:
        validate_indices(remove_indices, n)
        for i in remove_indices:
            keep[i] = False
    
    if remove_names:
        names_lower = {nm.lower() for nm in remove_names}
        for i, h in enumerate(hdul):
            # Don't allow removal of PRIMARY by name either
            if i == 0:
                continue
            extname = h.header.get(FITS_KEY_EXTNAME, "")
            if extname and extname.lower() in names_lower:
                keep[i] = False
    
    # Build output
    out = fits.HDUList()
    for i, h in enumerate(hdul):
        if keep[i]:
            out.append(h.copy())
    
    if len(out) == 0:
        raise HDUStructureError(
            "All HDUs removed; output would be empty.",
            issue="no HDUs remaining"
        )
    
    # Ensure first HDU is PrimaryHDU
    out = _ensure_primary_first(out)
    
    return out

def _op_add_hdus(hdul: fits.HDUList,
                 from_path: Optional[str],
                 from_index: Optional[Sequence[int]],
                 from_name: Optional[Sequence[str]],
                 create_blank: int,
                 blank_shape: Optional[Tuple[int, int]],
                 blank_dtype: Optional[str],
                 blank_extname: Optional[Sequence[str]],
                 insert_at: Optional[int],
                 insert_end: bool) -> fits.HDUList:
    """Add HDUs from another file or create blank ones. Ensures PRIMARY stays first."""
    # Build output starting with existing HDUs
    out = fits.HDUList()
    for h in hdul:
        out.append(h.copy())

    # Validate insert position doesn't put something before PRIMARY
    if insert_at is not None and insert_at == 0:
        raise HDUStructureError(
            "Cannot insert at position 0 - would displace PRIMARY HDU. "
            "FITS standard requires PrimaryHDU at index 0. Use position 1 or higher.",
            issue="attempted insertion at position 0"
        )

    # Copy from another FITS
    if from_path:
        with fits.open(from_path, memmap=True) as src:
            to_add = [h.copy() for h in _select_hdus(src, from_index, from_name)]
        
        if not to_add:
            raise InvalidParameterError(
                "No HDUs selected to add from source FITS.",
                parameter="from_index/from_name"
            )
        # Check if trying to add a PrimaryHDU (not at position 0)
        for i, h in enumerate(to_add):
            if isinstance(h, fits.PrimaryHDU):
                logging.warning(
                    "Source contains PrimaryHDU; converting to ImageHDU for insertion. "
                    "FITS files can only have one PRIMARY (at index 0)."
                )
                # Convert PrimaryHDU to ImageHDU
                new_hdu = fits.ImageHDU(data=h.data, header=h.header.copy())
                # Copy EXTNAME if it exists, or create one
                if FITS_KEY_EXTNAME not in new_hdu.header:
                    new_hdu.header[FITS_KEY_EXTNAME] = 'IMPORTED'
                to_add[i] = new_hdu
        
        if insert_end or insert_at is None:
            for h in to_add:
                out.append(h)
        else:
            validate_indices([insert_at], len(out) + 1, label="insert position")
            pos = int(insert_at)
            for j, h in enumerate(to_add):
                out.insert(pos + j, h)

    # Create blank HDUs
    if create_blank and create_blank > 0:
        # Determine shape: provided or from the first data HDU in 'out'
        shape = blank_shape
        if shape is None:
            # Find first data HDU with a 2D array
            for h in out:
                if isinstance(h, (fits.ImageHDU, fits.CompImageHDU, fits.PrimaryHDU)) and h.data is not None:
                    if h.data.ndim >= 2:
                        shape = (h.data.shape[-2], h.data.shape[-1])
                        break
        
        if shape is None:
            raise InvalidParameterError(
                "Could not infer shape for blank HDU(s); specify --blank-width/--blank-height.",
                parameter="blank_width/blank_height"
            )

        bdtype = blank_dtype or "float32"
        names = list(blank_extname) if blank_extname else []
        
        for k in range(create_blank):
            arr = np.zeros(shape, dtype=np.dtype(bdtype))
            hdu = fits.ImageHDU(arr)
            if k < len(names):
                hdu.header[FITS_KEY_EXTNAME] = (names[k], "created by fits_tools.edit")
            
            if insert_end or insert_at is None:
                out.append(hdu)
            else:
                validate_indices([insert_at], len(out) + 1, label="insert position")
                out.insert(int(insert_at), hdu)
                insert_at += 1  # subsequent inserts shift

    # Final validation: ensure PRIMARY is first
    out = _ensure_primary_first(out)
    
    return out

def _op_reorder_hdus(hdul: fits.HDUList,
                     order_indices: Optional[Sequence[int]],
                     order_names: Optional[Sequence[str]]) -> fits.HDUList:
    """Reorder HDUs by indices or names. Ensures first HDU is PrimaryHDU."""
    n = len(hdul)
    
    if order_indices:
        validate_indices(order_indices, n)
        
        # Check if index 0 (PRIMARY) is included
        if 0 not in order_indices:
            raise HDUStructureError(
                "Reorder by indices must include index 0 (PRIMARY HDU). "
                "FITS standard requires a PRIMARY HDU at position 0.",
                issue="PRIMARY HDU (index 0) not included in reorder"
            )
        
        # Check if index 0 is first in the order
        if order_indices[0] != 0:
            raise HDUStructureError(
                "Index 0 (PRIMARY HDU) must be first in --order-indices. "
                "FITS standard requires PRIMARY HDU at position 0.",
                issue="PRIMARY HDU not first in reorder sequence"
            )
        
        out = fits.HDUList()
        for i in order_indices:
            out.append(hdul[i].copy())
        
        return out
    
    if order_names:
        # Build name→list map to allow duplicates
        name_map = {}
        for i, h in enumerate(hdul):
            nm = h.header.get(FITS_KEY_EXTNAME, FITS_VAL_PRIMARY if i == 0 else f"EXT{i}")
            name_map.setdefault(nm.lower(), []).append((i, h))
        
        out = fits.HDUList()
        for nm in [s.lower() for s in order_names]:
            if nm not in name_map or not name_map[nm]:
                raise InvalidParameterError(
                    f"Name '{nm}' not found among EXTNAMEs.",
                    parameter="order_names",
                    value=nm
                )
            idx, hdu = name_map[nm].pop(0)
            out.append(hdu.copy())
        
        # Ensure first HDU is PrimaryHDU (convert if necessary)
        out = _ensure_primary_first(out)
        
        return out
    
    raise InvalidParameterError(
        "Provide either --order-indices or --order-names for reorder-hdus.",
        parameter="order_indices/order_names"
    )

