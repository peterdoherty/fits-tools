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

import logging
import tempfile
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
    FITS_KEY_EXTNAME,
    FITS_KEY_XTENSION,
    FITS_KEY_EXTVER,
    FITS_KEY_EXTLEVEL,
    FITS_KEY_PCOUNT,
    FITS_KEY_GCOUNT,
)

from fits_tools.exceptions import (
    InvalidParameterError,
    HDUStructureError,
)

from fits_tools.utils import (
    validate_indices,
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

def _ensure_primary_first(hdul: fits.HDUList) -> fits.HDUList:
    """
    Ensure that the first HDU in an HDUList is a valid PrimaryHDU.

    FITS standard requires a Primary HDU at index 0. This helper enforces
    that invariant by:

    - Leaving the HDUList unchanged if the first HDU is already a PrimaryHDU.
    - Converting an ImageHDU or CompImageHDU at index 0 into a PrimaryHDU,
      removing extension-only keywords (XTENSION, EXTNAME, etc.) that are
      invalid for a PRIMARY.
    - For other types (e.g. BinTableHDU) at index 0, inserting a minimal
      empty PrimaryHDU at the front of the list.

    Parameters
    ----------
    hdul : fits.HDUList
        Original HDU list (modified in-place and also returned).

    Returns
    -------
    fits.HDUList
        The same HDU list object with a PrimaryHDU guaranteed at index 0.

    Raises
    ------
    HDUStructureError
        If the HDUList is empty and a valid FITS file cannot be formed.
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


def _select_hdus(
    hdul: fits.HDUList,
    idx: Optional[Sequence[int]] = None,
    names: Optional[Sequence[str]] = None
) -> List[fits.hdu.base.ExtensionHDU]:
    """
    Select extension HDUs by index and/or EXTNAME, excluding PRIMARY by default.

    This helper is used when copying HDUs from one file into another.
    Selection rules:

    - If ``idx`` is provided, those integer indices are validated and selected.
    - If ``names`` is provided, HDUs with matching EXTNAME values are selected
      (case-insensitive). The PRIMARY HDU (index 0) is *not* selected unless
      the special name ``'PRIMARY'`` (as defined by ``FITS_VAL_PRIMARY``) is
      explicitly included in ``names``.
    - If neither ``idx`` nor ``names`` is provided, all extension HDUs
      (indices > 0) are selected.

    PRIMARY is excluded by default to avoid accidental duplication when adding
    HDUs from one file into another.

    Parameters
    ----------
    hdul : fits.HDUList
        Source HDU list.
    idx : sequence of int, optional
        HDU indices to select.
    names : sequence of str, optional
        EXTNAME values to select (case-insensitive). The special value
        matching ``FITS_VAL_PRIMARY`` is used to explicitly request the
        primary HDU.

    Returns
    -------
    list of ExtensionHDU
        Selected HDUs (references to the original objects; typically callers
        copy them).

    Raises
    ------
    InvalidParameterError
        If any index in ``idx`` is out of range.
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


def _op_remove_hdus(
    hdul: fits.HDUList,
    remove_indices: Sequence[int],
    remove_names: Sequence[str]
) -> fits.HDUList:
    """
    Remove selected HDUs from an HDUList, preserving a valid PRIMARY HDU.

    Removal criteria:

    - Any indices in ``remove_indices`` are removed (after validation).
    - Any HDUs whose EXTNAME matches an entry in ``remove_names`` (case-
      insensitive) are removed, except the PRIMARY HDU at index 0, which
      is never removed by name.

    At least one HDU must remain; otherwise a :class:`HDUStructureError` is
    raised. After removal, the resulting HDUList is passed through
    :func:`_ensure_primary_first` to guarantee that a PrimaryHDU remains at
    index 0.

    Parameters
    ----------
    hdul : fits.HDUList
        Original HDU list.
    remove_indices : sequence of int
        HDU indices to remove.
    remove_names : sequence of str
        EXTNAME values to remove (case-insensitive), excluding PRIMARY.

    Returns
    -------
    fits.HDUList
        New HDU list with the requested HDUs removed.

    Raises
    ------
    InvalidParameterError
        If any index in ``remove_indices`` is out of range.
    HDUStructureError
        If all HDUs would be removed or no valid PRIMARY could be preserved.
    """
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


def _op_add_hdus(
    hdul: fits.HDUList,
    from_path: Optional[str],
    from_index: Optional[Sequence[int]],
    from_name: Optional[Sequence[str]],
    create_blank: int,
    blank_shape: Optional[Tuple[int, int]],
    blank_dtype: Optional[str],
    blank_extname: Optional[Sequence[str]],
    insert_at: Optional[int],
    insert_end: bool
) -> fits.HDUList:
    """
    Add HDUs to an HDUList, either by copying from another FITS file or
    by creating blank image extensions.

    The operation is conservative with respect to the FITS standard:

    - The existing HDUs from ``hdul`` are copied into a new HDUList.
    - If ``from_path`` is provided, selected HDUs from that file are added:
      * Selection is controlled by ``from_index`` and/or ``from_name`` via
        :func:`_select_hdus`.
      * Any PrimaryHDU from the source is converted into an ImageHDU before
        insertion, to avoid multiple PRIMARY HDUs.
    - If ``create_blank`` > 0, that many blank ImageHDUs are created, using:
      * ``blank_shape`` if provided, or
      * the last-two-axis shape of the first 2D+ image HDU in the current
        list (if available).
      The new HDUs may be given EXTNAME values from ``blank_extname``.

    Insertion position:

    - If ``insert_end`` is True or ``insert_at`` is None, new HDUs are
      appended to the end.
    - Otherwise, new HDUs are inserted at position ``insert_at`` or
      shifted forward from that index. Insertion at position 0 is
      forbidden, as it would displace the PRIMARY HDU.

    Parameters
    ----------
    hdul : fits.HDUList
        Original HDU list.
    from_path : str or None
        Path to a source FITS file to copy HDUs from. If None, no HDUs
        are copied from external files.
    from_index : sequence of int or None
        Indices in the source file to copy.
    from_name : sequence of str or None
        EXTNAME values in the source file to copy.
    create_blank : int
        Number of blank image HDUs to create and insert.
    blank_shape : (int, int) or None
        Shape of blank HDUs (height, width). If None, inferred from the
        first 2D+ image HDU in the current list, if possible.
    blank_dtype : str or None
        NumPy dtype string for blank HDUs (e.g. ``'float32'``). Defaults
        to ``'float32'`` if None.
    blank_extname : sequence of str or None
        EXTNAMEs to assign to newly-created blank HDUs, one per HDU.
        Extra names are ignored if fewer blanks are created.
    insert_at : int or None
        Index at which to insert new HDUs. Must not be 0. If None, new
        HDUs are appended to the end.
    insert_end : bool
        If True, always append new HDUs to the end (overrides
        ``insert_at``).

    Returns
    -------
    fits.HDUList
        New HDU list with the added HDUs.

    Raises
    ------
    HDUStructureError
        If insertion at index 0 is requested, which would displace PRIMARY.
    InvalidParameterError
        If no HDUs are selected from the source file, or if a blank shape
        cannot be inferred when ``create_blank`` > 0.
    """
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

def _op_reorder_hdus(
    hdul: fits.HDUList,
    order_indices: Optional[Sequence[int]],
    order_names: Optional[Sequence[str]]
) -> fits.HDUList:
    """
    Reorder HDUs in an HDUList by indices or by names, preserving a valid PRIMARY.

    Two mutually-exclusive modes are supported:

    1. **Index-based reorder** (``order_indices``):
       - The new order is taken from the given index sequence.
       - Index 0 (PRIMARY) **must** be present and must be the first element
         of the sequence, otherwise an :class:`HDUStructureError` is raised.
       - The resulting HDUList is exactly the indices in the given order.

    2. **Name-based reorder** (``order_names``):
       - HDUs are looked up by EXTNAME, case-insensitive. The special name
         matching ``FITS_VAL_PRIMARY`` is used to refer to the primary HDU.
       - Names can be repeated to duplicate HDUs in the output order.
       - If a requested name is not found, an :class:`InvalidParameterError`
         is raised.
       - After assembly, :func:`_ensure_primary_first` is used to guarantee
         that index 0 is a PrimaryHDU (converting or inserting one if needed).

    Exactly one of ``order_indices`` or ``order_names`` must be provided.

    Parameters
    ----------
    hdul : fits.HDUList
        Original HDU list.
    order_indices : sequence of int or None
        New order of HDUs by index.
    order_names : sequence of str or None
        New order of HDUs by EXTNAME (and primary name).

    Returns
    -------
    fits.HDUList
        New HDU list with HDUs in the requested order.

    Raises
    ------
    HDUStructureError
        If index-based reordering does not include index 0 as the first
        element.
    InvalidParameterError
        If a requested EXTNAME is not found or if neither ordering
        argument is provided.
    """
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

