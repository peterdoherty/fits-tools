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

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU

# Import from fits_tools package
from .constants import (
    FITS_VAL_PRIMARY,
    FITS_KEY_EXTNAME,
    FITS_KEY_XTENSION,
    FITS_KEY_EXTVER,
    FITS_KEY_EXTLEVEL,
    FITS_KEY_PCOUNT,
    FITS_KEY_GCOUNT,
)

from .exceptions import (
    FileHandlingError,
    InvalidParameterError,
    HDUStructureError,
)

from .utils import safe_write
from .array_utils import validate_indices
from .types import StrOrPath

import logging
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

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
    logger.warning(
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
        logger.warning(
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
) -> List[_BaseHDU]:
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
    list of HDU
        Selected HDUs (may include the PRIMARY HDU if explicitly requested).

    Raises
    ------
    InvalidParameterError
        If any index in ``idx`` is out of range.
    """
    selected = []
    selected_indices = set()
    n = len(hdul)
    
    if idx:
        validate_indices(idx, n)
        for i in idx:
            selected.append(hdul[i])
            selected_indices.add(i)
    
    if names:
        names_l = {nm.lower() for nm in names}
        for i, h in enumerate(hdul):
            if i in selected_indices:  # Skip already selected
                continue
            if i == 0:
                if FITS_VAL_PRIMARY.lower() in names_l:
                    selected.append(h)
                    selected_indices.add(i) 
                continue 
            nm = h.header.get(FITS_KEY_EXTNAME, "")
            if nm and nm.lower() in names_l:
                selected.append(h)
                selected_indices.add(i)
    
    # If nothing specified, select all EXCEPT primary
    if not idx and not names:
        selected = [h for i, h in enumerate(hdul) if i > 0]
        if not selected:
            logger.warning("No extension HDUs found to select (only PRIMARY exists).")
    
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
    source_file: Optional[str],
    source_indices: Optional[Sequence[int]],
    source_names: Optional[Sequence[str]],
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
    - If ``source_file`` is provided, selected HDUs from that file are added:
      * Selection is controlled by ``source_indices`` and/or ``source_names`` via
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
    source_file : str or None
        Path to a source FITS file to copy HDUs from. If None, no HDUs
        are copied from external files.
    source_indices : sequence of int or None
        Indices in the source file to copy.
    source_names : sequence of str or None
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
    if source_file:
        with fits.open(source_file, memmap=True) as src:
            to_add = [h.copy() for h in _select_hdus(src, source_indices, source_names)]
        
        if not to_add:
            raise InvalidParameterError(
                "No HDUs selected to add from source FITS.",
                parameter="source_indices/source_names"
            )
        # Check if trying to add a PrimaryHDU (not at position 0)
        for i, h in enumerate(to_add):
            if isinstance(h, fits.PrimaryHDU):
                logger.warning(
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
        "Provide either order_indices or order_names for HDU reordering.",
        parameter="order_indices/order_names",
    )

# ---------------------------------------------------------------------
# File Level Operations (incomplete)
# ---------------------------------------------------------------------


# =============================================================================
# Public file-level operations
# =============================================================================

def remove_hdus_from_file(
    input_file: StrOrPath,
    remove_indices: Optional[Sequence[int]] = None,
    remove_names: Optional[Sequence[str]] = None,
    output_file: Optional[StrOrPath] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Remove HDUs from a FITS file and write the result.

    Parameters
    ----------
    input_file :
        Input FITS file path.
    remove_indices :
        HDU indices to remove. Indices are validated against the HDU list
        length. May include 0 to remove the existing PRIMARY; a new PRIMARY
        will be created if necessary.
    remove_names :
        EXTNAME values to remove (case-insensitive). PRIMARY (index 0) is
        never removed by name.
    output_file :
        Output file path. If None, writes back to ``input_file``.
    overwrite :
        Whether to allow overwriting an existing output file.
    dry_run :
        If True, do not write anything; a log message is emitted instead.

    Raises
    ------
    FileHandlingError
        If reading or writing the file fails.
    InvalidParameterError
        If indices are out of range.
    HDUStructureError
        If all HDUs would be removed and a valid FITS file cannot be formed.
    """
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileHandlingError(
            f"Input file not found: {input_file}",
            filepath=str(input_file),
        )
    
    output_file = Path(output_file) if output_file else None

    target = output_file or input_file
    remove_indices = remove_indices or []
    remove_names = remove_names or []
    if not remove_indices and not remove_names:
        raise InvalidParameterError(
            "Must provide at least one of remove_indices or remove_names.",
            parameter="remove_indices/remove_names",
        )

    try:
        with fits.open(input_file, memmap=True, mode="readonly") as hdul:
            new_hdul = _op_remove_hdus(
                hdul=hdul,
                remove_indices=remove_indices,
                remove_names=remove_names,
            )
        safe_write(
            new_hdul,
            path=str(target),
            overwrite=overwrite,
            dry_run=dry_run,
            operation="remove-hdus",
        )
    except (InvalidParameterError, HDUStructureError):
        # Propagate structural / parameter errors as-is
        raise
    except Exception as e:
        raise FileHandlingError(
            f"Failed to remove HDUs from {input_file}: {e}",
            filepath=str(input_file),
        ) from e


def add_hdus_to_file(
    input_file: StrOrPath,
    source_file: Optional[StrOrPath] = None,
    source_indices: Optional[Sequence[int]] = None,
    source_names: Optional[Sequence[str]] = None,
    create_blank: int = 0,
    blank_shape: Optional[Tuple[int, int]] = None,
    blank_dtype: Optional[str] = None,
    blank_extname: Optional[Sequence[str]] = None,
    insert_at: Optional[int] = None,
    insert_end: bool = True,
    output_file: Optional[StrOrPath] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Add HDUs to a FITS file (from another file or as blank HDUs) and write the result.

    This function provides two modes of adding HDUs to an existing FITS file:
    
    1. **Copy from another FITS file**: Specify ``source_file`` along with optional
       ``source_indices`` and/or ``source_names`` to select which HDUs to copy. If neither
       index nor name is specified, all extension HDUs (excluding PRIMARY) are copied.
       Any PrimaryHDU from the source file is automatically converted to an ImageHDU
       to maintain FITS standard compliance (only one PRIMARY allowed per file).
    
    2. **Create blank image HDUs**: Specify ``create_blank`` > 0 to create new empty
       ImageHDUs. The shape can be explicitly set via ``blank_shape`` or inferred from
       the first 2D+ image HDU in the target file. Blank HDUs can be assigned EXTNAME
       values via ``blank_extname``.
    
    Both modes can be used simultaneously. The insertion position is controlled by
    ``insert_at`` and ``insert_end``; by default, new HDUs are appended to the end.
    Insertion at position 0 is forbidden to preserve the PRIMARY HDU.
    
    The operation maintains FITS standard compliance by ensuring a valid PrimaryHDU
    remains at index 0 in the output file.

    Parameters
    ----------
    input_file : str or Path
        Path to the input FITS file to which HDUs will be added.
    source_file : str or Path, optional
        Path to a source FITS file from which to copy HDUs. If None, no HDUs
        are copied from external files (but blank HDUs may still be created).
    source_indices : sequence of int, optional
        Specific HDU indices to copy from the source file (``source_file``).
        Indices are validated against the source HDUList length. If None and
        ``source_names`` is also None, all extension HDUs (index > 0) from the
        source are copied.
    source_names : sequence of str, optional
        EXTNAME values to select from the source file (case-insensitive matching).
        The special name matching ``FITS_VAL_PRIMARY`` can be used to explicitly
        request the primary HDU from the source (it will be converted to an
        ImageHDU for insertion). If None and ``source_indices`` is also None, all
        extension HDUs from the source are copied.
    create_blank : int, default 0
        Number of blank (zero-filled) ImageHDUs to create and insert. Must be
        non-negative. If 0, no blank HDUs are created.
    blank_shape : tuple of (int, int), optional
        Shape of blank HDUs as (height, width) or (rows, columns). If None,
        the shape is inferred from the last two axes of the first 2D+ image HDU
        found in the input file. Must be provided or inferable if ``create_blank`` > 0.
    blank_dtype : str, optional
        NumPy dtype string for blank HDU data arrays (e.g., ``'float32'``,
        ``'float64'``, ``'int16'``). Defaults to ``'float32'`` if None.
    blank_extname : sequence of str, optional
        EXTNAME header values to assign to newly-created blank HDUs, one name
        per HDU. If provided, the i-th blank HDU receives the i-th name. Extra
        names are ignored if more names than blank HDUs are provided. If None
        or fewer names than HDUs, some blank HDUs will have no EXTNAME set.
    insert_at : int, optional
        Zero-based index position at which to insert new HDUs. Must be >= 1
        (cannot be 0, as that would displace the PRIMARY HDU). If None,
        ``insert_end`` controls the behavior. Ignored if ``insert_end`` is True.
    insert_end : bool, default True
        If True, all new HDUs are appended to the end of the HDU list,
        regardless of ``insert_at``. If False and ``insert_at`` is provided,
        new HDUs are inserted at that position (and subsequent HDUs are shifted).
    output_file : str or Path, optional
        Path for the output FITS file. If None, the input file (``input_file``)
        is overwritten with the modified HDUList.
    overwrite : bool, default False
        Whether to allow overwriting an existing file at ``output_file`` (or
        ``input_file`` if ``output_file`` is None). If False and the target file
        exists, an error is raised.
    dry_run : bool, default False
        If True, perform all operations but do not write the output file.
        A log message is emitted indicating what would have been written.
        Useful for testing and validation.

    Returns
    -------
    None
        This function writes output to disk and does not return a value.

    Raises
    ------
    FileHandlingError
        If reading the input file, reading the source file (``source_file``),
        or writing the output file fails for any reason (e.g., file not found,
        permission denied, disk full).
    InvalidParameterError
        If no HDUs are selected from the source file (when ``source_file`` is
        provided), or if a blank shape cannot be inferred when ``create_blank`` > 0,
        or if ``insert_at`` is invalid (e.g., negative or out of range), or if
        a requested EXTNAME in ``source_names`` is not found in the source file.
    HDUStructureError
        If insertion at position 0 is requested (which would displace the
        PRIMARY HDU), or if a valid FITS file structure cannot be maintained.

    Examples
    --------
    Append all extension HDUs from another file:
    
    >>> add_hdus_to_file('target.fits', source_file='source.fits')
    
    Copy specific HDUs by index from another file:
    
    >>> add_hdus_to_file('target.fits', source_file='source.fits',
    ...                  source_indices=[1, 3, 5], output_file='result.fits')
    
    Copy HDUs by name:
    
    >>> add_hdus_to_file('target.fits', source_file='source.fits',
    ...                  source_names=['SCI', 'ERR', 'DQ'])
    
    Create three blank HDUs with custom names:
    
    >>> add_hdus_to_file('target.fits', create_blank=3,
    ...                  blank_shape=(1024, 1024), blank_dtype='float32',
    ...                  blank_extname=['NEW1', 'NEW2', 'NEW3'])
    
    Insert blank HDUs at a specific position:
    
    >>> add_hdus_to_file('target.fits', create_blank=2,
    ...                  insert_at=2, insert_end=False,
    ...                  output_file='result.fits', overwrite=True)
    
    Perform a dry run to test without writing:
    
    >>> add_hdus_to_file('target.fits', source_file='source.fits',
    ...                  create_blank=1, dry_run=True)

    Notes
    -----
    - The FITS standard requires a PrimaryHDU at index 0. This function ensures
      that invariant is maintained; any PrimaryHDU from a source file is
      converted to an ImageHDU before insertion.
    - When both ``source_file`` and ``create_blank`` are specified, HDUs from
      the source file are added first, followed by blank HDUs.
    - The ``insert_at`` parameter is validated after each insertion, so
      multiple HDUs inserted at the same position are added sequentially
      (first HDU at position N, second at N+1, etc.).
    - If ``output_file`` is None, the original file is modified in place.
      Consider using ``dry_run=True`` first to verify the operation.

    See Also
    --------
    remove_hdus_from_file : Remove HDUs from a FITS file.
    reorder_hdus_in_file : Reorder HDUs within a FITS file.
    """
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileHandlingError(
            f"Input file not found: {input_file}",
            filepath=str(input_file),
        )
    
    source_file = Path(source_file) if source_file is not None else None
    output_file = Path(output_file) if output_file is not None else None

    # Validate inputs
    if source_file is None and create_blank <= 0:
        raise InvalidParameterError(
            "Must specify either source_file or create_blank > 0",
            parameter="source_file/create_blank"
        )
    
    if source_file is not None and not source_file.exists():
        raise FileHandlingError(
            f"Source file not found: {source_file}",
            filepath=str(source_file),
        )
    
    target = output_file or input_file
    
    logger.debug("Adding HDUs to %s", input_file)
    if source_file is not None:
        logger.debug("Source: %s", source_file)
    if create_blank > 0:
        logger.debug("Creating %d blank HDUs", create_blank)

    try:
        with fits.open(input_file, memmap=True, mode="readonly") as hdul:
            new_hdul = _op_add_hdus(
                hdul=hdul,
                source_file=str(source_file) if source_file is not None else None,
                source_indices=source_indices,
                source_names=source_names,
                create_blank=create_blank,
                blank_shape=blank_shape,
                blank_dtype=blank_dtype,
                blank_extname=blank_extname,
                insert_at=insert_at,
                insert_end=insert_end,
            )
        
        logger.debug("Result: %d HDUs total", len(new_hdul))
        
        safe_write(
            new_hdul,
            path=str(target),
            overwrite=overwrite,
            dry_run=dry_run,
            operation="add-hdus",
        )
    except (InvalidParameterError, HDUStructureError):
        raise
    except OSError as e:
        raise FileHandlingError(
            f"Failed to read/write file {input_file}: {e}",
            filepath=str(input_file),
        ) from e
    except Exception as e:
        logger.exception("Unexpected error adding HDUs to %s", input_file)
        raise FileHandlingError(
            f"Failed to add HDUs to {input_file}: {e}",
            filepath=str(input_file),
        ) from e


def reorder_hdus_in_file(
    input_file: StrOrPath,
    order_indices: Optional[Sequence[int]] = None,
    order_names: Optional[Sequence[str]] = None,
    output_file: Optional[StrOrPath] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Reorder HDUs in a FITS file and write the result.

    Exactly one of order_indices or order_names must be provided.

    Parameters
    ----------
    input_file :
        Input FITS file path.
    order_indices :
        New order of HDUs by index. Index 0 (PRIMARY) must be included
        and must be first; otherwise an HDUStructureError is raised.
    order_names :
        New order of HDUs by EXTNAME (and primary name). Uses
        FITS_VAL_PRIMARY to refer to the primary HDU. Names can be
        repeated to duplicate HDUs. Raises InvalidParameterError if a
        requested name is not found.
    output_file :
        Output file path. If None, writes back to ``input_file``.
    overwrite :
        Whether to allow overwriting an existing output file.
    dry_run :
        If True, do not write anything; a log message is emitted instead.

    Raises
    ------
    FileHandlingError
        If reading or writing the file fails.
    InvalidParameterError
        If neither ordering argument is provided or if a name is not found.
    HDUStructureError
        If index-based reordering violates PRIMARY position rules.
    """
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileHandlingError(
            f"Input file not found: {input_file}",
            filepath=str(input_file),
        )
    
    output_path: Optional[Path] = Path(output_file) if output_file is not None else None

    if order_indices is None and order_names is None:
        raise InvalidParameterError(
            "Must provide either order_indices or order_names for reordering.",
            parameter="order_indices/order_names",
        )
    
    if order_indices is not None and order_names is not None:
        raise InvalidParameterError(
            "Cannot specify both order_indices and order_names; choose one.",
            parameter="order_indices/order_names",
        )
    
    target = output_path or input_file

    try:
        with fits.open(input_file, memmap=True, mode="readonly") as hdul:
            new_hdul = _op_reorder_hdus(
                hdul=hdul,
                order_indices=order_indices,
                order_names=order_names,
            )
        safe_write(
            new_hdul,
            path=str(target),
            overwrite=overwrite,
            dry_run=dry_run,
            operation="reorder-hdus",
        )
    except (InvalidParameterError, HDUStructureError):
        raise
    except Exception as e:
        raise FileHandlingError(
            f"Failed to reorder HDUs in {input_file}: {e}",
            filepath=str(input_file),
        ) from e


__all__ = [
    "remove_hdus_from_file",
    "add_hdus_to_file",
    "reorder_hdus_in_file",
]
