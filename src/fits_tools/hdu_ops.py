#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Structural HDU-level FITS editing utilities.

This module operates on FITS structure rather than pixel values: selecting,
adding, removing, and reordering HDUs while preserving the FITS requirement
that a valid primary HDU remain at index 0.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from .array_validation import validate_indices
from .constants import (
    FITS_KEY_EXTLEVEL,
    FITS_KEY_EXTNAME,
    FITS_KEY_EXTVER,
    FITS_KEY_GCOUNT,
    FITS_KEY_PCOUNT,
    FITS_KEY_XTENSION,
    FITS_VAL_PRIMARY,
)
from .exceptions import FileHandlingError, HDUStructureError, InvalidParameterError
from .io import safe_write
from .types import StrOrPath

logger = logging.getLogger(__name__)

__all__ = [
    "add_hdus_to_file",
    "remove_hdus_from_file",
    "reorder_hdus_in_file",
]

_PRIMARY_ONLY_KEYWORDS = (
    FITS_KEY_XTENSION,
    FITS_KEY_EXTNAME,
    FITS_KEY_EXTVER,
    FITS_KEY_EXTLEVEL,
    FITS_KEY_PCOUNT,
    FITS_KEY_GCOUNT,
)


def _ensure_primary_first(hdul: fits.HDUList) -> fits.HDUList:
    """Return *hdul* with a valid primary HDU at index 0.

    If the first HDU is an image extension it is converted to a primary HDU.
    If the first HDU is not image-like, a minimal empty primary HDU is
    inserted at index 0.
    """
    if len(hdul) == 0:
        raise HDUStructureError(
            "HDUList is empty; cannot construct a valid FITS file.",
            issue="empty_hdulist",
        )

    first_hdu = hdul[0]
    if isinstance(first_hdu, fits.PrimaryHDU):
        return hdul

    if isinstance(first_hdu, (fits.ImageHDU, fits.CompImageHDU)):
        logger.warning(
            "First HDU is %s; converting to PrimaryHDU to preserve FITS compliance.",
            type(first_hdu).__name__,
        )
        primary = fits.PrimaryHDU(data=first_hdu.data, header=first_hdu.header.copy())
        for keyword in _PRIMARY_ONLY_KEYWORDS:
            primary.header.remove(keyword, ignore_missing=True)
        hdul[0] = primary
        return hdul

    logger.warning(
        "First HDU is %s and cannot be converted to PrimaryHDU; "
        "inserting a minimal PrimaryHDU.",
        type(first_hdu).__name__,
    )
    return fits.HDUList([fits.PrimaryHDU()] + [hdu.copy() for hdu in hdul])


def _select_hdus(
    hdul: fits.HDUList,
    indices: list[int] | tuple[int, ...] | None = None,
    names: list[str] | tuple[str, ...] | None = None,
) -> list[fits.hdu.base._BaseHDU]:
    """Select HDUs by index and/or EXTNAME.

    The primary HDU (index 0) is excluded by default to prevent accidental
    duplication when importing HDUs from another file. It may be selected
    explicitly by name using ``PRIMARY``.
    """
    selected: list[fits.hdu.base._BaseHDU] = []
    selected_indices: set[int] = set()
    hdu_count = len(hdul)

    if indices:
        validate_indices(indices, hdu_count, label="hdu index")
        for index in indices:
            selected.append(hdul[index])
            selected_indices.add(index)

    if names:
        normalized_names = {
            str(name).strip().lower() for name in names if str(name).strip()
        }
        for index, hdu in enumerate(hdul):
            if index in selected_indices:
                continue
            if index == 0:
                if FITS_VAL_PRIMARY.lower() in normalized_names:
                    selected.append(hdu)
                    selected_indices.add(index)
                continue
            extname = str(hdu.header.get(FITS_KEY_EXTNAME, "")).strip().lower()
            if extname and extname in normalized_names:
                selected.append(hdu)
                selected_indices.add(index)

    if not indices and not names:
        selected = [hdu for index, hdu in enumerate(hdul) if index > 0]
        if not selected:
            logger.warning("No extension HDUs were available for selection.")

    return selected


def _op_remove_hdus(
    hdul: fits.HDUList,
    remove_indices: list[int] | tuple[int, ...],
    remove_names: list[str] | tuple[str, ...],
) -> fits.HDUList:
    """Return a copy of *hdul* with selected HDUs removed."""
    hdu_count = len(hdul)
    keep = [True] * hdu_count

    if remove_indices:
        validate_indices(remove_indices, hdu_count, label="hdu index")
        for index in remove_indices:
            keep[index] = False

    if remove_names:
        normalized_names = {
            str(name).strip().lower() for name in remove_names if str(name).strip()
        }
        for index, hdu in enumerate(hdul):
            if index == 0:
                continue
            extname = str(hdu.header.get(FITS_KEY_EXTNAME, "")).strip().lower()
            if extname and extname in normalized_names:
                keep[index] = False

    output = fits.HDUList(
        [hdu.copy() for index, hdu in enumerate(hdul) if keep[index]]
    )
    if len(output) == 0:
        raise HDUStructureError(
            "All HDUs would be removed; output FITS file would be empty.",
            issue="no_hdus_remaining",
        )
    return _ensure_primary_first(output)


def _infer_blank_shape(hdul: fits.HDUList) -> tuple[int, int] | None:
    """Infer a 2-D blank-extension shape from the first image HDU with data."""
    for hdu in hdul:
        if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
            continue
        if hdu.data is None or getattr(hdu.data, "ndim", 0) < 2:
            continue
        return (int(hdu.data.shape[-2]), int(hdu.data.shape[-1]))
    return None


def _convert_primary_for_insertion(hdu: fits.PrimaryHDU) -> fits.ImageHDU:
    """Convert a primary HDU into an image extension suitable for insertion."""
    image_hdu = fits.ImageHDU(data=hdu.data, header=hdu.header.copy())
    image_hdu.header.setdefault(FITS_KEY_EXTNAME, "IMPORTED")
    return image_hdu


def _op_add_hdus(
    hdul: fits.HDUList,
    source_file: str | None,
    source_indices: list[int] | tuple[int, ...] | None,
    source_names: list[str] | tuple[str, ...] | None,
    create_blank: int,
    blank_shape: tuple[int, int] | None,
    blank_dtype: str | None,
    blank_extname: list[str] | tuple[str, ...] | None,
    insert_at: int | None,
    insert_end: bool,
) -> fits.HDUList:
    """Return a copy of *hdul* with additional HDUs inserted."""
    output = fits.HDUList([hdu.copy() for hdu in hdul])

    if insert_at == 0:
        raise HDUStructureError(
            "Cannot insert at index 0; the primary HDU must remain first.",
            issue="insert_at_zero",
        )

    if source_file is not None:
        with fits.open(source_file, memmap=True, mode="readonly") as source_hdul:
            to_add = [
                hdu.copy()
                for hdu in _select_hdus(source_hdul, source_indices, source_names)
            ]

        if not to_add:
            raise InvalidParameterError(
                "No HDUs were selected from the source FITS file.",
                parameter="source_indices/source_names",
            )

        normalized_to_add: list[fits.hdu.base._BaseHDU] = []
        for hdu in to_add:
            if isinstance(hdu, fits.PrimaryHDU):
                logger.warning(
                    "Converting imported PrimaryHDU to ImageHDU before insertion."
                )
                normalized_to_add.append(_convert_primary_for_insertion(hdu))
            else:
                normalized_to_add.append(hdu)

        if insert_end or insert_at is None:
            for hdu in normalized_to_add:
                output.append(hdu)
        else:
            validate_indices([insert_at], len(output) + 1, label="insert position")
            for offset, hdu in enumerate(normalized_to_add):
                output.insert(insert_at + offset, hdu)

    if create_blank > 0:
        shape = blank_shape or _infer_blank_shape(output)
        if shape is None:
            raise InvalidParameterError(
                "Could not infer blank HDU shape; specify blank_shape explicitly.",
                parameter="blank_shape",
            )

        dtype = np.dtype(blank_dtype or "float32")
        extnames = list(blank_extname or [])
        for index in range(create_blank):
            image_hdu = fits.ImageHDU(np.zeros(shape, dtype=dtype))
            if index < len(extnames):
                image_hdu.header[FITS_KEY_EXTNAME] = (
                    extnames[index],
                    "created by fits_tools",
                )
            if insert_end or insert_at is None:
                output.append(image_hdu)
            else:
                validate_indices([insert_at], len(output) + 1, label="insert position")
                output.insert(insert_at, image_hdu)
                insert_at += 1

    return _ensure_primary_first(output)


def _op_reorder_hdus(
    hdul: fits.HDUList,
    order_indices: list[int] | tuple[int, ...] | None,
    order_names: list[str] | tuple[str, ...] | None,
) -> fits.HDUList:
    """Return a reordered copy of *hdul*."""
    hdu_count = len(hdul)

    if order_indices:
        validate_indices(order_indices, hdu_count, label="hdu index")
        if 0 not in order_indices:
            raise HDUStructureError(
                "Index-based reordering must include HDU 0 (the primary HDU).",
                issue="missing_primary_in_reorder",
            )
        if order_indices[0] != 0:
            raise HDUStructureError(
                "HDU 0 (the primary HDU) must remain first in index-based reordering.",
                issue="primary_not_first",
            )
        return fits.HDUList([hdul[index].copy() for index in order_indices])

    if order_names:
        name_map: dict[str, list[fits.hdu.base._BaseHDU]] = {}
        for index, hdu in enumerate(hdul):
            name = hdu.header.get(
                FITS_KEY_EXTNAME,
                FITS_VAL_PRIMARY if index == 0 else f"HDU_{index}",
            )
            key = str(name).strip().lower()
            name_map.setdefault(key, []).append(hdu)

        reordered = fits.HDUList()
        for name in (entry.strip().lower() for entry in order_names):
            if name not in name_map or not name_map[name]:
                raise InvalidParameterError(
                    f"EXTNAME {name!r} was not found for reordering.",
                    parameter="order_names",
                    value=name,
                )
            reordered.append(name_map[name].pop(0).copy())

        return _ensure_primary_first(reordered)

    raise InvalidParameterError(
        "Provide exactly one of order_indices or order_names.",
        parameter="order_indices/order_names",
    )


def remove_hdus_from_file(
    input_file: StrOrPath,
    remove_indices: list[int] | tuple[int, ...] | None = None,
    remove_names: list[str] | tuple[str, ...] | None = None,
    output_file: StrOrPath | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Remove selected HDUs from a FITS file and write the result.

    Parameters
    ----------
    input_file : str or Path
        Path to the input FITS file.
    remove_indices : list or tuple of int, optional
        Zero-based indices of HDUs to remove.
    remove_names : list or tuple of str, optional
        EXTNAME values of HDUs to remove (case-insensitive).
    output_file : str or Path, optional
        Destination path. If None, the input file is overwritten in place.
    overwrite : bool, optional
        If True, allow overwriting an existing output file. Default is False.
    dry_run : bool, optional
        If True, perform all validation but do not write output. Default is
        False.

    Raises
    ------
    FileHandlingError
        If the input file does not exist or cannot be read or written.
    InvalidParameterError
        If neither *remove_indices* nor *remove_names* is provided.
    HDUStructureError
        If removing the selected HDUs would leave the file empty.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileHandlingError(f"Input file not found: {input_path}", filepath=input_path)

    remove_indices = remove_indices or []
    remove_names = remove_names or []
    if not remove_indices and not remove_names:
        raise InvalidParameterError(
            "At least one of remove_indices or remove_names must be provided.",
            parameter="remove_indices/remove_names",
        )

    target = Path(output_file) if output_file is not None else input_path

    try:
        with fits.open(input_path, memmap=True, mode="readonly") as hdul:
            output_hdul = _op_remove_hdus(hdul, remove_indices, remove_names)
        safe_write(
            output_hdul,
            path=target,
            overwrite=overwrite,
            dry_run=dry_run,
            operation="remove-hdus",
        )
    except (FileHandlingError, InvalidParameterError, HDUStructureError):
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected failure while removing HDUs from %s", input_path)
        raise FileHandlingError(
            f"Failed to remove HDUs from {input_path}: {exc}",
            filepath=input_path,
        ) from exc


def add_hdus_to_file(
    input_file: StrOrPath,
    source_file: StrOrPath | None = None,
    source_indices: list[int] | tuple[int, ...] | None = None,
    source_names: list[str] | tuple[str, ...] | None = None,
    create_blank: int = 0,
    blank_shape: tuple[int, int] | None = None,
    blank_dtype: str | None = None,
    blank_extname: list[str] | tuple[str, ...] | None = None,
    insert_at: int | None = None,
    insert_end: bool = True,
    output_file: StrOrPath | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Add HDUs to a FITS file from a source file or as blank extensions.

    Parameters
    ----------
    input_file : str or Path
        Path to the input FITS file.
    source_file : str or Path, optional
        Path to a second FITS file from which HDUs are copied.
    source_indices : list or tuple of int, optional
        HDU indices to copy from *source_file*.
    source_names : list or tuple of str, optional
        EXTNAME values to copy from *source_file* (case-insensitive).
    create_blank : int, optional
        Number of blank image extensions to create. Default is 0.
    blank_shape : tuple of int, optional
        Shape ``(ny, nx)`` for blank extensions. Inferred from existing
        image data if not provided.
    blank_dtype : str, optional
        NumPy dtype string for blank extensions. Default is ``"float32"``.
    blank_extname : list or tuple of str, optional
        EXTNAME values to assign to the created blank extensions.
    insert_at : int, optional
        Index at which to insert new HDUs. Cannot be 0. If None and
        *insert_end* is False, HDUs are appended.
    insert_end : bool, optional
        If True (default), append new HDUs at the end of the HDU list.
    output_file : str or Path, optional
        Destination path. If None, the input file is overwritten in place.
    overwrite : bool, optional
        If True, allow overwriting an existing output file. Default is False.
    dry_run : bool, optional
        If True, perform all validation but do not write output. Default is
        False.

    Raises
    ------
    FileHandlingError
        If the input or source file does not exist or cannot be read or
        written.
    InvalidParameterError
        If neither *source_file* nor *create_blank* > 0 is specified, or
        no HDUs are selected from the source file.
    HDUStructureError
        If *insert_at* is 0 or the resulting HDU list violates FITS
        structure requirements.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileHandlingError(f"Input file not found: {input_path}", filepath=input_path)

    source_path = Path(source_file) if source_file is not None else None
    if source_path is None and create_blank <= 0:
        raise InvalidParameterError(
            "Specify source_file, create_blank > 0, or both.",
            parameter="source_file/create_blank",
        )
    if source_path is not None and not source_path.exists():
        raise FileHandlingError(
            f"Source file not found: {source_path}", filepath=source_path
        )

    target = Path(output_file) if output_file is not None else input_path

    try:
        with fits.open(input_path, memmap=True, mode="readonly") as hdul:
            output_hdul = _op_add_hdus(
                hdul=hdul,
                source_file=str(source_path) if source_path is not None else None,
                source_indices=source_indices,
                source_names=source_names,
                create_blank=create_blank,
                blank_shape=blank_shape,
                blank_dtype=blank_dtype,
                blank_extname=blank_extname,
                insert_at=insert_at,
                insert_end=insert_end,
            )
        safe_write(
            output_hdul,
            path=target,
            overwrite=overwrite,
            dry_run=dry_run,
            operation="add-hdus",
        )
    except (FileHandlingError, InvalidParameterError, HDUStructureError):
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected failure while adding HDUs to %s", input_path)
        raise FileHandlingError(
            f"Failed to add HDUs to {input_path}: {exc}",
            filepath=input_path,
        ) from exc


def reorder_hdus_in_file(
    input_file: StrOrPath,
    order_indices: list[int] | tuple[int, ...] | None = None,
    order_names: list[str] | tuple[str, ...] | None = None,
    output_file: StrOrPath | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Reorder HDUs in a FITS file and write the result.

    Parameters
    ----------
    input_file : str or Path
        Path to the input FITS file.
    order_indices : list or tuple of int, optional
        Complete ordered sequence of HDU indices. Must include index 0,
        which must remain first.
    order_names : list or tuple of str, optional
        Complete ordered sequence of EXTNAME values (case-insensitive).
        Provide this or *order_indices*, not both.
    output_file : str or Path, optional
        Destination path. If None, the input file is overwritten in place.
    overwrite : bool, optional
        If True, allow overwriting an existing output file. Default is False.
    dry_run : bool, optional
        If True, perform all validation but do not write output. Default is
        False.

    Raises
    ------
    FileHandlingError
        If the input file does not exist or cannot be read or written.
    InvalidParameterError
        If neither or both of *order_indices* and *order_names* are provided,
        or if a specified EXTNAME is not found.
    HDUStructureError
        If the primary HDU (index 0) is absent or not first in the new order.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileHandlingError(f"Input file not found: {input_path}", filepath=input_path)

    if order_indices is None and order_names is None:
        raise InvalidParameterError(
            "Provide one of order_indices or order_names.",
            parameter="order_indices/order_names",
        )
    if order_indices is not None and order_names is not None:
        raise InvalidParameterError(
            "Provide only one of order_indices or order_names, not both.",
            parameter="order_indices/order_names",
        )

    target = Path(output_file) if output_file is not None else input_path

    try:
        with fits.open(input_path, memmap=True, mode="readonly") as hdul:
            output_hdul = _op_reorder_hdus(hdul, order_indices, order_names)
        safe_write(
            output_hdul,
            path=target,
            overwrite=overwrite,
            dry_run=dry_run,
            operation="reorder-hdus",
        )
    except (FileHandlingError, InvalidParameterError, HDUStructureError):
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected failure while reordering HDUs in %s", input_path)
        raise FileHandlingError(
            f"Failed to reorder HDUs in {input_path}: {exc}",
            filepath=input_path,
        ) from exc
