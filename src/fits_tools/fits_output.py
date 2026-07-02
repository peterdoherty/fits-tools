#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Shared output-disposition policy for FITS processing results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits

from .exceptions import ImageProcessingError
from .hdu import HDUIdentity, HDUSelector, get_hdu_identity, resolve_hdu_index
from .io import transactional_write_hdulist
from .types import StrOrPath

OutputMode = Literal["sef", "preserve", "inplace"]
StructurePolicy = Literal["target", "compatible", "strict"]


@dataclass(slots=True)
class ResolvedHDU:
    """Detached image data and identity resolved from a FITS file."""

    path: Path
    index: int
    identity: HDUIdentity
    data: np.ndarray
    header: fits.Header


def identity_selector(resolved: ResolvedHDU) -> HDUSelector:
    """Return the most stable exact selector available for *resolved*."""
    identity = resolved.identity
    if identity.extname is not None and identity.extver is not None:
        return (identity.extname, identity.extver)
    if identity.extname is not None:
        return identity.extname
    return identity.index


def _validate_destination_structure(
    source: ResolvedHDU,
    destination: fits.HDUList,
    target_index: int,
    result: np.ndarray,
    policy: StructurePolicy,
) -> None:
    """Validate a destination HDU according to *policy*."""
    target = get_hdu_identity(destination[target_index], target_index)
    if target.shape != result.shape:
        raise ImageProcessingError(
            f"Destination HDU shape {target.shape} does not match result shape "
            f"{result.shape}"
        )
    if policy == "target":
        return
    if source.identity.extname != target.extname or source.identity.extver != target.extver:
        raise ImageProcessingError(
            f"Destination identity {(target.extname, target.extver)!r} does not "
            f"match source identity "
            f"{(source.identity.extname, source.identity.extver)!r}"
        )
    for keyword in ("INSTRUME", "DETECTOR", "CCDNAME"):
        source_value = source.header.get(keyword)
        target_value = destination[target_index].header.get(keyword)
        if (
            source_value is not None
            and target_value is not None
            and source_value != target_value
        ):
            raise ImageProcessingError(
                f"Destination {keyword}={target_value!r} does not match source "
                f"{source_value!r}"
            )
    if policy == "strict":
        with fits.open(source.path, mode="readonly", memmap=False) as source_hdul:
            source_structure = [
                get_hdu_identity(candidate, index)
                for index, candidate in enumerate(source_hdul)
            ]
        destination_structure = [
            get_hdu_identity(candidate, index)
            for index, candidate in enumerate(destination)
        ]
        if source_structure != destination_structure:
            raise ImageProcessingError(
                "Destination MEF structure does not strictly match source"
            )


def _write_operation_result(
    source: ResolvedHDU,
    output_file: StrOrPath | None,
    data: np.ndarray,
    header: fits.Header,
    *,
    output_mode: OutputMode,
    destination_hdu: HDUSelector | None,
    structure_policy: StructurePolicy,
    overwrite: bool,
) -> None:
    """Commit a standalone, destination-MEF, or in-place result."""
    if output_mode not in ("sef", "preserve", "inplace"):
        raise ValueError(f"Unknown output_mode: {output_mode!r}")
    if structure_policy not in ("target", "compatible", "strict"):
        raise ValueError(f"Unknown structure_policy: {structure_policy!r}")

    if output_mode == "inplace":
        output_path = source.path
        if (
            output_file is not None
            and Path(output_file).resolve() != output_path.resolve()
        ):
            raise ValueError(
                "output_file must be omitted or equal input_file for inplace mode"
            )
    else:
        if output_file is None:
            raise ValueError(f"output_file is required for {output_mode!r} mode")
        output_path = Path(output_file)

    if output_mode == "sef":
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")
        transactional_write_hdulist(
            output_path,
            fits.HDUList([fits.PrimaryHDU(data=data, header=header)]),
        )
        return

    template_path = (
        source.path
        if output_mode == "inplace" or not output_path.exists()
        else output_path
    )
    with fits.open(template_path, mode="readonly", memmap=False) as template:
        hdul_out = fits.HDUList([candidate.copy() for candidate in template])

    target_index = resolve_hdu_index(
        hdul_out,
        destination_hdu or identity_selector(source),
        image_like=True,
        require_data=True,
        filepath=template_path,
    )
    _validate_destination_structure(
        source, hdul_out, target_index, np.asarray(data), structure_policy
    )
    target_identity = get_hdu_identity(hdul_out[target_index], target_index)
    hdul_out[target_index].data = np.array(data, copy=True)
    result_header = header.copy()
    if target_identity.extname is None:
        result_header.pop("EXTNAME", None)
    else:
        result_header["EXTNAME"] = target_identity.extname
    if target_identity.extver is None:
        result_header.pop("EXTVER", None)
    else:
        result_header["EXTVER"] = target_identity.extver
    hdul_out[target_index].header = result_header
    transactional_write_hdulist(output_path, hdul_out)


__all__ = [
    "OutputMode",
    "ResolvedHDU",
    "StructurePolicy",
    "identity_selector",
]
