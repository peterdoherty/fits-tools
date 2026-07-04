#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Shared output-disposition policy for FITS processing results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits

from .exceptions import ImageProcessingError
from .hdu import HDUSelector, get_hdu_identity, resolve_hdu_index
from .io import transactional_write_hdulist
from .resolved_hdu import ResolvedHDU, identity_selector
from .types import StrOrPath

OutputMode = Literal["sef", "preserve", "inplace"]
StructurePolicy = Literal["target", "compatible", "strict"]


@dataclass(slots=True)
class HDUResult:
    """One detached source/result pair for an MEF transaction."""

    source: ResolvedHDU
    data: np.ndarray
    header: fits.Header
    destination_hdu: HDUSelector | None = None
    allow_shape_change: bool = False


def _validate_destination_structure(
    source: ResolvedHDU,
    destination: fits.HDUList,
    target_index: int,
    result: np.ndarray,
    policy: StructurePolicy,
    allow_shape_change: bool = False,
) -> None:
    """Validate a destination HDU according to *policy*."""
    target = get_hdu_identity(destination[target_index], target_index)
    if not allow_shape_change and target.shape != result.shape:
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


def commit_hdu_result(
    source: ResolvedHDU,
    output_file: StrOrPath | None,
    data: np.ndarray,
    header: fits.Header,
    *,
    output_mode: OutputMode,
    destination_hdu: HDUSelector | None,
    structure_policy: StructurePolicy,
    overwrite: bool,
    allow_shape_change: bool = False,
) -> None:
    """Commit one image result as a standalone file or into an MEF.

    The source data and header must already be detached from all open FITS
    handles. Preserve and in-place modes copy the complete template HDU list,
    validate the selected destination, and perform one transactional write.

    *overwrite* applies only to ``"sef"`` mode, where it guards replacing an
    existing standalone output file. Preserve mode updates an existing
    destination MEF (or initializes a missing one) by design, and in-place
    mode rewrites the source file by definition, so neither is gated on
    *overwrite*.
    """
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

    commit_hdu_results(
        [
            HDUResult(
                source,
                np.asarray(data),
                header,
                destination_hdu,
                allow_shape_change,
            )
        ],
        output_file,
        output_mode=output_mode,
        structure_policy=structure_policy,
    )


def commit_hdu_results(
    results: Iterable[HDUResult],
    output_file: StrOrPath | None,
    *,
    output_mode: OutputMode = "preserve",
    structure_policy: StructurePolicy = "compatible",
) -> None:
    """Validate and commit several HDU results in one FITS transaction.

    Every result must originate from the same source file. All destination
    selectors are resolved and validated before the detached template is
    modified. Duplicate targets reject the operation. The destination is
    written exactly once after every update is ready.

    There is deliberately no ``overwrite`` guard: preserve mode updates an
    existing destination MEF (or initializes a missing one) by design, and
    in-place mode rewrites the source file by definition.
    """
    if output_mode not in ("preserve", "inplace"):
        raise ValueError(
            "commit_hdu_results supports only 'preserve' and 'inplace' modes"
        )
    if structure_policy not in ("target", "compatible", "strict"):
        raise ValueError(f"Unknown structure_policy: {structure_policy!r}")

    result_list = list(results)
    if not result_list:
        raise ValueError("results must contain at least one HDUResult")
    if not all(isinstance(result, HDUResult) for result in result_list):
        raise TypeError("results must contain only HDUResult instances")

    source_path = result_list[0].source.path
    if any(
        result.source.path.resolve() != source_path.resolve()
        for result in result_list[1:]
    ):
        raise ValueError("All HDU results must originate from the same source file")

    if output_mode == "inplace":
        output_path = source_path
        if (
            output_file is not None
            and Path(output_file).resolve() != output_path.resolve()
        ):
            raise ValueError(
                "output_file must be omitted or equal input_file for inplace mode"
            )
    else:
        if output_file is None:
            raise ValueError("output_file is required for 'preserve' mode")
        output_path = Path(output_file)

    template_path = (
        source_path
        if output_mode == "inplace" or not output_path.exists()
        else output_path
    )
    with fits.open(template_path, mode="readonly", memmap=False) as template:
        hdul_out = fits.HDUList([candidate.copy() for candidate in template])

    resolved: list[tuple[HDUResult, int]] = []
    target_indices: set[int] = set()
    for result in result_list:
        selector = (
            result.destination_hdu
            if result.destination_hdu is not None
            else identity_selector(result.source)
        )
        target_index = resolve_hdu_index(
            hdul_out,
            selector,
            image_like=True,
            require_data=True,
            filepath=template_path,
        )
        if target_index in target_indices:
            raise ImageProcessingError(
                f"Several results resolve to destination HDU {target_index}"
            )
        target_indices.add(target_index)
        _validate_destination_structure(
            result.source,
            hdul_out,
            target_index,
            np.asarray(result.data),
            structure_policy,
            result.allow_shape_change,
        )
        resolved.append((result, target_index))

    for result, target_index in resolved:
        target_identity = get_hdu_identity(hdul_out[target_index], target_index)
        hdul_out[target_index].data = np.array(result.data, copy=True)
        result_header = result.header.copy()
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
    "HDUResult",
    "OutputMode",
    "ResolvedHDU",
    "StructurePolicy",
    "commit_hdu_result",
    "commit_hdu_results",
    "identity_selector",
]
