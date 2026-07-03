#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Resolved image-HDU loading and cross-file association helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from .hdu import (
    AutoHDU,
    HDUIdentity,
    HDUSelector,
    get_hdu_identity,
    resolve_hdu_index,
)
from .types import StrOrPath


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


def derive_associated_selector(
    resolved: ResolvedHDU,
    source_selector: HDUSelector,
    *,
    extname: str | None = None,
) -> HDUSelector:
    """Derive a selector for the corresponding HDU in another file.

    Explicit numeric source selection retains numeric-position semantics for
    backward compatibility. Name, tuple, and ``"auto"`` selection derive the
    most exact selector available from the resolved source identity.

    Set *extname* for an associated product with a different conventional
    extension name. For example, ``extname="DQ"`` maps a resolved
    ``("SCI", 2)`` source to ``("DQ", 2)``. If the source has no ``EXTVER``,
    the requested bare name is returned and normal ambiguity checks apply.
    """
    if isinstance(source_selector, int) and not isinstance(source_selector, bool):
        return source_selector

    if extname is not None:
        normalized_name = extname.strip().upper()
        if not normalized_name:
            raise ValueError("extname must be a non-empty string")
        if resolved.identity.extver is not None:
            return (normalized_name, resolved.identity.extver)
        return normalized_name

    return identity_selector(resolved)


def load_resolved_hdu(
    filepath: StrOrPath,
    selector: HDUSelector = "auto",
    *,
    auto: AutoHDU = "first_sci_image",
    memmap: bool = False,
) -> ResolvedHDU:
    """Load and detach one image HDU using package-wide selection semantics."""
    path = Path(filepath)
    with fits.open(path, mode="readonly", memmap=memmap) as hdul:
        index = resolve_hdu_index(
            hdul,
            selector,
            auto=auto,
            image_like=True,
            require_data=True,
            filepath=path,
        )
        candidate = hdul[index]
        return ResolvedHDU(
            path=path,
            index=index,
            identity=get_hdu_identity(candidate, index),
            data=np.array(candidate.data, copy=True),
            header=candidate.header.copy(),
        )


__all__ = [
    "ResolvedHDU",
    "derive_associated_selector",
    "identity_selector",
    "load_resolved_hdu",
]
