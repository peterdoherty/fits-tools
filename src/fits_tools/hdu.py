#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""HDU selection helpers for FITS files.

This module centralizes HDU selection semantics used throughout
``fits_tools``. It defines the :data:`HDUSelector` and :data:`AutoHDU`
type aliases and provides two public resolution functions:

- :func:`resolve_hdu_index` -- resolve a selector against an open
  :class:`~astropy.io.fits.HDUList`.
- :func:`resolve_hdu_index_in_file` -- open a file and resolve a selector
  in one call.

Public APIs across the package should use the ``hdu`` parameter name and
the :data:`HDUSelector` type rather than ad-hoc ``ext``, ``fits_ext``, or
raw integer-only interfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from astropy.io import fits

from .exceptions import (
    FileHandlingError,
    InvalidHDUIndexError,
    InvalidParameterError,
)
from .types import StrOrPath

HDUSelector = int | str | tuple[str, int] | Literal["auto"]
AutoHDU = Literal["primary", "first_image", "first_sci_image"]

__all__ = [
    "AutoHDU",
    "HDUSelector",
    "is_image_like_hdu",
    "resolve_hdu_index",
    "resolve_hdu_index_in_file",
]


def is_image_like_hdu(hdu: Any) -> bool:
    """Return True if *hdu* is an image HDU with a non-trivial image shape.

    The check is header-based and does not require materializing ``.data``,
    avoiding unnecessary decompression or array allocation during
    header-only workflows.

    Parameters
    ----------
    hdu : fits.PrimaryHDU or fits.ImageHDU or fits.CompImageHDU
        The HDU to test.

    Returns
    -------
    bool
        True if *hdu* is a primary, image, or compressed-image HDU whose
        header declares at least two axes with positive NAXIS1 and NAXIS2.
    """
    if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        return False

    try:
        naxis = int(hdu.header.get("NAXIS", 0) or 0)
        naxis1 = int(hdu.header.get("NAXIS1", 0) or 0)
        naxis2 = int(hdu.header.get("NAXIS2", 0) or 0)
    except Exception:
        return False

    return naxis >= 2 and naxis1 > 0 and naxis2 > 0


def _extname_upper(hdu: Any) -> str:
    """Return normalized EXTNAME for *hdu* (missing -> empty string)."""
    return str(hdu.header.get("EXTNAME", "")).strip().upper()


def _extver_int(hdu: Any) -> int | None:
    """Return integer EXTVER for *hdu*, or None if unavailable or unparseable."""
    extver = hdu.header.get("EXTVER")
    if extver is None:
        return None
    try:
        return int(extver)
    except Exception:
        return None


def _selection_error(
    message: str,
    *,
    parameter: str = "hdu",
    value: object = None,
) -> InvalidParameterError:
    """Create a consistent selector-resolution exception."""
    return InvalidParameterError(message, parameter=parameter, value=value)


def resolve_hdu_index(
    hdul: fits.HDUList,
    hdu: HDUSelector = "auto",
    *,
    auto: AutoHDU = "primary",
    image_like: bool = False,
    require_data: bool = False,
    filepath: StrOrPath | None = None,
) -> int:
    """Resolve an HDU selector to an integer index.

    Parameters
    ----------
    hdul : fits.HDUList
        Already-open FITS HDU list.
    hdu : HDUSelector, optional
        HDU selector. Supported forms are:

        - ``int`` -- zero-based HDU index.
        - ``str`` -- case-insensitive EXTNAME (must be non-empty).
        - ``(str, int)`` -- (EXTNAME, EXTVER) tuple.
        - ``"auto"`` -- resolved according to *auto*.

        Default is ``"auto"``.
    auto : {"primary", "first_image", "first_sci_image"}, optional
        Policy used when ``hdu="auto"``. Default is ``"primary"``.
    image_like : bool, optional
        If True, require the selected HDU to be image-like by header
        inspection. Default is False.
    require_data : bool, optional
        If True, require the selected HDU to have non-None ``.data``.
        May force decompression for compressed image HDUs. Default is False.
    filepath : str or Path, optional
        File path used only to improve exception messages.

    Returns
    -------
    int
        Resolved HDU index.

    Raises
    ------
    InvalidHDUIndexError
        If an integer selector is out of bounds.
    InvalidParameterError
        If the selector is malformed, ambiguous, or does not satisfy the
        requested constraints.
    """

    def meets_constraints(index: int) -> bool:
        candidate = hdul[index]
        if image_like and not is_image_like_hdu(candidate):
            return False
        if require_data and candidate.data is None:
            return False
        return True

    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(hdul):
            raise InvalidHDUIndexError(filepath or "<memory>", hdu, len(hdul))
        if not meets_constraints(hdu):
            raise _selection_error(
                "HDU does not satisfy the requested selection constraints.",
                value=hdu,
            )
        return hdu

    if isinstance(hdu, tuple):
        if len(hdu) != 2:
            raise _selection_error(
                "Tuple HDU selector must be of the form (EXTNAME, EXTVER).",
                value=hdu,
            )

        extname, extver = hdu
        target_name = str(extname).strip().upper()
        if not target_name:
            raise _selection_error("EXTNAME must be a non-empty string.", value=hdu)

        try:
            target_ver = int(extver)
        except Exception as exc:
            raise _selection_error("EXTVER must be an integer.", value=hdu) from exc

        matches = [
            index
            for index, candidate in enumerate(hdul)
            if _extname_upper(candidate) == target_name
            and _extver_int(candidate) == target_ver
            and meets_constraints(index)
        ]

        if not matches:
            raise _selection_error(
                f"No HDU matches selector "
                f"(EXTNAME={target_name!r}, EXTVER={target_ver}).",
                value=hdu,
            )
        if len(matches) > 1:
            raise _selection_error(
                f"Ambiguous HDU selector "
                f"(EXTNAME={target_name!r}, EXTVER={target_ver}); "
                f"matched HDUs {matches}.",
                value=hdu,
            )
        return matches[0]

    if isinstance(hdu, str) and hdu != "auto":
        target_name = hdu.strip().upper()
        if not target_name:
            raise _selection_error(
                "EXTNAME selector must be a non-empty string.", value=hdu
            )

        matches = [
            index
            for index, candidate in enumerate(hdul)
            if _extname_upper(candidate) == target_name and meets_constraints(index)
        ]

        if not matches:
            raise _selection_error(
                f"No HDU with EXTNAME={hdu!r} satisfies the requested constraints.",
                value=hdu,
            )
        if len(matches) > 1:
            versions = [(_extver_int(hdul[index]), index) for index in matches]
            raise _selection_error(
                f"Ambiguous EXTNAME selector {hdu!r}; matched HDUs {versions}. "
                "Use (EXTNAME, EXTVER) to disambiguate.",
                value=hdu,
            )
        return matches[0]

    if hdu != "auto":
        raise _selection_error(
            "hdu must be an integer, EXTNAME string, "
            "(EXTNAME, EXTVER) tuple, or 'auto'.",
            value=hdu,
        )

    if auto == "primary":
        if len(hdul) == 0:
            raise _selection_error(
                "Cannot resolve 'primary' from an empty HDUList.", value=hdu
            )
        if not meets_constraints(0):
            raise _selection_error(
                "Primary HDU does not satisfy the requested selection constraints.",
                value=hdu,
            )
        return 0

    if auto == "first_sci_image":
        preferred_extnames = {"SCI", "IMAGE"}
        for index, candidate in enumerate(hdul):
            if _extname_upper(candidate) in preferred_extnames and meets_constraints(index):
                return index
        auto = "first_image"

    if auto == "first_image":
        for index in range(len(hdul)):
            if meets_constraints(index):
                return index
        raise _selection_error(
            "No HDU satisfies the requested selection constraints.",
            parameter="auto",
            value=auto,
        )

    raise _selection_error(
        f"Unknown auto-selection policy {auto!r}.",
        parameter="auto",
        value=auto,
    )


def resolve_hdu_index_in_file(
    filepath: StrOrPath,
    hdu: HDUSelector = "auto",
    *,
    auto: AutoHDU = "primary",
    image_like: bool = False,
    require_data: bool = False,
    memmap: bool = True,
) -> int:
    """Open a FITS file and resolve *hdu* to an integer index.

    Parameters
    ----------
    filepath : str or Path
        Path to the FITS file.
    hdu : HDUSelector, optional
        HDU selector; see :func:`resolve_hdu_index` for supported forms.
        Default is ``"auto"``.
    auto : {"primary", "first_image", "first_sci_image"}, optional
        Auto-selection policy when ``hdu="auto"``. Default is ``"primary"``.
    image_like : bool, optional
        If True, require the resolved HDU to be image-like. Default is False.
    require_data : bool, optional
        If True, require the resolved HDU to have non-None ``.data``.
        Default is False.
    memmap : bool, optional
        Whether to use memory mapping when opening the file. Default is True.

    Returns
    -------
    int
        Resolved HDU index.

    Raises
    ------
    FileHandlingError
        If the file does not exist or cannot be opened.
    InvalidHDUIndexError
        If an integer selector is out of bounds.
    InvalidParameterError
        If the selector is malformed, ambiguous, or does not satisfy the
        requested constraints.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)

    try:
        with fits.open(filepath, memmap=memmap, mode="readonly") as hdul:
            return resolve_hdu_index(
                hdul,
                hdu,
                auto=auto,
                image_like=image_like,
                require_data=require_data,
                filepath=filepath,
            )
    except (FileHandlingError, InvalidHDUIndexError, InvalidParameterError):
        raise
    except Exception as exc:
        raise FileHandlingError(
            f"Failed to open/read FITS file {filepath}: {exc}",
            filepath=filepath,
        ) from exc
