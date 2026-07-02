#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""HDU selection helpers for FITS files.

This module centralizes HDU selection semantics used throughout
``fits_tools``. It defines singular selectors, plural queries, stable HDU
identities, and public resolution functions:

- :func:`resolve_hdu_index` -- resolve a selector against an open
  :class:`~astropy.io.fits.HDUList`.
- :func:`resolve_hdu_indices` -- resolve several exact selectors.
- :func:`find_hdu_indices` -- discover HDUs matching an :class:`HDUQuery`.
- :func:`resolve_hdu_index_in_file` -- open a file and resolve a selector
  in one call.

Public APIs across the package should use the ``hdu`` parameter name and
the :data:`HDUSelector` type rather than ad-hoc ``ext``, ``fits_ext``, or
raw integer-only interfaces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class HDUIdentity:
    """Stable identity and basic structural metadata for one HDU."""

    index: int
    extname: str | None
    extver: int | None
    hdu_type: str
    shape: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class HDUQuery:
    """Criteria for finding zero or more HDUs in an HDUList.

    ``None`` means no restriction for ``extnames`` or ``extvers``. Empty
    sets intentionally match nothing.
    """

    extnames: frozenset[str] | None = None
    extvers: frozenset[int] | None = None
    include_primary: bool = False
    image_like: bool = True
    require_data: bool = True

    def __post_init__(self) -> None:
        if self.extnames is not None:
            normalized_names: set[str] = set()
            for name in self.extnames:
                if not isinstance(name, str) or not name.strip():
                    raise InvalidParameterError(
                        "HDUQuery extnames must contain non-empty strings.",
                        parameter="extnames",
                        value=name,
                    )
                normalized_names.add(name.strip().upper())
            object.__setattr__(self, "extnames", frozenset(normalized_names))

        if self.extvers is not None:
            normalized_versions: set[int] = set()
            for version in self.extvers:
                if isinstance(version, bool) or not isinstance(version, int):
                    raise InvalidParameterError(
                        "HDUQuery extvers must contain integers.",
                        parameter="extvers",
                        value=version,
                    )
                normalized_versions.add(version)
            object.__setattr__(self, "extvers", frozenset(normalized_versions))


__all__ = [
    "AutoHDU",
    "HDUIdentity",
    "HDUQuery",
    "HDUSelector",
    "find_hdu_indices",
    "get_hdu_identity",
    "is_image_like_hdu",
    "resolve_hdu_index",
    "resolve_hdu_index_in_file",
    "resolve_hdu_indices",
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


def get_hdu_identity(hdu: Any, index: int) -> HDUIdentity:
    """Return stable identity and structural metadata for *hdu*.

    Missing ``EXTNAME`` and ``EXTVER`` values are represented by ``None``.
    A present but non-integer ``EXTVER`` is rejected so malformed metadata
    is not silently confused with a missing version.
    """
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise InvalidParameterError(
            "HDU identity index must be a non-negative integer.",
            parameter="index",
            value=index,
        )
    if not hasattr(hdu, "header"):
        raise InvalidParameterError(
            "Object does not have a FITS header.",
            parameter="hdu",
            value=type(hdu).__name__,
        )

    raw_extname = hdu.header.get("EXTNAME")
    extname = (
        str(raw_extname).strip().upper()
        if raw_extname is not None and str(raw_extname).strip()
        else None
    )

    raw_extver = hdu.header.get("EXTVER")
    if raw_extver is None:
        extver = None
    else:
        try:
            extver = int(raw_extver)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidParameterError(
                f"HDU {index} has non-integer EXTVER={raw_extver!r}.",
                parameter="EXTVER",
                value=raw_extver,
            ) from exc

    shape: tuple[int, ...] | None = None
    if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        try:
            naxis = int(hdu.header.get("NAXIS", 0) or 0)
            if naxis > 0:
                fits_shape = tuple(
                    int(hdu.header.get(f"NAXIS{axis}", 0) or 0)
                    for axis in range(1, naxis + 1)
                )
                if all(length > 0 for length in fits_shape):
                    shape = tuple(reversed(fits_shape))
        except (TypeError, ValueError, OverflowError):
            shape = None

    return HDUIdentity(
        index=index,
        extname=extname,
        extver=extver,
        hdu_type=type(hdu).__name__,
        shape=shape,
    )


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

    if isinstance(hdu, bool):
        raise _selection_error(
            "Boolean values are not valid HDU indices.",
            value=hdu,
        )

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
        if not isinstance(extname, str) or not extname.strip():
            raise _selection_error(
                "EXTNAME must be a non-empty string.",
                value=hdu,
            )
        if isinstance(extver, bool) or not isinstance(extver, int):
            raise _selection_error("EXTVER must be an integer.", value=hdu)
        target_name = extname.strip().upper()
        target_ver = extver

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

    if isinstance(hdu, str) and hdu.strip().lower() == "auto":
        hdu = "auto"

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
            if (
                _extname_upper(candidate) in preferred_extnames
                and is_image_like_hdu(candidate)
                and meets_constraints(index)
            ):
                return index
        auto = "first_image"

    if auto == "first_image":
        for index, candidate in enumerate(hdul):
            if is_image_like_hdu(candidate) and meets_constraints(index):
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


def resolve_hdu_indices(
    hdul: fits.HDUList,
    selectors: Iterable[HDUSelector],
    *,
    image_like: bool = False,
    require_data: bool = False,
    filepath: StrOrPath | None = None,
) -> list[int]:
    """Resolve several exact selectors to unique HDU indices.

    Selector order is preserved. If multiple selectors resolve to the same
    HDU, the index appears only at its first position. Any invalid or
    ambiguous selector fails the complete operation.
    """
    if isinstance(selectors, (str, int, bool)):
        raise _selection_error(
            "selectors must be an iterable of HDU selectors, not one selector.",
            parameter="selectors",
            value=selectors,
        )
    if (
        isinstance(selectors, tuple)
        and len(selectors) == 2
        and isinstance(selectors[0], str)
        and isinstance(selectors[1], int)
    ):
        raise _selection_error(
            "Wrap a single (EXTNAME, EXTVER) selector in a list.",
            parameter="selectors",
            value=selectors,
        )

    try:
        selector_list = list(selectors)
    except TypeError as exc:
        raise _selection_error(
            "selectors must be an iterable of HDU selectors.",
            parameter="selectors",
            value=selectors,
        ) from exc

    resolved: list[int] = []
    seen: set[int] = set()
    for selector in selector_list:
        index = resolve_hdu_index(
            hdul,
            selector,
            image_like=image_like,
            require_data=require_data,
            filepath=filepath,
        )
        if index not in seen:
            resolved.append(index)
            seen.add(index)
    return resolved


def find_hdu_indices(
    hdul: fits.HDUList,
    query: HDUQuery,
    *,
    require_match: bool = True,
) -> list[int]:
    """Find all HDUs matching *query* in file order.

    Unlike :func:`resolve_hdu_indices`, this function performs discovery
    rather than exact resolution. Duplicate EXTNAME values are expected and
    are returned as separate indices.
    """
    if not isinstance(query, HDUQuery):
        raise InvalidParameterError(
            "query must be an HDUQuery instance.",
            parameter="query",
            value=type(query).__name__,
        )

    matches: list[int] = []
    for index, candidate in enumerate(hdul):
        if index == 0 and not query.include_primary:
            continue
        if query.image_like and not is_image_like_hdu(candidate):
            continue
        if query.require_data and candidate.data is None:
            continue
        if (
            query.extnames is not None
            and _extname_upper(candidate) not in query.extnames
        ):
            continue
        if (
            query.extvers is not None
            and _extver_int(candidate) not in query.extvers
        ):
            continue
        matches.append(index)

    if require_match and not matches:
        raise _selection_error(
            f"No HDUs match query {query!r}.",
            parameter="query",
            value=query,
        )
    return matches


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
