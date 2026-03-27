from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from astropy.io import fits

from .exceptions import FileHandlingError, InvalidHDUIndexError
from .types import StrOrPath

HDUSelector = Union[int, str, Tuple[str, int], Literal["auto"]]
AutoHDU = Literal["primary", "first_image", "first_sci_image"]


def is_image_like_hdu(hdu: fits.hdu.base.ExtensionHDU) -> bool:
    """
    Return True if the HDU appears to be an image HDU without accessing .data.

    Uses header inspection (NAXIS/NAXIS1/NAXIS2) to avoid decompression or
    materializing large arrays during header-only operations.
    """
    if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        return False
    try:
        naxis = int(hdu.header.get("NAXIS", 0) or 0)
        if naxis < 2:
            return False
        naxis1 = int(hdu.header.get("NAXIS1", 0) or 0)
        naxis2 = int(hdu.header.get("NAXIS2", 0) or 0)
        return (naxis1 > 0) and (naxis2 > 0)
    except Exception:
        return False


def _extname_upper(hdu: fits.hdu.base.ExtensionHDU) -> str:
    return str(hdu.header.get("EXTNAME", "")).strip().upper()


def _extver_int(hdu: fits.hdu.base.ExtensionHDU) -> Optional[int]:
    v = hdu.header.get("EXTVER", None)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def resolve_hdu_index(
    hdul: fits.HDUList,
    hdu: HDUSelector = "auto",
    *,
    auto: AutoHDU = "primary",
    image_like: bool = False,
    require_data: bool = False,
    filepath: Optional[StrOrPath] = None,
) -> int:
    """
    Resolve an HDU selector into an integer HDU index for the given HDUList.

    Parameters
    ----------
    hdul
        An already-open Astropy HDUList.
    hdu
        HDU selector:
        - int: HDU index (0-based)
        - str: EXTNAME match (case-insensitive)
        - (str, int): (EXTNAME, EXTVER) match
        - "auto": resolve using *auto* policy
    auto
        Policy used when hdu="auto":
        - "primary": return 0
        - "first_image": first image-like HDU meeting constraints
        - "first_sci_image": prefer EXTNAME in {"SCI","IMAGE"}, else fallback to first_image
    image_like
        If True, the selected HDU must be image-like (Primary/Image/CompImage and NAXIS>=2).
        This is checked using header inspection and does not require reading data.
    require_data
        If True, the selected HDU must have non-None ``.data`` (may trigger decompression).
    filepath
        Optional path used only to improve exception messages (e.g. for InvalidHDUIndexError).

    Returns
    -------
    int
        Resolved HDU index.

    Raises
    ------
    InvalidHDUIndexError
        If an integer index is out of bounds.
    ValueError
        If EXTNAME/EXTVER cannot be resolved, is missing, or is ambiguous.
    """
    # --- Helper for constraints ---
    def _meets_constraints(idx: int) -> bool:
        h = hdul[idx]
        if image_like and not is_image_like_hdu(h):
            return False
        if require_data and h.data is None:
            return False
        return True

    # --- int selector ---
    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(hdul):
            if filepath is not None:
                raise InvalidHDUIndexError(Path(filepath), hdu, len(hdul))
            raise InvalidHDUIndexError(Path("<memory>"), hdu, len(hdul))
        if not _meets_constraints(hdu):
            raise ValueError(f"HDU index {hdu} does not meet selection constraints")
        return hdu

    # --- (EXTNAME, EXTVER) selector ---
    if isinstance(hdu, tuple):
        if len(hdu) != 2:
            raise ValueError(f"HDU selector tuple must be (EXTNAME, EXTVER), got {hdu!r}")
        extname, extver = hdu
        target_name = str(extname).strip().upper()
        try:
            target_ver = int(extver)
        except Exception as e:
            raise ValueError(f"EXTVER must be an int, got {extver!r}") from e

        matches: list[int] = []
        for i in range(len(hdul)):
            if _extname_upper(hdul[i]) == target_name and _extver_int(hdul[i]) == target_ver:
                if _meets_constraints(i):
                    matches.append(i)

        if not matches:
            raise ValueError(f"No HDU with EXTNAME={target_name!r} and EXTVER={target_ver}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous HDU selector (EXTNAME={target_name!r}, EXTVER={target_ver}): "
                f"matched indices {matches}"
            )
        return matches[0]

    # --- str selector: EXTNAME or "auto" ---
    if isinstance(hdu, str):
        if hdu == "auto":
            # resolve below
            pass
        else:
            target_name = hdu.strip().upper()
            matches = [
                i for i in range(len(hdul))
                if _extname_upper(hdul[i]) == target_name and _meets_constraints(i)
            ]
            if not matches:
                raise ValueError(f"No HDU with EXTNAME={hdu!r}")
            if len(matches) > 1:
                # Force the user to disambiguate properly (professional behavior)
                vers = [(_extver_int(hdul[i]), i) for i in matches]
                raise ValueError(
                    f"Ambiguous EXTNAME selector {hdu!r}; matched HDUs {vers}. "
                    "Use (EXTNAME, EXTVER) to disambiguate."
                )
            return matches[0]
    else:
        raise TypeError(f"hdu must be int, str, (str,int), or 'auto'; got {type(hdu).__name__}")

    # --- auto selector ---
    if auto == "primary":
        idx = 0
        if idx >= len(hdul):
            raise ValueError("Empty HDUList")
        if not _meets_constraints(idx):
            raise ValueError("Primary HDU does not meet selection constraints")
        return idx

    if auto == "first_sci_image":
        preferred = {"SCI", "IMAGE"}
        for i in range(len(hdul)):
            if _extname_upper(hdul[i]) in preferred and _meets_constraints(i):
                return i
        # fallback to first_image
        auto = "first_image"

    if auto == "first_image":
        for i in range(len(hdul)):
            if _meets_constraints(i):
                return i
        raise ValueError("No HDU found matching selection constraints")

    raise ValueError(f"Unknown auto policy: {auto!r}")


def resolve_hdu_index_in_file(
    filepath: StrOrPath,
    hdu: HDUSelector = "auto",
    *,
    auto: AutoHDU = "primary",
    image_like: bool = False,
    require_data: bool = False,
    memmap: bool = True,
) -> int:
    """
    Convenience wrapper: open a FITS file and resolve an HDU selector to an index.

    Raises FileHandlingError on open failures; raises the same resolution errors
    as `resolve_hdu_index` for selector issues.
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
    except FileHandlingError:
        raise
    except OSError as e:
        raise FileHandlingError(f"FITS I/O error opening file: {e}", filepath=filepath) from e
    
    __all__ = {
        "HDUSelector", 
        "is_image_like_hdu",
        "resolve_hdu_index", 
        "resolve_hdu_index_in_file",
    }