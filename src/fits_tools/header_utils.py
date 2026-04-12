"""Low-level FITS header helper functions.

This module centralizes small, reusable operations on
:class:`astropy.io.fits.Header` objects. It is intentionally narrow in scope:

- no file I/O
- no HDU selection
- no workflow orchestration

The functions here are suitable for use across :mod:`fits_tools`, especially
when adding provenance records or making safe copies of headers prior to
writing output products.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from astropy.io import fits

__all__ = [
    "add_history",
    "add_comment",
    "copy_header",
]


def copy_header(
    source: fits.Header,
    exclude_keys: Iterable[str] | None = None,
) -> fits.Header:
    """Return a copy of a FITS header, optionally excluding keywords.

    Parameters
    ----------
    source : fits.Header
        Source FITS header to copy.
    exclude_keys : iterable of str, optional
        Keyword names to remove from the returned copy. Matching is
        case-insensitive. All cards for each matched keyword are removed.

    Returns
    -------
    fits.Header
        Independent copy of *source* with requested keywords removed.

    Raises
    ------
    TypeError
        If *source* is not a :class:`astropy.io.fits.Header`.

    Notes
    -----
    All matching cards for each excluded keyword are removed via
    :meth:`astropy.io.fits.Header.remove` with ``remove_all=True``.
    """
    if not isinstance(source, fits.Header):
        raise TypeError(
            f"source must be astropy.io.fits.Header, got {type(source).__name__}"
        )

    new_header = source.copy()
    if exclude_keys is None:
        return new_header

    for key in exclude_keys:
        key_str = str(key).strip().upper()
        if not key_str:
            continue
        new_header.remove(key_str, remove_all=True, ignore_missing=True)

    return new_header


def add_history(header: fits.Header, message: Any) -> None:
    """Append a HISTORY record to a FITS header.

    Parameters
    ----------
    header : fits.Header
        Header to modify in place.
    message : any
        Message to append. Converted to ``str`` and stripped. Empty
        messages are silently ignored.

    Raises
    ------
    TypeError
        If *header* is not a :class:`astropy.io.fits.Header`.

    Notes
    -----
    FITS allows repeated HISTORY cards. This function appends a new record
    and does not attempt to deduplicate existing entries.
    """
    if not isinstance(header, fits.Header):
        raise TypeError(
            f"header must be astropy.io.fits.Header, got {type(header).__name__}"
        )

    text = str(message).strip()
    if not text:
        return

    header.add_history(text)


def add_comment(header: fits.Header, message: Any) -> None:
    """Append a COMMENT record to a FITS header.

    Parameters
    ----------
    header : fits.Header
        Header to modify in place.
    message : any
        Message to append. Converted to ``str`` and stripped. Empty
        messages are silently ignored.

    Raises
    ------
    TypeError
        If *header* is not a :class:`astropy.io.fits.Header`.

    Notes
    -----
    FITS allows repeated COMMENT cards. This function appends a new record
    and does not attempt to deduplicate existing entries.
    """
    if not isinstance(header, fits.Header):
        raise TypeError(
            f"header must be astropy.io.fits.Header, got {type(header).__name__}"
        )

    text = str(message).strip()
    if not text:
        return

    header.add_comment(text)
