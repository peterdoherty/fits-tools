from __future__ import annotations

import pytest
from astropy.io import fits

from fits_tools.header_utils import add_comment, add_history, copy_header


def test_copy_header_returns_independent_copy_and_excludes_keys() -> None:
    header = fits.Header()
    header["A"] = 1
    header["B"] = 2

    copied = copy_header(header, exclude_keys=["b", ""])
    copied["A"] = 9

    assert "B" not in copied
    assert header["A"] == 1


def test_copy_header_requires_real_header() -> None:
    with pytest.raises(TypeError, match="source must be astropy.io.fits.Header"):
        copy_header({"A": 1})  # type: ignore[arg-type]


def test_add_history_and_comment_append_non_empty_messages() -> None:
    header = fits.Header()

    add_history(header, "  reduced  ")
    add_comment(header, "  checked  ")
    add_history(header, " ")
    add_comment(header, "")

    assert "reduced" in header["HISTORY"]
    assert "checked" in header["COMMENT"]
    assert len(header["HISTORY"]) == 1
    assert len(header["COMMENT"]) == 1


def test_add_history_and_comment_require_real_headers() -> None:
    with pytest.raises(TypeError, match="header must be astropy.io.fits.Header"):
        add_history({}, "x")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="header must be astropy.io.fits.Header"):
        add_comment({}, "x")  # type: ignore[arg-type]
