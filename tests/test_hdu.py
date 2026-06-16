#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import FileHandlingError, InvalidHDUIndexError, InvalidParameterError
from fits_tools.hdu import is_image_like_hdu, resolve_hdu_index, resolve_hdu_index_in_file


def test_is_image_like_hdu_distinguishes_image_and_table_hdus() -> None:
    image_hdu = fits.ImageHDU(data=np.ones((2, 3), dtype=np.float32), name="SCI")
    table_hdu = fits.BinTableHDU.from_columns([])

    assert is_image_like_hdu(image_hdu) is True
    assert is_image_like_hdu(table_hdu) is False


def test_resolve_hdu_index_supports_auto_name_and_tuple_selection() -> None:
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32), name="SCI"),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32) * 2, name="SCI"),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32) * 3, name="DQ"),
        ]
    )
    hdul[1].header["EXTVER"] = 1
    hdul[2].header["EXTVER"] = 2

    assert resolve_hdu_index(hdul, "auto", auto="primary") == 0
    assert resolve_hdu_index(hdul, "auto", auto="first_image", image_like=True) == 1
    assert resolve_hdu_index(hdul, "auto", auto="first_sci_image", image_like=True) == 1
    assert resolve_hdu_index(hdul, ("SCI", 2), image_like=True) == 2
    assert resolve_hdu_index(hdul, "DQ", image_like=True) == 3


def test_resolve_hdu_index_rejects_ambiguous_or_invalid_selectors() -> None:
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32), name="SCI"),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32) * 2, name="SCI"),
        ]
    )
    hdul[1].header["EXTVER"] = 1
    hdul[2].header["EXTVER"] = 2

    with pytest.raises(InvalidParameterError, match="Ambiguous EXTNAME selector"):
        resolve_hdu_index(hdul, "SCI", image_like=True)

    with pytest.raises(InvalidParameterError, match="EXTVER must be an integer"):
        resolve_hdu_index(hdul, ("SCI", "bad"))

    with pytest.raises(InvalidHDUIndexError):
        resolve_hdu_index(hdul, 9)


def test_resolve_hdu_index_in_file_opens_file_and_applies_constraints(tmp_path: Path) -> None:
    path = tmp_path / "multi.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((3, 3), dtype=np.float32), name="SCI"),
        ]
    ).writeto(path)

    assert resolve_hdu_index_in_file(path, "auto", auto="first_image", image_like=True) == 1

    with pytest.raises(FileHandlingError, match="File not found"):
        resolve_hdu_index_in_file(tmp_path / "missing.fits")


def test_hdu_helpers_cover_header_edge_cases_and_constraint_failures(tmp_path: Path) -> None:
    weird = fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32), name="SCI")
    weird.header["EXTVER"] = "bad"
    no_data = fits.ImageHDU(name="EMPTY")
    hdul = fits.HDUList([fits.PrimaryHDU(), weird, no_data])

    assert is_image_like_hdu(fits.PrimaryHDU()) is False

    with pytest.raises(InvalidParameterError, match="No HDU matches selector"):
        resolve_hdu_index(hdul, ("SCI", 1), image_like=True)

    with pytest.raises(InvalidParameterError, match="non-empty string"):
        resolve_hdu_index(hdul, "   ")

    with pytest.raises(InvalidParameterError, match="Tuple HDU selector"):
        resolve_hdu_index(hdul, ("SCI", 1, 2))  # type: ignore[arg-type]

    with pytest.raises(InvalidParameterError, match="No HDU with EXTNAME"):
        resolve_hdu_index(hdul, "EMPTY", require_data=True)

    with pytest.raises(InvalidParameterError, match="Primary HDU does not satisfy"):
        resolve_hdu_index(hdul, "auto", auto="primary", image_like=True)

    hdul_no_data = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(name="EMPTY")])
    with pytest.raises(InvalidParameterError, match="No HDU satisfies"):
        resolve_hdu_index(hdul_no_data, "auto", auto="first_image", require_data=True)

    with pytest.raises(InvalidParameterError, match="Unknown auto-selection policy"):
        resolve_hdu_index(hdul, "auto", auto="mystery")  # type: ignore[arg-type]

    bad_path = tmp_path / "bad.fits"
    bad_path.write_text("not fits", encoding="utf-8")
    with pytest.raises(FileHandlingError, match="Failed to open/read FITS file"):
        resolve_hdu_index_in_file(bad_path)
