from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import FileHandlingError, HeaderKeywordError
from fits_tools.keywords import (
    batch_delete_key,
    batch_set_key,
    check_keyword,
    delete_key,
    get_key,
    set_key,
)


@pytest.fixture
def keyword_fits_files(tmp_path: Path) -> list[Path]:
    files: list[Path] = []
    for idx, imagetyp in enumerate(("LIGHT", "BIAS"), start=1):
        path = tmp_path / f"file_{idx}.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(
                    data=np.ones((2, 2), dtype=np.float32) * idx,
                    header=fits.Header({"OBJECT": f"OBJ{idx}", "IMAGETYP": imagetyp}),
                ),
                fits.ImageHDU(
                    data=np.ones((2, 2), dtype=np.float32) * (idx + 10),
                    name="SCI",
                    header=fits.Header({"EXTVER": 1}),
                ),
            ]
        ).writeto(path)
        files.append(path)
    return files


def test_check_get_set_and_delete_keyword_roundtrip(keyword_fits_files: list[Path]) -> None:
    path = keyword_fits_files[0]

    assert check_keyword(path, "OBJECT") is True
    assert get_key(path, "OBJECT") == "OBJ1"
    assert set_key(path, "PIPELINE", "v1", comment="test comment") is True
    assert get_key(path, "PIPELINE") == "v1"
    assert delete_key(path, "PIPELINE") is True
    assert check_keyword(path, "PIPELINE") is False


def test_get_and_delete_keyword_raise_for_missing_keyword(keyword_fits_files: list[Path]) -> None:
    path = keyword_fits_files[0]

    with pytest.raises(HeaderKeywordError, match="Keyword 'MISSING' not found"):
        get_key(path, "MISSING")

    with pytest.raises(HeaderKeywordError, match="Keyword 'MISSING' not found"):
        delete_key(path, "MISSING")


def test_keyword_ops_support_named_hdu_selection(keyword_fits_files: list[Path]) -> None:
    path = keyword_fits_files[0]

    assert set_key(path, "GAIN", 1.5, hdu=("SCI", 1)) is True
    assert get_key(path, "GAIN", hdu="SCI") == 1.5
    assert check_keyword(path, "GAIN", hdu=("SCI", 1)) is True


def test_batch_set_and_delete_keyword_across_selected_files(keyword_fits_files: list[Path]) -> None:
    base_dir = keyword_fits_files[0].parent

    updated = batch_set_key(
        str(base_dir / "*.fits"),
        "BATCHED",
        True,
        keys={"IMAGETYP": "LIGHT"},
    )
    assert updated == 1
    assert get_key(keyword_fits_files[0], "BATCHED") is True
    with pytest.raises(HeaderKeywordError):
        get_key(keyword_fits_files[1], "BATCHED")

    updated_all_hdus = batch_set_key(keyword_fits_files, "ALLSCI", 5, hdu=None)
    assert updated_all_hdus == 2
    assert get_key(keyword_fits_files[0], "ALLSCI", hdu="SCI") == 5

    deleted = batch_delete_key(keyword_fits_files, "ALLSCI", hdu=None)
    assert deleted == 2
    with pytest.raises(HeaderKeywordError):
        get_key(keyword_fits_files[0], "ALLSCI", hdu="SCI")


def test_keyword_ops_raise_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.fits"

    with pytest.raises(FileHandlingError, match="File not found"):
        check_keyword(missing, "OBJECT")
