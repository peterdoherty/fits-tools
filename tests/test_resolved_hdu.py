#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.resolved_hdu import (
    derive_associated_selector,
    identity_selector,
    load_resolved_hdu,
)


@pytest.fixture
def mef_path(tmp_path: Path) -> Path:
    path = tmp_path / "multi.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 3), dtype=np.float32), name="SCI")
    sci2 = fits.ImageHDU(data=np.full((2, 3), 2.0, dtype=np.float32), name="SCI")
    sci1.header["EXTVER"] = 1
    sci2.header["EXTVER"] = 2
    fits.HDUList([fits.PrimaryHDU(), sci1, sci2]).writeto(path)
    return path


def test_load_resolved_hdu_detaches_data_header_and_identity(
    mef_path: Path,
) -> None:
    resolved = load_resolved_hdu(mef_path, ("SCI", 2))

    assert resolved.path == mef_path
    assert resolved.index == 2
    assert resolved.identity.extname == "SCI"
    assert resolved.identity.extver == 2
    assert np.array_equal(resolved.data, np.full((2, 3), 2.0))
    assert resolved.header["EXTVER"] == 2

    with fits.open(mef_path, mode="update") as hdul:
        hdul[2].data[0, 0] = 99
        hdul[2].header["CHANGED"] = True

    assert resolved.data[0, 0] == 2
    assert "CHANGED" not in resolved.header


def test_load_resolved_hdu_auto_prefers_science_image(mef_path: Path) -> None:
    resolved = load_resolved_hdu(mef_path)

    assert resolved.index == 1
    assert identity_selector(resolved) == ("SCI", 1)


def test_associated_selector_preserves_numeric_selection(mef_path: Path) -> None:
    resolved = load_resolved_hdu(mef_path, 2)

    assert derive_associated_selector(resolved, 2) == 2
    assert derive_associated_selector(resolved, 2, extname="DQ") == 2


@pytest.mark.parametrize(
    ("load_selector", "source_selector"),
    [
        (("SCI", 2), "auto"),
        (("SCI", 2), "SCI"),
        (("SCI", 2), ("SCI", 2)),
    ],
)
def test_associated_selector_uses_resolved_exact_identity(
    mef_path: Path,
    load_selector: object,
    source_selector: object,
) -> None:
    resolved = load_resolved_hdu(
        mef_path, load_selector  # type: ignore[arg-type]
    )

    assert derive_associated_selector(
        resolved, source_selector  # type: ignore[arg-type]
    ) == ("SCI", 2)


def test_associated_selector_can_map_science_to_dq(mef_path: Path) -> None:
    resolved = load_resolved_hdu(mef_path, ("SCI", 2))

    assert derive_associated_selector(
        resolved, ("SCI", 2), extname="dq"
    ) == ("DQ", 2)

    with pytest.raises(ValueError, match="non-empty"):
        derive_associated_selector(resolved, ("SCI", 2), extname=" ")
