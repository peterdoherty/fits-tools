#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_header() -> fits.Header:
    header = fits.Header()
    header["IMAGETYP"] = "LIGHT"
    header["EXPTIME"] = 30.0
    header["DATASEC"] = "[2:5,2:4]"
    header["BIASSEC"] = "[1:1,2:4]"
    header["TRIMSEC"] = "[2:4,2:4]"
    return header


@pytest.fixture
def sample_data() -> np.ndarray:
    return np.arange(20, dtype=np.float32).reshape(4, 5)


@pytest.fixture
def sample_fits_file(tmp_path: Path, sample_data: np.ndarray, sample_header: fits.Header) -> Path:
    path = tmp_path / "sample.fits"
    fits.PrimaryHDU(data=sample_data, header=sample_header).writeto(path)
    return path


@pytest.fixture
def mef_fits_file(tmp_path: Path) -> Path:
    path = tmp_path / "multi.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=np.arange(4, dtype=np.int16).reshape(2, 2)),
            fits.ImageHDU(data=np.ones((3, 3), dtype=np.float32), name="SCI"),
            fits.ImageHDU(data=np.full((2, 3), 7, dtype=np.int16), name="SCI"),
        ]
    )
    hdul.writeto(path)
    return path


@pytest.fixture
def mef_integration_file(tmp_path: Path) -> Path:
    """Representative instrument MEF used by cross-module acceptance tests."""
    path = tmp_path / "integration_mef.fits"

    def image(
        name: str,
        version: int,
        shape: tuple[int, int],
        value: float,
        dtype: np.dtype = np.dtype(np.float32),
    ) -> fits.ImageHDU:
        header = fits.Header(
            {
                "EXTVER": version,
                "INSTRUME": "TESTCAM",
                "DETECTOR": "SYNTHETIC",
                "CRPIX1": 10.0,
                "CRPIX2": 20.0,
            }
        )
        return fits.ImageHDU(
            data=np.full(shape, value, dtype=dtype),
            header=header,
            name=name,
        )

    compressed = fits.CompImageHDU(
        data=np.full((2, 4), 5.0, dtype=np.float32),
        name="COMP",
    )
    compressed.header["EXTVER"] = 1
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="VALUE", format="J", array=np.array([42]))],
        name="META",
    )
    fits.HDUList(
        [
            fits.PrimaryHDU(
                header=fits.Header(
                    {"INSTRUME": "TESTCAM", "DETECTOR": "SYNTHETIC"}
                )
            ),
            image("SCI", 1, (4, 4), 10.0),
            image("DQ", 1, (4, 4), 0, np.dtype(np.uint32)),
            table,
            image("SCI", 2, (3, 5), 20.0),
            image("DQ", 2, (3, 5), 0, np.dtype(np.uint32)),
            compressed,
        ]
    ).writeto(path, checksum=True)
    return path


@pytest.fixture
def bias_fits_file(tmp_path: Path) -> Path:
    path = tmp_path / "bias.fits"
    header = fits.Header()
    header["IMAGETYP"] = "BIAS"
    header["EXPTIME"] = 0.0
    fits.PrimaryHDU(
        data=np.full((4, 5), 100.0, dtype=np.float32),
        header=header,
    ).writeto(path)
    return path
