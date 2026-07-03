#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from fits_tools.image_processing.cosmic_rays import clean_cosmic_rays_file
from fits_tools.image_processing.transformations import transform_image_file
from fits_tools.simulation.noise import add_gaussian_noise


def _assert_preserved_structure(path: Path) -> None:
    with fits.open(path, checksum=True) as hdul:
        assert hdul[3].name == "META"
        assert hdul[3].data["VALUE"][0] == 42
        assert isinstance(hdul[6], fits.CompImageHDU)
        checksum_results = [hdu.verify_checksum() for hdu in hdul]
        assert checksum_results[:-1] == [1] * (len(hdul) - 1)
        assert checksum_results[-1] in (1, 2)


def test_transform_preserves_instrument_mef_wcs_tables_and_checksums(
    tmp_path: Path,
    mef_integration_file: Path,
) -> None:
    output = tmp_path / "transformed_mef.fits"

    transform_image_file(
        mef_integration_file,
        output,
        transformation="crop",
        hdu=("SCI", 1),
        region=(1, 3, 1, 3),
        adjust_wcs=True,
        output_mode="preserve",
    )

    with fits.open(output) as hdul:
        assert hdul[("SCI", 1)].data.shape == (2, 2)
        assert hdul[("SCI", 1)].header["CRPIX1"] == 9.0
        assert hdul[("SCI", 1)].header["CRPIX2"] == 19.0
        assert hdul[("DQ", 1)].data.shape == (4, 4)
        assert hdul[("SCI", 2)].data.shape == (3, 5)
    _assert_preserved_structure(output)


def test_cosmic_ray_cleaning_preserves_unrelated_mef_hdus(
    tmp_path: Path,
    mef_integration_file: Path,
) -> None:
    output = tmp_path / "cosmic_mef.fits"
    with fits.open(mef_integration_file, mode="update") as hdul:
        hdul[("SCI", 1)].data[2, 2] = 1000.0

    clean_cosmic_rays_file(
        mef_integration_file,
        output,
        method="median",
        box_size=3,
        hdu=("SCI", 1),
        output_mode="preserve",
    )

    with fits.open(output) as hdul:
        assert hdul[("SCI", 1)].header["CRMETH"] == "median"
        assert hdul[("SCI", 2)].data.shape == (3, 5)
        assert hdul[("DQ", 1)].data.shape == (4, 4)
    _assert_preserved_structure(output)


def test_noise_injection_preserves_selected_mef_container(
    tmp_path: Path,
    mef_integration_file: Path,
) -> None:
    output = tmp_path / "noisy_mef.fits"

    add_gaussian_noise(
        mef_integration_file,
        str(output),
        sigma=1.0,
        hdu=("SCI", 2),
        seed=7,
        output_mode="preserve",
    )

    with fits.open(output) as hdul:
        assert not np.array_equal(
            hdul[("SCI", 2)].data,
            np.full((3, 5), 20.0, dtype=np.float32),
        )
        assert np.all(hdul[("SCI", 1)].data == 10)
        assert hdul[("SCI", 2)].header["NOISEED"] == 7
    _assert_preserved_structure(output)
