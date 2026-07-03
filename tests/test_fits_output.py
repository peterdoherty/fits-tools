#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from fits_tools import commit_hdu_result
from fits_tools.resolved_hdu import load_resolved_hdu


def test_public_commit_hdu_result_updates_existing_destination_once(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.fits"
    destination_path = tmp_path / "destination.fits"

    source_sci = fits.ImageHDU(
        data=np.ones((2, 2), dtype=np.float32), name="SCI"
    )
    source_sci.header["EXTVER"] = 1
    fits.HDUList([fits.PrimaryHDU(), source_sci]).writeto(source_path)

    destination_sci = fits.ImageHDU(
        data=np.zeros((2, 2), dtype=np.float32), name="SCI"
    )
    destination_sci.header["EXTVER"] = 1
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="VALUE", format="J", array=np.array([42]))],
        name="META",
    )
    fits.HDUList(
        [fits.PrimaryHDU(), table, destination_sci]
    ).writeto(destination_path)

    source = load_resolved_hdu(source_path, ("SCI", 1))
    result_header = source.header.copy()
    result_header["CALDONE"] = True
    commit_hdu_result(
        source,
        destination_path,
        np.full((2, 2), 8.0, dtype=np.float32),
        result_header,
        output_mode="preserve",
        destination_hdu=("SCI", 1),
        structure_policy="compatible",
        overwrite=False,
    )

    with fits.open(destination_path, checksum=True) as hdul:
        assert hdul[1].name == "META"
        assert hdul[1].data["VALUE"][0] == 42
        assert np.array_equal(
            hdul[("SCI", 1)].data,
            np.full((2, 2), 8.0, dtype=np.float32),
        )
        assert hdul[("SCI", 1)].header["CALDONE"]
