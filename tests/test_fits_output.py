#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import fits_tools.fits_output as output_module
import fits_tools.io as io_module
from fits_tools import HDUResult, commit_hdu_result, commit_hdu_results
from fits_tools.exceptions import FileHandlingError, ImageProcessingError
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


def test_public_multi_result_commit_writes_once_and_updates_all_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "multi_source.fits"
    output_path = tmp_path / "multi_output.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1)
    sci2 = fits.ImageHDU(data=np.full((2, 2), 2.0), name="SCI", ver=2)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            sci1,
            fits.BinTableHDU.from_columns([], name="KEEP"),
            sci2,
        ]
    ).writeto(source_path)
    source1 = load_resolved_hdu(source_path, ("SCI", 1))
    source2 = load_resolved_hdu(source_path, ("SCI", 2))

    writes = 0
    real_write = output_module.transactional_write_hdulist

    def counted_write(path: Path, hdul: fits.HDUList) -> None:
        nonlocal writes
        writes += 1
        real_write(path, hdul)

    monkeypatch.setattr(
        output_module, "transactional_write_hdulist", counted_write
    )
    commit_hdu_results(
        [
            HDUResult(source1, np.full((2, 2), 11.0), source1.header),
            HDUResult(source2, np.full((2, 2), 22.0), source2.header),
        ],
        output_path,
    )

    assert writes == 1
    with fits.open(output_path) as hdul:
        assert np.all(hdul[("SCI", 1)].data == 11)
        assert hdul[2].name == "KEEP"
        assert np.all(hdul[("SCI", 2)].data == 22)


def test_multi_result_commit_rejects_duplicates_without_changing_destination(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "duplicate_source.fits"
    destination_path = tmp_path / "duplicate_destination.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1)
    sci2 = fits.ImageHDU(data=np.full((2, 2), 2.0), name="SCI", ver=2)
    fits.HDUList([fits.PrimaryHDU(), sci1, sci2]).writeto(source_path)
    fits.HDUList([fits.PrimaryHDU(), sci1.copy(), sci2.copy()]).writeto(
        destination_path
    )
    before = destination_path.read_bytes()
    source1 = load_resolved_hdu(source_path, ("SCI", 1))
    source2 = load_resolved_hdu(source_path, ("SCI", 2))

    with pytest.raises(ImageProcessingError, match="Several results"):
        commit_hdu_results(
            [
                HDUResult(
                    source1,
                    np.full((2, 2), 11.0),
                    source1.header,
                    ("SCI", 1),
                ),
                HDUResult(
                    source2,
                    np.full((2, 2), 22.0),
                    source2.header,
                    ("SCI", 1),
                ),
            ],
            destination_path,
            structure_policy="target",
        )

    assert destination_path.read_bytes() == before


def test_multi_result_late_validation_failure_rolls_back_destination(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "rollback_source.fits"
    destination_path = tmp_path / "rollback_destination.fits"
    sci1 = fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1)
    sci2 = fits.ImageHDU(data=np.full((2, 2), 2.0), name="SCI", ver=2)
    fits.HDUList([fits.PrimaryHDU(), sci1, sci2]).writeto(source_path)
    fits.HDUList([fits.PrimaryHDU(), sci1.copy(), sci2.copy()]).writeto(
        destination_path
    )
    before = destination_path.read_bytes()
    source1 = load_resolved_hdu(source_path, ("SCI", 1))
    source2 = load_resolved_hdu(source_path, ("SCI", 2))

    with pytest.raises(ImageProcessingError, match="shape"):
        commit_hdu_results(
            [
                HDUResult(source1, np.full((2, 2), 11.0), source1.header),
                HDUResult(source2, np.full((3, 3), 22.0), source2.header),
            ],
            destination_path,
        )

    assert destination_path.read_bytes() == before


def test_multi_result_write_failure_leaves_destination_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "write_source.fits"
    destination_path = tmp_path / "write_destination.fits"
    sci = fits.ImageHDU(data=np.ones((2, 2)), name="SCI", ver=1)
    fits.HDUList([fits.PrimaryHDU(), sci]).writeto(source_path)
    fits.HDUList([fits.PrimaryHDU(), sci.copy()]).writeto(destination_path)
    before = destination_path.read_bytes()
    source = load_resolved_hdu(source_path, ("SCI", 1))

    def fail_replace(source_name: Path, destination_name: Path) -> None:
        raise OSError(
            f"simulated replace failure: {source_name} -> {destination_name}"
        )

    monkeypatch.setattr(io_module.os, "replace", fail_replace)
    with pytest.raises(FileHandlingError, match="simulated replace failure"):
        commit_hdu_results(
            [HDUResult(source, np.full((2, 2), 9.0), source.header)],
            destination_path,
        )

    assert destination_path.read_bytes() == before
