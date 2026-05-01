from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import fits_tools.survey as survey
from fits_tools.survey import survey_fits


def test_survey_fits_writes_master_inventory_and_skips_reports_tree(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    reports = root / "reports"
    reports.mkdir()

    header = fits.Header({"IMAGETYP": "LIGHT", "EXPTIME": 30.0, "FILTER": "R"})
    fits.PrimaryHDU(data=np.arange(4, dtype=float).reshape(2, 2), header=header).writeto(root / "a.fits")
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=float), header=header).writeto(reports / "skip_me.fits")
    (root / "notes.txt").write_text("ignore")

    output = tmp_path / "master.csv"
    survey_fits(root, output_path=output, recursive=True, overwrite=True)

    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    assert rows[0]["filename"] == "a.fits"
    assert rows[0]["IMAGETYP"] == "LIGHT"
    assert rows[0]["EXPTIME"] == "30.0"


def test_survey_fits_split_by_dir_and_skip_existing_inventory(tmp_path: Path) -> None:
    root = tmp_path / "root"
    sub = root / "nested"
    sub.mkdir(parents=True)

    header = fits.Header({"IMAGETYP": "DARK", "EXPTIME": 5.0})
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=float), header=header).writeto(root / "root_file.fits")
    fits.PrimaryHDU(data=np.full((2, 2), 2.0), header=header).writeto(sub / "sub_file.fits")

    output_dir = tmp_path / "out"
    survey_fits(root, output_dir=output_dir, split_by_dir=True, recursive=True, overwrite=True)

    root_rows = list(csv.DictReader((output_dir / "inventory.csv").open()))
    sub_rows = list(csv.DictReader((output_dir / "nested" / "inventory.csv").open()))
    assert root_rows[0]["filename"] == "root_file.fits"
    assert sub_rows[0]["filename"] == "sub_file.fits"

    original = (output_dir / "inventory.csv").read_text()
    survey_fits(root, output_dir=output_dir, split_by_dir=True, recursive=True, overwrite=False)
    assert (output_dir / "inventory.csv").read_text() == original


def test_survey_fits_handles_row_build_failures_and_non_recursive_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    (root / "ok.fits").write_text("placeholder")
    (child / "child.fits").write_text("placeholder")

    def fake_make_row(
        file_path: Path,
        root_dir: Path,
        *,
        hdu: int,
        keywords: list[str],
        relative: bool,
    ) -> dict[str, object] | None:
        if file_path.name == "ok.fits":
            return {
                "filename": file_path.name,
                "mean": "1.00",
                "std": "0.10",
                "min": "0.00",
                "max": "2.00",
                "pixels": 4,
                "IMAGETYP": "LIGHT",
                "EXPTIME": 1.0,
                "FILTER": "N/A",
                "OBJECT": "N/A",
            }
        raise RuntimeError("bad row")

    monkeypatch.setattr(survey, "_make_row", fake_make_row)
    monkeypatch.setattr(survey, "_iter_fits_files_in_dir", lambda directory: sorted(directory.glob("*.fits")))

    output = tmp_path / "inventory.csv"
    survey_fits(root, output_path=output, recursive=False, overwrite=True)

    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    assert rows[0]["filename"] == "ok.fits"


def test_make_row_and_float_format_handle_na_and_header_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "x.fits"
    file_path.write_text("placeholder")

    class Result:
        mean_val = np.nan
        std_val = 2.345
        min_val = None
        max_val = np.inf
        num_pixels = 7

    monkeypatch.setattr(survey, "stats", lambda *args, **kwargs: Result())
    monkeypatch.setattr(survey, "get_header", lambda *args, **kwargs: {"FILTER": "I"})

    row = survey._make_row(
        file_path,
        tmp_path,
        hdu=0,
        keywords=["FILTER", "OBJECT"],
        relative=True,
    )

    assert row is not None
    assert row["filename"] == "x.fits"
    assert row["mean"] == "N/A"
    assert row["std"] == "2.35"
    assert row["min"] == "N/A"
    assert row["max"] == "N/A"
    assert row["FILTER"] == "I"
    assert row["OBJECT"] == "N/A"

    monkeypatch.setattr(survey, "stats", lambda *args, **kwargs: None)
    assert survey._make_row(file_path, tmp_path, hdu=0, keywords=[], relative=False) is None
