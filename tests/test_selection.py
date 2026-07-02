#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import SelectionError
from fits_tools.selection import (
    select_bias_files,
    select_dark_files,
    select_files,
    select_flat_files,
    select_light_files,
)


@pytest.fixture
def selection_tree(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()

    specs = [
        (root / "light1.fits", "LIGHT", "OBJ1"),
        (root / "bias1.fits", "BIAS", "OBJ2"),
        (nested / "dark1.fits", "DARK", "OBJ3"),
        (nested / "flat1.fits", "FLAT", "OBJ4"),
    ]
    for path, imagetyp, obj in specs:
        fits.PrimaryHDU(
            data=np.ones((2, 2), dtype=np.float32),
            header=fits.Header({"IMAGETYP": imagetyp, "OBJECT": obj}),
        ).writeto(path)

    (root / "notes.txt").write_text("not fits", encoding="utf-8")
    return root


def test_select_files_supports_directory_glob_and_header_filters(selection_tree: Path) -> None:
    all_non_recursive = select_files(selection_tree)
    recursive = select_files(selection_tree, recursive=True)
    globbed = select_files(str(selection_tree / "*.fits"))
    filtered = select_files(selection_tree, keys={"OBJECT": ["OBJ1", "OBJ4"]}, recursive=True)

    assert {path.name for path in all_non_recursive} == {"light1.fits", "bias1.fits"}
    assert {path.name for path in recursive} == {"light1.fits", "bias1.fits", "dark1.fits", "flat1.fits"}
    assert {path.name for path in globbed} == {"light1.fits", "bias1.fits"}
    assert {path.name for path in filtered} == {"light1.fits", "flat1.fits"}


def test_select_files_honors_max_files_and_rejects_invalid_inputs(selection_tree: Path) -> None:
    limited = select_files(selection_tree, recursive=True, max_files=2)
    assert len(limited) == 2

    with pytest.raises(SelectionError, match="positive integer"):
        select_files(selection_tree, max_files=0)

    with pytest.raises(SelectionError, match="not a valid file, directory, or glob pattern"):
        select_files(selection_tree / "missing_dir")


def test_frame_type_selectors_pick_expected_files(selection_tree: Path) -> None:
    assert [p.name for p in select_bias_files(selection_tree, recursive=True)] == ["bias1.fits"]
    assert [p.name for p in select_dark_files(selection_tree, recursive=True)] == ["dark1.fits"]
    assert [p.name for p in select_flat_files(selection_tree, recursive=True)] == ["flat1.fits"]
    assert [p.name for p in select_light_files(selection_tree, recursive=True)] == ["light1.fits"]


def test_select_files_supports_single_file_and_explicit_lists(selection_tree: Path) -> None:
    light = selection_tree / "light1.fits"
    bias = selection_tree / "bias1.fits"
    missing = selection_tree / "missing.fits"

    selected_single = select_files(light)
    selected_list = select_files([light, missing, bias])

    assert selected_single == [light]
    assert selected_list == [light, bias]


def test_select_files_filters_case_insensitively_and_skips_invalid_fits(
    tmp_path: Path, selection_tree: Path
) -> None:
    broken = selection_tree / "broken.fits"
    broken.write_text("not really fits", encoding="utf-8")

    selected = select_files(
        selection_tree,
        keys={"object": " obj1 "},
        recursive=True,
    )

    assert [path.name for path in selected] == ["light1.fits"]


def test_select_files_rejects_invalid_input_type() -> None:
    with pytest.raises(SelectionError, match="path, glob pattern, or list"):
        select_files(123)  # type: ignore[arg-type]


def test_frame_type_selection_uses_fallback_header_keys_and_limits(
    tmp_path: Path,
) -> None:
    obstype_file = tmp_path / "obstype.fits"
    imgtype_file = tmp_path / "imgtype.fits"
    broken = tmp_path / "broken.fits"

    fits.PrimaryHDU(
        data=np.ones((2, 2), dtype=np.float32),
        header=fits.Header({"OBSTYPE": "light"}),
    ).writeto(obstype_file)
    fits.PrimaryHDU(
        data=np.ones((2, 2), dtype=np.float32),
        header=fits.Header({"IMGTYPE": "bias"}),
    ).writeto(imgtype_file)
    broken.write_text("not fits", encoding="utf-8")

    assert select_light_files([obstype_file, broken], max_files=1) == [obstype_file]
    assert select_bias_files([imgtype_file, broken], max_files=1) == [imgtype_file]


def test_selection_supports_named_and_extver_hdu_headers(tmp_path: Path) -> None:
    light = tmp_path / "mef_light.fits"
    bias = tmp_path / "mef_bias.fits"
    for path, image_type, extver in (
        (light, "LIGHT", 1),
        (bias, "BIAS", 2),
    ):
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(
                    data=np.ones((2, 2), dtype=np.float32),
                    header=fits.Header(
                        {"IMAGETYP": image_type, "OBJECT": image_type}
                    ),
                    name="SCI",
                    ver=extver,
                ),
            ]
        ).writeto(path)

    assert select_files(
        [light, bias],
        keys={"OBJECT": "BIAS"},
        hdu=("SCI", 2),
    ) == [bias]
    assert select_light_files([light, bias], hdu=("SCI", 1)) == [light]
    assert select_bias_files([light, bias], hdu=("SCI", 2)) == [bias]
