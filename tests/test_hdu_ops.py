#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import HDUStructureError, InvalidParameterError
from fits_tools.hdu_ops import (
    _convert_primary_for_insertion,
    _ensure_primary_first,
    _infer_blank_shape,
    _select_hdus,
    add_hdus_to_file,
    remove_hdus_from_file,
    reorder_hdus_in_file,
)


@pytest.fixture
def base_fits_file(tmp_path: Path) -> Path:
    path = tmp_path / "base.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32), name="SCI"),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32) * 2, name="DQ"),
        ]
    ).writeto(path)
    return path


def test_add_hdus_to_file_can_import_named_hdus_and_create_blank_extensions(
    tmp_path: Path, base_fits_file: Path
) -> None:
    source_path = tmp_path / "source.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(data=np.full((2, 2), 9.0, dtype=np.float32)),
            fits.ImageHDU(data=np.full((2, 2), 5.0, dtype=np.float32), name="CAL"),
        ]
    ).writeto(source_path)
    output_path = tmp_path / "added.fits"

    add_hdus_to_file(
        base_fits_file,
        source_file=source_path,
        source_names=["PRIMARY", "CAL"],
        create_blank=1,
        blank_shape=(2, 2),
        blank_extname=["BLANK"],
        insert_end=True,
        output_file=output_path,
        overwrite=True,
    )

    with fits.open(output_path) as hdul:
        assert isinstance(hdul[0], fits.PrimaryHDU)
        assert hdul[3].header["EXTNAME"] == "IMPORTED"
        assert hdul[4].header["EXTNAME"] == "CAL"
        assert hdul[5].header["EXTNAME"] == "BLANK"
        assert np.array_equal(hdul[5].data, np.zeros((2, 2), dtype=np.float32))


def test_remove_hdus_from_file_can_remove_by_name(tmp_path: Path, base_fits_file: Path) -> None:
    output_path = tmp_path / "removed.fits"

    remove_hdus_from_file(
        base_fits_file,
        remove_names=["DQ"],
        output_file=output_path,
        overwrite=True,
    )

    with fits.open(output_path) as hdul:
        assert len(hdul) == 2
        assert hdul[1].header["EXTNAME"] == "SCI"


def test_reorder_hdus_in_file_supports_index_and_name_orders(
    tmp_path: Path, base_fits_file: Path
) -> None:
    by_index = tmp_path / "reordered_index.fits"
    by_name = tmp_path / "reordered_name.fits"

    reorder_hdus_in_file(
        base_fits_file,
        order_indices=[0, 2, 1],
        output_file=by_index,
        overwrite=True,
    )
    reorder_hdus_in_file(
        base_fits_file,
        order_names=["PRIMARY", "DQ", "SCI"],
        output_file=by_name,
        overwrite=True,
    )

    with fits.open(by_index) as hdul:
        assert hdul[1].header["EXTNAME"] == "DQ"
        assert hdul[2].header["EXTNAME"] == "SCI"

    with fits.open(by_name) as hdul:
        assert hdul[1].header["EXTNAME"] == "DQ"
        assert hdul[2].header["EXTNAME"] == "SCI"


def test_hdu_ops_raise_on_invalid_requests(tmp_path: Path, base_fits_file: Path) -> None:
    with pytest.raises(InvalidParameterError, match="At least one of remove_indices or remove_names"):
        remove_hdus_from_file(base_fits_file)

    with pytest.raises(HDUStructureError, match="Cannot insert at index 0"):
        add_hdus_to_file(base_fits_file, create_blank=1, blank_shape=(2, 2), insert_at=0)

    with pytest.raises(InvalidParameterError, match="Provide only one of order_indices or order_names"):
        reorder_hdus_in_file(base_fits_file, order_indices=[0, 1, 2], order_names=["PRIMARY"])

    with pytest.raises(HDUStructureError, match="must include HDU 0"):
        reorder_hdus_in_file(base_fits_file, order_indices=[1, 2])

    with pytest.raises(HDUStructureError, match="must remain first"):
        reorder_hdus_in_file(base_fits_file, order_indices=[1, 0, 2])


def test_internal_hdu_helpers_cover_primary_normalization_and_selection(
    base_fits_file: Path,
) -> None:
    image_first = fits.HDUList([fits.ImageHDU(data=np.ones((2, 2), dtype=np.float32), name="SCI")])
    table_first = fits.HDUList([fits.BinTableHDU(name="TAB"), fits.ImageHDU(data=np.ones((2, 2)))])

    converted = _ensure_primary_first(image_first)
    inserted = _ensure_primary_first(table_first)

    assert isinstance(converted[0], fits.PrimaryHDU)
    assert "XTENSION" not in converted[0].header
    assert isinstance(inserted[0], fits.PrimaryHDU)
    assert inserted[1].header["EXTNAME"] == "TAB"

    with pytest.raises(HDUStructureError, match="empty"):
        _ensure_primary_first(fits.HDUList([]))

    with fits.open(base_fits_file) as hdul:
        selected_default = _select_hdus(hdul)
        selected_mixed = _select_hdus(hdul, indices=[1], names=["PRIMARY", "SCI", "DQ"])
        inferred_shape = _infer_blank_shape(hdul)

    assert [hdu.header.get("EXTNAME") for hdu in selected_default] == ["SCI", "DQ"]
    assert selected_mixed[0].header["EXTNAME"] == "SCI"
    assert isinstance(selected_mixed[1], fits.PrimaryHDU)
    assert selected_mixed[2].header["EXTNAME"] == "DQ"
    assert inferred_shape == (2, 2)


def test_convert_primary_for_insertion_and_infer_blank_shape_edge_cases() -> None:
    primary = fits.PrimaryHDU(data=np.ones((2, 3), dtype=np.float32))
    image_hdu = _convert_primary_for_insertion(primary)
    no_shape = _infer_blank_shape(
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=np.array([1, 2, 3]))])
    )

    assert isinstance(image_hdu, fits.ImageHDU)
    assert image_hdu.header["EXTNAME"] == "IMPORTED"
    assert no_shape is None


def test_file_level_hdu_ops_cover_dry_run_insert_positions_and_all_removed_errors(
    tmp_path: Path, base_fits_file: Path
) -> None:
    inserted = tmp_path / "inserted.fits"
    dry_run_target = tmp_path / "dry_run.fits"

    add_hdus_to_file(
        base_fits_file,
        create_blank=2,
        blank_extname=["MID1", "MID2"],
        insert_at=1,
        insert_end=False,
        output_file=inserted,
        overwrite=True,
    )
    add_hdus_to_file(
        base_fits_file,
        create_blank=1,
        dry_run=True,
        output_file=dry_run_target,
    )

    with fits.open(inserted) as hdul:
        assert hdul[1].header["EXTNAME"] == "MID1"
        assert hdul[2].header["EXTNAME"] == "MID2"
        assert hdul[3].header["EXTNAME"] == "SCI"

    assert dry_run_target.exists() is False

    only_primary = tmp_path / "primary_only.fits"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(only_primary)
    with pytest.raises(HDUStructureError, match="would be empty"):
        remove_hdus_from_file(only_primary, remove_indices=[0], output_file=tmp_path / "bad.fits")


def test_reorder_hdus_by_name_can_rebuild_primary_and_support_dry_run(
    tmp_path: Path, base_fits_file: Path
) -> None:
    reordered = tmp_path / "reordered_single.fits"
    dry_run_target = tmp_path / "reordered_dry_run.fits"

    reorder_hdus_in_file(
        base_fits_file,
        order_names=["SCI"],
        output_file=reordered,
        overwrite=True,
    )
    reorder_hdus_in_file(
        base_fits_file,
        order_names=["PRIMARY", "DQ", "SCI"],
        dry_run=True,
        output_file=dry_run_target,
    )

    with fits.open(reordered) as hdul:
        assert isinstance(hdul[0], fits.PrimaryHDU)
        assert np.array_equal(hdul[0].data, np.ones((2, 2), dtype=np.float32))
        assert len(hdul) == 1

    assert dry_run_target.exists() is False
