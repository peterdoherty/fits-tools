#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.image_processing import cosmic_rays
from fits_tools.image_processing.cosmic_rays import (
    _calculate_mad,
    clean_cosmic_rays_file,
    detect_cosmic_rays,
    get_available_methods,
    reject_cosmic_rays,
)


def test_calculate_mad_returns_robust_sigma_estimate() -> None:
    data = np.array([1.0, 1.0, 2.0, 2.0, 100.0], dtype=np.float32)

    assert _calculate_mad(data) == pytest.approx(1.4826)


def test_detect_cosmic_rays_sigma_clip_finds_spike_and_adjusts_even_box_size() -> None:
    data = np.arange(49, dtype=np.float32).reshape(7, 7)
    data[3, 3] = 1000.0

    cleaned, mask, metadata = detect_cosmic_rays(
        data,
        method="sigma_clip",
        sigma=3.0,
        box_size=4,
        iterations=2,
    )

    assert mask[3, 3]
    assert cleaned[3, 3] != data[3, 3]
    assert metadata["box_size"] == 5
    assert metadata["iterations_performed"] >= 1


def test_reject_cosmic_rays_supports_interpolation_replacement() -> None:
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    data[2, 2] = 500.0

    result = reject_cosmic_rays(
        data,
        method="median",
        box_size=3,
        replace_with="interpolation",
    )

    assert result.num_cosmic_rays >= 1
    assert result.mask[2, 2]
    assert result.cleaned_data[2, 2] == pytest.approx(12.0, abs=1e-3)


def test_clean_cosmic_rays_file_writes_mask_extension_and_header_metadata(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw.fits"
    output_path = tmp_path / "clean.fits"
    data = np.arange(49, dtype=np.float32).reshape(7, 7)
    data[3, 3] = 500.0
    fits.PrimaryHDU(data=data).writeto(input_path)

    result = clean_cosmic_rays_file(
        input_path,
        output_file=output_path,
        method="median",
        box_size=3,
        overwrite=True,
        save_mask=True,
    )

    assert result.num_cosmic_rays >= 1
    with fits.open(output_path) as hdul:
        assert hdul[1].name == "CR_MASK"
        assert hdul[0].header["CRMETH"] == "median"
        assert hdul[0].header["CRNUM"] >= 1


def test_batch_clean_cosmic_rays_collects_successes_failures_and_mask_files(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.fits"
    missing = tmp_path / "missing.fits"
    data = np.arange(49, dtype=np.float32).reshape(7, 7)
    data[3, 3] = 900.0
    fits.PrimaryHDU(data=data).writeto(good)

    results = cosmic_rays.batch_clean_cosmic_rays(
        [good, missing],
        output_dir=tmp_path / "out",
        method="median",
        box_size=3,
        overwrite=True,
        save_masks=True,
    )

    assert results["successful"] == 1
    assert results["failed"] == 1
    assert results["total_cosmic_rays"] >= 1
    assert any("missing.fits" in error for error in results["errors"])
    assert (tmp_path / "out" / "good_clean.fits").exists()
    assert (tmp_path / "out" / "good_crmask.fits").exists()


def test_cosmic_ray_metadata_helpers_report_method_availability() -> None:
    methods = get_available_methods()
    deps = cosmic_rays.check_dependencies()

    assert "sigma_clip" in methods
    assert "available" in methods["median"]
    assert set(deps) == {"scipy", "astroscrappy", "tqdm"}
