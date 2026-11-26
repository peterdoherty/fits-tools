from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from fits_tools.simulation.core import (
    add_read_noise,
    create_bias_frame,
    create_bias_pair_for_noise_estimation,
    create_bias_time_series,
    create_calibration_set,
    create_dark_frame,
    create_flat_frame,
    create_mef_bias_frame,
    create_science_frame,
)
from fits_tools.simulation.noise import (
    add_gaussian_noise,
    add_noise_batch,
    add_poisson_noise,
)


def test_simulation_modules_import_and_core_noise_is_reproducible() -> None:
    data = np.zeros((3, 3), dtype=np.float32)

    first = add_read_noise(data, read_noise=2.0, seed=42)
    second = add_read_noise(data, read_noise=2.0, seed=42)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, data)


def test_frame_generators_are_reproducible_with_seed_zero(
    tmp_path: Path,
) -> None:
    generators = [
        (create_bias_frame, {}),
        (create_dark_frame, {"num_hot_pixels": 1}),
        (create_flat_frame, {"dust_specs": 1}),
        (
            create_science_frame,
            {"num_stars": 2, "num_cosmic_rays": 1},
        ),
    ]

    for index, (generator, kwargs) in enumerate(generators):
        first = tmp_path / f"frame_{index}_a.fits"
        second = tmp_path / f"frame_{index}_b.fits"
        generator(str(first), shape=(48, 48), seed=0, **kwargs)
        generator(str(second), shape=(48, 48), seed=0, **kwargs)

        assert np.array_equal(fits.getdata(first), fits.getdata(second))
        assert fits.getdata(first).shape == (48, 48)


def test_simulation_collection_generators_create_expected_products(
    tmp_path: Path,
) -> None:
    calibration_files = create_calibration_set(
        str(tmp_path / "calibration"),
        num_bias=1,
        num_dark=1,
        num_flat=1,
        shape=(6, 7),
        read_noise=0.0,
        seed=42,
    )
    assert {key: len(paths) for key, paths in calibration_files.items()} == {
        "bias": 1,
        "dark": 1,
        "flat": 1,
    }
    assert all(
        Path(path).exists() for paths in calibration_files.values() for path in paths
    )

    mef_path = tmp_path / "mef_bias.fits"
    create_mef_bias_frame(
        str(mef_path),
        num_amps=2,
        amp_shape=(5, 6),
        seed=0,
    )
    with fits.open(mef_path) as hdul:
        assert len(hdul) == 3
        assert [hdu.name for hdu in hdul[1:]] == ["AMP1", "AMP2"]
        assert all(hdu.data.shape == (5, 6) for hdu in hdul[1:])

    series = create_bias_time_series(
        str(tmp_path / "series"),
        num_frames=3,
        shape=(5, 5),
        thermal_cycle_period=2,
        seed=0,
    )
    assert len(series) == 3

    pairs = create_bias_pair_for_noise_estimation(
        str(tmp_path / "pairs"),
        num_pairs=2,
        shape=(5, 5),
        seed=0,
    )
    assert len(pairs["frame1"]) == len(pairs["frame2"]) == 2
    assert not np.array_equal(
        fits.getdata(pairs["frame1"][0]),
        fits.getdata(pairs["frame2"][0]),
    )


def test_science_generator_supports_small_images(tmp_path: Path) -> None:
    output = tmp_path / "small_science.fits"
    create_science_frame(
        str(output),
        shape=(8, 9),
        num_stars=2,
        num_cosmic_rays=0,
        seed=0,
    )

    assert fits.getdata(output).shape == (8, 9)


def test_noise_wrappers_use_requested_hdu(tmp_path: Path) -> None:
    source = tmp_path / "source.fits"
    gaussian_output = tmp_path / "gaussian.fits"
    poisson_output = tmp_path / "poisson.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.full((3, 3), 10.0, dtype=np.float32)),
        ]
    ).writeto(source)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        add_gaussian_noise(source, gaussian_output, sigma=1.0, hdu=1, seed=4)
        add_poisson_noise(source, poisson_output, gain=2.0, hdu=1, seed=4)

    assert fits.getdata(gaussian_output).shape == (3, 3)
    assert fits.getdata(poisson_output).shape == (3, 3)


def test_noise_batch_processes_selected_files(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    for index in range(2):
        fits.PrimaryHDU(data=np.full((3, 3), 10.0 + index, dtype=np.float32)).writeto(
            inputs / f"input_{index}.fits"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        output_files = add_noise_batch(
            str(inputs),
            str(outputs / "{stem}_noisy.fits"),
            noise_type="gaussian",
            num_workers=1,
            seed=0,
            sigma=1.0,
        )

    assert len(output_files) == 2
    assert all(Path(path).exists() for path in output_files)
