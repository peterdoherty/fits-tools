from __future__ import annotations

import numpy as np

from fits_tools.utils import ensure_extension, is_fits_file, prepare_for_json


def test_prepare_for_json_converts_numpy_scalars_and_nested_values() -> None:
    payload = {
        1: np.int32(5),
        "float": np.float64(1.25),
        "array": np.array([1, 2, 3], dtype=np.int16),
        "tuple": (np.bool_(True), complex(2, 3)),
        "bad_float": np.inf,
    }

    result = prepare_for_json(payload)

    assert result == {
        "1": 5,
        "float": 1.25,
        "array": [1, 2, 3],
        "tuple": [True, {"real": 2.0, "imag": 3.0}],
        "bad_float": None,
    }


def test_prepare_for_json_maps_nan_to_none() -> None:
    assert prepare_for_json(np.nan) is None


def test_is_fits_file_is_case_insensitive_and_supports_compression() -> None:
    assert is_fits_file("image.FITS")
    assert is_fits_file("image.fits.gz")
    assert is_fits_file("image.fit")
    assert not is_fits_file("image.txt")
    assert not is_fits_file(None)


def test_ensure_extension_appends_only_when_missing() -> None:
    assert str(ensure_extension("image")) == "image.fits"
    assert str(ensure_extension("image.fits")) == "image.fits"
    assert str(ensure_extension("notes.txt")) == "notes.txt"
