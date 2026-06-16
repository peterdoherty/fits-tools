#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import logging

import pytest

from fits_tools.constants import (
    FITS_VAL_ARC_IMGTYPE,
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_COMPARISON_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
    FITS_VAL_OBJECT_IMGTYPE,
    FITS_VAL_SCIENCE_IMGTYPE,
    FITS_VAL_SNAPSHOT_IMGTYPE,
    FITS_VAL_TEST_IMGTYPE,
)
from fits_tools.image_types import is_valid_image_type, normalize_image_type


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("bias frame", FITS_VAL_BIAS_IMGTYPE),
        ("ZERO", FITS_VAL_BIAS_IMGTYPE),
        ("dark current", FITS_VAL_DARK_IMGTYPE),
        ("flat field", FITS_VAL_FLAT_IMGTYPE),
        ("exposure", FITS_VAL_LIGHT_IMGTYPE),
        ("object frame", FITS_VAL_OBJECT_IMGTYPE),
        ("science", FITS_VAL_SCIENCE_IMGTYPE),
        ("arc lamp", FITS_VAL_ARC_IMGTYPE),
        ("comp lamp", FITS_VAL_COMPARISON_IMGTYPE),
        ("test frame", FITS_VAL_TEST_IMGTYPE),
        ("snap", FITS_VAL_SNAPSHOT_IMGTYPE),
    ],
)
def test_normalize_image_type_maps_real_world_variants(raw: str, expected: str) -> None:
    assert normalize_image_type(raw) == expected


def test_normalize_image_type_handles_none_empty_and_unrecognized(caplog: pytest.LogCaptureFixture) -> None:
    assert normalize_image_type(None) is None
    assert normalize_image_type("   ") is None
    assert normalize_image_type("unknown") is None

    with caplog.at_level(logging.WARNING):
        assert normalize_image_type("unknown", strict=True) is None

    assert "unrecognized image type" in caplog.text


def test_is_valid_image_type_is_canonical_only() -> None:
    assert is_valid_image_type(FITS_VAL_BIAS_IMGTYPE)
    assert not is_valid_image_type("bias")
    assert not is_valid_image_type(None)
