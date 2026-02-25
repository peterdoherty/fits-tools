from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import ImageProcessingError
from fits_tools.quality_control import (
    QualityThresholds,
    assess_quality,
    assess_quality_array,
)
from fits_tools.validate import ValidationResult


def test_assess_quality_array_default_filepath_is_path():
    data = np.ones((10, 10), dtype=float)
    header = fits.Header()

    metrics = assess_quality_array(data=data, header=header)

    assert metrics.filepath.name == "in-memory"


def test_assess_quality_array_nan_policy_ignores_nonfinite():
    data = np.array(
        [
            [1.0, 2.0, np.nan],
            [3.0, np.inf, 5.0],
        ],
        dtype=float,
    )
    header = fits.Header()

    metrics = assess_quality_array(data=data, header=header, compute_snr=False)

    # finite values are: 1, 2, 3, 5
    assert metrics.total_pixels == 4
    assert metrics.mean == pytest.approx(2.75)
    assert any("Ignored" in w for w in metrics.warnings)


def test_assess_quality_array_bpm_outlier_count_consistent():
    data = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1000.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    bpm = np.zeros_like(data, dtype=np.uint16)
    bpm[0, 0] = 1  # masked bad pixel
    header = fits.Header()

    metrics = assess_quality_array(data=data, header=header, bpm=bpm, compute_snr=False)

    assert metrics.total_pixels == 8
    assert 0 <= metrics.num_outliers <= metrics.total_pixels
    assert 0.0 <= metrics.outlier_fraction <= 1.0


def test_assess_quality_preserves_image_processing_error(monkeypatch):
    def _fake_quick_validate(filepath, frame_type=None):
        return ValidationResult(is_valid=True)

    def _fake_get_data_and_header(filepath, ext=0):
        return None, fits.Header()

    monkeypatch.setattr("fits_tools.quality_control.quick_validate", _fake_quick_validate)
    monkeypatch.setattr(
        "fits_tools.quality_control.get_data_and_header", _fake_get_data_and_header
    )

    with pytest.raises(ImageProcessingError, match="Could not load data from"):
        assess_quality("dummy.fits")


def test_assess_quality_region_none_raises_clear_error(monkeypatch):
    def _fake_quick_validate(filepath, frame_type=None):
        return ValidationResult(is_valid=True)

    def _fake_get_data_and_header(filepath, ext=0):
        return np.ones((4, 4), dtype=float), fits.Header()

    def _fake_get_data(filepath, ext=0, region="full"):
        return None

    monkeypatch.setattr("fits_tools.quality_control.quick_validate", _fake_quick_validate)
    monkeypatch.setattr(
        "fits_tools.quality_control.get_data_and_header", _fake_get_data_and_header
    )
    monkeypatch.setattr("fits_tools.quality_control.get_data", _fake_get_data)

    with pytest.raises(ImageProcessingError, match="Could not load region data"):
        assess_quality("dummy.fits", region="science")


def test_quality_thresholds_are_frozen_and_validate():
    t = QualityThresholds()
    with pytest.raises(FrozenInstanceError):
        t.min_snr = 7.0

    with pytest.raises(ValueError, match="max_outlier_fraction"):
        QualityThresholds(max_outlier_fraction=1.5)

