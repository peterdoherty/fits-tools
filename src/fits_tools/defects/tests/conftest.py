# tests/conftest.py

"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)

@pytest.fixture
def small_shape():
    """Small array size for fast tests."""
    return (100, 100)

@pytest.fixture
def realistic_shape():
    """Realistic detector size (marked as slow)."""
    return (2048, 2048)

@pytest.fixture
def clean_bias(small_shape):
    """Clean bias frame with Gaussian noise."""
    ny, nx = small_shape
    return np.random.normal(1000.0, 5.0, (ny, nx))

@pytest.fixture
def clean_dark(small_shape):
    """Clean dark frame with Poisson noise."""
    ny, nx = small_shape
    return np.random.poisson(10, (ny, nx)).astype(float)

@pytest.fixture
def clean_flat(small_shape):
    """Clean flat frame with uniform illumination."""
    ny, nx = small_shape
    return np.random.normal(10000.0, 100.0, (ny, nx))

@pytest.fixture
def defective_bias(small_shape):
    """Bias frame with known defects."""
    ny, nx = small_shape
    bias = np.random.normal(1000.0, 5.0, (ny, nx))
    
    # Add known defects
    bias[25, 25] = 1200  # Hot bias pixel
    bias[75, 75] = 800   # Cold bias pixel
    
    return bias

@pytest.fixture
def defective_dark(small_shape):
    """Dark frame with hot pixels."""
    ny, nx = small_shape
    dark = np.random.poisson(10, (ny, nx)).astype(float)
    
    # Hot pixels
    dark[30, 30] = 500
    dark[60, 60] = 1000
    
    return dark

@pytest.fixture
def defective_flat(small_shape):
    """Flat frame with hot and cold pixels."""
    ny, nx = small_shape
    flat = np.random.normal(10000.0, 100.0, (ny, nx))
    
    # Hot pixel (25% above median)
    flat[40, 40] = 12500
    
    # Dead pixels
    flat[50, 50] = 500
    flat[60, 60] = 100
    
    return flat

@pytest.fixture
def calibration_set(clean_bias, clean_dark, clean_flat):
    """Complete set of clean calibration frames."""
    return {
        'bias': clean_bias,
        'dark': clean_dark,
        'flat': clean_flat,
    }

@pytest.fixture
def defective_calibration_set(defective_bias, defective_dark, defective_flat):
    """Complete set of defective calibration frames."""
    return {
        'bias': defective_bias,
        'dark': defective_dark,
        'flat': defective_flat,
    }

@pytest.fixture(scope="session")
def benchmark_data_path():
    """Path to benchmark dataset (if exists)."""
    path = Path('benchmarks/benchmark_data')
    if path.exists():
        return path
    pytest.skip("Benchmark data not available")