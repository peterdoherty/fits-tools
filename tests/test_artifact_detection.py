#!/usr/bin/env python3
"""
Tests for per-frame artifact detection.
"""

import numpy as np
import pytest
from fits_tools.artifacts.detection import (
    detect_saturated_pixels,
    detect_cosmic_rays,
    detect_bloom,
    detect_statistical_outliers,
    flag_outliers,
)
from fits_tools.bitmask import (
    BPM_BIT_SATURATED,
    BPM_BIT_COSMIC,
    BPM_BIT_BLOOM,
    BPM_BIT_OUTLIER,
)


def test_saturated_pixels_16bit():
    """Test saturation detection for 16-bit ADC."""
    frame = np.ones((100, 100)) * 30000.0
    
    # Add saturated pixels
    frame[10, 10] = 65535  # Max value
    frame[20, 20] = 62500  # Above 95% threshold
    frame[30, 30] = 60000  # Below threshold
    
    dq, stats = detect_saturated_pixels(frame, bit_depth=16, saturation_threshold=0.95)
    
    assert dq[10, 10] & BPM_BIT_SATURATED
    assert dq[20, 20] & BPM_BIT_SATURATED
    assert not (dq[30, 30] & BPM_BIT_SATURATED)
    
    assert stats['num_saturated'] == 2
    assert stats['saturation_level'] == pytest.approx(62270.25)  # 0.95 * 65535


def test_cosmic_ray_sigma_clip():
    """Test sigma-clip cosmic ray detection."""
    np.random.seed(42)
    frame = np.random.randn(200, 200) * 10 + 100
    
    # Add cosmic rays (sharp spikes)
    frame[50, 50] = 1000
    frame[75, 75] = 800
    frame[100:103, 100] = 500  # Small trail
    
    dq, stats = detect_cosmic_rays(
        frame,
        method='sigma_clip',
        sigma=5.0,
        box_size=5,
        iterations=3
    )
    
    assert dq[50, 50] & BPM_BIT_COSMIC
    assert dq[75, 75] & BPM_BIT_COSMIC
    assert stats['method'] == 'sigma_clip'
    assert stats['num_cosmic_rays'] >= 5  # At least the planted ones


@pytest.mark.skipif(
    not hasattr(detect_cosmic_rays, '_LACOSMIC_AVAILABLE') or
    not detect_cosmic_rays._LACOSMIC_AVAILABLE,
    reason="astroscrappy not installed"
)
def test_cosmic_ray_lacosmic():
    """Test L.A.Cosmic detection (requires astroscrappy)."""
    np.random.seed(42)
    frame = np.random.randn(200, 200) * 5 + 100
    
    # Add cosmic rays
    frame[50, 50] = 500
    frame[75, 75] = 400
    
    dq, stats = detect_cosmic_rays(
        frame,
        method='lacosmic',
        sigclip=4.5,
        gain=1.0,
        readnoise=5.0
    )
    
    assert stats['method'] == 'lacosmic'
    assert stats['num_cosmic_rays'] >= 2


def test_bloom_detection_vertical():
    """Test vertical blooming detection."""
    frame = np.ones((200, 200)) * 100.0
    
    # Create a saturated star with vertical bloom trail
    saturation = 65535.0
    frame[100, 100] = saturation  # Saturated center
    
    # Vertical bloom trail (decreasing brightness)
    for offset in range(1, 20):
        brightness = saturation * 0.5 * np.exp(-offset / 5.0)
        frame[100 - offset, 100] = brightness  # Upward
        frame[100 + offset, 100] = brightness  # Downward
    
    dq, stats = detect_bloom(
        frame,
        saturation_level=60000,
        bleed_direction='vertical',
        min_trail_length=5,
        trail_threshold_factor=0.1
    )
    
    # Check that bloom pixels are flagged
    assert stats['num_saturated_seeds'] == 1
    assert stats['num_bloom_pixels'] > 10  # Should detect most of the trail
    
    # Verify bloom pixels along the column
    bloom_column = (dq[:, 100] & BPM_BIT_BLOOM) != 0
    assert np.sum(bloom_column) > 10


def test_bloom_no_saturation():
    """Test that bloom detection returns empty when no saturation."""
    frame = np.ones((100, 100)) * 1000.0  # All below saturation
    
    dq, stats = detect_bloom(frame, saturation_level=50000)
    
    assert stats['num_saturated_seeds'] == 0
    assert stats['num_bloom_pixels'] == 0
    assert np.all(dq == 0)


def test_statistical_outliers():
    """Test statistical outlier detection."""
    np.random.seed(42)
    frame = np.random.randn(100, 100) * 10 + 100
    
    # Add outliers
    frame[25, 25] = 500  # High outlier
    frame[75, 75] = -200  # Low outlier
    
    dq, stats = detect_statistical_outliers(frame, sigma=5.0)
    
    assert dq[25, 25] & BPM_BIT_OUTLIER
    assert dq[75, 75] & BPM_BIT_OUTLIER
    assert stats['num_outliers'] >= 2


def test_flag_outliers_wrapper():
    """Test the flag_outliers wrapper function."""
    outlier_mask = np.zeros((50, 50), dtype=bool)
    outlier_mask[10, 10] = True
    outlier_mask[20:25, 30] = True
    
    dq, stats = flag_outliers(outlier_mask)
    
    assert dq[10, 10] & BPM_BIT_OUTLIER
    assert np.all((dq[20:25, 30] & BPM_BIT_OUTLIER) != 0)
    assert stats['num_outliers'] == 6


def test_existing_bpm_exclusion():
    """Test that existing_bpm excludes pixels from CR statistics but not detection."""
    np.random.seed(42)
    frame = np.random.randn(100, 100) * 10 + 100
    
    # Create existing BPM with hot pixels
    existing_bpm = np.zeros((100, 100), dtype=np.uint32)
    existing_bpm[10:20, 10:20] = 1  # Hot region
    
    # Add cosmic ray in the hot region
    frame[15, 15] = 1000
    
    dq, stats = detect_cosmic_rays(
        frame,
        method='sigma_clip',
        existing_bpm=existing_bpm,
        sigma=5.0
    )
    
    # The CR should still be detected even though it's in a hot region
    assert dq[15, 15] & BPM_BIT_COSMIC


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
    