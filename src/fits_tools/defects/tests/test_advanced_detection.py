#!/usr/bin/env python3
"""
Tests for advanced defect detection algorithms.
"""

import numpy as np
import pytest
from fits_tools.defects.detection import (
    detect_nonlinear_pixels,
    detect_noisy_pixels,
    detect_rts_pixels,
    detect_charge_traps,
    detect_amplifier_glow,
    _detect_nonlinear_pixels_impl,
    _detect_noisy_pixels_impl,
    run_advanced_defect_detection,
)
from fits_tools.bitmask import (
    BPM_BIT_PIX_NONLIN,
    BPM_BIT_PIX_NOISE,
    BPM_BIT_PIX_RTS,
    BPM_BIT_PIX_TRAP,
    BPM_BIT_PIX_GLOW,
)


def test_placeholder_functions_return_empty():
    """Test that placeholder stubs return empty BPMs with warnings."""
    # Nonlinearity
    linearity_data = np.random.randn(5, 100, 100) * 100 + 1000
    exposure_times = np.array([1, 2, 3, 4, 5])
    bpm = detect_nonlinear_pixels(linearity_data, exposure_times)
    assert np.all(bpm == 0), "Placeholder should return empty BPM"
    
    # Noise
    bias_stack = np.random.randn(10, 100, 100) * 5 + 100
    bpm = detect_noisy_pixels(bias_stack)
    assert np.all(bpm == 0)
    
    # RTS
    dark_seq = np.random.randn(50, 100, 100) * 10 + 50
    bpm = detect_rts_pixels(dark_seq)
    assert np.all(bpm == 0)


def test_nonlinear_detection_draft():
    """Test draft nonlinearity detection with synthetic data."""
    ny, nx = 100, 100
    n_exp = 5
    exposure_times = np.linspace(1, 10, n_exp)
    
    # Create perfect linear response
    linearity_data = np.zeros((n_exp, ny, nx))
    for i, t in enumerate(exposure_times):
        linearity_data[i] = 1000 * t  # Perfect linearity
    
    # Add a nonlinear pixel at (50, 50)
    # Introduce quadratic component
    for i, t in enumerate(exposure_times):
        linearity_data[i, 50, 50] = 1000 * t + 50 * t**2  # 5% nonlinearity at t=10
    
    bpm = _detect_nonlinear_pixels_impl(
        linearity_data, 
        exposure_times, 
        threshold=0.03  # 3%
    )
    
    # Should detect the nonlinear pixel
    assert bpm[50, 50] & BPM_BIT_PIX_NONLIN
    # Should not flag linear pixels
    assert not (bpm[0, 0] & BPM_BIT_PIX_NONLIN)


def test_noisy_pixel_detection_draft():
    """Test draft noise detection with synthetic data."""
    ny, nx = 100, 100
    n_frames = 20
    
    # Create bias stack with normal noise
    bias_stack = np.random.randn(n_frames, ny, nx) * 5 + 100
    
    # Add a noisy pixel at (30, 40)
    bias_stack[:, 30, 40] += np.random.randn(n_frames) * 50  # 10x noisier
    
    bpm = _detect_noisy_pixels_impl(bias_stack, threshold_sigma=5.0)
    
    # Should detect the noisy pixel
    assert bpm[30, 40] & BPM_BIT_PIX_NOISE
    # Most normal pixels should be fine
    flagged = np.sum(bpm & BPM_BIT_PIX_NOISE)
    assert flagged < 10, f"Too many pixels flagged: {flagged}"


def test_rts_detection_synthetic():
    """Test RTS detection with synthetic telegraph signal."""
    ny, nx = 50, 50
    n_frames = 100
    
    # Create dark sequence with normal noise
    dark_seq = np.random.randn(n_frames, ny, nx) * 5 + 50
    
    # Add an RTS pixel at (25, 25) with discrete state switching
    rts_signal = np.zeros(n_frames)
    state = 0
    for i in range(n_frames):
        if i % 10 == 0:  # Switch every 10 frames
            state = 1 - state
        rts_signal[i] = 50 if state == 0 else 150  # 100 ADU amplitude
    
    dark_seq[:, 25, 25] = rts_signal + np.random.randn(n_frames) * 2
    
    from fits_tools.defects.detection import _detect_rts_pixels_impl
    bpm = _detect_rts_pixels_impl(
        dark_seq, 
        threshold_transitions=3, 
        min_dwell_frames=5,
        amplitude_threshold=3.0
    )
    
    # Should detect the RTS pixel
    assert bpm[25, 25] & BPM_BIT_PIX_RTS, "Failed to detect RTS pixel"


def test_integration_run_advanced():
    """Test the integration function with minimal data."""
    ny, nx = 100, 100
    
    sensor_data = {
        'linearity_sequence': (
            np.random.randn(5, ny, nx) * 100 + 1000,
            np.array([1, 2, 3, 4, 5])
        ),
        'bias_stack': np.random.randn(10, ny, nx) * 5 + 100,
        'dark_sequence': np.random.randn(20, ny, nx) * 10 + 50,
        'long_dark': np.random.randn(ny, nx) * 20 + 100,
    }
    
    # Test with placeholders (should return empty BPMs with warnings)
    bpm = run_advanced_defect_detection(
        sensor_data, 
        use_draft_implementations=False
    )
    
    assert bpm.shape == (ny, nx)
    assert bpm.dtype == np.uint32


def test_dead_pixel_detection():
    """Test that dead pixels are properly detected from flat."""
    ny, nx = 100, 100
    flat = np.ones((ny, nx)) * 1000.0
    
    # Add dead pixels (zero response)
    flat[10, 10] = 0
    flat[20, 20] = 5  # Very low but not quite zero
    
    from fits_tools.defects.detection import detect_dead_pixels
    from fits_tools.bitmask import BPM_BIT_PIX_DEAD
    
    bpm, stats = detect_dead_pixels(flat, threshold_percent=10.0)
    
    # Should detect both
    assert bpm[10, 10] & BPM_BIT_PIX_DEAD
    assert bpm[20, 20] & BPM_BIT_PIX_DEAD
    
    # Should not flag normal pixels
    assert not (bpm[50, 50] & BPM_BIT_PIX_DEAD)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])