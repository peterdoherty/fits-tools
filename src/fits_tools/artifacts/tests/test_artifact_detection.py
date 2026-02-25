import numpy as np
from fits_tools.artifacts.detection import *

frame = np.zeros((32, 32), dtype=float)

def test_saturated_pixels():
    f = frame.copy()
    f[10, 10] = 65535
    dq, stats = detect_saturated_pixels(f, bit_depth=16)
    assert (dq[10, 10] & BPM_BIT_SATURATED) != 0
    assert stats['num_saturated'] == 1

def test_cosmic_sigma_clip_zero_noise():
    f = np.full((32, 32), 1000.0)
    f[5, 5] = 2000.0
    dq, stats = detect_cosmic_rays(f, method='sigma_clip', sigma=5.0)
    assert (dq[5, 5] & BPM_BIT_COSMIC) != 0

def test_bloom_simple():
    f = np.zeros((32, 32))
    f[10, 10:13] = 65535  # saturated block
    f[11:20, 11] = 5000   # trail
    dq, stats = detect_bloom(f, bit_depth=16)
    assert stats['num_trails'] >= 1

def test_persistence_stats_shape():
    a = np.zeros((32, 32))
    b = np.zeros((32, 32))
    dq, stats = detect_persistence(a, b, threshold_sigma=3.0)
    # Ensure key always present after your fix:
    assert 'num_edge_pixels_masked' in stats

def test_statistical_outliers_global():
    f = np.zeros((32, 32))
    f[0, 0] = 10.0
    dq, stats = detect_statistical_outliers(f, sigma=3.0)
    assert stats['num_outliers'] >= 1

def test_thermal_gradient_runs():
    f = np.tile(np.linspace(0, 100, 32), (32, 1)).astype(float)
    dq, stats = detect_thermal_gradient(f)
    assert 'gradient_detected' in stats
    assert 'downsample_factor' in stats

def test_fringing_period_guard():
    try:
        detect_fringing(np.ones((64, 64)), fringe_period_range=(1, 10))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for min_period < 2")

def test_detect_all_and_report(tmp_path):
    f = np.zeros((64, 64))
    bpm = np.zeros_like(f, dtype=np.uint32)
    dq, stats = detect_all_artifacts(f, existing_bpm=bpm)
    assert 'saturation' in stats
    out = tmp_path / "report.pdf"
    create_artifact_report(f, dq, stats, output_path=str(out), show_plots=False)
    assert out.is_file()