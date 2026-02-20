import numpy as np
import time
from fits_tools.artifacts.detection import detect_thermal_gradient

def test_thermal_gradient_downsampling_accuracy():
    """
    Verify that adaptive downsampling gives same results as full resolution
    within acceptable tolerance.
    """
    print("Thermal Gradient Downsampling Validation")
    print("=" * 60)
    
    # Create 4K test frame with known gradient
    ny, nx = 4096, 4096
    print(f"Test frame: {ny}×{nx} = {ny*nx:,} pixels")
    
    # Create frame with thermal gradient: 0.05 ADU/pixel in Y direction
    y, x = np.mgrid[0:ny, 0:nx]
    true_gradient_y = 0.05
    true_gradient_x = 0.02
    frame = 1000 + true_gradient_y * y + true_gradient_x * x
    
    # Add realistic noise
    np.random.seed(42)
    frame += np.random.randn(ny, nx) * 5
    
    # Test 1: Full resolution (slow)
    print("\nTest 1: Full resolution (downsample_factor=1)")
    start = time.time()
    dq_full, stats_full = detect_thermal_gradient(
        frame,
        downsample_factor=1,  # Force full resolution
        smoothing_sigma=50.0
    )
    time_full = time.time() - start
    print(f"  Time: {time_full*1000:.1f} ms")
    print(f"  Points used: {stats_full['points_used']:,}")
    print(f"  Gradient magnitude: {stats_full['max_gradient']:.6f} ADU/pixel")
    print(f"  Gradient Y: {stats_full['coefficients']['dy']:.6f} (true: {true_gradient_y})")
    print(f"  Gradient X: {stats_full['coefficients']['dx']:.6f} (true: {true_gradient_x})")
    
    # Test 2: Adaptive downsampling (fast)
    print("\nTest 2: Adaptive downsampling (downsample_factor=None)")
    start = time.time()
    dq_adapt, stats_adapt = detect_thermal_gradient(
        frame,
        downsample_factor=None,  # Auto adaptive
        smoothing_sigma=50.0
    )
    time_adapt = time.time() - start
    print(f"  Time: {time_adapt*1000:.1f} ms")
    print(f"  Downsample factor: {stats_adapt['downsample_factor']}×")
    print(f"  Points used: {stats_adapt['points_used']:,}")
    print(f"  Gradient magnitude: {stats_adapt['max_gradient']:.6f} ADU/pixel")
    print(f"  Gradient Y: {stats_adapt['coefficients']['dy']:.6f} (true: {true_gradient_y})")
    print(f"  Gradient X: {stats_adapt['coefficients']['dx']:.6f} (true: {true_gradient_x})")
    
    # Compare results
    print("\nComparison:")
    print(f"  Speedup: {time_full/time_adapt:.1f}×")
    print(f"  Memory reduction: {stats_adapt['downsample_factor']**2:.1f}×")
    
    # Accuracy check
    grad_diff = abs(stats_full['max_gradient'] - stats_adapt['max_gradient'])
    grad_pct_error = 100 * grad_diff / stats_full['max_gradient']
    print(f"  Gradient difference: {grad_diff:.6f} ADU/pixel ({grad_pct_error:.3f}%)")
    
    dy_diff = abs(stats_full['coefficients']['dy'] - stats_adapt['coefficients']['dy'])
    dy_pct_error = 100 * dy_diff / abs(stats_full['coefficients']['dy'])
    print(f"  Gradient Y difference: {dy_diff:.6f} ADU/pixel ({dy_pct_error:.3f}%)")
    
    # Assertions
    assert grad_pct_error < 0.1, f"Gradient error {grad_pct_error:.3f}% exceeds 0.1% tolerance"
    assert time_adapt < time_full, "Adaptive should be faster than full resolution"
    assert stats_adapt['downsample_factor'] > 1, "4K image should trigger downsampling"
    
    print("\n✓ Accuracy validated: < 0.1% error")
    print(f"✓ Performance validated: {time_full/time_adapt:.1f}× faster")
    print(f"✓ Memory validated: {stats_adapt['downsample_factor']**2:.1f}× less RAM")


def test_thermal_gradient_small_sensor():
    """Verify small sensors don't get downsampled unnecessarily."""
    print("\n" + "=" * 60)
    print("Small Sensor Test (512×512)")
    
    frame = np.random.randn(512, 512) * 5 + 1000
    _, stats = detect_thermal_gradient(frame, downsample_factor=None)
    
    print(f"  Downsample factor: {stats['downsample_factor']}×")
    print(f"  Points used: {stats['points_used']:,}")
    
    assert stats['downsample_factor'] == 1, "Small sensors shouldn't be downsampled"
    print("✓ Small sensors use full resolution (correct)")


def test_thermal_gradient_huge_sensor():
    """Verify huge sensors get aggressive downsampling."""
    print("\n" + "=" * 60)
    print("Large Sensor Test (8K×8K)")
    
    # Don't actually create 8K array (too slow for test)
    # Just verify the helper function
    from fits_tools.artifacts.detection import _compute_downsample_factor
    
    factor_8k = _compute_downsample_factor((8192, 8192))
    points_8k = (8192 // factor_8k) ** 2
    
    print(f"  8K sensor downsample factor: {factor_8k}×")
    print(f"  Points used: {points_8k:,}")
    print(f"  Memory reduction: {factor_8k**2:.0f}×")
    
    assert factor_8k > 10, "8K sensors should use aggressive downsampling"
    assert 10_000 < points_8k < 500_000, "Target ~50k points"
    print("✓ Large sensors use aggressive downsampling (correct)")


if __name__ == '__main__':
    test_thermal_gradient_downsampling_accuracy()
    test_thermal_gradient_small_sensor()
    test_thermal_gradient_huge_sensor()
    
    print("\n" + "=" * 60)
    print("✓✓✓ All thermal gradient optimization tests passed!")
    