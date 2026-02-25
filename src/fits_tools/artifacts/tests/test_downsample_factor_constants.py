def test_downsample_factor_constants():
    """Verify downsampling constants are reasonable and documented."""
    from fits_tools.artifacts.detection import (
        _compute_downsample_factor,
        _THERMAL_FIT_TARGET_POINTS,
        _THERMAL_FIT_MAX_DOWNSAMPLE,
    )
    
    print("Testing Thermal Gradient Downsampling Constants")
    print("=" * 60)
    
    # Test 1: Constants are accessible
    print(f"\nModule constants:")
    print(f"  Target points: {_THERMAL_FIT_TARGET_POINTS:,}")
    print(f"  Max downsample factor: {_THERMAL_FIT_MAX_DOWNSAMPLE}")
    
    assert _THERMAL_FIT_TARGET_POINTS == 50_000, "Target should be 50k"
    assert _THERMAL_FIT_MAX_DOWNSAMPLE == 21, "Max should be 21"
    print("  ✓ Constants defined correctly")
    
    # Test 2: Small sensors → no downsampling
    print("\nTest 2: Small sensor (512×512)")
    factor = _compute_downsample_factor((512, 512))
    print(f"  Downsample factor: {factor}×")
    assert factor == 1, "Small sensors shouldn't downsample"
    print("  ✓ No downsampling for small sensors")
    
    # Test 3: Medium sensors → moderate downsampling
    print("\nTest 3: Medium sensor (2048×2048)")
    factor = _compute_downsample_factor((2048, 2048))
    points = (2048 // factor) ** 2
    print(f"  Downsample factor: {factor}×")
    print(f"  Points used: {points:,}")
    assert factor > 1, "Should downsample 2K sensors"
    assert factor % 2 == 1, "Should use odd factor"
    assert 10_000 < points < 500_000, "Should be near target"
    print("  ✓ Moderate downsampling applied")
    
    # Test 4: Large sensors → aggressive downsampling
    print("\nTest 4: Large sensor (4096×4096)")
    factor = _compute_downsample_factor((4096, 4096))
    points = (4096 // factor) ** 2
    print(f"  Downsample factor: {factor}×")
    print(f"  Points used: {points:,}")
    assert factor > 5, "Should aggressively downsample 4K sensors"
    assert factor % 2 == 1, "Should use odd factor"
    print("  ✓ Aggressive downsampling applied")
    
    # Test 5: Huge sensors → capped at max
    print("\nTest 5: Huge sensor (16384×16384)")
    factor = _compute_downsample_factor((16384, 16384))
    points = (16384 // factor) ** 2
    print(f"  Downsample factor: {factor}×")
    print(f"  Points used: {points:,}")
    assert factor == _THERMAL_FIT_MAX_DOWNSAMPLE, f"Should cap at {_THERMAL_FIT_MAX_DOWNSAMPLE}"
    print("  ✓ Capped at maximum factor")
    
    # Test 6: All factors are odd
    print("\nTest 6: Verify all factors are odd")
    for size in [1024, 2048, 3000, 4096, 6000, 8192]:
        factor = _compute_downsample_factor((size, size))
        assert factor % 2 == 1, f"Factor {factor} for {size}×{size} should be odd"
    print("  ✓ All factors are odd (good sampling)")
    
    print("\n" + "=" * 60)
    print("✓✓✓ All constant validation tests passed!")


if __name__ == '__main__':
    test_downsample_factor_constants()