def test_min_sigma_units_and_conversion():
    """Verify min_sigma units and helper function."""
    import numpy as np
    from fits_tools.artifacts.detection import detect_cosmic_rays, convert_readnoise_to_adu
    
    print("Testing min_sigma units and conversion")
    print("=" * 60)
    
    # Test 1: Unit conversion helper
    print("\nTest 1: Unit conversion")
    readnoise_e = 5.0  # electrons
    gain = 1.5  # e⁻/ADU
    
    min_sigma_adu = convert_readnoise_to_adu(readnoise_e, gain)
    expected = 5.0 / 1.5
    
    print(f"  Read noise: {readnoise_e} e⁻")
    print(f"  Gain: {gain} e⁻/ADU")
    print(f"  Converted: {min_sigma_adu:.3f} ADU")
    print(f"  Expected: {expected:.3f} ADU")
    
    assert abs(min_sigma_adu - expected) < 1e-6, "Conversion incorrect"
    print("  ✓ Conversion correct")
    
    # Test 2: Use in detection (noiseless frame)
    print("\nTest 2: Detection in noiseless frame with converted min_sigma")
    frame = np.ones((100, 100)) * 1000.0
    frame[50, 50] = 10000.0  # Strong spike
    
    dq, stats = detect_cosmic_rays(
        frame,
        method='sigma_clip',
        sigma=5.0,
        min_sigma=min_sigma_adu
    )
    
    print(f"  Cosmic rays detected: {stats['num_cosmic_rays']}")
    print(f"  min_sigma used: {stats['min_sigma']:.3f} ADU")
    
    assert stats['num_cosmic_rays'] >= 1, "Should detect the spike"
    assert abs(stats['min_sigma'] - min_sigma_adu) < 1e-6, "Stats should record min_sigma"
    print("  ✓ Detection works with converted min_sigma")
    
    # Test 3: Validation of negative/zero values
    print("\nTest 3: Validation of invalid inputs")
    try:
        convert_readnoise_to_adu(-5.0, 1.5)
        print("  ✗ Should reject negative readnoise")
        assert False
    except ValueError:
        print("  ✓ Correctly rejects negative readnoise")
    
    try:
        convert_readnoise_to_adu(5.0, 0.0)
        print("  ✗ Should reject zero gain")
        assert False
    except ValueError:
        print("  ✓ Correctly rejects zero gain")
    
    print("\n" + "=" * 60)
    print("✓✓✓ All min_sigma unit tests passed!")


if __name__ == '__main__':
    test_min_sigma_units_and_conversion()