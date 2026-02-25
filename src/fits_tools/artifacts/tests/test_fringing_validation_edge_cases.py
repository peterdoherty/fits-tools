def test_fringing_validation_edge_cases():
    """Test fringing validation with edge cases."""
    import numpy as np
    import pytest
    from fits_tools.artifacts.detection import detect_fringing
    
    print("Testing fringing validation edge cases")
    print("=" * 70)
    
    flat = np.ones((512, 512)) * 1000.0
    
    # Test 1: max_period = image dimension (should NOW work)
    print("\nTest 1: max_period exactly equals image dimension")
    try:
        dq, stats = detect_fringing(flat, fringe_period_range=(5, 512))
        print(f"  ✓ Accepted max_period=512 for 512×512 image")
        print(f"    This is correct - one full cycle is detectable")
    except ValueError as e:
        print(f"  ✗ Incorrectly rejected: {e}")
        raise
    
    # Test 2: max_period > image dimension (should still fail)
    print("\nTest 2: max_period exceeds image dimension")
    with pytest.raises(ValueError, match="exceeds minimum image dimension"):
        dq, stats = detect_fringing(flat, fringe_period_range=(5, 513))
    print(f"  ✓ Correctly rejected max_period=513 > 512")
    
    # Test 3: min_period = 1 (Nyquist violation, should fail with helpful message)
    print("\nTest 3: min_period violates Nyquist (enhanced error message)")
    try:
        dq, stats = detect_fringing(flat, fringe_period_range=(1, 10))
        print(f"  ✗ Should have rejected min_period=1")
        assert False
    except ValueError as e:
        error_msg = str(e)
        print(f"  ✓ Correctly rejected min_period=1")
        assert "Nyquist-Shannon" in error_msg
        assert "detect_fixed_pattern_noise" in error_msg
        print(f"    Error message includes educational content ✓")
        print(f"    Error message suggests alternative (FPN detector) ✓")
    
    # Test 4: Rectangular image (min dimension matters)
    print("\nTest 4: Rectangular image (1024×512)")
    flat_rect = np.ones((1024, 512)) * 1000.0
    
    # max_period=512 should work (equals min dimension)
    try:
        dq, stats = detect_fringing(flat_rect, fringe_period_range=(5, 512))
        print(f"  ✓ Accepted max_period=512 for 1024×512 image (min=512)")
    except ValueError as e:
        print(f"  ✗ Incorrectly rejected: {e}")
        raise
    
    # max_period=513 should fail
    with pytest.raises(ValueError, match="exceeds minimum image dimension"):
        dq, stats = detect_fringing(flat_rect, fringe_period_range=(5, 513))
    print(f"  ✓ Correctly rejected max_period=513 > min_dim=512")
    
    print("\n" + "=" * 70)
    print("✓✓✓ All fringing validation tests passed!")


if __name__ == '__main__':
    test_fringing_validation_edge_cases()