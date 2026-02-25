def test_grow_trail_edge_cases():
    """Test bloom trail growth at array boundaries."""
    import numpy as np
    from fits_tools.artifacts.detection import _grow_trail_1d
    
    print("Testing _grow_trail_1d boundary cases")
    print("=" * 60)
    
    # Test Case 1: Left edge, grow right
    print("\nCase 1: Saturation at left edge (idx=0), grow right")
    line = np.array([65000, 40000, 30000, 20000, 100, 100], dtype=float)
    trail = _grow_trail_1d(line, start_idx=0, direction=1, threshold=15000)
    print(f"  Line: {line}")
    print(f"  Trail indices: {trail}")
    print(f"  Trail values: {line[trail]}")
    assert 0 in trail, "Should include start pixel"
    assert np.all(line[trail] >= 15000), "All trail pixels should be >= threshold"
    expected = np.array([0, 1, 2, 3])
    assert np.array_equal(trail, expected), f"Expected {expected}, got {trail}"
    print("  ✓ Passed")
    
    # Test Case 2: Left edge, try to grow left (impossible)
    print("\nCase 2: Saturation at left edge (idx=0), try to grow left")
    trail = _grow_trail_1d(line, start_idx=0, direction=-1, threshold=15000)
    print(f"  Trail indices: {trail}")
    assert len(trail) == 1 and trail[0] == 0, "Should return only start pixel"
    print("  ✓ Passed")
    
    # Test Case 3: Right edge, grow left
    print("\nCase 3: Saturation at right edge (idx=5), grow left")
    line = np.array([100, 100, 20000, 30000, 40000, 65000], dtype=float)
    trail = _grow_trail_1d(line, start_idx=5, direction=-1, threshold=15000)
    print(f"  Line: {line}")
    print(f"  Trail indices: {trail}")
    print(f"  Trail values: {line[trail]}")
    assert 5 in trail, "Should include start pixel"
    assert np.all(line[trail] >= 15000), "All trail pixels should be >= threshold"
    expected = np.array([5, 4, 3, 2])
    assert np.array_equal(trail, expected), f"Expected {expected}, got {trail}"
    print("  ✓ Passed")
    
    # Test Case 4: Right edge, try to grow right (impossible)
    print("\nCase 4: Saturation at right edge (idx=5), try to grow right")
    trail = _grow_trail_1d(line, start_idx=5, direction=1, threshold=15000)
    print(f"  Trail indices: {trail}")
    assert len(trail) == 1 and trail[0] == 5, "Should return only start pixel"
    print("  ✓ Passed")
    
    # Test Case 5: Out of bounds - negative index
    print("\nCase 5: Out of bounds start_idx=-1")
    trail = _grow_trail_1d(line, start_idx=-1, direction=1, threshold=15000)
    print(f"  Trail indices: {trail}")
    assert len(trail) == 0, "Should return empty array for out-of-bounds"
    print("  ✓ Passed")
    
    # Test Case 6: Out of bounds - beyond array
    print("\nCase 6: Out of bounds start_idx=10 (array length=6)")
    trail = _grow_trail_1d(line, start_idx=10, direction=1, threshold=15000)
    print(f"  Trail indices: {trail}")
    assert len(trail) == 0, "Should return empty array for out-of-bounds"
    print("  ✓ Passed")
    
    # Test Case 7: Single-pixel trail (all neighbors below threshold)
    print("\nCase 7: Isolated hot pixel (no trail extension)")
    line = np.array([100, 100, 65000, 100, 100], dtype=float)
    trail = _grow_trail_1d(line, start_idx=2, direction=1, threshold=15000)
    print(f"  Trail indices: {trail}")
    assert len(trail) == 1 and trail[0] == 2, "Should return only start pixel"
    print("  ✓ Passed")
    
    # Test Case 8: Trail extends to frame edge
    print("\nCase 8: Trail extends all the way to frame edge")
    line = np.array([100, 100, 30000, 30000, 30000, 30000], dtype=float)
    trail = _grow_trail_1d(line, start_idx=2, direction=1, threshold=15000)
    print(f"  Trail indices: {trail}")
    expected = np.array([2, 3, 4, 5])
    assert np.array_equal(trail, expected), f"Expected {expected}, got {trail}"
    print("  ✓ Passed")
    
    print("\n" + "=" * 60)
    print("✓✓✓ All boundary case tests passed!")


if __name__ == '__main__':
    test_grow_trail_edge_cases()