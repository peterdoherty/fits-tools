def test_bloom_existing_bpm_filtering():
    """Test that existing_bpm properly filters saturation seeds."""
    import numpy as np
    from fits_tools.artifacts.detection import detect_bloom
    from fits_tools.bitmask import BPM_BIT_COL_BAD, BPM_BIT_BLOOM
    
    print("Testing bloom detection with bad pixel filtering")
    print("=" * 70)
    
    # Create test frame
    frame = np.ones((200, 200)) * 1000.0
    
    # Add saturated bad column
    frame[:, 100] = 65000
    
    # Add saturated good source
    frame[50:60, 150] = 65000
    
    # Create master BPM with bad column
    master_bpm = np.zeros((200, 200), dtype=np.uint32)
    master_bpm[:, 100] = BPM_BIT_COL_BAD
    
    print("\nTest 1: Without existing_bpm")
    dq1, stats1 = detect_bloom(frame, saturation_level=60000, existing_bpm=None)
    print(f"  Saturated pixels: {stats1['num_saturated_pixels']}")
    print(f"  Saturated clusters: {stats1['num_saturated_clusters']}")
    print(f"  Expected: 2 clusters (bad column + good source)")
    
    print("\nTest 2: With existing_bpm (filters bad column)")
    dq2, stats2 = detect_bloom(frame, saturation_level=60000, existing_bpm=master_bpm)
    print(f"  Saturated pixels (good): {stats2['num_saturated_pixels']}")
    print(f"  Saturated total: {stats2['num_saturated_total']}")
    print(f"  Saturated excluded: {stats2['num_saturated_excluded']}")
    print(f"  Saturated clusters: {stats2['num_saturated_clusters']}")
    print(f"  Expected: 1 cluster (only good source)")
    
    # Verify filtering worked
    assert stats2['num_saturated_excluded'] == 200, "Should exclude 200 pixels (bad column)"
    assert stats2['num_saturated_clusters'] == 1, "Should find only 1 cluster (good source)"
    assert stats1['num_saturated_clusters'] == 2, "Without filtering should find 2 clusters"
    
    print("\n✓ Bad pixel filtering working correctly")
    print(f"  Prevented {stats2['num_saturated_excluded']} false saturation seeds")


if __name__ == '__main__':
    test_bloom_existing_bpm_filtering()