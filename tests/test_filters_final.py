# test_filters_final.py
import numpy as np
from fits_tools.filters import _safe_cast_to_dtype, OPERATIONS

def test_safe_cast():
    """Test safe dtype casting."""
    # Test float to int with overflow
    data = np.array([300.7, -50.2, np.nan, np.inf], dtype=np.float32)
    result = _safe_cast_to_dtype(data, np.dtype(np.uint8))
    
    assert result[0] == 255  # Clipped to max
    assert result[1] == 0    # Clipped to min
    assert result[2] == 0    # NaN -> 0
    assert result[3] == 255  # Inf -> max
    print("✅ Safe cast works correctly")

def test_operations_registry():
    """Verify all operations are callable."""
    assert len(OPERATIONS) == 18, f"Expected 18 ops, got {len(OPERATIONS)}"
    
    # Check correction ops are proper functions, not lambdas
    for name in ['correct_row_median', 'correct_row_mean', 
                 'correct_col_median', 'correct_col_mean']:
        func = OPERATIONS[name]
        assert func.__name__.startswith('_apply'), \
            f"{name} should be a named function, not lambda"
    
    print("✅ All operations registered correctly")

def test_dtype_preservation():
    """Test that arithmetic filters return float32."""
    from fits_tools.filters import (
        _apply_unsharp_mask,
        _apply_high_pass,
        _apply_bilateral_filter,
        _correct_row_offset,
        _correct_col_offset
    )
    
    # Integer input
    data = np.array([[100, 150], [200, 250]], dtype=np.uint16)
    
    for func, name in [
        (_apply_unsharp_mask, 'unsharp_mask'),
        (_apply_high_pass, 'high_pass'),
        (_apply_bilateral_filter, 'bilateral'),
        (_correct_row_offset, 'row_offset'),
        (_correct_col_offset, 'col_offset'),
    ]:
        result = func(data)
        assert result.dtype == np.float32, \
            f"{name} should return float32, got {result.dtype}"
    
    print("✅ Arithmetic filters return float32")

if __name__ == "__main__":
    test_safe_cast()
    test_operations_registry()
    test_dtype_preservation()
    print("\n🎉 All verification tests passed!")
    