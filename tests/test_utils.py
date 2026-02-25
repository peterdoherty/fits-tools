import sys
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger.info("Running tests for utils.py functions...")

errors = []

# Test 1: prepare_for_json with mixed NumPy types
try:
    test_data_1 = {
        "mean_signal": np.float32(123.456),
        "variance": np.float32(7.89),
        "counts": np.int32(1000),
        "array_data": np.array([1, 2, 3], dtype=np.int16),
        "nested_list": [np.float32(1.1), np.float32(2.2), np.bool_(True)],
        "nested_dict": {"sub_key": np.int64(500)}
    }
    
    prepared_data_1 = prepare_for_json(test_data_1)
    logger.info("Test 1 (Mixed NumPy types): PASSED")
    logger.info(f"  Result: {prepared_data_1}")
    
    assert isinstance(prepared_data_1['mean_signal'], float)
    assert isinstance(prepared_data_1['variance'], float)
    assert isinstance(prepared_data_1['counts'], int)
    assert isinstance(prepared_data_1['array_data'], list)
    assert all(isinstance(x, int) for x in prepared_data_1['array_data'])
    assert isinstance(prepared_data_1['nested_list'][0], float)
    assert isinstance(prepared_data_1['nested_list'][2], bool)
    assert isinstance(prepared_data_1['nested_dict']['sub_key'], int)
except Exception as e:
    logger.error(f"Test 1 FAILED: {e}")
    errors.append(("Test 1", e))

# Test 2: Non-NumPy types
try:
    test_data_2 = {
        "empty_list": [],
        "empty_dict": {},
        "string_val": "hello",
        "bool_val": True
    }
    prepared_data_2 = prepare_for_json(test_data_2)
    logger.info("Test 2 (Non-NumPy types): PASSED")
    assert prepared_data_2 == test_data_2
except Exception as e:
    logger.error(f"Test 2 FAILED: {e}")
    errors.append(("Test 2", e))

# Test 3: is_fits_file
try:
    assert is_fits_file("image.fits") == True
    assert is_fits_file("IMAGE.FITS") == True
    assert is_fits_file("data.fits.gz") == True
    assert is_fits_file("image.jpg") == False
    logger.info("Test 3 (is_fits_file): PASSED")
except Exception as e:
    logger.error(f"Test 3 FAILED: {e}")
    errors.append(("Test 3", e))

# Test 4: validate_indices
try:
    validate_indices([0, 1, 2], max_index=5)  # Should pass
    try:
        validate_indices([0, 10], max_index=5)  # Should fail
        raise AssertionError("Should have raised IndexError")
    except IndexError:
        pass  # Expected
    logger.info("Test 4 (validate_indices): PASSED")
except Exception as e:
    logger.error(f"Test 4 FAILED: {e}")
    errors.append(("Test 4", e))

# Test 5: robust_mean
try:
    data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier
    mean = robust_mean(data, sigma=2.0)
    assert 2.5 < mean < 4.0, f"Expected mean around 3, got {mean}"
    logger.info(f"Test 5 (robust_mean): PASSED (mean={mean:.2f})")
except Exception as e:
    logger.error(f"Test 5 FAILED: {e}")
    errors.append(("Test 5", e))

# Summary
logger.info("\n" + "=" * 70)
if errors:
    logger.error(f"{len(errors)} test(s) FAILED:")
    for name, error in errors:
        logger.error(f"  {name}: {error}")
    return 1
else:
    logger.info("All tests PASSED!")
    return 0