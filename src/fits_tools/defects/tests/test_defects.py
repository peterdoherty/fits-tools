#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest test suite for fits_tools.defects
=========================================
Covers:
  - bitmask.py   : constants, enum, utility functions, BPM operations
  - detection.py : sigma_clip_stats, roi_stats, all detect_* functions,
                   create_defect_density_map, create_calibration_bpm
  - utils.py     : load/save round-trip, merge, analyze_bpm_statistics,
                   get_defect_mask, filter_bpm, flag_bad_rows_cols

Run with:
    pytest test_defects.py -v
    pytest test_defects.py -v --tb=short   # for compact tracebacks

No real FITS files or camera hardware required — all fixtures are
synthetic NumPy arrays.
"""

import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# The imports below assume the package is on sys.path. Adjust the import
# paths to match your project layout if needed.
# ---------------------------------------------------------------------------
from fits_tools.defects.bitmask import (
    BPM_BIT_BIAS, BPM_BIT_DARK, BPM_BIT_FLAT_HOT, BPM_BIT_FLAT_COLD,
    BPM_BIT_BAD_ROW, BPM_BIT_BAD_COL, BPM_BIT_TRAP, BPM_BIT_BLOCKED,
    BPM_BIT_HOT, BPM_BIT_DEAD, BPM_BIT_SATURATED, BPM_BIT_TELEGRAPH,
    BPM_BIT_COSMIC, BPM_BIT_OUTLIER,
    BPM_MASK_PERMANENT, BPM_MASK_HOT, BPM_MASK_COLD,
    BPM_MASK_STRUCTURAL, BPM_MASK_ALL, BPM_MASK_ALL_16, BPM_MASK_ALL_32,
    DefectType,
    DEFECT_NAMES, DEFECT_DESCRIPTIONS,
    has_defect, get_defect_types, count_defects,
    create_bpm, merge_bpms, apply_bpm,
    get_bpm_statistics, get_defect_mask, filter_bpm,
    bit_index, iter_defect_definitions,
    format_bit_comments, get_definitions_payload,
)
from fits_tools.defects.detection import (
    sigma_clip_stats, roi_stats,
    detect_bias_defects, detect_dark_hot_pixels,
    detect_flat_defects, detect_bad_rows_cols,
    create_defect_density_map, create_calibration_bpm,
    get_default_detection_config,
)
from fits_tools.defects.utils import (
    analyze_bpm_statistics,
    get_defect_mask as utils_get_defect_mask,
    filter_bpm as utils_filter_bpm,
    flag_bad_rows_cols, check_bad_rows_cols,
    describe_pixel_defects,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def small_bpm():
    """3×3 BPM with a variety of flags."""
    bpm = create_bpm((3, 3))
    bpm[0, 0] = BPM_BIT_BIAS
    bpm[0, 1] = BPM_BIT_DARK
    bpm[0, 2] = BPM_BIT_DARK | BPM_BIT_FLAT_HOT
    bpm[1, 0] = BPM_BIT_FLAT_COLD
    # Row 2 and pixel (1,1) are clean
    return bpm


@pytest.fixture
def uniform_data():
    """100×100 array of ones — perfect flat."""
    return np.ones((100, 100), dtype=np.float32)


@pytest.fixture
def noisy_data(rng=None):
    """100×100 array with Gaussian noise (mean=1000, std=10)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=1000.0, scale=10.0, size=(100, 100)).astype(np.float32)


@pytest.fixture
def hot_pixel_data(noisy_data):
    """noisy_data with 5 implanted hot pixels far above threshold."""
    data = noisy_data.copy()
    coords = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
    for r, c in coords:
        data[r, c] = 9000.0   # >> 5-sigma above 1000 ± 10
    return data, coords


@pytest.fixture
def flat_with_defects():
    """
    200×200 normalised flat (median ≈ 1.0) with:
      - 3 hot pixels  at value 1.5  (50% above median → flags at 20% threshold)
      - 3 cold pixels at value 0.4  (60% below median → flags at 20% threshold)
    """
    rng = np.random.default_rng(0)
    data = rng.normal(loc=1000.0, scale=5.0, size=(200, 200)).astype(np.float32)
    hot_coords  = [(10, 10), (50, 80), (100, 150)]
    cold_coords = [(20, 30), (60, 90), (110, 160)]
    for r, c in hot_coords:
        data[r, c] = 1500.0
    for r, c in cold_coords:
        data[r, c] = 400.0
    return data, hot_coords, cold_coords


# ===========================================================================
# bitmask.py tests
# ===========================================================================

class TestBitConstants:
    """Verify every bit constant is a distinct power of two."""

    ALL_BITS = [
        BPM_BIT_BIAS, BPM_BIT_DARK, BPM_BIT_FLAT_HOT, BPM_BIT_FLAT_COLD,
        BPM_BIT_BAD_ROW, BPM_BIT_BAD_COL, BPM_BIT_TRAP, BPM_BIT_BLOCKED,
        BPM_BIT_HOT, BPM_BIT_DEAD, BPM_BIT_SATURATED, BPM_BIT_TELEGRAPH,
        BPM_BIT_COSMIC, BPM_BIT_OUTLIER,
    ]

    def test_all_powers_of_two(self):
        for bit in self.ALL_BITS:
            assert bit > 0 and (bit & (bit - 1)) == 0, \
                f"{bit} is not a power of two"

    def test_all_distinct(self):
        assert len(set(self.ALL_BITS)) == len(self.ALL_BITS)

    def test_mask_all_covers_all_known_bits(self):
        for bit in self.ALL_BITS:
            assert BPM_MASK_ALL & bit, f"BPM_MASK_ALL does not cover bit {bit}"

    def test_mask_all_32_is_uint32_max(self):
        assert BPM_MASK_ALL_32 == 0xFFFF_FFFF

    def test_mask_all_16_is_uint16_max(self):
        assert BPM_MASK_ALL_16 == 0xFFFF

    def test_permanent_mask_excludes_saturated(self):
        assert not (BPM_MASK_PERMANENT & BPM_BIT_SATURATED)

    def test_permanent_mask_excludes_extended_bits(self):
        for bit in [BPM_BIT_COSMIC, BPM_BIT_OUTLIER]:
            assert not (BPM_MASK_PERMANENT & bit), \
                f"BPM_MASK_PERMANENT should not include extended bit {bit}"

    def test_hot_mask_composition(self):
        assert BPM_MASK_HOT == (BPM_BIT_DARK | BPM_BIT_FLAT_HOT | BPM_BIT_HOT)

    def test_cold_mask_composition(self):
        assert BPM_MASK_COLD == (BPM_BIT_FLAT_COLD | BPM_BIT_DEAD)

    def test_structural_mask_composition(self):
        assert BPM_MASK_STRUCTURAL == (BPM_BIT_BAD_ROW | BPM_BIT_BAD_COL | BPM_BIT_BLOCKED)


class TestDefectTypeEnum:
    """DefectType enum must mirror the plain constants."""

    def test_enum_values_match_constants(self):
        pairs = [
            (DefectType.BIAS,      BPM_BIT_BIAS),
            (DefectType.DARK,      BPM_BIT_DARK),
            (DefectType.FLAT_HOT,  BPM_BIT_FLAT_HOT),
            (DefectType.FLAT_COLD, BPM_BIT_FLAT_COLD),
            (DefectType.BAD_ROW,   BPM_BIT_BAD_ROW),
            (DefectType.BAD_COL,   BPM_BIT_BAD_COL),
            (DefectType.BLOCKED,   BPM_BIT_BLOCKED),
            (DefectType.HOT,       BPM_BIT_HOT),
            (DefectType.DEAD,      BPM_BIT_DEAD),
            (DefectType.COSMIC,    BPM_BIT_COSMIC),
        ]
        for member, constant in pairs:
            assert int(member) == constant

    def test_bitwise_combination(self):
        combo = DefectType.DARK | DefectType.FLAT_HOT
        assert DefectType.DARK in combo
        assert DefectType.FLAT_HOT in combo
        assert DefectType.BIAS not in combo

    def test_no_missing_members(self):
        """Every key in DEFECT_NAMES must be represented in DefectType."""
        enum_values = {int(m) for m in DefectType}
        for bit in DEFECT_NAMES:
            assert bit in enum_values, \
                f"DEFECT_NAMES bit {bit} ({DEFECT_NAMES[bit]}) missing from DefectType"


class TestDefectNameDicts:
    """DEFECT_NAMES and DEFECT_DESCRIPTIONS must be consistent."""

    def test_names_and_descriptions_same_keys(self):
        assert set(DEFECT_NAMES.keys()) == set(DEFECT_DESCRIPTIONS.keys())

    def test_names_are_strings(self):
        for bit, name in DEFECT_NAMES.items():
            assert isinstance(name, str) and name, f"Bad name for bit {bit}"

    def test_descriptions_are_strings(self):
        for bit, desc in DEFECT_DESCRIPTIONS.items():
            assert isinstance(desc, str) and desc, f"Bad description for bit {bit}"


class TestBitIndex:
    def test_known_values(self):
        assert bit_index(1)    == 0
        assert bit_index(2)    == 1
        assert bit_index(4)    == 2
        assert bit_index(1024) == 10

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            bit_index(0)

    def test_rejects_non_power_of_two(self):
        with pytest.raises(ValueError):
            bit_index(3)
        with pytest.raises(ValueError):
            bit_index(6)


class TestIterDefectDefinitions:
    def test_yields_tuples(self):
        results = list(iter_defect_definitions())
        assert len(results) == len(DEFECT_NAMES)
        for bit_val, name, desc in results:
            assert isinstance(bit_val, int)
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_ascending_bit_order(self):
        values = [bv for bv, _, _ in iter_defect_definitions()]
        assert values == sorted(values)


class TestFormatBitComments:
    def test_returns_list_of_strings(self):
        lines = format_bit_comments()
        assert isinstance(lines, list)
        assert all(isinstance(ln, str) for ln in lines)

    def test_first_line_is_header(self):
        lines = format_bit_comments()
        assert 'bit' in lines[0].lower() or 'flag' in lines[0].lower()

    def test_length_matches_definitions(self):
        # One header line + one line per known defect
        lines = format_bit_comments()
        assert len(lines) == 1 + len(DEFECT_NAMES)


class TestGetDefinitionsPayload:
    def test_schema_key_present(self):
        p = get_definitions_payload()
        assert 'schema' in p and 'nbits' in p and 'bits' in p

    def test_bits_sorted_ascending(self):
        bits = get_definitions_payload()['bits']
        indices = [b['bit'] for b in bits]
        assert indices == sorted(indices)

    def test_nbits_respected(self):
        p16 = get_definitions_payload(nbits=16)
        for b in p16['bits']:
            assert b['bit'] < 16

    def test_all_required_fields(self):
        for b in get_definitions_payload()['bits']:
            assert {'bit', 'value', 'name', 'desc'} <= b.keys()


class TestHasDefect:
    def test_single_bit_present(self):
        assert has_defect(BPM_BIT_DARK, BPM_BIT_DARK)

    def test_single_bit_absent(self):
        assert not has_defect(BPM_BIT_BIAS, BPM_BIT_DARK)

    def test_combined_bits(self):
        val = BPM_BIT_DARK | BPM_BIT_FLAT_HOT
        assert has_defect(val, BPM_BIT_DARK)
        assert has_defect(val, BPM_BIT_FLAT_HOT)
        assert not has_defect(val, BPM_BIT_BIAS)

    def test_zero_pixel(self):
        assert not has_defect(0, BPM_BIT_DARK)


class TestGetDefectTypes:
    def test_empty_pixel(self):
        assert get_defect_types(0) == []

    def test_single_defect(self):
        assert get_defect_types(BPM_BIT_DARK) == ['DARK']

    def test_multiple_defects_sorted_by_bit(self):
        val = BPM_BIT_DARK | BPM_BIT_FLAT_HOT  # bits 1 and 2
        result = get_defect_types(val)
        assert result == ['DARK', 'FLAT_HOT']

    def test_output_order_is_stable(self):
        """Calling twice must return the same order."""
        val = BPM_BIT_COSMIC | BPM_BIT_BIAS | BPM_BIT_TELEGRAPH
        assert get_defect_types(val) == get_defect_types(val)

    def test_all_known_bits(self):
        all_bits = 0
        for bit in DEFECT_NAMES:
            all_bits |= bit
        names = get_defect_types(all_bits)
        assert set(names) == set(DEFECT_NAMES.values())


class TestCountDefects:
    def test_count_specific_bit(self, small_bpm):
        # small_bpm has DARK at (0,1) and DARK|FLAT_HOT at (0,2) → 2 dark pixels
        assert count_defects(small_bpm, BPM_BIT_DARK) == 2

    def test_count_any_defect(self, small_bpm):
        assert count_defects(small_bpm) == 4

    def test_count_zero(self):
        bpm = create_bpm((10, 10))
        assert count_defects(bpm, BPM_BIT_DARK) == 0

    def test_count_all_defects(self):
        bpm = create_bpm((5, 5))
        bpm[:] = BPM_BIT_DARK
        assert count_defects(bpm, BPM_BIT_DARK) == 25


class TestCreateBpm:
    def test_shape_and_dtype(self):
        bpm = create_bpm((128, 256))
        assert bpm.shape == (128, 256)
        assert bpm.dtype == np.uint32

    def test_all_zeros(self):
        bpm = create_bpm((64, 64))
        assert np.all(bpm == 0)

    def test_custom_dtype(self):
        bpm = create_bpm((10, 10), dtype=np.uint16)
        assert bpm.dtype == np.uint16


class TestMergeBpms:
    def test_or_union(self):
        b1 = np.array([[BPM_BIT_DARK,  0]], dtype=np.uint32)
        b2 = np.array([[0, BPM_BIT_BIAS]], dtype=np.uint32)
        merged = merge_bpms(b1, b2, operation='or')
        assert merged[0, 0] == BPM_BIT_DARK
        assert merged[0, 1] == BPM_BIT_BIAS

    def test_and_intersection(self):
        val = BPM_BIT_DARK | BPM_BIT_BIAS
        b1 = np.array([[val, BPM_BIT_DARK]], dtype=np.uint32)
        b2 = np.array([[BPM_BIT_DARK, BPM_BIT_BIAS]], dtype=np.uint32)
        merged = merge_bpms(b1, b2, operation='and')
        assert merged[0, 0] == BPM_BIT_DARK   # only bit in common
        assert merged[0, 1] == 0               # no common bits

    def test_shape_mismatch_raises(self):
        b1 = create_bpm((10, 10))
        b2 = create_bpm((10, 11))
        with pytest.raises(ValueError):
            merge_bpms(b1, b2)

    def test_invalid_operation_raises(self):
        b = create_bpm((5, 5))
        with pytest.raises(ValueError):
            merge_bpms(b, b, operation='xor')

    def test_or_preserves_all_bits(self):
        b1 = np.array([[BPM_BIT_DARK | BPM_BIT_BIAS]], dtype=np.uint32)
        b2 = np.array([[BPM_BIT_FLAT_HOT]], dtype=np.uint32)
        merged = merge_bpms(b1, b2)
        assert merged[0, 0] == (BPM_BIT_DARK | BPM_BIT_BIAS | BPM_BIT_FLAT_HOT)


class TestApplyBpm:
    def test_bad_pixels_become_nan(self):
        data = np.ones((3, 3), dtype=np.float32)
        bpm = create_bpm((3, 3))
        bpm[1, 1] = BPM_BIT_DARK
        result = apply_bpm(data, bpm)
        assert np.isnan(result[1, 1])
        assert result[0, 0] == 1.0

    def test_good_pixels_unchanged(self, small_bpm):
        data = np.full((3, 3), 500.0)
        result = apply_bpm(data, small_bpm)
        assert result[2, 2] == 500.0   # pixel (2,2) has no defects

    def test_custom_fill_value(self):
        data = np.ones((3, 3), dtype=np.float32)
        bpm = create_bpm((3, 3))
        bpm[0, 0] = BPM_BIT_DEAD
        result = apply_bpm(data, bpm, fill_value=0.0)
        assert result[0, 0] == 0.0

    def test_selective_masking_by_defect_type(self):
        data = np.ones((3, 3), dtype=np.float32)
        bpm = create_bpm((3, 3))
        bpm[0, 0] = BPM_BIT_DARK
        bpm[1, 1] = BPM_BIT_COSMIC
        # Mask only permanent defects — cosmic ray should NOT be masked
        result = apply_bpm(data, bpm, defect_mask=BPM_MASK_PERMANENT)
        assert np.isnan(result[0, 0])
        assert result[1, 1] == 1.0

    def test_shape_mismatch_raises(self):
        data = np.ones((5, 5))
        bpm = create_bpm((5, 6))
        with pytest.raises(ValueError):
            apply_bpm(data, bpm)

    def test_integer_data_promoted_for_nan(self):
        data = np.ones((3, 3), dtype=np.int32)
        bpm = create_bpm((3, 3))
        bpm[0, 0] = BPM_BIT_BIAS
        result = apply_bpm(data, bpm)
        assert result.dtype == np.float64
        assert np.isnan(result[0, 0])

    def test_incompatible_fill_value_raises(self):
        data = np.ones((3, 3), dtype=np.int32)
        bpm = create_bpm((3, 3))
        bpm[0, 0] = BPM_BIT_BIAS
        with pytest.raises(ValueError):
            apply_bpm(data, bpm, fill_value=1.5)  # float into int: not safe


class TestGetBpmStatistics:
    def test_total_count(self, small_bpm):
        stats = get_bpm_statistics(small_bpm)
        assert stats['TOTAL']['count'] == 4

    def test_dark_count(self, small_bpm):
        stats = get_bpm_statistics(small_bpm)
        assert stats['DARK']['count'] == 2  # (0,1) and (0,2)

    def test_fraction_sums_correctly(self, small_bpm):
        stats = get_bpm_statistics(small_bpm)
        assert math.isclose(stats['TOTAL']['fraction'], 4 / 9, rel_tol=1e-6)

    def test_all_defect_names_present(self, small_bpm):
        stats = get_bpm_statistics(small_bpm)
        for name in DEFECT_NAMES.values():
            assert name in stats, f"Missing key: {name}"

    def test_empty_bpm(self):
        bpm = create_bpm((50, 50))
        stats = get_bpm_statistics(bpm)
        assert stats['TOTAL']['count'] == 0
        assert stats['TOTAL']['fraction'] == 0.0

    def test_keys_in_bit_order(self):
        """Returned defect keys must be in ascending bit order (excluding TOTAL)."""
        bpm = create_bpm((10, 10))
        stats = get_bpm_statistics(bpm)
        defect_keys = [k for k in stats if k != 'TOTAL']
        expected = list(DEFECT_NAMES.values())  # already sorted in fixture
        assert defect_keys == expected


class TestGetDefectMask:
    def test_none_returns_all_bad(self, small_bpm):
        mask = get_defect_mask(small_bpm)
        assert mask.sum() == 4

    def test_specific_type(self, small_bpm):
        mask = get_defect_mask(small_bpm, [BPM_BIT_DARK])
        assert mask.sum() == 2

    def test_combined_types(self, small_bpm):
        mask = get_defect_mask(small_bpm, [BPM_BIT_BIAS, BPM_BIT_FLAT_COLD])
        assert mask.sum() == 2  # (0,0) and (1,0)

    def test_empty_result(self, small_bpm):
        mask = get_defect_mask(small_bpm, [BPM_BIT_COSMIC])
        assert mask.sum() == 0


class TestFilterBpm:
    def test_retains_only_selected_bits(self):
        bpm = np.array([[BPM_BIT_DARK | BPM_BIT_BIAS, BPM_BIT_FLAT_HOT]], dtype=np.uint32)
        filtered = filter_bpm(bpm, [BPM_BIT_DARK])
        assert filtered[0, 0] == BPM_BIT_DARK
        assert filtered[0, 1] == 0

    def test_preserves_dtype(self):
        bpm = create_bpm((5, 5), dtype=np.uint16)
        bpm[0, 0] = BPM_BIT_BIAS
        filtered = filter_bpm(bpm, [BPM_BIT_BIAS])
        assert filtered.dtype == np.uint16


# ===========================================================================
# detection.py tests
# ===========================================================================

class TestSigmaClipStats:
    def test_clean_gaussian(self):
        rng = np.random.default_rng(1)
        data = rng.normal(100.0, 5.0, size=10000)
        median, std = sigma_clip_stats(data)
        assert abs(median - 100.0) < 0.5
        assert abs(std - 5.0) < 0.3

    def test_outliers_excluded(self):
        data = np.ones(1000) * 500.0
        data[0] = 1e6   # extreme outlier
        data[1] = -1e6
        median, std = sigma_clip_stats(data, sigma_low=3.0, sigma_high=3.0)
        assert abs(median - 500.0) < 1.0
        assert std < 1.0   # nearly constant after clipping

    def test_all_same_value(self):
        data = np.full(100, 42.0)
        median, std = sigma_clip_stats(data)
        assert median == 42.0
        assert std == 0.0

    def test_empty_after_masking(self):
        data = np.ones(10)
        mask = np.ones(10, dtype=bool)   # mask everything
        median, std = sigma_clip_stats(data, mask=mask)
        assert math.isnan(median)
        assert math.isnan(std)

    def test_mask_shape_mismatch_raises(self):
        data = np.ones((5, 5))
        mask = np.ones((4, 5), dtype=bool)
        with pytest.raises(ValueError):
            sigma_clip_stats(data, mask=mask)

    def test_mad_scale(self):
        rng = np.random.default_rng(2)
        data = rng.normal(200.0, 8.0, size=5000)
        median, std = sigma_clip_stats(data, scale='mad')
        assert abs(median - 200.0) < 1.0
        assert abs(std - 8.0) < 1.0

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            sigma_clip_stats(np.ones(10), scale='iqr')

    def test_convergence_within_max_iters(self):
        """Should converge in at most max_iters without infinite loop."""
        rng = np.random.default_rng(3)
        data = rng.normal(0, 1, 1000)
        median, std = sigma_clip_stats(data, max_iters=2)
        assert np.isfinite(median) and np.isfinite(std)

    def test_all_nan_input(self):
        data = np.full(50, np.nan)
        median, std = sigma_clip_stats(data)
        assert math.isnan(median)
        assert math.isnan(std)


class TestRoiStats:
    def test_roi_vs_full_frame(self, noisy_data):
        full_med, full_std = roi_stats(noisy_data)
        roi_med, roi_std = roi_stats(noisy_data, roi=(10, 90, 10, 90))
        # Both should be close to the true mean=1000, std=10
        assert abs(full_med - 1000.0) < 5.0
        assert abs(roi_med  - 1000.0) < 5.0

    def test_roi_out_of_bounds_raises(self, noisy_data):
        with pytest.raises(ValueError):
            roi_stats(noisy_data, roi=(0, 200, 0, 200))  # 200 > 100

    def test_degenerate_roi_raises(self, noisy_data):
        with pytest.raises(ValueError):
            roi_stats(noisy_data, roi=(50, 50, 10, 90))  # x2 == x1

    def test_no_sigma_clip(self, noisy_data):
        med, std = roi_stats(noisy_data, sigma_clip=False)
        assert np.isfinite(med) and np.isfinite(std)


class TestDetectBiasDefects:
    def test_detects_high_outliers_absolute(self):
        data = np.full((50, 50), 1000.0, dtype=np.float32)
        data[10, 10] = 1200.0   # 200 ADU above median
        bpm, stats = detect_bias_defects(data, threshold_adu=100.0, method='absolute', mode='high')
        assert bpm[10, 10] & BPM_BIT_BIAS
        assert stats['num_defects'] == 1

    def test_detects_both_sides(self):
        data = np.full((50, 50), 1000.0, dtype=np.float32)
        data[5,  5] = 1200.0   # high
        data[15, 15] = 800.0   # low
        bpm, stats = detect_bias_defects(data, threshold_adu=100.0, method='absolute', mode='both')
        assert bpm[5,  5]  & BPM_BIT_BIAS
        assert bpm[15, 15] & BPM_BIT_BIAS
        assert stats['num_defects'] == 2

    def test_sigma_method(self):
        rng = np.random.default_rng(10)
        data = rng.normal(500.0, 5.0, (100, 100)).astype(np.float32)
        data[50, 50] = 600.0   # 20-sigma spike
        bpm, stats = detect_bias_defects(data, method='sigma', sigma=5.0, mode='high')
        assert bpm[50, 50] & BPM_BIT_BIAS
        assert stats['num_defects'] >= 1

    def test_degenerate_data_no_crash(self):
        """All-NaN data should return empty BPM without raising."""
        data = np.full((10, 10), np.nan)
        bpm, stats = detect_bias_defects(data)
        assert stats['num_defects'] == 0
        assert np.all(bpm == 0)

    def test_degenerate_stats_schema_complete(self):
        data = np.full((10, 10), np.nan)
        _, stats = detect_bias_defects(data)
        for key in ('num_defects', 'fraction_defects', 'threshold_hi_adu', 'threshold_lo_adu'):
            assert key in stats, f"Missing key in degenerate stats: {key}"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            detect_bias_defects(np.ones((10, 10)), method='percentile')

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            detect_bias_defects(np.ones((10, 10)), mode='middle')

    def test_existing_bpm_excluded_from_stats(self):
        """A pixel already flagged should not bias the statistics."""
        data = np.full((50, 50), 1000.0, dtype=np.float32)
        data[0, 0] = 5000.0  # extreme — would pull median if included
        existing = create_bpm((50, 50))
        existing[0, 0] = BPM_BIT_COSMIC
        bpm, stats = detect_bias_defects(
            data, threshold_adu=100.0, method='absolute',
            mode='high', existing_bpm=existing
        )
        # median should still be ~1000, not pulled by the spike
        assert abs(stats['median'] - 1000.0) < 5.0


class TestDetectDarkHotPixels:
    def test_detects_hot_pixels(self, hot_pixel_data):
        data, coords = hot_pixel_data
        bpm, stats = detect_dark_hot_pixels(data, threshold_sigma=5.0)
        for r, c in coords:
            assert bpm[r, c] & BPM_BIT_DARK, f"Hot pixel at ({r},{c}) not flagged"
        assert stats['num_defects'] == len(coords)

    def test_does_not_flag_cold_pixels(self):
        data = np.full((50, 50), 100.0, dtype=np.float32)
        data[5, 5] = 0.0   # cold pixel — should NOT be flagged
        bpm, stats = detect_dark_hot_pixels(data, threshold_sigma=5.0)
        assert not (bpm[5, 5] & BPM_BIT_DARK)

    def test_stats_keys_present(self, hot_pixel_data):
        data, _ = hot_pixel_data
        _, stats = detect_dark_hot_pixels(data)
        for key in ('median', 'std', 'threshold_sigma', 'threshold_hi',
                    'num_defects', 'fraction_defects'):
            assert key in stats

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            detect_dark_hot_pixels(np.ones((10, 10)), threshold_sigma=-1.0)
        with pytest.raises(ValueError):
            detect_dark_hot_pixels(np.ones((10, 10)), threshold_sigma=0.0)

    def test_stats_roi(self, hot_pixel_data):
        data, coords = hot_pixel_data
        # ROI that excludes all hot pixels (they're at rows 10–50, cols 10–50)
        bpm, stats = detect_dark_hot_pixels(data, stats_roi=(0, 8, 0, 8))
        # Stats should be computed only from the clean corner
        assert np.isfinite(stats['median'])


class TestDetectFlatDefects:
    def test_detects_hot_and_cold(self, flat_with_defects):
        data, hot_coords, cold_coords = flat_with_defects
        bpm, stats = detect_flat_defects(
            data, hot_threshold_percent=20.0, cold_threshold_percent=20.0
        )
        for r, c in hot_coords:
            assert bpm[r, c] & BPM_BIT_FLAT_HOT, f"Hot pixel ({r},{c}) not flagged"
        for r, c in cold_coords:
            assert bpm[r, c] & BPM_BIT_FLAT_COLD, f"Cold pixel ({r},{c}) not flagged"

    def test_good_pixels_not_flagged(self, uniform_data):
        bpm, stats = detect_flat_defects(uniform_data)
        assert np.all(bpm == 0)
        assert stats['num_defects'] == 0

    def test_num_both_accounting(self, flat_with_defects):
        data, hot_coords, cold_coords = flat_with_defects
        _, stats = detect_flat_defects(
            data, hot_threshold_percent=20.0, cold_threshold_percent=20.0
        )
        # Standard thresholds: hot and cold sets don't overlap
        assert stats['num_both'] == 0
        assert stats['num_defects'] == stats['num_hot'] + stats['num_cold'] - stats['num_both']

    def test_stats_schema_complete(self, flat_with_defects):
        data, _, _ = flat_with_defects
        _, stats = detect_flat_defects(data)
        for key in ('median', 'std', 'normalize', 'hot_threshold_percent',
                    'cold_threshold_percent', 'hot_threshold', 'cold_threshold',
                    'num_hot', 'num_cold', 'num_both', 'num_defects', 'fraction_defects'):
            assert key in stats, f"Missing key in stats: {key}"

    def test_degenerate_median_stats_schema_complete(self):
        """Zero-data flat: degenerate early return must have the same schema."""
        data = np.zeros((20, 20), dtype=np.float32)
        _, stats = detect_flat_defects(data)
        for key in ('median', 'std', 'normalize', 'hot_threshold_percent',
                    'cold_threshold_percent', 'hot_threshold', 'cold_threshold',
                    'num_hot', 'num_cold', 'num_both', 'num_defects', 'fraction_defects'):
            assert key in stats, f"Missing key in degenerate stats: {key}"

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            detect_flat_defects(np.ones((10, 10)), hot_threshold_percent=-1.0)

    def test_normalize_false(self, flat_with_defects):
        data, _, _ = flat_with_defects
        bpm, stats = detect_flat_defects(data, normalize=False)
        assert stats['normalize'] is False
        assert np.isfinite(stats['median'])

    def test_existing_bpm_excluded(self, flat_with_defects):
        data, hot_coords, _ = flat_with_defects
        existing = create_bpm(data.shape)
        for r, c in hot_coords:
            existing[r, c] = BPM_BIT_COSMIC
        # With the hot pixels masked from stats, threshold should still catch them
        bpm, stats = detect_flat_defects(
            data, existing_bpm=existing,
            hot_threshold_percent=20.0, cold_threshold_percent=20.0
        )
        assert np.isfinite(stats['median'])


class TestDetectBadRowsCols:
    def test_bad_row_flagged(self):
        bpm_in = create_bpm((20, 20))
        bpm_in[5, :] = BPM_BIT_DARK   # entire row 5 is bad
        bpm_out, stats = detect_bad_rows_cols(bpm_in, row_threshold=0.5, col_threshold=0.9)
        assert bpm_out[5, 0] & BPM_BIT_BAD_ROW
        assert stats['num_bad_rows'] == 1
        assert 5 in stats['bad_rows']

    def test_bad_col_flagged(self):
        bpm_in = create_bpm((20, 20))
        bpm_in[:, 3] = BPM_BIT_DEAD
        bpm_out, stats = detect_bad_rows_cols(bpm_in, row_threshold=0.9, col_threshold=0.5)
        assert bpm_out[0, 3] & BPM_BIT_BAD_COL
        assert stats['num_bad_cols'] == 1

    def test_clean_bpm_no_flags(self):
        bpm_in = create_bpm((20, 20))
        bpm_in[5, 5] = BPM_BIT_DARK   # isolated pixel, far below threshold
        bpm_out, stats = detect_bad_rows_cols(bpm_in)
        assert stats['num_bad_rows'] == 0
        assert stats['num_bad_cols'] == 0

    def test_active_region_respected(self):
        bpm_in = create_bpm((20, 20))
        bpm_in[0, :] = BPM_BIT_DARK   # row 0 is bad but outside active region
        active = (slice(2, 18), slice(2, 18))
        bpm_out, stats = detect_bad_rows_cols(
            bpm_in, row_threshold=0.5, active_region=active
        )
        assert stats['num_bad_rows'] == 0

    def test_defect_mask_filter(self):
        bpm_in = create_bpm((20, 20))
        bpm_in[7, :] = BPM_BIT_COSMIC  # entire row, but only cosmic
        # Only count DARK bits — row 7 should NOT be flagged
        bpm_out, stats = detect_bad_rows_cols(
            bpm_in, row_threshold=0.5, defect_mask=BPM_BIT_DARK
        )
        assert stats['num_bad_rows'] == 0


class TestCreateDefectDensityMap:
    def test_basic_shape(self):
        bpm = create_bpm((128, 128))
        result = create_defect_density_map(bpm, tile_size=64)
        assert result['ny_tiles'] == 2
        assert result['nx_tiles'] == 2
        assert np.array(result['counts']).shape == (2, 2)

    def test_all_zeros_gives_zero_density(self):
        bpm = create_bpm((64, 64))
        result = create_defect_density_map(bpm, tile_size=32)
        assert np.all(np.array(result['counts']) == 0)

    def test_single_defect_in_tile(self):
        bpm = create_bpm((64, 64))
        bpm[5, 5] = BPM_BIT_DARK   # in tile (0,0) for tile_size=32
        result = create_defect_density_map(bpm, tile_size=32)
        counts = np.array(result['counts'])
        assert counts[0, 0] == 1
        assert counts.sum() == 1

    def test_too_small_image(self):
        bpm = create_bpm((10, 10))
        result = create_defect_density_map(bpm, tile_size=64)
        assert result['ny_tiles'] == 0
        assert result['nx_tiles'] == 0
        assert result['counts'] is None

    def test_defect_mask_respected(self):
        bpm = create_bpm((64, 64))
        bpm[5, 5] = BPM_BIT_COSMIC     # only cosmic — should not count with DARK mask
        bpm[10, 10] = BPM_BIT_DARK
        result_all  = create_defect_density_map(bpm, tile_size=32)
        result_dark = create_defect_density_map(bpm, tile_size=32, defect_mask=BPM_BIT_DARK)
        # All-defects map should see 2 bad pixels total
        assert np.array(result_all['counts']).sum() == 2
        # DARK-only map should see only 1
        assert np.array(result_dark['counts']).sum() == 1

    def test_non_contiguous_bpm(self):
        """Transposed (non-contiguous) array should not crash the reshape."""
        bpm = create_bpm((128, 64))
        bpm_T = np.ascontiguousarray(bpm.T)   # shape (64,128)
        result = create_defect_density_map(bpm_T, tile_size=32)
        assert result['ny_tiles'] == 2
        assert result['nx_tiles'] == 4

    def test_counts_match_manual_count(self):
        """Vectorised result must match a naive per-tile loop."""
        rng = np.random.default_rng(7)
        bpm = rng.integers(0, 2, size=(64, 64), dtype=np.uint32) * BPM_BIT_DARK
        result = create_defect_density_map(bpm, tile_size=16)
        counts = np.array(result['counts'])
        for iy in range(4):
            for ix in range(4):
                tile = bpm[iy*16:(iy+1)*16, ix*16:(ix+1)*16]
                assert counts[iy, ix] == int(np.sum(tile > 0))


class TestCreateCalibrationBpm:
    @pytest.fixture
    def calibration_frames(self):
        rng = np.random.default_rng(99)
        shape = (64, 64)
        bias = rng.normal(1000.0, 5.0, shape).astype(np.float32)
        dark = rng.normal(10.0,   2.0, shape).astype(np.float32)
        flat = rng.normal(50000., 100., shape).astype(np.float32)
        # Implant one obvious hot pixel per frame
        bias[5,  5] = 2000.0
        dark[10, 10] = 500.0
        flat[20, 20] = 80000.0
        flat[30, 30] = 100.0    # cold
        return bias, dark, flat, shape

    def test_returns_correct_types(self, calibration_frames):
        bias, dark, flat, _ = calibration_frames
        bpm, stats = create_calibration_bpm(bias, dark, flat)
        assert isinstance(bpm, np.ndarray)
        assert isinstance(stats, dict)

    def test_bpm_shape_matches_input(self, calibration_frames):
        bias, dark, flat, shape = calibration_frames
        bpm, _ = create_calibration_bpm(bias, dark, flat)
        assert bpm.shape == shape

    def test_all_stages_present_in_stats(self, calibration_frames):
        bias, dark, flat, _ = calibration_frames
        _, stats = create_calibration_bpm(bias, dark, flat)
        for key in ('bias', 'dark', 'flat', 'rows_cols', 'total_defects', 'total_fraction'):
            assert key in stats

    def test_shape_mismatch_raises(self, calibration_frames):
        bias, dark, flat, _ = calibration_frames
        wrong = np.ones((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="same shape"):
            create_calibration_bpm(bias, wrong, flat)

    def test_custom_config(self, calibration_frames):
        bias, dark, flat, _ = calibration_frames
        config = get_default_detection_config()
        config['bias']['threshold_adu'] = 50.0   # tighter
        bpm, stats = create_calibration_bpm(bias, dark, flat, config=config)
        assert stats['bias']['num_defects'] >= 1

    def test_total_fraction_in_range(self, calibration_frames):
        bias, dark, flat, _ = calibration_frames
        _, stats = create_calibration_bpm(bias, dark, flat)
        assert 0.0 <= stats['total_fraction'] <= 1.0


# ===========================================================================
# utils.py tests
# ===========================================================================

class TestAnalyzeBpmStatistics:
    def test_returns_same_as_get_bpm_statistics(self, small_bpm):
        from fits_tools.defects.bitmask import get_bpm_statistics
        expected = get_bpm_statistics(small_bpm)
        result = analyze_bpm_statistics(small_bpm, verbose=False)
        assert result == expected

    def test_verbose_does_not_raise(self, small_bpm, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            analyze_bpm_statistics(small_bpm, verbose=True)
        # At least one log line mentioning bad pixels
        assert any('bad pixel' in rec.message.lower() for rec in caplog.records)

    def test_missing_key_logs_warning(self, small_bpm, caplog, monkeypatch):
        """If get_bpm_statistics returns an incomplete dict, a warning must be logged."""
        import logging
        import fits_tools.defects.utils as utils_module

        original = utils_module.get_bpm_statistics

        def patched(bpm):
            stats = original(bpm)
            del stats['DARK']   # simulate missing key
            return stats

        monkeypatch.setattr(utils_module, 'get_bpm_statistics', patched)
        with caplog.at_level(logging.WARNING):
            analyze_bpm_statistics(small_bpm, verbose=True)
        assert any('DARK' in rec.message for rec in caplog.records)


class TestDescribePixelDefects:
    def test_good_pixel(self):
        assert describe_pixel_defects(0) == "Good pixel"

    def test_single_defect(self):
        desc = describe_pixel_defects(BPM_BIT_BIAS)
        assert 'BIAS' in desc

    def test_multiple_defects(self):
        val = BPM_BIT_BIAS | BPM_BIT_DARK
        desc = describe_pixel_defects(val)
        assert 'BIAS' in desc
        assert 'DARK' in desc


class TestCheckBadRowsCols:
    def test_bad_row_detected(self):
        bpm = create_bpm((20, 20))
        bpm[3, :] = BPM_BIT_DARK
        bad_rows, bad_cols = check_bad_rows_cols(bpm, row_threshold=0.5)
        assert bad_rows[3]
        assert not np.any(bad_cols)

    def test_bad_col_detected(self):
        bpm = create_bpm((20, 20))
        bpm[:, 7] = BPM_BIT_DEAD
        bad_rows, bad_cols = check_bad_rows_cols(bpm, col_threshold=0.5)
        assert bad_cols[7]
        assert not np.any(bad_rows)

    def test_clean_bpm(self):
        bpm = create_bpm((20, 20))
        bad_rows, bad_cols = check_bad_rows_cols(bpm)
        assert not np.any(bad_rows)
        assert not np.any(bad_cols)


class TestFlagBadRowsCols:
    def test_returns_copy_not_in_place(self):
        bpm = create_bpm((20, 20))
        bpm[5, :] = BPM_BIT_DARK
        original_id = id(bpm)
        result = flag_bad_rows_cols(bpm, row_threshold=0.5)
        assert id(result) != original_id
        # Original should be unchanged
        assert not (bpm[5, 0] & BPM_BIT_BAD_ROW)

    def test_bad_row_flagged_in_result(self):
        bpm = create_bpm((20, 20))
        bpm[5, :] = BPM_BIT_DARK
        result = flag_bad_rows_cols(bpm, row_threshold=0.5)
        assert result[5, 0] & BPM_BIT_BAD_ROW

    def test_original_pixel_bits_preserved(self):
        bpm = create_bpm((20, 20))
        bpm[5, :] = BPM_BIT_DARK
        result = flag_bad_rows_cols(bpm, row_threshold=0.5)
        # Original DARK bit should still be set alongside the new BAD_ROW bit
        assert result[5, 0] & BPM_BIT_DARK
        assert result[5, 0] & BPM_BIT_BAD_ROW

    def test_active_region(self):
        bpm = create_bpm((20, 20))
        bpm[0, :] = BPM_BIT_DARK   # row 0 bad, but outside active_region
        active = (slice(2, 18), slice(2, 18))
        result = flag_bad_rows_cols(bpm, row_threshold=0.5, active_region=active)
        assert not (result[0, 0] & BPM_BIT_BAD_ROW)


class TestUtilsGetDefectMask:
    """utils.get_defect_mask is a wrapper — confirm it delegates correctly."""

    def test_equivalence_with_bitmask_version(self, small_bpm):
        from fits_tools.defects.bitmask import get_defect_mask as bm_get
        assert np.array_equal(
            utils_get_defect_mask(small_bpm, [BPM_BIT_DARK]),
            bm_get(small_bpm, [BPM_BIT_DARK])
        )


class TestUtilsFilterBpm:
    """utils.filter_bpm is a wrapper — confirm it delegates correctly."""

    def test_equivalence_with_bitmask_version(self, small_bpm):
        from fits_tools.defects.bitmask import filter_bpm as bm_filter
        assert np.array_equal(
            utils_filter_bpm(small_bpm, [BPM_BIT_DARK]),
            bm_filter(small_bpm, [BPM_BIT_DARK])
        )


# ---------------------------------------------------------------------------
# Optional: FITS round-trip test
# Requires astropy and a writable tmpdir (provided by pytest's tmp_path).
# Skipped automatically if astropy is not installed.
# ---------------------------------------------------------------------------

try:
    from astropy.io import fits as _fits
    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy not installed")
class TestFitsRoundTrip:
    def test_save_and_load(self, tmp_path, small_bpm):
        from fits_tools.defects.utils import save_bad_pixel_map, load_bad_pixel_map
        path = tmp_path / "test_bpm.fits"
        save_bad_pixel_map(path, small_bpm)
        loaded, header = load_bad_pixel_map(path)
        assert loaded is not None
        assert np.array_equal(loaded, small_bpm)

    def test_header_contains_bpmver(self, tmp_path, small_bpm):
        from fits_tools.defects.utils import save_bad_pixel_map, load_bad_pixel_map
        path = tmp_path / "test_bpm_header.fits"
        save_bad_pixel_map(path, small_bpm)
        _, header = load_bad_pixel_map(path)
        assert 'BPMVER' in header

    def test_load_missing_file_returns_none(self, tmp_path):
        from fits_tools.defects.utils import load_bad_pixel_map
        result, header = load_bad_pixel_map(tmp_path / "nonexistent.fits")
        assert result is None
        assert header is None

    def test_load_corrupt_file_raises(self, tmp_path):
        from fits_tools.defects.utils import load_bad_pixel_map
        corrupt = tmp_path / "corrupt.fits"
        corrupt.write_bytes(b"this is not a FITS file")
        with pytest.raises(Exception):   # re-raised after logging
            load_bad_pixel_map(corrupt)

    def test_dq_def_extension_present(self, tmp_path, small_bpm):
        """save_bad_pixel_map should embed a DQ_DEF binary table extension."""
        from fits_tools.defects.utils import save_bad_pixel_map
        path = tmp_path / "bpm_with_dqdef.fits"
        save_bad_pixel_map(path, small_bpm)
        with _fits.open(path) as hdul:
            names = [h.name for h in hdul]
        assert 'DQ_DEF' in names

    def test_merge_bad_pixel_maps(self, tmp_path, small_bpm):
        from fits_tools.defects.utils import (
            save_bad_pixel_map, merge_bad_pixel_maps, load_bad_pixel_map
        )
        new_defects = create_bpm(small_bpm.shape)
        new_defects[2, 2] = BPM_BIT_COSMIC

        existing_path = tmp_path / "existing.fits"
        new_path      = tmp_path / "new.fits"
        output_path   = tmp_path / "merged.fits"
        save_bad_pixel_map(existing_path, small_bpm)
        save_bad_pixel_map(new_path, new_defects)

        stats = merge_bad_pixel_maps(existing_path, new_path, output_path, stage='COSMIC_STAGE')
        merged, _ = load_bad_pixel_map(output_path)

        # All original defects still present
        assert merged[0, 0] & BPM_BIT_BIAS
        # New defect added
        assert merged[2, 2] & BPM_BIT_COSMIC
        assert stats['TOTAL']['count'] == 5   # 4 original + 1 new

    def test_bpm_source_dynamic(self, tmp_path, small_bpm):
        """BPMSRC header should not be restricted to BIAS/DARK/FLAT."""
        from fits_tools.defects.utils import (
            save_bad_pixel_map, merge_bad_pixel_maps, load_bad_pixel_map
        )
        new_defects = create_bpm(small_bpm.shape)
        new_defects[2, 2] = BPM_BIT_TELEGRAPH

        existing_path = tmp_path / "ex.fits"
        new_path      = tmp_path / "new2.fits"
        output_path   = tmp_path / "merged2.fits"
        save_bad_pixel_map(existing_path, small_bpm)
        save_bad_pixel_map(new_path, new_defects)

        merge_bad_pixel_maps(
            existing_path, new_path, output_path, stage='CUSTOM_STAGE'
        )
        _, header = load_bad_pixel_map(output_path)
        bpmsrc = header.get('BPMSRC', '')
        # Must contain the custom stage name, not a hardcoded string
        assert 'CUSTOM_STAGE' in bpmsrc or 'TELEGRAPH' in bpmsrc
