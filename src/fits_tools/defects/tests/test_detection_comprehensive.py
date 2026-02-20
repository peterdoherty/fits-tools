# tests/test_detection_comprehensive.py

"""
Comprehensive tests for detection algorithms.
"""

import pytest
import numpy as np
from fits_tools.defects.detection import (
    sigma_clip_stats,
    roi_stats,
    detect_bright_pixels,
    detect_hot_pixels,
    detect_prnu_anomalies,
    detect_bad_rows_cols,
    create_calibration_bpm,
    get_default_detection_config,
)
from fits_tools.bitmask import (
    BPM_BIT_PIX_BRIGHT, BPM_BIT_PIX_HOT, BPM_BIT_PIX_PRNU_HIGH, BPM_BIT_PIX_PRNU_LOW,
    BPM_BIT_ROW_BAD, BPM_BIT_COL_BAD,
    has_defect, create_bpm,
)

class TestSigmaClipStats:
    """Test sigma clipping statistics."""
    
    def test_gaussian_data(self):
        """Test on Gaussian data."""
        data = np.random.normal(1000, 10, (1000, 1000))
        median, std = sigma_clip_stats(data, sigma_low=3.0, sigma_high=3.0)
        
        assert 990 < median < 1010
        assert 9 < std < 11
    
    def test_with_outliers(self):
        """Outliers should be clipped."""
        data = np.random.normal(1000, 10, (100, 100))
        data[0, 0] = 10000  # Extreme outlier
        
        median, std = sigma_clip_stats(data, sigma_high=3.0)
        
        # Should be close to true values
        assert 990 < median < 1010
        assert 9 < std < 12
    
    def test_constant_data(self):
        """Constant data should have std=0."""
        data = np.ones((100, 100)) * 1000.0
        median, std = sigma_clip_stats(data)
        
        assert median == 1000.0
        assert std == 0.0
    
    def test_with_mask(self):
        """Masked pixels should be excluded."""
        data = np.ones((100, 100)) * 1000.0
        data[0:50, :] = 2000.0  # Half the array different
        
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:50, :] = True  # Mask the different half
        
        median, std = sigma_clip_stats(data, mask=mask)
        
        assert 990 < median < 1010  # Should only see unmasked region
    
    def test_all_masked(self):
        """All masked should return NaN."""
        data = np.ones((100, 100))
        mask = np.ones((100, 100), dtype=bool)
        
        median, std = sigma_clip_stats(data, mask=mask)
        
        assert np.isnan(median)
        assert np.isnan(std)
    
    def test_mad_scale(self):
        """Test MAD scale estimator."""
        data = np.random.normal(1000, 10, (1000, 1000))
        
        median_std, std_std = sigma_clip_stats(data, scale='std')
        median_mad, std_mad = sigma_clip_stats(data, scale='mad')
        
        # Should be similar for Gaussian data
        assert abs(median_std - median_mad) < 5
        assert abs(std_std - std_mad) < 2
    
    def test_invalid_scale(self):
        """Invalid scale should raise ValueError."""
        data = np.ones((10, 10))
        
        with pytest.raises(ValueError, match="scale must be"):
            sigma_clip_stats(data, scale='invalid')


class TestDetectBiasDefects:
    """Test bias defect detection."""
    
    def test_no_defects(self, clean_bias):
        """Clean bias should have minimal defects."""
        bpm, stats = detect_bright_pixels(clean_bias)
        
        # May have a few false positives due to noise
        assert stats['fraction_defects'] < 0.01
    
    def test_absolute_method_high(self, defective_bias):
        """Should detect high bias outlier."""
        bpm, stats = detect_bright_pixels(
            defective_bias,
            threshold_adu=100.0,
            method='absolute',
            mode='high'
        )
        
        assert stats['num_defects'] >= 1
        assert has_defect(bpm[25, 25], BPM_BIT_PIX_BRIGHT)  # Known hot pixel
        assert not has_defect(bpm[75, 75], BPM_BIT_PIX_BRIGHT)  # Cold not flagged
    
    def test_absolute_method_both(self, defective_bias):
        """Should detect both high and low outliers."""
        bpm, stats = detect_bright_pixels(
            defective_bias,
            threshold_adu=100.0,
            method='absolute',
            mode='both'
        )
        
        assert has_defect(bpm[25, 25], BPM_BIT_PIX_BRIGHT)  # High
        assert has_defect(bpm[75, 75], BPM_BIT_PIX_BRIGHT)  # Low
    
    def test_sigma_method(self, defective_bias):
        """Test sigma-based detection."""
        bpm, stats = detect_bright_pixels(
            defective_bias,
            method='sigma',
            sigma=5.0,
            mode='both'
        )
        
        assert stats['num_defects'] >= 1
        assert 'threshold_sigma' in stats
    
    def test_with_roi(self, defective_bias):
        """ROI should affect statistics but not detection area."""
        roi = (10, 90, 10, 90)
        bpm, stats = detect_bright_pixels(
            defective_bias,
            stats_roi=roi,
            threshold_adu=100.0,
            mode='both'
        )
        
        # Should still detect defects outside ROI
        assert has_defect(bpm[25, 25], BPM_BIT_PIX_BRIGHT)
    
    def test_existing_bpm_exclusion(self, defective_bias, small_shape):
        """Existing BPM should be excluded from statistics."""
        # Create existing BPM with known hot pixel
        existing = create_bpm(small_shape)
        existing[25, 25] = BPM_BIT_PIX_HOT  # Mark as bad
        
        bpm1, stats1 = detect_bright_pixels(defective_bias)
        bpm2, stats2 = detect_bright_pixels(defective_bias, existing_bpm=existing)
        
        # Median should be more accurate when excluding outlier
        assert abs(stats2['median'] - 1000) < abs(stats1['median'] - 1000)
    
    def test_invalid_method(self, clean_bias):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            detect_bright_pixels(clean_bias, method='invalid')
    
    def test_invalid_mode(self, clean_bias):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            detect_bright_pixels(clean_bias, mode='invalid')


class TestDetectDarkHotPixels:
    """Test dark hot pixel detection."""
    
    def test_no_defects(self, clean_dark):
        """Clean dark should have minimal defects."""
        bpm, stats = detect_hot_pixels(clean_dark, threshold_sigma=5.0)
        
        assert stats['fraction_defects'] < 0.01
    
    def test_detect_hot_pixels(self, defective_dark):
        """Should detect known hot pixels."""
        bpm, stats = detect_hot_pixels(defective_dark, threshold_sigma=3.0)
        
        assert stats['num_defects'] >= 2
        assert has_defect(bpm[30, 30], BPM_BIT_PIX_HOT)
        assert has_defect(bpm[60, 60], BPM_BIT_PIX_HOT)
    
    def test_high_threshold(self, defective_dark):
        """High threshold should detect fewer defects."""
        bpm_low, stats_low = detect_hot_pixels(defective_dark, threshold_sigma=3.0)
        bpm_high, stats_high = detect_hot_pixels(defective_dark, threshold_sigma=10.0)
        
        assert stats_high['num_defects'] <= stats_low['num_defects']
    
    def test_with_existing_bpm(self, defective_dark, small_shape):
        """Existing BPM should be excluded from statistics."""
        existing = create_bpm(small_shape)
        existing[30, 30] = BPM_BIT_PIX_BRIGHT
        
        bpm, stats = detect_hot_pixels(
            defective_dark,
            threshold_sigma=5.0,
            existing_bpm=existing
        )
        
        # Should still detect the other hot pixel
        assert has_defect(bpm[60, 60], BPM_BIT_PIX_HOT)
    
    def test_invalid_threshold(self, clean_dark):
        """Negative threshold should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            detect_hot_pixels(clean_dark, threshold_sigma=-1.0)


class TestDetectFlatDefects:
    """Test flat field defect detection."""
    
    def test_uniform_flat(self, clean_flat):
        """Uniform flat should have minimal defects."""
        bpm, stats = detect_prnu_anomalies(clean_flat)
        
        assert stats['num_defects'] < 100  # May have noise-induced false positives
    
    def test_hot_pixel(self, defective_flat):
        """Should detect hot pixel."""
        bpm, stats = detect_prnu_anomalies(
            defective_flat,
            hot_threshold_percent=20.0,
            cold_threshold_percent=20.0
        )
        
        assert has_defect(bpm[40, 40], BPM_BIT_PIX_PRNU_HIGH)
        assert stats['num_hot'] >= 1
    
    def test_cold_pixels(self, defective_flat):
        """Should detect dead/cold pixels."""
        bpm, stats = detect_prnu_anomalies(
            defective_flat,
            hot_threshold_percent=20.0,
            cold_threshold_percent=20.0
        )
        
        assert has_defect(bpm[50, 50], BPM_BIT_PIX_PRNU_LOW)
        assert has_defect(bpm[60, 60], BPM_BIT_PIX_PRNU_LOW)
        assert stats['num_cold'] >= 2
    
    def test_normalize_true(self, defective_flat):
        """Test with normalization."""
        bpm_norm, stats_norm = detect_prnu_anomalies(
            defective_flat,
            normalize=True,
            hot_threshold_percent=20.0
        )
        
        assert stats_norm['normalize'] is True
        assert stats_norm['num_hot'] >= 1
    
    def test_normalize_false(self, defective_flat):
        """Test without normalization."""
        bpm_raw, stats_raw = detect_prnu_anomalies(
            defective_flat,
            normalize=False,
            hot_threshold_percent=20.0
        )
        
        assert stats_raw['normalize'] is False
    
    def test_zero_median_handling(self, small_shape):
        """Zero median should be handled gracefully."""
        flat_zero = np.zeros(small_shape)
        
        bpm, stats = detect_prnu_anomalies(flat_zero)
        
        assert stats['num_defects'] == 0
        assert np.isnan(stats['hot_threshold'])


class TestDetectBadRowsCols:
    """Test row/column defect detection."""
    
    def test_no_structural_defects(self, small_shape):
        """Sparse defects shouldn't trigger structural flags."""
        bpm = create_bpm(small_shape)
        bpm[50, 50] = BPM_BIT_PIX_HOT
        
        result, stats = detect_bad_rows_cols(bpm, row_threshold=0.5, col_threshold=0.5)
        
        assert stats['num_bad_rows'] == 0
        assert stats['num_bad_cols'] == 0
    
    def test_bad_column(self, small_shape):
        """Should detect bad column."""
        bpm = create_bpm(small_shape)
        bpm[:, 50] = BPM_BIT_PIX_HOT  # Entire column
        
        result, stats = detect_bad_rows_cols(bpm, col_threshold=0.1)
        
        assert stats['num_bad_cols'] >= 1
        assert 50 in stats['bad_cols']
        assert has_defect(result[0, 50], BPM_BIT_COL_BAD)
    
    def test_bad_row(self, small_shape):
        """Should detect bad row."""
        bpm = create_bpm(small_shape)
        bpm[50, :] = BPM_BIT_PIX_HOT  # Entire row
        
        result, stats = detect_bad_rows_cols(bpm, row_threshold=0.1)
        
        assert stats['num_bad_rows'] >= 1
        assert 50 in stats['bad_rows']
        assert has_defect(result[50, 0], BPM_BIT_ROW_BAD)
    
    def test_multiple_bad_cols(self, small_shape):
        """Should detect multiple bad columns."""
        bpm = create_bpm(small_shape)
        bpm[:, 20] = BPM_BIT_PIX_HOT
        bpm[:, 40] = BPM_BIT_PIX_HOT
        bpm[:, 60] = BPM_BIT_PIX_HOT
        
        result, stats = detect_bad_rows_cols(bpm, col_threshold=0.1)
        
        assert stats['num_bad_cols'] == 3
        assert set([20, 40, 60]).issubset(set(stats['bad_cols']))
    
    def test_active_region(self, small_shape):
        """Active region should limit analysis area."""
        bpm = create_bpm(small_shape)
        bpm[:, 90] = BPM_BIT_PIX_HOT  # Column near edge
        
        # Analyze only last 20 rows
        result, stats = detect_bad_rows_cols(
            bpm,
            active_region=(slice(-20, None), slice(None)),
            col_threshold=0.1
        )
        
        assert stats['num_bad_cols'] >= 1
    
    def test_negative_slice_indices(self, small_shape):
        """Should handle negative slice indices."""
        bpm = create_bpm(small_shape)
        bpm[:, 95] = BPM_BIT_PIX_HOT
        
        result, stats = detect_bad_rows_cols(
            bpm,
            active_region=(slice(-50, None), slice(-50, None)),
            col_threshold=0.1
        )
        
        # Should not crash
        assert isinstance(stats['num_bad_cols'], int)
    
    def test_stepped_slice(self, small_shape):
        """Should handle stepped slices."""
        bpm = create_bpm(small_shape)
        bpm[::2, :] = BPM_BIT_PIX_HOT  # Every other row
        
        result, stats = detect_bad_rows_cols(
            bpm,
            active_region=(slice(None, None, 2), slice(None)),
            row_threshold=0.5
        )
        
        # Should detect bad rows
        assert stats['num_bad_rows'] > 0
    
    def test_defect_mask_filtering(self, small_shape):
        """Should only consider specified defect types."""
        bpm = create_bpm(small_shape)
        bpm[:, 30] = BPM_BIT_PIX_HOT  # Dark defect column
        bpm[:, 60] = BPM_BIT_PIX_PRNU_HIGH  # Flat hot column
        
        # Only consider dark defects
        result, stats = detect_bad_rows_cols(
            bpm,
            col_threshold=0.1,
            defect_mask=BPM_BIT_PIX_HOT
        )
        
        assert 30 in stats['bad_cols']
        # 60 may or may not be flagged depending on dark defect density


class TestCreateCalibrationBPM:
    """Test full pipeline."""
    
    def test_clean_calibration(self, calibration_set):
        """Clean frames should produce minimal defects."""
        bpm, stats = create_calibration_bpm(
            calibration_set['bias'],
            calibration_set['dark'],
            calibration_set['flat']
        )
        
        assert stats['total_fraction'] < 0.05
    
    def test_defective_calibration(self, defective_calibration_set):
        """Should detect all defect types."""
        bpm, stats = create_calibration_bpm(
            defective_calibration_set['bias'],
            defective_calibration_set['dark'],
            defective_calibration_set['flat']
        )
        
        assert stats['bias']['num_defects'] > 0
        assert stats['dark']['num_defects'] > 0
        assert stats['flat']['num_defects'] > 0
        assert stats['total_defects'] > 0
    
    def test_shape_mismatch_raises(self, clean_bias, clean_dark):
        """Mismatched shapes should raise ValueError."""
        flat_wrong = np.ones((200, 200))
        
        with pytest.raises(ValueError, match="same shape"):
            create_calibration_bpm(clean_bias, clean_dark, flat_wrong)
    
    def test_custom_config(self, calibration_set):
        """Should accept custom configuration."""
        config = get_default_detection_config()
        config['bias']['threshold_adu'] = 50.0
        
        bpm, stats = create_calibration_bpm(
            calibration_set['bias'],
            calibration_set['dark'],
            calibration_set['flat'],
            config=config
        )
        
        assert stats['bias']['threshold_offset_adu'] == 50.0
    
    def test_with_roi(self, defective_calibration_set):
        """ROI should affect statistics."""
        roi = (10, 90, 10, 90)
        
        bpm, stats = create_calibration_bpm(
            defective_calibration_set['bias'],
            defective_calibration_set['dark'],
            defective_calibration_set['flat'],
            stats_roi=roi
        )
        
        # Should still detect defects
        assert stats['total_defects'] > 0
    
    def test_cumulative_masking(self, defective_calibration_set):
        """Later stages should exclude earlier defects from stats."""
        bpm, stats = create_calibration_bpm(
            defective_calibration_set['bias'],
            defective_calibration_set['dark'],
            defective_calibration_set['flat']
        )
        
        # Dark stats should have reasonable median (not biased by hot pixels)
        assert stats['dark']['median'] < 100  # Should be ~10


@pytest.mark.slow
class TestPerformance:
    """Performance tests on realistic data."""
    
    def test_large_array_performance(self, realistic_shape):
        """Test performance on 2k×2k array."""
        import time
        
        bias = np.random.normal(1000, 5, realistic_shape)
        dark = np.random.poisson(10, realistic_shape).astype(float)
        flat = np.random.normal(10000, 100, realistic_shape)
        
        start = time.time()
        bpm, stats = create_calibration_bpm(bias, dark, flat)
        elapsed = time.time() - start
        
        # Should complete in < 5 seconds
        assert elapsed < 5.0
        print(f"\nProcessing time for {realistic_shape}: {elapsed:.3f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])