# tests/test_detection.py

import pytest
import numpy as np
from fits_tools.defects.detection import *
from fits_tools.bitmask import *

class TestSigmaClipStats:
    """Test sigma clipping edge cases."""
    
    def test_constant_data(self):
        """Constant data should return std=0."""
        data = np.ones((100, 100)) * 1000.0
        median, std = sigma_clip_stats(data)
        assert median == 1000.0
        assert std == 0.0
    
    def test_all_nans(self):
        """All NaN data should return NaN stats."""
        data = np.full((100, 100), np.nan)
        median, std = sigma_clip_stats(data)
        assert np.isnan(median)
        assert np.isnan(std)
    
    def test_all_masked(self):
        """Fully masked data should return NaN stats."""
        data = np.ones((100, 100))
        mask = np.ones((100, 100), dtype=bool)
        median, std = sigma_clip_stats(data, mask=mask)
        assert np.isnan(median)
        assert np.isnan(std)
    
    def test_mad_scale(self):
        """Test MAD scale estimator."""
        data = np.random.normal(1000, 10, (100, 100))
        median_std, std_std = sigma_clip_stats(data, scale='std')
        median_mad, std_mad = sigma_clip_stats(data, scale='mad')
        
        # Medians should be similar
        assert np.abs(median_std - median_mad) < 1.0
        # MAD-derived sigma should be close to std (for Gaussian data)
        assert np.abs(std_std - std_mad) < 2.0
    
    def test_outliers_removed(self):
        """Sigma clipping should remove outliers."""
        data = np.random.normal(1000, 10, (100, 100))
        data[0, 0] = 10000  # Extreme outlier
        
        median, std = sigma_clip_stats(data, sigma_high=3.0)
        
        # Median should be unaffected
        assert 990 < median < 1010
        # Std should be close to 10
        assert 8 < std < 12


class TestDetectBiasDefects:
    """Test bias defect detection."""
    
    def test_no_defects(self):
        """Clean bias frame should have no defects."""
        bias = np.random.normal(1000, 5, (100, 100))
        bpm, stats = detect_bright_pixels(bias)
        
        assert stats['num_defects'] == 0
        assert np.sum(bpm) == 0
    
    def test_absolute_method(self):
        """Test absolute threshold method."""
        bias = np.ones((100, 100)) * 1000.0
        bias[50, 50] = 1200  # +200 ADU offset
        
        bpm, stats = detect_bright_pixels(
            bias, 
            threshold_adu=100.0, 
            method='absolute'
        )
        
        assert stats['num_defects'] == 1
        assert bpm[50, 50] == BPM_BIT_PIX_BRIGHT
    
    def test_sigma_method(self):
        """Test sigma threshold method."""
        bias = np.random.normal(1000, 5, (100, 100))
        bias[50, 50] = 1050  # 10-sigma outlier
        
        bpm, stats = detect_bright_pixels(
            bias, 
            method='sigma',
            sigma=5.0
        )
        
        assert stats['num_defects'] >= 1
        assert has_defect(bpm[50, 50], BPM_BIT_PIX_BRIGHT)
    
    def test_mode_high_only(self):
        """Mode='high' should only flag high outliers."""
        bias = np.ones((100, 100)) * 1000.0
        bias[40, 40] = 1200  # High
        bias[50, 50] = 800   # Low
        
        bpm, stats = detect_bright_pixels(
            bias,
            threshold_adu=100.0,
            mode='high'
        )
        
        assert has_defect(bpm[40, 40], BPM_BIT_PIX_BRIGHT)
        assert not has_defect(bpm[50, 50], BPM_BIT_PIX_BRIGHT)
    
    def test_mode_both(self):
        """Mode='both' should flag both high and low outliers."""
        bias = np.ones((100, 100)) * 1000.0
        bias[40, 40] = 1200  # High
        bias[50, 50] = 800   # Low
        
        bpm, stats = detect_bright_pixels(
            bias,
            threshold_adu=100.0,
            mode='both'
        )
        
        assert has_defect(bpm[40, 40], BPM_BIT_PIX_BRIGHT)
        assert has_defect(bpm[50, 50], BPM_BIT_PIX_BRIGHT)
        assert stats['num_defects'] == 2


class TestDetectDarkHotPixels:
    """Test dark hot pixel detection."""
    
    def test_clean_dark(self):
        """Clean dark frame should have minimal defects."""
        dark = np.random.poisson(10, (100, 100)).astype(float)
        bpm, stats = detect_hot_pixels(dark, threshold_sigma=5.0)
        
        # Should have very few false positives
        assert stats['fraction_defects'] < 0.01
    
    def test_hot_pixels(self):
        """Should detect hot pixels."""
        dark = np.ones((100, 100)) * 10.0
        dark[25, 25] = 100  # Hot pixel
        dark[75, 75] = 150  # Hotter pixel
        
        bpm, stats = detect_hot_pixels(dark, threshold_sigma=3.0)
        
        assert stats['num_defects'] >= 2
        assert has_defect(bpm[25, 25], BPM_BIT_PIX_HOT)
        assert has_defect(bpm[75, 75], BPM_BIT_PIX_HOT)
    
    def test_exclude_existing_bpm(self):
        """Should exclude existing defects from statistics."""
        dark = np.ones((100, 100)) * 10.0
        dark[25, 25] = 1000  # Extreme hot pixel
        
        # First pass - hot pixel biases statistics
        bpm1, stats1 = detect_hot_pixels(dark, threshold_sigma=5.0)
        
        # Second pass - exclude known hot pixel
        existing_bpm = create_bpm((100, 100))
        existing_bpm[25, 25] = BPM_BIT_PIX_BRIGHT  # Mark as bad
        bpm2, stats2 = detect_hot_pixels(
            dark, 
            threshold_sigma=5.0,
            existing_bpm=existing_bpm
        )
        
        # Median should be more accurate in second pass
        assert stats2['median'] < stats1['median']


class TestDetectFlatDefects:
    """Test flat field defect detection."""
    
    def test_uniform_flat(self):
        """Uniform flat should have no defects."""
        flat = np.ones((100, 100)) * 10000.0
        bpm, stats = detect_prnu_anomalies(flat)
        
        assert stats['num_defects'] == 0
    
    def test_hot_and_cold_pixels(self):
        """Should detect both hot and cold pixels."""
        flat = np.ones((100, 100)) * 10000.0
        flat[25, 25] = 12500  # 25% high = hot
        flat[75, 75] = 7500   # 25% low = cold
        
        bpm, stats = detect_prnu_anomalies(
            flat,
            hot_threshold_percent=20.0,
            cold_threshold_percent=20.0
        )
        
        assert has_defect(bpm[25, 25], BPM_BIT_PIX_PRNU_HIGH)
        assert has_defect(bpm[75, 75], BPM_BIT_PIX_PRNU_LOW)
        assert stats['num_hot'] >= 1
        assert stats['num_cold'] >= 1
    
    def test_normalize(self):
        """Test normalization."""
        flat = np.ones((100, 100)) * 5000.0
        flat[50, 50] = 6000  # 20% high
        
        bpm_norm, stats_norm = detect_prnu_anomalies(
            flat,
            normalize=True,
            hot_threshold_percent=15.0
        )
        
        bpm_raw, stats_raw = detect_prnu_anomalies(
            flat,
            normalize=False,
            hot_threshold_percent=15.0
        )
        
        # Normalized should flag the hot pixel
        assert has_defect(bpm_norm[50, 50], BPM_BIT_PIX_PRNU_HIGH)


class TestDetectBadRowsCols:
    """Test row/column defect detection."""
    
    def test_no_structural_defects(self):
        """Sparse defects shouldn't trigger row/col flags."""
        bpm = create_bpm((100, 100))
        bpm[50, 50] = BPM_BIT_PIX_HOT  # Single pixel
        
        result, stats = detect_bad_rows_cols(bpm, row_threshold=0.5)
        
        assert stats['num_bad_rows'] == 0
        assert stats['num_bad_cols'] == 0
    
    def test_bad_column(self):
        """Should detect bad column."""
        bpm = create_bpm((100, 100))
        bpm[:, 50] = BPM_BIT_PIX_HOT  # Entire column
        
        result, stats = detect_bad_rows_cols(bpm, col_threshold=0.1)
        
        assert stats['num_bad_cols'] >= 1
        assert 50 in stats['bad_cols']
        assert has_defect(result[0, 50], BPM_BIT_COL_BAD)
    
    def test_bad_row(self):
        """Should detect bad row."""
        bpm = create_bpm((100, 100))
        bpm[50, :] = BPM_BIT_PIX_HOT  # Entire row
        
        result, stats = detect_bad_rows_cols(bpm, row_threshold=0.1)
        
        assert stats['num_bad_rows'] >= 1
        assert 50 in stats['bad_rows']
        assert has_defect(result[50, 0], BPM_BIT_ROW_BAD)
    
    def test_active_region_negative_slice(self):
        """Test with negative slice indices."""
        bpm = create_bpm((100, 100))
        bpm[:, 90] = BPM_BIT_PIX_HOT  # Column near edge
        
        result, stats = detect_bad_rows_cols(
            bpm,
            active_region=(slice(-20, None), slice(None))
        )
        
        # Should detect bad column in active region
        assert stats['num_bad_cols'] >= 1
    
    def test_active_region_with_step(self):
        """Test with stepped slice."""
        bpm = create_bpm((100, 100))
        bpm[::2, :] = BPM_BIT_PIX_HOT  # Every other row
        
        result, stats = detect_bad_rows_cols(
            bpm,
            active_region=(slice(None, None, 2), slice(None)),
            row_threshold=0.5
        )
        
        # Should detect bad rows
        assert stats['num_bad_rows'] > 0


class TestCreateCalibrationBPM:
    """Test full calibration pipeline."""
    
    def test_clean_frames(self):
        """Clean calibration frames should produce minimal defects."""
        bias = np.random.normal(1000, 5, (100, 100))
        dark = np.random.poisson(10, (100, 100)).astype(float)
        flat = np.ones((100, 100)) * 10000.0
        
        bpm, stats = create_calibration_bpm(bias, dark, flat)
        
        # Should have very few defects
        assert stats['total_fraction'] < 0.05
    
    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        bias = np.ones((100, 100))
        dark = np.ones((100, 100))
        flat = np.ones((200, 200))  # Wrong shape
        
        with pytest.raises(ValueError, match="same shape"):
            create_calibration_bpm(bias, dark, flat)
    
    def test_cumulative_masking(self):
        """Later stages should exclude earlier defects."""
        bias = np.ones((100, 100)) * 1000.0
        bias[25, 25] = 1200  # Bias defect
        
        dark = np.ones((100, 100)) * 10.0
        dark[25, 25] = 1000  # Also high in dark
        
        flat = np.ones((100, 100)) * 10000.0
        
        bpm, stats = create_calibration_bpm(bias, dark, flat)
        
        # Pixel should be flagged in bias stage
        assert has_defect(bpm[25, 25], BPM_BIT_PIX_BRIGHT)
        
        # Dark stats should exclude this pixel
        # (hard to test directly, but check that it didn't
        # drastically affect dark median)
        assert 5 < stats['dark']['median'] < 15
    
    def test_custom_config(self):
        """Should accept custom configuration."""
        bias = np.ones((100, 100)) * 1000.0
        dark = np.ones((100, 100)) * 10.0
        flat = np.ones((100, 100)) * 10000.0
        
        config = get_default_detection_config()
        config['bias']['threshold_adu'] = 50.0  # More sensitive
        
        bpm, stats = create_calibration_bpm(bias, dark, flat, config=config)
        
        assert stats['bias']['threshold_offset_adu'] == 50.0


# Run with: pytest tests/test_detection.py -v