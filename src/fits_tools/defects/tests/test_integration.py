# tests/test_integration.py

import pytest
import numpy as np
from astropy.io import fits
from fits_tools.defects import create_calibration_bpm, apply_bpm

class TestRealWorldData:
    """Integration tests with synthetic 'real' data."""
    
    @pytest.fixture
    def synthetic_ccd_frames(self):
        """Create realistic synthetic CCD frames."""
        ny, nx = 2048, 2048
        
        # Bias: Gaussian with structure
        bias = np.random.normal(1000, 5, (ny, nx))
        bias += np.linspace(0, 20, nx)  # Ramp (overscan region)
        
        # Add bias defects
        bias[100:110, 500] = 1150  # Column with offset
        
        # Dark: Poisson with hot pixels
        dark = np.random.poisson(10, (ny, nx)).astype(float)
        dark[250, 750] = 500   # Hot pixel
        dark[800:810, 1200] = 200  # Hot column
        
        # Flat: Gaussian with vignetting
        y, x = np.ogrid[:ny, :nx]
        center_y, center_x = ny // 2, nx // 2
        r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        vignette = 1.0 - 0.3 * (r / r.max())**2
        
        flat = np.random.normal(10000, 100, (ny, nx)) * vignette
        flat[600, 1500] = 2000  # Dead pixel
        
        return bias, dark, flat
    
    def test_full_pipeline(self, synthetic_ccd_frames):
        """Test complete calibration pipeline."""
        bias, dark, flat = synthetic_ccd_frames
        
        bpm, stats = create_calibration_bpm(
            bias, dark, flat,
            stats_roi=(200, 1800, 200, 1800)  # Avoid edges
        )
        
        # Should detect known defects
        assert stats['total_defects'] > 0
        
        # Should detect hot column
        assert stats['rows_cols']['num_bad_cols'] > 0
        
        # BPM should be valid
        assert bpm.dtype == np.uint32
        assert bpm.shape == bias.shape
    
    def test_apply_to_science_frame(self, synthetic_ccd_frames):
        """Test applying BPM to science data."""
        bias, dark, flat = synthetic_ccd_frames
        
        # Create BPM
        bpm, _ = create_calibration_bpm(bias, dark, flat)
        
        # Create science frame
        science = np.random.poisson(1000, bias.shape).astype(float)
        
        # Apply BPM
        science_masked = apply_bpm(science, bpm, fill_value=np.nan)
        
        # Known defects should be masked
        assert np.isnan(science_masked[250, 750])  # Hot pixel
        
        # Good pixels should be unchanged
        good_pixel_mask = bpm == 0
        assert np.allclose(
            science[good_pixel_mask],
            science_masked[good_pixel_mask],
            equal_nan=True
        )