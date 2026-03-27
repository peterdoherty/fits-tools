# docs/user_guide.md

# User Guide: Defect Detection

## Quick Start

```python
from fits_tools.defects import create_calibration_bpm
from astropy.io import fits

# Load calibration frames
bias = fits.getdata('master_bias.fits')
dark = fits.getdata('master_dark.fits')
flat = fits.getdata('master_flat.fits')

# Detect defects
bpm, stats = create_calibration_bpm(bias, dark, flat)

# Print summary
print(f"Total defects: {stats['total_defects']} ({stats['total_fraction']*100:.2f}%)")

Understanding Detection Stages
1. Bias Defects
What: Pixels with anomalous offset levels
Threshold: Absolute ADU offset from median (default: 100 ADU)
Why: Identifies amplifier glitches, stuck pixels
2. Dark Hot Pixels
What: Pixels with excess dark current
Threshold: Sigma above median (default: 5σ)
Why: Thermal noise, radiation damage
Note: Only flags HIGH values (cold pixels detected in flat)
3. Flat Field Defects
What: Pixels with abnormal quantum efficiency
Threshold: ±20% from median response
Why: Dust, vignetting, dead pixels, QE variations
4. Structural Defects
What: Entire rows/columns with high defect density
Threshold: >10% bad pixels in row/column
Why: Readout issues, amplifier problems
Advanced Usage
Custom Thresholds
python

Copy code
config = get_default_detection_config()

# More sensitive bias detection
config['bias']['threshold_adu'] = 50.0

# More conservative dark detection
config['dark']['threshold_sigma'] = 7.0

# Stricter flat thresholds
config['flat']['hot_threshold_percent'] = 15.0
config['flat']['cold_threshold_percent'] = 15.0

bpm, stats = create_calibration_bpm(
    bias, dark, flat, config=config
)

ROI-Based Statistics
Use a clean central region for statistics:

python

Copy code
# Use center 1800×1800 region (avoiding edges)
bpm, stats = create_calibration_bpm(
    bias, dark, flat,
    stats_roi=(100, 1900, 100, 1900)  # x1, x2, y1, y2
)

Temporal Defect Detection
Detect RTS noise from multiple frames:


from fits_tools.defects import detect_temporal_defects

# Load 10 dark frames
dark_stack = np.array([fits.getdata(f'dark_{i:02d}.fits') for i in range(10)])

# Detect unstable pixels
temporal_bpm, stats = detect_temporal_defects(dark_stack, threshold_sigma=5.0)

Best Practices
Use high SNR calibration frames

Bias: Combine ≥10 frames
Dark: Combine ≥10 frames, long exposure
Flat: Combine ≥10 frames, uniform illumination
Exclude vignetted regions from statistics

python

Copy code
stats_roi=(200, 1800, 200, 1800)  # Avoid edges

Validate visually

python

Copy code
from fits_tools.defects import create_detection_report
create_detection_report(bpm, stats, output_path='report.pdf')

Save with metadata

python

Copy code
from fits_tools.defects.io import save_bpm_to_fits
save_bpm_to_fits(bpm, 'bpm.fits', stats=stats)

Update periodically

Detectors degrade over time
Regenerate BPM every 6-12 months
Track degradation with comparison plots