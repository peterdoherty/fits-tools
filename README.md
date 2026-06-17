# FITS Tools

A comprehensive Python toolkit for FITS file manipulation and astronomical image processing.

## Features

- FITS file I/O with robust error handling
- Image calibration (bias, dark, flat correction)
- Cosmic ray detection and removal
- Image combining and stacking
- Statistical analysis tools
- Bad pixel mask (BPM) management
- Quality control utilities

## Installation

### From source
```bash
git clone https://github.com/peterdoherty/fits-tools.git
cd fits-tools
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from fits_tools import get_data_and_header, write_simple_fits, stats
from fits_tools.image_processing.calibration import calibrate_image

# Load a FITS file
data, header = get_data_and_header("image.fits")

# Perform calibration
calibrated = calibrate_image("image.fits", bias="bias.fits", dark="dark.fits", flat="flat.fits")

# Calculate statistics
result = stats("image.fits")
print(f"Mean: {result.mean}, Std: {result.std}")
```

## Requirements

- Python >= 3.11
- numpy >= 1.23
- astropy >= 5.0
- scipy >= 1.10
- matplotlib >= 3.7
- tqdm

## License

GPL-3.0 — see [LICENSE](LICENSE) and [copyright](copyright) for details.
