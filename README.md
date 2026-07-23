# ISW FITS Tools

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

When installed from PyPI, the distribution name is `isw-fits-tools`:

```bash
pip install isw-fits-tools
```

The Python import package remains `fits_tools`:

```python
import fits_tools
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
calibrated = calibrate_image(
    "image.fits",
    master_bias="bias.fits",
    master_dark="dark.fits",
    master_flat="flat.fits",
)

# Calculate statistics
result = stats("image.fits")
print(f"Mean: {result.mean}, Std: {result.std}")
```

## Requirements

- Python >= 3.11
- numpy >= 1.23
- astropy >= 5.0
- scipy >= 1.10

Optional extras:

- `pip install "isw-fits-tools[plotting]"` for matplotlib-based reports
- `pip install "isw-fits-tools[progress]"` for tqdm progress bars

## Experimental and Unimplemented Detectors

The public crosstalk detector is not implemented and raises
`NotImplementedError` until instrument-specific channel geometry is defined.
The advanced nonlinearity, per-pixel noise, RTS, charge-trap, and
amplifier-glow detector entry points are placeholders that emit warnings and
return empty masks; they must not be used for scientific decisions. Draft
implementations are available only through the explicitly experimental
advanced-detection workflow.

## License

GPL-3.0 — see [LICENSE](LICENSE) and [copyright](copyright) for details.
