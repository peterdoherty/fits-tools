# FITS Tools

A comprehensive Python toolkit for FITS file manipulation and astronomical image processing.

## Features

- FITS file I/O with robust error handling
- Image calibration (bias, dark, flat correction)
- Cosmic ray detection and removal
- Image combining and stacking
- Statistical analysis tools
- Batch processing capabilities
- Quality control utilities

## Installation

### From PyPI (when published)
```bash
pip install fits-tools
```

### From source
```bash
git clone https://github.com/yourusername/fits-tools.git
cd fits-tools
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from fits_tools import io, statistics
from fits_tools.image_processing import calibration

# Load FITS file
data, header = io.read_fits("image.fits")

# Perform calibration
calibrated = calibration.apply_bias_dark_flat(
    data, bias, dark, flat
)

# Calculate statistics
stats = statistics.calculate_image_stats(calibrated)
print(f"Mean: {stats['mean']}, Std: {stats['std']}")
```

## Documentation

Full documentation is available at [link to docs]

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- astropy >= 5.0.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0

## License

MIT License (or your chosen license)

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
```

### .gitignore
```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db