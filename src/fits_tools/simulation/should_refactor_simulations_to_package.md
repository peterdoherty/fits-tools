# Should simulations.py be converted to a package?

## TL;DR: **YES** - This module is an excellent candidate for refactoring into a package.

---

## Current State Analysis

**File Statistics:**
- **Lines of Code:** ~1,977 lines
- **Functions:** 20+ public functions
- **Logical Components:** 6-7 distinct areas
- **Complexity:** High - handles multiple detector types, artifacts, and file formats

**Threshold Rules of Thumb:**
- ✅ **>500 lines**: Consider splitting
- ✅ **>1000 lines**: Should probably split
- ✅ **>1500 lines**: Definitely split
- ✅ **Multiple distinct responsibilities**: Split into modules
- ✅ **Difficult to navigate**: Split into package

**Your file:** 1,977 lines → **Strong candidate for package structure**

---

## Benefits of Converting to Package

### 1. **Improved Code Organization**
- Easier to find specific functionality
- Logical grouping of related functions
- Clearer separation of concerns

### 2. **Better Maintainability**
- Smaller files are easier to understand
- Reduces cognitive load when making changes
- Easier to review in pull requests

### 3. **Enhanced Testing**
- Can test each module independently
- Easier to achieve high test coverage
- Simpler to mock dependencies

### 4. **Improved Import Experience**
```python
# Current (all or nothing):
from fits_tools.simulations import *  # imports 20+ functions

# With package (selective):
from fits_tools.simulations.frames import create_bias_frame
from fits_tools.simulations.artifacts import add_cosmic_rays
from fits_tools.simulations.noise import add_read_noise
```

### 5. **Better Team Collaboration**
- Multiple developers can work on different modules simultaneously
- Reduced merge conflicts
- Clearer ownership of components

### 6. **Easier to Extend**
- Adding new detector types → new module
- Adding new artifacts → add to artifacts module
- Won't clutter existing code

---

## Proposed Package Structure

```
fits_tools/
└── simulations/
    ├── __init__.py                 # Public API exports
    ├── _base.py                    # Core/private utilities (65 lines)
    ├── _constants.py               # Magic numbers and defaults (50 lines)
    ├── noise.py                    # Noise models (200 lines)
    ├── artifacts.py                # Detector artifacts (300 lines)
    ├── frames.py                   # Calibration frame creation (600 lines)
    ├── mef.py                      # Multi-Extension FITS (200 lines)
    ├── timeseries.py               # Time-series generation (200 lines)
    ├── utils.py                    # Helper functions (100 lines)
    ├── cli.py                      # Command-line interface (250 lines)
    └── tests/
        ├── __init__.py
        ├── test_noise.py
        ├── test_artifacts.py
        ├── test_frames.py
        ├── test_mef.py
        └── test_timeseries.py
```

---

## Detailed Module Breakdown

### `__init__.py` - Public API (30-50 lines)
**Purpose:** Define and export the public API

```python
"""
Synthetic FITS data generation for CCD/CMOS testing and simulation.

This package provides utilities to generate realistic synthetic FITS files
for testing calibration pipelines, validation routines, and image processing
algorithms.
"""

__version__ = "0.3.0"
__author__ = "fits_tools team"

# Import main functions for convenient access
from .noise import (
    add_read_noise,
    add_photon_noise,
)

from .artifacts import (
    add_hot_pixels,
    add_cosmic_rays,
    add_cte_trails,
    add_tree_rings,
    add_brighter_fatter_effect,
    create_vignetting_pattern,
)

from .frames import (
    create_bias_frame,
    create_dark_frame,
    create_flat_frame,
    create_science_frame,
    create_calibration_set,
)

from .mef import (
    create_mef_bias_frame,
)

from .timeseries import (
    create_bias_time_series,
    create_bias_pair_for_noise_estimation,
)

# Maintain backward compatibility
from ._base import _create_dummy_fits

__all__ = [
    # Noise
    'add_read_noise',
    'add_photon_noise',
    
    # Artifacts
    'add_hot_pixels',
    'add_cosmic_rays',
    'add_cte_trails',
    'add_tree_rings',
    'add_brighter_fatter_effect',
    'create_vignetting_pattern',
    
    # Frames
    'create_bias_frame',
    'create_dark_frame',
    'create_flat_frame',
    'create_science_frame',
    'create_calibration_set',
    
    # MEF
    'create_mef_bias_frame',
    
    # Time series
    'create_bias_time_series',
    'create_bias_pair_for_noise_estimation',
    
    # Backward compatibility
    '_create_dummy_fits',
]
```

### `_constants.py` - Configuration Constants (50 lines)
**Purpose:** Centralize magic numbers and default values

```python
"""Internal constants for FITS simulations."""

# PSF parameters
DEFAULT_STAR_FLUX_RANGE = (5000, 50000)
DEFAULT_PSF_SIGMA_RANGE = (1.5, 2.5)
DEFAULT_STAR_EDGE_BUFFER = 20  # pixels from edge
DEFAULT_PSF_REGION_SIZE = 15  # half-width of PSF region

# Detector parameters
TYPICAL_CTE_RANGE = (0.999999, 1.0)
TYPICAL_READ_NOISE_RANGE = (1.0, 15.0)  # ADU
TYPICAL_DARK_CURRENT_RANGE = (0.001, 10.0)  # e-/pix/sec
TYPICAL_BIAS_LEVEL_RANGE = (50.0, 500.0)  # ADU

# Cosmic ray parameters
DEFAULT_CR_ENERGY_RANGE = (1000, 50000)
CR_PATTERN_TYPES = ['point', 'cross', 'blob']

# Hot pixel parameters
DEFAULT_HOT_PIXEL_RANGE = (1000, 10000)

# Noise parameters
DEFAULT_GAIN = 1.0  # e-/ADU
DEFAULT_READ_NOISE = 5.0  # ADU

# File naming
BIAS_FILENAME_TEMPLATE = 'bias_{:03d}.fits'
DARK_FILENAME_TEMPLATE = 'dark_{:03d}.fits'
FLAT_FILENAME_TEMPLATE = 'flat_{:03d}.fits'
SCIENCE_FILENAME_TEMPLATE = 'science_{:03d}.fits'
```

### `_base.py` - Core Utilities (65 lines)
**Purpose:** Low-level FITS file creation (backward compatible)

```python
"""Core FITS file creation utilities."""

import numpy as np
from astropy.io import fits
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _create_dummy_fits(
    filepath: str,
    nx: int = 10,
    ny: int = 10,
    primary_data_value: Any = 0.0,
    noise_level: float = 0.0,
    primary_keywords: Optional[Dict[str, Any]] = None,
    extensions_info: Optional[List[Dict[str, Any]]] = None,
    clip_negative: bool = True
) -> None:
    """
    Creates a dummy FITS file with a primary HDU and optional extensions.
    
    This is the core function used by all other frame creation functions.
    Kept for backward compatibility.
    """
    # ... implementation from lines 65-218 ...
```

### `noise.py` - Noise Models (~200 lines)
**Purpose:** All noise-related functions

```python
"""Noise models for CCD/CMOS detectors."""

import numpy as np
from typing import Optional
import logging

from ..utils import validate_numeric
from ._constants import DEFAULT_GAIN, DEFAULT_READ_NOISE

logger = logging.getLogger(__name__)


def add_read_noise(
    data: np.ndarray,
    read_noise: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """Add Gaussian read noise to image data."""
    # ... implementation from lines 225-261 ...


def add_photon_noise(
    data: np.ndarray,
    gain: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """Add Poisson photon noise to image data."""
    # ... implementation from lines 264-302 ...


class SeedGenerator:
    """Generate consistent, non-overlapping seeds from a base seed."""
    
    def __init__(self, base_seed: Optional[int] = None):
        self.base_seed = base_seed
        self.counter = 0
    
    def next(self) -> Optional[int]:
        """Get next seed in sequence."""
        if self.base_seed is None:
            return None
        seed = self.base_seed + self.counter * 1000
        self.counter += 1
        return seed
```

### `artifacts.py` - Detector Artifacts (~300 lines)
**Purpose:** All detector artifact functions

```python
"""Detector artifacts: hot pixels, cosmic rays, CTE trails, tree rings, etc."""

import numpy as np
from scipy import ndimage
from typing import Optional, Tuple
import logging

from ..utils import validate_numeric
from ._constants import (
    DEFAULT_HOT_PIXEL_RANGE,
    DEFAULT_CR_ENERGY_RANGE,
    CR_PATTERN_TYPES,
)

logger = logging.getLogger(__name__)


def add_hot_pixels(
    data: np.ndarray,
    num_hot_pixels: int = 0,
    hot_value_range: Tuple[float, float] = DEFAULT_HOT_PIXEL_RANGE,
    seed: Optional[int] = None
) -> np.ndarray:
    """Add hot pixels to image data."""
    # ... implementation from lines 305-356 ...


def add_cosmic_rays(
    data: np.ndarray,
    num_cosmic_rays: int = 0,
    energy_range: Tuple[float, float] = DEFAULT_CR_ENERGY_RANGE,
    seed: Optional[int] = None
) -> np.ndarray:
    """Add cosmic ray hits to image data."""
    # ... implementation from lines 359-424 ...


def create_vignetting_pattern(
    shape: Tuple[int, int],
    vignetting_factor: float = 0.2
) -> np.ndarray:
    """Create a vignetting (illumination fall-off) pattern."""
    # ... implementation from lines 427-468 ...


def add_cte_trails(
    data: np.ndarray,
    cte_efficiency: float = 0.999999,
    direction: str = 'vertical',
    seed: Optional[int] = None
) -> np.ndarray:
    """Add Charge Transfer Efficiency (CTE) trails."""
    # ... implementation from lines 475-534 ...


def add_tree_rings(
    data: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    amplitude: float = 0.02,
    num_rings: int = 5,
    seed: Optional[int] = None
) -> np.ndarray:
    """Add tree ring pattern (silicon crystal defects)."""
    # ... implementation from lines 537-604 ...


def add_brighter_fatter_effect(
    data: np.ndarray,
    strength: float = 1e-6,
    seed: Optional[int] = None
) -> np.ndarray:
    """Add brighter-fatter effect."""
    # ... implementation from lines 607-660 ...
```

### `frames.py` - Calibration Frames (~600 lines)
**Purpose:** All calibration frame creation functions

```python
"""Calibration frame creation: bias, dark, flat, science."""

import os
import numpy as np
from astropy.io import fits
from typing import Tuple, Dict, Any, Optional, List
import logging

from ..constants import (
    FITS_KEY_IMGTYPE,
    FITS_KEY_EXPTIME,
    FITS_KEY_RDNOISE,
    FITS_KEY_GAIN,
    FITS_KEY_DATASEC,
    FITS_KEY_BIASSEC,
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
)
from ..utils import validate_numeric
from ..exceptions import ImageProcessingError

from ._base import _create_dummy_fits
from ._constants import *
from .noise import add_read_noise, add_photon_noise, SeedGenerator
from .artifacts import (
    add_hot_pixels,
    add_cosmic_rays,
    create_vignetting_pattern,
)
from .utils import _apply_overscan_sections, _add_stars_to_region

logger = logging.getLogger(__name__)


def create_bias_frame(
    filepath: str,
    shape: Tuple[int, int] = (512, 512),
    bias_level: float = 100.0,
    read_noise: float = 5.0,
    num_hot_pixels: int = 0,
    add_sections: bool = False,
    overscan_width: int = 32,
    seed: Optional[int] = None,
    extra_keywords: Optional[Dict[str, Any]] = None
) -> None:
    """Create a realistic synthetic bias frame."""
    # ... implementation from lines 667-782 ...


def create_dark_frame(
    filepath: str,
    shape: Tuple[int, int] = (512, 512),
    exposure_time: float = 300.0,
    bias_level: float = 100.0,
    dark_current: float = 0.1,
    dark_variation: float = 0.1,
    read_noise: float = 5.0,
    gain: float = 1.0,
    num_hot_pixels: int = 5,
    add_sections: bool = False,
    overscan_width: int = 32,
    seed: Optional[int] = None,
    extra_keywords: Optional[Dict[str, Any]] = None
) -> None:
    """Create a realistic synthetic dark frame."""
    # ... implementation from lines 785-970 ...


def create_flat_frame(
    filepath: str,
    shape: Tuple[int, int] = (512, 512),
    exposure_time: float = 10.0,
    mean_level: float = 20000.0,
    bias_level: float = 100.0,
    read_noise: float = 5.0,
    gain: float = 1.0,
    vignetting: float = 0.2,
    dust_specs: int = 0,
    add_sections: bool = False,
    overscan_width: int = 32,
    seed: Optional[int] = None,
    extra_keywords: Optional[Dict[str, Any]] = None
) -> None:
    """Create a realistic synthetic flat field frame."""
    # ... implementation from lines 973-1132 ...


def create_science_frame(
    filepath: str,
    shape: Tuple[int, int] = (512, 512),
    exposure_time: float = 60.0,
    sky_level: float = 1000.0,
    bias_level: float = 100.0,
    read_noise: float = 5.0,
    gain: float = 1.0,
    num_stars: int = 10,
    num_cosmic_rays: int = 3,
    add_sections: bool = False,
    overscan_width: int = 32,
    seed: Optional[int] = None,
    extra_keywords: Optional[Dict[str, Any]] = None
) -> None:
    """Create a realistic synthetic science/light frame."""
    # ... implementation from lines 1135-1326 ...


def create_calibration_set(
    output_dir: str,
    num_bias: int = 10,
    num_dark: int = 10,
    num_flat: int = 10,
    shape: Tuple[int, int] = (512, 512),
    dark_exposure_time: float = 300.0,
    flat_exposure_time: float = 10.0,
    **kwargs
) -> Dict[str, List[str]]:
    """Create a complete set of calibration frames."""
    # ... implementation from lines 1329-1407 ...
```

### `utils.py` - Helper Functions (~100 lines)
**Purpose:** Shared utility functions used by multiple modules

```python
"""Helper functions for frame creation."""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable
import logging

from ..constants import (
    FITS_KEY_DATASEC,
    FITS_KEY_BIASSEC,
)

logger = logging.getLogger(__name__)


def _apply_overscan_sections(
    data: np.ndarray,
    keywords: Dict[str, Any],
    overscan_width: int,
    process_data_region_func: Callable,
    process_overscan_region_func: Callable,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply overscan sections to image data.
    
    Refactored helper to eliminate code duplication.
    """
    ny, nx = data.shape
    data_cols = nx - overscan_width
    
    if data_cols <= 0:
        raise ValueError(
            f"Overscan width ({overscan_width} pixels) must be less than "
            f"image width ({nx} pixels). Data section would have "
            f"{data_cols} pixels."
        )
    
    # Process data region
    data_region = data[:, :data_cols].copy()
    data_region = process_data_region_func(data_region, seed)
    data[:, :data_cols] = data_region
    
    # Process overscan region
    overscan_region = data[:, data_cols:]
    overscan_region = process_overscan_region_func(overscan_region, seed)
    data[:, data_cols:] = overscan_region
    
    # Add section keywords (FITS 1-indexed)
    keywords[FITS_KEY_DATASEC] = f'[1:{data_cols},1:{ny}]'
    keywords[FITS_KEY_BIASSEC] = f'[{data_cols+1}:{nx},1:{ny}]'
    
    return data, keywords


def _add_stars_to_region(
    data: np.ndarray,
    num_stars: int,
    rng: np.random.Generator,
    flux_range: Tuple[float, float] = (5000, 50000),
    sigma_range: Tuple[float, float] = (1.5, 2.5),
    edge_buffer: int = 20
) -> np.ndarray:
    """
    Add Gaussian PSF stars to data region.
    
    Refactored helper to eliminate code duplication.
    """
    ny, nx = data.shape
    result = data.copy()
    
    for _ in range(num_stars):
        # Random position (avoid edges)
        star_y = rng.integers(edge_buffer, ny - edge_buffer)
        star_x = rng.integers(edge_buffer, nx - edge_buffer)
        
        # Random brightness and PSF width
        star_flux = rng.uniform(*flux_range)
        sigma = rng.uniform(*sigma_range)
        
        # Create small region around star
        psf_size = 15
        y, x = np.ogrid[
            max(0, star_y - psf_size):min(ny, star_y + psf_size),
            max(0, star_x - psf_size):min(nx, star_x + psf_size)
        ]
        
        psf = star_flux * np.exp(
            -((x - star_x)**2 + (y - star_y)**2) / (2 * sigma**2)
        )
        
        result[
            max(0, star_y - psf_size):min(ny, star_y + psf_size),
            max(0, star_x - psf_size):min(nx, star_x + psf_size)
        ] += psf
    
    return result
```

### `mef.py` - Multi-Extension FITS (~200 lines)
**Purpose:** MEF-specific functionality

```python
"""Multi-Extension FITS (MEF) support."""

import os
import numpy as np
from astropy.io import fits
from typing import List, Tuple, Optional, Dict, Any
import logging

from ..utils import validate_numeric
from ._base import _create_dummy_fits
from .frames import create_bias_frame

logger = logging.getLogger(__name__)


def create_mef_bias_frame(
    filepath: str,
    num_amps: int = 4,
    amp_shape: Tuple[int, int] = (512, 512),
    bias_levels: Optional[List[float]] = None,
    read_noises: Optional[List[float]] = None,
    seed: Optional[int] = None,
    **kwargs
) -> None:
    """Create a Multi-Extension FITS bias frame with multiple amplifiers."""
    # ... implementation from lines 1414-1510 ...
```

### `timeseries.py` - Time Series (~200 lines)
**Purpose:** Time-series generation

```python
"""Time-series generation for stability and drift analysis."""

import os
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging

from ..utils import validate_numeric
from .frames import create_bias_frame

logger = logging.getLogger(__name__)


def create_bias_time_series(
    output_dir: str,
    num_frames: int = 100,
    shape: Tuple[int, int] = (512, 512),
    bias_level_start: float = 100.0,
    bias_drift_rate: float = 0.0,
    thermal_amplitude: float = 0.0,
    thermal_cycle_period: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
) -> List[str]:
    """Create a time series of bias frames with drift and thermal cycling."""
    # ... implementation from lines 1513-1606 ...


def create_bias_pair_for_noise_estimation(
    output_dir: str,
    num_pairs: int = 10,
    shape: Tuple[int, int] = (512, 512),
    bias_level: float = 100.0,
    read_noise: float = 5.0,
    **kwargs
) -> Dict[str, List[str]]:
    """Create pairs of bias frames for accurate read noise estimation."""
    # ... implementation from lines 1609-1688 ...
```

### `cli.py` - Command Line Interface (~250 lines)
**Purpose:** Command-line interface

```python
"""Command-line interface for FITS simulation tools."""

import argparse
import logging
import sys
import os
import shutil
import doctest

from . import __version__
from .frames import (
    create_bias_frame,
    create_dark_frame,
    create_flat_frame,
    create_science_frame,
    create_calibration_set,
)
from .mef import create_mef_bias_frame
from .timeseries import create_bias_time_series

logger = logging.getLogger(__name__)


def main():
    """Command-line interface and test suite for fits_simulations."""
    # ... implementation from lines 1731-1976 ...


if __name__ == "__main__":
    sys.exit(main())
```

---

## Migration Strategy

### Phase 1: Preparation (Low Risk)
1. ✅ Create comprehensive tests for current module
2. ✅ Document all current public APIs
3. ✅ Tag current version in git

### Phase 2: Create Package Structure (Medium Risk)
1. Create `simulations/` directory
2. Move code into new modules
3. Update imports
4. Keep `simulations.py` as compatibility shim:

```python
# fits_tools/simulations.py (legacy compatibility)
"""
DEPRECATED: Please use `fits_tools.simulations` package instead.

This module is maintained for backward compatibility and will be removed
in version 1.0.0.
"""

import warnings

warnings.warn(
    "Importing from fits_tools.simulations module is deprecated. "
    "Please use: from fits_tools.simulations import ... instead",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the package for backward compatibility
from fits_tools.simulations import *
```

### Phase 3: Testing (Critical)
1. Run all existing tests
2. Add integration tests
3. Test backward compatibility
4. Update documentation

### Phase 4: Gradual Deprecation
1. Release v0.4.0 with package structure + compatibility shim
2. Release v0.5.0 with deprecation warnings
3. Release v1.0.0 removing old module

---

## Backward Compatibility Strategy

### Option 1: Full Compatibility (Recommended)
Keep both for several versions:
```python
# Still works (with deprecation warning)
from fits_tools.simulations import create_bias_frame

# New preferred way
from fits_tools.simulations import create_bias_frame
# (imports from package, not module)
```

### Option 2: Immediate Break
Remove old module, force users to update. **Not recommended** unless:
- This is internal code only
- You control all users
- You're pre-1.0 and signal breaking changes

---

## Alternative: Partial Refactoring

If full package migration is too much work, consider a **hybrid approach**:

```
fits_tools/
└── simulations/
    ├── __init__.py          # Main public API (imports everything)
    ├── core.py              # Keep most code here (1500 lines)
    ├── artifacts.py         # Extract just artifacts (300 lines)
    ├── mef.py               # Extract MEF support (200 lines)
    └── cli.py               # Extract CLI (200 lines)
```

This gives you ~60% of the benefits with ~30% of the work.

---

## Recommendation

**YES, refactor into a package** because:

1. ✅ **Size**: 1,977 lines is definitely too large for a single module
2. ✅ **Complexity**: Multiple distinct responsibilities
3. ✅ **Maintainability**: Will improve significantly
4. ✅ **Testing**: Will become easier
5. ✅ **Extensibility**: Room to grow without clutter
6. ✅ **Team collaboration**: Better for multiple developers

**Suggested approach:**
1. Start with the **6-module structure** (noise, artifacts, frames, mef, timeseries, utils)
2. Use **Phase 1-4 migration strategy**
3. Maintain **backward compatibility** for 2-3 releases
4. Include **comprehensive tests** before and after

**Timeline estimate:**
- Planning: 1-2 hours
- Implementation: 4-8 hours
- Testing: 2-4 hours
- Documentation: 2-3 hours
- **Total: 1-2 days of focused work**

The investment will pay off in:
- Easier maintenance
- Better code organization
- Improved testability
- More pleasant development experience

Would you like me to create a detailed migration plan or help implement the refactoring?
