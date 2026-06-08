# Should simulations.py become a subpackage like image_processing?

## TL;DR: **YES** - Create `simulations/` as a subpackage at the same level as `image_processing/`

---

## Current fits_tools Architecture Analysis

```
src/fits_tools/
├── Core modules (single files):
│   ├── constants.py          # FITS keywords and constants
│   ├── data.py                # Data handling
│   ├── exceptions.py          # Custom exceptions
│   ├── hdu_ops.py             # HDU operations
│   ├── io.py                  # File I/O
│   ├── keywords.py            # Keyword handling
│   ├── noise.py               # Noise analysis
│   ├── quality_control.py     # QC functions
│   ├── sections.py            # Image sections
│   ├── selection.py           # File selection
│   ├── statistics.py          # Statistical operations
│   ├── survey.py              # Survey tools
│   ├── utils.py               # Utilities
│   └── validate.py            # Validation
│
├── Subpackages (directories):
│   ├── image_processing/      # Image processing (12 modules)
│   │   ├── batch.py
│   │   ├── calibration.py
│   │   ├── combining.py
│   │   ├── cosmic_rays.py
│   │   ├── filters.py
│   │   ├── operations.py
│   │   ├── pixel_edit.py
│   │   ├── streaming.py
│   │   ├── transformations.py
│   │   └── ...
│   │
│   └── simulations.py         # ← PROBLEM: Should be simulations/
```

---

## Why simulations.py Should Be a Subpackage

### 1. **Parallel to image_processing/**

**Comparison:**

| Aspect | image_processing/ | simulations.py |
|--------|------------------|----------------|
| **Size** | ~12 modules | 1,977 lines (should be ~6-7 modules) |
| **Scope** | Processing real FITS data | Generating synthetic FITS data |
| **Complexity** | High (multiple algorithms) | High (multiple artifact types) |
| **Relationship** | Complementary - processing vs. generation |
| **Structure** | Properly organized subpackage | Oversized single module |

**Observation:** `simulations.py` is **conceptually equivalent** to `image_processing/` - one processes data, one generates it. They should have parallel structures.

### 2. **Natural Organizational Symmetry**

```
fits_tools/
├── Core (single-purpose modules)
│   ├── io.py              # Read/write FITS
│   ├── keywords.py        # Handle keywords
│   ├── sections.py        # Manage sections
│   └── ...
│
└── Domains (multi-module subpackages)
    ├── image_processing/  # Process real FITS data
    │   ├── calibration.py
    │   ├── cosmic_rays.py
    │   ├── combining.py
    │   └── ...
    │
    └── simulations/       # Generate synthetic FITS data
        ├── frames.py
        ├── artifacts.py
        ├── noise.py
        └── ...
```

This creates a **clean, logical hierarchy**:
- **Core modules** = single-responsibility, reusable tools
- **Domain subpackages** = complex, multi-faceted functionality

### 3. **Usage Patterns Match image_processing/**

**Current image_processing usage:**
```python
# Users import from subpackage
from fits_tools.image_processing import calibrate_frame
from fits_tools.image_processing import remove_cosmic_rays
from fits_tools.image_processing.combining import median_combine
```

**Proposed simulations usage:**
```python
# Would mirror the same pattern
from fits_tools.simulations import create_bias_frame
from fits_tools.simulations import add_cosmic_rays
from fits_tools.simulations.artifacts import add_cte_trails
```

**Consistency principle:** Users already understand the subpackage pattern from `image_processing/`. Using it for `simulations/` provides a **familiar interface**.

### 4. **Both Are "Heavy" Features**

Looking at the module list, there are two "heavy" features:
1. **image_processing/** - processes data (12 modules)
2. **simulations.py** - generates data (1,977 lines → should be ~6-7 modules)

Everything else is lightweight:
- `constants.py` - just definitions
- `io.py` - basic read/write
- `keywords.py` - keyword utilities
- `sections.py` - section parsing
- etc.

**Principle:** Heavy features deserve subpackage structure.

### 5. **Future Growth Considerations**

**image_processing/** has grown to 12 modules because it handles many operations:
- Calibration
- Cosmic ray removal
- Combining
- Filtering
- Transformations
- Batch processing
- Streaming
- etc.

**simulations.py** will likely grow similarly:
- Different detector types (CCD, CMOS, IR)
- Different frame types (bias, dark, flat, science)
- Different artifact types (cosmic rays, hot pixels, CTE, etc.)
- Different noise models
- MEF support
- Time series
- **Future additions:**
  - Spectroscopic simulations
  - Polarimetric data
  - Different observatories
  - Detector quirks (e.g., HST, JWST, Gaia)

A subpackage structure **accommodates growth** without cluttering the main namespace.

---

## Recommended Structure

```
src/fits_tools/
├── __init__.py
├── constants.py
├── exceptions.py
├── [... other core modules ...]
│
├── image_processing/          # Process real data
│   ├── __init__.py
│   ├── calibration.py
│   ├── combining.py
│   ├── cosmic_rays.py
│   ├── [... 9 more modules ...]
│   └── cli.py
│
└── simulations/               # Generate synthetic data
    ├── __init__.py            # Public API
    ├── _base.py               # Core utilities (private)
    ├── _constants.py          # Simulation constants (private)
    ├── noise.py               # Noise models
    ├── artifacts.py           # Detector artifacts
    ├── frames.py              # Frame creation
    ├── mef.py                 # Multi-Extension FITS
    ├── timeseries.py          # Time series
    ├── utils.py               # Helper functions
    └── cli.py                 # Command-line interface
```

---

## Benefits of This Structure

### 1. **Consistency with Existing Architecture**
- Mirrors `image_processing/` structure
- Users familiar with one can understand the other
- Clear pattern: core modules vs. domain subpackages

### 2. **Clean Namespace**
```python
# Top-level: core utilities
from fits_tools import read_fits, write_fits
from fits_tools.keywords import update_keyword

# Subpackages: domain-specific
from fits_tools.image_processing import calibrate_frame
from fits_tools.simulations import create_bias_frame
```

### 3. **Better Discoverability**
```python
# Easy to explore
import fits_tools
dir(fits_tools.simulations)  # See all simulation functions
dir(fits_tools.image_processing)  # See all processing functions
```

### 4. **Parallel Development**
- `image_processing/` team can work independently
- `simulations/` team can work independently
- Minimal merge conflicts

### 5. **Logical Testing Structure**
```
tests/
├── test_io.py                      # Test core I/O
├── test_keywords.py                # Test keywords
├── image_processing/
│   ├── test_calibration.py
│   ├── test_combining.py
│   └── ...
└── simulations/
    ├── test_noise.py
    ├── test_artifacts.py
    ├── test_frames.py
    └── ...
```

---

## Comparison: Module vs. Subpackage

### Option A: Keep as Module (Current)
```
fits_tools/
├── image_processing/    # Subpackage (good)
└── simulations.py       # Module (inconsistent, too large)
```

**Problems:**
- ❌ Inconsistent with `image_processing/`
- ❌ 1,977 lines is too large for a single file
- ❌ Harder to navigate and maintain
- ❌ Asymmetric structure (one subpackage, one module)

### Option B: Convert to Subpackage (Recommended)
```
fits_tools/
├── image_processing/    # Subpackage
└── simulations/         # Subpackage (consistent)
```

**Benefits:**
- ✅ Consistent structure
- ✅ Appropriate size (6-7 modules of ~200-600 lines each)
- ✅ Easy to navigate
- ✅ Symmetric, logical organization

---

## Migration Impact

### Breaking Changes: **NONE** (with compatibility layer)

**Before (works today):**
```python
from fits_tools.simulations import create_bias_frame
from fits_tools.simulations import add_cosmic_rays
```

**After (same imports work):**
```python
# Still works - imports from simulations/__init__.py
from fits_tools.simulations import create_bias_frame
from fits_tools.simulations import add_cosmic_rays

# New capability - can import from submodules
from fits_tools.simulations.frames import create_bias_frame
from fits_tools.simulations.artifacts import add_cosmic_rays
```

**Implementation:**
```python
# fits_tools/simulations/__init__.py
"""Synthetic FITS data generation."""

# Re-export everything for backward compatibility
from .noise import add_read_noise, add_photon_noise
from .artifacts import add_cosmic_rays, add_hot_pixels, add_cte_trails, ...
from .frames import create_bias_frame, create_dark_frame, ...
from .mef import create_mef_bias_frame
from .timeseries import create_bias_time_series, ...

__all__ = [
    'add_read_noise',
    'add_photon_noise',
    'add_cosmic_rays',
    # ... etc.
]
```

**Result:** Zero breaking changes for users!

---

## Relation to Other fits_tools Modules

### Modules simulations/ Would Use:
- ✅ `constants.py` - FITS keywords
- ✅ `exceptions.py` - Error types
- ✅ `validate.py` - Input validation
- ✅ `utils.py` - General utilities

### Modules That Would Use simulations/:
- ✅ `image_processing/` - Could use synthetic data for testing
- ✅ Test suite - Generate test FITS files
- ✅ Documentation - Create example data

### Independence:
`simulations/` is **largely independent** - it generates data but doesn't depend on processing code. This makes it a **good candidate for a separate subpackage**.

---

## Recommended Architecture Principles

Based on the current structure, I'd suggest these principles:

### Core Modules (Single File)
**Criteria:**
- ✅ Single, focused responsibility
- ✅ < 500 lines of code
- ✅ Used by multiple other modules
- ✅ Provides utility/infrastructure

**Examples:**
- `constants.py` - Definitions
- `io.py` - File operations
- `keywords.py` - Keyword handling
- `sections.py` - Section parsing
- `utils.py` - Shared utilities

### Subpackages (Directory)
**Criteria:**
- ✅ Multiple related functions/classes
- ✅ > 1,000 lines of code total
- ✅ Complex domain with multiple aspects
- ✅ Benefits from internal organization

**Examples:**
- `image_processing/` - Process FITS data
- `simulations/` - Generate FITS data

---

## Decision Matrix

| Factor | Keep as Module | Convert to Subpackage |
|--------|---------------|---------------------|
| **Consistency with image_processing/** | ❌ Inconsistent | ✅ Parallel structure |
| **File size management** | ❌ 1,977 lines | ✅ 6-7 focused modules |
| **Namespace cleanliness** | ❌ All in one file | ✅ Organized by domain |
| **Discoverability** | ❌ Hard to browse | ✅ Easy to explore |
| **Maintainability** | ❌ Difficult | ✅ Much easier |
| **Testing** | ❌ One large test file | ✅ Modular tests |
| **Future growth** | ❌ Gets worse | ✅ Accommodates growth |
| **Breaking changes** | ✅ None needed | ✅ None needed |
| **Implementation effort** | ✅ No work | ⚠️ 1-2 days work |

**Score: 1 vs. 7 in favor of subpackage**

---

## Final Recommendation

**Convert `simulations.py` to `simulations/` subpackage** for these reasons:

1. **Consistency** - Matches `image_processing/` structure
2. **Size** - 1,977 lines needs to be split
3. **Symmetry** - Processing and generation should have parallel structures
4. **Maintainability** - Easier to work with 6-7 focused modules
5. **Scalability** - Ready for future additions
6. **Zero breaking changes** - Users won't be affected

**Timeline:**
- Planning: 2 hours
- Implementation: 1 day
- Testing: 0.5 days
- **Total: 1.5-2 days**

**ROI:** High - One-time investment for long-term maintainability improvement

---

## Proposed File Structure

```
src/fits_tools/simulations/
├── __init__.py              # Public API (50 lines)
│   """
│   from .noise import add_read_noise, add_photon_noise
│   from .artifacts import add_cosmic_rays, add_hot_pixels, ...
│   from .frames import create_bias_frame, create_dark_frame, ...
│   from .mef import create_mef_bias_frame
│   from .timeseries import create_bias_time_series, ...
│   
│   __all__ = [...]
│   """
│
├── _base.py                 # Core FITS creation (65 lines)
├── _constants.py            # Magic numbers (50 lines)
├── noise.py                 # Noise models (200 lines)
├── artifacts.py             # Detector artifacts (300 lines)
├── frames.py                # Frame creation (600 lines)
├── mef.py                   # Multi-Extension FITS (200 lines)
├── timeseries.py            # Time series (200 lines)
├── utils.py                 # Helpers (100 lines)
└── cli.py                   # CLI (250 lines)
```

**Total:** ~2,015 lines spread across 10 well-organized files
**vs. Current:** 1,977 lines in one file

The slight increase in total lines is due to:
- Proper imports in each module
- `__init__.py` re-exports
- Better documentation
- Eliminated duplication

**Worth it?** Absolutely!

---

## Next Steps

If you decide to proceed:

1. **Create structure:**
   ```bash
   mkdir src/fits_tools/simulations
   ```

2. **Split code** according to proposed modules

3. **Create `__init__.py`** with re-exports

4. **Update imports** in other modules (if any)

5. **Run tests** to verify nothing broke

6. **Update documentation**

7. **Optional:** Keep `simulations.py` as a shim with deprecation warning for one release cycle

Would you like me to create a detailed migration script or help implement the refactoring?
