#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bad pixel mask bitmask definitions and utilities for sensor metrology.

This module defines the bit flags used in bad pixel masks (BPM) throughout
the fits_tools package. These masks use a 32-bit integer architecture to 
encode hardware defects, structural anomalies, and transient data quality 
issues into a single array.

Bitmask Structure:
    Each pixel in a BPM array is a uint32 where individual bits indicate 
    specific defect types. The mask is organized into logical zones:
    - Bits 0-15:  Single-pixel hardware defects (Static)
    - Bits 16-26: Structural defects (Column/Row level)
    - Bits 27-31: Transient artifacts (Per-image Data Quality)

Bit Assignments:
    --- Pixel-Level (Bits 0-15) ---
    Bit 0 (1):      PIX_BAD       - General unusable pixel (Summary flag)
    Bit 1 (2):      PIX_DEAD      - Unresponsive pixel (No signal)
    Bit 2 (4):      PIX_BRIGHT    - High bias offset
    Bit 3 (8):      PIX_HOT       - Excessive dark current
    Bit 4 (16):     PIX_QE_HIGH   - High Quantum Efficiency response
    Bit 5 (32):     PIX_QE_LOW    - Low Quantum Efficiency response
    Bit 6 (64):     PIX_NONLIN    - Non-linear response behavior
    Bit 7 (128):    PIX_NOISE     - Excessive read noise
    Bit 8 (256):    PIX_RTS       - Random Telegraph Signal (Burst noise)
    Bit 9 (512):    PIX_TRAP      - Individual charge trap site
    Bit 10 (1024):  PIX_GLOW      - Amplifier/circuitry glow
    Bit 15 (32768): PIX_EXCLUDED  - Manually excluded from analysis

    --- Structural (Bits 16-26) ---
    Bit 16:         COL_BAD       - General bad column
    Bit 17:         COL_BRIGHT    - Excessively bright column
    Bit 18:         COL_DARK      - Dark or blocked column
    Bit 19:         COL_TRAP      - Column-level charge trap
    Bit 20:         COL_BLOCKED   - Vertical transfer obstructed
    Bit 21:         COL_CTE_LOW   - Poor Charge Transfer Efficiency
    Bit 22:         COL_BOUNDARY  - Amplifier/Segment boundary
    Bit 23:         COL_EXCLUDED  - Column excluded from measurements
    Bit 24:         ROW_BAD       - General bad row
    Bit 25:         ROW_RESERVED_25 - reserved for future use
    Bit 26:         ROW_EXCLUDED  - Row excluded from measurements

    --- Transient/DQ (Bits 27-31) ---
    Bit 27:         SATURATED     - Full-well capacity reached
    Bit 28:         PIX_COSMIC    - Cosmic ray hit
    Bit 29:         PIX_XTALK     - Electronic crosstalk artifact
    Bit 30:         PIX_BLOOM     - Saturated charge leakage
    Bit 31:         PIX_OUTLIER   - Pipeline-specific statistical outlier

Thread Safety
-------------
All functions in this module are thread-safe. Functions that modify
arrays (e.g., flag_bad_rows_cols) return copies rather than mutating
inputs, making them safe for concurrent processing of multiple sensors.

Notes
-----
When processing large batches in parallel:
    >>> from concurrent.futures import ProcessPoolExecutor
    >>> with ProcessPoolExecutor() as executor:
    ...     results = executor.map(process_sensor_bpm, sensor_list)

Usage Examples:
    Creating and marking a mask:
        >>> import numpy as np
        >>> from fits_tools.bitmask import BPM_BIT_PIX_HOT, BPM_BIT_PIX_NONLIN
        >>> bpm = np.zeros((2048, 2048), dtype=np.uint32)
        >>> # Mark a pixel as both hot and non-linear
        >>> bpm[100, 200] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN
    
    Checking for specific defect categories using group masks:
        >>> from fits_tools.bitmask  import BPM_MASK_COLD
        >>> if bpm[10, 10] & BPM_MASK_COLD:
        ...     print("Pixel is unresponsive or blocked")
    
    Using the Type-Safe Enum:
        >>> from fits_tools.bitmask import DefectType
        >>> pixel_status = DefectType(bpm[100, 200])
        >>> print(pixel_status.name)
        'PIX_HOT|PIX_NONLIN'

Performance Notes
-----------------
For arrays < 64 Mpixels, statistics functions use vectorized batch
processing (2 array passes). Larger arrays fall back to sequential
processing to limit memory usage.

Common Patterns
---------------
**Pattern 1: Master BPM + Per-Image DQ**
    >>> master_bpm = load_calibration_bpm()  # Permanent defects
    >>> image_dq = detect_cosmic_rays(data)  # Transient artifacts
    >>> final_mask = merge_bpms(master_bpm, image_dq, 'or')

**Pattern 2: Selective Masking**
    >>> # Use hot pixels as calibration sources, but mask dead pixels
    >>> calibration_mask = BPM_MASK_COLD | BPM_BIT_PIX_NONLIN
    >>> clean_data = apply_bpm(data, bpm, defect_mask=calibration_mask)

**Pattern 3: Quality Gating**
    >>> stats = get_bpm_statistics(bpm)
    >>> if stats['MASK_PERMANENT']['fraction'] > 0.05:
    ...     raise SensorQualityError("Defect rate exceeds 5% threshold")

Version History
---------------
2.0.0 (Current)
    - Added structural defects (columns/rows)
    - Added transient DQ flags
    - Performance improvements in statistics
1.0.0
    - Initial pixel-level defect tracking

Version: 2.0.0
"""

from typing import Dict, Iterator, List, Optional, Tuple, TypedDict, Union, Any
import numpy as np
from enum import IntFlag
import logging
import numpy.typing as npt
import textwrap
from pathlib import Path
from typing import Union, TextIO

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        """Dummy decorator when numba unavailable."""
        def wrapper(func):
            return func
        return wrapper
    

__version__ = "2.0.0"

logger = logging.getLogger(__name__)

_BPM_STATS_PIXEL_LIMIT: int = 64 * 1024 * 1024  # 64 Mpix threshold for batch processing


_DEFINITIONS_SCHEMA_VERSION: int = 1

# FITS standard: a COMMENT card's content field is 72 characters wide.
# "COMMENT " occupies the first 8 bytes of the 80-byte card record.
_FITS_COMMENT_WIDTH: int = 72

# ---------------------------------------------------------------------------
# TypedDict for the per-entry return structure.
# Gives callers type-checker support: stats['PIX_DEAD']['count'] is known int.
# Defined at module level so it can be referenced in external type annotations.
# ---------------------------------------------------------------------------
class DefectStatEntry(TypedDict):
    count:    int
    fraction: float
    type:     str   # 'Individual Bit' | 'Group Mask'

# =============================================================================
# Bitmask Constants
# =============================================================================

# --- PIXEL-LEVEL (Static/Hardware) Bits 0-15 ---
BPM_BIT_PIX_BAD           = 1 << 0   # General bad pixel
BPM_BIT_PIX_DEAD          = 1 << 1   # Unresponsive
BPM_BIT_PIX_BRIGHT        = 1 << 2   # High bias offset
BPM_BIT_PIX_HOT           = 1 << 3   # Excess dark current
BPM_BIT_PIX_QE_HIGH       = 1 << 4   # High QE
BPM_BIT_PIX_QE_LOW        = 1 << 5   # Low QE
BPM_BIT_PIX_NONLIN        = 1 << 6   # Non-linear response
BPM_BIT_PIX_NOISE         = 1 << 7   # High read noise
BPM_BIT_PIX_RTS           = 1 << 8   # Random Telegraph Signal
BPM_BIT_PIX_TRAP          = 1 << 9   # Pixel-level charge trap
BPM_BIT_PIX_GLOW          = 1 << 10  # Pixel subject to amp glow
BPM_BIT_PIX_RESERVED_11   = 1 << 11  # Reserved pixel mask bit
BPM_BIT_PIX_RESERVED_12   = 1 << 12  # Reserved pixel mask bit
BPM_BIT_PIX_RESERVED_13   = 1 << 13  # Reserved pixel mask bit
BPM_BIT_PIX_RESERVED_14   = 1 << 14  # Reserved pixel mask bit
BPM_BIT_PIX_EXCLUDED      = 1 << 15  # Manually excluded from analysis

# --- STRUCTURAL (Column/Row) Bits 16-26 ---
BPM_BIT_COL_BAD           = 1 << 16  # General bad column
BPM_BIT_COL_BRIGHT        = 1 << 17  # Bright column
BPM_BIT_COL_DARK          = 1 << 18  # Dark/blocked column
BPM_BIT_COL_TRAP          = 1 << 19  # Column-level trap
BPM_BIT_COL_BLOCKED       = 1 << 20  # Transfer blocked by defect
BPM_BIT_COL_CTE_LOW       = 1 << 21  # Poor Charge Transfer Efficiency
BPM_BIT_COL_BOUNDARY      = 1 << 22  # Segment/Amp boundary column
BPM_BIT_COL_EXCLUDED      = 1 << 23  # Column excluded from measurements

BPM_BIT_ROW_BAD           = 1 << 24  # General bad row
BPM_BIT_ROW_RESERVED_25   = 1 << 25  # Reserved
BPM_BIT_ROW_EXCLUDED      = 1 << 26  # Row excluded from measurements

# --- TRANSIENT (Per-Image DQ) Bits 27-31 ---
BPM_BIT_SATURATED     = 1 << 27  # Pixel at full-well
BPM_BIT_PIX_COSMIC        = 1 << 28  # Cosmic ray hit
BPM_BIT_PIX_XTALK         = 1 << 29  # Electronic crosstalk artifact
BPM_BIT_PIX_BLOOM         = 1 << 30  # Saturated charge leakage
BPM_BIT_PIX_OUTLIER       = 1 << 31  # Statistical outlier

# --- 0. The Physical Category Masks ---
BPM_MASK_PIX = (
    BPM_BIT_PIX_BAD | BPM_BIT_PIX_DEAD | BPM_BIT_PIX_BRIGHT | BPM_BIT_PIX_HOT | BPM_BIT_PIX_QE_HIGH | 
    BPM_BIT_PIX_QE_LOW | BPM_BIT_PIX_NONLIN | BPM_BIT_PIX_NOISE | BPM_BIT_PIX_RTS |    
    BPM_BIT_PIX_TRAP | BPM_BIT_PIX_GLOW  | BPM_BIT_PIX_EXCLUDED 
)

BPM_MASK_COL = (
    BPM_BIT_COL_BAD | BPM_BIT_COL_BRIGHT | BPM_BIT_COL_DARK |
    BPM_BIT_COL_TRAP | BPM_BIT_COL_BLOCKED | BPM_BIT_COL_CTE_LOW |
    BPM_BIT_COL_BOUNDARY  | BPM_BIT_COL_EXCLUDED 
)

BPM_MASK_ROW = (
    BPM_BIT_ROW_BAD | BPM_BIT_ROW_EXCLUDED
)

# --- 1. The High-Level Category Masks ---

# Combined mask for permanent (not transient) defects# Everything that is physically "wrong" with the silicon (Static)
# EXCLUDED bits are not part of either PERMANENT or TRANSIENT and live in BPM_MASK_IGNORE.
BPM_MASK_PERMANENT = (
    BPM_BIT_PIX_BAD | BPM_BIT_PIX_DEAD | BPM_BIT_PIX_BRIGHT | 
    BPM_BIT_PIX_HOT | BPM_BIT_PIX_QE_HIGH | BPM_BIT_PIX_QE_LOW |
    BPM_BIT_PIX_NONLIN | BPM_BIT_PIX_NOISE | BPM_BIT_PIX_RTS | 
    BPM_BIT_PIX_TRAP | BPM_BIT_PIX_GLOW |
    BPM_BIT_COL_BAD | BPM_BIT_COL_BRIGHT | BPM_BIT_COL_DARK | 
    BPM_BIT_COL_TRAP | BPM_BIT_COL_BLOCKED | BPM_BIT_COL_CTE_LOW |
    BPM_BIT_ROW_BAD
)

# Combined mask for Transient/conditional defects that result from the exposure/test conditions (Dynamic)
# EXCLUDED bits are not part of either PERMANENT or TRANSIENT and live in BPM_MASK_IGNORE.
BPM_MASK_TRANSIENT = (
    BPM_BIT_SATURATED | BPM_BIT_PIX_COSMIC | 
    BPM_BIT_PIX_XTALK | BPM_BIT_PIX_BLOOM | BPM_BIT_PIX_OUTLIER
)

# --- 2. Functional/Diagnostic Masks ---

# Combined mask for all "High Signal" anomalies (potential sources of false positives)
BPM_MASK_HOT = (
    BPM_BIT_PIX_BRIGHT | BPM_BIT_PIX_HOT | 
    BPM_BIT_PIX_QE_HIGH | BPM_BIT_COL_BRIGHT
)

# Combined mask for all cold/dead pixels
BPM_MASK_COLD = (
    BPM_BIT_PIX_DEAD | BPM_BIT_PIX_QE_LOW | 
    BPM_BIT_COL_DARK | BPM_BIT_COL_BLOCKED
)

# Combined mask for all temporal instability (noise that changes frame-to-frame)
BPM_MASK_NOISE = (
    BPM_BIT_PIX_NOISE | BPM_BIT_PIX_RTS | BPM_BIT_PIX_XTALK
)

# Combined mask for structural transport issues (charge transfer physics)
BPM_MASK_TRANSPORT = (
    BPM_BIT_PIX_TRAP | BPM_BIT_COL_TRAP | 
    BPM_BIT_COL_BLOCKED | BPM_BIT_COL_CTE_LOW
)

# Combined mask for all pixels, columns, and rows that are marked 'BAD'
BPM_MASK_BAD = BPM_BIT_PIX_BAD | BPM_BIT_ROW_BAD | BPM_BIT_COL_BAD   

# Combined mask for Charge trap defects
BPM_MASK_TRAPS = BPM_BIT_PIX_TRAP | BPM_BIT_COL_TRAP   

# Combined mask for Response anomalies
BPM_MASK_RESPONSE = BPM_BIT_PIX_DEAD | BPM_BIT_SATURATED | BPM_BIT_PIX_BLOOM 

# Mask for pixels that should be ignored entirely in statistical averages
BPM_MASK_IGNORE = (
    BPM_BIT_PIX_EXCLUDED | BPM_BIT_COL_EXCLUDED | BPM_BIT_ROW_EXCLUDED
)

BPM_MASK_ALL_16 = 0xFFFF
BPM_MASK_ALL_32 = 0xFFFFFFFF
# Default to 32-bit "all bits" mask because this module defines bits >= 16.
BPM_MASK_ALL = BPM_MASK_ALL_32


_MASK_ANY_DEFECT: int = BPM_MASK_PERMANENT | BPM_MASK_TRANSIENT | BPM_MASK_IGNORE 


# ---------------------------------------------------------------------------
# Module-level constant: pulled out of the function body so it is built once,
# not reconstructed on every call. Also makes it easy to extend or override.
# ---------------------------------------------------------------------------
_GROUP_MASKS: Dict[str, int] = {
    'MASK_PIX':       BPM_MASK_PIX,
    'MASK_COL':       BPM_MASK_COL,
    'MASK_ROW':       BPM_MASK_ROW,
    'MASK_PERMANENT': BPM_MASK_PERMANENT,
    'MASK_TRANSIENT': BPM_MASK_TRANSIENT,
    'MASK_HOT':       BPM_MASK_HOT,
    'MASK_COLD':      BPM_MASK_COLD,
    'MASK_NOISE':     BPM_MASK_NOISE,
    'MASK_TRANSPORT': BPM_MASK_TRANSPORT,
    'MASK_BAD':       BPM_MASK_BAD,
    'MASK_TRAPS':     BPM_MASK_TRAPS,
    'MASK_RESPONSE':  BPM_MASK_RESPONSE,
    'MASK_IGNORE':    BPM_MASK_IGNORE,
}


# =============================================================================
# Enum for Type-Safe Access
# =============================================================================

class DefectType(IntFlag):
    """
    Type-safe enumeration of sensor and data quality defects.
    
    Categorized by:
    - PIX: Single pixel hardware defects (0-15)
    - COL: Column structural defects (16-23)
    - ROW: Row structural defects (24-26)
    - ARTI: Transient/Dynamic artifacts (27-31)
    """

    # --- Single-pixel defects ---
    PIX_BAD       = BPM_BIT_PIX_BAD
    PIX_DEAD      = BPM_BIT_PIX_DEAD
    PIX_BRIGHT    = BPM_BIT_PIX_BRIGHT
    PIX_HOT       = BPM_BIT_PIX_HOT
    PIX_QE_HIGH   = BPM_BIT_PIX_QE_HIGH
    PIX_QE_LOW    = BPM_BIT_PIX_QE_LOW
    PIX_NONLIN    = BPM_BIT_PIX_NONLIN
    PIX_NOISE     = BPM_BIT_PIX_NOISE
    PIX_RTS       = BPM_BIT_PIX_RTS
    PIX_TRAP      = BPM_BIT_PIX_TRAP
    PIX_GLOW      = BPM_BIT_PIX_GLOW
    PIX_EXCLUDED  = BPM_BIT_PIX_EXCLUDED

    # --- Column level defects ---
    COL_BAD       = BPM_BIT_COL_BAD
    COL_BRIGHT    = BPM_BIT_COL_BRIGHT
    COL_DARK      = BPM_BIT_COL_DARK
    COL_TRAP      = BPM_BIT_COL_TRAP
    COL_BLOCKED   = BPM_BIT_COL_BLOCKED
    COL_CTE_LOW   = BPM_BIT_COL_CTE_LOW
    COL_BOUNDARY  = BPM_BIT_COL_BOUNDARY
    COL_EXCLUDED  = BPM_BIT_COL_EXCLUDED

    # --- Row level defects ---
    ROW_BAD       = BPM_BIT_ROW_BAD
    ROW_EXCLUDED  = BPM_BIT_ROW_EXCLUDED

    # --- Dynamic / Transient ---
    SATURATED     = BPM_BIT_SATURATED
    PIX_COSMIC    = BPM_BIT_PIX_COSMIC
    PIX_XTALK     = BPM_BIT_PIX_XTALK
    PIX_BLOOM     = BPM_BIT_PIX_BLOOM
    PIX_OUTLIER   = BPM_BIT_PIX_OUTLIER

    # --- Pre-defined Group Masks ---
    # decomposition may produce unexpected results when composite mask members overlap 
    # with single-bit members during flag decomposition. Keeping the composites as standalone 
    # module-level constants (as they already are) is cleaner. 
    # The enum should contain only single-bit atomic flags.
    # MASK_PIX       = BPM_MASK_PIX
    # MASK_COL       = BPM_MASK_COL
    # MASK_ROW       = BPM_MASK_ROW
    # MASK_PERMANENT = BPM_MASK_PERMANENT
    # MASK_TRANSIENT = BPM_MASK_TRANSIENT
    # MASK_HOT       = BPM_MASK_HOT
    # MASK_COLD      = BPM_MASK_COLD


# -----------------------------------------------------------------------------
# Defect Type Names and Descriptions
# -----------------------------------------------------------------------------

FLAG_NAMES = {
    BPM_BIT_PIX_BAD:      'PIX_BAD',
    BPM_BIT_PIX_DEAD:     'PIX_DEAD',
    BPM_BIT_PIX_BRIGHT:   'PIX_BRIGHT',
    BPM_BIT_PIX_HOT:      'PIX_HOT',
    BPM_BIT_PIX_QE_HIGH:  'PIX_QE_HIGH',
    BPM_BIT_PIX_QE_LOW:   'PIX_QE_LOW',
    BPM_BIT_PIX_NONLIN:   'PIX_NONLIN',
    BPM_BIT_PIX_NOISE:    'PIX_NOISE',
    BPM_BIT_PIX_RTS:      'PIX_RTS',
    BPM_BIT_PIX_TRAP:     'PIX_TRAP',
    BPM_BIT_PIX_GLOW:     'PIX_GLOW',
    BPM_BIT_PIX_EXCLUDED: 'PIX_EXCLUDED',
    
    BPM_BIT_COL_BAD:      'COL_BAD',
    BPM_BIT_COL_BRIGHT:   'COL_BRIGHT',
    BPM_BIT_COL_DARK:     'COL_DARK',
    BPM_BIT_COL_TRAP:     'COL_TRAP',
    BPM_BIT_COL_BLOCKED:  'COL_BLOCKED',
    BPM_BIT_COL_CTE_LOW:  'COL_CTE_LOW',
    BPM_BIT_COL_BOUNDARY: 'COL_BOUNDARY',
    BPM_BIT_COL_EXCLUDED: 'COL_EXCLUDED',

    BPM_BIT_ROW_BAD:      'ROW_BAD',
    BPM_BIT_ROW_RESERVED_25: 'ROW_RESERVED_25',
    BPM_BIT_ROW_EXCLUDED: 'ROW_EXCLUDED',

    BPM_BIT_SATURATED:    'SATURATED',
    BPM_BIT_PIX_COSMIC:   'PIX_COSMIC',
    BPM_BIT_PIX_XTALK:    'PIX_XTALK',
    BPM_BIT_PIX_BLOOM:    'PIX_BLOOM',
    BPM_BIT_PIX_OUTLIER:  'PIX_OUTLIER',
}

# Module level, after FLAG_NAMES is defined
_SORTED_FLAG_ITEMS: List[Tuple[int, str]] = sorted(FLAG_NAMES.items())

FLAG_DESCRIPTIONS = {
    BPM_BIT_PIX_BAD:      'General unusable pixel (summary flag)',
    BPM_BIT_PIX_DEAD:     'Unresponsive pixel (zero or near-zero signal)',
    BPM_BIT_PIX_BRIGHT:   'Anomalously high bias offset',
    BPM_BIT_PIX_HOT:      'Excessive dark current',
    BPM_BIT_PIX_QE_HIGH:  'Quantum Efficiency significantly above average',
    BPM_BIT_PIX_QE_LOW:   'Quantum Efficiency significantly below average',
    BPM_BIT_PIX_NONLIN:   'Pixel response deviates from linear model',
    BPM_BIT_PIX_NOISE:    'Pixel read noise exceeds specification',
    BPM_BIT_PIX_RTS:      'Random Telegraph Signal (burst/popcorn noise)',
    BPM_BIT_PIX_TRAP:     'Individual pixel charge trap site',
    BPM_BIT_PIX_GLOW:     'Pixel affected by amplifier or circuitry luminescence',
    BPM_BIT_PIX_EXCLUDED: 'Pixel manually excluded from hardware metrics',

    BPM_BIT_COL_BAD:      'General unusable column (summary flag)',
    BPM_BIT_COL_BRIGHT:   'Column with excessive bright pixel population',
    BPM_BIT_COL_DARK:     'Column with excessive dark/dead pixel population',
    BPM_BIT_COL_TRAP:     'Column containing significant charge traps',
    BPM_BIT_COL_BLOCKED:  'Vertical charge transfer physically obstructed',
    BPM_BIT_COL_CTE_LOW:  'Sub-standard Charge Transfer Efficiency (high CTI)',
    BPM_BIT_COL_BOUNDARY: 'Column located on physical segment/amplifier boundary',
    BPM_BIT_COL_EXCLUDED: 'Column excluded from measurement statistics',

    BPM_BIT_ROW_BAD:      'Entire row flagged as unusable',
    BPM_BIT_ROW_RESERVED_25: 'Reserved Row defect ID',
    BPM_BIT_ROW_EXCLUDED: 'Row excluded from measurement statistics',

    BPM_BIT_SATURATED: 'Full-well capacity exceeded',
    BPM_BIT_PIX_COSMIC:   'High-energy particle hit (transient)',
    BPM_BIT_PIX_XTALK:    'Signal from neighboring channel/pixel (crosstalk)',
    BPM_BIT_PIX_BLOOM:    'Charge overflow from adjacent saturated pixels',
    BPM_BIT_PIX_OUTLIER:  'Statistical outlier detected by processing pipeline',
}



def bit_index(bit_value: int, max_bits: int = 32) -> int:
    """Return the zero-based bit index for a single-bit flag value.
    
    Parameters
    ----------
    bit_value : int
        A power-of-two integer representing a single bit flag.
    max_bits : int, default 32
        Maximum bit index allowed (for validation).
    
    Raises
    ------
    ValueError
        If bit_value is not a power of two, is negative/zero,
        or exceeds the max_bits limit.
    """
    bit_value = int(bit_value)
    if bit_value <= 0 or (bit_value & (bit_value - 1)) != 0:
        raise ValueError(f"bit_value must be a single-bit power of two; got {bit_value}")
    
    index = bit_value.bit_length() - 1
    if index >= max_bits:
        raise ValueError(f"bit_index {index} exceeds maximum of {max_bits-1}")
    
    return index


def iter_flag_definitions() -> Iterator[Tuple[int, str, str]]:
    """Iterate over known defect bits in ascending bit order.

    Yields
    ------
    (bit_value, name, description)
    """
    for bit_val in sorted(FLAG_NAMES.keys()):
        yield int(bit_val), FLAG_NAMES[bit_val], FLAG_DESCRIPTIONS.get(bit_val, '')


def get_definitions_payload(nbits: int = 32) -> Dict[str, Any]:
    """Return a canonical JSON-serializable payload of defect bit definitions."""
    bits = []
    for bit_val, name, desc in iter_flag_definitions():
        idx = bit_index(bit_val)
        if 0 <= idx < nbits:
            bits.append({"bit": idx, "value": int(bit_val), "name": str(name), "desc": str(desc)})
    bits.sort(key=lambda d: d["bit"])
    return {"schema": _DEFINITIONS_SCHEMA_VERSION, "nbits": int(nbits), "bits": bits}


def format_bit_comments(
    width: int = _FITS_COMMENT_WIDTH,
    prefix: str = "",
    include_hex: bool = True,
) -> List[str]:
    """
    Return FITS-header-friendly COMMENT strings documenting the defect bits.

    Each string in the returned list maps directly to one FITS COMMENT card
    (i.e., one call to ``header['COMMENT'] = line``).  Lines are wrapped at
    ``width`` characters so that no card ever exceeds the 72-character FITS
    standard limit.  Continuation lines are indented to align the description
    text under the first character of the description on the opening line,
    producing a clean columnar layout in the header::

        Bit  0 (0x00000001) PIX_BAD      General unusable pixel (summary flag)
        Bit  1 (0x00000002) PIX_DEAD     Unresponsive pixel (zero or near-zero
                                         signal)
        Bit  3 (0x00000008) PIX_HOT      Excessive dark current
        ...

    Parameters
    ----------
    width : int, optional
        Maximum character width for each output line.  Default is 72, which
        matches the FITS COMMENT card content field (FITS Standard 4.0,
        Sect. 4.4.2).  Reduce to leave room for a ``prefix``.
    prefix : str, optional
        String prepended to every line before the content.  Useful for
        namespacing in crowded headers, e.g. ``prefix="BPM: "``.  The prefix
        length is automatically subtracted from the available wrapping width,
        so the 72-character guarantee is preserved.
    include_hex : bool, optional
        If True (default), each bit entry includes the 8-digit hexadecimal
        value ``(0xXXXXXXXX)``.  Set to False for a more compact layout when
        hex representation is redundant (e.g. the bit index is sufficient).

    Returns
    -------
    list of str
        Ordered as: one header line, then one-or-more lines per bit in
        ascending bit-index order.  Every string is guaranteed to be at most
        ``width`` characters long.

    Raises
    ------
    ValueError
        If ``width`` is too narrow to fit even the fixed preamble for any
        bit entry (i.e. no room remains for description text after accounting
        for the preamble and prefix).

    Examples
    --------
    Basic usage with astropy:

    >>> from astropy.io.fits import Header
    >>> hdr = Header()
    >>> for line in format_bit_comments():
    ...     hdr['COMMENT'] = line

    Compact layout without hex values:

    >>> lines = format_bit_comments(include_hex=False)

    Namespaced for a shared extension header:

    >>> lines = format_bit_comments(prefix="BPM: ", width=72)

    Notes
    -----
    **Width safety** — FITS COMMENT cards have a hard 72-character content
    limit.  The original implementation used ``{bit_val:5d}`` which silently
    truncated values above 99999 (bit 17+) and had no wrapping at all, so
    long descriptions produced invalid FITS cards.  This version uses an
    8-digit zero-padded hex field (always exactly 10 characters including the
    ``0x`` prefix) and delegates all line-length enforcement to
    ``textwrap.wrap``.

    **Column alignment** — the preamble width is computed once from the
    longest name in ``FLAG_NAMES``, so all description columns are
    vertically aligned regardless of name length.

    See Also
    --------
    iter_flag_definitions : The underlying source of bit definitions.
    get_definitions_payload : Machine-readable JSON equivalent.
    """
    if not isinstance(width, int) or width < 1:
        raise ValueError(f"width must be a positive integer; got {width!r}.")

    # ------------------------------------------------------------------
    # 1. Compute the fixed-width preamble once so all entries are aligned.
    #
    #    Format (include_hex=True):
    #      "Bit  0 (0x00000001) PIX_BAD       "
    #       ^^^^  ^^^^^^^^^^^^^ ^^^^^^^^^^^^^^
    #       4+2   2+8+2 = 12    name_col+1
    #
    #    Format (include_hex=False):
    #      "Bit  0 PIX_BAD       "
    # ------------------------------------------------------------------
    name_col: int = max(len(n) for n in FLAG_NAMES.values())

    if include_hex:
        # "Bit XX (0xXXXXXXXX) {name:<name_col} "
        preamble_width: int = len(prefix) + 4 + 2 + 1 + 12 + 1 + name_col + 1
    else:
        # "Bit XX {name:<name_col} "
        preamble_width = len(prefix) + 4 + 2 + 1 + name_col + 1

    desc_width: int = width - preamble_width
    if desc_width < 8:
        raise ValueError(
            f"width={width} is too narrow: the preamble alone requires "
            f"{preamble_width} characters, leaving only {desc_width} for "
            f"descriptions (minimum 8).  Increase width or shorten prefix."
        )

    continuation_indent: str = " " * preamble_width

    def _make_preamble(idx: int, bit_val: int, name: str) -> str:
        if include_hex:
            return f"{prefix}Bit {idx:2d} (0x{bit_val:08X}) {name:<{name_col}s} "
        return f"{prefix}Bit {idx:2d} {name:<{name_col}s} "

    # ------------------------------------------------------------------
    # 2. Build the output list.
    # ------------------------------------------------------------------
    lines: List[str] = [f"{prefix}Bit flag definitions for BPM/DQ array (bitwise OR):"]

    for bit_val, name, desc in iter_flag_definitions():
        idx       = bit_index(bit_val)
        preamble  = _make_preamble(idx, bit_val, name)
        text      = desc or name

        wrapped = textwrap.wrap(
            text,
            width=width,
            initial_indent=preamble,
            subsequent_indent=continuation_indent,
            break_long_words=False,    # never mangle a technical term mid-word
            break_on_hyphens=False,    # likewise; "sub-standard" stays intact
        )

        # textwrap.wrap returns [] for empty text — always emit at least
        # the preamble so every bit appears in the header.
        lines.extend(wrapped or [preamble])

    return lines


def format_pixel_status(value: int) -> str:
    """Standardized string representation for logs."""
    return str(DefectType(value)).replace("DefectType.", "")


# =============================================================================
# Utility Functions
# =============================================================================

def is_perfect_pixel(pixel_value: int) -> bool:
    """Fast check if a pixel has NO defects at all."""
    return pixel_value == 0

def is_usable_pixel(pixel_value: int, ignore_transient: bool = True) -> bool:
    """Fast check if a pixel is scientifically usable."""
    mask = BPM_MASK_PERMANENT if ignore_transient else _MASK_ANY_DEFECT
    return (pixel_value & mask) == 0

def has_defect(pixel_value: int, defect_mask: int) -> bool:
    """
    Check if a pixel value contains a specific defect type or any defect 
    from a specified group/mask.
    
    Parameters
    ----------
    pixel_value : int
        The value from a BPM/DQ array for a specific pixel.
    defect_mask : int
        The specific bit (e.g., BPM_BIT_PIX_HOT) or combined mask 
        (e.g., BPM_MASK_COLD) to check against.
    
    Returns
    -------
    bool
        True if the pixel value has the specified bit(s) set.
    
    Examples
    --------
    >>> # Check for a single specific bit
    >>> has_defect(pixel_val, BPM_BIT_PIX_DEAD)
    
    >>> # Check if a pixel has ANY "Hot" properties (using a group mask)
    >>> has_defect(pixel_val, BPM_MASK_HOT)
    
    See Also
    --------
    get_defect_types : Return a list of all defect names present in a pixel.
    DefectType : Enum-based inspection for more complex logic.
    """
    return bool(pixel_value & defect_mask)


def get_defect_types(pixel_value: int) -> List[str]:
    """
    Get a list of all defect type names present in a pixel value.
    
    This function performs a bitwise inspection of the provided value against
    the global FLAG_NAMES mapping.
    
    Parameters
    ----------
    pixel_value : int
        The 32-bit BPM/DQ value for a pixel.
    
    Returns
    -------
    List[str]
        A list of string names (e.g., ['PIX_HOT', 'PIX_NONLIN']). 
        Returns an empty list if no bits are set.
    
    Examples
    --------
    >>> # Pixel 130 is 128 (PIX_NOISE) + 2 (PIX_DEAD)
    >>> get_defect_types(130)
    ['PIX_DEAD', 'PIX_NOISE']
    
    See Also
    --------
    has_defect : Check for a specific bit or group mask
    FLAG_NAMES : Mapping of bit-integers to string identifiers
    """
    if not pixel_value:
        return []

    # Iterate through known bits. Using sorted() ensures the output 
    # follows the bit-order (0 to 31), which is easier for humans to read.
    defects = [
        name for bit, name in _SORTED_FLAG_ITEMS
        if (pixel_value & bit)
    ]
    
    return defects

@njit
def _count_defects_numba(
    flat_bpm: npt.NDArray[np.unsignedinteger],
    mask: int) -> int:
    """Numba-accelerated defect counter (~10x faster for large arrays)."""
    count = 0
    for val in flat_bpm:
        if val & mask:
            count += 1
    return count

def count_defects(
        bpm: npt.NDArray[np.unsignedinteger],
        defect_mask: Optional[int] = None
    ) -> int:
    """
    Count pixels with defects in a bad pixel mask (with optional Numba acceleration)
    
    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (typically uint32).
    defect_mask : int, optional
        Specific defect bit (e.g., BPM_BIT_PIX_HOT) or group mask 
        (e.g., BPM_MASK_PERMANENT) to count. 
        If None, counts any pixel where ANY bit is set.
    
    Returns
    -------
    int
        Number of pixels matching the specified defect criteria.
    
    Examples
    --------
    >>> # Count all pixels that have ANY flag set
    >>> count_defects(bpm)
    
    >>> # Count only the permanent hardware defects
    >>> count_defects(bpm, BPM_MASK_PERMANENT)
    
    >>> # Count only cosmic rays in this frame
    >>> count_defects(bpm, BPM_BIT_PIX_COSMIC)
    """
    if defect_mask is None:
        # Returns count of pixels where value != 0
        return int(np.count_nonzero(bpm))
    
    if HAS_NUMBA and bpm.size > 1_000_000:  # Only for large arrays
        return int(_count_defects_numba(bpm.ravel(), defect_mask))
    
    # Returns count of pixels where (pixel & defect_mask) != 0
    return int(np.count_nonzero(bpm & defect_mask))


def create_bpm(
        shape: Tuple[int, ...], 
        dtype: type = np.uint32
    ) -> np.ndarray:
    """
    Create an empty (zero-initialized) bad pixel mask.
    
    Parameters
    ----------
    shape : tuple
        Shape of the BPM array (rows, cols) or (planes, rows, cols).
    dtype : np.dtype, optional
        Data type of the array. Default is uint32 to accommodate all 32 bits
        of the metrology/DQ hierarchy (including structural and transient flags).
    
    Returns
    -------
    np.ndarray
        An array of zeros with the specified shape and depth.
    
    Notes
    -----
    - uint32: Supports bits 0-31 (Full metrology suite).
    - uint16: Supports bits 0-15 (Pixel-level defects only; structural/DQ bits 
      will be truncated).
    
    Warnings
    --------
    If using uint16, attempting to set a bit >= 16 (e.g., COL_BAD) will result 
    in the bit being lost due to integer overflow. Always use uint32 for 
    per-image Data Quality (DQ) extensions.
    """

    import warnings

    if np.dtype(dtype).itemsize < 4:
        warnings.warn(
            f"dtype={dtype} cannot represent structural/DQ flags (bits >= 16). "
            "Consider dtype=np.uint32.", UserWarning, stacklevel=2
        )
    return np.zeros(shape, dtype=dtype)


def merge_bpms(
    bpm1: npt.NDArray[np.unsignedinteger],
    bpm2: npt.NDArray[np.unsignedinteger],
    operation: str = 'or'
) -> np.ndarray:
    """
    Merge two bad pixel masks using a bitwise operation.

    Parameters
    ----------
    bpm1 : np.ndarray
        First bad pixel mask (Base mask).
    bpm2 : np.ndarray
        Second bad pixel mask (Comparison/Update mask).
    operation : str, optional
        Merge operation:
        'or'  : Union (Accumulate flags from both masks).
        'and' : Intersection (Keep only flags found in both).
        'xor' : Symmetric Difference (Highlight state changes/inconsistencies).
        'diff': Subtraction (bpm1 & ~bpm2 - Strip specific flags from bpm1).

    Returns
    -------
    np.ndarray
        Merged uint32 BPM array.

    Notes
    -----
    The 'diff' operation is useful for removing dynamic artifacts from a 
    persistent master mask or isolating hardware-only defects.
    """
    bpm1 = np.asarray(bpm1, dtype=np.uint32)
    bpm2 = np.asarray(bpm2, dtype=np.uint32)

    if bpm1.shape != bpm2.shape:
        raise ValueError(f"BPM shapes must match: {bpm1.shape} != {bpm2.shape}")

    # Ensure we are operating on the same bit-depth
    op = operation.lower()
    
    if op == 'or':
        return bpm1 | bpm2
    elif op == 'and':
        return bpm1 & bpm2
    elif op == 'xor':
        return bpm1 ^ bpm2
    elif op == 'diff':
        # Result: Flags in bpm1 that ARE NOT present in bpm2
        return bpm1 & ~bpm2
    else:
        raise ValueError(f"Invalid operation: '{operation}'.")


def apply_bpm(
    data: npt.NDArray[np.number],
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int = BPM_MASK_PERMANENT,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Apply a bad pixel mask to a data array, replacing flagged pixels with a fill value.
    
    This function creates a boolean selection mask based on the provided 
    defect_mask and applies it to a copy of the input data.
    
    Parameters
    ----------
    data : np.ndarray
        The input image data (e.g., raw ADU or calibrated flux).
    bpm : np.ndarray
        The 32-bit bad pixel mask array.
    defect_mask : int, optional
        Which bits to trigger masking. Default is BPM_MASK_PERMANENT.
        Allows selective masking (e.g., masking only HOT pixels but 
        keeping NOISY ones).
    fill_value : float, optional
        The value used to replace bad pixels. Default is np.nan.
    
    Returns
    -------
    np.ndarray
        A copy of the data with bad pixels replaced. If fill_value is NaN 
        and input data is integer, the result is promoted to float64.

    Examples
    --------
    >>> # Mask only permanent hardware flaws
    >>> clean_data = apply_bpm(raw_data, master_bpm, defect_mask=BPM_MASK_PERMANENT)
    
    >>> # Mask ONLY pixels that reached saturation
    >>> data_no_sat = apply_bpm(raw_data, dq_array, defect_mask=BPM_BIT_SATURATED)
    """
    if data.shape != bpm.shape:
        raise ValueError(
            f"Data and BPM shapes must match: {data.shape} != {bpm.shape}"
        )

    # Determine output dtype first.
    # Guard: np.isnan() raises TypeError on non-numeric types and returns
    # an array (ambiguous truth value) if fill_value is array-like.
    # np.isscalar() ensures we only call np.isnan() on a true scalar.
    is_nan_fill: bool = np.isscalar(fill_value) and np.isnan(fill_value)

    fv_array = np.asarray(fill_value)
    if np.issubdtype(fv_array.dtype, np.floating) and not np.issubdtype(data.dtype, np.floating):
        out_dtype = np.float64
    else:
        if not np.can_cast(fv_array.dtype, data.dtype, casting='safe'):
            raise ValueError(
                f"fill_value {fill_value!r} cannot be safely cast to {data.dtype}. "
                "Consider setting fill_value=0 or promoting your data to float."
            )
        out_dtype = data.dtype
    
    # Now make the copy with the correct dtype
    result = data.astype(out_dtype, copy=True)
    bad_pixels = (bpm & defect_mask) != 0

    result[bad_pixels] = fill_value
    return result


def get_bpm_statistics(
    bpm_array: npt.NDArray[np.unsignedinteger],
    force_uint32: bool = True,
) -> Dict[str, DefectStatEntry]:
    """
    Count the occurrence and area fraction of every known defect type.

    Analyses the BPM array against all individual bits defined in
    ``FLAG_NAMES`` and all high-level group masks defined in ``_GROUP_MASKS``.
    The two categories are distinguished by the ``'type'`` field in each entry.

    Parameters
    ----------
    bpm_array : ndarray of uint32 (or uint16)
        The 2-D (or N-D) integer array containing BPM/DQ flags.
        Must be an unsigned integer dtype; a ``TypeError`` is raised otherwise.
    force_uint32: bool
        If True, force passed bpm to uint32

    Returns
    -------
    dict[str, DefectStatEntry]
        Ordered as individual bits first (ascending bit order), then group
        masks.  Each value is a ``DefectStatEntry`` TypedDict with keys:

        ``count`` : int
            Number of pixels where at least one bit of the mask is set.
        ``fraction`` : float
            ``count / bpm_array.size``.
        ``type`` : str
            ``'Individual Bit'`` for single-bit entries from ``FLAG_NAMES``,
            ``'Group Mask'`` for composite entries from ``_GROUP_MASKS``.

    Raises
    ------
    TypeError
        If ``bpm_array`` is not an unsigned integer dtype.

    Examples
    --------
    >>> stats = get_bpm_statistics(master_bpm)
    >>> print(f"Dead pixels : {stats['PIX_DEAD']['count']}")
    >>> print(f"Bad columns : {stats['MASK_COL']['count']}")
    >>> # Iterate only individual-bit entries
    >>> hw_stats = {k: v for k, v in stats.items() if v['type'] == 'Individual Bit'}

    Notes
    -----
    **Vectorised batch strategy** — the original implementation made one
    ``np.count_nonzero(bpm_array & mask)`` call per mask, resulting in O(M)
    full-array traversals (one per mask).  This implementation instead:

    1. Flattens the array once to a contiguous uint32 1-D view (zero-copy
       when the input is already contiguous uint32).
    2. Stacks all ``N_bits`` individual-bit masks into a 1-D uint32 vector
       and evaluates ``(flat[:, None] & bits[None, :]) != 0`` in a single
       NumPy broadcast, yielding an ``(N_pixels, N_bits)`` bool array.
       ``np.count_nonzero`` along axis 0 then returns all N_bits counts in
       one C-level loop.
    3. Repeats step 2 for the smaller set of group masks.

    Total array traversals: **2** (down from O(M) ≈ 34 in the original).

    Memory trade-off: the intermediate boolean array is
    ``N_pixels × N_masks × 1 byte``.  For a 4 k × 4 k sensor with 25 bits
    this is ≈ 400 MB.  If ``bpm_array.size`` exceeds ``_STATS_PIXEL_LIMIT``
    (default 64 M pixels), the function falls back to sequential passes to
    keep peak memory under ~256 MB.

    See Also
    --------
    FLAG_NAMES : Mapping of individual bit values to string identifiers.
    _GROUP_MASKS : Mapping of composite mask names to combined bit values.
    count_defects : Lightweight single-mask pixel counter.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if not np.issubdtype(bpm_array.dtype, np.unsignedinteger):
        raise TypeError(
            f"bpm_array must be an unsigned integer dtype (e.g. uint32); "
            f"got {bpm_array.dtype!r}."
        )

    # ------------------------------------------------------------------
    # 2. Single contiguous 1-D uint32 view — zero-copy when possible.
    #    np.asarray avoids a copy if dtype and layout already match.
    # ------------------------------------------------------------------        
    if force_uint32 or bpm_array.dtype != np.uint32:
        flat = np.asarray(bpm_array, dtype=np.uint32).ravel()
    else:
        flat = bpm_array.ravel()

    total_pixels: int = bpm_array.size
    if total_pixels == 0:
        raise ValueError("bpm_array is empty (size == 0).")

    inv_total = 1.0 / total_pixels  # multiply is faster than repeated divides

    # ------------------------------------------------------------------
    # 3. Choose vectorised-batch or sequential strategy based on the
    #    expected peak memory of the intermediate boolean matrix.
    #
    #    Peak memory (bytes) ≈ N_pixels × N_masks
    #    Threshold: 64 M pixels keeps peak below ~256 MB for ~25 masks.
    # ------------------------------------------------------------------
    use_batch: bool = total_pixels <= _BPM_STATS_PIXEL_LIMIT

    stats: Dict[str, DefectStatEntry] = {}

    # ------------------------------------------------------------------
    # 4a. Individual-bit statistics (vectorised batch path)
    # ------------------------------------------------------------------
    # Preserve insertion order (ascending bit value) from FLAG_NAMES.
    ind_names  = list(FLAG_NAMES.values())               # ['PIX_BAD', ...]
    ind_bits   = np.array(list(FLAG_NAMES.keys()),        # shape (B,)
                          dtype=np.uint32)

    if use_batch:
        # Shape: (N_pixels, N_bits) — one broadcast + one reduction.
        ind_counts: npt.NDArray[np.intp] = np.count_nonzero(
            (flat[:, None] & ind_bits[None, :]),
            axis=0,
        )
    else:
        # Sequential fallback: still vectorised per-mask, but avoids the
        # large intermediate matrix.
        ind_counts = np.array(
            [np.count_nonzero(flat & b) for b in ind_bits],
            dtype=np.intp,
        )

    for name, count in zip(ind_names, ind_counts):
        stats[name] = DefectStatEntry(
            count=int(count),
            fraction=float(count) * inv_total,
            type='Individual Bit',
        )

    # ------------------------------------------------------------------
    # 4b. Group-mask statistics (vectorised batch path)
    # ------------------------------------------------------------------
    grp_names  = list(_GROUP_MASKS.keys())
    grp_values = np.array(list(_GROUP_MASKS.values()),    # shape (G,)
                          dtype=np.uint32)

    if use_batch:
        grp_counts: npt.NDArray[np.intp] = np.count_nonzero(
            (flat[:, None] & grp_values[None, :]),
            axis=0,
        )
    else:
        grp_counts = np.array(
            [np.count_nonzero(flat & g) for g in grp_values],
            dtype=np.intp,
        )

    for name, count in zip(grp_names, grp_counts):
        stats[name] = DefectStatEntry(
            count=int(count),
            fraction=float(count) * inv_total,
            type='Group Mask',
        )

    return stats


def combine_flags(defect_flags: List[int]) -> int:
    """
    Combine a list of defect bits or masks into a single bitwise integer.
    
    This is a low-level helper used to flatten multiple criteria into a 
    single mask for functions like apply_bpm() or count_defects().

    Parameters
    ----------
    defect_flags : List[int]
        List of integers (e.g., [BPM_BIT_PIX_HOT, BPM_BIT_PIX_DEAD]).
    
    Returns
    -------
    int
        The bitwise OR combination of all input flags.
    """
    if not defect_flags:
        return 0
        
    combined = 0
    for flag in defect_flags:
        # Cast to int to ensure compatibility with Enums or numpy types
        combined |= int(flag)
        
    return combined


def get_defect_mask(
    bpm_array: npt.NDArray[np.unsignedinteger],
    defect_types: Optional[List[int]] = None
) -> np.ndarray:
    """
    Return a boolean mask selecting specific defect types or groups.

    Parameters
    ----------
    bpm_array : np.ndarray
        The 32-bit integer array containing BPM/DQ flags.
    defect_types : List[int], optional
        List of defect bits or group masks to include. 
        If None, any non-zero value is treated as a defect.

    Returns
    -------
    np.ndarray
        A boolean array of the same shape as bpm_array, where True 
        indicates the presence of the requested defect(s).
        
    Examples
    --------
    >>> # Get all hardware-related defects
    >>> mask = get_defect_mask(bpm, [BPM_MASK_PERMANENT])
    
    >>> # Get only pixels that are either DEAD or COLD
    >>> mask = get_defect_mask(bpm, [BPM_BIT_PIX_DEAD, BPM_BIT_PIX_QE_LOW])
    """
    if defect_types is None:
        # Catch-all: anything that isn't 'Good' (0)
        return bpm_array != 0

    # Leverage the helper to flatten the list into a single bitwise filter
    combined = combine_flags(defect_types)
    
    # Standard bitwise logic: True if any requested bit overlaps
    return (bpm_array & combined) != 0


def filter_bpm(
    bpm_array: npt.NDArray[np.unsignedinteger],
    defect_types: List[int]
) -> np.ndarray:
    """
    Return a BPM array containing only the requested defect bits.

    Unlike get_defect_mask(), this preserves the specific bit values for 
    the selected defects, clearing all others.

    Parameters
    ----------
    bpm_array : np.ndarray
        The 32-bit integer array containing BPM/DQ flags.
    defect_types : List[int]
        List of defect bits or group masks to retain.

    Returns
    -------
    np.ndarray
        Array of the same shape and dtype as bpm_array, with bits 
        not in defect_types cleared to 0.
    """
    # Flatten the list into a single bitwise search filter
    combined = combine_flags(defect_types)
    
    # Cast the mask to match the array dtype to prevent bitwise 
    # overflows or unexpected sign-bit behavior.
    cm = np.asarray(combined, dtype=bpm_array.dtype)
    
    return bpm_array & cm


def describe_pixel_defects(bpm_value: int) -> str:
    """
    Return a human-readable string describing all defects present in a pixel.
    
    Parameters
    ----------
    bpm_value : int
        The 32-bit integer value for a single pixel in a BPM/DQ array.
        
    Returns
    -------
    str
        A comma-separated string of defect names, or "Good pixel" if value is 0.
        
    Examples
    --------
    >>> describe_pixel_defects(0)
    'Good pixel'
    >>> # Value 10 (8 + 2)
    >>> describe_pixel_defects(10)
    'PIX_DEAD, PIX_HOT'
    
    See Also
    --------
    get_defect_types : Returns the same information as a list of strings.
    FLAG_DESCRIPTIONS : Dictionary providing the long-form explanation of each bit.
    """
    if bpm_value == 0:
        return "Good pixel"
    
    # We use get_defect_types to maintain a single point of logic for bit-parsing
    defects = get_defect_types(bpm_value)
    
    if not defects:
        return f"Unknown defect (Value: {bpm_value})"
    
    return ", ".join(defects)


def describe_pixel_defects_verbose(bpm_value: int) -> str:
    """Returns a multi-line string with detailed descriptions of each defect.
    
    Parameters
    ----------
    bpm_value : int
        The 32-bit integer value for a single pixel.
        
    Returns
    -------
    str
        Multi-line string with bullet points, or "Good pixel" if value is 0.
        
    See Also
    --------
    describe_pixel_defects : Single-line summary version.
    FLAG_DESCRIPTIONS : Source of detailed explanations.
    """
    if bpm_value == 0:
        return "Good pixel (no defects detected)"  # More informative
    
    lines = []
    for bit, name in _SORTED_FLAG_ITEMS:
        if bpm_value & bit:
            desc = FLAG_DESCRIPTIONS.get(bit, "No description available")
            lines.append(f"  • {name}: {desc}")  # Better bullet character
    
    if not lines:
        return f"Unknown defect pattern (value: 0x{bpm_value:08X})"
    
    return "\n".join(lines)


def check_bad_rows_cols(
    bpm_array: npt.NDArray[np.unsignedinteger],
    row_threshold: float = 0.1,
    col_threshold: float = 0.1,
    defect_mask: int = BPM_MASK_PERMANENT,
    active_region: Optional[Tuple[slice, slice]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify rows and columns that exceed a defect density threshold.
    
    This function is used to promote clustered pixel defects to structural 
    row/column flags (e.g., flagging a column because it contains a dead tap).

    Parameters
    ----------
    bpm_array : np.ndarray
        The 32-bit BPM/DQ array. bpm_array must be 2-D (rows, cols)
    row_threshold : float
        Fraction (0.0 to 1.0) of bad pixels needed to flag a row.
    col_threshold : float
        Fraction (0.0 to 1.0) of bad pixels needed to flag a column.
    defect_mask : int
        Which bits trigger the "bad" count. Defaults to BPM_MASK_PERMANENT.
    active_region : tuple of slices, optional
        Used to ignore overscan or reference pixel regions.

    Returns
    -------
    bad_rows : np.ndarray (bool)
        Boolean mask of length N_rows.
    bad_cols : np.ndarray (bool)
        Boolean mask of length N_cols.
    """

    if bpm_array.ndim != 2:
        raise ValueError("check_bad_rows_cols expects a 2-D BPM array (rows, cols).")

    for name, val in [('row_threshold', row_threshold), ('col_threshold', col_threshold)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]; got {val!r}")
    
    if active_region is None:
        active_region = (slice(None), slice(None))
    
    row_slice, col_slice = active_region
    # Isolate relevant bits for the check
    check_mask = (bpm_array[row_slice, col_slice] & defect_mask) != 0
    
    # Calculate density using numpy axis means (very fast)
    row_bad_fractions = np.mean(check_mask, axis=1)
    col_bad_fractions = np.mean(check_mask, axis=0)
    
    bad_rows_active = row_bad_fractions > row_threshold
    bad_cols_active = col_bad_fractions > col_threshold

    # Map results back to full array size if active_region was used
    bad_rows = np.zeros(bpm_array.shape[0], dtype=bool)
    bad_cols = np.zeros(bpm_array.shape[1], dtype=bool)
    
    # Get the actual indices
    if isinstance(row_slice, slice):
        row_indices = np.arange(*row_slice.indices(bpm_array.shape[0]))
    else:
        row_indices = np.asarray(row_slice)
        
    if isinstance(col_slice, slice):
        col_indices = np.arange(*col_slice.indices(bpm_array.shape[1]))
    else:
        col_indices = np.asarray(col_slice)

    bad_rows[row_indices] = bad_rows_active
    bad_cols[col_indices] = bad_cols_active
    
    return bad_rows, bad_cols

def flag_bad_rows_cols(
    bpm_array: npt.NDArray[np.unsignedinteger],
    row_threshold: float = 0.1,
    col_threshold: float = 0.1,
    defect_mask: int = BPM_MASK_PERMANENT,
    active_region: Optional[Tuple[slice, slice]] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Apply structural flags (ROW_BAD/COL_BAD) to a BPM based on defect density.
    
    This function promotes high-density pixel defects to structural defects.
    """
    # 1. Get the boolean masks from your query function
    bad_rows, bad_cols = check_bad_rows_cols(
        bpm_array, row_threshold, col_threshold, defect_mask, active_region
    )

    # 2. Work on a copy to maintain functional purity
    new_bpm = bpm_array.copy()

    # 3. Use NumPy Broadcasting instead of for-loops (Much faster)
    if np.any(bad_rows):
        # This flags every pixel in the bad rows across all columns
        new_bpm[bad_rows, :] |= BPM_BIT_ROW_BAD
        
    if np.any(bad_cols):
        # This flags every pixel in the bad columns across all rows
        new_bpm[:, bad_cols] |= BPM_BIT_COL_BAD

    if verbose:
        n_rows = np.sum(bad_rows)
        n_cols = np.sum(bad_cols)
        if n_rows > 0 or n_cols > 0:
            logger.info(f"Flagged {n_rows} rows and {n_cols} columns as structurally bad.")

    return new_bpm

def to_boolean_mask(
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int = BPM_MASK_PERMANENT,
    invert: bool = False
) -> np.ndarray:
    """Convert BPM to boolean mask suitable for NumPy indexing.
    
    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array.
    defect_mask : int
        Which defects to include in the mask.
    invert : bool
        If True, return mask of GOOD pixels (True = usable).
        If False, return mask of BAD pixels (True = defective).
    
    Returns
    -------
    np.ndarray (bool)
        Boolean mask array.
        
    Examples
    --------
    >>> # Get all good pixels for statistical analysis
    >>> good_mask = to_boolean_mask(bpm, BPM_MASK_PERMANENT, invert=True)
    >>> median_good = np.median(data[good_mask])
    """
    mask = (bpm & defect_mask) != 0
    return ~mask if invert else mask


def export_to_ds9_region(
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int,
    filename: Union[str, Path, TextIO],
    region_type: str = "box",
    color: str = "red",
    coord_system: str = "physical",
    origin: int = 1,
) -> None:
    """
    Export flagged pixels as a DS9 region file for visualization.

    Parameters
    ----------
    bpm : ndarray of unsigned integer, shape (ny, nx)
        2-D bad-pixel mask array.
    defect_mask : int
        Bit or mask selecting which defects to export (e.g. BPM_BIT_PIX_HOT,
        BPM_MASK_PERMANENT, etc.). Any pixel with (bpm & defect_mask) != 0
        will produce a region entry.
    filename : str, pathlib.Path, or file-like
        Output target. May be:
          - a path (string or Path), in which case the file is opened here, or
          - an open file-like object with a .write() method.
    region_type : {'box', 'point', 'circle'}, optional
        DS9 region primitive to use. Defaults to 'box'.
        For 'box', each pixel is rendered as a 1x1 box centered on the pixel.
    color : str, optional
        DS9 color name for the regions. Default is 'red'.
    coord_system : str, optional
        DS9 coordinate system keyword written at the top of the file.
        Common values: 'physical', 'image'. Default is 'physical'.
    origin : int, optional
        Coordinate origin offset. DS9 is 1-based by default.
        If origin=1 (default), pixel (row=0, col=0) is written as (1,1).
        If origin=0, pixel (row=0, col=0) is written as (0,0).

    Raises
    ------
    ValueError
        If `bpm` is not 2-D or `region_type` is unsupported.

    Notes
    -----
    - This function assumes that the bpm indices map directly to DS9 pixel
      coordinates (no WCS transformation). For WCS-based regions, you would
      need to convert indices to world coordinates before writing.
    """
    if bpm.ndim != 2:
        raise ValueError("export_to_ds9_region expects a 2-D BPM array (rows, cols).")

    # Validate region type; you can easily extend this set later.
    supported_region_types = {"box", "point", "circle"}
    if region_type not in supported_region_types:
        raise ValueError(
            f"Unsupported region_type={region_type!r}. "
            f"Supported types: {sorted(supported_region_types)}"
        )

    # Find all flagged pixels
    mask = (bpm & defect_mask) != 0
    coords = np.argwhere(mask)  # shape (N, 2), entries are (row, col)

    # Prepare a small helper to write a single region line
    def _format_region_line(row: int, col: int) -> str:
        x = col + origin
        y = row + origin
        if region_type == "box":
            # 1x1 box centered on the pixel
            return f"box({x},{y},1,1) # color={color}\n"
        elif region_type == "point":
            return f"point({x},{y}) # point=x color={color}\n"
        elif region_type == "circle":
            # radius=0.5 pixel as a reasonable default
            return f"circle({x},{y},0.5) # color={color}\n"
        # Should never reach here due to earlier validation
        raise RuntimeError("Unexpected region_type in _format_region_line")

    # Open file if a path was provided; otherwise use the given file-like object.
    needs_close = False
    if isinstance(filename, (str, Path)):
        f = open(filename, "w", encoding="utf-8")
        needs_close = True
    else:
        f = filename  # assume file-like

    try:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"{coord_system}\n")
        for row, col in coords:
            f.write(_format_region_line(int(row), int(col)))
    finally:
        if needs_close:
            f.close()


def summarize_bpm(
        bpm: npt.NDArray[np.unsignedinteger],
        verbose: bool = True
    ) -> Dict[str, Any]:
    """Generate comprehensive BPM quality report.
    
        bpm_array must be 2-D (rows, cols)”.

    Returns
    -------
    dict
        Keys:
        - 'total_pixels' : int
        - 'defective_pixels' : int
        - 'yield_percent' : float
        - 'worst_row' : int
        - 'worst_row_count' : int
        - 'worst_col' : int
        - 'worst_col_count' : int
        - 'statistics' : dict[str, DefectStatEntry]
    """
    if bpm.ndim != 2:
        raise ValueError("check_bad_rows_cols expects a 2-D BPM array (rows, cols).")

    stats = get_bpm_statistics(bpm)
    
    total_defective = np.count_nonzero(bpm != 0)
    yield_pct = 100 * (1 - total_defective / bpm.size)
    
    # Find worst row/column
    row_counts = np.count_nonzero(bpm != 0, axis=1)
    col_counts = np.count_nonzero(bpm != 0, axis=0)
    
    report = {
        'total_pixels': bpm.size,
        'defective_pixels': int(total_defective),
        'yield_percent': float(yield_pct),
        'worst_row': int(np.argmax(row_counts)),
        'worst_row_count': int(row_counts.max()),
        'worst_col': int(np.argmax(col_counts)),
        'worst_col_count': int(col_counts.max()),
        'statistics': stats,
    }
    
    if verbose:
        logger.info(f"BPM Quality Report:")
        logger.info(f"  Array shape:      {bpm.shape}")
        logger.info(f"  Pixel yield:      {yield_pct:.2f}%")
        logger.info(f"  Total defects:    {total_defective:,}")
        logger.info(f"  Worst row:        {report['worst_row']} ({report['worst_row_count']} defects)")
        logger.info(f"  Worst column:     {report['worst_col']} ({report['worst_col_count']} defects)")
    
    return report

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # --- Pixel-Level Bit Constants (0-15) ---
    'BPM_BIT_PIX_BAD',
    'BPM_BIT_PIX_DEAD',
    'BPM_BIT_PIX_BRIGHT',
    'BPM_BIT_PIX_HOT',
    'BPM_BIT_PIX_QE_HIGH',
    'BPM_BIT_PIX_QE_LOW',
    'BPM_BIT_PIX_NONLIN',
    'BPM_BIT_PIX_NOISE',
    'BPM_BIT_PIX_RTS',
    'BPM_BIT_PIX_TRAP',
    'BPM_BIT_PIX_GLOW',
    'BPM_BIT_PIX_RESERVED_11',
    'BPM_BIT_PIX_RESERVED_12',
    'BPM_BIT_PIX_RESERVED_13',
    'BPM_BIT_PIX_RESERVED_14',
    'BPM_BIT_PIX_EXCLUDED',

    # --- Structural Bit Constants (16-26) ---
    'BPM_BIT_COL_BAD',
    'BPM_BIT_COL_BRIGHT',
    'BPM_BIT_COL_DARK',
    'BPM_BIT_COL_TRAP',
    'BPM_BIT_COL_BLOCKED',
    'BPM_BIT_COL_CTE_LOW',
    'BPM_BIT_COL_BOUNDARY',
    'BPM_BIT_COL_EXCLUDED',
    'BPM_BIT_ROW_BAD',
    'BPM_BIT_ROW_RESERVED_25',
    'BPM_BIT_ROW_EXCLUDED',

    # --- Transient / DQ Bit Constants (27-31) ---
    'BPM_BIT_SATURATED',
    'BPM_BIT_PIX_COSMIC',
    'BPM_BIT_PIX_XTALK',
    'BPM_BIT_PIX_BLOOM',
    'BPM_BIT_PIX_OUTLIER',

    # --- Combined Masks ---
    'BPM_MASK_PIX',
    'BPM_MASK_COL',
    'BPM_MASK_ROW',
    'BPM_MASK_PERMANENT',
    'BPM_MASK_TRANSIENT',
    'BPM_MASK_HOT',
    'BPM_MASK_COLD',
    'BPM_MASK_NOISE',
    'BPM_MASK_TRANSPORT',
    'BPM_MASK_IGNORE',
    'BPM_MASK_BAD', 
    'BPM_MASK_TRAPS', 
    'BPM_MASK_RESPONSE', 
    'BPM_MASK_ALL_16', 
    'BPM_MASK_ALL',
    'BPM_MASK_ALL_32',

    # --- Enum and Documentation ---
    'DefectType',
    'FLAG_NAMES',
    'FLAG_DESCRIPTIONS',

    # --- Utility Functions ---
    'has_defect',
    'get_defect_types',
    'count_defects',
    'combine_flags',
    'create_bpm',
    'merge_bpms',
    'apply_bpm',
    'get_bpm_statistics',
    'bit_index',
    'iter_flag_definitions',
    'format_bit_comments',
    'get_definitions_payload',
    'get_defect_mask', 'filter_bpm', 
    'describe_pixel_defects', 
    'describe_pixel_defects_verbose', 
    'check_bad_rows_cols', 
    'flag_bad_rows_cols', 
    'format_pixel_status',
    'to_boolean_mask',
    'export_to_ds9_region',
    'summarize_bpm',
]
