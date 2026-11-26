#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Bad pixel mask bitmask definitions and utilities for sensor metrology.

This module defines the bit flags used in bad pixel masks (BPM) throughout
the fits_tools package. These masks use a 32-bit integer architecture to
encode hardware defects, structural anomalies, and transient data quality
issues into a single array.

Bitmask Structure
-----------------
Each pixel in a BPM array is a uint32 where individual bits indicate
specific defect types, organized into logical zones:

- Bits 0–15:  Single-pixel hardware defects (static)
- Bits 16–26: Structural defects (column/row level)
- Bits 27–31: Transient artifacts (per-image data quality)

Bit Assignments
---------------
**Pixel-Level (Bits 0–15)**

=====  ==========  ==============  ==========================================
Bit    Value       Name            Description
=====  ==========  ==============  ==========================================
0      1           PIX_BAD         General unusable pixel (summary flag)
1      2           PIX_DEAD        Unresponsive pixel
2      4           PIX_BRIGHT      High bias offset
3      8           PIX_HOT         Excessive dark current
4      16          PIX_PRNU_HIGH   High PRNU (sensitivity outlier)
5      32          PIX_PRNU_LOW    Low PRNU (sensitivity outlier)
6      64          PIX_NONLIN      Non-linear response
7      128         PIX_NOISE       Excessive read noise
8      256         PIX_RTS         Random Telegraph Signal
9      512         PIX_TRAP        Individual charge trap
10     1024        PIX_GLOW        Amplifier/circuitry glow
11     2048        PIX_CLUSTER     Part of a cluster defect
15     32768       PIX_EXCLUDED    Manually excluded
=====  ==========  ==============  ==========================================

**Structural (Bits 16–26)**

=====  ============  ==============  ==========================================
Bit    Value         Name            Description
=====  ============  ==============  ==========================================
16     65536         COL_BAD         General bad column
17     131072        COL_BRIGHT      Excessively bright column
18     262144        COL_DARK        Dark or blocked column
19     524288        COL_TRAP        Column-level charge trap
20     1048576       COL_BLOCKED     Vertical transfer obstructed
21     2097152       COL_CTE_LOW     Poor Charge Transfer Efficiency
22     4194304       COL_BOUNDARY    Amplifier/segment boundary
23     8388608       COL_EXCLUDED    Column excluded from measurements
24     16777216      ROW_BAD         General bad row
26     67108864      ROW_EXCLUDED    Row excluded from measurements
=====  ============  ==============  ==========================================

**Transient/DQ (Bits 27–31)**

=====  ===========  ==============  ==========================================
Bit    Value        Name            Description
=====  ===========  ==============  ==========================================
27     134217728    SATURATED       Full-well capacity reached
28     268435456    COSMIC          Cosmic ray hit
29     536870912    CROSSTALK       Electronic crosstalk artifact
30     1073741824   BLOOM           Saturated charge leakage
31     2147483648   OUTLIER         Statistical outlier
=====  ===========  ==============  ==========================================

Thread Safety
-------------
All functions are thread-safe. Functions that modify arrays return copies
rather than mutating inputs, making them safe for concurrent use.

Examples
--------
Creating and inspecting a mask::

    import numpy as np
    from fits_tools.bitmask import BPM_BIT_PIX_HOT, BPM_BIT_PIX_NONLIN
    bpm = np.zeros((2048, 2048), dtype=np.uint32)
    bpm[100, 200] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN

Checking for defect categories::

    from fits_tools.bitmask import BPM_MASK_COLD
    if bpm[10, 10] & BPM_MASK_COLD:
        print("Pixel is unresponsive or blocked")

Using the type-safe enum::

    from fits_tools.bitmask import DefectType
    status = DefectType(bpm[100, 200])
    print(status)   # e.g. "PIX_HOT|PIX_NONLIN"
"""

from __future__ import annotations

import logging
import textwrap
import warnings
from collections.abc import Callable, Iterator
from enum import IntFlag
from pathlib import Path
from typing import Any, TextIO, TypedDict

import numpy as np
import numpy.typing as npt

from .types import StrOrPath

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Callable:  # type: ignore[misc]
        """No-op decorator substituted when numba is not installed."""
        def wrapper(func: Callable) -> Callable:
            return func
        return wrapper

logger = logging.getLogger(__name__)

_BPM_STATS_PIXEL_LIMIT: int = 64 * 1024 * 1024  # 64 Mpix threshold for batch mode
_DEFINITIONS_SCHEMA_VERSION: int = 1
# FITS COMMENT card content is 72 characters wide (80-byte card minus "COMMENT ").
_FITS_COMMENT_WIDTH: int = 72


# ---------------------------------------------------------------------------
# TypedDict for per-entry statistics return structure.
# Gives callers type-checker support for keys like stats['PIX_DEAD']['count'].
# ---------------------------------------------------------------------------

class DefectStatEntry(TypedDict):
    """Single entry in the dict returned by :func:`get_bpm_statistics`.

    Attributes
    ----------
    count : int
        Number of pixels matching this flag or mask.
    fraction : float
        Fraction of total pixels (count / total).
    type : str
        ``'Individual Bit'`` or ``'Group Mask'``.
    """

    count:    int
    fraction: float
    type:     str


# =============================================================================
# Bitmask Constants
# =============================================================================

# --- Pixel-Level (Static/Hardware) Bits 0–15 ---
BPM_BIT_PIX_BAD           = 1 << 0   # General bad pixel
BPM_BIT_PIX_DEAD          = 1 << 1   # Unresponsive
BPM_BIT_PIX_BRIGHT        = 1 << 2   # High bias offset
BPM_BIT_PIX_HOT           = 1 << 3   # Excess dark current
BPM_BIT_PIX_PRNU_HIGH     = 1 << 4   # High PRNU (sensitivity outlier)
BPM_BIT_PIX_PRNU_LOW      = 1 << 5   # Low PRNU (sensitivity outlier)
BPM_BIT_PIX_NONLIN        = 1 << 6   # Non-linear response
BPM_BIT_PIX_NOISE         = 1 << 7   # High read noise
BPM_BIT_PIX_RTS           = 1 << 8   # Random Telegraph Signal
BPM_BIT_PIX_TRAP          = 1 << 9   # Pixel-level charge trap
BPM_BIT_PIX_GLOW          = 1 << 10  # Amplifier/circuitry glow
BPM_BIT_PIX_CLUSTER       = 1 << 11  # Part of a cluster defect
BPM_BIT_PIX_RESERVED_12   = 1 << 12  # Reserved
BPM_BIT_PIX_RESERVED_13   = 1 << 13  # Reserved
BPM_BIT_PIX_RESERVED_14   = 1 << 14  # Reserved
BPM_BIT_PIX_EXCLUDED      = 1 << 15  # Manually excluded

# --- Structural (Column/Row) Bits 16–26 ---
BPM_BIT_COL_BAD           = 1 << 16  # General bad column
BPM_BIT_COL_BRIGHT        = 1 << 17  # Bright column
BPM_BIT_COL_DARK          = 1 << 18  # Dark/blocked column
BPM_BIT_COL_TRAP          = 1 << 19  # Column-level trap
BPM_BIT_COL_BLOCKED       = 1 << 20  # Transfer blocked
BPM_BIT_COL_CTE_LOW       = 1 << 21  # Poor Charge Transfer Efficiency
BPM_BIT_COL_BOUNDARY      = 1 << 22  # Segment/amplifier boundary column
BPM_BIT_COL_EXCLUDED      = 1 << 23  # Column excluded from measurements

BPM_BIT_ROW_BAD           = 1 << 24  # General bad row
BPM_BIT_ROW_RESERVED_25   = 1 << 25  # Reserved
BPM_BIT_ROW_EXCLUDED      = 1 << 26  # Row excluded from measurements

# --- Transient (Per-Image DQ) Bits 27–31 ---
BPM_BIT_SATURATED         = 1 << 27  # Pixel at full-well
BPM_BIT_COSMIC            = 1 << 28  # Cosmic ray hit
BPM_BIT_CROSSTALK         = 1 << 29  # Electronic crosstalk artifact
BPM_BIT_BLOOM             = 1 << 30  # Saturated charge leakage
BPM_BIT_OUTLIER           = 1 << 31  # Statistical outlier

# --- Physical category masks ---
BPM_MASK_PIX = (
    BPM_BIT_PIX_BAD | BPM_BIT_PIX_DEAD | BPM_BIT_PIX_BRIGHT
    | BPM_BIT_PIX_HOT | BPM_BIT_PIX_PRNU_HIGH | BPM_BIT_PIX_PRNU_LOW
    | BPM_BIT_PIX_NONLIN | BPM_BIT_PIX_NOISE | BPM_BIT_PIX_RTS
    | BPM_BIT_PIX_TRAP | BPM_BIT_PIX_GLOW | BPM_BIT_PIX_CLUSTER
    | BPM_BIT_PIX_EXCLUDED
    # Reserved bits 12–14 are deliberately excluded until assigned.
)

BPM_MASK_COL = (
    BPM_BIT_COL_BAD | BPM_BIT_COL_BRIGHT | BPM_BIT_COL_DARK
    | BPM_BIT_COL_TRAP | BPM_BIT_COL_BLOCKED | BPM_BIT_COL_CTE_LOW
    | BPM_BIT_COL_BOUNDARY | BPM_BIT_COL_EXCLUDED
)

BPM_MASK_ROW = BPM_BIT_ROW_BAD | BPM_BIT_ROW_EXCLUDED

# --- High-level category masks ---

# Permanent (static) hardware defects — everything physically wrong with the
# silicon. EXCLUDED bits are intentionally omitted; they live in BPM_MASK_IGNORE.
BPM_MASK_PERMANENT = (
    BPM_BIT_PIX_BAD | BPM_BIT_PIX_DEAD | BPM_BIT_PIX_BRIGHT
    | BPM_BIT_PIX_HOT | BPM_BIT_PIX_PRNU_HIGH | BPM_BIT_PIX_PRNU_LOW
    | BPM_BIT_PIX_NONLIN | BPM_BIT_PIX_NOISE | BPM_BIT_PIX_RTS
    | BPM_BIT_PIX_TRAP | BPM_BIT_PIX_GLOW | BPM_BIT_PIX_CLUSTER
    | BPM_BIT_COL_BAD | BPM_BIT_COL_BRIGHT | BPM_BIT_COL_DARK
    | BPM_BIT_COL_TRAP | BPM_BIT_COL_BLOCKED | BPM_BIT_COL_CTE_LOW
    | BPM_BIT_ROW_BAD
)

# Transient/conditional defects from exposure or test conditions.
# EXCLUDED bits intentionally omitted; they live in BPM_MASK_IGNORE.
BPM_MASK_TRANSIENT = (
    BPM_BIT_SATURATED | BPM_BIT_COSMIC
    | BPM_BIT_CROSSTALK | BPM_BIT_BLOOM | BPM_BIT_OUTLIER
)

# --- Functional / diagnostic masks ---

# High-signal anomalies (potential sources of false positives)
BPM_MASK_HOT = (
    BPM_BIT_PIX_BRIGHT | BPM_BIT_PIX_HOT
    | BPM_BIT_PIX_PRNU_HIGH | BPM_BIT_COL_BRIGHT
)

# Cold / dead pixels
BPM_MASK_COLD = (
    BPM_BIT_PIX_DEAD | BPM_BIT_PIX_PRNU_LOW
    | BPM_BIT_COL_DARK | BPM_BIT_COL_BLOCKED
)

# Temporal instability (noise changing frame-to-frame)
BPM_MASK_NOISE = BPM_BIT_PIX_NOISE | BPM_BIT_PIX_RTS | BPM_BIT_CROSSTALK

# Structural charge-transport issues
BPM_MASK_TRANSPORT = (
    BPM_BIT_PIX_TRAP | BPM_BIT_COL_TRAP
    | BPM_BIT_COL_BLOCKED | BPM_BIT_COL_CTE_LOW
)

# All 'BAD' summary flags
BPM_MASK_BAD = BPM_BIT_PIX_BAD | BPM_BIT_ROW_BAD | BPM_BIT_COL_BAD

# Charge trap defects
BPM_MASK_TRAPS = BPM_BIT_PIX_TRAP | BPM_BIT_COL_TRAP

# Response anomalies
BPM_MASK_RESPONSE = BPM_BIT_PIX_DEAD | BPM_BIT_SATURATED | BPM_BIT_BLOOM

# Pixels excluded entirely from statistical averages
BPM_MASK_IGNORE = (
    BPM_BIT_PIX_EXCLUDED | BPM_BIT_COL_EXCLUDED | BPM_BIT_ROW_EXCLUDED
)

BPM_MASK_ALL_16 = 0xFFFF
BPM_MASK_ALL_32 = 0xFFFFFFFF
# Default to 32-bit "all bits" mask because this module defines bits >= 16.
BPM_MASK_ALL = BPM_MASK_ALL_32

# COL_BOUNDARY is intentionally excluded from BPM_MASK_PERMANENT: flat-field
# calibration corrects boundary response differences, so those pixels remain
# scientifically usable.
_MASK_ANY_DEFECT: int = BPM_MASK_PERMANENT | BPM_MASK_TRANSIENT | BPM_MASK_IGNORE

# Group masks used by get_bpm_statistics — built once at module level.
_GROUP_MASKS: dict[str, int] = {
    "MASK_PIX":       BPM_MASK_PIX,
    "MASK_COL":       BPM_MASK_COL,
    "MASK_ROW":       BPM_MASK_ROW,
    "MASK_PERMANENT": BPM_MASK_PERMANENT,
    "MASK_TRANSIENT": BPM_MASK_TRANSIENT,
    "MASK_HOT":       BPM_MASK_HOT,
    "MASK_COLD":      BPM_MASK_COLD,
    "MASK_NOISE":     BPM_MASK_NOISE,
    "MASK_TRANSPORT": BPM_MASK_TRANSPORT,
    "MASK_BAD":       BPM_MASK_BAD,
    "MASK_TRAPS":     BPM_MASK_TRAPS,
    "MASK_RESPONSE":  BPM_MASK_RESPONSE,
    "MASK_IGNORE":    BPM_MASK_IGNORE,
}


# =============================================================================
# Enum for type-safe access
# =============================================================================


class DefectType(IntFlag):
    """Type-safe enumeration of sensor and data quality defects.

    Members are categorized by prefix:

    - ``PIX_*`` — single-pixel hardware defects (bits 0–15)
    - ``COL_*`` — column structural defects (bits 16–23)
    - ``ROW_*`` — row structural defects (bits 24–26)
    - ``SATURATED``, ``COSMIC``, etc. — transient artifacts (bits 27–31)

    Examples
    --------
    >>> status = DefectType(BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN)
    >>> print(status)
    PIX_HOT|PIX_NONLIN
    """

    # Single-pixel defects
    PIX_BAD       = BPM_BIT_PIX_BAD
    PIX_DEAD      = BPM_BIT_PIX_DEAD
    PIX_BRIGHT    = BPM_BIT_PIX_BRIGHT
    PIX_HOT       = BPM_BIT_PIX_HOT
    PIX_PRNU_HIGH = BPM_BIT_PIX_PRNU_HIGH
    PIX_PRNU_LOW  = BPM_BIT_PIX_PRNU_LOW
    PIX_NONLIN    = BPM_BIT_PIX_NONLIN
    PIX_NOISE     = BPM_BIT_PIX_NOISE
    PIX_RTS       = BPM_BIT_PIX_RTS
    PIX_TRAP      = BPM_BIT_PIX_TRAP
    PIX_GLOW      = BPM_BIT_PIX_GLOW
    PIX_CLUSTER   = BPM_BIT_PIX_CLUSTER
    PIX_EXCLUDED  = BPM_BIT_PIX_EXCLUDED

    # Column defects
    COL_BAD      = BPM_BIT_COL_BAD
    COL_BRIGHT   = BPM_BIT_COL_BRIGHT
    COL_DARK     = BPM_BIT_COL_DARK
    COL_TRAP     = BPM_BIT_COL_TRAP
    COL_BLOCKED  = BPM_BIT_COL_BLOCKED
    COL_CTE_LOW  = BPM_BIT_COL_CTE_LOW
    COL_BOUNDARY = BPM_BIT_COL_BOUNDARY
    COL_EXCLUDED = BPM_BIT_COL_EXCLUDED

    # Row defects
    ROW_BAD      = BPM_BIT_ROW_BAD
    ROW_EXCLUDED = BPM_BIT_ROW_EXCLUDED

    # Transient / DQ
    SATURATED = BPM_BIT_SATURATED
    COSMIC    = BPM_BIT_COSMIC
    CROSSTALK = BPM_BIT_CROSSTALK
    BLOOM     = BPM_BIT_BLOOM
    OUTLIER   = BPM_BIT_OUTLIER

    def __str__(self) -> str:
        if self.value == 0:
            return "GOOD"
        return super().__str__().replace("DefectType.", "")

    def describe(self) -> str:
        """Return a detailed multi-line description of all set flags."""
        return describe_pixel_defects_verbose(self.value)


# =============================================================================
# Flag name and description tables
# =============================================================================

FLAG_NAMES: dict[int, str] = {
    BPM_BIT_PIX_BAD:       "PIX_BAD",
    BPM_BIT_PIX_DEAD:      "PIX_DEAD",
    BPM_BIT_PIX_BRIGHT:    "PIX_BRIGHT",
    BPM_BIT_PIX_HOT:       "PIX_HOT",
    BPM_BIT_PIX_PRNU_HIGH: "PIX_PRNU_HIGH",
    BPM_BIT_PIX_PRNU_LOW:  "PIX_PRNU_LOW",
    BPM_BIT_PIX_NONLIN:    "PIX_NONLIN",
    BPM_BIT_PIX_NOISE:     "PIX_NOISE",
    BPM_BIT_PIX_RTS:       "PIX_RTS",
    BPM_BIT_PIX_TRAP:      "PIX_TRAP",
    BPM_BIT_PIX_GLOW:      "PIX_GLOW",
    BPM_BIT_PIX_CLUSTER:   "PIX_CLUSTER",
    # Reserved bits 12–14 omitted until assigned.
    BPM_BIT_PIX_EXCLUDED:  "PIX_EXCLUDED",
    BPM_BIT_COL_BAD:       "COL_BAD",
    BPM_BIT_COL_BRIGHT:    "COL_BRIGHT",
    BPM_BIT_COL_DARK:      "COL_DARK",
    BPM_BIT_COL_TRAP:      "COL_TRAP",
    BPM_BIT_COL_BLOCKED:   "COL_BLOCKED",
    BPM_BIT_COL_CTE_LOW:   "COL_CTE_LOW",
    BPM_BIT_COL_BOUNDARY:  "COL_BOUNDARY",
    BPM_BIT_COL_EXCLUDED:  "COL_EXCLUDED",
    BPM_BIT_ROW_BAD:       "ROW_BAD",
    # BPM_BIT_ROW_RESERVED_25 omitted until assigned.
    BPM_BIT_ROW_EXCLUDED:  "ROW_EXCLUDED",
    BPM_BIT_SATURATED:     "SATURATED",
    BPM_BIT_COSMIC:        "COSMIC",
    BPM_BIT_CROSSTALK:     "CROSSTALK",
    BPM_BIT_BLOOM:         "BLOOM",
    BPM_BIT_OUTLIER:       "OUTLIER",
}

FLAG_DESCRIPTIONS: dict[int, str] = {
    BPM_BIT_PIX_BAD:       "General unusable pixel (summary flag)",
    BPM_BIT_PIX_DEAD:      "Unresponsive pixel (zero or near-zero signal)",
    BPM_BIT_PIX_BRIGHT:    "Anomalously high bias offset",
    BPM_BIT_PIX_HOT:       "Excessive dark current",
    BPM_BIT_PIX_PRNU_HIGH: "Quantum Efficiency significantly above average",
    BPM_BIT_PIX_PRNU_LOW:  "Quantum Efficiency significantly below average",
    BPM_BIT_PIX_NONLIN:    "Pixel response deviates from linear model",
    BPM_BIT_PIX_NOISE:     "Pixel read noise exceeds specification",
    BPM_BIT_PIX_RTS:       "Random Telegraph Signal (burst/popcorn noise)",
    BPM_BIT_PIX_TRAP:      "Individual pixel charge trap site",
    BPM_BIT_PIX_GLOW:      "Pixel affected by amplifier or circuitry luminescence",
    BPM_BIT_PIX_CLUSTER:   "Pixel is part of a spatially connected defect cluster",
    BPM_BIT_PIX_EXCLUDED:  "Pixel manually excluded from hardware metrics",
    BPM_BIT_COL_BAD:       "General unusable column (summary flag)",
    BPM_BIT_COL_BRIGHT:    "Column with excessive bright pixel population",
    BPM_BIT_COL_DARK:      "Column with excessive dark/dead pixel population",
    BPM_BIT_COL_TRAP:      "Column containing significant charge traps",
    BPM_BIT_COL_BLOCKED:   "Vertical charge transfer physically obstructed",
    BPM_BIT_COL_CTE_LOW:   "Sub-standard Charge Transfer Efficiency (high CTI)",
    BPM_BIT_COL_BOUNDARY:  "Column on physical segment/amplifier boundary",
    BPM_BIT_COL_EXCLUDED:  "Column excluded from measurement statistics",
    BPM_BIT_ROW_BAD:       "Entire row flagged as unusable",
    BPM_BIT_ROW_EXCLUDED:  "Row excluded from measurement statistics",
    BPM_BIT_SATURATED:     "Full-well capacity exceeded",
    BPM_BIT_COSMIC:        "High-energy particle hit (transient)",
    BPM_BIT_CROSSTALK:     "Signal from neighboring channel/pixel",
    BPM_BIT_BLOOM:         "Charge overflow from adjacent saturated pixels",
    BPM_BIT_OUTLIER:       "Statistical outlier detected by processing pipeline",
}

# Pre-sorted list for consistent iteration (ascending bit index).
_SORTED_FLAG_ITEMS: list[tuple[int, str]] = sorted(FLAG_NAMES.items())


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Pixel-level bit constants (0–15)
    "BPM_BIT_PIX_BAD",
    "BPM_BIT_PIX_DEAD",
    "BPM_BIT_PIX_BRIGHT",
    "BPM_BIT_PIX_HOT",
    "BPM_BIT_PIX_PRNU_HIGH",
    "BPM_BIT_PIX_PRNU_LOW",
    "BPM_BIT_PIX_NONLIN",
    "BPM_BIT_PIX_NOISE",
    "BPM_BIT_PIX_RTS",
    "BPM_BIT_PIX_TRAP",
    "BPM_BIT_PIX_GLOW",
    "BPM_BIT_PIX_CLUSTER",
    # BPM_BIT_PIX_RESERVED_12/13/14 and BPM_BIT_ROW_RESERVED_25 are
    # intentionally excluded until the bits are formally assigned.
    "BPM_BIT_PIX_EXCLUDED",
    # Structural bit constants (16–26)
    "BPM_BIT_COL_BAD",
    "BPM_BIT_COL_BRIGHT",
    "BPM_BIT_COL_DARK",
    "BPM_BIT_COL_TRAP",
    "BPM_BIT_COL_BLOCKED",
    "BPM_BIT_COL_CTE_LOW",
    "BPM_BIT_COL_BOUNDARY",
    "BPM_BIT_COL_EXCLUDED",
    "BPM_BIT_ROW_BAD",
    # BPM_BIT_ROW_RESERVED_25 intentionally excluded until assigned.
    "BPM_BIT_ROW_EXCLUDED",
    # Transient / DQ bit constants (27–31)
    "BPM_BIT_SATURATED",
    "BPM_BIT_COSMIC",
    "BPM_BIT_CROSSTALK",
    "BPM_BIT_BLOOM",
    "BPM_BIT_OUTLIER",
    # Combined masks
    "BPM_MASK_PIX",
    "BPM_MASK_COL",
    "BPM_MASK_ROW",
    "BPM_MASK_PERMANENT",
    "BPM_MASK_TRANSIENT",
    "BPM_MASK_HOT",
    "BPM_MASK_COLD",
    "BPM_MASK_NOISE",
    "BPM_MASK_TRANSPORT",
    "BPM_MASK_IGNORE",
    "BPM_MASK_BAD",
    "BPM_MASK_TRAPS",
    "BPM_MASK_RESPONSE",
    "BPM_MASK_ALL_16",
    "BPM_MASK_ALL",
    "BPM_MASK_ALL_32",
    # Types and documentation tables
    "DefectType",
    "DefectStatEntry",
    "FLAG_NAMES",
    "FLAG_DESCRIPTIONS",
    # Utility functions
    "bit_index",
    "iter_flag_definitions",
    "get_definitions_payload",
    "format_bit_comments",
    "format_pixel_status",
    "is_perfect_pixel",
    "is_usable_pixel",
    "has_defect",
    "get_defect_types",
    "describe_pixel_defects",
    "describe_pixel_defects_verbose",
    "count_defects",
    "combine_flags",
    "create_bpm",
    "create_master_bpm",
    "merge_bpms",
    "apply_bpm",
    "build_good_pixel_mask",
    "get_bpm_statistics",
    "validate_bpm",
    "summarize_bpm",
    "get_defect_mask",
    "filter_bpm",
    "to_boolean_mask",
    "check_bad_rows_cols",
    "flag_bad_rows_cols",
    "flag_defect_clusters",
    "get_cluster_statistics",
    "flag_excluded_rows",
    "flag_excluded_cols",
    "flag_excluded_rows_cols",
    "flag_excluded_pixels",
    "flag_excluded_border",
    "export_to_ds9_region",
]


# =============================================================================
# Bit index and definition helpers
# =============================================================================


def bit_index(bit_value: int, max_bits: int = 32) -> int:
    """Return the zero-based bit index for a single-bit flag value.

    Parameters
    ----------
    bit_value : int
        A power-of-two integer representing a single bit flag.
    max_bits : int, optional
        Maximum permitted bit index (exclusive). Default is 32.

    Returns
    -------
    int
        Zero-based bit index.

    Raises
    ------
    ValueError
        If *bit_value* is not a positive power of two, or if the
        resulting index is >= *max_bits*.
    """
    bit_value = int(bit_value)
    if bit_value <= 0 or (bit_value & (bit_value - 1)) != 0:
        raise ValueError(
            f"bit_value must be a single-bit power of two; got {bit_value}"
        )
    index = bit_value.bit_length() - 1
    if index >= max_bits:
        raise ValueError(
            f"bit_index {index} exceeds maximum of {max_bits - 1}"
        )
    return index


def iter_flag_definitions() -> Iterator[tuple[int, str, str]]:
    """Iterate over known defect bits in ascending bit-index order.

    Yields
    ------
    tuple of (int, str, str)
        ``(bit_value, name, description)`` for each known flag.
    """
    for bit_val in sorted(FLAG_NAMES.keys()):
        yield int(bit_val), FLAG_NAMES[bit_val], FLAG_DESCRIPTIONS.get(bit_val, "")


def get_definitions_payload(nbits: int = 32) -> dict[str, Any]:
    """Return a JSON-serializable payload of defect bit definitions.

    Parameters
    ----------
    nbits : int, optional
        Only include bits with index < *nbits*. Default is 32.

    Returns
    -------
    dict
        Keys: ``'schema'`` (int), ``'nbits'`` (int), ``'bits'`` (list of
        dicts with keys ``'bit'``, ``'value'``, ``'name'``, ``'desc'``),
        sorted by bit index.
    """
    bits = []
    for bit_val, name, desc in iter_flag_definitions():
        idx = bit_index(bit_val)
        if 0 <= idx < nbits:
            bits.append({
                "bit":   idx,
                "value": int(bit_val),
                "name":  str(name),
                "desc":  str(desc),
            })
    bits.sort(key=lambda d: d["bit"])
    return {
        "schema": _DEFINITIONS_SCHEMA_VERSION,
        "nbits":  int(nbits),
        "bits":   bits,
    }


def format_bit_comments(
    width: int = _FITS_COMMENT_WIDTH,
    prefix: str = "",
    include_hex: bool = True,
) -> list[str]:
    """Return FITS-header COMMENT strings documenting the defect bits.

    Each string maps directly to one FITS COMMENT card (one call to
    ``header['COMMENT'] = line``). Lines are wrapped at *width* characters
    so no card exceeds the 72-character FITS standard limit. Continuation
    lines are indented to align description text under the first character
    of the description on the opening line::

        Bit  0 (0x00000001) PIX_BAD      General unusable pixel (summary flag)
        Bit  1 (0x00000002) PIX_DEAD     Unresponsive pixel (zero or near-zero
                                         signal)

    Parameters
    ----------
    width : int, optional
        Maximum character width per output line. Default is 72 (FITS
        COMMENT card content limit, FITS Standard 4.0 §4.4.2).
    prefix : str, optional
        String prepended to every line. Its length is subtracted from the
        available wrapping width, preserving the 72-character guarantee.
        Example: ``prefix="BPM: "``.
    include_hex : bool, optional
        If True (default), include the 8-digit hexadecimal value
        ``(0xXXXXXXXX)`` in each entry.

    Returns
    -------
    list of str
        One header line followed by one or more lines per bit in ascending
        bit-index order. Every string is at most *width* characters long.

    Raises
    ------
    ValueError
        If *width* is not a positive integer, or too narrow to accommodate
        the fixed preamble for any bit entry.

    Examples
    --------
    Basic usage with astropy::

        from astropy.io.fits import Header
        hdr = Header()
        for line in format_bit_comments():
            hdr["COMMENT"] = line

    Namespaced for a shared extension header::

        lines = format_bit_comments(prefix="BPM: ", width=72)
    """
    if not isinstance(width, int) or width < 1:
        raise ValueError(f"width must be a positive integer; got {width!r}.")

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
            f"width={width} is too narrow: the preamble requires "
            f"{preamble_width} characters, leaving only {desc_width} for "
            "descriptions (minimum 8). Increase width or shorten prefix."
        )

    continuation_indent = " " * preamble_width

    def _make_preamble(idx: int, bit_val: int, name: str) -> str:
        if include_hex:
            return f"{prefix}Bit {idx:2d} (0x{bit_val:08X}) {name:<{name_col}s} "
        return f"{prefix}Bit {idx:2d} {name:<{name_col}s} "

    lines: list[str] = [
        f"{prefix}Bit flag definitions for BPM/DQ array (bitwise OR):"
    ]

    for bit_val, name, desc in iter_flag_definitions():
        idx      = bit_index(bit_val)
        preamble = _make_preamble(idx, bit_val, name)
        text     = desc or name

        wrapped = textwrap.wrap(
            text,
            width=width,
            initial_indent=preamble,
            subsequent_indent=continuation_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [preamble])

    return lines


def format_pixel_status(value: int) -> str:
    """Return a standardized string representation for logs.

    Parameters
    ----------
    value : int
        BPM pixel value.

    Returns
    -------
    str
        Flag name string, e.g. ``"PIX_HOT|PIX_NONLIN"``.
    """
    defects = get_defect_types(int(value))
    return "GOOD" if not defects else "|".join(defects)


# =============================================================================
# Pixel-level query utilities
# =============================================================================


def is_perfect_pixel(pixel_value: int) -> bool:
    """Return True if a pixel has no defect bits set.

    Parameters
    ----------
    pixel_value : int
        BPM value for a single pixel.

    Returns
    -------
    bool
    """
    return pixel_value == 0


def is_usable_pixel(pixel_value: int, ignore_transient: bool = True) -> bool:
    """Return True if a pixel is scientifically usable.

    Parameters
    ----------
    pixel_value : int
        BPM value for a single pixel.
    ignore_transient : bool, optional
        If True (default), ignore transient DQ flags and only check
        permanent hardware defects.

    Returns
    -------
    bool
    """
    mask = BPM_MASK_PERMANENT if ignore_transient else _MASK_ANY_DEFECT
    return (pixel_value & mask) == 0


def has_defect(pixel_value: int, defect_mask: int) -> bool:
    """Return True if a pixel value has any bit from *defect_mask* set.

    Parameters
    ----------
    pixel_value : int
        BPM value for a single pixel.
    defect_mask : int
        Specific bit (e.g. ``BPM_BIT_PIX_HOT``) or combined mask
        (e.g. ``BPM_MASK_COLD``) to test.

    Returns
    -------
    bool

    See Also
    --------
    get_defect_types : Return a list of all defect names for a pixel.
    DefectType : Enum-based inspection for complex logic.
    """
    return bool(pixel_value & defect_mask)


def get_defect_types(pixel_value: int) -> list[str]:
    """Return all defect type names present in a pixel value.

    Parameters
    ----------
    pixel_value : int
        32-bit BPM/DQ value for a pixel.

    Returns
    -------
    list of str
        Flag names in ascending bit order, e.g. ``['PIX_DEAD', 'PIX_NOISE']``.
        Empty list if no bits are set.

    Examples
    --------
    >>> get_defect_types(130)   # 128 (PIX_NOISE) + 2 (PIX_DEAD)
    ['PIX_DEAD', 'PIX_NOISE']
    """
    if not pixel_value:
        return []
    return [name for bit, name in _SORTED_FLAG_ITEMS if pixel_value & bit]


def describe_pixel_defects(bpm_value: int) -> str:
    """Return a comma-separated string of defect names for a pixel.

    Parameters
    ----------
    bpm_value : int
        32-bit BPM/DQ value for a single pixel.

    Returns
    -------
    str
        Comma-separated defect names, or ``"Good pixel"`` if value is 0.

    Examples
    --------
    >>> describe_pixel_defects(0)
    'Good pixel'
    >>> describe_pixel_defects(10)   # bits 1 and 3
    'PIX_DEAD, PIX_HOT'
    """
    if bpm_value == 0:
        return "Good pixel"
    defects = get_defect_types(bpm_value)
    if not defects:
        return f"Unknown defect (value: {bpm_value})"
    return ", ".join(defects)


def describe_pixel_defects_verbose(bpm_value: int) -> str:
    """Return a multi-line description of all defects present in a pixel.

    Parameters
    ----------
    bpm_value : int
        32-bit BPM/DQ value for a single pixel.

    Returns
    -------
    str
        Bulleted multi-line string, or ``"Good pixel (no defects detected)"``
        if value is 0.

    See Also
    --------
    describe_pixel_defects : Single-line summary version.
    """
    if bpm_value == 0:
        return "Good pixel (no defects detected)"

    lines = []
    for bit, name in _SORTED_FLAG_ITEMS:
        if bpm_value & bit:
            desc = FLAG_DESCRIPTIONS.get(bit, "No description available")
            lines.append(f"  • {name}: {desc}")

    if not lines:
        return f"Unknown defect pattern (value: 0x{bpm_value:08X})"

    return "\n".join(lines)


# =============================================================================
# Array-level counting and masking utilities
# =============================================================================


@njit
def _count_defects_numba(
    flat_bpm: npt.NDArray[np.unsignedinteger],
    mask: int,
) -> int:
    """Numba-accelerated defect counter (~10× faster for large arrays)."""
    count = 0
    for val in flat_bpm:
        if val & mask:
            count += 1
    return count


def count_defects(
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int | None = None,
) -> int:
    """Count pixels matching a defect criterion.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (typically uint32).
    defect_mask : int, optional
        Bit or group mask to test against (e.g. ``BPM_MASK_PERMANENT``).
        If None, counts any pixel where any bit is set.

    Returns
    -------
    int
        Number of pixels matching the criterion.

    Notes
    -----
    Uses Numba acceleration for arrays larger than 1 M pixels when numba
    is installed.
    """
    if defect_mask is None:
        return int(np.count_nonzero(bpm))
    if HAS_NUMBA and bpm.size > 1_000_000:
        return int(_count_defects_numba(bpm.ravel(), defect_mask))
    return int(np.count_nonzero(bpm & defect_mask))


def combine_flags(defect_flags: list[int]) -> int:
    """Combine a list of defect bits or masks into a single integer.

    Parameters
    ----------
    defect_flags : list of int
        Flags to combine, e.g. ``[BPM_BIT_PIX_HOT, BPM_BIT_PIX_DEAD]``.

    Returns
    -------
    int
        Bitwise OR of all input flags. Returns 0 for an empty list.
    """
    combined = 0
    for flag in defect_flags:
        combined |= int(flag)
    return combined


def get_defect_mask(
    bpm_array: npt.NDArray[np.unsignedinteger],
    defect_types: list[int] | None = None,
) -> np.ndarray:
    """Return a boolean mask selecting pixels with specific defect types.

    Parameters
    ----------
    bpm_array : np.ndarray
        32-bit BPM/DQ array.
    defect_types : list of int, optional
        Defect bits or group masks to include. If None, any non-zero
        pixel is treated as defective.

    Returns
    -------
    np.ndarray of bool
        True where the requested defect(s) are present.
    """
    if defect_types is None:
        return bpm_array != 0
    return (bpm_array & combine_flags(defect_types)) != 0


def filter_bpm(
    bpm_array: npt.NDArray[np.unsignedinteger],
    defect_types: list[int],
) -> np.ndarray:
    """Return a BPM array retaining only the requested defect bits.

    Unlike :func:`get_defect_mask`, this preserves the specific bit
    values for selected defects, clearing all others.

    Parameters
    ----------
    bpm_array : np.ndarray
        32-bit BPM/DQ array.
    defect_types : list of int
        Defect bits or group masks to retain.

    Returns
    -------
    np.ndarray
        Same shape and dtype as *bpm_array*, with unselected bits zeroed.
    """
    cm = np.asarray(combine_flags(defect_types), dtype=bpm_array.dtype)
    return bpm_array & cm


def to_boolean_mask(
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int = BPM_MASK_PERMANENT,
    invert: bool = False,
) -> np.ndarray:
    """Convert a BPM to a boolean mask suitable for NumPy indexing.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array.
    defect_mask : int, optional
        Which defects to include. Default is :data:`BPM_MASK_PERMANENT`.
    invert : bool, optional
        If True, return a mask of good pixels (True = usable).
        If False (default), return a mask of bad pixels (True = defective).

    Returns
    -------
    np.ndarray of bool

    Examples
    --------
    >>> good = to_boolean_mask(bpm, BPM_MASK_PERMANENT, invert=True)
    >>> median_good = np.median(data[good])
    """
    mask = (bpm & defect_mask) != 0
    return ~mask if invert else mask


# =============================================================================
# BPM creation and modification
# =============================================================================


def create_bpm(
    shape: tuple[int, ...],
    dtype: type = np.uint32,
) -> np.ndarray:
    """Create a zero-initialized bad pixel mask.

    Parameters
    ----------
    shape : tuple of int
        Array shape, e.g. ``(rows, cols)`` or ``(planes, rows, cols)``.
    dtype : dtype, optional
        Array data type. Default is ``np.uint32``, which accommodates all
        32 bits of the metrology hierarchy.

    Returns
    -------
    np.ndarray
        Zero-initialized array of the given shape and dtype.

    Notes
    -----
    Dtype capabilities:

    - ``uint8``  — bits 0–7 only (basic pixel defects; structural and DQ
      flags will be silently lost).
    - ``uint16`` — bits 0–15 (all pixel-level defects; structural/DQ bits
      ≥ 16 will be silently truncated).
    - ``uint32`` — bits 0–31 (full metrology suite; **recommended**).

    Always use ``uint32`` when per-image Data Quality extensions or
    structural flags are required.
    """
    if np.dtype(dtype).itemsize < 4:
        warnings.warn(
            f"dtype={dtype} cannot represent structural/DQ flags (bits >= 16). "
            "Consider dtype=np.uint32.",
            UserWarning,
            stacklevel=2,
        )
    return np.zeros(shape, dtype=dtype)


def create_master_bpm(
    shape: tuple[int, int],
    dead_pixels: np.ndarray | None = None,
    hot_pixels: np.ndarray | None = None,
    bad_columns: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience constructor for master calibration BPMs.

    Parameters
    ----------
    shape : tuple of int
        Sensor dimensions as ``(nrows, ncols)``.
    dead_pixels : np.ndarray of shape (N, 2), optional
        ``(row, col)`` coordinates of dead pixels.
    hot_pixels : np.ndarray of shape (M, 2), optional
        ``(row, col)`` coordinates of hot pixels.
    bad_columns : array-like of int, optional
        Column indices to mark as :data:`BPM_BIT_COL_BAD`.

    Returns
    -------
    np.ndarray
        Initialized BPM with the specified defects marked.
    """
    bpm = create_bpm(shape)

    if dead_pixels is not None:
        dead = np.asarray(dead_pixels)
        bpm[dead[:, 0], dead[:, 1]] |= BPM_BIT_PIX_DEAD

    if hot_pixels is not None:
        hot = np.asarray(hot_pixels)
        bpm[hot[:, 0], hot[:, 1]] |= BPM_BIT_PIX_HOT

    if bad_columns is not None:
        bpm[:, bad_columns] |= BPM_BIT_COL_BAD

    return bpm


def merge_bpms(
    bpm1: npt.NDArray[np.unsignedinteger],
    bpm2: npt.NDArray[np.unsignedinteger],
    operation: str = "or",
) -> np.ndarray:
    """Merge two bad pixel masks using a bitwise operation.

    Parameters
    ----------
    bpm1 : np.ndarray
        Base mask.
    bpm2 : np.ndarray
        Update mask. Must have the same shape as *bpm1*.
    operation : {"or", "and", "xor", "diff"}, optional
        Merge strategy:

        - ``"or"``  — union; accumulate flags from both masks.
        - ``"and"`` — intersection; keep only flags in both.
        - ``"xor"`` — symmetric difference; highlight changes.
        - ``"diff"`` — subtraction; ``bpm1 & ~bpm2``.

        Default is ``"or"``.

    Returns
    -------
    np.ndarray
        Merged uint32 BPM array.

    Raises
    ------
    ValueError
        If shapes do not match or *operation* is not recognized.

    Notes
    -----
    The ``"diff"`` operation is useful for removing transient artifacts
    from a persistent master mask or isolating hardware-only defects.
    """
    bpm1 = np.asarray(bpm1, dtype=np.uint32)
    bpm2 = np.asarray(bpm2, dtype=np.uint32)

    if bpm1.shape != bpm2.shape:
        raise ValueError(
            f"BPM shapes must match: {bpm1.shape} != {bpm2.shape}"
        )

    op = operation.lower()
    if op == "or":
        return bpm1 | bpm2
    if op == "and":
        return bpm1 & bpm2
    if op == "xor":
        return bpm1 ^ bpm2
    if op == "diff":
        return bpm1 & ~bpm2
    raise ValueError(f"Invalid operation: {operation!r}.")



def _fill_output_dtype(data_dtype: np.dtype, fill_value: float) -> np.dtype:
    """Choose the output dtype for filling *data_dtype* with *fill_value*.

    The check is by value, not by type: NaN or any in-range float fits a
    float array of any precision (float32 images from FITS are the common
    case), and an integer fits an integer array if it is within range. A
    non-integral float in an integer array promotes the output to float64.
    """
    fv = np.asarray(fill_value)
    if np.issubdtype(data_dtype, np.floating):
        if fv.dtype.kind not in "biuf":
            raise ValueError(f"fill_value {fill_value!r} is not numeric")
        value = float(fv)
        if np.isfinite(value) and abs(value) > float(np.finfo(data_dtype).max):
            raise ValueError(
                f"fill_value {fill_value!r} is out of range for {data_dtype}."
            )
        return data_dtype
    if np.issubdtype(data_dtype, np.integer):
        value = float(fv)
        if fv.dtype.kind == "f" and not (np.isfinite(value) and value.is_integer()):
            return np.dtype(np.float64)
        info = np.iinfo(data_dtype)
        if not info.min <= value <= info.max:
            raise ValueError(
                f"fill_value {fill_value!r} cannot be represented as "
                f"{data_dtype}. Consider fill_value=0 or promoting data "
                "to float."
            )
        return data_dtype
    return np.dtype(np.float64)


def apply_bpm(
    data: npt.NDArray[np.number],
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int = BPM_MASK_PERMANENT,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Apply a bad pixel mask to a data array.

    Flagged pixels are replaced with *fill_value* in a copy of *data*.

    Parameters
    ----------
    data : np.ndarray
        Input image data.
    bpm : np.ndarray
        32-bit bad pixel mask. Must have the same shape as *data*.
    defect_mask : int, optional
        Which bits trigger masking. Default is :data:`BPM_MASK_PERMANENT`.
    fill_value : float, optional
        Replacement value for bad pixels. Default is ``np.nan``. If NaN
        and *data* is integer, the output is promoted to float64.

    Returns
    -------
    np.ndarray
        Copy of *data* with flagged pixels replaced.

    Raises
    ------
    ValueError
        If shapes do not match or *fill_value* cannot be safely cast to
        the output dtype.
    """
    if data.shape != bpm.shape:
        raise ValueError(
            f"Data and BPM shapes must match: {data.shape} != {bpm.shape}"
        )

    out_dtype = _fill_output_dtype(data.dtype, fill_value)

    result = data.astype(out_dtype, copy=True)
    result[(bpm & defect_mask) != 0] = fill_value
    return result


def build_good_pixel_mask(
    data: np.ndarray,
    bad_pixel_mask: np.ndarray | None = None,
    exclude_flags: int = BPM_MASK_PERMANENT,
    exclude_nan: bool = True,
    exclude_inf: bool = False,
    exclude_negative: bool = False,
    exclude_zero: bool = False,
    custom_condition: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Build a boolean mask of pixels suitable for analysis.

    Returns a boolean array where True = good (usable) pixel. All
    specified conditions are combined with logical AND, so a pixel must
    pass every criterion to be marked good.

    Parameters
    ----------
    data : np.ndarray
        Input image data (any shape, typically 2-D).
    bad_pixel_mask : np.ndarray, optional
        BPM array. If integer dtype, bitwise AND with *exclude_flags* is
        used. If boolean dtype, True = bad pixel. If None, only
        data-value conditions are applied.
    exclude_flags : int, optional
        Bitmask flags to exclude when *bad_pixel_mask* is integer.
        Default is :data:`BPM_MASK_PERMANENT`.
    exclude_nan : bool, optional
        Exclude NaN values. Default is True.
    exclude_inf : bool, optional
        Exclude Inf values. Default is False.
    exclude_negative : bool, optional
        Exclude negative values. Default is False.
    exclude_zero : bool, optional
        Exclude exact zero values. Default is False.
    custom_condition : callable, optional
        Function ``f(data) -> bool array`` where True = good pixel.
        Applied in addition to other conditions.

    Returns
    -------
    np.ndarray of bool
        Same shape as *data*. True = pixel is good.

    Raises
    ------
    ValueError
        If *bad_pixel_mask* shape does not match *data*, or if
        *custom_condition* returns a mismatched shape.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, np.nan, 4.0])
    >>> build_good_pixel_mask(data)
    array([ True,  True, False,  True])

    >>> # Exclude only dead pixels, keep saturated
    >>> good = build_good_pixel_mask(data, bpm, BPM_BIT_PIX_DEAD)

    See Also
    --------
    apply_bpm : Replace bad pixels with a fill value.
    to_boolean_mask : Convert BPM to boolean without data-value checks.
    """
    good_mask = np.ones(data.shape, dtype=bool)

    if bad_pixel_mask is not None:
        if bad_pixel_mask.shape != data.shape:
            raise ValueError(
                f"BPM shape {bad_pixel_mask.shape} does not match "
                f"data shape {data.shape}."
            )
        if np.issubdtype(bad_pixel_mask.dtype, np.integer):
            bad = (bad_pixel_mask & int(exclude_flags)) != 0
            good_mask &= ~bad
            logger.debug(
                "BPM: %d/%d pixels flagged with mask 0x%X",
                int(np.sum(bad)), data.size, int(exclude_flags),
            )
        else:
            good_mask &= ~bad_pixel_mask.astype(bool)
            logger.debug(
                "Boolean BPM: %d/%d pixels marked bad",
                int(np.sum(bad_pixel_mask)), data.size,
            )

    if exclude_nan:
        nan_mask = ~np.isnan(data)
        logger.debug("Excluding %d NaN pixel(s).", int(np.sum(~nan_mask)))
        good_mask &= nan_mask

    if exclude_inf:
        inf_mask = ~np.isinf(data)
        logger.debug("Excluding %d Inf pixel(s).", int(np.sum(~inf_mask)))
        good_mask &= inf_mask

    if exclude_negative:
        pos_mask = data >= 0
        logger.debug("Excluding %d negative pixel(s).", int(np.sum(~pos_mask)))
        good_mask &= pos_mask

    if exclude_zero:
        nonzero_mask = data != 0
        logger.debug("Excluding %d zero pixel(s).", int(np.sum(~nonzero_mask)))
        good_mask &= nonzero_mask

    if custom_condition is not None:
        try:
            custom_mask = np.asarray(custom_condition(data), dtype=bool)
            if custom_mask.shape != data.shape:
                raise ValueError(
                    f"Custom condition returned shape {custom_mask.shape}, "
                    f"expected {data.shape}."
                )
            logger.debug(
                "Custom condition excluded %d pixel(s).",
                int(np.sum(~custom_mask)),
            )
            good_mask &= custom_mask
        except Exception:
            logger.error("Custom condition raised an exception.", exc_info=True)
            raise

    n_good = int(np.sum(good_mask))
    n_bad  = data.size - n_good
    if n_bad > 0:
        logger.debug(
            "Pixel mask: %d/%d good (%.1f%%), %d excluded (%.1f%%)",
            n_good, data.size, 100.0 * n_good / data.size,
            n_bad, 100.0 * n_bad / data.size,
        )

    return good_mask


# =============================================================================
# Statistics and reporting
# =============================================================================


def get_bpm_statistics(
    bpm_array: npt.NDArray[np.unsignedinteger],
    force_sequential: bool = False,
) -> dict[str, DefectStatEntry]:
    """Count the occurrence and pixel fraction of every known defect type.

    Parameters
    ----------
    bpm_array : np.ndarray
        Bad pixel mask array (unsigned integer dtype).
    force_sequential : bool, optional
        If True, disable vectorized batch processing even for arrays
        within the size threshold. Useful for memory-constrained
        environments. Default is False.

    Returns
    -------
    dict of str to DefectStatEntry
        Keys are individual flag names (e.g. ``'PIX_DEAD'``) and group
        mask names (e.g. ``'MASK_PERMANENT'``). Each value has ``count``,
        ``fraction``, and ``type`` fields.

    Raises
    ------
    TypeError
        If *bpm_array* is not an unsigned integer dtype.
    ValueError
        If *bpm_array* is empty.
    """
    if not np.issubdtype(bpm_array.dtype, np.unsignedinteger):
        raise TypeError(
            f"bpm_array must be unsigned integer (e.g. uint32); "
            f"got {bpm_array.dtype!r}."
        )

    if bpm_array.ndim != 2:
        raise ValueError(
            f"bpm_array must be 2-D; got shape {bpm_array.shape}."
        )

    flat = np.asarray(bpm_array, dtype=np.uint32).ravel()
    total_pixels: int = bpm_array.size
    if total_pixels == 0:
        raise ValueError("bpm_array is empty (size == 0).")

    inv_total = 1.0 / total_pixels
    use_batch = (total_pixels <= _BPM_STATS_PIXEL_LIMIT) and not force_sequential

    if use_batch and total_pixels > 16_000_000:
        n_masks = len(FLAG_NAMES) + len(_GROUP_MASKS)
        est_mb  = (total_pixels * n_masks) / (1024 ** 2)
        logger.debug(
            "Batch statistics: ~%.1f MB temporary memory "
            "(%d pixels × %d masks).",
            est_mb, total_pixels, n_masks,
        )

    result: dict[str, DefectStatEntry] = {}

    # Individual-bit statistics
    ind_names = list(FLAG_NAMES.values())
    ind_bits  = np.array(list(FLAG_NAMES.keys()), dtype=np.uint32)

    if use_batch:
        ind_counts = np.count_nonzero(
            flat[:, None] & ind_bits[None, :], axis=0
        )
    else:
        ind_counts = np.array(
            [np.count_nonzero(flat & b) for b in ind_bits], dtype=np.intp
        )

    for name, count in zip(ind_names, ind_counts, strict=False):
        result[name] = DefectStatEntry(
            count=int(count),
            fraction=float(count) * inv_total,
            type="Individual Bit",
        )

    # Group-mask statistics
    grp_names  = list(_GROUP_MASKS.keys())
    grp_values = np.array(list(_GROUP_MASKS.values()), dtype=np.uint32)

    if use_batch:
        grp_counts = np.count_nonzero(
            flat[:, None] & grp_values[None, :], axis=0
        )
    else:
        grp_counts = np.array(
            [np.count_nonzero(flat & g) for g in grp_values], dtype=np.intp
        )

    for name, count in zip(grp_names, grp_counts, strict=False):
        result[name] = DefectStatEntry(
            count=int(count),
            fraction=float(count) * inv_total,
            type="Group Mask",
        )

    return result


def validate_bpm(
    bpm: npt.NDArray[np.unsignedinteger],
    max_defect_fraction: float = 0.20,
    max_bad_rows: int = 10,
    max_bad_cols: int = 10,
) -> tuple[bool, list[str]]:
    """Validate a BPM against sensor acceptance criteria.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D, unsigned integer).
    max_defect_fraction : float, optional
        Maximum acceptable fraction of permanently defective pixels.
        Default is 0.20.
    max_bad_rows : int, optional
        Maximum number of fully flagged bad rows. Default is 10.
    max_bad_cols : int, optional
        Maximum number of fully flagged bad columns. Default is 10.

    Returns
    -------
    is_valid : bool
        True if all criteria are met.
    messages : list of str
        Diagnostic messages for each failed criterion.

    Notes
    -----
    The bad-row and bad-column counts are derived by dividing the total
    ``ROW_BAD`` / ``COL_BAD`` pixel count by the array width/height.
    This heuristic is accurate only when the full row or column has been
    flagged; partially-flagged rows/columns will produce a fractional
    count that is rounded down.
    """
    messages: list[str] = []
    bpm_stats = get_bpm_statistics(bpm)

    perm_frac = bpm_stats["MASK_PERMANENT"]["fraction"]
    if perm_frac > max_defect_fraction:
        messages.append(
            f"Permanent defect fraction {perm_frac:.2%} exceeds "
            f"threshold {max_defect_fraction:.2%}."
        )

    bad_rows, bad_cols = check_bad_rows_cols(bpm)
    n_bad_rows = int(np.sum(bad_rows))
    n_bad_cols = int(np.sum(bad_cols))

    if n_bad_rows > max_bad_rows:
        messages.append(
            f"{n_bad_rows} bad row(s) exceeds limit of {max_bad_rows}."
        )
    if n_bad_cols > max_bad_cols:
        messages.append(
            f"{n_bad_cols} bad column(s) exceeds limit of {max_bad_cols}."
        )

    return len(messages) == 0, messages


def summarize_bpm(
    bpm: npt.NDArray[np.unsignedinteger],
) -> dict[str, Any]:
    """Generate a comprehensive BPM quality report.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D, unsigned integer).

    Returns
    -------
    dict
        Keys: ``'total_pixels'``, ``'defective_pixels'``,
        ``'yield_percent'``, ``'worst_row'``, ``'worst_row_count'``,
        ``'worst_col'``, ``'worst_col_count'``, ``'statistics'``
        (dict of :class:`DefectStatEntry`).

    Raises
    ------
    ValueError
        If *bpm* is not 2-D.
    """
    if bpm.ndim != 2:
        raise ValueError(
            f"summarize_bpm expects a 2-D BPM; got shape {bpm.shape}."
        )

    bpm_stats      = get_bpm_statistics(bpm)
    total_defective = int(np.count_nonzero(bpm != 0))
    yield_pct       = 100.0 * (1 - total_defective / bpm.size)

    row_counts = np.count_nonzero(bpm != 0, axis=1)
    col_counts = np.count_nonzero(bpm != 0, axis=0)

    report = {
        "total_pixels":    bpm.size,
        "defective_pixels": total_defective,
        "yield_percent":   float(yield_pct),
        "worst_row":       int(np.argmax(row_counts)),
        "worst_row_count": int(row_counts.max()),
        "worst_col":       int(np.argmax(col_counts)),
        "worst_col_count": int(col_counts.max()),
        "statistics":      bpm_stats,
    }

    logger.debug("BPM Quality Report:")
    logger.debug("  Array shape:   %s", bpm.shape)
    logger.debug("  Pixel yield:   %.2f%%", yield_pct)
    logger.debug("  Total defects: %d", total_defective)
    logger.debug(
        "  Worst row: %d (%d defects)",
        report["worst_row"], report["worst_row_count"],
    )
    logger.debug(
        "  Worst col: %d (%d defects)",
        report["worst_col"], report["worst_col_count"],
    )

    return report


# =============================================================================
# Row / column / pixel exclusion flags
# =============================================================================


def flag_excluded_rows(
    bpm: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Return a copy of *bpm* with specified rows marked as ROW_EXCLUDED.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D).
    rows : array-like of int
        Row indices to exclude.

    Returns
    -------
    np.ndarray
        Copy of *bpm* with :data:`BPM_BIT_ROW_EXCLUDED` set on all
        pixels in the specified rows.
    """
    return flag_excluded_rows_cols(bpm, rows=rows, cols=None)


def flag_excluded_cols(
    bpm: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Return a copy of *bpm* with specified columns marked as COL_EXCLUDED.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D).
    cols : array-like of int
        Column indices to exclude.

    Returns
    -------
    np.ndarray
        Copy of *bpm* with :data:`BPM_BIT_COL_EXCLUDED` set on all
        pixels in the specified columns.
    """
    return flag_excluded_rows_cols(bpm, rows=None, cols=cols)


def flag_excluded_rows_cols(
    bpm: np.ndarray,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
) -> np.ndarray:
    """Return a copy of *bpm* with specified rows and/or columns marked EXCLUDED.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D).
    rows : array-like of int, optional
        Row indices to exclude. :data:`BPM_BIT_ROW_EXCLUDED` is set on
        all pixels in each row.
    cols : array-like of int, optional
        Column indices to exclude. :data:`BPM_BIT_COL_EXCLUDED` is set
        on all pixels in each column.

    Returns
    -------
    np.ndarray
        Copy of *bpm* with exclusion bits applied.

    Raises
    ------
    ValueError
        If *bpm* is not 2-D.
    """
    if bpm.ndim != 2:
        raise ValueError(
            f"flag_excluded_rows_cols expects a 2-D BPM; got {bpm.shape}."
        )
    new_bpm = bpm.copy()
    if rows is not None:
        new_bpm[np.asarray(rows, dtype=int), :] |= BPM_BIT_ROW_EXCLUDED
    if cols is not None:
        new_bpm[:, np.asarray(cols, dtype=int)] |= BPM_BIT_COL_EXCLUDED
    return new_bpm


def flag_excluded_pixels(
    bpm: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Return a copy of *bpm* with specific pixels marked as PIX_EXCLUDED.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D).
    coords : array-like of shape (N, 2)
        Integer ``(row, col)`` coordinates. :data:`BPM_BIT_PIX_EXCLUDED`
        is set on each specified pixel.

    Returns
    -------
    np.ndarray
        Copy of *bpm* with pixel exclusion bits applied.

    Raises
    ------
    ValueError
        If *bpm* is not 2-D, or if *coords* does not have shape (N, 2).
    """
    if bpm.ndim != 2:
        raise ValueError(
            f"flag_excluded_pixels expects a 2-D BPM; got {bpm.shape}."
        )
    coords = np.asarray(coords, dtype=int)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"coords must have shape (N, 2); got {coords.shape}."
        )
    new_bpm = bpm.copy()
    new_bpm[coords[:, 0], coords[:, 1]] |= BPM_BIT_PIX_EXCLUDED
    return new_bpm


def flag_excluded_border(
    bpm: np.ndarray,
    nrows: int = 0,
    ncols: int = 0,
) -> np.ndarray:
    """Mark a border region of the array as EXCLUDED.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D).
    nrows : int, optional
        Number of rows to exclude at each vertical edge (top and bottom).
        Default is 0.
    ncols : int, optional
        Number of columns to exclude at each horizontal edge (left and
        right). Default is 0.

    Returns
    -------
    np.ndarray
        Copy of *bpm* with :data:`BPM_BIT_ROW_EXCLUDED`,
        :data:`BPM_BIT_COL_EXCLUDED`, and :data:`BPM_BIT_PIX_EXCLUDED`
        set on the border region.

    Raises
    ------
    ValueError
        If *bpm* is not 2-D.
    """
    if bpm.ndim != 2:
        raise ValueError(
            f"flag_excluded_border expects a 2-D BPM; got {bpm.shape}."
        )

    ny, nx  = bpm.shape
    new_bpm = bpm.copy()

    rows_to_exclude: np.ndarray = np.array([], dtype=int)
    cols_to_exclude: np.ndarray = np.array([], dtype=int)

    if nrows > 0:
        rows_to_exclude = np.unique(np.concatenate([
            np.arange(0, min(nrows, ny)),
            np.arange(max(0, ny - nrows), ny),
        ]))

    if ncols > 0:
        cols_to_exclude = np.unique(np.concatenate([
            np.arange(0, min(ncols, nx)),
            np.arange(max(0, nx - ncols), nx),
        ]))

    if rows_to_exclude.size > 0:
        new_bpm[rows_to_exclude, :] |= BPM_BIT_ROW_EXCLUDED | BPM_BIT_PIX_EXCLUDED
    if cols_to_exclude.size > 0:
        new_bpm[:, cols_to_exclude] |= BPM_BIT_COL_EXCLUDED | BPM_BIT_PIX_EXCLUDED

    return new_bpm


# =============================================================================
# Structural defect promotion
# =============================================================================


def check_bad_rows_cols(
    bpm_array: npt.NDArray[np.unsignedinteger],
    row_threshold: float = 0.1,
    col_threshold: float = 0.1,
    defect_mask: int = BPM_MASK_PERMANENT,
    active_region: tuple[slice, slice] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify rows and columns exceeding a defect density threshold.

    Parameters
    ----------
    bpm_array : np.ndarray
        2-D bad pixel mask array.
    row_threshold : float, optional
        Fraction of bad pixels in a row required to flag it. Default 0.1.
    col_threshold : float, optional
        Fraction of bad pixels in a column required to flag it.
        Default 0.1.
    defect_mask : int, optional
        Which bits count as bad. Default is :data:`BPM_MASK_PERMANENT`.
    active_region : tuple of slice, optional
        ``(row_slice, col_slice)`` to restrict the check (e.g. to exclude
        overscan). If None, the full array is used.

    Returns
    -------
    bad_rows : np.ndarray of bool
        Length ``bpm_array.shape[0]``. True = row exceeds threshold.
    bad_cols : np.ndarray of bool
        Length ``bpm_array.shape[1]``. True = column exceeds threshold.

    Raises
    ------
    ValueError
        If *bpm_array* is not 2-D, or if thresholds are outside [0, 1].
    TypeError
        If *active_region* is not a tuple of two slices.
    """
    if bpm_array.ndim != 2:
        raise ValueError(
            "check_bad_rows_cols expects a 2-D BPM array."
        )
    for pname, pval in (
        ("row_threshold", row_threshold),
        ("col_threshold", col_threshold),
    ):
        if not 0.0 <= pval <= 1.0:
            raise ValueError(
                f"{pname} must be in [0, 1]; got {pval!r}."
            )

    if active_region is None:
        active_region = (slice(None), slice(None))

    row_slice, col_slice = active_region
    if not isinstance(row_slice, slice) or not isinstance(col_slice, slice):
        raise TypeError(
            "active_region must be a tuple of two slice objects; "
            f"got ({type(row_slice).__name__}, {type(col_slice).__name__})."
        )

    check_mask = (bpm_array[row_slice, col_slice] & defect_mask) != 0
    bad_rows_active = np.mean(check_mask, axis=1) > row_threshold
    bad_cols_active = np.mean(check_mask, axis=0) > col_threshold

    bad_rows = np.zeros(bpm_array.shape[0], dtype=bool)
    bad_cols = np.zeros(bpm_array.shape[1], dtype=bool)

    row_indices = np.arange(*row_slice.indices(bpm_array.shape[0]))
    col_indices = np.arange(*col_slice.indices(bpm_array.shape[1]))

    bad_rows[row_indices] = bad_rows_active
    bad_cols[col_indices] = bad_cols_active

    return bad_rows, bad_cols


def flag_bad_rows_cols(
    bpm_array: npt.NDArray[np.unsignedinteger],
    row_threshold: float = 0.1,
    col_threshold: float = 0.1,
    defect_mask: int = BPM_MASK_PERMANENT,
    active_region: tuple[slice, slice] | None = None,
) -> np.ndarray:
    """Promote high-density pixel defects to structural ROW_BAD/COL_BAD flags.

    Parameters
    ----------
    bpm_array : np.ndarray
        2-D bad pixel mask array.
    row_threshold : float, optional
        Defect fraction threshold for rows. Default is 0.1.
    col_threshold : float, optional
        Defect fraction threshold for columns. Default is 0.1.
    defect_mask : int, optional
        Which bits count as bad. Default is :data:`BPM_MASK_PERMANENT`.
    active_region : tuple of slice, optional
        ``(row_slice, col_slice)`` to restrict the check.

    Returns
    -------
    np.ndarray
        Copy of *bpm_array* with :data:`BPM_BIT_ROW_BAD` and/or
        :data:`BPM_BIT_COL_BAD` set on qualifying rows/columns.
    """
    bad_rows, bad_cols = check_bad_rows_cols(
        bpm_array, row_threshold, col_threshold, defect_mask, active_region
    )
    new_bpm = bpm_array.copy()

    if np.any(bad_rows):
        new_bpm[bad_rows, :] |= BPM_BIT_ROW_BAD
    if np.any(bad_cols):
        new_bpm[:, bad_cols] |= BPM_BIT_COL_BAD

    n_rows = int(np.sum(bad_rows))
    n_cols = int(np.sum(bad_cols))
    if n_rows > 0 or n_cols > 0:
        logger.debug(
            "Flagged %d row(s) and %d column(s) as structurally bad.",
            n_rows, n_cols,
        )
    return new_bpm


# =============================================================================
# Cluster detection
# =============================================================================


def flag_defect_clusters(
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int = BPM_MASK_PERMANENT,
    min_size: int = 2,
    connectivity: int = 2,
    also_flag_defect: bool = False,
) -> np.ndarray:
    """Identify and flag spatially connected defect clusters.

    Sets :data:`BPM_BIT_PIX_CLUSTER` on pixels that are part of a
    connected defect cluster meeting the size threshold.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D, unsigned integer).
    defect_mask : int, optional
        Which defect bits to consider. Default is :data:`BPM_MASK_PERMANENT`.
    min_size : int, optional
        Minimum connected-component size to qualify as a cluster.
        Default is 2 (isolated single pixels are not marked).
    connectivity : {1, 2}, optional
        Pixel connectivity: 1 = 4-connected, 2 = 8-connected (default).
    also_flag_defect : bool, optional
        If True, also set :data:`BPM_BIT_PIX_BAD` on cluster pixels that
        have no other defect bits set. Default is False.

    Returns
    -------
    np.ndarray
        Copy of *bpm* with :data:`BPM_BIT_PIX_CLUSTER` set on qualifying
        pixels.

    Raises
    ------
    ValueError
        If *bpm* is not 2-D, *min_size* < 1, or *connectivity* not in
        {1, 2}.
    ImportError
        If scipy is not available.
    """
    try:
        from scipy import ndimage
    except ImportError as e:
        raise ImportError(
            "scipy is required for cluster detection. "
            "Install with: pip install scipy"
        ) from e

    if bpm.ndim != 2:
        raise ValueError(
            f"flag_defect_clusters expects a 2-D BPM; got {bpm.shape}."
        )
    if not isinstance(min_size, int) or min_size < 1:
        raise ValueError(
            f"min_size must be a positive integer; got {min_size!r}."
        )
    if connectivity not in (1, 2):
        raise ValueError(
            f"connectivity must be 1 or 2; got {connectivity!r}."
        )

    defect_pixels = (bpm & defect_mask) != 0
    if not np.any(defect_pixels):
        logger.debug("No defects match defect_mask; returning unchanged BPM.")
        return bpm.copy()

    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled_array, num_features = ndimage.label(defect_pixels, structure=structure)

    if num_features == 0:
        return bpm.copy()

    logger.debug(
        "%d connected component(s) before size filtering.", num_features
    )

    component_sizes = np.bincount(labeled_array.ravel())
    valid_ids = np.where(component_sizes >= min_size)[0]
    valid_ids = valid_ids[valid_ids > 0]  # exclude background label 0

    if valid_ids.size == 0:
        logger.debug(
            "No clusters >= %d pixel(s); returning unchanged BPM.", min_size
        )
        return bpm.copy()

    cluster_lut = np.zeros(num_features + 1, dtype=bool)
    cluster_lut[valid_ids] = True
    pixels_in_clusters = cluster_lut[labeled_array]

    new_bpm = bpm.copy()
    new_bpm[pixels_in_clusters] |= BPM_BIT_PIX_CLUSTER

    if also_flag_defect:
        # Check the original BPM (before PIX_CLUSTER was set) to find
        # cluster pixels that had no prior defect flags.
        had_no_defects = pixels_in_clusters & (
            (bpm & _MASK_ANY_DEFECT) == 0
        )
        if np.any(had_no_defects):
            new_bpm[had_no_defects] |= BPM_BIT_PIX_BAD

    logger.info(
        "Flagged %d pixel(s) as PIX_CLUSTER (%d cluster(s) >= %d pixel(s)).",
        int(np.count_nonzero(pixels_in_clusters)),
        int(valid_ids.size),
        min_size,
    )
    return new_bpm


def get_cluster_statistics(
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int = BPM_MASK_PERMANENT,
    connectivity: int = 2,
) -> dict[str, Any]:
    """Analyse defect cluster properties without modifying the BPM.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask array (2-D, unsigned integer).
    defect_mask : int, optional
        Which defect bits to analyse. Default is :data:`BPM_MASK_PERMANENT`.
    connectivity : {1, 2}, optional
        Pixel connectivity: 1 = 4-connected, 2 = 8-connected (default).

    Returns
    -------
    dict
        Keys: ``'num_clusters'``, ``'num_clustered_pixels'``,
        ``'cluster_sizes'`` (ndarray, descending), ``'largest_cluster_size'``,
        ``'mean_cluster_size'``, ``'median_cluster_size'``,
        ``'isolated_defects'``.

    Raises
    ------
    ValueError
        If *bpm* is not 2-D.
    ImportError
        If scipy is not available.
    """
    try:
        from scipy import ndimage
    except ImportError as e:
        raise ImportError(
            "scipy is required for cluster analysis. "
            "Install with: pip install scipy"
        ) from e

    if bpm.ndim != 2:
        raise ValueError(
            f"get_cluster_statistics expects a 2-D BPM; got {bpm.shape}."
        )

    defect_pixels = (bpm & defect_mask) != 0
    empty_result: dict[str, Any] = {
        "num_clusters":        0,
        "num_clustered_pixels": 0,
        "cluster_sizes":       np.array([], dtype=int),
        "largest_cluster_size": 0,
        "mean_cluster_size":   0.0,
        "median_cluster_size": 0.0,
        "isolated_defects":    0,
    }

    if not np.any(defect_pixels):
        return empty_result

    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled_array, num_features = ndimage.label(
        defect_pixels, structure=structure
    )

    if num_features == 0:
        empty_result["num_clustered_pixels"] = int(np.sum(defect_pixels))
        empty_result["isolated_defects"]     = int(np.sum(defect_pixels))
        return empty_result

    component_sizes = np.bincount(labeled_array.ravel())[1:]  # skip background
    cluster_sizes   = np.sort(component_sizes)[::-1]
    isolated        = int(np.sum(cluster_sizes == 1))

    return {
        "num_clusters":        int(num_features),
        "num_clustered_pixels": int(np.sum(cluster_sizes)),
        "cluster_sizes":       cluster_sizes,
        "largest_cluster_size": int(cluster_sizes[0]) if cluster_sizes.size else 0,
        "mean_cluster_size":   float(np.mean(cluster_sizes)) if cluster_sizes.size else 0.0,
        "median_cluster_size": float(np.median(cluster_sizes)) if cluster_sizes.size else 0.0,
        "isolated_defects":    isolated,
    }


# =============================================================================
# DS9 region export
# =============================================================================


def export_to_ds9_region(
    bpm: npt.NDArray[np.unsignedinteger],
    defect_mask: int,
    filename: StrOrPath | TextIO,
    region_type: str = "box",
    color: str = "red",
    coord_system: str = "physical",
    origin: int = 1,
    radius: float = 0.5,
    box_size: int = 1,
    chunk_size: int = 10000,
) -> None:
    """Export flagged pixels as a DS9 region file for visualization.

    Parameters
    ----------
    bpm : np.ndarray
        2-D bad pixel mask array.
    defect_mask : int
        Selects which defects to export. Any pixel with
        ``(bpm & defect_mask) != 0`` produces a region entry.
    filename : str, Path, or file-like
        Output target. A path (str or Path) is opened and closed here;
        a file-like object is written to directly and left open.
    region_type : {"box", "point", "circle"}, optional
        DS9 region primitive. Default is ``"box"`` (1×1 pixel box).
    color : str, optional
        DS9 color name. Default is ``"red"``.
    coord_system : str, optional
        DS9 coordinate system keyword. Default is ``"physical"``.
    origin : int, optional
        Coordinate origin offset. Default is 1 (DS9 is 1-based).
    radius : float, optional
        Circle radius in pixels when ``region_type="circle"``.
        Default is 0.5.
    box_size : int, optional
        Box side length in pixels when ``region_type="box"``.
        Default is 1.
    chunk_size : int, optional
        Lines per write batch for improved I/O performance. Default is
        10000.

    Raises
    ------
    ValueError
        If *bpm* is not 2-D or *region_type* is unsupported.
    """
    if bpm.ndim != 2:
        raise ValueError(
            "export_to_ds9_region expects a 2-D BPM array."
        )

    supported = {"box", "point", "circle"}
    if region_type not in supported:
        raise ValueError(
            f"Unsupported region_type={region_type!r}. "
            f"Supported: {sorted(supported)}."
        )

    coords = np.argwhere((bpm & defect_mask) != 0)
    if coords.shape[0] == 0:
        logger.warning(
            "No pixels match defect_mask; region file will be empty."
        )

    def _fmt(row: int, col: int) -> str:
        x, y = col + origin, row + origin
        if region_type == "box":
            return f"box({x},{y},{box_size},{box_size},0) # color={color}\n"
        if region_type == "point":
            return f"point({x},{y}) # point=x color={color}\n"
        # circle
        return f"circle({x},{y},{radius}) # color={color}\n"

    needs_close = isinstance(filename, (str, Path))
    f: TextIO = (
        Path(filename).open("w", encoding="utf-8")
        if needs_close
        else filename  # type: ignore[assignment]
    )

    try:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"{coord_system}\n")

        buf: list[str] = []
        for i, (row, col) in enumerate(coords):
            buf.append(_fmt(int(row), int(col)))
            if (i + 1) % chunk_size == 0:
                f.writelines(buf)
                buf.clear()
        if buf:
            f.writelines(buf)
    finally:
        if needs_close:
            f.close()
