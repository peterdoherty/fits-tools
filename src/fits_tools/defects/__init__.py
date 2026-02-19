#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bad pixel and defect detection for CCD/CMOS sensors.

This subpackage provides tools for creating, analyzing, and managing bad pixel
masks (BPMs) used in CCD/CMOS image calibration and quality control.

Key design points
-----------------
- **Bit-flag BPMs**: each pixel can carry multiple defect classes (bitwise OR).
- **Self-describing FITS products**: BPM FITS files can embed the bit
  definitions via HIERARCH header keys and a DQ_DEF table extension.
- **32-bit by default**: BPM arrays use uint32, covering all defined bits (0-31).
  uint16 is accepted for memory-constrained cases but must not be used with any
  bit >= 16 (e.g. BPM_BIT_COSMIC); NumPy will silently truncate such values.

Modules
-------
bitmask
    Bit flag constants and basic utilities (no FITS dependencies)
utils
    FITS file I/O and high-level analysis functions
detection
    Defect identification algorithms (bias/dark/flat/structural)

Quick start
-----------
>>> from fits_tools.defects import create_bpm, BPM_BIT_DARK
>>> bpm = create_bpm((512, 512))          # uint32 by default
>>> bpm[100, 200] |= BPM_BIT_DARK

>>> from fits_tools.defects import load_bad_pixel_map, get_defect_mask
>>> bpm, hdr = load_bad_pixel_map('master_bpm.fits')
>>> hot_mask = get_defect_mask(bpm, [BPM_BIT_DARK, BPM_BIT_HOT])
"""

# =============================================================================
# Re-exports: bitmask (constants + pure mask operations)
# =============================================================================
from .bitmask import (
    # Bit constants
    BPM_BIT_BIAS,
    BPM_BIT_DARK,
    BPM_BIT_FLAT_HOT,
    BPM_BIT_FLAT_COLD,
    BPM_BIT_BAD_ROW,
    BPM_BIT_BAD_COL,
    BPM_BIT_TRAP,
    BPM_BIT_TRAP_COL,
    BPM_BIT_BLOCKED,
    BPM_BIT_HOT,
    BPM_BIT_DEAD,
    BPM_BIT_SATURATED,
    BPM_BIT_TELEGRAPH,
    # Extended / higher-order bits (require uint32 once you cross bit 15)
    BPM_BIT_COSMIC,
    BPM_BIT_GLOW,
    BPM_BIT_BLOOM,
    BPM_BIT_CROSSTALK,
    BPM_BIT_OUTLIER,

    # Reserved bits
    BPM_BIT_RESERVED_13,
    BPM_BIT_RESERVED_14,
    BPM_BIT_RESERVED_15,
    BPM_BIT_RESERVED_21,
    BPM_BIT_RESERVED_22,
    BPM_BIT_RESERVED_23,
    BPM_BIT_RESERVED_24,
    BPM_BIT_RESERVED_25,
    BPM_BIT_RESERVED_26,
    BPM_BIT_RESERVED_27,
    BPM_BIT_RESERVED_28,
    BPM_BIT_RESERVED_29,
    BPM_BIT_RESERVED_30,
    BPM_BIT_RESERVED_31,

    # Combined masks
    BPM_MASK_PERMANENT,
    BPM_MASK_TRANSIENT,
    BPM_MASK_HOT,
    BPM_MASK_COLD,
    BPM_MASK_STRUCTURAL,
    BPM_MASK_TRAPS,
    BPM_MASK_RESPONSE,
    BPM_MASK_ALL,
    BPM_MASK_ALL_16,
    BPM_MASK_ALL_32,

    # Enum + metadata
    DefectType,
    DEFECT_NAMES,
    DEFECT_DESCRIPTIONS,

    # Basic utilities (no FITS dependencies)
    has_defect,
    get_defect_types,
    count_defects,
    create_bpm,
    merge_bpms,
    apply_bpm,
    get_bpm_statistics,

    # Definition helpers (for embedding / validation)
    iter_defect_definitions,
    get_definitions_payload,
)

# =============================================================================
# Re-exports: utils (FITS I/O + inspection)
# =============================================================================
from .utils import (
    STAGE_TO_FLAG,
    load_bad_pixel_map,
    save_bad_pixel_map,
    analyze_bpm_statistics,
    merge_bad_pixel_maps,
    get_defect_mask,
    filter_bpm,
    describe_pixel_defects,
    check_bad_rows_cols,
    flag_bad_rows_cols,
)

# =============================================================================
# Re-exports: detection (algorithms)
# =============================================================================
from .detection import (
    sigma_clip_stats,
    roi_stats,
    detect_bias_defects,
    detect_dark_hot_pixels,
    detect_flat_defects,
    detect_bad_rows_cols,
    create_defect_density_map,
    create_calibration_bpm,
    get_default_detection_config,
)

__version__ = "1.1.1"

__all__ = [
    # Bit constants
    "BPM_BIT_BIAS",
    "BPM_BIT_DARK",
    "BPM_BIT_FLAT_HOT",
    "BPM_BIT_FLAT_COLD",
    "BPM_BIT_BAD_ROW",
    "BPM_BIT_BAD_COL",
    "BPM_BIT_TRAP",
    "BPM_BIT_TRAP_COL",
    "BPM_BIT_BLOCKED",
    "BPM_BIT_HOT",
    "BPM_BIT_DEAD",
    "BPM_BIT_SATURATED",
    "BPM_BIT_TELEGRAPH",
    "BPM_BIT_COSMIC",
    "BPM_BIT_GLOW",
    "BPM_BIT_BLOOM",
    "BPM_BIT_CROSSTALK",
    "BPM_BIT_OUTLIER",

    # Reserved bits
    "BPM_BIT_RESERVED_13",
    "BPM_BIT_RESERVED_14",
    "BPM_BIT_RESERVED_15",
    "BPM_BIT_RESERVED_21",
    "BPM_BIT_RESERVED_22",
    "BPM_BIT_RESERVED_23",
    "BPM_BIT_RESERVED_24",
    "BPM_BIT_RESERVED_25",
    "BPM_BIT_RESERVED_26",
    "BPM_BIT_RESERVED_27",
    "BPM_BIT_RESERVED_28",
    "BPM_BIT_RESERVED_29",
    "BPM_BIT_RESERVED_30",
    "BPM_BIT_RESERVED_31",

    # Combined masks
    "BPM_MASK_PERMANENT",
    "BPM_MASK_TRANSIENT",
    "BPM_MASK_HOT",
    "BPM_MASK_COLD",
    "BPM_MASK_STRUCTURAL",
    "BPM_MASK_TRAPS",
    "BPM_MASK_RESPONSE",
    "BPM_MASK_ALL",
    "BPM_MASK_ALL_16",
    "BPM_MASK_ALL_32",

    # Enum + metadata
    "DefectType",
    "DEFECT_NAMES",
    "DEFECT_DESCRIPTIONS",
    "STAGE_TO_FLAG",

    # Core mask ops
    "has_defect",
    "get_defect_types",
    "count_defects",
    "create_bpm",
    "merge_bpms",
    "apply_bpm",
    "get_bpm_statistics",

    # Definition helpers
    "iter_defect_definitions",
    "get_definitions_payload",

    # FITS I/O + analysis
    "load_bad_pixel_map",
    "save_bad_pixel_map",
    "analyze_bpm_statistics",
    "merge_bad_pixel_maps",
    "get_defect_mask",
    "filter_bpm",
    "describe_pixel_defects",
    "check_bad_rows_cols",
    "flag_bad_rows_cols",

    # Detection algorithms
    "sigma_clip_stats",
    "roi_stats",
    "detect_bias_defects",
    "detect_dark_hot_pixels",
    "detect_flat_defects",
    "detect_bad_rows_cols",
    "create_defect_density_map",
    "create_calibration_bpm",
    "get_default_detection_config",

    "__version__",
]


# =============================================================================
# Documentation helpers
# =============================================================================

def get_defect_bit_info() -> str:
    """Return formatted information about all defect bit flags."""
    import numpy as np

    info = ["Defect Bit Flag System", "=" * 70]
    # Stable order: by bit position
    for bit_value, name in sorted(DEFECT_NAMES.items(), key=lambda kv: int(np.log2(kv[0]))):
        bit_pos = int(np.log2(bit_value))
        desc = DEFECT_DESCRIPTIONS.get(bit_value, "")
        info.append(f"Bit {bit_pos:2d} (value {bit_value:10d}): {name:12s} - {desc}")
    info.append("=" * 70)
    info.append("Note: Pixels can have multiple flags (bitwise OR combination)")
    return "\n".join(info)


def get_combined_masks_info() -> str:
    """Return information about predefined combined masks."""
    info = ["Predefined Combined Masks", "=" * 70]

    masks = {
        "BPM_MASK_PERMANENT": (BPM_MASK_PERMANENT, "All permanent defects"),
        "BPM_MASK_TRANSIENT": (BPM_MASK_TRANSIENT, "Transient/conditional defects"),
        "BPM_MASK_HOT": (BPM_MASK_HOT, "Hot pixels (dark + flat + hot)"),
        "BPM_MASK_COLD": (BPM_MASK_COLD, "Cold/dead pixels"),
        "BPM_MASK_STRUCTURAL": (BPM_MASK_STRUCTURAL, "Bad rows/columns/blocked"),
        "BPM_MASK_TRAPS": (BPM_MASK_TRAPS, "Charge trap defects"),
        "BPM_MASK_RESPONSE": (BPM_MASK_RESPONSE, "Response anomalies"),
        "BPM_MASK_ALL": (BPM_MASK_ALL, "All possible defined defects"),
    }

    for name, (value, desc) in masks.items():
        included_flags = []
        for bit_value, flag_name in DEFECT_NAMES.items():
            if value & bit_value:
                included_flags.append(flag_name)
        flags_str = ", ".join(sorted(included_flags)) if included_flags else "ALL"
        info.append(f"{name:22s} (0x{int(value):08X}): {desc}")
        info.append(f"{'':22s}   Includes: {flags_str}")

    info.append("=" * 70)
    return "\n".join(info)


__all__.extend(["get_defect_bit_info", "get_combined_masks_info"])
