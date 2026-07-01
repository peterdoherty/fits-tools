#!/usr/bin/env python3
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Bad pixel and defect detection for CCD/CMOS sensors.

This subpackage provides tools for creating, analyzing, and managing bad pixel
masks (BPMs) used in CCD/CMOS image calibration and quality control.

Key design points
-----------------
- **Bit-flag BPMs**: each pixel can carry multiple defect classes (bitwise OR).
- **Self-describing FITS products**: BPM FITS files embed the bit definitions
  via HIERARCH header keywords and a DQ_DEF binary table extension.
- **32-bit by default**: BPM arrays use uint32, covering all defined bits (0-31).
  uint16 is accepted for memory-constrained cases but silently truncates any
  bit >= 16 (e.g. BPM_BIT_COL_BAD, BPM_BIT_ROW_BAD).

Modules
-------
fits_tools.bitmask
    Bit flag constants and all pure array-level operations (no FITS I/O).
fits_tools.defects.io
    FITS file I/O: save, load, merge, and validate BPM FITS files.
fits_tools.defects.detection
    Defect identification algorithms (bias / dark / flat / structural).

Quick start
-----------
>>> from fits_tools.defects import create_bpm, BPM_BIT_PIX_HOT
>>> bpm = create_bpm((512, 512))          # uint32 by default
>>> bpm[100, 200] |= BPM_BIT_PIX_HOT

>>> from fits_tools.defects import load_bpm_from_fits
>>> bpm, meta = load_bpm_from_fits('master_bpm.fits')

>>> from fits_tools.defects import detect_bright_pixels
>>> defect_map, stats = detect_bright_pixels(master_bias, threshold_adu=20.0)

Version History
---------------
1.0.0  Initial release.
1.1.0  Migrated FITS I/O from utils.py to io.py.
1.2.0  Removed utils.py and bad_pixel_map_utils.py; all I/O now in io.py.
       Removed STAGE_TO_FLAG (callers should use FLAG_NAMES directly).
       Removed load_bad_pixel_map / save_bad_pixel_map aliases; use
       load_bpm_from_fits / save_bpm_to_fits.
       Removed BPM_MASK_STRUCTURAL (use BPM_MASK_COL | BPM_MASK_ROW).
       Fixed all bit constant names to match fits_tools.bitmask.
"""

# =============================================================================
# Re-exports: bitmask (constants + pure mask operations, no FITS)
# =============================================================================
from ..bitmask import (
    BPM_BIT_BLOOM,
    # --- Structural bit constants (bits 16-26) ---
    BPM_BIT_COL_BAD,
    BPM_BIT_COL_BLOCKED,
    BPM_BIT_COL_BOUNDARY,
    BPM_BIT_COL_BRIGHT,
    BPM_BIT_COL_CTE_LOW,
    BPM_BIT_COL_DARK,
    BPM_BIT_COL_EXCLUDED,
    BPM_BIT_COL_TRAP,
    BPM_BIT_COSMIC,
    BPM_BIT_CROSSTALK,
    BPM_BIT_OUTLIER,
    # --- Pixel-level bit constants (bits 0-15) ---
    BPM_BIT_PIX_BAD,
    BPM_BIT_PIX_BRIGHT,
    BPM_BIT_PIX_CLUSTER,
    BPM_BIT_PIX_DEAD,
    BPM_BIT_PIX_EXCLUDED,
    BPM_BIT_PIX_GLOW,
    BPM_BIT_PIX_HOT,
    BPM_BIT_PIX_NOISE,
    BPM_BIT_PIX_NONLIN,
    BPM_BIT_PIX_PRNU_HIGH,
    BPM_BIT_PIX_PRNU_LOW,
    BPM_BIT_PIX_RESERVED_12,
    BPM_BIT_PIX_RESERVED_13,
    BPM_BIT_PIX_RESERVED_14,
    BPM_BIT_PIX_RTS,
    BPM_BIT_PIX_TRAP,
    BPM_BIT_ROW_BAD,
    BPM_BIT_ROW_EXCLUDED,
    BPM_BIT_ROW_RESERVED_25,
    # --- Transient / DQ bit constants (bits 27-31) ---
    BPM_BIT_SATURATED,
    BPM_MASK_ALL,
    BPM_MASK_ALL_16,
    BPM_MASK_ALL_32,
    BPM_MASK_BAD,
    BPM_MASK_COL,
    BPM_MASK_COLD,
    BPM_MASK_HOT,
    BPM_MASK_IGNORE,
    BPM_MASK_NOISE,
    BPM_MASK_PERMANENT,
    # --- Combined masks ---
    BPM_MASK_PIX,
    BPM_MASK_RESPONSE,
    BPM_MASK_ROW,
    BPM_MASK_TRANSIENT,
    BPM_MASK_TRANSPORT,
    BPM_MASK_TRAPS,
    FLAG_DESCRIPTIONS,
    FLAG_NAMES,
    # --- Enum and documentation ---
    DefectType,
    apply_bpm,
    # --- Core utilities ---
    bit_index,
    build_good_pixel_mask,
    check_bad_rows_cols,
    combine_flags,
    count_defects,
    create_bpm,
    create_master_bpm,
    describe_pixel_defects,
    describe_pixel_defects_verbose,
    export_to_ds9_region,
    filter_bpm,
    flag_bad_rows_cols,
    flag_defect_clusters,
    flag_excluded_border,
    flag_excluded_cols,
    flag_excluded_pixels,
    flag_excluded_rows,
    flag_excluded_rows_cols,
    format_bit_comments,
    format_pixel_status,
    get_bpm_statistics,
    get_cluster_statistics,
    get_defect_mask,
    get_defect_types,
    get_definitions_payload,
    has_defect,
    is_perfect_pixel,
    is_usable_pixel,
    iter_flag_definitions,
    merge_bpms,
    summarize_bpm,
    to_boolean_mask,
    validate_bpm,
)

# =============================================================================
# Re-exports: detection (algorithms — no FITS dependencies)
# =============================================================================
from .detection import (
    # High-level pipeline
    create_calibration_bpm,
    create_calibration_bpm_batch,
    create_defect_density_map,
    detect_amplifier_glow,
    detect_bad_rows_cols,
    # Detection functions
    detect_bright_pixels,
    detect_charge_traps,
    detect_dead_pixels,
    # Analysis utilities
    detect_defect_clusters,
    detect_hot_pixels,
    detect_noisy_pixels,
    # Advanced detection (placeholders — return empty BPMs with warnings)
    detect_nonlinear_pixels,
    detect_prnu_anomalies,
    detect_rts_pixels,
    detect_temporal_defects,
    estimate_prnu_model,
    get_default_detection_config,
    roi_stats,
    # Statistical utilities
    sigma_clip_stats,
)

# =============================================================================
# Re-exports: io (FITS I/O — the only place that opens/writes BPM FITS files)
# =============================================================================
from .io import (
    analyze_bpm_statistics,
    load_bpm_from_fits,
    merge_bad_pixel_maps,
    save_bpm_to_fits,
    save_calibration_bpm,
    validate_bpm_fits,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # --- Pixel-level bit constants ---
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
    "BPM_BIT_PIX_RESERVED_12",
    "BPM_BIT_PIX_RESERVED_13",
    "BPM_BIT_PIX_RESERVED_14",
    "BPM_BIT_PIX_EXCLUDED",

    # --- Structural bit constants ---
    "BPM_BIT_COL_BAD",
    "BPM_BIT_COL_BRIGHT",
    "BPM_BIT_COL_DARK",
    "BPM_BIT_COL_TRAP",
    "BPM_BIT_COL_BLOCKED",
    "BPM_BIT_COL_CTE_LOW",
    "BPM_BIT_COL_BOUNDARY",
    "BPM_BIT_COL_EXCLUDED",
    "BPM_BIT_ROW_BAD",
    "BPM_BIT_ROW_RESERVED_25",
    "BPM_BIT_ROW_EXCLUDED",

    # --- Transient / DQ bit constants ---
    "BPM_BIT_SATURATED",
    "BPM_BIT_COSMIC",
    "BPM_BIT_CROSSTALK",
    "BPM_BIT_BLOOM",
    "BPM_BIT_OUTLIER",

    # --- Combined masks ---
    "BPM_MASK_PIX",
    "BPM_MASK_COL",
    "BPM_MASK_ROW",
    "BPM_MASK_PERMANENT",
    "BPM_MASK_TRANSIENT",
    "BPM_MASK_HOT",
    "BPM_MASK_COLD",
    "BPM_MASK_NOISE",
    "BPM_MASK_TRANSPORT",
    "BPM_MASK_BAD",
    "BPM_MASK_TRAPS",
    "BPM_MASK_RESPONSE",
    "BPM_MASK_IGNORE",
    "BPM_MASK_ALL_16",
    "BPM_MASK_ALL_32",
    "BPM_MASK_ALL",

    # --- Enum and documentation ---
    "DefectType",
    "FLAG_NAMES",
    "FLAG_DESCRIPTIONS",

    # --- Core utilities ---
    "bit_index",
    "iter_flag_definitions",
    "get_definitions_payload",
    "format_bit_comments",
    "format_pixel_status",
    "is_perfect_pixel",
    "is_usable_pixel",
    "has_defect",
    "get_defect_types",
    "count_defects",
    "combine_flags",
    "create_bpm",
    "create_master_bpm",
    "merge_bpms",
    "apply_bpm",
    "get_bpm_statistics",
    "build_good_pixel_mask",
    "get_defect_mask",
    "filter_bpm",
    "describe_pixel_defects",
    "describe_pixel_defects_verbose",
    "to_boolean_mask",
    "check_bad_rows_cols",
    "flag_bad_rows_cols",
    "flag_excluded_rows",
    "flag_excluded_cols",
    "flag_excluded_rows_cols",
    "flag_excluded_pixels",
    "flag_excluded_border",
    "flag_defect_clusters",
    "get_cluster_statistics",
    "export_to_ds9_region",
    "summarize_bpm",
    "validate_bpm",

    # --- FITS I/O ---
    "save_bpm_to_fits",
    "load_bpm_from_fits",
    "validate_bpm_fits",
    "merge_bad_pixel_maps",
    "save_calibration_bpm",
    "analyze_bpm_statistics",

    # --- Statistical utilities ---
    "sigma_clip_stats",
    "roi_stats",
    "estimate_prnu_model",

    # --- Detection functions ---
    "detect_bright_pixels",
    "detect_hot_pixels",
    "detect_dead_pixels",
    "detect_prnu_anomalies",
    "detect_bad_rows_cols",
    "detect_temporal_defects",
    "detect_nonlinear_pixels",
    "detect_noisy_pixels",
    "detect_rts_pixels",
    "detect_charge_traps",
    "detect_amplifier_glow",

    # --- Analysis utilities ---
    "detect_defect_clusters",
    "create_defect_density_map",

    # --- High-level pipeline ---
    "create_calibration_bpm",
    "create_calibration_bpm_batch",
    "get_default_detection_config",

    # --- Package helpers ---
    "get_defect_bit_info",
    "get_combined_masks_info",
]


# =============================================================================
# Package-level documentation helpers
# =============================================================================

def get_defect_bit_info() -> str:
    """Return a formatted table of all defect bit flags in ascending bit order."""
    lines = ["Defect Bit Flag System", "=" * 70]
    for bit_val, name, desc in iter_flag_definitions():
        idx = bit_index(bit_val)
        lines.append(f"Bit {idx:2d} (0x{bit_val:08X})  {name:<20s}  {desc}")
    lines.append("=" * 70)
    lines.append("Pixels can carry multiple flags via bitwise OR.")
    return "\n".join(lines)


def get_combined_masks_info() -> str:
    """Return a formatted summary of all predefined combined masks."""
    masks = {
        "BPM_MASK_PIX":       (BPM_MASK_PIX,       "All single-pixel defect bits"),
        "BPM_MASK_COL":       (BPM_MASK_COL,       "All column-level defect bits"),
        "BPM_MASK_ROW":       (BPM_MASK_ROW,       "All row-level defect bits"),
        "BPM_MASK_PERMANENT": (BPM_MASK_PERMANENT, "All permanent hardware defects"),
        "BPM_MASK_TRANSIENT": (BPM_MASK_TRANSIENT, "Transient / per-image DQ flags"),
        "BPM_MASK_HOT":       (BPM_MASK_HOT,       "Hot pixels (all types)"),
        "BPM_MASK_COLD":      (BPM_MASK_COLD,       "Cold / dead pixels (all types)"),
        "BPM_MASK_NOISE":     (BPM_MASK_NOISE,     "Temporal noise / instability"),
        "BPM_MASK_TRANSPORT": (BPM_MASK_TRANSPORT, "Charge transport defects"),
        "BPM_MASK_BAD":       (BPM_MASK_BAD,       "General bad: pixel, row, column"),
        "BPM_MASK_TRAPS":     (BPM_MASK_TRAPS,     "Charge trap defects"),
        "BPM_MASK_RESPONSE":  (BPM_MASK_RESPONSE,  "Response anomalies"),
        "BPM_MASK_IGNORE":    (BPM_MASK_IGNORE,    "Pixels excluded from statistics"),
        "BPM_MASK_ALL":       (BPM_MASK_ALL,       "All 32 defined bits"),
    }

    lines = ["Predefined Combined Masks", "=" * 70]
    for name, (value, desc) in masks.items():
        included = [
            flag_name for bit_val, flag_name in FLAG_NAMES.items()
            if value & bit_val
        ]
        flags_str = ", ".join(included) if included else "(none)"
        lines.append(f"{name:<22s} (0x{int(value):08X})  {desc}")
        lines.append(f"{'':22s}  Includes: {flags_str}")
    lines.append("=" * 70)
    return "\n".join(lines)
