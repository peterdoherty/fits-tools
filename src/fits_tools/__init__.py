#!/usr/bin/env python3
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
fits_tools — A comprehensive toolkit for CCD/CMOS image processing and
quality control.

This package provides utilities for FITS file I/O, header keyword management,
statistical analysis, noise simulation, and automated quality assessment.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fits-tools")
except PackageNotFoundError:  # running from source tree without installation
    __version__ = "0.1.0"

__author__ = "FITS Tools Development Team"


# --- Core I/O ---
# --- Bitmask and Bad Pixel Map ---
from . import bitmask
from .array_validation import (
    validate_indices,
    validate_numeric,
    validate_positive,
    validate_shape,
    validate_shapes,
)
from .bitmask import (
    BPM_BIT_PIX_DEAD,
    BPM_BIT_PIX_HOT,
    BPM_MASK_PERMANENT,
    BPM_MASK_TRANSIENT,
    apply_bpm,
    get_bpm_statistics,
)

# --- Exceptions ---
from .exceptions import (
    FileHandlingError,
    FitsToolsError,
    HeaderKeywordError,
    ImageProcessingError,
    ValidationError,
)
from .fits_output import (
    HDUResult,
    OutputMode,
    StructurePolicy,
    commit_hdu_result,
    commit_hdu_results,
)

# --- HDU Selection ---
from .hdu import (
    HDUIdentity,
    HDUQuery,
    HDUSelector,
    find_hdu_indices,
    get_hdu_identity,
    resolve_hdu_index,
    resolve_hdu_indices,
)

# --- Structural HDU Operations ---
from .hdu_ops import (
    remove_hdus_from_file as remove_hdus,
)
from .hdu_ops import (
    reorder_hdus_in_file as reorder_hdus,
)
from .header_utils import copy_header
from .image_types import normalize_image_type
from .io import (
    count_hdus,
    get_all_data,
    get_data,
    get_data_and_header,
    get_header,
    get_image_hdus,
    get_image_shape,
    inspect_fits,
    transactional_write_hdulist,
    write_fits,  # deprecated; use write_simple_fits
    write_simple_fits,
)

# --- Header Keyword Management ---
from .keywords import (
    batch_delete_key,
    batch_set_key,
    check_keyword,
    delete_key,
    get_key,
    set_key,
)

# --- Quality Control ---
from .quality_control import (
    QualityMetrics,
    QualityThresholds,
    assess_calibration_quality,
    assess_quality,
    assess_quality_array,
    batch_quality_assessment,
)
from .resolved_hdu import (
    ResolvedHDU,
    derive_associated_selector,
    identity_selector,
    load_resolved_hdu,
)

# --- File Selection ---
from .selection import select_files

# --- Statistical Analysis ---
from .statistics import (
    array_stats,
    calculate_snr,
    robust_stats,
    stats,
)

# --- Data Inventory ---
from .survey import survey_fits

# --- Constants / image-type helpers ---
from .types import RegionType
from .utils import is_fits_file

# --- Validation ---
from .validate import (
    quick_validate,
    validate_fits_file,
)

__all__ = [
    # I/O
    "copy_header",
    "count_hdus",
    "get_all_data",
    "get_data",
    "get_data_and_header",
    "get_header",
    "get_image_hdus",
    "get_image_shape",
    "inspect_fits",
    "transactional_write_hdulist",
    "write_simple_fits",
    "write_fits",  # deprecated; use write_simple_fits
    # Bitmask and BPM
    "bitmask",
    "BPM_BIT_PIX_HOT",
    "BPM_BIT_PIX_DEAD",
    "BPM_MASK_PERMANENT",
    "BPM_MASK_TRANSIENT",
    "apply_bpm",
    "get_bpm_statistics",
    # Data inventory
    "survey_fits",
    # Header keywords
    "batch_delete_key",
    "batch_set_key",
    "check_keyword",
    "delete_key",
    "get_key",
    "set_key",
    # Quality control
    "QualityMetrics",
    "QualityThresholds",
    "assess_calibration_quality",
    "assess_quality",
    "assess_quality_array",
    "batch_quality_assessment",
    # Statistics
    "array_stats",
    "calculate_snr",
    "robust_stats",
    "stats",
    # Validation
    "quick_validate",
    "validate_fits_file",
    "validate_indices",
    "validate_numeric",
    "validate_positive",
    "validate_shape",
    "validate_shapes",
    # HDU operations
    "remove_hdus",
    "reorder_hdus",
    # HDU selection
    "HDUIdentity",
    "HDUQuery",
    "HDUSelector",
    "find_hdu_indices",
    "get_hdu_identity",
    "resolve_hdu_index",
    "resolve_hdu_indices",
    "ResolvedHDU",
    "derive_associated_selector",
    "identity_selector",
    "load_resolved_hdu",
    "OutputMode",
    "StructurePolicy",
    "HDUResult",
    "commit_hdu_result",
    "commit_hdu_results",
    # File selection
    "select_files",
    # Constants / helpers
    "RegionType",
    "is_fits_file",
    "normalize_image_type",
    # Exceptions
    "FileHandlingError",
    "FitsToolsError",
    "HeaderKeywordError",
    "ImageProcessingError",
    "ValidationError",
]
