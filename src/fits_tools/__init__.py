#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from .header_utils import copy_header
from .io import (
    count_hdus,
    get_all_data,
    get_data,
    get_data_and_header,
    get_header,
    get_image_hdus,
    get_image_shape,
    inspect_fits,
    write_fits,  # deprecated; use write_simple_fits
    write_simple_fits,
)

# --- Bitmask and Bad Pixel Map ---
from . import bitmask
from .bitmask import (
    BPM_BIT_PIX_DEAD,
    BPM_BIT_PIX_HOT,
    BPM_MASK_PERMANENT,
    BPM_MASK_TRANSIENT,
    apply_bpm,
    get_bpm_statistics,
)

# --- Data Inventory ---
from .survey import survey_fits

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

# --- Statistical Analysis ---
from .statistics import (
    array_stats,
    calculate_snr,
    robust_stats,
    stats,
)

# --- Validation ---
from .validate import (
    quick_validate,
    validate_fits_file,
)
from .array_validation import (
    validate_indices,
    validate_numeric,
    validate_positive,
    validate_shape,
    validate_shapes,
)

# --- Structural HDU Operations ---
from .hdu_ops import (
    _op_remove_hdus as remove_hdus,
    _op_reorder_hdus as reorder_hdus,
)

# --- File Selection ---
from .selection import select_files

# --- Constants / image-type helpers ---
from .types import RegionType
from .image_types import normalize_image_type
from .utils import is_fits_file

# --- Exceptions ---
from .exceptions import (
    FileHandlingError,
    FitsToolsError,
    HeaderKeywordError,
    ImageProcessingError,
    ValidationError,
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
