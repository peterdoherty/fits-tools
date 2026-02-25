#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fits_tools - A comprehensive toolkit for CCD/CMOS image processing and quality control.

This package provides utilities for FITS file I/O, header keyword management, 
statistical analysis, noise simulation, and automated quality assessment.
"""

__version__ = "1.0.0"
__author__ = "FITS Tools Development Team"

# Core I/O and Inspection
from .io import (
    get_data,
    get_header,
    get_data_and_header,
    get_all_data,
    write_fits,
    get_image_hdus,
    count_hdus,
    inspect_fits,
)

from . import bitmask
from . import bpm_builder

# Optional: expose key items at package level
from .bitmask import (
    BPM_BIT_PIX_HOT,
    BPM_BIT_PIX_DEAD,
    BPM_MASK_PERMANENT,
    BPM_MASK_TRANSIENT,
    apply_bpm,
    get_bpm_statistics,
)

from .bpm_builder import BPMBuilder

# Data Inventory and Surveying
from .survey import (
    survey_fits,
)

# Header Keyword Management
from .keywords import (
    get_key,
    set_key,
    delete_key,
    check_keyword,
    batch_set_key,
    batch_delete_key,
)

# Quality Control and Assessment
from .quality_control import (
    assess_quality,
    assess_quality_array,
    assess_calibration_quality,
    batch_quality_assessment,
    QualityMetrics,
    QualityThresholds,
)

# Statistical Analysis
from .statistics import (
    stats,
    robust_stats,
    calculate_snr,
)

# Validation
from .validate import (
    validate_fits_file,
    quick_validate,
)

from .array_utils import (
    validate_indices,
    validate_shape,
    validate_shapes,
    validate_positive,
    validate_numeric,
)

# Noise Addition and Simulation
from .simulation.noise import (
    add_gaussian_noise,
    add_poisson_noise,
    add_noise_batch,
)

# Structural HDU Operations
from .hdu_ops import (
    _op_remove_hdus as remove_hdus,
    _op_reorder_hdus as reorder_hdus,
)

# File Selection and Filtering
from .selection import (
    select_files,
)

# Constants, Enums, and Helpers
from .constants import (
    RegionType,
    normalize_image_type,
    is_fits_file,
)

# Custom Exceptions
from .exceptions import (
    FitsToolsError,
    FileHandlingError,
    ImageProcessingError,
    HeaderKeywordError,
    ValidationError,
)

__all__ = [
    # I/O
    'get_data',
    'get_header',
    'get_data_and_header',
    'get_all_data',
    'write_fits',
    'get_image_hdus',
    'count_hdus',
    'inspect_fits',
    
    # bitmasks
    'bitmask',
    'bpm_builder',
    'BPMBuilder',


    # defect detection


    # artifact detection

    
    # Surveying
    'survey_fits',
    
    # Keywords
    'get_key',
    'set_key',
    'delete_key',
    'check_keyword',
    'batch_set_key',
    'batch_delete_key',
    
    # Quality Control
    'assess_quality',
    'assess_quality_array',
    'assess_calibration_quality',
    'batch_quality_assessment',
    'QualityMetrics',
    'QualityThresholds',
    
    # Statistics
    'stats',
    'robust_stats',
    'calculate_snr',
    
 

    # Validation
    'validate_fits_file',
    'quick_validate',
    
    # Noise
    'add_gaussian_noise',
    'add_poisson_noise',
    'add_noise_batch',
    
    # Selection
    'select_files',
    
    # HDU Ops
    'remove_hdus',
    'reorder_hdus',
    
    # Constants
    'RegionType',
    'normalize_image_type',
    'is_fits_file',
    
    # Exceptions
    'FitsToolsError',
    'FileHandlingError',
    'ImageProcessingError',
    'HeaderKeywordError',
    'ValidationError',
]