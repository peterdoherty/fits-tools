"""fits_tools.image_processing

CCD/CMOS image-processing routines centered on FITS I/O.

The command-line interface is implemented in :mod:`fits_tools.image_processing.cli`
(and is intentionally not imported here to keep package import side effects
minimal).
"""

from __future__ import annotations

# Result types
from .result_types import (
    BaseResult,
    CalibrationResult,
    CombiningResult,
    ImageOperationResult,
    ProcessingResult,
    TransformationResult,
)

# Calibration
from .calibration import (
    apply_calibration,
    calibrate_image,
    calibrate_batch,
    preview_calibration,
)

# Combining
from .combining import (
    combine_images,
    combine_images_robust,
    create_master_frame,
    mean_combine,
    median_combine,
)

# Operations
from .operations import (
    apply_unary_operation,
    apply_scalar_operation,
    create_mask,
    threshold_image,
    apply_bitwise_operation,
    subtract_images,
    add_images,
    multiply_images,
    divide_images,
    scale_image,
    apply_operation_to_file,
    list_operations,
    get_operation_description,
    UNARY_OPS,
    SCALAR_OPS,
    COMPARISON_OPS,
    BITWISE_OPS,
    BINARY_IMAGE_OPS,
    CLI_ALIASES,
    fill_masked,
    fill_where,
    set_value,
)

# Transformations
from .transformations import (
    crop_image,
    bin_image,
    rotate_image,
    flip_image,
    transpose_image,
    normalize_image,
    rescale_intensity,
    invert_image,
    transform_image_file,
)

# Cosmic rays
from .cosmic_rays import (
    detect_cosmic_rays,
    reject_cosmic_rays,
    clean_cosmic_rays_file,
    batch_clean_cosmic_rays,
    get_available_methods,
    check_dependencies,
)

# Streaming
from .streaming import (
    streaming_statistics,
    streaming_combine,
)

# Batch convenience wrappers
from .batch import (
    process_batch,
    batch_apply_operation,
    batch_transform,
)

__all__ = [
    # Result types
    "BaseResult",
    "CalibrationResult",
    "CombiningResult",
    "ImageOperationResult",
    "ProcessingResult",
    "TransformationResult",

    # Calibration
    "apply_calibration",
    "calibrate_image",
    "calibrate_batch",
    "preview_calibration",

    # Combining
    "combine_images",
    "combine_images_robust",
    "create_master_frame",
    "median_combine",
    "mean_combine",

    # Operations
    "apply_unary_operation",
    "apply_scalar_operation",
    "create_mask",
    "threshold_image",
    "apply_bitwise_operation",
    "subtract_images",
    "add_images",
    "multiply_images",
    "divide_images",
    "scale_image",
    "apply_operation_to_file",
    "list_operations",
    "get_operation_description",
    "UNARY_OPS",
    "SCALAR_OPS",
    "COMPARISON_OPS",
    "BITWISE_OPS",
    "BINARY_IMAGE_OPS",
    "CLI_ALIASES",

    # Transformations
    "crop_image",
    "bin_image",
    "rotate_image",
    "flip_image",
    "transpose_image",
    "normalize_image",
    "rescale_intensity",
    "invert_image",
    "transform_image_file",

    # Cosmic rays
    "detect_cosmic_rays",
    "reject_cosmic_rays",
    "clean_cosmic_rays_file",
    "batch_clean_cosmic_rays",
    "get_available_methods",
    "check_dependencies",

    # Streaming
    "streaming_statistics",
    "streaming_combine",

    # Batch
    "process_batch",
    "batch_apply_operation",
    "batch_transform",
    "fill_masked",
    "fill_where",
    "set_value",
]
