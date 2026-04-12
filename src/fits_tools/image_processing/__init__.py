"""Image-processing routines for CCD/CMOS FITS data.

This subpackage exposes the primary public APIs for calibration, combining,
pixel-domain operations, geometric transformations, cosmic-ray cleaning, and
streaming statistics/combining.

The package initializer intentionally re-exports only the stable, user-facing
entry points from the underlying modules. Low-level helpers remain in their
respective modules.
"""

from __future__ import annotations

from .result_types import (
    BaseResult,
    CalibrationResult,
    CombiningResult,
    ImageOperationResult,
    ProcessingResult,
    TransformationResult,
)
from .calibration import apply_calibration, calibrate_batch, calibrate_image, preview_calibration
from .combining import combine_images, combine_images_robust, create_master_frame, mean_combine, median_combine
from .operations import (
    BINARY_IMAGE_OPS,
    BITWISE_OPS,
    CLI_ALIASES,
    COMPARISON_OPS,
    SCALAR_OPS,
    UNARY_OPS,
    add_images,
    apply_bitwise_operation,
    apply_operation_to_file,
    apply_scalar_operation,
    apply_unary_operation,
    create_mask,
    divide_images,
    fill_masked,
    fill_where,
    get_operation_description,
    list_operations,
    multiply_images,
    scale_image,
    set_value,
    subtract_images,
    threshold_image,
)
from .transformations import (
    bin_image,
    crop_image,
    flip_image,
    invert_image,
    normalize_image,
    rescale_intensity,
    rotate_image,
    transform_image_file,
    transpose_image,
)
from .cosmic_rays import (
    batch_clean_cosmic_rays,
    check_dependencies,
    clean_cosmic_rays_file,
    detect_cosmic_rays,
    get_available_methods,
    reject_cosmic_rays,
)
from .streaming import streaming_combine, streaming_statistics

__all__ = [
    "BaseResult", "CalibrationResult", "CombiningResult", "ImageOperationResult", "ProcessingResult", "TransformationResult",
    "apply_calibration", "calibrate_batch", "calibrate_image", "preview_calibration",
    "combine_images", "combine_images_robust", "create_master_frame", "mean_combine", "median_combine",
    "BINARY_IMAGE_OPS", "BITWISE_OPS", "CLI_ALIASES", "COMPARISON_OPS", "SCALAR_OPS", "UNARY_OPS",
    "add_images", "apply_bitwise_operation", "apply_operation_to_file", "apply_scalar_operation", "apply_unary_operation",
    "create_mask", "divide_images", "fill_masked", "fill_where", "get_operation_description", "list_operations",
    "multiply_images", "scale_image", "set_value", "subtract_images", "threshold_image",
    "bin_image", "crop_image", "flip_image", "invert_image", "normalize_image", "rescale_intensity",
    "rotate_image", "transform_image_file", "transpose_image",
    "batch_clean_cosmic_rays", "check_dependencies", "clean_cosmic_rays_file", "detect_cosmic_rays",
    "get_available_methods", "reject_cosmic_rays",
    "streaming_combine", "streaming_statistics",
]
