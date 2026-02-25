#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image calibration workflows for CCD/CMOS data.

This module provides calibration workflows for bias, dark, and flat-field
correction with proper exposure time scaling, validation, and optional
quality control integration.

Calibration Pipeline
--------------------
1. Bias correction: ``raw - master_bias``
2. Dark correction: ``(raw - bias) - (master_dark * exptime_scale)``
3. Flat correction: ``(raw - bias - dark) / normalized_flat``
"""

import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import numpy as np
from astropy.io import fits

from ..io import get_data, get_data_and_header, write_fits
from ..constants import FITS_KEY_EXPTIME
from ..exceptions import FileHandlingError, ImageProcessingError
from ..utils import add_history
from ..array_utils import validate_shapes
from ..validate import (
    ValidationResult,
    validate_bias_frame,
    validate_dark_frame,
    validate_flat_frame,
    validate_fits_file,
)
from ..quality_control import (
    assess_quality_array,
    DEFAULT_THRESHOLDS,
)

from ..bitmask import (
    BPM_BIT_SATURATED,
    BPM_BIT_OUTLIER,
    create_bpm,
    merge_bpms,
    format_bit_comments,
)
from ..types import StrOrPath
from .result_types import CalibrationBatchResult, CalibrationResult

logger = logging.getLogger(__name__)
__version__ = "2.0.0"


# ============================================================================
# Helpers
# ============================================================================

def _file_checksum(path: Path) -> str:
    """Return the SHA256 hex-digest of a file, computed in streaming chunks
    to avoid loading large master frames into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _validate_and_normalize_region(
    region: Optional[Tuple[int, int, int, int]],
    shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Validate and normalize a region specification against an image shape.

    Parameters
    ----------
    region :
        Region specified as ``(x1, x2, y1, y2)`` in **0-based Python/numpy
        indices** (not FITS 1-based pixel coordinates):

        - x1, x2 : column range ``[x1, x2)`` — left-inclusive, right-exclusive
        - y1, y2 : row range    ``[y1, y2)`` — left-inclusive, right-exclusive

        Applied to the array as ``data[y1:y2, x1:x2]``.

        .. note::
            **FITS pixel coordinate users:** FITS uses 1-based pixel indices
            and (column, row) order, consistent with this tuple's (x, y)
            ordering. To convert from FITS pixel coordinates (FITS_X, FITS_Y)
            subtract 1 from each::

                region = (FITS_X1 - 1, FITS_X2, FITS_Y1 - 1, FITS_Y2)

            The end indices are left as-is because FITS conventions are
            inclusive on both ends while Python slicing is exclusive on the
            right.

    shape :
        Image shape as ``(ny, nx)`` — i.e. ``(nrows, ncols)`` as returned
        by ``array.shape``. Note this is (y, x) order, the reverse of the
        region tuple.

    Returns
    -------
    (int, int, int, int) or None
        Normalized integer region if *region* is not None and valid;
        otherwise None.

    Raises
    ------
    ImageProcessingError
        If region cannot be parsed as four integers, or lies outside the
        bounds of the given image shape.

    Examples
    --------
    Crop to the bottom-left 100x100 pixels (0-based):

    >>> _validate_and_normalize_region((0, 100, 0, 100), (2048, 2048))
    (0, 100, 0, 100)

    Convert from FITS pixel coordinates (1-based, inclusive):
    FITS [101:200, 101:200] → region = (100, 200, 100, 200)

    >>> _validate_and_normalize_region((100, 200, 100, 200), (2048, 2048))
    (100, 200, 100, 200)
    """
    if region is None:
        return None

    try:
        x1, x2, y1, y2 = map(int, region)
    except Exception as e:
        raise ImageProcessingError(f"Invalid region {region!r}: {e}") from e

    ny, nx = shape
    if not (0 <= x1 < x2 <= nx and 0 <= y1 < y2 <= ny):
        raise ImageProcessingError(
            f"Region {region} is out of bounds for image shape {shape}"
        )
    return x1, x2, y1, y2


# ============================================================================
# Core Calibration
# ============================================================================

def apply_calibration(
    raw_data: np.ndarray,
    raw_header: fits.Header,
    master_bias_data: Optional[np.ndarray] = None,
    master_dark_data: Optional[np.ndarray] = None,
    master_dark_header: Optional[fits.Header] = None,
    master_flat_data: Optional[np.ndarray] = None,
    input_bpm: Optional[np.ndarray] = None,
    saturation_level: Optional[float] = None,
    working_dtype: np.dtype = np.float32,
    normalize_flat: bool = False,
) -> CalibrationResult:
    """
    Apply bias, dark, and flat calibration to a raw image array.

    Calibration order:
        1. raw - bias
        2. result - (dark * exptime_scale)
        3. Saturation check → DQ flag
        4. result / flat
        5. Invalid pixel check → DQ flag
        6. Merge per-image DQ with input_bpm

    Parameters
    ----------
    raw_data : np.ndarray
        Raw image array.
    raw_header : fits.Header
        FITS header of the raw image (used for EXPTIME keyword).
    master_bias_data : np.ndarray, optional
        Master bias array. Must match raw_data shape.
    master_dark_data : np.ndarray, optional
        Master dark array. Must match raw_data shape.
    master_dark_header : fits.Header, optional
        Header of the master dark frame. Required if master_dark_data is
        provided, so that exposure time scaling can be applied.
    master_flat_data : np.ndarray, optional
        Master flat array. Must match raw_data shape and be normalized
        (median ≈ 1.0). Pass normalize_flat=True to normalize automatically.
    input_bpm : np.ndarray, optional
        Master bad pixel mask (uint32) encoding permanent hardware defects.
        Per-image DQ flags are OR-merged into this mask and returned.
        If None, only per-image DQ flags are returned.
    saturation_level : float, optional
        Pixel value (in calibrated ADU) at or above which a pixel is flagged
        as BPM_BIT_SATURATED. Checked after dark correction, before flat.
        If None, no saturation check is performed.
    working_dtype : np.dtype, optional
        Dtype for internal arithmetic. Defaults to np.float32, which is
        sufficient for single-frame 16-bit calibration. Pass np.float64
        for higher precision if needed.
    normalize_flat : bool, optional
        If True, divide the master flat by its median before applying.
        Defaults to False; caller is responsible for pre-normalizing.
        A warning is emitted if the flat median deviates from 1.0 by >1%.

    Returns
    -------
    CalibrationResult
        Calibrated data, updated header, per-image DQ/BPM array, and
        flags indicating which corrections were applied.
    """

    calibrated_data = raw_data.astype(working_dtype, copy=True)
    corrections = {"bias": False, "dark": False, "flat": False}

    # Per-image data quality array — accumulates transient flags
    dq = create_bpm(raw_data.shape)

    # ------------------------------------------------------------------
    # 1. Bias subtraction
    # ------------------------------------------------------------------
    if master_bias_data is not None:
        validate_shapes(calibrated_data, master_bias_data, operation="bias correction")
        calibrated_data -= master_bias_data.astype(working_dtype)
        corrections["bias"] = True
        logger.debug("Subtracted master bias.")

    # ------------------------------------------------------------------
    # 2. Dark subtraction (scaled by exposure time)
    # ------------------------------------------------------------------
    if master_dark_data is not None:
        if master_dark_header is None:
            raise ImageProcessingError(
                "master_dark_header is required for dark scaling. "
                "Cannot determine dark exposure time."
            )

        raw_exptime = raw_header.get(FITS_KEY_EXPTIME)
        dark_exptime = master_dark_header.get(FITS_KEY_EXPTIME)

        if raw_exptime is None:
            logger.warning(
                "Raw image missing %s keyword. Skipping dark correction.",
                FITS_KEY_EXPTIME,
            )
        elif dark_exptime is None:
            logger.warning(
                "Dark frame missing %s keyword. Skipping dark correction.",
                FITS_KEY_EXPTIME,
            )
        elif float(dark_exptime) == 0:
            logger.warning("Dark frame has EXPTIME=0. Skipping dark correction.")
        else:
            validate_shapes(calibrated_data, master_dark_data, operation="dark correction")
            scale_factor = float(raw_exptime) / float(dark_exptime)
            calibrated_data -= master_dark_data.astype(working_dtype) * scale_factor
            corrections["dark"] = True
            logger.debug(
                "Subtracted scaled master dark "
                "(raw_exp=%.2fs, dark_exp=%.2fs, scale=%.3f).",
                float(raw_exptime),
                float(dark_exptime),
                scale_factor,
            )

    # ------------------------------------------------------------------
    # 3. Saturation check (after dark, before flat)
    # ------------------------------------------------------------------
    if saturation_level is not None:
        saturated = calibrated_data >= saturation_level
        n_sat = int(np.count_nonzero(saturated))
        if n_sat > 0:
            dq[saturated] |= BPM_BIT_SATURATED
            logger.warning(
                "%d pixels at or above saturation level (%.1f); "
                "flagged as SATURATED.",
                n_sat,
                saturation_level,
            )

    # ------------------------------------------------------------------
    # 4. Flat-field correction
    # ------------------------------------------------------------------
    if master_flat_data is not None:
        validate_shapes(calibrated_data, master_flat_data, operation="flat correction")

        flat = master_flat_data.astype(working_dtype)

        if input_bpm is not None:
            flat_median = float(np.median(flat[input_bpm == 0]))
        else:
            flat_median = float(np.median(flat))

        if normalize_flat:
            if flat_median == 0:
                raise ImageProcessingError(
                    "Master flat has a median of zero and cannot be normalized."
                )
            flat = flat / flat_median
            logger.info("Master flat normalized by median (%.4f).", flat_median)
        elif abs(flat_median - 1.0) > 0.01:
            logger.warning(
                "Master flat median is %.4f (expected ~1.0). "
                "The flat may not be normalized. Pass normalize_flat=True "
                "to normalize automatically, or pre-normalize before passing.",
                flat_median,
            )

        if input_bpm is not None:
            good_flat = flat[input_bpm == 0]
            flat_min = float(np.min(good_flat)) if good_flat.size > 0 else 0.0
        else:
            flat_min = float(np.min(flat))

        if flat_min <= 0:
            logger.warning(
                "Master flat contains zero or negative values (min=%.4f). "
                "This may cause division issues.",
                flat_min,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            calibrated_data /= flat

        # ------------------------------------------------------------------
        # 5. Flag invalid pixels produced by flat division
        # ------------------------------------------------------------------
        invalid_mask = ~np.isfinite(calibrated_data)
        n_invalid = int(np.count_nonzero(invalid_mask))
        if n_invalid > 0:
            dq[invalid_mask] |= BPM_BIT_OUTLIER
            calibrated_data[invalid_mask] = 0.0
            logger.warning(
                "Flat correction produced %d invalid pixels; "
                "flagged as OUTLIER in DQ array.",
                n_invalid,
            )

        corrections["flat"] = True
        logger.debug(
            "Divided by master flat (flat range: [%.4f, %.4f]).",
            flat_min,
            float(np.max(flat)),
        )

    # ------------------------------------------------------------------
    # 6. Merge per-image DQ with input master BPM
    # ------------------------------------------------------------------
    if input_bpm is not None:
        output_bpm = merge_bpms(input_bpm, dq, "or")
    else:
        # Only carry a BPM if there's actually something flagged
        output_bpm = dq if np.any(dq) else None

    # ------------------------------------------------------------------
    # 7. Update header
    # ------------------------------------------------------------------
    new_header = raw_header.copy()
    if corrections["bias"]:
        add_history(new_header, "Bias corrected")
    if corrections["dark"]:
        add_history(new_header, "Dark corrected")
    if corrections["flat"]:
        add_history(new_header, "Flat corrected")

    return CalibrationResult(
        data=calibrated_data,
        header=new_header,
        bpm=output_bpm,
        bias_corrected=corrections["bias"],
        dark_corrected=corrections["dark"],
        flat_corrected=corrections["flat"],
    )

# ============================================================================
# High-Level Calibration (file I/O, validation, QC)
# ============================================================================

def calibrate_image(
    raw_file: StrOrPath,
    output_file: Optional[StrOrPath] = None,
    master_bias: Optional[Union[StrOrPath, np.ndarray]] = None,
    master_dark: Optional[Union[StrOrPath, np.ndarray]] = None,
    master_dark_header: Optional[fits.Header] = None,
    master_flat: Optional[Union[str, Path, np.ndarray]] = None,
    bias_checksum: Optional[str] = None,
    dark_checksum: Optional[str] = None,
    flat_checksum: Optional[str] = None,
    master_bpm: Optional[Union[str, Path, np.ndarray]] = None,
    saturation_level: Optional[float] = None,
    write_bpm: bool = True,
    ext: int = 0,
    overwrite: bool = False,
    validate_inputs: bool = True,
    compute_quality: bool = False,
    working_dtype: np.dtype = np.float32,
    normalize_flat: bool = False,
    # TODO: compare_with_raw: bool = False,
) -> CalibrationResult:
    """Calibrate a FITS image with bias, dark, and flat correction."""
  
    raw_file = Path(raw_file)
    if output_file is not None:
        output_file = Path(output_file)
    if isinstance(master_bias, (str, Path)):
        master_bias = Path(master_bias)
    if isinstance(master_dark, (str, Path)):
        master_dark = Path(master_dark)
    if isinstance(master_flat, (str, Path)):
        master_flat = Path(master_flat)
    if isinstance(master_bpm, (str, Path)):
        master_bpm = Path(master_bpm)

    if output_file is not None and output_file.exists() and not overwrite:
        raise FileHandlingError(
            f"Output file exists and overwrite=False: {output_file}. "
            "Use overwrite=True to overwrite."
        )

    if validate_inputs:
        # Validate master bias
        if isinstance(master_bias, Path):
            bias_validation = validate_bias_frame(master_bias, ext=ext)
            if not bias_validation.is_valid:
                logger.warning(
                    "Master bias validation issues: %s", bias_validation.errors
                )
            # Optionally: store this on batch_result.metadata later if you want

        # Validate master dark
        if isinstance(master_dark, Path):
            dark_validation = validate_dark_frame(master_dark, ext=ext)
            if not dark_validation.is_valid:
                logger.warning(
                    "Master dark validation issues: %s", dark_validation.errors
                )

        # Validate master flat
        if isinstance(master_flat, Path):
            flat_validation = validate_flat_frame(master_flat, ext=ext)
            if not flat_validation.is_valid:
                logger.warning(
                    "Master flat validation issues: %s", flat_validation.errors
                )

        # Optionally validate master BPM if you add a validator for it later
        # if isinstance(master_bpm, Path):
        #     bpm_validation = validate_bpm_file(master_bpm, ext=ext)
        #     ...

    logger.debug("Calibrating: %s", raw_file)

    validation_results = ValidationResult()

    try:
        # 1. Validation
        if validate_inputs:
            logger.debug("Validating input files...")

            raw_validation = validate_fits_file(raw_file)
            if not raw_validation.is_valid:
                raise ImageProcessingError(
                    f"Raw file validation failed: {raw_validation.errors}"
                )
            validation_results.merge(raw_validation, prefix="raw")

            if isinstance(master_bias, Path):
                bias_validation = validate_bias_frame(
                    master_bias, ext=ext
                )
                if not bias_validation.is_valid:
                    logger.warning(
                        "Master bias validation issues: %s", bias_validation.errors
                    )
                validation_results.merge(bias_validation, prefix="bias")

            if isinstance(master_dark, Path):
                dark_validation = validate_dark_frame(master_dark, ext=ext)
                if not dark_validation.is_valid:
                    logger.warning(
                        "Master dark validation issues: %s", dark_validation.errors
                    )
                validation_results.merge(dark_validation, prefix="dark")

            if isinstance(master_flat, Path):
                flat_validation = validate_flat_frame(master_flat, ext=ext)
                if not flat_validation.is_valid:
                    logger.warning(
                        "Master flat validation issues: %s", flat_validation.errors
                    )
                validation_results.merge(flat_validation, prefix="flat")

        # 2. Load raw
        logger.debug("Loading raw data...")
        raw_data, raw_header = get_data_and_header(raw_file, ext=ext)
        if raw_data is None:
            raise ImageProcessingError(
                f"HDU {ext} in {raw_file} contains no image data"
            )

        # 3. Load calibration frames
        bias_data: Optional[np.ndarray] = None
        dark_data: Optional[np.ndarray] = None
        dark_header: Optional[fits.Header] = None
        flat_data: Optional[np.ndarray] = None
        bpm_data: Optional[np.ndarray] = None

        if master_bias is not None:
            if isinstance(master_bias, Path):
                logger.debug("Loading master bias: %s", master_bias)
                bias_data = get_data(master_bias, ext=ext)
            else:
                logger.debug("Using in-memory master bias")
                bias_data = master_bias

        if master_dark is not None:
            if isinstance(master_dark, Path):
                logger.debug("Loading master dark: %s", master_dark)
                dark_data, dark_header = get_data_and_header(master_dark, ext=ext)
            else:
                dark_data = master_dark
                if master_dark_header is not None:
                    # Caller supplied the header (e.g. from calibrate_batch);
                    # dark scaling will work correctly.
                    dark_header = master_dark_header
                else:
                    # No header available — fall back to assuming same
                    # exposure as the raw frame and warn loudly.
                    dark_header = fits.Header()
                    logger.warning(
                        "Master dark provided as array without a header. "
                        "Cannot perform exposure time scaling. "
                        "Pass master_dark_header to enable scaling. "
                        "Assuming dark EXPTIME == raw EXPTIME."
                    )
                    if FITS_KEY_EXPTIME in raw_header:
                        dark_header[FITS_KEY_EXPTIME] = raw_header[FITS_KEY_EXPTIME]

        if master_flat is not None:
            if isinstance(master_flat, Path):
                logger.debug("Loading master flat: %s", master_flat)
                flat_data = get_data(master_flat, ext=ext)
            else:
                logger.debug("Using in-memory master flat")
                flat_data = master_flat

        if master_bpm is not None:
            if isinstance(master_bpm, Path):
                bpm_data = get_data(master_bpm, ext=ext)
            else:
                bpm_data = np.asarray(master_bpm, dtype=np.uint32)

        # 4. Apply calibration
        logger.debug("Applying calibration...")
        result = apply_calibration(
            raw_data,
            raw_header,
            master_bias_data=bias_data,
            master_dark_data=dark_data,
            master_dark_header=dark_header,
            master_flat_data=flat_data,
            input_bpm=bpm_data,
            saturation_level=saturation_level,
            working_dtype=working_dtype,
            normalize_flat=normalize_flat,
        )

        result.input_file = raw_file
        result.output_file = output_file

        # 5. Quality control
        if compute_quality:
            logger.debug("Computing quality metrics...")
            try:
                qc = assess_quality_array(
                    data=result.data,
                    header=result.header,
                    filepath=raw_file,
                    frame_type=result.header.get("IMAGETYP"),
                    thresholds=DEFAULT_THRESHOLDS,
                    compute_snr=True,
                )
                result.quality_metrics = qc.to_dict()

                result.header["QCGRADE"] = (
                    qc.quality_grade,
                    "Quality control grade",
                )
                result.header["QCSCORE"] = (
                    qc.quality_score,
                    "Quality control score (0-100)",
                )
                result.header["QCPASS"] = (
                    qc.passed,
                    "Passed quality control",
                )

                if qc.snr is not None:
                    result.header["QCSNR"] = (qc.snr, "QC SNR estimate")
                if qc.num_negative_pixels > 0:
                    result.header["QCNEGPIX"] = (
                        qc.num_negative_pixels,
                        "Number of negative pixels",
                    )
                if qc.num_saturated_pixels > 0:
                    result.header["QCNSAT"] = (
                        qc.num_saturated_pixels,
                        "Number of saturated pixels",
                    )

                if not qc.passed:
                    add_history(
                        result.header,
                        f"Quality control: {qc.quality_grade} (FAILED)",
                    )
                    logger.warning(
                        "Quality control FAILED: %s", qc.quality_grade
                    )
                    for warning in qc.warnings[:3]:
                        logger.warning("  - %s", warning)
                else:
                    add_history(
                        result.header,
                        f"Quality control: {qc.quality_grade}",
                    )

                logger.debug(
                    "  Quality grade: %s", qc.quality_grade
                )
                logger.debug(
                    "  Quality score: %.1f/100", qc.quality_score
                )
                if qc.snr is not None:
                    logger.debug("  SNR: %.2f", qc.snr)
                if qc.num_negative_pixels > 0:
                    frac = (
                        qc.num_negative_pixels / qc.total_pixels * 100.0
                        if qc.total_pixels > 0
                        else 0.0
                    )
                    logger.warning(
                        "  Negative pixels: %d (%.2f%%)",
                        qc.num_negative_pixels,
                        frac,
                    )

            except Exception as e:
                logger.warning("Could not compute quality metrics: %s", e)
                logger.exception("Quality assessment error:")
                result.quality_metrics = {
                    "error": str(e),
                    "exception_type": type(e).__name__,
                }

        # 6. Attach validation + metadata
        result.validation = validation_results

        add_history(result.header, f"Calibrated with fits_tools v{__version__}")
        result.header["PROCDATE"] = (datetime.now(timezone.utc).isoformat(), "Processing date")

        if isinstance(master_bias, Path):
            result.header["BIASFILE"] = (master_bias.name, "Master bias file")
            result.header["BIASSUM"] = (
                bias_checksum or _file_checksum(master_bias),
                "Master bias SHA256 checksum",
            )
        if isinstance(master_dark, Path):
            result.header["DARKFILE"] = (master_dark.name, "Master dark file")
            result.header["DARKSUM"] = (
                dark_checksum or _file_checksum(master_dark),
                "Master dark SHA256 checksum",
            )
        if isinstance(master_flat, Path):
            result.header["FLATFILE"] = (master_flat.name, "Master flat file")
            result.header["FLATSUM"] = (
                flat_checksum or _file_checksum(master_flat),
                "Master flat SHA256 checksum",
            )
        if isinstance(master_bpm, Path):
            result.header["BPMFILE"] = (master_bpm.name, "Master BPM file")

        if validation_results and not validation_results.is_valid:
            result.header["VALWARN"] = (True, "Validation warnings present")
        else:
            result.header["VALWARN"] = (False, "No validation warnings")

        # 7. Write output
        if output_file is not None:
            # Write DQ/BPM sidecar
            if result.bpm is not None and write_bpm:
                bpm_file = output_file.with_name(
                    output_file.stem + "_dq" + output_file.suffix
                )
                bpm_header = fits.Header()
                bpm_header["BITDEPTH"] = (32, "BPM bit depth")
                bpm_header["SCIFILE"] = (output_file.name, "Corresponding science file")
                for comment in format_bit_comments():
                    bpm_header["COMMENT"] = comment
                write_fits(bpm_file, result.bpm, bpm_header, overwrite=overwrite)
                result.bpm_file = bpm_file
                result.header["DQFILE"] = (bpm_file.name, "Data quality / BPM file")
                logger.debug("Writing DQ file: %s", bpm_file)

            # Write calibrated science image
            logger.debug("Writing calibrated image: %s", output_file)
            write_fits(
                output_file,
                result.data,
                result.header,
                overwrite=overwrite,
            )
            logger.debug("Calibration complete: %s", output_file)
        else:
            logger.debug("Calibration complete (in-memory)")

        return result

    except Exception as e:
        logger.error(
            "Calibration failed for %s: %s", raw_file, e, exc_info=True
        )
        raise ImageProcessingError(f"Failed to calibrate image: {e}") from e


# ============================================================================
# Preview (no write)
# ============================================================================

def preview_calibration(
    raw_file: StrOrPath,
    output_file: Optional[StrOrPath] = None,
    master_bias: Optional[Union[StrOrPath, np.ndarray]] = None,
    master_dark: Optional[Union[StrOrPath, np.ndarray]] = None,
    master_flat: Optional[Union[StrOrPath, np.ndarray]] = None,
    ext: int = 0,
    region: Optional[Tuple[int, int, int, int]] = None,
    working_dtype: np.dtype = np.float32,
    normalize_flat: bool = False, 
) -> Dict[str, np.ndarray]:
    """
    Preview calibration on an image (or sub-region) without writing output.

    # TODO: add master_bpm support

    Returns
    -------
    dict
        Keys: 'raw', 'bias_subtracted', 'dark_subtracted', 'flat_corrected'
        (only present if the corresponding calibration step is applied).
    """
    raw_file = Path(raw_file)
    if output_file is not None:
        output_file = Path(output_file)
    if isinstance(master_bias, (str, Path)):
        master_bias = Path(master_bias)
    if isinstance(master_dark, (str, Path)):
        master_dark = Path(master_dark)
    if isinstance(master_flat, (str, Path)):
        master_flat = Path(master_flat)

    raw_data, raw_header = get_data_and_header(raw_file, ext=ext)
    if raw_data is None:
        raise ImageProcessingError(
            f"HDU {ext} in {raw_file} contains no image data"
        )

    if region is not None:
        x1, x2, y1, y2 = _validate_and_normalize_region(region, raw_data.shape)
        raw_data = raw_data[y1:y2, x1:x2]
    else:
        x1 = x2 = y1 = y2 = None  # for slicing later

    stages: Dict[str, np.ndarray] = {"raw": raw_data.astype(working_dtype, copy=True)}
    working_data = raw_data.astype(working_dtype, copy=True)

    # Bias
    if master_bias is not None:
        bias_data = (
            get_data(master_bias, ext=ext)
            if isinstance(master_bias, Path)
            else np.asarray(master_bias)
        )
        if region is not None and x1 is not None:
            bias_data = bias_data[y1:y2, x1:x2]
        working_data -= bias_data.astype(working_dtype)
        stages["bias_subtracted"] = working_data.copy()

    # Dark
    if master_dark is not None:
        if isinstance(master_dark, Path):
            dark_data, dark_header = get_data_and_header(master_dark, ext=ext)
        else:
            dark_data = np.asarray(master_dark)
            dark_header = fits.Header()
            dark_header[FITS_KEY_EXPTIME] = raw_header.get(FITS_KEY_EXPTIME, 1.0)

        if region is not None and x1 is not None:
            dark_data = dark_data[y1:y2, x1:x2]

        raw_exptime = raw_header.get(FITS_KEY_EXPTIME, 1.0)
        dark_exptime = dark_header.get(FITS_KEY_EXPTIME, 1.0)
        scale = float(raw_exptime) / float(dark_exptime)
        working_data -= dark_data.astype(working_dtype) * scale
        stages["dark_subtracted"] = working_data.copy()

    # Flat
    if master_flat is not None:
        flat_data = (
            get_data(master_flat, ext=ext)
            if isinstance(master_flat, Path)
            else np.asarray(master_flat)
        )
        if region is not None and x1 is not None:
            flat_data = flat_data[y1:y2, x1:x2]

        with np.errstate(divide="ignore", invalid="ignore"):
            working_data /= flat_data.astype(working_dtype)
        working_data = np.nan_to_num(working_data, nan=0.0, posinf=0.0, neginf=0.0)
        stages["flat_corrected"] = working_data.copy()

    return stages


# ============================================================================
# Batch Calibration
# ============================================================================

def calibrate_batch(
    raw_files: List[StrOrPath],
    output_dir: StrOrPath,
    master_bias: Optional[Union[str, Path, np.ndarray]] = None,
    master_dark: Optional[Union[str, Path, np.ndarray]] = None,
    master_dark_header: Optional[fits.Header] = None,
    master_flat: Optional[Union[str, Path, np.ndarray]] = None,
    master_bpm: Optional[Union[str, Path, np.ndarray]] = None, 
    saturation_level: Optional[float] = None,
    ext: int = 0,
    overwrite: bool = False,
    validate_inputs: bool = True,
    compute_quality: bool = False,
    working_dtype: np.dtype = np.float32,
    normalize_flat: bool = False,
    write_bpm: bool = True,
    # TODO: num_workers: int = 1,
) -> CalibrationBatchResult:
    """Calibrate multiple images in batch mode (currently serial)."""
 
    output_dir = Path(output_dir)
    raw_files = [Path(f) for f in raw_files]
    if isinstance(master_bias, (str, Path)):
        master_bias = Path(master_bias)
    if isinstance(master_dark, (str, Path)):
        master_dark = Path(master_dark)
    if isinstance(master_flat, (str, Path)):
        master_flat = Path(master_flat)
    if isinstance(master_bpm, (str, Path)):
        master_bpm = Path(master_bpm)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_files:
        raise ValueError("Empty raw_files list")

    logger.debug("Calibrating %d images...", len(raw_files))
    logger.debug("Output directory: %s", output_dir)

    bias_data: Optional[np.ndarray] = None
    dark_ref: Optional[Union[str, np.ndarray]] = None
    flat_data: Optional[np.ndarray] = None

    if master_bias is not None:
        if isinstance(master_bias, Path):
            logger.debug("Loading master bias: %s", master_bias)
            bias_data = get_data(master_bias, ext=ext)
        else:
            bias_data = master_bias

    dark_data: Optional[np.ndarray] = None
    dark_header_resolved: Optional[fits.Header] = None   # replaces dark_ref

    if master_dark is not None:
        if isinstance(master_dark, Path):
            logger.debug("Loading master dark: %s", master_dark)
            dark_data, dark_header_resolved = get_data_and_header(master_dark, ext=ext)
        else:
            dark_data = master_dark
            if master_dark_header is not None:
                dark_header_resolved = master_dark_header
            else:
                logger.warning(
                    "Master dark provided as array without master_dark_header. "
                    "Dark scaling will assume EXPTIME matches each raw frame."
                )

    if master_flat is not None:
        if isinstance(master_flat, Path):
            logger.debug("Loading master flat: %s", master_flat)
            flat_data = get_data(master_flat, ext=ext)
        else:
            flat_data = master_flat

    bpm_data: Optional[np.ndarray] = None
    if master_bpm is not None:
        if isinstance(master_bpm, Path):
            logger.debug("Loading master BPM: %s", master_bpm)
            bpm_data = get_data(master_bpm, ext=ext)
        else:
            bpm_data = np.asarray(master_bpm, dtype=np.uint32)

    batch_result = CalibrationBatchResult()
    batch_result.metadata.update(
        {
            "master_bias": master_bias if isinstance(master_bias, Path) else None,
            "master_dark": master_dark if isinstance(master_dark, Path) else None,
            "master_flat": master_flat if isinstance(master_flat, Path) else None,
        }
    )

    if isinstance(master_bias, Path):
        bias_checksum = _file_checksum(master_bias)
    else:
        bias_checksum = None

    if isinstance(master_dark, Path):
        dark_checksum = _file_checksum(master_dark)
    else:
        dark_checksum = None

    if isinstance(master_flat, Path):
        flat_checksum = _file_checksum(master_flat)
    else:
        flat_checksum = None

    if isinstance(master_bpm, Path):
        bpm_checksum = _file_checksum(master_bpm)
    else:
        bpm_checksum = None

    # TODO: parallelize when num_workers > 1 using ProcessPoolExecutor
    for i, raw_file in enumerate(raw_files, 1):
        logger.debug(
            "[%d/%d] Processing: %s",
            i,
            len(raw_files),
            raw_file.name,
        )
        try:
            output_file = output_dir / f"{raw_file.stem}_cal{raw_file.suffix}"

            result = calibrate_image(
                raw_file,
                output_file=output_file,
                master_bias=bias_data,
                master_dark=dark_data,
                master_dark_header=dark_header_resolved,
                master_flat=flat_data,
                master_bpm=bpm_data,
                saturation_level=saturation_level,
                ext=ext,
                overwrite=overwrite,
                validate_inputs=validate_inputs,
                compute_quality=compute_quality,
                working_dtype=working_dtype,
                normalize_flat=normalize_flat,
                write_bpm=write_bpm,
                bias_checksum=bias_checksum,
                dark_checksum=dark_checksum,
                flat_checksum=flat_checksum,
                bpm_checksum=bpm_checksum,
            )

            batch_result.add_success(result)
            logger.debug("[%d/%d] Processing: %s", i, len(raw_files), raw_file.name)

        except Exception as e:
            logger.error("  ✗ Failed: %s: %s", raw_file, e)
            batch_result.add_failure(raw_file, e)

    logger.info(
        "Batch calibration complete: %d/%d successful (%.1f%%)",
        batch_result.success_count,
        batch_result.total_count,
        batch_result.success_rate * 100.0,
        )

    return batch_result


__all__ = [
    "apply_calibration",
    "calibrate_image",
    "calibrate_batch",
    "preview_calibration",
]