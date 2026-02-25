#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FITS I/O for bad pixel masks and calibration frames.

This module provides load, save, merge, and validation operations for bad pixel
maps (BPMs) stored as FITS files.  It is the only place in fits_tools that
should open or write FITS files containing BPM data.

On-disk format
--------------
A BPM FITS file written by this module has two HDUs:

``PRIMARY``
    The BPM integer array (uint32 recommended, covers bits 0-31).  The header
    contains human-readable provenance (creation date, software version,
    defect counts) and a machine-readable SHA-256 hash of the bit definitions
    embedded as HIERARCH keywords.

``DQ_DEF`` (BinTable)
    A binary table extension that stores the complete set of bit definitions
    (bit index, bit value, name, description) in a machine-readable form.
    This makes the file self-describing — any reader can reconstruct the flag
    schema without consulting the source code.

Relationship to other modules
------------------------------
- ``fits_tools.bitmask`` — all bit constants, array-level operations (no FITS)
- ``fits_tools.defects.detection`` — defect detection algorithms (no FITS)
- This module — FITS I/O only; imports from both of the above

Version History
---------------
1.0.0  Initial implementation from io.py (detection module)
1.1.0  Merged with utils.py: added BPMDEFS binary table, SHA-256 hash,
       merge_bad_pixel_maps, analyze_bpm_statistics, validate_bpm_fits,
       save_calibration_bpm.
1.2.0  Default BPM dtype changed to uint32 to cover all defined bits (0-31).
       STAGE_TO_FLAG removed; callers should use DEFECT_NAMES directly.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.time import Time

from ..bitmask import (
    FLAG_NAMES,
    FLAG_DESCRIPTIONS,
    format_bit_comments,
    get_bpm_statistics,
    get_definitions_payload,
)

logger = logging.getLogger(__name__)

__version__ = "1.2.0"

__all__ = [
    # Primary I/O
    'save_bpm_to_fits',
    'load_bpm_from_fits',
    'validate_bpm_fits',
    # Multi-frame operations
    'merge_bad_pixel_maps',
    'save_calibration_bpm',
    # Analysis with logging
    'analyze_bpm_statistics',
]


# =============================================================================
# Private helpers
# =============================================================================

def _bpm_nbits_from_dtype(dtype: np.dtype) -> int:
    """Return the number of usable bits for a given BPM dtype."""
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return 8
    if dtype == np.uint16:
        return 16
    if dtype == np.uint32:
        return 32
    if dtype == np.uint64:
        return 64
    return int(dtype.itemsize * 8)


def _sha256_defs(payload: dict) -> str:
    """Return a hex SHA-256 digest of a JSON-serialised definitions payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _embed_bpm_definitions_hierarch(header: fits.Header, payload: dict, defs_hash: str) -> None:
    """
    Embed defect bit definitions in a FITS header using HIERARCH keywords.

    These keywords allow any reader to verify that the bit schema matches
    what the file was written with, without opening the DQ_DEF extension.
    """
    schema = int(payload.get("schema", 1))
    nbits  = int(payload.get("nbits", 0))
    header["HIERARCH BPM.DEF.SCHEMA"] = (schema,    "BPM definition schema version")
    header["HIERARCH BPM.DEF.NBITS"]  = (nbits,     "Number of defined bits")
    header["HIERARCH BPM.DEF.HASH"]   = (defs_hash, "sha256(bit definitions)")
    for item in payload.get("bits", []):
        b = int(item["bit"])
        header[f"HIERARCH BPM.DEF.BIT.{b:02d}.VALUE"] = (int(item["value"]),   "Bit value")
        header[f"HIERARCH BPM.DEF.BIT.{b:02d}.NAME"]  = (str(item["name"]),    "Bit name")
        header[f"HIERARCH BPM.DEF.BIT.{b:02d}.DESC"]  = (str(item["desc"]),    "Bit meaning")


def _build_bpmdefs_extension(payload: dict, defs_hash: str) -> fits.BinTableHDU:
    """
    Build the DQ_DEF binary table HDU that stores the complete bit definitions.

    The table has four columns: BIT (index), VALUE (power-of-two integer),
    NAME (up to 20 characters), DESC (up to 80 characters).
    """
    bits = payload.get("bits", [])
    cols = [
        fits.Column(
            name="BIT",   format="I",
            array=np.array([int(b["bit"])   for b in bits], dtype=np.uint16),
        ),
        fits.Column(
            name="VALUE", format="K",
            array=np.array([int(b["value"]) for b in bits], dtype=np.int64),
        ),
        fits.Column(
            name="NAME",  format="20A",
            array=np.array(
                [str(b["name"])[:20].encode("ascii", "replace") for b in bits],
                dtype="S20",
            ),
        ),
        fits.Column(
            name="DESC",  format="80A",
            array=np.array(
                [str(b["desc"])[:80].encode("ascii", "replace") for b in bits],
                dtype="S80",
            ),
        ),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="DQ_DEF")
    hdu.header["HIERARCH BPM.DEF.SCHEMA"] = (int(payload.get("schema", 1)), "BPM definition schema version")
    hdu.header["HIERARCH BPM.DEF.NBITS"]  = (int(payload.get("nbits",  0)), "Number of defined bits")
    hdu.header["HIERARCH BPM.DEF.HASH"]   = (defs_hash,                     "sha256(bit definitions)")
    return hdu


# =============================================================================
# Save
# =============================================================================

def save_bpm_to_fits(
    bpm: np.ndarray,
    filename: StrOrPath,
    stats: Optional[Dict] = None,
    header_cards: Optional[Dict] = None,
    overwrite: bool = True,
) -> None:
    """
    Save a bad pixel mask to a FITS file with complete metadata.

    The output file has two HDUs:

    - ``PRIMARY``: the BPM integer array with provenance header.
    - ``DQ_DEF``: binary table of bit definitions (machine-readable).

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask (unsigned integer dtype, uint32 recommended).
    filename : str or Path
        Output FITS file path.
    stats : dict, optional
        Detection statistics dict returned by
        ``defects.detection.create_calibration_bpm``.  When provided, the
        per-stage thresholds and medians are recorded in the header.
    header_cards : dict, optional
        Extra header cards to include.  Values may be plain scalars or
        ``(value, comment)`` tuples.
    overwrite : bool
        Overwrite an existing file.  Default: True.

    Examples
    --------
    >>> bpm, stats = create_calibration_bpm(bias, dark, flat)
    >>> save_bpm_to_fits(bpm, 'master_bpm.fits', stats=stats)

    See Also
    --------
    load_bpm_from_fits : Load a BPM saved by this function.
    save_calibration_bpm : Save BPM and master calibration frames together.
    """
    filename = Path(filename)

    # Ensure unsigned integer dtype
    if bpm.dtype.kind != "u":
        logger.warning(
            "save_bpm_to_fits: BPM dtype is %s (not unsigned integer); "
            "converting to uint32.", bpm.dtype,
        )
        bpm = bpm.astype(np.uint32)

    # --- Primary HDU ---
    hdu = fits.PrimaryHDU(bpm)
    hdr = hdu.header

    # Provenance
    hdr["INSTRUME"] = ("CCD",          "Instrument type")
    hdr["IMAGETYP"] = ("BPM",          "Image type: Bad Pixel Mask")
    hdr["DATE"]     = (Time.now().fits, "File creation date (UTC)")
    hdr["ORIGIN"]   = ("fits_tools",   "Software origin")
    hdr["BPMVERS"]  = (__version__,    "fits_tools BPM I/O version")
    hdr["BITDEPTH"] = (bpm.dtype.itemsize * 8, "Bits per pixel in mask")
    hdr["BPMDTYPE"] = (str(bpm.dtype),         "NumPy dtype of mask")

    # Overall defect counts
    bpm_stats = get_bpm_statistics(bpm)

    total_bad   = int(np.count_nonzero(bpm))
    total_frac  = total_bad / bpm.size

    hdr["NBADPIX"] = (total_bad,  "Total bad pixels")
    hdr["BADFRAC"] = (total_frac, "Fraction of bad pixels")

    # Per-defect-type counts via HIERARCH (one keyword per known bit name)
    for defect_name in FLAG_NAMES.values():
        entry = bpm_stats.get(defect_name, {})
        hdr[f"HIERARCH BPM.N.{defect_name}"] = (
            int(entry.get("count", 0)),
            f"Pixels flagged as {defect_name}",
        )

    # Embed bit definitions (HIERARCH summary; full table is in DQ_DEF extension)
    try:
        nbits   = _bpm_nbits_from_dtype(bpm.dtype)
        payload = get_definitions_payload(nbits=nbits)
        defs_hash = _sha256_defs(payload)
        _embed_bpm_definitions_hierarch(hdr, payload, defs_hash)
    except Exception as exc:
        logger.warning("save_bpm_to_fits: could not embed bit definitions: %s", exc)
        payload, defs_hash = None, None

    # Human-readable bit flag documentation as COMMENT cards
    hdr["COMMENT"] = "=" * 70
    hdr["COMMENT"] = "BIT FLAG DEFINITIONS"
    hdr["COMMENT"] = "=" * 70
    for line in format_bit_comments():
        hdr["COMMENT"] = line

    # Detection statistics (optional)
    if stats:
        hdr["COMMENT"] = "=" * 70
        hdr["COMMENT"] = "DETECTION PARAMETERS"
        hdr["COMMENT"] = "=" * 70

        bias_s = stats.get("bias", {})
        if bias_s:
            hdr["BIASTHR"] = (bias_s.get("threshold_offset_adu", np.nan), "Bias threshold (ADU)")
            hdr["BIASMED"] = (float(bias_s.get("median", np.nan)),         "Bias median (ADU)")
            hdr["BIASSTD"] = (float(bias_s.get("std",    np.nan)),         "Bias std dev (ADU)")

        dark_s = stats.get("dark", {})
        if dark_s:
            hdr["DARKSIG"] = (float(dark_s.get("threshold_sigma", np.nan)), "Dark threshold (sigma)")
            hdr["DARKMED"] = (float(dark_s.get("median",           np.nan)), "Dark median (ADU)")
            hdr["DARKSTD"] = (float(dark_s.get("std",              np.nan)), "Dark std dev (ADU)")

        flat_s = stats.get("flat", {})
        if flat_s:
            hdr["FLATHOT"]  = (float(flat_s.get("hot_threshold_percent",  np.nan)), "Flat hot threshold (%)")
            hdr["FLATCOLD"] = (float(flat_s.get("cold_threshold_percent", np.nan)), "Flat cold threshold (%)")
            hdr["FLATMED"]  = (float(flat_s.get("median",                 np.nan)), "Flat median (ADU)")

    # Caller-supplied cards
    if header_cards:
        for key, value in header_cards.items():
            hdr[key] = value  # accepts plain scalars or (value, comment) tuples

    # --- DQ_DEF binary table ---
    hdus = [hdu]
    if payload is not None:
        try:
            hdus.append(_build_bpmdefs_extension(payload, defs_hash))
        except Exception as exc:
            logger.warning("save_bpm_to_fits: could not build DQ_DEF extension: %s", exc)

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite, checksum=True)
    logger.info(
        "Saved BPM to %s (%d bad pixels, dtype=%s)",
        filename, bpm_stats["TOTAL"]["count"], bpm.dtype,
    )


# =============================================================================
# Load
# =============================================================================

def load_bpm_from_fits(
    filename: StrOrPath,
    ext: int = 0,
    validate: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Load a bad pixel mask from a FITS file.

    Parameters
    ----------
    filename : str or Path
        Path to the BPM FITS file.
    ext : int
        HDU index of the BPM array.  Default: 0 (primary).
    validate : bool
        If True, call :func:`validate_bpm_fits` after loading.

    Returns
    -------
    bpm : np.ndarray
        Bad pixel mask array.  Dtype is preserved from the file.
    metadata : dict
        Provenance and statistics extracted from the header::

            {
                'filename':        str,
                'date':            str,
                'instrume':        str,
                'bitdepth':        int,
                'total_bad':       int,
                'bad_fraction':    float,
                'defect_counts':   {name: count, ...},
                'detection_params': {key: value, ...},
            }

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If validation fails (when ``validate=True``).

    Examples
    --------
    >>> bpm, meta = load_bpm_from_fits('master_bpm.fits')
    >>> print(f"Total bad pixels: {meta['total_bad']}")

    See Also
    --------
    save_bpm_to_fits : Write a BPM in the format expected by this function.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"BPM file not found: {filename}")

    with fits.open(filename) as hdul:
        data   = hdul[ext].data
        header = hdul[ext].header

        if data is None:
            raise ValueError(f"HDU {ext} of {filename} contains no data.")

        bpm = data.copy()

        metadata: Dict = {
            "filename":     str(filename),
            "date":         header.get("DATE",     "Unknown"),
            "instrume":     header.get("INSTRUME", "Unknown"),
            "bitdepth":     header.get("BITDEPTH", bpm.dtype.itemsize * 8),
            "total_bad":    int(header.get("NBADPIX", int(np.sum(bpm > 0)))),
            "bad_fraction": float(header.get("BADFRAC",  float(np.sum(bpm > 0) / bpm.size))),
        }

        # Per-type counts
        defect_counts: Dict[str, int] = {}
        for name in FLAG_NAMES.values():
            key = f"HIERARCH BPM.N.{name}"
            if key in header:
                defect_counts[name] = int(header[key])
        metadata["defect_counts"] = defect_counts

        # Detection parameters
        detection_params: Dict[str, float] = {}
        for key in ("BIASTHR", "BIASMED", "BIASSTD",
                    "DARKSIG", "DARKMED", "DARKSTD",
                    "FLATHOT", "FLATCOLD", "FLATMED"):
            if key in header:
                detection_params[key] = header[key]
        metadata["detection_params"] = detection_params

    if validate:
        validate_bpm_fits(bpm, metadata)

    logger.info(
        "Loaded BPM from %s (%d bad pixels, dtype=%s)",
        filename, metadata["total_bad"], bpm.dtype,
    )
    return bpm, metadata


# =============================================================================
# Validation
# =============================================================================

def validate_bpm_fits(bpm: np.ndarray, metadata: Dict) -> None:
    """
    Validate BPM structure and consistency with its metadata.

    Parameters
    ----------
    bpm : np.ndarray
        BPM array to validate.
    metadata : dict
        Metadata dict as returned by :func:`load_bpm_from_fits`.

    Raises
    ------
    ValueError
        If the dtype is not unsigned integer, or the array is not 2-D.

    Notes
    -----
    A mismatch between the header defect count and the actual array count
    produces a warning rather than an error, because floating-point round-trip
    and partial reads can cause harmless discrepancies.
    """
    if not np.issubdtype(bpm.dtype, np.unsignedinteger):
        raise ValueError(
            f"BPM must have an unsigned integer dtype; got {bpm.dtype}. "
            f"The file may have been written incorrectly or converted after saving."
        )

    if bpm.ndim != 2:
        raise ValueError(
            f"BPM must be 2-D; got shape {bpm.shape}."
        )

    actual_bad   = int(np.sum(bpm > 0))
    reported_bad = int(metadata.get("total_bad", actual_bad))
    if actual_bad != reported_bad:
        logger.warning(
            "validate_bpm_fits: defect count mismatch — "
            "header reports %d but array contains %d bad pixels.",
            reported_bad, actual_bad,
        )

    logger.debug(
        "BPM validation passed: shape=%s, dtype=%s, bad_pixels=%d",
        bpm.shape, bpm.dtype, actual_bad,
    )


# =============================================================================
# Merge
# =============================================================================

def merge_bad_pixel_maps(
    existing_bpm_path: Optional[StrOrPath],
    new_defects_path:  StrOrPath,
    output_path:       StrOrPath,
    stage:             str = "UNKNOWN",
) -> Dict[str, Dict[str, float]]:
    """
    Merge an existing cumulative BPM with a stage-specific defect map.

    Uses bitwise OR so that every defect flag from both inputs is preserved
    and multiple flags can coexist on a single pixel.

    Parameters
    ----------
    existing_bpm_path : str, Path, or None
        Path to the cumulative master BPM.  If None or the file does not
        exist, only the new defects are used to create a fresh BPM.
    new_defects_path : str or Path
        Path to the stage-specific defect map produced by a detection function.
    output_path : str or Path
        Destination for the merged BPM.
    stage : str
        Name of the pipeline stage creating this update (e.g. ``'DARK'``).
        Recorded in the output header for provenance.  Default: ``'UNKNOWN'``.

    Returns
    -------
    dict
        Statistics for the merged BPM, as returned by
        :func:`~fits_tools.bitmask.get_bpm_statistics`.

    Raises
    ------
    ValueError
        If the new defects file cannot be loaded, or if the array shapes
        of the two inputs do not match.

    Examples
    --------
    >>> # First stage — create the initial master BPM from bias detections
    >>> merge_bad_pixel_maps(None, 'bias_defects.fits', 'master_bpm.fits', stage='BIAS')
    >>>
    >>> # Second stage — accumulate dark detections
    >>> merge_bad_pixel_maps('master_bpm.fits', 'dark_defects.fits',
    ...                      'master_bpm.fits', stage='DARK')

    See Also
    --------
    save_bpm_to_fits : Underlying save used to write the merged output.
    """
    new_defects_path = Path(new_defects_path)
    output_path      = Path(output_path)

    # Load new defects — required
    new_bpm, new_header_meta = load_bpm_from_fits(new_defects_path, validate=False)

    # Load existing BPM — optional
    if existing_bpm_path is not None:
        existing_bpm_path = Path(existing_bpm_path)

    if existing_bpm_path and existing_bpm_path.exists():
        existing_bpm, _ = load_bpm_from_fits(existing_bpm_path, validate=False)
        if existing_bpm.shape != new_bpm.shape:
            raise ValueError(
                f"Shape mismatch: existing BPM {existing_bpm.shape} "
                f"vs new defects {new_bpm.shape}."
            )
    else:
        existing_bpm = np.zeros_like(new_bpm)
        logger.debug("merge_bad_pixel_maps: no existing BPM found; starting fresh.")

    # Merge
    merged_bpm = existing_bpm | new_bpm

    # Statistics
    stats_before = get_bpm_statistics(existing_bpm)
    stats_new    = get_bpm_statistics(new_bpm)
    stats_after  = analyze_bpm_statistics(merged_bpm)

    n_before = int(np.count_nonzero(existing_bpm))
    n_new    = int(np.count_nonzero(new_bpm))
    n_after  = int(np.count_nonzero(merged_bpm))
    n_added  = n_after - n_before

    logger.debug(
        "merge_bad_pixel_maps [%s]: existing=%d  new=%d  merged=%d  added=%d",
        stage, n_before, n_new, n_after, n_added,
    )

    # Build BPMSRC from defect types that contributed new pixels
    contributing = [
        name for name, info in stats_new.items()
        if name != "TOTAL" and int(info.get("count", 0)) > 0
    ]
    stage_upper = stage.upper()
    if stage_upper not in contributing:
        contributing.insert(0, stage_upper)
    bpm_source = "+".join(contributing) if contributing else stage_upper

    # Save
    extra_cards = {
        "BPMTYPE":  ("CUMULATIVE",         "Cumulative bad pixel map"),
        "BPMSRC":   (bpm_source[:68],      "Defect sources included"),
        "BPMSTAGE": (stage_upper,          "Last stage to update this map"),
        "NNEW":     (n_added,              f"New pixels flagged by {stage}"),
    }
    save_bpm_to_fits(merged_bpm, output_path, header_cards=extra_cards, overwrite=True)

    return stats_after


# =============================================================================
# Save calibration set
# =============================================================================

def save_calibration_bpm(
    bpm:          np.ndarray,
    master_bias:  np.ndarray,
    master_dark:  np.ndarray,
    master_flat:  np.ndarray,
    stats:        Dict,
    output_prefix: StrOrPath,
    overwrite:    bool = True,
) -> Dict[str, str]:
    """
    Save a complete calibration set — BPM and master frames — to FITS files.

    All output files are linked: the master frame headers carry a ``BPMFILE``
    keyword pointing to the BPM, so the provenance chain is preserved on disk.

    Parameters
    ----------
    bpm : np.ndarray
        Bad pixel mask.
    master_bias, master_dark, master_flat : np.ndarray
        Master calibration frames (bias-subtracted dark, bias+dark-corrected flat).
    stats : dict
        Detection statistics returned by ``create_calibration_bpm``.
    output_prefix : str or Path
        Prefix for all output filenames.  The BPM will be saved as
        ``{prefix}_bpm.fits`` and masters as ``{prefix}_master_{type}.fits``.
    overwrite : bool
        Overwrite existing files.  Default: True.

    Returns
    -------
    dict
        Maps frame type to filename::

            {'bpm': '...', 'bias': '...', 'dark': '...', 'flat': '...'}

    Examples
    --------
    >>> bpm, stats = create_calibration_bpm(bias, dark, flat)
    >>> files = save_calibration_bpm(
    ...     bpm, master_bias, master_dark, master_flat,
    ...     stats, output_prefix='calib_20230101'
    ... )

    See Also
    --------
    save_bpm_to_fits : Underlying BPM writer.
    """
    output_prefix = Path(output_prefix)
    files: Dict[str, str] = {}

    # Save BPM
    bpm_filename = str(output_prefix) + "_bpm.fits"
    save_bpm_to_fits(bpm, bpm_filename, stats=stats, overwrite=overwrite)
    files["bpm"] = bpm_filename

    # Save master frames
    for frame, frame_type in [
        (master_bias, "bias"),
        (master_dark, "dark"),
        (master_flat, "flat"),
    ]:
        filename = f"{output_prefix}_master_{frame_type}.fits"
        hdu = fits.PrimaryHDU(frame)
        hdr = hdu.header
        hdr["IMAGETYP"] = (f"MASTER_{frame_type.upper()}", "Image type")
        hdr["DATE"]     = (Time.now().fits,                 "File creation date (UTC)")
        hdr["ORIGIN"]   = ("fits_tools",                    "Software origin")
        hdr["BPMFILE"]  = (bpm_filename,                    "Associated bad pixel mask")
        fits.HDUList([hdu]).writeto(filename, overwrite=overwrite, checksum=True)
        files[frame_type] = filename
        logger.info("Saved master %s to %s", frame_type, filename)

    return files


# =============================================================================
# Analysis with logging
# =============================================================================

def analyze_bpm_statistics(
    bpm_array: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute defect statistics for a BPM, optionally logging the results.

    This is a thin wrapper around :func:`~fits_tools.bitmask.get_bpm_statistics`
    that adds logging support for pipeline reporting.  Use the underlying
    function directly when logging is not needed.

    Parameters
    ----------
    bpm_array : np.ndarray
        Bad pixel mask array.

    Returns
    -------
    dict
        Per-defect-type statistics as returned by
        :func:`fits_tools.bitmask.get_bpm_statistics`.

    Examples
    --------
    >>> stats = analyze_bpm_statistics(bpm)
    >>> print(f"Hot pixels: {stats['PIX_HOT']['count']}")
    """
    stats = get_bpm_statistics(bpm_array)

    total_bad  = int(np.count_nonzero(bpm_array))
    total_frac = total_bad / bpm_array.size
    logger.debug("Bad pixel map statistics:")
    logger.debug("  Total bad pixels: %d (%.3f%%)", total_bad, total_frac * 100)

    # Log only known individual bit names (FLAG_NAMES)
    for defect_name in FLAG_NAMES.values():
        entry = stats.get(defect_name)
        if entry is None:
            # This can happen if a bit is defined but unused.
            continue
        count = int(entry["count"])
        frac  = float(entry["fraction"])
        if count > 0:
            logger.debug("  %-12s: %6d  (%.3f%%)", defect_name, count, frac * 100)

    return stats
