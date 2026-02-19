#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-frame artifact detection for CCD/CMOS images.

Detection Philosophy
--------------------
Artifacts are transient, exposure-specific events.  Each function in this
module operates on a **single science frame** and returns a DQ array for that
frame only.  Results must not be merged into the master bad pixel map (which
describes permanent sensor properties); instead they belong in a per-frame
Data Quality (DQ) extension.

A typical reduction step::

    master_bpm, _ = load_bpm_from_fits('master_bpm.fits')

    cr_dq,  cr_stats  = detect_cosmic_rays(science, existing_bpm=master_bpm)
    sat_dq, sat_stats = detect_saturated_pixels(science)

    frame_dq = master_bpm | cr_dq | sat_dq   # full per-frame DQ array

The ``existing_bpm`` parameter accepted by several functions serves one
purpose only: to exclude permanently bad pixels from the statistics used to
set detection thresholds.  It does not suppress flagging — if a pixel is
both a known hot pixel and hit by a cosmic ray in this frame, both bits will
be set in the returned DQ array.

Implemented
-----------
``detect_saturated_pixels``
    Flags pixels at or above the ADC saturation level.
    Flags: ``BPM_BIT_SATURATED``

``detect_cosmic_rays``
    Sigma-clipping (local median residual) and L.A.Cosmic (Laplacian edge
    detection) methods.
    Flags: ``BPM_BIT_COSMIC``

Stubbed (not yet implemented)
------------------------------
``detect_bloom``
    Blooming / bleed trails from saturated bright sources.
    Flags: ``BPM_BIT_BLOOM``

``detect_crosstalk``
    Electronic crosstalk between amplifier channels.  Highly
    instrument-specific; requires channel geometry information.
    Flags: ``BPM_BIT_CROSSTALK``

Performance notes
-----------------
- Sigma-clip cosmic ray detection: O(N), ~20-50 ms for a 4K × 4K frame
- L.A.Cosmic detection: O(N), ~100-500 ms (depends on astroscrappy build)
- Both methods are dominated by the median filter pass

Dependencies
------------
- ``scipy`` (required for sigma-clip method)
- ``astroscrappy`` (optional, required for L.A.Cosmic method)

Version: 1.0.0
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

# Optional dependencies — checked at call time, not import time
try:
    from scipy.ndimage import median_filter as _scipy_median_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_median_filter = None
    _SCIPY_AVAILABLE = False

try:
    import astroscrappy as _astroscrappy
    _LACOSMIC_AVAILABLE = True
except ImportError:
    _astroscrappy = None
    _LACOSMIC_AVAILABLE = False

from ..bitmask import (
    BPM_BIT_SATURATED,
    BPM_BIT_COSMIC,
    BPM_BIT_BLOOM,
    BPM_BIT_CROSSTALK,
    create_bpm,
)
from ..defects.detection import sigma_clip_stats

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

_MAD_TO_STD = 1.4826  # Conversion factor for normal distribution

__all__ = [
    'detect_saturated_pixels',
    'detect_cosmic_rays',
    'detect_bloom',
    'detect_crosstalk',
]


# =============================================================================
# Saturation
# =============================================================================

def detect_saturated_pixels(
    frame: np.ndarray,
    saturation_threshold: float = 0.95,
    bit_depth: int = 16,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Flag pixels at or above the ADC saturation level in a single frame.

    Saturation is an exposure-specific condition — a pixel that saturates
    on a bright star is perfectly functional in other frames.  The returned
    DQ array belongs to this frame only and must not be merged into the
    master bad pixel map.

    Parameters
    ----------
    frame : np.ndarray
        Science frame to inspect (ADU).
    saturation_threshold : float
        Fraction of the ADC full-well to treat as saturated.  Default: 0.95.
        Values above ``(2^bit_depth - 1) × saturation_threshold`` are flagged.
    bit_depth : int
        ADC bit depth that determines the maximum possible ADU value.
        Default: 16 (max = 65535).
    bpm_dtype : np.dtype
        Dtype of the output DQ array.  Default: np.uint32.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_SATURATED`` set where applicable.
    stats : dict
        Detection statistics::

            {
                'bit_depth':            int,
                'max_value':            int,
                'saturation_threshold': float,
                'saturation_level':     float,
                'num_saturated':        int,
                'fraction_saturated':   float,
            }

    Raises
    ------
    ValueError
        If ``saturation_threshold`` is not in (0, 1].

    Examples
    --------
    >>> dq, stats = detect_saturated_pixels(science_frame, bit_depth=16)
    >>> print(f"Saturated pixels: {stats['num_saturated']}")
    """
    if not (0.0 < saturation_threshold <= 1.0):
        raise ValueError(
            f"saturation_threshold must be in (0, 1]; got {saturation_threshold!r}"
        )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)

    max_value = (2 ** bit_depth) - 1
    sat_level = max_value * saturation_threshold
    sat_mask  = frame >= sat_level

    dq[sat_mask] |= BPM_BIT_SATURATED

    stats = {
        'bit_depth':            int(bit_depth),
        'max_value':            int(max_value),
        'saturation_threshold': float(saturation_threshold),
        'saturation_level':     float(sat_level),
        'num_saturated':        int(np.sum(sat_mask)),
        'fraction_saturated':   float(np.sum(sat_mask)) / frame.size,
    }

    return dq, stats


# =============================================================================
# Cosmic rays
# =============================================================================

def detect_cosmic_rays(
    frame: np.ndarray,
    method: str = 'sigma_clip',
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
    # Sigma-clip parameters
    sigma: float = 5.0,
    box_size: int = 5,
    iterations: int = 3,
    # L.A.Cosmic parameters
    sigclip: float = 4.5,
    sigfrac: float = 0.3,
    objlim: float = 5.0,
    gain: float = 1.0,
    readnoise: float = 6.5,
    lacosmic_iterations: int = 4,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect cosmic ray hits in a single science frame.

    Two methods are available:

    ``'sigma_clip'``
        Compares each pixel to the local median (computed with a box filter)
        and flags positive outliers beyond ``sigma`` × MAD of the residuals.
        Fast and dependency-light (requires only scipy).  Works well for
        isolated, point-like cosmic rays.  Less effective for extended tracks.

    ``'lacosmic'``
        Laplacian edge detection algorithm (van Dokkum 2001), implemented by
        the ``astroscrappy`` package.  More sensitive to extended tracks and
        better at separating cosmic rays from stars when gain and readnoise
        are well-characterised.  Requires ``astroscrappy`` to be installed.

    Only detection is performed here.  To replace cosmic ray pixels with
    interpolated values, use ``image_processing.cosmic_rays.reject_cosmic_rays``
    with the boolean mask derived from the returned DQ array.

    Parameters
    ----------
    frame : np.ndarray
        Science frame (bias- and dark-subtracted, ADU or e⁻).
    method : str
        Detection algorithm: ``'sigma_clip'`` or ``'lacosmic'``.
        Default: ``'sigma_clip'``.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask.  Permanently bad pixels are excluded from the
        residual statistics used to set the detection threshold, preventing
        hot pixels from inflating the noise floor.  Any pixel that is
        flagged in the returned DQ array may also be in the master BPM —
        the two are independent and should be OR'd by the caller.
    bpm_dtype : np.dtype
        Dtype of the output DQ array.  Default: np.uint32.

    Sigma-clip parameters (``method='sigma_clip'``)
    ------------------------------------------------
    sigma : float
        Detection threshold above the local median residual in units of
        the robust MAD-derived sigma.  Default: 5.0.
    box_size : int
        Size of the median filter box (forced to odd if even given).
        Larger boxes smooth more aggressively but can miss clustered hits.
        Default: 5.
    iterations : int
        Number of sigma-clip iterations.  Detected pixels are replaced with
        the local median before the next pass, allowing extended tracks to
        be detected progressively.  Default: 3.

    L.A.Cosmic parameters (``method='lacosmic'``)
    ----------------------------------------------
    sigclip : float
        Laplacian-to-noise threshold.  Default: 4.5.
    sigfrac : float
        Fractional detection limit for neighbouring pixels.  Default: 0.3.
    objlim : float
        Minimum contrast between Laplacian image and fine-structure image.
        Increase to avoid flagging bright stars.  Default: 5.0.
    gain : float
        CCD gain in e⁻/ADU.  Default: 1.0.
    readnoise : float
        Read noise in e⁻.  Default: 6.5.
    lacosmic_iterations : int
        Number of L.A.Cosmic iterations.  Default: 4.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_COSMIC`` set for detected pixels.
    stats : dict
        Detection statistics.  Keys depend on method; common keys::

            {
                'method':           str,
                'num_cosmic_rays':  int,
                'fraction':         float,
                'iterations':       int,
            }

    Raises
    ------
    ImportError
        If the chosen method's required package is not installed.
    ValueError
        If ``method`` is not recognised.

    Examples
    --------
    >>> dq, stats = detect_cosmic_rays(science_frame, method='sigma_clip',
    ...                                sigma=5.0, existing_bpm=master_bpm)
    >>> print(f"Cosmic ray pixels: {stats['num_cosmic_rays']}")
    >>>
    >>> # Use the boolean mask for cleaning
    >>> cr_mask = (dq & BPM_BIT_COSMIC) != 0
    >>> from fits_tools.image_processing.cosmic_rays import reject_cosmic_rays
    >>> result = reject_cosmic_rays(science_frame, mask=cr_mask)

    References
    ----------
    .. [1] van Dokkum, P. G. (2001). "Cosmic-Ray Rejection by Laplacian Edge
           Detection." PASP, 113, 1420.
    """
    method = str(method).lower()
    if method not in {'sigma_clip', 'lacosmic'}:
        raise ValueError(
            f"method must be 'sigma_clip' or 'lacosmic'; got {method!r}"
        )

    if method == 'sigma_clip':
        return _detect_cosmic_rays_sigma_clip(
            frame=frame,
            existing_bpm=existing_bpm,
            sigma=sigma,
            box_size=box_size,
            iterations=iterations,
            bpm_dtype=bpm_dtype,
        )
    else:
        return _detect_cosmic_rays_lacosmic(
            frame=frame,
            existing_bpm=existing_bpm,
            sigclip=sigclip,
            sigfrac=sigfrac,
            objlim=objlim,
            gain=gain,
            readnoise=readnoise,
            iterations=lacosmic_iterations,
            bpm_dtype=bpm_dtype,
        )


def _detect_cosmic_rays_sigma_clip(
    frame: np.ndarray,
    existing_bpm: Optional[np.ndarray],
    sigma: float,
    box_size: int,
    iterations: int,
    bpm_dtype: np.dtype,
) -> Tuple[np.ndarray, Dict]:
    """
    Sigma-clip cosmic ray detection using local median residuals.

    For each iteration:
    1. Compute local median with a box filter.
    2. Compute per-pixel residual (frame - local_median).
    3. Compute robust MAD of residuals, excluding existing bad pixels.
    4. Flag positive outliers beyond sigma × MAD.
    5. Replace flagged pixels with the local median before the next pass
       so that extended tracks are detected progressively.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "detect_cosmic_rays with method='sigma_clip' requires scipy. "
            "Install with: pip install scipy"
        )

    if box_size % 2 == 0:
        box_size += 1
        logger.debug("detect_cosmic_rays: box_size adjusted to %d (must be odd)", box_size)

    dq       = create_bpm(frame.shape, dtype=bpm_dtype)
    work     = frame.astype(float, copy=True)
    cr_mask  = np.zeros(frame.shape, dtype=bool)

    # Build the exclusion mask for statistics: existing bad pixels.
    # We do NOT use this to suppress new detections — a hot pixel can also
    # be hit by a cosmic ray and should receive both bits.
    stat_mask = (existing_bpm > 0) if existing_bpm is not None else None

    iters_run = 0
    for iteration in range(iterations):
        local_median = _scipy_median_filter(work, size=box_size)
        residuals    = work - local_median

        # Robust scale of residuals, excluding known-bad pixels
        _, mad_sigma = sigma_clip_stats(residuals, mask=stat_mask, scale='mad')

        if not np.isfinite(mad_sigma) or mad_sigma <= 0:
            logger.debug(
                "detect_cosmic_rays (sigma_clip): degenerate MAD at iteration %d; stopping.",
                iteration + 1,
            )
            break

        # Threshold on the positive tail only — cosmic rays are always spikes
        threshold  = float(sigma) * mad_sigma
        new_hits   = (residuals > threshold) & ~cr_mask

        num_new = int(np.sum(new_hits))
        logger.debug(
            "detect_cosmic_rays (sigma_clip): iteration %d — %d new pixels flagged",
            iteration + 1, num_new,
        )

        cr_mask |= new_hits

        # Replace newly flagged pixels with local median before next pass
        work[new_hits] = local_median[new_hits]

        iters_run += 1
        if num_new == 0:
            break

    dq[cr_mask] |= BPM_BIT_COSMIC

    stats = {
        'method':          'sigma_clip',
        'sigma':           float(sigma),
        'box_size':        int(box_size),
        'iterations':      iters_run,
        'num_cosmic_rays': int(np.sum(cr_mask)),
        'fraction':        float(np.sum(cr_mask)) / frame.size,
    }
    return dq, stats


def _detect_cosmic_rays_lacosmic(
    frame: np.ndarray,
    existing_bpm: Optional[np.ndarray],
    sigclip: float,
    sigfrac: float,
    objlim: float,
    gain: float,
    readnoise: float,
    iterations: int,
    bpm_dtype: np.dtype,
) -> Tuple[np.ndarray, Dict]:
    """
    L.A.Cosmic cosmic ray detection via astroscrappy.

    The ``existing_bpm`` is passed to astroscrappy as an ``inmask`` so that
    permanently bad pixels are excluded from the Laplacian edge image
    statistics.  Again, this does not suppress detection — it only improves
    the noise model.
    """
    if not _LACOSMIC_AVAILABLE:
        raise ImportError(
            "detect_cosmic_rays with method='lacosmic' requires astroscrappy. "
            "Install with: pip install astroscrappy"
        )

    # astroscrappy expects a C-contiguous float32 array
    frame_f32 = np.ascontiguousarray(frame, dtype=np.float32)

    # Build input mask: True = ignore this pixel (existing bad pixels)
    if existing_bpm is not None:
        inmask = (existing_bpm > 0).astype(bool)
    else:
        inmask = None

    cr_mask_bool, _ = _astroscrappy.detect_cosmics(
        frame_f32,
        inmask=inmask,
        sigclip=float(sigclip),
        sigfrac=float(sigfrac),
        objlim=float(objlim),
        gain=float(gain),
        readnoise=float(readnoise),
        niter=int(iterations),
        cleantype='median',   # astroscrappy needs a cleantype even for detection-only
        fsmode='median',
    )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)
    dq[cr_mask_bool] |= BPM_BIT_COSMIC

    stats = {
        'method':          'lacosmic',
        'sigclip':         float(sigclip),
        'sigfrac':         float(sigfrac),
        'objlim':          float(objlim),
        'gain':            float(gain),
        'readnoise':       float(readnoise),
        'iterations':      int(iterations),
        'num_cosmic_rays': int(np.sum(cr_mask_bool)),
        'fraction':        float(np.sum(cr_mask_bool)) / frame.size,
    }
    return dq, stats


# =============================================================================
# Stubs
# =============================================================================

def detect_bloom(
    frame: np.ndarray,
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect blooming and bleed trails from saturated sources.

    Blooming occurs when a pixel overfills and excess charge bleeds into
    adjacent pixels along the serial register direction.  Detection requires
    knowledge of the saturation level, the bleed direction, and typically
    a source catalogue or saturation map to seed the trail search.

    Parameters
    ----------
    frame : np.ndarray
        Science frame (ADU).
    existing_bpm : np.ndarray, optional
        Master bad pixel mask (used to exclude known defects from statistics).
    bpm_dtype : np.dtype
        Dtype of the output DQ array.  Default: np.uint32.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_BLOOM`` set for detected pixels.
    stats : dict
        Detection statistics.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Notes
    -----
    Implementation will require at minimum:

    - Saturation level (ADU or fraction of full-well)
    - Bleed direction (typically parallel to columns for front-illuminated CCDs)
    - A connected-component or run-length search seeded from saturated pixels
    - Threshold on trail length / brightness to separate true blooms from
      charge-transfer inefficiency artefacts
    """
    raise NotImplementedError(
        "detect_bloom is not yet implemented.  "
        "Planned approach: seed from saturated pixels (detect_saturated_pixels), "
        "then grow connected components along the bleed direction."
    )


def detect_crosstalk(
    frame: np.ndarray,
    channel_geometry: Dict,
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect electronic crosstalk between amplifier channels.

    Crosstalk appears as a faint ghost of a bright source in one channel
    mirrored or offset into an adjacent channel.  The geometry (number of
    channels, their layout on the detector, the crosstalk coefficient matrix)
    is entirely instrument-specific.

    Parameters
    ----------
    frame : np.ndarray
        Science frame (ADU).
    channel_geometry : dict
        Instrument-specific description of the amplifier channel layout and
        crosstalk coefficients.  The required keys are not yet defined pending
        a concrete instrument use case.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask.
    bpm_dtype : np.dtype
        Dtype of the output DQ array.  Default: np.uint32.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_CROSSTALK`` set for detected pixels.
    stats : dict
        Detection statistics.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Notes
    -----
    Implementation will require at minimum:

    - Channel boundary definitions (which rows/columns belong to which amp)
    - A crosstalk coefficient matrix (N_channels × N_channels)
    - A source detection step to identify bright sources that generate
      detectable crosstalk in adjacent channels
    - Threshold on predicted crosstalk signal vs. background noise
    """
    raise NotImplementedError(
        "detect_crosstalk is not yet implemented.  "
        "This function is instrument-specific and requires channel_geometry "
        "to be defined for the target detector before implementation can proceed."
    )
