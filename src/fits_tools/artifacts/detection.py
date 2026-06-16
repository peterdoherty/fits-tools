#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Per-frame artifact detection for CCD/CMOS images.

Detection Philosophy
--------------------
Artifacts are transient, exposure-specific events. Each function operates
on a **single science frame** and returns a DQ array for that frame only.
Results must not be merged into the master bad pixel map (which describes
permanent sensor properties); they belong in a per-frame Data Quality (DQ)
extension.

A typical reduction step::

    master_bpm, _ = load_bpm_from_fits("master_bpm.fits")

    cr_dq,    cr_stats    = detect_cosmic_rays(science, existing_bpm=master_bpm)
    sat_dq,   sat_stats   = detect_saturated_pixels(science)
    bloom_dq, bloom_stats = detect_bloom(science)

    frame_dq = master_bpm | cr_dq | sat_dq | bloom_dq

The ``existing_bpm`` parameter accepted by several functions serves one
purpose: to exclude permanently bad pixels from the statistics used to set
detection thresholds. It does not suppress flagging — if a pixel is both a
known hot pixel and hit by a cosmic ray, both bits will be set in the
returned DQ array.

Implemented — Core Artifacts
-----------------------------
- :func:`detect_saturated_pixels` — flags ``BPM_BIT_SATURATED``
- :func:`detect_cosmic_rays` — flags ``BPM_BIT_COSMIC``
- :func:`detect_bloom` — flags ``BPM_BIT_BLOOM``
- :func:`detect_persistence` — flags ``BPM_BIT_OUTLIER``
- :func:`detect_statistical_outliers` — flags ``BPM_BIT_OUTLIER``
- :func:`flag_outliers` — encodes external outlier masks as ``BPM_BIT_OUTLIER``

Implemented — Lab Metrology
----------------------------
- :func:`detect_fringing` — FFT-based thin-film interference detection
- :func:`detect_fixed_pattern_noise` — banding/checkerboard from readout electronics
- :func:`detect_thermal_gradient` — large-scale temperature gradients (QC metric)

Integration & Utilities
-----------------------
- :func:`detect_all_artifacts` — full suite with configurable parameters
- :func:`get_default_artifact_config` — default configuration dictionary
- :func:`align_frames_cross_correlation` — sub-pixel frame registration
- :func:`create_edge_mask_from_shift` — mask unreliable edge pixels after shift
- :func:`interpolate_bad_pixels` — smooth bad-pixel fill before FFT
- :func:`convert_readnoise_to_adu` — unit conversion utility

Stubbed (Not Yet Implemented)
------------------------------
- :func:`detect_crosstalk` — requires instrument channel geometry definition

Statistics Dictionary Conventions
----------------------------------
All detection functions return ``(dq, stats)`` where ``stats`` is a dict
with detection metadata. Common keys across detectors:

- ``"fraction"`` — fraction of frame affected
- ``"num_*"`` — pixel counts for the relevant feature
- ``"threshold"`` or ``"threshold_sigma"`` — detection threshold used
- ``"method"`` — algorithm variant (where multiple are available)

Detectors may include additional diagnostic keys appropriate to their
algorithm; see individual function docstrings for details.

Performance
-----------
Typical processing times for a 4K × 4K frame (16 Mpixels):

=================================  ============  ========================
Operation                          Time          Notes
=================================  ============  ========================
``detect_saturated_pixels``        5–10 ms       Vectorized threshold
``detect_cosmic_rays`` (sigma)     20–50 ms      Dominated by median filter
``detect_cosmic_rays`` (lacosmic)  100–500 ms    Laplacian computation
``detect_bloom``                   10–30 ms      Vectorized trail growth
``detect_persistence``             20–40 ms      Difference + CCL
``detect_fringing``                50–200 ms     FFT analysis
``detect_fixed_pattern_noise``     30–100 ms     Temporal statistics
``detect_thermal_gradient``        15–25 ms      Adaptive downsampling
``detect_all_artifacts`` (full)    150–400 ms    Sum of enabled detectors
=================================  ============  ========================

Dependencies
------------
**Required:** numpy, scipy

**Optional:**

- ``astroscrappy`` — L.A.Cosmic cosmic ray detection
- ``scikit-image`` — sub-pixel frame registration
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import ndimage

try:
    import astroscrappy as _astroscrappy
    _LACOSMIC_AVAILABLE = True
except ImportError:
    _astroscrappy = None
    _LACOSMIC_AVAILABLE = False

from ..bitmask import (
    BPM_BIT_BLOOM,
    BPM_BIT_COSMIC,
    BPM_BIT_CROSSTALK,
    BPM_BIT_OUTLIER,
    BPM_BIT_SATURATED,
    create_bpm,
)
from ..defects.detection import sigma_clip_stats
from ..types import StrOrPath

logger = logging.getLogger(__name__)

__all__ = [
    # Core artifact detectors
    "detect_saturated_pixels",
    "detect_cosmic_rays",
    "detect_bloom",
    "detect_crosstalk",
    "detect_statistical_outliers",
    "detect_persistence",
    "flag_outliers",
    # Lab-specific detectors
    "detect_fringing",
    "detect_fixed_pattern_noise",
    "detect_thermal_gradient",
    # Integration
    "detect_all_artifacts",
    "get_default_artifact_config",
    # Utilities
    "interpolate_bad_pixels",
    "create_edge_mask_from_shift",
    "align_frames_cross_correlation",
    "convert_readnoise_to_adu",
]

# =============================================================================
# Module-level constants
# =============================================================================

# Target point count for thermal gradient plane fitting (~50k provides
# 16,000× overdetermination for the 3-parameter linear model while keeping
# lstsq fast at ~20 ms).
_THERMAL_FIT_TARGET_POINTS: int = 50_000

# Maximum downsampling factor for thermal gradient detection. Capped at 21×
# to ensure at least ~400×400 sampling for 8K sensors (100× overdetermined).
# Odd values are preferred to avoid Nyquist ambiguity at factor/2 frequency.
_THERMAL_FIT_MAX_DOWNSAMPLE: int = 21


# =============================================================================
# Saturation
# =============================================================================


def detect_saturated_pixels(
    frame: np.ndarray,
    saturation_threshold: float = 0.95,
    bit_depth: int = 16,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Flag pixels at or above the ADC saturation level.

    Saturation is exposure-specific — a pixel that saturates on a bright
    star is perfectly functional in other frames. The returned DQ array
    belongs to this frame only and must not be merged into the master BPM.

    Parameters
    ----------
    frame : np.ndarray
        Science frame to inspect (ADU).
    saturation_threshold : float, optional
        Fraction of ADC full-well to treat as saturated. Pixels above
        ``(2^bit_depth - 1) × saturation_threshold`` are flagged.
        Default is 0.95.
    bit_depth : int, optional
        ADC bit depth determining the maximum ADU value. Default is 16
        (max = 65535).
    bpm_dtype : np.dtype, optional
        Dtype of the output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_SATURATED`` set where applicable.
    stats : dict
        Keys: ``'bit_depth'``, ``'max_value'``, ``'saturation_threshold'``,
        ``'saturation_level'``, ``'num_saturated'``, ``'fraction_saturated'``.

    Raises
    ------
    ValueError
        If ``saturation_threshold`` is not in (0, 1].
    """
    if not (0.0 < saturation_threshold <= 1.0):
        raise ValueError(
            f"saturation_threshold must be in (0, 1]; "
            f"got {saturation_threshold!r}."
        )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)
    max_value = (2 ** bit_depth) - 1
    sat_level = max_value * saturation_threshold
    sat_mask  = frame >= sat_level
    dq[sat_mask] |= BPM_BIT_SATURATED

    stats: dict[str, Any] = {
        "bit_depth":            int(bit_depth),
        "max_value":            int(max_value),
        "saturation_threshold": float(saturation_threshold),
        "saturation_level":     float(sat_level),
        "num_saturated":        int(np.sum(sat_mask)),
        "fraction_saturated":   float(np.sum(sat_mask)) / frame.size,
    }
    return dq, stats


# =============================================================================
# Cosmic rays
# =============================================================================


def convert_readnoise_to_adu(
    readnoise_electrons: float,
    gain_e_per_adu: float,
) -> float:
    """Convert read noise from electrons to ADU.

    Parameters
    ----------
    readnoise_electrons : float
        Sensor read noise in electrons (e⁻ RMS).
    gain_e_per_adu : float
        Detector gain in electrons per ADU (e⁻/ADU).

    Returns
    -------
    float
        Read noise in ADU, suitable for ``detect_cosmic_rays(min_sigma=...)``.

    Raises
    ------
    ValueError
        If *gain_e_per_adu* is not positive or *readnoise_electrons* is
        negative.

    Notes
    -----
    The conversion is ``noise_adu = noise_electrons / gain_e_per_adu``.
    Do not multiply by gain — that is the inverse conversion.

    Examples
    --------
    >>> min_sigma = convert_readnoise_to_adu(readnoise_electrons=5.0,
    ...                                      gain_e_per_adu=1.5)
    >>> print(f"min_sigma = {min_sigma:.2f} ADU")
    min_sigma = 3.33 ADU
    """
    if gain_e_per_adu <= 0:
        raise ValueError(f"gain must be positive; got {gain_e_per_adu}.")
    if readnoise_electrons < 0:
        raise ValueError(
            f"readnoise must be non-negative; got {readnoise_electrons}."
        )
    return float(readnoise_electrons / gain_e_per_adu)


def detect_cosmic_rays(
    frame: np.ndarray,
    method: str = "sigma_clip",
    existing_bpm: np.ndarray | None = None,
    bpm_dtype: np.dtype = np.uint32,
    # Sigma-clip parameters
    sigma: float = 5.0,
    box_size: int = 5,
    iterations: int = 3,
    min_sigma: float | None = None,
    # L.A.Cosmic parameters
    sigclip: float = 4.5,
    sigfrac: float = 0.3,
    objlim: float = 5.0,
    gain: float = 1.0,
    readnoise: float = 6.5,
    lacosmic_iterations: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect cosmic ray hits in a single science frame.

    Parameters
    ----------
    frame : np.ndarray
        Science frame (bias- and dark-subtracted, ADU or e⁻).
    method : {"sigma_clip", "lacosmic"}, optional
        Detection algorithm. Default is ``"sigma_clip"``.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask. Permanently bad pixels are excluded from
        residual statistics to prevent them from inflating the noise floor.
        Detection is not suppressed — a pixel may be flagged in both the
        master BPM and the returned DQ array.
    bpm_dtype : np.dtype, optional
        Dtype of the output DQ array. Default is ``np.uint32``.
    sigma : float, optional
        Sigma-clip threshold above local median residual. Default is 5.0.
    box_size : int, optional
        Median filter box size (forced to odd). Default is 5.
    iterations : int, optional
        Sigma-clip iteration count. Default is 3.
    min_sigma : float, optional
        Minimum noise floor in ADU. Use when MAD of residuals may be zero
        (noiseless synthetic data or very low-noise calibration frames).
        Convert from electrons via :func:`convert_readnoise_to_adu`.
        Default is None (uses 1e-10 ADU).
    sigclip : float, optional
        L.A.Cosmic Laplacian-to-noise threshold. Default is 4.5.
    sigfrac : float, optional
        L.A.Cosmic fractional detection limit for neighbours. Default is 0.3.
    objlim : float, optional
        L.A.Cosmic minimum contrast vs fine-structure image. Increase to
        avoid flagging bright stars. Default is 5.0.
    gain : float, optional
        CCD gain in e⁻/ADU for L.A.Cosmic. Default is 1.0.
    readnoise : float, optional
        Read noise in e⁻ for L.A.Cosmic. Default is 6.5.
    lacosmic_iterations : int, optional
        Number of L.A.Cosmic iterations. Default is 4.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_COSMIC`` set for detected pixels.
    stats : dict
        Detection statistics. Common keys: ``"method"``,
        ``"num_cosmic_rays"``, ``"fraction"``, ``"iterations"``.

    Raises
    ------
    ImportError
        If ``method="lacosmic"`` and ``astroscrappy`` is not installed.
    ValueError
        If *method* is not recognised.

    Notes
    -----
    **sigma_clip**: Compares each pixel to its local median (box filter) and
    flags positive outliers beyond ``sigma × MAD`` of the residuals. Fast,
    dependency-light. Works well for isolated point-like hits.

    **lacosmic**: Laplacian edge detection (van Dokkum 2001) via
    ``astroscrappy``. More sensitive to extended tracks and better at
    separating cosmic rays from stars when gain and readnoise are known.

    For cosmic ray *replacement* (interpolation), use
    ``image_processing.cosmic_rays.reject_cosmic_rays`` with the boolean
    mask derived from the returned DQ array.

    References
    ----------
    .. [1] van Dokkum, P. G. (2001). PASP, 113, 1420.
    """
    method = str(method).lower()
    if method not in {"sigma_clip", "lacosmic"}:
        raise ValueError(
            f"method must be 'sigma_clip' or 'lacosmic'; got {method!r}."
        )

    if method == "sigma_clip":
        return _detect_cosmic_rays_sigma_clip(
            frame=frame,
            existing_bpm=existing_bpm,
            sigma=sigma,
            box_size=box_size,
            iterations=iterations,
            min_sigma=min_sigma,
            bpm_dtype=bpm_dtype,
        )
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
    existing_bpm: np.ndarray | None,
    sigma: float,
    box_size: int,
    iterations: int,
    bpm_dtype: np.dtype,
    min_sigma: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sigma-clip cosmic ray detection using local median residuals.

    For each iteration: compute local median, compute residuals, estimate
    robust MAD, flag positive outliers beyond ``sigma × MAD``, replace
    flagged pixels with the local median before the next pass.

    The ``min_sigma`` floor handles zero-MAD cases (noiseless synthetic
    images or constant-background regions) where any deviation from the
    local median should be detected.
    """
    if box_size % 2 == 0:
        box_size += 1
        logger.debug(
            "detect_cosmic_rays: box_size adjusted to %d (must be odd).",
            box_size,
        )

    if min_sigma is None:
        min_sigma = 1e-10
    elif min_sigma <= 0:
        raise ValueError(f"min_sigma must be positive; got {min_sigma}.")

    dq      = create_bpm(frame.shape, dtype=bpm_dtype)
    work    = frame.astype(float, copy=True)
    cr_mask = np.zeros(frame.shape, dtype=bool)

    stat_mask = (existing_bpm > 0) if existing_bpm is not None else None

    iters_run = 0
    for iteration in range(iterations):
        local_median = ndimage.median_filter(work, size=box_size)
        residuals    = work - local_median

        _, mad_sigma = sigma_clip_stats(residuals, mask=stat_mask, scale="mad")

        if not np.isfinite(mad_sigma):
            logger.warning(
                "detect_cosmic_rays (sigma_clip): non-finite MAD at "
                "iteration %d; stopping.",
                iteration + 1,
            )
            break

        if mad_sigma == 0:
            logger.debug(
                "detect_cosmic_rays (sigma_clip): zero MAD at iteration %d; "
                "using min_sigma=%.2e.",
                iteration + 1, min_sigma,
            )
            effective_sigma = min_sigma
        elif mad_sigma < min_sigma:
            logger.debug(
                "detect_cosmic_rays (sigma_clip): MAD %.2e < min_sigma %.2e "
                "at iteration %d; using min_sigma.",
                mad_sigma, min_sigma, iteration + 1,
            )
            effective_sigma = min_sigma
        else:
            effective_sigma = mad_sigma

        threshold = float(sigma) * effective_sigma
        new_hits  = (residuals > threshold) & ~cr_mask
        num_new   = int(np.sum(new_hits))

        logger.debug(
            "detect_cosmic_rays (sigma_clip): iteration %d — "
            "%d new pixel(s) flagged (threshold=%.2e, sigma=%.2e).",
            iteration + 1, num_new, threshold, effective_sigma,
        )

        cr_mask |= new_hits
        work[new_hits] = local_median[new_hits]
        iters_run += 1

        if num_new == 0:
            break

    dq[cr_mask] |= BPM_BIT_COSMIC

    stats: dict[str, Any] = {
        "method":          "sigma_clip",
        "sigma":           float(sigma),
        "box_size":        int(box_size),
        "min_sigma":       float(min_sigma),
        "iterations_run":  iters_run,
        "num_cosmic_rays": int(np.sum(cr_mask)),
        "fraction":        float(np.sum(cr_mask)) / frame.size,
    }
    return dq, stats


def _detect_cosmic_rays_lacosmic(
    frame: np.ndarray,
    existing_bpm: np.ndarray | None,
    sigclip: float,
    sigfrac: float,
    objlim: float,
    gain: float,
    readnoise: float,
    iterations: int,
    bpm_dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, Any]]:
    """L.A.Cosmic cosmic ray detection via astroscrappy.

    The ``existing_bpm`` is passed as ``inmask`` so that permanently bad
    pixels are excluded from the Laplacian edge image statistics. This
    improves the noise model without suppressing detection.
    """
    if not _LACOSMIC_AVAILABLE:
        raise ImportError(
            "detect_cosmic_rays with method='lacosmic' requires astroscrappy. "
            "Install with: pip install astroscrappy"
        )

    frame_f32 = np.ascontiguousarray(frame, dtype=np.float32)
    inmask = (existing_bpm > 0).astype(bool) if existing_bpm is not None else None

    cr_mask_bool, _ = _astroscrappy.detect_cosmics(
        frame_f32,
        inmask=inmask,
        sigclip=float(sigclip),
        sigfrac=float(sigfrac),
        objlim=float(objlim),
        gain=float(gain),
        readnoise=float(readnoise),
        niter=int(iterations),
        cleantype="median",
        fsmode="median",
    )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)
    dq[cr_mask_bool] |= BPM_BIT_COSMIC

    stats: dict[str, Any] = {
        "method":               "lacosmic",
        "sigclip":              float(sigclip),
        "sigfrac":              float(sigfrac),
        "objlim":               float(objlim),
        "gain":                 float(gain),
        "readnoise":            float(readnoise),
        "iterations_requested": int(iterations),
        "num_cosmic_rays":      int(np.sum(cr_mask_bool)),
        "fraction":             float(np.sum(cr_mask_bool)) / frame.size,
    }
    return dq, stats


# =============================================================================
# Bloom detection
# =============================================================================


def detect_bloom(
    frame: np.ndarray,
    saturation_level: float | None = None,
    bit_depth: int = 16,
    bleed_direction: str = "vertical",
    min_trail_length: int = 5,
    trail_threshold_factor: float = 0.1,
    existing_bpm: np.ndarray | None = None,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect blooming and bleed trails from saturated sources.

    Blooming occurs when a pixel overfills its full-well capacity and excess
    charge bleeds into adjacent pixels along the readout direction.

    Parameters
    ----------
    frame : np.ndarray
        Science frame (ADU).
    saturation_level : float, optional
        ADU value considered saturated. If None, computed as
        ``0.95 × (2^bit_depth - 1)``.
    bit_depth : int, optional
        ADC bit depth. Default is 16.
    bleed_direction : {"vertical", "horizontal", "both"}, optional
        Direction of charge bleeding. ``"vertical"`` (along columns) is
        typical for CCDs. Default is ``"vertical"``.
    min_trail_length : int, optional
        Minimum contiguous pixels to classify as a bleed trail. Default is 5.
    trail_threshold_factor : float, optional
        Fraction of saturation level below which pixels are excluded from
        the trail. Default is 0.1.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask. Permanently bad pixels are excluded from
        saturation seed identification to avoid seeding bloom detection
        from defective hardware (e.g. bad columns). Trail pixels from
        such defects that affect neighbouring good pixels are still flagged.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_BLOOM`` set for bleed trail pixels.
    stats : dict
        Keys: ``'saturation_level'``, ``'bleed_direction'``,
        ``'min_trail_length'``, ``'trail_threshold'``,
        ``'num_saturated_pixels'`` (good seeds used),
        ``'num_saturated_total'``, ``'num_saturated_excluded'``,
        ``'num_saturated_clusters'``, ``'num_bloom_pixels'``,
        ``'fraction'``, ``'num_trails'``.

    Raises
    ------
    ValueError
        If *bleed_direction* is not recognised, or *existing_bpm* shape
        does not match *frame*.
    """
    if saturation_level is None:
        saturation_level = 0.95 * ((2 ** bit_depth) - 1)

    bleed_direction = str(bleed_direction).lower()
    if bleed_direction not in {"vertical", "horizontal", "both"}:
        raise ValueError(
            f"bleed_direction must be 'vertical', 'horizontal', or 'both'; "
            f"got {bleed_direction!r}."
        )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)

    saturated_mask_all = frame >= saturation_level
    num_saturated_total = int(np.sum(saturated_mask_all))

    if existing_bpm is not None:
        if existing_bpm.shape != frame.shape:
            raise ValueError(
                f"existing_bpm shape {existing_bpm.shape} does not match "
                f"frame shape {frame.shape}."
            )
        bad_pixel_mask       = existing_bpm > 0
        saturated_mask       = saturated_mask_all & ~bad_pixel_mask
        num_saturated_pixels = int(np.sum(saturated_mask))
        num_excluded         = num_saturated_total - num_saturated_pixels
        if num_excluded > 0:
            logger.debug(
                "detect_bloom: excluded %d/%d saturated pixel(s) "
                "(in existing_bpm) from bloom seed detection.",
                num_excluded, num_saturated_total,
            )
    else:
        saturated_mask       = saturated_mask_all
        num_saturated_pixels = num_saturated_total
        num_excluded         = 0

    trail_threshold = float(saturation_level) * trail_threshold_factor

    empty_stats: dict[str, Any] = {
        "saturation_level":       float(saturation_level),
        "bleed_direction":        bleed_direction,
        "min_trail_length":       int(min_trail_length),
        "trail_threshold":        trail_threshold,
        "num_saturated_pixels":   num_saturated_pixels,
        "num_saturated_total":    num_saturated_total,
        "num_saturated_excluded": num_excluded,
        "num_saturated_clusters": 0,
        "num_bloom_pixels":       0,
        "fraction":               0.0,
        "num_trails":             0,
    }

    if num_saturated_pixels == 0:
        logger.debug(
            "detect_bloom: no good saturated pixels found; skipping bloom detection."
        )
        return dq, empty_stats

    labeled_sat, num_sat_clusters = ndimage.label(saturated_mask)
    if num_sat_clusters == 0:
        return dq, empty_stats

    logger.debug(
        "detect_bloom: found %d saturated source(s) from %d good pixel(s).",
        num_sat_clusters, num_saturated_pixels,
    )

    bloom_mask = np.zeros_like(frame, dtype=bool)
    ny, nx = frame.shape

    for cluster_id in range(1, num_sat_clusters + 1):
        cluster_mask = labeled_sat == cluster_id
        coords       = np.argwhere(cluster_mask)
        seed_y = int(np.clip(coords.mean(axis=0)[0], 0, ny - 1))
        seed_x = int(np.clip(coords.mean(axis=0)[1], 0, nx - 1))

        if bleed_direction in {"vertical", "both"}:
            trail_up   = _grow_trail_1d(frame[:, seed_x], seed_y, -1, trail_threshold)
            trail_down = _grow_trail_1d(frame[:, seed_x], seed_y, +1, trail_threshold)
            trail_idx  = np.concatenate([trail_up, trail_down])
            trail_idx  = trail_idx[~np.isin(trail_idx, coords[:, 0])]
            if len(trail_idx) >= min_trail_length:
                bloom_mask[trail_idx, seed_x] = True

        if bleed_direction in {"horizontal", "both"}:
            trail_left  = _grow_trail_1d(frame[seed_y, :], seed_x, -1, trail_threshold)
            trail_right = _grow_trail_1d(frame[seed_y, :], seed_x, +1, trail_threshold)
            trail_idx   = np.concatenate([trail_left, trail_right])
            trail_idx   = trail_idx[~np.isin(trail_idx, coords[:, 1])]
            if len(trail_idx) >= min_trail_length:
                bloom_mask[seed_y, trail_idx] = True

    _, num_trails = ndimage.label(bloom_mask)
    dq[bloom_mask] |= BPM_BIT_BLOOM

    stats: dict[str, Any] = {
        "saturation_level":       float(saturation_level),
        "bleed_direction":        bleed_direction,
        "min_trail_length":       int(min_trail_length),
        "trail_threshold":        trail_threshold,
        "num_saturated_pixels":   num_saturated_pixels,
        "num_saturated_total":    num_saturated_total,
        "num_saturated_excluded": num_excluded,
        "num_saturated_clusters": num_sat_clusters,
        "num_bloom_pixels":       int(np.sum(bloom_mask)),
        "fraction":               float(np.sum(bloom_mask)) / frame.size,
        "num_trails":             num_trails,
    }
    return dq, stats


def _grow_trail_1d(
    line: np.ndarray,
    start_idx: int,
    direction: int,
    threshold: float,
) -> np.ndarray:
    """Grow a bleed trail along a 1-D slice from a seed pixel.

    Parameters
    ----------
    line : np.ndarray
        1-D row or column of the frame.
    start_idx : int
        Starting pixel index (saturated seed).
    direction : int
        Growth direction: ``+1`` forward, ``-1`` backward.
    threshold : float
        Minimum pixel value to continue the trail.

    Returns
    -------
    np.ndarray of int
        Pixel indices belonging to the trail, including *start_idx*.
        Returns an empty array if *start_idx* is out of bounds.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1; got {direction}.")

    if not (0 <= start_idx < len(line)):
        return np.array([], dtype=int)

    if direction == 1:
        if start_idx >= len(line) - 1:
            return np.array([start_idx], dtype=int)
        segment = line[start_idx + 1:]
        offset  = start_idx + 1
    else:
        if start_idx <= 0:
            return np.array([start_idx], dtype=int)
        segment = line[:start_idx][::-1]
        offset  = start_idx - 1

    if segment.size == 0:
        return np.array([start_idx], dtype=int)

    above = segment >= threshold
    if not np.any(above):
        return np.array([start_idx], dtype=int)

    below = ~above
    trail_length = int(np.argmax(below)) if np.any(below) else len(segment)

    if direction == 1:
        return np.arange(start_idx, offset + trail_length, dtype=int)
    return np.arange(start_idx, offset - trail_length, -1, dtype=int)


# =============================================================================
# Persistence
# =============================================================================


def detect_persistence(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    threshold_sigma: float = 3.0,
    min_region_size: int = 10,
    source_mask: np.ndarray | None = None,
    align_frames: bool = False,
    existing_bpm: np.ndarray | None = None,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect image persistence (latent images from previous exposures).

    Compares the current frame to a reference taken before or long after
    a suspected bright exposure. Residual positive signal indicates
    persistence.

    Parameters
    ----------
    frame : np.ndarray
        Current science frame suspected of having persistence.
    reference_frame : np.ndarray
        Reference frame of the same field. Must have the same shape as
        *frame*. The *align_frames* parameter corrects small translational
        shifts only; it does not handle different array dimensions.
    threshold_sigma : float, optional
        Detection threshold in sigma units on the difference image.
        Default is 3.0.
    min_region_size : int, optional
        Minimum connected-pixel count to classify as persistence. Filters
        isolated noise pixels. Default is 10.
    source_mask : np.ndarray of bool, optional
        Mask of known astrophysical sources (True = source pixel). Excluded
        from statistics and detection to avoid confusing new sources with
        persistence.
    align_frames : bool, optional
        If True, align *reference_frame* to *frame* via cross-correlation
        before differencing. Corrects translational shifts only. Default
        is False.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask. Excluded from statistics and detection.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_OUTLIER`` set for persistence
        regions.
    stats : dict
        Keys: ``'threshold_sigma'``, ``'threshold'``, ``'num_regions'``,
        ``'num_pixels'``, ``'fraction'``, ``'max_persistence'``,
        ``'mean_persistence'``, ``'alignment_offset'``,
        ``'num_edge_pixels_masked'``.

    Raises
    ------
    ValueError
        If *frame* and *reference_frame* have different shapes.

    Notes
    -----
    Persistence detection cannot distinguish persistence from new
    astrophysical sources that appeared between the reference and current
    frame. Use *source_mask* to exclude known source regions.
    Even 1-pixel misalignment can create spurious dipole patterns around
    stars; use ``align_frames=True`` or pre-register frames externally.

    References
    ----------
    .. [1] Smith, R. M. et al. (2008). Proc. SPIE, 7021, 70210J.
    """
    if frame.shape != reference_frame.shape:
        raise ValueError(
            f"frame and reference_frame must have the same shape; "
            f"got {frame.shape} vs {reference_frame.shape}."
        )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)

    alignment_offset    = None
    edge_mask_from_shift = None
    ref_aligned          = reference_frame

    if align_frames:
        try:
            ref_aligned, alignment_offset, edge_mask_from_shift = \
                align_frames_cross_correlation(reference_frame, frame)
            logger.debug(
                "detect_persistence: applied alignment offset dy=%.2f, dx=%.2f pixel(s).",
                alignment_offset[0], alignment_offset[1],
            )
        except Exception as e:
            logger.warning(
                "detect_persistence: alignment failed (%s); proceeding without.",
                e,
            )
            alignment_offset    = None
            edge_mask_from_shift = None

    diff = frame.astype(float) - ref_aligned.astype(float)

    stat_mask = np.zeros(frame.shape, dtype=bool)
    if existing_bpm is not None:
        stat_mask |= existing_bpm > 0
    if source_mask is not None:
        stat_mask |= source_mask
        logger.debug(
            "detect_persistence: excluding %d source pixel(s) from statistics.",
            int(np.sum(source_mask)),
        )

    median_diff, sigma_diff = sigma_clip_stats(
        diff,
        mask=stat_mask if np.any(stat_mask) else None,
        scale="mad",
    )

    n_edge = int(np.sum(edge_mask_from_shift)) if edge_mask_from_shift is not None else 0

    empty: dict[str, Any] = {
        "threshold_sigma":      float(threshold_sigma),
        "threshold":            float("nan"),
        "num_regions":          0,
        "num_pixels":           0,
        "fraction":             0.0,
        "max_persistence":      0.0,
        "mean_persistence":     0.0,
        "alignment_offset":     alignment_offset,
        "num_edge_pixels_masked": n_edge,
    }

    if not np.isfinite(sigma_diff) or sigma_diff <= 0:
        logger.warning(
            "detect_persistence: degenerate sigma (%s); no persistence detected.",
            sigma_diff,
        )
        return dq, empty

    threshold = median_diff + threshold_sigma * sigma_diff
    candidates = diff > threshold
    if np.any(stat_mask):
        candidates &= ~stat_mask

    if not np.any(candidates):
        empty["threshold"] = float(threshold)
        return dq, empty

    labeled, num_regions = ndimage.label(candidates)
    if num_regions == 0:
        empty["threshold"] = float(threshold)
        return dq, empty

    region_sizes  = np.bincount(labeled.ravel())[1:]
    valid_regions = np.where(region_sizes >= min_region_size)[0] + 1
    final_mask    = np.isin(labeled, valid_regions)

    dq[final_mask] |= BPM_BIT_OUTLIER
    pv = diff[final_mask]

    stats: dict[str, Any] = {
        "threshold_sigma":      float(threshold_sigma),
        "threshold":            float(threshold),
        "num_regions":          int(len(valid_regions)),
        "num_pixels":           int(np.sum(final_mask)),
        "fraction":             float(np.sum(final_mask)) / frame.size,
        "max_persistence":      float(np.max(pv)) if pv.size > 0 else 0.0,
        "mean_persistence":     float(np.mean(pv)) if pv.size > 0 else 0.0,
        "alignment_offset":     alignment_offset,
        "num_edge_pixels_masked": n_edge,
    }
    return dq, stats


def align_frames_cross_correlation(
    reference: np.ndarray,
    target: np.ndarray,
    upsample_factor: int = 10,
) -> tuple[np.ndarray, tuple[float, float], np.ndarray | None]:
    """Align a reference frame to a target using phase cross-correlation.

    Parameters
    ----------
    reference : np.ndarray
        Reference frame to be shifted.
    target : np.ndarray
        Target frame (fixed; used as alignment reference).
    upsample_factor : int, optional
        Upsampling factor for sub-pixel precision (1 = integer only,
        10 = 0.1 pixel, 100 = 0.01 pixel). Default is 10.

    Returns
    -------
    aligned : np.ndarray
        Shifted reference frame aligned to *target*.
    offset : tuple of (dy, dx)
        Applied shift in pixels. Positive dy = shifted down,
        positive dx = shifted right.
    edge_mask : np.ndarray of bool or None
        True where shifted edge pixels contain extrapolated values and
        should be excluded from difference imaging.

    Notes
    -----
    Uses ``skimage.registration.phase_cross_correlation`` when available
    (sub-pixel accuracy). Falls back to integer-pixel scipy cross-correlation
    when scikit-image is not installed with a logged warning.

    References
    ----------
    .. [1] Guizar-Sicairos et al. (2008). Optics Letters, 33(2), 156–158.
    """
    try:
        from skimage.registration import phase_cross_correlation
        has_phase_corr = True
    except ImportError:
        has_phase_corr = False

    if has_phase_corr:
        shift, error, _ = phase_cross_correlation(
            target, reference, upsample_factor=upsample_factor,
            return_error=True,
        )
        offset = tuple(shift.tolist())
        logger.debug(
            "align_frames_cross_correlation: shift=%s, error=%.6f.",
            offset, error,
        )
        from scipy.ndimage import shift as scipy_shift
        aligned    = scipy_shift(reference, shift=shift, mode="nearest", order=3)
        edge_mask  = create_edge_mask_from_shift(reference.shape, shift)
        return aligned, offset, edge_mask

    logger.warning(
        "align_frames_cross_correlation: scikit-image not available; "
        "falling back to integer-pixel alignment. "
        "Install with: pip install scikit-image"
    )
    try:
        from scipy.signal import correlate
        from scipy.ndimage import shift as scipy_shift
    except ImportError as e:
        raise ImportError(
            "Frame alignment requires scipy or scikit-image. "
            "Install with: pip install scipy"
        ) from e

    corr      = correlate(target, reference, mode="same", method="fft")
    peak_idx  = np.unravel_index(np.argmax(corr), corr.shape)
    center    = np.array(target.shape) // 2
    shift_vec = np.array(peak_idx) - center
    offset    = tuple(shift_vec.tolist())

    logger.debug(
        "align_frames_cross_correlation: shift=%s (integer pixel fallback).",
        offset,
    )
    aligned   = scipy_shift(reference, shift=shift_vec, mode="nearest", order=1)
    edge_mask = create_edge_mask_from_shift(reference.shape, shift_vec)
    return aligned, offset, edge_mask


def create_edge_mask_from_shift(
    shape: tuple[int, int],
    shift: np.ndarray,
) -> np.ndarray:
    """Create a boolean mask marking pixels affected by frame shifting.

    After alignment, pixels near the edges are filled with extrapolated
    values. This mask identifies those unreliable pixels.

    Parameters
    ----------
    shape : tuple of (ny, nx)
        Shape of the frame.
    shift : np.ndarray
        Applied shift as ``[dy, dx]``.

    Returns
    -------
    np.ndarray of bool
        True where pixels contain extrapolated edge values and should
        be excluded from analysis.

    Examples
    --------
    >>> mask = create_edge_mask_from_shift((100, 100), np.array([2.3, -1.7]))
    >>> # mask[0:3, :] is True (top edge); mask[:, 98:] is True (right edge)
    """
    ny, nx = shape
    dy, dx = shift
    edge_mask = np.zeros((ny, nx), dtype=bool)

    if dy > 0:
        edge_mask[:int(np.ceil(abs(dy))), :] = True
    elif dy < 0:
        edge_mask[-int(np.ceil(abs(dy))):, :] = True

    if dx > 0:
        edge_mask[:, :int(np.ceil(abs(dx)))] = True
    elif dx < 0:
        edge_mask[:, -int(np.ceil(abs(dx))):] = True

    return edge_mask


# =============================================================================
# Stub
# =============================================================================


def detect_crosstalk(
    frame: np.ndarray,
    channel_geometry: dict[str, Any],
    existing_bpm: np.ndarray | None = None,
    bpm_dtype: np.dtype = np.uint32,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect electronic crosstalk between amplifier channels.

    Parameters
    ----------
    frame : np.ndarray
        Science frame (ADU).
    channel_geometry : dict
        Instrument-specific amplifier layout and crosstalk coefficients.
        Required keys are not yet defined.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_CROSSTALK`` set.
    stats : dict
        Detection statistics.

    Raises
    ------
    NotImplementedError
        Always. Implementation awaits a concrete instrument use case.
    """
    raise NotImplementedError(
        "detect_crosstalk is not yet implemented. "
        "This function is instrument-specific and requires channel_geometry "
        "to be defined for the target detector before implementation."
    )


# =============================================================================
# Statistical outliers
# =============================================================================


def flag_outliers(
    outlier_mask: np.ndarray,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode an externally-detected outlier mask into a DQ array.

    Unlike other detectors, outlier detection is pipeline-specific and
    performed externally. This function encodes the result using the
    correct bit flag, ensuring consistent flag usage across pipelines.

    Parameters
    ----------
    outlier_mask : np.ndarray of bool
        Boolean array where True indicates an outlier pixel.
    bpm_dtype : np.dtype, optional
        Dtype of the output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        DQ array with ``BPM_BIT_OUTLIER`` set for each True pixel in
        *outlier_mask*.
    stats : dict
        Keys: ``'num_outliers'``, ``'fraction'``.
    """
    dq = create_bpm(outlier_mask.shape, dtype=bpm_dtype)
    dq[outlier_mask] |= BPM_BIT_OUTLIER
    stats: dict[str, Any] = {
        "num_outliers": int(np.sum(outlier_mask)),
        "fraction":     float(np.sum(outlier_mask)) / outlier_mask.size,
    }
    return dq, stats


def detect_statistical_outliers(
    frame: np.ndarray,
    reference_frames: np.ndarray | None = None,
    reference_median: np.ndarray | None = None,
    reference_sigma: np.ndarray | None = None,
    sigma: float = 5.0,
    tail: str = "both",
    existing_bpm: np.ndarray | None = None,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect pixels that are statistical outliers relative to a reference.

    Three operating modes depending on what reference data is supplied:

    1. **reference_frames** (full stack) — median of the stack is used as
       the expected value; MAD of residuals sets the threshold.
    2. **reference_median + reference_sigma** (pre-computed maps) —
       per-pixel z-score is computed; threshold is in z units. Memory-
       efficient for large stacks.
    3. **neither** (global statistics) — global frame median and MAD are
       used. Useful as a basic sanity check.

    Parameters
    ----------
    frame : np.ndarray
        Science frame to check.
    reference_frames : np.ndarray of shape (N, ny, nx), optional
        Stack of reference frames. Cannot be combined with
        *reference_median* / *reference_sigma*.
    reference_median : np.ndarray, optional
        Per-pixel reference median. Must be provided together with
        *reference_sigma*.
    reference_sigma : np.ndarray, optional
        Per-pixel reference sigma. Must be provided together with
        *reference_median*.
    sigma : float, optional
        Detection threshold in sigma units. Default is 5.0.
    tail : {"both", "high", "low"}, optional
        Which tail(s) to flag. Default is ``"both"``.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask (excluded from statistics).
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_OUTLIER`` set for outlier pixels.
    stats : dict
        Keys: ``'method'``, ``'tail'``, ``'sigma'``, ``'scale'``,
        ``'threshold'``, ``'num_outliers'``, ``'fraction'``.
        For ``tail="both"``, also ``'num_low'``, ``'num_high'``,
        ``'threshold_low'``, ``'threshold_high'``.

    Raises
    ------
    ValueError
        If incompatible reference inputs are provided, or if *tail* is
        not recognised, or if reference map shapes do not match *frame*.
    """
    tail = str(tail).lower()
    if tail not in {"both", "high", "low"}:
        raise ValueError(
            f"tail must be 'both', 'high', or 'low'; got {tail!r}."
        )

    if reference_frames is not None and (
        reference_median is not None or reference_sigma is not None
    ):
        raise ValueError(
            "Cannot provide both reference_frames and "
            "reference_median/reference_sigma. Choose one approach."
        )

    if (reference_median is None) != (reference_sigma is None):
        raise ValueError(
            "reference_median and reference_sigma must be provided together."
        )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)
    mask_exclude = (existing_bpm > 0) if existing_bpm is not None else None

    # Determine method and compute data array for thresholding
    if reference_frames is not None:
        method      = "reference_stack"
        residuals   = frame - np.median(reference_frames, axis=0)
        _, scale    = sigma_clip_stats(residuals, mask=mask_exclude, scale="mad")
        data        = residuals
        threshold_value = float(sigma) * float(scale)
        valid_mask  = None

    elif reference_median is not None and reference_sigma is not None:
        method = "reference_maps"
        for name, arr in (
            ("reference_median", reference_median),
            ("reference_sigma",  reference_sigma),
        ):
            if arr.shape != frame.shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match "
                    f"frame shape {frame.shape}."
                )

        residuals  = frame - reference_median
        sigma_map  = np.asarray(reference_sigma, dtype=float)
        invalid    = ~np.isfinite(sigma_map) | (sigma_map <= 0)
        if mask_exclude is not None:
            invalid |= mask_exclude

        valid_mask = ~invalid
        if not np.any(valid_mask):
            logger.warning(
                "detect_statistical_outliers: no valid pixels in reference_sigma "
                "after masking; no outliers flagged."
            )
            return dq, {
                "method": method, "tail": tail, "sigma": float(sigma),
                "scale": float("nan"), "threshold": float("nan"),
                "num_outliers": 0, "fraction": 0.0,
            }

        data            = np.zeros_like(residuals, dtype=float)
        data[valid_mask] = residuals[valid_mask] / sigma_map[valid_mask]
        scale           = float(np.nanmedian(sigma_map[valid_mask]))
        threshold_value  = float(sigma)

    else:
        method   = "global"
        med, scale = sigma_clip_stats(frame, mask=mask_exclude, scale="mad")
        residuals   = frame - med
        data        = residuals
        threshold_value = float(sigma) * float(scale)
        valid_mask  = None

    # Early exit on degenerate scale
    if not np.isfinite(scale) or scale <= 0:
        logger.warning(
            "detect_statistical_outliers: degenerate scale (%s); "
            "no outliers flagged.", scale,
        )
        return dq, {
            "method": method, "tail": tail, "sigma": float(sigma),
            "scale": float(scale), "threshold": float("nan"),
            "num_outliers": 0, "fraction": 0.0,
        }

    def _apply_valid(mask: np.ndarray) -> np.ndarray:
        if method == "reference_maps" and valid_mask is not None:
            return mask & valid_mask
        return mask

    if tail == "both":
        out_low  = _apply_valid(data < -threshold_value)
        out_high = _apply_valid(data >  threshold_value)
        outlier_mask = out_low | out_high
        stats: dict[str, Any] = {
            "method":          method,
            "tail":            tail,
            "sigma":           float(sigma),
            "scale":           float(scale),
            "threshold":       float(threshold_value),
            "threshold_low":  -float(threshold_value),
            "threshold_high":  float(threshold_value),
            "num_outliers":    int(np.sum(outlier_mask)),
            "num_low":         int(np.sum(out_low)),
            "num_high":        int(np.sum(out_high)),
            "fraction":        float(np.sum(outlier_mask)) / frame.size,
        }
    elif tail == "high":
        outlier_mask = _apply_valid(data > threshold_value)
        stats = {
            "method":       method, "tail": tail, "sigma": float(sigma),
            "scale":        float(scale), "threshold": float(threshold_value),
            "num_outliers": int(np.sum(outlier_mask)),
            "num_high":     int(np.sum(outlier_mask)),
            "fraction":     float(np.sum(outlier_mask)) / frame.size,
        }
    else:  # low
        outlier_mask = _apply_valid(data < -threshold_value)
        stats = {
            "method":       method, "tail": tail, "sigma": float(sigma),
            "scale":        float(scale), "threshold": float(threshold_value),
            "num_outliers": int(np.sum(outlier_mask)),
            "num_low":      int(np.sum(outlier_mask)),
            "fraction":     float(np.sum(outlier_mask)) / frame.size,
        }

    dq[outlier_mask] |= BPM_BIT_OUTLIER
    return dq, stats


# =============================================================================
# Integration
# =============================================================================


def detect_all_artifacts(
    frame: np.ndarray,
    existing_bpm: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
    reference_frame: np.ndarray | None = None,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the full artifact detection suite on a single science frame.

    Runs detectors in order: saturation → bloom → cosmic rays →
    persistence (if *reference_frame* provided). Lab-specific detectors
    (fringing, FPN, thermal gradient) are not included and should be run
    separately as part of a calibration characterization workflow.

    Parameters
    ----------
    frame : np.ndarray
        Science frame to analyse (ADU).
    existing_bpm : np.ndarray, optional
        Master bad pixel mask. Used by each detector to exclude permanent
        defects from noise statistics. Does not suppress detection.
    config : dict, optional
        Per-detector configuration. If None, uses
        :func:`get_default_artifact_config`.
    reference_frame : np.ndarray, optional
        Reference frame for persistence detection. If None, persistence
        detection is skipped.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        Combined per-frame DQ array with all detected artifacts flagged.
    stats : dict
        Nested statistics from each detector under keys
        ``'saturation'``, ``'bloom'``, ``'cosmic_rays'``,
        ``'persistence'``, plus summary keys ``'total_flagged'``,
        ``'total_fraction'``, ``'artifacts_only'``,
        ``'artifacts_fraction'``.

    Notes
    -----
    Each detector is self-contained and independently identifies the
    features it needs (e.g. bloom re-detects saturation internally).
    This ensures detectors can be run individually in any order and that
    each detector's stats are self-consistent.
    """
    if config is None:
        config = get_default_artifact_config()

    all_stats: dict[str, Any] = {}
    dq = create_bpm(frame.shape, dtype=bpm_dtype)

    logger.debug("detect_all_artifacts: running saturation detection.")
    sat_dq, sat_stats = detect_saturated_pixels(
        frame, bpm_dtype=bpm_dtype, **config.get("saturation", {})
    )
    dq |= sat_dq
    all_stats["saturation"] = sat_stats

    logger.debug("detect_all_artifacts: running bloom detection.")
    bloom_dq, bloom_stats = detect_bloom(
        frame, existing_bpm=existing_bpm, bpm_dtype=bpm_dtype,
        **config.get("bloom", {}),
    )
    dq |= bloom_dq
    all_stats["bloom"] = bloom_stats

    logger.debug("detect_all_artifacts: running cosmic ray detection.")
    cr_dq, cr_stats = detect_cosmic_rays(
        frame, existing_bpm=existing_bpm, bpm_dtype=bpm_dtype,
        **config.get("cosmic_rays", {}),
    )
    dq |= cr_dq
    all_stats["cosmic_rays"] = cr_stats

    if reference_frame is not None:
        logger.debug("detect_all_artifacts: running persistence detection.")
        pers_dq, pers_stats = detect_persistence(
            frame, reference_frame=reference_frame,
            existing_bpm=existing_bpm, bpm_dtype=bpm_dtype,
            **config.get("persistence", {}),
        )
        dq |= pers_dq
        all_stats["persistence"] = pers_stats
    else:
        all_stats["persistence"] = {"skipped": True, "reason": "No reference frame"}

    combined = (dq | existing_bpm) if existing_bpm is not None else dq
    all_stats["total_flagged"]     = int(np.sum(combined > 0))
    all_stats["total_fraction"]    = float(all_stats["total_flagged"]) / frame.size
    all_stats["artifacts_only"]    = int(np.sum(dq > 0))
    all_stats["artifacts_fraction"] = float(all_stats["artifacts_only"]) / frame.size

    return dq, all_stats


def get_default_artifact_config() -> dict[str, Any]:
    """Return default configuration for all artifact detectors.

    Returns
    -------
    dict
        Nested dict with one sub-dict per detector, containing the
        default keyword arguments passed to each detection function by
        :func:`detect_all_artifacts`.

    Examples
    --------
    >>> config = get_default_artifact_config()
    >>> config["cosmic_rays"]["sigma"] = 7.0  # more conservative
    >>> dq, stats = detect_all_artifacts(frame, config=config)
    """
    return {
        "saturation": {
            "saturation_threshold": 0.95,
            "bit_depth": 16,
        },
        "bloom": {
            "bleed_direction":        "vertical",
            "min_trail_length":       5,
            "trail_threshold_factor": 0.1,
        },
        "cosmic_rays": {
            "method":     "sigma_clip",
            "sigma":      5.0,
            "box_size":   5,
            "iterations": 3,
            "min_sigma":  None,
        },
        "persistence": {
            "threshold_sigma": 3.0,
            "min_region_size": 10,
        },
    }


# =============================================================================
# Lab metrology detectors
# =============================================================================


def detect_fringing(
    flat_frame: np.ndarray,
    wavelength: float | None = None,
    fringe_period_range: tuple[int, int] = (5, 50),
    threshold_amplitude: float = 0.01,
    existing_bpm: np.ndarray | None = None,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect fringing patterns in flat fields (thin-film interference).

    Fringing appears as sinusoidal brightness variations caused by
    interference in the thin silicon substrate (back-illuminated CCDs) or
    anti-reflection coatings. Most pronounced in the near-IR (700–1000 nm).

    Parameters
    ----------
    flat_frame : np.ndarray
        Normalized flat field (e.g. master flat / median).
    wavelength : float, optional
        Illumination wavelength in nanometres. If provided, used for
        context only (expected period estimation).
    fringe_period_range : tuple of (int, int), optional
        Expected fringe period range in pixels. Must satisfy the Nyquist
        criterion: ``min_period >= 2``. Default is ``(5, 50)``.
    threshold_amplitude : float, optional
        Minimum fringe amplitude (fractional) to flag. Default is 0.01.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask. Bad pixels are interpolated over before
        FFT to prevent spurious high-frequency peaks, then excluded from
        the fringe detection output to avoid double-flagging.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        DQ array with ``BPM_BIT_OUTLIER`` set in fringing regions.
        Pixels already in *existing_bpm* are never flagged here.
    stats : dict
        Keys: ``'fringe_detected'``, ``'dominant_period'``,
        ``'mean_amplitude'``, ``'max_amplitude'``,
        ``'num_pixels_affected'``, ``'wavelength'``,
        ``'num_bad_pixels_masked'``.

    Raises
    ------
    TypeError
        If *fringe_period_range* does not contain integers.
    ValueError
        If ``min_period < 2`` (Nyquist violation), ``max_period`` exceeds
        the image dimension, or ``min_period >= max_period``.

    Notes
    -----
    **Algorithm:**

    1. Interpolate over bad pixels to prevent FFT artifacts.
    2. Apply 2-D FFT to the normalized flat.
    3. Search for peaks in the power spectrum within *fringe_period_range*.
    4. If a significant peak is found (SNR > 10 vs. background), inverse
       FFT to isolate the fringe pattern.
    5. Flag pixels where fringe amplitude exceeds *threshold_amplitude*.
    6. Exclude *existing_bpm* pixels from the output.
    """
    from numpy.fft import fft2, ifft2, fftshift, fftfreq

    min_period, max_period = fringe_period_range

    if not isinstance(min_period, (int, np.integer)) or not isinstance(
        max_period, (int, np.integer)
    ):
        raise TypeError(
            f"fringe_period_range must contain integers; "
            f"got ({type(min_period).__name__}, {type(max_period).__name__})."
        )

    if min_period < 2:
        raise ValueError(
            f"Minimum fringe period must be >= 2 pixels (Nyquist limit); "
            f"got {min_period}. Use detect_fixed_pattern_noise() for "
            f"pixel-to-pixel patterns."
        )

    min_dimension = min(flat_frame.shape)
    if max_period > min_dimension:
        raise ValueError(
            f"Maximum fringe period {max_period} exceeds minimum image "
            f"dimension {min_dimension}. Set max_period <= {min_dimension}."
        )

    if min_period >= max_period:
        raise ValueError(
            f"Invalid period range: min_period ({min_period}) must be "
            f"less than max_period ({max_period})."
        )

    dq = create_bpm(flat_frame.shape, dtype=bpm_dtype)
    num_bad_pixels = 0
    flat_work      = flat_frame.copy()

    if existing_bpm is not None:
        bad_pixel_mask = existing_bpm > 0
        num_bad_pixels = int(np.sum(bad_pixel_mask))
        if num_bad_pixels > 0:
            logger.debug(
                "detect_fringing: interpolating over %d bad pixel(s) "
                "(%.2f%%) before FFT.",
                num_bad_pixels, 100.0 * num_bad_pixels / flat_frame.size,
            )
            flat_work = interpolate_bad_pixels(flat_frame, bad_pixel_mask)

    median_flat = np.median(flat_work)
    if not np.isfinite(median_flat) or median_flat <= 0:
        logger.warning("detect_fringing: invalid flat field median.")
        return dq, {
            "fringe_detected": False, "dominant_period": None,
            "mean_amplitude": 0.0, "max_amplitude": 0.0,
            "num_pixels_affected": 0, "wavelength": wavelength,
            "num_bad_pixels_masked": num_bad_pixels,
        }

    flat_norm     = flat_work / median_flat
    fft_flat      = fft2(flat_norm)
    power_spectrum = np.abs(fftshift(fft_flat)) ** 2

    ny, nx    = flat_frame.shape
    freq_y    = fftshift(fftfreq(ny))
    freq_x    = fftshift(fftfreq(nx))
    fy_grid, fx_grid = np.meshgrid(freq_y, freq_x, indexing="ij")
    freq_radial = np.sqrt(fx_grid ** 2 + fy_grid ** 2)

    freq_min  = 1.0 / max_period
    freq_max  = 1.0 / min_period
    freq_mask = (freq_radial >= freq_min) & (freq_radial <= freq_max)

    cy, cx = ny // 2, nx // 2
    freq_mask[cy - 2:cy + 3, cx - 2:cx + 3] = False

    if not np.any(freq_mask):
        return dq, {
            "fringe_detected": False, "dominant_period": None,
            "mean_amplitude": 0.0, "max_amplitude": 0.0,
            "num_pixels_affected": 0, "wavelength": wavelength,
            "num_bad_pixels_masked": num_bad_pixels,
        }

    masked_power     = power_spectrum * freq_mask
    peak_power       = float(np.max(masked_power))
    background_power = float(np.median(power_spectrum[freq_mask]))

    if peak_power < 10 * background_power:
        return dq, {
            "fringe_detected": False, "dominant_period": None,
            "mean_amplitude": 0.0, "max_amplitude": 0.0,
            "num_pixels_affected": 0, "wavelength": wavelength,
            "num_bad_pixels_masked": num_bad_pixels,
        }

    peak_idx        = np.unravel_index(np.argmax(masked_power), masked_power.shape)
    peak_freq       = float(freq_radial[peak_idx])
    dominant_period = 1.0 / peak_freq if peak_freq > 0 else None

    filter_width = 0.1 * peak_freq
    bandpass     = np.exp(-((freq_radial - peak_freq) / filter_width) ** 2)
    fringe_pat   = np.real(ifft2(fft_flat * fftshift(bandpass)))
    fringe_amp   = np.abs(fringe_pat)

    fringe_mask = fringe_amp > threshold_amplitude
    if existing_bpm is not None:
        fringe_mask &= ~bad_pixel_mask

    dq[fringe_mask] |= BPM_BIT_OUTLIER

    stats: dict[str, Any] = {
        "fringe_detected":     True,
        "dominant_period":     float(dominant_period) if dominant_period else None,
        "mean_amplitude":      float(np.mean(fringe_amp)),
        "max_amplitude":       float(np.max(fringe_amp)),
        "num_pixels_affected": int(np.sum(fringe_mask)),
        "fraction":            float(np.sum(fringe_mask)) / flat_frame.size,
        "wavelength":          float(wavelength) if wavelength else None,
        "num_bad_pixels_masked": num_bad_pixels,
    }
    return dq, stats


def interpolate_bad_pixels(
    image: np.ndarray,
    bad_mask: np.ndarray,
) -> np.ndarray:
    """Interpolate over bad pixels using a distance-weighted approach.

    Prepares images for FFT analysis or other operations sensitive to
    discontinuities by smoothly filling bad-pixel locations.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    bad_mask : np.ndarray of bool
        True where pixels should be interpolated over.

    Returns
    -------
    np.ndarray
        Copy of *image* with bad pixels filled by nearest-good-pixel
        interpolation followed by light Gaussian smoothing.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if not np.any(bad_mask):
        return image.copy()

    result  = image.copy()
    indices = distance_transform_edt(
        bad_mask, return_distances=False, return_indices=True
    )
    result[bad_mask] = image[tuple(indices[:, bad_mask])]

    if np.sum(bad_mask) > 10:
        from scipy.ndimage import binary_dilation
        smooth_mask = binary_dilation(bad_mask, iterations=2)
        smoothed    = gaussian_filter(result, sigma=1.0)
        result[smooth_mask] = smoothed[smooth_mask]

    return result


def detect_fixed_pattern_noise(
    dark_stack: np.ndarray,
    threshold_sigma: float = 3.0,
    pattern_types: list[str] | None = None,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect fixed-pattern noise (FPN) from a stack of dark frames.

    Identifies persistent spatial patterns (banding, checkerboard) that
    are stable across exposures but not captured by the master dark.

    Parameters
    ----------
    dark_stack : np.ndarray of shape (N, ny, nx)
        Stack of dark frames (bias-subtracted, same exposure time).
        Minimum 3 frames required; 10+ recommended for robust detection.
    threshold_sigma : float, optional
        Detection threshold relative to the noise floor. Default is 3.0.
    pattern_types : list of str, optional
        Pattern types to search for. Supported values:
        ``"banding_horizontal"``, ``"banding_vertical"``,
        ``"checkerboard"``. If None, all three are checked.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        DQ array with ``BPM_BIT_OUTLIER`` set where FPN is detected.
    stats : dict
        Keys: ``'n_frames'``, ``'patterns_detected'`` (list),
        ``'num_pixels_affected'``, ``'fraction'``, ``'noise_floor'``.
        Each detected pattern adds a sub-dict with detailed metrics.

    Raises
    ------
    ValueError
        If *dark_stack* has fewer than 3 frames.

    Notes
    -----
    Uses NaN-safe statistics (``np.nanmedian``, ``np.nanstd``) throughout,
    so frames pre-processed with NaN bad-pixel masks are handled correctly.

    For checkerboard detection, the image is truncated to even dimensions
    to ensure equal odd/even sample sizes. Excluded pixels are reported in
    ``stats['checkerboard']['analysis_region']``.
    """
    if pattern_types is None:
        pattern_types = ["banding_horizontal", "banding_vertical", "checkerboard"]

    n_frames = dark_stack.shape[0]
    if n_frames < 3:
        raise ValueError(
            f"Need at least 3 frames for FPN detection; got {n_frames}."
        )

    ny, nx = dark_stack.shape[1:]
    dq = create_bpm((ny, nx), dtype=bpm_dtype)

    temporal_median = np.nanmedian(dark_stack, axis=0)
    temporal_std    = np.nanstd(dark_stack, axis=0)
    noise_floor     = float(np.nanmedian(temporal_std))

    if not np.isfinite(noise_floor) or noise_floor <= 0:
        logger.warning(
            "detect_fixed_pattern_noise: degenerate noise floor (%.4g); "
            "no patterns detected.",
            noise_floor,
        )
        return dq, {
            "n_frames": n_frames, "patterns_detected": [],
            "num_pixels_affected": 0, "fraction": 0.0,
            "noise_floor": float(noise_floor),
        }

    patterns_detected: list[str]      = []
    pattern_stats:     dict[str, Any] = {}
    fpn_mask = np.zeros((ny, nx), dtype=bool)

    # --- Horizontal banding ---
    if "banding_horizontal" in pattern_types:
        row_medians       = np.nanmedian(temporal_median, axis=1)
        row_residuals     = row_medians - float(np.nanmedian(row_medians))
        row_variation     = float(np.nanstd(row_residuals))

        if np.isfinite(row_variation) and row_variation > threshold_sigma * noise_floor:
            patterns_detected.append("banding_horizontal")
            bad_rows     = np.abs(row_residuals) > threshold_sigma * row_variation
            bad_row_idx  = np.where(bad_rows)[0]
            fpn_mask[bad_rows, :] = True
            pattern_stats["banding_horizontal"] = {
                "amplitude":          row_variation,
                "peak_deviation":     float(np.nanmax(np.abs(row_residuals))),
                "affected_rows":      bad_row_idx.tolist(),
                "num_affected_rows":  int(len(bad_row_idx)),
            }

    # --- Vertical banding ---
    if "banding_vertical" in pattern_types:
        col_medians   = np.nanmedian(temporal_median, axis=0)
        col_residuals = col_medians - float(np.nanmedian(col_medians))
        col_variation = float(np.nanstd(col_residuals))

        if np.isfinite(col_variation) and col_variation > threshold_sigma * noise_floor:
            patterns_detected.append("banding_vertical")
            bad_cols    = np.abs(col_residuals) > threshold_sigma * col_variation
            bad_col_idx = np.where(bad_cols)[0]
            fpn_mask[:, bad_cols] = True
            pattern_stats["banding_vertical"] = {
                "amplitude":         col_variation,
                "peak_deviation":    float(np.nanmax(np.abs(col_residuals))),
                "affected_cols":     bad_col_idx.tolist(),
                "num_affected_cols": int(len(bad_col_idx)),
            }

    # --- Checkerboard ---
    if "checkerboard" in pattern_types:
        h_even = ny - (ny % 2)
        w_even = nx - (nx % 2)

        if h_even < 2 or w_even < 2:
            logger.warning(
                "detect_fixed_pattern_noise: image too small for checkerboard "
                "detection (%d×%d).", ny, nx,
            )
        else:
            crop           = temporal_median[:h_even, :w_even]
            odd_rows_mean  = float(np.nanmean(crop[::2, :]))
            even_rows_mean = float(np.nanmean(crop[1::2, :]))
            odd_cols_mean  = float(np.nanmean(crop[:, ::2]))
            even_cols_mean = float(np.nanmean(crop[:, 1::2]))

            row_offset = abs(odd_rows_mean - even_rows_mean)
            col_offset = abs(odd_cols_mean - even_cols_mean)

            for name, val in (("row_offset", row_offset), ("col_offset", col_offset)):
                if not np.isfinite(val):
                    logger.warning(
                        "detect_fixed_pattern_noise: %s is NaN "
                        "(possibly all-NaN data in odd or even row/col).", name,
                    )

            row_offset = row_offset if np.isfinite(row_offset) else 0.0
            col_offset = col_offset if np.isfinite(col_offset) else 0.0
            max_offset = max(row_offset, col_offset)

            if np.isfinite(max_offset) and max_offset > threshold_sigma * noise_floor:
                patterns_detected.append("checkerboard")
                pattern_stats["checkerboard"] = {
                    "amplitude":  float(max_offset),
                    "row_offset": float(row_offset),
                    "col_offset": float(col_offset),
                    "dominant":   "row" if row_offset > col_offset else "col",
                    "analysis_region": {
                        "rows_used":     h_even,
                        "cols_used":     w_even,
                        "rows_excluded": ny - h_even,
                        "cols_excluded": nx - w_even,
                    },
                }

    dq[fpn_mask] |= BPM_BIT_OUTLIER

    stats: dict[str, Any] = {
        "n_frames":            n_frames,
        "patterns_detected":   patterns_detected,
        "num_pixels_affected": int(np.sum(fpn_mask)),
        "fraction":            float(np.sum(fpn_mask)) / (ny * nx),
        "noise_floor":         float(noise_floor),
    }
    stats.update(pattern_stats)
    return dq, stats


def _compute_downsample_factor(shape: tuple[int, int]) -> int:
    """Compute the optimal downsampling factor for thermal gradient fitting.

    Targets ``_THERMAL_FIT_TARGET_POINTS`` (~50k) for excellent
    overdetermination of the 3-parameter linear model while keeping
    ``lstsq`` fast. Capped at ``_THERMAL_FIT_MAX_DOWNSAMPLE`` (21×) and
    rounded to the nearest odd integer to avoid Nyquist ambiguity.

    Parameters
    ----------
    shape : tuple of (ny, nx)
        Image dimensions.

    Returns
    -------
    int
        Downsampling factor in ``[1, _THERMAL_FIT_MAX_DOWNSAMPLE]``.
    """
    n_pixels = shape[0] * shape[1]
    if n_pixels <= _THERMAL_FIT_TARGET_POINTS:
        return 1

    factor = int(np.sqrt(n_pixels / _THERMAL_FIT_TARGET_POINTS))
    if factor % 2 == 0:
        factor += 1
    return min(max(factor, 1), _THERMAL_FIT_MAX_DOWNSAMPLE)


def detect_thermal_gradient(
    frame: np.ndarray,
    max_gradient: float = 0.01,
    smoothing_sigma: float = 50.0,
    downsample_factor: int | None = None,
    existing_bpm: np.ndarray | None = None,
    bpm_dtype: np.dtype = np.uint32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect large-scale thermal gradients in dark frames.

    Thermal gradients appear as smooth, monotonic brightness variations
    across the detector due to uneven cooling or temperature stabilisation.
    This is a QC check rather than a per-pixel flagging tool — large
    gradients indicate the test conditions are not stable.

    Parameters
    ----------
    frame : np.ndarray
        Dark frame to analyse (bias-subtracted).
    max_gradient : float, optional
        Maximum acceptable gradient in ADU/pixel. Default is 0.01.
    smoothing_sigma : float, optional
        Gaussian smoothing kernel size in pixels. Default is 50.0.
    downsample_factor : int or None, optional
        Downsampling factor for plane fitting.
        ``None`` (default) uses adaptive downsampling via
        :func:`_compute_downsample_factor` targeting ~50k points.
        ``1`` disables downsampling (full resolution; slower with no
        accuracy benefit). Odd values are recommended for larger factors.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask. Excluded from the gradient fit.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default is ``np.uint32``.

    Returns
    -------
    dq : np.ndarray
        DQ array (always empty — gradient is a QC flag, not per-pixel).
    stats : dict
        Keys: ``'gradient_detected'``, ``'max_gradient'`` (ADU/pixel),
        ``'gradient_direction'`` (degrees), ``'peak_to_valley'`` (ADU),
        ``'rms_gradient'``, ``'downsample_factor'``, ``'points_used'``,
        ``'coefficients'`` (sub-dict with ``'dy'``, ``'dx'``, ``'offset'``).

    Raises
    ------
    ValueError
        If *downsample_factor* is less than 1.

    Notes
    -----
    For large sensors, plane fitting on the full image is both
    memory-intensive and slow. Thermal gradients are smooth (spatial
    scale >> pixel size), so adaptive downsampling provides 10–200×
    speedup with negligible accuracy loss (<0.01% gradient coefficient
    error for 4K sensors at 9× downsampling).
    """
    from scipy.ndimage import gaussian_filter

    dq     = create_bpm(frame.shape, dtype=bpm_dtype)
    ny, nx = frame.shape
    smoothed = gaussian_filter(frame.astype(float), sigma=smoothing_sigma)

    if downsample_factor is None:
        downsample_factor = _compute_downsample_factor(frame.shape)
        logger.debug(
            "detect_thermal_gradient: adaptive %dx downsampling.",
            downsample_factor,
        )
    elif downsample_factor < 1:
        raise ValueError(
            f"downsample_factor must be >= 1; got {downsample_factor}."
        )

    if downsample_factor > 1:
        smoothed_ds = smoothed[::downsample_factor, ::downsample_factor]
    else:
        smoothed_ds = smoothed

    ny_ds, nx_ds = smoothed_ds.shape
    y_coords_ds, x_coords_ds = np.mgrid[
        0:ny:downsample_factor, 0:nx:downsample_factor
    ]

    mask = np.zeros((ny_ds, nx_ds), dtype=bool)
    if existing_bpm is not None:
        mask |= existing_bpm[::downsample_factor, ::downsample_factor] > 0

    valid = ~mask
    if int(np.sum(valid)) < 100:
        logger.warning(
            "detect_thermal_gradient: fewer than 100 valid pixels; "
            "skipping gradient fit."
        )
        return dq, {
            "gradient_detected": False, "max_gradient": None,
            "gradient_direction": None, "peak_to_valley": None,
            "rms_gradient": None,
            "downsample_factor": downsample_factor,
            "points_used": int(np.sum(valid)),
        }

    y_flat = y_coords_ds[valid]
    x_flat = x_coords_ds[valid]
    z_flat = smoothed_ds[valid]

    A = np.column_stack([y_flat, x_flat, np.ones_like(y_flat)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    a, b, c = coeffs

    gradient_magnitude = float(np.sqrt(a ** 2 + b ** 2))
    gradient_direction = float(np.degrees(np.arctan2(a, b)))

    y_full, x_full = np.mgrid[0:ny, 0:nx]
    plane_fit = a * y_full + b * x_full + c
    p2v = float(np.max(plane_fit) - np.min(plane_fit))

    grad_y, grad_x = np.gradient(smoothed)
    if existing_bpm is not None:
        valid_full = ~(existing_bpm > 0)
    else:
        valid_full = np.ones((ny, nx), dtype=bool)
    rms_grad = float(
        np.sqrt(np.mean(grad_y[valid_full] ** 2 + grad_x[valid_full] ** 2))
    )

    stats: dict[str, Any] = {
        "gradient_detected": bool(gradient_magnitude > max_gradient),
        "max_gradient":      gradient_magnitude,
        "gradient_direction": gradient_direction,
        "peak_to_valley":    p2v,
        "rms_gradient":      rms_grad,
        "downsample_factor": int(downsample_factor),
        "points_used":       int(np.sum(valid)),
        "coefficients": {
            "dy":     float(a),
            "dx":     float(b),
            "offset": float(c),
        },
    }
    return dq, stats
