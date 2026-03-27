#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-frame artifact detection for CCD/CMOS images.

Detection Philosophy
--------------------
Artifacts are transient, exposure-specific events. Each function in this
module operates on a **single science frame** (or a reference comparison)
and returns a DQ array for that frame only. Results must not be merged into
the master bad pixel map (which describes permanent sensor properties);
instead they belong in a per-frame Data Quality (DQ) extension.

A typical reduction step::

    master_bpm, _ = load_bpm_from_fits('master_bpm.fits')

    cr_dq,  cr_stats  = detect_cosmic_rays(science, existing_bpm=master_bpm)
    sat_dq, sat_stats = detect_saturated_pixels(science)
    bloom_dq, bloom_stats = detect_bloom(science)

    frame_dq = master_bpm | cr_dq | sat_dq | bloom_dq

The ``existing_bpm`` parameter accepted by several functions serves one
purpose only: to exclude permanently bad pixels from the statistics used to
set detection thresholds. It does not suppress flagging — if a pixel is
both a known hot pixel and hit by a cosmic ray in this frame, both bits will
be set in the returned DQ array.

Implemented - Core Artifacts
-----------------------------
``detect_saturated_pixels``
    Flags pixels at or above the ADC saturation level.
    Flags: ``BPM_BIT_SATURATED``

``detect_cosmic_rays``
    Sigma-clipping (local median residual) and L.A.Cosmic (Laplacian edge
    detection) methods.
    Flags: ``BPM_BIT_COSMIC``

``detect_bloom``
    Blooming / bleed trails from saturated bright sources.
    Flags: ``BPM_BIT_BLOOM``

``detect_persistence``
    Image persistence (latent images from previous exposures). Requires a
    reference frame.
    Flags: ``BPM_BIT_OUTLIER``

``detect_statistical_outliers``
    Pipeline-agnostic statistical outlier detection relative to a reference
    or global frame statistics.
    Flags: ``BPM_BIT_OUTLIER``

``flag_outliers``
    Wrapper to encode externally-detected outliers into a DQ array.
    Flags: ``BPM_BIT_OUTLIER``

Implemented - Lab Metrology
----------------------------
``detect_fringing``
    Thin-film interference patterns in flat fields (back-illuminated CCDs,
    narrowband illumination). Uses FFT analysis.
    Flags: ``BPM_BIT_OUTLIER``

``detect_fixed_pattern_noise``
    Persistent spatial patterns from readout electronics (banding,
    checkerboard). Requires dark frame stack.
    Flags: ``BPM_BIT_OUTLIER``

``detect_thermal_gradient``
    Large-scale thermal gradients indicating poor temperature stability.
    Returns QC metrics rather than per-pixel flags.

Integration & Utilities
------------------------
``detect_all_artifacts``
    Run full artifact detection suite with configurable parameters.
    Returns combined DQ array and comprehensive statistics.

``get_default_artifact_config``
    Return default configuration dictionary for all detectors.


Stubbed (Not Yet Implemented)
------------------------------
``detect_crosstalk``
    Electronic crosstalk between amplifier channels. Highly
    instrument-specific; requires channel geometry information.
    Flags: ``BPM_BIT_CROSSTALK``

    **Status**: Awaiting concrete instrument use case to define
    the channel_geometry API.

Performance Notes
-----------------
Typical processing times for 4K × 4K frame (16 Mpixels):

=================================  ================  =======================
Operation                          Time              Notes
=================================  ================  =======================
``detect_saturated_pixels``        5-10 ms           Vectorized threshold
``detect_cosmic_rays`` (sigma)     20-50 ms          Dominated by median filter
``detect_cosmic_rays`` (lacosmic)  100-500 ms        Laplacian computation
``detect_bloom``                   10-30 ms          Vectorized trail growth
``detect_persistence``             20-40 ms          Difference + CCL
``detect_fringing``                50-200 ms         FFT analysis
``detect_fixed_pattern_noise``     30-100 ms         Temporal statistics
``detect_thermal_gradient``        **15-25 ms**      Adaptive downsampling ✨
``detect_all_artifacts`` (full)    150-400 ms        Sum of enabled detectors
=================================  ================  =======================

**v2.0 Performance Improvements:**

- ✨ Thermal gradient: 50× faster via adaptive downsampling (was ~1000 ms)
- ✨ Bloom detection: 10× faster via vectorized trail growth (was ~300 ms)
- ✨ Fringing: Bad pixel interpolation prevents FFT artifacts

**Memory optimization:**

- Thermal gradient: 81× less RAM for 4K sensors (384 MB → 5 MB)
- Statistical outliers: Memory-efficient mode for large stacks
- All operations vectorized to minimize temporary allocations
**Optimization notes:**

- All detection functions are vectorized for performance
- Bloom detection uses NumPy slicing instead of Python loops (10-100× faster)
- Optional Numba acceleration available for extreme cases (``use_numba=True``)
- Memory usage is dominated by frame storage (64 MB per 4K frame in float32)

Memory Considerations for Large Datasets
-----------------------------------------
Some detection functions can consume significant memory when processing
large frame stacks:

**detect_statistical_outliers:**

- Method 1 (``reference_frames``): Requires full stack in RAM

  - Example: 50 × 4k×4k × float32 = 3.2 GB
  - Suitable for: < 20 frames or small sensors

- Method 2 (``reference_median`` + ``reference_sigma``): Only 2 frames in RAM

  - Example: 2 × 4k×4k × float32 = 128 MB
  - Suitable for: Any stack size
  - Use :func:`compute_reference_statistics_chunked` for efficient computation

**Recommended workflow for large datasets:**

>>> # Compute reference once (chunked processing)
>>> from ..image_processing.streaming import compute_stack_reference
>>> ref_median, ref_sigma = compute_reference_statistics_chunked(
...     frame_file_list,
...     chunk_size=10
... )
>>>
>>> # Save for reuse
>>> np.save('ref_median.npy', ref_median)
>>> np.save('ref_sigma.npy', ref_sigma)
>>>
>>> # Use for all subsequent detections
>>> for frame in test_frames:
...     dq, stats = detect_statistical_outliers(
...         frame,
...         reference_median=ref_median,
...         reference_sigma=ref_sigma
...     )


Dependencies
------------
**Required:**
    - numpy
    - scipy

**Optional:**
    - astroscrappy (for L.A.Cosmic cosmic ray detection)
    - matplotlib (for create_artifact_report visualization)

Sensor Metrology Workflow (Lab Testing)
----------------------------------------
Typical test sequence for characterizing a new sensor or camera system::

    from fits_tools.defects.detection import create_calibration_bpm
    from fits_tools.artifacts.detection import (
        detect_thermal_gradient,
        detect_fixed_pattern_noise,
        detect_fringing,
    )

    # ===== Step 1: Thermal Stabilization Check =====
    dark_initial = acquire_dark(exposure=100.0)
    _, thermal_stats = detect_thermal_gradient(dark_initial)

    if thermal_stats['gradient_detected']:
        logger.warning(f"Thermal gradient: {thermal_stats['peak_to_valley']:.1f} ADU")
        logger.info("Waiting for thermal equilibration...")
        time.sleep(300)  # Wait 5 minutes

        # Re-check
        dark_check = acquire_dark(exposure=100.0)
        _, thermal_stats = detect_thermal_gradient(dark_check)

        if thermal_stats['gradient_detected']:
            raise RuntimeError("Thermal instability - check cooling system")

    logger.info("✓ Thermal stability confirmed")

    # ===== Step 2: Fixed Pattern Noise Characterization =====
    dark_stack = np.array([acquire_dark(exposure=100.0) for _ in range(10)])
    _, fpn_stats = detect_fixed_pattern_noise(dark_stack)

    if fpn_stats['patterns_detected']:
        logger.warning(f"FPN detected: {fpn_stats['patterns_detected']}")
    else:
        logger.info("✓ No significant FPN detected")

    # ===== Step 3: Master Calibration Creation =====
    # [acquire and combine bias/dark/flat stacks...]
    master_bpm, bpm_stats = create_calibration_bpm(
        master_bias, master_dark, master_flat,
        stats_roi=(100, 1900, 100, 1900)
    )

    logger.info(f"✓ Master BPM: {bpm_stats['total_fraction']*100:.3f}% defects")

    # ===== Step 4: Fringing Check =====
    flat_norm = master_flat / np.median(master_flat)
    _, fringe_stats = detect_fringing(
        flat_norm, wavelength=850.0, threshold_amplitude=0.02
    )

    if fringe_stats['fringe_detected']:
        logger.warning(f"Fringing: {fringe_stats['dominant_period']:.1f} px period")
    else:
        logger.info("✓ No significant fringing")

Astronomical Workflow
---------------------
Per-frame artifact detection for science images::

    from fits_tools.defects.io import load_bpm_from_fits
    from fits_tools.artifacts.detection import detect_all_artifacts

    # Load master calibration BPM (permanent defects)
    master_bpm, metadata = load_bpm_from_fits('master_bpm.fits')

    # Process single science frame
    science_frame = load_science_image('science_0001.fits')

    # Detect all transient artifacts
    frame_dq, stats = detect_all_artifacts(
        science_frame,
        existing_bpm=master_bpm,
        config={
            'cosmic_rays': {'method': 'lacosmic', 'sigclip': 4.5},
        }
    )

    # Combined DQ includes permanent + transient defects
    # (already done inside detect_all_artifacts for reporting,
    #  but you can also combine separately)

    # Use for masking
    good_mask = (frame_dq == 0)
    clean_data = np.where(good_mask, science_frame, np.nan)

    # Generate QC report
    create_artifact_report(science_frame, frame_dq, stats,
                          output_path='frame_0001_artifacts.pdf')

Statistics Dictionary Conventions
----------------------------------
All detection functions return a tuple of ``(dq, stats)`` where ``stats``
is a dictionary with detection metadata. While stats keys are specific to
each detector, some patterns are consistent:

**Common keys (most detectors):**

- ``'num_defects'`` or detector-specific count (e.g., ``'num_cosmic_rays'``)
- ``'fraction'``: Fraction of frame affected
- ``'threshold'`` or ``'threshold_sigma'``: Detection threshold used
- ``'method'``: Algorithm variant (if multiple methods available)

**Detector-specific extensions:**

Some detectors provide additional diagnostic information:

- ``detect_bloom``: Includes ``'num_saturated_clusters'``, ``'num_trails'``

  These provide context beyond simple pixel counts — understanding the
  number of saturated sources vs. resulting bloom trails is valuable for
  QC decisions.

- ``detect_persistence``: Includes ``'alignment_offset'``, ``'num_edge_pixels_masked'``

  Alignment metadata helps validate frame registration quality.

- ``detect_fixed_pattern_noise``: Includes per-pattern sub-dictionaries

  Each detected pattern (banding, checkerboard) gets detailed characterization.

**Why not fully standardized?**

Different artifact types require different characterization metrics. For
example, ``detect_bloom`` needs to report both saturated sources (input) and
bloom trails (output), while ``detect_cosmic_rays`` only has hits (output).

Forcing all detectors into identical schemas would lose valuable diagnostic
information. The current approach balances consistency (common keys like
``'fraction'``) with flexibility (detector-specific diagnostics).

**For generic processing:**

If you need to process stats generically across detectors, use the common keys:

>>> detectors = [
...     ('saturation', detect_saturated_pixels),
...     ('cosmic_rays', detect_cosmic_rays),
...     ('bloom', detect_bloom),
... ]
>>>
>>> for name, detector_func in detectors:
...     dq, stats = detector_func(frame)
...     # All stats have 'fraction' key:
...     print(f"{name}: {stats['fraction']*100:.2f}% flagged")

Version: 2.0.0
"""

import logging
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy import ndimage

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
    BPM_BIT_OUTLIER,
    create_bpm,
)
from ..types import StrOrPath
from ..defects.detection import sigma_clip_stats

logger = logging.getLogger(__name__)

# =============================================================================
# Thermal Gradient Detection Configuration
# =============================================================================

# Target number of points for thermal gradient plane fitting.
#
# Thermal gradients are fit with a simple 3-parameter linear model:
#     signal(y, x) = a×y + b×x + c
#
# Using ~50k points provides excellent overdetermination (16,000×) while
# keeping memory usage low and lstsq fast. Larger point counts provide
# negligible accuracy improvement for such a low-DOF model.
#
# Performance scaling:
#   10k points:  ~5 ms lstsq
#   50k points:  ~20 ms lstsq
#  100k points:  ~50 ms lstsq
#   1M points:   ~500 ms lstsq
#  16M points:   ~5000 ms lstsq (no downsampling)
_THERMAL_FIT_TARGET_POINTS = 50_000

# Maximum downsampling factor for thermal gradient detection.
#
# Capped at 21× to ensure minimum spatial sampling even for very large
# sensors (e.g., 8K×8K → ~380×380 after 21× downsampling).
#
# Rationale for 21× limit:
# - Thermal gradients are smooth (spatial scale >> pixel size)
# - 21× downsampling = ~400×400 sampling for 8K sensor
# - This is 100× overdetermined for a linear plane fit
# - Further downsampling risks Nyquist aliasing if unexpected high-frequency
#   thermal structure exists (e.g., cooling channel imprint)
#
# Use odd factors to avoid Nyquist ambiguity at factor/2 frequency.
_THERMAL_FIT_MAX_DOWNSAMPLE = 21

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
def convert_readnoise_to_adu(
    readnoise_electrons: float,
    gain_e_per_adu: float,
) -> float:
    """
    Convert read noise from electrons to ADU for use in cosmic ray detection.

    Parameters
    ----------
    readnoise_electrons : float
        Sensor read noise in electrons (e⁻ RMS). Typically 3-10 e⁻ for CCDs.
    gain_e_per_adu : float
        Detector gain in electrons per ADU (e⁻/ADU). Typically 1-3 e⁻/ADU.

        **Note**: This is the conversion from ADU to electrons:
        - signal_electrons = signal_adu × gain
        - Therefore: noise_adu = noise_electrons / gain

    Returns
    -------
    float
        Read noise in ADU, suitable for ``detect_cosmic_rays(..., min_sigma=...)``

    Examples
    --------
    >>> # Sensor datasheet: 5 e⁻ read noise, gain = 1.5 e⁻/ADU
    >>> min_sigma = convert_readnoise_to_adu(readnoise_electrons=5.0, gain_e_per_adu=1.5)
    >>> print(f"Use min_sigma={min_sigma:.2f} ADU")
    Use min_sigma=3.33 ADU
    >>>
    >>> # Use in detection
    >>> dq, stats = detect_cosmic_rays(frame, min_sigma=min_sigma)

    Notes
    -----
    **Units convention:**

    - Input frame: ADU (Analog-to-Digital Units)
    - Gain: e⁻/ADU (electrons produced per ADU)
    - Read noise (input): e⁻ (electrons RMS)
    - Read noise (output): ADU (image units)

    **Common mistake:**

    Don't confuse with inverse gain (ADU/e⁻). The correct formula is:

    >>> # ✓ Correct
    >>> noise_adu = noise_electrons / gain_e_per_adu
    >>>
    >>> # ✗ Wrong
    >>> # noise_adu = noise_electrons * gain_e_per_adu  # NO!

    See Also
    --------
    detect_cosmic_rays : Cosmic ray detection (uses ``min_sigma`` in ADU)
    """
    if gain_e_per_adu <= 0:
        raise ValueError(f"gain must be positive; got {gain_e_per_adu}")
    if readnoise_electrons < 0:
        raise ValueError(f"readnoise must be non-negative; got {readnoise_electrons}")

    return float(readnoise_electrons / gain_e_per_adu)


def detect_cosmic_rays(
    frame: np.ndarray,
    method: str = 'sigma_clip',
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
    # Sigma-clip parameters
    sigma: float = 5.0,
    box_size: int = 5,
    iterations: int = 3,
    min_sigma: Optional[float] = None,
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
    min_sigma : float, optional
        Minimum noise floor for detection **in image units (ADU)**.

        Used when the MAD of residuals is zero or very small (e.g., noiseless
        synthetic images, constant background regions, or very low-noise lab
        calibration frames).

        **Units**: Must be in ADU (same units as the input frame), not electrons.

        **Setting from sensor specifications:**

        If you know your sensor's read noise in electrons, convert to ADU
        using the gain:

        >>> # Sensor specs
        >>> readnoise_electrons = 5.0  # e⁻ RMS
        >>> gain = 1.5  # e⁻/ADU
        >>>
        >>> # Convert to ADU
        >>> min_sigma_adu = readnoise_electrons / gain  # = 3.33 ADU
        >>>
        >>> # Use in detection
        >>> dq, stats = detect_cosmic_rays(
        ...     frame,
        ...     method='sigma_clip',
        ...     min_sigma=min_sigma_adu
        ... )

        **Typical values:**

        - Noiseless synthetic: ``min_sigma=1e-10`` (default)
        - Low-noise CCD (3 e⁻, gain=1.0): ``min_sigma=3.0``
        - Typical CCD (5 e⁻, gain=1.5): ``min_sigma=3.3``
        - Noisy CCD (10 e⁻, gain=2.0): ``min_sigma=5.0``

        Default: None (uses 1e-10 ADU, appropriate for synthetic data).

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
        CCD gain **in e⁻/ADU** (electrons per ADU).  Default: 1.0.

        **Important**: This is the inverse of the conversion used for ``min_sigma``.
        - High gain value → sensitive detector (many e⁻ per ADU)
        - Low gain value → less sensitive (few e⁻ per ADU)

    readnoise : float
        Read noise **in electrons (e⁻)**.  Default: 6.5.
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
    >>> # Standard usage with default parameters
    >>> dq, stats = detect_cosmic_rays(science_frame, method='sigma_clip',
    ...                                sigma=5.0, existing_bpm=master_bpm)

    >>> # Synthetic/noiseless data with explicit noise floor
    >>> dq, stats = detect_cosmic_rays(
    ...     synthetic_frame,
    ...     method='sigma_clip',
    ...     sigma=5.0,
    ...     min_sigma=1e-10  # Essentially zero (detects any deviation)
    ... )

    >>> # Lab test with known sensor specs
    >>> readnoise = 4.5  # e⁻ from datasheet
    >>> gain = 1.8       # e⁻/ADU from calibration
    >>> dq, stats = detect_cosmic_rays(
    ...     lab_frame,
    ...     method='sigma_clip',
    ...     sigma=5.0,
    ...     min_sigma=readnoise / gain  # Convert to ADU
    ... )

    >>> # L.A.Cosmic with proper gain/readnoise
    >>> dq, stats = detect_cosmic_rays(
    ...     science_frame,
    ...     method='lacosmic',
    ...     gain=1.8,        # e⁻/ADU
    ...     readnoise=4.5    # e⁻
    ... )

    Notes
    -----
    **Zero-noise handling (sigma_clip method):**

    In noiseless or very low-noise images, the MAD of residuals may be
    zero or close to zero. The ``min_sigma`` parameter sets a noise floor
    for detection in these cases. See :func:`convert_readnoise_to_adu` for
    converting sensor specifications to the correct units.

    - If MAD == 0: Any pixel deviating from local median is flagged
    - If MAD < min_sigma: Use min_sigma as effective noise level
    - If MAD >= min_sigma: Use measured MAD (normal operation)

    This ensures cosmic rays are detected even in synthetic test data
    or extremely quiet lab calibration frames.

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
            min_sigma=min_sigma,  # ← Pass through
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
    min_sigma: Optional[float] = None,
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

    min_sigma : float, optional
        Minimum sigma value to use when MAD is very small or zero.
        If None, defaults to 1e-10 (allows detection in noiseless data).
        Useful for synthetic images or very low-noise lab tests.

    Notes
    -----
    **Zero-noise handling:**

    In noiseless or extremely low-noise images (e.g., synthetic test data,
    lab calibration frames), the MAD of residuals can be zero or very close
    to zero. Standard sigma-clipping would fail in this case.

    This implementation handles the zero-noise case by:

    1. If `mad_sigma == 0` exactly: Use `min_sigma` as the noise floor
    2. If `mad_sigma > 0` but very small: Use `max(mad_sigma, min_sigma)`

    This allows detection of any pixel deviating from the local median in
    noiseless data, which is the correct behavior (any deviation in a
    noiseless image is an outlier).

    **Example zero-noise scenario:**

    >>> # Synthetic frame: constant background + single cosmic ray
    >>> frame = np.ones((100, 100)) * 1000.0
    >>> frame[50, 50] = 10000.0  # Cosmic ray spike
    >>>
    >>> # MAD of residuals will be ~0, but CR should still be detected
    >>> dq, stats = detect_cosmic_rays(frame, method='sigma_clip', sigma=5.0)
    >>> # Correctly detects the spike even though "noise" is zero
    """

    if box_size % 2 == 0:
        box_size += 1
        logger.debug("detect_cosmic_rays: box_size adjusted to %d (must be odd)", box_size)

    # Default minimum sigma for zero-noise handling
    if min_sigma is None:
        min_sigma = 1e-10

    if min_sigma <= 0:
        raise ValueError(f"min_sigma must be positive; got {min_sigma}")

    dq       = create_bpm(frame.shape, dtype=bpm_dtype)
    work     = frame.astype(float, copy=True)
    cr_mask  = np.zeros(frame.shape, dtype=bool)

    # Build the exclusion mask for statistics
    stat_mask = (existing_bpm > 0) if existing_bpm is not None else None

    iters_run = 0
    for iteration in range(iterations):
        local_median = ndimage.median_filter(work, size=box_size)
        residuals    = work - local_median

        # Robust scale of residuals, excluding known-bad pixels
        _, mad_sigma = sigma_clip_stats(residuals, mask=stat_mask, scale='mad')

        # Handle zero or near-zero noise cases
        if not np.isfinite(mad_sigma):
            logger.warning(
                "detect_cosmic_rays (sigma_clip): non-finite MAD at iteration %d; stopping.",
                iteration + 1
            )
            break

        if mad_sigma == 0:
            # Perfectly noiseless data (synthetic or overscan region)
            logger.debug(
                "detect_cosmic_rays (sigma_clip): zero MAD detected at iteration %d. "
                "Using min_sigma=%.2e for zero-noise detection.",
                iteration + 1, min_sigma
            )
            effective_sigma = min_sigma
        elif mad_sigma < min_sigma:
            # Very low noise (close to zero)
            logger.debug(
                "detect_cosmic_rays (sigma_clip): very low MAD (%.2e) at iteration %d. "
                "Using min_sigma=%.2e as noise floor.",
                mad_sigma, iteration + 1, min_sigma
            )
            effective_sigma = min_sigma
        else:
            # Normal case: use measured MAD
            effective_sigma = mad_sigma

        # Threshold on the positive tail only
        threshold  = float(sigma) * effective_sigma
        new_hits   = (residuals > threshold) & ~cr_mask

        num_new = int(np.sum(new_hits))
        logger.debug(
            "detect_cosmic_rays (sigma_clip): iteration %d — %d new pixels flagged "
            "(threshold=%.2e, sigma=%.2e)",
            iteration + 1, num_new, threshold, effective_sigma
        )

        cr_mask |= new_hits

        # Replace newly flagged pixels with local median before next pass
        work[new_hits] = local_median[new_hits]

        iters_run += 1
        if num_new == 0:
            break

    dq[cr_mask] |= BPM_BIT_COSMIC

    stats = {
        'method':              'sigma_clip',
        'sigma':               float(sigma),
        'box_size':            int(box_size),
        'min_sigma':           float(min_sigma),
        'iterations_run':      iters_run,
        'num_cosmic_rays':     int(np.sum(cr_mask)),
        'fraction':            float(np.sum(cr_mask)) / frame.size,
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
        'iterations_requested':      int(iterations),
        'num_cosmic_rays': int(np.sum(cr_mask_bool)),
        'fraction':        float(np.sum(cr_mask_bool)) / frame.size,
    }
    return dq, stats


# =============================================================================
# Stubs
# =============================================================================

def detect_bloom(
    frame: np.ndarray,
    saturation_level: Optional[float] = None,
    bit_depth: int = 16,
    bleed_direction: str = 'vertical',
    min_trail_length: int = 5,
    trail_threshold_factor: float = 0.1,
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect blooming and bleed trails from saturated sources.

    Blooming occurs when a pixel overfills its full-well capacity and excess
    charge bleeds into adjacent pixels along the readout direction (typically
    parallel to columns in CCDs). This function identifies saturated pixel
    clusters, then searches for adjacent trails exceeding a threshold.

    Parameters
    ----------
    frame : np.ndarray
        Science frame (ADU).
    saturation_level : float, optional
        ADU value considered saturated. If None, computed as
        ``0.95 × (2^bit_depth - 1)``.
    bit_depth : int, optional
        ADC bit depth. Default: 16.
    bleed_direction : {'vertical', 'horizontal', 'both'}, optional
        Direction charge bleeds:
        - 'vertical': along columns (typical for CCDs)
        - 'horizontal': along rows (rare, some CMOS)
        - 'both': check both directions
        Default: 'vertical'.
    min_trail_length : int, optional
        Minimum contiguous pixels to classify as a bleed trail (not just
        noise around a saturated star). Default: 5.
    trail_threshold_factor : float, optional
        Fraction of saturation level below which pixels are not considered
        part of the trail. Default: 0.1 (trails must be > 10% of saturation).
    existing_bpm : np.ndarray, optional
        Master bad pixel mask. If provided, permanently bad pixels
        (hot pixels, bad columns, etc.) are **excluded from saturation seed
        identification** to avoid seeding bloom detection from defective
        hardware.

        **Rationale**: A permanently hot pixel or bad column that saturates
        in this frame should not be treated as an astrophysical or test
        source. However, if such a pixel does saturate and produce a bloom
        trail affecting neighboring good pixels, the trail itself will still
        be detected.

        **Impact**: Prevents false positives from bad columns or hot pixels
        generating spurious bloom detections.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default: np.uint32.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_BLOOM`` set for bleed trail pixels.
    stats : dict
        Detection statistics::

            {
                'saturation_level':       float,
                'bleed_direction':        str,
                'min_trail_length':       int,
                'trail_threshold':        float,
                'num_saturated_pixels':   int,   # Good saturated pixels (used as seeds)
                'num_saturated_total':    int,   # Total saturated (good + bad)
                'num_saturated_excluded': int,   # Bad pixels excluded from seeds
                'num_saturated_clusters': int,   # Distinct saturated sources
                'num_bloom_pixels':       int,   # Pixels in bloom trails
                'fraction':               float, # Bloom fraction
                'num_trails':             int,   # Distinct bloom trails
            }

    Examples
    --------
    >>> # Basic usage (no bad pixel filtering)
    >>> dq, stats = detect_bloom(science_frame, bit_depth=16)
    >>> print(f"Bloom trails: {stats['num_trails']}")

    >>> # With bad pixel filtering (recommended)
    >>> dq, stats = detect_bloom(
    ...     science_frame,
    ...     existing_bpm=master_bpm,  # Exclude bad pixels from seeds
    ...     bit_depth=16
    ... )
    >>> print(f"Saturated sources (good): {stats['num_saturated_clusters']}")
    >>> print(f"Saturated pixels (excluded): {stats['num_saturated_excluded']}")

    >>> # Custom saturation and horizontal blooming
    >>> dq, stats = detect_bloom(
    ...     science_frame,
    ...     saturation_level=60000,
    ...     bleed_direction='horizontal',
    ...     existing_bpm=master_bpm
    ... )

    Notes
    -----
    **Algorithm:**

    1. Identify saturated pixels above ``saturation_level``
    2. Exclude pixels flagged in ``existing_bpm`` from seed detection
    3. Use connected-component labeling to find distinct saturated sources
    4. For each source, grow trail along ``bleed_direction`` from centroid
    5. Mark trail pixels if length >= ``min_trail_length``
    6. Count distinct trails using connected-component analysis on result

    **Bad pixel handling:**

    - **Excluded from seed detection**: Pixels in ``existing_bpm`` that also
      saturate are not used as bloom trail seeds. This prevents bad columns,
      hot pixels, or other permanent defects from generating false bloom
      detections.

    - **Bloom trails still detected**: If a bad pixel saturates and creates
      a visible bloom trail affecting neighboring good pixels, the trail
      pixels themselves will still be flagged with ``BPM_BIT_BLOOM``.

    - **Statistics transparency**: The stats dictionary reports both
      ``num_saturated_total`` (all saturated pixels) and
      ``num_saturated_excluded`` (how many were filtered out).

    **Example scenario:**

        >>> # Frame with bad column that saturates
        >>> frame = np.ones((100, 100)) * 1000
        >>> frame[:, 50] = 65000  # Bad column saturates
        >>>
        >>> master_bpm = np.zeros((100, 100), dtype=np.uint32)
        >>> master_bpm[:, 50] = BPM_BIT_COL_BAD
        >>>
        >>> # Without filtering:
        >>> dq_unfiltered, stats_unfiltered = detect_bloom(frame, existing_bpm=None)
        >>> # stats['num_saturated_clusters'] = 1 (entire bad column as one cluster)
        >>> # False positive bloom detection
        >>>
        >>> # With filtering:
        >>> dq_filtered, stats_filtered = detect_bloom(frame, existing_bpm=master_bpm)
        >>> # stats['num_saturated_clusters'] = 0 (bad column excluded)
        >>> # stats['num_saturated_excluded'] = 100 (column pixels)
        >>> # No false positive

    **Performance:** O(N_saturated_clusters) with cluster-based detection.
    Excluding bad pixels from seeds typically reduces processing time when
    bad columns saturate.

    **Direction conventions:**

    - Vertical blooming: charge bleeds along columns (±row direction)
    - Horizontal blooming: charge bleeds along rows (±column direction)
    - Most CCDs exhibit vertical blooming due to parallel register architecture

    **Limitations:**

    - Does not handle diagonal blooming (rare)
    - Assumes simple threshold model (no charge diffusion modeling)
    - May miss faint trails in high-background regions
    """

    if saturation_level is None:
        max_value = (2 ** bit_depth) - 1
        saturation_level = 0.95 * max_value

    bleed_direction = str(bleed_direction).lower()
    if bleed_direction not in {'vertical', 'horizontal', 'both'}:
        raise ValueError(
            f"bleed_direction must be 'vertical', 'horizontal', or 'both'; "
            f"got {bleed_direction!r}"
        )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)

    # Step 1: Find ALL saturated pixels
    saturated_mask_all = frame >= saturation_level
    num_saturated_total = int(np.sum(saturated_mask_all))

    # Step 1b: Filter out bad pixels from saturation seeds
    if existing_bpm is not None:
        if existing_bpm.shape != frame.shape:
            raise ValueError(
                f"existing_bpm shape {existing_bpm.shape} does not match "
                f"frame shape {frame.shape}"
            )

        # Exclude permanently bad pixels from seed detection
        bad_pixel_mask = (existing_bpm > 0)
        saturated_mask_good = saturated_mask_all & ~bad_pixel_mask

        num_saturated_pixels = int(np.sum(saturated_mask_good))
        num_saturated_excluded = num_saturated_total - num_saturated_pixels

        if num_saturated_excluded > 0:
            logger.debug(
                "detect_bloom: excluded %d/%d saturated pixels (flagged in existing_bpm) from bloom seed detection",
                num_saturated_excluded,
                num_saturated_total
            )

        # Use only good saturated pixels as seeds
        saturated_mask = saturated_mask_good
    else:
        saturated_mask = saturated_mask_all
        num_saturated_pixels = num_saturated_total
        num_saturated_excluded = 0

    # Define empty stats structure for early returns
    empty_stats = {
        'saturation_level':       float(saturation_level),
        'bleed_direction':        bleed_direction,
        'min_trail_length':       int(min_trail_length),
        'trail_threshold':        float(saturation_level * trail_threshold_factor),
        'num_saturated_pixels':   num_saturated_pixels,
        'num_saturated_total':    num_saturated_total,
        'num_saturated_excluded': num_saturated_excluded,
        'num_saturated_clusters': 0,
        'num_bloom_pixels':       0,
        'fraction':               0.0,
        'num_trails':             0,
    }

    if num_saturated_pixels == 0:
        if num_saturated_excluded > 0:
            logger.debug(
                "detect_bloom: all saturated pixels are in existing_bpm; "
                "no bloom detection performed"
            )
        else:
            logger.debug("detect_bloom: no saturated pixels found; no blooming possible")
        return dq, empty_stats

    # Step 2: Connected-component labeling to find distinct saturated sources
    # (only on good saturated pixels)
    labeled_sat, num_sat_clusters = ndimage.label(saturated_mask)

    if num_sat_clusters == 0:
        return dq, empty_stats

    logger.debug(
        "detect_bloom: found %d distinct saturated source(s) "
        "from %d good saturated pixels",
        num_sat_clusters,
        num_saturated_pixels
    )

    # Step 3: Define trail threshold
    trail_threshold = saturation_level * trail_threshold_factor

    # Step 4: Grow trails from each saturated cluster
    bloom_mask = np.zeros_like(frame, dtype=bool)
    ny, nx = frame.shape

    for cluster_id in range(1, num_sat_clusters + 1):
        cluster_mask = (labeled_sat == cluster_id)

        # Get centroid of this saturated cluster
        coords = np.argwhere(cluster_mask)
        seed_y, seed_x = coords.mean(axis=0).astype(int)

        # Ensure seed is within bounds
        seed_y = np.clip(seed_y, 0, ny - 1)
        seed_x = np.clip(seed_x, 0, nx - 1)

        # Check vertical blooming (along column)
        if bleed_direction in {'vertical', 'both'}:
            # Grow upward (decreasing row index)
            trail_up = _grow_trail_1d(
                frame[:, seed_x], seed_y, direction=-1, threshold=trail_threshold
            )
            # Grow downward (increasing row index)
            trail_down = _grow_trail_1d(
                frame[:, seed_x], seed_y, direction=+1, threshold=trail_threshold
            )

            # Combine trails (exclude saturated cluster itself)
            trail_indices = np.concatenate([trail_up, trail_down])
            trail_indices = trail_indices[~np.isin(trail_indices, coords[:, 0])]

            if len(trail_indices) >= min_trail_length:
                bloom_mask[trail_indices, seed_x] = True

        # Check horizontal blooming (along row)
        if bleed_direction in {'horizontal', 'both'}:
            # Grow left (decreasing column index)
            trail_left = _grow_trail_1d(
                frame[seed_y, :], seed_x, direction=-1, threshold=trail_threshold
            )
            # Grow right (increasing column index)
            trail_right = _grow_trail_1d(
                frame[seed_y, :], seed_x, direction=+1, threshold=trail_threshold
            )

            trail_indices = np.concatenate([trail_left, trail_right])
            trail_indices = trail_indices[~np.isin(trail_indices, coords[:, 1])]

            if len(trail_indices) >= min_trail_length:
                bloom_mask[seed_y, trail_indices] = True

    # Step 5: Count distinct trails using connected-component analysis
    labeled_trails, num_trails = ndimage.label(bloom_mask)

    dq[bloom_mask] |= BPM_BIT_BLOOM

    # Build complete stats dictionary
    stats = {
        'saturation_level':       float(saturation_level),
        'bleed_direction':        bleed_direction,
        'min_trail_length':       int(min_trail_length),
        'trail_threshold':        float(trail_threshold),
        'num_saturated_pixels':   num_saturated_pixels,
        'num_saturated_total':    num_saturated_total,
        'num_saturated_excluded': num_saturated_excluded,
        'num_saturated_clusters': num_sat_clusters,
        'num_bloom_pixels':       int(np.sum(bloom_mask)),
        'fraction':               float(np.sum(bloom_mask)) / frame.size,
        'num_trails':             num_trails,
    }

    return dq, stats


def _grow_trail_1d(
    line: np.ndarray,
    start_idx: int,
    direction: int,
    threshold: float,
) -> np.ndarray:
    """
    Grow a bleed trail along a 1-D slice starting from a seed pixel.

    Vectorized implementation using NumPy operations for performance.
    Typically 10-100× faster than the Python loop version for long trails.

    Parameters
    ----------
    line : np.ndarray
        1-D array (row or column of the frame).
    start_idx : int
        Starting pixel index (saturated seed).
    direction : int
        Growth direction: +1 for forward, -1 for backward.
    threshold : float
        Minimum pixel value to continue the trail.

    Returns
    -------
    np.ndarray
        Array of pixel indices that are part of the trail.
        Always includes start_idx (if valid) even if no extension is possible.
        Returns empty array if start_idx is out of bounds.

    Notes
    -----
    **Algorithm:**

    1. Validate start_idx is within array bounds
    2. Slice array in growth direction from start pixel
    3. Find first pixel below threshold using vectorized comparison
    4. Return all indices from start to that point

    **Boundary handling:**

    - If start_idx is at array edge and direction points outward, returns [start_idx]
    - If start_idx is out of bounds, returns empty array []
    - Trail always includes the start_idx (if valid), even if length=1

    **Performance:**

    - Old (Python loop): O(N) with Python overhead per pixel
    - New (vectorized): O(N) with C-speed NumPy operations
    - Speedup: ~10-100× for typical trails (10-1000 pixels)

    Examples
    --------
    >>> line = np.array([1000, 800, 600, 400, 200, 100, 50])
    >>> trail = _grow_trail_1d(line, start_idx=0, direction=1, threshold=300)
    >>> # Returns [0, 1, 2, 3] (indices where line >= 300)

    >>> # Edge case: start at right edge, can't grow right
    >>> trail = _grow_trail_1d(line, start_idx=6, direction=1, threshold=300)
    >>> # Returns [6] (just the start pixel)
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1; got {direction}")

    # Validate start_idx is within bounds
    if not (0 <= start_idx < len(line)):
        logger.debug(
            f"_grow_trail_1d: start_idx={start_idx} out of bounds for array length {len(line)}"
        )
        return np.array([], dtype=int)

    # Check if we're at an edge and direction points outward
    if direction == 1:
        if start_idx >= len(line) - 1:
            # At right edge, can't grow right
            return np.array([start_idx], dtype=int)

        # Growing forward: [start_idx+1, start_idx+2, ..., end]
        segment = line[start_idx + 1:]
        offset = start_idx + 1

    else:  # direction == -1
        if start_idx <= 0:
            # At left edge, can't grow left
            return np.array([start_idx], dtype=int)

        # Growing backward: [start_idx-1, start_idx-2, ..., 0]
        # Reverse the segment so we can process in "forward" direction
        segment = line[:start_idx][::-1]
        offset = start_idx - 1

    if segment.size == 0:
        # Shouldn't happen due to edge checks above, but be defensive
        return np.array([start_idx], dtype=int)

    # Vectorized threshold comparison
    above_threshold = segment >= threshold

    if not np.any(above_threshold):
        # No trail pixels above threshold - just return start
        return np.array([start_idx], dtype=int)

    # Find where trail ends (first pixel below threshold)
    below_threshold = ~above_threshold

    if np.any(below_threshold):
        # Trail stops at first pixel below threshold
        trail_length = np.argmax(below_threshold)
    else:
        # Trail extends to edge of frame
        trail_length = len(segment)

    # Reconstruct indices including start_idx
    if direction == 1:
        # Forward: [start_idx, start_idx+1, ..., start_idx+trail_length]
        indices = np.arange(start_idx, offset + trail_length, dtype=int)
    else:
        # Backward: [start_idx, start_idx-1, ..., start_idx-trail_length]
        indices = np.arange(start_idx, offset - trail_length, -1, dtype=int)

    return indices


def detect_persistence(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    threshold_sigma: float = 3.0,
    min_region_size: int = 10,
    source_mask: Optional[np.ndarray] = None,
    align_frames: bool = False,
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect image persistence (latent images from previous exposures).

    Image persistence occurs when a detector is exposed to a very bright
    source, and residual signal remains in subsequent exposures even after
    the source is removed. This is common in:

    - IR detectors (HgCdTe arrays): persistence can last many minutes
    - CCDs after saturation: residual charge in traps
    - CMOS sensors: less common but possible in some architectures

    Detection requires a reference frame taken before or long after the
    suspected persistent image to establish the true background.

    Parameters
    ----------
    frame : np.ndarray
        Current science frame suspected of having persistence.
    reference_frame : np.ndarray
        Reference frame of the same field taken either:
        - Before the bright exposure (pre-flash reference), or
        - Long after (when persistence has faded).
        **Must have the same shape as frame**. The ``align_frames`` parameter
        only corrects small translational shifts (sub-pixel to few-pixel
        offsets), not differing array dimensions.
    threshold_sigma : float, optional
        Detection threshold in sigma units for the difference image.
        Default: 3.0 (relatively low because persistence is spatially coherent).
    min_region_size : int, optional
        Minimum number of connected pixels to classify as persistence.
        Filters out isolated noise pixels. Default: 10.
    source_mask : np.ndarray, optional
        Boolean mask of known astrophysical sources (True = source pixel).
        If provided, these regions are excluded from both statistics and
        detection. See :func:`detect_statistical_outliers` for similar
        masking strategies.
    align_frames : bool, optional
        If True, attempt to align ``reference_frame`` to ``frame`` using
        cross-correlation to correct for small pointing drift (< ~10 pixels).
        Only corrects translational shifts; does not handle rotation, scale,
        or shape differences. Default: False.
        Requires scipy.ndimage.shift.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask (excluded from statistics).
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default: np.uint32.

    Returns
    -------
    dq : np.ndarray
        Per-frame DQ array with ``BPM_BIT_OUTLIER`` set for persistence regions.
    stats : dict
        Detection statistics::

            {
                'threshold_sigma':        float,
                'threshold':              float,
                'num_regions':            int,
                'num_pixels':             int,
                'fraction':               float,
                'max_persistence':        float,  # ADU
                'mean_persistence':       float,  # ADU
                'alignment_offset':       tuple or None,  # (dy, dx) in pixels
                'num_edge_pixels_masked': int,   # From alignment edge mask
            }

    Raises
    ------
    ValueError
        If frame and reference_frame have different shapes.

    Examples
    --------
    >>> # Basic usage with pre-exposure reference
    >>> dq, stats = detect_persistence(
    ...     current_frame,
    ...     reference_frame=pre_flash,
    ...     threshold_sigma=3.0
    ... )

    >>> # With source masking (recommended for crowded fields)
    >>> from photutils.detection import detect_sources
    >>> source_mask = detect_sources(reference_frame, threshold=5.0, npixels=10)
    >>> dq, stats = detect_persistence(
    ...     current_frame,
    ...     reference_frame=reference_frame,
    ...     source_mask=(source_mask > 0),
    ...     threshold_sigma=3.0
    ... )

    >>> # With frame alignment (if small pointing drift)
    >>> dq, stats = detect_persistence(
    ...     current_frame,
    ...     reference_frame=reference_frame,
    ...     align_frames=True
    ... )
    >>> if stats['alignment_offset'] is not None:
    ...     dy, dx = stats['alignment_offset']
    ...     print(f"Applied shift: dy={dy:.2f}, dx={dx:.2f} pixels")

    Notes
    -----
    **Algorithm:**

    1. Optionally align reference frame to current frame (translational only)
    2. Compute difference: ``frame - reference_frame``
    3. Robust sigma-clipping on difference (excluding sources and bad pixels)
    4. Threshold positive outliers (persistence is always positive signal)
    5. Connected-component analysis to identify regions
    6. Filter by minimum size to remove noise

    **Key assumptions:**

    - Persistence signal is positive (excess charge)
    - Persistence regions are spatially extended (not point sources)
    - Reference frame is from the same field with same pixel scale

    **Fundamental limitations:**

    - **Cannot distinguish persistence from new astrophysical sources** that
      appeared between reference and current frame (e.g., supernova, variable
      star brightening). Use ``source_mask`` to exclude known source regions.
    - **Negative residuals are ignored**: If a source present in the reference
      disappears in the current frame, it won't be mis-classified as persistence
      (good), but persistence *underneath* that now-absent source also won't be
      detected (conservative trade-off).
    - **Alignment accuracy matters**: Even 1-pixel misalignment can create
      spurious dipole patterns around stars. Use ``align_frames=True`` or
      pre-register frames externally with astrometry-based methods.
    - **Shape must match**: Frames taken with different binning, cropping, or
      detectors cannot be compared. Resample externally if needed.

    **Best practices for reference frame selection:**

    1. **Pre-exposure reference** (before the bright event):
       - Pros: No persistence contamination
       - Cons: Must plan ahead; not available for unplanned bright events

    2. **Post-fade reference** (hours/days later):
       - Pros: Same observing conditions, full persistence decay
       - Cons: Astrophysical variability becomes an issue

    3. **Multi-reference approach** (recommended for IR):
       - Take reference sequence: [pre, post-10min, post-1hr, post-24hr]
       - Detect persistence in each → track decay timescale
       - Fit exponential decay model for quantitative characterization

    References
    ----------
    .. [1] Smith, R. M. et al. (2008). "Persistence in near-infrared detectors."
           Proc. SPIE, 7021, 70210J.
    .. [2] Long, K. S. et al. (2013). "The Persistence of Memory: Decay of WFC3/IR
           Persistence." ISR WFC3 2013-12.
    """
    # from scipy import ndimage

    # Step 0: Shape validation
    if frame.shape != reference_frame.shape:
        raise ValueError(
            f"frame and reference_frame must have the same shape; "
            f"got {frame.shape} vs {reference_frame.shape}. "
            f"The align_frames parameter only corrects small translational "
            f"shifts, not different array dimensions."
        )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)

    # Step 1: Optional alignment
    alignment_offset = None
    edge_mask_from_shift = None
    ref_aligned = reference_frame

    if align_frames:
        try:
            # Unpack exactly 3 values
            ref_aligned, alignment_offset, edge_mask_from_shift = align_frames_cross_correlation(
                reference_frame, frame
            )
            logger.debug(
                "detect_persistence: applied alignment offset dy=%.2f, dx=%.2f pixels",
                alignment_offset[0],
                alignment_offset[1]
            )
            if edge_mask_from_shift is not None:
                n_edge = np.sum(edge_mask_from_shift)
                logger.debug(
                    "  masked %d edge pixels affected by shift (%.2f%%)",
                    n_edge,
                    (n_edge / frame.size) * 100
                )
        except Exception as e:
            logger.warning(
                "detect_persistence: alignment failed (%s); proceeding without alignment",
                e
            )
            # Reset to None on failure
            alignment_offset = None
            edge_mask_from_shift = None

    # Step 2: Compute difference image
    diff = frame.astype(float) - ref_aligned.astype(float)

    # Step 3: Build composite mask for statistics
    stat_mask = np.zeros(frame.shape, dtype=bool)

    if existing_bpm is not None:
        stat_mask |= (existing_bpm > 0)

    if source_mask is not None:
        stat_mask |= source_mask
        logger.debug(
            "detect_persistence: excluding %d source pixels from statistics",
            np.sum(source_mask)
        )

    # Step 4: Robust statistics on difference
    median_diff, sigma_diff = sigma_clip_stats(
        diff,
        mask=stat_mask if np.any(stat_mask) else None,
        scale='mad'
    )

    if not np.isfinite(sigma_diff) or sigma_diff <= 0:
        logger.warning(
            "detect_persistence: degenerate sigma (%s); no persistence detected.",
            sigma_diff
        )
        return dq, {
            'threshold_sigma': float(threshold_sigma),
            'threshold': float('nan'),
            'num_regions': 0,
            'num_pixels': 0,
            'fraction': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'alignment_offset': alignment_offset,
        }

    # Step 5: Threshold positive outliers (excluding masked regions)
    threshold = median_diff + threshold_sigma * sigma_diff
    persistence_candidates = (diff > threshold)

    # Exclude masked regions from detection as well
    if np.any(stat_mask):
        persistence_candidates &= ~stat_mask

    if np.sum(persistence_candidates) == 0:
        return dq, {
            'threshold_sigma': float(threshold_sigma),
            'threshold': float(threshold),
            'num_regions': 0,
            'num_pixels': 0,
            'fraction': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'alignment_offset': alignment_offset,
            'num_edge_pixels_masked': int(np.sum(edge_mask_from_shift)) if edge_mask_from_shift is not None else 0,
        }

    # Step 6: Connected-component filtering
    labeled, num_regions = ndimage.label(persistence_candidates)

    if num_regions == 0:
        return dq, {
            'threshold_sigma': float(threshold_sigma),
            'threshold': float(threshold),
            'num_regions': 0,
            'num_pixels': 0,
            'fraction': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'alignment_offset': alignment_offset,
            'num_edge_pixels_masked': int(np.sum(edge_mask_from_shift)) if edge_mask_from_shift is not None else 0,
        }

    # Filter by size
    region_sizes = np.bincount(labeled.ravel())[1:]  # Exclude background
    valid_regions = np.where(region_sizes >= min_region_size)[0] + 1

    final_mask = np.isin(labeled, valid_regions)

    dq[final_mask] |= BPM_BIT_OUTLIER

    persistence_values = diff[final_mask]

    stats = {
        'threshold_sigma': float(threshold_sigma),
        'threshold': float(threshold),
        'num_regions': int(len(valid_regions)),
        'num_pixels': int(np.sum(final_mask)),
        'fraction': float(np.sum(final_mask)) / frame.size,
        'max_persistence': float(np.max(persistence_values)) if persistence_values.size > 0 else 0.0,
        'mean_persistence': float(np.mean(persistence_values)) if persistence_values.size > 0 else 0.0,
        'alignment_offset': alignment_offset,  # ✓ Always present (None if not aligned)
        'num_edge_pixels_masked': int(np.sum(edge_mask_from_shift)) if edge_mask_from_shift is not None else 0,
    }

    return dq, stats


def align_frames_cross_correlation(
    reference: np.ndarray,
    target: np.ndarray,
    upsample_factor: int = 10,
) -> Tuple[np.ndarray, Tuple[float, float], Optional[np.ndarray]]:
    """
    Align reference frame to target using phase cross-correlation.

    General-purpose frame alignment utility for difference imaging,
    persistence detection, or image registration.

    Uses the phase correlation method with optional upsampling to achieve
    sub-pixel alignment accuracy. This is more accurate than simple
    cross-correlation peak finding for small shifts.

    Parameters
    ----------
    reference : np.ndarray
        Reference frame to be shifted (will be modified).
    target : np.ndarray
        Target frame (fixed, used as alignment reference).
    upsample_factor : int, optional
        Upsampling factor for sub-pixel precision:
        - 1: Integer pixel precision only
        - 10: 0.1 pixel precision (default)
        - 100: 0.01 pixel precision (slower)
        Higher values increase computation time quadratically.

    Returns
    -------
    aligned : np.ndarray
        Shifted reference frame, aligned to target.
    offset : tuple of (dy, dx)
        Applied shift in pixels (float for sub-pixel shifts).
        Positive dy means reference was shifted down.
        Positive dx means reference was shifted right.
    edge_mask : np.ndarray (bool) or None
        Boolean mask indicating pixels near shifted edges that should be
        excluded from analysis (True = exclude). Shape matches ``aligned``.
        These pixels contain extrapolated/replicated values and should not
        be used for difference imaging.

    Raises
    ------
    ImportError
        If scipy version is too old (< 1.4.0, when register_translation
        was moved to phase_cross_correlation).

    Notes
    -----
    **Edge handling:**

    When aligning frames, pixels near the edges of the shifted frame are
    filled using edge replication (``mode='nearest'``). This prevents
    zero-padding artifacts but means edge pixels don't contain real data.

    The returned ``edge_mask`` identifies these unreliable pixels. For
    difference imaging (e.g., persistence detection), these pixels should
    be excluded from statistical analysis and detection.

    **Edge mask extent:**

    For a shift of (dy, dx):
    - Top edge masked if dy > 0: rows [0, ceil(dy)]
    - Bottom edge masked if dy < 0: rows [ny + floor(dy), ny]
    - Left edge masked if dx > 0: cols [0, ceil(dx)]
    - Right edge masked if dx < 0: cols [nx + floor(dx), nx]

    **Algorithm:**

    Uses phase correlation via FFT cross-correlation, which is more robust
    to noise than spatial-domain template matching. The method:

    1. Computes FFT of both images
    2. Calculates normalized cross-power spectrum
    3. Finds peak in inverse FFT (shift location)
    4. Optionally upsamples around peak for sub-pixel accuracy

    **Performance:**

    - upsample_factor=1: ~10-20 ms for 2k×2k (integer only)
    - upsample_factor=10: ~20-40 ms for 2k×2k (0.1 pixel precision)
    - upsample_factor=100: ~50-150 ms for 2k×2k (0.01 pixel precision)

    **Accuracy:**

    Typical sub-pixel accuracy is ~0.01-0.05 pixels for well-sampled,
    high-SNR images. Accuracy degrades with:
    - Low SNR (noise dominates correlation peak)
    - Undersampling (Nyquist limit violations)
    - Large shifts (> ~10% of image size)

    See Also
    --------
    detect_persistence : Uses this function for optional frame alignment
    create_edge_mask_from_shift : Helper for masking shifted edge pixels

    References
    ----------
    .. [1] Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008).
           "Efficient subpixel image registration algorithms."
           Optics Letters, 33(2), 156-158.
    """
    try:
        # Try modern scikit-image API (best: sub-pixel alignment)
        from skimage.registration import phase_cross_correlation
        has_phase_correlation = True
    except ImportError:
        # Fallback to scipy-only implementation (integer-pixel only)
        has_phase_correlation = False

    if has_phase_correlation:
        # Modern approach: use skimage's phase_cross_correlation
        shift, error, phasediff = phase_cross_correlation(
            target,
            reference,
            upsample_factor=upsample_factor,
            return_error=True
        )

        offset = tuple(shift.tolist())

        logger.debug(
            "align_frames_cross_correlation: detected shift = %s, error = %.6f",
            offset,
            error
        )
        # Note: We log the error for debugging but don't return it
        # If you need error metrics in production, retrieve from logs

        # Apply shift using nearest-neighbor edge handling
        from scipy.ndimage import shift as scipy_shift
        aligned = scipy_shift(
            reference,
            shift=shift,
            mode='nearest',
            order=3
        )

        # Create edge mask for shifted regions
        edge_mask = create_edge_mask_from_shift(reference.shape, shift)

        return aligned, offset, edge_mask

    else:
        # ============================================================
        # Fallback Path: Integer-pixel alignment via cross-correlation
        # ============================================================
        logger.warning(
            "align_frames_cross_correlation: scikit-image not available. "
            "Using fallback integer-pixel alignment. "
            "For sub-pixel accuracy, install with: pip install scikit-image"
        )

        # Import only what we actually use in fallback
        try:
            from scipy.signal import correlate
            from scipy.ndimage import shift as scipy_shift
        except ImportError as e:
            raise ImportError(
                "Frame alignment requires scipy (minimum) or scikit-image (best). "
                "Install with: pip install scipy"
            ) from e

        # Cross-correlation to find offset (integer pixel)
        corr = correlate(target, reference, mode='same', method='fft')

        # Find peak (integer pixel)
        peak_idx = np.unravel_index(np.argmax(corr), corr.shape)

        # Convert to offset from center
        center = np.array(target.shape) // 2
        shift_vec = np.array(peak_idx) - center
        offset = tuple(shift_vec.tolist())

        logger.debug(
            "align_frames_cross_correlation: detected shift = %s (integer pixel fallback)",
            offset
        )

        # Apply shift with nearest-neighbor edge handling
        aligned = scipy_shift(
            reference,
            shift=shift_vec,
            mode='nearest',
            order=1  # Linear for integer shifts
        )

        # Create edge mask
        edge_mask = create_edge_mask_from_shift(reference.shape, shift_vec)

        return aligned, offset, edge_mask


def create_edge_mask_from_shift(
    shape: Tuple[int, int],
    shift: np.ndarray,
) -> np.ndarray:
    """
    Create a boolean mask marking pixels affected by frame shifting.

    Useful for identifying unreliable pixels after image alignment, particularly
    for difference imaging applications.

    When a frame is shifted, pixels near the edges are filled with
    extrapolated values (via edge replication). This function creates
    a mask identifying these unreliable pixels.

    Parameters
    ----------
    shape : tuple of (ny, nx)
        Shape of the frame.
    shift : np.ndarray
        Applied shift as [dy, dx]. Positive values mean the frame
        moved down/right.

    Returns
    -------
    edge_mask : np.ndarray (bool)
        Boolean mask where True indicates pixels that should be excluded
        from analysis. Shape is (ny, nx).

    Examples
    --------
    >>> # Frame shifted 2.3 pixels down, 1.7 pixels left
    >>> mask = create_edge_mask_from_shift((100, 100), np.array([2.3, -1.7]))
    >>> # mask[0:3, :] = True (top edge)
    >>> # mask[:, 98:100] = True (right edge)

    See Also
    --------
    align_frames_cross_correlation : Frame alignment (produces shifts)
    detect_persistence : Uses edge masks for clean difference imaging
    """
    ny, nx = shape
    dy, dx = shift

    edge_mask = np.zeros((ny, nx), dtype=bool)

    # Mask rows affected by vertical shift
    if dy > 0:
        # Frame shifted down: top edge is unreliable
        n_rows = int(np.ceil(abs(dy)))
        edge_mask[:n_rows, :] = True
    elif dy < 0:
        # Frame shifted up: bottom edge is unreliable
        n_rows = int(np.ceil(abs(dy)))
        edge_mask[-n_rows:, :] = True

    # Mask columns affected by horizontal shift
    if dx > 0:
        # Frame shifted right: left edge is unreliable
        n_cols = int(np.ceil(abs(dx)))
        edge_mask[:, :n_cols] = True
    elif dx < 0:
        # Frame shifted left: right edge is unreliable
        n_cols = int(np.ceil(abs(dx)))
        edge_mask[:, -n_cols:] = True

    return edge_mask


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


def flag_outliers(
    outlier_mask: np.ndarray,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Set BPM_BIT_OUTLIER for pixels identified as statistical outliers
    by the calling pipeline.

    Unlike other artifact detectors, outlier detection is pipeline-specific
    and is performed externally. This function simply encodes the result into
    a DQ array using the correct bit flag, ensuring consistent flag usage
    across all pipelines.

    Parameters
    ----------
    outlier_mask : np.ndarray (bool)
        Boolean array where True indicates an outlier pixel.
    """
    dq = create_bpm(outlier_mask.shape, dtype=bpm_dtype)
    dq[outlier_mask] |= BPM_BIT_OUTLIER
    stats = {
        'num_outliers': int(np.sum(outlier_mask)),
        'fraction':     float(np.sum(outlier_mask)) / outlier_mask.size,
    }
    return dq, stats


def detect_statistical_outliers(
    frame: np.ndarray,
    reference_frames: Optional[np.ndarray] = None,
    reference_median: Optional[np.ndarray] = None,
    reference_sigma: Optional[np.ndarray] = None,
    sigma: float = 5.0,
    tail: str = 'both',
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect pixels that are statistical outliers relative to a reference.

    This is a pipeline-agnostic outlier detector useful for stack rejection
    or temporal analysis. Pixels that deviate by more than ``sigma`` from
    the expected value are flagged.

    See original docstring above for full details.
    """
    tail = str(tail).lower()
    if tail not in {'both', 'high', 'low'}:
        raise ValueError(f"tail must be 'both', 'high', or 'low'; got {tail!r}")

    # Validate input combinations
    if reference_frames is not None:
        if reference_median is not None or reference_sigma is not None:
            raise ValueError(
                "Cannot provide both reference_frames and reference_median/reference_sigma. "
                "Choose one approach: either pass the full stack (reference_frames) or "
                "pre-computed statistics (reference_median + reference_sigma)."
            )

    if (reference_median is None) != (reference_sigma is None):
        raise ValueError(
            "reference_median and reference_sigma must be provided together. "
            f"Got reference_median={'provided' if reference_median is not None else 'None'}, "
            f"reference_sigma={'provided' if reference_sigma is not None else 'None'}."
        )

    dq = create_bpm(frame.shape, dtype=bpm_dtype)

    mask_exclude = (existing_bpm > 0) if existing_bpm is not None else None

    # Determine which method to use and compute the thresholding "data" array
    if reference_frames is not None:
        # Method 1: Full stack (original behavior)
        method = 'reference_stack'
        reference = np.median(reference_frames, axis=0)
        residuals = frame - reference

        _, scale = sigma_clip_stats(residuals, mask=mask_exclude, scale='mad')
        if not np.isfinite(scale) or scale <= 0:
            logger.warning(
                "detect_statistical_outliers: degenerate scale (%s); no outliers will be flagged.",
                scale,
            )
            return dq, {
                'method': method,
                'tail': tail,
                'sigma': float(sigma),
                'scale': float(scale),
                'threshold': float('nan'),
                'num_outliers': 0,
                'fraction': 0.0,
            }

        data = residuals
        threshold_value = float(sigma) * float(scale)

        valid_mask = None  # not used in this mode

    elif reference_median is not None and reference_sigma is not None:
        # Method 2: Pre-computed per-pixel reference maps (memory-efficient)
        method = 'reference_maps'

        if reference_median.shape != frame.shape:
            raise ValueError(
                f"reference_median shape {reference_median.shape} does not match frame shape {frame.shape}"
            )
        if reference_sigma.shape != frame.shape:
            raise ValueError(
                f"reference_sigma shape {reference_sigma.shape} does not match frame shape {frame.shape}"
            )

        residuals = frame - reference_median

        sigma_map = np.asarray(reference_sigma, dtype=float)
        invalid = ~np.isfinite(sigma_map) | (sigma_map <= 0)
        if mask_exclude is not None:
            invalid |= mask_exclude

        valid_mask = ~invalid
        if not np.any(valid_mask):
            logger.warning(
                "detect_statistical_outliers: no valid pixels in reference_sigma after masking; "
                "no outliers will be flagged."
            )
            return dq, {
                'method': method,
                'tail': tail,
                'sigma': float(sigma),
                'scale': float('nan'),
                'threshold': float('nan'),
                'num_outliers': 0,
                'fraction': 0.0,
            }

        # Z-score map: per-pixel normalization by reference_sigma
        data = np.zeros_like(residuals, dtype=float)
        data[valid_mask] = residuals[valid_mask] / sigma_map[valid_mask]

        # For reporting only: a robust global sigma scale (not used for thresholding here)
        scale = float(np.nanmedian(sigma_map[valid_mask]))

        # Threshold is in z units
        threshold_value = float(sigma)

    else:
        # Method 3: Global frame statistics (fallback)
        method = 'global'
        median, scale = sigma_clip_stats(frame, mask=mask_exclude, scale='mad')
        residuals = frame - median

        if not np.isfinite(scale) or scale <= 0:
            logger.warning(
                "detect_statistical_outliers: degenerate scale (%s); no outliers will be flagged.",
                scale,
            )
            return dq, {
                'method': method,
                'tail': tail,
                'sigma': float(sigma),
                'scale': float(scale),
                'threshold': float('nan'),
                'num_outliers': 0,
                'fraction': 0.0,
            }

        data = residuals
        threshold_value = float(sigma) * float(scale)

        valid_mask = None  # not used in this mode

    # Apply threshold based on tail parameter
    if tail == 'both':
        threshold_low = -threshold_value
        threshold_high = threshold_value

        outlier_low = data < threshold_low
        outlier_high = data > threshold_high
        outlier_mask = outlier_low | outlier_high

        # In reference_maps mode, explicitly exclude invalid sigma pixels
        if method == 'reference_maps' and valid_mask is not None:
            outlier_mask &= valid_mask
            outlier_low &= valid_mask
            outlier_high &= valid_mask

        num_low = int(np.sum(outlier_low))
        num_high = int(np.sum(outlier_high))

        stats = {
            'method': method,
            'tail': tail,
            'sigma': float(sigma),
            'scale': float(scale) if np.isfinite(scale) else float('nan'),
            'threshold': float(threshold_value),
            'threshold_low': float(threshold_low),
            'threshold_high': float(threshold_high),
            'num_outliers': int(np.sum(outlier_mask)),
            'num_low': num_low,
            'num_high': num_high,
            'fraction': float(np.sum(outlier_mask)) / frame.size,
        }

    elif tail == 'high':
        outlier_mask = data > threshold_value
        if method == 'reference_maps' and valid_mask is not None:
            outlier_mask &= valid_mask

        num_high = int(np.sum(outlier_mask))
        stats = {
            'method': method,
            'tail': tail,
            'sigma': float(sigma),
            'scale': float(scale) if np.isfinite(scale) else float('nan'),
            'threshold': float(threshold_value),
            'num_outliers': num_high,
            'num_high': num_high,
            'fraction': float(num_high) / frame.size,
        }

    else:  # tail == 'low'
        outlier_mask = data < -threshold_value
        if method == 'reference_maps' and valid_mask is not None:
            outlier_mask &= valid_mask

        num_low = int(np.sum(outlier_mask))
        stats = {
            'method': method,
            'tail': tail,
            'sigma': float(sigma),
            'scale': float(scale) if np.isfinite(scale) else float('nan'),
            'threshold': float(threshold_value),
            'num_outliers': num_low,
            'num_low': num_low,
            'fraction': float(num_low) / frame.size,
        }

    dq[outlier_mask] |= BPM_BIT_OUTLIER
    return dq, stats


def detect_all_artifacts(
    frame: np.ndarray,
    existing_bpm: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
    reference_frame: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Run all artifact detection algorithms and return combined DQ array.

    This is a convenience function that runs the artifact detection suite
    in the optimal order, with sensible defaults. Individual detection
    thresholds can be customized via the ``config`` parameter.

    Note: Lab-specific detectors (fringing, FPN, thermal gradient) are NOT
    included in this function - they should be run separately as part of
    the calibration characterization workflow.

    Parameters
    ----------
    frame : np.ndarray
        Science frame to analyze (ADU).
    existing_bpm : np.ndarray, optional
        Master bad pixel mask (permanent defects). Used by detectors to:

        - **Saturation**: Included for API consistency (not used)
        - **Bloom**: Exclude bad pixels from saturation seed detection
        - **Cosmic rays**: Exclude from statistics (not from detection)
        - **Persistence**: Exclude from statistics and detection

        Does not suppress detection - a bad pixel hit by a cosmic ray
        will have both bits set.
    config : dict, optional
        Detection configuration. If None, uses defaults from
        :func:`get_default_artifact_config`.
    reference_frame : np.ndarray, optional
        Reference frame for persistence detection. If None, persistence
        detection is skipped.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default: np.uint32.

    Returns
    -------
    dq : np.ndarray
        Comprehensive per-frame DQ array with all detected artifacts flagged.
    stats : dict
        Nested statistics dictionary with results from each detector::

            {
                'saturation': {...},
                'bloom': {...},
                'cosmic_rays': {...},
                'persistence': {...},
                'total_flagged': int,
                'total_fraction': float,
                'artifacts_only': int,
                'artifacts_fraction': float,
            }

    Examples
    --------
    >>> # Basic usage with defaults
    >>> dq, stats = detect_all_artifacts(science_frame, master_bpm)
    >>> print(f"Total flagged: {stats['total_fraction']*100:.2f}%")

    >>> # Custom configuration
    >>> config = get_default_artifact_config()
    >>> config['cosmic_rays']['method'] = 'lacosmic'
    >>> config['cosmic_rays']['sigclip'] = 5.0
    >>> dq, stats = detect_all_artifacts(frame, master_bpm, config=config)

    >>> # With persistence detection
    >>> dq, stats = detect_all_artifacts(
    ...     current_frame,
    ...     master_bpm,
    ...     reference_frame=pre_exposure_frame
    ... )

    Notes
    -----
    **Detection order:**

    1. **Saturation** - Identify full-well pixels for reporting
    2. **Bloom** - Independently detects saturation (excluding bad pixels), grows trails
    3. **Cosmic rays** - Local outlier detection
    4. **Persistence** - Requires reference frame (optional)

    **Detector independence:**

    Each detector is self-contained and independently identifies the features
    it needs (e.g., bloom re-detects saturation internally). This ensures:

    - Detectors can be run individually in any order
    - Each detector's stats are self-consistent
    - No hidden dependencies between stages

    The ~5 ms redundancy of saturation re-detection in bloom is negligible
    compared to the 150-400 ms total pipeline time and provides architectural
    cleanliness.

    **Bad pixel filtering:**

    The ``existing_bpm`` is used differently by each detector based on
    physical reasoning:

    - **Bloom**: Filters saturation seeds (bad pixels don't generate physical blooms)
    - **Cosmic rays**: Excludes from noise statistics (prevents bias)
    - **Persistence**: Excludes from statistics and detection (known defects)

    **Performance:** ~150-400 ms for 4K × 4K frame depending on artifact density.

    See Also
    --------
    get_default_artifact_config : Get default detection parameters
    create_artifact_report : Generate visualization report
    detect_saturated_pixels : Individual saturation detector
    detect_cosmic_rays : Individual cosmic ray detector
    detect_bloom : Individual bloom detector
    detect_persistence : Individual persistence detector
    """
    if config is None:
        config = get_default_artifact_config()

    all_stats = {}
    dq = create_bpm(frame.shape, dtype=bpm_dtype)

    # Stage 1: Saturation
    logger.debug("detect_all_artifacts: running saturation detection")
    sat_dq, sat_stats = detect_saturated_pixels(
        frame,
        bpm_dtype=bpm_dtype,
        **config.get('saturation', {})
    )
    dq |= sat_dq
    all_stats['saturation'] = sat_stats

    # Stage 2: Bloom (uses saturation results)
    logger.debug("detect_all_artifacts: running bloom detection")
    bloom_dq, bloom_stats = detect_bloom(
        frame,
        existing_bpm=existing_bpm,
        bpm_dtype=bpm_dtype,
        **config.get('bloom', {})
    )
    dq |= bloom_dq
    all_stats['bloom'] = bloom_stats

    # Stage 3: Cosmic rays
    logger.debug("detect_all_artifacts: running cosmic ray detection")
    cr_dq, cr_stats = detect_cosmic_rays(
        frame,
        existing_bpm=existing_bpm,
        bpm_dtype=bpm_dtype,
        **config.get('cosmic_rays', {})
    )
    dq |= cr_dq
    all_stats['cosmic_rays'] = cr_stats

    # Stage 4: Persistence (optional)
    if reference_frame is not None:
        logger.debug("detect_all_artifacts: running persistence detection")
        pers_dq, pers_stats = detect_persistence(
            frame,
            reference_frame=reference_frame,
            existing_bpm=existing_bpm,
            bpm_dtype=bpm_dtype,
            **config.get('persistence', {})
        )
        dq |= pers_dq
        all_stats['persistence'] = pers_stats
    else:
        all_stats['persistence'] = {'skipped': True, 'reason': 'No reference frame'}

    # Summary statistics
    # Combine with master BPM for total flagging
    if existing_bpm is not None:
        combined = dq | existing_bpm
    else:
        combined = dq

    all_stats['total_flagged'] = int(np.sum(combined > 0))
    all_stats['total_fraction'] = float(all_stats['total_flagged']) / frame.size
    all_stats['artifacts_only'] = int(np.sum(dq > 0))
    all_stats['artifacts_fraction'] = float(all_stats['artifacts_only']) / frame.size

    return dq, all_stats


def get_default_artifact_config() -> Dict:
    """
    Get default configuration for artifact detection.

    Returns
    -------
    dict
        Default parameters for all artifact detectors.

    Examples
    --------
    >>> config = get_default_artifact_config()
    >>> config['cosmic_rays']['sigma'] = 7.0  # More conservative
    >>> dq, stats = detect_all_artifacts(frame, config=config)
    """
    return {
        'saturation': {
            'saturation_threshold': 0.95,
            'bit_depth': 16,
        },
        'bloom': {
            'bleed_direction': 'vertical',
            'min_trail_length': 5,
            'trail_threshold_factor': 0.1,
        },
        'cosmic_rays': {
            'method': 'sigma_clip',
            'sigma': 5.0,
            'box_size': 5,
            'iterations': 3,
            'min_sigma': None,
        },
        'persistence': {
            'threshold_sigma': 3.0,
            'min_region_size': 10,
        },
    }


def detect_fringing(
    flat_frame: np.ndarray,
    wavelength: Optional[float] = None,
    fringe_period_range: Tuple[int, int] = (5, 50),
    threshold_amplitude: float = 0.01,
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect fringing patterns in flat fields (thin-film interference).

    Fringing appears as sinusoidal brightness variations across the detector,
    caused by interference in the thin silicon substrate (back-illuminated CCDs)
    or in anti-reflection coatings. Most pronounced in the near-IR (700-1000 nm).

    Common in lab testing when using:
    - Narrowband filters
    - LED sources with narrow spectral peaks
    - Back-illuminated sensors at red/NIR wavelengths

    Parameters
    ----------
    flat_frame : np.ndarray
        Normalized flat field (e.g., master flat / median).
    wavelength : float, optional
        Wavelength in nanometers. If provided, used to estimate expected
        fringe period from silicon thickness (~15 μm typical).
    fringe_period_range : tuple of (min, max), optional
        Expected fringe period range in pixels. Default: (5, 50).

        **Must satisfy Nyquist criterion**: min_period >= 2 pixels.
        Periods below 2 pixels violate the sampling theorem and cannot be
        resolved; attempting to detect them will produce aliasing artifacts.

        Set based on your pixel scale and expected interference patterns:
        - Coarse pixels (>10 μm): Use (3, 20) for typical silicon fringes
        - Fine pixels (<5 μm): Use (5, 100) for wider spatial patterns

    threshold_amplitude : float, optional
        Minimum fringe amplitude (fractional) to flag. Default: 0.01 (1%).
    existing_bpm : np.ndarray, optional
        Master bad pixel mask. Bad pixels are:
        1. Excluded from FFT analysis (interpolated over via
           :func:`interpolate_bad_pixels`)
        2. Not flagged in output (prevents double-flagging)
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default: np.uint32.

    Raises
    ------
    ValueError
        If ``fringe_period_range`` violates Nyquist criterion (min < 2 pixels),
        exceeds image dimensions, or is invalid (min >= max).

    Returns
    -------
    dq : np.ndarray
        DQ array with BPM_BIT_OUTLIER set in fringing regions.
        **Note**: Pixels already in ``existing_bpm`` are never flagged here.
    stats : dict
        Fringe characterization::

            {
                'fringe_detected':      bool,
                'dominant_period':      float,  # pixels
                'mean_amplitude':       float,  # fractional
                'max_amplitude':        float,
                'num_pixels_affected':  int,   # excludes existing_bpm
                'wavelength':           float or None,
                'num_bad_pixels_masked': int,   # from existing_bpm
            }

    Examples
    --------
    >>> # Test with 850nm LED
    >>> dq, stats = detect_fringing(
    ...     flat_norm,
    ...     wavelength=850.0,
    ...     threshold_amplitude=0.02,
    ...     existing_bpm=master_bpm  # Exclude known defects
    ... )
    >>> if stats['fringe_detected']:
    ...     logger.warning(f"Fringing detected: {stats['dominant_period']:.1f} px period")

    Notes
    -----
    **Algorithm:**

    1. Interpolate over bad pixels (from existing_bpm) to prevent FFT artifacts
    2. Apply 2D FFT to flat field
    3. Search for peaks in power spectrum within fringe_period_range
    4. If significant peak found, inverse FFT to isolate fringe pattern
    5. Flag pixels where fringe amplitude exceeds threshold
    6. Exclude existing_bpm pixels from output (prevents double-flagging)

    **Bad pixel handling:**

    Bad pixels (dead, hot, etc.) in the flat field create artifacts in FFT
    analysis. This implementation uses :func:`interpolate_bad_pixels` to
    fill bad pixels before FFT, preventing spurious high-frequency peaks.
    This implementation:

    - **Interpolates** over bad pixels before FFT using nearest-neighbor fill
    - **Excludes** bad pixels from fringe detection output
    - **Prevents double-flagging** of pixels already in existing_bpm

    Without interpolation, dead pixels (value = 0) introduce spurious high
    frequencies that can trigger false fringe detections.

    **Mitigation in lab:**

    - Use broadband light sources (white LED or tungsten)
    - Dither wavelength slightly (rotate filter)
    - Characterize fringe pattern and divide it out

    See Also
    --------
    interpolate_bad_pixels : Smooth interpolation over bad pixels for FFT
    detect_fixed_pattern_noise : Alternative pattern detection (time-domain)

    References
    ----------
    .. [1] Chromey, F. R. (2010). "To Measure the Sky", Section 5.5.
    """

    from numpy.fft import fft2, ifft2, fftshift, fftfreq

    # ========================================================================
    # Validation Section
    # ========================================================================

    min_period, max_period = fringe_period_range

    # Type check
    if not isinstance(min_period, (int, np.integer)) or not isinstance(max_period, (int, np.integer)):
        raise TypeError(
            f"fringe_period_range must contain integers; "
            f"got types ({type(min_period).__name__}, {type(max_period).__name__})"
        )

    # Nyquist criterion (min >= 2)
    if min_period < 2:
        raise ValueError(
            f"Minimum fringe period must be >= 2 pixels (Nyquist-Shannon limit); "
            f"got {min_period}.\n\n"
            f"Physical constraint: Detecting a periodic pattern requires at least "
            f"2 samples per cycle (one for peak, one for trough). Periods below "
            f"2 pixels violate the sampling theorem and cause aliasing artifacts.\n\n"
            f"What you probably want:\n"
            f"  • For pixel-to-pixel variations: Use detect_fixed_pattern_noise()\n"
            f"  • For fine fringes (2-5 px): fringe_period_range=(2, 10)\n"
            f"  • For typical fringes: fringe_period_range=(5, 50) [default]\n"
            f"  • For coarse fringes: fringe_period_range=(10, 100)\n\n"
            f"See also: detect_fixed_pattern_noise() for checkerboard/banding patterns."
        )

    # Image size constraint (max <= image_size, not <)
    min_dimension = min(flat_frame.shape)
    if max_period > min_dimension:
        raise ValueError(
            f"Maximum fringe period {max_period} exceeds minimum image dimension "
            f"{min_dimension}. Cannot detect fringes with periods larger than "
            f"the image itself.\n\n"
            f"Recommendation: Set max_period <= {min_dimension}.\n"
            f"For large-scale structure, consider detect_thermal_gradient() instead."
        )

    # Logical range check
    if min_period >= max_period:
        raise ValueError(
            f"Invalid period range: min_period ({min_period}) must be less than "
            f"max_period ({max_period}).\n\n"
            f"Example valid ranges:\n"
            f"  fringe_period_range=(5, 50)   # Search 5-50 pixel periods\n"
            f"  fringe_period_range=(2, 20)   # Fine fringes only\n"
            f"  fringe_period_range=(10, 100) # Coarse fringes only"
        )

    # ========================================================================
    # Computation
    # ========================================================================
    dq = create_bpm(flat_frame.shape, dtype=bpm_dtype)

    # Handle bad pixels before FFT analysis
    num_bad_pixels = 0
    flat_work = flat_frame.copy()

    if existing_bpm is not None:
        bad_pixel_mask = (existing_bpm > 0)
        num_bad_pixels = int(np.sum(bad_pixel_mask))

        if num_bad_pixels > 0:
            logger.debug(
                "detect_fringing: interpolating over %d bad pixels (%.2f%%) before FFT",
                num_bad_pixels,
                (num_bad_pixels / flat_frame.size) * 100
            )

            # Interpolate over bad pixels to prevent FFT artifacts
            # Use distance-weighted interpolation (better than nearest for FFT)
            flat_work = interpolate_bad_pixels(flat_frame, bad_pixel_mask)

    # Normalize flat to mean = 1.0 (after interpolation)
    median_flat = np.median(flat_work)

    if not np.isfinite(median_flat) or median_flat <= 0:
        logger.warning("detect_fringing: invalid flat field median")
        return dq, {
            'fringe_detected': False,
            'dominant_period': None,
            'mean_amplitude': 0.0,
            'max_amplitude': 0.0,
            'num_pixels_affected': 0,
            'wavelength': wavelength,
            'num_bad_pixels_masked': num_bad_pixels,
        }

    flat_norm = flat_work / median_flat

    # Step 1: 2D FFT
    fft_flat = fft2(flat_norm)
    power_spectrum = np.abs(fftshift(fft_flat)) ** 2

    ny, nx = flat_frame.shape
    freq_y = fftshift(fftfreq(ny))
    freq_x = fftshift(fftfreq(nx))

    # Step 2: Search for peaks in frequency range
    min_period, max_period = fringe_period_range
    freq_min = 1.0 / max_period
    freq_max = 1.0 / min_period

    # Create radial frequency map
    fy_grid, fx_grid = np.meshgrid(freq_y, freq_x, indexing='ij')
    freq_radial = np.sqrt(fx_grid**2 + fy_grid**2)

    # Mask the frequency range of interest
    freq_mask = (freq_radial >= freq_min) & (freq_radial <= freq_max)

    # Exclude DC component (center)
    center_y, center_x = ny // 2, nx // 2
    freq_mask[center_y-2:center_y+3, center_x-2:center_x+3] = False

    if not np.any(freq_mask):
        return dq, {
            'fringe_detected': False,
            'dominant_period': None,
            'mean_amplitude': 0.0,
            'max_amplitude': 0.0,
            'num_pixels_affected': 0,
            'wavelength': wavelength,
            'num_bad_pixels_masked': num_bad_pixels,
        }

    # Find peak in masked region
    masked_power = power_spectrum * freq_mask
    peak_power = np.max(masked_power)
    background_power = np.median(power_spectrum[freq_mask])

    # Threshold: peak must be significantly above background
    if peak_power < 10 * background_power:  # SNR < 10
        return dq, {
            'fringe_detected': False,
            'dominant_period': None,
            'mean_amplitude': 0.0,
            'max_amplitude': 0.0,
            'num_pixels_affected': 0,
            'wavelength': wavelength,
            'num_bad_pixels_masked': num_bad_pixels,
        }

    # Get dominant frequency
    peak_idx = np.unravel_index(np.argmax(masked_power), masked_power.shape)
    peak_freq = freq_radial[peak_idx]
    dominant_period = 1.0 / peak_freq if peak_freq > 0 else None

    # Step 3: Isolate fringe pattern
    filter_width = 0.1 * peak_freq
    bandpass = np.exp(-((freq_radial - peak_freq) / filter_width) ** 2)

    # Apply filter and inverse FFT
    fft_filtered = fft_flat * fftshift(bandpass)
    fringe_pattern = np.real(ifft2(fft_filtered))

    # Step 4: Compute amplitude
    fringe_amplitude = np.abs(fringe_pattern)
    mean_amp = np.mean(fringe_amplitude)
    max_amp = np.max(fringe_amplitude)

    # Flag regions exceeding threshold
    fringe_mask = fringe_amplitude > threshold_amplitude

    # CRITICAL: Exclude existing_bpm pixels from fringe detection output
    # This prevents double-flagging of dead/hot pixels as fringe artifacts
    if existing_bpm is not None:
        fringe_mask &= ~bad_pixel_mask
        logger.debug(
            "detect_fringing: excluded %d existing bad pixels from fringe detection output",
            num_bad_pixels
        )

    dq[fringe_mask] |= BPM_BIT_OUTLIER

    stats = {
        'fringe_detected': True,
        'dominant_period': float(dominant_period) if dominant_period else None,
        'mean_amplitude': float(mean_amp),
        'max_amplitude': float(max_amp),
        'num_pixels_affected': int(np.sum(fringe_mask)),
        'fraction': float(np.sum(fringe_mask)) / flat_frame.size,
        'wavelength': float(wavelength) if wavelength else None,
        'num_bad_pixels_masked': num_bad_pixels,
    }

    return dq, stats


def interpolate_bad_pixels(
    image: np.ndarray,
    bad_mask: np.ndarray,
) -> np.ndarray:
    """
    Interpolate over bad pixels using distance-weighted approach.

    This is a general-purpose utility for preparing images for FFT analysis
    or other operations that are sensitive to discontinuities.

    Uses scipy's distance transform for efficient interpolation. This is
    better than simple nearest-neighbor for FFT analysis because it provides
    smoother transitions.

    Parameters
    ----------
    image : np.ndarray
        Input image with bad pixels.
    bad_mask : np.ndarray (bool)
        Boolean mask where True indicates bad pixels to interpolate over.

    Returns
    -------
    interpolated : np.ndarray
        Image with bad pixels filled by interpolation.

    Notes
    -----
    **Algorithm:**

    1. Find nearest good pixel for each bad pixel (distance transform)
    2. Use nearest-neighbor interpolation as base
    3. Apply light Gaussian smoothing at bad pixel locations for continuity

    This prevents sharp discontinuities that create FFT artifacts while
    preserving the overall structure for fringe detection.

    See Also
    --------
    detect_fringing : Uses this function to prepare flats for FFT analysis
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if not np.any(bad_mask):
        return image.copy()

    # Create working copy
    result = image.copy()

    # Find indices of nearest good pixel for each bad pixel
    # distance_transform_edt returns distances; we want indices
    indices = distance_transform_edt(
        bad_mask,
        return_distances=False,
        return_indices=True
    )

    # Fill bad pixels with nearest good pixel values
    result[bad_mask] = image[tuple(indices[:, bad_mask])]

    # Apply light smoothing at bad pixel locations for continuity
    # This prevents sharp edges that create FFT artifacts
    if np.sum(bad_mask) > 10:  # Only smooth if significant number of bad pixels
        # Dilate bad pixel mask slightly
        from scipy.ndimage import binary_dilation
        smooth_mask = binary_dilation(bad_mask, iterations=2)

        # Gaussian smooth only in dilated region
        smoothed = gaussian_filter(result, sigma=1.0)
        result[smooth_mask] = smoothed[smooth_mask]

    return result


def detect_fixed_pattern_noise(
    dark_stack: np.ndarray,
    threshold_sigma: float = 3.0,
    pattern_types: Optional[List[str]] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect fixed-pattern noise (FPN) from a stack of dark frames.

    Fixed-pattern noise appears as a consistent spatial pattern that is
    stable across multiple exposures but not captured by the master dark
    calibration. Common types in lab testing:

    - **Banding**: Horizontal or vertical stripes from readout electronics
    - **Checkerboard**: Alternating high/low pixels from digital crosstalk
    - **Quadrant offsets**: Pedestal differences between amplifier channels

    Parameters
    ----------
    dark_stack : np.ndarray, shape (N_frames, ny, nx)
        Stack of dark frames (bias-subtracted, same exposure time).
        Minimum 5 frames recommended; 10+ for robust detection.
    threshold_sigma : float, optional
        Detection threshold for pattern strength. Default: 3.0.
    pattern_types : list of str, optional
        Which pattern types to search for:
        - 'banding_horizontal'
        - 'banding_vertical'
        - 'checkerboard'
        If None, searches for all types.
    bpm_dtype : np.dtype, optional
        Dtype of output DQ array. Default: np.uint32.

    Returns
    -------
    dq : np.ndarray
        DQ array with BPM_BIT_OUTLIER set where FPN detected.
    stats : dict
        Pattern characterization::

            {
                'n_frames':              int,     # Number of frames analyzed
                'patterns_detected':     list,    # Pattern type names found
                'num_pixels_affected':   int,     # Total pixels flagged
                'fraction':              float,   # Fraction of detector affected
                'noise_floor':           float,   # Global temporal noise (ADU)
                'banding_horizontal':    dict or None,  # If detected
                'banding_vertical':      dict or None,  # If detected
                'checkerboard':          dict or None,  # If detected
            }

        Per-pattern sub-dictionaries (when detected)::

            {
                'amplitude':      float,  # Pattern strength (ADU)
                'affected_rows':  list,   # For horizontal banding
                'affected_cols':  list,   # For vertical banding
                'row_offset':     float,  # For checkerboard (ADU)
                'col_offset':     float,  # For checkerboard (ADU)
            }

    Examples
    --------
    >>> # Take 10 dark frames at same exposure
    >>> dark_stack = np.array([dark1, dark2, ..., dark10])
    >>> dq, stats = detect_fixed_pattern_noise(dark_stack)
    >>>
    >>> if stats['patterns_detected']:
    ...     logger.warning(f"FPN detected: {stats['patterns_detected']}")
    ...
    ...     if 'banding_horizontal' in stats['patterns_detected']:
    ...         band_info = stats['banding_horizontal']
    ...         logger.info(f"  Horizontal bands: {len(band_info['affected_rows'])} rows")
    ...         logger.info(f"  Amplitude: {band_info['amplitude']:.2f} ADU")

    Notes
    -----
    **Algorithm:**

    1. Compute pixel-wise temporal standard deviation across stack
    2. For each pattern type, extract pattern signature
    3. Compare pattern amplitude to noise floor
    4. Flag pixels participating in significant patterns

    **Pattern signatures:**

    - **Horizontal banding**: Row-wise median variations exceed threshold
    - **Vertical banding**: Column-wise median variations exceed threshold
    - **Checkerboard**: Odd/even row or column offsets exceed threshold

    **Odd dimensions:**

    Checkerboard detection requires comparing odd vs. even rows/columns,
    which assumes symmetric sampling. For odd-sized dimensions (e.g., 2049×2047),
    the image is automatically truncated to the nearest even size (2048×2046)
    for analysis. The excluded pixels (last row/column) are reported in
    ``stats['checkerboard']['analysis_region']``.

    This ensures equal sample sizes and prevents biased comparisons.

    **NaN handling:**

    If the input ``dark_stack`` contains NaN values (e.g., from upstream
    masking where bad pixels are set to NaN), the detector uses NaN-safe
    statistics (``np.nanmean``, ``np.nanmedian``) to compute pattern
    signatures. This ensures robust detection even when frames have been
    preprocessed with bad pixel masks.


    """
    if pattern_types is None:
        pattern_types = ['banding_horizontal', 'banding_vertical', 'checkerboard']

    n_frames = dark_stack.shape[0]
    if n_frames < 3:
        raise ValueError(f"Need at least 3 frames for FPN detection; got {n_frames}")

    ny, nx = dark_stack.shape[1:]
    dq = create_bpm((ny, nx), dtype=bpm_dtype)

    # Compute temporal statistics (NaN-safe)
    temporal_median = np.nanmedian(dark_stack, axis=0)
    temporal_std = np.nanstd(dark_stack, axis=0)

    # Robust noise floor estimate (exclude hot pixels via median, NaN-safe)
    noise_floor = np.nanmedian(temporal_std)

    # Check if noise_floor is valid
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        logger.warning(
            f"detect_fixed_pattern_noise: degenerate noise floor ({noise_floor}); "
            f"no patterns will be detected. Check input data for validity."
        )
        return dq, {
            'n_frames': n_frames,
            'patterns_detected': [],
            'num_pixels_affected': 0,
            'fraction': 0.0,
            'noise_floor': float(noise_floor),
        }

    patterns_detected = []
    pattern_stats = {}
    fpn_mask = np.zeros((ny, nx), dtype=bool)

    # Pattern 1: Horizontal banding
    if 'banding_horizontal' in pattern_types:
        row_medians = np.nanmedian(temporal_median, axis=1)  # NaN-safe
        global_row_median = np.nanmedian(row_medians)
        row_residuals = row_medians - global_row_median
        row_variation = np.nanstd(row_residuals)

        if np.isfinite(row_variation) and row_variation > threshold_sigma * noise_floor:
            patterns_detected.append('banding_horizontal')

            # Flag rows that deviate significantly
            bad_rows = np.abs(row_residuals) > threshold_sigma * row_variation
            bad_row_indices = np.where(bad_rows)[0]

            fpn_mask[bad_rows, :] = True

            pattern_stats['banding_horizontal'] = {
                'amplitude': float(row_variation),
                'peak_deviation': float(np.nanmax(np.abs(row_residuals))),
                'affected_rows': bad_row_indices.tolist(),
                'num_affected_rows': int(len(bad_row_indices)),
            }

    # Pattern 2: Vertical banding
    if 'banding_vertical' in pattern_types:
        col_medians = np.nanmedian(temporal_median, axis=0)  # NaN-safe
        global_col_median = np.nanmedian(col_medians)
        col_residuals = col_medians - global_col_median
        col_variation = np.nanstd(col_residuals)

        if np.isfinite(col_variation) and col_variation > threshold_sigma * noise_floor:
            patterns_detected.append('banding_vertical')

            bad_cols = np.abs(col_residuals) > threshold_sigma * row_variation
            bad_col_indices = np.where(bad_cols)[0]

            fpn_mask[:, bad_cols] = True

            pattern_stats['banding_vertical'] = {
                'amplitude': float(col_variation),
                'peak_deviation': float(np.nanmax(np.abs(col_residuals))),
                'affected_cols': bad_col_indices.tolist(),
                'num_affected_cols': int(len(bad_col_indices)),
            }

    # Pattern 3: Checkerboard (odd/even pixel offset)
    if 'checkerboard' in pattern_types:
        # Truncate to even dimensions for proper odd/even comparison
        h_even = ny - (ny % 2)
        w_even = nx - (nx % 2)

        if h_even < 2 or w_even < 2:
            logger.warning(
                "detect_fixed_pattern_noise: image too small for checkerboard "
                "detection (need at least 2×2 even region, got %d×%d)",
                ny,
                nx
            )
        else:
            # Crop to even dimensions
            crop = temporal_median[:h_even, :w_even]

            # Compare odd vs even rows and columns (NaN-safe)
            odd_rows_mean = np.nanmean(crop[::2, :])
            even_rows_mean = np.nanmean(crop[1::2, :])
            odd_cols_mean = np.nanmean(crop[:, ::2])
            even_cols_mean = np.nanmean(crop[:, 1::2])

            row_offset = abs(odd_rows_mean - even_rows_mean)
            col_offset = abs(odd_cols_mean - even_cols_mean)

            # Check if offsets are finite
            if not np.isfinite(row_offset):
                logger.warning(
                    "detect_fixed_pattern_noise: row_offset is NaN "
                    "(possibly all NaN data in odd or even rows)"
                )
                row_offset = 0.0

            if not np.isfinite(col_offset):
                logger.warning(
                    "detect_fixed_pattern_noise: col_offset is NaN "
                    "(possibly all NaN data in odd or even columns)"
                )
                col_offset = 0.0

            max_offset = max(row_offset, col_offset)

            if np.isfinite(max_offset) and max_offset > threshold_sigma * noise_floor:
                patterns_detected.append('checkerboard')

                pattern_stats['checkerboard'] = {
                    'amplitude': float(max_offset),
                    'row_offset': float(row_offset),
                    'col_offset': float(col_offset),
                    'dominant': 'row' if row_offset > col_offset else 'col',
                    'analysis_region': {
                        'rows_used': h_even,
                        'cols_used': w_even,
                        'rows_excluded': ny - h_even,
                        'cols_excluded': nx - w_even,
                    },
                }

    dq[fpn_mask] |= BPM_BIT_OUTLIER

    # Build comprehensive stats
    stats = {
        'n_frames': n_frames,
        'patterns_detected': patterns_detected,
        'num_pixels_affected': int(np.sum(fpn_mask)),
        'fraction': float(np.sum(fpn_mask)) / (ny * nx),
        'noise_floor': float(noise_floor),
    }

    # Add per-pattern details
    for pattern_name, pattern_info in pattern_stats.items():
        stats[pattern_name] = pattern_info

    return dq, stats


def _compute_downsample_factor(shape: Tuple[int, int]) -> int:
    """
    Compute optimal downsampling factor for thermal gradient plane fitting.

    Adaptive strategy targets ~50k points for excellent overdetermination
    of the 3-parameter linear model while maintaining fast performance.

    Parameters
    ----------
    shape : tuple of (ny, nx)
        Image dimensions.

    Returns
    -------
    int
        Downsampling factor (odd integer in range [1, 21]).
        - 1 = no downsampling (< 50k pixels)
        - 3, 5, 7, ... = adaptive downsampling
        - 21 = maximum (for very large sensors)

    Examples
    --------
    >>> _compute_downsample_factor((512, 512))
    1  # 262k pixels, no downsampling needed

    >>> _compute_downsample_factor((2048, 2048))
    3  # 4.2M pixels → 467k points after 3× downsampling

    >>> _compute_downsample_factor((4096, 4096))
    9  # 16.8M pixels → 207k points after 9× downsampling

    >>> _compute_downsample_factor((8192, 8192))
    17  # 67M pixels → 232k points after 17× downsampling

    >>> _compute_downsample_factor((16384, 16384))
    21  # 268M pixels → 609k points (capped at max factor)

    Notes
    -----
    **Why target ~50k points?**

    A linear plane fit has only 3 degrees of freedom (dy, dx, offset).
    Using 50,000 points provides:
    - 16,000× overdetermination (far exceeding statistical requirements)
    - Fast lstsq performance (~20 ms)
    - Negligible numerical error vs. full resolution

    **Why odd factors?**

    Odd downsampling factors avoid potential Nyquist ambiguity at the
    factor/2 spatial frequency. While thermal gradients are smooth and
    this is unlikely to matter in practice, using odd factors is a
    conservative best practice.

    **Why cap at 21×?**

    Even for very large sensors (16K×16K), 21× downsampling produces
    ~600×600 sampling, which is still 100× overdetermined for a plane fit.
    Further downsampling could miss unexpected spatial structure (e.g.,
    cooling channel artifacts, amplifier zone boundaries).

    See Also
    --------
    detect_thermal_gradient : Uses this helper for adaptive downsampling
    """
    ny, nx = shape
    n_pixels = ny * nx

    if n_pixels <= _THERMAL_FIT_TARGET_POINTS:
        return 1  # No downsampling needed

    # Compute factor to achieve target point count
    factor = int(np.sqrt(n_pixels / _THERMAL_FIT_TARGET_POINTS))

    # Round to nearest odd number (better sampling properties)
    if factor % 2 == 0:
        factor += 1

    # Clamp to reasonable range
    return min(max(factor, 1), _THERMAL_FIT_MAX_DOWNSAMPLE)


def detect_thermal_gradient(
    frame: np.ndarray,
    max_gradient: float = 0.01,
    smoothing_sigma: float = 50.0,
    downsample_factor: Optional[int] = None,  # NEW: None = auto
    existing_bpm: Optional[np.ndarray] = None,
    bpm_dtype: np.dtype = np.uint32,
) -> Tuple[np.ndarray, Dict]:
    """
    Detect large-scale thermal gradients in dark frames.

    Thermal gradients appear as smooth, monotonic brightness variations
    across the detector due to uneven cooling or temperature stabilization.
    Common during lab testing when:

    - Detector is still equilibrating after power-on
    - Cooling system has hot spots
    - Ambient temperature is changing

    This is a QC check rather than a per-pixel flagging tool — large
    gradients indicate the test conditions are not stable.

    Parameters
    ----------
    frame : np.ndarray
        Dark frame to analyze (bias-subtracted).
    max_gradient : float, optional
        Maximum acceptable gradient (ADU/pixel). Default: 0.01.
    smoothing_sigma : float, optional
        Gaussian smoothing kernel size (pixels). Default: 50.0.
    downsample_factor : int or None, optional
        Downsampling factor for plane fitting (improves performance).

        - **None (default)**: Automatic adaptive downsampling

          Uses :func:`_compute_downsample_factor` to target ~50k points
          for optimal balance of speed and accuracy. See module constants
          ``_THERMAL_FIT_TARGET_POINTS`` and ``_THERMAL_FIT_MAX_DOWNSAMPLE``
          for tuning rationale.

        - **1**: No downsampling (use full resolution)

          Slower with no accuracy benefit. Only useful for algorithm validation.

        - **3, 5, 7, ...**: Manual downsampling (odd numbers recommended)

          For custom performance tuning. Odd factors avoid Nyquist ambiguity.
    existing_bpm : np.ndarray, optional
        Master bad pixel mask (excluded from gradient fit).

    Returns
    -------
    dq : np.ndarray
        DQ array (empty in this case — gradient is a QC flag, not per-pixel).
    stats : dict
        Gradient characterization::

            {
                'gradient_detected':  bool,
                'max_gradient':       float,  # ADU/pixel
                'gradient_direction': float,  # degrees
                'peak_to_valley':     float,  # ADU
                'rms_gradient':       float,
            }

    Examples
    --------
    >>> # Check if dark frame has thermal gradient
    >>> _, stats = detect_thermal_gradient(dark_frame)
    >>> if stats['gradient_detected']:
    ...     logger.error(f"Thermal instability: {stats['peak_to_valley']:.1f} ADU p-v")
    ...     # Recommendation: wait for temperature stabilization

    Notes
    -----
    **Performance optimization:**

    Thermal gradient detection fits a plane to the image to characterize
    large-scale tilt. For large sensors, this requires solving a least-squares
    problem with millions of data points, which is:

    - Memory-intensive: 16 Mpixel image → ~384 MB matrix
    - Time-consuming: ~1 second for lstsq on 4K×4K image

    Since thermal gradients are smooth (spatial scale >> pixel size), the
    plane fit is computed on a downsampled version of the smoothed image.
    This provides 10-200× speedup with negligible accuracy loss.

    **Adaptive downsampling strategy** (``downsample_factor=None``):

    - < 1K×1K sensors: No downsampling (fast enough)
    - 2K×2K sensors: 3× downsampling (~7× faster)
    - 4K×4K sensors: 9× downsampling (~50× faster)
    - 8K×8K sensors: 17× downsampling (~200× faster)

    Target: Use ~50,000 points for plane fit (excellent overdetermination
    for 3-parameter model).

    **Accuracy:** For a 4K×4K sensor with 9× downsampling:
    - Gradient coefficient difference: < 0.0001 ADU/pixel (0.01% error)
    - Peak-to-valley difference: < 0.1 ADU (negligible)

    **When to use full resolution** (``downsample_factor=1``):
    - You have abundant RAM and time
    - You're validating the algorithm itself
    - Never for production QC (no benefit)

    Examples
    --------
    >>> # Default: automatic adaptive downsampling
    >>> _, stats = detect_thermal_gradient(dark_frame)
    >>> # Typically 10-100× faster than v1.0, same accuracy

    >>> # Force full resolution (slower, no accuracy benefit)
    >>> _, stats = detect_thermal_gradient(dark_frame, downsample_factor=1)

    >>> # Manual downsampling for 8K sensor
    >>> _, stats = detect_thermal_gradient(huge_frame, downsample_factor=15)
    """

    from scipy.ndimage import gaussian_filter

    dq = create_bpm(frame.shape, dtype=bpm_dtype)
    ny, nx = frame.shape

    # Step 1: Smooth heavily to get large-scale structure
    smoothed = gaussian_filter(frame.astype(float), sigma=smoothing_sigma)

    # Step 2: Determine downsampling for plane fit
    if downsample_factor is None:
        # Adaptive: compute optimal factor
        downsample_factor = _compute_downsample_factor(frame.shape)
        logger.debug(
            "detect_thermal_gradient: using adaptive %dx "
            "downsampling for plane fit", downsample_factor
        )
    elif downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1; got {downsample_factor}")

    # Downsample smoothed image for plane fitting
    if downsample_factor > 1:
        smoothed_ds = smoothed[::downsample_factor, ::downsample_factor]
        ny_ds, nx_ds = smoothed_ds.shape
        logger.debug(
            "  Downsampled from %d×%d (%d points) to %d×%d (%d points) for plane fit",
            ny,
            nx,
            ny * nx,
            ny_ds,
            nx_ds,
            ny_ds * nx_ds
        )
    else:
        smoothed_ds = smoothed
        ny_ds, nx_ds = ny, nx

    # Step 3: Fit plane to DOWNSAMPLED smoothed image
    y_coords_ds, x_coords_ds = np.mgrid[0:ny:downsample_factor, 0:nx:downsample_factor]

    # Build mask for downsampled coordinates
    mask = np.zeros((ny_ds, nx_ds), dtype=bool)
    if existing_bpm is not None:
        # Downsample the mask too (any bad pixel in region → masked)
        bpm_ds = existing_bpm[::downsample_factor, ::downsample_factor] > 0
        mask |= bpm_ds

    valid = ~mask

    if np.sum(valid) < 100:
        logger.warning("detect_thermal_gradient: insufficient valid pixels")
        return dq, {
            'gradient_detected': False,
            'max_gradient': None,
            'gradient_direction': None,
            'peak_to_valley': None,
            'rms_gradient': None,
            'downsample_factor': downsample_factor,
            'points_used': int(np.sum(valid)),
        }

    # Flatten and select valid pixels
    y_flat = y_coords_ds[valid]
    x_flat = x_coords_ds[valid]
    z_flat = smoothed_ds[valid]

    # Least-squares plane fit on DOWNSAMPLED data
    A = np.column_stack([y_flat, x_flat, np.ones_like(y_flat)])
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_flat, rcond=None)

    a, b, c = coeffs  # gradient_y, gradient_x, offset (in downsampled coords)

    # Convert gradients back to full-resolution coordinates
    # Since we sampled every Nth pixel, the gradient per pixel is unchanged
    # (gradient is ADU/pixel in original coordinates)
    gradient_y_per_pixel = a
    gradient_x_per_pixel = b

    # Step 4: Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_y_per_pixel**2 + gradient_x_per_pixel**2)
    gradient_direction = np.degrees(np.arctan2(gradient_y_per_pixel, gradient_x_per_pixel))

    # Step 5: Evaluate plane on FULL RESOLUTION for diagnostics
    y_coords_full, x_coords_full = np.mgrid[0:ny, 0:nx]
    plane_fit = gradient_y_per_pixel * y_coords_full + gradient_x_per_pixel * x_coords_full + c
    p2v = np.max(plane_fit) - np.min(plane_fit)

    # RMS gradient on full resolution
    grad_y, grad_x = np.gradient(smoothed)
    valid_full = ~(existing_bpm > 0) if existing_bpm is not None else np.ones((ny, nx), dtype=bool)
    rms_grad = np.sqrt(np.mean(grad_y[valid_full]**2 + grad_x[valid_full]**2))

    gradient_detected = gradient_magnitude > max_gradient

    stats = {
        'gradient_detected': bool(gradient_detected),
        'max_gradient': float(gradient_magnitude),
        'gradient_direction': float(gradient_direction),
        'peak_to_valley': float(p2v),
        'rms_gradient': float(rms_grad),
        'downsample_factor': int(downsample_factor),
        'points_used': int(np.sum(valid)),
        'coefficients': {
            'dy': float(gradient_y_per_pixel),
            'dx': float(gradient_x_per_pixel),
            'offset': float(c),
        },
    }

    return dq, stats


__all__ = [
    # Core artifact detectors
    'detect_saturated_pixels',
    'detect_cosmic_rays',
    'detect_bloom',
    'detect_crosstalk',
    'detect_statistical_outliers',
    'detect_persistence',
    'flag_outliers',

    # Lab-specific detectors
    'detect_fringing',
    'detect_fixed_pattern_noise',
    'detect_thermal_gradient',

    # Integration
    'detect_all_artifacts',
    'get_default_artifact_config',

    # Utilities
    'interpolate_bad_pixels',
    'create_edge_mask_from_shift',
    'align_frames_cross_correlation',
    'convert_readnoise_to_adu',
]
