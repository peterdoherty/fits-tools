#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Per-frame artifact detection for CCD/CMOS images.

Artifacts differ from defects in a fundamental way: they are properties of a
specific exposure, not the sensor itself.  A pixel is not "a cosmic ray pixel"
— it was hit by a cosmic ray in one frame and is perfectly good in all others.

Detection functions in this subpackage write flags into a per-frame Data
Quality (DQ) array using the same bitmask scheme as fits_tools.defects, but
the resulting array belongs to the individual science frame, not the master
bad pixel map.

Bit flags used here (from fits_tools.bitmask):
    BPM_BIT_SATURATED   Pixel at or above saturation level in this frame
    BPM_BIT_COSMIC      Cosmic ray hit in this frame
    BPM_BIT_BLOOM       Blooming / bleed trail in this frame
    BPM_BIT_CROSSTALK   Electronic crosstalk artifact in this frame
    BPM_BIT_OUTLIER     Statistical outlier flagged by the calling pipeline

Typical usage
-------------
    >>> from fits_tools.artifacts.detection import detect_cosmic_rays, detect_saturated_pixels
    >>> cr_dq,  cr_stats  = detect_cosmic_rays(science_frame, existing_bpm=master_bpm)
    >>> sat_dq, sat_stats = detect_saturated_pixels(science_frame)
    >>> frame_dq = cr_dq | sat_dq  # combine into single per-frame DQ array
"""

from .detection import (
    detect_bloom,
    detect_cosmic_rays,
    detect_crosstalk,  # Unimplemented placeholder; warns and returns an empty DQ.
    detect_saturated_pixels,
    flag_outliers,
)

__all__ = [
    "detect_saturated_pixels",
    "detect_cosmic_rays",
    "detect_bloom",
    "detect_crosstalk",
    "flag_outliers",
]
