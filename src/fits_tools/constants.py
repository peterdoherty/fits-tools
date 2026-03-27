#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package-wide constants for fits_tools.

This module provides a single, authoritative source for:

- **Keyword constants** (``FITS_KEY_*``) — string names of FITS header
  keywords used throughout the package, avoiding raw string literals in
  calling code.
- **Value constants** (``FITS_VAL_*``) — canonical values for header
  keywords such as ``IMGTYPE`` and ``XTENSION``.
- **Numeric constants** — physical limits (exposure times, temperatures,
  ADU ranges, etc.) used for validation.
- **Immutable collections** — grouped sets of keywords and values for
  iteration and membership testing.
- **Data-type mappings** — bidirectional lookup between NumPy dtype strings
  and FITS ``BITPIX`` codes.
- **:class:`RegionType` enum** — named region types for image section
  extraction (``FULL``, ``DATA``, ``BIAS``, ``TRIM``, ``DARK``).
- **Helper functions** — lightweight utilities for image-type classification
  and normalisation, FITS section string parsing, and filename checks.

Usage Examples
--------------
Basic keyword access::

    from fits_tools.constants import FITS_KEY_IMGTYPE, FITS_KEY_EXPTIME
    imgtype = header[FITS_KEY_IMGTYPE]
    exptime = header[FITS_KEY_EXPTIME]

Image-type classification::

    from fits_tools.constants import is_science_frame, normalize_image_type
    if is_science_frame(header['IMAGETYP']):
        process_science(header)
    canonical = normalize_image_type('ZERO')   # → 'BIAS'

Validation with alias awareness::

    from fits_tools.constants import is_valid_image_type
    is_valid_image_type('skyflat')   # → True  (alias for FLAT)
    is_valid_image_type('UNKNOWN')   # → False

Membership testing with collections::

    from fits_tools.constants import TEMPERATURE_KEYWORDS
    for kw in TEMPERATURE_KEYWORDS:
        if kw in header:
            print(f"Temperature keyword {kw}: {header[kw]}")

Version History
---------------
0.1.0
    Initial release.
0.2.0
    Added temperature, gain, readout mode, and extra image-type keywords.
0.3.0
    Added WCS keywords, extension keywords, immutable collections,
    type-normalisation, type hints, validation constants, BITPIX/BUNIT
    constants, and :class:`RegionType`.
0.4.0
    Moved ``Enum`` import to module top level (PEP 8); fixed
    ``is_valid_image_type`` to recognise aliases; general cleanup.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

__all__ = [
    # --- Primary / structure keywords ---
    "FITS_KEY_SIMPLE",
    "FITS_KEY_BITPIX",
    "FITS_KEY_NAXIS",
    "FITS_KEY_NAXIS1",
    "FITS_KEY_NAXIS2",
    "FITS_KEY_NAXIS3",
    "FITS_KEY_EXTEND",
    # --- Extension keywords ---
    "FITS_KEY_XTENSION",
    "FITS_KEY_EXTNAME",
    "FITS_KEY_EXTVER",
    "FITS_KEY_EXTLEVEL",
    "FITS_KEY_PCOUNT",
    "FITS_KEY_GCOUNT",
    "FITS_KEY_TFIELDS",
    # --- Data scaling ---
    "FITS_KEY_BZERO",
    "FITS_KEY_BSCALE",
    "FITS_KEY_BUNIT",
    "FITS_KEY_BLANK",
    "FITS_KEY_DATAMIN",
    "FITS_KEY_DATAMAX",
    # --- Special / commentary keywords ---
    "FITS_KEY_COMMENT",
    "FITS_KEY_HISTORY",
    "FITS_KEY_CONTINUE",
    # --- WCS keywords ---
    "FITS_KEY_CTYPE1",
    "FITS_KEY_CTYPE2",
    "FITS_KEY_CRVAL1",
    "FITS_KEY_CRVAL2",
    "FITS_KEY_CRPIX1",
    "FITS_KEY_CRPIX2",
    "FITS_KEY_CDELT1",
    "FITS_KEY_CDELT2",
    "FITS_KEY_CROTA1",
    "FITS_KEY_CROTA2",
    "FITS_KEY_CD1_1",
    "FITS_KEY_CD1_2",
    "FITS_KEY_CD2_1",
    "FITS_KEY_CD2_2",
    "FITS_KEY_PC1_1",
    "FITS_KEY_PC1_2",
    "FITS_KEY_PC2_1",
    "FITS_KEY_PC2_2",
    "FITS_KEY_CUNIT1",
    "FITS_KEY_CUNIT2",
    "FITS_KEY_WCSAXES",
    "FITS_KEY_RADESYS",
    "FITS_KEY_EQUINOX",
    # --- Section keywords ---
    "FITS_KEY_BIASSEC",
    "FITS_KEY_DATASEC",
    "FITS_KEY_TRIMSEC",
    "FITS_KEY_DARKSEC",
    "FITS_KEY_CCDSEC",
    "FITS_KEY_DETSEC",
    "FITS_KEY_AMPSEC",
    # --- Observation information ---
    "FITS_KEY_DATE_OBS",
    "FITS_KEY_TIME_OBS",
    "FITS_KEY_DATE_END",
    "FITS_KEY_MJD_OBS",
    "FITS_KEY_MJD_END",
    "FITS_KEY_UTC",
    "FITS_KEY_LST",
    "FITS_KEY_JD",
    # --- Exposure and timing ---
    "FITS_KEY_EXPTIME",
    "FITS_KEY_DARKTIME",
    "FITS_KEY_ELAPSED",
    # --- Image type / classification ---
    "FITS_KEY_IMGTYPE",
    "FITS_KEY_IMAGETYP",
    "FITS_KEY_OBSTYPE",
    "FITS_KEY_DATASET",
    "FITS_KEY_SUBSET",
    "FITS_KEY_SEQNUM",
    # --- Target and observing setup ---
    "FITS_KEY_OBJECT",
    "FITS_KEY_TARGET",
    "FITS_KEY_TARGNAME",
    "FITS_KEY_RA",
    "FITS_KEY_DEC",
    "FITS_KEY_EPOCH",
    # --- Telescope and instrument ---
    "FITS_KEY_TELESCOP",
    "FITS_KEY_INSTRUME",
    "FITS_KEY_DETECTOR",
    "FITS_KEY_DETTYPE",
    "FITS_KEY_CAMERA",
    "FITS_KEY_SENSOR",
    "FITS_KEY_MODEL",
    # --- Filter and wavelength ---
    "FITS_KEY_FILTER",
    "FITS_KEY_FILTER1",
    "FITS_KEY_FILTER2",
    "FITS_KEY_WAVELNTH",
    "FITS_KEY_WAVEBAND",
    "FITS_KEY_BANDWDTH",
    "FITS_KEY_BANDPASS",
    # --- Detector temperature ---
    "FITS_KEY_CCD_TEMP",
    "FITS_KEY_SET_TEMP",
    "FITS_KEY_TEMP",
    "FITS_KEY_CCDTEMP",
    "FITS_KEY_TEMPERAT",
    "FITS_KEY_COOLSTAT",
    "FITS_KEY_COOLPWR",
    # --- Detector characterisation ---
    "FITS_KEY_GAIN",
    "FITS_KEY_EGAIN",
    "FITS_KEY_RDNOISE",
    "FITS_KEY_READNOIS",
    "FITS_KEY_DARKCUR",
    "FITS_KEY_DARKCURR",
    "FITS_KEY_SATURATE",
    "FITS_KEY_MAXADU",
    "FITS_KEY_PIXWELL",
    # --- Readout configuration ---
    "FITS_KEY_READOUT",
    "FITS_KEY_READMODE",
    "FITS_KEY_GAINMODE",
    "FITS_KEY_CAMGAIN",
    "FITS_KEY_READRATE",
    "FITS_KEY_AMPNAME",
    # --- Binning ---
    "FITS_KEY_BINNING",
    "FITS_KEY_XBINNING",
    "FITS_KEY_YBINNING",
    "FITS_KEY_BINROWS",
    "FITS_KEY_BINCOLS",
    # --- CCD geometry / sections ---
    "FITS_KEY_PREROWS",
    "FITS_KEY_READROWS",
    "FITS_KEY_POSTROWS",
    "FITS_KEY_PRECOLS",
    "FITS_KEY_READCOLS",
    "FITS_KEY_POSTCOLS",
    "FITS_KEY_OVERCOLS",
    "FITS_KEY_OVERROWS",
    "FITS_KEY_SEGROWS",
    "FITS_KEY_SEGCOLS",
    "FITS_KEY_SEREXT",
    "FITS_KEY_PAIRNUM",
    # --- Software and processing ---
    "FITS_KEY_ORIGIN",
    "FITS_KEY_DATE",
    "FITS_KEY_CREATOR",
    "FITS_KEY_SWCREATE",
    "FITS_KEY_SWMODIFY",
    "FITS_KEY_SOFTWARE",
    "FITS_KEY_PROCVER",
    "FITS_KEY_PROCDATE",
    # --- Observer / program ---
    "FITS_KEY_OBSERVER",
    "FITS_KEY_PROPOSER",
    "FITS_KEY_PROGID",
    "FITS_KEY_PROPOSAL",
    # --- Checksum and verification ---
    "FITS_KEY_CHECKSUM",
    "FITS_KEY_DATASUM",
    # --- Image type values ---
    "FITS_VAL_BIAS_IMGTYPE",
    "FITS_VAL_DARK_IMGTYPE",
    "FITS_VAL_FLAT_IMGTYPE",
    "FITS_VAL_LIGHT_IMGTYPE",
    "FITS_VAL_OBJECT_IMGTYPE",
    "FITS_VAL_SCIENCE_IMGTYPE",
    "FITS_VAL_TEST_IMGTYPE",
    "FITS_VAL_SNAPSHOT_IMGTYPE",
    "FITS_VAL_ARC_IMGTYPE",
    "FITS_VAL_COMPARISON_IMGTYPE",
    # --- Dataset type values ---
    "FITS_VAL_DARK_DATASET",
    "FITS_VAL_FLAT_DATASET",
    "FITS_VAL_SBIAS_DATASET",
    "FITS_VAL_SCIENCE_DATASET",
    # --- Standard HDU / extension names ---
    "FITS_VAL_PRIMARY",
    "FITS_VAL_SCI",
    "FITS_VAL_ERR",
    "FITS_VAL_DQ",
    "FITS_VAL_WHT",
    "FITS_VAL_VAR",
    "FITS_VAL_MASK",
    # --- XTENSION values ---
    "FITS_VAL_IMAGE_XTENSION",
    "FITS_VAL_BINTABLE_XTENSION",
    "FITS_VAL_TABLE_XTENSION",
    # --- BITPIX values ---
    "FITS_BITPIX_UINT8",
    "FITS_BITPIX_INT16",
    "FITS_BITPIX_INT32",
    "FITS_BITPIX_INT64",
    "FITS_BITPIX_FLOAT32",
    "FITS_BITPIX_FLOAT64",
    # --- BUNIT values ---
    "FITS_BUNIT_ADU",
    "FITS_BUNIT_ELECTRON",
    "FITS_BUNIT_ELECTRONS",
    "FITS_BUNIT_DN",
    "FITS_BUNIT_COUNT",
    "FITS_BUNIT_COUNTS",
    "FITS_BUNIT_PHOTON",
    "FITS_BUNIT_PHOTONS",
    # --- File format ---
    "FITS_FILE_EXTENSIONS",
    "FITS_SECTION_PATTERN",
    "FITS_SECTION_REGEX",
    # --- Physical / validation limits ---
    "MIN_EXPTIME",
    "MAX_EXPTIME",
    "MIN_GAIN",
    "MAX_GAIN",
    "DEFAULT_GAIN",
    "MIN_READNOISE",
    "MAX_READNOISE",
    "DEFAULT_READNOISE",
    "MIN_CCD_TEMP",
    "MAX_CCD_TEMP",
    "MIN_ADU",
    "MAX_ADU_16BIT",
    "MAX_ADU_14BIT",
    "MAX_ADU_12BIT",
    "MIN_BINNING",
    "MAX_BINNING",
    # --- Immutable collections ---
    "ALL_IMAGE_TYPES",
    "CALIBRATION_IMAGE_TYPES",
    "SCIENCE_IMAGE_TYPES",
    "SECTION_KEYWORDS",
    "TEMPERATURE_KEYWORDS",
    "DETECTOR_CHAR_KEYWORDS",
    "WCS_KEYWORDS",
    "EXTENSION_KEYWORDS",
    "ESSENTIAL_KEYWORDS",
    "CALIBRATION_RECOMMENDED_KEYWORDS",
    "IMAGE_TYPE_KEYWORDS",
    # --- Data-type mappings ---
    "NUMPY_TO_FITS_DTYPE",
    "FITS_TO_NUMPY_DTYPE",
    # --- Enum ---
    "RegionType",
    # --- Helper functions ---
    "is_calibration_frame",
    "is_science_frame",
    "is_valid_image_type",
    "normalize_image_type",
    "get_image_type_description",
    "parse_fits_section",
    "is_fits_file",
]


# =============================================================================
# FITS Standard Keywords
# =============================================================================

# --- Primary HDU and Structure ---
FITS_KEY_SIMPLE   = "SIMPLE"
FITS_KEY_BITPIX   = "BITPIX"
FITS_KEY_NAXIS    = "NAXIS"
FITS_KEY_NAXIS1   = "NAXIS1"
FITS_KEY_NAXIS2   = "NAXIS2"
FITS_KEY_NAXIS3   = "NAXIS3"
FITS_KEY_EXTEND   = "EXTEND"

# --- Extension Keywords (Multi-Extension FITS support) ---
FITS_KEY_XTENSION = "XTENSION"
FITS_KEY_EXTNAME  = "EXTNAME"
FITS_KEY_EXTVER   = "EXTVER"
FITS_KEY_EXTLEVEL = "EXTLEVEL"
FITS_KEY_PCOUNT   = "PCOUNT"
FITS_KEY_GCOUNT   = "GCOUNT"
FITS_KEY_TFIELDS  = "TFIELDS"

# --- Data Scaling and Range ---
FITS_KEY_BZERO   = "BZERO"
FITS_KEY_BSCALE  = "BSCALE"
FITS_KEY_BUNIT   = "BUNIT"
FITS_KEY_BLANK   = "BLANK"
FITS_KEY_DATAMIN = "DATAMIN"
FITS_KEY_DATAMAX = "DATAMAX"

# --- Special / Commentary Keywords ---
FITS_KEY_COMMENT  = "COMMENT"
FITS_KEY_HISTORY  = "HISTORY"
FITS_KEY_CONTINUE = "CONTINUE"

# --- WCS Keywords (World Coordinate System, FITS WCS Paper I/II) ---
FITS_KEY_CTYPE1  = "CTYPE1"
FITS_KEY_CTYPE2  = "CTYPE2"
FITS_KEY_CRVAL1  = "CRVAL1"
FITS_KEY_CRVAL2  = "CRVAL2"
FITS_KEY_CRPIX1  = "CRPIX1"
FITS_KEY_CRPIX2  = "CRPIX2"
FITS_KEY_CDELT1  = "CDELT1"
FITS_KEY_CDELT2  = "CDELT2"
FITS_KEY_CROTA1  = "CROTA1"   # deprecated but still encountered
FITS_KEY_CROTA2  = "CROTA2"   # deprecated but still encountered
FITS_KEY_CD1_1   = "CD1_1"
FITS_KEY_CD1_2   = "CD1_2"
FITS_KEY_CD2_1   = "CD2_1"
FITS_KEY_CD2_2   = "CD2_2"
FITS_KEY_PC1_1   = "PC1_1"
FITS_KEY_PC1_2   = "PC1_2"
FITS_KEY_PC2_1   = "PC2_1"
FITS_KEY_PC2_2   = "PC2_2"
FITS_KEY_CUNIT1  = "CUNIT1"
FITS_KEY_CUNIT2  = "CUNIT2"
FITS_KEY_WCSAXES = "WCSAXES"
FITS_KEY_RADESYS = "RADESYS"
FITS_KEY_EQUINOX = "EQUINOX"

# --- FITS Section Keywords (Region Extraction) ---
# These keywords describe pixel regions using 1-based [x1:x2,y1:y2] notation.
FITS_KEY_BIASSEC = "BIASSEC"   # overscan / bias region
FITS_KEY_DATASEC = "DATASEC"   # illuminated data region
FITS_KEY_TRIMSEC = "TRIMSEC"   # trimmed data region (post-overscan removal)
FITS_KEY_DARKSEC = "DARKSEC"   # shielded / dark region
FITS_KEY_CCDSEC  = "CCDSEC"    # position of this amplifier on the CCD mosaic
FITS_KEY_DETSEC  = "DETSEC"    # position on the detector focal plane
FITS_KEY_AMPSEC  = "AMPSEC"    # amplifier region

# --- Observation Information ---
FITS_KEY_DATE_OBS = "DATE-OBS"  # ISO-8601 UTC start of observation
FITS_KEY_TIME_OBS = "TIME-OBS"  # UTC time (legacy; prefer DATE-OBS with T)
FITS_KEY_DATE_END = "DATE-END"  # ISO-8601 UTC end of observation
FITS_KEY_MJD_OBS  = "MJD-OBS"
FITS_KEY_MJD_END  = "MJD-END"
FITS_KEY_UTC      = "UTC"
FITS_KEY_LST      = "LST"
FITS_KEY_JD       = "JD"

# --- Exposure and Timing ---
FITS_KEY_EXPTIME  = "EXPTIME"   # requested exposure time (s)
FITS_KEY_DARKTIME = "DARKTIME"  # actual dark/shutter-closed time (s)
FITS_KEY_ELAPSED  = "ELAPSED"   # elapsed wall-clock time (s)

# --- Image Type and Classification ---
FITS_KEY_IMGTYPE  = "IMGTYPE"   # primary image-type keyword (fits_tools preferred)
FITS_KEY_IMAGETYP = "IMAGETYP"  # common alternative (SBIG, MaxIM, etc.)
FITS_KEY_OBSTYPE  = "OBSTYPE"   # ESO/VLT convention
FITS_KEY_DATASET  = "DATASET"   # dataset classification
FITS_KEY_SUBSET   = "SUBSET"    # filter / band for flat subsets
FITS_KEY_SEQNUM   = "SEQNUM"    # sequence number within a series

# --- Target and Observing Setup ---
FITS_KEY_OBJECT   = "OBJECT"
FITS_KEY_TARGET   = "TARGET"
FITS_KEY_TARGNAME = "TARGNAME"
FITS_KEY_RA       = "RA"
FITS_KEY_DEC      = "DEC"
FITS_KEY_EPOCH    = "EPOCH"

# --- Telescope and Instrument ---
FITS_KEY_TELESCOP = "TELESCOP"
FITS_KEY_INSTRUME = "INSTRUME"
FITS_KEY_DETECTOR = "DETECTOR"
FITS_KEY_DETTYPE  = "DETTYPE"
FITS_KEY_CAMERA   = "CAMERA"
FITS_KEY_SENSOR   = "SENSOR"
FITS_KEY_MODEL    = "MODEL"

# --- Filter and Wavelength ---
FITS_KEY_FILTER   = "FILTER"
FITS_KEY_FILTER1  = "FILTER1"
FITS_KEY_FILTER2  = "FILTER2"
FITS_KEY_WAVELNTH = "WAVELNTH"
FITS_KEY_WAVEBAND = "WAVEBAND"
FITS_KEY_BANDWDTH = "BANDWDTH"
FITS_KEY_BANDPASS = "BANDPASS"

# --- Detector Temperature ---
FITS_KEY_CCD_TEMP = "CCD-TEMP"  # actual CCD temperature (°C)
FITS_KEY_SET_TEMP = "SET-TEMP"  # requested setpoint temperature (°C)
FITS_KEY_TEMP     = "TEMP"
FITS_KEY_CCDTEMP  = "CCDTEMP"
FITS_KEY_TEMPERAT = "TEMPERAT"
FITS_KEY_COOLSTAT = "COOLSTAT"  # cooler status string
FITS_KEY_COOLPWR  = "COOLPWR"   # cooler power (%)

# --- Detector Characterisation ---
FITS_KEY_GAIN     = "GAIN"      # system gain (e⁻/ADU)
FITS_KEY_EGAIN    = "EGAIN"     # electrons per ADU (alias)
FITS_KEY_RDNOISE  = "RDNOISE"   # read noise (e⁻ RMS)
FITS_KEY_READNOIS = "READNOIS"  # read noise (alias)
FITS_KEY_DARKCUR  = "DARKCUR"   # dark current (e⁻/s/pix)
FITS_KEY_DARKCURR = "DARKCURR"  # dark current (alias)
FITS_KEY_SATURATE = "SATURATE"  # full-well saturation level (ADU)
FITS_KEY_MAXADU   = "MAXADU"    # maximum ADU value of the digitiser
FITS_KEY_PIXWELL  = "PIXWELL"   # pixel well depth (e⁻)

# --- Readout Configuration ---
FITS_KEY_READOUT  = "READOUT"   # readout mode name / label
FITS_KEY_READMODE = "READMODE"
FITS_KEY_GAINMODE = "GAINMODE"  # gain switch position
FITS_KEY_CAMGAIN  = "CAMGAIN"   # camera gain setting (integer index)
FITS_KEY_READRATE = "READRATE"  # readout speed (kHz or MHz)
FITS_KEY_AMPNAME  = "AMPNAME"   # amplifier identifier

# --- Binning ---
FITS_KEY_BINNING  = "BINNING"   # combined binning string, e.g. "2x2"
FITS_KEY_XBINNING = "XBINNING"  # column binning factor
FITS_KEY_YBINNING = "YBINNING"  # row binning factor
FITS_KEY_BINROWS  = "BINROWS"
FITS_KEY_BINCOLS  = "BINCOLS"

# --- CCD Geometry and Pixel Counts ---
FITS_KEY_PREROWS  = "PREROWS"   # pre-scan rows
FITS_KEY_READROWS = "READROWS"  # active read rows
FITS_KEY_POSTROWS = "POSTROWS"  # post-scan rows
FITS_KEY_PRECOLS  = "PRECOLS"   # pre-scan columns
FITS_KEY_READCOLS = "READCOLS"  # active read columns
FITS_KEY_POSTCOLS = "POSTCOLS"  # post-scan columns
FITS_KEY_OVERCOLS = "OVERCOLS"  # serial overscan columns
FITS_KEY_OVERROWS = "OVERROWS"  # parallel overscan rows
FITS_KEY_SEGROWS  = "SEGROWS"   # segment / channel row count
FITS_KEY_SEGCOLS  = "SEGCOLS"   # segment / channel column count
FITS_KEY_SEREXT   = "SEREXT"    # serial extension pixel count
FITS_KEY_PAIRNUM  = "PAIRNUM"   # correlated pair number (for CDS)

# --- Software and Processing Provenance ---
FITS_KEY_ORIGIN   = "ORIGIN"    # organisation / site
FITS_KEY_DATE     = "DATE"      # date the file was written
FITS_KEY_CREATOR  = "CREATOR"   # software that created the file
FITS_KEY_SWCREATE = "SWCREATE"
FITS_KEY_SWMODIFY = "SWMODIFY"
FITS_KEY_SOFTWARE = "SOFTWARE"
FITS_KEY_PROCVER  = "PROCVER"   # processing pipeline version
FITS_KEY_PROCDATE = "PROCDATE"  # date of processing

# --- Observer and Programme ---
FITS_KEY_OBSERVER = "OBSERVER"
FITS_KEY_PROPOSER = "PROPOSER"
FITS_KEY_PROGID   = "PROGID"
FITS_KEY_PROPOSAL = "PROPOSAL"

# --- Data Integrity ---
FITS_KEY_CHECKSUM = "CHECKSUM"  # FITS standard HDU checksum
FITS_KEY_DATASUM  = "DATASUM"   # FITS standard data checksum


# =============================================================================
# FITS Standard Values
# =============================================================================

# --- Canonical Image Type Values (IMGTYPE / IMAGETYP) ---
FITS_VAL_BIAS_IMGTYPE       = "BIAS"
FITS_VAL_DARK_IMGTYPE       = "DARK"
FITS_VAL_FLAT_IMGTYPE       = "FLAT"
FITS_VAL_LIGHT_IMGTYPE      = "LIGHT"
FITS_VAL_OBJECT_IMGTYPE     = "OBJECT"
FITS_VAL_SCIENCE_IMGTYPE    = "SCIENCE"
FITS_VAL_TEST_IMGTYPE       = "TEST"
FITS_VAL_SNAPSHOT_IMGTYPE   = "SNAPSHOT"
FITS_VAL_ARC_IMGTYPE        = "ARC"
FITS_VAL_COMPARISON_IMGTYPE = "COMPARISON"

# --- Dataset Type Values (DATASET keyword) ---
FITS_VAL_DARK_DATASET    = "DARK"
FITS_VAL_FLAT_DATASET    = "FLAT"
FITS_VAL_SBIAS_DATASET   = "SBIAS"
FITS_VAL_SCIENCE_DATASET = "SCIENCE"

# --- Standard HDU / Extension Names ---
FITS_VAL_PRIMARY = "PRIMARY"
FITS_VAL_SCI     = "SCI"
FITS_VAL_ERR     = "ERR"
FITS_VAL_DQ      = "DQ"
FITS_VAL_WHT     = "WHT"
FITS_VAL_VAR     = "VAR"
FITS_VAL_MASK    = "MASK"

# --- XTENSION Values ---
FITS_VAL_IMAGE_XTENSION    = "IMAGE"
FITS_VAL_BINTABLE_XTENSION = "BINTABLE"
FITS_VAL_TABLE_XTENSION    = "TABLE"

# --- BITPIX Values (FITS data-type encoding) ---
FITS_BITPIX_UINT8   =   8   # unsigned 8-bit integer (via BZERO=128)
FITS_BITPIX_INT16   =  16   # signed 16-bit integer
FITS_BITPIX_INT32   =  32   # signed 32-bit integer
FITS_BITPIX_INT64   =  64   # signed 64-bit integer
FITS_BITPIX_FLOAT32 = -32   # IEEE 754 single-precision float
FITS_BITPIX_FLOAT64 = -64   # IEEE 754 double-precision float

# --- BUNIT Values (Brightness Unit) ---
FITS_BUNIT_ADU      = "adu"
FITS_BUNIT_ELECTRON = "electron"
FITS_BUNIT_ELECTRONS = "electrons"
FITS_BUNIT_DN       = "DN"
FITS_BUNIT_COUNT    = "count"
FITS_BUNIT_COUNTS   = "counts"
FITS_BUNIT_PHOTON   = "photon"
FITS_BUNIT_PHOTONS  = "photons"


# =============================================================================
# RegionType Enum
# =============================================================================

class RegionType(Enum):
    """Named region types for image section extraction.

    These correspond to standard FITS section keywords or special behaviours
    used by ``io.get_data`` and related functions.

    Members
    -------
    FULL
        Return the entire array without any sectioning.
    DATA
        Extract the illuminated data region (``DATASEC``).
    BIAS
        Extract the overscan / bias region (``BIASSEC``).
    TRIM
        Extract the trimmed data region after overscan removal (``TRIMSEC``).
    DARK
        Extract the shielded / dark reference region (``DARKSEC``).

    Examples
    --------
    ::

        from fits_tools.constants import RegionType
        data = get_data('image.fits', region=RegionType.BIAS)
        data = get_data('image.fits', region=RegionType.DATA.value)  # or as string
    """

    FULL = "full"
    DATA = "data"
    BIAS = "bias"
    TRIM = "trim"
    DARK = "dark"


# =============================================================================
# File Format Constants
# =============================================================================

#: Recognised FITS filename extensions (lower-case), including compressed forms.
FITS_FILE_EXTENSIONS: tuple = (
    ".fits",
    ".fit",
    ".fts",
    ".fits.gz",
    ".fits.fz",
    ".fit.fz",
    ".fts.fz",
)

#: Regex pattern for a FITS section string: ``[x1:x2,y1:y2]`` (1-based).
FITS_SECTION_PATTERN: str = (
    r"\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]"
)

#: Compiled version of :data:`FITS_SECTION_PATTERN`.
FITS_SECTION_REGEX: re.Pattern = re.compile(FITS_SECTION_PATTERN)


# =============================================================================
# Physical / Validation Limits
# =============================================================================

# --- Exposure Time (seconds) ---
MIN_EXPTIME: float =     0.0       # bias frames are legitimate at 0 s
MAX_EXPTIME: float = 86400.0       # 24 hours — a practical upper bound

# --- System Gain (e⁻/ADU) ---
MIN_GAIN: float   =   0.01
MAX_GAIN: float   = 100.0
DEFAULT_GAIN: float = 1.0

# --- Read Noise (electrons RMS) ---
MIN_READNOISE: float     =    0.1
MAX_READNOISE: float     = 1000.0
DEFAULT_READNOISE: float =    5.0

# --- Detector Temperature (°C) ---
MIN_CCD_TEMP: float = -100.0
MAX_CCD_TEMP: float =   50.0

# --- ADU Limits for common digitiser widths ---
MIN_ADU: int         =     0
MAX_ADU_16BIT: int   = 65535
MAX_ADU_14BIT: int   = 16383
MAX_ADU_12BIT: int   =  4095

# --- Binning Factors ---
MIN_BINNING: int =  1
MAX_BINNING: int = 16


# =============================================================================
# Immutable Collections
# =============================================================================

#: All canonical image type values recognised by fits_tools.
ALL_IMAGE_TYPES: tuple = (
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE,
    FITS_VAL_OBJECT_IMGTYPE,
    FITS_VAL_SCIENCE_IMGTYPE,
    FITS_VAL_TEST_IMGTYPE,
    FITS_VAL_SNAPSHOT_IMGTYPE,
    FITS_VAL_ARC_IMGTYPE,
    FITS_VAL_COMPARISON_IMGTYPE,
)

#: Canonical calibration frame types.
CALIBRATION_IMAGE_TYPES: tuple = (
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
)

#: Canonical science / object frame types.
SCIENCE_IMAGE_TYPES: tuple = (
    FITS_VAL_LIGHT_IMGTYPE,
    FITS_VAL_OBJECT_IMGTYPE,
    FITS_VAL_SCIENCE_IMGTYPE,
)

#: Standard FITS section keywords.
SECTION_KEYWORDS: tuple = (
    FITS_KEY_BIASSEC,
    FITS_KEY_DATASEC,
    FITS_KEY_TRIMSEC,
    FITS_KEY_DARKSEC,
    FITS_KEY_CCDSEC,
    FITS_KEY_DETSEC,
    FITS_KEY_AMPSEC,
)

#: Keywords that may carry detector temperature information.
TEMPERATURE_KEYWORDS: tuple = (
    FITS_KEY_CCD_TEMP,
    FITS_KEY_SET_TEMP,
    FITS_KEY_TEMP,
    FITS_KEY_CCDTEMP,
    FITS_KEY_TEMPERAT,
    FITS_KEY_COOLSTAT,
    FITS_KEY_COOLPWR,
)

#: Keywords describing detector electro-optical characterisation.
DETECTOR_CHAR_KEYWORDS: tuple = (
    FITS_KEY_GAIN,
    FITS_KEY_EGAIN,
    FITS_KEY_RDNOISE,
    FITS_KEY_READNOIS,
    FITS_KEY_DARKCUR,
    FITS_KEY_DARKCURR,
    FITS_KEY_SATURATE,
    FITS_KEY_MAXADU,
)

#: Core WCS keywords (minimal set required for a valid WCS solution).
WCS_KEYWORDS: tuple = (
    FITS_KEY_CTYPE1,
    FITS_KEY_CTYPE2,
    FITS_KEY_CRVAL1,
    FITS_KEY_CRVAL2,
    FITS_KEY_CRPIX1,
    FITS_KEY_CRPIX2,
    FITS_KEY_CDELT1,
    FITS_KEY_CDELT2,
    FITS_KEY_CROTA2,
    FITS_KEY_CD1_1,
    FITS_KEY_CD1_2,
    FITS_KEY_CD2_1,
    FITS_KEY_CD2_2,
)

#: Keywords that identify and describe FITS extensions (MEF files).
EXTENSION_KEYWORDS: tuple = (
    FITS_KEY_XTENSION,
    FITS_KEY_EXTNAME,
    FITS_KEY_EXTVER,
    FITS_KEY_EXTLEVEL,
)

#: Minimum set of keywords expected in every science or calibration frame.
ESSENTIAL_KEYWORDS: tuple = (
    FITS_KEY_IMGTYPE,
    FITS_KEY_EXPTIME,
    FITS_KEY_DATE_OBS,
)

#: Recommended keywords for calibration frames to enable reliable matching.
CALIBRATION_RECOMMENDED_KEYWORDS: tuple = (
    FITS_KEY_IMGTYPE,
    FITS_KEY_EXPTIME,
    FITS_KEY_DATE_OBS,
    FITS_KEY_CCD_TEMP,
    FITS_KEY_GAIN,
    FITS_KEY_READOUT,
    FITS_KEY_XBINNING,
    FITS_KEY_YBINNING,
)

#: Alternative keywords used for image-type classification across instruments.
IMAGE_TYPE_KEYWORDS: tuple = (
    FITS_KEY_IMGTYPE,
    FITS_KEY_IMAGETYP,
    FITS_KEY_OBSTYPE,
)


# =============================================================================
# Data-Type Mappings
# =============================================================================

#: Map from NumPy dtype string to FITS BITPIX code.
NUMPY_TO_FITS_DTYPE: dict = {
    "uint8":   FITS_BITPIX_UINT8,
    "int16":   FITS_BITPIX_INT16,
    "int32":   FITS_BITPIX_INT32,
    "int64":   FITS_BITPIX_INT64,
    "float32": FITS_BITPIX_FLOAT32,
    "float64": FITS_BITPIX_FLOAT64,
}

#: Map from FITS BITPIX code to NumPy dtype string.
FITS_TO_NUMPY_DTYPE: dict = {v: k for k, v in NUMPY_TO_FITS_DTYPE.items()}





def parse_fits_section(section_str: Optional[str]) -> Optional[tuple]:
    """Parse a FITS section string into integer coordinates.

    FITS sections use 1-based pixel indexing with the format
    ``[x1:x2,y1:y2]`` where *x* is the column axis and *y* is the row axis
    (FITS convention).

    Parameters
    ----------
    section_str:
        Section string from a keyword such as ``BIASSEC`` or ``DATASEC``,
        e.g. ``'[1:512,1:1024]'``.

    Returns
    -------
    tuple of (int, int, int, int) or None
        ``(x1, x2, y1, y2)`` in 1-based FITS coordinates, or ``None`` if
        *section_str* is empty, ``None``, or does not match the expected
        format.

    Notes
    -----
    This function intentionally returns ``None`` rather than raising on bad
    input.  The caller is responsible for raising
    :class:`~fits_tools.exceptions.SectionParsingError` with full context
    if ``None`` is unexpected.

    Examples
    --------
    ::

        >>> parse_fits_section('[1:512,1:1024]')
        (1, 512, 1, 1024)
        >>> parse_fits_section('[100:200, 50:150]')
        (100, 200, 50, 150)
        >>> parse_fits_section('invalid') is None
        True
        >>> parse_fits_section(None) is None
        True
    """
    if not section_str:
        return None

    match = FITS_SECTION_REGEX.match(section_str.strip())
    if not match:
        return None

    x1, x2, y1, y2 = map(int, match.groups())
    return (x1, x2, y1, y2)


def is_fits_file(filepath: Optional[str]) -> bool:
    """Return ``True`` if *filepath* ends with a recognised FITS extension.

    The check is case-insensitive and also handles double-suffix compressed
    formats such as ``.fits.gz`` and ``.fits.fz``.

    Parameters
    ----------
    filepath:
        Filename or path string to check.  ``None`` and empty strings
        return ``False``.

    Returns
    -------
    bool

    Examples
    --------
    ::

        >>> is_fits_file('image.fits')
        True
        >>> is_fits_file('IMAGE.FIT')
        True
        >>> is_fits_file('data.fits.gz')
        True
        >>> is_fits_file('image.jpg')
        False
        >>> is_fits_file(None)
        False
    """
    if not filepath:
        return False
    lower = str(filepath).lower()
    return any(lower.endswith(ext) for ext in FITS_FILE_EXTENSIONS)

