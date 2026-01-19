#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FITS keyword and value constants.

This module defines common FITS header keywords and standardized values used
across astronomical data processing and CCD/CMOS imager testing.

The module provides:
    - Keyword constants (FITS_KEY_*) for standardized header access
    - Value constants (FITS_VAL_*) for standardized header values
    - Helper collections for validation and filtering
    - Utility functions for image type checking and normalization
    - Data type and unit constants for FITS compliance

Usage:
    Basic keyword access:
        >>> from fits_tools.constants import FITS_KEY_IMGTYPE, FITS_KEY_EXPTIME
        >>> imgtype = header[FITS_KEY_IMGTYPE]
        >>> exptime = header[FITS_KEY_EXPTIME]
    
    Image type validation:
        >>> from fits_tools.constants import is_science_frame, normalize_image_type
        >>> if is_science_frame(imgtype):
        ...     # Process science frame
        ...     pass
        >>> normalized = normalize_image_type('zero')  # Returns 'BIAS'
    
    Using collections for validation:
        >>> from fits_tools.constants import ALL_IMAGE_TYPES, TEMPERATURE_KEYWORDS
        >>> if imgtype in ALL_IMAGE_TYPES:
        ...     print("Valid image type")
        >>> for key in TEMPERATURE_KEYWORDS:
        ...     if key in header:
        ...         print(f"Temperature: {header[key]}")

Version History:
    0.1.0 - Initial release
    0.2.0 - Added missing standard keywords for temperature, gain, readout modes,
            and additional image type values for better validation support
    0.3.0 - Added WCS keywords, extension keywords, helper improvements,
            made collections immutable, added type normalization, type hints,
            validation constants, and BITPIX/BUNIT constants
"""
from typing import Optional
import re


# =============================================================================
# FITS Standard Keywords
# =============================================================================

# --- Primary HDU and Structure ---
FITS_KEY_SIMPLE = 'SIMPLE'
FITS_KEY_BITPIX = 'BITPIX'
FITS_KEY_NAXIS = 'NAXIS'
FITS_KEY_NAXIS1 = 'NAXIS1'
FITS_KEY_NAXIS2 = 'NAXIS2'
FITS_KEY_NAXIS3 = 'NAXIS3'
FITS_KEY_EXTEND = 'EXTEND'

# --- Extension Keywords (MEF support) ---
FITS_KEY_XTENSION = 'XTENSION'
FITS_KEY_EXTNAME = 'EXTNAME'
FITS_KEY_EXTVER = 'EXTVER'
FITS_KEY_EXTLEVEL = 'EXTLEVEL'
FITS_KEY_PCOUNT = 'PCOUNT'
FITS_KEY_GCOUNT = 'GCOUNT'
FITS_KEY_TFIELDS = 'TFIELDS'

# --- Data Scaling and Range ---
FITS_KEY_BZERO = 'BZERO'
FITS_KEY_BSCALE = 'BSCALE'
FITS_KEY_BUNIT = 'BUNIT'
FITS_KEY_BLANK = 'BLANK'
FITS_KEY_DATAMIN = 'DATAMIN'
FITS_KEY_DATAMAX = 'DATAMAX'

# --- Special Keywords ---
FITS_KEY_COMMENT = 'COMMENT'
FITS_KEY_HISTORY = 'HISTORY'
FITS_KEY_CONTINUE = 'CONTINUE'

# --- WCS Keywords (World Coordinate System) ---
FITS_KEY_CTYPE1 = 'CTYPE1'
FITS_KEY_CTYPE2 = 'CTYPE2'
FITS_KEY_CRVAL1 = 'CRVAL1'
FITS_KEY_CRVAL2 = 'CRVAL2'
FITS_KEY_CRPIX1 = 'CRPIX1'
FITS_KEY_CRPIX2 = 'CRPIX2'
FITS_KEY_CDELT1 = 'CDELT1'
FITS_KEY_CDELT2 = 'CDELT2'
FITS_KEY_CROTA1 = 'CROTA1'
FITS_KEY_CROTA2 = 'CROTA2'
FITS_KEY_CD1_1 = 'CD1_1'
FITS_KEY_CD1_2 = 'CD1_2'
FITS_KEY_CD2_1 = 'CD2_1'
FITS_KEY_CD2_2 = 'CD2_2'
FITS_KEY_PC1_1 = 'PC1_1'
FITS_KEY_PC1_2 = 'PC1_2'
FITS_KEY_PC2_1 = 'PC2_1'
FITS_KEY_PC2_2 = 'PC2_2'
FITS_KEY_CUNIT1 = 'CUNIT1'
FITS_KEY_CUNIT2 = 'CUNIT2'
FITS_KEY_WCSAXES = 'WCSAXES'
FITS_KEY_RADESYS = 'RADESYS'
FITS_KEY_EQUINOX = 'EQUINOX'

# --- FITS Section Keywords (Region Extraction) ---
FITS_KEY_BIASSEC = 'BIASSEC'
FITS_KEY_DATASEC = 'DATASEC'
FITS_KEY_TRIMSEC = 'TRIMSEC'
FITS_KEY_DARKSEC = 'DARKSEC'
FITS_KEY_CCDSEC = 'CCDSEC'
FITS_KEY_DETSEC = 'DETSEC'
FITS_KEY_AMPSEC = 'AMPSEC'

# --- Observation Information ---
FITS_KEY_DATE_OBS = 'DATE-OBS'
FITS_KEY_TIME_OBS = 'TIME-OBS'
FITS_KEY_DATE_END = 'DATE-END'
FITS_KEY_MJD_OBS = 'MJD-OBS'
FITS_KEY_MJD_END = 'MJD-END'
FITS_KEY_UTC = 'UTC'
FITS_KEY_LST = 'LST'
FITS_KEY_JD = 'JD'

# --- Exposure and Timing ---
FITS_KEY_EXPTIME = 'EXPTIME'
FITS_KEY_DARKTIME = 'DARKTIME'
FITS_KEY_ELAPSED = 'ELAPSED'

# --- Image Type and Classification ---
FITS_KEY_IMGTYPE = 'IMGTYPE'
FITS_KEY_IMAGETYP = 'IMAGETYP'
FITS_KEY_OBSTYPE = 'OBSTYPE'
FITS_KEY_DATASET = 'DATASET'
FITS_KEY_SUBSET = 'SUBSET'
FITS_KEY_SEQNUM = 'SEQNUM'

# --- Target and Observing Setup ---
FITS_KEY_OBJECT = 'OBJECT'
FITS_KEY_TARGET = 'TARGET'
FITS_KEY_TARGNAME = 'TARGNAME'
FITS_KEY_RA = 'RA'
FITS_KEY_DEC = 'DEC'
FITS_KEY_EPOCH = 'EPOCH'

# --- Telescope and Instrument ---
FITS_KEY_TELESCOP = 'TELESCOP'
FITS_KEY_INSTRUME = 'INSTRUME'
FITS_KEY_DETECTOR = 'DETECTOR'
FITS_KEY_DETTYPE = 'DETTYPE'
FITS_KEY_CAMERA = 'CAMERA'
FITS_KEY_SENSOR = 'SENSOR'
FITS_KEY_MODEL = 'MODEL'

# --- Filter and Wavelength ---
FITS_KEY_FILTER = 'FILTER'
FITS_KEY_FILTER1 = 'FILTER1'
FITS_KEY_FILTER2 = 'FILTER2'
FITS_KEY_WAVELNTH = 'WAVELNTH'
FITS_KEY_WAVEBAND = 'WAVEBAND'
FITS_KEY_BANDWDTH = 'BANDWDTH'
FITS_KEY_BANDPASS = 'BANDPASS'

# --- Detector Temperature ---
FITS_KEY_CCD_TEMP = 'CCD-TEMP'
FITS_KEY_SET_TEMP = 'SET-TEMP'
FITS_KEY_TEMP = 'TEMP'
FITS_KEY_CCDTEMP = 'CCDTEMP'
FITS_KEY_TEMPERAT = 'TEMPERAT'
FITS_KEY_COOLSTAT = 'COOLSTAT'
FITS_KEY_COOLPWR = 'COOLPWR'

# --- Detector Characterization ---
FITS_KEY_GAIN = 'GAIN'
FITS_KEY_EGAIN = 'EGAIN'
FITS_KEY_RDNOISE = 'RDNOISE'
FITS_KEY_READNOIS = 'READNOIS'
FITS_KEY_DARKCUR = 'DARKCUR'
FITS_KEY_DARKCURR = 'DARKCURR'
FITS_KEY_SATURATE = 'SATURATE'
FITS_KEY_MAXADU = 'MAXADU'
FITS_KEY_PIXWELL = 'PIXWELL'

# --- Readout Configuration ---
FITS_KEY_READOUT = 'READOUT'
FITS_KEY_READMODE = 'READMODE'
FITS_KEY_GAINMODE = 'GAINMODE'
FITS_KEY_CAMGAIN = 'CAMGAIN'
FITS_KEY_READRATE = 'READRATE'
FITS_KEY_AMPNAME = 'AMPNAME'

# --- Binning ---
FITS_KEY_BINNING = 'BINNING'
FITS_KEY_XBINNING = 'XBINNING'
FITS_KEY_YBINNING = 'YBINNING'
FITS_KEY_BINROWS = 'BINROWS'
FITS_KEY_BINCOLS = 'BINCOLS'

# --- CCD Geometry and Sections ---
FITS_KEY_PREROWS = 'PREROWS'
FITS_KEY_READROWS = 'READROWS'
FITS_KEY_POSTROWS = 'POSTROWS'
FITS_KEY_PRECOLS = 'PRECOLS'
FITS_KEY_READCOLS = 'READCOLS'
FITS_KEY_POSTCOLS = 'POSTCOLS'
FITS_KEY_OVERCOLS = 'OVERCOLS'
FITS_KEY_OVERROWS = 'OVERROWS'
FITS_KEY_SEGROWS = 'SEGROWS'
FITS_KEY_SEGCOLS = 'SEGCOLS'
FITS_KEY_SEREXT = 'SEREXT'
FITS_KEY_PAIRNUM = 'PAIRNUM'

# --- Software and Processing ---
FITS_KEY_ORIGIN = 'ORIGIN'
FITS_KEY_DATE = 'DATE'
FITS_KEY_CREATOR = 'CREATOR'
FITS_KEY_SWCREATE = 'SWCREATE'
FITS_KEY_SWMODIFY = 'SWMODIFY'
FITS_KEY_SOFTWARE = 'SOFTWARE'
FITS_KEY_PROCVER = 'PROCVER'
FITS_KEY_PROCDATE = 'PROCDATE'

# --- Observer and Program ---
FITS_KEY_OBSERVER = 'OBSERVER'
FITS_KEY_PROPOSER = 'PROPOSER'
FITS_KEY_PROGID = 'PROGID'
FITS_KEY_PROPOSAL = 'PROPOSAL'

# --- Checksum and Verification ---
FITS_KEY_CHECKSUM = 'CHECKSUM'
FITS_KEY_DATASUM = 'DATASUM'


# =============================================================================
# FITS Standard Values
# =============================================================================

# --- Image Type Values ---
FITS_VAL_BIAS_IMGTYPE = 'BIAS'
FITS_VAL_DARK_IMGTYPE = 'DARK'
FITS_VAL_FLAT_IMGTYPE = 'FLAT'
FITS_VAL_LIGHT_IMGTYPE = 'LIGHT'
FITS_VAL_OBJECT_IMGTYPE = 'OBJECT'
FITS_VAL_SCIENCE_IMGTYPE = 'SCIENCE'
FITS_VAL_TEST_IMGTYPE = 'TEST'
FITS_VAL_SNAPSHOT_IMGTYPE = 'SNAPSHOT'
FITS_VAL_ARC_IMGTYPE = 'ARC'
FITS_VAL_COMPARISON_IMGTYPE = 'COMPARISON'

# --- Dataset Type Values ---
FITS_VAL_DARK_DATASET = 'DARK'
FITS_VAL_FLAT_DATASET = 'FLAT'
FITS_VAL_SBIAS_DATASET = 'SBIAS'
FITS_VAL_SCIENCE_DATASET = 'SCIENCE'

# --- Special HDU Names ---
FITS_VAL_PRIMARY = 'PRIMARY'
FITS_VAL_SCI = 'SCI'
FITS_VAL_ERR = 'ERR'
FITS_VAL_DQ = 'DQ'
FITS_VAL_WHT = 'WHT'
FITS_VAL_VAR = 'VAR'
FITS_VAL_MASK = 'MASK'

# --- XTENSION Values ---
FITS_VAL_IMAGE_XTENSION = 'IMAGE'
FITS_VAL_BINTABLE_XTENSION = 'BINTABLE'
FITS_VAL_TABLE_XTENSION = 'TABLE'

# --- BITPIX Values (FITS data types) ---
FITS_BITPIX_UINT8 = 8
FITS_BITPIX_INT16 = 16
FITS_BITPIX_INT32 = 32
FITS_BITPIX_INT64 = 64
FITS_BITPIX_FLOAT32 = -32
FITS_BITPIX_FLOAT64 = -64

# --- BUNIT (Brightness Unit) Standard Values ---
FITS_BUNIT_ADU = 'adu'
FITS_BUNIT_ELECTRON = 'electron'
FITS_BUNIT_ELECTRONS = 'electrons'
FITS_BUNIT_DN = 'DN'
FITS_BUNIT_COUNT = 'count'
FITS_BUNIT_COUNTS = 'counts'
FITS_BUNIT_PHOTON = 'photon'
FITS_BUNIT_PHOTONS = 'photons'

# =============================================================================
# Region Type Enum for Data Extraction
# =============================================================================

from enum import Enum

class RegionType(Enum):
    """
    Enumeration of valid region types for data extraction from FITS files.
    
    These represent standard FITS section keywords or special behaviors:
        - FULL: Return entire array (no sectioning)
        - DATA: Extract DATASEC region
        - BIAS: Extract BIASSEC (overscan/bias) region
        - TRIM: Extract TRIMSEC (useful data after trimming overscan)
        - DARK: Extract DARKSEC (dark/shielded region)
    
    Examples:
        >>> RegionType.DATA.value
        'data'
        >>> RegionType.BIAS.value
        'bias'
    """
    FULL = 'full'
    DATA = 'data'
    BIAS = 'bias'
    TRIM = 'trim'
    DARK = 'dark'


# =============================================================================
# File and Format Constants
# =============================================================================

# FITS file extensions (including compressed formats)
FITS_FILE_EXTENSIONS = ('.fits', '.fit', '.fts', '.fit.fz', '.fts.fz', '.fits.gz', '.fits.fz')

# Section format pattern for parsing BIASSEC, DATASEC, etc.
# Matches formats like [x1:x2,y1:y2] (1-based FITS convention)
FITS_SECTION_PATTERN = r'\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]'
FITS_SECTION_REGEX = re.compile(FITS_SECTION_PATTERN)


# =============================================================================
# Validation Constants
# =============================================================================

# --- Exposure Time Limits (seconds) ---
MIN_EXPTIME = 0.0
MAX_EXPTIME = 86400.0  # 24 hours

# --- Gain Limits (e-/ADU) ---
MIN_GAIN = 0.01
MAX_GAIN = 100.0
DEFAULT_GAIN = 1.0

# --- Read Noise Limits (electrons RMS) ---
MIN_READNOISE = 0.0
MAX_READNOISE = 1000.0
DEFAULT_READNOISE = 0.0

# --- Temperature Limits (Celsius) ---
MIN_CCD_TEMP = -100.0
MAX_CCD_TEMP = 50.0

# --- ADU Limits ---
MIN_ADU = 0
MAX_ADU_16BIT = 65535
MAX_ADU_14BIT = 16383
MAX_ADU_12BIT = 4095

# --- Binning Limits ---
MIN_BINNING = 1
MAX_BINNING = 16


# =============================================================================
# Helper Collections (Immutable)
# =============================================================================

# --- Image Type Collections ---
ALL_IMAGE_TYPES = (
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

CALIBRATION_IMAGE_TYPES = (
    FITS_VAL_BIAS_IMGTYPE,
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
)

SCIENCE_IMAGE_TYPES = (
    FITS_VAL_LIGHT_IMGTYPE,
    FITS_VAL_OBJECT_IMGTYPE,
    FITS_VAL_SCIENCE_IMGTYPE,
)

# --- Keyword Collections ---
SECTION_KEYWORDS = (
    FITS_KEY_BIASSEC,
    FITS_KEY_DATASEC,
    FITS_KEY_TRIMSEC,
    FITS_KEY_DARKSEC,
    FITS_KEY_CCDSEC,
    FITS_KEY_DETSEC,
    FITS_KEY_AMPSEC,
)

TEMPERATURE_KEYWORDS = (
    FITS_KEY_CCD_TEMP,
    FITS_KEY_SET_TEMP,
    FITS_KEY_TEMP,
    FITS_KEY_CCDTEMP,
    FITS_KEY_TEMPERAT,
    FITS_KEY_COOLSTAT,
    FITS_KEY_COOLPWR,
)

DETECTOR_CHAR_KEYWORDS = (
    FITS_KEY_GAIN,
    FITS_KEY_EGAIN,
    FITS_KEY_RDNOISE,
    FITS_KEY_READNOIS,
    FITS_KEY_DARKCUR,
    FITS_KEY_DARKCURR,
    FITS_KEY_SATURATE,
    FITS_KEY_MAXADU,
)

WCS_KEYWORDS = (
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

EXTENSION_KEYWORDS = (
    FITS_KEY_XTENSION,
    FITS_KEY_EXTNAME,
    FITS_KEY_EXTVER,
    FITS_KEY_EXTLEVEL,
)

ESSENTIAL_KEYWORDS = (
    FITS_KEY_IMGTYPE,
    FITS_KEY_EXPTIME,
    FITS_KEY_DATE_OBS,
)

CALIBRATION_RECOMMENDED_KEYWORDS = (
    FITS_KEY_IMGTYPE,
    FITS_KEY_EXPTIME,
    FITS_KEY_DATE_OBS,
    FITS_KEY_CCD_TEMP,
    FITS_KEY_GAIN,
    FITS_KEY_READOUT,
    FITS_KEY_XBINNING,
    FITS_KEY_YBINNING,
)

# Image type keyword alternatives (different systems use different names)
IMAGE_TYPE_KEYWORDS = (
    FITS_KEY_IMGTYPE,
    FITS_KEY_IMAGETYP,
    FITS_KEY_OBSTYPE,
)

# --- Data Type Mappings ---
NUMPY_TO_FITS_DTYPE = {
    'uint8': FITS_BITPIX_UINT8,
    'int16': FITS_BITPIX_INT16,
    'int32': FITS_BITPIX_INT32,
    'int64': FITS_BITPIX_INT64,
    'float32': FITS_BITPIX_FLOAT32,
    'float64': FITS_BITPIX_FLOAT64,
}

FITS_TO_NUMPY_DTYPE = {
    FITS_BITPIX_UINT8: 'uint8',
    FITS_BITPIX_INT16: 'int16',
    FITS_BITPIX_INT32: 'int32',
    FITS_BITPIX_INT64: 'int64',
    FITS_BITPIX_FLOAT32: 'float32',
    FITS_BITPIX_FLOAT64: 'float64',
}


# =============================================================================
# Image Type Aliases and Normalization
# =============================================================================

# Mapping of common aliases to standard image type values
IMAGE_TYPE_ALIASES = {
    # Bias frames
    'BIAS': FITS_VAL_BIAS_IMGTYPE,
    'ZERO': FITS_VAL_BIAS_IMGTYPE,
    'BIAS FRAME': FITS_VAL_BIAS_IMGTYPE,
    
    # Dark frames
    'DARK': FITS_VAL_DARK_IMGTYPE,
    'DARK FRAME': FITS_VAL_DARK_IMGTYPE,
    
    # Flat frames
    'FLAT': FITS_VAL_FLAT_IMGTYPE,
    'FLATFIELD': FITS_VAL_FLAT_IMGTYPE,
    'FLAT FIELD': FITS_VAL_FLAT_IMGTYPE,
    'SKYFLAT': FITS_VAL_FLAT_IMGTYPE,
    'DOMEFLAT': FITS_VAL_FLAT_IMGTYPE,
    
    # Science frames
    'LIGHT': FITS_VAL_LIGHT_IMGTYPE,
    'OBJECT': FITS_VAL_OBJECT_IMGTYPE,
    'SCIENCE': FITS_VAL_SCIENCE_IMGTYPE,
    'SCI': FITS_VAL_SCIENCE_IMGTYPE,
    'TARGET': FITS_VAL_SCIENCE_IMGTYPE,
    'OBSERVATION': FITS_VAL_SCIENCE_IMGTYPE,
    
    # Other
    'TEST': FITS_VAL_TEST_IMGTYPE,
    'SNAPSHOT': FITS_VAL_SNAPSHOT_IMGTYPE,
    'SNAP': FITS_VAL_SNAPSHOT_IMGTYPE,
}


# =============================================================================
# Helper Functions
# =============================================================================

def is_calibration_frame(imgtype_value: Optional[str]) -> bool:
    """
    Check if an image type value represents a calibration frame.
    Case-insensitive comparison with whitespace trimming.
    
    Args:
        imgtype_value: Value from IMGTYPE/IMAGETYP header keyword.
    
    Returns:
        True if the frame is a calibration frame (BIAS, DARK, or FLAT).
    
    Examples:
        >>> is_calibration_frame('BIAS')
        True
        >>> is_calibration_frame('bias')
        True
        >>> is_calibration_frame('LIGHT')
        False
        >>> is_calibration_frame(None)
        False
    """
    if not imgtype_value:
        return False
    return imgtype_value.upper().strip() in CALIBRATION_IMAGE_TYPES


def is_science_frame(imgtype_value: Optional[str]) -> bool:
    """
    Check if an image type value represents a science/object frame.
    Case-insensitive comparison with whitespace trimming.
    
    Args:
        imgtype_value: Value from IMGTYPE/IMAGETYP header keyword.
    
    Returns:
        True if the frame is a science frame (LIGHT, OBJECT, or SCIENCE).
    
    Examples:
        >>> is_science_frame('OBJECT')
        True
        >>> is_science_frame('light')
        True
        >>> is_science_frame('BIAS')
        False
        >>> is_science_frame(None)
        False
    """
    if not imgtype_value:
        return False
    return imgtype_value.upper().strip() in SCIENCE_IMAGE_TYPES


def is_valid_image_type(imgtype_value: Optional[str]) -> bool:
    """
    Check if an image type value is recognized.
    Case-insensitive comparison with whitespace trimming.
    
    Args:
        imgtype_value: Value from IMGTYPE/IMAGETYP header keyword.
    
    Returns:
        True if the image type is in the list of recognized types.
    
    Examples:
        >>> is_valid_image_type('DARK')
        True
        >>> is_valid_image_type('dark')
        True
        >>> is_valid_image_type('UNKNOWN')
        False
        >>> is_valid_image_type(None)
        False
    """
    if not imgtype_value:
        return False
    return imgtype_value.upper().strip() in ALL_IMAGE_TYPES


def normalize_image_type(imgtype_value: Optional[str]) -> Optional[str]:
    """
    Normalize image type value to standard form using alias mapping.
    Case-insensitive with whitespace trimming.
    
    Args:
        imgtype_value: Raw image type value from header.
    
    Returns:
        Normalized image type value, or the original (uppercased) if not found
        in aliases. Returns None if input is None or empty.
    
    Examples:
        >>> normalize_image_type('zero')
        'BIAS'
        >>> normalize_image_type('SCI')
        'SCIENCE'
        >>> normalize_image_type('FLAT FIELD')
        'FLAT'
        >>> normalize_image_type('UNKNOWN')
        'UNKNOWN'
        >>> normalize_image_type(None)
        None
    """
    if not imgtype_value:
        return None
    
    normalized = imgtype_value.upper().strip()
    return IMAGE_TYPE_ALIASES.get(normalized, normalized)


def get_image_type_description(imgtype_value: Optional[str]) -> str:
    """
    Get a human-readable description of an image type.
    Case-insensitive with whitespace trimming.
    
    Args:
        imgtype_value: Value from IMGTYPE/IMAGETYP header keyword.
    
    Returns:
        Description of the image type.
    
    Examples:
        >>> get_image_type_description('BIAS')
        'Bias/Zero frame (no exposure, for readout bias measurement)'
        >>> get_image_type_description('bias')
        'Bias/Zero frame (no exposure, for readout bias measurement)'
        >>> get_image_type_description(None)
        'Unknown or missing image type'
        >>> get_image_type_description('CUSTOM')
        'Unknown image type: CUSTOM'
    """
    if not imgtype_value:
        return 'Unknown or missing image type'
    
    # Normalize first to handle aliases
    normalized = normalize_image_type(imgtype_value)
    
    descriptions = {
        FITS_VAL_BIAS_IMGTYPE: 'Bias/Zero frame (no exposure, for readout bias measurement)',
        FITS_VAL_DARK_IMGTYPE: 'Dark frame (no illumination, for dark current measurement)',
        FITS_VAL_FLAT_IMGTYPE: 'Flat field frame (uniform illumination, for sensitivity correction)',
        FITS_VAL_LIGHT_IMGTYPE: 'Light/Science frame (astronomical target)',
        FITS_VAL_OBJECT_IMGTYPE: 'Object/Science frame (astronomical target)',
        FITS_VAL_SCIENCE_IMGTYPE: 'Science frame (astronomical target)',
        FITS_VAL_TEST_IMGTYPE: 'Test frame (for system testing)',
        FITS_VAL_SNAPSHOT_IMGTYPE: 'Snapshot/Preview frame (quick test exposure)',
        FITS_VAL_ARC_IMGTYPE: 'Arc/Comparison frame (wavelength calibration)',
        FITS_VAL_COMPARISON_IMGTYPE: 'Comparison frame (calibration lamp)',
    }
    
    return descriptions.get(normalized, f'Unknown image type: {imgtype_value}')


def parse_fits_section(section_str: str) -> Optional[tuple]:
    """
    Parse a FITS section string (e.g., '[1:512,1:512]') into coordinates.
    
    FITS sections use 1-based indexing: [x1:x2,y1:y2]
    
    Args:
        section_str: Section string from BIASSEC, DATASEC, etc.
    
    Returns:
        Tuple of (x1, x2, y1, y2) as integers (1-based FITS convention),
        or None if parsing fails.
    
    Examples:
        >>> parse_fits_section('[1:512,1:512]')
        (1, 512, 1, 512)
        >>> parse_fits_section('[100:200,50:150]')
        (100, 200, 50, 150)
        >>> parse_fits_section('invalid')
        None
    """
    if not section_str:
        return None
    
    match = FITS_SECTION_REGEX.match(section_str.strip())
    if not match:
        return None
    
    x1, x2, y1, y2 = map(int, match.groups())
    return (x1, x2, y1, y2)


def is_fits_file(filename: str) -> bool:
    """
    Check if a filename has a recognized FITS extension.
    Case-insensitive comparison.
    
    Args:
        filename: Filename to check.
    
    Returns:
        True if the filename ends with a FITS extension.
    
    Examples:
        >>> is_fits_file('image.fits')
        True
        >>> is_fits_file('IMAGE.FITS')
        True
        >>> is_fits_file('data.fits.gz')
        True
        >>> is_fits_file('image.jpg')
        False
    """
    if not filename:
        return False
    
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in FITS_FILE_EXTENSIONS)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    
    # --- Primary Keywords ---
    'FITS_KEY_SIMPLE',
    'FITS_KEY_BITPIX',
    'FITS_KEY_NAXIS',
    'FITS_KEY_NAXIS1',
    'FITS_KEY_NAXIS2',
    'FITS_KEY_NAXIS3',
    'FITS_KEY_EXTEND',
    
    # --- Extension Keywords ---
    'FITS_KEY_XTENSION',
    'FITS_KEY_EXTNAME',
    'FITS_KEY_EXTVER',
    'FITS_KEY_EXTLEVEL',
    'FITS_KEY_PCOUNT',
    'FITS_KEY_GCOUNT',
    'FITS_KEY_TFIELDS',
    
    # --- Data Scaling ---
    'FITS_KEY_BZERO',
    'FITS_KEY_BSCALE',
    'FITS_KEY_BUNIT',
    'FITS_KEY_BLANK',
    'FITS_KEY_DATAMIN',
    'FITS_KEY_DATAMAX',
    
    # --- Special Keywords ---
    'FITS_KEY_COMMENT',
    'FITS_KEY_HISTORY',
    'FITS_KEY_CONTINUE',
    
    # --- WCS Keywords ---
    'FITS_KEY_CTYPE1',
    'FITS_KEY_CTYPE2',
    'FITS_KEY_CRVAL1',
    'FITS_KEY_CRVAL2',
    'FITS_KEY_CRPIX1',
    'FITS_KEY_CRPIX2',
    'FITS_KEY_CDELT1',
    'FITS_KEY_CDELT2',
    'FITS_KEY_CROTA1',
    'FITS_KEY_CROTA2',
    'FITS_KEY_CD1_1',
    'FITS_KEY_CD1_2',
    'FITS_KEY_CD2_1',
    'FITS_KEY_CD2_2',
    'FITS_KEY_PC1_1',
    'FITS_KEY_PC1_2',
    'FITS_KEY_PC2_1',
    'FITS_KEY_PC2_2',
    'FITS_KEY_CUNIT1',
    'FITS_KEY_CUNIT2',
    'FITS_KEY_WCSAXES',
    'FITS_KEY_RADESYS',
    'FITS_KEY_EQUINOX',
    
    # --- Section Keywords ---
    'FITS_KEY_BIASSEC',
    'FITS_KEY_DATASEC',
    'FITS_KEY_TRIMSEC',
    'FITS_KEY_DARKSEC',
    'FITS_KEY_CCDSEC',
    'FITS_KEY_DETSEC',
    'FITS_KEY_AMPSEC',
    
    # --- Observation Keywords ---
    'FITS_KEY_DATE_OBS',
    'FITS_KEY_TIME_OBS',
    'FITS_KEY_DATE_END',
    'FITS_KEY_MJD_OBS',
    'FITS_KEY_MJD_END',
    'FITS_KEY_UTC',
    'FITS_KEY_LST',
    'FITS_KEY_JD',
    
    # --- Exposure and Timing ---
    'FITS_KEY_EXPTIME',
    'FITS_KEY_DARKTIME',
    'FITS_KEY_ELAPSED',
    
    # --- Image Type ---
    'FITS_KEY_IMGTYPE',
    'FITS_KEY_IMAGETYP',
    'FITS_KEY_OBSTYPE',
    'FITS_KEY_DATASET',
    'FITS_KEY_SUBSET',
    'FITS_KEY_SEQNUM',
    
    # --- Target Keywords ---
    'FITS_KEY_OBJECT',
    'FITS_KEY_TARGET',
    'FITS_KEY_TARGNAME',
    'FITS_KEY_RA',
    'FITS_KEY_DEC',
    'FITS_KEY_EPOCH',
    
    # --- Telescope and Instrument ---
    'FITS_KEY_TELESCOP',
    'FITS_KEY_INSTRUME',
    'FITS_KEY_DETECTOR',
    'FITS_KEY_DETTYPE',
    'FITS_KEY_CAMERA',
    'FITS_KEY_SENSOR',
    'FITS_KEY_MODEL',
    
    # --- Filter and Wavelength ---
    'FITS_KEY_FILTER',
    'FITS_KEY_FILTER1',
    'FITS_KEY_FILTER2',
    'FITS_KEY_WAVELNTH',
    'FITS_KEY_WAVEBAND',
    'FITS_KEY_BANDWDTH',
    'FITS_KEY_BANDPASS',
    
    # --- Temperature Keywords ---
    'FITS_KEY_CCD_TEMP',
    'FITS_KEY_SET_TEMP',
    'FITS_KEY_TEMP',
    'FITS_KEY_CCDTEMP',
    'FITS_KEY_TEMPERAT',
    'FITS_KEY_COOLSTAT',
    'FITS_KEY_COOLPWR',
    
    # --- Detector Characterization ---
    'FITS_KEY_GAIN',
    'FITS_KEY_EGAIN',
    'FITS_KEY_RDNOISE',
    'FITS_KEY_READNOIS',
    'FITS_KEY_DARKCUR',
    'FITS_KEY_DARKCURR',
    'FITS_KEY_SATURATE',
    'FITS_KEY_MAXADU',
    'FITS_KEY_PIXWELL',
    
    # --- Readout Configuration ---
    'FITS_KEY_READOUT',
    'FITS_KEY_READMODE',
    'FITS_KEY_GAINMODE',
    'FITS_KEY_CAMGAIN',
    'FITS_KEY_READRATE',
    'FITS_KEY_AMPNAME',
    
    # --- Binning ---
    'FITS_KEY_BINNING',
    'FITS_KEY_XBINNING',
    'FITS_KEY_YBINNING',
    'FITS_KEY_BINROWS',
    'FITS_KEY_BINCOLS',
    
    # --- CCD Geometry ---
    'FITS_KEY_PREROWS',
    'FITS_KEY_READROWS',
    'FITS_KEY_POSTROWS',
    'FITS_KEY_PRECOLS',
    'FITS_KEY_READCOLS',
    'FITS_KEY_POSTCOLS',
    'FITS_KEY_OVERCOLS',
    'FITS_KEY_OVERROWS',
    'FITS_KEY_SEGROWS',
    'FITS_KEY_SEGCOLS',
    'FITS_KEY_SEREXT',
    'FITS_KEY_PAIRNUM',
    
    # --- Software and Processing ---
    'FITS_KEY_ORIGIN',
    'FITS_KEY_DATE',
    'FITS_KEY_CREATOR',
    'FITS_KEY_SWCREATE',
    'FITS_KEY_SWMODIFY',
    'FITS_KEY_SOFTWARE',
    'FITS_KEY_PROCVER',
    'FITS_KEY_PROCDATE',
    
    # --- Observer ---
    'FITS_KEY_OBSERVER',
    'FITS_KEY_PROPOSER',
    'FITS_KEY_PROGID',
    'FITS_KEY_PROPOSAL',
    
    # --- Checksum ---
    'FITS_KEY_CHECKSUM',
    'FITS_KEY_DATASUM',
    
    # --- Image Type Values ---
    'FITS_VAL_BIAS_IMGTYPE',
    'FITS_VAL_DARK_IMGTYPE',
    'FITS_VAL_FLAT_IMGTYPE',
    'FITS_VAL_LIGHT_IMGTYPE',
    'FITS_VAL_OBJECT_IMGTYPE',
    'FITS_VAL_SCIENCE_IMGTYPE',
    'FITS_VAL_TEST_IMGTYPE',
    'FITS_VAL_SNAPSHOT_IMGTYPE',
    'FITS_VAL_ARC_IMGTYPE',
    'FITS_VAL_COMPARISON_IMGTYPE',
    
    # --- Dataset Type Values ---
    'FITS_VAL_DARK_DATASET',
    'FITS_VAL_FLAT_DATASET',
    'FITS_VAL_SBIAS_DATASET',
    'FITS_VAL_SCIENCE_DATASET',
    
    # --- Special HDU Names ---
    'FITS_VAL_PRIMARY',
    'FITS_VAL_SCI',
    'FITS_VAL_ERR',
    'FITS_VAL_DQ',
    'FITS_VAL_WHT',
    'FITS_VAL_VAR',
    'FITS_VAL_MASK',
    
    # --- XTENSION Values ---
    'FITS_VAL_IMAGE_XTENSION',
    'FITS_VAL_BINTABLE_XTENSION',
    'FITS_VAL_TABLE_XTENSION',
    
    # --- BITPIX Values ---
    'FITS_BITPIX_UINT8',
    'FITS_BITPIX_INT16',
    'FITS_BITPIX_INT32',
    'FITS_BITPIX_INT64',
    'FITS_BITPIX_FLOAT32',
    'FITS_BITPIX_FLOAT64',
    
    # --- BUNIT Values ---
    'FITS_BUNIT_ADU',
    'FITS_BUNIT_ELECTRON',
    'FITS_BUNIT_ELECTRONS',
    'FITS_BUNIT_DN',
    'FITS_BUNIT_COUNT',
    'FITS_BUNIT_COUNTS',
    'FITS_BUNIT_PHOTON',
    'FITS_BUNIT_PHOTONS',
    
    # --- File and Format ---
    'FITS_FILE_EXTENSIONS',
    'FITS_SECTION_PATTERN',
    'FITS_SECTION_REGEX',
    
    # --- Validation Constants ---
    'MIN_EXPTIME',
    'MAX_EXPTIME',
    'MIN_GAIN',
    'MAX_GAIN',
    'DEFAULT_GAIN',
    'MIN_READNOISE',
    'MAX_READNOISE',
    'DEFAULT_READNOISE',
    'MIN_CCD_TEMP',
    'MAX_CCD_TEMP',
    'MIN_ADU',
    'MAX_ADU_16BIT',
    'MAX_ADU_14BIT',
    'MAX_ADU_12BIT',
    'MIN_BINNING',
    'MAX_BINNING',
    
    # --- Collections ---
    'ALL_IMAGE_TYPES',
    'CALIBRATION_IMAGE_TYPES',
    'SCIENCE_IMAGE_TYPES',
    'SECTION_KEYWORDS',
    'TEMPERATURE_KEYWORDS',
    'DETECTOR_CHAR_KEYWORDS',
    'WCS_KEYWORDS',
    'EXTENSION_KEYWORDS',
    'ESSENTIAL_KEYWORDS',
    'CALIBRATION_RECOMMENDED_KEYWORDS',
    'IMAGE_TYPE_KEYWORDS',
    
    # --- Region types ---
    'RegionType',
    
    # --- Mappings ---
    'NUMPY_TO_FITS_DTYPE',
    'FITS_TO_NUMPY_DTYPE',
    'IMAGE_TYPE_ALIASES',
    
    # --- Helper Functions ---
    'is_calibration_frame',
    'is_science_frame',
    'is_valid_image_type',
    'normalize_image_type',
    'get_image_type_description',
    'parse_fits_section',
    'is_fits_file',
]