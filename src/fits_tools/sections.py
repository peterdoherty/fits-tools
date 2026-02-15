#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for parsing FITS image sections from headers.

This module provides utilities to extract and interpret image section
definitions (e.g., BIASSEC, DATASEC, TRIMSEC, DARKSEC) from FITS headers,
converting them into 0-indexed Python slicing coordinates.

FITS Section Format:
    FITS section strings follow the format '[col1:col2,row1:row2]' where
    coordinates are 1-indexed and inclusive. This module converts them to
    0-indexed Python slicing coordinates (start inclusive, end exclusive).

The Section NamedTuple:
    This module provides a Section type that wraps the coordinate tuple
    with convenient properties and methods for common operations.

Version History:
    0.1.0 - Initial release
    0.2.0 - Added validation, doctests, utility functions, optional sections,
            improved regex handling, and better error messages
    0.3.0 - Refactored to use fits_tools constants and exceptions, added
            Section NamedTuple for type safety, improved integration
"""

import logging
import sys
from typing import Dict, Optional, Tuple, Union

from astropy.io.fits import Header

from fits_tools.constants import (
    FITS_KEY_DATASEC,
    FITS_KEY_BIASSEC,
    FITS_KEY_TRIMSEC,
    FITS_KEY_DARKSEC,
    SECTION_KEYWORDS,
    FITS_SECTION_REGEX,
)

from fits_tools.exceptions import (
    HeaderKeywordError,
    SectionParsingError,
)

# Module logger
logger = logging.getLogger(__name__)

__version__ = "0.3.0"


# =============================================================================
# Section Type Definition
# =============================================================================

class Section:
    """
    Represents a FITS section in 0-indexed Python coordinates.
    
    FITS sections use 1-indexed, inclusive coordinates like '[1:100,1:200]'.
    This class stores them as 0-indexed Python coordinates with exclusive end
    indices for direct use with numpy slicing.
    
    Attributes:
        start_col (int): Starting column (inclusive, 0-indexed).
        end_col (int): Ending column (exclusive, 0-indexed).
        start_row (int): Starting row (inclusive, 0-indexed).
        end_row (int): Ending row (exclusive, 0-indexed).
    
    Examples:
        >>> sec = Section(0, 100, 0, 200)
        >>> sec.width
        100
        >>> sec.height
        200
        >>> sec.area
        20000
        >>> sec.to_fits_string()
        '[1:100,1:200]'
    """
    
    def __init__(self, start_col: int, end_col: int, start_row: int, end_row: int):
        """
        Initialize a Section.
        
        Args:
            start_col: Starting column (0-indexed, inclusive).
            end_col: Ending column (0-indexed, exclusive).
            start_row: Starting row (0-indexed, inclusive).
            end_row: Ending row (0-indexed, exclusive).
        """
        self.start_col = int(start_col)
        self.end_col = int(end_col)
        self.start_row = int(start_row)
        self.end_row = int(end_row)
    
    def __repr__(self) -> str:
        return (
            f"Section(start_col={self.start_col}, end_col={self.end_col}, "
            f"start_row={self.start_row}, end_row={self.end_row})"
        )
    
    def __str__(self) -> str:
        return self.to_fits_string()
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Section):
            return (self.start_col == other.start_col and
                    self.end_col == other.end_col and
                    self.start_row == other.start_row and
                    self.end_row == other.end_row)
        elif isinstance(other, tuple) and len(other) == 4:
            return (self.start_col, self.end_col, self.start_row, self.end_row) == other
        return False
    
    def __iter__(self):
        """Allow unpacking: start_col, end_col, start_row, end_row = section"""
        return iter((self.start_col, self.end_col, self.start_row, self.end_row))
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple (start_col, end_col, start_row, end_row)."""
        return (self.start_col, self.end_col, self.start_row, self.end_row)
    
    @property
    def width(self) -> int:
        """Width of section in pixels."""
        return self.end_col - self.start_col
    
    @property
    def height(self) -> int:
        """Height of section in pixels."""
        return self.end_row - self.start_row
    
    @property
    def area(self) -> int:
        """Area in pixels (width × height)."""
        return self.width * self.height
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape as (height, width) - numpy convention."""
        return (self.height, self.width)
    
    def to_slices(self) -> Tuple[slice, slice]:
        """
        Convert to numpy slice objects.
        
        Returns:
            Tuple of (row_slice, col_slice) for numpy array indexing.
        
        Examples:
            >>> sec = Section(0, 100, 0, 200)
            >>> row_slice, col_slice = sec.to_slices()
            >>> import numpy as np
            >>> data = np.zeros((512, 512))
            >>> section_data = data[row_slice, col_slice]
            >>> section_data.shape
            (200, 100)
        """
        return slice(self.start_row, self.end_row), slice(self.start_col, self.end_col)
    
    def to_fits_string(self) -> str:
        """
        Convert to FITS format string '[col1:col2,row1:row2]' (1-indexed).
        
        Returns:
            FITS section string.
        
        Examples:
            >>> Section(0, 100, 0, 200).to_fits_string()
            '[1:100,1:200]'
            >>> Section(50, 100, 0, 50).to_fits_string()
            '[51:100,1:50]'
        """
        col1 = self.start_col + 1
        col2 = self.end_col
        row1 = self.start_row + 1
        row2 = self.end_row
        return f"[{col1}:{col2},{row1}:{row2}]"
    
    def contains_point(self, col: int, row: int) -> bool:
        """
        Check if a point (col, row) is within this section.
        
        Args:
            col: Column coordinate (0-indexed).
            row: Row coordinate (0-indexed).
        
        Returns:
            True if the point is within the section.
        
        Examples:
            >>> sec = Section(0, 100, 0, 200)
            >>> sec.contains_point(50, 100)
            True
            >>> sec.contains_point(150, 50)
            False
        """
        return (self.start_col <= col < self.end_col and 
                self.start_row <= row < self.end_row)
    
    def overlaps(self, other: 'Section') -> bool:
        """
        Check if this section overlaps with another.
        
        Args:
            other: Another Section to check.
        
        Returns:
            True if the sections overlap.
        
        Examples:
            >>> sec1 = Section(0, 100, 0, 200)
            >>> sec2 = Section(50, 150, 100, 250)
            >>> sec1.overlaps(sec2)
            True
            >>> sec3 = Section(200, 300, 0, 100)
            >>> sec1.overlaps(sec3)
            False
        """
        no_overlap = (
            self.end_col <= other.start_col or
            other.end_col <= self.start_col or
            self.end_row <= other.start_row or
            other.end_row <= self.start_row
        )
        return not no_overlap
    
    def fits_within(self, image_shape: Tuple[int, int]) -> bool:
        """
        Check if this section fits within an image's dimensions.
        
        Args:
            image_shape: Image dimensions as (rows, cols) (numpy shape).
        
        Returns:
            True if section is completely within image bounds.
        
        Examples:
            >>> Section(0, 100, 0, 200).fits_within((512, 512))
            True
            >>> Section(0, 600, 0, 200).fits_within((512, 512))
            False
        """
        rows, cols = image_shape
        return (self.start_col >= 0 and self.start_row >= 0 and
                self.end_col <= cols and self.end_row <= rows)


# =============================================================================
# Section Parsing Functions
# =============================================================================

def parse_section_string(
    section_str: str, 
    keyword_name: Optional[str] = None,
    validate: bool = True
) -> Section:
    """
    Parse a FITS section string (e.g., '[col1:col2,row1:row2]') to a Section object.

    FITS coordinates are 1-indexed and inclusive. Python slicing is 0-indexed with
    exclusive end. For example, FITS '[1:100,1:200]' becomes Section(0, 100, 0, 200).

    Args:
        section_str: The FITS section string to parse. Accepts formats with or
            without spaces: '[1:100,1:200]' or '[ 1 : 100 , 1 : 200 ]'
        keyword_name: The name of the FITS keyword (for error messages).
        validate: If True, validate that the parsed coordinates are sensible.

    Returns:
        Section object with 0-indexed Python slicing coordinates.

    Raises:
        SectionParsingError: If section_str is malformed or validation fails.

    Examples:
        >>> parse_section_string('[1:100,1:200]')
        Section(start_col=0, end_col=100, start_row=0, end_row=200)
        
        >>> parse_section_string('[51:100,1:50]', 'BIASSEC')
        Section(start_col=50, end_col=100, start_row=0, end_row=50)
        
        >>> parse_section_string('[ 1 : 100 , 1 : 200 ]')
        Section(start_col=0, end_col=100, start_row=0, end_row=200)
    """
    if section_str is None or not section_str:
        raise SectionParsingError(
            "FITS section string is empty or None.",
            section_string=str(section_str),
            keyword=keyword_name
        )

    # Remove spaces from section string for more flexible parsing
    section_str_clean = section_str.strip()# Use regex from constants
    match = FITS_SECTION_REGEX.match(section_str_clean)
    if not match:
        raise SectionParsingError(
            f"Malformed FITS section string: '{section_str}'. "
            "Expected format '[col1:col2,row1:row2]' with positive integers.",
            section_string=section_str,
            keyword=keyword_name
        )

    # Extract 1-indexed FITS coordinates
    col1, col2, row1, row2 = map(int, match.groups())

    # Validate FITS coordinates if requested
    if validate:
        # Check for zero values (FITS is 1-indexed)
        if col1 == 0 or col2 == 0 or row1 == 0 or row2 == 0:
            raise SectionParsingError(
                f"Invalid section: FITS coordinates must be >= 1. "
                f"Got: [{col1}:{col2},{row1}:{row2}]",
                section_string=section_str,
                keyword=keyword_name
            )
        
        # Check that start <= end
        if col1 > col2:
            raise SectionParsingError(
                f"Invalid section: column start ({col1}) > end ({col2}). "
                f"Section string: '{section_str}'",
                section_string=section_str,
                keyword=keyword_name
            )
        if row1 > row2:
            raise SectionParsingError(
                f"Invalid section: row start ({row1}) > end ({row2}). "
                f"Section string: '{section_str}'",
                section_string=section_str,
                keyword=keyword_name
            )

    # Convert to 0-indexed Python slicing coordinates
    start_col = col1 - 1
    end_col = col2
    start_row = row1 - 1
    end_row = row2

    return Section(start_col, end_col, start_row, end_row)


def get_section_by_keyword(
    header: Header, 
    keyword_name: str,
    required: bool = True,
    validate: bool = True
) -> Optional[Section]:
    """
    Extract section coordinates from a FITS header based on a given keyword.

    This retrieves the section string from the header and parses it using
    parse_section_string().

    Args:
        header: The FITS header object.
        keyword_name: The name of the FITS keyword (e.g., 'DATASEC', 'BIASSEC').
        required: If True, raise an error if keyword is missing. If False,
            return None when keyword is not found.
        validate: If True, validate that the parsed coordinates are sensible.

    Returns:
        Section object or None if keyword is missing and required=False.

    Raises:
        HeaderKeywordError: If keyword is missing (when required=True).
        SectionParsingError: If the keyword value is malformed.

    Examples:
        >>> from astropy.io import fits
        >>> header = fits.Header()
        >>> header['DATASEC'] = '[1:100,1:200]'
        >>> sec = get_section_by_keyword(header, 'DATASEC')
        >>> sec.width, sec.height
        (100, 200)
        
        >>> get_section_by_keyword(header, 'MISSING', required=False) is None
        True
    """
    if keyword_name not in header:
        if required:
            available_keys = [k for k in header.keys() if 'SEC' in k]
            msg = f"FITS header missing required keyword: {keyword_name}"
            if available_keys:
                msg += f". Available section keywords: {available_keys}"
            raise HeaderKeywordError(msg, keyword=keyword_name)
        else:
            return None
    
    section_str = header[keyword_name]
    return parse_section_string(section_str, keyword_name, validate=validate)


# =============================================================================
# Convenience Functions for Standard Section Keywords
# =============================================================================

def get_datasec(header: Header, required: bool = True) -> Optional[Section]:
    """
    Extract data section coordinates from a FITS header.

    Assumes 'DATASEC' keyword is present and in the format '[col1:col2,row1:row2]'.

    Args:
        header: The FITS header object.
        required: If True, raise an error if keyword is missing.

    Returns:
        Section object or None if DATASEC is missing and required=False.

    Raises:
        HeaderKeywordError: If 'DATASEC' keyword is missing (when required=True).
        SectionParsingError: If DATASEC is malformed.

    Examples:
        >>> from astropy.io import fits
        >>> header = fits.Header()
        >>> header['DATASEC'] = '[1:512,1:512]'
        >>> sec = get_datasec(header)
        >>> sec.shape
        (512, 512)
    """
    return get_section_by_keyword(header, FITS_KEY_DATASEC, required=required)


def get_biassec(header: Header, required: bool = True) -> Optional[Section]:
    """
    Extract bias section coordinates from a FITS header.

    Assumes 'BIASSEC' keyword is present and in the format '[col1:col2,row1:row2]'.

    Args:
        header: The FITS header object.
        required: If True, raise an error if keyword is missing.

    Returns:
        Section object or None if BIASSEC is missing and required=False.

    Raises:
        HeaderKeywordError: If 'BIASSEC' keyword is missing (when required=True).
        SectionParsingError: If BIASSEC is malformed.

    Examples:
        >>> from astropy.io import fits
        >>> header = fits.Header()
        >>> header['BIASSEC'] = '[513:544,1:512]'
        >>> sec = get_biassec(header)
        >>> sec.width
        32
    """
    return get_section_by_keyword(header, FITS_KEY_BIASSEC, required=required)


def get_trimsec(header: Header, required: bool = True) -> Optional[Section]:
    """
    Extract trim section coordinates from a FITS header.

    Assumes 'TRIMSEC' keyword is present and in the format '[col1:col2,row1:row2]'.

    Args:
        header: The FITS header object.
        required: If True, raise an error if keyword is missing.

    Returns:
        Section object or None if TRIMSEC is missing and required=False.

    Raises:
        HeaderKeywordError: If 'TRIMSEC' keyword is missing (when required=True).
        SectionParsingError: If TRIMSEC is malformed.

    Examples:
        >>> from astropy.io import fits
        >>> header = fits.Header()
        >>> header['TRIMSEC'] = '[11:502,21:512]'
        >>> sec = get_trimsec(header)
        >>> sec.to_tuple()
        (10, 502, 20, 512)
    """
    return get_section_by_keyword(header, FITS_KEY_TRIMSEC, required=required)


def get_darksec(header: Header, required: bool = True) -> Optional[Section]:
    """
    Extract dark section coordinates from a FITS header.

    Assumes 'DARKSEC' keyword is present and in the format '[col1:col2,row1:row2]'.

    Args:
        header: The FITS header object.
        required: If True, raise an error if keyword is missing.

    Returns:
        Section object or None if DARKSEC is missing and required=False.

    Raises:
        HeaderKeywordError: If 'DARKSEC' keyword is missing (when required=True).
        SectionParsingError: If DARKSEC is malformed.

    Examples:
        >>> from astropy.io import fits
        >>> header = fits.Header()
        >>> header['DARKSEC'] = '[1:50,1:512]'
        >>> sec = get_darksec(header)
        >>> sec.area
        25600
        
        >>> header2 = fits.Header()
        >>> get_darksec(header2, required=False) is None
        True
    """
    return get_section_by_keyword(header, FITS_KEY_DARKSEC, required=required)


def get_all_sections(
    header: Header, 
    required: bool = False
) -> Dict[str, Section]:
    """
    Extract all standard section keywords from a header.

    Args:
        header: The FITS header object.
        required: If True, raise errors for missing sections. If False,
            skip missing sections.

    Returns:
        Dictionary with section keyword names as keys and Section objects
        as values. Only includes sections that are present in the header.

    Examples:
        >>> from astropy.io import fits
        >>> header = fits.Header()
        >>> header['DATASEC'] = '[1:512,1:512]'
        >>> header['BIASSEC'] = '[513:544,1:512]'
        >>> sections = get_all_sections(header)
        >>> 'DATASEC' in sections and 'BIASSEC' in sections
        True
        >>> sections['DATASEC'].area
        262144
    """
    sections = {}
    
    for keyword in SECTION_KEYWORDS:
        section = get_section_by_keyword(header, keyword, required=required)
        if section is not None:
            sections[keyword] = section
    
    return sections

"""
New overscan region functions to add to fits_tools/sections.py

Insert these functions after the get_all_sections() function and before
the "Deprecated Utility Functions" section.

Also requires:
  - Adding 'import numpy as np' to the imports at the top of sections.py
  - Adding the new function names to __all__
  - Updating __version__ to "0.4.0"
"""

# =============================================================================
# Overscan Region Detection
# =============================================================================

def _detect_active_region_from_image(
    image: 'np.ndarray',
    threshold_fraction: float = 0.1,
    min_gradient_percentile: float = 90.0,
) -> Optional[Section]:
    """
    Attempt to detect the active (illuminated) region of an image by finding
    sharp signal transitions at the overscan boundaries.

    This is a last-resort fallback when neither DATASEC nor caller-provided
    geometry is available. It works best on flat-field images with clear
    contrast between the illuminated active area and the overscan.

    The algorithm:
        1. Compute median column profile and median row profile.
        2. Compute the absolute gradient of each profile.
        3. Identify the two strongest gradient peaks in each profile as the
           boundaries of the active region.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    threshold_fraction : float, optional
        Minimum signal fraction (relative to median) to consider a pixel
        as part of the illuminated region. Used as a sanity check.
        Defaults to 0.1.
    min_gradient_percentile : float, optional
        Percentile threshold for identifying gradient peaks. Higher values
        require sharper transitions. Defaults to 90.0.

    Returns
    -------
    Section or None
        Detected active region, or None if detection fails.

    Notes
    -----
    This function is intentionally conservative. It returns None rather than
    guessing when the image structure is ambiguous (e.g., dark frames, very
    low signal, or images without overscan).
    """
    import numpy as np

    if image.ndim != 2:
        logger.warning("_detect_active_region_from_image: image is not 2D.")
        return None

    nrows, ncols = image.shape

    try:
        # Median profiles along each axis
        col_profile = np.median(image, axis=0)  # shape: (ncols,)
        row_profile = np.median(image, axis=1)  # shape: (nrows,)

        # Check that there is meaningful signal
        median_signal = np.median(image)
        if median_signal <= 0:
            logger.warning("_detect_active_region_from_image: no positive signal detected.")
            return None

        def _find_edges(profile):
            """Find the start and end of the high-signal region in a 1D profile."""
            gradient = np.abs(np.diff(profile.astype(np.float64)))
            if gradient.max() == 0:
                return None, None

            grad_threshold = np.percentile(gradient, min_gradient_percentile)
            peaks = np.where(gradient >= grad_threshold)[0]

            if len(peaks) < 2:
                return None, None

            # The first and last strong gradient positions are the edges
            start = int(peaks[0]) + 1   # +1 because diff shifts by one
            end = int(peaks[-1]) + 1    # exclusive end for slicing

            # Sanity check: the region between edges should have higher signal
            region_median = np.median(profile[start:end])
            outside_left = np.median(profile[:max(start, 1)])
            outside_right = np.median(profile[min(end, len(profile) - 1):])

            if region_median <= max(outside_left, outside_right) * (1.0 + threshold_fraction):
                return None, None

            return start, end

        col_start, col_end = _find_edges(col_profile)
        row_start, row_end = _find_edges(row_profile)

        if any(v is None for v in (col_start, col_end, row_start, row_end)):
            logger.warning("_detect_active_region_from_image: could not reliably "
                           "detect active region boundaries.")
            return None

        detected = Section(col_start, col_end, row_start, row_end)
        logger.info(f"Auto-detected active region: {detected} "
                    f"(width={detected.width}, height={detected.height})")
        return detected

    except Exception as e:
        logger.warning(f"_detect_active_region_from_image: detection failed: {e}")
        return None


def _resolve_datasec(
    header: Optional['Header'] = None,
    datasec: Optional[Section] = None,
    image: Optional['np.ndarray'] = None,
) -> Section:
    """
    Resolve the active data section from available sources.

    Resolution order:
        1. Caller-provided ``datasec`` (highest priority).
        2. DATASEC keyword from ``header``.
        3. Auto-detection from ``image`` data (last resort).

    Parameters
    ----------
    header : astropy.io.fits.Header or None
        FITS header that may contain DATASEC.
    datasec : Section or None
        Explicitly provided active region geometry.
    image : np.ndarray or None
        Image data for auto-detection fallback.

    Returns
    -------
    Section
        Resolved active data section.

    Raises
    ------
    ValueError
        If the active region cannot be determined from any source.
    """
    # Priority 1: Caller-provided geometry
    if datasec is not None:
        logger.debug(f"Using caller-provided DATASEC: {datasec}")
        return datasec

    # Priority 2: Header keyword
    if header is not None:
        header_datasec = get_datasec(header, required=False)
        if header_datasec is not None:
            logger.debug(f"Using DATASEC from header: {header_datasec}")
            return header_datasec

    # Priority 3: Auto-detect from image
    if image is not None:
        detected = _detect_active_region_from_image(image)
        if detected is not None:
            logger.info(f"Using auto-detected active region: {detected}")
            return detected

    raise ValueError(
        "Cannot determine active data region. Provide one of: "
        "datasec parameter, DATASEC header keyword, or image data for auto-detection."
    )


def get_serial_overscan(
    header: Optional['Header'] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    datasec: Optional[Section] = None,
    image: Optional['np.ndarray'] = None,
    keyword: str = FITS_KEY_BIASSEC,
    required: bool = False,
) -> Optional[Section]:
    """
    Get the serial (horizontal) overscan region.

    Resolution order:
        1. Header keyword (default: ``BIASSEC``; configurable via ``keyword``).
        2. Infer from image geometry: columns beyond DATASEC on either side.
           If columns exist on both sides of the active area, the wider
           region is returned (typically the readout-side overscan).

    The active region (DATASEC) is resolved via :func:`_resolve_datasec`
    using the ``header``, ``datasec``, and ``image`` parameters.

    Parameters
    ----------
    header : astropy.io.fits.Header or None, optional
        FITS header. Used to look up the overscan keyword and DATASEC.
    image_shape : tuple of (int, int) or None, optional
        Full image dimensions as ``(nrows, ncols)``. Required for geometry
        inference if ``image`` is not provided.
    datasec : Section or None, optional
        Explicitly provided active region geometry. Takes priority over
        DATASEC from header.
    image : np.ndarray or None, optional
        Image data. Used as fallback for DATASEC auto-detection, and its
        shape is used for geometry inference if ``image_shape`` is None.
    keyword : str, optional
        Header keyword for serial overscan. Defaults to ``BIASSEC``.
    required : bool, optional
        If True, raise an error when the overscan cannot be determined.
        If False, return None. Defaults to False.

    Returns
    -------
    Section or None
        Serial overscan region, or None if it cannot be determined and
        ``required`` is False.

    Raises
    ------
    HeaderKeywordError
        If ``required`` is True and the overscan cannot be determined.

    Examples
    --------
    >>> from astropy.io import fits
    >>> header = fits.Header()
    >>> header['DATASEC'] = '[1:512,1:512]'
    >>> header['BIASSEC'] = '[513:544,1:512]'
    >>> sec = get_serial_overscan(header, image_shape=(512, 544))
    >>> sec.width
    32
    """
    # Try header keyword first
    if header is not None:
        keyword_sec = get_section_by_keyword(header, keyword, required=False)
        if keyword_sec is not None:
            logger.debug(f"Serial overscan from {keyword}: {keyword_sec}")
            return keyword_sec

    # Resolve image_shape
    effective_shape = image_shape
    if effective_shape is None and image is not None:
        effective_shape = image.shape

    if effective_shape is None:
        if required:
            raise HeaderKeywordError(
                f"Cannot determine serial overscan: no {keyword} keyword, "
                "and no image_shape or image provided for inference.",
                keyword=keyword
            )
        return None

    # Resolve DATASEC for geometry inference
    try:
        resolved_datasec = _resolve_datasec(header=header, datasec=datasec, image=image)
    except ValueError:
        if required:
            raise HeaderKeywordError(
                f"Cannot determine serial overscan: no {keyword} keyword "
                "and DATASEC could not be resolved for inference.",
                keyword=keyword
            )
        return None

    nrows, ncols = effective_shape

    # Check for columns on both sides of the active area
    left_cols = resolved_datasec.start_col  # columns before active area
    right_cols = ncols - resolved_datasec.end_col  # columns after active area

    # Use the row range of the active area for the overscan region
    os_start_row = resolved_datasec.start_row
    os_end_row = resolved_datasec.end_row

    if right_cols > 0 and right_cols >= left_cols:
        # Overscan to the right of the active area (most common)
        result = Section(resolved_datasec.end_col, ncols, os_start_row, os_end_row)
        logger.debug(f"Serial overscan inferred (right): {result}")
        return result
    elif left_cols > 0:
        # Overscan to the left of the active area
        result = Section(0, resolved_datasec.start_col, os_start_row, os_end_row)
        logger.debug(f"Serial overscan inferred (left): {result}")
        return result

    if required:
        raise HeaderKeywordError(
            f"Cannot determine serial overscan: no {keyword} keyword and "
            "no extra columns found outside DATASEC.",
            keyword=keyword
        )
    return None


def get_parallel_overscan(
    header: Optional['Header'] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    datasec: Optional[Section] = None,
    image: Optional['np.ndarray'] = None,
    keyword: Optional[str] = None,
    required: bool = False,
) -> Optional[Section]:
    """
    Get the parallel (vertical) overscan region.

    Resolution order:
        1. Header keyword (if ``keyword`` is provided and found in header).
        2. Infer from image geometry: rows beyond DATASEC above or below.
           If rows exist on both sides, the larger region is returned.

    The active region (DATASEC) is resolved via :func:`_resolve_datasec`
    using the ``header``, ``datasec``, and ``image`` parameters.

    Parameters
    ----------
    header : astropy.io.fits.Header or None, optional
        FITS header. Used to look up the overscan keyword and DATASEC.
    image_shape : tuple of (int, int) or None, optional
        Full image dimensions as ``(nrows, ncols)``. Required for geometry
        inference if ``image`` is not provided.
    datasec : Section or None, optional
        Explicitly provided active region geometry. Takes priority over
        DATASEC from header.
    image : np.ndarray or None, optional
        Image data. Used as fallback for DATASEC auto-detection, and its
        shape is used for geometry inference if ``image_shape`` is None.
    keyword : str or None, optional
        Header keyword for parallel overscan. There is no universal standard
        for this keyword; common choices include ``"POSCAN"``, ``"PARSEC"``,
        or instrument-specific names. If None, skips keyword lookup and
        relies on geometry inference. Defaults to None.
    required : bool, optional
        If True, raise an error when the overscan cannot be determined.
        If False, return None. Defaults to False.

    Returns
    -------
    Section or None
        Parallel overscan region, or None if it cannot be determined and
        ``required`` is False.

    Raises
    ------
    HeaderKeywordError
        If ``required`` is True and the overscan cannot be determined.

    Examples
    --------
    >>> from astropy.io import fits
    >>> header = fits.Header()
    >>> header['DATASEC'] = '[1:512,1:512]'
    >>> sec = get_parallel_overscan(header, image_shape=(544, 512))
    >>> sec.height
    32
    """
    # Try header keyword first (if provided)
    if keyword is not None and header is not None:
        keyword_sec = get_section_by_keyword(header, keyword, required=False)
        if keyword_sec is not None:
            logger.debug(f"Parallel overscan from {keyword}: {keyword_sec}")
            return keyword_sec

    # Resolve image_shape
    effective_shape = image_shape
    if effective_shape is None and image is not None:
        effective_shape = image.shape

    if effective_shape is None:
        if required:
            kw_msg = f"no {keyword} keyword, " if keyword else ""
            raise HeaderKeywordError(
                f"Cannot determine parallel overscan: {kw_msg}"
                "and no image_shape or image provided for inference.",
                keyword=keyword or "PARALLEL_OVERSCAN"
            )
        return None

    # Resolve DATASEC for geometry inference
    try:
        resolved_datasec = _resolve_datasec(header=header, datasec=datasec, image=image)
    except ValueError:
        if required:
            kw_msg = f"no {keyword} keyword " if keyword else "no keyword configured "
            raise HeaderKeywordError(
                f"Cannot determine parallel overscan: {kw_msg}"
                "and DATASEC could not be resolved for inference.",
                keyword=keyword or "PARALLEL_OVERSCAN"
            )
        return None

    nrows, ncols = effective_shape

    # Check for rows on both sides of the active area
    top_rows = resolved_datasec.start_row  # rows above active area
    bottom_rows = nrows - resolved_datasec.end_row  # rows below active area

    # Use the column range of the active area for the overscan region
    os_start_col = resolved_datasec.start_col
    os_end_col = resolved_datasec.end_col

    if bottom_rows > 0 and bottom_rows >= top_rows:
        # Overscan below the active area (most common)
        result = Section(os_start_col, os_end_col, resolved_datasec.end_row, nrows)
        logger.debug(f"Parallel overscan inferred (bottom): {result}")
        return result
    elif top_rows > 0:
        # Overscan above the active area
        result = Section(os_start_col, os_end_col, 0, resolved_datasec.start_row)
        logger.debug(f"Parallel overscan inferred (top): {result}")
        return result

    if required:
        kw_msg = f"no {keyword} keyword and " if keyword else ""
        raise HeaderKeywordError(
            f"Cannot determine parallel overscan: {kw_msg}"
            "no extra rows found outside DATASEC.",
            keyword=keyword or "PARALLEL_OVERSCAN"
        )
    return None


def infer_overscan_regions(
    header: Optional['Header'] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    datasec: Optional[Section] = None,
    image: Optional['np.ndarray'] = None,
    serial_keyword: str = FITS_KEY_BIASSEC,
    parallel_keyword: Optional[str] = None,
) -> Dict[str, Section]:
    """
    Determine all available overscan regions from header and image geometry.

    This is a convenience function that calls :func:`get_serial_overscan` and
    :func:`get_parallel_overscan` and returns a dictionary of the results.
    Only regions that could be determined are included in the output.

    Parameters
    ----------
    header : astropy.io.fits.Header or None, optional
        FITS header.
    image_shape : tuple of (int, int) or None, optional
        Full image dimensions as ``(nrows, ncols)``.
    datasec : Section or None, optional
        Explicitly provided active region geometry.
    image : np.ndarray or None, optional
        Image data for auto-detection fallback.
    serial_keyword : str, optional
        Header keyword for serial overscan. Defaults to ``BIASSEC``.
    parallel_keyword : str or None, optional
        Header keyword for parallel overscan. Defaults to None (infer only).

    Returns
    -------
    dict of str to Section
        Dictionary with keys ``'serial'`` and/or ``'parallel'``, containing
        the corresponding overscan Section objects. Only includes regions
        that could be determined. May also include ``'datasec'`` with the
        resolved active area.

    Examples
    --------
    >>> from astropy.io import fits
    >>> header = fits.Header()
    >>> header['DATASEC'] = '[1:512,1:512]'
    >>> header['BIASSEC'] = '[513:544,1:512]'
    >>> regions = infer_overscan_regions(header, image_shape=(544, 544))
    >>> 'serial' in regions
    True
    >>> 'parallel' in regions
    True
    >>> regions['serial'].width
    32
    >>> regions['parallel'].height
    32
    """
    regions = {}

    # Resolve DATASEC first and include it in the output
    try:
        resolved_datasec = _resolve_datasec(header=header, datasec=datasec, image=image)
        regions['datasec'] = resolved_datasec
    except ValueError as e:
        logger.warning(f"Could not resolve DATASEC: {e}")
        return regions

    # Serial overscan
    serial = get_serial_overscan(
        header=header,
        image_shape=image_shape,
        datasec=resolved_datasec,
        image=image,
        keyword=serial_keyword,
        required=False,
    )
    if serial is not None:
        regions['serial'] = serial

    # Parallel overscan
    parallel = get_parallel_overscan(
        header=header,
        image_shape=image_shape,
        datasec=resolved_datasec,
        image=image,
        keyword=parallel_keyword,
        required=False,
    )
    if parallel is not None:
        regions['parallel'] = parallel

    # Log summary
    found = [k for k in ('serial', 'parallel') if k in regions]
    if found:
        logger.info(f"Overscan regions found: {', '.join(found)}")
        for name in found:
            sec = regions[name]
            logger.info(f"  {name}: {sec} "
                        f"({'width=' + str(sec.width) if name == 'serial' else 'height=' + str(sec.height)})")
    else:
        logger.warning("No overscan regions could be determined.")

    return regions

# =============================================================================
# Deprecated Utility Functions (for backward compatibility)
# =============================================================================
# These maintain the old tuple-based API while using Section internally

def get_section_dimensions(section: Union[Section, Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """
    Calculate the dimensions (width, height) of a section.
    
    Deprecated: Use Section.width and Section.height properties instead.

    Args:
        section: Section object or tuple (start_col, end_col, start_row, end_row).

    Returns:
        (width, height) of the section in pixels.

    Examples:
        >>> get_section_dimensions((0, 100, 0, 200))
        (100, 200)
        
        >>> sec = Section(50, 100, 20, 120)
        >>> get_section_dimensions(sec)
        (50, 100)
    """
    if isinstance(section, Section):
        return section.width, section.height
    start_col, end_col, start_row, end_row = section
    return end_col - start_col, end_row - start_row


def get_section_area(section: Union[Section, Tuple[int, int, int, int]]) -> int:
    """
    Calculate the area (number of pixels) of a section.
    
    Deprecated: Use Section.area property instead.

    Args:
        section: Section object or tuple (start_col, end_col, start_row, end_row).

    Returns:
        Number of pixels in the section.

    Examples:
        >>> get_section_area((0, 100, 0, 200))
        20000
        
        >>> get_section_area(Section(0, 10, 0, 10))
        100
    """
    if isinstance(section, Section):
        return section.area
    width, height = get_section_dimensions(section)
    return width * height


def sections_overlap(
    section1: Union[Section, Tuple[int, int, int, int]], 
    section2: Union[Section, Tuple[int, int, int, int]]
) -> bool:
    """
    Check if two sections overlap.
    
    Deprecated: Use Section.overlaps() method instead.

    Args:
        section1: First section.
        section2: Second section.

    Returns:
        True if the sections overlap.

    Examples:
        >>> sections_overlap((0, 100, 0, 200), (50, 150, 100, 250))
        True
        
        >>> sections_overlap((0, 100, 0, 100), (100, 200, 0, 100))
        False
    """
    if not isinstance(section1, Section):
        section1 = Section(*section1)
    if not isinstance(section2, Section):
        section2 = Section(*section2)
    return section1.overlaps(section2)


def section_contains_point(
    section: Union[Section, Tuple[int, int, int, int]], 
    col: int, 
    row: int
) -> bool:
    """
    Check if a point (col, row) is within a section.
    
    Deprecated: Use Section.contains_point() method instead.

    Args:
        section: Section object or tuple.
        col: Column coordinate (0-indexed).
        row: Row coordinate (0-indexed).

    Returns:
        True if the point is within the section.

    Examples:
        >>> section_contains_point((0, 100, 0, 200), 50, 100)
        True
        
        >>> section_contains_point((0, 100, 0, 200), 150, 50)
        False
    """
    if not isinstance(section, Section):
        section = Section(*section)
    return section.contains_point(col, row)


def section_to_fits_string(section: Union[Section, Tuple[int, int, int, int]]) -> str:
    """
    Convert Python section coordinates back to FITS format string.
    
    Deprecated: Use Section.to_fits_string() method instead.

    Args:
        section: Section object or tuple (start_col, end_col, start_row, end_row).

    Returns:
        FITS format section string '[col1:col2,row1:row2]'.

    Examples:
        >>> section_to_fits_string((0, 100, 0, 200))
        '[1:100,1:200]'
        
        >>> section_to_fits_string(Section(50, 100, 0, 50))
        '[51:100,1:50]'
    """
    if not isinstance(section, Section):
        section = Section(*section)
    return section.to_fits_string()


def section_to_slices(section: Union[Section, Tuple[int, int, int, int]]) -> Tuple[slice, slice]:
    """
    Convert section coordinates to Python slice objects.
    
    Deprecated: Use Section.to_slices() method instead.

    Args:
        section: Section object or tuple.

    Returns:
        (row_slice, col_slice) for numpy array indexing.

    Examples:
        >>> row_slice, col_slice = section_to_slices((0, 100, 0, 200))
        >>> print(row_slice, col_slice)
        slice(0, 200, None) slice(0, 100, None)
    """
    if not isinstance(section, Section):
        section = Section(*section)
    return section.to_slices()


def validate_section_in_image(
    section: Union[Section, Tuple[int, int, int, int]],
    image_shape: Tuple[int, int]
) -> bool:
    """
    Validate that a section fits within an image's dimensions.
    
    Deprecated: Use Section.fits_within() method instead.

    Args:
        section: Section object or tuple.
        image_shape: Image dimensions as (rows, cols).

    Returns:
        True if section is completely within image bounds.

    Raises:
        ValueError: If section coordinates are negative or inverted.

    Examples:
        >>> validate_section_in_image((0, 100, 0, 200), (512, 512))
        True
        
        >>> validate_section_in_image((0, 600, 0, 200), (512, 512))
        False
    """
    if not isinstance(section, Section):
        start_col, end_col, start_row, end_row = section
        # Validate coordinates
        if start_col < 0 or start_row < 0:
            raise ValueError(f"Section has negative coordinates: {section}")
        if start_col >= end_col or start_row >= end_row:
            raise ValueError(f"Section has inverted coordinates: {section}")
        section = Section(start_col, end_col, start_row, end_row)
    
    return section.fits_within(image_shape)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Section type
    'Section',
    
    # Parsing functions
    'parse_section_string',
    'get_section_by_keyword',
    
    # Convenience functions for standard keywords
    'get_datasec',
    'get_biassec',
    'get_trimsec',
    'get_darksec',
    'get_all_sections',
    
    # Deprecated utility functions (backward compatibility)
    'get_section_dimensions',
    'get_section_area',
    'sections_overlap',
    'section_contains_point',
    'section_to_fits_string',
    'section_to_slices',
    'validate_section_in_image',
]


# =============================================================================
# Main / Testing
# =============================================================================

def main():
    """Run doctests for the sections module."""
    import argparse
    import doctest
    
    parser = argparse.ArgumentParser(
        description='FITS Section Parser - Testing and Utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Run doctests')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output for doctests')
    parser.add_argument('--version', action='version',
                       version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.test or len(sys.argv) == 1:
        logger.info("Running doctests for sections.py...")
        result = doctest.testmod(verbose=args.verbose)
        
        if result.failed == 0:
            logger.info(f"All {result.attempted} doctests passed!")
            return 0
        else:
            logger.error(f"{result.failed} of {result.attempted} doctests failed!")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
