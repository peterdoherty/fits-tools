#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Functions for parsing and working with FITS image sections.

This module provides utilities to extract and interpret image section
definitions (e.g., BIASSEC, DATASEC, TRIMSEC, DARKSEC) from FITS headers,
converting them into 0-indexed Python slicing coordinates. It also includes
helpers for inferring overscan regions when explicit keywords are missing.

FITS Section Format
-------------------
FITS section strings follow the format ``'[col1:col2,row1:row2]'`` where
coordinates are 1-indexed and inclusive. This module converts them to
0-indexed Python slicing coordinates (start inclusive, end exclusive).

Section Class
-------------
The :class:`Section` type wraps a region definition with convenient
properties and methods for common operations (width, height, overlaps,
conversion to FITS strings or NumPy slices).

Version History
---------------
0.1.0
    Initial release.
0.2.0
    Added validation, doctests, utility functions, optional sections,
    improved regex handling, and better error messages.
0.3.0
    Refactored to use fits_tools constants and exceptions, added
    :class:`Section` for type safety, improved integration.
0.4.0
    Added overscan region inference helpers; integrated NumPy directly;
    removed CLI/doctest harness; general cleanup.
"""

from __future__ import annotations

import logging

import numpy as np
from astropy.io.fits import Header

from .constants import (
    FITS_KEY_BIASSEC,
    FITS_KEY_DARKSEC,
    FITS_KEY_DATASEC,
    FITS_KEY_TRIMSEC,
    FITS_SECTION_REGEX,
    SECTION_KEYWORDS,
)
from .exceptions import HeaderKeywordError, SectionParsingError

logger = logging.getLogger(__name__)


# =============================================================================
# Section Type Definition
# =============================================================================

class Section:
    """
    Represents a FITS section in 0-indexed Python coordinates.

    FITS sections use 1-indexed, inclusive coordinates like ``'[1:100,1:200]'``.
    This class stores them as 0-indexed Python coordinates with exclusive end
    indices for direct use with NumPy slicing.

    Attributes
    ----------
    start_col, end_col, start_row, end_row : int
        0-indexed, with ``end_*`` exclusive.

    Examples
    --------
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
            return (
                self.start_col == other.start_col
                and self.end_col == other.end_col
                and self.start_row == other.start_row
                and self.end_row == other.end_row
            )
        if isinstance(other, tuple) and len(other) == 4:
            return (self.start_col, self.end_col, self.start_row, self.end_row) == other
        return False

    def __iter__(self):
        """Allow unpacking: ``start_col, end_col, start_row, end_row = section``."""
        return iter((self.start_col, self.end_col, self.start_row, self.end_row))

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return ``(start_col, end_col, start_row, end_row)``."""
        return (self.start_col, self.end_col, self.start_row, self.end_row)

    @property
    def width(self) -> int:
        """Width of the section in pixels."""
        return self.end_col - self.start_col

    @property
    def height(self) -> int:
        """Height of the section in pixels."""
        return self.end_row - self.start_row

    @property
    def area(self) -> int:
        """Area in pixels (width × height)."""
        return self.width * self.height

    @property
    def shape(self) -> tuple[int, int]:
        """Shape as ``(height, width)`` (NumPy convention)."""
        return (self.height, self.width)

    def to_slices(self) -> tuple[slice, slice]:
        """
        Convert to NumPy slice objects.

        Returns
        -------
        (row_slice, col_slice) : tuple of slice
            Suitable for indexing ``data[row_slice, col_slice]``.

        Examples
        --------
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
        Convert to FITS format string ``'[col1:col2,row1:row2]'`` (1-indexed).

        Examples
        --------
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
        Check if a point (col, row) is within this section (0-indexed).

        Examples
        --------
        >>> sec = Section(0, 100, 0, 200)
        >>> sec.contains_point(50, 100)
        True
        >>> sec.contains_point(150, 50)
        False
        """
        return (
            self.start_col <= col < self.end_col
            and self.start_row <= row < self.end_row
        )

    def overlaps(self, other: Section) -> bool:
        """
        Check if this section overlaps with another.

        Examples
        --------
        >>> sec1 = Section(0, 100, 0, 200)
        >>> sec2 = Section(50, 150, 100, 250)
        >>> sec1.overlaps(sec2)
        True
        >>> sec3 = Section(200, 300, 0, 100)
        >>> sec1.overlaps(sec3)
        False
        """
        no_overlap = (
            self.end_col <= other.start_col
            or other.end_col <= self.start_col
            or self.end_row <= other.start_row
            or other.end_row <= self.start_row
        )
        return not no_overlap

    def fits_within(self, image_shape: tuple[int, int]) -> bool:
        """
        Check if this section fits within an image's dimensions.

        Parameters
        ----------
        image_shape :
            Image dimensions as ``(rows, cols)``.

        Examples
        --------
        >>> Section(0, 100, 0, 200).fits_within((512, 512))
        True
        >>> Section(0, 600, 0, 200).fits_within((512, 512))
        False
        """
        rows, cols = image_shape
        return (
            self.start_col >= 0
            and self.start_row >= 0
            and self.end_col <= cols
            and self.end_row <= rows
        )


# =============================================================================
# Section Parsing
# =============================================================================

def parse_section_string(
    section_str: str,
    keyword_name: str | None = None,
    validate: bool = True,
) -> Section:
    """
    Parse a FITS section string to a :class:`Section` object.

    FITS coordinates are 1-indexed and inclusive. Python slicing is 0-indexed
    with exclusive end. For example, FITS ``'[1:100,1:200]'`` becomes
    ``Section(0, 100, 0, 200)``.

    Parameters
    ----------
    section_str :
        FITS section string to parse. Whitespace is allowed.
    keyword_name :
        FITS keyword name (e.g. ``'DATASEC'``) for error messages.
    validate :
        If True, check for obviously invalid coordinates.

    Returns
    -------
    Section

    Raises
    ------
    SectionParsingError
        If the string is malformed or validation fails.
    """
    if section_str is None or not section_str:
        raise SectionParsingError(
            "FITS section string is empty or None.",
            section_string=str(section_str),
            keyword=keyword_name,
        )

    section_str_clean = section_str.strip()
    match = FITS_SECTION_REGEX.match(section_str_clean)
    if not match:
        raise SectionParsingError(
            f"Malformed FITS section string: {section_str!r}. "
            "Expected format '[col1:col2,row1:row2]' with positive integers.",
            section_string=section_str,
            keyword=keyword_name,
        )

    col1, col2, row1, row2 = map(int, match.groups())

    if validate:
        if col1 == 0 or col2 == 0 or row1 == 0 or row2 == 0:
            raise SectionParsingError(
                "Invalid section: FITS coordinates must be >= 1. "
                f"Got: [{col1}:{col2},{row1}:{row2}]",
                section_string=section_str,
                keyword=keyword_name,
            )
        if col1 > col2:
            raise SectionParsingError(
                f"Invalid section: column start ({col1}) > end ({col2}). "
                f"Section string: {section_str!r}",
                section_string=section_str,
                keyword=keyword_name,
            )
        if row1 > row2:
            raise SectionParsingError(
                f"Invalid section: row start ({row1}) > end ({row2}). "
                f"Section string: {section_str!r}",
                section_string=section_str,
                keyword=keyword_name,
            )

    start_col = col1 - 1
    end_col = col2
    start_row = row1 - 1
    end_row = row2

    return Section(start_col, end_col, start_row, end_row)


def get_section_by_keyword(
    header: Header,
    keyword_name: str,
    required: bool = True,
    validate: bool = True,
) -> Section | None:
    """
    Extract section coordinates from a FITS header based on a given keyword.

    Parameters
    ----------
    header :
        FITS header.
    keyword_name :
        Name of the FITS keyword (e.g. ``'DATASEC'``, ``'BIASSEC'``).
    required :
        If True, raise if keyword is missing. If False, return None when
        keyword is not present.
    validate :
        If True, validate the parsed coordinates.

    Returns
    -------
    Section or None

    Raises
    ------
    HeaderKeywordError
        If the keyword is missing when ``required`` is True.
    SectionParsingError
        If the keyword value is malformed.
    """
    if keyword_name not in header:
        if required:
            available_keys = [k for k in header.keys() if "SEC" in k]
            msg = f"FITS header missing required keyword: {keyword_name}"
            if available_keys:
                msg += f". Available section keywords: {available_keys}"
            raise HeaderKeywordError(msg, keyword=keyword_name)
        return None

    section_str = header[keyword_name]
    return parse_section_string(section_str, keyword_name, validate=validate)


# =============================================================================
# Convenience Functions for Standard Section Keywords
# =============================================================================

def get_datasec(header: Header, required: bool = True) -> Section | None:
    """Extract ``DATASEC`` as a :class:`Section`."""
    return get_section_by_keyword(header, FITS_KEY_DATASEC, required=required)


def get_biassec(header: Header, required: bool = True) -> Section | None:
    """Extract ``BIASSEC`` as a :class:`Section`."""
    return get_section_by_keyword(header, FITS_KEY_BIASSEC, required=required)


def get_trimsec(header: Header, required: bool = True) -> Section | None:
    """Extract ``TRIMSEC`` as a :class:`Section`."""
    return get_section_by_keyword(header, FITS_KEY_TRIMSEC, required=required)


def get_darksec(header: Header, required: bool = True) -> Section | None:
    """Extract ``DARKSEC`` as a :class:`Section`."""
    return get_section_by_keyword(header, FITS_KEY_DARKSEC, required=required)


def get_all_sections(header: Header, required: bool = False) -> dict[str, Section]:
    """
    Extract all standard section keywords from a header.

    Parameters
    ----------
    header :
        FITS header.
    required :
        If True, raise for missing sections; if False, skip them.

    Returns
    -------
    dict
        Mapping from section keyword (e.g. ``'DATASEC'``) to :class:`Section`.
    """
    sections: dict[str, Section] = {}
    for keyword in SECTION_KEYWORDS:
        section = get_section_by_keyword(header, keyword, required=required)
        if section is not None:
            sections[keyword] = section
    return sections


# =============================================================================
# Overscan Region Detection / Inference
# =============================================================================

# def _detect_active_region_from_image(
#     image: np.ndarray,
#     threshold_fraction: float = 0.1,
#     min_gradient_percentile: float = 90.0,
# ) -> Section | None:
#     """
#     Attempt to detect the active (illuminated) region from image content.

#     This is a last-resort fallback when neither DATASEC nor caller-provided
#     geometry is available. It works best on flat-field images with clear
#     contrast between the illuminated active area and the overscan.

#     Algorithm (simplified):
#         1. Compute median column and row profiles.
#         2. Compute absolute gradients of each profile.
#         3. Identify two strongest gradient peaks as boundaries of the
#            active region along each axis, with sanity checks.

#     Returns None rather than guessing when the structure is ambiguous.
#     """
#     if image.ndim != 2:
#         logger.warning("_detect_active_region_from_image: image is not 2D.")
#         return None

#     nrows, ncols = image.shape

#     try:
#         col_profile = np.median(image, axis=0)
#         row_profile = np.median(image, axis=1)

#         median_signal = np.median(image)
#         if median_signal <= 0:
#             logger.warning(
#                 "_detect_active_region_from_image: no positive signal detected."
#             )
#             return None

#         def _find_edges(profile: np.ndarray):
#             gradient = np.abs(np.diff(profile.astype(np.float64)))
#             if gradient.max() == 0:
#                 return None, None

#             grad_threshold = np.percentile(gradient, min_gradient_percentile)
#             peaks = np.where(gradient >= grad_threshold)[0]
#             if len(peaks) < 2:
#                 return None, None

#             start = int(peaks[0]) + 1
#             end = int(peaks[-1]) + 1

#             region_median = np.median(profile[start:end])
#             outside_left = np.median(profile[: max(start, 1)])
#             outside_right = np.median(profile[min(end, len(profile) - 1) :])

#             if region_median <= max(outside_left, outside_right) * (
#                 1.0 + threshold_fraction
#             ):
#                 return None, None

#             return start, end

#         col_start, col_end = _find_edges(col_profile)
#         row_start, row_end = _find_edges(row_profile)

#         if any(v is None for v in (col_start, col_end, row_start, row_end)):
#             logger.warning(
#                 "_detect_active_region_from_image: could not reliably detect "
#                 "active region boundaries."
#             )
#             return None

#         detected = Section(col_start, col_end, row_start, row_end)
#         logger.info(
#             "Auto-detected active region: %s (width=%d, height=%d)",
#             detected,
#             detected.width,
#             detected.height,
#         )
#         return detected

#     except Exception as e:
#         logger.warning("_detect_active_region_from_image: detection failed: %s", e)
#         return None


def _resolve_datasec(
    header: Header | None = None,
    datasec: Section | None = None,
    image: np.ndarray | None = None,
) -> Section:
    """
    Resolve the active data section from available sources.

    Resolution order:
        1. Caller-provided ``datasec`` (highest priority).
        2. ``DATASEC`` keyword from *header*.
        3. Auto-detection from *image* data.

    Raises
    ------
    ValueError
        If no active region can be determined.
    """
    if datasec is not None:
        logger.debug("Using caller-provided DATASEC: %s", datasec)
        return datasec

    if header is not None:
        header_datasec = get_datasec(header, required=False)
        if header_datasec is not None:
            logger.debug("Using DATASEC from header: %s", header_datasec)
            return header_datasec

    # Image-based auto-detection removed — unreliable on flat-field images
    # due to illumination rolloff near the active-area boundary.
    # if image is not None:
    #     detected = _detect_active_region_from_image(image)
    #     if detected is not None:
    #         logger.info("Using auto-detected active region: %s", detected)
    #         return detected

    raise ValueError(
        "Cannot determine active data region. Provide one of: "
        "datasec parameter, DATASEC header keyword, or image data for auto-detection."
    )


def get_serial_overscan(
    header: Header | None = None,
    image_shape: tuple[int, int] | None = None,
    datasec: Section | None = None,
    image: np.ndarray | None = None,
    keyword: str = FITS_KEY_BIASSEC,
    required: bool = False,
) -> Section | None:
    """
    Get the serial (horizontal) overscan region.

    Resolution order:
        1. Header keyword (default: ``BIASSEC``; configurable via ``keyword``).
        2. Inferred from geometry: columns beyond DATASEC on either side.

    If both sides have extra columns, the wider region is returned (typically
    the readout-side overscan).

    Raises
    ------
    HeaderKeywordError
        If ``required`` is True and overscan cannot be determined.
    """
    # Header keyword, if available
    if header is not None:
        keyword_sec = get_section_by_keyword(header, keyword, required=False)
        if keyword_sec is not None:
            logger.debug("Serial overscan from %s: %s", keyword, keyword_sec)
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
                keyword=keyword,
            )
        return None

    try:
        resolved_datasec = _resolve_datasec(header=header, datasec=datasec, image=image)
    except ValueError as exc:
        if required:
            raise HeaderKeywordError(
                "Cannot determine serial overscan: no {keyword} keyword "
                "and DATASEC could not be resolved for inference.",
                keyword=keyword,
            ) from exc
        return None

    nrows, ncols = effective_shape
    left_cols = resolved_datasec.start_col
    right_cols = ncols - resolved_datasec.end_col

    os_start_row = resolved_datasec.start_row
    os_end_row = resolved_datasec.end_row

    if right_cols > 0 and right_cols >= left_cols:
        result = Section(resolved_datasec.end_col, ncols, os_start_row, os_end_row)
        logger.debug("Serial overscan inferred (right): %s", result)
        return result
    if left_cols > 0:
        result = Section(0, resolved_datasec.start_col, os_start_row, os_end_row)
        logger.debug("Serial overscan inferred (left): %s", result)
        return result

    if required:
        raise HeaderKeywordError(
            f"Cannot determine serial overscan: no {keyword} keyword and "
            "no extra columns found outside DATASEC.",
            keyword=keyword,
        )
    return None


def get_parallel_overscan(
    header: Header | None = None,
    image_shape: tuple[int, int] | None = None,
    datasec: Section | None = None,
    image: np.ndarray | None = None,
    keyword: str | None = None,
    required: bool = False,
) -> Section | None:
    """
    Get the parallel (vertical) overscan region.

    Resolution order:
        1. Header keyword (if provided via ``keyword``).
        2. Inferred from geometry: rows beyond DATASEC above or below.

    If both sides have extra rows, the larger region is returned.

    Raises
    ------
    HeaderKeywordError
        If ``required`` is True and overscan cannot be determined.
    """
    # Header keyword, if requested
    if keyword is not None and header is not None:
        keyword_sec = get_section_by_keyword(header, keyword, required=False)
        if keyword_sec is not None:
            logger.debug("Parallel overscan from %s: %s", keyword, keyword_sec)
            return keyword_sec

    effective_shape = image_shape
    if effective_shape is None and image is not None:
        effective_shape = image.shape

    if effective_shape is None:
        if required:
            kw_msg = f"no {keyword} keyword, " if keyword else ""
            raise HeaderKeywordError(
                f"Cannot determine parallel overscan: {kw_msg}"
                "and no image_shape or image provided for inference.",
                keyword=keyword or "PARALLEL_OVERSCAN",
            )
        return None

    try:
        resolved_datasec = _resolve_datasec(header=header, datasec=datasec, image=image)
    except ValueError as exc:
        if required:
            kw_msg = f"no {keyword} keyword " if keyword else "no keyword configured "
            raise HeaderKeywordError(
                f"Cannot determine parallel overscan: {kw_msg}"
                "and DATASEC could not be resolved for inference.",
                keyword=keyword or "PARALLEL_OVERSCAN",
            ) from exc
        return None

    nrows, ncols = effective_shape
    top_rows = resolved_datasec.start_row
    bottom_rows = nrows - resolved_datasec.end_row

    os_start_col = resolved_datasec.start_col
    os_end_col = resolved_datasec.end_col

    if bottom_rows > 0 and bottom_rows >= top_rows:
        result = Section(os_start_col, os_end_col, resolved_datasec.end_row, nrows)
        logger.debug("Parallel overscan inferred (bottom): %s", result)
        return result
    if top_rows > 0:
        result = Section(os_start_col, os_end_col, 0, resolved_datasec.start_row)
        logger.debug("Parallel overscan inferred (top): %s", result)
        return result

    if required:
        kw_msg = f"no {keyword} keyword and " if keyword else ""
        raise HeaderKeywordError(
            f"Cannot determine parallel overscan: {kw_msg}"
            "no extra rows found outside DATASEC.",
            keyword=keyword or "PARALLEL_OVERSCAN",
        )
    return None


def infer_overscan_regions(
    header: Header | None = None,
    image_shape: tuple[int, int] | None = None,
    datasec: Section | None = None,
    image: np.ndarray | None = None,
    serial_keyword: str = FITS_KEY_BIASSEC,
    parallel_keyword: str | None = None,
) -> dict[str, Section]:
    """
    Determine available overscan regions from header and geometry.

    Returns a dictionary that may contain:

    - ``'datasec'``   : resolved active region
    - ``'serial'``    : serial overscan region
    - ``'parallel'``  : parallel overscan region

    Only regions that can be determined are included.
    """
    regions: dict[str, Section] = {}

    try:
        resolved_datasec = _resolve_datasec(header=header, datasec=datasec, image=image)
        regions["datasec"] = resolved_datasec
    except ValueError as e:
        logger.warning("Could not resolve DATASEC: %s", e)
        return regions

    serial = get_serial_overscan(
        header=header,
        image_shape=image_shape,
        datasec=resolved_datasec,
        image=image,
        keyword=serial_keyword,
        required=False,
    )
    if serial is not None:
        regions["serial"] = serial

    parallel = get_parallel_overscan(
        header=header,
        image_shape=image_shape,
        datasec=resolved_datasec,
        image=image,
        keyword=parallel_keyword,
        required=False,
    )
    if parallel is not None:
        regions["parallel"] = parallel

    found = [k for k in ("serial", "parallel") if k in regions]
    if found:
        logger.info("Overscan regions found: %s", ", ".join(found))
        for name in found:
            sec = regions[name]
            if name == "serial":
                logger.info("  %s: %s (width=%d)", name, sec, sec.width)
            else:
                logger.info("  %s: %s (height=%d)", name, sec, sec.height)
    else:
        logger.warning("No overscan regions could be determined.")

    return regions


# =============================================================================
# Regions of Interest
# =============================================================================

def resolve_roi(
    image_shape: tuple[int, int],
    custom_roi: Section | tuple[int, int, int, int] | None = None,
    header: Header | None = None,
) -> Section | None:
    """
    Resolve and validate a Region of Interest (ROI).

    Resolution order:
        1. custom_roi (user-provided)
        2. DATASEC from header
        3. None (use full frame)

    Raises ValueError if the resolved ROI exceeds the image_shape.
    """
    # 1. Resolve
    roi: Section | None = None

    if custom_roi is not None:
        if isinstance(custom_roi, Section):
            roi = custom_roi
        else:
            roi = Section(*custom_roi)
    elif header is not None:
        try:
            roi = get_datasec(header, required=False)
        except SectionParsingError as exc:
            logger.warning(
                "Ignoring invalid optional DATASEC and using full frame: %s",
                exc,
            )
            roi = None

    if roi is None:
        return None

    # 2. Validate using the built-in Section method
    if not roi.fits_within(image_shape):
        raise ValueError(
            f"Invalid ROI: {roi} exceeds image dimensions {image_shape}."
        )

    return roi

def create_roi_mask(image_shape: tuple[int, int], roi: Section) -> np.ndarray:
    """
    Create a 2D boolean mask where the Region of Interest is True.

    This is useful for applying Numpy masked arrays (np.ma.array)
    without changing the underlying coordinate system of the full frame.
    """
    if not roi.fits_within(image_shape):
        raise ValueError(f"ROI {roi} exceeds image shape {image_shape}.")

    mask = np.zeros(image_shape, dtype=bool)
    mask[roi.to_slices()] = True
    return mask

def extract_roi(data: np.ndarray, roi: Section) -> np.ndarray:
    """
    Extract the sub-array defined by the ROI from the data.

    Parameters
    ----------
    data : np.ndarray
        The 2D image array.
    roi : Section
        The region of interest to extract.

    Returns
    -------
    np.ndarray
        A view of the data array bounded by the ROI.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data array, got {data.ndim}D.")
    if not roi.fits_within(data.shape):
        raise ValueError(f"ROI {roi} cannot be extracted; it exceeds data shape {data.shape}.")

    return data[roi.to_slices()]

def expand_roi(roi: Section, margin: int, image_shape: tuple[int, int]) -> Section:
    """
    Expand (or shrink) an ROI by a uniform margin, constrained by image boundaries.

    Positive margins expand the ROI. Negative margins shrink it.

    Parameters
    ----------
    roi : Section
        The original region of interest.
    margin : int
        Number of pixels to add/subtract from all edges.
    image_shape : tuple[int, int]
        The (rows, cols) of the full image to constrain expansion.

    Returns
    -------
    Section
        A new Section object with the expanded coordinates.
    """
    rows, cols = image_shape

    start_col = max(0, roi.start_col - margin)
    end_col = min(cols, roi.end_col + margin)
    start_row = max(0, roi.start_row - margin)
    end_row = min(rows, roi.end_row + margin)

    # Check if a negative margin completely collapsed the ROI
    if start_col >= end_col or start_row >= end_row:
        raise ValueError(f"Margin {margin} collapsed ROI {roi} to zero or negative area.")

    return Section(start_col, end_col, start_row, end_row)


# =============================================================================
# Deprecated Utility Functions (Backward Compatibility)
# =============================================================================

def get_section_dimensions(
    section: Section | tuple[int, int, int, int]
) -> tuple[int, int]:
    """
    Calculate the dimensions (width, height) of a section.

    Deprecated
    ----------
    Use :attr:`Section.width` and :attr:`Section.height` instead.
    """
    if isinstance(section, Section):
        return section.width, section.height
    start_col, end_col, start_row, end_row = section
    return end_col - start_col, end_row - start_row


def get_section_area(section: Section | tuple[int, int, int, int]) -> int:
    """
    Calculate the area (pixels) of a section.

    Deprecated
    ----------
    Use :attr:`Section.area` instead.
    """
    if isinstance(section, Section):
        return section.area
    width, height = get_section_dimensions(section)
    return width * height


def sections_overlap(
    section1: Section | tuple[int, int, int, int],
    section2: Section | tuple[int, int, int, int],
) -> bool:
    """
    Check if two sections overlap.

    Deprecated
    ----------
    Use :meth:`Section.overlaps` instead.
    """
    if not isinstance(section1, Section):
        section1 = Section(*section1)
    if not isinstance(section2, Section):
        section2 = Section(*section2)
    return section1.overlaps(section2)


def section_contains_point(
    section: Section | tuple[int, int, int, int],
    col: int,
    row: int,
) -> bool:
    """
    Check if a point (col, row) is within a section.

    Deprecated
    ----------
    Use :meth:`Section.contains_point` instead.
    """
    if not isinstance(section, Section):
        section = Section(*section)
    return section.contains_point(col, row)


def section_to_fits_string(
    section: Section | tuple[int, int, int, int]
) -> str:
    """
    Convert Python section coordinates back to FITS format string.

    Deprecated
    ----------
    Use :meth:`Section.to_fits_string` instead.
    """
    if not isinstance(section, Section):
        section = Section(*section)
    return section.to_fits_string()


def section_to_slices(
    section: Section | tuple[int, int, int, int]
) -> tuple[slice, slice]:
    """
    Convert section coordinates to Python slice objects.

    Deprecated
    ----------
    Use :meth:`Section.to_slices` instead.
    """
    if not isinstance(section, Section):
        section = Section(*section)
    return section.to_slices()


def validate_section_in_image(
    section: Section | tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> bool:
    """
    Validate that a section fits within an image's dimensions.

    Deprecated
    ----------
    Use :meth:`Section.fits_within` instead.
    """
    if not isinstance(section, Section):
        start_col, end_col, start_row, end_row = section
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
    # Core type
    "Section",
    # Parsing
    "parse_section_string",
    "get_section_by_keyword",
    # Standard keywords
    "get_datasec",
    "get_biassec",
    "get_trimsec",
    "get_darksec",
    "get_all_sections",
    # Overscan helpers
    "get_serial_overscan",
    "get_parallel_overscan",
    "infer_overscan_regions",
    # Deprecated utilities
    "get_section_dimensions",
    "get_section_area",
    "sections_overlap",
    "section_contains_point",
    "section_to_fits_string",
    "section_to_slices",
    "validate_section_in_image",
]
