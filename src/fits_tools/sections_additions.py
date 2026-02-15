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
