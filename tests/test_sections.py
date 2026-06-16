#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.constants import FITS_KEY_BIASSEC, FITS_KEY_DATASEC
from fits_tools.exceptions import HeaderKeywordError, RegionError, SectionParsingError
from fits_tools.sections import (
    Section,
    create_roi_mask,
    expand_roi,
    extract_roi,
    get_all_sections,
    get_parallel_overscan,
    get_section_area,
    get_section_by_keyword,
    get_section_dimensions,
    get_serial_overscan,
    infer_overscan_regions,
    parse_section_string,
    resolve_roi,
    section_contains_point,
    section_to_fits_string,
    section_to_slices,
    sections_overlap,
    validate_section_in_image,
)
from fits_tools.io import region_to_slices


def test_parse_section_string_round_trips_to_fits_coordinates() -> None:
    section = parse_section_string("[2:5,2:4]")

    assert section == Section(1, 5, 1, 4)
    assert section.width == 4
    assert section.height == 3
    assert section.to_fits_string() == "[2:5,2:4]"


def test_parse_section_string_rejects_malformed_input() -> None:
    with pytest.raises(SectionParsingError, match="Malformed FITS section string"):
        parse_section_string("2:5,2:4")


def test_parse_section_string_rejects_zero_and_reversed_coordinates() -> None:
    with pytest.raises(SectionParsingError, match="must be >= 1"):
        parse_section_string("[0:5,1:4]")

    with pytest.raises(SectionParsingError, match="column start"):
        parse_section_string("[5:2,1:4]")

    with pytest.raises(SectionParsingError, match="row start"):
        parse_section_string("[1:4,5:2]")


def test_get_section_by_keyword_handles_missing_optional_key(sample_header: fits.Header) -> None:
    assert get_section_by_keyword(sample_header, "DARKSEC", required=False) is None


def test_get_section_by_keyword_raises_for_missing_required_key(sample_header: fits.Header) -> None:
    with pytest.raises(HeaderKeywordError, match="missing required keyword"):
        get_section_by_keyword(sample_header, "DARKSEC")


def test_region_to_slices_supports_keyword_tuple_and_section(sample_header: fits.Header) -> None:
    assert region_to_slices((4, 5), sample_header, "data") == (slice(1, 4), slice(1, 5))
    assert region_to_slices((4, 5), sample_header, (0, 2, 1, 3)) == (slice(1, 3), slice(0, 2))
    assert region_to_slices((4, 5), sample_header, Section(0, 1, 0, 4)) == (slice(0, 4), slice(0, 1))


def test_region_to_slices_rejects_out_of_bounds_region(sample_header: fits.Header) -> None:
    with pytest.raises(RegionError, match="exceeds data dimensions"):
        region_to_slices((4, 5), sample_header, (0, 6, 0, 2))


def test_infer_overscan_regions_uses_header_and_geometry(sample_header: fits.Header) -> None:
    regions = infer_overscan_regions(header=sample_header, image_shape=(4, 5))

    assert regions["datasec"] == get_section_by_keyword(sample_header, FITS_KEY_DATASEC)
    assert regions["serial"] == get_section_by_keyword(sample_header, FITS_KEY_BIASSEC)


def test_section_methods_and_deprecated_wrappers_cover_geometry_helpers() -> None:
    sec = Section(1, 4, 2, 6)
    other = Section(3, 5, 4, 7)

    assert repr(sec).startswith("Section(")
    assert str(sec) == "[2:4,3:6]"
    assert tuple(sec) == (1, 4, 2, 6)
    assert sec.to_tuple() == (1, 4, 2, 6)
    assert sec.shape == (4, 3)
    assert sec.contains_point(2, 3) is True
    assert sec.contains_point(0, 0) is False
    assert sec.overlaps(other) is True
    assert sec.fits_within((10, 10)) is True
    assert get_section_dimensions(sec) == (3, 4)
    assert get_section_area(sec) == 12
    assert sections_overlap(sec, other) is True
    assert section_contains_point(sec, 2, 3) is True
    assert section_to_fits_string(sec) == "[2:4,3:6]"
    assert section_to_slices(sec) == (slice(2, 6), slice(1, 4))
    assert validate_section_in_image(sec, (10, 10)) is True


def test_get_all_sections_and_overscan_helpers_infer_from_geometry() -> None:
    header = fits.Header({"DATASEC": "[2:5,2:5]"})

    sections = get_all_sections(header, required=False)
    serial = get_serial_overscan(header=header, image_shape=(6, 7), required=True)
    parallel = get_parallel_overscan(header=header, image_shape=(6, 7), required=True)

    assert sections["DATASEC"] == Section(1, 5, 1, 5)
    assert serial == Section(5, 7, 1, 5)
    assert parallel == Section(1, 5, 5, 6)


def test_overscan_helpers_cover_left_top_keyword_and_no_overscan_paths() -> None:
    left_top_datasec = Section(2, 6, 2, 5)
    header = fits.Header({"DATASEC": "[3:6,3:5]", "PARSEC": "[3:6,1:2]"})

    serial = get_serial_overscan(datasec=left_top_datasec, image_shape=(5, 6), required=True)
    parallel = get_parallel_overscan(datasec=left_top_datasec, image_shape=(5, 6), required=True)
    keyword_parallel = get_parallel_overscan(
        header=header,
        image_shape=(5, 6),
        keyword="PARSEC",
        required=True,
    )
    no_overscan = infer_overscan_regions(
        datasec=Section(0, 6, 0, 5),
        image_shape=(5, 6),
    )

    assert serial == Section(0, 2, 2, 5)
    assert parallel == Section(2, 6, 0, 2)
    assert keyword_parallel == Section(2, 6, 0, 2)
    assert no_overscan == {"datasec": Section(0, 6, 0, 5)}


def test_overscan_helpers_raise_clear_errors_when_inference_is_impossible() -> None:
    with pytest.raises(HeaderKeywordError, match="no BIASSEC keyword"):
        get_serial_overscan(required=True)

    with pytest.raises(HeaderKeywordError, match="DATASEC could not be resolved"):
        get_serial_overscan(image_shape=(4, 4), required=True)

    with pytest.raises(HeaderKeywordError, match="no extra columns found"):
        get_serial_overscan(datasec=Section(0, 4, 0, 4), image_shape=(4, 4), required=True)

    with pytest.raises(HeaderKeywordError, match="no TESTSEC keyword"):
        get_parallel_overscan(keyword="TESTSEC", required=True)

    with pytest.raises(HeaderKeywordError, match="DATASEC could not be resolved"):
        get_parallel_overscan(image_shape=(4, 4), keyword="TESTSEC", required=True)

    with pytest.raises(HeaderKeywordError, match="no extra rows found"):
        get_parallel_overscan(
            datasec=Section(0, 4, 0, 4),
            image_shape=(4, 4),
            keyword="TESTSEC",
            required=True,
        )


def test_resolve_roi_mask_extract_and_expand_cover_roi_utilities(sample_header: fits.Header) -> None:
    roi = resolve_roi((4, 5), header=sample_header)
    assert roi == Section(1, 5, 1, 4)

    mask = create_roi_mask((4, 5), roi)
    data = np.arange(20).reshape(4, 5)
    extracted = extract_roi(data, roi)
    expanded = expand_roi(roi, margin=1, image_shape=(4, 5))

    assert mask.sum() == roi.area
    assert np.array_equal(extracted, data[1:4, 1:5])
    assert expanded == Section(0, 5, 0, 4)


def test_roi_helpers_reject_invalid_bounds_and_collapsing_margins() -> None:
    roi = Section(1, 3, 1, 3)
    assert resolve_roi((2, 2)) is None

    with pytest.raises(ValueError, match="exceeds image dimensions"):
        resolve_roi((2, 2), custom_roi=roi)

    with pytest.raises(ValueError, match="exceeds image shape"):
        create_roi_mask((2, 2), roi)

    with pytest.raises(ValueError, match="Expected 2D data array"):
        extract_roi(np.zeros((2, 2, 2)), roi)

    with pytest.raises(ValueError, match="collapsed ROI"):
        expand_roi(roi, margin=-2, image_shape=(10, 10))


def test_resolve_roi_ignores_invalid_optional_header_datasec() -> None:
    header = fits.Header({"DATASEC": "NONE"})

    assert resolve_roi((4, 5), header=header) is None


def test_deprecated_section_wrappers_accept_tuples_and_validate_errors() -> None:
    assert get_section_dimensions((1, 4, 2, 6)) == (3, 4)
    assert get_section_area((1, 4, 2, 6)) == 12
    assert sections_overlap((0, 2, 0, 2), (1, 3, 1, 3)) is True
    assert section_contains_point((0, 2, 0, 2), 1, 1) is True
    assert section_to_fits_string((1, 4, 2, 6)) == "[2:4,3:6]"
    assert section_to_slices((1, 4, 2, 6)) == (slice(2, 6), slice(1, 4))

    with pytest.raises(ValueError, match="negative coordinates"):
        validate_section_in_image((-1, 2, 0, 2), (4, 4))

    with pytest.raises(ValueError, match="inverted coordinates"):
        validate_section_in_image((2, 2, 0, 2), (4, 4))
