from __future__ import annotations

from fits_tools import constants


def test_section_regex_and_file_extensions_are_consistent() -> None:
    match = constants.FITS_SECTION_REGEX.match("[1:5,2:7]")

    assert match is not None
    assert match.groups() == ("1", "5", "2", "7")
    assert ".fits" in constants.FITS_FILE_EXTENSIONS
    assert ".fits.gz" in constants.FITS_FILE_EXTENSIONS


def test_keyword_groups_include_expected_members_without_duplicates() -> None:
    assert constants.FITS_KEY_DATASEC in constants.SECTION_KEYWORDS
    assert constants.FITS_KEY_CCD_TEMP in constants.TEMPERATURE_KEYWORDS
    assert constants.FITS_KEY_GAIN in constants.DETECTOR_CHAR_KEYWORDS
    assert constants.FITS_KEY_IMGTYPE in constants.IMAGE_TYPE_KEYWORDS
    assert len(constants.SECTION_KEYWORDS) == len(set(constants.SECTION_KEYWORDS))
    assert len(constants.ALL_IMAGE_TYPES) == len(set(constants.ALL_IMAGE_TYPES))


def test_dtype_mappings_and_limits_are_self_consistent() -> None:
    for np_name, bitpix in constants.NUMPY_TO_FITS_DTYPE.items():
        assert constants.FITS_TO_NUMPY_DTYPE[bitpix] == np_name

    assert constants.MIN_EXPTIME <= constants.MAX_EXPTIME
    assert constants.MIN_GAIN < constants.DEFAULT_GAIN < constants.MAX_GAIN
    assert constants.MIN_READNOISE < constants.DEFAULT_READNOISE < constants.MAX_READNOISE
    assert constants.MIN_BINNING < constants.MAX_BINNING
