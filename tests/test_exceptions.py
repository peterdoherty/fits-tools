#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import pytest

from fits_tools.exceptions import (
    DataShapeError,
    DimensionalityError,
    FileHandlingError,
    FileOverwriteError,
    FitsToolsError,
    HeaderKeywordError,
    HDUStructureError,
    InvalidHDUIndexError,
    InvalidParameterError,
    MasterBiasCreationError,
    MasterDarkCreationError,
    MasterFlatCreationError,
    NoInputFilesFoundError,
    NormalizationError,
    RegionError,
    SectionParsingError,
    SelectionError,
    UnsupportedOperationError,
    ValidationError,
    WCSError,
    fits_operation_context,
    get_exception_hierarchy,
)


def test_from_exception_preserves_cause_and_supports_subclasses() -> None:
    original = ValueError("nope")
    wrapped = FileHandlingError.from_exception(original, filepath="a.fits")

    assert isinstance(wrapped, FileHandlingError)
    assert wrapped.__cause__ is original
    assert "nope" in str(wrapped)


def test_exception_reprs_include_structured_fields() -> None:
    assert "filepath" in repr(FileOverwriteError("x.fits"))
    assert "hdu_index" in repr(InvalidHDUIndexError("x.fits", 3, 2))
    assert "issue" in repr(HDUStructureError("bad", filepath="x.fits", issue="missing_primary"))
    assert "expected_shape" in repr(DataShapeError("bad", expected_shape=(2, 2), actual_shape=(3, 3)))
    assert "required_ndim" in repr(DimensionalityError("bad", ndim=3, required_ndim=2, operation="stack"))
    assert "operation" in repr(RegionError("bad", region=(1, 2), image_shape=(10, 10), operation="crop"))
    assert "keyword='GAIN'" in repr(HeaderKeywordError("bad", keyword="GAIN", filepath="x.fits"))
    assert "section_string='[1:2,3:4]'" in repr(SectionParsingError("bad", "[1:2,3:4]", keyword="BIASSEC"))
    assert "validation_type" in repr(ValidationError("bad", filepath="x.fits", validation_type="schema", errors=["a"]))
    assert "parameter='sigma'" in repr(InvalidParameterError("bad", parameter="sigma", value=-1, expected="> 0"))
    assert "reason" in repr(UnsupportedOperationError("bad", operation="warp", reason="missing wcs"))
    assert "mean_value" in repr(NormalizationError(mean_value=0.0))
    assert "wcs_operation" in repr(WCSError("bad", filepath="x.fits", wcs_operation="solve"))


def test_default_master_frame_exceptions_have_expected_messages() -> None:
    assert "BIAS" in repr(MasterBiasCreationError())
    assert "DARK" in repr(MasterDarkCreationError())
    assert "FLAT" in repr(MasterFlatCreationError())
    assert "search_param" in repr(NoInputFilesFoundError("BIAS", "*.fits"))


def test_fits_operation_context_passthrough_and_wrapping() -> None:
    with pytest.raises(InvalidParameterError):
        with fits_operation_context("testing", reraise_as=FitsToolsError):
            raise InvalidParameterError("already fits")

    with pytest.raises(FileHandlingError) as excinfo:
        with fits_operation_context("read", filepath="a.fits", reraise_as=FileHandlingError):
            raise RuntimeError("boom")

    assert "Error during read on 'a.fits'" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


def test_fits_operation_context_falls_back_when_constructor_signature_is_incompatible() -> None:
    with pytest.raises(MasterBiasCreationError) as excinfo:
        with fits_operation_context("combine", filepath="a.fits", reraise_as=MasterBiasCreationError):
            raise RuntimeError("bad stack")

    assert "Error during combine on 'a.fits'" in str(excinfo.value)


def test_from_exception_and_context_cover_additional_constructor_paths() -> None:
    class OnlyMessage(FitsToolsError):
        def __init__(self, message: str) -> None:
            super().__init__(message)

    wrapped = OnlyMessage.from_exception(ValueError("boom"), filepath="ignored.fits")
    assert isinstance(wrapped, OnlyMessage)
    assert wrapped.__cause__ is not None

    with pytest.raises(WCSError) as excinfo:
        with fits_operation_context("solve", filepath="wcs.fits", reraise_as=WCSError):
            raise RuntimeError("bad wcs")

    assert excinfo.value.filepath == Path("wcs.fits")
    assert excinfo.value.wcs_operation == "solve"

    with pytest.raises(OnlyMessage) as fallback_info:
        with fits_operation_context("open", filepath="demo.fits", reraise_as=OnlyMessage):
            raise RuntimeError("invalid")

    assert "Error during open on 'demo.fits'" in str(fallback_info.value)


def test_exception_hierarchy_string_mentions_key_classes() -> None:
    hierarchy = get_exception_hierarchy()

    assert "FitsToolsError (base)" in hierarchy
    assert "InvalidParameterError" in hierarchy
    assert "NormalizationError" in hierarchy
    assert isinstance(SelectionError("bad"), FitsToolsError)
