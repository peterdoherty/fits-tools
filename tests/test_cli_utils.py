#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import logging

import pytest

import fits_tools.cli_utils as cli_utils
from fits_tools.cli_utils import (
    derive_out_path,
    ensure_single_or_dir,
    is_tqdm_available,
    make_output_path,
    maybe_tqdm,
    setup_logger,
)
from fits_tools.exceptions import InvalidParameterError


def test_setup_logger_reuses_handlers_and_accepts_string_levels() -> None:
    logger = setup_logger("fits_tools.tests.cli", level="DEBUG")
    handler_count = len(logger.handlers)
    logger2 = setup_logger("fits_tools.tests.cli", level=logging.WARNING)

    assert logger is logger2
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == handler_count


def test_ensure_single_or_dir_validates_single_and_multiple_inputs() -> None:
    ensure_single_or_dir(["one.fits"], "out.fits", None)
    ensure_single_or_dir(["one.fits", "two.fits"], None, "outdir")

    with pytest.raises(InvalidParameterError):
        ensure_single_or_dir(["one.fits"], None, None)

    with pytest.raises(InvalidParameterError):
        ensure_single_or_dir(["one.fits", "two.fits"], "out.fits", None)


def test_make_output_path_preserves_fits_suffixes_and_can_force_extension(tmp_path) -> None:
    gz = make_output_path("image.fits.gz", output_dir=tmp_path, suffix="_proc")
    txt = make_output_path("notes.txt", output_dir=tmp_path, suffix="_copy")
    bare = make_output_path("raw", output_dir=tmp_path, suffix="_v1")

    assert gz == tmp_path / "image_proc.fits.gz"
    assert txt == tmp_path / "notes_copy.fits"
    assert bare == tmp_path / "raw_v1.fits"


def test_derive_out_path_handles_directory_and_explicit_file(tmp_path) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    from_dir = derive_out_path("input.fits", str(out_dir), None, "_crop")
    from_output_dir = derive_out_path("input.fits", None, str(out_dir), "_crop")
    explicit = derive_out_path("input.fits", str(tmp_path / "fixed.fits"), None, "_crop")

    assert from_dir.endswith("input_crop.fits")
    assert from_output_dir.endswith("input_crop.fits")
    assert explicit.endswith("fixed.fits")

    with pytest.raises(InvalidParameterError):
        derive_out_path("input.fits", None, None, "_crop")


def test_maybe_tqdm_and_availability_cover_present_and_absent_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    iterable = [1, 2, 3]

    monkeypatch.setattr(cli_utils, "_tqdm", None)
    assert maybe_tqdm(iterable) is iterable
    assert is_tqdm_available() is False

    monkeypatch.setattr(
        cli_utils,
        "_tqdm",
        lambda values, desc="Processing", **kwargs: ("wrapped", list(values), desc, kwargs),
    )
    wrapped = maybe_tqdm(iterable, desc="Work", total=3)

    assert wrapped[0] == "wrapped"
    assert wrapped[1] == iterable
    assert wrapped[2] == "Work"
    assert wrapped[3]["total"] == 3
    assert is_tqdm_available() is True
