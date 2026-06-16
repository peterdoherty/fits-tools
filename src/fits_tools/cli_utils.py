#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""CLI-focused helpers for fits_tools.

This module provides small utilities intended primarily for command-line entry
points, including:

- consistent logger setup
- output path derivation (single-file vs multi-file semantics)
- optional progress bars via tqdm
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from .exceptions import InvalidParameterError
from .types import StrOrPath
from .utils import is_fits_file

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

_T = TypeVar("_T")

__all__ = [
    "setup_logger",
    "ensure_single_or_dir",
    "make_output_path",
    "derive_out_path",
    "maybe_tqdm",
    "is_tqdm_available",
]


# =============================================================================
# Logging utilities
# =============================================================================


def setup_logger(
    name: str | None = None,
    level: str | int = "INFO",
    format_string: str | None = None,
) -> logging.Logger:
    """Configure and return a logger with consistent formatting.

    This is intended for CLI entry points. Library code should generally not
    configure logging handlers globally.

    Parameters
    ----------
    name
        Logger name. If None, returns the root logger.
    level
        Logging level as a string (e.g. ``"INFO"``) or an integer constant from
        :mod:`logging`.
    format_string
        Logging formatter string. If None, a simple default format is used.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    if format_string is None:
        format_string = "%(levelname)s: %(message)s"

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    log = logging.getLogger(name) if name else logging.getLogger()

    # Only attach a handler if the logger is not already configured.
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_string))
        log.addHandler(handler)

    log.setLevel(level)
    return log


# =============================================================================
# Output-path helpers
# =============================================================================


def ensure_single_or_dir(
    inputs: list[str],
    output_path: str | None,
    output_dir: str | None,
) -> None:
    """Validate output path specification for single vs multiple input files.

    CLI rule:
    - For a single input file: either ``output_path`` or ``output_dir`` must be provided.
    - For multiple input files: ``output_dir`` must be provided.

    Parameters
    ----------
    inputs
        Input FITS file paths.
    output_path
        Output file path for a single input (or a directory; see
        :func:`derive_out_path`).
    output_dir
        Output directory for one-output-per-input workflows.

    Raises
    ------
    InvalidParameterError
        If the combination is inconsistent.
    """
    if len(inputs) == 1 and (output_path or output_dir):
        return
    if len(inputs) > 1 and output_dir:
        return

    raise InvalidParameterError(
        "Invalid output specification: for multiple inputs, provide --output-dir; "
        "for a single input, provide --output or --output-dir.",
        parameter="output_path/output_dir",
        value=(output_path, output_dir),
        expected="single input: output_path or output_dir; multiple inputs: output_dir",
    )


def _split_name_and_fits_extension(path: Path) -> tuple[str, str]:
    """Split a filename into (stem, extension) preserving multi-suffix FITS names.

    Examples
    --------
    - ``image.fits``    -> (``image``, ``.fits``)
    - ``image.fits.gz`` -> (``image``, ``.fits.gz``)
    - ``image``         -> (``image``, ````)
    """
    suffixes = path.suffixes
    if not suffixes:
        return path.name, ""

    ext = "".join(suffixes)
    stem = path.name[: -len(ext)]
    return stem, ext


def make_output_path(
    input_path: StrOrPath,
    output_dir: StrOrPath | None = None,
    suffix: str = "",
    ensure_fits_ext: bool = True,
) -> Path:
    """Construct an output path from an input path.

    The output filename is derived from the input filename by inserting
    ``suffix`` before the extension. For FITS filenames, multi-suffix extensions
    (e.g. ``.fits.gz``) are preserved.

    Parameters
    ----------
    input_path
        Input file path.
    output_dir
        Output directory. If None, uses the input file's parent directory.
    suffix
        Suffix to insert before the extension (e.g. ``"_crop"``).
    ensure_fits_ext
        If True, and the derived output path does not look like a FITS filename,
        ``.fits`` is appended.

    Returns
    -------
    pathlib.Path
        Derived output path.
    """
    input_path = Path(input_path)

    if output_dir is None:
        out_dir = input_path.parent
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    stem, ext = _split_name_and_fits_extension(input_path)

    # If the input isn't a FITS file, decide what to do with its extension.
    # For CLI workflows, it is typically safer to force .fits when requested.
    if ensure_fits_ext and not is_fits_file(input_path):
        ext = ".fits" if not ext else ext

    out_name = f"{stem}{suffix}{ext}"
    result = out_dir / out_name

    if ensure_fits_ext and not is_fits_file(result):
        result = result.with_suffix(".fits")

    return result


def derive_out_path(
    in_path: str,
    output_path: str | None,
    output_dir: str | None,
    suffix: str,
) -> str:
    """Derive an output file path for a single input image.

    Parameters
    ----------
    in_path
        Input FITS file path.
    output_path
        User-specified output path (file or directory). Required if
        ``output_dir`` is not provided.
    output_dir
        Output directory for derived filenames.
    suffix
        Suffix to apply when constructing output filenames (e.g. ``"_crop"``).

    Returns
    -------
    str
        Resolved output file path.

    Raises
    ------
    InvalidParameterError
        If neither ``output_dir`` nor ``output_path`` is provided.
    """
    if output_dir:
        return str(make_output_path(in_path, output_dir=output_dir, suffix=suffix))

    if output_path is None:
        raise InvalidParameterError(
            "output_path must be provided when output_dir is not set.",
            parameter="output_path",
            value=None,
            expected="str (file path) or existing directory",
        )

    out = Path(output_path)
    if out.exists() and out.is_dir():
        return str(make_output_path(in_path, output_dir=out, suffix=suffix))

    return str(out)


# =============================================================================
# Progress helpers
# =============================================================================


def maybe_tqdm(iterable: Iterable[_T], desc: str = "Processing", **kwargs: object) -> Iterable[_T]:
    """Wrap an iterable with tqdm if available.

    Parameters
    ----------
    iterable
        Iterable to wrap.
    desc
        Progress bar description.
    **kwargs
        Forwarded to :class:`tqdm.tqdm`.

    Returns
    -------
    iterable
        tqdm-wrapped iterable if tqdm is available; otherwise the original iterable.
    """
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, **kwargs)


def is_tqdm_available() -> bool:
    """Return True if tqdm is importable."""
    return _tqdm is not None
    