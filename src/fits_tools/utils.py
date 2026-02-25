#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for FITS data processing.

This module provides common utilities used across the fits_tools package,
including:
    - Data type conversion and JSON serialization
    - File path manipulation and validation
    - Array and index validation
    - Logging configuration
    - Common FITS operations

Version History:
    0.1.0 - Initial release with JSON preparation utility
    0.2.0 - Added file utilities, validation, logging, and FITS helpers
            consolidated from other modules
"""

import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
from collections.abc import Iterable
import numpy as np
from astropy.io import fits

from .exceptions import (
    InvalidParameterError,
    FileOverwriteError,
    FileHandlingError,
)
from .constants import FITS_FILE_EXTENSIONS, is_fits_file  
from .types import StrOrPath

# Optional dependencies
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

__version__ = "0.2.0"

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Logging Utilities
# =============================================================================

def setup_logger(
    name: Optional[str] = None,
    level: Union[str, int] = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.
    
    Args:
        name: Logger name. If None, returns root logger.
        level: Logging level (string like "INFO" or logging constant).
        format_string: Custom format string. If None, uses default.
    
    Returns:
        Configured logger instance.
    
    Examples:
        >>> logger = setup_logger(__name__, "DEBUG")
        >>> logger.debug("Debug message")
        
        >>> logger = setup_logger(level=logging.WARNING)
        >>> logger.warning("Warning message")
    """
    if format_string is None:
        format_string = "%(levelname)s: %(message)s"
    
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Get or create logger
    log = logging.getLogger(name) if name else logging.getLogger()
    
    # Only configure if not already configured
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_string))
        log.addHandler(handler)
    
    log.setLevel(level)
    return log


def maybe_tqdm(iterable: Iterable, desc: str = "Processing", **kwargs) -> Iterable:
    """
    Wrap iterable with tqdm progress bar if available.
    
    Parameters
    ----------
    iterable : iterable
        Any iterable to wrap.
    desc : str, optional
        Description for the progress bar. Defaults to "Processing".
    **kwargs
        Additional arguments passed to tqdm.
    
    Returns
    -------
    iterable
        tqdm-wrapped iterable if tqdm is available, otherwise the original iterable.
    """
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, **kwargs)
    return iterable

def is_tqdm_available() -> bool:
    """Check if tqdm is available for progress bars."""
    return _tqdm is not None


# =============================================================================
# Data Type Conversion
# =============================================================================

def prepare_for_json(data: Any) -> Any:
    """
    Recursively convert numpy types to standard Python types for JSON serialization.
    
    Handles dictionaries, lists, numpy arrays, and scalar numpy types.
    
    Args:
        data: Data structure (dict, list, array, or scalar) potentially
              containing numpy types.
    
    Returns:
        Data structure with numpy types converted to Python native types.
    
    Examples:
        >>> import numpy as np
        >>> data = {"mean": np.float32(1.5), "counts": np.int32(100)}
        >>> result = prepare_for_json(data)
        >>> isinstance(result["mean"], float)
        True
        >>> isinstance(result["counts"], int)
        True
        
        >>> arr = np.array([1, 2, 3])
        >>> prepare_for_json(arr)
        [1, 2, 3]
    """
    if isinstance(data, dict):
        return {k: prepare_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [prepare_for_json(elem) for elem in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.complexfloating):
        return complex(data)
    else:
        return data


def ensure_native_types(value: Any) -> Any:
    """
    Convert a single value from numpy type to Python native type.
    
    Simpler alternative to prepare_for_json for scalar values.
    
    Args:
        value: Value to convert.
    
    Returns:
        Python native type equivalent.
    
    Examples:
        >>> import numpy as np
        >>> ensure_native_types(np.int32(42))
        42
        >>> ensure_native_types(np.float32(3.14))
        3.14
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.complexfloating):
        return complex(value)
    return value


# =============================================================================
# File Path Utilities
# =============================================================================


def normalize_path(path: StrOrPath) -> Path:
    """
    Normalize a file path to absolute Path object.
    
    Expands user home directory (~) and resolves relative paths.
    
    Args:
        path: Input path (string or Path).
    
    Returns:
        Absolute Path object.
    
    Examples:
        >>> p = normalize_path("~/data/image.fits")
        >>> p.is_absolute()
        True
        >>> p = normalize_path("./image.fits")
        >>> p.is_absolute()
        True
    """
    return Path(path).expanduser().resolve()


def ensure_extension(filepath: StrOrPath, default_ext: str = ".fits") -> Path:
    """
    Ensure a filepath has a FITS extension, adding default if missing.
    
    Args:
        filepath: Input path.
        default_ext: Extension to add if none present (default: ".fits").
    
    Returns:
        Path with FITS extension.
    
    Examples:
        >>> ensure_extension("output")
        PosixPath('output.fits')
        >>> ensure_extension("output.fit")
        PosixPath('output.fit')
    """
    path = Path(filepath)
    if not is_fits_file(path):
        path = path.with_suffix(default_ext)
    return path


def make_output_path(
    input_path: StrOrPath,
    output_dir: Optional[StrOrPath] = None,
    suffix: str = "",
    ensure_fits: bool = True
) -> Path:
    """
    Generate output path from input path with optional suffix and directory.
    
    Args:
        input_path: Input file path.
        output_dir: Output directory. If None, uses input file's directory.
        suffix: Suffix to add before extension (e.g., "_crop").
        ensure_fits: If True, ensures output has FITS extension.
    
    Returns:
        Output path.
    
    Examples:
        >>> make_output_path("data/image.fits", suffix="_crop")
        PosixPath('data/image_crop.fits')
        >>> make_output_path("image.fits", output_dir="output", suffix="_proc")
        PosixPath('output/image_proc.fits')
    """
    input_path = Path(input_path)
    
    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = input_path.parent
    
    # Build filename with suffix
    stem = input_path.stem
    ext = input_path.suffix
    
    # Handle compressed FITS (.fits.gz, .fits.fz)
    if stem.endswith('.fits'):
        stem = stem[:-5]
        ext = '.fits' + ext
    
    new_name = f"{stem}{suffix}{ext}"
    result = out_dir / new_name
    
    if ensure_fits and not is_fits_file(result):
        result = result.with_suffix('.fits')
    
    return result



# =============================================================================
# FITS Utilities
# =============================================================================

def add_history(header, msg: str) -> None:
    """Add HISTORY card to FITS header with fallback."""
    try:
        header.add_history(msg)
    except Exception:
        header["HISTORY"] = msg

def add_comment(header, msg: str) -> None:
    """Add COMMENT card to FITS header with fallback."""
    try:
        header.add_comment(msg)
    except Exception:
        header["COMMENT"] = msg


def get_hdu_count(filepath: StrOrPath) -> int:
    """
    Get the number of HDUs in a FITS file without loading data.
    
    Args:
        filepath: Path to FITS file.
    
    Returns:
        Number of HDUs.
    
    Examples:
        >>> # count = get_hdu_count("image.fits")
        >>> # print(f"File has {count} HDUs")
    """
    with fits.open(filepath, memmap=True) as hdul:
        return len(hdul)


def get_image_shape(
    filepath: StrOrPath,
    hdu_index: int = 0
) -> Optional[Tuple[int, ...]]:
    """
    Get the shape of an image HDU without loading the data.
    
    Args:
        filepath: Path to FITS file.
        hdu_index: HDU index to check (default 0).
    
    Returns:
        Shape tuple or None if HDU has no data.
    
    Examples:
        >>> # shape = get_image_shape("image.fits")
        >>> # print(f"Image shape: {shape}")
    """
    with fits.open(filepath, memmap=True) as hdul:
        if hdul[hdu_index].data is None:
            return None
        return hdul[hdu_index].data.shape


def copy_header(source: fits.Header, exclude_keys: Optional[List[str]] = None) -> fits.Header:
    """
    Create a copy of a FITS header, optionally excluding specific keywords.
    
    Args:
        source: Source header to copy.
        exclude_keys: Keywords to exclude from copy.
    
    Returns:
        Copied header.
    
    Examples:
        >>> from astropy.io import fits
        >>> hdr = fits.Header()
        >>> hdr['EXPTIME'] = 10.0
        >>> hdr['OBJECT'] = 'M31'
        >>> new_hdr = copy_header(hdr, exclude_keys=['OBJECT'])
        >>> 'EXPTIME' in new_hdr
        True
        >>> 'OBJECT' in new_hdr
        False
    """
    new_header = source.copy()
    if exclude_keys:
        for key in exclude_keys:
            new_header.remove(key, ignore_missing=True)
    return new_header

def ensure_single_or_dir(inputs: List[str], output_path: Optional[str], output_dir: Optional[str]) -> None:
    """
    Validate output path specification for single vs multiple input files.

    This helper enforces a simple rule used by CLI entry points:

    - If there is exactly **one** input file, either ``output_path`` or
      ``output_dir`` must be provided.
    - If there are **multiple** input files, ``output_dir`` must be provided
      (``output_path`` is not sufficient / not allowed).

    Parameters
    ----------
    inputs : list of str
        List of input FITS file paths.
    output_path : str or None
        Explicit output file path for a single input, or an existing directory
        (in which case the output filename is derived from the input). Ignored
        when multiple inputs are given.
    output_dir : str or None
        Output directory that will receive one output file per input.

    Raises
    ------
    InvalidParameterError
        If the combination of ``inputs``, ``output_path`` and ``output_dir``
        is inconsistent (e.g. multiple inputs with only ``output_path``).
    """
    if len(inputs) == 1 and output_path:
        return
    if output_dir:
        return
    raise InvalidParameterError(
        "For multiple inputs, provide --output-dir; for a single input, provide --output or --output-dir.",
        parameter="output_path/output_dir"
    )

def derive_out_path(in_path: str, output_path: Optional[str], output_dir: Optional[str], suffix: str) -> str:
    """
    Derive an output file path for a single input image.

    This helper applies the CLI semantics:

    - If ``output_dir`` is provided, the output path is derived from
      ``in_path`` placed inside ``output_dir`` with the provided ``suffix``.
    - Otherwise, ``output_path`` must be provided and is interpreted as:
      * A directory: use its basename plus the original filename.
      * A file path: use it exactly.

    Parameters
    ----------
    in_path : str
        Input FITS file path.
    output_path : str or None
        User-specified output path (file or directory). Must be non-None if
        ``output_dir`` is not provided.
    output_dir : str or None
        Output directory for derived filenames.
    suffix : str
        Suffix to apply when constructing output filenames inside
        ``output_dir`` (e.g. ``'_crop'``, ``'_mask'``).

    Returns
    -------
    str
        Resolved output file path.
    """
    if output_dir:
        return str(make_output_path(in_path, output_dir=output_dir, suffix=suffix))
    
    assert output_path is not None
    if Path(output_path).is_dir():
        return str(Path(output_path) / Path(in_path).name)
    return output_path

def safe_write(hdul: fits.HDUList, 
               path: StrOrPath,
               overwrite: bool, 
               dry_run: bool, 
               operation: str = "edit") -> None:
    """
    Safely write an HDUList to disk with atomic replace and HISTORY update.

    This is a conservative writer intended for CLI-style tools where failures
    must never corrupt an existing file:

    - If ``dry_run`` is True, nothing is written; a log message is emitted.
    - If ``overwrite`` is False and ``path`` exists, a :class:`FileOverwriteError`
      is raised.
    - A HISTORY line documenting the operation and UTC timestamp is added to
      the primary header when possible.
    - The file is first written to a temporary file in the destination
      directory, then atomically moved into place using :func:`os.replace`.
      If writing fails, the temporary file is removed and the original target
      file is left untouched.

    Parameters
    ----------
    hdul : fits.HDUList
        HDU list to write.
    path : str
        Destination file path.
    overwrite : bool
        Whether to allow overwriting an existing file.
    dry_run : bool
        If True, do not actually write anything to disk.
    operation : str, optional
        Short operation name recorded in the HISTORY card, e.g. ``'crop'``,
        ``'mask'``. Defaults to ``'edit'``.

    Raises
    ------
    FileOverwriteError
        If the destination exists and ``overwrite`` is False.
    FileHandlingError
        If writing the file (or renaming the temporary file) fails.
    """
    if dry_run:
        logger.info(f"[dry-run] Would write: {path}")
        return

    target_path = Path(path)

    if target_path.exists() and not overwrite:
        raise FileOverwriteError(path)

    # Add HISTORY card to document the operation
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    history_msg = f"fits_tools: {operation} at {timestamp} UTC"

    try:
        hdr = hdul[0].header  # will raise if hdul empty, etc.
        add_history(hdr, history_msg)
    except Exception as e:
        logger.debug(f"Could not add HISTORY card: {e}")

    # Atomic write: write to a temp file in the destination directory, then replace.
    dest_dir = target_path.resolve().parent
    base = target_path.name
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{base}.tmp.",
            dir=dest_dir,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        # Always overwrite temp path
        hdul.writeto(tmp_path, overwrite=True, checksum=True)
        tmp_path.replace(target_path)
    except Exception as e:
        # Clean up temp file if present; do NOT delete existing destination.
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise FileHandlingError(f"Failed to write {path}: {e}", filepath=path) from e
    

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Logging
    'setup_logger',
    
    # Data conversion
    'prepare_for_json',
    'ensure_native_types',
    
    # File utilities
    'is_fits_file',
    'normalize_path',
    'ensure_extension',
    'make_output_path',
    
    # Validation
    'validate_indices',
    'validate_shape',
    'validate_shapes',
    'validate_positive',
    
    # FITS utilities
    'add_history',
    'add_comment',
    'get_hdu_count',
    'get_image_shape',
    'copy_header',
    'ensure_single_or_dir',
    'derive_out_path',
    'safe_write',
    
]

if __name__ == '__main__':
    sys.exit()
