
import logging
from pathlib import Path
from typing import  List, Optional, Union
from collections.abc import Iterable
import numpy as np
from astropy.io import fits

from .exceptions import (
    InvalidParameterError,
)
from .constants import FITS_FILE_EXTENSIONS, is_fits_file  
from .types import StrOrPath

# Optional dependencies
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


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
    hdu = input_path.suffix
    
    # Handle compressed FITS (.fits.gz, .fits.fz)
    if stem.endswith('.fits'):
        stem = stem[:-5]
        ext = '.fits' + ext
    
    new_name = f"{stem}{suffix}{ext}"
    result = out_dir / new_name
    
    if ensure_fits and not is_fits_file(result):
        result = result.with_suffix('.fits')
    
    return result

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



