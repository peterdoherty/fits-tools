#!/usr/bin/env python3
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Noise addition and injection for FITS files.

This module provides batch noise addition capabilities for testing and
simulation workflows, building on the noise models in simulations.py.

Features:
    - Batch processing with multiprocessing
    - Uses validated noise models from simulations module
    - Region-specific noise addition
    - Reproducible with seed parameter
    - Integration with fits_tools infrastructure

Version History:
    0.1.0 - Initial standalone version
    0.2.0 - Refactored to use fits_tools infrastructure, integrated with
            simulations.py noise models
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ..array_validation import validate_numeric
from ..exceptions import (
    FileHandlingError,
    ImageProcessingError,
)

# Use refactored package infrastructure
from ..io import get_data, get_header, write_fits
from ..selection import select_files
from ..types import StrOrPath
from .core import (
    add_photon_noise,  # Poisson noise
    add_read_noise,  # Gaussian noise
)

logger = logging.getLogger(__name__)

# ============================================================================
# Noise Operation Wrappers
# ============================================================================

def add_gaussian_noise(
    input_file: StrOrPath,
    output: str,
    sigma: float = 1.0,
    region: str | tuple[int, int, int, int] = 'full',
    hdu: int = 0,
    seed: int | None = None,
) -> None:
    """
    Add Gaussian noise to a FITS file.

    This is a convenience wrapper around simulations.add_read_noise()
    that handles file I/O.

    Parameters
    ----------
    input_file : str
        Input FITS file path.
    output : str
        Output FITS file path.
    sigma : float, optional
        Standard deviation of Gaussian noise. Defaults to 1.0.
    region : str or tuple, optional
        Region to add noise to. Defaults to 'full'.
    ext : int, optional
        HDU extension. Defaults to 0.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create a test file
    ...     from .simulations import create_bias_frame
    ...     infile = os.path.join(tmpdir, 'input.fits')
    ...     outfile = os.path.join(tmpdir, 'output.fits')
    ...     create_bias_frame(infile, shape=(10, 10), bias_level=100, read_noise=0)
    ...
    ...     # Add noise
    ...     add_gaussian_noise(infile, outfile, sigma=5.0, seed=42)
    ...
    ...     # Check it was created
    ...     print(os.path.exists(outfile))
    True
    """
    # Validate parameters
    validate_numeric(sigma, 'sigma', minimum=0.0, allow_zero=False)

    try:
        # Read data and header
        data = get_data(input_file, hdu=hdu, region=region)
        header = get_header(input_file, hdu=hdu)

        if data is None:
            raise ImageProcessingError(f"No data in {input_file} extension {hdu}")

        # Add noise using simulations module
        noisy_data = add_read_noise(data, sigma, seed=seed)

        # Update header
        header.add_history(f'Added Gaussian noise (sigma={sigma})')
        if seed is not None:
            header['NOISEED'] = (seed, 'Random seed for noise')

        # Write output
        write_fits(output, noisy_data, header=header, overwrite=True)

        logger.debug(f"Added Gaussian noise (σ={sigma}) to {input_file} -> {output}")

    except Exception as e:
        logger.error(f"Error adding noise to {input_file}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to add Gaussian noise: {e}") from e


def add_poisson_noise(
    input_file: StrOrPath,
    output: StrOrPath,
    gain: float = 1.0,
    region: str | tuple[int, int, int, int] = 'full',
    hdu: int = 0,
    seed: int | None = None,
) -> None:
    """
    Add Poisson photon noise to a FITS file.

    Wrapper around simulations.add_photon_noise().

    Parameters
    ----------
    input_file : str
        Input FITS file path.
    output : str
        Output FITS file path.
    gain : float, optional
        Gain in electrons/ADU. Defaults to 1.0.
    region : str or tuple, optional
        Region to add noise to. Defaults to 'full'.
    ext : int, optional
        HDU extension. Defaults to 0.
    seed : int, optional
        Random seed for reproducibility.
    """
    validate_numeric(gain, 'gain', minimum=0.0, allow_zero=False)

    try:
        data = get_data(input_file, hdu=hdu, region=region)
        header = get_header(input_file, hdu=hdu)

        if data is None:
            raise ImageProcessingError(f"No data in {input_file} extension {hdu}")

        noisy_data = add_photon_noise(data, gain=gain, seed=seed)

        header.add_history(f'Added Poisson noise (gain={gain})')
        if seed is not None:
            header['NOISEED'] = (seed, 'Random seed for noise')

        write_fits(output, noisy_data, header=header, overwrite=True)

        logger.debug(f"Added Poisson noise (gain={gain}) to {input_file} -> {output}")

    except Exception as e:
        logger.error(f"Error adding noise to {input_file}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to add Poisson noise: {e}") from e


# ============================================================================
# Batch Processing
# ============================================================================

def add_noise_batch(
    input_files: str | list[str],
    output_pattern: str,
    noise_type: str = 'gaussian',
    sigma: float = 1.0,
    gain: float = 1.0,
    num_workers: int = 1,
    seed: int | None = None,
    **kwargs
) -> list[str]:
    """
    Add noise to multiple FITS files in batch mode.

    Parameters
    ----------
    input_files : str or list
        Directory, glob pattern, or list of files.
    output_pattern : str
        Output naming pattern. Use {stem} for input filename stem.
        Example: "noisy_{stem}.fits"
    noise_type : str, optional
        Type of noise: 'gaussian', 'poisson'. Defaults to 'gaussian'.
    sigma : float, optional
        Sigma for Gaussian noise. Defaults to 1.0.
    gain : float, optional
        Gain for Poisson noise. Defaults to 1.0.
    num_workers : int, optional
        Number of parallel workers. Defaults to 1.
    seed : int, optional
        Base random seed. Each file gets seed + file_index.
    **kwargs
        Additional parameters passed to noise functions.

    Returns
    -------
    list of str
        List of output file paths created.

    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create test files
    ...     from .simulations import create_bias_frame
    ...     for i in range(3):
    ...         fpath = os.path.join(tmpdir, f'input_{i}.fits')
    ...         create_bias_frame(fpath, shape=(10, 10), bias_level=100)
    ...
    ...     # Add noise to all files
    ...     outputs = add_noise_batch(
    ...         tmpdir,
    ...         os.path.join(tmpdir, 'noisy_{stem}.fits'),
    ...         noise_type='gaussian',
    ...         sigma=5.0,
    ...         seed=42
    ...     )
    ...     print(len(outputs))
    3
    """
    # Select files
    try:
        selection_input: StrOrPath | list[StrOrPath] = (
            input_files
            if isinstance(input_files, str)
            else [Path(item) for item in input_files]
        )
        files = select_files(selection_input)
    except Exception as e:
        raise FileHandlingError(f"Failed to select input files: {e}") from e

    if not files:
        logger.warning("No input files found")
        return []
    if noise_type not in {"gaussian", "poisson"}:
        raise ValueError(
            f"Unknown noise type: {noise_type!r}; expected 'gaussian' or 'poisson'"
        )

    logger.debug(f"Processing {len(files)} files with {num_workers} workers")

    # Prepare tasks
    tasks = []
    output_files = []

    for idx, infile in enumerate(files):
        # Generate output filename
        stem = Path(infile).stem
        outfile = output_pattern.format(stem=stem, index=idx)
        output_files.append(outfile)

        # Determine seed for this file
        file_seed = (seed + idx) if seed is not None else None

        noise_parameters = (
            {"sigma": sigma} if noise_type == "gaussian" else {"gain": gain}
        )
        tasks.append({
            'input': infile,
            'output': outfile,
            'noise_type': noise_type,
            'seed': file_seed,
            **noise_parameters,
            **kwargs
        })

    # Process files
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            executor.map(_process_single_file, tasks)
    else:
        for task in tasks:
            _process_single_file(task)

    return output_files


def _process_single_file(task: dict[str, Any]) -> None:
    """Helper function for parallel processing."""
    task = task.copy()
    infile = task.get("input", "unknown")
    try:
        noise_type = task.pop('noise_type')
        task.pop('input')
        outfile = task.pop('output')

        if noise_type == 'gaussian':
            add_gaussian_noise(infile, outfile, **task)
        elif noise_type == 'poisson':
            add_poisson_noise(infile, outfile, **task)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    except Exception as e:
        logger.error("Failed to process %s: %s", infile, e)
        raise


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'add_gaussian_noise',
    'add_poisson_noise',
    'add_noise_batch',
]
