#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np
from astropy.io import fits

# Use refactored package infrastructure
from ..io import get_data, write_fits, get_header
from .core import (
    add_read_noise,      # Gaussian noise
    add_photon_noise,    # Poisson noise
    add_hot_pixels,
    add_cosmic_rays,
)
from ..exceptions import (
    ImageProcessingError,
    FileHandlingError,
    FitsToolsError,
)
from ..utils import (
    is_fits_file,
    validate_numeric,
)
from ..selection import select_files
from ..types import StrOrPath

logger = logging.getLogger(__name__)

# ============================================================================
# Noise Operation Wrappers
# ============================================================================

def add_gaussian_noise(
    input_file: StrOrPath,
    output: str,
    sigma: float = 1.0,
    region: Union[str, Tuple[int, int, int, int]] = 'full',
    hdu: int = 0,
    seed: Optional[int] = None,
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
        data = get_data(input_file, ext=ext, region=region)
        header = get_header(input_file, ext=ext)
        
        if data is None:
            raise ImageProcessingError(f"No data in {input_file} extension {ext}")
        
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
        raise ImageProcessingError(f"Failed to add Gaussian noise: {e}")


def add_poisson_noise(
    input_file: StrOrPath,
    output: StrOrPath,
    gain: float = 1.0,
    region: Union[str, Tuple[int, int, int, int]] = 'full',
    hdu: int = 0,
    seed: Optional[int] = None,
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
        data = get_data(input_file, ext=ext, region=region)
        header = get_header(input_file, ext=ext)
        
        if data is None:
            raise ImageProcessingError(f"No data in {input_file} extension {ext}")
        
        noisy_data = add_photon_noise(data, gain=gain, seed=seed)
        
        header.add_history(f'Added Poisson noise (gain={gain})')
        if seed is not None:
            header['NOISEED'] = (seed, 'Random seed for noise')
        
        write_fits(output, noisy_data, header=header, overwrite=True)
        
        logger.debug(f"Added Poisson noise (gain={gain}) to {input_file} -> {output}")
            
    except Exception as e:
        logger.error(f"Error adding noise to {input_file}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to add Poisson noise: {e}")


# ============================================================================
# Batch Processing
# ============================================================================

def add_noise_batch(
    input_files: Union[str, List[str]],
    output_pattern: str,
    noise_type: str = 'gaussian',
    sigma: float = 1.0,
    gain: float = 1.0,
    num_workers: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> List[str]:
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
        files = select_files(input_files)
    except Exception as e:
        raise FileHandlingError(f"Failed to select input files: {e}")
    
    if not files:
        logger.warning("No input files found")
        return []
    
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
        
        tasks.append({
            'input': infile,
            'output': outfile,
            'noise_type': noise_type,
            'sigma': sigma,
            'gain': gain,
            'seed': file_seed,
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


def _process_single_file(task: Dict[str, Any]) -> None:
    """Helper function for parallel processing."""
    try:
        noise_type = task.pop('noise_type')
        infile = task.pop('input')
        outfile = task.pop('output')
        
        if noise_type == 'gaussian':
            add_gaussian_noise(infile, outfile, **task)
        elif noise_type == 'poisson':
            add_poisson_noise(infile, outfile, **task)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
    except Exception as e:
        logger.error(f"Failed to process {task.get('input', 'unknown')}: {e}")
        raise


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'add_gaussian_noise',
    'add_poisson_noise',
    'add_noise_batch',
]

    