#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processing for FITS images with parallel execution support.

This module provides high-level batch processing capabilities that coordinate
operations across the entire image_processing package. It supports parallel
execution, progress tracking, and comprehensive error handling for processing
large numbers of files.

Features:
    - Unified interface for batch operations
    - Parallel processing with multiprocessing
    - Progress tracking with tqdm
    - Comprehensive error collection and reporting
    - Support for all image_processing operations
    - Automatic output directory management
    - Flexible file naming schemes
    - Memory-efficient processing options

Supported Operations:
    - Unary operations (log, sqrt, etc.)
    - Scalar operations (add, multiply, etc.)
    - Binary operations (image arithmetic)
    - Cosmic ray rejection
    - Image transformations (crop, bin, normalize)
    - Custom processing functions

Note:
    For batch calibration, use calibrate_batch() from calibration.py.
    This module focuses on general-purpose batch processing.

Usage Examples:
    Basic batch operation:
        >>> from fits_tools.image_processing.batch import process_batch
        >>> files = ['img1.fits', 'img2.fits', 'img3.fits']
        >>> result = process_batch(
        ...     files,
        ...     operation_type='unary',
        ...     operation='log',
        ...     output_dir='processed/'
        ... )
        >>> print(f"Success rate: {result.success_rate*100:.1f}%")
    
    Parallel processing:
        >>> result = process_batch(
        ...     files,
        ...     operation_type='cosmic_rays',
        ...     output_dir='cleaned/',
        ...     num_workers=4,
        ... )
    
    Batch cosmic ray cleaning:
        >>> result = batch_clean_cosmic_rays(files, output_dir='cleaned/')

Version: 2.0.0
"""

import logging
from pathlib import Path
from typing import List, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from astropy.io import fits

# Import from fits_tools
from ..io import  write_fits, get_data_and_header
from ..exceptions import ImageProcessingError, FileHandlingError
from ..utils import add_history, maybe_tqdm, is_tqdm_available


# Import from image_processing submodules
from .result_types import ProcessingResult, ImageOperationResult
from .operations import (
    apply_unary_operation,
    apply_scalar_operation,
    create_mask,
    apply_bitwise_operation,
)
from .transformations import (
    crop_image,
    bin_image,
    normalize_image,
    rotate_image,
    flip_image,
)
from .cosmic_rays import clean_cosmic_rays_file

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def _generate_output_filename(
    input_file: StrOrPath,
    output_dir: StrOrPath,
    suffix: Optional[str] = None,
    prefix: Optional[str] = None
) -> str:
    """
    Generate output filename based on input filename.
    
    Parameters
    ----------
    input_file : str
        Input file path.
    output_dir : str
        Output directory.
    suffix : str, optional
        Suffix to add before extension (e.g., '_processed').
    prefix : str, optional
        Prefix to add before filename.
    
    Returns
    -------
    str
        Output file path.
    """
    input_path = Path(input_file)
    name, ext = input_path.stem, input_path.suffix
    
    if prefix:
        name = f"{prefix}{name}"
    if suffix:
        name = f"{name}{suffix}"
    
    output_filename = f"{name}{ext}"
    return str(Path(output_dir) / output_filename)


def _process_single_file(
    filepath: str,
    operation_type: str,
    output_dir: str,
    hdu: int = 0,
    overwrite: bool = False,
    dry_run: bool = False, 
    **kwargs
) -> ImageOperationResult:
    """
    Process a single file (helper for parallel processing).
    
    This function is designed to be picklable for multiprocessing.
    
    Parameters
    ----------
    filepath : str
        Input FITS file.
    operation_type : str
        Type of operation to perform.
    output_dir : str
        Output directory.
    ext : int
        HDU extension.
    overwrite : bool
        Overwrite existing files.
    dry_run : bool
        If True, simulate processing without writing files.
    **kwargs
        Operation-specific parameters.
    
    Returns
    -------
    ImageOperationResult
        Result of processing.
    """
    try:
        # Generate output filename
        suffix = kwargs.get('suffix', '_processed')
        output_file = _generate_output_filename(
            filepath,
            output_dir,
            suffix=suffix
        )
        
        # Check if output exists
        if Path(output_file).exists() and not overwrite:
            return ImageOperationResult(
                success=False,
                output_file=output_file,
                operation=operation_type,
                error=f"Output exists and overwrite=False"
            )
        
        # DRY RUN: Simulate success without processing
        if dry_run:
            return ImageOperationResult(
                success=True,
                output_file=output_file,
                operation=operation_type,
                metadata={'input_file': filepath, 'dry_run': True}
            )
        
        # Load data
        data, header = get_data_and_header(filepath, ext=ext)
        
        # Perform operation based on type
        if operation_type == 'unary':
            operation = kwargs.get('operation')
            if not operation:
                raise ValueError("operation parameter required for unary operations")
            
            result_data = apply_unary_operation(
                data,
                operation,
                preserve_dtype=kwargs.get('preserve_dtype', False)
            )
            add_history(header, f"Applied unary operation: {operation}")
        
        elif operation_type == 'scalar':
            operation = kwargs.get('operation')
            scalar = kwargs.get('scalar')
            if not operation or scalar is None:
                raise ValueError("operation and scalar parameters required")
            
            result_data = apply_scalar_operation(
                data,
                operation,
                scalar,
                preserve_dtype=kwargs.get('preserve_dtype', False)
            )
            add_history(header, f"Applied scalar operation: {operation} {scalar}")
        
        elif operation_type == 'comparison':
            operation = kwargs.get('operation')
            threshold = kwargs.get('threshold')
            if not operation or threshold is None:
                raise ValueError("operation and threshold parameters required")
            
            result_data = create_mask(data, operation, threshold)
            result_data = result_data.astype(np.uint8)
            add_history(header, f"Created mask: {operation} {threshold}")
        
        elif operation_type == 'bitwise':
            operation = kwargs.get('operation')
            operand = kwargs.get('operand')
            
            result_data = apply_bitwise_operation(data, operation, operand)
            add_history(header, f"Applied bitwise operation: {operation}")
        
        elif operation_type == 'crop':
            region = kwargs.get('region')
            if not region:
                raise ValueError("region parameter required for crop")
            
            result_data = crop_image(data, region)
            add_history(header, f"Cropped to region: {region}")
        
        elif operation_type == 'bin':
            bin_factor = kwargs.get('bin_factor', 2)
            method = kwargs.get('method', 'mean')
            
            result_data = bin_image(data, bin_factor=bin_factor, method=method)
            add_history(header, f"Binned by {bin_factor}x using {method}")
        
        elif operation_type == 'normalize':
            method = kwargs.get('method', 'median')
            target = kwargs.get('target_value', 1.0)
            
            result_data = normalize_image(data, method=method, target_value=target)
            add_history(header, f"Normalized using {method} to {target}")
        
        elif operation_type == 'rotate':
            angle = kwargs.get('angle')
            if angle is None:
                raise ValueError("angle parameter required for rotate")
            
            result_data = rotate_image(data, angle)
            add_history(header, f"Rotated by {angle} degrees")
        
        elif operation_type == 'flip':
            axis = kwargs.get('axis')
            if axis is None:
                raise ValueError("axis parameter required for flip")
            
            result_data = flip_image(data, axis)
            add_history(header, f"Flipped along axis: {axis}")
        
        elif operation_type == 'cosmic_rays':
            # Use cosmic ray module's function
            return ImageOperationResult(
                success=True,
                output_file=clean_cosmic_rays_file(
                    filepath,
                    output_file=output_file,
                    ext=ext,
                    overwrite=overwrite,
                    **kwargs
                ),
                operation='cosmic_ray_rejection'
            )
        
        else:
            raise ValueError(f"Unknown operation_type: {operation_type}")
        
        # Write output
        write_fits(output_file, result_data, header, overwrite=overwrite)
        
        return ImageOperationResult(
            success=True,
            output_file=output_file,
            operation=operation_type,
            metadata={'input_file': filepath}
        )
    
    except Exception as e:
        return ImageOperationResult(
            success=False,
            operation=operation_type,
            error=str(e),
            metadata={'input_file': filepath}
        )

# ============================================================================
# Main Batch Processing Function
# ============================================================================

def process_batch(
    file_list: List[str],
    operation_type: str,
    output_dir: str,
    hdu: int = 0,
    overwrite: bool = False,
    skip_existing: bool = False,  
    dry_run: bool = False,  
    num_workers: int = 1,
    **kwargs
) -> ProcessingResult:
    """
    Process multiple FITS files in batch mode with optional parallelization.
    
    This is a unified interface for batch processing that supports all
    image_processing operations. For calibration, use calibrate_batch()
    from the calibration module.
    
    Parameters
    ----------
    file_list : list of str
        List of input FITS files.
    operation_type : str
        Type of operation to perform:
        - 'unary': Unary operations (requires 'operation' kwarg)
        - 'scalar': Scalar operations (requires 'operation' and 'scalar')
        - 'comparison': Comparison operations (requires 'operation' and 'threshold')
        - 'bitwise': Bitwise operations (requires 'operation' and 'operand')
        - 'crop': Crop images (requires 'region')
        - 'bin': Bin images (requires 'bin_factor')
        - 'normalize': Normalize images (requires 'method')
        - 'rotate': Rotate images (requires 'angle')
        - 'flip': Flip images (requires 'axis')
        - 'cosmic_rays': Cosmic ray rejection (optional: 'method', 'sigma')
    output_dir : str
        Output directory for processed files.
    ext : int, optional
        HDU extension to process. Defaults to 0.
    overwrite : bool, optional
        Overwrite existing output files. Defaults to False.
    skip_existing : bool, optional
        If True, skip files where output already exists (useful for resuming
        interrupted batch jobs). Defaults to False.
    dry_run : bool, optional
        If True, simulate processing without writing files. Useful for testing
        batch operations before committing. Defaults to False.
    num_workers : int, optional
        Number of parallel workers. If 1, processes serially. Defaults to 1.
    verbose : bool, optional
        Enable verbose logging and progress bars.
    **kwargs
        Operation-specific parameters (e.g., operation, scalar, threshold, etc.).
    
    Returns
    -------
    ProcessingResult
        Batch processing results with success/failure counts and error list.
    
    Raises
    ------
    ValueError
        If file_list is empty or operation_type is invalid.
    
    Examples
    --------
    Basic unary operation:
    
    >>> from fits_tools.image_processing.batch import process_batch
    >>> files = ['img1.fits', 'img2.fits', 'img3.fits']
    >>> result = process_batch(
    ...     files,
    ...     operation_type='unary',
    ...     operation='log',
    ...     output_dir='processed/',
    ...     overwrite=True
    ... )
    >>> print(f"Processed {result.success_count}/{result.total_count} files")
    
    Resume interrupted batch job:
    
    >>> result = process_batch(
    ...     files,
    ...     operation_type='cosmic_rays',
    ...     output_dir='cleaned/',
    ...     skip_existing=True,  # Skip already processed files
    ...     num_workers=4
    ... )
    
    Dry run (test without processing):
    
    >>> result = process_batch(
    ...     files,
    ...     operation_type='unary',
    ...     operation='sqrt',
    ...     output_dir='test/',
    ...     dry_run=True,  # No files written
    ...     verbose=True
    ... )
    >>> print(f"Would process {result.success_count} files")
    
    See Also
    --------
    calibrate_batch : Batch calibration (in calibration module)
    batch_clean_cosmic_rays : Convenience wrapper for cosmic ray cleaning
    
    Notes
    -----
    - Serial processing (num_workers=1) is more memory-efficient
    - Parallel processing (num_workers>1) is faster but uses more memory
    - Progress bars require tqdm package
    - For calibration, use calibration.calibrate_batch() instead
    - skip_existing is useful for resuming long batch jobs after interruption
    - dry_run is useful for validating parameters before processing
    """
    if not file_list:
        raise ValueError("Empty file list")
    
    valid_operation_types = [
        'unary', 'scalar', 'comparison', 'bitwise',
        'crop', 'bin', 'normalize', 'rotate', 'flip',
        'cosmic_rays'
    ]
    
    if operation_type not in valid_operation_types:
        raise ValueError(
            f"Unknown operation_type: '{operation_type}'. "
            f"Valid options: {', '.join(valid_operation_types)}"
        )
    
    # Create output directory
    if not dry_run:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Batch processing {len(file_list)} files")
    logger.debug(f"Operation type: {operation_type}")
    logger.debug(f"Output directory: {output_dir}")
    if dry_run:
        logger.debug("DRY RUN MODE - No files will be written")
    if num_workers > 1:
        logger.debug(f"Using {num_workers} parallel workers")
    
    # Initialize result
    result = ProcessingResult()
    
    # Filter files if skip_existing is enabled
    files_to_process = file_list
    if skip_existing and not overwrite:
        original_count = len(file_list)
        files_to_process = []
        skipped_count = 0
        
        suffix = kwargs.get('suffix', '_processed')
        
        for filepath in file_list:
            output_file = _generate_output_filename(
                filepath,
                output_dir,
                suffix=suffix
            )
            
            if Path(output_file).exists():
                skipped_count += 1
                logger.debug(f"Skipping (exists): {Path(filepath).name}")
            else:
                files_to_process.append(filepath)
        
        if skipped_count > 0:
            logger.debug(f"Skipping {skipped_count} existing files")
        
        if not files_to_process:
            logger.debug("All files already processed")
            return result
    
    # Serial processing
    if num_workers == 1:
        iterator = maybe_tqdm(
            files_to_process,
            desc=f"Processing ({operation_type})",
            unit="file"
        ) if verbose else files_to_process
        
        for filepath in iterator:
            file_result = _process_single_file(
                filepath,
                operation_type,
                output_dir,
                ext=ext,
                overwrite=overwrite,
                dry_run=dry_run,  # Pass dry_run parameter
                **kwargs
            )
            
            if file_result.success:
                result.add_success(file_result.output_file)
            else:
                error_msg = f"{Path(filepath).name}: {file_result.error}"
                result.add_failure(error_msg)
    
    # Parallel processing
    else:
        # Create partial function with fixed parameters
        process_func = partial(
            _process_single_file,
            operation_type=operation_type,
            output_dir=output_dir,
            ext=ext,
            overwrite=overwrite,
            dry_run=dry_run,  # Pass dry_run parameter
            **kwargs
        )
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(process_func, filepath): filepath
                for filepath in files_to_process
            }
            
            # Collect results with progress bar
            iterator = as_completed(futures)
            iterator = maybe_tqdm(
                iterator,
                total=len(files_to_process),
                desc=f"Processing ({operation_type})",
                unit="file"
            )
            
            for future in iterator:
                filepath = futures[future]
                try:
                    file_result = future.result()
                    
                    if file_result.success:
                        result.add_success(file_result.output_file)
                        if verbose and not is_tqdm_available():
                            logger.info(f"✓ {Path(filepath).name}")
                    else:
                        error_msg = f"{Path(filepath).name}: {file_result.error}"
                        result.add_failure(error_msg)
                        if verbose and not is_tqdm_available():
                            logger.error(f"✗ {Path(filepath).name}")
                
                except Exception as e:
                    error_msg = f"{Path(filepath).name}: {str(e)}"
                    result.add_failure(error_msg)
                    if verbose and not is_tqdm_available():
                        logger.error(f"✗ {Path(filepath).name}: {e}")
    
    if verbose:
        logger.info(f"Batch processing complete:")
        logger.info(f"  Success: {result.success_count}/{result.total_count}")
        logger.info(f"  Failed: {result.failure_count}")
        logger.info(f"  Success rate: {result.success_rate*100:.1f}%")
        if dry_run:
            logger.info("  (DRY RUN - no files were actually written)")
    
    return result

# ============================================================================
# Convenience Functions
# ============================================================================

def batch_clean_cosmic_rays(
    file_list: List[str],
    output_dir: str,
    method: str = 'sigma_clip',
    sigma: float = 5.0,
    hdu: int = 0,
    num_workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
    **kwargs
) -> ProcessingResult:
    """
    Clean cosmic rays from multiple files (convenience wrapper).
    
    Parameters
    ----------
    file_list : list of str
        Input FITS files.
    output_dir : str
        Output directory.
    method : str, optional
        Cosmic ray detection method. Defaults to 'sigma_clip'.
    sigma : float, optional
        Detection threshold. Defaults to 5.0.
    ext : int, optional
        HDU extension. Defaults to 0.
    num_workers : int, optional
        Number of parallel workers. Defaults to 1.
    overwrite : bool, optional
        Overwrite existing files. Defaults to False.
    verbose : bool, optional
        Enable verbose output.
    **kwargs
        Additional arguments passed to cosmic ray rejection.
    
    Returns
    -------
    ProcessingResult
        Batch processing results.
    
    Examples
    --------
    >>> from fits_tools.image_processing.batch import batch_clean_cosmic_rays
    >>> files = ['sci_001.fits', 'sci_002.fits', 'sci_003.fits']
    >>> result = batch_clean_cosmic_rays(
    ...     files,
    ...     output_dir='cleaned/',
    ...     method='sigma_clip',
    ...     sigma=5.0,
    ...     num_workers=4
    ... )
    """
    return process_batch(
        file_list,
        operation_type='cosmic_rays',
        output_dir=output_dir,
        method=method,
        sigma=sigma,
        ext=ext,
        num_workers=num_workers,
        overwrite=overwrite,
        verbose=verbose,
        suffix='_clean',
        **kwargs
    )


def batch_apply_operation(
    file_list: List[str],
    operation: str,
    output_dir: str,
    operation_type: str = 'unary',
    hdu: int = 0,
    num_workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
    **kwargs
) -> ProcessingResult:
    """
    Apply an operation to multiple files (convenience wrapper).
    
    Parameters
    ----------
    file_list : list of str
        Input FITS files.
    operation : str
        Operation name (e.g., 'log', 'sqrt', 'add', etc.).
    output_dir : str
        Output directory.
    operation_type : str, optional
        Type of operation ('unary', 'scalar', etc.). Defaults to 'unary'.
    ext : int, optional
        HDU extension. Defaults to 0.
    num_workers : int, optional
        Number of parallel workers. Defaults to 1.
    overwrite : bool, optional
        Overwrite existing files. Defaults to False.
    verbose : bool, optional
        Enable verbose output.
    **kwargs
        Operation-specific parameters (e.g., scalar, threshold).
    
    Returns
    -------
    ProcessingResult
        Batch processing results.
    
    Examples
    --------
    >>> from fits_tools.image_processing.batch import batch_apply_operation
    >>> # Apply log to all files
    >>> result = batch_apply_operation(
    ...     files,
    ...     operation='log',
    ...     output_dir='log/',
    ...     operation_type='unary'
    ... )
    
    >>> # Add constant to all files
    >>> result = batch_apply_operation(
    ...     files,
    ...     operation='add',
    ...     output_dir='offset/',
    ...     operation_type='scalar',
    ...     scalar=100
    ... )
    """
    return process_batch(
        file_list,
        operation_type=operation_type,
        operation=operation,
        output_dir=output_dir,
        ext=ext,
        num_workers=num_workers,
        overwrite=overwrite,
        verbose=verbose,
        suffix=f'_{operation}',
        **kwargs
    )


def batch_transform(
    file_list: List[str],
    transform_type: str,
    output_dir: str,
    hdu: int = 0,
    num_workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
    **kwargs
) -> ProcessingResult:
    """
    Apply geometric transformation to multiple files.
    
    Parameters
    ----------
    file_list : list of str
        Input FITS files.
    transform_type : str
        Transformation type ('crop', 'bin', 'normalize', 'rotate', 'flip').
    output_dir : str
        Output directory.
    ext : int, optional
        HDU extension. Defaults to 0.
    num_workers : int, optional
        Number of parallel workers. Defaults to 1.
    overwrite : bool, optional
        Overwrite existing files. Defaults to False.
    verbose : bool, optional
        Enable verbose output.
    **kwargs
        Transform-specific parameters.
    
    Returns
    -------
    ProcessingResult
        Batch processing results.
    
    Examples
    --------
    >>> from fits_tools.image_processing.batch import batch_transform
    >>> # Bin all images by 2x2
    >>> result = batch_transform(
    ...     files,
    ...     transform_type='bin',
    ...     output_dir='binned/',
    ...     bin_factor=2,
    ...     method='mean'
    ... )
    
    >>> # Crop all images
    >>> result = batch_transform(
    ...     files,
    ...     transform_type='crop',
    ...     output_dir='cropped/',
    ...     region=(100, 900, 100, 900)
    ... )
    """
    valid_transforms = ['crop', 'bin', 'normalize', 'rotate', 'flip']
    if transform_type not in valid_transforms:
        raise ValueError(
            f"Unknown transform_type: '{transform_type}'. "
            f"Valid options: {', '.join(valid_transforms)}"
        )
    
    return process_batch(
        file_list,
        operation_type=transform_type,
        output_dir=output_dir,
        ext=ext,
        num_workers=num_workers,
        overwrite=overwrite,
        verbose=verbose,
        suffix=f'_{transform_type}',
        **kwargs
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main function
    'process_batch',
    
    # Convenience functions
    'batch_clean_cosmic_rays',
    'batch_apply_operation',
    'batch_transform',
]

