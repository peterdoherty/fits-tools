#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for FITS image processing.

This module provides a unified command-line interface (CLI) for all
image_processing operations. It serves as the primary user-facing entry
point for the fits_tools.image_processing package.

Entry Point:
    fits-process <subcommand> [options] files...

Subcommands:
    calibrate      - Calibrate raw images with bias/dark/flat correction
    combine        - Combine/stack multiple images
    operation      - Apply mathematical operations (unary, scalar, binary)
    transform      - Apply geometric transformations (crop, bin, normalize)
    cosmic-rays    - Detect and remove cosmic rays
    batch          - Batch process multiple files
    stream         - Memory-efficient processing for large datasets
    quality        - Quality control and assessment
    info           - Display information and available operations

Examples:
    # Calibrate a raw image
    fits-process calibrate raw.fits --bias bias.fits --output calibrated.fits
    
    # Combine bias frames
    fits-process combine bias_*.fits --method median --output master_bias.fits
    
    # Apply log operation
    fits-process operation --unary log image.fits --output log_image.fits
    
    # Clean cosmic rays from multiple files
    fits-process cosmic-rays *.fits --output-dir cleaned/ --workers 4
    
    # Batch process with dry run
    fits-process batch --op unary log *.fits --output-dir processed/ --dry-run

TODO Features (for v2):
    - Pipeline mode: Chain multiple operations in single command
    - Config file support: ~/.fits-tools.conf for default parameters
    - Shell completion: bash/zsh completion scripts
    - Multiple output formats: Support NPY, PNG, TIFF (currently FITS only)
    - Logging to file: --log option to save processing log

Version: 2.0.0
"""

import sys
import os
import logging
import argparse
from typing import List, Optional
from pathlib import Path
from glob import glob

import numpy as np
from astropy.io import fits

# Import from fits_tools
from ..io import get_data, get_header
from ..exceptions import ImageProcessingError, FileHandlingError
from ..utils import add_history

# Import from image_processing submodules
from .calibration import calibrate_image, calibrate_batch
from .combining import combine_images_robust, median_combine, mean_combine
from .operations import (
    list_operations,
    get_operation_description,
    UNARY_OPS,
    SCALAR_OPS,
    COMPARISON_OPS,
    BITWISE_OPS,
)
from .transformations import crop_image, bin_image, normalize_image
from .cosmic_rays import clean_cosmic_rays_file, batch_clean_cosmic_rays, get_available_methods
from .batch import process_batch, batch_apply_operation, batch_transform
from .streaming import streaming_combine, streaming_statistics
from .quality_control import assess_quality, batch_quality_assessment

logger = logging.getLogger(__name__)
__version__ = "2.0.0"
__author__ = "FITS Tools Development Team"



# ---------------------------------------------------------------------
# Utilities & helpers
# ---------------------------------------------------------------------

def expand_file_patterns(patterns: List[str]) -> List[str]:
    """
    Expand file patterns into a list of file paths.
    
    Supports:
    - Direct file paths
    - Glob patterns (*.fits)
    - Text files containing lists of paths (@filelist.txt)
    
    Parameters
    ----------
    patterns : list of str
        File patterns to expand.
    
    Returns
    -------
    list of str
        Expanded list of file paths.
    """
    files = []
    
    for pattern in patterns:
        # @file.txt - read list from file
        if pattern.startswith('@'):
            filepath = pattern[1:]
            if not os.path.exists(filepath):
                logger.error(f"File list not found: {filepath}")
                continue
            
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        files.append(line)
        
        # Glob pattern
        elif any(ch in pattern for ch in '*?[]'):
            matched = sorted(glob(pattern))
            if not matched:
                logger.warning(f"No files matched pattern: {pattern}")
            files.extend(matched)
        
        # Direct file path
        else:
            files.append(pattern)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def setup_logging(verbose: bool = False, quiet: bool = False):
    """
    Setup logging configuration.
    
    Parameters
    ----------
    verbose : bool
        Enable verbose (DEBUG) logging.
    quiet : bool
        Suppress all but ERROR messages.
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def handle_error(error: Exception, verbose: bool = False) -> int:
    """
    Handle and display error messages.
    
    Parameters
    ----------
    error : Exception
        The exception to handle.
    verbose : bool
        If True, show full traceback.
    
    Returns
    -------
    int
        Exit code (1 for error).
    """
    logger.error(f"Error: {error}")
    
    if verbose:
        import traceback
        traceback.print_exc()
    
    return 1


# ============================================================================
# Subcommand: calibrate
# ============================================================================

def cmd_calibrate(args) -> int:
    """Handle calibrate subcommand."""
    try:
        files = expand_file_patterns(args.files)
        
        if not files:
            logger.error("No input files specified")
            return 1
        
        # Single file mode
        if len(files) == 1 and args.output:
            result = calibrate_image(
                files[0],
                output_file=args.output,
                master_bias=args.bias,
                master_dark=args.dark,
                master_flat=args.flat,
                ext=args.ext,
                overwrite=args.overwrite,
                validate_inputs=args.validate,
                compute_quality=args.quality,
                verbose=args.verbose
            )
            
            logger.info(f"Calibration complete: {result}")
            return 0
        
        # Batch mode
        elif len(files) > 1 or args.output_dir:
            if not args.output_dir:
                logger.error("--output-dir required for multiple files")
                return 1
            
            if args.dry_run:
                logger.info(f"DRY RUN: Would calibrate {len(files)} files")
                logger.info(f"  Output directory: {args.output_dir}")
                logger.info(f"  Bias: {args.bias or 'None'}")
                logger.info(f"  Dark: {args.dark or 'None'}")
                logger.info(f"  Flat: {args.flat or 'None'}")
                return 0
            
            results = calibrate_batch(
                files,
                output_dir=args.output_dir,
                master_bias=args.bias,
                master_dark=args.dark,
                master_flat=args.flat,
                ext=args.ext,
                overwrite=args.overwrite,
                validate_inputs=args.validate,
                compute_quality=args.quality,
                num_workers=args.workers,
                verbose=args.verbose
            )
            
            logger.info(f"Batch calibration complete: {len(results)} files processed")
            return 0
        
        else:
            logger.error("Specify --output for single file or --output-dir for batch")
            return 1
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Subcommand: combine
# ============================================================================

def cmd_combine(args) -> int:
    """Handle combine subcommand."""
    try:
        files = expand_file_patterns(args.files)
        
        if not files:
            logger.error("No input files specified")
            return 1
        
        if len(files) < 2:
            logger.error("At least 2 files required for combining")
            return 1
        
        if not args.output:
            logger.error("--output required")
            return 1
        
        if args.dry_run:
            logger.info(f"DRY RUN: Would combine {len(files)} files")
            logger.info(f"  Method: {args.method}")
            logger.info(f"  Output: {args.output}")
            return 0
        
        result = combine_images_robust(
            files,
            method=args.method,
            sigma=args.sigma,
            maxiters=args.maxiters,
            output_file=args.output,
            ext=args.ext,
            overwrite=args.overwrite,
            verbose=args.verbose
        )
        
        logger.info(f"Combine complete: {result}")
        return 0
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Subcommand: operation
# ============================================================================

def cmd_operation(args) -> int:
    """Handle operation subcommand."""
    try:
        files = expand_file_patterns(args.files)
        
        if not files:
            logger.error("No input files specified")
            return 1
        
        # Determine operation type and parameters
        op_kwargs = {}
        
        if args.unary:
            op_type = 'unary'
            op_kwargs['operation'] = args.unary
            suffix = f'_{args.unary}'
        
        elif args.scalar:
            op_type = 'scalar'
            if args.value is None:
                logger.error("--value required for scalar operations")
                return 1
            op_kwargs['operation'] = args.scalar
            op_kwargs['scalar'] = args.value
            suffix = f'_{args.scalar}'
        
        elif args.comparison:
            op_type = 'comparison'
            if args.invert:
                op_kwargs["invert"] = True
            if args.threshold is None:
                logger.error("--threshold required for comparison operations")
                return 1
            op_kwargs['operation'] = args.comparison
            op_kwargs['threshold'] = args.threshold
            suffix = f"_{args.comparison}"
        
        elif args.fill:
            op_type = "fill"
            op_kwargs["operation"] = args.fill

            if args.value is None:
                logger.error("--value required for fill operations (fill_value)")
                return 1
            op_kwargs["scalar"] = args.value

            # Optional invert
            if args.invert:
                op_kwargs["invert"] = True

            if args.fill == "mask":
                if not args.mask:
                    logger.error("--mask required for --fill mask")
                    return 1
                op_kwargs["operand"] = args.mask

                # Only pass mask_ext if user provided it
                if args.mask_ext is not None:
                    op_kwargs["mask_ext"] = args.mask_ext

                suffix = "_fillmask"
            else:
                # Comparison-based fill requires threshold
                if args.threshold is None:
                    logger.error("--threshold required for comparison-based fill operations")
                    return 1

                op_kwargs["threshold"] = args.threshold
                suffix = f"_fill_{args.fill}"

        elif args.bitwise:
            op_type = 'bitwise'
            op_kwargs['operation'] = args.bitwise
            if args.bitwise != 'bitwise_not' and args.operand is None:
                logger.error("--operand required for bitwise operations (except NOT)")
                return 1
            if args.operand is not None:
                op_kwargs['operand'] = args.operand
            suffix = f'_{args.bitwise}'
        
        else:
            logger.error("Specify operation type: --unary, --scalar, --comparison, --bitwise, or --fill")
            return 1
        
        # Single file mode
        if len(files) == 1 and args.output:
            if args.dry_run:
                logger.info(f"DRY RUN: Would process {files[0]}")
                logger.info(f"  Operation: {op_type} {op_kwargs.get('operation')}")
                logger.info(f"  Output: {args.output}")
                return 0
            
            from .operations import apply_operation_to_file
            
            result = apply_operation_to_file(
                files[0],
                args.output,
                op_type,
                ext=args.ext,
                overwrite=args.overwrite,
                verbose=args.verbose,
                **op_kwargs
            )
            
            if result.success:
                logger.info(f"Operation complete: {result.output_file}")
                return 0
            else:
                logger.error(f"Operation failed: {result.error}")
                return 1
        
        # Batch mode
        elif len(files) > 1 or args.output_dir:
            if not args.output_dir:
                logger.error("--output-dir required for multiple files")
                return 1
            
            result = process_batch(
                files,
                operation_type=op_type,
                output_dir=args.output_dir,
                ext=args.ext,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                num_workers=args.workers,
                verbose=args.verbose,
                suffix=suffix,
                **op_kwargs
            )
            
            logger.info(f"Batch operation complete: {result.success_count}/{result.total_count} successful")
            return 0 if result.failure_count == 0 else 1
        
        else:
            logger.error("Specify --output for single file or --output-dir for batch")
            return 1
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Subcommand: transform
# ============================================================================

def cmd_transform(args) -> int:
    """Handle transform subcommand."""
    try:
        files = expand_file_patterns(args.files)
        
        if not files:
            logger.error("No input files specified")
            return 1
        
        # Determine transformation type and parameters
        transform_kwargs = {}
        
        if args.crop:
            transform_type = 'crop'
            # Parse region: x1,x2,y1,y2
            try:
                coords = [int(x) for x in args.crop.split(',')]
                if len(coords) != 4:
                    raise ValueError("Crop requires 4 coordinates")
                transform_kwargs['region'] = tuple(coords)
            except ValueError as e:
                logger.error(f"Invalid crop region: {e}")
                return 1
        
        elif args.bin:
            transform_type = 'bin'
            transform_kwargs['bin_factor'] = args.bin
            transform_kwargs['method'] = args.bin_method
        
        elif args.normalize:
            transform_type = 'normalize'
            transform_kwargs['method'] = args.normalize
            transform_kwargs['target_value'] = args.target
        
        elif args.rotate:
            transform_type = 'rotate'
            transform_kwargs['angle'] = args.rotate
        
        elif args.flip:
            transform_type = 'flip'
            transform_kwargs['axis'] = args.flip
        
        else:
            logger.error("Specify transformation: --crop, --bin, --normalize, --rotate, or --flip")
            return 1
        
        # Single file mode
        if len(files) == 1 and args.output:
            if args.dry_run:
                logger.info(f"DRY RUN: Would transform {files[0]}")
                logger.info(f"  Transformation: {transform_type}")
                logger.info(f"  Output: {args.output}")
                return 0
            
            result = process_batch(
                [files[0]],
                operation_type=transform_type,
                output_dir=os.path.dirname(args.output) or '.',
                ext=args.ext,
                overwrite=args.overwrite,
                verbose=args.verbose,
                suffix='',
                **transform_kwargs
            )
            
            # Rename to desired output
            if result.success_count > 0 and result.output_files[0] != args.output:
                import shutil
                shutil.move(result.output_files[0], args.output)
            
            logger.info(f"Transform complete: {args.output}")
            return 0
        
        # Batch mode
        elif len(files) > 1 or args.output_dir:
            if not args.output_dir:
                logger.error("--output-dir required for multiple files")
                return 1
            
            result = batch_transform(
                files,
                transform_type=transform_type,
                output_dir=args.output_dir,
                ext=args.ext,
                num_workers=args.workers,
                overwrite=args.overwrite,
                verbose=args.verbose,
                **transform_kwargs
            )
            
            logger.info(f"Batch transform complete: {result.success_count}/{result.total_count} successful")
            return 0 if result.failure_count == 0 else 1
        
        else:
            logger.error("Specify --output for single file or --output-dir for batch")
            return 1
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Subcommand: cosmic-rays
# ============================================================================

def cmd_cosmic_rays(args) -> int:
    """Handle cosmic-rays subcommand."""
    try:
        files = expand_file_patterns(args.files)
        
        if not files:
            logger.error("No input files specified")
            return 1
        
        # Single file mode
        if len(files) == 1 and args.output:
            if args.dry_run:
                logger.info(f"DRY RUN: Would clean {files[0]}")
                logger.info(f"  Method: {args.method}")
                logger.info(f"  Output: {args.output}")
                return 0
            
            result = clean_cosmic_rays_file(
                files[0],
                output_file=args.output,
                method=args.method,
                sigma=args.sigma,
                iterations=args.iterations,
                ext=args.ext,
                overwrite=args.overwrite,
                save_mask=args.save_mask,
                verbose=args.verbose
            )
            
            logger.info(f"Cosmic ray cleaning complete: {result}")
            return 0
        
        # Batch mode
        elif len(files) > 1 or args.output_dir:
            if not args.output_dir:
                logger.error("--output-dir required for multiple files")
                return 1
            
            if args.dry_run:
                logger.info(f"DRY RUN: Would clean {len(files)} files")
                logger.info(f"  Method: {args.method}")
                logger.info(f"  Output directory: {args.output_dir}")
                return 0
            
            result = batch_clean_cosmic_rays(
                files,
                output_dir=args.output_dir,
                method=args.method,
                sigma=args.sigma,
                ext=args.ext,
                num_workers=args.workers,
                overwrite=args.overwrite,
                verbose=args.verbose
            )
            
            logger.info(f"Batch cleaning complete: {result.success_count}/{result.total_count} successful")
            return 0 if result.failure_count == 0 else 1
        
        else:
            logger.error("Specify --output for single file or --output-dir for batch")
            return 1
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Subcommand: batch
# ============================================================================

def cmd_batch(args) -> int:
    """Handle batch subcommand."""
    try:
        files = expand_file_patterns(args.files)
        
        if not files:
            logger.error("No input files specified")
            return 1
        
        if not args.output_dir:
            logger.error("--output-dir required for batch processing")
            return 1
        
        # Build operation kwargs
        op_kwargs = {}
        
        if args.operation:
            op_kwargs['operation'] = args.operation
        
        if args.scalar is not None:
            op_kwargs['scalar'] = args.scalar
        
        if args.threshold is not None:
            op_kwargs['threshold'] = args.threshold
        
        if args.operand is not None:
            op_kwargs['operand'] = args.operand
        
        result = process_batch(
            files,
            operation_type=args.op_type,
            output_dir=args.output_dir,
            ext=args.ext,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            num_workers=args.workers,
            verbose=args.verbose,
            **op_kwargs
        )
        
        logger.info(f"Batch processing complete:")
        logger.info(f"  Success: {result.success_count}/{result.total_count}")
        logger.info(f"  Failed: {result.failure_count}")
        
        if result.errors and args.verbose:
            logger.info("Errors:")
            for error in result.errors[:10]:  # Show first 10
                logger.info(f"  - {error}")
        
        return 0 if result.failure_count == 0 else 1
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Subcommand: stream
# ============================================================================

def cmd_stream(args) -> int:
    """Handle stream subcommand."""
    try:
        files = expand_file_patterns(args.files)
        
        if not files:
            logger.error("No input files specified")
            return 1
        
        if not args.output:
            logger.error("--output required")
            return 1
        
        if args.dry_run:
            logger.info(f"DRY RUN: Would process {len(files)} files using streaming")
            logger.info(f"  Method: {args.method}")
            logger.info(f"  Output: {args.output}")
            return 0
        
        if args.method in ('median', 'mean', 'min', 'max'):
            # Streaming combine
            result = streaming_combine(
                files,
                method=args.method,
                output_file=args.output,
                ext=args.ext,
                overwrite=args.overwrite,
                verbose=args.verbose
            )
            
            logger.info(f"Streaming combine complete: {result}")
            return 0
        
        elif args.method == 'stats':
            # Streaming statistics
            stats = streaming_statistics(
                files,
                ext=args.ext,
                compute_quantiles=args.quantiles,
                verbose=args.verbose
            )
            
            # Write mean to output
            from ..io import write_fits
            header = get_header(files[0], ext=args.ext, verbose=False)
            add_history(header, f'Streaming statistics: mean of {len(files)} files')
            
            write_fits(args.output, stats['mean'], header, overwrite=args.overwrite, verbose=args.verbose)
            
            logger.info(f"Streaming statistics complete:")
            logger.info(f"  Files processed: {stats['count']}")
            logger.info(f"  Mean written to: {args.output}")
            
            return 0
        
        else:
            logger.error(f"Unknown streaming method: {args.method}")
            return 1
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Subcommand: quality
# ============================================================================

def cmd_quality(args) -> int:
    """Handle quality subcommand."""
    try:
        files = expand_file_patterns(args.files)
        
        if not files:
            logger.error("No input files specified")
            return 1
        
        # Single file mode
        if len(files) == 1:
            qc = assess_quality(
                files[0],
                ext=args.ext,
                frame_type=args.frame_type,
                verbose=args.verbose
            )
            
            print(qc)
            
            if args.report:
                qc.to_json(args.report)
                logger.info(f"Report saved to: {args.report}")
            
            return 0 if qc.passed else 1
        
        # Batch mode
        else:
            results = batch_quality_assessment(
                files,
                ext=args.ext,
                frame_type=args.frame_type,
                verbose=args.verbose
            )
            
            from .quality_control import generate_quality_report
            
            report = generate_quality_report(results, format='text')
            print(report)
            
            if args.report:
                with open(args.report, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to: {args.report}")
            
            return 0 if results['pass_rate'] == 1.0 else 1
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Subcommand: info
# ============================================================================

def cmd_info(args) -> int:
    """Handle info subcommand."""
    try:
        if args.list_operations:
            print("Available Operations:")
            print("=" * 70)
            
            ops = list_operations()
            for op_type, op_list in ops.items():
                print(f"\n{op_type.upper()} Operations:")
                for op in op_list:
                    desc = get_operation_description(op)
                    print(f"  {op:20s} - {desc}")
            
            return 0
        
        elif args.check_dependencies:
            print("Dependency Status:")
            print("=" * 70)
            
            # Check scipy
            try:
                import scipy
                print("✓ scipy available")
            except ImportError:
                print("✗ scipy not available (required for some operations)")
                print("  Install with: pip install scipy")
            
            # Check astroscrappy
            try:
                import astroscrappy
                print("✓ astroscrappy available (L.A.Cosmic support)")
            except ImportError:
                print("✗ astroscrappy not available (optional, for L.A.Cosmic)")
                print("  Install with: pip install astroscrappy")
            
            # Check tqdm
            try:
                import tqdm
                print("✓ tqdm available (progress bars)")
            except ImportError:
                print("✗ tqdm not available (optional, for progress bars)")
                print("  Install with: pip install tqdm")
            
            # Check PIL
            try:
                from PIL import Image
                print("✓ PIL/Pillow available (image export)")
            except ImportError:
                print("✗ PIL/Pillow not available (optional, for PNG/TIFF export)")
                print("  Install with: pip install Pillow")
            
            return 0
        
        elif args.cosmic_ray_methods:
            print("Cosmic Ray Detection Methods:")
            print("=" * 70)
            
            methods = get_available_methods()
            for name, info in methods.items():
                status = "✓" if info['available'] else "✗"
                print(f"\n{status} {name} - {info['name']}")
                print(f"  {info['description']}")
                if not info['available']:
                    print(f"  Requires: {info['requires']}")
                print(f"  Recommended for: {info['recommended_for']}")
            
            return 0
        
        else:
            print("fits-process info")
            print("=" * 70)
            print(f"Version: {__version__}")
            print("Use --list-operations, --check-dependencies, or --cosmic-ray-methods")
            return 0
    
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Main Parser Setup
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    
    parser = argparse.ArgumentParser(
        prog='fits-process',
        description='FITS Image Processing Command-Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate a raw image
  fits-process calibrate raw.fits --bias bias.fits --output calibrated.fits
  
  # Combine bias frames
  fits-process combine bias_*.fits --method median --output master_bias.fits
  
  # Apply log operation to multiple files
  fits-process operation --unary log *.fits --output-dir processed/
  
  # Clean cosmic rays with parallel processing
  fits-process cosmic-rays *.fits --output-dir cleaned/ --workers 4
  
  # Batch process with dry run
  fits-process batch --op-type unary --operation log *.fits --output-dir test/ --dry-run
  
  # Check available operations
  fits-process info --list-operations

For more information: https://github.com/yourusername/fits_tools
        """
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Subcommand to execute')
    subparsers.required = True
    
    # Common arguments for all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    common.add_argument('-q', '--quiet', action='store_true', help='Quiet mode (errors only)')
    common.add_argument('--ext', type=int, default=0, help='HDU extension (default: 0)')
    common.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
    # ========================================================================
    # Subcommand: calibrate
    # ========================================================================
    
    parser_calibrate = subparsers.add_parser(
        'calibrate',
        parents=[common],
        help='Calibrate raw images',
        description='Apply bias, dark, and flat field correction to raw images.'
    )
    parser_calibrate.add_argument('files', nargs='+', help='Input FITS files')
    parser_calibrate.add_argument('-o', '--output', help='Output file (single file mode)')
    parser_calibrate.add_argument('--output-dir', help='Output directory (batch mode)')
    parser_calibrate.add_argument('--bias', help='Master bias file')
    parser_calibrate.add_argument('--dark', help='Master dark file')
    parser_calibrate.add_argument('--flat', help='Master flat file')
    parser_calibrate.add_argument('--validate', action='store_true', help='Validate inputs')
    parser_calibrate.add_argument('--quality', action='store_true', help='Compute quality metrics')
    parser_calibrate.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser_calibrate.add_argument('--dry-run', action='store_true', help='Simulate without processing')
    parser_calibrate.set_defaults(func=cmd_calibrate)
    
    # ========================================================================
    # Subcommand: combine
    # ========================================================================
    
    parser_combine = subparsers.add_parser(
        'combine',
        parents=[common],
        help='Combine/stack multiple images',
        description='Combine multiple images using robust statistics.'
    )
    parser_combine.add_argument('files', nargs='+', help='Input FITS files')
    parser_combine.add_argument('-o', '--output', required=True, help='Output file')
    parser_combine.add_argument('--method', default='sigma_clipped_median',
                                choices=['sigma_clipped_median', 'sigma_clipped_mean', 'median', 'mean'],
                                help='Combining method (default: sigma_clipped_median)')
    parser_combine.add_argument('--sigma', type=float, default=3.0, help='Sigma threshold (default: 3.0)')
    parser_combine.add_argument('--maxiters', type=int, default=5, help='Max iterations (default: 5)')
    parser_combine.add_argument('--dry-run', action='store_true', help='Simulate without processing')
    parser_combine.set_defaults(func=cmd_combine)
    
    # ========================================================================
    # Subcommand: operation
    # ========================================================================

    parser_operation = subparsers.add_parser(
        "operation",
        parents=[common],
        help="Apply mathematical operations",
        description="Apply unary, scalar, comparison, bitwise, or fill operations.",
    )
    parser_operation.add_argument("files", nargs="+", help="Input FITS files")
    parser_operation.add_argument("-o", "--output", help="Output file (single file mode)")
    parser_operation.add_argument("--output-dir", help="Output directory (batch mode)")

    # Operation type group
    op_group = parser_operation.add_mutually_exclusive_group(required=True)
    op_group.add_argument("--unary", choices=sorted(UNARY_OPS.keys()),
                          help="Unary operation (log, sqrt, etc.)")
    op_group.add_argument("--scalar", choices=sorted(SCALAR_OPS.keys()),
                          help="Scalar operation (add, multiply, etc.)")
    op_group.add_argument("--comparison", choices=sorted(COMPARISON_OPS.keys()),
                          help="Comparison operation (greater, less, etc.)")
    op_group.add_argument("--bitwise", choices=sorted(BITWISE_OPS.keys()),
                          help="Bitwise operation (and, or, xor, not)")

    fill_choices = ["mask"] + sorted(COMPARISON_OPS.keys())
    op_group.add_argument(
        "--fill",
        choices=fill_choices,
        help="Fill pixels selected by a mask: use 'mask' with --mask/--mask-ext, or a comparison op with --threshold.",
    )

    # Common operation parameters
    parser_operation.add_argument(
        "--value",
        type=float,
        help="Scalar value (required for --scalar; fill_value for --fill).",
    )
    parser_operation.add_argument(
        "--threshold",
        type=float,
        help="Threshold (required for --comparison; also used for --fill with comparison ops).",
    )
    parser_operation.add_argument(
        "--operand",
        type=int,
        help="Operand (required for most bitwise ops; ignored otherwise).",
    )
    parser_operation.add_argument(
        "--invert",
        action="store_true",
        help="Invert selection (for --fill and comparison-based fills).",
    )

    # Fill-specific parameters (kept in their own group to reduce noise)
    fill_group = parser_operation.add_argument_group("fill options")

    fill_group.add_argument(
        "--mask",
        help="Mask FITS file (used only with --fill mask).",
    )
    fill_group.add_argument(
        "--mask-ext",
        type=int,
        default=None,
        help="Mask HDU extension for --mask; defaults to --ext.",
    )

    parser_operation.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser_operation.add_argument("--dry-run", action="store_true", help="Simulate without processing")

    parser_operation.set_defaults(func=cmd_operation)

    
    # ========================================================================
    # Subcommand: transform
    # ========================================================================
    
    parser_transform = subparsers.add_parser(
        'transform',
        parents=[common],
        help='Apply geometric transformations',
        description='Crop, bin, normalize, rotate, or flip images.'
    )
    parser_transform.add_argument('files', nargs='+', help='Input FITS files')
    parser_transform.add_argument('-o', '--output', help='Output file (single file mode)')
    parser_transform.add_argument('--output-dir', help='Output directory (batch mode)')
    
    # Transform type group
    transform_group = parser_transform.add_mutually_exclusive_group(required=True)
    transform_group.add_argument('--crop', help='Crop region (x1,x2,y1,y2)')
    transform_group.add_argument('--bin', type=int, help='Bin factor (e.g., 2 for 2x2)')
    transform_group.add_argument('--normalize', choices=['median', 'mean', 'max'],
                                help='Normalization method')
    transform_group.add_argument('--rotate', type=float, help='Rotation angle in degrees')
    transform_group.add_argument('--flip', choices=['horizontal', 'vertical', 'both', '0', '1'],
                                help='Flip axis')
    
    # Transform parameters
    parser_transform.add_argument('--bin-method', choices=['mean', 'sum'], default='mean',
                                  help='Binning method (default: mean)')
    parser_transform.add_argument('--target', type=float, default=1.0,
                                  help='Target value for normalization (default: 1.0)')
    parser_transform.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser_transform.add_argument('--dry-run', action='store_true', help='Simulate without processing')
    parser_transform.set_defaults(func=cmd_transform)
    
    # ========================================================================
    # Subcommand: cosmic-rays
    # ========================================================================
    
    parser_cr = subparsers.add_parser(
        'cosmic-rays',
        parents=[common],
        help='Detect and remove cosmic rays',
        description='Clean cosmic rays from images using various detection methods.'
    )
    parser_cr.add_argument('files', nargs='+', help='Input FITS files')
    parser_cr.add_argument('-o', '--output', help='Output file (single file mode)')
    parser_cr.add_argument('--output-dir', help='Output directory (batch mode)')
    parser_cr.add_argument('--method', default='sigma_clip',
                          choices=['sigma_clip', 'lacosmic', 'median'],
                          help='Detection method (default: sigma_clip)')
    parser_cr.add_argument('--sigma', type=float, default=5.0,
                          help='Detection threshold (default: 5.0)')
    parser_cr.add_argument('--iterations', type=int, default=1,
                          help='Number of iterations (default: 1)')
    parser_cr.add_argument('--save-mask', action='store_true',
                          help='Save cosmic ray mask as additional HDU')
    parser_cr.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser_cr.add_argument('--dry-run', action='store_true', help='Simulate without processing')
    parser_cr.set_defaults(func=cmd_cosmic_rays)
    
    # ========================================================================
    # Subcommand: batch
    # ========================================================================
    
    parser_batch = subparsers.add_parser(
        'batch',
        parents=[common],
        help='Batch process multiple files',
        description='Apply operations to multiple files with parallel processing.'
    )
    parser_batch.add_argument('files', nargs='+', help='Input FITS files')
    parser_batch.add_argument('--output-dir', required=True, help='Output directory')
    parser_batch.add_argument('--op-type', required=True,
                             choices=['unary', 'scalar', 'comparison', 'bitwise',
                                     'crop', 'bin', 'normalize', 'rotate', 'flip',
                                     'cosmic_rays'],
                             help='Operation type')
    parser_batch.add_argument('--operation', help='Operation name (e.g., log, add)')
    parser_batch.add_argument('--scalar', type=float, help='Scalar value')
    parser_batch.add_argument('--threshold', type=float, help='Threshold value')
    parser_batch.add_argument('--operand', type=int, help='Bitwise operand')
    parser_batch.add_argument('--skip-existing', action='store_true',
                             help='Skip files where output exists')
    parser_batch.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser_batch.add_argument('--dry-run', action='store_true', help='Simulate without processing')
    parser_batch.set_defaults(func=cmd_batch)
    
    # ========================================================================
    # Subcommand: stream
    # ========================================================================
    
    parser_stream = subparsers.add_parser(
        'stream',
        parents=[common],
        help='Memory-efficient streaming operations',
        description='Process large datasets with constant memory usage.'
    )
    parser_stream.add_argument('files', nargs='+', help='Input FITS files')
    parser_stream.add_argument('-o', '--output', required=True, help='Output file')
    parser_stream.add_argument('--method', default='median',
                              choices=['median', 'mean', 'min', 'max', 'stats'],
                              help='Streaming method (default: median)')
    parser_stream.add_argument('--quantiles', action='store_true',
                              help='Compute quantiles (for stats method)')
    parser_stream.add_argument('--dry-run', action='store_true', help='Simulate without processing')
    parser_stream.set_defaults(func=cmd_stream)
    
    # ========================================================================
    # Subcommand: quality
    # ========================================================================
    
    parser_quality = subparsers.add_parser(
        'quality',
        parents=[common],
        help='Quality control and assessment',
        description='Assess image quality and generate QC reports.'
    )
    parser_quality.add_argument('files', nargs='+', help='Input FITS files')
    parser_quality.add_argument('--frame-type', choices=['BIAS', 'DARK', 'FLAT', 'LIGHT'],
                               help='Expected frame type')
    parser_quality.add_argument('--report', help='Save report to file')
    parser_quality.set_defaults(func=cmd_quality)
    
    # ========================================================================
    # Subcommand: info
    # ========================================================================
    
    parser_info = subparsers.add_parser(
        'info',
        parents=[common],
        help='Display information',
        description='Show available operations, dependencies, and system info.'
    )
    info_group = parser_info.add_mutually_exclusive_group()
    info_group.add_argument('--list-operations', action='store_true',
                           help='List all available operations')
    info_group.add_argument('--check-dependencies', action='store_true',
                           help='Check dependency status')
    info_group.add_argument('--cosmic-ray-methods', action='store_true',
                           help='Show cosmic ray detection methods')
    parser_info.set_defaults(func=cmd_info)
    
    return parser


# ============================================================================
# Main Entry Point
# ============================================================================

def main(argv=None):
    """
    Main entry point for the CLI.
    
    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If None, uses sys.argv.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    # Execute subcommand
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        return handle_error(e, args.verbose)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'main',
    'create_parser',
]


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == '__main__':
    sys.exit(main())