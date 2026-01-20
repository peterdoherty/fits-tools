#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for selecting and filtering FITS files.

This module provides utilities to select FITS files based on various criteria
including directory scanning, glob patterns, and header keyword filtering.
Particularly useful for batch processing of CCD/CMOS calibration frames.

Features:
    - Select files from directories, glob patterns, or lists
    - Filter by FITS header keywords
    - Recursive directory scanning
    - Convenience functions for bias, dark, flat frames
    - Maximum file limits for large datasets

Version History:
    0.1.0 - Initial release
    0.2.0 - Refactored to use fits_tools modules, improved error handling,
            added doctest examples, removed code duplication
"""

import argparse
import doctest
import glob
import logging
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Union

from astropy.io import fits

from fits_tools.constants import (
    FITS_KEY_IMGTYPE, 
    FITS_VAL_BIAS_IMGTYPE, 
    FITS_VAL_DARK_IMGTYPE,
    FITS_VAL_FLAT_IMGTYPE,
    FITS_VAL_LIGHT_IMGTYPE)

from fits_tools.exceptions import SelectionError
from fits_tools.utils import (
    is_fits_file,
)

# Module logger
logger = logging.getLogger(__name__)

def _normalize_for_compare(value: Any) -> Any:
    """Normalize values for keyword comparison.

    Strings are stripped and uppercased for case-insensitive matching.
    Other types are returned unchanged.
    """
    if isinstance(value, str):
        return value.strip().upper()
    return value


__version__ = "0.2.0"


# =============================================================================
# Main Selection Function
# =============================================================================

def select_files(
    input_param: Union[str, List[str]], 
    keys: Optional[Dict[str, Any]] = None, 
    recursive: bool = False, 
    verbose: bool = False, 
    max_files: Optional[int] = None
) -> List[str]:
    """
    Select FITS files based on input parameter and optional keyword/value pairs.

    This function can handle various input types: a list of file paths, a directory,
    or a glob pattern. It filters files based on FITS header keywords and can limit
    the number of returned files.

    Args:
        input_param: Either a list of FITS files, a directory path, or a search
            string (glob pattern).
        keys: Dictionary of FITS header keyword/value pairs to filter files.
            Values in the dictionary can be:
            - A single value (str, int, float, bool) for an exact match.
            - A list of values (e.g., ['BIAS', 'DARK']) to match if the header
              keyword's value is any one of the values in the list.
        recursive: If True, search recursively through subdirectories when
            input_param is a directory or a glob pattern.
        verbose: If True, produce diagnostic text output.
        max_files: If provided, limit the number of files returned to this value.
            If more files match the criteria than max_files, only the first
            max_files found will be returned.

    Returns:
        List of full paths to FITS files that match the criteria, up to the
        specified max_files limit. Returns an empty list if no files are found.

    Raises:
        ValueError: If max_files is provided but is not a positive integer.
        SelectionError: If the initial input_param (when it's a single string) is
            not a valid FITS file or directory.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> # Create temporary directory with test files
        >>> tmpdir = tempfile.mkdtemp()
        >>> # Create a BIAS file
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['IMGTYPE'] = 'BIAS'
        >>> hdu.writeto(os.path.join(tmpdir, 'bias.fits'), overwrite=True)
        >>> # Create a DARK file
        >>> hdu.header['IMGTYPE'] = 'DARK'
        >>> hdu.writeto(os.path.join(tmpdir, 'dark.fits'), overwrite=True)
        >>> # Select all files
        >>> files = select_files(tmpdir)
        >>> len(files)
        2
        >>> # Select only BIAS files
        >>> bias_files = select_files(tmpdir, keys={'IMGTYPE': 'BIAS'})
        >>> len(bias_files)
        1
        >>> # Cleanup
        >>> shutil.rmtree(tmpdir)
    """
    # Initialize list to store matching files
    matching_files = []
    
    # Ensure max_files is a positive integer if provided
    if max_files is not None:
        if not isinstance(max_files, int) or max_files <= 0:
            raise ValueError("The 'max_files' parameter must be a positive integer if provided.")

    # Step 1: Get initial list of files based on input_param
    potential_files_iterator = []
    
    if isinstance(input_param, list):
        # Input is already a list of files
        potential_files_iterator = input_param
        if verbose:
            logger.info(f"Processing list of {len(potential_files_iterator)} files")
            
    elif isinstance(input_param, str):
        if os.path.isdir(input_param):
            # Input is a directory
            if verbose:
                logger.info(
                    f"Processing directory: {input_param}"
                    f"{' recursively' if recursive else ''}"
                )
            
            if recursive:
                # Use os.walk for recursive search
                potential_files_iterator = (
                    os.path.join(root, file)
                    for root, _, files in os.walk(input_param)
                    for file in files if is_fits_file(file)
                )
            else:
                # Non-recursive search in a directory
                potential_files_iterator = (
                    os.path.join(input_param, file)
                    for file in os.listdir(input_param)
                    if is_fits_file(file)
                )
                
        elif '*' in input_param or '?' in input_param:
            # Input is a glob pattern
            if verbose:
                logger.info(f"Processing glob pattern: {input_param}")
            
            if recursive:
                # glob.iglob with recursive=True
                potential_files_iterator = glob.iglob(input_param, recursive=True)
            else:
                potential_files_iterator = glob.iglob(input_param)
                
        else:
            # Input is a single file path - validate existence and FITS extension
            if os.path.isfile(input_param) and is_fits_file(input_param):
                potential_files_iterator = [input_param]
                if verbose:
                    logger.info(f"Processing single file: {input_param}")
            else:
                # Raise an error if it's a single string but not a valid FITS file/directory
                raise SelectionError(
                    f"Input '{input_param}' is not a valid FITS file, directory, or glob pattern."
                )
    else:
        raise SelectionError(
            "Invalid input_param type. Must be string or list of strings."
        )

    # Step 2: Filter files based on FITS header keywords if keys are provided,
    # and apply max_files limit
    files_processed_count = 0
    
    for file_path in potential_files_iterator:
        # Check if we've reached the limit
        if max_files is not None and len(matching_files) >= max_files:
            if verbose:
                logger.info(f"Reached max_files limit ({max_files}). Stopping file processing.")
            break

        # Ensure the file is a FITS file before attempting to open
        if not is_fits_file(file_path):
            if verbose:
                logger.debug(f"Skipping non-FITS file: {file_path}")
            continue

        try:
            # Open the FITS file and check header keywords
            with fits.open(file_path, memmap=True) as hdul:
                # Assume primary HDU for filtering
                header = hdul[0].header

                if keys is None:
                    # No keyword filtering needed, add directly
                    matching_files.append(file_path)
                    if verbose:
                        logger.debug(f"Match (no keyword filter): {file_path}")
                else:
                    # Check if all keys match
                    all_keys_match = True
                    
                    for key, value in keys.items():
                        # Convert key to uppercase for comparison
                        header_key_upper = key.upper()

                        if header_key_upper not in header:
                            all_keys_match = False
                            if verbose:
                                logger.debug(
                                    f"  No match for {file_path}: "
                                    f"Keyword '{key}' not found in header."
                                )
                            break
                        
                        header_value = _normalize_for_compare(header[header_key_upper])

                        if isinstance(value, list):
                            # If the expected value is a list, check if header_value is in that list
                            expected_list = [_normalize_for_compare(v) for v in value]
                            if header_value not in expected_list:
                                all_keys_match = False
                                if verbose:
                                    logger.debug(
                                        f"  No match for {file_path}: "
                                        f"Value '{header_value}' for '{key}' "
                                        f"not in expected list {value}."
                                    )
                                break
                        else:
                            # Otherwise, perform exact equality check
                            expected_value = _normalize_for_compare(value)
                            if header_value != expected_value:
                                all_keys_match = False
                                if verbose:
                                    logger.debug(
                                        f"  No match for {file_path}: "
                                        f"Value '{header_value}' for '{key}' "
                                        f"does not match expected '{value}'."
                                    )
                                break
                    
                    if all_keys_match:
                        matching_files.append(file_path)
                        if verbose:
                            logger.debug(f"Match: {file_path}")
                    else:
                        if verbose:
                            logger.debug(f"No match for {file_path} based on keywords.")
                            
            files_processed_count += 1
            
        except FileNotFoundError:
            if verbose:
                logger.warning(
                    f"Error: File not found during processing - {file_path}. "
                    f"Skipping this file."
                )
            files_processed_count += 1
            continue
            
        except (OSError, fits.verify.VerifyError) as e:
            # Catch FITS-specific errors
            if verbose:
                logger.warning(
                    f"Error processing {file_path}: Not a valid FITS file or "
                    f"other I/O issue: {e}. Skipping this file."
                )
            files_processed_count += 1
            continue
            
        except Exception as e:
            if verbose:
                logger.warning(
                    f"An unexpected error occurred while processing {file_path}: {e}. "
                    f"Skipping this file.",
                    exc_info=True
                )
            files_processed_count += 1
            continue
    
    if verbose:
        logger.info(f"Found {len(matching_files)} matching files out of {files_processed_count} processed.")
    
    return matching_files


# =============================================================================
# Convenience Functions for Common Frame Types
# =============================================================================

def select_bias_files(
    input_param: Union[str, List[str]], 
    recursive: bool = False, 
    verbose: bool = False, 
    max_files: Optional[int] = None
) -> List[str]:
    """
    Select FITS bias files. Convenience wrapper around select_files.

    Args:
        input_param: Either a list of FITS files, a directory path, or a search
            string (glob pattern).
        recursive: If True, search recursively through subdirectories.
        verbose: If True, produce diagnostic text output.
        max_files: If provided, limit the number of files returned to this value.

    Returns:
        List of full paths to FITS bias files.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> tmpdir = tempfile.mkdtemp()
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['IMGTYPE'] = 'BIAS'
        >>> hdu.writeto(os.path.join(tmpdir, 'bias.fits'), overwrite=True)
        >>> files = select_bias_files(tmpdir)
        >>> len(files)
        1
        >>> shutil.rmtree(tmpdir)
    """
    if verbose:
        logger.info(
            f"select_bias_files: Calling select_files with "
            f"keys={{'{FITS_KEY_IMGTYPE}': '{FITS_VAL_BIAS_IMGTYPE}'}}"
        )
    
    return select_files(
        input_param=input_param,
        keys={FITS_KEY_IMGTYPE: FITS_VAL_BIAS_IMGTYPE},
        recursive=recursive,
        verbose=verbose,
        max_files=max_files
    )


def select_dark_files(
    input_param: Union[str, List[str]], 
    recursive: bool = False, 
    verbose: bool = False, 
    max_files: Optional[int] = None
) -> List[str]:
    """
    Select FITS dark files. Convenience wrapper around select_files.

    Args:
        input_param: Either a list of FITS files, a directory path, or a search
            string (glob pattern).
        recursive: If True, search recursively through subdirectories.
        verbose: If True, produce diagnostic text output.
        max_files: If provided, limit the number of files returned to this value.

    Returns:
        List of full paths to FITS dark files.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> tmpdir = tempfile.mkdtemp()
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['IMGTYPE'] = 'DARK'
        >>> hdu.writeto(os.path.join(tmpdir, 'dark.fits'), overwrite=True)
        >>> files = select_dark_files(tmpdir)
        >>> len(files)
        1
        >>> shutil.rmtree(tmpdir)
    """
    if verbose:
        logger.info(
            f"select_dark_files: Calling select_files with "
            f"keys={{'{FITS_KEY_IMGTYPE}': '{FITS_VAL_DARK_IMGTYPE}'}}"
        )
    
    return select_files(
        input_param=input_param,
        keys={FITS_KEY_IMGTYPE: FITS_VAL_DARK_IMGTYPE},
        recursive=recursive,
        verbose=verbose,
        max_files=max_files
    )


def select_flat_files(
    input_param: Union[str, List[str]], 
    recursive: bool = False, 
    verbose: bool = False, 
    max_files: Optional[int] = None
) -> List[str]:
    """
    Select FITS flat files. Convenience wrapper around select_files.

    Args:
        input_param: Either a list of FITS files, a directory path, or a search
            string (glob pattern).
        recursive: If True, search recursively through subdirectories.
        verbose: If True, produce diagnostic text output.
        max_files: If provided, limit the number of files returned to this value.

    Returns:
        List of full paths to FITS flat files.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> tmpdir = tempfile.mkdtemp()
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['IMGTYPE'] = 'FLAT'
        >>> hdu.writeto(os.path.join(tmpdir, 'flat.fits'), overwrite=True)
        >>> files = select_flat_files(tmpdir)
        >>> len(files)
        1
        >>> shutil.rmtree(tmpdir)
    """
    if verbose:
        logger.info(
            f"select_flat_files: Calling select_files with "
            f"keys={{'{FITS_KEY_IMGTYPE}': '{FITS_VAL_FLAT_IMGTYPE}'}}"
        )
    
    return select_files(
        input_param=input_param,
        keys={FITS_KEY_IMGTYPE: FITS_VAL_FLAT_IMGTYPE},
        recursive=recursive,
        verbose=verbose,
        max_files=max_files
    )


def select_light_files(
    input_param: Union[str, List[str]], 
    recursive: bool = False, 
    verbose: bool = False, 
    max_files: Optional[int] = None
) -> List[str]:
    """
    Select FITS light/science files. Convenience wrapper around select_files.

    Args:
        input_param: Either a list of FITS files, a directory path, or a search
            string (glob pattern).
        recursive: If True, search recursively through subdirectories.
        verbose: If True, produce diagnostic text output.
        max_files: If provided, limit the number of files returned to this value.

    Returns:
        List of full paths to FITS light/science files.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> tmpdir = tempfile.mkdtemp()
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['IMGTYPE'] = 'LIGHT'
        >>> hdu.writeto(os.path.join(tmpdir, 'light.fits'), overwrite=True)
        >>> files = select_light_files(tmpdir)
        >>> len(files)
        1
        >>> shutil.rmtree(tmpdir)
    """
    if verbose:
        logger.info(
            f"select_light_files: Calling select_files with "
            f"keys={{'{FITS_KEY_IMGTYPE}': '{FITS_VAL_LIGHT_IMGTYPE}'}}"
        )
    
    return select_files(
        input_param=input_param,
        keys={FITS_KEY_IMGTYPE: FITS_VAL_LIGHT_IMGTYPE},
        recursive=recursive,
        verbose=verbose,
        max_files=max_files
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main function
    'select_files',
    
    # Convenience functions
    'select_bias_files',
    'select_dark_files',
    'select_flat_files',
    'select_light_files',
]


# =============================================================================
# Main / Testing
# =============================================================================

def main():
    """
    Command-line interface for FITS file selection utilities and testing.
    """
    parser = argparse.ArgumentParser(
        description='FITS File Selection Utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run doctests
  %(prog)s --test
  
  # Select all FITS files in directory
  %(prog)s --dir /path/to/fits
  
  # Select BIAS files recursively
  %(prog)s --dir /path/to/fits --type BIAS --recursive
  
  # Select with custom keyword filter
  %(prog)s --dir /path/to/fits --filter EXPTIME=10.0
        """
    )
    
    parser.add_argument('--test', action='store_true', help='Run doctests')
    parser.add_argument('--dir', help='Directory to search for FITS files')
    parser.add_argument('--type', choices=['BIAS', 'DARK', 'FLAT', 'LIGHT'],
                       help='Select specific frame type')
    parser.add_argument('--filter', action='append',
                       help='Filter by keyword (format: KEYWORD=VALUE)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search recursively through subdirectories')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to return')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run doctests
    if args.test:
        logger.info("Running doctests for selection.py...")
        result = doctest.testmod(verbose=args.verbose)
        
        if result.failed == 0:
            logger.info(f"All {result.attempted} doctests passed!")
            return 0
        else:
            logger.error(f"{result.failed} of {result.attempted} doctests failed!")
            return 1
    
    # Select files
    if args.dir:
        # Build filter dictionary from --filter arguments
        keys = {}
        if args.filter:
            for f in args.filter:
                if '=' not in f:
                    print(f"Error: Invalid filter format '{f}'. Use KEYWORD=VALUE")
                    return 1
                key, value = f.split('=', 1)
                # Try to convert value to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
                keys[key] = value
        
        # Select based on type if specified
        if args.type == 'BIAS':
            files = select_bias_files(
                args.dir,
                recursive=args.recursive,
                verbose=args.verbose,
                max_files=args.max_files
            )
        elif args.type == 'DARK':
            files = select_dark_files(
                args.dir,
                recursive=args.recursive,
                verbose=args.verbose,
                max_files=args.max_files
            )
        elif args.type == 'FLAT':
            files = select_flat_files(
                args.dir,
                recursive=args.recursive,
                verbose=args.verbose,
                max_files=args.max_files
            )
        elif args.type == 'LIGHT':
            files = select_light_files(
                args.dir,
                recursive=args.recursive,
                verbose=args.verbose,
                max_files=args.max_files
            )
        else:
            # General selection
            files = select_files(
                args.dir,
                keys=keys if keys else None,
                recursive=args.recursive,
                verbose=args.verbose,
                max_files=args.max_files
            )
        
        # Print results
        print(f"\nFound {len(files)} matching files:")
        for f in files:
            print(f"  {f}")
        
        return 0
    
    # No arguments
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

    
