import os
import csv
import logging
import numpy as np
from astropy.io import fits
from typing import List, Optional, Dict, Any

# Internal Project Imports
from fits_tools.constants import FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME
from fits_tools.statistics import get_basic_stats

# Configure Logging
logger = logging.getLogger(__name__)

def survey_fits(
    root_dir: str,
    output_path: str,
    keywords: Optional[List[str]] = None,
    recursive: bool = True,
    extension: int = 0
) -> None:
    """
    Recursively scans a directory for FITS files, calculates statistics, 
    and exports metadata to a CSV file.
    """
    if keywords is None:
        keywords = [FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME, 'FILTER', 'OBJECT']

    inventory: List[Dict[str, Any]] = []
    
    logger.info(f"Starting survey of: {root_dir}")

    for root, _, files in os.walk(root_dir):
        # Filter for FITS files
        fits_files = sorted([f for f in files if f.lower().endswith(('.fits', '.fit', '.fts'))])
        
        for filename in fits_files:
            file_path = os.path.join(root, filename)
            
            try:
                with fits.open(file_path) as hdul:
                    if len(hdul) <= extension:
                        logger.warning(f"Extension {extension} not found in {filename}")
                        continue
                        
                    header = hdul[extension].header
                    data = hdul[extension].data
                    
                    if data is None:
                        logger.warning(f"No image data in {filename} at extension {extension}")
                        continue

                    # Use your existing statistics module for core logic
                    stats = get_basic_stats(data)
                    
                    # Build the row dictionary
                    row: Dict[str, Any] = {
                        'filename': filename,
                        'rel_path': os.path.relpath(file_path, root_dir),
                        'mean': round(float(stats['mean']), 2),
                        'median': round(float(stats['median']), 2),
                        'std': round(float(stats['std']), 2),
                        'min': int(stats['min']),
                        'max': int(stats['max'])
                    }
                    
                    # Add requested metadata keywords from the header
                    for kw in keywords:
                        row[kw] = header.get(kw, 'N/A')
                        
                    inventory.append(row)
                    
            except Exception as e:
                logger.error(f"Could not process {filename}: {e}")

        if not recursive:
            break

    if not inventory:
        logger.warning("No valid FITS data found to survey.")
        return

    # Write results to CSV
    try:
        with open(output_path, 'w', newline='') as f:
            # Fieldnames are taken from the first entry
            writer = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
            writer.writeheader()
            writer.writerows(inventory)
        logger.info(f"Survey complete. Results saved to: {output_path}")
    except IOError as e:
        logger.error(f"Failed to write survey results: {e}")