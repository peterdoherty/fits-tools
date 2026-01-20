import os
import csv
import logging
from typing import List, Optional

# Internal Project Imports
from .constants import FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME
from .statistics import stats
from .io import get_header

logger = logging.getLogger(__name__)

def survey_fits(
    root_dir: str,
    output_path: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    recursive: bool = True,
    extension: int = 0,
    split_by_dir: bool = False,
    overwrite: bool = False  # Added overwrite toggle
) -> None:
    """
    Scans directories and writes FITS metadata to CSV.
    """
    if keywords is None:
        keywords = [FITS_KEY_IMAGETYP, FITS_KEY_EXPTIME, 'FILTER', 'OBJECT']

    # Fieldnames for CSV output
    fieldnames = ['filename', 'mean', 'median', 'std', 'min', 'max', 'pixels'] + keywords
    
    # Determine the centralized reports folder if it exists
    base_report_dir = os.path.join(root_dir, "reports") if os.path.exists(os.path.join(root_dir, "reports")) else root_dir

    for root, dirs, files in os.walk(root_dir):
        # Prevent the script from scanning the reports folder itself
        if "reports" in root:
            continue
            
        fits_files = sorted([f for f in files if f.lower().endswith(('.fits', '.fit', '.fts'))])
        if not fits_files:
            continue

        # Handle split file naming
        if split_by_dir:
            dir_name = os.path.basename(root)
            current_output = os.path.join(base_report_dir, f"{dir_name}_inventory.csv")
        else:
            current_output = output_path if output_path else os.path.join(base_report_dir, "master_inventory.csv")

        # Skip logic: If file exists and overwrite is False, move to next directory
        if os.path.exists(current_output) and not overwrite:
            logger.info(f"Skipping directory; inventory already exists: {current_output}")
            continue

        try:
            mode = 'w' # We overwrite for fresh inventories
            with open(current_output, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for filename in fits_files:
                    file_path = os.path.join(root, filename)
                    try:
                        res = stats(file_path, ext=extension, stats_type='simple')
                        if res is None: continue
                        
                        header = get_header(file_path, ext=extension)
                        
                        # Corrected Mapping to StatsResult attributes 
                        row = {
                            'filename': filename,
                            'mean': f"{res.mean_val:.2f}",
                            'median': "N/A",  # simple_stats sets this to NaN
                            'std': f"{res.std_val:.2f}",
                            'min': f"{res.min_val:.2f}",
                            'max': f"{res.max_val:.2f}",
                            'pixels': res.num_pixels
                        }
                        
                        for kw in keywords:
                            row[kw] = header.get(kw, 'N/A')
                        
                        writer.writerow(row)
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                
                f.flush()
                logger.info(f"Inventory saved: {current_output}")

        except IOError as e:
            logger.error(f"File I/O Error for {current_output}: {e}")

        if not recursive:
            break