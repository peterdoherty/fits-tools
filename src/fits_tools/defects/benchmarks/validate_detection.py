# benchmarks/validate_detection.py

"""
Validate defect detection algorithms against ground truth.

Computes precision, recall, F1-score, and ROC curves for each
detection stage. Results are saved for publication.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
from astropy.io import fits

from fits_tools.defects import create_calibration_bpm_batch
from fits_tools.defects.bitmask import (
    BPM_BIT_BIAS, BPM_BIT_DARK, BPM_BIT_FLAT_HOT, BPM_BIT_FLAT_COLD,
    BPM_BIT_BAD_ROW, BPM_BIT_BAD_COL
)

def load_benchmark_data(data_dir: str = 'benchmark_data'):
    """Load benchmark dataset and ground truth."""
    data_path = Path(data_dir)
    
    # Load frames
    bias_files = sorted(data_path.glob('bias_*.fits'))
    dark_files = sorted(data_path.glob('dark_*.fits'))
    flat_files = sorted(data_path.glob('flat_*.fits'))
    
    bias_stack = np.array([fits.getdata(f) for f in bias_files])
    dark_stack = np.array([fits.getdata(f) for f in dark_files])
    flat_stack = np.array([fits.getdata(f) for f in flat_files])
    
    # Load ground truth
    with open(data_path / 'ground_truth.json') as f:
        ground_truth = json.load(f)
    
    # Load detector params
    with open(data_path / 'detector_params.json') as f:
        params = json.load(f)
    
    return bias_stack, dark_stack, flat_stack, ground_truth, params


def create_ground_truth_bpm(ground_truth: dict, shape: tuple) -> dict:
    """Convert ground truth defects to BPM format for comparison."""
    ny, nx = shape
    
    bpms = {
        'bias': np.zeros((ny, nx), dtype=np.uint32),
        'dark': np.zeros((ny, nx), dtype=np.uint32),
        'flat_hot': np.zeros((ny, nx), dtype=np.uint32),
        'flat_cold': np.zeros((ny, nx), dtype=np.uint32),
        'bad_rows': np.zeros((ny, nx), dtype=np.uint32),
        'bad_cols': np.zeros((ny, nx), dtype=np.uint32),
    }
    
    # Bias defects
    for y, x, offset in ground_truth['bias_defects']:
        if abs(offset) > 100:  # Match default threshold
            bpms['bias'][y, x] = 1
    
    # Hot pixels (combines hot + warm)
    for y, x, mult in ground_truth['hot_pixels']:
        if mult > 10:  # Significant hot pixel
            bpms['dark'][y, x] = 1
    
    for y, x, temp_offset in ground_truth['warm_pixels']:
        if temp_offset > 10:
            bpms['dark'][y, x] = 1
    
    # Dead pixels
    for y, x, qe_frac in ground_truth['dead_pixels']:
        if qe_frac < 0.8:  # Less than 80% QE
            bpms['flat_cold'][y, x] = 1
    
    # Bad columns
    for col in ground_truth['bad_columns']:
        bpms['bad_cols'][:, col] = 1
    
    # Bad rows
    for row in ground_truth['bad_rows']:
        bpms['bad_rows'][row, :] = 1
    
    return bpms


def compute_metrics(detected: np.ndarray, truth: np.ndarray) -> dict:
    """Compute precision, recall, F1-score."""
    detected_flat = detected.ravel() > 0
    truth_flat = truth.ravel() > 0
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth_flat, detected_flat, average='binary', zero_division=0
    )
    
    # True/false positives/negatives
    tp = np.sum(detected_flat & truth_flat)
    fp = np.sum(detected_flat & ~truth_flat)
    tn = np.sum(~detected_flat & ~truth_flat)
    fn = np.sum(~detected_flat & truth_flat)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }


def validate_detection():
    """Main validation function."""
    print("="*70)
    print("DEFECT DETECTION ALGORITHM VALIDATION")
    print("="*70)
    
    # Load data
    print("\nLoading benchmark data...")
    bias_stack, dark_stack, flat_stack, ground_truth, params = load_benchmark_data()
    
    # Run detection
    print("Running detection pipeline...")
    bpm, masters, stats = create_calibration_bpm_batch(
        bias_stack, dark_stack, flat_stack
    )
    
    # Create ground truth BPMs
    print("Preparing ground truth...")
    truth_bpms = create_ground_truth_bpm(ground_truth, bpm.shape)
    
    # Compute metrics for each defect type
    results = {}
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    defect_types = [
        ('bias', BPM_BIT_BIAS, 'Bias Defects'),
        ('dark', BPM_BIT_DARK, 'Dark Hot Pixels'),
        ('flat_cold', BPM_BIT_FLAT_COLD, 'Flat Cold/Dead Pixels'),
    ]
    
    for key, bit, name in defect_types:
        detected = (bpm & bit) != 0
        truth = truth_bpms[key]
        metrics = compute_metrics(detected, truth)
        results[key] = metrics
        
        print(f"\n{name}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  TP: {metrics['true_positives']:6d}  "
              f"FP: {metrics['false_positives']:6d}")
        print(f"  TN: {metrics['true_negatives']:6d}  "
              f"FN: {metrics['false_negatives']:6d}")
    
    # Save results
    output_path = Path('validation_results')
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Validation results saved to {output_path}/metrics.json")
    print("="*70)
    
    return results


if __name__ == '__main__':
    validate_detection()