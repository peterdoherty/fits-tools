# cli.py (new file)

"""
Command-line interface for defect detection.
"""
import sys
import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np
from .detection import (
    create_calibration_bpm, 
    get_default_detection_config, 
    create_detection_report, 
    plot_defect_comparison
)
from .io import (
    save_calibration_bpm, 
    save_bpm_to_fits, 
    load_bpm_from_fits
)
from ..bitmask import get_bpm_statistics
import json
import logging

def cmd_create(args):
    """Handle 'create' command."""
    
    logging.info("Loading calibration frames...")
    bias = fits.getdata(args.bias)
    dark = fits.getdata(args.dark)
    flat = fits.getdata(args.flat)
    
    # Parse ROI
    roi = None
    if args.roi:
        try:
            roi = tuple(map(int, args.roi.split(',')))
            if len(roi) != 4:
                raise ValueError
        except Exception:
            logging.error(f"Invalid ROI format: {args.roi}. Expected: x1,x2,y1,y2")
            sys.exit(1)
    
    # Create config
    config = get_default_detection_config()
    config['bias']['threshold_adu'] = args.bias_threshold
    config['dark']['threshold_sigma'] = args.dark_sigma
    config['flat']['hot_threshold_percent'] = args.flat_hot
    config['flat']['cold_threshold_percent'] = args.flat_cold
    
    logging.info("Detecting defects...")
    bpm, stats = create_calibration_bpm(
        bias, dark, flat,
        config=config,
        stats_roi=roi
    )
    
    logging.info(f"Detected {stats['total_defects']} total defects "
                f"({stats['total_fraction']*100:.2f}%)")
    
    # Save BPM
    if args.save_masters:
        files = save_calibration_bpm(
            bpm, bias, dark, flat, stats,
            output_prefix=Path(args.output).stem
        )
        logging.info(f"Saved calibration set: {files}")
    else:
        save_bpm_to_fits(bpm, args.output, stats=stats)
        logging.info(f"Saved BPM to {args.output}")
    
    # Generate report
    if args.report:
        logging.info(f"Generating report: {args.report}")
        create_detection_report(bpm, stats, output_path=args.report, show_plots=False)


def cmd_analyze(args):
    """Handle 'analyze' command."""
    
    logging.info(f"Loading BPM: {args.bpm}")
    bpm, metadata = load_bpm_from_fits(args.bpm)
    
    # Get statistics
    stats = get_bpm_statistics(bpm)
    
    # Print summary
    print("\n" + "="*70)
    print("BPM ANALYSIS SUMMARY")
    print("="*70)
    print(f"File: {args.bpm}")
    print(f"Shape: {bpm.shape}")
    print(f"Dtype: {bpm.dtype}")

    total_bad = int(np.count_nonzero(bpm))
    total_fraction = total_bad / bpm.size

    print(f"Total bad pixels: {total_bad} "
        f"({total_fraction*100:.3f}%)")

    print("\nDefect breakdown:")
    for name, info in stats.items():
        if info['count'] > 0:
            print(f"  {name:15s}: {info['count']:6d} ({info['fraction']*100:.3f}%)")            
    print("="*70 + "\n")
    
    # Export JSON
    if args.json:
        output = {
            'metadata': metadata,
            'statistics': stats
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        logging.info(f"Exported statistics to {args.json}")
    
    # Generate report
    if args.report:
        logging.error(
            "Report generation for 'analyze' is not implemented yet; "
            "reports currently require the full detection stats from create_calibration_bpm()."
        )
        sys.exit(1)


def cmd_compare(args):
    """Handle 'compare' command."""
    bpms = []
    labels = args.labels if args.labels else [Path(f).stem for f in args.bpms]
    
    if len(labels) != len(args.bpms):
        logging.error(f"Number of labels ({len(labels)}) must match number of BPMs ({len(args.bpms)})")
        sys.exit(1)
    
    for filename in args.bpms:
        logging.info(f"Loading: {filename}")
        bpm, _ = load_bpm_from_fits(filename)
        bpms.append(bpm)
    
    # Print comparison table
    print("\n" + "="*70)
    print("BPM COMPARISON")
    print("="*70)
    print(f"{'Label':<20} {'Shape':<15} {'Bad Pixels':<15} {'Fraction':<10}")
    print("-"*70)
    for label, bpm in zip(labels, bpms):
        count = np.sum(bpm > 0)
        frac = count / bpm.size
        print(f"{label:<20} {str(bpm.shape):<15} {count:<15} {frac*100:>6.2f}%")
    print("="*70 + "\n")
    
    # Generate comparison plot
    if args.report:
        logging.warning(
            "--report for 'compare' is not yet implemented; "
            "showing interactive comparison only."
        )
    plot_defect_comparison(bpms, labels)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CCD/CMOS defect detection and bad pixel mask generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  fits-defects create --bias bias.fits --dark dark.fits --flat flat.fits --output bpm.fits
  
  # Custom thresholds
  fits-defects create --bias bias.fits --dark dark.fits --flat flat.fits \\
      --bias-threshold 150 --dark-sigma 7.0 --output bpm.fits
  
  # With ROI and report
  fits-defects create --bias bias.fits --dark dark.fits --flat flat.fits \\
      --roi 100,1900,100,1900 --report bpm_report.pdf --output bpm.fits
  
  # Analyze existing BPM
  fits-defects analyze --bpm bpm.fits
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # CREATE command
    create_parser = subparsers.add_parser('create', help='Create BPM from calibration frames')
    create_parser.add_argument('--bias', required=True, help='Master bias FITS file')
    create_parser.add_argument('--dark', required=True, help='Master dark FITS file')
    create_parser.add_argument('--flat', required=True, help='Master flat FITS file')
    create_parser.add_argument('--output', '-o', required=True, help='Output BPM FITS file')
    create_parser.add_argument('--roi', help='Stats ROI as x1,x2,y1,y2')
    create_parser.add_argument('--bias-threshold', type=float, default=100.0,
                               help='Bias threshold (ADU)')
    create_parser.add_argument('--dark-sigma', type=float, default=5.0,
                               help='Dark hot pixel threshold (sigma)')
    create_parser.add_argument('--flat-hot', type=float, default=20.0,
                               help='Flat hot threshold (percent)')
    create_parser.add_argument('--flat-cold', type=float, default=20.0,
                               help='Flat cold threshold (percent)')
    create_parser.add_argument('--save-masters', action='store_true',
                               help='Save master calibration frames')
    create_parser.add_argument('--verbose', '-v', action='store_true')
    
    # ANALYZE command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing BPM')
    analyze_parser.add_argument('--bpm', required=True, help='BPM FITS file')
    analyze_parser.add_argument(
        '--report',
        help='(Reserved for future use) Generate PDF report from full detection stats'
    )
    analyze_parser.add_argument('--json', help='Export statistics as JSON')
    analyze_parser.add_argument('--verbose', '-v', action='store_true')
    
    # COMPARE command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple BPMs')
    compare_parser.add_argument('--bpms', nargs='+', required=True, help='BPM FITS files')
    compare_parser.add_argument('--labels', nargs='+', help='Labels for each BPM')
    compare_parser.add_argument('--report', help='Generate PDF comparison report')
    compare_parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    if args.command == 'create':
        cmd_create(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()

    