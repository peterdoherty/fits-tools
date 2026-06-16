#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

# benchmarks/generate_synthetic_data.py

"""
Generate validated synthetic CCD data with known defects.

This dataset serves as ground truth for algorithm validation and
can be published alongside the software for reproducibility.

References
----------
Based on Janesick (2001) Sections 5.1-5.4 and realistic CCD characteristics
from Gaia, HST/ACS, and LSST Camera specifications.
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time
import json

class SyntheticCCD:
    """
    Generate realistic synthetic CCD frames with known defects.
    
    Parameters match typical scientific-grade CCDs:
    - e2v CCD250 (4k×4k, 15μm pixels)
    - Teledyne H4RG (4k×4k, 10μm pixels)
    
    Attributes
    ----------
    ny, nx : int
        Detector dimensions (pixels)
    gain : float
        Conversion gain (e-/ADU)
    read_noise : float
        Read noise (e- RMS)
    dark_current : float
        Median dark current (e-/pix/sec)
    qe : float
        Quantum efficiency (0-1)
    bias_level : float
        Bias/pedestal level (ADU)
    full_well : int
        Full well capacity (e-)
    
    Examples
    --------
    >>> ccd = SyntheticCCD(ny=2048, nx=2048, temp=-100)
    >>> bias = ccd.generate_bias(n_frames=10)
    >>> dark = ccd.generate_dark(exposure=300, n_frames=10)
    >>> flat = ccd.generate_flat(counts=30000, n_frames=10)
    """
    
    def __init__(
        self,
        ny: int = 2048,
        nx: int = 2048,
        gain: float = 1.5,
        read_noise: float = 3.0,
        dark_current: float = 0.01,
        qe: float = 0.85,
        bias_level: float = 1000.0,
        temp: float = -100.0,
        full_well: int = 100000,
    ):
        self.ny = ny
        self.nx = nx
        self.gain = gain
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.qe = qe
        self.bias_level = bias_level
        self.temp = temp
        self.full_well = full_well
        
        # Generate defect map (ground truth)
        self.true_defects = self._generate_realistic_defects()
    
    def _generate_realistic_defects(self) -> dict:
        """Generate realistic defect distribution based on Janesick Ch. 5."""
        defects = {
            'bias_defects': [],      # (y, x, offset_adu)
            'hot_pixels': [],        # (y, x, dark_rate_multiplier)
            'dead_pixels': [],       # (y, x, qe_fraction)
            'trap_pixels': [],       # (y, x, trap_depth_e)
            'bad_columns': [],       # column_index
            'bad_rows': [],          # row_index
            'warm_pixels': [],       # (y, x, temp_offset_K)
        }
        
        rng = np.random.default_rng(42)  # Reproducible
        
        # Bias defects: ~0.1% of pixels, ±100-500 ADU
        n_bias = int(0.001 * self.ny * self.nx)
        for _ in range(n_bias):
            y = rng.integers(0, self.ny)
            x = rng.integers(0, self.nx)
            offset = rng.uniform(-500, 500)
            defects['bias_defects'].append((y, x, offset))
        
        # Hot pixels: ~0.5% of pixels, 10-1000× normal dark current
        n_hot = int(0.005 * self.ny * self.nx)
        for _ in range(n_hot):
            y = rng.integers(0, self.ny)
            x = rng.integers(0, self.nx)
            multiplier = rng.lognormal(3.0, 1.5)  # Log-normal distribution
            defects['hot_pixels'].append((y, x, multiplier))
        
        # Dead pixels: ~0.3% of pixels, QE < 20%
        n_dead = int(0.003 * self.ny * self.nx)
        for _ in range(n_dead):
            y = rng.integers(0, self.ny)
            x = rng.integers(0, self.nx)
            qe_frac = rng.uniform(0.0, 0.2)
            defects['dead_pixels'].append((y, x, qe_frac))
        
        # Warm pixels: ~0.2% with elevated temp (higher dark current)
        n_warm = int(0.002 * self.ny * self.nx)
        for _ in range(n_warm):
            y = rng.integers(0, self.ny)
            x = rng.integers(0, self.nx)
            temp_offset = rng.uniform(5, 30)  # 5-30K warmer
            defects['warm_pixels'].append((y, x, temp_offset))
        
        # Bad columns: ~2-5 per detector (typical for 4k×4k)
        n_bad_cols = rng.integers(2, 6)
        defects['bad_columns'] = rng.choice(self.nx, n_bad_cols, replace=False).tolist()
        
        # Bad rows: ~1-3 per detector (less common)
        n_bad_rows = rng.integers(1, 4)
        defects['bad_rows'] = rng.choice(self.ny, n_bad_rows, replace=False).tolist()
        
        # Charge traps: ~0.1% of pixels (radiation damage)
        n_traps = int(0.001 * self.ny * self.nx)
        for _ in range(n_traps):
            y = rng.integers(0, self.ny)
            x = rng.integers(0, self.nx)
            trap_depth = rng.uniform(100, 5000)  # electrons
            defects['trap_pixels'].append((y, x, trap_depth))
        
        return defects
    
    def generate_bias(self, n_frames: int = 10) -> np.ndarray:
        """Generate bias frame stack with read noise and defects."""
        rng = np.random.default_rng()
        frames = []
        
        for _ in range(n_frames):
            # Base bias level + read noise
            frame = np.random.normal(
                self.bias_level,
                self.read_noise / self.gain,
                (self.ny, self.nx)
            )
            
            # Add bias defects
            for y, x, offset in self.true_defects['bias_defects']:
                frame[y, x] += offset
            
            # Add bad columns (elevated/depressed bias)
            for col in self.true_defects['bad_columns']:
                offset = rng.uniform(-200, 200)
                frame[:, col] += offset
            
            frames.append(frame)
        
        return np.array(frames)
    
    def generate_dark(self, exposure: float, n_frames: int = 10) -> np.ndarray:
        """Generate dark frame stack (bias-subtracted equivalent)."""
        rng = np.random.default_rng()
        frames = []
        
        for _ in range(n_frames):
            # Poisson dark current + read noise
            dark_e = np.random.poisson(
                self.dark_current * exposure,
                (self.ny, self.nx)
            )
            read_e = np.random.normal(0, self.read_noise, (self.ny, self.nx))
            
            frame = (dark_e + read_e) / self.gain
            
            # Add hot pixels
            for y, x, multiplier in self.true_defects['hot_pixels']:
                hot_e = np.random.poisson(
                    self.dark_current * exposure * multiplier
                )
                frame[y, x] = hot_e / self.gain
            
            # Add warm pixels (T-dependent dark current)
            for y, x, temp_offset in self.true_defects['warm_pixels']:
                # Dark current doubles every ~6-8K (Janesick Eq. 5.1)
                multiplier = 2 ** (temp_offset / 7.0)
                warm_e = np.random.poisson(
                    self.dark_current * exposure * multiplier
                )
                frame[y, x] = warm_e / self.gain
            
            # Bad columns have elevated dark current
            for col in self.true_defects['bad_columns']:
                multiplier = rng.uniform(5, 50)
                frame[:, col] *= multiplier
            
            frames.append(frame)
        
        return np.array(frames)
    
    def generate_flat(
        self, 
        counts: float = 30000, 
        n_frames: int = 10,
        vignetting: bool = True
    ) -> np.ndarray:
        """Generate flat field stack with realistic illumination."""
        rng = np.random.default_rng()
        frames = []
        
        # Create vignetting pattern (typical for telescopes)
        if vignetting:
            y, x = np.ogrid[:self.ny, :self.nx]
            cy, cx = self.ny // 2, self.nx // 2
            r = np.sqrt((y - cy)**2 + (x - cx)**2)
            r_max = np.sqrt(cy**2 + cx**2)
            # 30% vignetting at corners
            illumination = 1.0 - 0.3 * (r / r_max)**2
        else:
            illumination = np.ones((self.ny, self.nx))
        
        for _ in range(n_frames):
            # Poisson photon statistics
            signal_e = np.random.poisson(
                counts * self.qe * illumination,
                (self.ny, self.nx)
            )
            read_e = np.random.normal(0, self.read_noise, (self.ny, self.nx))
            
            frame = (signal_e + read_e) / self.gain
            
            # Apply dead pixels (low QE)
            for y, x, qe_frac in self.true_defects['dead_pixels']:
                frame[y, x] *= qe_frac
            
            # Apply charge traps (incomplete charge transfer)
            for y, x, trap_depth in self.true_defects['trap_pixels']:
                # Trap captures fraction of signal
                trapped_frac = min(trap_depth / counts, 0.5)
                frame[y, x] *= (1 - trapped_frac)
            
            # Bad columns have low/high QE
            for col in self.true_defects['bad_columns']:
                qe_mult = rng.uniform(0.1, 0.3)  # Very low QE
                frame[:, col] *= qe_mult
            
            frames.append(frame)
        
        return np.array(frames)
    
    def save_benchmark_dataset(self, output_dir: str = 'benchmark_data'):
        """Save complete benchmark dataset with ground truth."""
        from pathlib import Path
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating benchmark dataset...")
        
        # Generate frames
        bias_stack = self.generate_bias(n_frames=10)
        dark_stack = self.generate_dark(exposure=300, n_frames=10)
        flat_stack = self.generate_flat(counts=30000, n_frames=10)
        
        # Save as FITS
        print("Saving FITS files...")
        for i, frame in enumerate(bias_stack):
            hdu = fits.PrimaryHDU(frame)
            hdu.header['IMAGETYP'] = 'BIAS'
            hdu.header['EXPTIME'] = 0.0
            hdu.header['GAIN'] = self.gain
            hdu.header['RDNOISE'] = self.read_noise
            hdu.writeto(output_path / f'bias_{i:02d}.fits', overwrite=True)
        
        for i, frame in enumerate(dark_stack):
            hdu = fits.PrimaryHDU(frame)
            hdu.header['IMAGETYP'] = 'DARK'
            hdu.header['EXPTIME'] = 300.0
            hdu.header['GAIN'] = self.gain
            hdu.header['RDNOISE'] = self.read_noise
            hdu.writeto(output_path / f'dark_{i:02d}.fits', overwrite=True)
        
        for i, frame in enumerate(flat_stack):
            hdu = fits.PrimaryHDU(frame)
            hdu.header['IMAGETYP'] = 'FLAT'
            hdu.header['EXPTIME'] = 1.0
            hdu.header['GAIN'] = self.gain
            hdu.header['RDNOISE'] = self.read_noise
            hdu.writeto(output_path / f'flat_{i:02d}.fits', overwrite=True)
        
        # Save ground truth defects
        print("Saving ground truth...")
        with open(output_path / 'ground_truth.json', 'w') as f:
            json.dump(self.true_defects, f, indent=2)
        
        # Save detector parameters
        params = {
            'ny': self.ny,
            'nx': self.nx,
            'gain': self.gain,
            'read_noise': self.read_noise,
            'dark_current': self.dark_current,
            'qe': self.qe,
            'bias_level': self.bias_level,
            'temp': self.temp,
            'full_well': self.full_well,
        }
        with open(output_path / 'detector_params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"✓ Benchmark dataset saved to {output_path}")
        print(f"  - {len(bias_stack)} bias frames")
        print(f"  - {len(dark_stack)} dark frames")
        print(f"  - {len(flat_stack)} flat frames")
        print(f"  - Ground truth defect map")
        
        return str(output_path)


if __name__ == '__main__':
    # Generate benchmark dataset
    ccd = SyntheticCCD(ny=2048, nx=2048, temp=-100)
    ccd.save_benchmark_dataset('benchmark_data')