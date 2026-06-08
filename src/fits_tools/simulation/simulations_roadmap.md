# Simulations Subpackage - Future Enhancements Roadmap

## Current State (v0.3.0)
The existing `simulations.py` module provides:
- ✅ Basic calibration frames (bias, dark, flat, science)
- ✅ Realistic noise models (read noise, photon noise)
- ✅ Detector artifacts (hot pixels, cosmic rays, CTE trails, tree rings)
- ✅ Multi-Extension FITS (MEF) support
- ✅ Time-series generation
- ✅ Overscan region simulation

## Immediate Priority: Refactoring (v0.4.0)
Convert `simulations.py` → `simulations/` subpackage structure:
```
simulations/
├── __init__.py
├── noise.py          # Noise models
├── artifacts.py      # Detector artifacts
├── frames.py         # Frame creation
├── mef.py           # Multi-Extension FITS
├── timeseries.py    # Time series
└── utils.py         # Helpers
```

---

## Future Enhancements

### Phase 1: Spectroscopic & Atmospheric (v0.5.0)
**Your existing code to integrate:**

#### 1. Atmospheric Transmission (`atmosphere.py`)
```python
"""
Atmospheric transmission modeling for ground-based observations.

Features:
- Wavelength-dependent atmospheric extinction
- Airmass calculations
- Standard atmospheric models (US Standard, tropical, etc.)
- Water vapor absorption bands
- Rayleigh and Mie scattering
- Telluric line absorption (O2, H2O, O3)
"""

def calculate_atmospheric_transmission(
    wavelength: np.ndarray,
    airmass: float = 1.0,
    pwv: float = 5.0,  # precipitable water vapor in mm
    aerosol_optical_depth: float = 0.05,
    site: str = 'generic'
) -> np.ndarray:
    """
    Calculate atmospheric transmission as function of wavelength.
    
    Parameters
    ----------
    wavelength : array
        Wavelength in Angstroms
    airmass : float
        Airmass (1.0 = zenith)
    pwv : float
        Precipitable water vapor in mm
    aerosol_optical_depth : float
        Aerosol optical depth at 550nm
    site : str
        Observatory site ('mauna_kea', 'la_silla', 'paranal', 'generic')
    
    Returns
    -------
    transmission : array
        Atmospheric transmission (0-1) at each wavelength
    """
    pass

def get_standard_atmosphere(site: str = 'generic') -> dict:
    """Return standard atmospheric parameters for observatory."""
    pass
```

#### 2. Filter Transmission (`filters.py`)
```python
"""
Standard photometric filter transmission curves.

Features:
- Johnson-Cousins (UBVRI)
- Sloan Digital Sky Survey (ugriz)
- 2MASS (JHKs)
- Pan-STARRS (grizy)
- HST filters (WFC3, ACS)
- JWST filters
- Custom filter support
"""

def get_filter_transmission(
    filter_name: str,
    wavelength: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get filter transmission curve.
    
    Parameters
    ----------
    filter_name : str
        Filter identifier (e.g., 'SDSS_g', 'Johnson_V', 'F606W')
    wavelength : array, optional
        Wavelength grid for interpolation
    
    Returns
    -------
    wavelength : array
        Wavelength in Angstroms
    transmission : array
        Filter transmission (0-1)
    """
    pass

def calculate_effective_wavelength(
    wavelength: np.ndarray,
    transmission: np.ndarray,
    spectrum: Optional[np.ndarray] = None
) -> float:
    """Calculate effective wavelength of filter."""
    pass

def apply_filter_to_spectrum(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    filter_name: str,
    include_atmosphere: bool = False,
    airmass: float = 1.0
) -> float:
    """
    Calculate observed flux through filter.
    
    Combines:
    - Source spectrum
    - Filter transmission
    - Atmospheric transmission (optional)
    - Detector QE (optional)
    """
    pass
```

#### 3. Spectral Sources (`spectra.py`)
```python
"""
Synthetic astronomical spectra.

Features:
- Blackbody spectra (stars, planets)
- Stellar spectral types (O, B, A, F, G, K, M)
- Galaxy SEDs (elliptical, spiral, starburst)
- QSO/AGN spectra
- Sky emission lines
- Emission line objects
"""

def create_blackbody_spectrum(
    temperature: float,
    wavelength: np.ndarray
) -> np.ndarray:
    """Create blackbody spectrum (Planck function)."""
    pass

def get_stellar_spectrum(
    spectral_type: str,
    wavelength: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get template stellar spectrum.
    
    Parameters
    ----------
    spectral_type : str
        Spectral type (e.g., 'A0V', 'G2V', 'M5III')
    
    Returns
    -------
    wavelength : array
        Wavelength in Angstroms
    flux : array
        Flux (erg/s/cm^2/Angstrom)
    """
    pass

def add_emission_lines(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    lines: Dict[float, float],  # {wavelength: flux}
    line_width: float = 2.0  # FWHM in Angstroms
) -> np.ndarray:
    """Add emission lines to spectrum."""
    pass
```

#### 4. Sky Background (`sky.py`)
```python
"""
Sky background modeling.

Features:
- Moonlight (phase-dependent)
- Zodiacal light
- Airglow emission lines
- Scattered starlight
- Aurora (high latitude sites)
- Light pollution
"""

def calculate_sky_background(
    wavelength: np.ndarray,
    moon_phase: float = 0.0,
    moon_separation: float = 90.0,
    ecliptic_latitude: float = 0.0,
    solar_activity: str = 'medium',
    site: str = 'generic'
) -> np.ndarray:
    """
    Calculate sky background spectrum.
    
    Parameters
    ----------
    wavelength : array
        Wavelength in Angstroms
    moon_phase : float
        Moon phase (0=new, 1=full)
    moon_separation : float
        Angular separation from moon in degrees
    ecliptic_latitude : float
        Ecliptic latitude in degrees (for zodiacal light)
    solar_activity : str
        Solar activity level ('low', 'medium', 'high')
    site : str
        Observatory site
    
    Returns
    -------
    sky_flux : array
        Sky background in mag/arcsec^2
    """
    pass
```

---

### Phase 2: Advanced Detector Effects (v0.6.0)

#### 5. Quantum Efficiency (`qe.py`)
```python
"""
Detector quantum efficiency and response.

Features:
- CCD QE curves (front-illuminated, back-illuminated)
- CMOS sensor response
- Near-IR detectors (HgCdTe, InSb)
- Fringing patterns (thin CCD)
- Red leak (UV filters)
"""

def get_detector_qe(
    detector_type: str,
    wavelength: np.ndarray
) -> np.ndarray:
    """Get detector QE as function of wavelength."""
    pass

def add_fringing(
    data: np.ndarray,
    wavelength: float,
    amplitude: float = 0.02,
    pattern_type: str = 'thin_ccd'
) -> np.ndarray:
    """Add interference fringing pattern to image."""
    pass
```

#### 6. Point Spread Functions (`psf.py`)
```python
"""
Realistic PSF models.

Features:
- Atmospheric seeing (Kolmogorov, von Karman turbulence)
- Adaptive optics PSFs
- Telescope diffraction
- Aberrations (coma, astigmatism, spherical)
- Pixel response function
- Focus variations across field
"""

def create_atmospheric_psf(
    seeing: float,  # FWHM in arcsec
    wavelength: float,
    pixel_scale: float,
    size: int = 51,
    profile: str = 'moffat'  # or 'kolmogorov'
) -> np.ndarray:
    """Create atmospheric seeing PSF."""
    pass

def create_telescope_psf(
    diameter: float,  # meters
    wavelength: float,
    pixel_scale: float,
    obscuration_ratio: float = 0.3,
    n_spiders: int = 4,
    aberrations: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """Create telescope diffraction PSF with aberrations."""
    pass

def convolve_psfs(*psfs: np.ndarray) -> np.ndarray:
    """Convolve multiple PSF components."""
    pass
```

---

### Phase 3: Advanced Calibration (v0.7.0)

#### 7. Realistic Flat Fields (`advanced_flats.py`)
```python
"""
Advanced flat field generation.

Features:
- Twilight flats (color gradients)
- Dome flats (illumination patterns)
- Sky flats (gradient removal)
- Dust shadows (realistic sizes and shapes)
- Ghost reflections
- Filter-dependent vignetting
"""

def create_twilight_flat(
    shape: Tuple[int, int],
    wavelength: float,
    gradient_direction: float = 45.0,
    gradient_strength: float = 0.1
) -> np.ndarray:
    """Create twilight flat with color gradient."""
    pass

def add_dust_shadows(
    flat: np.ndarray,
    num_dust: int = 10,
    size_range: Tuple[int, int] = (5, 50),
    depth_range: Tuple[float, float] = (0.7, 0.95)
) -> np.ndarray:
    """Add realistic dust donuts to flat."""
    pass

def add_ghost_reflections(
    flat: np.ndarray,
    num_ghosts: int = 2,
    intensity_range: Tuple[float, float] = (0.01, 0.05)
) -> np.ndarray:
    """Add ghost reflections from filter."""
    pass
```

#### 8. Persistence & Non-Linearity (`detector_effects.py`)
```python
"""
Advanced detector effects.

Features:
- Charge persistence (IR detectors)
- Non-linearity correction needs
- Warm/cold pixels (temperature-dependent)
- Reset anomaly (IR arrays)
- Shutter effects
- Electronic crosstalk
"""

def add_persistence(
    current_frame: np.ndarray,
    previous_frames: List[np.ndarray],
    persistence_coefficient: float = 0.01,
    persistence_timescale: float = 100.0  # frames
) -> np.ndarray:
    """Add charge persistence from previous exposures."""
    pass

def add_nonlinearity(
    data: np.ndarray,
    coefficients: List[float],
    full_well: float = 100000.0
) -> np.ndarray:
    """Add detector non-linearity."""
    pass
```

---

### Phase 4: Observatory Simulation (v0.8.0)

#### 9. Full Observatory Simulator (`observatory.py`)
```python
"""
Complete end-to-end observatory simulation.

Combines all components:
- Source spectrum
- Atmospheric transmission
- Telescope optics
- Filter transmission
- Detector QE
- PSF convolution
- Detector effects
- Calibration artifacts
"""

class Observatory:
    """Complete observatory simulation."""
    
    def __init__(
        self,
        telescope_diameter: float,
        site: str,
        detector_type: str,
        filter_set: str = 'SDSS'
    ):
        """Initialize observatory configuration."""
        pass
    
    def observe(
        self,
        target_spectrum: np.ndarray,
        filter_name: str,
        exposure_time: float,
        airmass: float = 1.0,
        seeing: float = 1.0,
        moon_phase: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """
        Simulate complete observation.
        
        Returns FITS-formatted image with realistic:
        - Signal
        - Sky background
        - Atmospheric effects
        - PSF
        - Detector noise and artifacts
        - Proper headers
        """
        pass
```

#### 10. Survey Simulators (`surveys.py`)
```python
"""
Survey-specific simulators.

Pre-configured for:
- SDSS
- Pan-STARRS
- HST (various instruments)
- JWST (various instruments)
- Rubin Observatory (LSST)
- Euclid
- Roman Space Telescope
"""

def create_sdss_observation(
    ra: float,
    dec: float,
    filter_name: str,
    exposure_time: float = 54.0,
    seeing: float = 1.4
) -> np.ndarray:
    """Simulate SDSS observation."""
    pass
```

---

### Phase 5: Specialized Simulations (v0.9.0)

#### 11. Spectroscopy (`spectroscopy.py`)
```python
"""
Spectroscopic data simulation.

Features:
- Long-slit spectra
- Multi-object spectroscopy (MOS)
- Integral field spectroscopy (IFS)
- Echelle orders
- Wavelength calibration lamps
- Spectroscopic flats
"""

def create_longslit_spectrum(
    spatial_profile: np.ndarray,
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    dispersion: float,  # Angstrom/pixel
    slit_width: float  # arcsec
) -> np.ndarray:
    """Create 2D longslit spectrum."""
    pass
```

#### 12. Time Domain (`transients.py`)
```python
"""
Time-domain and variable sources.

Features:
- Variable stars (Cepheids, RR Lyrae, etc.)
- Supernovae light curves
- Transiting exoplanets
- Stellar flares
- Microlensing events
- Binary systems
"""

def create_lightcurve(
    times: np.ndarray,
    source_type: str,
    parameters: Dict[str, float]
) -> np.ndarray:
    """Generate time-series light curve."""
    pass

def create_transit_lightcurve(
    times: np.ndarray,
    period: float,
    depth: float,
    duration: float
) -> np.ndarray:
    """Generate exoplanet transit light curve."""
    pass
```

#### 13. Astrometry (`astrometry.py`)
```python
"""
Astrometric simulations.

Features:
- Proper motion
- Parallax
- Distortion patterns
- Differential chromatic refraction
- Field rotation
"""

def add_proper_motion(
    positions: np.ndarray,
    proper_motions: np.ndarray,
    time_baseline: float
) -> np.ndarray:
    """Apply proper motion to source positions."""
    pass

def add_parallax(
    positions: np.ndarray,
    parallaxes: np.ndarray,
    observation_date: float
) -> np.ndarray:
    """Apply parallax displacement."""
    pass
```

---

## Proposed Final Structure (v1.0.0)

```
simulations/
├── __init__.py
│
├── Core (current functionality)
│   ├── noise.py              # Noise models
│   ├── artifacts.py          # Detector artifacts
│   ├── frames.py             # Basic frame creation
│   ├── mef.py                # Multi-Extension FITS
│   ├── timeseries.py         # Time series
│   └── utils.py              # Helpers
│
├── Spectroscopic & Atmospheric
│   ├── atmosphere.py         # Atmospheric transmission
│   ├── filters.py            # Filter transmission curves
│   ├── spectra.py            # Source spectra
│   └── sky.py                # Sky background
│
├── Advanced Detector
│   ├── qe.py                 # Quantum efficiency
│   ├── psf.py                # PSF models
│   └── detector_effects.py  # Persistence, non-linearity
│
├── Advanced Calibration
│   └── advanced_flats.py     # Twilight, dome flats, dust
│
├── Observatory & Surveys
│   ├── observatory.py        # End-to-end simulator
│   └── surveys.py            # Survey-specific configs
│
└── Specialized
    ├── spectroscopy.py       # Spectroscopic simulations
    ├── transients.py         # Time-domain sources
    └── astrometry.py         # Astrometric effects
```

---

## Integration Examples

### Example 1: Realistic Science Frame with Atmospheric Effects
```python
from fits_tools.simulations import create_science_frame
from fits_tools.simulations.atmosphere import calculate_atmospheric_transmission
from fits_tools.simulations.filters import apply_filter_to_spectrum
from fits_tools.simulations.spectra import get_stellar_spectrum
from fits_tools.simulations.sky import calculate_sky_background

# Get source spectrum
wavelength, spectrum = get_stellar_spectrum('G2V')

# Apply atmospheric transmission
transmission = calculate_atmospheric_transmission(
    wavelength, 
    airmass=1.5,
    site='mauna_kea'
)

# Calculate observed flux through filter
observed_flux = apply_filter_to_spectrum(
    wavelength,
    spectrum * transmission,
    filter_name='SDSS_g'
)

# Add sky background
sky_flux = calculate_sky_background(
    wavelength,
    moon_phase=0.3,
    moon_separation=45.0
)

# Create final frame
create_science_frame(
    'science.fits',
    sky_level=sky_flux,
    num_stars=50,
    # ... other parameters
)
```

### Example 2: Full Observatory Simulation
```python
from fits_tools.simulations.observatory import Observatory

# Configure observatory
obs = Observatory(
    telescope_diameter=8.2,  # meters (VLT)
    site='paranal',
    detector_type='E2V_CCD231-84',
    filter_set='SDSS'
)

# Observe target
image = obs.observe(
    target_spectrum=my_galaxy_spectrum,
    filter_name='g',
    exposure_time=300.0,
    airmass=1.2,
    seeing=0.8,
    moon_phase=0.0
)
```

---

## Data Files Needed

### Filter Transmission Files
```
data/filters/
├── johnson_cousins/
│   ├── U.dat
│   ├── B.dat
│   ├── V.dat
│   ├── R.dat
│   └── I.dat
├── sdss/
│   ├── u.dat
│   ├── g.dat
│   └── ...
├── 2mass/
├── panstarrs/
├── hst/
└── jwst/
```

### Atmospheric Models
```
data/atmosphere/
├── us_standard.dat
├── tropical.dat
├── sub_arctic_winter.dat
└── site_specific/
    ├── mauna_kea.dat
    ├── paranal.dat
    └── ...
```

### Stellar Spectra Library
```
data/spectra/
├── stellar/
│   ├── pickles_library/
│   └── synthetic_grids/
├── galaxies/
└── qso/
```

---

## Dependencies to Add

```toml
# pyproject.toml additions

[project.optional-dependencies]
spectroscopy = [
    "scipy>=1.9.0",
    "synphot>=1.2.0",       # For synthetic photometry
    "specutils>=1.9.0",      # Spectroscopic utilities
]

atmosphere = [
    "skyfield>=1.45",        # Astronomical calculations
    "astropy>=5.0",
    "scipy>=1.9.0",
]

full = [
    "fits_tools[spectroscopy]",
    "fits_tools[atmosphere]",
]
```

---

## Testing Strategy

Each new module should include:

1. **Unit tests** - Test individual functions
2. **Integration tests** - Test combined functionality
3. **Validation tests** - Compare to known results
4. **Regression tests** - Ensure outputs don't change

Example validation test:
```python
def test_atmospheric_transmission_validation():
    """Validate against MODTRAN results."""
    # Load reference MODTRAN transmission
    ref_data = load_reference_data('modtran_output.dat')
    
    # Calculate with our model
    transmission = calculate_atmospheric_transmission(
        wavelength=ref_data['wavelength'],
        airmass=1.0,
        pwv=5.0
    )
    
    # Should match within 5%
    assert np.allclose(transmission, ref_data['transmission'], rtol=0.05)
```

---

## Documentation Needs

1. **User Guide** - How to use each component
2. **Tutorials** - End-to-end examples
3. **API Reference** - Complete function documentation
4. **Theory** - Physical models and assumptions
5. **Validation** - Comparison to real data

---

## Community Contributions

Consider accepting contributions for:
- Additional filter sets
- Observatory configurations
- Detector models
- Spectral templates
- Atmospheric models for specific sites

---

## Notes

**Priority for integration:**
1. ✅ Refactor to subpackage structure (enables everything else)
2. 🔧 Add your atmospheric transmission code
3. 🔧 Add your filter transmission code
4. 🌟 Build out from there based on user needs

**Philosophy:**
- Start simple, add complexity as needed
- Validate against real data
- Document assumptions and limitations
- Make it easy to extend

Would you like me to help create any of these modules, or would you like to discuss the integration of your existing atmospheric/filter code?
