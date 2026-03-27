# Basic run
pytest test_defects.py -v

# Stop on first failure
pytest test_defects.py -x

# Skip the FITS I/O tests (no astropy needed)
pytest test_defects.py -v -k "not FitsRoundTrip"

# Run just one class
pytest test_defects.py::TestDetectFlatDefects -v

# With coverage (needs pytest-cov)
pytest test_defects.py --cov=fits_tools.defects --cov-report=term-missing