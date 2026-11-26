#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from fits_tools.image_processing.calibration import calibrate_image
from fits_tools.image_processing.combining import combine_images
from fits_tools.image_processing.cosmic_rays import (
    batch_clean_cosmic_rays,
    clean_cosmic_rays_file,
)
from fits_tools.image_processing.filters import filter_hdu, filter_hdu_batch
from fits_tools.image_processing.operations import apply_operation_to_file
from fits_tools.image_processing.streaming import (
    streaming_combine,
    streaming_statistics,
)
from fits_tools.image_processing.transformations import transform_image_file


def test_end_to_end_image_processing_pipeline(tmp_path: Path) -> None:
    bias_path = tmp_path / "master_bias.fits"
    fits.PrimaryHDU(
        data=np.full((6, 6), 10.0, dtype=np.float32),
        header=fits.Header({"IMAGETYP": "BIAS", "EXPTIME": 0.0}),
    ).writeto(bias_path)

    base = np.arange(36, dtype=np.float32).reshape(6, 6) + 100.0
    raw1_path = tmp_path / "raw1.fits"
    raw2_path = tmp_path / "raw2.fits"
    fits.PrimaryHDU(
        data=base + 1.0, header=fits.Header({"EXPTIME": 30.0, "IMAGETYP": "LIGHT"})
    ).writeto(raw1_path)
    raw2_data = base + 3.0
    raw2_data[2, 3] = 1000.0
    fits.PrimaryHDU(
        data=raw2_data, header=fits.Header({"EXPTIME": 30.0, "IMAGETYP": "LIGHT"})
    ).writeto(raw2_path)

    cal1_path = tmp_path / "raw1_cal.fits"
    cal2_path = tmp_path / "raw2_cal.fits"
    cal1 = calibrate_image(
        raw1_path,
        output_file=cal1_path,
        master_bias=bias_path,
        overwrite=True,
        validate_inputs=False,
        write_bpm=False,
    )
    cal2 = calibrate_image(
        raw2_path,
        output_file=cal2_path,
        master_bias=bias_path,
        overwrite=True,
        validate_inputs=False,
        write_bpm=False,
    )

    assert cal1.header["BIASFILE"] == "master_bias.fits"
    assert cal2.data[2, 3] > 900.0

    clean1_path = tmp_path / "raw1_clean.fits"
    clean2_path = tmp_path / "raw2_clean.fits"
    clean_cosmic_rays_file(
        cal1_path, output_file=clean1_path, method="median", box_size=3, overwrite=True
    )
    cr_result = clean_cosmic_rays_file(
        cal2_path, output_file=clean2_path, method="median", box_size=3, overwrite=True
    )

    assert cr_result.num_cosmic_rays >= 1
    assert fits.getdata(clean2_path)[2, 3] < 200.0

    filtered1_path = tmp_path / "raw1_filtered.fits"
    filtered2_path = tmp_path / "raw2_filtered.fits"
    filter_hdu(
        clean1_path, filtered1_path, operation="gaussian", sigma=0.6, overwrite=True
    )
    filter_hdu(
        clean2_path, filtered2_path, operation="gaussian", sigma=0.6, overwrite=True
    )

    cropped1_path = tmp_path / "raw1_crop.fits"
    cropped2_path = tmp_path / "raw2_crop.fits"
    transform_image_file(
        filtered1_path,
        cropped1_path,
        transformation="crop",
        region=(1, 5, 1, 5),
        overwrite=True,
    )
    transform_image_file(
        filtered2_path,
        cropped2_path,
        transformation="crop",
        region=(1, 5, 1, 5),
        overwrite=True,
    )

    scaled1_path = tmp_path / "raw1_scaled.fits"
    scaled2_path = tmp_path / "raw2_scaled.fits"
    result1 = apply_operation_to_file(
        cropped1_path,
        scaled1_path,
        operation_type="scalar",
        operation="multiply",
        scalar=2.0,
        overwrite=True,
    )
    result2 = apply_operation_to_file(
        cropped2_path,
        scaled2_path,
        operation_type="scalar",
        operation="multiply",
        scalar=2.0,
        overwrite=True,
    )

    assert result1.success is True
    assert result2.success is True
    assert fits.getdata(scaled1_path).shape == (4, 4)

    combined_path = tmp_path / "stack_mean.fits"
    combined = combine_images(
        [scaled1_path, scaled2_path],
        method="mean",
        output_file=combined_path,
        overwrite=True,
        return_type="result",
    )
    streaming_mean = streaming_combine(
        [scaled1_path, scaled2_path], method="mean", show_progress=False
    )

    assert combined.output_file == combined_path
    assert combined.header["NCOMBINE"] == 2
    assert np.allclose(combined.data, streaming_mean)
    assert np.allclose(fits.getdata(combined_path), streaming_mean)


def test_batch_workflow_integrates_clean_filter_and_streaming_stats(
    tmp_path: Path,
) -> None:
    input_files: list[Path] = []
    for idx, offset in enumerate((0.0, 5.0), start=1):
        path = tmp_path / f"batch_{idx}.fits"
        data = np.arange(25, dtype=np.float32).reshape(5, 5) + 50.0 + offset
        data[1 + idx, 2] = 800.0 + offset
        fits.PrimaryHDU(
            data=data, header=fits.Header({"EXPTIME": 20.0, "IMAGETYP": "LIGHT"})
        ).writeto(path)
        input_files.append(path)

    clean_results = batch_clean_cosmic_rays(
        input_files,
        output_dir=tmp_path / "cleaned",
        method="median",
        box_size=3,
        overwrite=True,
        save_masks=True,
    )

    assert clean_results["successful"] == 2
    cleaned_files = [Path(path) for path in clean_results["output_files"]]
    assert all(path.exists() for path in cleaned_files)

    batch_result = filter_hdu_batch(
        cleaned_files,
        tmp_path / "filtered" / "{stem}_flt.fits",
        operation="median",
        size=3,
        overwrite=True,
        validate_outputs=True,
        show_progress=False,
    )

    assert batch_result.success_count == 2
    filtered_files = batch_result.successful
    stats = streaming_statistics(
        filtered_files, compute_quantiles=True, quantiles=[0.5], show_progress=False
    )
    streamed_median = streaming_combine(
        filtered_files, method="median", show_progress=False
    )

    assert stats["count"] == 2
    assert stats["mean"].shape == (5, 5)
    assert stats["quantiles"][0.5].shape == (5, 5)
    assert np.allclose(streamed_median, stats["quantiles"][0.5], atol=1.0)
