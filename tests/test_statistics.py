#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from fits_tools.exceptions import ImageProcessingError
from fits_tools.statistics import (
    ComparisonResult,
    RobustStatsResult,
    StatsResult,
    _array_stats,
    _simple_stats,
    analyze_statistics_trend,
    array_stats,
    calculate_average_stats,
    calculate_dynamic_range,
    calculate_snr,
    calculate_statistics_across_files,
    compare_stats,
    estimate_read_noise,
    get_average_maximum,
    get_average_mean,
    get_average_median,
    get_average_minimum,
    get_average_std,
    get_average_stddev,
    get_image_max,
    get_image_mean,
    get_image_median,
    get_image_min,
    get_image_std,
    robust_stats,
    stats,
    stats_to_dict,
)


def test_array_stats_computes_expected_values_for_roi() -> None:
    data = np.arange(16, dtype=float).reshape(4, 4)

    result = array_stats(data, roi=(1, 3, 1, 4))

    roi = data[1:4, 1:3]
    assert result.mean_val == pytest.approx(float(np.mean(roi)))
    assert result.median_val == pytest.approx(float(np.median(roi)))
    assert result.num_pixels == roi.size
    assert result.range_val == pytest.approx(float(np.max(roi) - np.min(roi)))


def test_array_stats_skipna_counts_only_finite_pixels() -> None:
    data = np.array([[1.0, np.nan], [np.inf, 4.0]])

    result = array_stats(data, skipna=True)

    assert result.num_pixels == 2
    assert result.mean_val == pytest.approx(2.5)


def test_stats_reads_named_region_from_fits_file(sample_fits_file) -> None:
    result = stats(sample_fits_file, region="data", stats_type="all")

    assert result.mean_val == pytest.approx(12.5)
    assert result.num_pixels == 12


def test_stats_simple_omits_extended_fields(sample_fits_file) -> None:
    result = stats(sample_fits_file, stats_type="simple")

    assert np.isnan(result.median_val)
    assert np.isnan(result.percentile_95)
    assert result.mean_val == pytest.approx(9.5)


def test_robust_stats_clips_outlier_from_array() -> None:
    data = np.array([1, 1, 1, 1, 100], dtype=float)

    result = robust_stats(data, sigma=1.5)

    assert result.num_clipped >= 1
    assert result.mean == pytest.approx(1.0)


def test_calculate_snr_uses_signal_region_std_when_noise_region_missing(sample_fits_file) -> None:
    snr = calculate_snr(sample_fits_file, signal_region="data")

    data = np.arange(20, dtype=np.float32).reshape(4, 5)[1:4, 1:5]
    assert snr == pytest.approx(float(np.mean(data) / np.std(data)))


def test_array_stats_handles_empty_arrays_and_invalid_roi() -> None:
    empty = array_stats(np.array([], dtype=float))
    assert empty.num_pixels == 0
    assert np.isnan(empty.mean_val)

    with pytest.raises(TypeError, match="NumPy array"):
        array_stats([1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ROI is only supported for 2-D arrays"):
        array_stats(np.zeros((2, 2, 2)), roi=(0, 1, 0, 1))

    with pytest.raises(ValueError, match="tuple of 4 integers"):
        array_stats(np.zeros((2, 2)), roi=(0, 1, 0))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid roi"):
        array_stats(np.zeros((2, 2)), roi=(0, 3, 0, 1))


def test_deprecated_stats_helpers_still_return_expected_results() -> None:
    result = array_stats(np.arange(4, dtype=float).reshape(2, 2))

    with pytest.deprecated_call():
        as_dict = stats_to_dict(result)
    with pytest.deprecated_call():
        alias_result = _array_stats(np.arange(4, dtype=float).reshape(2, 2))

    assert as_dict["mean_val"] == pytest.approx(1.5)
    assert alias_result.mean_val == pytest.approx(1.5)


def test_statistics_result_types_have_useful_string_and_dict_views() -> None:
    basic = StatsResult(1.0, 5.0, 3.0, 3.0, 1.0, 1.0, num_pixels=4)
    robust = RobustStatsResult(3.0, 3.0, 1.0, 1.0, 4, num_clipped=1, clip_fraction=0.2)
    comparison = ComparisonResult(
        file1="a.fits",
        file2="b.fits",
        stats1=StatsResult(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, num_pixels=1),
        stats2=StatsResult(0.0, 1.0, 0.0, 0.0, 1.0, 1.0, num_pixels=1),
    )

    assert basic.to_dict()["mean_val"] == pytest.approx(3.0)
    assert "Statistics Summary" in str(basic)
    assert "StatsResult" in repr(basic)
    assert "mask" not in robust.to_dict()
    assert "Robust Statistics" in str(robust)
    assert np.isnan(comparison.relative_mean_diff)
    assert "Comparison Results" in str(comparison)


def test_robust_stats_supports_roi_and_return_mask() -> None:
    data = np.array([[1.0, 1.0], [1.0, 100.0]])

    result = robust_stats(data, region=(0, 2, 0, 2), sigma=1.5, return_mask=True)

    assert result.num_clipped >= 1
    assert result.mask is not None
    assert result.mask.shape == data.shape


def test_array_stats_flatten_simple_stats_and_stats_aliases_cover_helper_paths(
    sample_fits_file,
) -> None:
    cube = np.arange(8, dtype=float).reshape(2, 2, 2)
    flat_result = array_stats(cube, compute_extended=False)
    simple = _simple_stats(np.array([1.0, np.nan, 3.0]), skipna=True)

    assert flat_result.num_pixels == 8
    assert np.isnan(flat_result.percentile_95)
    assert simple.num_pixels == 2
    assert simple.mean_val == pytest.approx(2.0)

    assert get_image_min(sample_fits_file, region="full") == pytest.approx(0.0)
    assert get_image_max(sample_fits_file, region="full") == pytest.approx(19.0)
    assert get_image_mean(sample_fits_file, region="full") == pytest.approx(9.5)
    assert get_image_median(sample_fits_file, region="full") == pytest.approx(9.5)
    assert get_image_std(sample_fits_file, region="full") == pytest.approx(np.std(np.arange(20)))


def test_dynamic_range_and_read_noise_cover_special_cases(tmp_path) -> None:
    zero_min = tmp_path / "zero_min.fits"
    fits.PrimaryHDU(data=np.array([[0.0, 10.0]], dtype=np.float32)).writeto(zero_min)
    assert calculate_dynamic_range(zero_min, region="full") == float("inf")

    bias1 = tmp_path / "bias1.fits"
    bias2 = tmp_path / "bias2.fits"
    fits.PrimaryHDU(
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        header=fits.Header({"DATASEC": "[1:2,1:2]"}),
    ).writeto(bias1)
    fits.PrimaryHDU(
        data=np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        header=fits.Header({"DATASEC": "[1:2,1:2]"}),
    ).writeto(bias2)

    assert estimate_read_noise([bias1, bias2], region="full") == pytest.approx(
        np.mean([np.std(fits.getdata(bias1)), np.std(fits.getdata(bias2))])
    )

    with pytest.raises(ValueError, match="No bias files provided"):
        estimate_read_noise([])


def test_stats_and_robust_stats_cover_scalar_regions_and_error_paths(
    monkeypatch: pytest.MonkeyPatch, sample_fits_file
) -> None:
    tuple_mean = stats(sample_fits_file, region=(1, 3, 1, 3), stats_type="mean")
    explicit_noise_snr = calculate_snr(
        sample_fits_file,
        signal_region="full",
        noise_region=(0, 1, 0, 2),
    )
    finite_dynamic_range = calculate_dynamic_range(sample_fits_file, region="data")

    assert tuple_mean == pytest.approx(np.mean(np.arange(20, dtype=float).reshape(4, 5)[1:3, 1:3]))
    assert explicit_noise_snr == pytest.approx(9.5 / 2.5)
    assert finite_dynamic_range == pytest.approx(19.0 / 6.0)

    with pytest.raises(ValueError, match="Unsupported stats_type"):
        stats(sample_fits_file, stats_type="not-a-stat")

    with pytest.raises(TypeError, match="Expected file path"):
        robust_stats(123)  # type: ignore[arg-type]

    monkeypatch.setattr("fits_tools.statistics.get_data", lambda *args, **kwargs: None)

    assert stats(sample_fits_file) is None

    with pytest.raises(ImageProcessingError, match="No valid data provided"):
        robust_stats(sample_fits_file)


def test_estimate_read_noise_and_multi_file_averages_cover_skip_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    bias1 = tmp_path / "bias1.fits"
    bias2 = tmp_path / "bias2.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(bias1)
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32) * 2).writeto(bias2)

    original_stats = stats

    def fake_stats(path, *args, **kwargs):
        path = str(path)
        if path.endswith("bias1.fits"):
            return 1.5 if kwargs.get("stats_type") == "std" else original_stats(path, *args, **kwargs)
        if path.endswith("bias2.fits"):
            raise FileNotFoundError("missing during test")
        return original_stats(path, *args, **kwargs)

    monkeypatch.setattr("fits_tools.statistics.stats", fake_stats)

    assert estimate_read_noise([bias1, bias2], region="full") == pytest.approx(1.5)

    monkeypatch.setattr("fits_tools.statistics.stats", lambda *args, **kwargs: None)
    assert calculate_statistics_across_files([bias1], single_statistic="mean") is None


def test_multi_file_statistics_compare_and_trend_analysis(tmp_path) -> None:
    paths = []
    for idx, mean in enumerate((10.0, 20.0, 30.0), start=1):
        path = tmp_path / f"file_{idx}.fits"
        fits.PrimaryHDU(
            data=np.full((2, 2), mean, dtype=np.float32),
            header=fits.Header({"IMAGETYP": "LIGHT"}),
        ).writeto(path)
        paths.append(path)

    averages = calculate_statistics_across_files(paths, single_statistic=None)
    avg_mean = get_average_mean(paths)
    comparison = compare_stats(paths[0], paths[1])
    trend = analyze_statistics_trend(paths, statistic="mean")

    assert averages["mean"] == pytest.approx(20.0)
    assert avg_mean == pytest.approx(20.0)
    assert comparison.mean_diff == pytest.approx(-10.0)
    assert trend["trend"] > 0
    assert trend["num_files"] == 3


def test_statistics_across_files_and_trend_handle_invalid_inputs(tmp_path) -> None:
    bad = tmp_path / "bad.fits"
    bad.write_text("not fits", encoding="utf-8")

    assert calculate_statistics_across_files([bad]) is None

    with pytest.raises(ValueError, match="Invalid single_statistic"):
        calculate_statistics_across_files([], single_statistic="bad")

    with pytest.raises(ValueError, match="Empty file list provided"):
        analyze_statistics_trend([])

    with pytest.raises(ImageProcessingError, match="No valid statistics obtained"):
        analyze_statistics_trend([bad])

    with pytest.deprecated_call():
        assert get_average_stddev([bad]) is None


def test_average_stat_wrappers_and_deprecated_alias_cover_remaining_paths(tmp_path) -> None:
    paths = []
    for idx, value in enumerate((2.0, 6.0), start=1):
        path = tmp_path / f"avg_{idx}.fits"
        fits.PrimaryHDU(data=np.full((2, 2), value, dtype=np.float32)).writeto(path)
        paths.append(path)

    with pytest.deprecated_call():
        all_stats = calculate_average_stats(paths)

    assert all_stats is not None
    assert get_average_mean(paths) == pytest.approx(4.0)
    assert get_average_median(paths) == pytest.approx(4.0)
    assert get_average_minimum(paths) == pytest.approx(4.0)
    assert get_average_maximum(paths) == pytest.approx(4.0)
    assert get_average_std(paths) == pytest.approx(0.0)
