#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import numpy as np
import pytest

from fits_tools.bitmask import (
    BPM_BIT_COL_BAD,
    BPM_BIT_PIX_BRIGHT,
    BPM_BIT_PIX_DEAD,
    BPM_BIT_PIX_HOT,
    BPM_BIT_PIX_NOISE,
    BPM_BIT_PIX_NONLIN,
    BPM_BIT_PIX_PRNU_HIGH,
    BPM_BIT_PIX_PRNU_LOW,
    BPM_BIT_PIX_TRAP,
    BPM_BIT_ROW_BAD,
    has_defect,
)
from fits_tools.defects.detection import (
    _detect_amplifier_glow_impl,
    _detect_charge_traps_impl,
    _detect_noisy_pixels_impl,
    _detect_nonlinear_pixels_impl,
    _detect_rts_pixels_impl,
    _sigma_clip_combine,
    _validate_roi,
    create_calibration_bpm,
    create_calibration_bpm_batch,
    create_defect_density_map,
    create_detection_report,
    detect_amplifier_glow,
    detect_bad_rows_cols,
    detect_bright_pixels,
    detect_charge_traps,
    detect_dead_pixels,
    detect_defect_clusters,
    detect_hot_pixels,
    detect_noisy_pixels,
    detect_nonlinear_pixels,
    detect_prnu_anomalies,
    detect_rts_pixels,
    detect_temporal_defects,
    estimate_prnu_model,
    get_default_detection_config,
    plot_defect_comparison,
    roi_stats,
    run_advanced_defect_detection,
    sigma_clip_stats,
)


def test_validate_roi_normalizes_and_rejects_invalid_regions() -> None:
    assert _validate_roi((20, 30), (1, 10, 2, 12)) == (1, 10, 2, 12)

    with pytest.raises(ValueError, match="roi must have 4 elements"):
        _validate_roi((20, 30), (1, 2, 3))

    with pytest.raises(ValueError, match="out of bounds"):
        _validate_roi((20, 30), (0, 40, 0, 10))

    with pytest.raises(ValueError, match="degenerate"):
        _validate_roi((20, 30), (5, 5, 0, 10))


def test_sigma_clip_stats_handles_constant_and_all_nan_data() -> None:
    median, std = sigma_clip_stats(np.ones((10, 10)) * 1000.0)
    assert median == pytest.approx(1000.0)
    assert std == pytest.approx(0.0)

    nan_median, nan_std = sigma_clip_stats(np.full((4, 4), np.nan))
    assert np.isnan(nan_median)
    assert np.isnan(nan_std)


def test_sigma_clip_stats_validates_inputs_and_supports_mask_and_mad() -> None:
    data = np.array([[1.0, 2.0], [3.0, 100.0]])
    mask = np.array([[False, False], [True, False]])
    median, scale = sigma_clip_stats(data, mask=mask, scale="mad")

    assert median == pytest.approx(1.5)
    assert scale >= 0.0

    with pytest.raises(TypeError, match="data must be ndarray"):
        sigma_clip_stats([[1.0, 2.0]])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="numeric dtype"):
        sigma_clip_stats(np.array([["x"]], dtype=object))

    with pytest.raises(ValueError, match="must be non-negative"):
        sigma_clip_stats(np.ones((2, 2)), sigma_low=-1.0)

    with pytest.raises(ValueError, match="max_iters must be >= 1"):
        sigma_clip_stats(np.ones((2, 2)), max_iters=0)

    with pytest.raises(ValueError, match="mask shape"):
        sigma_clip_stats(np.ones((2, 2)), mask=np.ones((3, 3), dtype=bool))

    with pytest.raises(ValueError, match="scale must be 'std' or 'mad'"):
        sigma_clip_stats(np.ones((2, 2)), scale="iqr")


def test_sigma_clip_combine_rejects_outlier_samples() -> None:
    frames = np.stack(
        [
            np.ones((2, 2), dtype=float) * 10.0,
            np.ones((2, 2), dtype=float) * 11.0,
            np.array([[10.0, 100.0], [10.0, 10.0]], dtype=float),
        ]
    )

    combined, rejected = _sigma_clip_combine(frames, sigma=1.0)

    assert combined[0, 1] < 50.0
    assert rejected[0, 1] >= 1


def test_roi_stats_supports_roi_mask_and_non_clipped_path() -> None:
    data = np.arange(25, dtype=float).reshape(5, 5)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True

    median, std = roi_stats(data, roi=(1, 4, 1, 4), sigma_clip=False, mask=mask)

    assert median == pytest.approx(np.nanmedian(data[1:4, 1:4][~mask[1:4, 1:4]]))
    assert std >= 0.0


def test_estimate_prnu_model_builds_block_model_and_validates_inputs() -> None:
    flat = np.arange(64, dtype=float).reshape(8, 8)
    existing_bpm = np.zeros((8, 8), dtype=np.uint32)
    existing_bpm[0, 0] = 1

    model = estimate_prnu_model(flat, tile_size=4, sigma_clip=False, existing_bpm=existing_bpm)

    assert model.shape == flat.shape
    assert np.all(model[:4, :4] == model[0, 0])

    with pytest.raises(ValueError, match="expects a 2-D array"):
        estimate_prnu_model(np.ones((2, 2, 2)))

    with pytest.raises(ValueError, match="tile_size must be positive"):
        estimate_prnu_model(np.ones((4, 4)), tile_size=0)

    with pytest.raises(ValueError, match="existing_bpm shape"):
        estimate_prnu_model(np.ones((4, 4)), existing_bpm=np.zeros((2, 2), dtype=np.uint32))


def test_detect_bright_pixels_absolute_high_mode_flags_only_high_outlier() -> None:
    bias = np.ones((50, 50), dtype=float) * 1000.0
    bias[10, 10] = 1200.0
    bias[20, 20] = 800.0

    bpm, stats = detect_bright_pixels(bias, threshold_adu=100.0, method="absolute", mode="high")

    assert has_defect(bpm[10, 10], BPM_BIT_PIX_BRIGHT)
    assert not has_defect(bpm[20, 20], BPM_BIT_PIX_BRIGHT)
    assert stats["num_defects"] == 1


def test_detect_bright_pixels_sigma_mode_flags_both_sides() -> None:
    rng = np.random.default_rng(123)
    bias = rng.normal(1000.0, 5.0, size=(100, 100))
    bias[5, 5] = 1050.0
    bias[10, 10] = 950.0

    bpm, stats = detect_bright_pixels(bias, method="sigma", sigma=5.0, mode="both")

    assert has_defect(bpm[5, 5], BPM_BIT_PIX_BRIGHT)
    assert has_defect(bpm[10, 10], BPM_BIT_PIX_BRIGHT)
    assert stats["threshold_mode"] == "sigma"


def test_detect_bright_pixels_validates_method_mode_and_handles_degenerate_data() -> None:
    bias = np.full((6, 6), np.nan)
    bpm, stats = detect_bright_pixels(bias)

    assert stats["num_defects"] == 0
    assert np.isnan(stats["threshold_hi_adu"])
    assert not np.any(bpm)

    with pytest.raises(ValueError, match="method must be 'absolute' or 'sigma'"):
        detect_bright_pixels(np.ones((4, 4)), method="median")

    with pytest.raises(ValueError, match="mode must be 'high', 'low', or 'both'"):
        detect_bright_pixels(np.ones((4, 4)), mode="middle")


def test_detect_hot_pixels_finds_high_dark_current_pixels() -> None:
    rng = np.random.default_rng(7)
    dark = rng.normal(10.0, 1.0, size=(80, 80))
    dark[25, 25] = 100.0
    dark[60, 60] = 150.0

    bpm, stats = detect_hot_pixels(dark, threshold_sigma=3.0)

    assert has_defect(bpm[25, 25], BPM_BIT_PIX_HOT)
    assert has_defect(bpm[60, 60], BPM_BIT_PIX_HOT)
    assert stats["num_defects"] >= 2


def test_detect_hot_pixels_validates_threshold_and_handles_zero_std() -> None:
    with pytest.raises(ValueError, match="threshold_sigma must be positive"):
        detect_hot_pixels(np.ones((4, 4)), threshold_sigma=0.0)

    bpm, stats = detect_hot_pixels(np.ones((8, 8)), threshold_sigma=5.0)

    assert stats["num_defects"] == 0
    assert np.isnan(stats["threshold_hi"])
    assert not np.any(bpm)


def test_detect_dead_pixels_flags_low_response_and_validates_threshold() -> None:
    flat = np.full((20, 20), 1000.0)
    flat[4, 4] = 10.0

    bpm, stats = detect_dead_pixels(flat, threshold_percent=20.0)

    assert has_defect(bpm[4, 4], BPM_BIT_PIX_DEAD)
    assert stats["num_defects"] >= 1

    with pytest.raises(ValueError, match="threshold_percent must be in"):
        detect_dead_pixels(flat, threshold_percent=0.0)


def test_detect_dead_pixels_handles_degenerate_median() -> None:
    bpm, stats = detect_dead_pixels(np.zeros((10, 10)))

    assert stats["num_defects"] == 0
    assert np.isnan(stats["threshold"])
    assert not np.any(bpm)


def test_detect_prnu_anomalies_flags_hot_and_cold_response_pixels() -> None:
    flat = np.ones((80, 80), dtype=float) * 10000.0
    flat[20, 20] = 13000.0
    flat[40, 40] = 7000.0

    bpm, stats = detect_prnu_anomalies(
        flat,
        hot_threshold_percent=20.0,
        cold_threshold_percent=20.0,
        normalize=True,
    )

    assert has_defect(bpm[20, 20], BPM_BIT_PIX_PRNU_HIGH)
    assert has_defect(bpm[40, 40], BPM_BIT_PIX_PRNU_LOW)
    assert stats["num_hot"] >= 1
    assert stats["num_cold"] >= 1


def test_detect_prnu_anomalies_validates_thresholds_and_local_model() -> None:
    flat = np.full((12, 12), 1000.0)
    local_model = np.full_like(flat, 1000.0)
    flat[3, 3] = 1400.0

    bpm, stats = detect_prnu_anomalies(
        flat,
        hot_threshold_percent=10.0,
        cold_threshold_percent=10.0,
        normalize=True,
        local_model=local_model,
    )

    assert has_defect(bpm[3, 3], BPM_BIT_PIX_PRNU_HIGH)
    assert stats["num_hot"] >= 1

    with pytest.raises(ValueError, match="hot_threshold_percent must be >= 0"):
        detect_prnu_anomalies(flat, hot_threshold_percent=-1.0)

    with pytest.raises(ValueError, match="cold_threshold_percent must be >= 0"):
        detect_prnu_anomalies(flat, cold_threshold_percent=-1.0)

    with pytest.raises(ValueError, match="local_model shape"):
        detect_prnu_anomalies(flat, local_model=np.ones((6, 6)))


def test_detect_prnu_anomalies_handles_degenerate_median() -> None:
    bpm, stats = detect_prnu_anomalies(np.zeros((8, 8)))

    assert stats["num_defects"] == 0
    assert np.isnan(stats["hot_threshold"])
    assert not np.any(bpm)


def test_detect_bad_rows_cols_handles_negative_and_stepped_slices() -> None:
    defect_map = np.zeros((20, 20), dtype=np.uint32)
    defect_map[:, 18] = BPM_BIT_PIX_HOT
    defect_map[::2, :] |= BPM_BIT_PIX_HOT

    bpm, stats = detect_bad_rows_cols(
        defect_map,
        row_threshold=0.5,
        col_threshold=0.1,
        active_region=(slice(-20, None, 2), slice(None)),
    )

    assert stats["num_bad_rows"] > 0
    assert stats["num_bad_cols"] >= 1
    assert has_defect(bpm[0, 18], BPM_BIT_COL_BAD)
    assert has_defect(bpm[0, 0], BPM_BIT_ROW_BAD)


def test_detect_bad_rows_cols_validates_thresholds_and_dimensionality() -> None:
    with pytest.raises(ValueError, match="defect_map must be 2-D"):
        detect_bad_rows_cols(np.zeros((2, 2, 2), dtype=np.uint32))

    with pytest.raises(ValueError, match="row_threshold must be in \\[0, 1\\]"):
        detect_bad_rows_cols(np.zeros((4, 4), dtype=np.uint32), row_threshold=2.0)

    with pytest.raises(ValueError, match="col_threshold must be in \\[0, 1\\]"):
        detect_bad_rows_cols(np.zeros((4, 4), dtype=np.uint32), col_threshold=-0.1)


def test_detect_temporal_defects_requires_at_least_three_frames() -> None:
    stack = np.ones((2, 10, 10), dtype=float)

    with pytest.raises(ValueError, match="at least 3 frames"):
        detect_temporal_defects(stack)


def test_detect_temporal_defects_flags_rts_like_pixel() -> None:
    rng = np.random.default_rng(99)
    stack = rng.normal(10.0, 0.5, size=(5, 20, 20))
    stack[:, 6, 6] = np.array([10.0, 10.0, 40.0, 10.0, 40.0])

    bpm, stats = detect_temporal_defects(stack, threshold_sigma=2.0)

    assert stats["num_defects"] >= 1
    assert np.count_nonzero(bpm) >= 1


def test_detect_temporal_defects_validates_shape_and_handles_existing_bpm_and_degenerate_stats() -> None:
    with pytest.raises(ValueError, match="must be 3D"):
        detect_temporal_defects(np.ones((4, 4)))

    with pytest.raises(ValueError, match="threshold_sigma must be positive"):
        detect_temporal_defects(np.ones((3, 4, 4)), threshold_sigma=0.0)

    with pytest.raises(ValueError, match="existing_bpm shape"):
        detect_temporal_defects(np.ones((3, 4, 4)), existing_bpm=np.ones((3, 3), dtype=np.uint32))

    stack = np.ones((3, 5, 5), dtype=float)
    existing_bpm = np.zeros((5, 5), dtype=np.uint32)
    existing_bpm[0, 0] = 1
    bpm, stats = detect_temporal_defects(stack, existing_bpm=existing_bpm)

    assert stats["num_defects"] == 0
    assert np.isnan(stats["threshold"])
    assert not np.any(bpm)


def test_placeholder_detectors_return_empty_masks() -> None:
    nonlin = detect_nonlinear_pixels(np.ones((3, 4, 4)), np.array([1.0, 2.0, 3.0]))
    noise = detect_noisy_pixels(np.ones((4, 4, 4)))
    rts = detect_rts_pixels(np.ones((4, 4, 4)))
    traps = detect_charge_traps(np.ones((4, 4)))
    glow = detect_amplifier_glow(np.ones((4, 4)))

    assert nonlin.shape == (4, 4)
    assert noise.shape == (4, 4)
    assert rts.shape == (4, 4)
    assert traps.shape == (4, 4)
    assert glow.shape == (4, 4)
    assert not np.any(nonlin | noise | rts | traps | glow)


def test_draft_nonlinear_noisy_rts_trap_and_glow_impls_exercise_real_paths() -> None:
    linearity_data = np.array(
        [
            [[10.0, 10.0], [10.0, 10.0]],
            [[20.0, 20.0], [20.0, 20.0]],
            [[30.0, 35.0], [30.0, 30.0]],
            [[40.0, 60.0], [40.0, 40.0]],
        ]
    )
    nonlin = _detect_nonlinear_pixels_impl(
        linearity_data,
        np.array([1.0, 2.0, 3.0, 4.0]),
        threshold=0.05,
        saturation_level=1e9,
    )
    assert has_defect(nonlin[0, 1], BPM_BIT_PIX_NONLIN)

    bias_frames = np.random.default_rng(1).normal(0.0, 1.0, size=(6, 10, 10))
    bias_frames[:, 2, 2] = np.array([0.0, 8.0, -8.0, 7.0, -7.0, 6.0])
    noisy = _detect_noisy_pixels_impl(bias_frames, threshold_sigma=2.0)
    assert has_defect(noisy[2, 2], BPM_BIT_PIX_NOISE)

    dark_sequence = np.random.default_rng(2).normal(10.0, 0.2, size=(25, 6, 6))
    dark_sequence[:, 3, 3] = np.array(
        [10.0, 10.2, 9.9, 10.1, 10.0]
        + [20.0, 20.1, 19.9, 20.2, 20.0]
        + [10.1, 9.8, 10.0, 10.2, 9.9]
        + [20.2, 19.8, 20.0, 20.1, 19.9]
        + [10.0, 10.1, 9.9, 10.2, 10.0]
    )
    rts = _detect_rts_pixels_impl(
        dark_sequence,
        threshold_transitions=2,
        min_dwell_frames=2,
        amplitude_threshold=1.0,
    )
    assert rts.shape == (6, 6)

    trap_data = np.zeros((15, 15), dtype=float)
    trap_data[7, 7] = 10.0
    trap = _detect_charge_traps_impl(trap_data, threshold=2.0)
    assert np.any(trap & BPM_BIT_PIX_TRAP)

    glow_frame = np.zeros((20, 20), dtype=float)
    glow_frame[2:8, 2:8] = 50.0
    glow = _detect_amplifier_glow_impl(
        glow_frame,
        threshold_sigma=1.0,
        spatial_filter_size=4,
        amp_regions={"A": (slice(0, 10), slice(0, 10))},
    )
    assert glow.shape == (20, 20)


def test_run_advanced_defect_detection_supports_stub_and_draft_modes() -> None:
    sensor_data = {
        "linearity_sequence": (np.ones((3, 4, 4)), np.array([1.0, 2.0, 3.0])),
        "bias_stack": np.random.default_rng(3).normal(0.0, 1.0, size=(6, 4, 4)),
        "dark_sequence": np.random.default_rng(4).normal(0.0, 1.0, size=(25, 4, 4)),
        "long_dark": np.ones((4, 4), dtype=float),
    }

    bpm_stub = run_advanced_defect_detection(sensor_data, use_draft_implementations=False)
    bpm_draft = run_advanced_defect_detection(sensor_data, use_draft_implementations=True)

    assert bpm_stub.shape == (4, 4)
    assert bpm_draft.shape == (4, 4)

    with pytest.raises(ValueError, match="No sensor data provided"):
        run_advanced_defect_detection({}, enable_tests={"nonlinearity": False, "noise": False, "rts": False, "traps": False, "glow": False})


def test_detect_defect_clusters_and_density_map_cover_empty_and_filtered_cases() -> None:
    bpm = np.zeros((8, 8), dtype=np.uint32)
    bpm[1:3, 1:3] = BPM_BIT_PIX_HOT
    bpm[5, 5] = BPM_BIT_PIX_BRIGHT

    clusters = detect_defect_clusters(bpm, min_cluster_size=2, defect_mask=BPM_BIT_PIX_HOT)
    empty_clusters = detect_defect_clusters(np.zeros_like(bpm))

    assert clusters["num_clusters"] >= 1
    assert clusters["num_large_clusters"] >= 1
    assert empty_clusters["num_clusters"] == 0

    density = create_defect_density_map(bpm, tile_size=4, defect_mask=BPM_BIT_PIX_HOT)
    tiny_density = create_defect_density_map(np.zeros((2, 2), dtype=np.uint32), tile_size=8)

    assert density["ny_tiles"] == 2
    assert density["nx_tiles"] == 2
    assert density["counts"][0][0] >= 1
    assert tiny_density["counts"] is None


def test_create_detection_report_and_plot_comparison(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)

    stats = {
        "bias": {"method": "absolute", "median": 1000.0, "std": 2.0, "num_defects": 1, "fraction_defects": 0.01},
        "dark": {"threshold_sigma": 5.0, "median": 10.0, "std": 1.0, "num_defects": 1, "fraction_defects": 0.01},
        "flat": {
            "hot_threshold_percent": 20.0,
            "cold_threshold_percent": 20.0,
            "median": 10000.0,
            "num_hot": 1,
            "num_cold": 1,
            "num_defects": 2,
            "fraction_defects": 0.02,
        },
        "rows_cols": {"num_bad_rows": 0, "num_bad_cols": 1},
        "total_defects": 3,
        "total_fraction": 0.03,
    }
    bpm = np.zeros((10, 10), dtype=np.uint32)
    bpm[1, 1] = BPM_BIT_PIX_BRIGHT
    bpm[2, 2] = BPM_BIT_PIX_HOT
    bpm[:, 5] |= BPM_BIT_COL_BAD

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    output_path = tmp_path / "report.pdf"
    create_detection_report(bpm, stats, output_path=str(output_path), show_plots=False)
    plot_defect_comparison([bpm, bpm], ["A", "B"], title="Compare")

    assert output_path.exists()


def test_create_calibration_bpm_combines_stage_outputs_and_exposes_stage_stats() -> None:
    rng = np.random.default_rng(1234)
    bias = rng.normal(1000.0, 3.0, size=(60, 60))
    dark = rng.normal(10.0, 1.0, size=(60, 60))
    flat = rng.normal(10000.0, 100.0, size=(60, 60))

    bias[10, 10] = 1200.0
    dark[20, 20] = 200.0
    flat[30, 30] = 13000.0
    flat[40, 40] = 500.0
    flat[:, 5] = 0.0

    bpm, stats = create_calibration_bpm(bias, dark, flat)

    assert has_defect(bpm[10, 10], BPM_BIT_PIX_BRIGHT)
    assert has_defect(bpm[20, 20], BPM_BIT_PIX_HOT)
    assert has_defect(bpm[30, 30], BPM_BIT_PIX_PRNU_HIGH)
    assert stats["total_defects"] >= 4
    assert {"bias", "dark", "dead", "flat", "rows_cols"}.issubset(stats)


def test_create_calibration_bpm_rejects_shape_mismatch() -> None:
    bias = np.ones((10, 10), dtype=float)
    dark = np.ones((10, 10), dtype=float)
    flat = np.ones((11, 10), dtype=float)

    with pytest.raises(ValueError, match="same shape"):
        create_calibration_bpm(bias, dark, flat)


def test_create_calibration_bpm_rejects_non_numeric_input() -> None:
    bias = np.ones((10, 10), dtype=float)
    dark = np.ones((10, 10), dtype=float)
    flat = np.ones((10, 10), dtype=object)

    with pytest.raises(TypeError, match="numeric dtype"):
        create_calibration_bpm(bias, dark, flat)


def test_default_detection_config_is_mutable_for_custom_thresholds() -> None:
    config = get_default_detection_config()
    config["bias"]["threshold_adu"] = 50.0

    bias = np.ones((20, 20), dtype=float) * 1000.0
    dark = np.ones((20, 20), dtype=float) * 10.0
    flat = np.ones((20, 20), dtype=float) * 10000.0
    bias[3, 3] = 1060.0

    bpm, stats = create_calibration_bpm(bias, dark, flat, config=config)

    assert has_defect(bpm[3, 3], BPM_BIT_PIX_BRIGHT)
    assert stats["bias"]["threshold_offset_adu"] == pytest.approx(50.0)


def test_create_calibration_bpm_batch_supports_combine_modes_and_validation() -> None:
    rng = np.random.default_rng(55)
    bias_frames = rng.normal(1000.0, 1.0, size=(3, 12, 12))
    dark_frames = rng.normal(10.0, 1.0, size=(3, 12, 12))
    flat_frames = rng.normal(10000.0, 10.0, size=(3, 12, 12))
    dark_frames[:, 4, 4] = np.array([10.0, 30.0, 60.0])

    bpm, masters, stats = create_calibration_bpm_batch(
        bias_frames,
        dark_frames,
        flat_frames,
        combine_method="sigma_clip",
    )

    assert bpm.shape == (12, 12)
    assert masters.shape == (3, 12, 12)
    assert stats["combine_method"] == "sigma_clip"
    assert "spatial" in stats and "temporal" in stats

    _, _, mean_stats = create_calibration_bpm_batch(
        bias_frames,
        dark_frames,
        flat_frames,
        combine_method="mean",
    )
    assert mean_stats["combine_method"] == "mean"

    with pytest.raises(ValueError, match="combine_method must be"):
        create_calibration_bpm_batch(bias_frames, dark_frames, flat_frames, combine_method="sum")

    with pytest.raises(ValueError, match="bias_frames must be 3-D"):
        create_calibration_bpm_batch(np.ones((12, 12)), dark_frames, flat_frames)  # type: ignore[arg-type]
