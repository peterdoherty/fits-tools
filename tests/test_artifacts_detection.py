#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import numpy as np
import pytest

import fits_tools.artifacts.detection as artifact_detection
from fits_tools.artifacts.detection import (
    _compute_downsample_factor,
    _grow_trail_1d,
    align_frames_cross_correlation,
    convert_readnoise_to_adu,
    detect_all_artifacts,
    detect_bloom,
    detect_cosmic_rays,
    detect_crosstalk,
    detect_fixed_pattern_noise,
    detect_fringing,
    detect_persistence,
    detect_saturated_pixels,
    detect_statistical_outliers,
    detect_thermal_gradient,
    create_edge_mask_from_shift,
    flag_outliers,
    get_default_artifact_config,
    interpolate_bad_pixels,
)
from fits_tools.bitmask import (
    BPM_BIT_BLOOM,
    BPM_BIT_COSMIC,
    BPM_BIT_OUTLIER,
    BPM_BIT_SATURATED,
)


def test_detect_saturated_pixels_flags_pixels_above_adc_threshold() -> None:
    frame = np.full((32, 32), 1000.0)
    frame[10, 10] = 65535.0
    frame[20, 20] = 63000.0

    dq, stats = detect_saturated_pixels(frame, bit_depth=16, saturation_threshold=0.95)

    assert dq[10, 10] & BPM_BIT_SATURATED
    assert dq[20, 20] & BPM_BIT_SATURATED
    assert stats["num_saturated"] == 2


def test_detect_cosmic_rays_sigma_clip_detects_isolated_spike() -> None:
    frame = np.full((64, 64), 1000.0)
    frame[12, 13] = 2000.0

    dq, stats = detect_cosmic_rays(
        frame,
        method="sigma_clip",
        sigma=5.0,
        min_sigma=1.0,
    )

    assert dq[12, 13] & BPM_BIT_COSMIC
    assert stats["method"] == "sigma_clip"
    assert stats["num_cosmic_rays"] >= 1


def test_detect_bloom_finds_vertical_trail_from_saturated_seed() -> None:
    frame = np.full((128, 128), 100.0)
    frame[64, 64] = 65535.0
    for offset in range(1, 10):
        frame[64 + offset, 64] = 9000.0 - offset * 250.0

    dq, stats = detect_bloom(
        frame,
        saturation_level=60000.0,
        bleed_direction="vertical",
        min_trail_length=5,
        trail_threshold_factor=0.1,
    )

    assert stats["num_saturated_pixels"] == 1
    assert stats["num_bloom_pixels"] >= 5
    assert np.count_nonzero(dq[:, 64] & BPM_BIT_BLOOM) >= 5


def test_detect_persistence_reports_edge_mask_stat_consistently() -> None:
    frame = np.zeros((32, 32), dtype=float)
    reference = np.zeros_like(frame)

    dq, stats = detect_persistence(frame, reference, threshold_sigma=3.0)

    assert dq.shape == frame.shape
    assert "num_edge_pixels_masked" in stats
    assert stats["num_pixels"] == 0


def test_detect_statistical_outliers_flags_high_and_low_outliers() -> None:
    rng = np.random.default_rng(42)
    frame = rng.normal(100.0, 5.0, size=(50, 50))
    frame[5, 5] = 300.0
    frame[40, 40] = -100.0

    dq, stats = detect_statistical_outliers(frame, sigma=3.0)

    assert dq[5, 5] & BPM_BIT_OUTLIER
    assert dq[40, 40] & BPM_BIT_OUTLIER
    assert stats["num_outliers"] >= 2


def test_flag_outliers_wraps_boolean_mask_into_dq_bits() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True
    mask[5:8, 2] = True

    dq, stats = flag_outliers(mask)

    assert dq[1, 1] & BPM_BIT_OUTLIER
    assert np.all((dq[5:8, 2] & BPM_BIT_OUTLIER) != 0)
    assert stats["num_outliers"] == 4


def test_detect_fringing_rejects_nyquist_violating_periods() -> None:
    with pytest.raises(ValueError, match="Minimum fringe period"):
        detect_fringing(np.ones((64, 64)), fringe_period_range=(1, 10))


def test_detect_thermal_gradient_reports_adaptive_downsampling() -> None:
    frame = np.tile(np.linspace(0.0, 100.0, 512), (512, 1))

    dq, stats = detect_thermal_gradient(frame)

    assert dq.shape == frame.shape
    assert "gradient_detected" in stats
    assert "downsample_factor" in stats
    assert stats["points_used"] > 0


def test_detect_all_artifacts_returns_nested_stats_and_skips_persistence_without_reference() -> None:
    frame = np.full((64, 64), 1000.0)
    frame[4, 4] = 65535.0
    frame[20, 20] = 2000.0

    dq, stats = detect_all_artifacts(frame)

    assert dq.shape == frame.shape
    assert set(["saturation", "bloom", "cosmic_rays", "persistence"]).issubset(stats)
    assert stats["persistence"]["skipped"] is True
    assert stats["artifacts_only"] >= 1


def test_default_artifact_config_is_customizable() -> None:
    config = get_default_artifact_config()
    config["cosmic_rays"]["min_sigma"] = 0.5

    frame = np.full((32, 32), 1000.0)
    frame[3, 3] = 2000.0
    dq, stats = detect_all_artifacts(frame, config=config)

    assert dq[3, 3] & BPM_BIT_COSMIC
    assert stats["cosmic_rays"]["min_sigma"] == pytest.approx(0.5)


def test_convert_readnoise_to_adu_uses_inverse_gain_conversion() -> None:
    assert convert_readnoise_to_adu(5.0, 1.5) == pytest.approx(5.0 / 1.5)


def test_convert_readnoise_to_adu_validates_inputs() -> None:
    with pytest.raises(ValueError, match="gain must be positive"):
        convert_readnoise_to_adu(5.0, 0.0)

    with pytest.raises(ValueError, match="readnoise must be non-negative"):
        convert_readnoise_to_adu(-1.0, 1.5)


def test_detect_cosmic_rays_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be 'sigma_clip' or 'lacosmic'"):
        detect_cosmic_rays(np.ones((8, 8)), method="unknown")


def test_detect_cosmic_rays_sigma_clip_adjusts_even_box_size_and_validates_min_sigma() -> None:
    frame = np.full((32, 32), 100.0)
    frame[8, 8] = 300.0

    dq, stats = detect_cosmic_rays(
        frame,
        method="sigma_clip",
        box_size=4,
        sigma=3.0,
        min_sigma=1.0,
    )

    assert dq[8, 8] & BPM_BIT_COSMIC
    assert stats["box_size"] == 5

    with pytest.raises(ValueError, match="min_sigma must be positive"):
        detect_cosmic_rays(frame, method="sigma_clip", min_sigma=0.0)


def test_detect_cosmic_rays_lacosmic_import_error_when_backend_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_detection, "_LACOSMIC_AVAILABLE", False)
    monkeypatch.setattr(artifact_detection, "_astroscrappy", None)

    with pytest.raises(ImportError, match="requires astroscrappy"):
        detect_cosmic_rays(np.ones((8, 8)), method="lacosmic")


def test_detect_cosmic_rays_lacosmic_passes_mask_and_flags_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeAstroScrappy:
        @staticmethod
        def detect_cosmics(frame: np.ndarray, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
            captured["frame_dtype"] = frame.dtype
            captured["inmask"] = kwargs["inmask"]
            return np.array([[False, True], [False, False]]), frame

    monkeypatch.setattr(artifact_detection, "_LACOSMIC_AVAILABLE", True)
    monkeypatch.setattr(artifact_detection, "_astroscrappy", FakeAstroScrappy())

    existing_bpm = np.array([[0, 1], [0, 0]], dtype=np.uint32)
    dq, stats = detect_cosmic_rays(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        method="lacosmic",
        existing_bpm=existing_bpm,
        lacosmic_iterations=2,
    )

    assert dq[0, 1] & BPM_BIT_COSMIC
    assert stats["method"] == "lacosmic"
    assert stats["iterations_requested"] == 2
    assert captured["frame_dtype"] == np.float32
    assert np.array_equal(captured["inmask"], existing_bpm.astype(bool))


def test_detect_bloom_validates_direction_and_mask_shape() -> None:
    frame = np.ones((8, 8))

    with pytest.raises(ValueError, match="bleed_direction must be"):
        detect_bloom(frame, bleed_direction="diagonal")

    with pytest.raises(ValueError, match="existing_bpm shape"):
        detect_bloom(frame, existing_bpm=np.zeros((4, 4), dtype=np.uint32))


def test_detect_bloom_handles_horizontal_and_masked_seeds() -> None:
    frame = np.full((20, 20), 10.0)
    frame[10, 10] = 65000.0
    frame[10, 11:16] = 9000.0
    existing_bpm = np.zeros_like(frame, dtype=np.uint32)
    existing_bpm[10, 10] = 1

    dq, stats = detect_bloom(
        frame,
        saturation_level=60000.0,
        bleed_direction="horizontal",
        min_trail_length=3,
        trail_threshold_factor=0.1,
        existing_bpm=existing_bpm,
    )

    assert stats["num_saturated_total"] == 1
    assert stats["num_saturated_excluded"] == 1
    assert stats["num_bloom_pixels"] == 0
    assert not np.any(dq & BPM_BIT_BLOOM)


def test_grow_trail_1d_covers_forward_backward_and_validation() -> None:
    line = np.array([1.0, 8.0, 8.0, 8.0, 1.0])

    assert np.array_equal(_grow_trail_1d(line, 1, 1, 5.0), np.array([1, 2, 3]))
    assert np.array_equal(_grow_trail_1d(line, 3, -1, 5.0), np.array([3, 2, 1]))
    assert np.array_equal(_grow_trail_1d(line, 9, 1, 5.0), np.array([], dtype=int))

    with pytest.raises(ValueError, match="direction must be -1 or 1"):
        _grow_trail_1d(line, 1, 0, 5.0)


def test_detect_persistence_validates_shape_and_can_align_frames() -> None:
    with pytest.raises(ValueError, match="must have the same shape"):
        detect_persistence(np.zeros((4, 4)), np.zeros((5, 5)))

    reference = np.zeros((32, 32), dtype=float)
    reference[10:14, 10:14] = 20.0
    frame = np.roll(reference, shift=(1, -2), axis=(0, 1))
    frame[20:23, 20:23] += 15.0

    dq, stats = detect_persistence(
        frame,
        reference,
        align_frames=True,
        threshold_sigma=2.0,
        min_region_size=4,
    )

    assert stats["alignment_offset"] is not None
    assert stats["num_edge_pixels_masked"] >= 1
    assert stats["num_pixels"] == 0
    assert not np.any(dq & BPM_BIT_OUTLIER)


def test_detect_persistence_handles_alignment_failure_and_source_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
        raise RuntimeError("boom")

    monkeypatch.setattr(artifact_detection, "align_frames_cross_correlation", explode)

    frame = np.zeros((16, 16), dtype=float)
    reference = np.zeros_like(frame)
    frame[5:8, 5:8] = 20.0
    source_mask = np.zeros_like(frame, dtype=bool)
    source_mask[5:8, 5:8] = True

    dq, stats = detect_persistence(
        frame,
        reference,
        align_frames=True,
        source_mask=source_mask,
        threshold_sigma=1.0,
        min_region_size=1,
    )

    assert stats["alignment_offset"] is None
    assert stats["num_pixels"] == 0
    assert not np.any(dq & BPM_BIT_OUTLIER)


def test_align_frames_cross_correlation_returns_edge_mask_for_shifted_reference() -> None:
    reference = np.zeros((32, 32), dtype=float)
    reference[8:12, 8:12] = 5.0
    target = np.roll(reference, shift=(2, -1), axis=(0, 1))

    aligned, offset, edge_mask = align_frames_cross_correlation(reference, target)

    assert aligned.shape == reference.shape
    assert edge_mask is not None
    assert edge_mask.shape == reference.shape
    assert abs(offset[0] - 2) < 0.6
    assert abs(offset[1] + 1) < 0.6


def test_create_edge_mask_from_shift_marks_expected_edges() -> None:
    mask = create_edge_mask_from_shift((6, 7), np.array([2.2, -1.1]))

    assert np.all(mask[:3, :])
    assert np.all(mask[:, -2:])
    assert not mask[3, 2]


def test_detect_crosstalk_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        detect_crosstalk(np.zeros((4, 4)), channel_geometry={})


def test_detect_statistical_outliers_validates_tail_and_reference_inputs() -> None:
    frame = np.ones((4, 4))

    with pytest.raises(ValueError, match="tail must be 'both', 'high', or 'low'"):
        detect_statistical_outliers(frame, tail="middle")

    with pytest.raises(ValueError, match="Cannot provide both reference_frames"):
        detect_statistical_outliers(
            frame,
            reference_frames=np.ones((2, 4, 4)),
            reference_median=np.ones((4, 4)),
            reference_sigma=np.ones((4, 4)),
        )

    with pytest.raises(ValueError, match="must be provided together"):
        detect_statistical_outliers(frame, reference_median=np.ones((4, 4)))

    with pytest.raises(ValueError, match="reference_median shape"):
        detect_statistical_outliers(
            frame,
            reference_median=np.ones((5, 5)),
            reference_sigma=np.ones((5, 5)),
        )


def test_detect_statistical_outliers_supports_reference_maps_and_one_sided_tails() -> None:
    frame = np.array([[10.0, 40.0], [1.0, 10.0]])
    reference_median = np.full((2, 2), 10.0)
    reference_sigma = np.full((2, 2), 2.0)

    dq_high, stats_high = detect_statistical_outliers(
        frame,
        reference_median=reference_median,
        reference_sigma=reference_sigma,
        sigma=5.0,
        tail="high",
    )
    dq_low, stats_low = detect_statistical_outliers(
        frame,
        reference_median=reference_median,
        reference_sigma=reference_sigma,
        sigma=4.0,
        tail="low",
    )

    assert dq_high[0, 1] & BPM_BIT_OUTLIER
    assert stats_high["method"] == "reference_maps"
    assert stats_high["num_high"] == 1
    assert dq_low[1, 0] & BPM_BIT_OUTLIER
    assert stats_low["num_low"] == 1


def test_detect_statistical_outliers_returns_empty_when_reference_sigma_has_no_valid_pixels() -> None:
    frame = np.ones((3, 3))
    dq, stats = detect_statistical_outliers(
        frame,
        reference_median=np.ones((3, 3)),
        reference_sigma=np.zeros((3, 3)),
    )

    assert stats["num_outliers"] == 0
    assert np.isnan(stats["threshold"])
    assert not np.any(dq)


def test_detect_statistical_outliers_supports_reference_stack_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    reference_frames = np.stack([np.full((4, 4), 10.0) for _ in range(3)])
    frame = np.full((4, 4), 10.0)
    frame[1, 2] = 30.0

    monkeypatch.setattr(artifact_detection, "sigma_clip_stats", lambda *args, **kwargs: (0.0, 1.0))

    dq, stats = detect_statistical_outliers(
        frame,
        reference_frames=reference_frames,
        sigma=3.0,
    )

    assert dq[1, 2] & BPM_BIT_OUTLIER
    assert stats["method"] == "reference_stack"


def test_detect_all_artifacts_counts_existing_bpm_in_total_only() -> None:
    frame = np.full((16, 16), 1000.0)
    existing_bpm = np.zeros_like(frame, dtype=np.uint32)
    existing_bpm[0, 0] = 1

    dq, stats = detect_all_artifacts(frame, existing_bpm=existing_bpm)

    assert stats["artifacts_only"] == int(np.sum(dq > 0))
    assert stats["total_flagged"] >= stats["artifacts_only"]
    assert stats["total_flagged"] == int(np.sum((dq | existing_bpm) > 0))


def test_detect_fringing_validates_period_range_and_handles_invalid_flat() -> None:
    flat = np.ones((16, 16))

    with pytest.raises(TypeError, match="must contain integers"):
        detect_fringing(flat, fringe_period_range=(5.5, 10))

    with pytest.raises(ValueError, match="exceeds minimum image dimension"):
        detect_fringing(flat, fringe_period_range=(5, 32))

    with pytest.raises(ValueError, match="must be less than max_period"):
        detect_fringing(flat, fringe_period_range=(8, 8))

    dq, stats = detect_fringing(np.zeros((16, 16)), fringe_period_range=(5, 8))

    assert stats["fringe_detected"] is False
    assert stats["num_pixels_affected"] == 0
    assert not np.any(dq)


def test_detect_fringing_returns_empty_when_no_significant_peak_found(monkeypatch: pytest.MonkeyPatch) -> None:
    flat = np.ones((64, 64), dtype=float)
    existing_bpm = np.zeros_like(flat, dtype=np.uint32)
    existing_bpm[0, 0] = 1

    monkeypatch.setattr(
        artifact_detection.np,
        "max",
        lambda arr, *args, **kwargs: 1.0,
    )
    monkeypatch.setattr(
        artifact_detection.np,
        "median",
        lambda arr, *args, **kwargs: 1.0,
    )
    dq, stats = detect_fringing(
        flat,
        fringe_period_range=(5, 20),
        threshold_amplitude=0.001,
        existing_bpm=existing_bpm,
    )

    assert stats["fringe_detected"] is False
    assert stats["dominant_period"] is None
    assert not np.any(dq)


def test_interpolate_bad_pixels_returns_copy_and_smooths_large_masks() -> None:
    image = np.arange(100, dtype=float).reshape(10, 10)
    bad_mask = np.zeros((10, 10), dtype=bool)
    bad_mask[2:6, 2:5] = True

    filled = interpolate_bad_pixels(image, bad_mask)

    assert filled.shape == image.shape
    assert not np.array_equal(filled[bad_mask], image[bad_mask])
    assert np.array_equal(interpolate_bad_pixels(image, np.zeros_like(bad_mask)), image)


def test_detect_fixed_pattern_noise_validates_stack_and_detects_banding() -> None:
    with pytest.raises(ValueError, match="Need at least 3 frames"):
        detect_fixed_pattern_noise(np.zeros((2, 8, 8)))

    stack = np.stack([np.full((8, 8), 10.0 + i) for i in range(3)])
    for frame in stack:
        frame[3, :] += 50.0
        frame[:, 5] += 50.0

    dq, stats = detect_fixed_pattern_noise(stack, threshold_sigma=1.0)

    assert "banding_horizontal" in stats["patterns_detected"]
    assert "banding_vertical" in stats["patterns_detected"]
    assert stats["num_pixels_affected"] > 0
    assert np.any(dq & BPM_BIT_OUTLIER)


def test_detect_fixed_pattern_noise_handles_degenerate_noise_floor() -> None:
    dark_stack = np.full((3, 8, 8), np.nan)
    dq, stats = detect_fixed_pattern_noise(dark_stack)

    assert stats["patterns_detected"] == []
    assert stats["num_pixels_affected"] == 0
    assert not np.any(dq)


def test_detect_fixed_pattern_noise_flags_checkerboard_pattern() -> None:
    pattern = np.zeros((8, 8), dtype=float)
    pattern[::2, :] = 20.0
    stack = np.stack([pattern + i for i in range(3)])

    dq, stats = detect_fixed_pattern_noise(
        stack,
        threshold_sigma=1.0,
        pattern_types=["checkerboard"],
    )

    assert "checkerboard" in stats["patterns_detected"]
    assert stats["checkerboard"]["analysis_region"]["rows_used"] == 8
    assert not np.any(dq & BPM_BIT_OUTLIER)


def test_detect_thermal_gradient_reports_detected_gradient_with_explicit_downsample() -> None:
    frame = np.add.outer(np.linspace(0.0, 20.0, 128), np.linspace(0.0, 10.0, 128))

    dq, stats = detect_thermal_gradient(
        frame,
        max_gradient=0.001,
        downsample_factor=3,
    )

    assert dq.shape == frame.shape
    assert stats["gradient_detected"] is True
    assert stats["downsample_factor"] == 3


def test_compute_downsample_factor_covers_small_and_large_shapes() -> None:
    assert _compute_downsample_factor((100, 100)) == 1

    factor = _compute_downsample_factor((4000, 4000))
    assert factor % 2 == 1
    assert factor >= 1


def test_detect_thermal_gradient_validates_downsample_and_handles_too_few_valid_pixels() -> None:
    frame = np.tile(np.linspace(0.0, 1.0, 32), (32, 1))

    with pytest.raises(ValueError, match="downsample_factor must be >= 1"):
        detect_thermal_gradient(frame, downsample_factor=0)

    existing_bpm = np.ones_like(frame, dtype=np.uint32)
    existing_bpm[:9, :11] = 0
    dq, stats = detect_thermal_gradient(frame, existing_bpm=existing_bpm, downsample_factor=1)

    assert stats["gradient_detected"] is False
    assert stats["points_used"] < 100
    assert not np.any(dq)
