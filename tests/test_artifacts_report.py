#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from fits_tools.artifacts.report import (
    _format_summary_text,
    _validate_artifact_stats,
    create_artifact_report,
    create_artifact_stats_dict,
)
from fits_tools.bitmask import (
    BPM_BIT_BLOOM,
    BPM_BIT_COSMIC,
    BPM_BIT_OUTLIER,
    BPM_BIT_SATURATED,
)


@pytest.fixture
def artifact_stats() -> dict[str, object]:
    return {
        "saturation": {
            "saturation_level": 60000.0,
            "num_saturated": 2,
            "fraction_saturated": 2.0 / 25.0,
        },
        "cosmic_rays": {
            "method": "sigma_clip",
            "num_cosmic_rays": 1,
            "fraction": 1.0 / 25.0,
        },
        "bloom": {
            "bleed_direction": "vertical",
            "num_saturated_clusters": 1,
            "num_saturated_total": 2,
            "num_saturated_excluded": 0,
            "num_trails": 1,
            "num_bloom_pixels": 3,
            "fraction": 3.0 / 25.0,
        },
        "persistence": {
            "num_regions": 1,
            "num_pixels": 2,
            "mean_persistence": 120.0,
            "max_persistence": 150.0,
        },
        "artifacts_only": 5,
        "artifacts_fraction": 5.0 / 25.0,
        "total_flagged": 5,
        "total_fraction": 5.0 / 25.0,
    }


def test_create_artifact_stats_dict_uses_dq_for_summary_counts() -> None:
    dq = np.zeros((4, 4), dtype=np.uint32)
    dq[0, 0] = BPM_BIT_SATURATED
    dq[1, 1] = BPM_BIT_COSMIC

    stats = create_artifact_stats_dict(
        sat_stats={"saturation_level": 60000.0, "num_saturated": 1, "fraction_saturated": 1 / 16},
        cr_stats={"method": "sigma_clip", "num_cosmic_rays": 1, "fraction": 1 / 16},
        bloom_stats={
            "bleed_direction": "vertical",
            "num_saturated_clusters": 1,
            "num_saturated_total": 1,
            "num_saturated_excluded": 0,
            "num_trails": 0,
            "num_bloom_pixels": 0,
            "fraction": 0.0,
        },
        dq=dq,
    )

    assert stats["artifacts_only"] == 2
    assert stats["total_flagged"] == 2
    assert stats["persistence"]["skipped"] is True


def test_create_artifact_stats_dict_without_dq_uses_zero_summary_with_note() -> None:
    stats = create_artifact_stats_dict(
        sat_stats={"saturation_level": 60000.0, "num_saturated": 0, "fraction_saturated": 0.0},
        cr_stats={"method": "sigma_clip", "num_cosmic_rays": 0, "fraction": 0.0},
        bloom_stats={
            "bleed_direction": "vertical",
            "num_saturated_clusters": 0,
            "num_saturated_total": 0,
            "num_saturated_excluded": 0,
            "num_trails": 0,
            "num_bloom_pixels": 0,
            "fraction": 0.0,
        },
        frame_size=100,
    )

    assert stats["artifacts_only"] == 0
    assert "overlap" in str(stats["note"])


def test_validate_artifact_stats_rejects_missing_required_keys(artifact_stats: dict[str, object]) -> None:
    broken = dict(artifact_stats)
    broken.pop("cosmic_rays")

    with pytest.raises(ValueError, match="missing required key 'cosmic_rays'"):
        _validate_artifact_stats(broken)


def test_format_summary_text_covers_detected_and_skipped_persistence(artifact_stats: dict[str, object]) -> None:
    detected = _format_summary_text(artifact_stats)
    skipped_stats = dict(artifact_stats)
    skipped_stats["persistence"] = {"skipped": True}
    skipped = _format_summary_text(skipped_stats)

    assert "ARTIFACT DETECTION SUMMARY" in detected
    assert "Mean flux" in detected
    assert "Skipped" in skipped


def test_create_artifact_report_saves_figure_for_mixed_masks(tmp_path: Path, artifact_stats: dict[str, object]) -> None:
    frame = np.arange(25, dtype=np.float32).reshape(5, 5) - 5.0
    dq = np.zeros((5, 5), dtype=np.uint32)
    dq[0, 0] = BPM_BIT_SATURATED
    dq[1, 2] = BPM_BIT_COSMIC
    dq[3, 1] = BPM_BIT_BLOOM
    dq[4, 4] = BPM_BIT_OUTLIER
    output = tmp_path / "artifact_report.png"

    create_artifact_report(frame, dq, artifact_stats, output_path=output, show_plots=False)

    assert output.exists()
    assert output.stat().st_size > 0


def test_create_artifact_report_rejects_shape_mismatch(artifact_stats: dict[str, object]) -> None:
    frame = np.zeros((4, 4), dtype=np.float32)
    dq = np.zeros((5, 5), dtype=np.uint32)

    with pytest.raises(ValueError, match="does not match dq shape"):
        create_artifact_report(frame, dq, artifact_stats, show_plots=False)
