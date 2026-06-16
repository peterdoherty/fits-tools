#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import io

import numpy as np
import pytest

from fits_tools.bitmask import (
    BPM_BIT_COL_EXCLUDED,
    BPM_BIT_COL_BAD,
    BPM_BIT_PIX_BAD,
    BPM_BIT_PIX_CLUSTER,
    BPM_BIT_PIX_DEAD,
    BPM_BIT_PIX_EXCLUDED,
    BPM_BIT_PIX_HOT,
    BPM_BIT_PIX_NONLIN,
    BPM_BIT_ROW_BAD,
    BPM_BIT_ROW_EXCLUDED,
    BPM_BIT_SATURATED,
    BPM_MASK_COL,
    BPM_MASK_HOT,
    BPM_MASK_PERMANENT,
    BPM_MASK_TRANSIENT,
    DefectType,
    apply_bpm,
    bit_index,
    build_good_pixel_mask,
    check_bad_rows_cols,
    combine_flags,
    create_bpm,
    create_master_bpm,
    count_defects,
    describe_pixel_defects,
    describe_pixel_defects_verbose,
    export_to_ds9_region,
    filter_bpm,
    flag_bad_rows_cols,
    flag_defect_clusters,
    flag_excluded_border,
    flag_excluded_cols,
    flag_excluded_pixels,
    flag_excluded_rows,
    flag_excluded_rows_cols,
    format_bit_comments,
    format_pixel_status,
    get_cluster_statistics,
    get_defect_mask,
    get_definitions_payload,
    get_bpm_statistics,
    get_defect_types,
    has_defect,
    is_perfect_pixel,
    is_usable_pixel,
    merge_bpms,
    summarize_bpm,
    to_boolean_mask,
    validate_bpm,
)


def test_create_bpm_defaults_to_uint32() -> None:
    bpm = create_bpm((3, 4))
    assert bpm.dtype == np.uint32
    assert np.count_nonzero(bpm) == 0


def test_get_defect_types_and_has_defect_support_combined_flags() -> None:
    value = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN

    assert set(get_defect_types(value)) == {"PIX_HOT", "PIX_NONLIN"}
    assert has_defect(value, BPM_MASK_HOT)
    assert not has_defect(value, BPM_MASK_COL)


def test_merge_bpms_supports_or_and_diff_operations() -> None:
    bpm1 = create_bpm((2, 2))
    bpm2 = create_bpm((2, 2))
    bpm1[0, 0] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN
    bpm2[0, 0] = BPM_BIT_PIX_HOT
    bpm2[1, 1] = BPM_BIT_SATURATED

    merged = merge_bpms(bpm1, bpm2, operation="or")
    diff = merge_bpms(bpm1, bpm2, operation="diff")

    assert set(get_defect_types(merged[0, 0])) == {"PIX_HOT", "PIX_NONLIN"}
    assert get_defect_types(merged[1, 1]) == ["SATURATED"]
    assert get_defect_types(diff[0, 0]) == ["PIX_NONLIN"]


def test_apply_bpm_masks_flagged_pixels_and_promotes_integer_arrays() -> None:
    data = np.arange(9, dtype=np.int16).reshape(3, 3)
    bpm = create_bpm((3, 3))
    bpm[1, 1] = BPM_BIT_PIX_HOT
    bpm[:, 0] |= BPM_BIT_COL_BAD

    masked = apply_bpm(data, bpm, defect_mask=BPM_MASK_PERMANENT, fill_value=np.nan)

    assert masked.dtype == np.float64
    assert np.isnan(masked[1, 1])
    assert np.isnan(masked[0, 0])
    assert masked[0, 1] == pytest.approx(1.0)


def test_get_bpm_statistics_counts_individual_and_group_flags() -> None:
    bpm = create_bpm((2, 3))
    bpm[0, 0] = BPM_BIT_PIX_HOT
    bpm[:, 1] |= BPM_BIT_COL_BAD

    stats = get_bpm_statistics(bpm)

    assert stats["PIX_HOT"]["count"] == 1
    assert stats["COL_BAD"]["count"] == 2
    assert stats["MASK_PERMANENT"]["count"] == 3


def test_defect_type_enum_exposes_membership() -> None:
    status = DefectType(BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN)

    assert DefectType.PIX_HOT in status
    assert DefectType.PIX_NONLIN in status


def test_bit_definition_helpers_and_descriptions_cover_edge_cases() -> None:
    payload = get_definitions_payload(nbits=4)
    comments = format_bit_comments(width=80, prefix="BPM: ", include_hex=False)
    value = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN

    assert bit_index(BPM_BIT_PIX_HOT) == 3
    assert payload["nbits"] == 4
    assert [entry["bit"] for entry in payload["bits"]] == sorted(entry["bit"] for entry in payload["bits"])
    assert comments[0].startswith("BPM: Bit flag definitions")
    assert all(len(line) <= 80 for line in comments)
    assert format_pixel_status(value) == "PIX_HOT|PIX_NONLIN"
    assert is_perfect_pixel(0) is True
    assert is_usable_pixel(BPM_BIT_SATURATED, ignore_transient=True) is True
    assert is_usable_pixel(BPM_BIT_SATURATED, ignore_transient=False) is False
    assert describe_pixel_defects(0) == "Good pixel"
    assert "PIX_HOT" in describe_pixel_defects_verbose(value)

    with pytest.raises(ValueError, match="single-bit power of two"):
        bit_index(3)

    with pytest.raises(ValueError, match="exceeds maximum"):
        bit_index(1 << 5, max_bits=5)

    with pytest.raises(ValueError, match="positive integer"):
        format_bit_comments(width=0)


def test_bitmask_mask_building_and_master_bpm_helpers() -> None:
    bpm = create_master_bpm(
        (3, 4),
        dead_pixels=np.array([[0, 1]]),
        hot_pixels=np.array([[2, 3]]),
        bad_columns=np.array([0]),
    )

    assert count_defects(bpm) == 5
    assert combine_flags([BPM_BIT_PIX_HOT, BPM_BIT_PIX_DEAD]) == (BPM_BIT_PIX_HOT | BPM_BIT_PIX_DEAD)

    selected = get_defect_mask(bpm, [BPM_BIT_PIX_HOT, BPM_BIT_COL_BAD])
    filtered = filter_bpm(bpm, [BPM_BIT_PIX_HOT, BPM_BIT_COL_BAD])
    bad_mask = to_boolean_mask(bpm)
    good_mask = to_boolean_mask(bpm, invert=True)

    assert selected.dtype == bool
    assert np.count_nonzero(selected) == 4
    assert filtered[0, 1] == 0
    assert np.array_equal(good_mask, ~bad_mask)


def test_apply_bpm_and_good_pixel_mask_cover_shape_casting_and_custom_conditions() -> None:
    data = np.array([[1.0, np.nan], [-1.0, 0.0]])
    bpm = create_bpm((2, 2))
    bpm[0, 0] = BPM_BIT_PIX_HOT
    bpm[1, 1] = BPM_BIT_SATURATED

    masked = apply_bpm(data, bpm, defect_mask=BPM_MASK_PERMANENT, fill_value=-99.0)
    custom = build_good_pixel_mask(
        data,
        bad_pixel_mask=bpm,
        exclude_inf=False,
        exclude_negative=True,
        exclude_zero=True,
        custom_condition=lambda arr: arr < 2,
    )
    bool_bpm_mask = build_good_pixel_mask(
        data,
        bad_pixel_mask=np.array([[False, True], [False, False]]),
    )

    assert masked[0, 0] == pytest.approx(-99.0)
    assert custom.tolist() == [[False, False], [False, False]]
    assert bool_bpm_mask.tolist() == [[True, False], [True, True]]

    with pytest.raises(ValueError, match="shapes must match"):
        apply_bpm(np.zeros((2, 2)), np.zeros((3, 3), dtype=np.uint32))

    promoted = apply_bpm(
        np.zeros((2, 2), dtype=np.int16),
        np.zeros((2, 2), dtype=np.uint32),
        fill_value=1.5,
    )
    assert promoted.dtype == np.float64

    with pytest.raises(ValueError, match="does not match data shape"):
        build_good_pixel_mask(np.zeros((2, 2)), bad_pixel_mask=np.zeros((3, 3), dtype=bool))

    with pytest.raises(ValueError, match="Custom condition returned shape"):
        build_good_pixel_mask(np.zeros((2, 2)), custom_condition=lambda arr: np.ones((3, 3), dtype=bool))


def test_bpm_statistics_validation_summary_and_exclusion_helpers() -> None:
    bpm = create_bpm((4, 4))
    bpm[0, :] |= BPM_BIT_PIX_HOT
    bpm[:, 1] |= BPM_BIT_COL_BAD

    stats = get_bpm_statistics(bpm, force_sequential=True)
    valid, messages = validate_bpm(bpm, max_defect_fraction=0.1, max_bad_rows=0, max_bad_cols=0)
    summary = summarize_bpm(bpm)
    row_col = flag_bad_rows_cols(bpm, row_threshold=0.5, col_threshold=0.5)
    excluded = flag_excluded_rows(flag_excluded_cols(bpm, np.array([2])), np.array([3]))
    excluded = flag_excluded_pixels(excluded, np.array([[1, 1]]))
    bordered = flag_excluded_border(excluded, nrows=1, ncols=1)

    assert stats["PIX_HOT"]["count"] == 4
    assert valid is False
    assert messages
    assert summary["worst_row_count"] >= 4
    assert np.any((row_col & BPM_BIT_ROW_BAD) != 0)
    assert np.any((row_col & BPM_BIT_COL_BAD) != 0)
    assert np.any((bordered & BPM_BIT_ROW_EXCLUDED) != 0)
    assert np.any((bordered & BPM_BIT_COL_EXCLUDED) != 0)
    assert np.any((bordered & BPM_BIT_PIX_EXCLUDED) != 0)

    with pytest.raises(TypeError, match="unsigned integer"):
        get_bpm_statistics(np.zeros((2, 2), dtype=np.int32))

    with pytest.raises(ValueError, match="must be 2-D"):
        get_bpm_statistics(np.zeros((2, 2, 2), dtype=np.uint32))

    with pytest.raises(ValueError, match="expects a 2-D BPM"):
        flag_excluded_rows_cols(np.zeros((2, 2, 2), dtype=np.uint32))  # type: ignore[name-defined]

    with pytest.raises(ValueError, match="shape \\(N, 2\\)"):
        flag_excluded_pixels(np.zeros((2, 2), dtype=np.uint32), np.array([1, 2, 3]))


def test_bad_row_col_cluster_and_ds9_export_helpers() -> None:
    scipy = pytest.importorskip("scipy")
    assert scipy is not None

    bpm = create_bpm((4, 4))
    bpm[0, 0] = BPM_BIT_PIX_HOT
    bpm[0, 1] = BPM_BIT_PIX_HOT
    bpm[3, 3] = BPM_BIT_PIX_DEAD

    bad_rows, bad_cols = check_bad_rows_cols(
        bpm,
        row_threshold=0.2,
        col_threshold=0.2,
        active_region=(slice(0, 2), slice(0, 2)),
    )
    clustered = flag_defect_clusters(bpm, min_size=2, connectivity=1, also_flag_defect=True)
    cluster_stats = get_cluster_statistics(bpm, connectivity=2)

    output = io.StringIO()
    export_to_ds9_region(clustered, BPM_BIT_PIX_CLUSTER, output, region_type="point", color="green")
    text = output.getvalue()

    assert bad_rows.tolist() == [True, False, False, False]
    assert bad_cols.tolist() == [True, True, False, False]
    assert np.any((clustered & BPM_BIT_PIX_CLUSTER) != 0)
    assert cluster_stats["num_clusters"] >= 1
    assert "point(" in text
    assert "green" in text

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        check_bad_rows_cols(bpm, row_threshold=2.0)

    with pytest.raises(TypeError, match="tuple of two slice objects"):
        check_bad_rows_cols(bpm, active_region=(0, slice(None)))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported region_type"):
        export_to_ds9_region(bpm, BPM_BIT_PIX_HOT, io.StringIO(), region_type="triangle")
