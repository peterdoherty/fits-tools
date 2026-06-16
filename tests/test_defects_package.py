#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

from __future__ import annotations

import fits_tools.defects as defects


def test_defects_package_exports_key_symbols() -> None:
    assert "create_bpm" in defects.__all__
    assert "detect_hot_pixels" in defects.__all__
    assert "save_bpm_to_fits" in defects.__all__
    assert defects.BPM_BIT_PIX_HOT in defects.FLAG_NAMES


def test_get_defect_bit_info_contains_named_bits() -> None:
    text = defects.get_defect_bit_info()

    assert "Defect Bit Flag System" in text
    assert "PIX_HOT" in text
    assert "Pixels can carry multiple flags" in text


def test_get_combined_masks_info_lists_predefined_masks_and_members() -> None:
    text = defects.get_combined_masks_info()

    assert "Predefined Combined Masks" in text
    assert "BPM_MASK_PERMANENT" in text
    assert "Includes:" in text
