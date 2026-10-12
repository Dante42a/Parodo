"""Smoke-тесты stage2: геометрия убыли кости и правиловая классификация (без весов моделей)."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.pipeline.stage1_segment import ToothInstance
from src.pipeline.stage2_classify import classify_image
from src.pipeline.stage2_measure import ToothMeasurement, measure_bone_loss

H, W = 200, 100
CFG = OmegaConf.load("configs/base.yaml")


def lower_tooth(y1=110, y2=190, x1=40, x2=60):
    # Зуб нижней челюсти: центр ниже H/2 → коронка сверху, корень вниз
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2 + 1, x1:x2 + 1] = True
    return ToothInstance(instance_id=1, mask=mask, bbox=(y1, x1, y2, x2), confidence=1.0)


def measurement(ratio, tooth_id=1):
    return ToothMeasurement(
        tooth_id=tooth_id, label="tooth", bbox=(0, 0, 1, 1), root_length_px=100,
        bone_loss_height_px=int(ratio * 100), bone_loss_ratio=ratio, cortical_coverage=0.0,
        cej_y=0, root_tip_y=0, bone_crest_y=0, confidence=1.0,
    )


def test_bone_at_cej_gives_no_loss():
    tooth = lower_tooth()
    cej_y = 110 + int(0.40 * 80)  # = 142
    bone = np.zeros((H, W), dtype=bool)
    bone[cej_y:, :] = True
    [m] = measure_bone_loss([tooth], bone, np.zeros_like(bone), cej_fraction=0.40)
    assert m.cej_y == cej_y
    assert m.bone_loss_ratio == 0.0


def test_half_root_bone_loss():
    tooth = lower_tooth()
    # корень: cej_y=142 … root_tip=190 → длина 48; гребень на 166 → потеря 24/48
    bone = np.zeros((H, W), dtype=bool)
    bone[166:, :] = True
    [m] = measure_bone_loss([tooth], bone, np.zeros_like(bone), cej_fraction=0.40)
    assert m.bone_loss_ratio == pytest.approx(0.5)


def test_no_bone_is_conservative_zero():
    tooth = lower_tooth()
    empty = np.zeros((H, W), dtype=bool)
    [m] = measure_bone_loss([tooth], empty, empty)
    assert m.bone_loss_ratio == 0.0


def test_tiny_instance_skipped():
    empty = np.zeros((H, W), dtype=bool)
    assert measure_bone_loss([lower_tooth(y1=150, y2=152)], empty, empty) == []


def test_all_normal():
    diag = classify_image([measurement(0.0), measurement(0.05, 2)], CFG)
    assert diag.overall_condition == "Normal"
    assert diag.num_affected == 0
    assert diag.spread is None


@pytest.mark.parametrize("ratio, severity", [(0.2, "Mild"), (0.5, "Medium"), (0.9, "Severe")])
def test_severity_thresholds(ratio, severity):
    diag = classify_image([measurement(ratio)], CFG)
    assert diag.overall_condition == "Periodontitis"
    assert diag.overall_severity == severity
    assert diag.spread == "Local"


def test_generalized_spread():
    n = CFG.stage2.generalized_min_teeth
    diag = classify_image([measurement(0.5, i) for i in range(n)], CFG)
    assert diag.spread == "Generalized"
    assert diag.num_affected == n
