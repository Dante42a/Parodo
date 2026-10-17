"""
Stage 2 – rule-based classification from per-tooth measurements.

Текущая реализация — классификация только по убыли кости:
  Normal           – bone_loss_ratio < normal_threshold
  Periodontitis    – bone_loss_ratio >= normal_threshold

Целевая (полная) логика с учётом кортикальной пластинки пока отключена:
пластинка размечена лишь на 60 снимках и сегментируется ненадёжно (Dice ≤ 0,12),
поэтому дифференциация пародонтита и пародонтоза по ней временно не выполняется.
  Normal           – cortical plate present, bone_loss_ratio < normal_threshold
  Periodontitis    – cortical plate absent  (regardless of bone loss ratio)
  Periodontosis    – cortical plate present, bone_loss_ratio >= normal_threshold

Spread (for Periodontitis/Periodontosis):
  Local            – fewer than generalized_min_teeth affected
  Generalized      – generalized_min_teeth or more affected

Severity (per affected tooth):
  Mild   – bone_loss_ratio <= mild_threshold (≈ 1/3 root)
  Medium – bone_loss_ratio <= severe_threshold (≈ 2/3 root)
  Severe – bone_loss_ratio > severe_threshold
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from omegaconf import DictConfig

from src.pipeline.stage2_measure import ToothMeasurement


@dataclass
class ToothDiagnosis:
    tooth_id: int
    condition: str      # "Normal" | "Periodontitis" | "Periodontosis"
    severity: Optional[str]  # "Mild" | "Medium" | "Severe" | None (for Normal)
    bone_loss_ratio: float
    cortical_coverage: float
    rule_id: str


@dataclass
class ImageDiagnosis:
    per_tooth: List[ToothDiagnosis]
    overall_condition: str    # most severe condition across teeth
    spread: Optional[str]     # "Local" | "Generalized" | None
    overall_severity: Optional[str]  # worst severity
    num_affected: int
    status: str = "ok"
    reasons: Optional[List[str]] = None
    rule_version: str = "research-rules-v1"
    disclaimer: str = (
        "ВАЖНО: Данный результат предназначен исключительно для исследовательских целей. "
        "Система не является диагностическим медицинским инструментом. "
        "Интерпретация результатов должна выполняться квалифицированным врачом-стоматологом."
    )


def classify_tooth(m: ToothMeasurement, cfg_stage2) -> ToothDiagnosis:
    if not math.isfinite(m.bone_loss_ratio) or not 0.0 <= m.bone_loss_ratio <= 1.0:
        raise ValueError(f"Invalid bone_loss_ratio for tooth {m.tooth_id}")
    if m.root_length_px <= 0:
        raise ValueError(f"Invalid root length for tooth {m.tooth_id}")
    # кортикальная пластинка размечена, но сегментируется ненадёжно (Dice ≤ 0,12) —
    # пока классифицируем только по убыли кости; cortical-логика отключена
    if m.bone_loss_ratio < cfg_stage2.normal_threshold:
        condition = "Normal"
        rule_id = "KB-02"
    else:
        condition = "Periodontitis"
        rule_id = "KB-03-05"

    if condition == "Normal":
        severity = None
    elif m.bone_loss_ratio <= cfg_stage2.mild_threshold:
        severity = "Mild"
    elif m.bone_loss_ratio <= cfg_stage2.severe_threshold:
        severity = "Medium"
    else:
        severity = "Severe"

    return ToothDiagnosis(
        tooth_id=m.tooth_id,
        condition=condition,
        severity=severity,
        bone_loss_ratio=m.bone_loss_ratio,
        cortical_coverage=m.cortical_coverage,
        rule_id=rule_id,
    )


_SEVERITY_ORDER = {"Mild": 0, "Medium": 1, "Severe": 2}
_CONDITION_ORDER = {"Normal": 0, "Periodontosis": 1, "Periodontitis": 2}


def classify_image(
    measurements: List[ToothMeasurement],
    cfg: DictConfig,
) -> ImageDiagnosis:
    s2 = cfg.stage2
    per_tooth = [classify_tooth(m, s2) for m in measurements]

    if not per_tooth:
        return ImageDiagnosis(
            per_tooth=[],
            overall_condition="InsufficientData",
            spread=None,
            overall_severity=None,
            num_affected=0,
            status="needs_review",
            reasons=["Не удалось получить достоверные измерения зубов"],
        )

    affected = [t for t in per_tooth if t.condition != "Normal"]
    num_affected = len(affected)

    if num_affected == 0:
        return ImageDiagnosis(
            per_tooth=per_tooth,
            overall_condition="Normal",
            spread=None,
            overall_severity=None,
            num_affected=0,
        )

    spread = "Generalized" if num_affected >= s2.generalized_min_teeth else "Local"

    worst_condition = max(affected, key=lambda t: _CONDITION_ORDER[t.condition]).condition
    worst_severity = max(
        (t.severity for t in affected if t.severity),
        key=lambda s: _SEVERITY_ORDER[s],
        default=None,
    )

    return ImageDiagnosis(
        per_tooth=per_tooth,
        overall_condition=worst_condition,
        spread=spread,
        overall_severity=worst_severity,
        num_affected=num_affected,
    )
