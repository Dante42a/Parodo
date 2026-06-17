"""
Stage 2 – rule-based classification from per-tooth measurements.

Conditions:
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


@dataclass
class ImageDiagnosis:
    per_tooth: List[ToothDiagnosis]
    overall_condition: str    # most severe condition across teeth
    spread: Optional[str]     # "Local" | "Generalized" | None
    overall_severity: Optional[str]  # worst severity
    num_affected: int
    disclaimer: str = (
        "ВАЖНО: Данный результат предназначен исключительно для исследовательских целей. "
        "Система не является диагностическим медицинским инструментом. "
        "Интерпретация результатов должна выполняться квалифицированным врачом-стоматологом."
    )


def classify_tooth(m: ToothMeasurement, cfg_stage2) -> ToothDiagnosis:
    cortical_present = m.cortical_coverage >= cfg_stage2.cortical_min_coverage

    if not cortical_present:
        condition = "Periodontitis"
    elif m.bone_loss_ratio < cfg_stage2.normal_threshold:
        condition = "Normal"
    else:
        condition = "Periodontosis"

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
    )


_SEVERITY_ORDER = {"Mild": 0, "Medium": 1, "Severe": 2}
_CONDITION_ORDER = {"Normal": 0, "Periodontosis": 1, "Periodontitis": 2}


def classify_image(
    measurements: List[ToothMeasurement],
    cfg: DictConfig,
) -> ImageDiagnosis:
    s2 = cfg.stage2
    per_tooth = [classify_tooth(m, s2) for m in measurements]

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
