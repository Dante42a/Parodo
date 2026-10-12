from types import SimpleNamespace

from src.pipeline.stage2_classify import classify_image, classify_tooth
from src.pipeline.stage2_measure import ToothMeasurement

CFG = SimpleNamespace(
    stage2=SimpleNamespace(
        normal_threshold=0.1,
        mild_threshold=1 / 3,
        severe_threshold=2 / 3,
        generalized_min_teeth=8,
    )
)


def measurement(ratio: float, tooth_id: int = 11) -> ToothMeasurement:
    return ToothMeasurement(tooth_id, "test", (0, 0, 20, 10), 10, int(ratio * 10), ratio, 0, 4, 14, 5, 1)


def test_rule_boundaries_are_deterministic():
    assert classify_tooth(measurement(0.099), CFG.stage2).condition == "Normal"
    assert classify_tooth(measurement(0.1), CFG.stage2).severity == "Mild"
    assert classify_tooth(measurement(1 / 3), CFG.stage2).severity == "Mild"
    assert classify_tooth(measurement(2 / 3), CFG.stage2).severity == "Medium"
    assert classify_tooth(measurement(0.9), CFG.stage2).severity == "Severe"


def test_empty_measurements_require_review():
    result = classify_image([], CFG)
    assert result.status == "needs_review"
    assert result.overall_condition == "InsufficientData"


def test_invalid_measurement_is_rejected():
    bad = measurement(1.2)
    try:
        classify_tooth(bad, CFG.stage2)
    except ValueError as exc:
        assert "bone_loss_ratio" in str(exc)
    else:
        raise AssertionError("invalid ratio was accepted")
