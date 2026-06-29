"""
Visualization utilities: colored mask overlay, bone ratio annotations.
"""

from typing import List, Optional
import cv2
import numpy as np


# class index → BGR color
CLASS_COLORS = {
    0: (0, 0, 0),         # background — transparent
    1: (0, 0, 200),       # tooth — red
    2: (200, 100, 0),     # bone — blue-ish
    3: (0, 200, 200),     # cortical — yellow
}

ALPHA = 0.45   # mask overlay transparency


def overlay_masks(
    image: np.ndarray,
    mask: np.ndarray,
    measurements=None,
    instances=None,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    image:    (H, W, 3) uint8 RGB
    mask:     (H, W) int с индексами классов 0-3 (от Swin-UNet)
    instances: список ToothInstance от YOLO — если передан, зубы рендерятся по ним
               (более точно, чем семантический блоб Swin-UNet)
    measurements: список ToothMeasurement для отрисовки линий потери кости
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlay = image.copy()

    # Bone и cortical всегда из Swin-UNet семантики
    for cls, color in CLASS_COLORS.items():
        if cls in (0, 1):
            continue
        overlay[mask == cls] = color

    # Зубы: YOLO-инстансы если есть, иначе семантическая маска
    if instances:
        tooth_color = CLASS_COLORS[1]
        for inst in instances:
            overlay[inst.mask] = tooth_color
    else:
        overlay[mask == 1] = CLASS_COLORS[1]

    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    if measurements:
        for m in measurements:
            y_min, x_min, y_max, x_max = m.bbox
            x_mid = (x_min + x_max) // 2
            # CEJ line
            cv2.line(result, (x_min, m.cej_y), (x_max, m.cej_y), (255, 255, 0), 1)
            # bone crest line
            cv2.line(result, (x_min, m.bone_crest_y), (x_max, m.bone_crest_y), (0, 255, 255), 1)
            # bone loss ratio text
            cv2.putText(
                result,
                f"{m.bone_loss_ratio:.0%}",
                (x_mid - 10, m.cej_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return result


def draw_diagnosis_card(
    image: np.ndarray,
    diagnosis,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Append a text summary below the image."""
    h, w = image.shape[:2]
    card_h = 80
    card = np.zeros((card_h, w, 3), dtype=np.uint8)
    lines = [
        f"Condition: {diagnosis.overall_condition}  |  Spread: {diagnosis.spread or '-'}  |  Severity: {diagnosis.overall_severity or '-'}",
        f"Affected teeth: {diagnosis.num_affected}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(card, line, (10, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (200, 230, 200), 1, cv2.LINE_AA)
    return np.vstack([image, card])
