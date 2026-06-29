"""
Stage 2 — geometric measurement of bone loss per tooth instance.

Algorithm per tooth instance:
  1. Bounding box from instance mask.
  2. CEJ estimated at cej_fraction from top of tooth (anatomical approximation).
  3. Root apex = lowest y-pixel of instance mask.
  4. Bone crest = topmost y-pixel of mask_bone within tooth's x-range.
  5. bone_loss_ratio = (bone_crest_y - cej_y) / root_length  clipped [0, 1]
  6. Cortical coverage = fraction of tooth x-range where mask_cortical is
     present within ±CORTICAL_BAND_PX of bone crest.

Accepts ToothInstance objects from YOLOv8-seg (stage1_segment.run_tooth_instances).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from src.pipeline.stage1_segment import ToothInstance

CORTICAL_BAND_PX = 5


@dataclass
class ToothMeasurement:
    tooth_id: int
    bbox: tuple             # (y1, x1, y2, x2)
    root_length_px: int
    bone_loss_height_px: int
    bone_loss_ratio: float  # [0, 1]
    cortical_coverage: float
    cej_y: int
    root_tip_y: int
    bone_crest_y: int
    confidence: float       # from YOLO (1.0 for fallback)


def measure_bone_loss(
    instances: List[ToothInstance],
    mask_bone: np.ndarray,
    mask_cortical: np.ndarray,
    cej_fraction: float = 0.40,
) -> List[ToothMeasurement]:
    """
    instances: per-tooth instance masks from YOLO or fallback_tooth_instances().
    mask_bone, mask_cortical: (H, W) bool arrays from Swin-UNet.

    Jaw direction:
      Upper jaw (tooth center Y < H/2): crown at bottom, root tip at top (y_min).
        - CEJ at y2 - cej_fraction * height
        - bone crest = bottommost bone row in x-range (max y)
        - bone_loss = max(cej_y - bone_crest_y, 0) / root_length
      Lower jaw (tooth center Y >= H/2): crown at top, root tip at bottom (y_max).
        - CEJ at y1 + cej_fraction * height
        - bone crest = topmost bone row in x-range (min y)
        - bone_loss = max(bone_crest_y - cej_y, 0) / root_length
    """
    img_h = mask_bone.shape[0]
    measurements = []

    for inst in instances:
        y1, x1, y2, x2 = inst.bbox
        tooth_height = y2 - y1
        if tooth_height < 5:
            continue

        tooth_center_y = (y1 + y2) / 2
        upper_jaw = tooth_center_y < img_h / 2

        bone_col = mask_bone[:, x1:x2 + 1]
        bone_rows = np.where(bone_col.any(axis=1))[0]

        if upper_jaw:
            # Root goes upward: root_tip = y_min, crown = y_max
            root_tip_y = y1
            cej_y = y2 - int(cej_fraction * tooth_height)
            root_length = max(cej_y - root_tip_y, 1)
            # Bone crest is the lowest (max y) bone edge near this tooth
            bone_crest_y = int(bone_rows.max()) if len(bone_rows) else root_tip_y
            bone_loss_height = max(cej_y - bone_crest_y, 0)
        else:
            # Root goes downward: root_tip = y_max, crown = y_min
            root_tip_y = y2
            cej_y = y1 + int(cej_fraction * tooth_height)
            root_length = max(root_tip_y - cej_y, 1)
            # Bone crest is the highest (min y) bone edge near this tooth
            bone_crest_y = int(bone_rows.min()) if len(bone_rows) else root_tip_y
            bone_loss_height = max(bone_crest_y - cej_y, 0)

        bone_loss_ratio = min(bone_loss_height / root_length, 1.0)

        # Cortical coverage near bone crest
        tooth_width = max(x2 - x1 + 1, 1)
        cort_cols = 0
        for col in range(x1, x2 + 1):
            y_lo = max(bone_crest_y - CORTICAL_BAND_PX, 0)
            y_hi = min(bone_crest_y + CORTICAL_BAND_PX + 1, mask_cortical.shape[0])
            if mask_cortical[y_lo:y_hi, col].any():
                cort_cols += 1
        cortical_coverage = cort_cols / tooth_width

        measurements.append(ToothMeasurement(
            tooth_id=inst.instance_id,
            bbox=inst.bbox,
            root_length_px=root_length,
            bone_loss_height_px=bone_loss_height,
            bone_loss_ratio=round(bone_loss_ratio, 3),
            cortical_coverage=round(cortical_coverage, 3),
            cej_y=cej_y,
            root_tip_y=root_tip_y,
            bone_crest_y=bone_crest_y,
            confidence=inst.confidence,
        ))

    return measurements
