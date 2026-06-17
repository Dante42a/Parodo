"""
Stage 2 – geometric measurement of bone loss per tooth.

Algorithm per tooth instance:
  1. Extract bounding box from tooth mask instance.
  2. Estimate CEJ (cemento-enamel junction) at `cej_fraction` from crown top.
     Anatomical approximation: upper 40 % = crown, lower 60 % = root.
  3. Find root apex = lowest y-pixel of the tooth mask.
  4. Find bone crest = topmost y-pixel of mask_bone within tooth's x-range.
  5. bone_loss_ratio = (bone_crest_y - cej_y) / root_length  clipped to [0, 1]
     (positive when bone has receded below CEJ)
  6. Cortical plate coverage = fraction of tooth x-range where mask_cortical
     is present near the bone crest level (within ±CORTICAL_BAND_PX pixels).

Limitation documented in the report:
  CEJ is not directly visible on OPG; the 40/60 split is an approximation
  that may introduce systematic error, particularly for teeth with unusual
  crown-to-root ratios (wisdom teeth, short roots).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import ndimage


CORTICAL_BAND_PX = 5   # pixels above/below bone crest to search for cortical plate


@dataclass
class ToothMeasurement:
    tooth_id: int
    bbox: tuple                 # (y_min, x_min, y_max, x_max) in pixels
    root_length_px: int
    bone_loss_height_px: int
    bone_loss_ratio: float      # [0, 1]
    cortical_coverage: float    # [0, 1]  fraction of tooth width with cortical plate
    cej_y: int
    root_tip_y: int
    bone_crest_y: int


def _extract_tooth_instances(mask_tooth: np.ndarray):
    """Label connected components in tooth mask → list of per-tooth binary masks."""
    labeled, n = ndimage.label(mask_tooth)
    instances = []
    for i in range(1, n + 1):
        instances.append((i, labeled == i))
    return instances


def measure_bone_loss(
    mask_tooth: np.ndarray,
    mask_bone: np.ndarray,
    mask_cortical: np.ndarray,
    cej_fraction: float = 0.40,
) -> List[ToothMeasurement]:
    """
    mask_tooth, mask_bone, mask_cortical: (H, W) bool arrays.
    Returns a list of ToothMeasurement, one per detected tooth instance.
    """
    instances = _extract_tooth_instances(mask_tooth)
    measurements = []

    for tooth_id, tooth_mask in instances:
        rows, cols = np.where(tooth_mask)
        if len(rows) < 10:
            continue

        y_min, y_max = int(rows.min()), int(rows.max())
        x_min, x_max = int(cols.min()), int(cols.max())
        tooth_height = y_max - y_min

        # CEJ: anatomical approximation
        cej_y = y_min + int(cej_fraction * tooth_height)

        # Root apex: lowest y of this tooth mask
        root_tip_y = y_max

        root_length = max(root_tip_y - cej_y, 1)

        # Bone crest: topmost y in mask_bone within this tooth's x-range
        bone_col_slice = mask_bone[:, x_min:x_max + 1]
        bone_rows = np.where(bone_col_slice.any(axis=1))[0]
        if len(bone_rows) == 0:
            # No bone detected near this tooth
            bone_crest_y = root_tip_y
        else:
            # Topmost = smallest y (image y increases downward)
            bone_crest_y = int(bone_rows.min())

        bone_loss_height = max(bone_crest_y - cej_y, 0)
        bone_loss_ratio = min(bone_loss_height / root_length, 1.0)

        # Cortical plate coverage: scan each column in tooth x-range
        cortical_cols_present = 0
        tooth_width = max(x_max - x_min + 1, 1)
        for col in range(x_min, x_max + 1):
            y_band_min = max(bone_crest_y - CORTICAL_BAND_PX, 0)
            y_band_max = min(bone_crest_y + CORTICAL_BAND_PX + 1, mask_cortical.shape[0])
            if mask_cortical[y_band_min:y_band_max, col].any():
                cortical_cols_present += 1
        cortical_coverage = cortical_cols_present / tooth_width

        measurements.append(ToothMeasurement(
            tooth_id=tooth_id,
            bbox=(y_min, x_min, y_max, x_max),
            root_length_px=root_length,
            bone_loss_height_px=bone_loss_height,
            bone_loss_ratio=round(bone_loss_ratio, 3),
            cortical_coverage=round(cortical_coverage, 3),
            cej_y=cej_y,
            root_tip_y=root_tip_y,
            bone_crest_y=bone_crest_y,
        ))

    # Sort teeth left to right by x_min
    measurements.sort(key=lambda m: m.bbox[1])
    return measurements
