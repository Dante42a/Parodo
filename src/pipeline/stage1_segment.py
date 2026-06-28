"""
Stage 1: semantic segmentation (bone/cortical) + tooth instance detection.

Two separate models:
  - Swin-UNet (or any semantic model): produces mask_bone, mask_cortical
  - YOLOv8-seg: produces per-tooth instance masks

If no YOLO model is available, falls back to connected-components on mask_tooth.
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_BACKGROUND = 0
CLASS_TOOTH = 1
CLASS_BONE = 2
CLASS_CORTICAL = 3


@dataclass
class SegmentationOutput:
    mask_full: np.ndarray       # (H, W) int, values 0-3
    mask_tooth: np.ndarray      # (H, W) bool — semantic tooth pixels
    mask_bone: np.ndarray       # (H, W) bool
    mask_cortical: np.ndarray   # (H, W) bool
    probs: np.ndarray           # (4, H, W) float32


@dataclass
class ToothInstance:
    instance_id: int
    mask: np.ndarray            # (H, W) bool
    bbox: tuple                 # (y1, x1, y2, x2) pixels
    confidence: float


@torch.no_grad()
def run_segmentation(
    model: nn.Module,
    image_tensor: torch.Tensor,   # (1, 3, H, W)
    device: str = "cuda",
) -> SegmentationOutput:
    model.eval()
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)       # (1, 4, H, W)
    probs = F.softmax(logits, dim=1)[0]
    mask = probs.argmax(dim=0).cpu().numpy()
    probs_np = probs.cpu().numpy()
    return SegmentationOutput(
        mask_full=mask,
        mask_tooth=(mask == CLASS_TOOTH),
        mask_bone=(mask == CLASS_BONE),
        mask_cortical=(mask == CLASS_CORTICAL),
        probs=probs_np,
    )


def run_tooth_instances(
    yolo_model,
    image_np: np.ndarray,           # (H, W, 3) uint8 RGB at target resolution
    target_hw: tuple = (512, 1024),
    conf_thresh: float = 0.25,
) -> List[ToothInstance]:
    """Run YOLOv8-seg and return per-tooth instance masks at target_hw resolution."""
    H, W = target_hw
    results = yolo_model.predict(image_np, imgsz=W, conf=conf_thresh, verbose=False)
    result = results[0]

    instances: List[ToothInstance] = []
    if result.masks is None:
        return instances

    masks_raw = result.masks.data.cpu().numpy()   # (N, H_m, W_m)
    confs = result.boxes.conf.cpu().numpy()

    for i, (mask_raw, conf) in enumerate(zip(masks_raw, confs)):
        if mask_raw.shape != (H, W):
            mask_resized = cv2.resize(
                mask_raw.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask_raw
        bool_mask = mask_resized > 0.5

        rows, cols = np.where(bool_mask)
        if len(rows) == 0:
            continue
        instances.append(ToothInstance(
            instance_id=i + 1,
            mask=bool_mask,
            bbox=(int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
            confidence=float(conf),
        ))

    # Sort left-to-right, re-number
    instances.sort(key=lambda t: t.bbox[1])
    for idx, inst in enumerate(instances):
        inst.instance_id = idx + 1

    return instances


def fallback_tooth_instances(mask_tooth: np.ndarray) -> List[ToothInstance]:
    """Connected-components fallback when no YOLO model is available."""
    from scipy import ndimage
    labeled, n = ndimage.label(mask_tooth)
    instances = []
    for i in range(1, n + 1):
        m = labeled == i
        rows, cols = np.where(m)
        if len(rows) < 10:
            continue
        instances.append(ToothInstance(
            instance_id=i,
            mask=m,
            bbox=(int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
            confidence=1.0,
        ))
    instances.sort(key=lambda t: t.bbox[1])
    for idx, inst in enumerate(instances):
        inst.instance_id = idx + 1
    return instances
