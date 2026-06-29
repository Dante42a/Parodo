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
    image_np: np.ndarray,           # (H, W, 3) uint8 RGB
    target_hw: tuple = (512, 1024),
    conf_thresh: float = 0.03,
    iou_thresh: float = 0.30,
    max_det: int = 40,
    mask_tooth: Optional[np.ndarray] = None,  # Swin-UNet tooth mask для det-режима
) -> List[ToothInstance]:
    """
    Запускает YOLO и возвращает per-tooth инстансы.

    Два режима:
    - seg-модель (yolov8*-seg.pt): маски берутся из result.masks
    - det-модель (yolov8*.pt):     маски = bbox ∩ mask_tooth (Swin-UNet)
    """
    H, W = target_hw
    results = yolo_model.predict(
        image_np, imgsz=W,
        conf=conf_thresh, iou=iou_thresh, max_det=max_det,
        verbose=False,
    )
    result = results[0]

    instances: List[ToothInstance] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return instances

    confs = boxes.conf.cpu().numpy()
    # Координаты в пикселях исходного изображения (YOLO их масштабирует сам)
    xyxy = boxes.xyxy.cpu().numpy()   # (N, 4): x1 y1 x2 y2

    is_seg = result.masks is not None

    for i, (box, conf) in enumerate(zip(xyxy, confs)):
        bx1, by1, bx2, by2 = box
        # Масштабируем координаты к target_hw
        scale_x = W / result.orig_shape[1]
        scale_y = H / result.orig_shape[0]
        x1 = int(np.clip(bx1 * scale_x, 0, W - 1))
        y1 = int(np.clip(by1 * scale_y, 0, H - 1))
        x2 = int(np.clip(bx2 * scale_x, 0, W - 1))
        y2 = int(np.clip(by2 * scale_y, 0, H - 1))

        if is_seg:
            mask_raw = result.masks.data[i].cpu().numpy()
            if mask_raw.shape != (H, W):
                mask_raw = cv2.resize(
                    mask_raw.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
                )
            bool_mask = mask_raw > 0.5
        else:
            # Det-режим: маска = прямоугольник бокса ∩ Swin-UNet mask_tooth
            bool_mask = np.zeros((H, W), dtype=bool)
            bool_mask[y1:y2 + 1, x1:x2 + 1] = True
            if mask_tooth is not None:
                bool_mask &= mask_tooth

        rows, cols = np.where(bool_mask)
        if len(rows) < 10:
            continue

        instances.append(ToothInstance(
            instance_id=i + 1,
            mask=bool_mask,
            bbox=(int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
            confidence=float(conf),
        ))

    # Фильтр ложных позитивов
    filtered = []
    for inst in instances:
        y1, x1, y2, x2 = inst.bbox
        h_bbox = y2 - y1
        w_bbox = x2 - x1
        cx = (x1 + x2) / 2
        if cx < W * 0.04 or cx > W * 0.96:
            continue
        if h_bbox < 30 or h_bbox > H * 0.65:
            continue
        if w_bbox > h_bbox * 2.2:
            continue
        filtered.append(inst)

    filtered.sort(key=lambda t: t.bbox[1])
    for idx, inst in enumerate(filtered):
        inst.instance_id = idx + 1

    return filtered


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
