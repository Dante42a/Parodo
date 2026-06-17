"""
Segmentation metrics: mIoU and Dice per class.
"""

import torch
import numpy as np


def compute_iou_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """
    preds:   (B, H, W) int64 — argmax of logits
    targets: (B, H, W) int64
    returns: (num_classes,) float array of per-class IoU
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        tgt_cls = targets == cls
        if ignore_index is not None:
            valid = targets != ignore_index
            pred_cls = pred_cls & valid
            tgt_cls = tgt_cls & valid

        intersection = (pred_cls & tgt_cls).sum().item()
        union = (pred_cls | tgt_cls).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return np.array(ious)


def compute_dice_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1.0,
) -> np.ndarray:
    dices = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls = (targets == cls).float()
        intersection = (pred_cls * tgt_cls).sum().item()
        denom = pred_cls.sum().item() + tgt_cls.sum().item()
        if denom == 0:
            dices.append(float("nan"))
        else:
            dices.append((2.0 * intersection + smooth) / (denom + smooth))
    return np.array(dices)


def miou(preds, targets, num_classes):
    iou = compute_iou_per_class(preds, targets, num_classes)
    return float(np.nanmean(iou))
