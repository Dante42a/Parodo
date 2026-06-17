"""
Stage 1: run a segmentation model on a preprocessed image tensor
and return per-class binary masks.
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SegmentationOutput:
    mask_full: np.ndarray      # (H, W) int, values 0-3
    mask_tooth: np.ndarray     # (H, W) bool
    mask_bone: np.ndarray      # (H, W) bool
    mask_cortical: np.ndarray  # (H, W) bool
    probs: np.ndarray          # (4, H, W) float32 — softmax probs per class


CLASS_BACKGROUND = 0
CLASS_TOOTH = 1
CLASS_BONE = 2
CLASS_CORTICAL = 3


@torch.no_grad()
def run_segmentation(
    model: nn.Module,
    image_tensor: torch.Tensor,   # (1, 3, H, W) float32
    device: str = "cuda",
) -> SegmentationOutput:
    model.eval()
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)  # (1, 4, H, W)
    probs = F.softmax(logits, dim=1)[0]  # (4, H, W)
    mask = probs.argmax(dim=0).cpu().numpy()
    probs_np = probs.cpu().numpy()

    return SegmentationOutput(
        mask_full=mask,
        mask_tooth=(mask == CLASS_TOOTH),
        mask_bone=(mask == CLASS_BONE),
        mask_cortical=(mask == CLASS_CORTICAL),
        probs=probs_np,
    )
