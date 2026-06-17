"""
SegFormer wrapper using HuggingFace transformers.

forward() returns logits at 1/4 input resolution;
the trainer upsamples to full resolution before computing loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from omegaconf import DictConfig


class SegFormerWrapper(nn.Module):
    def __init__(self, pretrained_model: str, num_labels: int):
        super().__init__()
        id2label = {i: f"class_{i}" for i in range(num_labels)}
        label2id = {v: k for k, v in id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        out = self.model(pixel_values=x)
        logits = out.logits  # (B, C, H/4, W/4)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return logits


def build_segformer(cfg: DictConfig) -> nn.Module:
    m = cfg.model
    return SegFormerWrapper(
        pretrained_model=m.pretrained_model,
        num_labels=m.num_labels,
    )
