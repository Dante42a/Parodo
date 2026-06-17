"""
Combined segmentation loss: weighted Dice + weighted CrossEntropy.
Optionally replace CE with Focal loss for heavy class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # gather target class probabilities
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - target_probs) ** self.gamma
        loss = -focal_weight * target_log_probs
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg: DictConfig, class_weights=None):
        super().__init__()
        self.dice_w = cfg.loss.dice_weight
        self.bce_w = cfg.loss.bce_weight
        self.dice = DiceLoss()

        weight = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(logits.device)
        return self.dice_w * self.dice(logits, targets) + self.bce_w * self.ce(logits, targets)


def build_loss(cfg: DictConfig) -> nn.Module:
    weights = list(cfg.data.class_weights) if cfg.data.get("class_weights") else None
    return CombinedLoss(cfg, class_weights=weights)
