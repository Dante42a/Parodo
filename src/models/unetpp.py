import segmentation_models_pytorch as smp
from omegaconf import DictConfig
import torch.nn as nn


def build_unetpp(cfg: DictConfig) -> nn.Module:
    m = cfg.model
    return smp.UnetPlusPlus(
        encoder_name=m.encoder_name,
        encoder_weights=m.encoder_weights,
        in_channels=m.in_channels,
        classes=m.classes,
        activation=None,
    )
