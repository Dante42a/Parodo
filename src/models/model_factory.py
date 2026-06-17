"""
Central dispatcher: create_model(cfg) returns the correct nn.Module.
"""

import torch.nn as nn
from omegaconf import DictConfig


def create_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name.lower()

    if name == "unet":
        from src.models.unet import build_unet
        return build_unet(cfg)

    if name == "unetpp":
        from src.models.unetpp import build_unetpp
        return build_unetpp(cfg)

    if name == "swin_unet":
        from src.models.swin_unet import build_swin_unet
        return build_swin_unet(cfg)

    if name == "segformer":
        from src.models.segformer import build_segformer
        return build_segformer(cfg)

    if name == "deeplabv3plus":
        from src.models.deeplabv3plus import build_deeplabv3plus
        return build_deeplabv3plus(cfg)

    if name == "yolov8_seg":
        # YOLOv8 is trained separately; return the evaluation wrapper
        from src.models.yolov8_seg import build_yolov8_seg
        ckpt = cfg.get("yolo_checkpoint", "checkpoints/yolov8/periodontal/weights/best.pt")
        return build_yolov8_seg(ckpt, cfg)

    raise ValueError(f"Unknown model name: '{name}'. "
                     f"Choose from: unet, unetpp, swin_unet, segformer, deeplabv3plus, yolov8_seg")
