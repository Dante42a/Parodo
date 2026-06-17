"""
YOLOv8-seg wrapper.

YOLOv8 is trained separately via the ultralytics CLI:
  yolo segment train data=data/yolo_data.yaml model=yolov8m-seg.pt epochs=60 imgsz=1024

This wrapper loads a trained checkpoint and exposes the same
forward(image) -> semantic_mask interface as other models
so evaluation uses the unified src/metrics/ pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


class YOLOv8SegWrapper(nn.Module):
    def __init__(self, checkpoint_path: str, num_classes: int, img_size: int = 1024):
        super().__init__()
        from ultralytics import YOLO
        self.yolo = YOLO(checkpoint_path)
        self.num_classes = num_classes
        self.img_size = img_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) float32 tensor
        returns: (B, num_classes, H, W) logits (one-hot style scores)
        """
        B, C, H, W = x.shape
        semantic = torch.zeros(B, self.num_classes, H, W, device=x.device)

        for i in range(B):
            # ultralytics expects numpy HWC uint8
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            results = self.yolo.predict(img_np, imgsz=self.img_size, verbose=False)
            result = results[0]

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()   # (N, H, W)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for mask, cls in zip(masks, classes):
                    if cls < self.num_classes:
                        semantic[i, cls] = torch.maximum(
                            semantic[i, cls],
                            torch.from_numpy(mask).float().to(x.device),
                        )

        # Background: where no class assigned
        semantic[:, 0] = 1.0 - semantic[:, 1:].max(dim=1).values
        return semantic

    def train_yolo(self, cfg: DictConfig):
        """Launch YOLO training via ultralytics API."""
        from ultralytics import YOLO
        model = YOLO(f"{cfg.model.variant}.pt")
        model.train(
            data=cfg.model.ultralytics_data_yaml,
            epochs=cfg.model.ultralytics_epochs,
            imgsz=cfg.model.ultralytics_imgsz,
            batch=cfg.model.ultralytics_batch,
            device=cfg.model.ultralytics_device,
            project="checkpoints/yolov8",
            name="periodontal",
        )


def build_yolov8_seg(checkpoint_path: str, cfg: DictConfig) -> nn.Module:
    return YOLOv8SegWrapper(
        checkpoint_path=checkpoint_path,
        num_classes=cfg.data.num_classes,
        img_size=cfg.data.image_size[1],
    )
