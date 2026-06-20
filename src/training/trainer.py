"""
PyTorch Lightning module for segmentation training.
Handles encoder freeze phase, differential LRs, AMP, and MLflow logging.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import mlflow
import numpy as np
from omegaconf import DictConfig

from src.losses.combined import build_loss
from src.metrics.segmentation import compute_iou_per_class, compute_dice_per_class, miou


class SegmentationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.criterion = build_loss(cfg)
        self.num_classes = cfg.data.num_classes
        self._freeze_epochs = cfg.training.freeze_encoder_epochs

    def on_train_epoch_start(self):
        # Unfreeze encoder after freeze_encoder_epochs
        if self.current_epoch == self._freeze_epochs:
            for param in self.model.parameters():
                param.requires_grad = True
            print(f"Epoch {self.current_epoch}: encoder unfrozen.")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        images, masks, _ = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = logits.argmax(dim=1)
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)
        iou = compute_iou_per_class(preds, masks, self.num_classes)
        dice = compute_dice_per_class(preds, masks, self.num_classes)
        mean_iou = float(np.nanmean(iou))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mIoU", mean_iou, prog_bar=True)

        names = self.cfg.data.class_names
        for i, name in enumerate(names):
            if not np.isnan(iou[i]):
                self.log(f"val_IoU_{name}", iou[i])
            if not np.isnan(dice[i]):
                self.log(f"val_Dice_{name}", dice[i])

        return {"val_loss": loss, "val_mIoU": mean_iou}

    def on_validation_epoch_end(self):
        # Log epoch metrics to MLflow
        metrics = self.trainer.callback_metrics
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                mlflow.log_metric(k.replace("/", "_"), float(v), step=self.current_epoch)

    def configure_optimizers(self):
        cfg = self.cfg.training

        # Try to get encoder/decoder parameter groups
        try:
            encoder_params = list(self.model.encoder.parameters())
            decoder_params = [p for p in self.model.parameters()
                              if not any(p is ep for ep in encoder_params)]
        except AttributeError:
            # Model doesn't expose .encoder — use single LR
            encoder_params = []
            decoder_params = list(self.model.parameters())

        # Freeze encoder during initial epochs
        for p in encoder_params:
            p.requires_grad = False

        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": cfg.encoder_lr})
        param_groups.append({"params": decoder_params, "lr": cfg.decoder_lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs - cfg.warmup_epochs, eta_min=1e-6
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
        )
        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, scheduler], milestones=[cfg.warmup_epochs]
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": combined, "interval": "epoch"}}
