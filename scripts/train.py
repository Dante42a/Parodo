"""
Main training entry point.

Usage:
    python scripts/train.py --config configs/unet.yaml
    python scripts/train.py --config configs/segformer.yaml --fast_dev_run
    python scripts/train.py --config configs/unet.yaml --resume checkpoints/last.ckpt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from omegaconf import OmegaConf

from src.data.datamodule import PeriodontalDataModule
from src.models.model_factory import create_model
from src.training.trainer import SegmentationModule
from src.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    # Load and merge configs (supports single-level defaults inheritance)
    base_cfg = OmegaConf.load("configs/base.yaml")
    model_cfg = OmegaConf.load(args.config)
    defaults = OmegaConf.to_container(model_cfg, resolve=False).get("defaults", [])
    if defaults:
        parent_name = defaults[0].lstrip("- ").strip()
        parent_cfg = OmegaConf.load(f"configs/{parent_name}.yaml")
        cfg = OmegaConf.merge(base_cfg, parent_cfg, model_cfg)
    else:
        cfg = OmegaConf.merge(base_cfg, model_cfg)

    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.seed)

    model = create_model(cfg)
    module = SegmentationModule(model, cfg)
    dm = PeriodontalDataModule(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint.dirpath,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            filename=f"{cfg.model.name}-{{epoch}}-{{val_mIoU:.4f}}",
            save_last=True,
        ),
        EarlyStopping(monitor=cfg.checkpoint.monitor, patience=10, mode=cfg.checkpoint.mode),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if __import__("torch").cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if cfg.training.amp else 32,
        gradient_clip_val=cfg.training.grad_clip,
        accumulate_grad_batches=cfg.training.get("grad_accum_steps", 1),
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )

    mlflow.set_experiment(cfg.logging.mlflow_experiment)
    with mlflow.start_run(run_name=cfg.model.name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        trainer.fit(module, dm, ckpt_path=args.resume)

    print(f"\nTraining complete. Best checkpoint in {cfg.checkpoint.dirpath}/")


if __name__ == "__main__":
    main()
