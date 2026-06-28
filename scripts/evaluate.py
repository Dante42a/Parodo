"""
Evaluate a trained model on the test set and dump metrics to CSV.

Usage:
    python scripts/evaluate.py --config configs/unet.yaml
                               --checkpoint checkpoints/unet-epoch=42-val_mIoU=0.7123.ckpt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import mlflow
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PeriodontalDataset
from src.data.transforms import build_val_transforms
from src.models.model_factory import create_model
from src.metrics.segmentation import compute_iou_per_class, compute_dice_per_class


def evaluate(cfg, checkpoint_path: str):
    model = create_model(cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = state.get("state_dict", state)
    # Lightning wraps model as self.model → strip "model." prefix if present
    if all(k.startswith("model.") for k in raw if not k.startswith("criterion")):
        raw = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    model.load_state_dict(raw)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ds = PeriodontalDataset(
        root=cfg.data.root,
        split_file=f"{cfg.data.split_dir}/test.txt",
        transform=build_val_transforms(cfg),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers)

    all_iou = []
    all_dice = []
    inference_times = []

    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            t0 = time.perf_counter()
            logits = model(images)
            torch.cuda.synchronize() if device == "cuda" else None
            inference_times.append((time.perf_counter() - t0) * 1000)

            preds = logits.argmax(dim=1).cpu()
            iou = compute_iou_per_class(preds, masks, cfg.data.num_classes)
            dice = compute_dice_per_class(preds, masks, cfg.data.num_classes)
            all_iou.append(iou)
            all_dice.append(dice)

    mean_iou = np.nanmean(all_iou, axis=0)
    mean_dice = np.nanmean(all_dice, axis=0)
    fps = 1000.0 / np.mean(inference_times)

    results = {
        "model": cfg.model.name,
        "mIoU": float(np.nanmean(mean_iou)),
        "FPS": round(fps, 1),
    }
    for i, name in enumerate(cfg.data.class_names):
        results[f"IoU_{name}"] = float(mean_iou[i])
        results[f"Dice_{name}"] = float(mean_dice[i])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="reports/results/metrics_summary.csv")
    args = parser.parse_args()

    base_cfg = OmegaConf.load("configs/base.yaml")
    model_cfg = OmegaConf.load(args.config)
    defaults = OmegaConf.to_container(model_cfg, resolve=False).get("defaults", [])
    if defaults:
        parent_name = defaults[0].lstrip("- ").strip()
        parent_cfg = OmegaConf.load(f"configs/{parent_name}.yaml")
        cfg = OmegaConf.merge(base_cfg, parent_cfg, model_cfg)
    else:
        cfg = OmegaConf.merge(base_cfg, model_cfg)

    results = evaluate(cfg, args.checkpoint)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        df = pd.read_csv(out_path)
        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    else:
        df = pd.DataFrame([results])

    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()
