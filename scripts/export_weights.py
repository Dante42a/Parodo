"""
Export only the model weights from a Lightning checkpoint — for the API and the server.

Lightning-чекпойнт хранит, кроме весов, состояние оптимизатора и планировщика (нужно, чтобы продолжить
обучение). Для анализа снимков оно лишнее, но при загрузке читается в память целиком: 549 МБ вместо ~180 МБ.
Скрипт оставляет только веса (значения не меняются), проверяет их строгой загрузкой в модель из конфига
и печатает sha256 — его потом видно в /readyz как model_info.semantic_sha256.

Usage:
    python scripts/export_weights.py --config configs/deeplabv3plus.yaml \
        --checkpoint "checkpoints/deeplabv3plus-epoch=51-val_mIoU=0.6886.ckpt" \
        --out checkpoints/semantic.ckpt
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.models.model_factory import create_model


def extract_state_dict(checkpoint: dict) -> dict:
    """Веса модели без префикса Lightning `model.` и без весов функции потерь (`criterion.*`)."""
    raw = checkpoint.get("state_dict", checkpoint)
    return {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in raw.items()
        if not k.startswith("criterion")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="checkpoints/semantic.ckpt")
    args = parser.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load("configs/base.yaml"), OmegaConf.load(args.config))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)

    # Проверка: веса строго подходят к архитектуре из конфига (каждый слой, без лишних и недостающих)
    model = create_model(OmegaConf.merge(cfg, {"model": {"encoder_weights": None}}))
    model.load_state_dict(state_dict, strict=True)

    meta = {
        "source": Path(args.checkpoint).name,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "arch": f"{cfg.model.name}/{cfg.model.get('encoder_name', '-')}",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "meta": meta}, out)

    # Результат читается и в безопасном режиме weights_only (только тензоры и простые типы, без кода)
    torch.load(out, map_location="cpu", weights_only=True)

    sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
    src_mb = Path(args.checkpoint).stat().st_size / 2**20
    out_mb = out.stat().st_size / 2**20
    print(f"{meta['arch']}, epoch {meta['epoch']}: {src_mb:.0f} МБ → {out_mb:.0f} МБ")
    print(f"{out}  sha256 {sha256}")


if __name__ == "__main__":
    main()
