"""
Export the best PyTorch checkpoint to ONNX for use in the Gradio app.

Usage:
    python scripts/export_onnx.py --config configs/unet.yaml
                                  --checkpoint checkpoints/best.ckpt
                                  --out app/models/best_model.onnx
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.model_factory import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="app/models/best_model.onnx")
    args = parser.parse_args()

    base_cfg = OmegaConf.load("configs/base.yaml")
    model_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, model_cfg)

    model = create_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    h, w = cfg.data.image_size
    dummy = torch.randn(1, 3, h, w)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    print(f"Exported ONNX model to {out_path}")
    print(f"Input shape: (1, 3, {h}, {w})")
    print(f"Output shape: (1, {cfg.data.num_classes}, {h}, {w})")


if __name__ == "__main__":
    main()
