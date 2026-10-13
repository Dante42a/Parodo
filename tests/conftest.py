"""Общие фикстуры тестов."""

import pytest

# Маленькая DeepLabV3+ (resnet18, без предобученных весов) — собирается за секунды и без скачиваний.
TINY_CONFIG = """
model:
  name: deeplabv3plus
  encoder_name: resnet18
  encoder_weights: null
  in_channels: 3
  classes: 4
"""


@pytest.fixture
def tiny_config(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(TINY_CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def tiny_weights(tmp_path, tiny_config):
    import torch
    from omegaconf import OmegaConf

    from src.models.model_factory import create_model

    cfg = OmegaConf.merge(OmegaConf.load("configs/base.yaml"), OmegaConf.load(tiny_config))
    path = tmp_path / "semantic.ckpt"
    torch.save({"state_dict": create_model(cfg).state_dict()}, path)
    return path
