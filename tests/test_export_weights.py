"""scripts/export_weights.py: из Lightning-чекпойнта остаются только веса, и API их загружает."""

import subprocess
import sys

import torch
from omegaconf import OmegaConf

from src.api.service import AnalysisService, ModelState
from src.models.model_factory import create_model
from tests.test_model_loading import TINY_CONFIG


def test_export_keeps_only_weights_and_api_loads_them(tmp_path, monkeypatch):
    config = tmp_path / "tiny.yaml"
    config.write_text(TINY_CONFIG, encoding="utf-8")
    model = create_model(OmegaConf.merge(OmegaConf.load("configs/base.yaml"), OmegaConf.load(config)))

    # чекпойнт в формате Lightning: префикс model., веса функции потерь и состояние оптимизатора
    lightning = tmp_path / "deeplabv3plus-epoch=3.ckpt"
    torch.save(
        {
            "epoch": 3,
            "global_step": 42,
            "state_dict": {
                **{f"model.{k}": v for k, v in model.state_dict().items()},
                "criterion.ce.weight": torch.ones(4),
            },
            "optimizer_states": [{"state": {i: torch.zeros(1000) for i in range(50)}}],
        },
        lightning,
    )
    out = tmp_path / "semantic.ckpt"
    subprocess.run(
        [sys.executable, "scripts/export_weights.py", "--config", str(config), "--checkpoint", str(lightning),
         "--out", str(out)],
        check=True,
    )

    exported = torch.load(out, weights_only=True)
    assert set(exported) == {"state_dict", "meta"}
    assert exported["meta"]["epoch"] == 3
    assert not any(k.startswith(("model.", "criterion")) for k in exported["state_dict"])
    for name, tensor in model.state_dict().items():  # веса не изменились
        assert torch.equal(exported["state_dict"][name], tensor)
    assert out.stat().st_size < lightning.stat().st_size

    monkeypatch.setenv("PARODO_MODEL_PATH", str(out))
    monkeypatch.setenv("PARODO_CONFIG", str(config))
    monkeypatch.delenv("PARODO_YOLO_PATH", raising=False)
    service = AnalysisService()
    service.load_safely()
    assert service.state == ModelState.READY
