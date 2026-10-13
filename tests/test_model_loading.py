"""Загрузка модели и /readyz: сервис честно сообщает, готов ли он к анализу."""

import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.api.service import AnalysisService, ModelState


@pytest.fixture
def make_client(monkeypatch):
    """Клиент с новым сервисом, собранным под заданные переменные окружения."""

    def _make(**env):
        for name in ("PARODO_MODEL_PATH", "PARODO_YOLO_PATH"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, str(value))
        service = AnalysisService()
        monkeypatch.setattr(api, "service", service)
        return TestClient(api.app), service

    return _make


def test_readyz_without_model_path_is_not_configured(make_client):
    client, service = make_client()
    service.load_safely()
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == ModelState.NOT_CONFIGURED
    assert "PARODO_MODEL_PATH" in response.json()["error"]


def test_readyz_with_missing_file_reports_load_failed(make_client, tmp_path):
    missing = tmp_path / "semantic.ckpt"
    client, service = make_client(PARODO_MODEL_PATH=missing)
    service.load_safely()
    body = client.get("/readyz").json()
    assert body["status"] == ModelState.LOAD_FAILED
    assert str(missing) in body["error"]


def test_startup_triggers_loading(make_client):
    client, service = make_client()
    with client:  # контекстный менеджер запускает lifespan, как при реальном старте
        assert client.get("/healthz").status_code == 200
        service.loader.join(timeout=10)
    assert service.state == ModelState.NOT_CONFIGURED


def test_analyze_reports_reason_when_model_not_ready(make_client, tmp_path):
    from tests.test_input_validation import png_bytes

    client, service = make_client(PARODO_MODEL_PATH=tmp_path / "nope.ckpt")
    service.load_safely()
    response = client.post("/v1/analyze", files={"file": ("opg.png", png_bytes(), "image/png")})
    assert response.status_code == 503
    assert "nope.ckpt" in response.json()["detail"]


# ── Понятные причины ошибок загрузки ─────────────────────────────────────────
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


def loaded(make_client, **env):
    client, service = make_client(**env)
    service.load_safely()
    return client, service


def test_missing_config_is_explained(make_client, tiny_weights, tmp_path):
    _, service = loaded(make_client, PARODO_MODEL_PATH=tiny_weights, PARODO_CONFIG=tmp_path / "none.yaml")
    assert service.state == ModelState.LOAD_FAILED
    assert "PARODO_CONFIG" in service.error


def test_corrupt_weights_are_explained(make_client, tiny_config, tmp_path):
    broken = tmp_path / "semantic.ckpt"
    broken.write_bytes(b"definitely not a checkpoint")
    _, service = loaded(make_client, PARODO_MODEL_PATH=broken, PARODO_CONFIG=tiny_config)
    assert service.state == ModelState.LOAD_FAILED
    assert "повреждён" in service.error


def test_wrong_architecture_is_explained(make_client, tiny_config, tmp_path):
    import torch

    foreign = tmp_path / "semantic.ckpt"
    torch.save({"state_dict": {"some.other.layer": torch.zeros(1)}}, foreign)
    _, service = loaded(make_client, PARODO_MODEL_PATH=foreign, PARODO_CONFIG=tiny_config)
    assert service.state == ModelState.LOAD_FAILED
    assert "не подходят к архитектуре deeplabv3plus/resnet18" in service.error


def test_missing_ml_dependency_is_explained(make_client, tiny_weights, tiny_config, monkeypatch):
    import sys

    # имитируем образ без segmentation-models-pytorch
    monkeypatch.delitem(sys.modules, "src.models.deeplabv3plus", raising=False)
    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", None)
    _, service = loaded(make_client, PARODO_MODEL_PATH=tiny_weights, PARODO_CONFIG=tiny_config)
    assert service.state == ModelState.LOAD_FAILED
    assert "segmentation_models_pytorch" in service.error


def test_yolo_path_without_file_fails_instead_of_silent_fallback(make_client, tiny_weights, tiny_config, tmp_path):
    _, service = loaded(
        make_client,
        PARODO_MODEL_PATH=tiny_weights,
        PARODO_CONFIG=tiny_config,
        PARODO_YOLO_PATH=tmp_path / "yolo.pt",
    )
    assert service.state == ModelState.LOAD_FAILED
    assert "PARODO_YOLO_PATH" in service.error


def test_ready_without_yolo_warns(make_client, tiny_weights, tiny_config):
    client, service = loaded(make_client, PARODO_MODEL_PATH=tiny_weights, PARODO_CONFIG=tiny_config)
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == ModelState.READY
    assert any("PARODO_YOLO_PATH" in w for w in response.json()["warnings"])
