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
