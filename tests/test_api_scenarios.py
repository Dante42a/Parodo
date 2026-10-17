"""Приёмочные сценарии API (задача Vikunja): успешный анализ, неверный формат, превышение размера,
отсутствие весов, недоступность модели.

Все проверки идут через HTTP — так же, как к API обращается фронтенд. Каждый тест проверяет
и код ответа, и причину в тексте: результат однозначен, а не «что-то не упало».
"""

import base64

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from starlette.datastructures import UploadFile

import src.api.main as api
import src.api.service as service_module
from src.api.service import AnalysisService, ModelState
from src.models.model_factory import create_model
from tests.conftest import TINY_CONFIG
from tests.test_input_validation import png_bytes


def analyze(client, payload: bytes, filename="opg.png", content_type="image/png"):
    return client.post("/v1/analyze", files={"file": (filename, payload, content_type)})


def noise_png(size: int) -> bytes:
    """PNG из шума — почти не сжимается, поэтому размер файла предсказуемо большой."""
    rng = np.random.default_rng(0)
    ok, encoded = cv2.imencode(".png", rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.fixture(scope="module")
def ready(tmp_path_factory):
    """API с загруженной маленькой моделью — как на сервере: старт → фоновая загрузка → ready."""
    tmp = tmp_path_factory.mktemp("model")
    config = tmp / "tiny.yaml"
    config.write_text(TINY_CONFIG, encoding="utf-8")
    weights = tmp / "semantic.ckpt"
    cfg = OmegaConf.merge(OmegaConf.load("configs/base.yaml"), OmegaConf.load(config))
    torch.save({"state_dict": create_model(cfg).state_dict()}, weights)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PARODO_MODEL_PATH", str(weights))
        mp.setenv("PARODO_CONFIG", str(config))
        mp.delenv("PARODO_YOLO_PATH", raising=False)
        service = AnalysisService()
        mp.setattr(api, "service", service)
        with TestClient(api.app) as client:
            service.loader.join(timeout=120)
            assert service.state == ModelState.READY, service.error
            yield client, service


@pytest.fixture
def unloaded(monkeypatch):
    """API с сервисом, загруженным под заданное окружение (для сценариев без готовой модели)."""

    def _make(**env):
        monkeypatch.delenv("PARODO_MODEL_PATH", raising=False)
        monkeypatch.delenv("PARODO_YOLO_PATH", raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, str(value))
        service = AnalysisService()
        monkeypatch.setattr(api, "service", service)
        return TestClient(api.app), service

    return _make


# ── 1. Успешный анализ ───────────────────────────────────────────────────────


def test_successful_analysis_returns_full_result(ready):
    client, _ = ready
    readyz = client.get("/readyz")
    assert readyz.status_code == 200

    response = analyze(client, png_bytes(256))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] in {"ok", "needs_review"}
    assert body["overall_condition"]
    assert isinstance(body["teeth"], list)
    assert body["model_version"] == readyz.json()["model_version"]
    overlay = cv2.imdecode(np.frombuffer(base64.b64decode(body["overlay_png_base64"]), np.uint8), cv2.IMREAD_COLOR)
    assert overlay is not None
    assert body["disclaimer"]


# ── 2. Неверный формат ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename, payload, content_type, reason",
    [
        ("notes.txt", b"plain text", "text/plain", "Поддерживаются PNG, JPEG и WebP"),
        ("scan.pdf", png_bytes(), "application/pdf", "Поддерживаются PNG, JPEG и WebP"),
        ("broken.png", b"definitely not an image", "image/png", "не удалось декодировать"),
        ("empty.png", b"", "image/png", "Файл пуст"),
        ("tiny.png", png_bytes(32), "image/png", "64×64"),
    ],
    ids=["txt", "pdf", "not-an-image", "empty", "too-small"],
)
def test_wrong_format_is_rejected_with_reason(ready, filename, payload, content_type, reason):
    client, _ = ready  # модель готова — значит, отказ именно из-за файла
    response = analyze(client, payload, filename, content_type)
    assert response.status_code == 422
    assert reason in response.json()["detail"]


# ── 3. Превышение размера ────────────────────────────────────────────────────


def test_oversized_file_is_rejected_without_reading_it_whole(ready, monkeypatch):
    client, _ = ready
    limit = 10_000
    monkeypatch.setattr(service_module, "MAX_UPLOAD_BYTES", limit)
    requested = []
    real_read = UploadFile.read

    async def spy_read(self, size=-1):
        requested.append(size)
        return await real_read(self, size)

    monkeypatch.setattr(UploadFile, "read", spy_read)
    payload = noise_png(256)
    assert len(payload) > limit

    response = analyze(client, payload)
    assert response.status_code == 422
    assert "превышает лимит" in response.json()["detail"]
    assert requested == [limit + 1]  # прочитано не больше лимита + 1 байт, а не весь файл


def test_file_exactly_at_limit_is_accepted(ready, monkeypatch):
    client, _ = ready
    payload = png_bytes(256)
    monkeypatch.setattr(service_module, "MAX_UPLOAD_BYTES", len(payload))
    assert analyze(client, payload).status_code == 200


# ── 4. Отсутствие весов ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "case, expected_state",
    [("path-not-set", ModelState.NOT_CONFIGURED), ("file-missing", ModelState.LOAD_FAILED)],
)
def test_missing_weights_gives_503_with_reason(unloaded, tmp_path, case, expected_state):
    env = {} if case == "path-not-set" else {"PARODO_MODEL_PATH": tmp_path / "semantic.ckpt"}
    client, service = unloaded(**env)
    service.load_safely()

    readyz = client.get("/readyz")
    assert readyz.status_code == 503
    assert readyz.json()["status"] == expected_state
    assert "PARODO_MODEL_PATH" in readyz.json()["error"]

    response = analyze(client, png_bytes())
    assert response.status_code == 503
    assert "PARODO_MODEL_PATH" in response.json()["detail"]


# ── 5. Недоступность модели ──────────────────────────────────────────────────


def test_model_still_loading_gives_503(unloaded):
    client, service = unloaded()
    service.state = ModelState.LOADING  # загрузка идёт — запрос пришёл слишком рано

    readyz = client.get("/readyz")
    assert readyz.status_code == 503
    assert readyz.json()["status"] == ModelState.LOADING
    response = analyze(client, png_bytes())
    assert response.status_code == 503
    assert "loading" in response.json()["detail"]


def test_model_failed_to_load_gives_503_with_reason(unloaded, tmp_path, tiny_config):
    broken = tmp_path / "semantic.ckpt"
    broken.write_bytes(b"definitely not a checkpoint")
    client, service = unloaded(PARODO_MODEL_PATH=broken, PARODO_CONFIG=tiny_config)
    service.load_safely()

    assert client.get("/readyz").json()["status"] == ModelState.LOAD_FAILED
    response = analyze(client, png_bytes())
    assert response.status_code == 503
    assert "повреждён" in response.json()["detail"]


def test_model_busy_gives_503(ready, monkeypatch):
    client, service = ready
    monkeypatch.setattr(service_module, "ANALYZE_WAIT_S", 0.05)
    service._infer_lock.acquire()  # идёт другой анализ
    try:
        response = analyze(client, png_bytes())
    finally:
        service._infer_lock.release()
    assert response.status_code == 503
    assert "занят" in response.json()["detail"]
