"""Критерий приёмки задачи: /readyz сообщает готовность, валидный снимок возвращает результат анализа.

test_ready_then_valid_image_gives_result — полный путь на маленькой модели (в CI, без настоящих весов):
проверяет, что цепочка «старт → загрузка → /readyz → /v1/analyze → ответ» работает, а не точность модели.

test_real_weights_on_real_image — то же на боевых весах и реальном снимке. В CI весов нет, поэтому
по умолчанию пропускается. Запуск (semantic.ckpt — из scripts/export_weights.py):
    PARODO_E2E_MODEL_PATH=checkpoints/semantic.ckpt \
    PARODO_E2E_YOLO_PATH=checkpoints/yolov8/teeth_akudental/weights/best.pt pytest -s tests/test_acceptance.py
"""

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.api.service import AnalysisService, ModelState
from tests.test_input_validation import png_bytes

SAMPLE_IMAGE = Path("data/processed/images/train_0.png")


def start_service(monkeypatch, model_path, config=None, yolo_path=None) -> tuple[TestClient, AnalysisService]:
    """Запустить приложение как на сервере: lifespan стартует загрузку, ждём её окончания."""
    monkeypatch.setenv("PARODO_MODEL_PATH", str(model_path))
    if config:
        monkeypatch.setenv("PARODO_CONFIG", str(config))
    if yolo_path:
        monkeypatch.setenv("PARODO_YOLO_PATH", str(yolo_path))
    else:
        monkeypatch.delenv("PARODO_YOLO_PATH", raising=False)
    service = AnalysisService()
    monkeypatch.setattr(api, "service", service)
    client = TestClient(api.app)
    client.__enter__()
    service.loader.join(timeout=300)
    return client, service


def analyze(client, payload: bytes):
    return client.post("/v1/analyze", files={"file": ("opg.png", payload, "image/png")})


def decode_overlay(body) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(body["overlay_png_base64"]), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def test_ready_then_valid_image_gives_result(monkeypatch, tiny_weights, tiny_config):
    client, _ = start_service(monkeypatch, tiny_weights, config=tiny_config)
    try:
        ready = client.get("/readyz")
        assert ready.status_code == 200, ready.json()
        assert ready.json()["status"] == ModelState.READY

        response = analyze(client, png_bytes(256))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] in {"ok", "needs_review"}
        assert body["overall_condition"]
        assert isinstance(body["teeth"], list)
        assert body["model_version"] == ready.json()["model_version"]
        assert any("PARODO_YOLO_PATH" in r for r in body["reasons"])  # работа без YOLO видна в результате
        overlay = decode_overlay(body)
        assert overlay is not None and overlay.shape == (512, 1024, 3)  # размер из configs/base.yaml
    finally:
        client.__exit__(None, None, None)


@pytest.mark.skipif(
    not os.getenv("PARODO_E2E_MODEL_PATH") or not SAMPLE_IMAGE.is_file(),
    reason="нужны боевые веса (PARODO_E2E_MODEL_PATH) и снимок data/processed/images/train_0.png",
)
def test_real_weights_on_real_image(monkeypatch):
    client, service = start_service(
        monkeypatch, os.environ["PARODO_E2E_MODEL_PATH"], yolo_path=os.getenv("PARODO_E2E_YOLO_PATH")
    )
    try:
        assert client.get("/readyz").status_code == 200, service.error
        response = analyze(client, SAMPLE_IMAGE.read_bytes())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "ok"
        assert len(body["teeth"]) > 0  # на реальном снимке зубы находятся
        if os.getenv("PARODO_E2E_YOLO_PATH"):
            assert not any("PARODO_YOLO_PATH" in r for r in body["reasons"])
        print(f"\n{body['model_version']}: {body['overall_condition']} / {body['overall_severity']}, "
              f"зубов {len(body['teeth'])}, затронуто {body['num_affected']}, {body['processing_ms']:.0f} мс")
    finally:
        client.__exit__(None, None, None)
