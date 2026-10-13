"""Поведение /v1/analyze во время работы: сервис не зависает, анализы идут по одному, ошибки различимы."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api
import src.api.service as service_module
from src.api.service import AnalysisService, ModelState
from src.pipeline.stage2_classify import ImageDiagnosis
from tests.test_input_validation import png_bytes


class StubPipeline:
    """Заглушка вместо нейросети: может «думать» до сигнала или упасть."""

    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def run(self, image):
        self.started.set()
        if self.fail:
            raise self.fail
        self.release.wait(timeout=10)
        diagnosis = ImageDiagnosis(
            per_tooth=[], overall_condition="Normal", spread=None, overall_severity=None, num_affected=0
        )
        return SimpleNamespace(diagnosis=diagnosis, overlay=np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.fixture
def ready_with(monkeypatch):
    """Сервис в состоянии ready с заданной заглушкой пайплайна.

    Клиент открыт через `with`: все запросы идут через один цикл событий, как на реальном сервере —
    иначе блокировка цикла долгим анализом не воспроизводится.
    """
    clients = []

    def _make(pipeline: StubPipeline):
        service = AnalysisService()
        service._pipeline = pipeline
        service.state = ModelState.READY
        monkeypatch.setattr(api, "service", service)
        client = TestClient(api.app)
        client.__enter__()
        clients.append(client)
        return client, service

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


def post_image(client):
    return client.post("/v1/analyze", files={"file": ("opg.png", png_bytes(), "image/png")})


def test_healthz_answers_while_analysis_runs(ready_with):
    pipeline = StubPipeline()
    pipeline.release.clear()  # анализ «думает», пока не отпустим
    client, _ = ready_with(pipeline)
    result = {}
    worker = threading.Thread(target=lambda: result.update(response=post_image(client)))
    worker.start()
    assert pipeline.started.wait(timeout=5)

    began = time.perf_counter()
    assert client.get("/healthz").status_code == 200
    assert time.perf_counter() - began < 2  # раньше /healthz ждал бы конца анализа

    pipeline.release.set()
    worker.join(timeout=10)
    assert result["response"].status_code == 200


def test_busy_service_returns_503(ready_with, monkeypatch):
    monkeypatch.setattr(service_module, "ANALYZE_WAIT_S", 0.05)
    client, service = ready_with(StubPipeline())
    service._infer_lock.acquire()  # имитируем идущий анализ
    try:
        response = post_image(client)
    finally:
        service._infer_lock.release()
    assert response.status_code == 503
    assert "занят" in response.json()["detail"]


def test_analysis_failure_returns_500_with_error_code(ready_with):
    client, _ = ready_with(StubPipeline(fail=ValueError("internal details")))
    response = post_image(client)
    assert response.status_code == 500
    assert "код ошибки" in response.json()["detail"]
    assert "internal details" not in response.json()["detail"]  # внутренности — только в лог
