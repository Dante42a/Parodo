from __future__ import annotations

import base64
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.api.schemas import AnalysisResponse, ToothResult

logger = logging.getLogger("parodo.api")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_UPLOAD_BYTES = int(os.getenv("PARODO_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
DISCLAIMER = (
    "Исследовательский прототип. Не является медицинским изделием и не предназначен "
    "для постановки диагноза или принятия клинических решений. Требуется проверка врачом."
)


class InputError(ValueError):
    pass


def decode_image(payload: bytes, content_type: str | None) -> np.ndarray:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise InputError("Поддерживаются PNG, JPEG и WebP")
    if not payload:
        raise InputError("Файл пуст")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InputError(f"Файл превышает лимит {MAX_UPLOAD_BYTES} байт")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InputError("Файл не удалось декодировать как изображение")
    h, w = image.shape[:2]
    if h < 64 or w < 64:
        raise InputError("Минимальный размер изображения — 64×64")
    return image


class ModelState:
    """Состояние модели — то, что показывает /readyz."""

    NOT_STARTED = "not_started"        # загрузку ещё не запускали
    LOADING = "loading"                # веса читаются в фоне
    READY = "ready"                    # можно анализировать
    NOT_CONFIGURED = "not_configured"  # PARODO_MODEL_PATH не задан
    LOAD_FAILED = "load_failed"        # загрузка упала, причина — в error


class ModelUnavailable(RuntimeError):
    """Модель не готова к анализу (→ HTTP 503)."""


class AnalysisService:
    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()
        self.loader: threading.Thread | None = None
        self.state = ModelState.NOT_STARTED
        self.error: str | None = None
        self.model_path = os.getenv("PARODO_MODEL_PATH", "")
        self.yolo_path = os.getenv("PARODO_YOLO_PATH", "")
        self.model_version = os.getenv("PARODO_MODEL_VERSION", "unconfigured")

    @property
    def loaded(self) -> bool:
        return self.state == ModelState.READY

    def start_loading(self) -> None:
        """Запустить загрузку в фоне, чтобы сервис отвечал на /healthz сразу после старта."""
        self.loader = threading.Thread(target=self.load_safely, name="model-loader", daemon=True)
        self.loader.start()

    def load_safely(self) -> None:
        """Загрузить модель; любую ошибку сохранить в состоянии вместо падения процесса."""
        try:
            self.load()
        except ModelUnavailable:
            pass  # not_configured — состояние и причину уже выставил load()
        except Exception as exc:  # причину отдаём в /readyz
            logger.exception("model_load_failed")
            self.state = ModelState.LOAD_FAILED
            self.error = f"{type(exc).__name__}: {exc}"

    def load(self) -> None:
        with self._lock:
            if self.state == ModelState.READY:
                return
            if not self.model_path:
                self.state = ModelState.NOT_CONFIGURED
                self.error = "PARODO_MODEL_PATH does not point to model weights"
                raise ModelUnavailable(self.error)
            self.state = ModelState.LOADING
            self.error = None
            if not Path(self.model_path).is_file():
                raise FileNotFoundError(f"PARODO_MODEL_PATH does not point to model weights: {self.model_path}")
            from src.pipeline.full_pipeline import PeriodontalPipeline

            cfg = OmegaConf.load(os.getenv("PARODO_CONFIG", "configs/deeplabv3plus.yaml"))
            base = OmegaConf.load("configs/base.yaml")
            self._pipeline = PeriodontalPipeline(
                self.model_path,
                cfg=OmegaConf.merge(base, cfg),
                yolo_path=self.yolo_path or None,
                device=os.getenv("PARODO_DEVICE", "cpu"),
            )
            self.state = ModelState.READY
            logger.info("model_ready path=%s", self.model_path)

    def analyze(self, image: np.ndarray) -> AnalysisResponse:
        if self.state == ModelState.NOT_STARTED:
            # загрузку при старте не запускали (например, в тестах без lifespan) — грузим сейчас
            self.load_safely()
        if self.state != ModelState.READY:
            raise ModelUnavailable(self.error or f"model is {self.state}")
        started = time.perf_counter()
        result = self._pipeline.run(image)
        elapsed = (time.perf_counter() - started) * 1000
        ok, overlay = cv2.imencode(".png", result.overlay)
        if not ok:
            raise RuntimeError("Unable to encode overlay")
        diagnosis = result.diagnosis
        return AnalysisResponse(
            request_id=str(uuid.uuid4()),
            status=diagnosis.status,
            overall_condition=diagnosis.overall_condition,
            spread=diagnosis.spread,
            overall_severity=diagnosis.overall_severity,
            num_affected=diagnosis.num_affected,
            reasons=diagnosis.reasons or [],
            rule_version=diagnosis.rule_version,
            model_version=self.model_version,
            processing_ms=round(elapsed, 2),
            teeth=[ToothResult(**vars(tooth)) for tooth in diagnosis.per_tooth],
            overlay_png_base64=base64.b64encode(overlay.tobytes()).decode("ascii"),
            disclaimer=DISCLAIMER,
        )
