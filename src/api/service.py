from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.api.schemas import AnalysisResponse, ToothResult

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


class AnalysisService:
    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()
        self.model_path = os.getenv("PARODO_MODEL_PATH", "")
        self.yolo_path = os.getenv("PARODO_YOLO_PATH", "")
        self.model_version = os.getenv("PARODO_MODEL_VERSION", "unconfigured")

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            if not self.model_path or not Path(self.model_path).is_file():
                raise RuntimeError("PARODO_MODEL_PATH does not point to model weights")
            from src.pipeline.full_pipeline import PeriodontalPipeline

            cfg = OmegaConf.load(os.getenv("PARODO_CONFIG", "configs/deeplabv3plus.yaml"))
            base = OmegaConf.load("configs/base.yaml")
            self._pipeline = PeriodontalPipeline(
                self.model_path,
                cfg=OmegaConf.merge(base, cfg),
                yolo_path=self.yolo_path or None,
                device=os.getenv("PARODO_DEVICE", "cpu"),
            )

    def analyze(self, image: np.ndarray) -> AnalysisResponse:
        self.load()
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
