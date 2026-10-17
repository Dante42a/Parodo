from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.api.schemas import AnalysisResponse, ModelInfo, ToothResult

logger = logging.getLogger("parodo.api")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_UPLOAD_BYTES = int(os.getenv("PARODO_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
# Сколько запрос ждёт своей очереди на инференс, прежде чем получить 503 «занято»
ANALYZE_WAIT_S = float(os.getenv("PARODO_ANALYZE_WAIT_S", "120"))
DISCLAIMER = (
    "Исследовательский прототип. Не является медицинским изделием и не предназначен "
    "для постановки диагноза или принятия клинических решений. Требуется проверка врачом."
)


class InputError(ValueError):
    pass


def decode_image(payload: bytes, content_type: str | None) -> np.ndarray:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise InputError("Поддерживаются PNG, JPEG и WebP.")
    if not payload:
        raise InputError("Файл пуст.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InputError(f"Файл превышает лимит {MAX_UPLOAD_BYTES / 2**20:g} МБ.")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InputError("Файл не удалось декодировать как изображение.")
    h, w = image.shape[:2]
    if h < 64 or w < 64:
        raise InputError("Минимальный размер изображения — 64×64.")
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


class ServiceBusy(ModelUnavailable):
    """Очередь на инференс не подошла за ANALYZE_WAIT_S (→ HTTP 503, можно повторить)."""


class ModelLoadError(RuntimeError):
    """Загрузка не удалась; текст — понятная причина для /readyz."""


YOLO_NOT_CONFIGURED = (
    "PARODO_YOLO_PATH не задан: зубы выделяются запасным алгоритмом (связные компоненты), точность ниже"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_version(arch: str, semantic: str, yolo: str | None, config: str) -> str:
    """Версия = архитектура + короткие хэши весов и конфига: сменились веса или правила — сменилась версия."""
    version = f"{arch}@{semantic[:8]}"
    if yolo:
        version += f"+yolo@{yolo[:8]}"
    return f"{version}+cfg@{config[:8]}"


def missing_dependency(exc: ImportError) -> str:
    return f"в окружении нет ML-зависимости «{exc.name or exc}» — образ API собран без библиотек инференса"


class AnalysisService:
    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()
        # Инференс строго по одному: на 1 CPU параллельность не ускоряет, а память удваивает
        self._infer_lock = threading.Lock()
        self.loader: threading.Thread | None = None
        self.state = ModelState.NOT_STARTED
        self.error: str | None = None
        self.warnings: list[str] = []  # модель работает, но с оговорками (например, без YOLO)
        self.model_path = os.getenv("PARODO_MODEL_PATH", "")
        self.yolo_path = os.getenv("PARODO_YOLO_PATH", "")
        self.model_version = "unconfigured"
        self.model_info: ModelInfo | None = None  # полные хэши и пути — для сверки с файлами на сервере

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
        except ModelLoadError as exc:
            logger.error("model_load_failed reason=%s", exc, exc_info=exc.__cause__)
            self.state = ModelState.LOAD_FAILED
            self.error = str(exc)
        except Exception as exc:  # непредвиденная ошибка — отдаём как есть
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
            self.warnings = []
            self.model_version = "unconfigured"  # версия появляется только у успешно загруженной модели
            self.model_info = None

            # Этап 1: конфиг (архитектура модели и параметры пайплайна)
            config_path = os.getenv("PARODO_CONFIG", "configs/deeplabv3plus.yaml")
            try:
                cfg = OmegaConf.merge(OmegaConf.load("configs/base.yaml"), OmegaConf.load(config_path))
            except Exception as exc:
                raise ModelLoadError(f"конфиг PARODO_CONFIG={config_path} не прочитан: {exc}") from exc
            if "model" not in cfg:
                raise ModelLoadError(f"в конфиге {config_path} нет секции model — не задана архитектура")
            arch = f"{cfg.model.name}/{cfg.model.get('encoder_name', '-')}"
            # хэш итогового конфига: архитектура + пороги классификации (правила диагноза)
            config_sha256 = hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()

            # Этап 2: семантическая модель (кость, кортикальная пластинка, зубы)
            if not Path(self.model_path).is_file():
                raise ModelLoadError(f"PARODO_MODEL_PATH does not point to model weights: {self.model_path}")
            semantic_sha256 = sha256_file(self.model_path)
            try:
                from src.pipeline.full_pipeline import PeriodontalPipeline

                pipeline = PeriodontalPipeline(
                    self.model_path, cfg=cfg, yolo_path=None, device=os.getenv("PARODO_DEVICE", "cpu")
                )
            except ImportError as exc:
                raise ModelLoadError(f"семантическая модель: {missing_dependency(exc)}") from exc
            except Exception as exc:
                if "state_dict" in str(exc):
                    detail = str(exc).splitlines()[0]
                    raise ModelLoadError(
                        f"веса {self.model_path} не подходят к архитектуре {arch} из {config_path}: {detail}"
                    ) from exc
                raise ModelLoadError(
                    f"файл {self.model_path} повреждён или не является чекпойнтом PyTorch "
                    f"({type(exc).__name__}: {exc})"
                ) from exc

            # Этап 3: YOLO (отдельные зубы). Путь задан — файл обязан быть; не задан — работаем с оговоркой
            yolo_sha256 = None
            if self.yolo_path:
                try:
                    pipeline.load_yolo(self.yolo_path)
                    yolo_sha256 = sha256_file(self.yolo_path)
                except FileNotFoundError as exc:
                    raise ModelLoadError(f"PARODO_YOLO_PATH не указывает на файл: {self.yolo_path}") from exc
                except ImportError as exc:
                    raise ModelLoadError(f"YOLO: {missing_dependency(exc)}") from exc
                except Exception as exc:
                    raise ModelLoadError(
                        f"YOLO {self.yolo_path} не загрузился ({type(exc).__name__}: {exc})"
                    ) from exc
            else:
                self.warnings.append(YOLO_NOT_CONFIGURED)
                logger.warning("yolo_not_configured: %s", YOLO_NOT_CONFIGURED)

            self.model_info = ModelInfo(
                arch=arch,
                config=config_path,
                config_sha256=config_sha256,
                semantic_path=self.model_path,
                semantic_sha256=semantic_sha256,
                yolo_path=self.yolo_path or None,
                yolo_sha256=yolo_sha256,
            )
            self.model_version = build_model_version(
                arch.replace("/", "-"), semantic_sha256, yolo_sha256, config_sha256
            )
            self._pipeline = pipeline
            self.state = ModelState.READY
            logger.info("model_ready version=%s", self.model_version)

    def analyze(self, image: np.ndarray) -> AnalysisResponse:
        if self.state == ModelState.NOT_STARTED:
            # загрузку при старте не запускали (например, в тестах без lifespan) — грузим сейчас
            self.load_safely()
        if self.state != ModelState.READY:
            raise ModelUnavailable(self.error or f"model is {self.state}")
        if not self._infer_lock.acquire(timeout=ANALYZE_WAIT_S):
            raise ServiceBusy(f"сервис занят другим анализом дольше {ANALYZE_WAIT_S:.0f} с — повторите позже")
        try:
            started = time.perf_counter()
            result = self._pipeline.run(image)
        finally:
            self._infer_lock.release()
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
            reasons=[*(diagnosis.reasons or []), *self.warnings],
            rule_version=diagnosis.rule_version,
            model_version=self.model_version,
            processing_ms=round(elapsed, 2),
            teeth=[ToothResult(**vars(tooth)) for tooth in diagnosis.per_tooth],
            overlay_png_base64=base64.b64encode(overlay.tobytes()).decode("ascii"),
            disclaimer=DISCLAIMER,
        )
