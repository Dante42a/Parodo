import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from prometheus_client import Counter, Histogram, make_asgi_app

from src.api.schemas import AnalysisResponse, HealthResponse
from src.api.service import AnalysisService, InputError, ModelUnavailable, decode_image

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("parodo.api")

REQUESTS = Counter("parodo_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("parodo_analysis_seconds", "Analysis latency")

service = AnalysisService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Загрузка модели стартует сразу, а не при первом анализе — /readyz отражает реальное состояние
    service.start_loading()
    yield


app = FastAPI(
    title="Parodo Research API",
    version="0.1.0",
    description="Research-only periodontal image analysis. Not a medical device.",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.get("/healthz", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=service.loaded, model_version=service.model_version)


@app.get("/readyz", response_model=HealthResponse)
def ready(response: Response) -> HealthResponse:
    # 200 — модель готова, 503 — нет (status и error объясняют почему)
    if not service.loaded:
        response.status_code = 503
    return HealthResponse(
        status=service.state,
        model_loaded=service.loaded,
        model_version=service.model_version,
        error=service.error,
        warnings=service.warnings,
        model_info=service.model_info,
    )


@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    started = time.perf_counter()
    try:
        payload = await file.read()
        image = decode_image(payload, file.content_type)
        with LATENCY.time():
            # инференс — в отдельном потоке: пока он идёт, /healthz, /readyz и /metrics отвечают
            response = await run_in_threadpool(service.analyze, image)
        REQUESTS.labels("analyze", "ok").inc()
        logger.info("analysis_complete request_id=%s ms=%.2f", response.request_id, response.processing_ms)
        return response
    except InputError as exc:
        REQUESTS.labels("analyze", "invalid_input").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        # модель не готова или сервис занят — состояние временное, запрос можно повторить
        REQUESTS.labels("analyze", "unavailable").inc()
        logger.warning("analysis_unavailable reason=%s elapsed=%.3f", exc, time.perf_counter() - started)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # сбой на конкретном снимке: подробности — в лог по error_id, клиенту — короткое сообщение
        error_id = uuid.uuid4().hex[:12]
        REQUESTS.labels("analyze", "error").inc()
        logger.exception("analysis_failed error_id=%s", error_id)
        raise HTTPException(status_code=500, detail=f"анализ не выполнен, код ошибки {error_id}") from exc
    finally:
        await file.close()
