import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Histogram, make_asgi_app

from src.api.schemas import AnalysisResponse, HealthResponse
from src.api.service import AnalysisService, InputError, decode_image

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("parodo.api")

REQUESTS = Counter("parodo_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("parodo_analysis_seconds", "Analysis latency")

app = FastAPI(
    title="Parodo Research API",
    version="0.1.0",
    description="Research-only periodontal image analysis. Not a medical device.",
)
service = AnalysisService()
app.mount("/metrics", make_asgi_app())


@app.get("/healthz", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=service.loaded, model_version=service.model_version)


@app.get("/readyz", response_model=HealthResponse)
def ready() -> HealthResponse:
    status = "ready" if service.loaded else "model_not_loaded"
    return HealthResponse(status=status, model_loaded=service.loaded, model_version=service.model_version)


@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    started = time.perf_counter()
    try:
        payload = await file.read()
        image = decode_image(payload, file.content_type)
        with LATENCY.time():
            response = service.analyze(image)
        REQUESTS.labels("analyze", "ok").inc()
        logger.info("analysis_complete request_id=%s ms=%.2f", response.request_id, response.processing_ms)
        return response
    except InputError as exc:
        REQUESTS.labels("analyze", "invalid_input").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        REQUESTS.labels("analyze", "unavailable").inc()
        logger.warning("analysis_unavailable reason=%s elapsed=%.3f", exc, time.perf_counter() - started)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()
