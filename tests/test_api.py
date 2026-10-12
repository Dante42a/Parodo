from fastapi.testclient import TestClient

from src.api.main import app
from tests.test_input_validation import png_bytes

client = TestClient(app)


def test_health_is_available_without_weights():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_invalid_upload_returns_422():
    response = client.post("/v1/analyze", files={"file": ("bad.txt", b"bad", "text/plain")})
    assert response.status_code == 422


def test_valid_image_without_weights_returns_503():
    response = client.post("/v1/analyze", files={"file": ("opg.png", png_bytes(), "image/png")})
    assert response.status_code == 503
    assert "PARODO_MODEL_PATH" in response.json()["detail"]
