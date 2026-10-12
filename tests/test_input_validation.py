import cv2
import numpy as np
import pytest

from src.api.service import InputError, decode_image


def png_bytes(size=128):
    ok, encoded = cv2.imencode(".png", np.zeros((size, size, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_valid_png_decodes():
    assert decode_image(png_bytes(), "image/png").shape == (128, 128, 3)


@pytest.mark.parametrize(
    "payload,content_type,message",
    [
        (b"", "image/png", "пуст"),
        (b"not-image", "image/png", "декод"),
        (png_bytes(), "application/pdf", "PNG"),
    ],
)
def test_invalid_input_rejected(payload, content_type, message):
    with pytest.raises(InputError, match=message):
        decode_image(payload, content_type)


def test_tiny_image_rejected():
    with pytest.raises(InputError, match="64"):
        decode_image(png_bytes(32), "image/png")
