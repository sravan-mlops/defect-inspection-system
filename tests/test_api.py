"""
test_api.py

Tests for the FastAPI serving layer. Uses FastAPI's TestClient, which
runs the app in-process - no need for a live uvicorn server. This is
what runs automatically in CI on every push.

IMPORTANT: uses a synthetically generated image, not a file from data/,
because data/ is gitignored and won't exist in a fresh CI checkout.
Tests must never depend on local-only files.
"""

import io

from fastapi.testclient import TestClient
from PIL import Image

from src.serving.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_returns_valid_structure():
    # Synthetic image - we're testing response STRUCTURE and status code,
    # not detection accuracy, so a real dataset image isn't necessary.
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("synthetic.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "defect_found" in data
    assert isinstance(data["detections"], list)