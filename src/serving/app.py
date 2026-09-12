"""
app.py

Purpose: Serve the trained defect-detection model as a REST API.

Key design decisions (see docs/design_doc.md for the full reasoning):
- Model is loaded ONCE at startup, not per-request (latency budget: <200ms/image)
- Confidence threshold is set lower than YOLO's default (0.25 -> 0.15) because
  our design doc prioritizes recall (catching defects) over precision
  (avoiding false alarms) - a missed defect is costlier than a false alarm.
- Detections in a middle confidence band are flagged "needs_review" instead
  of a forced pass/fail, per the design doc's failure-handling policy.
- Every prediction is logged as structured JSON (not free-form text) so
  logs are queryable in a real log aggregator (CloudWatch, Datadog, etc.)
  - this is what lets you debug production issues after the fact and spot
  model behavior drifting over time.
- Prometheus metrics are exposed at /metrics for Grafana dashboarding -
  logs are for debugging one incident, metrics are for "how's the system
  doing right now" at a glance.

Run: uvicorn src.serving.app:app --reload --port 8000
Then open http://localhost:8000/docs for interactive API docs (auto-generated
by FastAPI - one of its biggest advantages over writing raw Flask/Django).
"""

import json
import logging
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
import numpy as np
import cv2

# Structured (JSON-per-line) logging setup. Each log call below produces
# one JSON object per line - this is deliberately NOT human-readable
# prose, because in production these logs get shipped to a tool that
# parses and queries them as data, not read directly in a terminal.
logger = logging.getLogger("defect_api")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def log_event(event: str, **fields):
    logger.info(json.dumps({"event": event, "timestamp": time.time(), **fields}))

MODEL_PATH = Path("models/best.pt")

# Confidence thresholds - see docstring above for reasoning.
CONF_THRESHOLD = 0.15   # detections below this are discarded entirely
REVIEW_THRESHOLD = 0.40  # detections between CONF_THRESHOLD and this are
                          # flagged "needs_review" rather than a firm defect call

app = FastAPI(title="Defect Inspection API")

# Auto-instruments the app with standard metrics (request count, latency
# histogram, in-progress requests) and exposes them at GET /metrics in
# Prometheus's text format - this single line is what Prometheus scrapes.
Instrumentator().instrument(app).expose(app)

# Custom, domain-specific metrics beyond the generic HTTP ones above.
# These are what make the dashboard actually useful for THIS system,
# not just "is the server up" - e.g. "are we suddenly seeing way more
# needs_review cases than usual" is a real operational signal.
DEFECTS_FOUND = Counter(
    "defects_found_total", "Total defects detected, by class", ["class_name"]
)
NEEDS_REVIEW_TOTAL = Counter(
    "needs_review_total", "Total detections flagged for human review"
)

# Loaded once at module import time (i.e. once when the server starts),
# not inside the endpoint function. This is the single most important
# performance decision in this file - see docstring above.
model = YOLO(str(MODEL_PATH))

# Warm up the model with a dummy prediction at startup. PyTorch's first
# inference call does one-time setup (thread pools, memory allocation)
# that adds real latency (~2 seconds, measured) - we pay that cost once
# here at server startup, not on whichever real user's request happens
# to be first.
_dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(_dummy_image, verbose=False)


class Detection(BaseModel):
    class_name: str
    confidence: float
    box_xyxy: List[float]  # [xmin, ymin, xmax, ymax] in pixels
    needs_review: bool


class PredictionResponse(BaseModel):
    filename: str
    defect_found: bool
    detections: List[Detection]


@app.get("/health")
def health_check():
    """
    Standard readiness/liveness endpoint. Real deployments (load balancers,
    container orchestrators, uptime monitors) poll this to confirm the
    service is alive before routing traffic to it.
    """
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs defect detection, returns structured
    results. This is the core inference endpoint a dashboard or client
    system would call.
    """
    if not file.content_type.startswith("image/"):
        log_event("predict_rejected", reason="not_an_image", content_type=file.content_type)
        raise HTTPException(status_code=400, detail="File must be an image")

    start = time.perf_counter()

    # Read uploaded bytes into an OpenCV image (BGR numpy array) without
    # writing to disk first - keeps this endpoint fast and stateless.
    raw_bytes = await file.read()
    np_array = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        log_event("predict_rejected", reason="decode_failed", filename=file.filename)
        raise HTTPException(status_code=400, detail="Could not decode image")

    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    detections = []
    for box in results.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        xyxy = box.xyxy[0].tolist()

        detections.append(Detection(
            class_name=class_name,
            confidence=round(confidence, 3),
            box_xyxy=[round(v, 1) for v in xyxy],
            needs_review=confidence < REVIEW_THRESHOLD,
        ))

        DEFECTS_FOUND.labels(class_name=class_name).inc()
        if confidence < REVIEW_THRESHOLD:
            NEEDS_REVIEW_TOTAL.inc()

    log_event(
        "prediction_made",
        filename=file.filename,
        latency_ms=latency_ms,
        n_detections=len(detections),
        n_needs_review=sum(1 for d in detections if d.needs_review),
        classes_found=[d.class_name for d in detections],
    )

    return PredictionResponse(
        filename=file.filename,
        defect_found=len(detections) > 0,
        detections=detections,
    )