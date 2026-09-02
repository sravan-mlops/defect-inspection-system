# Design Doc: Real-Time Manufacturing Defect Inspection System

## 1. Problem
A QA inspector on a steel/casting production line manually checks products for surface
defects. This is slow, inconsistent, and misses subtle defects under fatigue.

## 2. Users
- **Primary:** QA inspector — needs a pass/fail decision + defect location, fast.
- **Secondary:** Plant manager — needs a dashboard of defect trends over time.

## 3. Success criteria
- Model: recall on defect class ≥ 90% (missing a real defect is costlier than a false alarm)
- Latency: < 200ms per image on CPU (assume no GPU on factory floor edge device)
- System: inspector can upload/stream an image and get a decision in the UI within 1s

## 4. Constraints
- No GPU at inference time (realistic edge deployment assumption)
- Small labeled dataset (~1,800 images) — must handle class imbalance, use augmentation
- Must be explainable enough that inspector trusts it (show bounding box, not just a score)

## 5. Failure handling
- If model confidence < threshold → flag as "needs human review" instead of forcing a
  pass/fail. Never silently auto-pass an uncertain case.

## 6. Architecture (high level)
```
Camera/Upload -> FastAPI inference service -> Model (YOLOv8n) -> Postgres (results log)
                                                    |
                                              Streamlit dashboard (client demo)
                                                    |
                                          Prometheus/logs (monitoring)
```

## 7. Tradeoffs considered
- **YOLOv8n vs a heavier detector:** chose nano variant for CPU latency budget over
  raw accuracy — documents a real deployment tradeoff for interviews.
- **Self-hosted MLflow vs no tracking:** added MLflow so training runs are reproducible
  and comparable, mirrors real ML infra practice.

## 8. Out of scope (v1)
- Multi-camera fusion, active learning loop, on-device (embedded) deployment.
