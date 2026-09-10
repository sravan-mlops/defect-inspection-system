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

## 9. Known Limitations & Future Improvements

**Model accuracy is uneven across defect classes.** Overall mAP50 is 0.737,
but recall specifically is weak on `crazing` (35%) and `rolled-in_scale`
(45-55%) versus `patches`/`scratches` (~90%). These two weak classes are
low-contrast, texture-based defects, structurally harder for a nano-sized
model to distinguish from normal surface texture than the sharp-edged
defect types.

**What we tried:** a second training run (100 epochs vs 50, added rotation/
scale/mixup augmentation) improved `rolled-in_scale` recall (45%→55%) but
left `crazing` unchanged and slightly hurt the already-strong classes. This
was a legitimate, measured experiment with a genuinely mixed result — not
every intervention helps, and recognizing that (rather than continuing to
tune indefinitely) was a deliberate scope decision given diminishing
returns without a fundamentally different lever (bigger model = more
latency cost; more/better labeled data = not available for this dataset).

**What would likely help, if continued:**

- A larger model variant (YOLOv8s/m) — but this trades away CPU latency
  budget, so would require either accepting a higher latency target or
  moving inference to GPU.
- Class-weighted loss specifically up-weighting `crazing` and
  `rolled-in_scale` during training.
- More labeled examples of the weak classes specifically, rather than
  more epochs on the existing imbalanced set.

**Deployment/infra limitations, by deliberate scope choice for a portfolio project:**

- The ECS security group allows inbound traffic from `0.0.0.0/0` (anywhere).
  Fine for a demo; a real deployment would restrict this to a load balancer
  or known client IP ranges.
- CI runs tests and verifies the Docker image builds on every push, but does
  NOT auto-deploy to AWS — deploying costs money and requires storing cloud
  credentials as CI secrets, a tradeoff not worth it for iterating on a
  personal project. A real team would add a manual-approval deploy step.
- No experiment-tracking server (MLflow) in the cloud training run (Colab)
  since Colab sessions are ephemeral — MLflow is used for local training
  runs only. A persistent tracking server would be added in a team setting.
