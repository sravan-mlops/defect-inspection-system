"""
benchmark_inference_only.py

Purpose: Time the model's raw inference speed directly, bypassing the API/
network layer entirely. This isolates whether slowness is in the model
itself or somewhere in the FastAPI serving layer.

Run: python src/evaluation/benchmark_inference_only.py
"""

import time
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
TEST_IMAGES_DIR = Path("data/processed/images/val")
N_REQUESTS = 10

model = YOLO(MODEL_PATH)

image_paths = list(TEST_IMAGES_DIR.glob("*.jpg"))[:N_REQUESTS]

print("Warmup run (not counted - first call includes one-time setup)...")
model.predict(str(image_paths[0]), verbose=False)

print(f"\nTiming {len(image_paths)} predictions...")
times_ms = []
for img_path in image_paths:
    start = time.perf_counter()
    model.predict(str(img_path), verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    times_ms.append(elapsed_ms)
    print(f"  {img_path.name}: {elapsed_ms:.1f} ms")

print(f"\nMean: {sum(times_ms)/len(times_ms):.1f} ms")