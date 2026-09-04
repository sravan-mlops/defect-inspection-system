"""
benchmark_latency.py

Purpose: Measure actual inference latency against our design doc's stated
requirement (<200ms per image on CPU). Never assume a requirement is met -
measure it.

Run: python src/evaluation/benchmark_latency.py
(requires the API server to already be running: uvicorn src.serving.app:app)
"""

import time
from pathlib import Path
from statistics import mean, median

import requests

API_URL = "http://127.0.0.1:8000/predict"
TEST_IMAGES_DIR = Path("data/processed/images/val")
N_REQUESTS = 30
LATENCY_TARGET_MS = 200


def benchmark():
    image_paths = list(TEST_IMAGES_DIR.glob("*.jpg"))[:N_REQUESTS]
    if not image_paths:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return

    latencies_ms = []

    for img_path in image_paths:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}

            start = time.perf_counter()
            response = requests.post(API_URL, files=files)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                latencies_ms.append(elapsed_ms)
            else:
                print(f"  Request failed for {img_path.name}: {response.status_code}")

    print(f"\nRan {len(latencies_ms)} requests")
    print(f"  Mean latency:   {mean(latencies_ms):.1f} ms")
    print(f"  Median latency: {median(latencies_ms):.1f} ms")
    print(f"  Min / Max:      {min(latencies_ms):.1f} / {max(latencies_ms):.1f} ms")
    print(f"\nDesign doc target: <{LATENCY_TARGET_MS} ms")
    if mean(latencies_ms) < LATENCY_TARGET_MS:
        print("PASS - within latency budget")
    else:
        print("FAIL - exceeds latency budget, needs investigation")


if __name__ == "__main__":
    benchmark()