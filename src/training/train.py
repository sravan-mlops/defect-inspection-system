"""
train.py

Purpose: Fine-tune a COCO-pretrained YOLOv8n model on our defect dataset,
using transfer learning (not training from scratch - see design_doc.md
for why this matters with a small dataset).

Logs the run to MLflow so every experiment (settings + results) is recorded
and comparable.

Run: python src/training/train.py
"""

from ultralytics import YOLO
import mlflow


DATA_YAML = "data/processed/data.yaml"
EPOCHS = 50
IMG_SIZE = 640  # YOLO's standard training resolution; it resizes our 200x200
                # source images up to this internally
BATCH_SIZE = 16
MODEL_VARIANT = "yolov8n.pt"  # "n" = nano: smallest/fastest, matches our
                               # no-GPU, <200ms latency constraint from the design doc


def main():
    mlflow.set_experiment("defect-inspection")

    with mlflow.start_run():
        # Log the settings we chose and WHY, so a future you (or an
        # interviewer) can see the reasoning without re-reading code
        mlflow.log_params({
            "model_variant": MODEL_VARIANT,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "transfer_learning": True,
        })

        # Load COCO-pretrained weights - this IS the transfer learning step.
        # We are not starting from random weights.
        model = YOLO(MODEL_VARIANT)

        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project="runs/train",
            name="defect_yolov8n",
        )

        # Log final validation metrics to MLflow so runs are comparable
        metrics = results.results_dict
        mlflow.log_metrics({
            "mAP50": metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall": metrics.get("metrics/recall(B)", 0),
        })

        print("\nTraining complete.")
        print(f"Best weights saved to: runs/train/defect_yolov8n/weights/best.pt")


if __name__ == "__main__":
    main()