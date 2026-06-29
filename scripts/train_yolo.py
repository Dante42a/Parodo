"""
Train YOLOv8-seg for per-tooth instance segmentation.

Run prepare_yolo_dataset.py first, then:
    python scripts/train_yolo.py
"""

import os
from pathlib import Path

# Disable ultralytics' built-in MLflow callback (conflicts with file-store backend)
os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from ultralytics import YOLO

DATA_YAML = str(Path("data/yolo_teeth/data.yaml").resolve())
BASE_MODEL = "yolov8m-seg.pt"   # COCO-pretrained, ~52 MB
EPOCHS = 60
IMGSZ = 1024
BATCH = 2
DEVICE = 0


def main():
    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="checkpoints/yolov8",
        name="teeth",
        exist_ok=True,
        patience=20,
        save=True,
        cache=False,       # network drive — no caching
        workers=0,         # network drive — no multiprocess workers
        rect=False,
        cos_lr=True,
        close_mosaic=10,
    )
    best = Path("checkpoints/yolov8/teeth/weights/best.pt")
    print(f"\nBest checkpoint: {best}")
    print("Next step: python app/app.py --yolo", best)


if __name__ == "__main__":
    main()
