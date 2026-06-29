"""
Обучение YOLOv8m-seg на датасете AKUDENTAL.

333 снимка, 35 классов (FDI нотация зубов + Bridge/Crown/Implant), 16 точек/зуб.

Запуск:
    python scripts/train_yolo_akudental.py
"""

import os

os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

model.train(
    data="data/yolo_akudental/data.yaml",
    epochs=100,
    imgsz=1024,
    batch=2,
    workers=0,
    cache=False,
    mosaic=0.0,
    close_mosaic=0,
    half=True,
    patience=30,
    project="checkpoints/yolov8",
    name="teeth_akudental",
    exist_ok=True,
    plots=True,
)

print("\nЛучший чекпойнт: checkpoints/yolov8/teeth_akudental/weights/best.pt")
print("Следующий шаг:")
print("  python app/app.py --model checkpoints/swin_unet-epoch=10-val_mIoU=0.7739.ckpt --yolo checkpoints/yolov8/teeth_akudental/weights/best.pt")
