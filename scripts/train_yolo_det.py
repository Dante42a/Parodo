"""
Обучение YOLOv8m detection (только боксы, не сегментация).

Запуск:
    python scripts/train_yolo_det.py
"""

import os

os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="data/yolo_teeth_det/data.yaml",
    epochs=80,
    imgsz=1024,
    batch=4,          # detection легче чем seg, можно batch=4
    workers=0,
    cache=False,
    mosaic=0.0,
    close_mosaic=0,
    half=True,
    patience=25,
    project="checkpoints/yolov8",
    name="teeth_det",
    exist_ok=True,
    plots=True,
)

print("\nЛучший чекпойнт: checkpoints/yolov8/teeth_det/weights/best.pt")
print("Следующий шаг:")
print("  python app/app.py --model checkpoints/swin_unet-epoch=10-val_mIoU=0.7739.ckpt --yolo checkpoints/yolov8/teeth_det/weights/best.pt")
