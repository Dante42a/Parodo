"""
Конвертация DENTEX quadrant_enumeration в YOLOv8 detection формат (только боксы).

Боксы точнее 6-точечных полигонов — модель детекции обучается лучше.
Per-tooth маски потом получаются пересечением бокса с Swin-UNet mask_tooth.

Создаёт:
  data/yolo_teeth_det/
    images/train|val/   — те же что в yolo_teeth (symlink-like copy)
    labels/train|val/   — YOLO det формат: "0 cx cy w h"
    data.yaml
"""

import json
import random
import shutil
from pathlib import Path

DENTEX_JSON = "data/raw/dentex/training_data/quadrant_enumeration/train_quadrant_enumeration.json"
PROC_DIR = Path("data/processed/images")
OUT_DIR = Path("data/yolo_teeth_det")
VAL_RATIO = 0.15
SEED = 42


def coco_bbox_to_yolo(x, y, w, h, orig_w, orig_h):
    """COCO [x_top_left, y_top_left, w, h] → YOLO [cx, cy, w, h] нормализованные."""
    cx = (x + w / 2) / orig_w
    cy = (y + h / 2) / orig_h
    nw = w / orig_w
    nh = h / orig_h
    return cx, cy, nw, nh


def main():
    with open(DENTEX_JSON) as f:
        d = json.load(f)

    id_to_img = {img["id"]: img for img in d["images"]}

    id_to_anns: dict = {}
    for ann in d["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    valid_ids = [
        img_id for img_id, img in id_to_img.items()
        if (PROC_DIR / img["file_name"]).exists()
    ]
    print(f"Снимков в DENTEX QE: {len(id_to_img)}, с обработанной версией: {len(valid_ids)}")

    random.seed(SEED)
    random.shuffle(valid_ids)
    n_val = max(1, int(len(valid_ids) * VAL_RATIO))
    val_ids = set(valid_ids[:n_val])
    train_ids = set(valid_ids[n_val:])
    print(f"Сплит → train: {len(train_ids)}, val: {len(val_ids)}")

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def process_split(img_ids, split):
        ok = skipped = 0
        total_boxes = 0
        for img_id in img_ids:
            img_info = id_to_img[img_id]
            fname = img_info["file_name"]
            orig_w = img_info["width"]
            orig_h = img_info["height"]

            src = PROC_DIR / fname
            if not src.exists():
                skipped += 1
                continue

            shutil.copy2(str(src), str(OUT_DIR / "images" / split / fname))

            anns = id_to_anns.get(img_id, [])
            lines = []
            for ann in anns:
                bbox = ann.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                bx, by, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    continue
                cx, cy, nw, nh = coco_bbox_to_yolo(bx, by, bw, bh, orig_w, orig_h)
                # Клипируем на случай выхода за границы
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.001, min(1.0, nw))
                nh = max(0.001, min(1.0, nh))
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                total_boxes += 1

            label_path = OUT_DIR / "labels" / split / (Path(fname).stem + ".txt")
            label_path.write_text("\n".join(lines), encoding="utf-8")
            ok += 1

        print(f"  {split}: {ok} снимков, {total_boxes} боксов, пропущено: {skipped}")

    process_split(train_ids, "train")
    process_split(val_ids, "val")

    abs_out = OUT_DIR.resolve().as_posix()
    (OUT_DIR / "data.yaml").write_text(
        f"path: {abs_out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['tooth']\n",
        encoding="utf-8",
    )
    print(f"\nДатасет готов → {OUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()
