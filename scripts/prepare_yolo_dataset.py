"""
Convert DENTEX quadrant_enumeration COCO annotations to YOLOv8 instance segmentation format.

All teeth are treated as one class (class 0 = "tooth") since we only need
individual tooth instances, not their type.

Creates:
  data/yolo_teeth/
    images/train/   copies of data/processed/images/
    images/val/
    labels/train/   YOLO polygon format: "0 x1 y1 x2 y2 ..."
    labels/val/
    data.yaml
"""

import json
import random
import shutil
from pathlib import Path

DENTEX_JSON = "data/raw/dentex/training_data/quadrant_enumeration/train_quadrant_enumeration.json"
PROC_DIR = Path("data/processed/images")
OUT_DIR = Path("data/yolo_teeth")
VAL_RATIO = 0.15
SEED = 42


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
    print(f"Images in DENTEX QE: {len(id_to_img)}, with processed version: {len(valid_ids)}")

    random.seed(SEED)
    random.shuffle(valid_ids)
    n_val = max(1, int(len(valid_ids) * VAL_RATIO))
    val_ids = set(valid_ids[:n_val])
    train_ids = set(valid_ids[n_val:])
    print(f"Split → train: {len(train_ids)}, val: {len(val_ids)}")

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def process_split(img_ids, split):
        ok = skipped = 0
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
                seg = ann.get("segmentation", [])
                if not seg or not isinstance(seg, list):
                    continue
                poly = seg[0]
                if len(poly) < 6:
                    continue
                # Normalize to [0,1]: divide x by orig_w, y by orig_h.
                # The preprocess pipeline resizes (orig_w x orig_h) → (1024 x 512)
                # so relative coords are preserved exactly.
                pts = []
                for i in range(0, len(poly) - 1, 2):
                    pts.append(f"{poly[i] / orig_w:.6f}")
                    pts.append(f"{poly[i + 1] / orig_h:.6f}")
                lines.append("0 " + " ".join(pts))

            label_path = OUT_DIR / "labels" / split / (Path(fname).stem + ".txt")
            label_path.write_text("\n".join(lines), encoding="utf-8")
            ok += 1

        print(f"  {split}: {ok} written, {skipped} skipped")

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
    print(f"\nDataset ready → {OUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()
