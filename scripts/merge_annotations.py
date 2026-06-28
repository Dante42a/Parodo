"""
Merge LabelMe bone/cortical annotations into existing tooth masks.

For each JSON in data/annotations/:
  1. Load existing mask (background=0, tooth=1, bone=2, cortical=3)
  2. Draw bone polygons  → class 2
  3. Draw cortical polygons → class 3
  4. Save updated mask

Then create bone_train.txt / bone_val.txt splits from annotated images.

Usage:
    python scripts/merge_annotations.py
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

ANN_DIR   = Path("data/annotations")
MASK_DIR  = Path("data/processed/masks")
SPLIT_DIR = Path("data/splits")
LABEL_MAP = {"bone": 2}  # cortical (3) будет добавлен после завершения его разметки
VAL_RATIO = 0.15
SEED      = 42


def rasterize_polygon(mask: np.ndarray, points: list, value: int) -> None:
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color=value)


def merge_one(json_path: Path) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    stem = json_path.stem  # e.g. "train_0"

    mask_path = MASK_DIR / f"{stem}.png"
    if not mask_path.exists():
        print(f"  [skip] mask not found: {mask_path}")
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [skip] could not read mask: {mask_path}")
        return None

    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower()
        if label not in LABEL_MAP:
            continue
        rasterize_polygon(mask, shape["points"], LABEL_MAP[label])

    cv2.imwrite(str(mask_path), mask)
    return stem


def main():
    json_files = sorted(ANN_DIR.glob("*.json"))
    print(f"Found {len(json_files)} annotation files")

    annotated = []
    for jf in json_files:
        stem = merge_one(jf)
        if stem:
            annotated.append(stem)

    print(f"Merged {len(annotated)} masks")

    # Create bone splits from annotated images only
    random.seed(SEED)
    random.shuffle(annotated)
    n_val = max(1, int(len(annotated) * VAL_RATIO))
    val   = sorted(annotated[:n_val])
    train = sorted(annotated[n_val:])

    bone_split_dir = SPLIT_DIR / "bone"
    bone_split_dir.mkdir(parents=True, exist_ok=True)
    (bone_split_dir / "train.txt").write_text("\n".join(train))
    (bone_split_dir / "val.txt").write_text("\n".join(val))
    # test = val for now (no held-out annotated test set)
    (bone_split_dir / "test.txt").write_text("\n".join(val))

    print(f"Splits: train={len(train)}, val={len(val)} → data/splits/bone/")
    print(f"Run training with: python scripts/train.py --config configs/swin_unet_bone.yaml")


if __name__ == "__main__":
    main()
