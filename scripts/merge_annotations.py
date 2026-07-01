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
from pathlib import Path

import cv2
import numpy as np

ANN_DIR   = Path("data/annotations")
MASK_DIR  = Path("data/processed/masks")
LABEL_MAP = {"bone": 2, "cortical": 3}
# порядок отрисовки: bone (2) первым, cortical (3) поверх — тонкая кортикальная
# линия не должна перекрываться костью
DRAW_ORDER = ["bone", "cortical"]


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

    # рисуем строго в порядке DRAW_ORDER: сперва bone, затем cortical поверх
    for target in DRAW_ORDER:
        for shape in data.get("shapes", []):
            if shape.get("label", "").lower() != target:
                continue
            rasterize_polygon(mask, shape["points"], LABEL_MAP[target])

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
    # сплиты генерирует отдельный скрипт (стратификация по cortical, 3-way без утечек):
    print("Далее: python scripts/create_bone_splits.py")


if __name__ == "__main__":
    main()
