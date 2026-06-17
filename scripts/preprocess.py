"""
Preprocess raw DENTEX images into standardized PNGs and generate metadata.csv.

Steps:
  1. Load each OPG from data/raw/dentex/
  2. Apply CLAHE for local contrast enhancement
  3. Resize to target dimensions (default 1024x512)
  4. Save to data/processed/images/<id>.png
  5. Generate per-tooth masks from DENTEX COCO annotations
     (initial masks; require manual refinement for bone/cortical layers)
  6. Save masks to data/processed/masks/<id>.png  (single-channel, values 0-3)
  7. Write data/processed/metadata.csv

Usage:
    python scripts/preprocess.py [--src data/raw/dentex] [--out data/processed]
                                 [--width 1024] [--height 512]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


CLASS_BACKGROUND = 0
CLASS_TOOTH = 1
CLASS_BONE = 2       # placeholder — requires annotation
CLASS_CORTICAL = 3   # placeholder — requires annotation


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def coco_segmentation_to_mask(
    segmentation,
    height: int,
    width: int,
) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def process_dentex(src: Path, out: Path, target_w: int, target_h: int):
    img_out = out / "images"
    mask_out = out / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # Find annotation JSON (disease-level split has richest annotations)
    ann_dir = src / "training_data" / "quadrant_enumeration_disease"
    ann_files = sorted(ann_dir.glob("*.json")) if ann_dir.exists() else []

    rows = []
    image_dirs = [
        src / "training_data" / "quadrant_enumeration_disease",
        src / "xrays",
    ]

    image_paths = []
    for d in image_dirs:
        if d.exists():
            image_paths += sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))

    # Build id→annotation map
    ann_map = {}
    for ann_file in ann_files:
        with open(ann_file) as f:
            coco = json.load(f)
        id_to_fname = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        for ann in coco.get("annotations", []):
            fname = id_to_fname.get(ann["image_id"], "")
            stem = Path(fname).stem
            ann_map.setdefault(stem, []).append(ann)

    for img_path in tqdm(image_paths, desc="Preprocessing"):
        stem = img_path.stem

        # Load as grayscale
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        orig_h, orig_w = gray.shape
        enhanced = apply_clahe(gray)
        resized = cv2.resize(enhanced, (target_w, target_h))

        out_img_path = img_out / f"{stem}.png"
        cv2.imwrite(str(out_img_path), resized)

        # Build mask (tooth layer only; bone/cortical placeholders = 0)
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        for ann in ann_map.get(stem, []):
            if "segmentation" in ann and ann["segmentation"]:
                scaled_seg = []
                for poly in ann["segmentation"]:
                    pts = np.array(poly).reshape(-1, 2)
                    pts[:, 0] = (pts[:, 0] * scale_x).clip(0, target_w - 1)
                    pts[:, 1] = (pts[:, 1] * scale_y).clip(0, target_h - 1)
                    scaled_seg.append(pts.flatten().tolist())
                tooth_mask = coco_segmentation_to_mask(scaled_seg, target_h, target_w)
                mask[tooth_mask == 1] = CLASS_TOOTH

        mask_path = mask_out / f"{stem}.png"
        cv2.imwrite(str(mask_path), mask)

        rows.append({
            "image_id": stem,
            "source": "dentex",
            "has_bone_mask": False,    # requires manual annotation
            "has_cortical_mask": False,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)
    print(f"\nProcessed {len(rows)} images → {out}")
    print("NOTE: bone and cortical masks need manual annotation in LabelMe/CVAT.")
    print("      Export annotations as COCO JSON and merge into the mask PNGs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/dentex")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    process_dentex(Path(args.src), Path(args.out), args.width, args.height)


if __name__ == "__main__":
    main()
