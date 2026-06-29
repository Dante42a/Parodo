"""
Подготовка датасета AKUDENTAL для обучения YOLOv8-seg.

Источник: data/raw/akudental/AKUDENTAL/
  - images/        — 333 оригинальных JPG (~2943x1435)
  - folds/AKUDENTAL_YOLO_SEG_FOLD_*/labels/ — YOLO-seg метки, 35 классов, 16 точек/зуб

Классы сохраняются как есть (FDI нотация + протезы):
  0-7   — зубы верхней правой четверти (11-18)
  8-15  — зубы верхней левой четверти  (21-28)
  16-23 — зубы нижней левой четверти   (31-38)
  24-31 — зубы нижней правой четверти  (41-48)
  32    — Bridge
  33    — Filling / Crown
  34    — Implant

Результат: data/yolo_akudental/
  - images/train|val/
  - labels/train|val/
  - data.yaml
"""

import random
import shutil
from pathlib import Path

SRC_ROOT  = Path("data/raw/akudental/AKUDENTAL")
OUT_DIR   = Path("data/yolo_akudental")
VAL_RATIO = 0.15
SEED      = 42

NAMES = [
    "11 - Central Incisor", "12 - Lateral Incisor", "13 - Canine",
    "14 - 1st Premolar",    "15 - 2nd Premolar",    "16 - 1st Molar",
    "17 - 2nd Molar",       "18 - 3rd Molar",
    "21 - Central Incisor", "22 - Lateral Incisor", "23 - Canine",
    "24 - 1st Premolar",    "25 - 2nd Premolar",    "26 - 1st Molar",
    "27 - 2nd Molar",       "28 - 3rd Molar",
    "31 - Central Incisor", "32 - Lateral Incisor", "33 - Canine",
    "34 - 1st Premolar",    "35 - 2nd Premolar",    "36 - 1st Molar",
    "37 - 2nd Molar",       "38 - 3rd Molar",
    "41 - Central Incisor", "42 - Lateral Incisor", "43 - Canine",
    "44 - 1st Premolar",    "45 - 2nd Premolar",    "46 - 1st Molar",
    "47 - 2nd Molar",       "48 - 3rd Molar",
    "Bridge", "Filling - Crown", "Implant",
]


def collect_labels():
    """Собираем метки из всех 5 фолдов — каждый снимок берём один раз."""
    labels = {}
    for fold_n in range(5):
        fold_dir = SRC_ROOT / "folds" / f"AKUDENTAL_YOLO_SEG_FOLD_{fold_n}" / "labels"
        for split in ("train", "test"):
            split_dir = fold_dir / split
            if not split_dir.exists():
                continue
            for lbl_file in split_dir.glob("*.txt"):
                if lbl_file.stem not in labels:
                    labels[lbl_file.stem] = lbl_file
    return labels


def main():
    img_dir = SRC_ROOT / "images"

    labels = collect_labels()
    print(f"Собрано меток: {len(labels)}")

    valid = {stem: lbl for stem, lbl in labels.items()
             if (img_dir / f"{stem}.jpg").exists()}
    print(f"С изображением: {len(valid)}")

    stems = sorted(valid.keys())
    random.seed(SEED)
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * VAL_RATIO))
    val_stems   = set(stems[:n_val])
    train_stems = set(stems[n_val:])
    print(f"Сплит → train: {len(train_stems)}, val: {len(val_stems)}")

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, split_stems in (("train", train_stems), ("val", val_stems)):
        for stem in split_stems:
            shutil.copy2(img_dir / f"{stem}.jpg",
                         OUT_DIR / "images" / split / f"{stem}.jpg")
            shutil.copy2(valid[stem],
                         OUT_DIR / "labels" / split / f"{stem}.txt")

    abs_out = OUT_DIR.resolve().as_posix()
    names_yaml = "\n".join(f"    {i}: '{n}'" for i, n in enumerate(NAMES))
    (OUT_DIR / "data.yaml").write_text(
        f"path: {abs_out}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(NAMES)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )
    print(f"\nДатасет готов → {OUT_DIR}/data.yaml")
    print("Следующий шаг: python scripts/train_yolo_akudental.py")


if __name__ == "__main__":
    main()
