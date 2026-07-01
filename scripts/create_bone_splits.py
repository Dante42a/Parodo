"""
Создание непересекающихся train/val/test сплитов (70/15/15) для bone-датасета.

В сплит попадают только снимки с разметкой класса `bone`. Стратификация — по
наличию класса `cortical`, чтобы кортикальная пластинка равномерно распределилась
по всем трём выборкам (иначе в test её может не оказаться вовсе, и Dice_cortical
будет невозможно посчитать).

Usage:
    python scripts/create_bone_splits.py [--ann data/annotations]
                                         [--out data/splits/bone]
                                         [--seed 42]
"""

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def scan_annotations(ann_dir: Path):
    """Возвращает (stems, cortical_flags) по всем снимкам с разметкой bone."""
    stems, cortical = [], []
    for f in sorted(ann_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        labels = {s["label"] for s in data.get("shapes", [])}
        if "bone" not in labels:
            continue
        stems.append(f.stem)
        cortical.append("cortical" in labels)
    return stems, cortical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default="data/annotations")
    parser.add_argument("--out", default="data/splits/bone")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stems, cortical = scan_annotations(Path(args.ann))
    n_cort = sum(cortical)
    print(f"bone-размечено: {len(stems)} (из них cortical: {n_cort})")

    # сначала отделяем 30% (val+test), стратифицируя по cortical
    train, rest, _, rest_cort = train_test_split(
        stems, cortical, test_size=0.30, random_state=args.seed, stratify=cortical
    )
    # оставшиеся 30% делим пополам → val 15% / test 15%
    val, test = train_test_split(
        rest, test_size=0.50, random_state=args.seed, stratify=rest_cort
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(sorted(train)) + "\n")
    (out / "val.txt").write_text("\n".join(sorted(val)) + "\n")
    (out / "test.txt").write_text("\n".join(sorted(test)) + "\n")

    def n_cort_in(subset):
        idx = {s: c for s, c in zip(stems, cortical)}
        return sum(idx[s] for s in subset)

    print(f"train={len(train)} (cortical {n_cort_in(train)}), "
          f"val={len(val)} (cortical {n_cort_in(val)}), "
          f"test={len(test)} (cortical {n_cort_in(test)})")
    print(f"Сохранено в {out}/")


if __name__ == "__main__":
    main()
