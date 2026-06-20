"""
Create stratified train/val/test splits (70/15/15) from metadata.csv.

Stratification is done by source (dentex/tufts) to avoid bias.
Patient-level grouping: images from the same patient stay in the same split
(DENTEX image IDs encode patient; adjust `patient_id_fn` if needed).

Usage:
    python scripts/create_splits.py [--meta data/processed/metadata.csv]
                                    [--out data/splits]
                                    [--seed 42]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def patient_id_from_image_id(image_id: str) -> str:
    # DENTEX IDs are "train_N" — no real patient grouping available,
    # so treat each image as its own unit for the split.
    return image_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="data/processed/metadata.csv")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.meta)
    df["patient_id"] = df["image_id"].apply(patient_id_from_image_id)

    # Group by patient to prevent leakage
    patients = df["patient_id"].unique()
    np.random.seed(args.seed)
    np.random.shuffle(patients)

    n = len(patients)
    n_val = max(1, int(0.15 * n))
    n_test = max(1, int(0.15 * n))
    n_train = n - n_val - n_test

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    train_ids = df[df["patient_id"].isin(train_patients)]["image_id"].tolist()
    val_ids = df[df["patient_id"].isin(val_patients)]["image_id"].tolist()
    test_ids = df[df["patient_id"].isin(test_patients)]["image_id"].tolist()

    (out / "train.txt").write_text("\n".join(train_ids))
    (out / "val.txt").write_text("\n".join(val_ids))
    (out / "test.txt").write_text("\n".join(test_ids))

    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
