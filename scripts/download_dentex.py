"""
Download and unzip the DENTEX 2023 dataset from Zenodo.

Usage:
    python scripts/download_dentex.py [--out data/raw/dentex]

DENTEX Zenodo DOI: 10.5281/zenodo.7812257
Dataset contains:
  - training_data/quadrant/          — 693 OPGs, quadrant-level bbox
  - training_data/quadrant_enumeration/  — 634 OPGs, quadrant+tooth# bbox
  - training_data/quadrant_enumeration_disease/  — 1005 OPGs, COCO JSON with disease labels
  - xrays/                          — raw images

After download, images go to data/raw/dentex/images/
and annotations to data/raw/dentex/annotations/
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


ZENODO_BASE = "https://zenodo.org/records/7812323/files/training_data.zip?download=1"
FILES = {
    "training_data.zip": f"{ZENODO_BASE}/training_data.zip?download=1",
}


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {dest.name} (~10.9 GB, this will take a while) ...")
    with requests.get(url, stream=True, timeout=30) as r:  # timeout = connect only
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/dentex")
    args = parser.parse_args()
    out = Path(args.out)

    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    for fname, url in FILES.items():
        zip_path = tmp / fname
        download_file(url, zip_path)
        print(f"  Extracting {fname} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)

    shutil.rmtree(tmp)
    print(f"\nDENTEX dataset extracted to: {out}")
    print("Next step: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
