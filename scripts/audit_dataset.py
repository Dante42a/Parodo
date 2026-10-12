"""Create a reproducible manifest for committed LabelMe annotations."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(annotations: Path) -> dict:
    files = sorted(annotations.glob("*.json"))
    records = []
    labels = Counter()
    image_names = Counter()
    embedded_images = 0
    invalid = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append({"file": path.name, "error": str(exc)})
            continue
        image_name = Path(payload.get("imagePath") or "").name
        image_names[image_name] += 1
        embedded_images += int(payload.get("imageData") is not None)
        file_labels = Counter(shape.get("label", "<missing>") for shape in payload.get("shapes", []))
        labels.update(file_labels)
        records.append(
            {
                "annotation": path.name,
                "image": image_name,
                "sha256": sha256(path),
                "labels": dict(sorted(file_labels.items())),
            }
        )
    return {
        "schema_version": 1,
        "annotation_count": len(files),
        "valid_annotation_count": len(records),
        "invalid": invalid,
        "embedded_image_count": embedded_images,
        "label_counts": dict(sorted(labels.items())),
        "duplicate_image_names": sorted(k for k, count in image_names.items() if k and count > 1),
        "patient_grouping_available": False,
        "patient_grouping_note": (
            "Patient identifiers are not present; patient-level leakage cannot be excluded."
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_manifest.json"))
    args = parser.parse_args()
    result = audit(args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
