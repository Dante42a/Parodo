import json

from scripts.audit_dataset import audit


def test_audit_reports_labels_and_embedded_data(tmp_path):
    annotation = {
        "imagePath": "sample.png",
        "imageData": None,
        "shapes": [{"label": "bone"}, {"label": "cortical"}],
    }
    (tmp_path / "one.json").write_text(json.dumps(annotation), encoding="utf-8")
    result = audit(tmp_path)
    assert result["valid_annotation_count"] == 1
    assert result["embedded_image_count"] == 0
    assert result["label_counts"] == {"bone": 1, "cortical": 1}
