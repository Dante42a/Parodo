"""
Classification metrics: AUROC, F1, confusion matrix for periodontal diagnosis.
"""

from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
)


CONDITION_LABELS = ["Normal", "Periodontitis_Local", "Periodontitis_Generalized", "Periodontosis"]
SEVERITY_LABELS = ["Mild", "Medium", "Severe"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
) -> Dict[str, float]:
    """
    y_true:  (N,) int   — ground truth class indices
    y_pred:  (N,) int   — predicted class indices
    y_score: (N, C) float — predicted probabilities per class
    labels:  list of class name strings
    """
    metrics = {}
    num_classes = len(labels)

    # Per-class AUROC (one-vs-rest)
    try:
        auroc_per = roc_auc_score(
            y_true, y_score, multi_class="ovr", average=None, labels=list(range(num_classes))
        )
        for i, lbl in enumerate(labels):
            metrics[f"AUROC_{lbl}"] = float(auroc_per[i])
        metrics["AUROC_macro"] = float(np.mean(auroc_per))
    except ValueError:
        # If a class is absent from y_true, skip AUROC
        metrics["AUROC_macro"] = float("nan")

    # F1 per class and macro
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    for i, lbl in enumerate(labels):
        metrics[f"F1_{lbl}"] = float(f1_per[i])
    metrics["F1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Cohen's Kappa
    metrics["Cohen_Kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # Confusion matrix (returned separately as numpy array)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["_confusion_matrix"] = cm  # not a scalar; handled by caller

    return metrics
