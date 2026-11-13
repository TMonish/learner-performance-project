"""
Evaluation helpers shared by the classifier scripts (Days 5-7).
Project : Learner Performance Prediction & Early Learning Support Dashboard

DATA IS SYNTHETIC. Results demonstrate the implementation, not effectiveness on real learners.

The positive class is below_benchmark = 1 (learner finishing below the benchmark).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def md_table(df: pd.DataFrame) -> str:
    """Small markdown table writer (avoids an extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.astype(object).iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def positive_class_probability(pipeline, X) -> np.ndarray:
    """Probability of below_benchmark = 1. Verifies column order so it can never be flipped."""
    classes = list(pipeline.classes_)
    assert classes == [0, 1], f"unexpected class order {classes}"
    return pipeline.predict_proba(X)[:, 1]


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """Threshold-free metrics (ROC-AUC, PR-AUC) plus label metrics at the given threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": round(float(roc_auc_score(y_true, prob)), 3),
        "PR-AUC": round(float(average_precision_score(y_true, prob)), 3),
        "Precision": round(float(precision_score(y_true, pred, zero_division=0)), 3),
        "Recall": round(float(recall_score(y_true, pred, zero_division=0)), 3),
        "F1": round(float(f1_score(y_true, pred, zero_division=0)), 3),
        "Accuracy": round(float(accuracy_score(y_true, pred)), 3),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
