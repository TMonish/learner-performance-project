"""
Model loading and prediction for the dashboard (no Streamlit imports).
Reuses the saved pipelines and the frozen settings. Nothing is retrained here.
DATA IS SYNTHETIC.
"""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

import paths  # noqa: F401
from pipeline_utils import FEATURES, MODELS_DIR, predict_score
from validation import csv_safe

SUPPORT_LABEL = "May benefit from extra support"
ON_TRACK_LABEL = "On track"


def load_config() -> dict:
    path = MODELS_DIR / "final_config.json"
    if not path.exists():
        raise FileNotFoundError("models/final_config.json not found. Run the Day 7 scripts first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen():
    path = MODELS_DIR / "frozen_settings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def integrity_warnings(frozen) -> list:
    """Compare model files with the fingerprints recorded when the settings were frozen."""
    if not frozen:
        return []
    bad = []
    for fname, expected in frozen.get("file_sha256", {}).items():
        path = MODELS_DIR / fname
        if not path.exists():
            bad.append(f"{fname} is missing")
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            bad.append(f"{fname} differs from the frozen version")
    return bad


def load_models(config: dict):
    reg = joblib.load(MODELS_DIR / config["regression_pipeline"])
    clf = joblib.load(MODELS_DIR / config["selected_classifier"])
    return reg, clf


def make_frame(rows) -> pd.DataFrame:
    """Exactly the FEATURES columns in the training order. A blank quiz stays NaN (never 0)."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df = df[FEATURES].copy()
    for col in FEATURES:
        if col != "course_category":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df.reset_index(drop=True)


def support_score(clf, X: pd.DataFrame, threshold_type: str) -> np.ndarray:
    """Probability of below_benchmark = 1, or the decision score for margin-based models."""
    assert list(clf.classes_) == [0, 1], "unexpected class order"
    if threshold_type == "probability":
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def predict(models, config: dict, X: pd.DataFrame) -> pd.DataFrame:
    reg, clf = models
    score = np.round(predict_score(reg, X), 1)
    s = support_score(clf, X, config["threshold_type"])
    flagged = s >= config["threshold"]
    return pd.DataFrame({
        "predicted_final_score": score,
        "support_score": s,
        "flagged": flagged,
        "predicted_support_class": np.where(flagged, SUPPORT_LABEL, ON_TRACK_LABEL),
    })


def build_export(valid: pd.DataFrame, preds: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prediction file: original order, matched by enrollment_id, spreadsheet-safe text."""
    out = pd.DataFrame({
        "enrollment_id": [csv_safe(v) for v in valid["enrollment_id"]],
        "predicted_final_score": preds["predicted_final_score"].to_numpy(),
        "predicted_support_class": preds["predicted_support_class"].to_numpy(),
    })
    if config["threshold_type"] == "probability" and config["probability_available"]:
        out["below_benchmark_probability"] = np.round(preds["support_score"].to_numpy(), 3)
    else:
        out["decision_score"] = np.round(preds["support_score"].to_numpy(), 3)
    return out
