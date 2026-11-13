"""
Day 7 shared helpers: model registry, threshold selection, file hashing.
Project : Learner Performance Prediction & Early Learning Support Dashboard

DATA IS SYNTHETIC. Results demonstrate the implementation, not effectiveness on real learners.

ASSUMPTIONS THAT BELONG TO THE CLIENT (change them here BEFORE running select_and_freeze.py,
and confirm them with Career Solutions):
  CAPACITY - the largest share of learners that coordinators can realistically support.
  BETA     - how much more a missed struggling learner matters than a false alarm.
             BETA = 2 means recall counts twice as much as precision (an F2 score).
"""

import hashlib

import joblib
import numpy as np
import pandas as pd

from eval_utils import positive_class_probability
from pipeline_utils import MODELS_DIR

# ---------------- client-owned assumptions (documented, configurable) ----------------
CAPACITY = 0.40            # at most 40% of learners can be flagged for support
BETA = 2.0                 # F-beta used to pick the threshold (recall weighted 2x precision)

# ---------------- selection rules ----------------
PRAUC_TOLERANCE = 0.01     # models within this PR-AUC of the best are treated as tied on validation
TIE_BREAK_ORDER = ["Logistic regression", "SVM", "Decision tree"]   # interpretable, native probabilities first
MIN_BRIER_SKILL = 0.25     # probability shown in the dashboard only if validation Brier skill >= this

MODELS = {
    "Logistic regression": {"file": "logistic_regression_pipeline.joblib", "score_type": "probability"},
    "Decision tree": {"file": "decision_tree_pipeline.joblib", "score_type": "probability"},
    "SVM": {"file": "svm_pipeline.joblib", "score_type": "decision_score"},
}
REGRESSION_FILE = "linear_regression_pipeline.joblib"
FROZEN_FILE = MODELS_DIR / "frozen_settings.json"
FINAL_CONFIG_FILE = MODELS_DIR / "final_config.json"
LOCK_FILE = MODELS_DIR / "test_evaluation_lock.json"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_models() -> dict:
    missing = [m["file"] for m in MODELS.values() if not (MODELS_DIR / m["file"]).exists()]
    if missing:
        raise SystemExit(f"Missing saved pipelines {missing}. Run the Day 5 and Day 6 scripts first.")
    return {name: joblib.load(MODELS_DIR / m["file"]) for name, m in MODELS.items()}


def get_scores(pipeline, X, score_type: str) -> np.ndarray:
    """Probability of below_benchmark = 1, or the SVM decision score. Class order is verified."""
    if score_type == "probability":
        return positive_class_probability(pipeline, X)
    assert list(pipeline.classes_) == [0, 1], "unexpected class order"
    return pipeline.decision_function(X)


def sweep_thresholds(y, scores) -> pd.DataFrame:
    """Evaluate every candidate threshold (midpoints between consecutive distinct scores).
    A learner is flagged for support when score >= threshold."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    uniq = np.unique(s)
    if len(uniq) < 2:
        raise SystemExit("Scores have no variation; cannot choose a threshold.")
    cands = (uniq[:-1] + uniq[1:]) / 2
    positives = int(y.sum())
    rows = []
    for t in cands:
        flag = s >= t
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        fn = positives - tp
        tn = len(y) - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / positives if positives else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fb = (1 + BETA**2) * prec * rec / (BETA**2 * prec + rec) if (prec + rec) else 0.0
        rows.append({"threshold": float(t), "flagged_share": float(flag.mean()),
                     "precision": prec, "recall": rec, "f1": f1, "f_beta": fb,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame) -> pd.Series:
    """Highest F-beta among thresholds that flag no more than CAPACITY of learners.
    Ties go to the higher threshold (flags fewer learners)."""
    feasible = sweep[sweep["flagged_share"] <= CAPACITY + 1e-12]
    if feasible.empty:
        raise SystemExit(f"No threshold keeps flagged share within {CAPACITY:.0%}.")
    best = feasible["f_beta"].max()
    tied = feasible[feasible["f_beta"] >= best - 1e-12]
    return tied.sort_values("threshold", ascending=False).iloc[0]
