"""
Shared helpers for model training (Days 4-8).
Project : Learner Performance Prediction & Early Learning Support Dashboard

DATA IS SYNTHETIC (see docs/source_notes.md).

One preprocessing definition is used by all four models, so the comparison is fair.
Every step is fitted on TRAINING data only, because the preprocessor lives inside
the scikit-learn Pipeline and is fitted by pipeline.fit(X_train, y_train).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

SEED = 42
BENCHMARK = 50                 # must match prepare_data.py; configurable to real policy
SCORE_MIN, SCORE_MAX = 0, 100  # valid final-score range (used for prediction clipping)

# The ONLY model inputs. IDs, outcome_status and both targets are never features.
QUIZ_COL = ["quiz_average"]
OTHER_NUMERIC = ["attendance_pct", "assignment_completion_pct", "active_learning_minutes",
                 "days_since_last_activity", "practice_attempt_count"]
CATEGORICAL = ["course_category"]
FEATURES = QUIZ_COL + OTHER_NUMERIC + CATEGORICAL
TARGET_SCORE = "final_score"
TARGET_CLASS = "below_benchmark"
FORBIDDEN_FEATURES = {"enrollment_id", "learner_group_id", "outcome_status",
                      TARGET_SCORE, TARGET_CLASS}

assert not set(FEATURES) & FORBIDDEN_FEATURES, "feature list contains ID or outcome fields"


def load_split(name: str):
    """Load one split ('train', 'validation' or 'test'). Returns (X, y_score, y_class, ids)."""
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    return df[FEATURES].copy(), df[TARGET_SCORE].copy(), df[TARGET_CLASS].copy(), df["enrollment_id"].copy()


def build_preprocessor() -> ColumnTransformer:
    """
    quiz_average : median imputation + 'quiz missing' indicator, then scaling
                   (a blank quiz average is not the same as a score of 0)
    other numeric: median imputation (safety net) + scaling
    category     : one-hot encoding; unseen categories are ignored, not crashed on
    """
    quiz = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("quiz", quiz, QUIZ_COL),
        ("numeric", numeric, OTHER_NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def predict_score(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Score prediction with the documented rule: clip to the valid 0-100 range."""
    return np.clip(pipeline.predict(X), SCORE_MIN, SCORE_MAX)
