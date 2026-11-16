"""Fake learner rows for the CSV template and the tests. Never reads any file in data/."""

import io

import numpy as np
import pandas as pd

from validation import CATEGORIES, REQUIRED_COLUMNS


def fake_rows(n: int, seed: int = 42, blank_quiz_share: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quiz = np.round(np.clip(rng.normal(60, 18, n), 0, 100), 1).astype(object)
    quiz[rng.random(n) < blank_quiz_share] = ""
    return pd.DataFrame({
        "enrollment_id": [f"DEMO-{i:04d}" for i in range(1, n + 1)],
        "quiz_average": quiz,
        "attendance_pct": np.round(np.clip(rng.normal(72, 15, n), 0, 100), 1),
        "assignment_completion_pct": np.round(np.clip(rng.normal(70, 18, n), 0, 100), 1),
        "active_learning_minutes": np.round(np.exp(rng.normal(5.4, 0.4, n))).astype(int),
        "days_since_last_activity": np.minimum(rng.poisson(2.5, n), 14),
        "practice_attempt_count": rng.poisson(6, n),
        "course_category": rng.choice(CATEGORIES, n),
    })[REQUIRED_COLUMNS]


def template_csv(n: int = 5) -> bytes:
    buf = io.StringIO()
    fake_rows(n).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
