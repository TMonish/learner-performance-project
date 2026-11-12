
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42                    
N_OBSERVED = 1500            
N_WITHDRAWN = 40             
N_MISSING_ASSESSMENT = 20    
BENCHMARK = 50               
OUTPUT_FILE = Path(__file__).parent / "synthetic_learner_data.csv"

CATEGORIES = ["Python", "Data Analytics", "Machine Learning", "Web Development", "SQL"]
CATEGORY_PROBS = [0.28, 0.24, 0.18, 0.16, 0.14]

CATEGORY_EFFECT = {
    "Python": 1.0,
    "Data Analytics": 0.0,
    "Machine Learning": -4.0,
    "Web Development": 1.5,
    "SQL": 2.0,
}


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_OBSERVED + N_WITHDRAWN + N_MISSING_ASSESSMENT


    engagement = rng.normal(0, 1, n)
    ability = 0.4 * engagement + np.sqrt(1 - 0.4**2) * rng.normal(0, 1, n)

    status = np.array(["observed"] * N_OBSERVED
                      + ["withdrawn"] * N_WITHDRAWN
                      + ["missing_assessment"] * N_MISSING_ASSESSMENT)

    engagement[status == "withdrawn"] -= 1.0
    engagement[status == "missing_assessment"] -= 0.3

    category = rng.choice(CATEGORIES, size=n, p=CATEGORY_PROBS)

    # ---- Week-2 features ---------------------------------------------
    attendance = np.clip(72 + 14 * engagement + rng.normal(0, 8, n), 0, 100)
    quiz = np.clip(60 + 13 * ability + 4 * engagement + rng.normal(0, 9, n), 0, 100)
    assignment = np.clip(70 + 16 * engagement + 4 * ability + rng.normal(0, 10, n), 0, 100)
    minutes = np.exp(5.4 + 0.35 * engagement + rng.normal(0, 0.35, n))
    days_since = np.minimum(rng.poisson(np.exp(0.8 - 0.5 * engagement)), 14)
    practice = rng.poisson(np.exp(1.7 + 0.3 * engagement + 0.1 * ability))

    # ---- Final score (week 6) ----------------------------------------
    # Quiz average may be missing; the score generator uses the true latent
    # quiz level so missingness does not distort the target.
    cat_shift = np.array([CATEGORY_EFFECT[c] for c in category])
    final = (
        8
        + 0.20 * attendance
        + 0.30 * quiz
        + 0.22 * assignment
        + 0.012 * np.minimum(minutes, 600)
        - 0.9 * days_since
        + 0.15 * np.minimum(practice, 25)
        + cat_shift
        + rng.normal(0, 9, n)            # noise -> overlapping outcomes
    )
    final = np.clip(final, 0, 100)

    # ---- Missing quiz averages (kept distinct from genuine zeros) ------
    # Learners with low engagement are more likely to have no quiz completed.
    p_missing = np.clip(0.05 + 0.04 * (-engagement), 0.01, 0.30)
    quiz_missing = rng.random(n) < p_missing
    quiz_out = np.round(quiz, 1).astype(object)
    quiz_out[quiz_missing] = np.nan
    # A handful of genuine zero scores (completed quizzes, scored 0).
    zero_idx = rng.choice(np.where(~quiz_missing)[0], size=10, replace=False)
    for i in zero_idx:
        quiz_out[i] = 0.0

    df = pd.DataFrame({
        "attendance_pct": np.round(attendance, 1),
        "quiz_average": quiz_out,
        "assignment_completion_pct": np.round(assignment, 1),
        "active_learning_minutes": np.round(minutes).astype(int),
        "days_since_last_activity": days_since.astype(int),
        "practice_attempt_count": practice.astype(int),
        "course_category": category,
        "outcome_status": status,
        "final_score": np.round(final, 1),
    })
    df["quiz_average"] = pd.to_numeric(df["quiz_average"])

    # Non-observed outcomes: blank, never silently 0 or below benchmark.
    df.loc[df["outcome_status"] != "observed", "final_score"] = np.nan

    df["below_benchmark"] = np.where(
        df["final_score"].isna(), np.nan, (df["final_score"] < BENCHMARK).astype(float)
    )
    df["below_benchmark"] = df["below_benchmark"].astype("Int64")

    # ---- Anonymous identifiers --------------------------------------
    # One enrollment per learner for the first prototype.
    order = rng.permutation(n)                      # shuffle row order
    df = df.iloc[order].reset_index(drop=True)
    df.insert(0, "enrollment_id", [f"ENR-{i:05d}" for i in range(1, n + 1)])
    group_numbers = rng.permutation(n) + 1
    df.insert(1, "learner_group_id", [f"LG-{g:05d}" for g in group_numbers])
    return df


def validate(df: pd.DataFrame) -> None:
    """Hard checks. The script stops if any of these fail."""
    assert df["enrollment_id"].is_unique, "duplicate enrollment_id"
    assert df["learner_group_id"].is_unique, "duplicate learner_group_id"
    for col in ["attendance_pct", "quiz_average", "assignment_completion_pct", "final_score"]:
        s = df[col].dropna()
        assert s.between(0, 100).all(), f"{col} outside 0-100"
    for col in ["active_learning_minutes", "days_since_last_activity", "practice_attempt_count"]:
        assert (df[col] >= 0).all(), f"{col} has negative values"
    assert set(df["course_category"]) <= set(CATEGORIES), "unknown category"
    observed = df[df["outcome_status"] == "observed"]
    assert observed["final_score"].notna().all(), "observed rows missing final_score"
    assert set(observed["below_benchmark"].unique()) == {0, 1}, "both classes must be present"
    others = df[df["outcome_status"] != "observed"]
    assert others["final_score"].isna().all(), "non-observed rows must have blank final_score"
    assert others["below_benchmark"].isna().all(), "non-observed rows must have blank below_benchmark"


def main() -> None:
    df = generate()
    validate(df)
    df.to_csv(OUTPUT_FILE, index=False)

    obs = df[df["outcome_status"] == "observed"]
    print(f"Saved {len(df)} rows -> {OUTPUT_FILE.name}  (SYNTHETIC DATA)")
    print("\nOutcome status counts:")
    print(df["outcome_status"].value_counts().to_string())
    print(f"\nBelow-benchmark prevalence (observed rows, benchmark={BENCHMARK}): "
          f"{obs['below_benchmark'].mean():.1%}")
    print(f"Final score mean/std (observed): {obs['final_score'].mean():.1f} / {obs['final_score'].std():.1f}")
    print(f"Missing quiz_average: {df['quiz_average'].isna().sum()} rows; "
          f"genuine zeros: {(df['quiz_average'] == 0).sum()} rows")
    print("\nCourse category counts:")
    print(df["course_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
