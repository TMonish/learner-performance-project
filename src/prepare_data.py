"""
Day 3 - Data validation, missing-outcome handling and train/validation/test split.
Project : Learner Performance Prediction & Early Learning Support Dashboard
Client  : Career Solutions (EduTech)

DATA IS SYNTHETIC (see docs/source_notes.md).

What this script does
  1. Validates the raw CSV and writes reports/quality_report.md
  2. Separates withdrawn / missing-assessment rows (never labelled 0 or below benchmark)
  3. Recomputes below_benchmark from final_score using the configurable BENCHMARK
  4. Creates ONE reproducible 60/20/20 split, stratified by below_benchmark
  5. Writes data/processed/*.csv and reports/split_summary.md

What it deliberately does NOT do
  - No imputation, scaling or encoding. Those are fitted on training data only,
    inside scikit-learn Pipelines (Days 4-6), to prevent leakage.

Usage:  python src/prepare_data.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Configuration (change here only)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "synthetic_learner_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

SEED = 42
BENCHMARK = 50                      # must match the course policy / generator
SPLIT = {"train": 0.60, "validation": 0.20, "test": 0.20}

ID_COLS = ["enrollment_id", "learner_group_id"]
PCT_COLS = ["attendance_pct", "quiz_average", "assignment_completion_pct"]
COUNT_COLS = ["active_learning_minutes", "days_since_last_activity", "practice_attempt_count"]
CAT_COLS = ["course_category"]
FEATURES = PCT_COLS + COUNT_COLS + CAT_COLS          # the ONLY model inputs
TARGET_SCORE = "final_score"
TARGET_CLASS = "below_benchmark"
STATUS_COL = "outcome_status"
VALID_STATUS = {"observed", "withdrawn", "missing_assessment"}
KNOWN_CATEGORIES = ["Python", "Data Analytics", "Machine Learning", "Web Development", "SQL"]
REQUIRED_COLUMNS = ID_COLS + FEATURES + [STATUS_COL, TARGET_SCORE, TARGET_CLASS]


def md_table(df: pd.DataFrame) -> str:
    """Small markdown table writer (avoids an extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def run_checks(df: pd.DataFrame) -> list:
    """Return a list of (check, PASS/FAIL, detail)."""
    results = []

    def add(name, ok, detail=""):
        results.append((name, "PASS" if ok else "FAIL", detail))

    add("File is not empty", len(df) > 0, f"{len(df)} rows")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    add("All required columns present", not missing_cols, f"missing: {missing_cols}" if missing_cols else "")
    if missing_cols:                       # cannot continue safely
        return results

    add("Feature list has no ID or outcome fields",
        not set(FEATURES) & set(ID_COLS + [TARGET_SCORE, TARGET_CLASS, STATUS_COL]),
        f"features: {FEATURES}")
    add("enrollment_id is unique", df["enrollment_id"].is_unique,
        f"{df['enrollment_id'].duplicated().sum()} duplicates")
    add("learner_group_id is unique (one enrollment per learner)", df["learner_group_id"].is_unique,
        f"{df['learner_group_id'].duplicated().sum()} duplicates")

    for col in PCT_COLS + [TARGET_SCORE]:
        s = df[col].dropna()
        bad = int((~s.between(0, 100)).sum())
        add(f"{col} within 0-100", bad == 0, f"{bad} out-of-range values")
    for col in COUNT_COLS:
        s = df[col].dropna()
        neg = int((s < 0).sum())
        nonint = int((s != s.round()).sum())
        add(f"{col} non-negative whole numbers", neg == 0 and nonint == 0,
            f"{neg} negative, {nonint} non-integer")

    unknown = sorted(set(df["course_category"].dropna()) - set(KNOWN_CATEGORIES))
    add("course_category values are known", not unknown, f"unknown: {unknown}" if unknown else "")
    add("course_category has no blanks", df["course_category"].notna().all(),
        f"{int(df['course_category'].isna().sum())} blanks")
    other_missing = [c for c in FEATURES if c not in ("quiz_average",) and df[c].isna().any()]
    add("No unexpected missing feature values (only quiz_average may be blank)",
        not other_missing, f"columns with blanks: {other_missing}" if other_missing else "")

    bad_status = sorted(set(df[STATUS_COL].dropna()) - VALID_STATUS)
    add("outcome_status values are valid", not bad_status and df[STATUS_COL].notna().all(),
        f"invalid: {bad_status}" if bad_status else "")

    obs = df[df[STATUS_COL] == "observed"]
    non = df[df[STATUS_COL] != "observed"]
    add("Observed rows all have final_score", obs[TARGET_SCORE].notna().all(),
        f"{int(obs[TARGET_SCORE].isna().sum())} blank")
    add("Non-observed rows have blank final_score and below_benchmark",
        non[TARGET_SCORE].isna().all() and non[TARGET_CLASS].isna().all(), "")

    recomputed = (obs[TARGET_SCORE] < BENCHMARK).astype(int)
    mismatch = int((recomputed != obs[TARGET_CLASS].astype(int)).sum())
    add(f"Stored below_benchmark matches final_score < {BENCHMARK}", mismatch == 0,
        f"{mismatch} mismatches")
    add("Both classes present among observed rows", recomputed.nunique() == 2,
        f"prevalence {recomputed.mean():.1%}")
    return results


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)
    checks = run_checks(df)
    failed = [c for c in checks if c[1] == "FAIL"]

    # ---------------- quality report ----------------
    report = ["# Data Quality Report", "",
              "**Data origin: SYNTHETIC.** Results on this data demonstrate the implementation, "
              "not effectiveness on real learners.", "",
              f"Source file: `{RAW_FILE.name}` | Rows: {len(df)} | Benchmark: {BENCHMARK}", "",
              "## Validation checks",
              md_table(pd.DataFrame(checks, columns=["Check", "Result", "Detail"])), ""]
    if not failed:
        miss = df[FEATURES + [TARGET_SCORE]].isna().sum().reset_index()
        miss.columns = ["Column", "Blank values"]
        miss["Percent"] = (miss["Blank values"] / len(df) * 100).round(1).astype(str) + "%"
        report += ["## Missing values by column", md_table(miss), "",
                   "## Outcome status (withdrawals and missing assessments are reported separately)",
                   md_table(df[STATUS_COL].value_counts().rename_axis("outcome_status")
                            .reset_index(name="rows")), "",
                   "## Quiz average: blank vs genuine zero",
                   f"- Blank (no quiz completed before cutoff): {int(df['quiz_average'].isna().sum())}",
                   f"- Genuine 0.0 scores: {int((df['quiz_average'] == 0).sum())}",
                   "- Blanks are kept as blanks. They are imputed later inside the training "
                   "pipeline, fitted on training data only.", ""]
    (REPORTS_DIR / "quality_report.md").write_text("\n".join(report), encoding="utf-8")

    if failed:
        print("VALIDATION FAILED - see reports/quality_report.md")
        for name, _, detail in failed:
            print(f"  FAIL: {name} {detail}")
        raise SystemExit(1)

    # ---------------- missing outcomes ----------------
    non_obs = df[df[STATUS_COL] != "observed"].copy()
    obs = df[df[STATUS_COL] == "observed"].copy()
    obs[TARGET_CLASS] = (obs[TARGET_SCORE] < BENCHMARK).astype(int)   # recomputed from config
    non_obs.to_csv(PROCESSED_DIR / "excluded_non_observed.csv", index=False)

    # ---------------- 60/20/20 split ----------------
    train, rest = train_test_split(
        obs, test_size=SPLIT["validation"] + SPLIT["test"],
        stratify=obs[TARGET_CLASS], random_state=SEED)
    val, test = train_test_split(
        rest, test_size=SPLIT["test"] / (SPLIT["validation"] + SPLIT["test"]),
        stratify=rest[TARGET_CLASS], random_state=SEED)
    parts = {"train": train, "validation": val, "test": test}

    # safety assertions
    assert len(train) + len(val) + len(test) == len(obs), "rows lost in split"
    ids = [set(p["enrollment_id"]) for p in parts.values()]
    groups = [set(p["learner_group_id"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "enrollment overlap"
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), \
        "learner appears in two partitions"

    assignments = []
    for name, part in parts.items():
        part.sort_values("enrollment_id").to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        assignments.append(part[["enrollment_id", "learner_group_id"]].assign(split=name))
    pd.concat(assignments).sort_values("enrollment_id").to_csv(
        PROCESSED_DIR / "split_assignments.csv", index=False)

    # ---------------- split summary ----------------
    rows = []
    for name, part in parts.items():
        rows.append({
            "Split": name,
            "Rows": len(part),
            "Share": f"{len(part) / len(obs):.1%}",
            "Below benchmark": int(part[TARGET_CLASS].sum()),
            "Prevalence": f"{part[TARGET_CLASS].mean():.1%}",
            "Mean final score": round(part[TARGET_SCORE].mean(), 1),
        })
    summary = ["# Split Summary", "",
               "**Data origin: SYNTHETIC.**", "",
               f"- Random seed: {SEED}",
               f"- Rows used (observed final assessment): {len(obs)}",
               f"- Excluded and reported separately: {len(non_obs)} "
               f"({int((non_obs[STATUS_COL] == 'withdrawn').sum())} withdrawn, "
               f"{int((non_obs[STATUS_COL] == 'missing_assessment').sum())} missing assessment)",
               f"- Stratified by `{TARGET_CLASS}`; one enrollment per learner, so no learner "
               "appears in two partitions (asserted)",
               "- The same partitions will be used by all four models",
               "- The test set is not used for tuning, threshold choice or model selection", "",
               md_table(pd.DataFrame(rows)), "",
               "The model applies to the observed-assessment population only. "
               "Withdrawn and missing-assessment learners are not scored as zero or below benchmark.", ""]
    (REPORTS_DIR / "split_summary.md").write_text("\n".join(summary), encoding="utf-8")

    # ---------------- console summary ----------------
    print(f"All {len(checks)} validation checks passed.")
    print(f"Observed rows: {len(obs)} | excluded (withdrawn/missing): {len(non_obs)}")
    print(md_table(pd.DataFrame(rows)))
    print("Saved: data/processed/*.csv, reports/quality_report.md, reports/split_summary.md")


if __name__ == "__main__":
    main()
