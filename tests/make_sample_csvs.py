"""
Creates tests/sample_csv/: one valid CSV and one bad CSV per validation case.
Uses fake rows from numpy default_rng(42). It NEVER reads anything in data/.

Usage:  python tests/make_sample_csvs.py [--include-large]
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import samples  # noqa: E402
from validation import REQUIRED_COLUMNS  # noqa: E402

OUT = HERE / "sample_csv"


def fr(n: int, seed: int = 42) -> pd.DataFrame:
    """Fake rows as plain objects, so any cell can later be overwritten with text."""
    return samples.fake_rows(n, seed=seed).astype(object)


def write(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / name, index=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = fr(20, seed=42)

    write("valid_20_rows.csv", base)

    d = base.copy(); d.loc[2, "attendance_pct"] = ""; d.loc[5, "course_category"] = ""
    d.loc[7, "active_learning_minutes"] = ""; write("bad_blank_required.csv", d)

    d = base.copy(); d["attendance_pct"] = d["attendance_pct"].astype(object); d.loc[3, "attendance_pct"] = "abc"
    d.loc[9, "assignment_completion_pct"] = "N/A"; write("bad_non_numeric.csv", d)

    d = base.copy(); d.loc[1, "attendance_pct"] = 120; d.loc[4, "assignment_completion_pct"] = -5
    d.loc[8, "quiz_average"] = 101; write("bad_percent_out_of_range.csv", d)

    d = base.copy(); d.loc[2, "practice_attempt_count"] = -3; d.loc[6, "active_learning_minutes"] = -10
    write("bad_negative_count.csv", d)

    d = base.copy(); d["practice_attempt_count"] = d["practice_attempt_count"].astype(object)
    d.loc[3, "practice_attempt_count"] = 2.5; d.loc[10, "days_since_last_activity"] = 1.2
    write("bad_non_integer_count.csv", d)

    d = base.copy(); d.loc[4, "course_category"] = "Quantum Basket Weaving"; d.loc[11, "course_category"] = "python"
    write("bad_unknown_category.csv", d)

    write("bad_missing_columns.csv", base.drop(columns=["attendance_pct", "course_category"]))
    (OUT / "bad_empty_file.csv").write_bytes(b"")
    (OUT / "bad_header_only.csv").write_text(",".join(REQUIRED_COLUMNS) + "\n", encoding="utf-8")
    base.to_csv(OUT / "bad_wrong_file_type.txt", index=False)
    (OUT / "bad_undecodable.csv").write_bytes(b"\xff\xfe\x00\x01 not a csv \x80\x81\x82")

    d = base.copy(); d.loc[5, "enrollment_id"] = d.loc[1, "enrollment_id"]; d.loc[12, "enrollment_id"] = d.loc[1, "enrollment_id"]
    write("bad_duplicate_ids.csv", d)

    d = base.copy(); d.loc[3, "enrollment_id"] = ""; write("bad_blank_id.csv", d)

    many = fr(10_001, seed=1); write("bad_too_many_rows.csv", many)

    mixed = base.copy(); mixed["attendance_pct"] = mixed["attendance_pct"].astype(object)
    mixed.loc[0, "attendance_pct"] = 250; mixed.loc[3, "course_category"] = "Unknown"
    mixed.loc[6, "enrollment_id"] = mixed.loc[8, "enrollment_id"]; mixed.loc[11, "practice_attempt_count"] = -1
    mixed.loc[14, "assignment_completion_pct"] = ""; write("mixed_several_bad_rows.csv", mixed)

    inj = fr(4, seed=7)
    inj["enrollment_id"] = ["=HYPERLINK(\"http://example.com\")", "+cmd|calc", "@SUM(A1)", "<b>bold</b><script>alert(1)</script>"]
    write("valid_special_characters.csv", inj)

    if "--include-large" in sys.argv:
        big = fr(5, seed=3); big["padding"] = ["x" * 1_500_000] * 5
        write("bad_too_large_over_5MB.csv", big)
    print(f"Sample files written to {OUT}")


if __name__ == "__main__":
    main()
