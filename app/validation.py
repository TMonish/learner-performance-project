"""
Input validation for the dashboard (single entries and batch CSV files).
No Streamlit imports here, so it can be tested on its own (see tests/test_validation.py).

Rules: bad input is FLAGGED or REJECTED with a clear reason. It is never predicted, never
silently dropped, and never allowed to crash the app. DATA IS SYNTHETIC.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import paths  # noqa: F401  (puts src/ on sys.path)
from pipeline_utils import FEATURES

CATEGORIES = ["Python", "Data Analytics", "Machine Learning", "Web Development", "SQL"]
PCT_COLS = ["quiz_average", "attendance_pct", "assignment_completion_pct"]
COUNT_COLS = ["active_learning_minutes", "days_since_last_activity", "practice_attempt_count"]
COUNT_MAX = {"active_learning_minutes": 20160,      # 14 days x 24 h x 60 min
             "days_since_last_activity": 14,        # the cutoff is the end of week 2
             "practice_attempt_count": None}
BLANK_ALLOWED = {"quiz_average"}                    # blank = no quiz completed (NOT zero)
REQUIRED_COLUMNS = ["enrollment_id"] + FEATURES

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10_000
MAX_ID_LEN = 64
_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", re.ASCII)


def _short(value, n: int = 40) -> str:
    s = str(value)
    return s if len(s) <= n else s[: n - 1] + "..."


def _fmt(x: float) -> str:
    return f"{x:g}"


def _parse(text):
    """Return (number, None) or (None, 'blank' | 'invalid')."""
    s = str(text).strip()
    if s == "":
        return None, "blank"
    if not _NUMBER.fullmatch(s):
        return None, "invalid"
    x = float(s)
    if not math.isfinite(x):
        return None, "invalid"
    return x, None


def validate_row(raw: dict):
    """Validate one learner's 7 feature values (given as text or numbers).
    Returns (clean_dict_or_None, [(column, reason), ...])."""
    problems, clean = [], {}

    for col in PCT_COLS:
        x, err = _parse(raw.get(col, ""))
        if err == "blank":
            if col in BLANK_ALLOWED:
                clean[col] = np.nan
            else:
                problems.append((col, "required value is blank"))
        elif err:
            problems.append((col, f"'{_short(raw.get(col))}' is not a valid number"))
        elif not 0 <= x <= 100:
            problems.append((col, f"{_fmt(x)} is outside the allowed range 0-100"))
        else:
            clean[col] = x

    for col in COUNT_COLS:
        x, err = _parse(raw.get(col, ""))
        if err == "blank":
            problems.append((col, "required value is blank"))
        elif err:
            problems.append((col, f"'{_short(raw.get(col))}' is not a valid number"))
        elif x < 0:
            problems.append((col, f"{_fmt(x)} is negative; counts cannot be negative"))
        elif x != int(x):
            problems.append((col, f"{_fmt(x)} is not a whole number"))
        elif COUNT_MAX[col] is not None and x > COUNT_MAX[col]:
            problems.append((col, f"{_fmt(x)} is above the maximum of {COUNT_MAX[col]}"))
        else:
            clean[col] = int(x)

    cat = str(raw.get("course_category", "")).strip()
    if cat == "":
        problems.append(("course_category", "required value is blank"))
    elif cat not in CATEGORIES:
        problems.append(("course_category",
                         f"unknown category '{_short(cat)}' (allowed: {', '.join(CATEGORIES)})"))
    else:
        clean["course_category"] = cat

    return (None if problems else clean), problems


@dataclass
class BatchResult:
    fatal: str | None = None                  # set when the whole file is unusable
    n_read: int = 0
    valid: pd.DataFrame | None = None         # columns: file_row, enrollment_id, + FEATURES
    problems: list = field(default_factory=list)   # dicts: file_row, enrollment_id, column, reason

    @property
    def n_valid(self) -> int:
        return 0 if self.valid is None else len(self.valid)

    @property
    def n_rejected(self) -> int:
        return len({p["file_row"] for p in self.problems})


def validate_csv_bytes(filename: str, data: bytes) -> BatchResult:
    """Validate an uploaded CSV. Row numbers follow the file: the header is row 1."""
    if not str(filename).lower().endswith(".csv"):
        return BatchResult(fatal="Wrong file type. Please upload a .csv file.")
    if len(data) == 0:
        return BatchResult(fatal="The file is empty.")
    if len(data) > MAX_BYTES:
        return BatchResult(fatal=f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return BatchResult(fatal="The file could not be read as UTF-8 text. "
                                 "Please save it as a CSV (UTF-8) and try again.")
    if "\x00" in text:
        return BatchResult(fatal="The file does not look like a text CSV.")
    if text.strip() == "":
        return BatchResult(fatal="The file is empty.")

    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, skip_blank_lines=False)
    except pd.errors.EmptyDataError:
        return BatchResult(fatal="The file is empty.")
    except (pd.errors.ParserError, ValueError):
        return BatchResult(fatal="The file could not be parsed as a CSV. Check that every row has the "
                                 "same number of columns.")

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return BatchResult(fatal="Missing required column(s): " + ", ".join(missing))
    if len(df) == 0:
        return BatchResult(fatal="The file has a header but no data rows.")
    if len(df) > MAX_ROWS:
        return BatchResult(fatal=f"The file has more than {MAX_ROWS:,} rows.")

    ids = df["enrollment_id"].astype(str).str.strip()
    problems, good_rows = [], []

    for i in range(len(df)):
        file_row = i + 2
        eid = ids.iloc[i]
        row_problems = []
        if eid == "":
            row_problems.append(("enrollment_id", "required value is blank"))
        elif len(eid) > MAX_ID_LEN:
            row_problems.append(("enrollment_id", f"longer than {MAX_ID_LEN} characters"))
        clean, feat_problems = validate_row({c: df.at[i, c] for c in FEATURES})
        row_problems += feat_problems
        for col, reason in row_problems:
            problems.append({"file_row": file_row, "enrollment_id": _short(eid),
                             "column": col, "reason": reason})
        if not row_problems:
            good_rows.append({"file_row": file_row, "enrollment_id": eid, **clean})

    valid = pd.DataFrame(good_rows, columns=["file_row", "enrollment_id"] + FEATURES)

    # Flag EVERY row that shares an enrollment_id (none of them is predicted).
    nonblank = ids[ids != ""]
    dup_ids = set(nonblank[nonblank.duplicated(keep=False)])
    if dup_ids:
        where = {d: [i + 2 for i in range(len(df)) if ids.iloc[i] == d] for d in dup_ids}
        for d, rows in where.items():
            for r in rows:
                others = ", ".join(str(x) for x in rows if x != r)
                problems.append({"file_row": r, "enrollment_id": _short(d), "column": "enrollment_id",
                                 "reason": f"duplicate enrollment_id (also on row {others})"})
        valid = valid[~valid["enrollment_id"].isin(dup_ids)]

    problems.sort(key=lambda p: (p["file_row"], p["column"]))
    return BatchResult(n_read=len(df), valid=valid.reset_index(drop=True), problems=problems)


def csv_safe(value):
    """Protect exported text from spreadsheet formula injection."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value
