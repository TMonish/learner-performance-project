"""
Plain-assert tests for the dashboard (no extra packages).
Run from the project root:   python tests/test_validation.py
Needs tests/sample_csv/ (python tests/make_sample_csvs.py) and the saved models. Uses no data/ files.
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import engine  # noqa: E402
import samples  # noqa: E402
import theme  # noqa: E402
import ui  # noqa: E402
import validation as v  # noqa: E402
from pipeline_utils import FEATURES  # noqa: E402

SAMPLES = ROOT / "tests" / "sample_csv"
RESULTS = []


def check(name):
    def wrap(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception as exc:                       # noqa: BLE001
            RESULTS.append((name, False, "".join(traceback.format_exception_only(type(exc), exc)).strip()))
        return fn
    return wrap


def load(name):
    path = SAMPLES / name
    return v.validate_csv_bytes(path.name, path.read_bytes())


def rows_flagged(res):
    return sorted({p["file_row"] for p in res.problems})


@check("valid CSV: all 20 rows accepted, blank quiz stays blank")
def _():
    r = load("valid_20_rows.csv")
    assert r.fatal is None and r.n_read == 20 and r.n_valid == 20 and not r.problems
    assert r.valid["quiz_average"].isna().sum() >= 1
    assert (r.valid["quiz_average"].dropna() > 0).all()


@check("blank required entries are flagged (rows 4, 7, 9)")
def _():
    assert rows_flagged(load("bad_blank_required.csv")) == [4, 7, 9]


@check("non-numeric text is flagged (rows 5, 11)")
def _():
    assert rows_flagged(load("bad_non_numeric.csv")) == [5, 11]


@check("percentages outside 0-100 are flagged (rows 3, 6, 10)")
def _():
    assert rows_flagged(load("bad_percent_out_of_range.csv")) == [3, 6, 10]


@check("negative counts are flagged (rows 4, 8)")
def _():
    assert rows_flagged(load("bad_negative_count.csv")) == [4, 8]


@check("non-integer counts are flagged (rows 5, 12)")
def _():
    assert rows_flagged(load("bad_non_integer_count.csv")) == [5, 12]


@check("unknown course_category flagged, exact match required (rows 6, 13)")
def _():
    assert rows_flagged(load("bad_unknown_category.csv")) == [6, 13]


@check("missing columns are named")
def _():
    r = load("bad_missing_columns.csv")
    assert r.fatal and "attendance_pct" in r.fatal and "course_category" in r.fatal and r.valid is None


@check("empty file, header-only file, wrong type, undecodable file")
def _():
    assert "empty" in load("bad_empty_file.csv").fatal.lower()
    assert "no data rows" in load("bad_header_only.csv").fatal
    assert "Wrong file type" in load("bad_wrong_file_type.txt").fatal
    assert load("bad_undecodable.csv").fatal


@check("duplicate enrollment_id: EVERY duplicate flagged and none predicted (rows 3, 7, 14)")
def _():
    r = load("bad_duplicate_ids.csv")
    assert rows_flagged(r) == [3, 7, 14] and r.n_valid == 17
    assert "DEMO-0002" not in set(r.valid["enrollment_id"])


@check("blank enrollment_id flagged (row 5)")
def _():
    assert rows_flagged(load("bad_blank_id.csv")) == [5]


@check("more than 10,000 rows rejected; file over 5 MB rejected")
def _():
    assert "more than" in load("bad_too_many_rows.csv").fatal
    big = b"enrollment_id\n" + b"x" * (v.MAX_BYTES + 10)
    assert "larger than" in v.validate_csv_bytes("big.csv", big).fatal


@check("mixed file: only valid rows are predicted, original order kept, rejected rows never appear")
def _():
    r = load("mixed_several_bad_rows.csv")
    flagged = set(rows_flagged(r))
    assert flagged and r.n_valid + r.n_rejected == r.n_read
    assert list(r.valid["file_row"]) == sorted(r.valid["file_row"])
    assert not set(r.valid["file_row"]) & flagged
    cfg = engine.load_config(); models = engine.load_models(cfg)
    export = engine.build_export(r.valid, engine.predict(models, cfg, engine.make_frame(r.valid[FEATURES])), cfg)
    assert len(export) == r.n_valid
    assert list(export["enrollment_id"]) == list(r.valid["enrollment_id"])


@check("single-row prediction equals batch prediction for the same rows")
def _():
    r = load("valid_20_rows.csv")
    cfg = engine.load_config(); models = engine.load_models(cfg)
    batch = engine.predict(models, cfg, engine.make_frame(r.valid[FEATURES]))
    for i in range(len(r.valid)):
        raw = {c: ("" if pd.isna(r.valid.at[i, c]) else str(r.valid.at[i, c])) for c in FEATURES}
        clean, problems = v.validate_row(raw)
        assert clean is not None, problems
        one = engine.predict(models, cfg, engine.make_frame([clean])).iloc[0]
        assert abs(one["predicted_final_score"] - batch.at[i, "predicted_final_score"]) < 1e-9
        assert abs(one["support_score"] - batch.at[i, "support_score"]) < 1e-9
        assert bool(one["flagged"]) == bool(batch.at[i, "flagged"])


@check("blank quiz is NaN (not zero) and gives a different result from a quiz score of 0")
def _():
    base = {"attendance_pct": "70", "assignment_completion_pct": "70", "active_learning_minutes": "200",
            "days_since_last_activity": "2", "practice_attempt_count": "5", "course_category": "SQL"}
    blank, _p = v.validate_row({**base, "quiz_average": ""})
    zero, _p = v.validate_row({**base, "quiz_average": "0"})
    assert np.isnan(blank["quiz_average"]) and zero["quiz_average"] == 0
    cfg = engine.load_config(); models = engine.load_models(cfg)
    a = engine.predict(models, cfg, engine.make_frame([blank])).iloc[0]["predicted_final_score"]
    b = engine.predict(models, cfg, engine.make_frame([zero])).iloc[0]["predicted_final_score"]
    assert a != b


@check("number parsing: spaces ok; NaN, inf, underscores, non-ASCII digits and bounds rejected")
def _():
    base = {"quiz_average": "50", "attendance_pct": " 55 ", "assignment_completion_pct": "1e2",
            "active_learning_minutes": "200", "days_since_last_activity": "14",
            "practice_attempt_count": "5", "course_category": " SQL "}
    clean, p = v.validate_row(base)
    assert clean is not None and clean["attendance_pct"] == 55 and clean["course_category"] == "SQL", p
    for col, bad in [("attendance_pct", "nan"), ("attendance_pct", "inf"), ("active_learning_minutes", "1_000"),
                     ("practice_attempt_count", "\u0661\u0662"), ("days_since_last_activity", "15"),
                     ("active_learning_minutes", "20161")]:
        c, problems = v.validate_row({**base, col: bad})
        assert c is None and any(pc == col for pc, _ in problems), (col, bad)


@check("special characters: export is formula-safe and HTML is escaped")
def _():
    r = load("valid_special_characters.csv")
    assert r.n_valid == 4
    cfg = engine.load_config(); models = engine.load_models(cfg)
    export = engine.build_export(r.valid, engine.predict(models, cfg, engine.make_frame(r.valid[FEATURES])), cfg)
    assert all(str(x).startswith("'") for x in export["enrollment_id"][:3])
    table = ui.html_table(export.to_dict("records"), list(export.columns))
    assert "<script" not in table and "&lt;script" in table


@check("export has the right columns and never labels a decision score as a probability")
def _():
    cfg = engine.load_config(); models = engine.load_models(cfg)
    r = load("valid_20_rows.csv")
    preds = engine.predict(models, cfg, engine.make_frame(r.valid[FEATURES]))
    out = engine.build_export(r.valid, preds, cfg)
    assert list(out.columns[:3]) == ["enrollment_id", "predicted_final_score", "predicted_support_class"]
    assert ("below_benchmark_probability" in out) != ("decision_score" in out)
    svm_cfg = dict(cfg, threshold_type="decision_score", probability_available=False)
    assert "decision_score" in engine.build_export(r.valid, preds, svm_cfg)
    assert "below_benchmark_probability" not in engine.build_export(r.valid, preds, svm_cfg)


@check("sample template downloads as a valid CSV")
def _():
    r = v.validate_csv_bytes("learner_template.csv", samples.template_csv())
    assert r.fatal is None and r.n_valid == 5


# ------------------------------- app smoke tests (Streamlit AppTest) -------------------------------
def _texts(at):
    return " ".join(m.value for m in at.markdown)


@check("app starts without errors in light mode and shows both required notices")
def _():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    t = _texts(at)
    assert "Synthetic data" in t and "must not decide admission, certification, grades or access" in t
    assert len(at.tabs) == 4
    assert theme.LIGHT["bg"] in t and "Estimated final score" in t
    assert ("May benefit from extra support" in t) or ("On track" in t)


@check("dark mode switches the palette; presets change the result; no errors")
def _():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=90).run()
    at.toggle(key="dark_mode").set_value(True).run()
    assert not at.exception, at.exception
    assert theme.DARK["bg"] in _texts(at)
    at.button(key="preset_nudge").click().run()
    assert not at.exception, at.exception
    t = _texts(at)
    assert "May benefit from extra support" in t
    assert at.session_state["no_quiz"] is True
    at.button(key="preset_engaged").click().run()
    assert not at.exception and "On track" in _texts(at)


if __name__ == "__main__":
    failed = 0
    for name, ok, msg in RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n        {msg}"))
        failed += 0 if ok else 1
    print(f"\n{len(RESULTS) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
