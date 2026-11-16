"""
Early Learning Support - local Streamlit dashboard.
Project : Learner Performance Prediction & Early Learning Support Dashboard (Career Solutions)

DATA IS SYNTHETIC. Predictions support human review only.

Run from the project root (virtual environment active):
    streamlit run app/app.py

This app only LOADS the saved, frozen models. It never trains anything, never reads the data/ folder,
never saves uploaded files and sends no data anywhere.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401  (puts src/ on sys.path)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import engine  # noqa: E402
import samples  # noqa: E402
import theme  # noqa: E402
import ui  # noqa: E402
import validation  # noqa: E402
from pipeline_utils import FEATURES  # noqa: E402

st.set_page_config(page_title="Early Learning Support", layout="wide")


@st.cache_resource(show_spinner=False)
def get_models():
    """Load the frozen config and the saved pipelines once per server session."""
    config = engine.load_config()
    frozen = engine.load_frozen()
    return config, engine.load_models(config), frozen, engine.integrity_warnings(frozen)


# ----------------------------------------------------------------------------------------------
# Theme (light / dark) - the toggle's state is read BEFORE the CSS is injected
# ----------------------------------------------------------------------------------------------
dark = st.session_state.setdefault("dark_mode", False)
st.markdown(theme.build_css(dark), unsafe_allow_html=True)

try:
    config, models, frozen, warnings_list = get_models()
except FileNotFoundError as exc:
    st.markdown(ui.note("error", "Model settings not found.", str(exc)), unsafe_allow_html=True)
    st.stop()

head_left, head_right = st.columns([5, 1.3], vertical_alignment="center")
with head_left:
    st.markdown(ui.header(), unsafe_allow_html=True)
with head_right:
    st.toggle("Dark mode", key="dark_mode")
st.markdown(ui.rule(), unsafe_allow_html=True)

st.markdown(ui.hero(), unsafe_allow_html=True)
st.markdown(ui.banners(), unsafe_allow_html=True)
if warnings_list:
    st.markdown(ui.note("warn", "Model files differ from the frozen versions.", "; ".join(warnings_list)),
                unsafe_allow_html=True)

BENCHMARK = config["benchmark"]
PROB_OK = config["threshold_type"] == "probability" and config["probability_available"]
CLF_NAME = config["selected_classifier_name"]
CLF_METRICS = config["selected_classifier_test_metrics"]

# ----------------------------------------------------------------------------------------------
# Single-learner inputs (shared by tabs 1 and 2). Presets are fictional examples.
# ----------------------------------------------------------------------------------------------
DEFAULTS = {"category": "Data Analytics", "attendance": 72.0, "assignment": 70.0, "no_quiz": False,
            "quiz": 60.0, "minutes": 245, "days": 2, "practice": 6}
PRESETS = {
    "engaged": {"category": "Python", "attendance": 94.0, "assignment": 96.0, "no_quiz": False, "quiz": 84.0,
                "minutes": 420, "days": 0, "practice": 14},
    "typical": dict(DEFAULTS),
    "nudge": {"category": "SQL", "attendance": 41.0, "assignment": 35.0, "no_quiz": True, "quiz": 55.0,
              "minutes": 90, "days": 9, "practice": 1},
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


tab_score, tab_support, tab_compare, tab_batch = st.tabs(
    ["Score prediction", "Learning support", "Model comparison", "Batch prediction"])

# ============================== Tab 1: score prediction ==============================
with tab_score:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.container(key="card_inputs"):
            st.markdown(ui.card_title("Week-2 activity", "Enter what the platform recorded in the first two weeks."),
                        unsafe_allow_html=True)
            with st.container(key="presets_row"):
                p1, p2, p3 = st.columns(3)
                p1.button("Highly engaged", key="preset_engaged", on_click=apply_preset, args=("engaged",), use_container_width=True)
                p2.button("Typical", key="preset_typical", on_click=apply_preset, args=("typical",), use_container_width=True)
                p3.button("Needs a nudge", key="preset_nudge", on_click=apply_preset, args=("nudge",), use_container_width=True)
            st.selectbox("Course category", validation.CATEGORIES, key="category")
            st.slider("Attendance (%)", 0.0, 100.0, step=0.5, key="attendance")
            st.slider("Assignment completion (%)", 0.0, 100.0, step=0.5, key="assignment")
            no_quiz = st.checkbox("No quiz completed yet", key="no_quiz")
            st.slider("Quiz average (%)", 0.0, 100.0, step=0.5, key="quiz", disabled=no_quiz)
            c1, c2 = st.columns(2)
            c1.number_input("Active learning minutes", 0, validation.COUNT_MAX["active_learning_minutes"], step=10, key="minutes")
            c2.number_input("Practice attempts", 0, 500, step=1, key="practice")
            st.slider("Days since last activity", 0, validation.COUNT_MAX["days_since_last_activity"], key="days")

    # Same validation and the same prediction code as the batch tab.
    raw = {"quiz_average": "" if st.session_state["no_quiz"] else str(st.session_state["quiz"]),
           "attendance_pct": str(st.session_state["attendance"]),
           "assignment_completion_pct": str(st.session_state["assignment"]),
           "active_learning_minutes": str(st.session_state["minutes"]),
           "days_since_last_activity": str(st.session_state["days"]),
           "practice_attempt_count": str(st.session_state["practice"]),
           "course_category": st.session_state["category"]}
    clean, problems = validation.validate_row(raw)
    result = None
    if clean is not None:
        result = engine.predict(models, config, engine.make_frame([clean])).iloc[0]

    with right:
        if result is None:
            st.markdown(ui.note("error", "Please check the inputs.", "; ".join(f"{c}: {r}" for c, r in problems)),
                        unsafe_allow_html=True)
        else:
            st.markdown(ui.score_card(result["predicted_final_score"], config["regression_test_mae"], BENCHMARK),
                        unsafe_allow_html=True)

# ============================== Tab 2: learning support ==============================
with tab_support:
    if result is None:
        st.markdown(ui.note("error", "Please fix the inputs on the Score prediction tab first."), unsafe_allow_html=True)
    else:
        st.markdown(ui.support_card(
            flagged=bool(result["flagged"]), label=result["predicted_support_class"], clf_name=CLF_NAME,
            score=result["support_score"], score_type=config["threshold_type"], show_probability=PROB_OK,
            threshold=config["threshold"], benchmark=BENCHMARK,
            recall=CLF_METRICS["Recall"], precision=CLF_METRICS["Precision"]), unsafe_allow_html=True)
        with st.expander("How is this flag decided?"):
            how = (f"The selected model is **{CLF_NAME}**. It was chosen and its threshold frozen using validation "
                   f"data only, then checked once on held-out data. A learner is flagged when the "
                   f"{'estimated likelihood' if PROB_OK else 'decision score'} is at or above "
                   f"{config['threshold']:.3f}. Model weights describe patterns in this synthetic data; they do not "
                   f"show what causes learning outcomes.")
            st.markdown(how)

# ============================== Tab 3: model comparison ==============================
with tab_compare:
    reg_mae, base_mae = config["regression_test_mae"], config["regression_baseline_test_mae"]
    st.markdown(ui.kpis([
        ("Score model error (MAE)", f"{reg_mae:g}", f"score points, vs {base_mae:g} for the training-mean baseline"),
        ("Score model R\u00b2", f"{config['regression_test_r2']:.2f}", "share of score variation explained, held-out data"),
        ("Selected classifier", CLF_NAME, f"flags learners at {config['threshold']:.3f}"),
        ("Data origin", "Synthetic", "generated records, not real learners"),
    ]), unsafe_allow_html=True)
    st.write("")

    metrics = config["classifier_test_metrics"]
    rows = []
    for name, m in metrics.items():
        rows.append({"Model": name + (" (selected)" if name == CLF_NAME else ""), "Score type": m["score_type"].replace("_", " "),
                     "ROC-AUC": f"{m['ROC-AUC']:.3f}", "PR-AUC": f"{m['PR-AUC']:.3f}", "Precision": f"{m['Precision']:.3f}",
                     "Recall": f"{m['Recall']:.3f}", "F1": f"{m['F1']:.3f}", "Flagged": f"{m['flagged_share'] * 100:.1f}%",
                     "_sel": name == CLF_NAME})
    st.markdown(ui.card_title("Classifiers on held-out data", "Each model uses the threshold frozen on validation data."),
                unsafe_allow_html=True)
    cols = ["Model", "Score type", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Flagged"]
    st.markdown(ui.html_table(rows, cols, selected_col="_sel", selected_value=True,
                              numeric_cols=("ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Flagged")), unsafe_allow_html=True)
    st.write("")
    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown(ui.bars("Ranking quality (PR-AUC)", [(n, m["PR-AUC"], n == CLF_NAME) for n, m in metrics.items()],
                            baseline=config["test_baseline"]["prevalence"], baseline_label="baseline = share of learners below benchmark"),
                    unsafe_allow_html=True)
    with cb:
        st.markdown(ui.bars("Learners caught (recall)", [(n, m["Recall"], n == CLF_NAME) for n, m in metrics.items()]),
                    unsafe_allow_html=True)
    st.write("")
    st.markdown(ui.note("info", "Reading these numbers.",
                        "The SVM produces a decision score, not a probability. Results come from one synthetic cohort, so "
                        "they show the workflow works rather than how it would perform on real learners."), unsafe_allow_html=True)
    if frozen and frozen.get("assumptions_to_confirm_with_client"):
        a = frozen["assumptions_to_confirm_with_client"]
        st.write("")
        st.markdown(ui.note("warn", "Assumptions awaiting client confirmation.",
                            f"Support capacity: at most {a['capacity_max_flagged_share'] * 100:.0f}% of learners flagged. "
                            f"Recall weighted {a['beta_recall_weight']:g}x precision when setting the threshold."), unsafe_allow_html=True)

# ============================== Tab 4: batch prediction ==============================
with tab_batch:
    with st.container(key="card_upload"):
        st.markdown(ui.card_title("Upload a CSV", f"Columns: enrollment_id, {', '.join(FEATURES)}. "
                                  f"Up to {validation.MAX_ROWS:,} rows and 5 MB. Files are processed in memory and never saved."),
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV file", accept_multiple_files=False, key="batch_file", label_visibility="collapsed")
        st.download_button("Download a sample CSV template", samples.template_csv(), file_name="learner_template.csv",
                           mime="text/csv", key="template_dl")

    if uploaded is not None:
        res = validation.validate_csv_bytes(uploaded.name, uploaded.getvalue())
        st.write("")
        if res.fatal:
            st.markdown(ui.note("error", "We couldn't use this file.", res.fatal), unsafe_allow_html=True)
        else:
            st.markdown(ui.count_kpis([("Rows read", res.n_read, "data rows in the file"),
                                       ("Ready to predict", res.n_valid, "passed every check"),
                                       ("Needs attention", res.n_rejected, "rows not predicted")]), unsafe_allow_html=True)
            st.write("")
            if res.problems:
                shown = res.problems[:300]
                st.markdown(ui.card_title("Rows that need attention",
                                          "These rows were not predicted. Row numbers match the file (the header is row 1)."),
                            unsafe_allow_html=True)
                st.markdown(ui.html_table(shown, ["file_row", "enrollment_id", "column", "reason"], numeric_cols=("file_row",)),
                            unsafe_allow_html=True)
                if len(res.problems) > len(shown):
                    st.caption(f"Showing the first {len(shown)} of {len(res.problems)} issues. Download the full list below.")
                prob_df = pd.DataFrame(res.problems)
                for col in ("enrollment_id", "reason"):
                    prob_df[col] = prob_df[col].map(validation.csv_safe)
                st.download_button("Download the issue list (CSV)", prob_df.to_csv(index=False).encode("utf-8"),
                                   file_name="rows_needing_attention.csv", mime="text/csv", key="problems_dl")
                st.write("")
            if res.n_valid:
                preds = engine.predict(models, config, engine.make_frame(res.valid[FEATURES]))
                export = engine.build_export(res.valid, preds, config)
                score_col = "below_benchmark_probability" if "below_benchmark_probability" in export else "decision_score"
                n_flag = int(preds["flagged"].sum())
                st.markdown(ui.card_title("Predictions",
                                          f"{n_flag} of {res.n_valid} learners are flagged as a priority for human review."
                                          + ("" if PROB_OK else " The score column is a decision score, not a probability.")),
                            unsafe_allow_html=True)
                preview = export.head(100).to_dict("records")
                st.markdown(ui.html_table(preview, list(export.columns),
                                          numeric_cols=("predicted_final_score", score_col)), unsafe_allow_html=True)
                if len(export) > 100:
                    st.caption("Showing the first 100 rows. The download contains all of them.")
                st.write("")
                st.download_button("Download predictions (CSV)", export.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv", mime="text/csv", key="pred_dl")

st.markdown('<div class="rule" style="margin-top:3.2rem"></div><p class="fine" style="margin-top:.9rem">Prototype for Career Solutions. '
            'Synthetic data. Anonymous IDs only.</p>', unsafe_allow_html=True)
