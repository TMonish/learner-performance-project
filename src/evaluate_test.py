"""
Day 7, phase B - One-time evaluation of all FROZEN models on the held-out TEST split.
Project : Learner Performance Prediction & Early Learning Support Dashboard

DATA IS SYNTHETIC. Results demonstrate the implementation, not effectiveness on real learners.

Safety rules enforced by this script
  - Nothing is fitted or tuned here. Saved pipelines and frozen thresholds are only APPLIED.
  - The fingerprints (SHA-256) of the saved pipelines and of test.csv must match the values
    recorded when the settings were frozen; otherwise the script stops.
  - After a real run it writes models/test_evaluation_lock.json. A second real run is refused,
    because reusing the test set after seeing results invalidates it.
  - `--rehearsal` runs the identical code on the VALIDATION split (outputs go to reports/rehearsal/)
    so the reporting code can be checked without touching the test set.

Usage:
    python src/evaluate_test.py --rehearsal     (optional dry run on validation data)
    python src/evaluate_test.py                 (the real, one-time test evaluation)
"""

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             brier_score_loss, mean_absolute_error, r2_score, root_mean_squared_error)

from eval_utils import classification_metrics, md_table
from final_utils import (FINAL_CONFIG_FILE, FROZEN_FILE, LOCK_FILE, MODELS, REGRESSION_FILE,
                         get_scores, load_models, sha256_file)
from pipeline_utils import (FIG_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_split, predict_score)

MIN_GROUP_ROWS = 30        # smaller course-category groups are flagged as unreliable
MIN_GROUP_POSITIVES = 10   # fewer below-benchmark learners than this -> recall is unreliable
NUMERIC_PROFILE = ["attendance_pct", "quiz_average", "assignment_completion_pct",
                   "active_learning_minutes", "days_since_last_activity", "practice_attempt_count"]


def reg_metrics(y, pred) -> dict:
    return {"MAE": round(float(mean_absolute_error(y, pred)), 2),
            "RMSE": round(float(root_mean_squared_error(y, pred)), 2),
            "R2": round(float(r2_score(y, pred)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true",
                    help="dry run on the VALIDATION split; the test split is not touched")
    ap.add_argument("--i-understand-this-reuses-the-test-set", dest="reuse", action="store_true")
    args = ap.parse_args()
    split = "validation" if args.rehearsal else "test"

    out_dir = REPORTS_DIR / "rehearsal" if args.rehearsal else REPORTS_DIR
    fig_dir = out_dir / "figures" if args.rehearsal else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = " [REHEARSAL ON VALIDATION DATA - NOT TEST RESULTS]" if args.rehearsal else ""

    # ---------------- safety checks ----------------
    if not FROZEN_FILE.exists():
        raise SystemExit("Settings are not frozen. Run  python src\\select_and_freeze.py  first.")
    if not args.rehearsal and LOCK_FILE.exists() and not args.reuse:
        raise SystemExit("The test set has ALREADY been evaluated (models/test_evaluation_lock.json). "
                         "Results are in reports/test_evaluation_report_day7.md and models/final_config.json. "
                         "Re-running would reuse the test set. Refusing.")
    frozen = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
    for fname, expected in frozen["file_sha256"].items():
        if sha256_file(MODELS_DIR / fname) != expected:
            raise SystemExit(f"{fname} has changed since the settings were frozen. "
                             "Do not retrain after freezing. Stopping.")
    if not args.rehearsal and sha256_file(PROCESSED_DIR / "test.csv") != frozen["test_csv_sha256"]:
        raise SystemExit("test.csv has changed since the settings were frozen. Stopping.")
    print(f"Fingerprints verified. Evaluating on the {split.upper()} split.{tag}")

    benchmark = frozen["benchmark"]
    selected = frozen["selected_classifier_name"]
    X, y_score, y_class, ids = load_split(split)
    df = pd.read_csv(PROCESSED_DIR / f"{split}.csv")
    _, y_train_score, y_train_class, _ = load_split("train")
    models = load_models()
    reg = joblib.load(MODELS_DIR / REGRESSION_FILE)
    prevalence = float(y_class.mean())
    print(f"Rows: {len(X)} | prevalence {prevalence:.1%} | selected classifier: {selected}")

    # ================= regression =================
    pred = predict_score(reg, X)
    base_pred = np.full(len(y_score), float(y_train_score.mean()))
    reg_tab = pd.DataFrame([{"Model": "Baseline: training-mean score", **reg_metrics(y_score, base_pred)},
                            {"Model": "Linear regression (clipped 0-100)", **reg_metrics(y_score, pred)}])
    reg_m = reg_metrics(y_score, pred)
    reg_b = reg_metrics(y_score, base_pred)
    print("\nREGRESSION:")
    print(md_table(reg_tab))

    resid = y_score - pred
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(y_score, pred, alpha=0.5, s=18, color="#2a7f9e")
    axes[0].plot([0, 100], [0, 100], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual final score"); axes[0].set_ylabel("Predicted final score")
    axes[0].set_title(f"Actual vs predicted ({split})"); axes[0].legend()
    axes[1].scatter(pred, resid, alpha=0.5, s=18, color="#2a7f9e"); axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted final score"); axes[1].set_ylabel("Residual (actual - predicted)")
    axes[1].set_title(f"Residuals vs predicted ({split})")
    axes[2].hist(resid, bins=25, color="#2a7f9e"); axes[2].axvline(0, color="r", linestyle="--")
    axes[2].set_xlabel("Residual (score points)"); axes[2].set_title(f"Residual distribution ({split})")
    plt.tight_layout(); plt.savefig(fig_dir / "16_regression_heldout.png", dpi=150); plt.close()

    # ================= classifiers (frozen thresholds) =================
    scores, cls_rows, cls_metrics = {}, [], {}
    prior = DummyClassifier(strategy="prior").fit(X, y_class)    # constant-probability reference
    base_cm = classification_metrics(y_class, prior.predict_proba(X)[:, 1], 0.5)
    cls_rows.append({"Model": "Majority-class baseline (always predict 0)", "Score type": "-",
                     "ROC-AUC": 0.5, "PR-AUC": round(prevalence, 3), "Precision": 0.0, "Recall": 0.0,
                     "F1": 0.0, "Flagged share": "0.0%", "Brier": "-"})
    for name, pipe in models.items():
        stype = frozen["models"][name]["score_type"]
        thr = frozen["models"][name]["threshold"]
        s = get_scores(pipe, X, stype)
        scores[name] = s
        m = classification_metrics(y_class, s, thr)
        flagged = float((s >= thr).mean())
        brier = float(brier_score_loss(y_class, s)) if stype == "probability" else None
        cls_metrics[name] = {"score_type": stype, "threshold": thr, **m,
                             "flagged_share": round(flagged, 3),
                             "brier": None if brier is None else round(brier, 3)}
        cls_rows.append({"Model": name + (" (SELECTED)" if name == selected else ""),
                         "Score type": stype.replace("_", " "), "ROC-AUC": m["ROC-AUC"], "PR-AUC": m["PR-AUC"],
                         "Precision": m["Precision"], "Recall": m["Recall"], "F1": m["F1"],
                         "Flagged share": f"{flagged:.1%}",
                         "Brier": "n/a" if brier is None else round(brier, 3)})
    cls_tab = pd.DataFrame(cls_rows)
    print("\nCLASSIFIERS (each at its frozen threshold):")
    print(md_table(cls_tab))

    base_brier = float(brier_score_loss(y_class, np.full(len(y_class), float(y_train_class.mean()))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in models:
        label = f"{name} ({'probability' if frozen['models'][name]['score_type']=='probability' else 'decision score'})"
        RocCurveDisplay.from_predictions(y_class, scores[name], name=label, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_class, scores[name], name=label, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", label="Constant baseline"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"ROC curves ({split})")
    axes[1].axhline(prevalence, color="k", linestyle="--", label=f"Baseline ({prevalence:.2f})")
    axes[1].legend(fontsize=8); axes[1].set_title(f"Precision-recall curves ({split})")
    plt.tight_layout(); plt.savefig(fig_dir / "17_classifier_curves_heldout.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, models):
        thr = frozen["models"][name]["threshold"]
        ConfusionMatrixDisplay.from_predictions(y_class, (scores[name] >= thr).astype(int), ax=ax, cmap="Blues",
                                                display_labels=["At/above", "Below"], colorbar=False)
        ax.set_title(f"{name}\nfrozen threshold {thr:.3f} ({split})", fontsize=10)
    plt.tight_layout(); plt.savefig(fig_dir / "18_confusion_matrices_heldout.png", dpi=150); plt.close()

    # ================= error analysis (selected classifier) =================
    thr_sel = frozen["threshold"]
    s_sel = scores[selected]
    flag = (s_sel >= thr_sel).astype(int)
    y = y_class.to_numpy()
    etype = np.select([(flag == 1) & (y == 1), (flag == 1) & (y == 0), (flag == 0) & (y == 1)],
                      ["true positive", "false positive", "false negative"], default="true negative")
    an = df.copy()
    an["predicted_final_score"] = np.round(pred, 1)
    an["selected_model_score"] = np.round(s_sel, 4)
    an["flagged_for_support"] = flag
    an["error_type"] = etype
    an["score_error"] = np.round(an["final_score"] - an["predicted_final_score"], 1)
    case_cols = ["enrollment_id", "course_category", "final_score", "below_benchmark", "predicted_final_score",
                 "selected_model_score", "flagged_for_support", "error_type"]
    an[case_cols].to_csv(out_dir / f"{split}_predictions_selected_model.csv", index=False)

    counts = an["error_type"].value_counts().reindex(
        ["true positive", "false negative", "false positive", "true negative"]).fillna(0).astype(int)
    counts_tab = counts.rename_axis("Outcome").reset_index(name="Learners")

    prof = an.groupby("error_type")[NUMERIC_PROFILE].mean().round(1)
    prof.insert(0, "n", an["error_type"].value_counts())
    prof = prof.reindex(["true positive", "false negative", "false positive", "true negative"]).reset_index()
    prof = prof.rename(columns={"error_type": "group"})

    dist = (an["final_score"] - benchmark).abs()
    an["band"] = pd.cut(dist, [-0.01, 5, 10, 200], labels=["within 5 points of benchmark",
                                                           "5 to 10 points away", "more than 10 points away"])
    band_tab = (an.groupby("band", observed=True)
                .apply(lambda g: pd.Series({"Learners": len(g),
                                            "Misclassified": int((g["error_type"].isin(["false positive", "false negative"])).sum())}),
                       include_groups=False).reset_index().rename(columns={"band": "Distance from benchmark"}))
    band_tab["Error rate"] = (band_tab["Misclassified"] / band_tab["Learners"]).round(3)

    cat_rows = []
    for cat, g in an.groupby("course_category"):
        pos = int((g["below_benchmark"] == 1).sum())
        tp = int((g["error_type"] == "true positive").sum())
        fp = int((g["error_type"] == "false positive").sum())
        rec = tp / pos if pos else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        note = []
        if len(g) < MIN_GROUP_ROWS:
            note.append(f"fewer than {MIN_GROUP_ROWS} learners")
        if pos < MIN_GROUP_POSITIVES:
            note.append(f"fewer than {MIN_GROUP_POSITIVES} below-benchmark learners (recall unreliable)")
        cat_rows.append({"Course category": cat, "Learners": len(g), "Below benchmark": pos,
                         "Score MAE": round(float(g["score_error"].abs().mean()), 2),
                         "Recall": "n/a" if np.isnan(rec) else round(rec, 3),
                         "Precision": "n/a" if np.isnan(prec) else round(prec, 3),
                         "Caution": "; ".join(note) if note else ""})
    cat_tab = pd.DataFrame(cat_rows).sort_values("Learners", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    c = cat_tab.sort_values("Course category")
    axes[0].bar(c["Course category"], c["Score MAE"], color="#2a7f9e")
    for i, (m, n) in enumerate(zip(c["Score MAE"], c["Learners"])):
        axes[0].text(i, m + 0.1, f"n={n}", ha="center", fontsize=8)
    axes[0].axhline(reg_m["MAE"], color="r", linestyle="--", label=f"Overall MAE {reg_m['MAE']}")
    axes[0].set_ylabel("Mean absolute error (score points)"); axes[0].legend()
    axes[0].set_title(f"Score error by course category ({split})")
    plt.setp(axes[0].get_xticklabels(), rotation=20)
    axes[1].bar(band_tab["Distance from benchmark"].astype(str), band_tab["Error rate"], color="#c8553d")
    for i, (r, n) in enumerate(zip(band_tab["Error rate"], band_tab["Learners"])):
        axes[1].text(i, r + 0.01, f"n={n}", ha="center", fontsize=8)
    axes[1].set_ylabel("Share misclassified")
    axes[1].set_title(f"{selected}: errors by distance from benchmark")
    plt.setp(axes[1].get_xticklabels(), rotation=15, fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "19_error_analysis_heldout.png", dpi=150); plt.close()

    excl = pd.read_csv(PROCESSED_DIR / "excluded_non_observed.csv")
    excl_counts = excl["outcome_status"].value_counts()

    sel_m = cls_metrics[selected]
    print(f"\nSELECTED ({selected}) at frozen threshold {thr_sel:.4f}: "
          f"precision {sel_m['Precision']}, recall {sel_m['Recall']}, flagged {sel_m['flagged_share']:.1%}")
    print(md_table(counts_tab))

    # ================= write config + report =================
    final_cfg = {
        "data_origin": "SYNTHETIC", "evaluated_on": split, "benchmark": benchmark,
        "selected_classifier": frozen["selected_classifier"], "selected_classifier_name": selected,
        "threshold": thr_sel, "threshold_type": frozen["threshold_type"],
        "probability_available": frozen["probability_available"],
        "support_rule": "flag for support when score >= threshold",
        "regression_pipeline": REGRESSION_FILE, "regression_prediction_rule": "clipped to 0-100",
        "regression_test_mae": reg_m["MAE"], "regression_test_rmse": reg_m["RMSE"],
        "regression_test_r2": reg_m["R2"], "regression_baseline_test_mae": reg_b["MAE"],
        "classifier_test_metrics": cls_metrics,
        "selected_classifier_test_metrics": cls_metrics[selected],
        "test_baseline": {"prevalence": round(prevalence, 3), "majority_class_accuracy": round(1 - prevalence, 3),
                          "constant_brier": round(base_brier, 3)},
        "frozen_settings_sha256": sha256_file(FROZEN_FILE),
    }
    cfg_path = out_dir / "final_config_rehearsal.json" if args.rehearsal else FINAL_CONFIG_FILE
    cfg_path.write_text(json.dumps(final_cfg, indent=2), encoding="utf-8")

    beats_reg = reg_m["MAE"] < reg_b["MAE"]
    beats_cls = cls_metrics[selected]["PR-AUC"] > prevalence
    name_split = "Held-out test" if not args.rehearsal else "REHEARSAL (validation data)"
    rep = [
        f"# {name_split} Evaluation (Day 7, phase B)", "",
        "**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, "
        "not that it is effective on real learners.", "",
    ]
    if args.rehearsal:
        rep += ["> **REHEARSAL: these numbers come from the validation split, not the test split.**", ""]
    rep += [
        f"- Split evaluated: **{split}** ({len(X)} learners with an observed final assessment; "
        f"{prevalence:.1%} below the benchmark of {benchmark}).",
        "- All models and thresholds were frozen before this evaluation. Nothing was fitted or tuned "
        "here, and the file fingerprints were verified.",
        f"- Selected classifier: **{selected}**, frozen threshold {thr_sel:.4f} "
        f"({frozen['threshold_type'].replace('_', ' ')}). A learner is flagged for support when the score "
        "is at or above the threshold.",
        f"- Assumptions behind the threshold (to confirm with the client): capacity "
        f"{frozen['assumptions_to_confirm_with_client']['capacity_max_flagged_share']:.0%} of learners, "
        f"recall weighted {frozen['assumptions_to_confirm_with_client']['beta_recall_weight']:g}x precision.",
        f"- Excluded and reported separately: {int(excl_counts.get('withdrawn', 0))} withdrawn and "
        f"{int(excl_counts.get('missing_assessment', 0))} missing-assessment enrollments. They were not "
        "scored as zero or below benchmark. The model applies to the observed-assessment population only.", "",
        "## Score prediction (linear regression)", md_table(reg_tab), "",
        f"MAE and RMSE are in score points. {'The model improves on the training-mean baseline.' if beats_reg else 'The model does NOT improve on its baseline; a data improvement should be proposed before any readiness claim.'}", "",
        "## Classifiers at their frozen thresholds", md_table(cls_tab), "",
        f"Prevalence {prevalence:.1%}; constant-probability Brier reference {base_brier:.3f}. "
        "SVM values are decision scores (signed margins), not probabilities; its ROC-AUC and PR-AUC use "
        "those scores. "
        + ("The selected model's PR-AUC is above the prevalence baseline." if beats_cls
           else "The selected model does NOT beat the prevalence baseline on PR-AUC."), "",
        "## Error analysis for the selected classifier", "",
        "### Outcomes at the frozen threshold", md_table(counts_tab), "",
        "False negatives are learners below the benchmark who were not flagged (missed). False positives "
        "were flagged but finished at or above the benchmark.", "",
        "### Average inputs by outcome group (descriptive only)", md_table(prof), "",
        "### Errors by distance from the benchmark", md_table(band_tab), "",
        "Learners close to the benchmark are the hardest to classify: a learner scoring 48 and one scoring 52 "
        "look almost the same to any model.", "",
        "### By course category",
        md_table(cat_tab), "",
        "Groups flagged in the Caution column are too small for reliable conclusions.", "",
        "## Limitations",
        "- Synthetic data: the relationships were built into the generator, so these results cannot be "
        "read as effectiveness on real learners.",
        "- One cohort, one split: this measures generalization within the sampled population. A deployment "
        "claim requires evaluation on later cohorts.",
        "- Coefficients, tree splits and feature importance are associations, not causes of learning outcomes.",
        "- Predictions support human review and optional learning assistance only. They must not decide "
        "admission, certification, grades or access to learning opportunities.", "",
        "## Charts",
        f"- `{fig_dir.relative_to(REPORTS_DIR.parent).as_posix()}/16_regression_heldout.png`",
        f"- `{fig_dir.relative_to(REPORTS_DIR.parent).as_posix()}/17_classifier_curves_heldout.png`",
        f"- `{fig_dir.relative_to(REPORTS_DIR.parent).as_posix()}/18_confusion_matrices_heldout.png`",
        f"- `{fig_dir.relative_to(REPORTS_DIR.parent).as_posix()}/19_error_analysis_heldout.png`", "",
    ]
    rep_name = "rehearsal_report_day7.md" if args.rehearsal else "test_evaluation_report_day7.md"
    (out_dir / rep_name).write_text("\n".join(rep), encoding="utf-8")

    if not args.rehearsal:
        LOCK_FILE.write_text(json.dumps({
            "evaluated_split": "test",
            "final_config_sha256": sha256_file(FINAL_CONFIG_FILE),
            "note": "The test set has been used once. Settings are frozen. Re-running or retuning "
                    "would invalidate these results."}, indent=2), encoding="utf-8")
        print("\nTest evaluation complete and LOCKED (models/test_evaluation_lock.json).")
    print(f"Saved: {cfg_path.relative_to(REPORTS_DIR.parent).as_posix()}, "
          f"{(out_dir / rep_name).relative_to(REPORTS_DIR.parent).as_posix()}, "
          f"{split}_predictions_selected_model.csv, charts 16-19")


if __name__ == "__main__":
    main()
