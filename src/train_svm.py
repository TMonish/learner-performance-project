"""
Day 6 - Support Vector Machine classifier (below_benchmark) and three-model comparison.
Project : Learner Performance Prediction & Early Learning Support Dashboard

DATA IS SYNTHETIC. Results demonstrate the implementation, not effectiveness on real learners.

Rules followed here
  - Same preprocessing and same train/validation split as the other models.
  - Fitted on the TRAINING split only. Tuning uses the VALIDATION split only.
  - The TEST split is not opened (it is used once, on Day 7).
  - The SVM output is a SIGNED MARGIN ("decision score"), NOT a probability. Ranking metrics
    (ROC-AUC, PR-AUC) use the decision score. A positive score means the below-benchmark class.
  - A calibrated SVM (training-only cross-validation) is built SEPARATELY and its calibration is
    evaluated separately, in case a probability is needed for the dashboard.
  - No support-priority threshold is chosen today (Day 7). Label metrics use provisional cut-offs:
    probability >= 0.5 for logistic regression / tree, decision score >= 0 for the SVM.

Usage:  python src/train_svm.py        (run Day 5 first: it loads the saved Day 5 pipelines)
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eval_utils import classification_metrics, md_table, positive_class_probability
from pipeline_utils import (BENCHMARK, FEATURES, FIG_DIR, MODELS_DIR, REPORTS_DIR, SEED,
                            build_preprocessor, load_split)

PROB_THRESHOLD = 0.5           # provisional, logistic regression / tree
SCORE_THRESHOLD = 0.0          # provisional, SVM decision score (the SVM's own natural boundary)
C_VALUES = [0.1, 1, 10]        # limited search, as the brief asks
KERNELS = ["linear", "rbf"]    # rbf uses gamma="scale"
AUC_TOLERANCE = 0.005          # prefer the simplest candidate within this of the best validation ROC-AUC
KEYS = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"]


def make_svm(kernel: str, c: float) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()),
                     ("model", SVC(kernel=kernel, C=c, gamma="scale", random_state=SEED))])


def decision_score(pipeline, X) -> np.ndarray:
    """Signed margin. Positive = below-benchmark class. Verified so the sign can never be flipped."""
    assert list(pipeline.classes_) == [0, 1], f"unexpected class order {list(pipeline.classes_)}"
    return pipeline.decision_function(X)


def load_day5(name: str):
    path = MODELS_DIR / f"{name}_pipeline.joblib"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run  python src\\train_classifiers.py  (Day 5) first.")
    return joblib.load(path)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, _, y_train, _ = load_split("train")
    X_val, _, y_val, _ = load_split("validation")
    prevalence = float(y_val.mean())
    print(f"Train rows: {len(X_train)} | Validation rows: {len(X_val)} | "
          f"Validation prevalence: {prevalence:.1%}")

    # ---------------- SVM candidates: linear kernel vs small RBF search ----------------
    cand_rows = []
    for kernel in KERNELS:
        for c in C_VALUES:
            p = make_svm(kernel, c).fit(X_train, y_train)
            m_tr = classification_metrics(y_train, decision_score(p, X_train), SCORE_THRESHOLD)
            m_va = classification_metrics(y_val, decision_score(p, X_val), SCORE_THRESHOLD)
            cand_rows.append({"kernel": kernel, "C": c,
                              "train ROC-AUC": m_tr["ROC-AUC"], "val ROC-AUC": m_va["ROC-AUC"],
                              "val PR-AUC": m_va["PR-AUC"], "val F1 (score>=0)": m_va["F1"]})
    cand = pd.DataFrame(cand_rows)
    best_auc = cand["val ROC-AUC"].max()
    near = cand[cand["val ROC-AUC"] >= best_auc - AUC_TOLERANCE].copy()
    near["order"] = near["kernel"].map({"linear": 0, "rbf": 1})
    chosen = near.sort_values(["order", "C"]).iloc[0]
    kernel, c = str(chosen["kernel"]), float(chosen["C"])
    print(f"SVM chosen on validation: kernel={kernel}, C={c} "
          f"(best validation ROC-AUC among candidates = {best_auc})")

    svm = make_svm(kernel, c).fit(X_train, y_train)
    svm_val_score = decision_score(svm, X_val)
    svm_train_score = decision_score(svm, X_train)
    svm_val = classification_metrics(y_val, svm_val_score, SCORE_THRESHOLD)
    svm_tr = classification_metrics(y_train, svm_train_score, SCORE_THRESHOLD)

    # ---------------- load Day 5 models for the three-way comparison ----------------
    logreg, tree = load_day5("logistic_regression"), load_day5("decision_tree")
    lr_val_prob = positive_class_probability(logreg, X_val)
    tree_val_prob = positive_class_probability(tree, X_val)
    lr_val = classification_metrics(y_val, lr_val_prob, PROB_THRESHOLD)
    tree_val = classification_metrics(y_val, tree_val_prob, PROB_THRESHOLD)
    tree_label = f"Decision tree (depth {tree.named_steps['model'].max_depth}, " \
                 f"leaf {tree.named_steps['model'].min_samples_leaf})"
    svm_label = f"SVM ({kernel}, C={c:g}) - decision score"

    base_rows = [
        {"Model": "Majority-class baseline (always predict 0)", "ROC-AUC": 0.5,
         "PR-AUC": round(prevalence, 3), "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
         "Accuracy": round(1 - prevalence, 3)},
    ]
    comparison = pd.DataFrame(base_rows + [
        {"Model": "Logistic regression", **{k: lr_val[k] for k in KEYS}},
        {"Model": tree_label, **{k: tree_val[k] for k in KEYS}},
        {"Model": svm_label, **{k: svm_val[k] for k in KEYS}},
    ])
    print("\nVALIDATION comparison (same rows for every model):")
    print(md_table(comparison))
    print("  Label metrics use provisional cut-offs: prob >= 0.5 (LR, tree) and score >= 0 (SVM).")

    overfit = pd.DataFrame([
        {"Model": svm_label, "Train ROC-AUC": svm_tr["ROC-AUC"], "Validation ROC-AUC": svm_val["ROC-AUC"],
         "Gap": round(svm_tr["ROC-AUC"] - svm_val["ROC-AUC"], 3),
         "Train F1": svm_tr["F1"], "Validation F1": svm_val["F1"]}])
    print("\nSVM training vs validation:")
    print(md_table(overfit))

    # ---------------- calibrated SVM (separate model, training-only cross-validation) ----------------
    calibrated = CalibratedClassifierCV(estimator=make_svm(kernel, c), method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    assert list(calibrated.classes_) == [0, 1]
    cal_val_prob = calibrated.predict_proba(X_val)[:, 1]

    brier_base = brier_score_loss(y_val, np.full(len(y_val), float(y_train.mean())))
    calib = pd.DataFrame([
        {"Probability source": "Constant baseline (training prevalence)", "Brier score": round(brier_base, 3),
         "ROC-AUC": 0.5},
        {"Probability source": "Logistic regression", "Brier score": round(brier_score_loss(y_val, lr_val_prob), 3),
         "ROC-AUC": lr_val["ROC-AUC"]},
        {"Probability source": tree_label, "Brier score": round(brier_score_loss(y_val, tree_val_prob), 3),
         "ROC-AUC": tree_val["ROC-AUC"]},
        {"Probability source": "SVM, sigmoid-calibrated (5-fold on training data)",
         "Brier score": round(brier_score_loss(y_val, cal_val_prob), 3),
         "ROC-AUC": classification_metrics(y_val, cal_val_prob, PROB_THRESHOLD)["ROC-AUC"]},
    ])
    print("\nCalibration check on validation (lower Brier score = better probabilities):")
    print(md_table(calib))

    # ---------------- charts ----------------
    labels = [f"{r.kernel}\nC={r.C:g}" for r in cand.itertuples()]
    x = np.arange(len(cand))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, cand["train ROC-AUC"], 0.4, label="Training ROC-AUC", color="#c8553d")
    ax.bar(x + 0.2, cand["val ROC-AUC"], 0.4, label="Validation ROC-AUC", color="#2a7f9e")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.7, 1.0)
    ax.set_title("SVM candidates: training vs validation ROC-AUC (decision scores)")
    ax.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "11_svm_candidates.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    series = [("Logistic regression (probability)", lr_val_prob),
              (tree_label, tree_val_prob), (f"SVM {kernel} (decision score)", svm_val_score)]
    for name, s in series:
        RocCurveDisplay.from_predictions(y_val, s, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_val, s, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", label="Constant baseline"); axes[0].legend(fontsize=8)
    axes[0].set_title("ROC curves (validation)")
    axes[1].axhline(prevalence, color="k", linestyle="--", label=f"Baseline ({prevalence:.2f})")
    axes[1].legend(fontsize=8); axes[1].set_title("Precision-recall curves (validation)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "12_three_classifier_curves.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, p in [("Logistic regression", lr_val_prob), (tree_label, tree_val_prob),
                    ("SVM, calibrated", cal_val_prob)]:
        frac, mean_pred = calibration_curve(y_val, p, n_bins=6, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed share below benchmark")
    ax.set_title("Calibration (validation, 6 bins - small sample, read with care)")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIG_DIR / "13_calibration_curves.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("Logistic regression\n(prob >= 0.5)", (lr_val_prob >= PROB_THRESHOLD).astype(int)),
              (f"{tree_label}\n(prob >= 0.5)", (tree_val_prob >= PROB_THRESHOLD).astype(int)),
              (f"SVM {kernel}\n(score >= 0)", (svm_val_score >= SCORE_THRESHOLD).astype(int))]
    for ax, (title, pred) in zip(axes, panels):
        ConfusionMatrixDisplay.from_predictions(y_val, pred, ax=ax, cmap="Blues",
                                                display_labels=["At/above", "Below"], colorbar=False)
        ax.set_title(title + "\nvalidation, provisional cut-off", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG_DIR / "14_confusion_matrices_day6.png", dpi=150); plt.close()

    # ---------------- save, then prove reload reproduces outputs ----------------
    joblib.dump(svm, MODELS_DIR / "svm_pipeline.joblib")
    joblib.dump(calibrated, MODELS_DIR / "svm_calibrated_pipeline.joblib")
    assert np.allclose(decision_score(joblib.load(MODELS_DIR / "svm_pipeline.joblib"), X_val), svm_val_score)
    assert np.allclose(joblib.load(MODELS_DIR / "svm_calibrated_pipeline.joblib").predict_proba(X_val)[:, 1],
                       cal_val_prob)
    print("\nSaved SVM pipelines reload and reproduce identical outputs: OK")

    comparison.to_csv(REPORTS_DIR / "classifier_comparison_validation.csv", index=False)
    meta = {
        "data_origin": "SYNTHETIC", "trained_on": "train split only", "seed": SEED,
        "features": FEATURES, "benchmark": BENCHMARK, "positive_class": "below_benchmark = 1",
        "scikit_learn_version": sklearn.__version__,
        "svm": {"kernel": kernel, "C": c, "gamma": "scale", "output": "decision score (signed margin), NOT a probability",
                "selection_rule": f"simplest candidate within {AUC_TOLERANCE} of best validation ROC-AUC "
                                  "(linear before rbf, then smaller C)",
                "provisional_score_threshold": SCORE_THRESHOLD, "validation": svm_val, "train": svm_tr},
        "svm_calibrated": {"method": "sigmoid", "cv": 5, "fitted_on": "training data only",
                           "validation_brier": float(calib.iloc[3]["Brier score"])},
        "threshold_status": "NOT chosen yet - to be chosen on validation data on Day 7, then frozen",
    }
    (MODELS_DIR / "svm_day6_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------- report ----------------
    report = [
        "# SVM and Classifier Comparison - Validation Results (Day 6)", "",
        "**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, "
        "not that it is effective on real learners.", "",
        "- Same preprocessing and same train/validation split as the other classifiers. Fitted on the "
        "training split only. The test split has not been used.",
        f"- Candidates: a linear kernel and an RBF kernel, each with C in {C_VALUES} and gamma = \"scale\".",
        f"- Selection rule: the simplest candidate (linear before RBF, then smaller C) whose validation "
        f"ROC-AUC is within {AUC_TOLERANCE} of the best ({best_auc}). Chosen: **{kernel}, C = {c:g}**.",
        "- **The SVM output is a decision score (signed margin), not a probability.** A positive score means "
        "the below-benchmark class. ROC-AUC and PR-AUC use this score.",
        "- Label metrics use provisional cut-offs (probability >= 0.5 for logistic regression and the tree; "
        "decision score >= 0 for the SVM). The support-priority threshold is chosen on validation data on "
        "Day 7 and then frozen.", "",
        "## SVM candidates (validation)", md_table(cand), "",
        "## Three-classifier comparison (validation)", md_table(comparison), "",
        "## SVM overfitting check", md_table(overfit), "",
        "## Probability calibration (evaluated separately)",
        "A sigmoid-calibrated SVM was fitted with 5-fold cross-validation on training data only. The "
        "Brier score measures probability quality (lower is better). The validation set has 300 rows, "
        "so calibration curves are rough.", "",
        md_table(calib), "",
        "## Charts",
        "- `reports/figures/11_svm_candidates.png`",
        "- `reports/figures/12_three_classifier_curves.png`",
        "- `reports/figures/13_calibration_curves.png`",
        "- `reports/figures/14_confusion_matrices_day6.png`", "",
    ]
    (REPORTS_DIR / "svm_validation_report_day6.md").write_text("\n".join(report), encoding="utf-8")
    print("Saved: models/svm_pipeline.joblib, models/svm_calibrated_pipeline.joblib, models/svm_day6_metadata.json,")
    print("       reports/svm_validation_report_day6.md, reports/classifier_comparison_validation.csv, charts 11-14")


if __name__ == "__main__":
    main()
