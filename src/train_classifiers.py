"""
Day 5 - Logistic regression and decision tree classifiers (below_benchmark).
Project : Learner Performance Prediction & Early Learning Support Dashboard

DATA IS SYNTHETIC. Results demonstrate the implementation, not effectiveness on real learners.

Rules followed here
  - Both models use the SAME preprocessing and the SAME train/validation split.
  - Everything is fitted on the TRAINING split. Tuning uses the VALIDATION split only.
  - The TEST split is not opened (it is used once, on Day 7).
  - No support-priority threshold is chosen today. Label metrics are shown at a provisional 0.5
    cut-off; the real threshold is chosen on validation data on Day 7 and then frozen.
  - The decision score / probability always refers to the BELOW-BENCHMARK class.

Usage:  python src/train_classifiers.py
"""

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from eval_utils import classification_metrics, md_table, positive_class_probability
from pipeline_utils import (BENCHMARK, FEATURES, FIG_DIR, MODELS_DIR, REPORTS_DIR, SEED,
                            build_preprocessor, load_split)

PROVISIONAL_THRESHOLD = 0.5
TREE_DEPTHS = [2, 3, 4, 5, 6]
TREE_LEAVES = [1, 5, 10, 20, 40]
AUC_TOLERANCE = 0.005          # prefer the simplest tree within this of the best validation ROC-AUC


def make_logistic() -> Pipeline:
    return Pipeline([("prep", build_preprocessor()),
                     ("model", LogisticRegression(max_iter=1000, random_state=SEED))])


def make_tree(max_depth=None, min_samples_leaf=1) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()),
                     ("model", DecisionTreeClassifier(max_depth=max_depth,
                                                      min_samples_leaf=min_samples_leaf,
                                                      random_state=SEED))])


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, _, y_train, _ = load_split("train")
    X_val, _, y_val, _ = load_split("validation")
    prevalence = float(y_val.mean())
    print(f"Train rows: {len(X_train)} | Validation rows: {len(X_val)} | "
          f"Validation prevalence: {prevalence:.1%}")

    # ---------------- baselines ----------------
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    prior = DummyClassifier(strategy="prior").fit(X_train, y_train)      # constant probability
    maj_pred = majority.predict(X_val)
    maj_acc = float((maj_pred == y_val).mean())
    base_prob = prior.predict_proba(X_val)[:, 1]
    base = classification_metrics(y_val, base_prob, PROVISIONAL_THRESHOLD)

    # ---------------- logistic regression (convergence is checked, not assumed) ----------------
    logreg = make_logistic()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)        # a warning becomes a hard stop
        logreg.fit(X_train, y_train)
    n_iter = int(logreg.named_steps["model"].n_iter_[0])
    print(f"Logistic regression converged in {n_iter} iterations (limit 1000).")
    lr_val_prob = positive_class_probability(logreg, X_val)
    lr_train_prob = positive_class_probability(logreg, X_train)

    # ---------------- decision tree: small tuning grid on VALIDATION data ----------------
    grid = []
    for d in TREE_DEPTHS:
        for leaf in TREE_LEAVES:
            t = make_tree(d, leaf).fit(X_train, y_train)
            m_tr = classification_metrics(y_train, positive_class_probability(t, X_train))
            m_va = classification_metrics(y_val, positive_class_probability(t, X_val))
            grid.append({"max_depth": d, "min_samples_leaf": leaf,
                         "train ROC-AUC": m_tr["ROC-AUC"], "val ROC-AUC": m_va["ROC-AUC"],
                         "val PR-AUC": m_va["PR-AUC"]})
    grid_df = pd.DataFrame(grid)
    grid_df[["max_depth", "min_samples_leaf"]] = grid_df[["max_depth", "min_samples_leaf"]].astype(int)
    best_auc = grid_df["val ROC-AUC"].max()
    near_best = grid_df[grid_df["val ROC-AUC"] >= best_auc - AUC_TOLERANCE]
    chosen = near_best.sort_values(["max_depth", "min_samples_leaf"], ascending=[True, False]).iloc[0]
    depth, leaf = int(chosen["max_depth"]), int(chosen["min_samples_leaf"])
    print(f"Decision tree chosen on validation: max_depth={depth}, min_samples_leaf={leaf} "
          f"(best validation ROC-AUC in grid = {best_auc})")

    tree = make_tree(depth, leaf).fit(X_train, y_train)
    tree_val_prob = positive_class_probability(tree, X_val)
    tree_train_prob = positive_class_probability(tree, X_train)

    # ---------------- comparison table (validation) ----------------
    rows = [
        {"Model": "Majority-class baseline (always predict 0)", "ROC-AUC": 0.5,
         "PR-AUC": round(prevalence, 3), "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
         "Accuracy": round(maj_acc, 3)},
        {"Model": "Constant-probability baseline (prevalence)", **{k: base[k] for k in
         ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"]}},
    ]
    lr_val = classification_metrics(y_val, lr_val_prob, PROVISIONAL_THRESHOLD)
    tr_val = classification_metrics(y_val, tree_val_prob, PROVISIONAL_THRESHOLD)
    lr_tr = classification_metrics(y_train, lr_train_prob, PROVISIONAL_THRESHOLD)
    tr_tr = classification_metrics(y_train, tree_train_prob, PROVISIONAL_THRESHOLD)
    keys = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"]
    rows.append({"Model": "Logistic regression", **{k: lr_val[k] for k in keys}})
    rows.append({"Model": f"Decision tree (depth {depth}, leaf {leaf})", **{k: tr_val[k] for k in keys}})
    comparison = pd.DataFrame(rows)
    print("\nVALIDATION metrics (label metrics at provisional threshold 0.5):")
    print(md_table(comparison))

    overfit = pd.DataFrame([
        {"Model": "Logistic regression", "Train ROC-AUC": lr_tr["ROC-AUC"],
         "Validation ROC-AUC": lr_val["ROC-AUC"], "Gap": round(lr_tr["ROC-AUC"] - lr_val["ROC-AUC"], 3),
         "Train F1": lr_tr["F1"], "Validation F1": lr_val["F1"]},
        {"Model": f"Decision tree (depth {depth}, leaf {leaf})", "Train ROC-AUC": tr_tr["ROC-AUC"],
         "Validation ROC-AUC": tr_val["ROC-AUC"], "Gap": round(tr_tr["ROC-AUC"] - tr_val["ROC-AUC"], 3),
         "Train F1": tr_tr["F1"], "Validation F1": tr_val["F1"]},
    ])
    print("\nTraining vs validation (overfitting check):")
    print(md_table(overfit))

    # ---------------- overfitting demonstration: unrestricted depth sweep ----------------
    sweep = []
    for d in list(range(1, 13)):
        t = make_tree(d, 1).fit(X_train, y_train)
        sweep.append({"depth": d,
                      "train": classification_metrics(y_train, positive_class_probability(t, X_train))["ROC-AUC"],
                      "validation": classification_metrics(y_val, positive_class_probability(t, X_val))["ROC-AUC"]})
    sweep = pd.DataFrame(sweep)

    # ---------------- charts ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, prob in [("Logistic regression", lr_val_prob), (f"Decision tree (depth {depth})", tree_val_prob)]:
        RocCurveDisplay.from_predictions(y_val, prob, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_val, prob, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", label="Constant baseline")
    axes[0].set_title("ROC curve (validation)"); axes[0].legend()
    axes[1].axhline(prevalence, color="k", linestyle="--", label=f"Baseline (prevalence {prevalence:.2f})")
    axes[1].set_title("Precision-recall curve (validation)"); axes[1].legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "07_classifier_curves_validation.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep["depth"], sweep["train"], "o-", label="Training ROC-AUC", color="#c8553d")
    ax.plot(sweep["depth"], sweep["validation"], "o-", label="Validation ROC-AUC", color="#2a7f9e")
    ax.axvline(depth, color="gray", linestyle="--", label=f"Chosen depth ({depth}); chosen tree also uses min_samples_leaf={leaf}")
    ax.set_xlabel("max_depth (min_samples_leaf = 1)"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Decision tree overfitting check"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "08_decision_tree_overfitting.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, prob) in zip(axes, [("Logistic regression", lr_val_prob),
                                       (f"Decision tree (depth {depth})", tree_val_prob)]):
        ConfusionMatrixDisplay.from_predictions(
            y_val, (prob >= PROVISIONAL_THRESHOLD).astype(int), ax=ax, cmap="Blues",
            display_labels=["At/above", "Below"], colorbar=False)
        ax.set_title(f"{name}\n(validation, threshold {PROVISIONAL_THRESHOLD})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "10_confusion_matrices_day5.png", dpi=150); plt.close()

    # shallow tree for interpretation only (separate depth-3 model, fitted on training data)
    shallow = make_tree(3, 20).fit(X_train, y_train)
    names = [n.split("__", 1)[1] for n in shallow.named_steps["prep"].get_feature_names_out()]
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(shallow.named_steps["model"], feature_names=names, class_names=["At/above", "Below"],
              filled=True, rounded=True, fontsize=8, ax=ax, impurity=False, proportion=True)
    ax.set_title("Shallow decision tree (depth 3) for interpretation only - synthetic data; split values are in standardized units")
    plt.tight_layout(); plt.savefig(FIG_DIR / "09_decision_tree_shallow.png", dpi=150); plt.close()

    # ---------------- logistic coefficients (descriptive, NOT causal) ----------------
    lr_names = [n.split("__", 1)[1] for n in logreg.named_steps["prep"].get_feature_names_out()]
    coefs = pd.DataFrame({"Input (after preprocessing)": lr_names,
                          "Coefficient": logreg.named_steps["model"].coef_[0].round(2)})
    coefs = coefs.reindex(coefs["Coefficient"].abs().sort_values(ascending=False).index)

    # ---------------- save pipelines, prove reload reproduces predictions ----------------
    for name, pipe, prob in [("logistic_regression", logreg, lr_val_prob),
                             ("decision_tree", tree, tree_val_prob)]:
        path = MODELS_DIR / f"{name}_pipeline.joblib"
        joblib.dump(pipe, path)
        reloaded = joblib.load(path)
        assert np.allclose(positive_class_probability(reloaded, X_val), prob), f"{name} reload differs!"
    print("\nSaved pipelines reload and reproduce identical probabilities: OK")

    meta = {
        "data_origin": "SYNTHETIC", "trained_on": "train split only", "seed": SEED,
        "features": FEATURES, "benchmark": BENCHMARK, "positive_class": "below_benchmark = 1",
        "provisional_threshold": PROVISIONAL_THRESHOLD,
        "threshold_status": "NOT chosen yet - to be chosen on validation data on Day 7, then frozen",
        "scikit_learn_version": sklearn.__version__,
        "logistic_regression": {"max_iter": 1000, "iterations_used": n_iter,
                                "validation": lr_val, "train": lr_tr},
        "decision_tree": {"max_depth": depth, "min_samples_leaf": leaf, "validation": tr_val, "train": tr_tr,
                          "selection_rule": f"simplest tree within {AUC_TOLERANCE} of best validation ROC-AUC"},
    }
    (MODELS_DIR / "classifiers_day5_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------- report ----------------
    report = [
        "# Logistic Regression and Decision Tree - Validation Results (Day 5)", "",
        "**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, "
        "not that it is effective on real learners.", "",
        "- Both models use the same preprocessing and the same train/validation split, and were fitted on "
        "the training split only. The test split has not been used.",
        f"- Positive class: below_benchmark = 1 (benchmark = {BENCHMARK}). Probabilities refer to that class.",
        f"- Prevalence of the positive class in validation: {prevalence:.1%}.",
        f"- Logistic regression converged in {n_iter} iterations (limit 1000), checked with warnings "
        "treated as errors.",
        f"- Label metrics use a **provisional** 0.5 cut-off. The support-priority threshold is not chosen "
        "yet; it will be chosen on validation data on Day 7 and then frozen before testing.",
        "- ROC-AUC and PR-AUC do not depend on the cut-off.", "",
        "## Validation comparison", md_table(comparison), "",
        "## Overfitting check (training vs validation)", md_table(overfit), "",
        f"Decision tree selection: the simplest tree whose validation ROC-AUC was within {AUC_TOLERANCE} "
        f"of the best in the grid (best = {best_auc}), giving max_depth = {depth} and "
        f"min_samples_leaf = {leaf}. Charts: `08_decision_tree_overfitting.png` shows how an "
        "unrestricted tree fits training data much better than validation data.", "",
        "## Tree tuning grid (validation)", md_table(grid_df), "",
        "## Logistic regression coefficients (standardized inputs)",
        "Positive values raise the modelled probability of finishing below the benchmark. "
        "These are associations in synthetic data, not causes of learning outcomes.", "",
        md_table(coefs), "",
        "## Charts",
        "- `reports/figures/07_classifier_curves_validation.png`",
        "- `reports/figures/08_decision_tree_overfitting.png`",
        "- `reports/figures/09_decision_tree_shallow.png` (depth-3 tree for interpretation only)",
        "- `reports/figures/10_confusion_matrices_day5.png`", "",
    ]
    (REPORTS_DIR / "classifier_validation_report_day5.md").write_text("\n".join(report), encoding="utf-8")
    print("Saved: models/logistic_regression_pipeline.joblib, models/decision_tree_pipeline.joblib,")
    print("       models/classifiers_day5_metadata.json, reports/classifier_validation_report_day5.md,")
    print("       reports/figures/07-10 charts")


if __name__ == "__main__":
    main()
