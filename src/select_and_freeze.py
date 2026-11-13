"""
Day 7, phase A - Select the classifier and the support-priority thresholds, then FREEZE.
Project : Learner Performance Prediction & Early Learning Support Dashboard

DATA IS SYNTHETIC. Results demonstrate the implementation, not effectiveness on real learners.

This script uses the TRAINING and VALIDATION splits only. The test split is never read or analysed
here (the file is only fingerprinted with a hash so later tampering can be detected).

What it does
  1. Scores every saved classifier on the validation split.
  2. Picks a threshold for EACH model with the same documented rule (see final_utils.py):
     highest F-beta among thresholds that flag no more than CAPACITY of learners.
  3. Selects the classifier: models within PRAUC_TOLERANCE of the best validation PR-AUC are tied;
     ties go to the more interpretable model (logistic regression first).
  4. Decides whether a probability may be shown in the dashboard (validation Brier skill rule).
  5. Writes models/frozen_settings.json (settings + file fingerprints). Nothing is re-tuned afterwards.

Usage:  python src/select_and_freeze.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from eval_utils import md_table
from final_utils import (BETA, CAPACITY, FROZEN_FILE, MIN_BRIER_SKILL, MODELS, PRAUC_TOLERANCE,
                         REGRESSION_FILE, TIE_BREAK_ORDER, choose_threshold, get_scores,
                         load_models, sha256_file, sweep_thresholds)
from pipeline_utils import (BENCHMARK, FEATURES, FIG_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED,
                            load_split)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lock = MODELS_DIR / "test_evaluation_lock.json"
    if lock.exists():
        raise SystemExit("The test set has already been evaluated. Settings are frozen; "
                         "changing them now would invalidate the test results.")
    if FROZEN_FILE.exists():
        print(f"NOTE: {FROZEN_FILE.name} already exists and will be overwritten. "
              "Do not re-run this after the test evaluation has been done.")

    X_val, _, y_val, _ = load_split("validation")
    _, _, y_train, _ = load_split("train")
    models = load_models()
    base_brier = brier_score_loss(y_val, np.full(len(y_val), float(y_train.mean())))
    print(f"Validation rows: {len(X_val)} | prevalence {y_val.mean():.1%} | "
          f"capacity assumption {CAPACITY:.0%} | beta {BETA:g}")

    summary, per_model, sweeps = [], {}, {}
    for name, pipe in models.items():
        stype = MODELS[name]["score_type"]
        s = get_scores(pipe, X_val, stype)
        sweep = sweep_thresholds(y_val, s)
        row = choose_threshold(sweep)
        sweeps[name] = sweep
        brier = float(brier_score_loss(y_val, s)) if stype == "probability" else None
        skill = round(1 - brier / base_brier, 3) if brier is not None else None
        per_model[name] = {
            "file": MODELS[name]["file"], "score_type": stype, "threshold": float(row["threshold"]),
            "validation": {"roc_auc": round(float(roc_auc_score(y_val, s)), 3),
                           "pr_auc": round(float(average_precision_score(y_val, s)), 3),
                           "precision": round(float(row["precision"]), 3),
                           "recall": round(float(row["recall"]), 3),
                           "f1": round(float(row["f1"]), 3),
                           "flagged_share": round(float(row["flagged_share"]), 3),
                           "brier": None if brier is None else round(brier, 3),
                           "brier_skill": skill},
        }
        v = per_model[name]["validation"]
        summary.append({"Model": name, "Score type": stype, "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"],
                        "Frozen threshold": round(float(row["threshold"]), 3),
                        "Flagged share": f"{v['flagged_share']:.1%}", "Precision": v["precision"],
                        "Recall": v["recall"], "F1": v["f1"],
                        "Brier skill": "n/a" if skill is None else skill})
    summary_df = pd.DataFrame(summary)
    print("\nVALIDATION summary with frozen thresholds:")
    print(md_table(summary_df))

    # ---------------- classifier selection ----------------
    best_pr = max(v["validation"]["pr_auc"] for v in per_model.values())
    tied = [n for n, v in per_model.items() if v["validation"]["pr_auc"] >= best_pr - PRAUC_TOLERANCE]
    selected = next(n for n in TIE_BREAK_ORDER if n in tied)
    sel = per_model[selected]
    prob_ok = (sel["score_type"] == "probability" and sel["validation"]["brier_skill"] is not None
               and sel["validation"]["brier_skill"] >= MIN_BRIER_SKILL)
    print(f"\nTied within {PRAUC_TOLERANCE} PR-AUC of the best ({best_pr}): {tied}")
    print(f"SELECTED CLASSIFIER: {selected} | threshold {sel['threshold']:.4f} ({sel['score_type']}) | "
          f"probability shown in dashboard: {prob_ok}")

    # ---------------- freeze ----------------
    hashed_files = [m["file"] for m in MODELS.values()] + [REGRESSION_FILE]
    frozen = {
        "status": "FROZEN - chosen on validation data only; do not change after the test evaluation",
        "data_origin": "SYNTHETIC", "seed": SEED, "benchmark": BENCHMARK, "features": FEATURES,
        "assumptions_to_confirm_with_client": {
            "capacity_max_flagged_share": CAPACITY,
            "beta_recall_weight": BETA,
            "threshold_rule": f"highest F{BETA:g} among thresholds flagging <= {CAPACITY:.0%} of learners; "
                              "a learner is flagged when score >= threshold",
        },
        "selection_rule": f"models within {PRAUC_TOLERANCE} validation PR-AUC of the best are tied; "
                          f"tie-break order {TIE_BREAK_ORDER}",
        "selected_classifier_name": selected,
        "selected_classifier": sel["file"],
        "threshold": sel["threshold"],
        "threshold_type": sel["score_type"],
        "probability_available": bool(prob_ok),
        "probability_rule": f"native probability and validation Brier skill >= {MIN_BRIER_SKILL}",
        "models": per_model,
        "regression_pipeline": REGRESSION_FILE,
        "regression_prediction_rule": "clipped to 0-100",
        "file_sha256": {f: sha256_file(MODELS_DIR / f) for f in hashed_files},
        "test_csv_sha256": sha256_file(PROCESSED_DIR / "test.csv"),
        "scikit_learn_version": sklearn.__version__,
    }
    FROZEN_FILE.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    # ---------------- chart: threshold trade-off for the selected model (validation) ----------------
    sw = sweeps[selected]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sw["threshold"], sw["recall"], label="Recall", color="#c8553d")
    ax.plot(sw["threshold"], sw["precision"], label="Precision", color="#2a7f9e")
    ax.plot(sw["threshold"], sw["flagged_share"], label="Share of learners flagged", color="#6b6b6b")
    ax.axvline(sel["threshold"], color="k", linestyle="--", label=f"Frozen threshold ({sel['threshold']:.3f})")
    ax.axhline(CAPACITY, color="gray", linestyle=":", label=f"Capacity assumption ({CAPACITY:.0%})")
    ax.set_xlabel(f"Threshold ({sel['score_type'].replace('_', ' ')})"); ax.set_ylabel("Value")
    ax.set_title(f"{selected}: threshold trade-off on VALIDATION data (synthetic)")
    ax.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "15_threshold_selection_validation.png", dpi=150); plt.close()

    # ---------------- report ----------------
    report = [
        "# Classifier Selection and Threshold Freeze (Day 7, phase A)", "",
        "**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, "
        "not that it is effective on real learners.", "",
        "All choices below were made on the **validation** split. The test split was not read or "
        "analysed. Once frozen, these settings are not changed after seeing test results.", "",
        "## Assumptions to confirm with the client",
        f"- **Support capacity:** coordinators can support at most **{CAPACITY:.0%}** of learners.",
        f"- **Cost balance:** missing a learner who needs help matters more than a false alarm. "
        f"Recall is weighted {BETA:g}x precision (F{BETA:g}).",
        f"- **Threshold rule:** highest F{BETA:g} among thresholds that flag no more than {CAPACITY:.0%} of "
        "learners. A learner is flagged when score >= threshold.",
        "- If the client's capacity or cost balance differs, change them in `src/final_utils.py` "
        "and redo this step BEFORE the test evaluation.", "",
        "## Validation results with each model's frozen threshold", md_table(summary_df), "",
        "## Selection",
        f"- Best validation PR-AUC: {best_pr}. Models within {PRAUC_TOLERANCE} are treated as tied: "
        f"{', '.join(tied)}.",
        f"- Tie-break: prefer the more interpretable model with native probabilities (order: "
        f"{', '.join(TIE_BREAK_ORDER)}).",
        f"- **Selected: {selected}**, threshold {sel['threshold']:.4f} on the "
        f"{sel['score_type'].replace('_', ' ')}.",
        f"- Probability shown in the dashboard: **{'yes' if prob_ok else 'no (show decision score instead)'}** "
        f"(rule: native probability and validation Brier skill >= {MIN_BRIER_SKILL}; "
        f"baseline Brier = {base_brier:.3f}).",
        "- SVM values are decision scores (signed margins), never probabilities.", "",
        "## Caution",
        f"The threshold was chosen on only {len(y_val)} validation rows, so it carries sampling noise. "
        "Test results show how it behaves on unseen rows from the same synthetic cohort; a real "
        "deployment claim would need later cohorts.", "",
        "Chart: `reports/figures/15_threshold_selection_validation.png`", "",
    ]
    (REPORTS_DIR / "model_selection_report_day7.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nFROZEN. Saved {FROZEN_FILE.name}, reports/model_selection_report_day7.md, chart 15.")
    print("The test set has NOT been used. Next: python src\\evaluate_test.py")


if __name__ == "__main__":
    main()
