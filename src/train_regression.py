"""
Day 4 - Linear regression for final-score prediction (with a training-mean baseline).
Project : Learner Performance Prediction & Early Learning Support Dashboard

DATA IS SYNTHETIC. Results demonstrate the implementation, not effectiveness on real learners.

Rules followed here
  - Fit on the TRAINING split only. Report validation metrics.
  - The TEST split is not opened in this script (it is used once, on Day 7).
  - Predictions are clipped to 0-100 (documented rule). Raw and clipped metrics are both shown.
  - The fitted pipeline is saved and reloaded to prove it reproduces the same predictions.

Usage:  python src/train_regression.py
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")                      # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from pipeline_utils import (BENCHMARK, FEATURES, FIG_DIR, MODELS_DIR, REPORTS_DIR, SCORE_MAX,
                            SCORE_MIN, SEED, build_preprocessor, load_split, predict_score)


def metrics(y_true, y_pred) -> dict:
    return {"MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 2),
            "R2": round(float(r2_score(y_true, y_pred)), 3)}


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _, _ = load_split("train")
    X_val, y_val, _, _ = load_split("validation")
    print(f"Train rows: {len(X_train)} | Validation rows: {len(X_val)} | Features: {FEATURES}")

    # ---------------- baseline: always predict the training-mean score ----------------
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    base_val = baseline.predict(X_val)

    # ---------------- linear regression pipeline ----------------
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())])
    pipe.fit(X_train, y_train)                       # preprocessing is fitted on train only

    train_raw = pipe.predict(X_train)
    val_raw = pipe.predict(X_val)
    train_clip = predict_score(pipe, X_train)
    val_clip = predict_score(pipe, X_val)

    n_out = int(((val_raw < SCORE_MIN) | (val_raw > SCORE_MAX)).sum())

    rows = [
        {"Model": "Baseline: training-mean score", "Data": "validation", **metrics(y_val, base_val)},
        {"Model": "Linear regression (raw)", "Data": "validation", **metrics(y_val, val_raw)},
        {"Model": "Linear regression (clipped 0-100)", "Data": "validation", **metrics(y_val, val_clip)},
        {"Model": "Linear regression (clipped 0-100)", "Data": "training", **metrics(y_train, train_clip)},
    ]
    table = pd.DataFrame(rows)
    print("\n" + md_table(table))
    print(f"\nValidation predictions outside 0-100 before clipping: {n_out} of {len(val_raw)}")

    base_mae = metrics(y_val, base_val)["MAE"]
    lr_mae = metrics(y_val, val_clip)["MAE"]
    beats_baseline = lr_mae < base_mae
    print(f"Beats baseline on validation MAE: {beats_baseline} ({lr_mae} vs {base_mae})")

    # ---------------- coefficients (descriptive, NOT causal) ----------------
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({"Input (after preprocessing)": names,
                          "Coefficient": pipe.named_steps["model"].coef_.round(2)})
    coefs = coefs.reindex(coefs["Coefficient"].abs().sort_values(ascending=False).index)

    # ---------------- charts (validation data) ----------------
    residuals = y_val - val_clip
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(y_val, val_clip, alpha=0.5, s=18, color="#2a7f9e")
    axes[0].plot([0, 100], [0, 100], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual final score"); axes[0].set_ylabel("Predicted final score")
    axes[0].set_title("Actual vs predicted (validation)"); axes[0].legend()
    axes[1].scatter(val_clip, residuals, alpha=0.5, s=18, color="#2a7f9e")
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted final score"); axes[1].set_ylabel("Residual (actual - predicted)")
    axes[1].set_title("Residuals vs predicted (validation)")
    axes[2].hist(residuals, bins=25, color="#2a7f9e")
    axes[2].axvline(0, color="r", linestyle="--")
    axes[2].set_xlabel("Residual (score points)"); axes[2].set_title("Residual distribution (validation)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_linear_regression_validation.png", dpi=150)
    plt.close()

    # ---------------- save pipeline, then prove reload reproduces predictions ----------------
    model_path = MODELS_DIR / "linear_regression_pipeline.joblib"
    joblib.dump(pipe, model_path)
    reloaded = joblib.load(model_path)
    assert np.allclose(predict_score(reloaded, X_val), val_clip), "reloaded pipeline differs!"
    print("\nSaved pipeline reloads and reproduces identical predictions: OK")

    meta = {
        "model": "LinearRegression", "trained_on": "train split only", "seed": SEED,
        "data_origin": "SYNTHETIC", "features": FEATURES, "benchmark": BENCHMARK,
        "prediction_rule": f"clipped to [{SCORE_MIN}, {SCORE_MAX}]",
        "scikit_learn_version": sklearn.__version__,
        "validation_metrics_clipped": metrics(y_val, val_clip),
        "validation_metrics_baseline": metrics(y_val, base_val),
        "beats_baseline": bool(beats_baseline),
    }
    (MODELS_DIR / "linear_regression_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------- report ----------------
    report = [
        "# Linear Regression - Validation Results (Day 4)", "",
        "**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, "
        "not that it is effective on real learners.", "",
        "- Fitted on the training split only. Validation metrics are reported here; "
        "the test split has not been used.",
        f"- Preprocessing (inside the pipeline): median imputation plus a 'quiz missing' indicator for "
        f"`quiz_average`, scaling of numeric inputs, one-hot encoding of `course_category`.",
        f"- Prediction rule: predictions are clipped to {SCORE_MIN}-{SCORE_MAX}. "
        f"{n_out} of {len(val_raw)} validation predictions fell outside that range before clipping.",
        "- MAE and RMSE are in score points. R-squared is unitless.", "",
        "## Metrics", md_table(table), "",
        f"**Beats the training-mean baseline on validation MAE:** {'yes' if beats_baseline else 'NO'} "
        f"({lr_mae} vs {base_mae} score points).", "",
        "## Coefficients (standardized inputs)",
        "These describe how the fitted model weighs inputs in this synthetic data. "
        "They are associations, not causes of learning outcomes.", "",
        md_table(coefs), "",
        "Chart: `reports/figures/06_linear_regression_validation.png`", "",
    ]
    (REPORTS_DIR / "regression_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print("Saved: models/linear_regression_pipeline.joblib, models/linear_regression_metadata.json,")
    print("       reports/regression_validation_report.md, reports/figures/06_linear_regression_validation.png")


if __name__ == "__main__":
    main()
