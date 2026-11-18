# Learner Performance Prediction & Early Learning Support Dashboard

Machine Learning project for the EduTech industry (Career Solutions). A 9-day, solo prototype.

> **All data in this project is synthetic.** No real learner records were used. Results show that the
> implementation works; they do not show effectiveness on real learners.

## What it does

Using only a learner's activity from the first two weeks of a six-week online course, the project:

1. Estimates the **final score (0-100)** with linear regression.
2. Estimates whether the learner will finish **below the benchmark (50)** with logistic regression.
3. Compares a **decision tree** and an **SVM** on the same task.
4. Presents everything in a local **Streamlit dashboard** for coordinator review.

Predictions support human review and optional learning assistance only. They must not decide admission,
certification, grades or access to learning opportunities.

## Key results (held-out test set, 300 learners, synthetic data)

| Item | Result |
|---|---|
| Score model (linear regression) | MAE 7.33 points, RMSE 9.20, R2 0.706 (baseline MAE 13.75) |
| Selected classifier | Logistic regression, support threshold 0.259 |
| Selected classifier, ranking | ROC-AUC 0.889, PR-AUC 0.789 (best possible on this data: 0.901 and 0.831) |
| Selected classifier, at threshold | Accuracy 0.787, precision 0.624, recall 0.821, F1 0.709, 41.7% of learners flagged |
| SVM | ROC-AUC 0.887, accuracy 0.793, precision 0.634, recall 0.821 |
| Decision tree | ROC-AUC 0.839, accuracy 0.747, precision 0.590, recall 0.653 |

Full details are in `reports/test_evaluation_report_day7.md` and the final report.

## Folder map

```
app/        Streamlit dashboard (app.py, engine.py, validation.py, theme.py, ui.py, samples.py, paths.py)
data/       synthetic_learner_data.csv and data/processed/ (train, validation, test, excluded rows)
docs/       field_dictionary.md, source_notes.md, input_testing_checklist.md, daily_task_log.md
models/     four saved pipelines (.joblib), frozen_settings.json, final_config.json, test lock file
notebooks/  01_eda.ipynb (exploratory analysis, training split only)
reports/    quality, split, validation and test reports; reports/figures/ charts
src/        data generation, preparation, training, selection and evaluation scripts
tests/      make_sample_csvs.py and test_validation.py
```

## Setup (Windows, PowerShell)

```
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Run order

Run from the project root with the virtual environment active.

| Step | Command | What you should see |
|---|---|---|
| 1 | `python src\generate_synthetic_data.py` | 1,560 rows saved; 31.7% below benchmark. Move the CSV into `data\`. |
| 2 | `python src\prepare_data.py` | 20 validation checks passed; split 900 / 300 / 300 |
| 3 | `python src\train_regression.py` | Validation MAE 7.68 against baseline 13.12 |
| 4 | `python src\train_classifiers.py` | Logistic regression and decision tree validation tables |
| 5 | `python src\train_svm.py` | SVM and three-classifier comparison |
| 6 | `python src\select_and_freeze.py` | Logistic regression selected, threshold 0.2593 |
| 7 | `python src\evaluate_test.py` | One-time test evaluation, then locked |
| 8 | `python tests\make_sample_csvs.py` and `python tests\test_validation.py` | 21 passed, 0 failed |
| 9 | `streamlit run app\app.py` | Dashboard opens in the browser |

Notes:

- `src\evaluate_test.py --rehearsal` runs the same code on the validation split without touching the test set.
- The test set is evaluated **once**. After step 7 a lock file (`models/test_evaluation_lock.json`) makes the
  script refuse a second run. Do not retrain models after freezing; the fingerprint check will stop the run.
- All random seeds are fixed (42), so every step reproduces identical results.

## The dashboard

Four tabs: **Score prediction**, **Learning support**, **Model comparison** and **Batch prediction**.
It loads the saved, frozen models and never trains. Uploads are processed in memory and never saved.
It supports light and dark themes. Validation covers blank entries, out-of-range percentages, negative or
non-integer counts, unknown categories, missing columns, empty files, wrong file types and duplicate IDs.
Work through `docs/input_testing_checklist.md` to test it by hand.

## Limitations

- Synthetic data from one cohort: validate on later cohorts of permissioned, anonymised real data before any
  readiness claim.
- Patterns in the models are associations, not causes of learning outcomes.
- The 40% support-capacity assumption and the recall weighting behind the threshold need client confirmation.
- Withdrawn learners and learners without an assessment (60 records) are reported separately and not scored.
