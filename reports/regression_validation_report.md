# Linear Regression - Validation Results (Day 4)

**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, not that it is effective on real learners.

- Fitted on the training split only. Validation metrics are reported here; the test split has not been used.
- Preprocessing (inside the pipeline): median imputation plus a 'quiz missing' indicator for `quiz_average`, scaling of numeric inputs, one-hot encoding of `course_category`.
- Prediction rule: predictions are clipped to 0-100. 0 of 300 validation predictions fell outside that range before clipping.
- MAE and RMSE are in score points. R-squared is unitless.

## Metrics
| Model | Data | MAE | RMSE | R2 |
|---|---|---|---|---|
| Baseline: training-mean score | validation | 13.12 | 16.16 | -0.001 |
| Linear regression (raw) | validation | 7.68 | 9.73 | 0.637 |
| Linear regression (clipped 0-100) | validation | 7.68 | 9.73 | 0.637 |
| Linear regression (clipped 0-100) | training | 7.55 | 9.39 | 0.66 |

**Beats the training-mean baseline on validation MAE:** yes (7.68 vs 13.12 score points).

## Coefficients (standardized inputs)
These describe how the fitted model weighs inputs in this synthetic data. They are associations, not causes of learning outcomes.

| Input (after preprocessing) | Coefficient |
|---|---|
| numeric__assignment_completion_pct | 5.2 |
| quiz__quiz_average | 4.57 |
| numeric__attendance_pct | 2.85 |
| category__course_category_Machine Learning | -2.68 |
| numeric__days_since_last_activity | -2.45 |
| category__course_category_SQL | 2.07 |
| numeric__active_learning_minutes | 0.85 |
| category__course_category_Web Development | 0.57 |
| numeric__practice_attempt_count | 0.5 |
| category__course_category_Python | 0.36 |
| quiz__missingindicator_quiz_average | -0.33 |
| category__course_category_Data Analytics | -0.32 |

Chart: `reports/figures/06_linear_regression_validation.png`
