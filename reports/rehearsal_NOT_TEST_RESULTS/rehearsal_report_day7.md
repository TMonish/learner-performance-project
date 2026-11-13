# REHEARSAL (validation data) Evaluation (Day 7, phase B)

**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, not that it is effective on real learners.

> **REHEARSAL: these numbers come from the validation split, not the test split.**

- Split evaluated: **validation** (300 learners with an observed final assessment; 31.7% below the benchmark of 50).
- All models and thresholds were frozen before this evaluation. Nothing was fitted or tuned here, and the file fingerprints were verified.
- Selected classifier: **Logistic regression**, frozen threshold 0.2593 (probability). A learner is flagged for support when the score is at or above the threshold.
- Assumptions behind the threshold (to confirm with the client): capacity 40% of learners, recall weighted 2x precision.
- Excluded and reported separately: 40 withdrawn and 20 missing-assessment enrollments. They were not scored as zero or below benchmark. The model applies to the observed-assessment population only.

## Score prediction (linear regression)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Baseline: training-mean score | 13.12 | 16.16 | -0.001 |
| Linear regression (clipped 0-100) | 7.68 | 9.73 | 0.637 |

MAE and RMSE are in score points. The model improves on the training-mean baseline.

## Classifiers at their frozen thresholds
| Model | Score type | ROC-AUC | PR-AUC | Precision | Recall | F1 | Flagged share | Brier |
|---|---|---|---|---|---|---|---|---|
| Majority-class baseline (always predict 0) | - | 0.5 | 0.317 | 0.0 | 0.0 | 0.0 | 0.0% | - |
| Logistic regression (SELECTED) | probability | 0.918 | 0.861 | 0.667 | 0.842 | 0.744 | 40.0% | 0.105 |
| Decision tree | probability | 0.867 | 0.753 | 0.647 | 0.811 | 0.72 | 39.7% | 0.131 |
| SVM | decision score | 0.918 | 0.858 | 0.667 | 0.842 | 0.744 | 40.0% | n/a |

Prevalence 31.7%; constant-probability Brier reference 0.216. SVM values are decision scores (signed margins), not probabilities; its ROC-AUC and PR-AUC use those scores. The selected model's PR-AUC is above the prevalence baseline.

## Error analysis for the selected classifier

### Outcomes at the frozen threshold
| Outcome | Learners |
|---|---|
| true positive | 80 |
| false negative | 15 |
| false positive | 40 |
| true negative | 165 |

False negatives are learners below the benchmark who were not flagged (missed). False positives were flagged but finished at or above the benchmark.

### Average inputs by outcome group (descriptive only)
| group | n | attendance_pct | quiz_average | assignment_completion_pct | active_learning_minutes | days_since_last_activity | practice_attempt_count |
|---|---|---|---|---|---|---|---|
| true positive | 80 | 57.4 | 44.4 | 47.9 | 160.9 | 4.3 | 3.6 |
| false negative | 15 | 71.2 | 55.8 | 75.3 | 276.3 | 2.1 | 6.4 |
| false positive | 40 | 65.8 | 49.4 | 58.5 | 201.4 | 2.8 | 4.2 |
| true negative | 165 | 80.1 | 67.5 | 81.1 | 294.7 | 1.6 | 7.0 |

### Errors by distance from the benchmark
| Distance from benchmark | Learners | Misclassified | Error rate |
|---|---|---|---|
| within 5 points of benchmark | 69 | 25 | 0.362 |
| 5 to 10 points away | 49 | 19 | 0.388 |
| more than 10 points away | 182 | 11 | 0.06 |

Learners close to the benchmark are the hardest to classify: a learner scoring 48 and one scoring 52 look almost the same to any model.

### By course category
| Course category | Learners | Below benchmark | Score MAE | Recall | Precision | Caution |
|---|---|---|---|---|---|---|
| Python | 90 | 24 | 7.89 | 0.958 | 0.676 |  |
| Data Analytics | 79 | 25 | 8.08 | 0.76 | 0.576 |  |
| Web Development | 52 | 14 | 7.15 | 0.857 | 0.571 |  |
| Machine Learning | 44 | 23 | 7.6 | 0.87 | 0.833 |  |
| SQL | 35 | 9 | 7.15 | 0.667 | 0.75 | fewer than 10 below-benchmark learners (recall unreliable) |

Groups flagged in the Caution column are too small for reliable conclusions.

## Limitations
- Synthetic data: the relationships were built into the generator, so these results cannot be read as effectiveness on real learners.
- One cohort, one split: this measures generalization within the sampled population. A deployment claim requires evaluation on later cohorts.
- Coefficients, tree splits and feature importance are associations, not causes of learning outcomes.
- Predictions support human review and optional learning assistance only. They must not decide admission, certification, grades or access to learning opportunities.

## Charts
- `reports/rehearsal/figures/16_regression_heldout.png`
- `reports/rehearsal/figures/17_classifier_curves_heldout.png`
- `reports/rehearsal/figures/18_confusion_matrices_heldout.png`
- `reports/rehearsal/figures/19_error_analysis_heldout.png`
