# Held-out test Evaluation (Day 7, phase B)

**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, not that it is effective on real learners.

- Split evaluated: **test** (300 learners with an observed final assessment; 31.7% below the benchmark of 50).
- All models and thresholds were frozen before this evaluation. Nothing was fitted or tuned here, and the file fingerprints were verified.
- Selected classifier: **Logistic regression**, frozen threshold 0.2593 (probability). A learner is flagged for support when the score is at or above the threshold.
- Assumptions behind the threshold (to confirm with the client): capacity 40% of learners, recall weighted 2x precision.
- Excluded and reported separately: 40 withdrawn and 20 missing-assessment enrollments. They were not scored as zero or below benchmark. The model applies to the observed-assessment population only.

## Score prediction (linear regression)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Baseline: training-mean score | 13.75 | 16.97 | -0.0 |
| Linear regression (clipped 0-100) | 7.33 | 9.2 | 0.706 |

MAE and RMSE are in score points. The model improves on the training-mean baseline.

## Classifiers at their frozen thresholds
| Model | Score type | ROC-AUC | PR-AUC | Precision | Recall | F1 | Flagged share | Brier |
|---|---|---|---|---|---|---|---|---|
| Majority-class baseline (always predict 0) | - | 0.5 | 0.317 | 0.0 | 0.0 | 0.0 | 0.0% | - |
| Logistic regression (SELECTED) | probability | 0.889 | 0.789 | 0.624 | 0.821 | 0.709 | 41.7% | 0.125 |
| Decision tree | probability | 0.839 | 0.687 | 0.59 | 0.653 | 0.62 | 35.0% | 0.158 |
| SVM | decision score | 0.887 | 0.787 | 0.634 | 0.821 | 0.716 | 41.0% | n/a |

Prevalence 31.7%; constant-probability Brier reference 0.216. SVM values are decision scores (signed margins), not probabilities; its ROC-AUC and PR-AUC use those scores. The selected model's PR-AUC is above the prevalence baseline.

## Error analysis for the selected classifier

### Outcomes at the frozen threshold
| Outcome | Learners |
|---|---|
| true positive | 78 |
| false negative | 17 |
| false positive | 47 |
| true negative | 158 |

False negatives are learners below the benchmark who were not flagged (missed). False positives were flagged but finished at or above the benchmark.

### Average inputs by outcome group (descriptive only)
| group | n | attendance_pct | quiz_average | assignment_completion_pct | active_learning_minutes | days_since_last_activity | practice_attempt_count |
|---|---|---|---|---|---|---|---|
| true positive | 78 | 55.6 | 44.4 | 48.3 | 159.9 | 4.0 | 3.4 |
| false negative | 17 | 71.9 | 64.0 | 70.2 | 234.5 | 1.8 | 5.6 |
| false positive | 47 | 65.1 | 48.6 | 59.5 | 210.0 | 3.2 | 4.3 |
| true negative | 158 | 81.7 | 71.1 | 82.6 | 309.3 | 1.6 | 7.2 |

### Errors by distance from the benchmark
| Distance from benchmark | Learners | Misclassified | Error rate |
|---|---|---|---|
| within 5 points of benchmark | 67 | 35 | 0.522 |
| 5 to 10 points away | 56 | 18 | 0.321 |
| more than 10 points away | 177 | 11 | 0.062 |

Learners close to the benchmark are the hardest to classify: a learner scoring 48 and one scoring 52 look almost the same to any model.

### By course category
| Course category | Learners | Below benchmark | Score MAE | Recall | Precision | Caution |
|---|---|---|---|---|---|---|
| Data Analytics | 74 | 23 | 8.03 | 0.783 | 0.643 |  |
| Python | 69 | 22 | 6.83 | 0.727 | 0.64 |  |
| Machine Learning | 55 | 31 | 7.51 | 0.903 | 0.718 |  |
| Web Development | 53 | 9 | 6.49 | 0.778 | 0.5 | fewer than 10 below-benchmark learners (recall unreliable) |
| SQL | 49 | 10 | 7.67 | 0.9 | 0.474 |  |

Groups flagged in the Caution column are too small for reliable conclusions.

## Limitations
- Synthetic data: the relationships were built into the generator, so these results cannot be read as effectiveness on real learners.
- One cohort, one split: this measures generalization within the sampled population. A deployment claim requires evaluation on later cohorts.
- Coefficients, tree splits and feature importance are associations, not causes of learning outcomes.
- Predictions support human review and optional learning assistance only. They must not decide admission, certification, grades or access to learning opportunities.

## Charts
- `reports/figures/16_regression_heldout.png`
- `reports/figures/17_classifier_curves_heldout.png`
- `reports/figures/18_confusion_matrices_heldout.png`
- `reports/figures/19_error_analysis_heldout.png`
