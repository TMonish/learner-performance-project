# Logistic Regression and Decision Tree - Validation Results (Day 5)

**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, not that it is effective on real learners.

- Both models use the same preprocessing and the same train/validation split, and were fitted on the training split only. The test split has not been used.
- Positive class: below_benchmark = 1 (benchmark = 50). Probabilities refer to that class.
- Prevalence of the positive class in validation: 31.7%.
- Logistic regression converged in 18 iterations (limit 1000), checked with warnings treated as errors.
- Label metrics use a **provisional** 0.5 cut-off. The support-priority threshold is not chosen yet; it will be chosen on validation data on Day 7 and then frozen before testing.
- ROC-AUC and PR-AUC do not depend on the cut-off.

## Validation comparison
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|
| Majority-class baseline (always predict 0) | 0.5 | 0.317 | 0.0 | 0.0 | 0.0 | 0.683 |
| Constant-probability baseline (prevalence) | 0.5 | 0.317 | 0.0 | 0.0 | 0.0 | 0.683 |
| Logistic regression | 0.918 | 0.861 | 0.812 | 0.684 | 0.743 | 0.85 |
| Decision tree (depth 5, leaf 40) | 0.867 | 0.753 | 0.693 | 0.642 | 0.667 | 0.797 |

## Overfitting check (training vs validation)
| Model | Train ROC-AUC | Validation ROC-AUC | Gap | Train F1 | Validation F1 |
|---|---|---|---|---|---|
| Logistic regression | 0.893 | 0.918 | -0.025 | 0.724 | 0.743 |
| Decision tree (depth 5, leaf 40) | 0.889 | 0.867 | 0.022 | 0.694 | 0.667 |

Decision tree selection: the simplest tree whose validation ROC-AUC was within 0.005 of the best in the grid (best = 0.87), giving max_depth = 5 and min_samples_leaf = 40. Charts: `08_decision_tree_overfitting.png` shows how an unrestricted tree fits training data much better than validation data.

## Tree tuning grid (validation)
| max_depth | min_samples_leaf | train ROC-AUC | val ROC-AUC | val PR-AUC |
|---|---|---|---|---|
| 2 | 1 | 0.824 | 0.82 | 0.655 |
| 2 | 5 | 0.824 | 0.82 | 0.655 |
| 2 | 10 | 0.824 | 0.82 | 0.655 |
| 2 | 20 | 0.824 | 0.82 | 0.655 |
| 2 | 40 | 0.824 | 0.82 | 0.655 |
| 3 | 1 | 0.867 | 0.831 | 0.687 |
| 3 | 5 | 0.865 | 0.828 | 0.685 |
| 3 | 10 | 0.864 | 0.838 | 0.71 |
| 3 | 20 | 0.865 | 0.834 | 0.695 |
| 3 | 40 | 0.86 | 0.852 | 0.716 |
| 4 | 1 | 0.906 | 0.836 | 0.734 |
| 4 | 5 | 0.904 | 0.843 | 0.734 |
| 4 | 10 | 0.902 | 0.852 | 0.75 |
| 4 | 20 | 0.896 | 0.851 | 0.744 |
| 4 | 40 | 0.884 | 0.86 | 0.744 |
| 5 | 1 | 0.927 | 0.83 | 0.744 |
| 5 | 5 | 0.919 | 0.844 | 0.758 |
| 5 | 10 | 0.915 | 0.859 | 0.771 |
| 5 | 20 | 0.906 | 0.864 | 0.77 |
| 5 | 40 | 0.889 | 0.867 | 0.753 |
| 6 | 1 | 0.946 | 0.839 | 0.741 |
| 6 | 5 | 0.937 | 0.841 | 0.754 |
| 6 | 10 | 0.93 | 0.863 | 0.772 |
| 6 | 20 | 0.917 | 0.861 | 0.767 |
| 6 | 40 | 0.892 | 0.87 | 0.757 |

## Logistic regression coefficients (standardized inputs)
Positive values raise the modelled probability of finishing below the benchmark. These are associations in synthetic data, not causes of learning outcomes.

| Input (after preprocessing) | Coefficient |
|---|---|
| quiz_average | -0.84 |
| assignment_completion_pct | -0.75 |
| course_category_Machine Learning | 0.58 |
| days_since_last_activity | 0.51 |
| attendance_pct | -0.45 |
| active_learning_minutes | -0.3 |
| practice_attempt_count | -0.24 |
| course_category_Web Development | -0.19 |
| course_category_Python | -0.16 |
| course_category_Data Analytics | -0.15 |
| missingindicator_quiz_average | 0.13 |
| course_category_SQL | -0.05 |

## Charts
- `reports/figures/07_classifier_curves_validation.png`
- `reports/figures/08_decision_tree_overfitting.png`
- `reports/figures/09_decision_tree_shallow.png` (depth-3 tree for interpretation only)
- `reports/figures/10_confusion_matrices_day5.png`
