# SVM and Classifier Comparison - Validation Results (Day 6)

**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, not that it is effective on real learners.

- Same preprocessing and same train/validation split as the other classifiers. Fitted on the training split only. The test split has not been used.
- Candidates: a linear kernel and an RBF kernel, each with C in [0.1, 1, 10] and gamma = "scale".
- Selection rule: the simplest candidate (linear before RBF, then smaller C) whose validation ROC-AUC is within 0.005 of the best (0.918). Chosen: **linear, C = 0.1**.
- **The SVM output is a decision score (signed margin), not a probability.** A positive score means the below-benchmark class. ROC-AUC and PR-AUC use this score.
- Label metrics use provisional cut-offs (probability >= 0.5 for logistic regression and the tree; decision score >= 0 for the SVM). The support-priority threshold is chosen on validation data on Day 7 and then frozen.

## SVM candidates (validation)
| kernel | C | train ROC-AUC | val ROC-AUC | val PR-AUC | val F1 (score>=0) |
|---|---|---|---|---|---|
| linear | 0.1 | 0.893 | 0.918 | 0.858 | 0.761 |
| linear | 1.0 | 0.892 | 0.918 | 0.859 | 0.751 |
| linear | 10.0 | 0.892 | 0.918 | 0.859 | 0.751 |
| rbf | 0.1 | 0.877 | 0.903 | 0.837 | 0.72 |
| rbf | 1.0 | 0.893 | 0.884 | 0.834 | 0.752 |
| rbf | 10.0 | 0.948 | 0.852 | 0.792 | 0.713 |

## Three-classifier comparison (validation)
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|
| Majority-class baseline (always predict 0) | 0.5 | 0.317 | 0.0 | 0.0 | 0.0 | 0.683 |
| Logistic regression | 0.918 | 0.861 | 0.812 | 0.684 | 0.743 | 0.85 |
| Decision tree (depth 5, leaf 40) | 0.867 | 0.753 | 0.693 | 0.642 | 0.667 | 0.797 |
| SVM (linear, C=0.1) - decision score | 0.918 | 0.858 | 0.827 | 0.705 | 0.761 | 0.86 |

## SVM overfitting check
| Model | Train ROC-AUC | Validation ROC-AUC | Gap | Train F1 | Validation F1 |
|---|---|---|---|---|---|
| SVM (linear, C=0.1) - decision score | 0.893 | 0.918 | -0.025 | 0.721 | 0.761 |

## Probability calibration (evaluated separately)
A sigmoid-calibrated SVM was fitted with 5-fold cross-validation on training data only. The Brier score measures probability quality (lower is better). The validation set has 300 rows, so calibration curves are rough.

| Probability source | Brier score | ROC-AUC |
|---|---|---|
| Constant baseline (training prevalence) | 0.216 | 0.5 |
| Logistic regression | 0.105 | 0.918 |
| Decision tree (depth 5, leaf 40) | 0.131 | 0.867 |
| SVM, sigmoid-calibrated (5-fold on training data) | 0.108 | 0.917 |

## Charts
- `reports/figures/11_svm_candidates.png`
- `reports/figures/12_three_classifier_curves.png`
- `reports/figures/13_calibration_curves.png`
- `reports/figures/14_confusion_matrices_day6.png`
