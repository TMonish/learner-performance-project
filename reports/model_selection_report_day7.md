# Classifier Selection and Threshold Freeze (Day 7, phase A)

**Data origin: SYNTHETIC.** Results demonstrate that the implementation works, not that it is effective on real learners.

All choices below were made on the **validation** split. The test split was not read or analysed. Once frozen, these settings are not changed after seeing test results.

## Assumptions to confirm with the client
- **Support capacity:** coordinators can support at most **40%** of learners.
- **Cost balance:** missing a learner who needs help matters more than a false alarm. Recall is weighted 2x precision (F2).
- **Threshold rule:** highest F2 among thresholds that flag no more than 40% of learners. A learner is flagged when score >= threshold.
- If the client's capacity or cost balance differs, change them in `src/final_utils.py` and redo this step BEFORE the test evaluation.

## Validation results with each model's frozen threshold
| Model | Score type | ROC-AUC | PR-AUC | Frozen threshold | Flagged share | Precision | Recall | F1 | Brier skill |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | probability | 0.918 | 0.861 | 0.259 | 40.0% | 0.667 | 0.842 | 0.744 | 0.516 |
| Decision tree | probability | 0.867 | 0.753 | 0.365 | 39.7% | 0.647 | 0.811 | 0.72 | 0.397 |
| SVM | decision_score | 0.918 | 0.858 | -0.782 | 40.0% | 0.667 | 0.842 | 0.744 | n/a |

## Selection
- Best validation PR-AUC: 0.861. Models within 0.01 are treated as tied: Logistic regression, SVM.
- Tie-break: prefer the more interpretable model with native probabilities (order: Logistic regression, SVM, Decision tree).
- **Selected: Logistic regression**, threshold 0.2593 on the probability.
- Probability shown in the dashboard: **yes** (rule: native probability and validation Brier skill >= 0.25; baseline Brier = 0.216).
- SVM values are decision scores (signed margins), never probabilities.

## Caution
The threshold was chosen on only 300 validation rows, so it carries sampling noise. Test results show how it behaves on unseen rows from the same synthetic cohort; a real deployment claim would need later cohorts.

Chart: `reports/figures/15_threshold_selection_validation.png`
