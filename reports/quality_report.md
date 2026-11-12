# Data Quality Report

**Data origin: SYNTHETIC.** Results on this data demonstrate the implementation, not effectiveness on real learners.

Source file: `synthetic_learner_data.csv` | Rows: 1560 | Benchmark: 50

## Validation checks
| Check | Result | Detail |
|---|---|---|
| File is not empty | PASS | 1560 rows |
| All required columns present | PASS |  |
| Feature list has no ID or outcome fields | PASS | features: ['attendance_pct', 'quiz_average', 'assignment_completion_pct', 'active_learning_minutes', 'days_since_last_activity', 'practice_attempt_count', 'course_category'] |
| enrollment_id is unique | PASS | 0 duplicates |
| learner_group_id is unique (one enrollment per learner) | PASS | 0 duplicates |
| attendance_pct within 0-100 | PASS | 0 out-of-range values |
| quiz_average within 0-100 | PASS | 0 out-of-range values |
| assignment_completion_pct within 0-100 | PASS | 0 out-of-range values |
| final_score within 0-100 | PASS | 0 out-of-range values |
| active_learning_minutes non-negative whole numbers | PASS | 0 negative, 0 non-integer |
| days_since_last_activity non-negative whole numbers | PASS | 0 negative, 0 non-integer |
| practice_attempt_count non-negative whole numbers | PASS | 0 negative, 0 non-integer |
| course_category values are known | PASS |  |
| course_category has no blanks | PASS | 0 blanks |
| No unexpected missing feature values (only quiz_average may be blank) | PASS |  |
| outcome_status values are valid | PASS |  |
| Observed rows all have final_score | PASS | 0 blank |
| Non-observed rows have blank final_score and below_benchmark | PASS |  |
| Stored below_benchmark matches final_score < 50 | PASS | 0 mismatches |
| Both classes present among observed rows | PASS | prevalence 31.7% |

## Missing values by column
| Column | Blank values | Percent |
|---|---|---|
| attendance_pct | 0 | 0.0% |
| quiz_average | 87 | 5.6% |
| assignment_completion_pct | 0 | 0.0% |
| active_learning_minutes | 0 | 0.0% |
| days_since_last_activity | 0 | 0.0% |
| practice_attempt_count | 0 | 0.0% |
| course_category | 0 | 0.0% |
| final_score | 60 | 3.8% |

## Outcome status (withdrawals and missing assessments are reported separately)
| outcome_status | rows |
|---|---|
| observed | 1500 |
| withdrawn | 40 |
| missing_assessment | 20 |

## Quiz average: blank vs genuine zero
- Blank (no quiz completed before cutoff): 87
- Genuine 0.0 scores: 10
- Blanks are kept as blanks. They are imputed later inside the training pipeline, fitted on training data only.
