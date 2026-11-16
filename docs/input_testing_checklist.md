# Input Testing Checklist - Early Learning Support Dashboard

**Data origin: SYNTHETIC.** Fill in the Pass/Fail column yourself while testing.

How to test
1. Activate the virtual environment, then run `python tests/make_sample_csvs.py` once to create `tests/sample_csv/`.
2. Run `streamlit run app/app.py` and open the **Batch prediction** tab.
3. Upload each file below and compare what you see with the expected result.
4. Automated checks: `python tests/test_validation.py` (must report 0 failed).

Row numbers in messages match the file, where the header is row 1.

## A. Batch CSV validation

| # | Case | Sample file | Expected result | Pass/Fail |
|---|---|---|---|---|
| 1 | Valid file | `valid_20_rows.csv` | 20 rows read, 20 ready, 0 need attention. Predictions table and download appear. Blank quiz averages are accepted. | |
| 2 | Blank required entries | `bad_blank_required.csv` | Rows 4, 7 and 9 listed as "required value is blank" (attendance, course category, active minutes). Other rows are still predicted. | |
| 3 | Non-numeric text | `bad_non_numeric.csv` | Rows 5 and 11 listed as "is not a valid number". Not predicted. | |
| 4 | Percentage outside 0-100 | `bad_percent_out_of_range.csv` | Rows 3, 6 and 10 listed as outside 0-100 (120, -5, 101). | |
| 5 | Negative count | `bad_negative_count.csv` | Rows 4 and 8 listed as negative. | |
| 6 | Non-integer count | `bad_non_integer_count.csv` | Rows 5 and 12 listed as "not a whole number". | |
| 7 | Unknown course category | `bad_unknown_category.csv` | Rows 6 and 13 listed as unknown category, with the allowed list. Lowercase "python" is also rejected (exact match). | |
| 8 | Missing columns | `bad_missing_columns.csv` | Whole file refused. Message names `attendance_pct` and `course_category`. | |
| 9 | Empty file | `bad_empty_file.csv` | Whole file refused: "The file is empty." | |
| 10 | Header only | `bad_header_only.csv` | Whole file refused: header but no data rows. | |
| 11 | Wrong file type | `bad_wrong_file_type.txt` | Whole file refused: "Wrong file type. Please upload a .csv file." | |
| 12 | Undecodable file | `bad_undecodable.csv` | Whole file refused with a readable message; the app does not crash. | |
| 13 | Duplicate enrollment IDs | `bad_duplicate_ids.csv` | Rows 3, 7 and 14 ALL flagged as duplicates. None of them is predicted (17 ready). | |
| 14 | Blank enrollment ID | `bad_blank_id.csv` | Row 5 flagged. Not predicted. | |
| 15 | More than 10,000 rows | `bad_too_many_rows.csv` | Whole file refused: more than 10,000 rows. | |
| 16 | File over 5 MB | run `python tests/make_sample_csvs.py --include-large`, upload `bad_too_large_over_5MB.csv` | Refused by the upload limit or by the app. No crash. | |
| 17 | Several problems at once | `mixed_several_bad_rows.csv` | Each bad row listed with its reason. Only valid rows predicted. Order kept. | |
| 18 | Special characters in IDs | `valid_special_characters.csv` | 4 rows ready. Text like `<script>` is shown as plain text, never executed. In `predictions.csv`, IDs beginning with `=`, `+` or `@` start with an apostrophe. | |

## B. Output checks

| # | Check | Expected result | Pass/Fail |
|---|---|---|---|
| 19 | Download predictions | Columns: enrollment_id, predicted_final_score, predicted_support_class, below_benchmark_probability. Same order as the upload. | |
| 20 | Rejected rows excluded | No rejected enrollment ID appears in `predictions.csv`. | |
| 21 | Issue list download | `rows_needing_attention.csv` lists file_row, enrollment_id, column, reason for every problem. | |
| 22 | Row matching | Pick three IDs; their scores in the download match the preview table. | |
| 23 | Single equals batch | Enter the same learner values on the Score prediction tab as one batch row. Score and support result match. | |

## C. Single-learner form

| # | Case | Steps | Expected result | Pass/Fail |
|---|---|---|---|---|
| 24 | Blank quiz | Tick "No quiz completed yet" | Quiz slider greys out. A score still appears. Result differs from entering a quiz score of 0. | |
| 25 | Bounds | Try to push sliders and number boxes past their limits | They stop at 0-100, 0-14 days, 0-20,160 minutes, 0-500 attempts. | |
| 26 | Example presets | Click Highly engaged / Typical / Needs a nudge | Fields change. Needs a nudge shows "May benefit from extra support". | |
| 27 | MAE note | Look under the score | Shows the test MAE in score points and "MAE is an average error, not a confidence interval." | |

## D. Wording, privacy and look

| # | Check | Expected result | Pass/Fail |
|---|---|---|---|
| 28 | Required notices | "Synthetic data" notice and "Predictions support human review..." notice are visible above every tab. | |
| 29 | Supportive wording | Flag text reads "May benefit from extra support" / "On track". No names or emails anywhere. | |
| 30 | Probability labelling | Probability shown only when the selected model provides one; an SVM would show "decision score". | |
| 31 | Theme toggle | Switch Dark mode on and off. Text, cards, inputs, dropdown menus, tabs and the upload area stay readable in both. | |
| 32 | Animations | Score ring and number animate; cards lift on hover; tabs fade in. | |
| 33 | Reduced motion | Turn on "Reduce motion" in the operating system. Animations stop; everything still displays. | |
| 34 | Small screen | Narrow the browser window. Layout stacks into one column without overlapping. | |
| 35 | Frozen model files | Normal start shows no warning. (Optional) after changing a model file, the app warns that files differ from the frozen versions. | |
