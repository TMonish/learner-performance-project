# Split Summary

**Data origin: SYNTHETIC.**

- Random seed: 42
- Rows used (observed final assessment): 1500
- Excluded and reported separately: 60 (40 withdrawn, 20 missing assessment)
- Stratified by `below_benchmark`; one enrollment per learner, so no learner appears in two partitions (asserted)
- The same partitions will be used by all four models
- The test set is not used for tuning, threshold choice or model selection

| Split | Rows | Share | Below benchmark | Prevalence | Mean final score |
|---|---|---|---|---|---|
| train | 900 | 60.0% | 285 | 31.7% | 57.3 |
| validation | 300 | 20.0% | 95 | 31.7% | 57.8 |
| test | 300 | 20.0% | 95 | 31.7% | 57.5 |

The model applies to the observed-assessment population only. Withdrawn and missing-assessment learners are not scored as zero or below benchmark.
