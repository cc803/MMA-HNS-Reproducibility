# Comment 7 Cross-Rate Generalization Summary

## Reviewer Focus

Comment 7 asks whether the method generalizes when the model is trained under one text-missing rate but tested under a heavier text-missing rate. The current same-rate results in Table 4 cover 10->10, 30->30, and 50->50. This summary organizes the additional MKG-Y random-mask experiments for 10->30 and 10->50.

## Recommended Scope

- Dataset: MKG-Y.
- Missingness: random entity-level text-missing masks.
- Training missing rate: 10%.
- Test missing rates: 30% and 50%.
- Main methods to report: A, A+B, and A+B+C.
- Metric scale below: percentages, matching Table 4 style.

Do not use `abc_train10_to_test50_seed0.log` in the formal table. It is an old-grid run and is superseded by `abc_train10_to_test50_seed0_widegrid.log`.

## Main Cross-Rate Table

| Train rate | Test rate | Method | Seeds | MRR | H@1 | H@3 | H@10 | Missing-text MRR | Missing-text H@10 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10% | 30% | A | 3 | 31.30 +/- 1.11 | 25.82 +/- 1.01 | 34.35 +/- 1.23 | 40.89 +/- 1.09 | 27.35 +/- 1.97 | 37.75 +/- 2.16 |
| 10% | 30% | A+B | 3 | 33.02 +/- 0.90 | 27.14 +/- 1.16 | 36.63 +/- 0.66 | 42.61 +/- 0.45 | 29.73 +/- 1.59 | 40.33 +/- 0.99 |
| 10% | 30% | A+B+C | 3 | 33.34 +/- 0.87 | 27.22 +/- 1.10 | 36.92 +/- 0.82 | 43.38 +/- 0.37 | 29.64 +/- 1.57 | 40.17 +/- 1.00 |
| 10% | 50% | A | 3 | 27.32 +/- 1.76 | 21.34 +/- 1.61 | 30.56 +/- 1.78 | 38.25 +/- 1.92 | 24.59 +/- 2.33 | 36.13 +/- 2.72 |
| 10% | 50% | A+B | 3 | 30.32 +/- 0.89 | 23.98 +/- 1.07 | 34.08 +/- 0.71 | 41.14 +/- 0.59 | 28.16 +/- 1.21 | 39.75 +/- 0.75 |
| 10% | 50% | A+B+C | 3 | 30.59 +/- 1.01 | 24.10 +/- 1.18 | 34.38 +/- 0.83 | 41.65 +/- 0.55 | 28.14 +/- 1.36 | 39.62 +/- 0.87 |

## Key Takeaways

- The cross-rate setting is substantially harder than same-rate Table 4. A trained at 10% missingness drops from about 37.15 MRR at 10->10 to 31.30 at 10->30 and 27.32 at 10->50.
- Retrieval compensation is useful under train-test missing-rate mismatch. Compared with A, A+B improves MRR by +1.72 points at 10->30 and +3.00 points at 10->50. Missing-text MRR improves by +2.38 and +3.57 points.
- A+B+C is stable in both cross-rate settings. It improves overall MRR/H@10 over A+B at 10->30 (+0.32 MRR and +0.77 H@10) and at 10->50 (+0.27 MRR and +0.51 H@10). Missing-text MRR is comparable to A+B, indicating that HPSAC mainly stabilizes overall ranking while preserving the missing-text subset performance.

## Seed-Level Sources

| Method | Train->Test | Seed | MRR | H@10 | Missing-text MRR | Source |
|---|---:|---:|---:|---:|---:|---|
| A | 10->30 | 0 | 32.41 | 41.83 | 29.04 | `results/mkgy_text_missing/cross_rate_soft10_to_30_with_soft.json` |
| A | 10->30 | 1 | 30.18 | 39.69 | 25.19 | `results/mkgy_text_missing/cross_rate_seed12/test_text10_to_text30_with_soft_seed1.log` |
| A | 10->30 | 2 | 31.32 | 41.14 | 27.81 | `results/mkgy_text_missing/cross_rate_seed12/test_text10_to_text30_with_soft_seed2.log` |
| A+B | 10->30 | 0 | 33.98 | 43.13 | 31.15 | `results/mkgy_text_missing/cross_rate_ab_abc/ab_train10_to_test30_seed0.log` |
| A+B | 10->30 | 1 | 32.19 | 42.40 | 28.01 | `results/mkgy_text_missing/cross_rate_ab_abc/ab_train10_to_test30_seed1.log` |
| A+B | 10->30 | 2 | 32.89 | 42.32 | 30.02 | `results/mkgy_text_missing/cross_rate_ab_abc/ab_train10_to_test30_seed2.log` |
| A+B+C | 10->30 | 0 | 34.26 | 43.80 | 31.07 | `results/mkgy_text_missing/cross_rate_ab_abc/abc_train10_to_test30_seed0_widegrid.log` |
| A+B+C | 10->30 | 1 | 32.53 | 43.24 | 27.96 | `results/mkgy_text_missing/cross_rate_ab_abc/abc_train10_to_test30_seed1_widegrid.log` |
| A+B+C | 10->30 | 2 | 33.22 | 43.11 | 29.89 | `results/mkgy_text_missing/cross_rate_ab_abc/abc_train10_to_test30_seed2_widegrid.log` |
| A | 10->50 | 0 | 28.93 | 39.92 | 26.65 | `results/mkgy_text_missing/cross_rate_soft10_to_50_with_soft.json` |
| A | 10->50 | 1 | 25.44 | 36.14 | 22.06 | `results/mkgy_text_missing/cross_rate_seed12/test_text10_to_text50_with_soft_seed1.log` |
| A | 10->50 | 2 | 27.58 | 38.68 | 25.07 | `results/mkgy_text_missing/cross_rate_seed12/test_text10_to_text50_with_soft_seed2.log` |
| A+B | 10->50 | 0 | 31.34 | 41.70 | 29.52 | `results/mkgy_text_missing/cross_rate_ab_abc/ab_train10_to_test50_seed0.log` |
| A+B | 10->50 | 1 | 29.76 | 41.21 | 27.17 | `results/mkgy_text_missing/cross_rate_ab_abc/ab_train10_to_test50_seed1.log` |
| A+B | 10->50 | 2 | 29.84 | 40.52 | 27.81 | `results/mkgy_text_missing/cross_rate_ab_abc/ab_train10_to_test50_seed2.log` |
| A+B+C | 10->50 | 0 | 31.73 | 42.26 | 29.62 | `results/mkgy_text_missing/cross_rate_ab_abc/abc_train10_to_test50_seed0_widegrid.log` |
| A+B+C | 10->50 | 1 | 29.82 | 41.48 | 26.94 | `results/mkgy_text_missing/cross_rate_ab_abc/abc_train10_to_test50_seed1_widegrid.log` |
| A+B+C | 10->50 | 2 | 30.22 | 41.21 | 27.84 | `results/mkgy_text_missing/cross_rate_ab_abc/abc_train10_to_test50_seed2_widegrid.log` |

## Completeness

All rows in the main cross-rate table now use three seeds.

## Paper Revision Suggestion

Add a short paragraph at the end of Section 4.4, after Table 4:

> To further evaluate robustness under train-test missingness mismatch, we conduct an additional cross-rate evaluation on MKG-Y. Models are trained with 10% randomly masked textual features and directly evaluated under heavier text-missing rates of 30% and 50%, using fixed entity-level masks for each seed. The results in Table A7 show that retrieval-based compensation improves robustness in this setting. Compared with A, A+B improves MRR by 1.72 and 3.00 percentage points under 10->30 and 10->50 evaluation, respectively, and improves missing-text MRR by 2.38 and 3.57 percentage points. The full A+B+C model further improves overall MRR and H@10 over A+B in both cross-rate settings, showing that HPSAC keeps retrieval compensation stable when test-time text missingness is more severe than training-time missingness.

Add a new appendix table after Table A6:

> Table A7. Cross-rate generalization on MKG-Y. Models are trained under 10% random text missingness and evaluated under 30% or 50% text missingness. Results are reported as mean +/- standard deviation over three seeds.

Use the "Main Cross-Rate Table" above as Table A7.

## Rebuttal Draft

> We thank the reviewer for this suggestion. In the revised manuscript, we added a cross-rate generalization analysis on MKG-Y. Specifically, we train the models under 10% random text missingness and evaluate the same checkpoints under heavier 30% and 50% text-missing rates, using fixed entity-level masks for each seed. The results show that the setting is more challenging than same-rate evaluation, but retrieval-based compensation improves robustness under missing-rate mismatch. Compared with A, A+B improves MRR by 1.72 and 3.00 percentage points under 10->30 and 10->50 settings, respectively, and improves missing-text MRR by 2.38 and 3.57 points. The full A+B+C model further improves overall MRR and H@10 over A+B in both cross-rate settings, indicating that HPSAC keeps retrieval compensation stable when test-time text missingness is more severe than training-time missingness. We added these results and the corresponding discussion in Section 4.4 and Appendix A.
