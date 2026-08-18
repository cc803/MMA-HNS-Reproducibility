| Method | Approach | Workload | Test MRR (Mean+/-Std) | H@1 | H@3 | H@10 |
|---|---|---|---:|---:|---:|---:|
| Full HPSAC (Table 5) | Original groupwise search over lambda and alpha. | reference | 0.282490 +/- 0.013031 | 0.219740 +/- 0.014021 | 0.317061 +/- 0.013240 | 0.394668 +/- 0.008357 |
| Fixed parameters | No group search; use lambda=0.25 and alpha=0.0 for every group. | very_low | 0.232857 +/- 0.011511 | 0.178808 +/- 0.007779 | 0.260358 +/- 0.016261 | 0.332770 +/- 0.017264 |
| Alpha-only calibration | Fix lambda=0.25 and run groupwise Pareto-safe search over alpha. | low | 0.261467 +/- 0.012289 | 0.201903 +/- 0.010678 | 0.292027 +/- 0.016071 | 0.371636 +/- 0.008723 |
| Temperature scaling | Search a global positive T on validation and evaluate score/T. | low | 0.232857 +/- 0.011511 | 0.178808 +/- 0.007779 | 0.260358 +/- 0.016261 | 0.332770 +/- 0.017264 |
