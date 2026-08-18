| Method | Approach | Workload | Fallback lambda | Test MRR (Mean+/-Std) | H@1 | H@3 | H@10 |
|---|---|---|---:|---:|---:|---:|---:|
| Fixed parameters | No group search; use lambda=1.00 and alpha=0.0 for every group. | very_low | 1.00 | 0.368019 +/- 0.000206 | 0.322193 +/- 0.001045 | 0.394042 +/- 0.001563 | 0.441795 +/- 0.001314 |
| Alpha-only calibration | Fix lambda=1.00 and run groupwise Pareto-safe search over alpha. | low | 1.00 | 0.368452 +/- 0.000165 | 0.322506 +/- 0.000782 | 0.394230 +/- 0.001358 | 0.442546 +/- 0.001637 |
| Temperature scaling | Search a global positive T on validation and evaluate score/T. | low | 1.00 | 0.368019 +/- 0.000206 | 0.322193 +/- 0.001045 | 0.394042 +/- 0.001563 | 0.441795 +/- 0.001314 |
