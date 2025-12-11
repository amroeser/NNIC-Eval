# NNIC Significance Report

This report summarises paired comparisons between test accuracy before (Acc pre) and after IC application (Acc post) across multiple runs for each experiment.

Note that n_runs is relatively small for most experiments; therefore, classical significance tests have limited power. We primarily interpret the direction and magnitude of the mean accuracy change (ΔAcc) and its confidence interval.

| Experiment | n runs (IC) | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | ΔAcc std | 95% CI low | 95% CI high | Boot 95% CI (B=1000) | Boot 95% CI (B=2000) | t-stat | p (t-test) | W-stat | p (Wilcoxon) |
|-----------|-------------:|---------------:|----------------:|------------:|------------:|---------:|-----------:|------------:|-----------------------:|-----------------------:|-------:|-----------:|-------:|-------------:|
| covertype_47_47_5 | 5 | 0.597 | 0.804 | 0.207 | 34.703 | 0.088 | 0.130 | 0.284 | [0.143, 0.285] | [0.146, 0.285] | 5.276 | 0.006 | 0.000 | 0.062 |
| covertype_72_22_5 | 4 | 0.739 | 0.761 | 0.022 | 3.030 | 0.009 | 0.013 | 0.031 | [0.014, 0.030] | [0.014, 0.030] | 4.819 | 0.017 | 0.000 | 0.125 |
| openml_creditg_47_47_5 | 5 | 0.620 | 0.760 | 0.140 | 22.581 | 0.089 | 0.062 | 0.218 | [0.100, 0.220] | [0.100, 0.220] | 3.500 | 0.025 | 0.000 | 0.062 |
| openml_creditg_72_22_5 | 5 | 0.584 | 0.668 | 0.084 | 14.384 | 0.054 | 0.037 | 0.131 | [0.060, 0.132] | [0.060, 0.132] | 3.500 | 0.025 | 0.000 | 0.062 |
| openml_electricity_47_47_5 | 5 | 0.741 | 0.871 | 0.130 | 17.572 | 0.035 | 0.100 | 0.160 | [0.106, 0.157] | [0.106, 0.157] | 8.437 | 0.001 | 0.000 | 0.062 |
| openml_electricity_72_22_5 | 5 | 0.767 | 0.889 | 0.122 | 15.934 | 0.011 | 0.112 | 0.132 | [0.114, 0.131] | [0.114, 0.131] | 23.971 | 0.000 | 0.000 | 0.062 |
| openml_magic_47_47_5 | 5 | 0.792 | 0.913 | 0.121 | 15.329 | 0.048 | 0.079 | 0.163 | [0.078, 0.150] | [0.079, 0.152] | 5.656 | 0.005 | 0.000 | 0.062 |
| openml_magic_72_22_5 | 5 | 0.814 | 0.919 | 0.105 | 12.865 | 0.042 | 0.068 | 0.142 | [0.067, 0.132] | [0.069, 0.132] | 5.576 | 0.005 | 0.000 | 0.062 |
| openml_phoneme_47_47_5 | 5 | 0.827 | 0.908 | 0.081 | 9.821 | 0.005 | 0.077 | 0.086 | [0.077, 0.086] | [0.077, 0.086] | 34.785 | 0.000 | 0.000 | 0.062 |
| openml_phoneme_72_22_5 | 5 | 0.843 | 0.887 | 0.044 | 5.254 | 0.006 | 0.039 | 0.049 | [0.040, 0.049] | [0.040, 0.049] | 16.971 | 0.000 | 0.000 | 0.062 |
| openml_spambase_47_47_5 | 5 | 0.787 | 0.809 | 0.022 | 2.750 | 0.014 | 0.010 | 0.034 | [0.010, 0.031] | [0.010, 0.031] | 3.536 | 0.024 | 0.000 | 0.125 |
| openml_spambase_72_22_5 | 5 | 0.835 | 0.840 | 0.004 | 0.518 | 0.005 | -0.000 | 0.009 | [0.001, 0.009] | [0.001, 0.009] | 1.826 | 0.142 | 0.000 | 0.250 |

## Summary visualisation of ΔAcc

![Per-experiment mean ΔAcc with bootstrap 95% CIs](plots/significance/delta_acc_forest.png)
