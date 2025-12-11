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

## Interpretation and methodological considerations

Across all experiments, the application of IC consistently improves test accuracy relative to the corresponding baseline models. The mean accuracy differences (ΔAcc) are strictly positive, i.e. the IC-augmented models never underperform their baselines on average. The magnitude of ΔAcc varies across datasets: for *covertype*, *openml_creditg*, *openml_electricity*, *openml_magic* and *openml_phoneme*, we observe sizeable absolute gains (approximately 0.08–0.21 accuracy points), corresponding to relative improvements of roughly 10–30%. In contrast, for *openml_spambase* the effects are comparatively small, with mean improvements on the order of 0.02 and 0.004 absolute points, respectively.

From a statistical inference perspective, the paired design is a central strength of the analysis. By comparing pre- and post-IC performance within the same run, we control for run-level variability and obtain more efficient estimates of ΔAcc than with an unpaired design. For most datasets, the 95% confidence intervals (both analytical and bootstrap-based) lie entirely above zero, indicating that the observed improvements are unlikely to be due solely to random fluctuations, despite the relatively small number of runs (typically n = 5). This is consistent with the paired t-tests, which yield p-values that are mostly well below conventional significance thresholds (α = 0.05). The Wilcoxon signed-rank tests broadly corroborate this pattern, with a few p-values around 0.062–0.125 reflecting the limited power and discreteness of nonparametric tests at very small sample sizes rather than providing substantive evidence against a systematic effect.

In terms of classical quality criteria, **reliability** is supported by multiple random runs per configuration and by comparatively narrow bootstrap confidence intervals for most datasets, suggesting that the observed gains are stable across repetitions. **Internal validity** benefits from the within-run pairing and from the fact that IC is the only systematic difference between pre- and post-conditions, which minimises confounding by other hyperparameter changes. **Statistical conclusion validity** is, however, constrained by the small n, which reduces test power, particularly for small effect sizes as in the *openml_spambase* settings. Non-significant p-values in such cases should therefore be interpreted as inconclusive rather than as evidence for the absence of an effect.

With respect to **external validity**, the experiments cover a diverse set of tabular classification benchmarks, suggesting that the beneficial effect of IC is not confined to a single dataset. Nonetheless, the number of datasets and configurations is finite, and all results are obtained under a specific training pipeline and model family. Generalisation to other data modalities, architectures, or training regimes should therefore be made with caution. Overall, the available evidence indicates that IC yields consistent and, in many cases, substantial accuracy gains under the studied conditions, with the main methodological limitations arising from the small sample sizes and the resulting constraints on statistical power.

Beyond these aspects, additional quality criteria can be considered. **Objectivity** is largely ensured by the fully specified and automated training and evaluation pipeline, which precludes manual tuning or run-specific interventions once the configuration is fixed. **Construct validity** is supported by the use of widely adopted benchmark datasets and standard accuracy as the primary outcome, such that the measured performance plausibly reflects the intended construct of predictive quality for tabular classification. **Ecological validity** is more limited, as benchmark datasets may only partially capture the data characteristics and constraints of real-world deployment scenarios (e.g. non-stationarity, changing cost structures, or domain shift). Finally, **reproducibility and robustness** are facilitated by the use of fixed random seeds and publicly available datasets, but would be further strengthened by independent re-implementations and by systematically stress-testing IC under alternative architectures, hyperparameter settings, and data perturbations (e.g. noise or missing values).

