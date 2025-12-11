# NNIC Batch Experiments Report

## Metrics

- **Acc pre / Acc post**: Test accuracy of the NN without vs. with NNIC (IC corrections).
- **ΔAcc**: Absolute difference `Acc post - Acc pre`.
- **Enh%**: Relative improvement in test accuracy due to NNIC, `100 * (Acc post - Acc pre) / Acc pre` (only defined if `Acc pre > 0` and ICs are applied).
- **F1 pre / F1 post**: F1-score before vs. after applying ICs.
- **#IC rules**: Number of generated and applied IC rules.
- **frac flagged**: Fraction of test examples for which at least one IC fired and changed the NN prediction (flagging/correction rate).
- **Err pre / Err post / ErrRed%**: Number of errors before vs. after applying ICs and the relative error reduction, `100 * (Err pre - Err post) / Err pre`.
- **Acc/F1 RF (optional)**: Accuracy and F1-score of a random-forest baseline classifier (if configured).

## Green-AI (potential benefit)

The following experiments record, among other things, the number of NN parameters as well as training and IC runtimes. These quantities are indicators of resource consumption but **are not direct energy measurements**. Any Green-AI statements about NNIC therefore refer to a **potential** reduction in compute effort and error rates, not to measured CO₂ emissions.

### Global effect of NNIC (weighted means over all experiments with ICs)

- **Acc pre (weighted, only experiments with ICs)**: 0.635
- **Acc post (weighted, only experiments with ICs)**: 0.817
- **ΔAcc (weighted)**: 0.181
- **Enh% (weighted)**: 28.54%
- **Number of test examples (only experiments with ICs)**: 65640

### Comparison: NNIC with half training data vs. 95% NN baseline (mean over runs)

The following overview compares, for selected datasets, the performance of NNIC with half the training data (47.5% train, 47.5% residual, 5% test; metric: `Acc post (mean)`) to the plain NN baseline with 95% training data (95% train, 0% residual, 5% test; metric: `Acc pre (mean)`).

| Dataset          | NN baseline (95% train, Acc pre mean) | NNIC (47.5/47.5, Acc post mean) | ΔAcc (NNIC - 95% NN) | Rel. ΔAcc vs. 95% NN |
|------------------|----------------------------------------|----------------------------------|----------------------|----------------------|
| covertype        |                                  0.753 |                            0.804 |                 +0.051 |                +6.7% |
| credit-g         |                                  0.692 |                            0.760 |                 +0.068 |                +9.8% |
| electricity      |                                  0.773 |                            0.871 |                 +0.098 |               +12.7% |
| magic-telescope  |                                  0.821 |                            0.913 |                 +0.092 |               +11.2% |
| phoneme          |                                  0.848 |                            0.908 |                 +0.060 |                +7.0% |
| spambase         |                                  0.876 |                            0.809 |                 -0.068 |                -7.7% |

### Overview: IC rules vs. accuracy improvement (all experiments)

![IC rules vs. accuracy improvement – all experiments](../plots/report/all_experiments_ic_effect.png)

## Methodology and interpretation

The batch experiments follow a common protocol across all tabular benchmarks. For each dataset, we consider three data splits with identical test size (5% of the data) but varying proportions of training and residual data: (i) 47.5% train and 47.5% residual, (ii) 72.5% train and 22.5% residual, and (iii) 95% train and 0% residual. The constant 5% test portion is held out in all configurations to provide directly comparable and unbiased estimates of generalisation performance while keeping 95% of the data available for learning (either via gradient-based training or IC induction). The symmetric 47.5/47.5 split represents a regime with minimal gradient-based training data but maximal residual information for rule induction; the 72.5/22.5 split provides an intermediate operating point; and the 95/0 split realises a strong pure-NN baseline where all non-test data are used for gradient-based learning only. For the NNIC configurations, a neural network is first trained solely on the training portion. The residual portion is then used to induce a set of IC rules from the model's errors and the available features. At test time, the trained NN produces initial predictions on the 5% test set, which are subsequently checked and possibly corrected by the IC rules, yielding the post-IC predictions. The baselines with 95% training data correspond to the same NN architecture and training pipeline but without any IC stage (residual portion = 0%).

The report summarises these runs using the metrics defined above, including test accuracy and F1-score before and after ICs, absolute and relative accuracy improvements (ΔAcc, Enh%), the number of IC rules, the fraction of flagged (i.e. corrected) test points, and absolute as well as relative error reductions (Err pre, Err post, ErrRed%). In addition, the section "Global effect of NNIC" aggregates all experiments with ICs via test-size-weighted means, and the table "Comparison: NNIC with half training data vs. 95% NN baseline" compares the post-IC performance of the 47.5/47.5/5.0 setting directly to the 95/0/5.0 NN baseline for selected datasets. The figure "IC rules vs. accuracy improvement – all experiments" provides an overall visual summary of how the number of applied IC rules relates to the achieved accuracy gains across all configurations.

On a global level, NNIC substantially improves the predictive performance of the underlying neural networks. When averaging over all IC configurations with test-size weights, test accuracy increases from 0.635 (pre-IC) to 0.817 (post-IC), corresponding to an absolute gain of 0.181 and a relative improvement of approximately 28.5%. This aggregated effect is reflected in the per-dataset summaries: for most train/residual splits, the application of ICs yields pronounced accuracy gains, sizeable reductions in the number of test errors, and consistent improvements in F1-score. At the same time, the number of induced rules and the fraction of flagged test examples remain moderate, indicating that relatively compact rule sets are sufficient to correct a substantial portion of the baseline model's errors (cf. the global IC-effect figure above).

A central result from a data-efficiency perspective is the explicit comparison between NNIC with half of the available training data and a plain NN with 95% training data. The corresponding table shows that, for covertype, credit-g, electricity, magic-telescope and phoneme, the configuration with 47.5% train, 47.5% residual and 5% test achieves higher mean post-IC accuracy than the 95% NN baseline (measured via its pre-IC accuracy on the 95/0/5.0 split). The absolute gains range from +0.051 to +0.098 accuracy points, corresponding to relative improvements between roughly +6.7% and +12.7% over a considerably more data-hungry baseline model. Only for spambase does NNIC with half the training data underperform the 95% baseline (−0.068 absolute, −7.7% relative), which is plausible given the already very high baseline accuracy and the resulting limited headroom for further improvements.

The detailed per-dataset sections further illustrate these patterns. For covertype, NNIC leads to large absolute and relative improvements in accuracy and F1-score, particularly in the 47.5/47.5/5.0 setting, where a moderate number of IC rules with a relatively high flagging rate corrects a substantial fraction of the baseline errors. For credit-g, accuracy and F1-score increase across both residual-based settings, again with noticeable error reductions and modest rule set sizes. Electricity and magic-telescope exhibit particularly strong gains, where ICs can correct a large share of misclassifications despite only a small fraction of test instances being flagged, as visualised in the corresponding accuracy and error plots. Phoneme shows consistent, though somewhat smaller, improvements, with ICs mainly refining an already strong baseline. For spambase, the per-experiment tables and plots reveal that ICs still yield positive ΔAcc when comparing pre- vs. post-IC performance at fixed training fractions, but the 95% NN baseline remains the strongest configuration overall, highlighting a dataset where NNIC brings limited additional benefit in the half-data regime.

From a methodological viewpoint, these results support the interpretation that a substantial portion of the NN's residual errors can be captured by a relatively small number of interpretable IC rules, especially when a dedicated residual set is available. This enables a trade-off between gradient-based training data and rule-based correction capacity: on several benchmarks, the combination of a smaller NN training set with an IC stage not only matches but exceeds the performance of a much more data-demanding baseline. In a paper, the global weighted averages, the half-vs-95% comparison table and a small selection of representative plots (e.g. accuracy/error plots for one or two datasets together with the global IC-effect figure) would suffice to convey the main empirical message without reproducing the full set of visualisations shown in this report.

## Dataset: covertype

For *covertype*, all three splits are evaluated to study how much of the large dataset can be reallocated from gradient-based training to IC induction without degrading performance. The 47.5/47.5/5.0 configuration in particular probes a regime with relatively little training data but a large residual set, enabling dense observation of the NN's error patterns. The tables below report accuracy, error counts and F1-scores across splits, while the accompanying covertype accuracy and error plots, together with the IC-effect plot, visualise how ICs leverage the residual data to improve performance at test time.

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre | Acc post | ΔAcc | Enh% | Err pre | Err post | ErrRed% | F1 pre | F1 post | #IC rules | frac flagged | Acc RF | F1 RF |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------:|---------:|-----:|------:|--------:|---------:|--------:|-------:|--------:|----------:|-------------:|-------:|------:|
| 47.500 | 47.500 | 5.000 | 275980 | 275981 | 29051 | covertype_47_47_5 | 0.494 | 0.838 | 0.344 | 69.491 | 14688 | 4707 | 67.953 | 0.425 | 0.787 | 10 | 0.482 | - | - |
| 72.500 | 22.500 | 5.000 | 421233 | 130728 | 29051 | covertype_72_22_5 | 0.740 | 0.770 | 0.030 | 3.996 | 7552 | 6693 | 11.374 | 0.728 | 0.756 | 2 | 0.076 | - | - |
| 95.000 | 0.000 | 5.000 | 551961 | 0 | 29051 | covertype_95_5 | 0.760 | - | - | - | 6976 | - | - | 0.749 | - | 0 | 0.000 | - | - |

### Mittelwerte über Runs (sofern n_runs > 1)

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | F1 pre (mean) | F1 post (mean) | #IC rules (mean) | frac flagged (mean) |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------------:|----------------:|------------:|------------:|--------------:|---------------:|------------------:|--------------------:|
| 47.500 | 47.500 | 5.000 | 275980 | 275981 | 29051 | covertype_47_47_5 | 0.597 | 0.804 | 0.207 | 34.703 | 0.538 | 0.773 | 8.200 | 0.364 |
| 72.500 | 22.500 | 5.000 | 421233 | 130728 | 29051 | covertype_72_22_5 | 0.742 | 0.761 | 0.020 | 2.651 | 0.730 | 0.748 | 1.200 | 0.048 |
| 95.000 | 0.000 | 5.000 | 551961 | 0 | 29051 | covertype_95_5 | 0.753 | - | - | - | 0.743 | - | 0.000 | 0.000 |

![Accuracy pre/post with NNIC for covertype](../plots/report/covertype_acc.png)

![Errors before/after NNIC for covertype](../plots/report/covertype_errors.png)

![Effect of IC rules for covertype](../plots/report/covertype_ic_effect.png)

![Fehler vor/nach NNIC für covertype](../plots/report/covertype_errors.png)

![Effekt der IC-Regeln für covertype](../plots/report/covertype_ic_effect.png)

## Dataset: credit-g

For *credit-g*, the same three-way split design is applied to a substantially smaller dataset, which makes the trade-off between training and residual data particularly delicate. Here, the 47.5/47.5/5.0 and 72.5/22.5/5.0 configurations investigate whether ICs can compensate for the reduced number of gradient-based updates by exploiting a larger residual set. The tables and the corresponding credit-g accuracy, error and IC-effect plots illustrate how, even under these data constraints, NNIC can reduce the number of errors and improve F1-score compared to the plain NN.

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre | Acc post | ΔAcc | Enh% | Err pre | Err post | ErrRed% | F1 pre | F1 post | #IC rules | frac flagged | Acc RF | F1 RF |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------:|---------:|-----:|------:|--------:|---------:|--------:|-------:|--------:|----------:|-------------:|-------:|------:|
| 47.500 | 47.500 | 5.000 | 475 | 475 | 50 | openml_creditg_47_47_5 | 0.700 | 0.800 | 0.100 | 14.286 | 15 | 10 | 33.333 | 0.576 | 0.762 | 7 | 0.100 | - | - |
| 72.500 | 22.500 | 5.000 | 725 | 225 | 50 | openml_creditg_72_22_5 | 0.700 | 0.760 | 0.060 | 8.571 | 15 | 12 | 20.000 | 0.576 | 0.698 | 3 | 0.060 | - | - |
| 95.000 | 0.000 | 5.000 | 950 | 0 | 50 | openml_creditg_95_5 | 0.700 | - | - | - | 15 | - | - | 0.576 | - | 0 | 0.000 | - | - |

### Mittelwerte über Runs (sofern n_runs > 1)

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | F1 pre (mean) | F1 post (mean) | #IC rules (mean) | frac flagged (mean) |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------------:|----------------:|------------:|------------:|--------------:|---------------:|------------------:|--------------------:|
| 47.500 | 47.500 | 5.000 | 475 | 475 | 50 | openml_creditg_47_47_5 | 0.620 | 0.760 | 0.140 | 22.581 | 0.489 | 0.730 | 7.200 | 0.140 |
| 72.500 | 22.500 | 5.000 | 725 | 225 | 50 | openml_creditg_72_22_5 | 0.584 | 0.668 | 0.084 | 14.384 | 0.488 | 0.630 | 4.200 | 0.084 |
| 95.000 | 0.000 | 5.000 | 950 | 0 | 50 | openml_creditg_95_5 | 0.692 | - | - | - | 0.589 | - | 0.000 | 0.000 |

![Accuracy pre/post with NNIC for credit-g](../plots/report/credit-g_acc.png)

![Errors before/after NNIC for credit-g](../plots/report/credit-g_errors.png)

![Effect of IC rules for credit-g](../plots/report/credit-g_ic_effect.png)

![Fehler vor/nach NNIC für credit-g](../plots/report/credit-g_errors.png)

![Effekt der IC-Regeln für credit-g](../plots/report/credit-g_ic_effect.png)

## Dataset: electricity

For *electricity*, the experimental design again contrasts the three splits with a fixed 5% test set. Given the temporal and potentially non-stationary nature of this dataset, the residual portion is particularly important for capturing systematic error patterns of the NN that may not be fully addressed by gradient-based training alone. The tables and the electricity accuracy/error plots show that ICs can correct a large fraction of misclassifications, especially in the 47.5/47.5/5.0 and 72.5/22.5/5.0 settings, while the IC-effect plot summarises how many rules are required to obtain these gains.

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre | Acc post | ΔAcc | Enh% | Err pre | Err post | ErrRed% | F1 pre | F1 post | #IC rules | frac flagged | Acc RF | F1 RF |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------:|---------:|-----:|------:|--------:|---------:|--------:|-------:|--------:|----------:|-------------:|-------:|------:|
| 47.500 | 47.500 | 5.000 | 21523 | 21523 | 2266 | openml_electricity_47_47_5 | 0.735 | 0.918 | 0.183 | 24.910 | 600 | 185 | 69.167 | 0.727 | 0.919 | 9 | 0.261 | - | - |
| 72.500 | 22.500 | 5.000 | 32850 | 10196 | 2266 | openml_electricity_72_22_5 | 0.767 | 0.906 | 0.139 | 18.135 | 529 | 214 | 59.546 | 0.763 | 0.905 | 6 | 0.224 | - | - |
| 95.000 | 0.000 | 5.000 | 43046 | 0 | 2266 | openml_electricity_95_5 | 0.778 | - | - | - | 503 | - | - | 0.777 | - | 0 | 0.000 | - | - |

### Mittelwerte über Runs (sofern n_runs > 1)

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | F1 pre (mean) | F1 post (mean) | #IC rules (mean) | frac flagged (mean) |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------------:|----------------:|------------:|------------:|--------------:|---------------:|------------------:|--------------------:|
| 47.500 | 47.500 | 5.000 | 21523 | 21523 | 2266 | openml_electricity_47_47_5 | 0.741 | 0.871 | 0.130 | 17.572 | 0.738 | 0.870 | 9.000 | 0.256 |
| 72.500 | 22.500 | 5.000 | 32850 | 10196 | 2266 | openml_electricity_72_22_5 | 0.767 | 0.889 | 0.122 | 15.934 | 0.765 | 0.889 | 5.800 | 0.225 |
| 95.000 | 0.000 | 5.000 | 43046 | 0 | 2266 | openml_electricity_95_5 | 0.773 | - | - | - | 0.771 | - | 0.000 | 0.000 |

![Accuracy pre/post with NNIC for electricity](../plots/report/electricity_acc.png)

![Errors before/after NNIC for electricity](../plots/report/electricity_errors.png)

![Effect of IC rules for electricity](../plots/report/electricity_ic_effect.png)

![Fehler vor/nach NNIC für electricity](../plots/report/electricity_errors.png)

![Effekt der IC-Regeln für electricity](../plots/report/electricity_ic_effect.png)

## Dataset: magic-telescope

For *magic-telescope*, the three splits allow us to analyse whether NNIC can maintain or improve performance on a medium-sized benchmark when a substantial fraction of the data is reserved for IC induction. The 47.5/47.5/5.0 configuration tests the extreme half-data regime, while 72.5/22.5/5.0 provides an intermediate point between that and the 95/0/5.0 baseline. The tables, together with the magic-telescope accuracy and error plots and the IC-effect plot, demonstrate that ICs can substantially reduce the number of errors even when the NN itself is trained on considerably fewer examples.

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre | Acc post | ΔAcc | Enh% | Err pre | Err post | ErrRed% | F1 pre | F1 post | #IC rules | frac flagged | Acc RF | F1 RF |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------:|---------:|-----:|------:|--------:|---------:|--------:|-------:|--------:|----------:|-------------:|-------:|------:|
| 47.500 | 47.500 | 5.000 | 9034 | 9035 | 951 | openml_magic_47_47_5 | 0.800 | 0.952 | 0.151 | 18.922 | 190 | 46 | 75.789 | 0.787 | 0.952 | 4 | 0.186 | - | - |
| 72.500 | 22.500 | 5.000 | 13789 | 4280 | 951 | openml_magic_72_22_5 | 0.810 | 0.953 | 0.143 | 17.662 | 181 | 45 | 75.138 | 0.797 | 0.953 | 7 | 0.168 | - | - |
| 95.000 | 0.000 | 5.000 | 18069 | 0 | 951 | openml_magic_95_5 | 0.832 | - | - | - | 160 | - | - | 0.825 | - | 0 | 0.000 | - | - |

### Mittelwerte über Runs (sofern n_runs > 1)

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | F1 pre (mean) | F1 post (mean) | #IC rules (mean) | frac flagged (mean) |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------------:|----------------:|------------:|------------:|--------------:|---------------:|------------------:|--------------------:|
| 47.500 | 47.500 | 5.000 | 9034 | 9035 | 951 | openml_magic_47_47_5 | 0.792 | 0.913 | 0.121 | 15.329 | 0.780 | 0.909 | 4.200 | 0.193 |
| 72.500 | 22.500 | 5.000 | 13789 | 4280 | 951 | openml_magic_72_22_5 | 0.814 | 0.919 | 0.105 | 12.865 | 0.807 | 0.919 | 5.200 | 0.147 |
| 95.000 | 0.000 | 5.000 | 18069 | 0 | 951 | openml_magic_95_5 | 0.821 | - | - | - | 0.814 | - | 0.000 | 0.000 |

![Accuracy pre/post with NNIC for magic-telescope](../plots/report/magic-telescope_acc.png)

![Errors before/after NNIC for magic-telescope](../plots/report/magic-telescope_errors.png)

![Effect of IC rules for magic-telescope](../plots/report/magic-telescope_ic_effect.png)

![Fehler vor/nach NNIC für magic-telescope](../plots/report/magic-telescope_errors.png)

![Effekt der IC-Regeln für magic-telescope](../plots/report/magic-telescope_ic_effect.png)

## Dataset: phoneme

For *phoneme*, the split design examines whether ICs can further refine an already strong NN baseline. Because the dataset is relatively small and the baseline accuracies are high, reallocating data from training to residual poses a stringent test of NNIC's ability to extract additional structure from residual errors. The tables and the phoneme accuracy/error and IC-effect plots show that, across both residual-based splits, ICs provide consistent gains in accuracy and F1-score while relying on a moderate number of rules.

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre | Acc post | ΔAcc | Enh% | Err pre | Err post | ErrRed% | F1 pre | F1 post | #IC rules | frac flagged | Acc RF | F1 RF |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------:|---------:|-----:|------:|--------:|---------:|--------:|-------:|--------:|----------:|-------------:|-------:|------:|
| 47.500 | 47.500 | 5.000 | 2566 | 2567 | 271 | openml_phoneme_47_47_5 | 0.838 | 0.923 | 0.085 | 10.132 | 44 | 21 | 52.273 | 0.838 | 0.923 | 7 | 0.096 | - | - |
| 72.500 | 22.500 | 5.000 | 3917 | 1216 | 271 | openml_phoneme_72_22_5 | 0.852 | 0.904 | 0.052 | 6.061 | 40 | 26 | 35.000 | 0.851 | 0.904 | 9 | 0.077 | - | - |
| 95.000 | 0.000 | 5.000 | 5133 | 0 | 271 | openml_phoneme_95_5 | 0.860 | - | - | - | 38 | - | - | 0.862 | - | 0 | 0.000 | - | - |

### Mittelwerte über Runs (sofern n_runs > 1)

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | F1 pre (mean) | F1 post (mean) | #IC rules (mean) | frac flagged (mean) |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------------:|----------------:|------------:|------------:|--------------:|---------------:|------------------:|--------------------:|
| 47.500 | 47.500 | 5.000 | 2566 | 2567 | 271 | openml_phoneme_47_47_5 | 0.827 | 0.908 | 0.081 | 9.821 | 0.826 | 0.908 | 7.600 | 0.103 |
| 72.500 | 22.500 | 5.000 | 3917 | 1216 | 271 | openml_phoneme_72_22_5 | 0.843 | 0.887 | 0.044 | 5.254 | 0.844 | 0.888 | 8.400 | 0.068 |
| 95.000 | 0.000 | 5.000 | 5133 | 0 | 271 | openml_phoneme_95_5 | 0.848 | - | - | - | 0.848 | - | 0.000 | 0.000 |

![Accuracy pre/post with NNIC for phoneme](../plots/report/phoneme_acc.png)

![Errors before/after NNIC for phoneme](../plots/report/phoneme_errors.png)

![Effect of IC rules for phoneme](../plots/report/phoneme_ic_effect.png)

![Fehler vor/nach NNIC für phoneme](../plots/report/phoneme_errors.png)

![Effekt der IC-Regeln für phoneme](../plots/report/phoneme_ic_effect.png)

## Dataset: spambase

For *spambase*, the three splits again differ only in the allocation between training and residual data at a fixed 5% test set. Here, the 95/0/5.0 baseline already achieves very high accuracy, leaving limited room for further improvement. The per-split tables and the spambase accuracy/error and IC-effect plots make clear that, while ICs still achieve positive ΔAcc within a given split (e.g. 47.5/47.5/5.0 vs. its own pre-IC baseline), the 95%-train NN remains the strongest overall configuration. This dataset thus serves as an informative counter-example where NNIC offers only marginal additional benefit in the half-data regime.

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre | Acc post | ΔAcc | Enh% | Err pre | Err post | ErrRed% | F1 pre | F1 post | #IC rules | frac flagged | Acc RF | F1 RF |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------:|---------:|-----:|------:|--------:|---------:|--------:|-------:|--------:|----------:|-------------:|-------:|------:|
| 47.500 | 47.500 | 5.000 | 2185 | 2185 | 231 | openml_spambase_47_47_5 | 0.818 | 0.848 | 0.030 | 3.704 | 42 | 35 | 16.667 | 0.814 | 0.847 | 6 | 0.035 | - | - |
| 72.500 | 22.500 | 5.000 | 3335 | 1035 | 231 | openml_spambase_72_22_5 | 0.857 | 0.870 | 0.013 | 1.515 | 33 | 30 | 9.091 | 0.855 | 0.868 | 5 | 0.017 | - | - |
| 95.000 | 0.000 | 5.000 | 4370 | 0 | 231 | openml_spambase_95_5 | 0.896 | - | - | - | 24 | - | - | 0.897 | - | 0 | 0.000 | - | - |

### Mittelwerte über Runs (sofern n_runs > 1)

| Train% | Residual% | Test% | Train_n | Residual_n | Test_n | Exp. Name | Acc pre (mean) | Acc post (mean) | ΔAcc (mean) | Enh% (mean) | F1 pre (mean) | F1 post (mean) | #IC rules (mean) | frac flagged (mean) |
|--------|-----------|-------|--------:|-----------:|-------:|-----------|--------------:|----------------:|------------:|------------:|--------------:|---------------:|------------------:|--------------------:|
| 47.500 | 47.500 | 5.000 | 2185 | 2185 | 231 | openml_spambase_47_47_5 | 0.787 | 0.809 | 0.022 | 2.750 | 0.776 | 0.801 | 6.000 | 0.023 |
| 72.500 | 22.500 | 5.000 | 3335 | 1035 | 231 | openml_spambase_72_22_5 | 0.835 | 0.840 | 0.004 | 0.518 | 0.834 | 0.838 | 4.600 | 0.007 |
| 95.000 | 0.000 | 5.000 | 4370 | 0 | 231 | openml_spambase_95_5 | 0.876 | - | - | - | 0.877 | - | 0.000 | 0.000 |

![Accuracy pre/post with NNIC for spambase](../plots/report/spambase_acc.png)

![Errors before/after NNIC for spambase](../plots/report/spambase_errors.png)

![Effect of IC rules for spambase](../plots/report/spambase_ic_effect.png)

![Fehler vor/nach NNIC für spambase](../plots/report/spambase_errors.png)

![Effekt der IC-Regeln für spambase](../plots/report/spambase_ic_effect.png)
