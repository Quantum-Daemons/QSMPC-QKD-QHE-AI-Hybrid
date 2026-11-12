---
license: cdla-sharing-1.0
library_name: onnx
pipeline_tag: tabular-classification
tags:
  - quantum-safe
  - privacy-preserving
  - finance-aml
  - onnx
  - onnxruntime-web
---

# MLP on per-transaction features, under CKKS and garbled circuits

MLP [114, 128, 64, 2], ReLU: the 32 per-account AMLworld features plus 79 per-transaction window features and 3 pair features, in batches of 2,048

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **published non-linear model** the `finance_aml` page runs under encryption: an MLP on the page's gradient-boosting teacher's 114 inputs, 32 per-account features,
79 per-transaction window features and 3 pair features, trained with class weights towards 0.7 times the label plus 0.3 times the probability IBM's
Multi-PNA+EU gives the transaction on the same windows (generalised distillation, Lopez-Paz et al., ICLR 2016; the weight chosen on validation).
Its linear layers run under CKKS and its ReLUs and argmax in garbled circuits, and its features see transactions up to the end of their batch of 2,048,
as the Graph Feature Preprocessor paper processes them. Its scaling and seed were chosen on the validation window, and the record below carries every
seed's figure. `weights.json` also carries 16 held-out rows for checks, under the data's licence.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.998652 |
| `auprc` | 0.643563 |
| `auprc_baseline` | 0.00152346 |
| `auroc` | 0.988564 |
| `batch_reach_minutes` | 17 |
| `batch_window` | 2048 |
| `max_of_seeds_minority_f1` | 0.635851 |
| `median_of_seeds_minority_f1` | 0.598592 |
| `min_of_seeds_minority_f1` | 0.595308 |
| `minority_class_f1` | 0.595308 |
| `n_features` | 114 |
| `n_parameters` | 23106 |
| `n_test` | 1523504 |
| `n_train` | 2843872 |
| `n_val` | 710969 |
| `precision` | 0.548692 |
| `recall` | 0.650582 |
| `seed` | 0 |
| `share_clipped_on_fit_window` | 0.00013282 |
| `teacher_agreement` | 0.998861 |
| `threshold_on_validation` | 0.909758 |
| `wall_clock_s` | 245.8 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17). Same metric and corpus, different protocol: that figure uses a 60/20/20 split by transaction order and IBM's Graph Feature Preprocessor updated over the test window in batches of 128 transactions, where the GFP paper reports 64.77 +/- 0.47 at the batches of 2,048 this project's features use (Blanusa et al., arXiv:2402.08593, Table 4); this project tests on the last 30 percent of transactions (fit 56, validation 14) with its own graph features, whose structural block is fitted on the training window.
- **Published ceiling**: 0.7581 - Multi-FraudGT, the best AML Small HI row of Table 16 ('Minority class F1 scores (%) for six AML datasets') in Bilgi and Atasu, 'MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes', TMLR (08/2026), arXiv:2412.00241v3: 75.81 +/- 0.75. In the same column the paper's own MEGA-PNA scores 74.01 +/- 1.55 and MEGA-GenAgg 74.48 +/- 0.84, and Multi-PNA 67.35 +/- 2.89; Small HI is the one AML dataset of the six where Multi-FraudGT is ahead of MEGA-PNA.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

On HI-Small the published figures are 63.23 for GFP + XGBoost, 62.86 for GFP + LightGBM and 56.77 for PNA (Altman et al., NeurIPS 2023, Table 2), the two GFP figures at a batch of 128 transactions; at the batch of 2,048 these features use, GFP + XGBoost reaches 64.77 (Blanusa et al., ICAIF 2024, Table 4). Graph models reach 68.16 for Multi-PNA+EU (Egressy et al., AAAI 2024) and 63.72 for a GNN also evaluated under CKKS (Bilgi et al., arXiv:2610.18054). The Graph Feature Preprocessor paper's own decomposition (Table 5) puts basic features at 19.75 and fan-in and fan-out at 56.88 to 59.71 for XGBoost. These 114 features are the 32 per-account aggregates plus 3 pair features and per-transaction fan-in and fan-out, vertex statistics and scatter-gather in time windows, in batches of 2048 reaching at most 17 minutes past the transaction, with no cycle enumeration, on a 56/14/30 temporal split rather than the papers' 60/20/20. The comparison that means most here is with the booster on the same features: AUPRC 0.6436 against 0.6368. This project's published teacher, XGBoost with its own settings on the same 114 inputs, has AUPRC 0.6428. AMLworld is synthetic by IBM's own statement.

## Calibration map

The demo page shows this model's logits and its decision, not a probability, and reads no score through this map. It states what the score would mean as a risk, for a reader who needs one. Fitted on the validation split and scored on the held-out test split.

$$p = \frac{1}{1 + \exp\!\big(a\,\operatorname{logit}\big(\operatorname{clip}(q,\ \varepsilon,\ 1-\varepsilon)\big) + b\big)}$$

$$\operatorname{logit}(q) = \log\frac{q}{1-q},\qquad \varepsilon = 10^{-12}$$

Here \\(q\\) is the MLP's P(laundering), \\(q = \sigma(z_1 - z_0)\\), the softmax of its two logits \\(z_0, z_1\\). The tuned threshold is folded into the last layer's bias, so the operating point is \\(q = 0.5\\).

| quantity | value |
|---|---|
| `a` | -2.0908748724881656 |
| `b` | 1.4390414513645886 |
| fitted on | 710969 validation rows, 801 positive |
| scored on | 1523504 test rows, 2321 positive |
| Brier score, raw to calibrated | 0.00277745 to 0.00081516 |
| log-loss, raw to calibrated | 0.01759709 to 0.00404743 |
| CORP miscalibration (MCB), raw to calibrated | 0.00200017 to 0.00003789 |
| base-rate Brier score (CORP UNC) | 0.00152114 |

Brier skill against the base rate, \\(1 - \mathrm{BS}/\mathrm{UNC}\\), is 0.4641.

Raw is the MLP's own softmax probability. The folded threshold puts 0.5 at the operating point and the class weights of its training raise it, so it is not a risk.

Why Platt and not isotonic: Measured on this MLP's windows, not carried from the gradient-boosting teacher's record. Isotonic's Brier 0.00082835 is above Platt's 0.00081516 and its log-loss 0.00445331 is above 0.00404743; lower is better on both. Platt is kept as a choice, not as the better score: it is strictly monotone, so the curve ranks rows as the student does (AUPRC 0.64356317 against the raw 0.64356317, with 3 test rows tied at the logit clip; isotonic's 0.62678417 comes from the test window's scores tied at 245 values, 1,225,466 of them exactly zero), and it is the method of the teacher's and the student's records, so the four finance records differ by the model and not by the map.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists. THE TEST WINDOW RUNS PAST THE DATA: HI-Small holds ten days, September 1-10, 2022 (IBM's dataset card), and the last 30 percent of its transactions in time order run to September 18. The 1,108 rows dated September 11-18 are there, the dataset's creator explains, so that laundering patterns begun earlier can complete; 655 of them are laundering, 28.2% of the test window's 2,321 positives. Without them the window's rate is 0.109 percent rather than 0.152, the teacher's minority-class F1 0.5499 rather than 0.6161 and its AUPRC 0.5408 rather than 0.6428, and the CKKS student's F1 0.2083 rather than 0.3088. The published 60/20/20 split's test set carries the same 1,108 rows (docs/measurements/finance-test-window-tail-2026-11-11.json).

## Notes and limitations

The finance page's non-linear model under encryption, the linear layers under CKKS and the ReLUs and argmax in garbled circuits. Trained on the fit window with class weights, distilled from IBM's Multi-PNA+EU run on the same windows (the target 0.7 times the label plus 0.3 times its probability, the weight chosen on validation); scaling and seed chosen on the validation window, every seed's figure in the record. Its features see transactions up to the end of their batch of 2,048, as the Graph Feature Preprocessor paper processes them. Different from `finance_aml.gnn`, the Multi-PNA+EU card, which is a target architecture: this project ran IBM's code as a teacher and ships no artifact of it.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
