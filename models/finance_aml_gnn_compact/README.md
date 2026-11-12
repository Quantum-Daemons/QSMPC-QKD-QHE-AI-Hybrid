---
license: cdla-sharing-1.0
pipeline_tag: tabular-classification
tags:
  - quantum-safe
  - privacy-preserving
  - finance-aml
---

# Compact transaction GNN, under CKKS and garbled circuits

Message passing on a fixed tree: 2 endpoints and 4 neighbour slots per direction at each, a shared per-slot map with no bias and a sum per direction, a second layer per endpoint, beside the 114-feature MLP channel; 608 ReLUs

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **graph model** the `finance_aml` page runs under encryption: message passing on a fixed, public tree of 16 neighbouring edges
per transaction, its linear maps under CKKS and its 608 ReLUs and argmax in garbled circuits, beside an MLP channel over the same 114 inputs as the
page's gradient-boosting teacher. Trained with class weights towards 0.7 times the label plus 0.3 times the probability IBM's Multi-PNA+EU gives the transaction on the same
windows (generalised distillation, Lopez-Paz et al., ICLR 2016; the weight chosen on validation); PNA's neighbourhoods include later
transactions of the training window, which reach this model only through its training targets. Its neighbours and features see transactions up to
the end of their batch of 2,048. The endpoint scaling and seed were chosen on the validation window, and the record below carries every seed's figure.
`weights.json` also carries 16 held-out rows with their neighbourhoods for checks, and `eval_sample.json` the same rows in the layout
of the other models' evaluation samples, with their split and the logits to reproduce, both under the data's licence.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.99903 |
| `auprc` | 0.667649 |
| `auprc_baseline` | 0.00152346 |
| `auroc` | 0.989573 |
| `batch_reach_minutes` | 17 |
| `batch_window` | 2048 |
| `max_of_seeds_minority_f1` | 0.658661 |
| `median_of_seeds_minority_f1` | 0.645966 |
| `min_of_seeds_minority_f1` | 0.640242 |
| `minority_class_f1` | 0.658661 |
| `n_features` | 114 |
| `n_parameters` | 34962 |
| `n_slots_per_direction` | 4 |
| `n_test` | 1523504 |
| `n_train` | 2843872 |
| `n_val` | 710969 |
| `precision` | 0.709806 |
| `recall` | 0.61439 |
| `relu_count` | 608 |
| `seed` | 0 |
| `threshold_on_validation` | 0.919443 |
| `wall_clock_s` | 242.8 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17). Same metric and corpus, different protocol: that figure uses a 60/20/20 split by transaction order and IBM's Graph Feature Preprocessor updated over the test window in batches of 128 transactions, where the GFP paper reports 64.77 +/- 0.47 at the batches of 2,048 this project's features use (Blanusa et al., arXiv:2402.08593, Table 4); this project tests on the last 30 percent of transactions (fit 56, validation 14) with its own graph features, whose structural block is fitted on the training window.
- **Published ceiling**: 0.7581 - Multi-FraudGT, the best AML Small HI row of Table 16 ('Minority class F1 scores (%) for six AML datasets') in Bilgi and Atasu, 'MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes', TMLR (08/2026), arXiv:2412.00241v3: 75.81 +/- 0.75. In the same column the paper's own MEGA-PNA scores 74.01 +/- 1.55 and MEGA-GenAgg 74.48 +/- 0.84, and Multi-PNA 67.35 +/- 2.89; Small HI is the one AML dataset of the six where Multi-FraudGT is ahead of MEGA-PNA.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Calibration map

The demo page shows this model's logits and its decision, not a probability, and reads no score through this map. It states what the score would mean as a risk, for a reader who needs one. Fitted on the validation split and scored on the held-out test split.

$$p = \frac{1}{1 + \exp\!\big(a\,\operatorname{logit}\big(\operatorname{clip}(q,\ \varepsilon,\ 1-\varepsilon)\big) + b\big)}$$

$$\operatorname{logit}(q) = \log\frac{q}{1-q},\qquad \varepsilon = 10^{-12}$$

Here \\(q\\) is the GNN's P(laundering), \\(q = \sigma(z_1 - z_0)\\), the softmax of its two logits \\(z_0, z_1\\). The tuned threshold is folded into the last layer's bias, so the operating point is \\(q = 0.5\\).

| quantity | value |
|---|---|
| `a` | -2.193338071016361 |
| `b` | 0.776467242175192 |
| fitted on | 710969 validation rows, 801 positive |
| scored on | 1523504 test rows, 2321 positive |
| Brier score, raw to calibrated | 0.00210277 to 0.00077607 |
| log-loss, raw to calibrated | 0.01462592 to 0.00388651 |
| CORP miscalibration (MCB), raw to calibrated | 0.00136497 to 0.00003826 |
| base-rate Brier score (CORP UNC) | 0.00152114 |

Brier skill against the base rate, \\(1 - \mathrm{BS}/\mathrm{UNC}\\), is 0.4898.

Raw is the GNN's own softmax probability. The folded threshold puts 0.5 at the operating point and the class weights of its training raise it, so it is not a risk.

Why Platt and not isotonic: Measured on this GNN's windows, not carried from the gradient-boosting teacher's record. Isotonic's Brier 0.00078341 is above Platt's 0.00077607 and its log-loss 0.00397688 is above 0.00388651; lower is better on both. Platt is kept as a choice, not as the better score: it is strictly monotone, so the curve ranks rows as the student does (AUPRC 0.66764883 against the raw 0.66764883, no row tied at the logit clip; isotonic's 0.64986910 comes from the test window's scores tied at 168 values, 1,303,483 of them exactly zero), and it is the method of the teacher's and the student's records, so the four finance records differ by the model and not by the map.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists. THE TEST WINDOW RUNS PAST THE DATA: HI-Small holds ten days, September 1-10, 2022 (IBM's dataset card), and the last 30 percent of its transactions in time order run to September 18. The 1,108 rows dated September 11-18 are there, the dataset's creator explains, so that laundering patterns begun earlier can complete; 655 of them are laundering, 28.2% of the test window's 2,321 positives. Without them the window's rate is 0.109 percent rather than 0.152, the teacher's minority-class F1 0.5499 rather than 0.6161 and its AUPRC 0.5408 rather than 0.6428, and the CKKS student's F1 0.2083 rather than 0.3088. The published 60/20/20 split's test set carries the same 1,108 rows (docs/measurements/finance-test-window-tail-2026-11-11.json).

## Notes and limitations

The finance page's graph model under encryption. Trained on the same windows as the published MLP, with the shape, learning rate, endpoint scaling and seed chosen on validation, and distilled from IBM's Multi-PNA+EU run on those windows (the target 0.7 times the label plus 0.3 times its probability, the weight chosen on validation; `scripts/train/ibm_multignn/` reproduces the teacher). Its neighbours and features see transactions up to the end of their batch of 2,048. It reads 16 edges per transaction, where the official Multi-GNN models read a sample of the whole test-window graph, so the two are different settings. It is not the `finance_aml.gnn` card, which is the official architecture's target.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
