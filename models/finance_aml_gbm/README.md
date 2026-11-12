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

# Graph-feature gradient boosting

per-account and structural graph features computed by this project (not IBM's Graph Feature Preprocessor) + XGBoost

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **plaintext** model for the `finance_aml` use case. No code path
evaluates this model under encryption; what each use case encrypts, if anything,
is stated in the notes below.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.9991335762820446 |
| `auprc` | 0.6428339350667854 |
| `auprc_baseline` | 0.0015234617040716665 |
| `auroc` | 0.9732365980304583 |
| `batch_reach_minutes` | 17 |
| `decision_threshold` | 0.745127 |
| `minority_class_f1` | 0.6160558464223386 |
| `n_features` | 114 |
| `n_test` | 1523504 |
| `n_train` | 3554841 |
| `positive_rate` | 0.00101943 |
| `wall_clock_s` | 557.4 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17). Same metric and corpus, different protocol: that figure uses a 60/20/20 split by transaction order and IBM's Graph Feature Preprocessor updated over the test window in batches of 128 transactions, where the GFP paper reports 64.77 +/- 0.47 at the batches of 2,048 this project's features use (Blanusa et al., arXiv:2402.08593, Table 4); this project tests on the last 30 percent of transactions (fit 56, validation 14) with its own graph features, whose structural block is fitted on the training window.
- **Published ceiling**: 0.7581 - Multi-FraudGT, the best AML Small HI row of Table 16 ('Minority class F1 scores (%) for six AML datasets') in Bilgi and Atasu, 'MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes', TMLR (08/2026), arXiv:2412.00241v3: 75.81 +/- 0.75. In the same column the paper's own MEGA-PNA scores 74.01 +/- 1.55 and MEGA-GenAgg 74.48 +/- 0.84, and Multi-PNA 67.35 +/- 2.89; Small HI is the one AML dataset of the six where Multi-FraudGT is ahead of MEGA-PNA.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

On this project's split, 56/14/30 by time with nothing read past a transaction's horizon, the 114 inputs reach minority-class F1 0.6161, against the published 0.6323 for GFP + XGBoost on HI-Small (Altman et al., NeurIPS 2023 D&B), measured on the paper's 60/20/20 split at a batch of 128 transactions; at the batch of 2,048 these features use, the GFP paper reports 64.77 (Blanusa et al., arXiv:2402.08593, Table 4). The inputs are the 32 per-account aggregates, 79 per-transaction fan-in, fan-out, vertex-statistic and scatter-gather features in the Graph Feature Preprocessor's style, computed in batches of 2,048 and at most 17 minutes past the transaction, and 3 pair counts; there is no cycle enumeration. Until Round 86 the teacher read the 32 aggregates alone and reached 0.1519, the level of the GFP paper's basic-features row (Table 5: 19.75). Full workings: references/finance-gap-investigation.md.

## Calibration map

The demo page reads this model's score \\(q\\) as a risk through this map before it compares it with a threshold probability, as its decision curve does. Fitted on the validation split and scored on the held-out test split.

$$p = \frac{1}{1 + \exp\!\big(a\,\operatorname{logit}\big(\operatorname{clip}(q,\ \varepsilon,\ 1-\varepsilon)\big) + b\big)}$$

$$\operatorname{logit}(q) = \log\frac{q}{1-q},\qquad \varepsilon = 10^{-12}$$

| quantity | value |
|---|---|
| `a` | -0.650727139917704 |
| `b` | 1.504235103165 |
| fitted on | 710969 validation rows, 801 positive |
| scored on | 1523504 test rows, 2321 positive |
| Brier score, raw to calibrated | 0.00077072 to 0.00082539 |
| log-loss, raw to calibrated | 0.00475425 to 0.00449744 |
| CORP miscalibration (MCB), raw to calibrated | 0.00001292 to 0.0000676 |
| base-rate Brier score (CORP UNC) | 0.00152114 |

Brier skill against the base rate, \\(1 - \mathrm{BS}/\mathrm{UNC}\\), is 0.4574.

Raw is the published graph's float32 P(laundering). It is exactly 0 on 63,274 of the 1,523,504 test rows, 2 of them laundering, where the probability is below float32's resolution; the log-loss clips at 1e-15, so each of those 2 costs 34.5 nats. The training run's booster, which keeps them above zero, scored the raw log-loss 0.00475425 where the graph scores 0.00475425, and the calibrated 0.00449744 where the graph scores 0.00449744; the Brier scores agree to 8 decimals.

Why Platt and not isotonic: Measured in the run that wrote this record, on the teacher's own windows. The map raises the Brier score from 0.00077072 to 0.00082539 and lowers the log-loss from 0.00475425 to 0.00449744 on the test window, where 0.152% of rows are laundering; on the validation rows it was fitted on, 0.113% of them laundering, it lowers the Brier score from 0.00070321 to 0.00061622. Isotonic regression fitted on the same validation rows scores Brier 0.00081774, below Platt's, and log-loss 0.00450874, above Platt's; lower is better on both. Platt is kept as a choice, not as the better score: it is strictly monotone and two floats the browser applies, while isotonic ties the test window's 32,828 distinct scores at 97 values, 63,274 rows at exactly zero, 2 of them laundering, and it is the method of the student's and the networks' records, so the finance records differ by the model and not by the map.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists. THE TEST WINDOW RUNS PAST THE DATA: HI-Small holds ten days, September 1-10, 2022 (IBM's dataset card), and the last 30 percent of its transactions in time order run to September 18. The 1,108 rows dated September 11-18 are there, the dataset's creator explains, so that laundering patterns begun earlier can complete; 655 of them are laundering, 28.2% of the test window's 2,321 positives. Without them the window's rate is 0.109 percent rather than 0.152, the teacher's minority-class F1 0.5499 rather than 0.6161 and its AUPRC 0.5408 rather than 0.6428, and the CKKS student's F1 0.2083 rather than 0.3088. The published 60/20/20 split's test set carries the same 1,108 rows (docs/measurements/finance-test-window-tail-2026-11-11.json).

## Notes and limitations

The browser-tier finance model: it scores one feature row per transaction with no message passing at scoring time. That row is not stateless - its per-account and structural features are computed from the account's earlier transactions - so a deployment needs that history to score a new transaction. It is not evaluated under encryption: the public page encrypts finance_aml.student, distilled from this model, on the 8,192 test-window rows of this model's sample, and a four-feature PaySim surrogate for its one-row CKKS step.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
