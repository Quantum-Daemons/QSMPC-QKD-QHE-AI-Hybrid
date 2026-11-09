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

MLP [111, 128, 64, 2], ReLU: the 32 per-account AMLworld features plus 79 per-transaction window features, in batches of 2,048

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **published non-linear model** the `finance_aml` page runs under encryption: an MLP on the 32 per-account features of the teacher plus 79 per-transaction
window features, trained on the labels with class weights rather than distilled. Its linear layers run under CKKS and its ReLUs and argmax in garbled circuits, and
its features see transactions up to the end of their batch of 2,048, as the Graph Feature Preprocessor paper processes them. Its scaling and seed were chosen on the validation window,
and the record below carries every seed's figure. `weights.json` also carries 16 held-out rows for checks, under the data's licence.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.997894 |
| `auprc` | 0.305865 |
| `auprc_baseline` | 0.00152346 |
| `auroc` | 0.969699 |
| `batch_window` | 2048 |
| `max_of_seeds_minority_f1` | 0.404806 |
| `median_of_seeds_minority_f1` | 0.369612 |
| `min_of_seeds_minority_f1` | 0.334992 |
| `minority_class_f1` | 0.334992 |
| `n_features` | 111 |
| `n_parameters` | 22722 |
| `n_test` | 1523504 |
| `n_train` | 2843872 |
| `n_val` | 710969 |
| `precision` | 0.322813 |
| `recall` | 0.348126 |
| `seed` | 2 |
| `share_clipped_on_fit_window` | 0.00012958 |
| `teacher_agreement` | 0.998077 |
| `threshold_on_validation` | 0.995822 |
| `wall_clock_s` | 237.1 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17). Same metric and corpus, different protocol: that figure uses a 60/20/20 split by transaction order and IBM's Graph Feature Preprocessor updated over the test window; this project tests on the last 30 percent of transactions (fit 56, validation 14) with its own graph features, whose structural block is fitted on the training window.
- **Published ceiling**: 0.7581 - Multi-FraudGT, the best AML Small HI row of Table 16 ('Minority class F1 scores (%) for six AML datasets') in Bilgi and Atasu, 'MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes', TMLR (08/2026), arXiv:2412.00241v3: 75.81 +/- 0.75. In the same column the paper's own MEGA-PNA scores 74.01 +/- 1.55 and MEGA-GenAgg 74.48 +/- 0.84, and Multi-PNA 67.35 +/- 2.89; Small HI is the one AML dataset of the six where Multi-FraudGT is ahead of MEGA-PNA.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

On HI-Small the published figures are 63.23 for GFP + XGBoost, 62.86 for GFP + LightGBM and 56.77 for PNA (Altman et al., NeurIPS 2023, Table 2), 68.16 for Multi-PNA+EU (Egressy et al., AAAI 2024), and 63.72 for a GNN also evaluated under CKKS (Bilgi et al., arXiv:2610.18054). The Graph Feature Preprocessor paper's own decomposition (Blanusa et al., ICAIF 2024, Table 5) puts basic features at 19.75 and fan-in and fan-out at 56.88 to 59.71 for XGBoost. These 111 features are the 32 per-account aggregates plus per-transaction fan-in and fan-out, vertex statistics and scatter-gather in time windows, in batches of 2048, with no cycle enumeration, on a 56/14/30 temporal split rather than the papers' 60/20/20. The comparison that means most here is with the booster on the same features: AUPRC 0.3059 against 0.3358. This project's published teacher, on the 32 aggregates, has AUPRC 0.1233. AMLworld is synthetic by IBM's own statement.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists. THE TEST WINDOW RUNS PAST THE DATA: HI-Small holds ten days, September 1-10, 2022 (IBM's dataset card), and the last 30 percent of its transactions in time order run to September 18. The 1,108 rows dated September 11-18 are there, the dataset's creator explains, so that laundering patterns begun earlier can complete; 655 of them are laundering, 28.2% of the test window's 2,321 positives. Without them the window's rate is 0.109 percent rather than 0.152, the teacher's minority-class F1 0.1387 rather than 0.1519 and its AUPRC 0.1030 rather than 0.1233, and the CKKS student's F1 0.0473 rather than 0.0614. The published 60/20/20 split's test set carries the same 1,108 rows (docs/measurements/finance-test-window-tail-2026-11-06.json).

## Notes and limitations

The finance page's non-linear model under encryption, the linear layers under CKKS and the ReLUs and argmax in garbled circuits. Trained on the fit window with class weights, on labels rather than distilled; scaling and seed chosen on the validation window, every seed's figure in the record. Its features see transactions up to the end of their batch of 2,048, as the Graph Feature Preprocessor paper processes them. Different from `finance_aml.gnn`, the never-trained Multi-PNA+EU card, which is a target architecture and not a model that runs here.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
