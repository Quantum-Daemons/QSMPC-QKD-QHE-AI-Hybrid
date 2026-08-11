---
license: other
license_name: cdla-sharing-1.0
library_name: onnx
pipeline_tag: tabular-classification
tags:
  - quantum-safe
  - privacy-preserving
  - finance-aml
  - onnx
  - onnxruntime-web
---

# CKKS-linear fraud student

logistic regression distilled from the gradient-boosting teacher

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **encrypted-tier** model for the `finance_aml` use case: it is a
distilled student small enough to evaluate under real CKKS, and the demo runs it
that way. It is not the plaintext teacher, and its numbers should be read against
that teacher rather than against the published literature.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.982943 |
| `disagreement_on_positives` | 0.361913 |
| `metric_delta` | 0.090554 |
| `metric_delta_pp` | 9.0554 |
| `n_eval` | 1523504 |
| `student_metric` | 0.061394 |
| `student_params` | 33 |
| `teacher_metric` | 0.151948 |
| `teacher_params` | 141154 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17)
- **Published ceiling**: 0.7401 - MEGA-PNA, arXiv:2412.00241v2 Table 2 'Minority-class F1 scores (%) on AML edge classification task', AML Small HI (74.01 +/- 1.55). PREPRINT - no venue confirmed, which is why it is named as such. It replaces Multi-PNA+EU (AAAI-24, 68.16 +/- 2.65): that row is in the SAME table with two better ones above it (MEGA-GIN 70.83, MEGA-PNA 74.01) and has not been the ceiling since December 2024. Verified by reading the table, not the abstract.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists.

## Notes and limitations

Runs under real CKKS. crypto_fidelity is measured against the teacher every run.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
