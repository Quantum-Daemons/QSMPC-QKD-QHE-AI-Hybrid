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
This is the **encrypted-tier** model for the `finance_aml` use case: a
distilled student small enough to evaluate under CKKS. Whether the public demo
encrypts THIS model differs by use case; the notes below say what actually runs.
It is not the plaintext teacher, and its numbers should be read against that
teacher rather than against the published literature. Its agreement with the
teacher was measured offline, in plaintext.

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
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17). Same metric and corpus, different protocol: that figure uses a 60/20/20 split by transaction order and IBM's Graph Feature Preprocessor updated over the test window; this project tests on the last 30 percent of transactions (fit 56, validation 14) with its own graph features, whose structural block is fitted on the training window.
- **Published ceiling**: 0.7401 - MEGA-PNA, arXiv:2412.00241v2 Table 2 'Minority-class F1 scores (%) on AML edge classification task', AML Small HI (74.01 +/- 1.55); the paper's v3 is accepted for publication in TMLR (2026). It is not the table's highest row: the same table prints MEGA-GenAgg at 74.88 +/- 0.38 and MEGA(PNA)-GIN at 74.69 +/- 0.50, and Multi-PNA (68.16 +/- 2.65) below all three.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists.

## Notes and limitations

Built to run under CKKS, but no code path here encrypts THIS model. The public page's CKKS step evaluates the 8-feature linear model bundled with the PaySim sample and re-scores this 33-parameter AMLworld student in plaintext ONNX; the full deployment's finance pipeline encrypts its own PaySim-trained surrogate. Its agreement with the teacher (`agreement`) is measured offline, at training time, in plaintext; each browser run measures `encrypted_agreement`, CKKS against plaintext for the model it encrypts.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
