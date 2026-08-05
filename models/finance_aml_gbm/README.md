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

# Graph-feature-preprocessed gradient boosting

GFP (graph features) + XGBoost

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo. This is the **plaintext** model for the
`finance_aml` use case; the encrypted path runs a distilled student, not this model.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.9983866140161103 |
| `auprc` | 0.07323010834680115 |
| `auprc_baseline` | 0.0015234617040716665 |
| `auroc` | 0.8980183471873905 |
| `decision_threshold` | 0.928426 |
| `minority_class_f1` | 0.08077786088257292 |
| `n_test` | 1523504 |
| `n_train` | 3554841 |
| `positive_rate` | 0.00101943 |
| `wall_clock_s` | 73.7 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 — GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17)
- **Published ceiling**: 0.6816 — Multi-PNA+EU, AAAI-24 vol.38 no.10 pp.11838-11846 (68.16 +/- 2.65)
- **Companion metric shown alongside**: `auprc` — reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

This reaches minority-class F1 well below the published 0.6323. Two differences account for most of it and both are deliberate. (1) The published baseline uses IBM's Graph Feature Preprocessor, which ships in `snapml` under a Proprietary licence and is therefore excluded by this project's commercial-OK-only rule; what is reimplemented here is the documented typology set (degree, fan-in/out, reciprocity, k-hop cycle closure, burst degree, scatter-gather). (2) The corpus injects CYCLE attempts of up to 10 hops (HI-Small_Patterns.txt), while reachability here is computed to 4 hops because the boolean sparse powers grow quickly beyond that. The evaluation is also strictly inductive: graph aggregates are fitted on the training window only, so no test-time edge informs a training-time feature. Reported as measured; not adjusted to close the gap.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists.

## Notes and limitations

Stateless at inference, so this is the browser-tier finance model.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
