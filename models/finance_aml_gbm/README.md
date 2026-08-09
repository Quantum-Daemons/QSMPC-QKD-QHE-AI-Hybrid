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
| `accuracy` | 0.9982855312490154 |
| `auprc` | 0.12334101789973353 |
| `auprc_baseline` | 0.0015234617040716665 |
| `auroc` | 0.9525582593503528 |
| `decision_threshold` | 0.386869 |
| `minority_class_f1` | 0.15194805194805194 |
| `n_test` | 1523504 |
| `n_train` | 3554841 |
| `positive_rate` | 0.00101943 |
| `wall_clock_s` | 84.7 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17)
- **Published ceiling**: 0.6816 - Multi-PNA+EU, AAAI-24 vol.38 no.10 pp.11838-11846 (68.16 +/- 2.65)
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

Minority-class F1 0.1519 against the published 0.6323. Round 24 measured the causes rather than assuming them, and then closed the one that was actionable. RULED OUT: the decision threshold (0.6323 needs AUPRC >= ~0.40, so it is unreachable at any threshold - a ranking gap, not a tuning gap) and accounts unseen in the training window (0.39% of test rows). RULED OUT as a main cause: the 4-hop reachability limit against the corpus's 10-hop CYCLE attempts - the Graph Feature Preprocessor paper (arXiv:2402.08593) caps the multi-hop cycle and scatter-gather contribution at 'up to 4%' relative while attributing 'more than 30%' to fan-in/fan-out, which this project already implements. FIXED: the per-account aggregates were fitted with a groupby over the whole training window, so a training row's feature included its own future (median transaction had only 46.4% of its out-degree accrued by its own timestamp) while test rows got the same table as a stale past - the booster was fitted on one distribution and scored on another. Prior-at-time aggregates removed the asymmetry and lifted minority-class F1 from 0.0808 to 0.1519 (+88%), AUPRC 0.0732 to 0.1233 (+68%), AUROC 0.8980 to 0.9526, with NO leakage - verified by truncation, every causal feature being bit-identical whether or not later rows exist. WHAT REMAINS: IBM's Graph Feature Preprocessor ships in snapml under a Proprietary licence and is excluded by this project's commercial-OK-only rule; the mechanism that matters is architectural rather than legal - GFP computes per-transaction subgraph patterns over a graph updated as each transaction arrives, whereas this pipeline computes per-account scalars plus training-window structural features. Full workings: references/finance-gap-investigation.md. Reported as measured; not adjusted to close the gap.

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
