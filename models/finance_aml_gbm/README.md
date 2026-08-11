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
a quantum-safe orchestration demo.
This is the **plaintext** model for the `finance_aml` use case; the encrypted
path runs a distilled student, not this model.

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
| `wall_clock_s` | 96.8 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17)
- **Published ceiling**: 0.7401 - MEGA-PNA, arXiv:2412.00241v2 Table 2 'Minority-class F1 scores (%) on AML edge classification task', AML Small HI (74.01 +/- 1.55). PREPRINT - no venue confirmed, which is why it is named as such. It replaces Multi-PNA+EU (AAAI-24, 68.16 +/- 2.65): that row is in the SAME table with two better ones above it (MEGA-GIN 70.83, MEGA-PNA 74.01) and has not been the ceiling since December 2024. Verified by reading the table, not the abstract.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

This reaches minority-class F1 well below the published 0.6323 (GFP+XGBoost, AMLworld HI-Small, NeurIPS 2023 D&B). The cause is a feature ARCHITECTURE difference, not a tuning one: the gap is unreachable at any threshold, because measured AUPRC is far below what that F1 requires. IBM's Graph Feature Preprocessor computes per-transaction subgraph patterns over a dynamically updated graph; this project computes per-account scalar aggregates. A degree is one number per account, whereas a GFP feature is a pattern match in that transaction's own neighbourhood at the moment it arrives. GFP ships in `snapml` under a Proprietary licence and is excluded by this project's commercial-OK-only rule, but the licence is not itself the cause - the architecture is, and it is reimplementable without `snapml`, simply at more cost than renaming aggregates. Reachability is computed to 4 hops against injected CYCLE attempts of up to 10; that was previously recorded as a main cause and is NOT one - the GFP paper (arXiv:2402.08593) decomposes its own contribution and caps multi-hop cycle and scatter-gather features at about 4% relative, while fan-in/fan-out, which is reimplemented here, carries more than 30%. The evaluation is strictly inductive and temporally split. Per-account aggregates are causal (each row sees only edges strictly before it); the structural block - cycle closure, scatter-gather, bank diversity, burst degree - is still fitted on the training window, because recomputing sparse reachability at every timestep is not tractable here. Reported as measured; not adjusted to close the gap. Full workings: references/finance-gap-investigation.md.

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
