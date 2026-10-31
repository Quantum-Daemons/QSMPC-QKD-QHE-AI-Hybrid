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

# Graph-feature gradient boosting

per-account and structural graph features computed by this project (not IBM's Graph Feature Preprocessor) + XGBoost

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
| `wall_clock_s` | 96.6 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17). Same metric and corpus, different protocol: that figure uses a 60/20/20 split by transaction order and IBM's Graph Feature Preprocessor updated over the test window; this project tests on the last 30 percent of transactions (fit 56, validation 14) with its own graph features, whose structural block is fitted on the training window.
- **Published ceiling**: 0.7401 - MEGA-PNA, arXiv:2412.00241v2 Table 2 'Minority-class F1 scores (%) on AML edge classification task', AML Small HI (74.01 +/- 1.55); the paper's v3 is accepted for publication in TMLR (2026). It is not the table's highest row: the same table prints MEGA-GenAgg at 74.88 +/- 0.38 and MEGA(PNA)-GIN at 74.69 +/- 0.50, and Multi-PNA (68.16 +/- 2.65) below all three.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

This reaches minority-class F1 well below the published 0.6323 (GFP+XGBoost, AMLworld HI-Small, NeurIPS 2023 D&B). The cause is a feature ARCHITECTURE difference, not a tuning one: the gap is unreachable at any threshold, because measured AUPRC is far below what that F1 requires. IBM's Graph Feature Preprocessor computes per-transaction subgraph patterns over a dynamically updated graph; this project computes per-account scalar aggregates. A degree is one number per account, whereas a GFP feature is a pattern match in that transaction's own neighbourhood at the moment it arrives. GFP ships in `snapml` under a Proprietary licence and is excluded by this project's commercial-OK-only rule, but the licence is not itself the cause - the architecture is, and it is reimplementable without `snapml`, simply at more cost than renaming aggregates. Reachability is computed to 4 hops against injected CYCLE attempts of up to 10; that was previously recorded as a main cause and is NOT one - the GFP paper (arXiv:2402.08593) decomposes its own contribution, and its Table 5 (HI-Small, XGBoost, batch 128) gives fan-in/fan-out, which is reimplemented here, 37 F1 points (19.75 to 56.88) and multi-hop cycle and scatter-gather patterns 1.7 more (to 58.60; 'up to 4%' in its text is F1 points, not a relative gain). The evaluation is strictly inductive and temporally split. Per-account aggregates are causal (each row sees only edges strictly before it); the structural block - cycle closure, scatter-gather, bank diversity, burst degree - is still fitted on the training window, because recomputing sparse reachability at every timestep is not tractable here. Reported as measured; not adjusted to close the gap. Full workings: references/finance-gap-investigation.md.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists.

## Notes and limitations

The browser-tier finance model: it scores one feature row per transaction with no message passing at scoring time. That row is not stateless - its per-account and structural features are computed from the account's earlier transactions - so a deployment needs that history to score a new transaction.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
