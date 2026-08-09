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
| `wall_clock_s` | 77.5 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17)
- **Published ceiling**: 0.6816 - Multi-PNA+EU, AAAI-24 vol.38 no.10 pp.11838-11846 (68.16 +/- 2.65)
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

This reaches minority-class F1 well below the published 0.6323. Round 24 measured the causes rather
than assuming them; the full workings are in
[`references/finance-gap-investigation.md`](../../references/finance-gap-investigation.md).

**Ruled out by measurement.** The decision threshold: reaching F1 0.6323 requires precision around
0.63 held to recall around 0.63, so AUPRC of at least ~0.40, and the measured AUPRC is 0.0732 - the
figure is unreachable at *any* threshold, which makes this a ranking gap rather than a tuning gap.
Accounts unseen in the training window: only 0.39% of test rows have either side unseen, too rare to
matter.

**Also ruled out as a main cause: the hop limit.** The corpus injects CYCLE attempts of up to 10 hops
while reachability here is computed to 4. An earlier version of this section gave that as one of two
main causes. It cannot be: the Graph Feature Preprocessor paper (arXiv:2402.08593) attributes "more
than 30%" of its minority-F1 improvement to fan-in/fan-out features - which this project already
implements - and only "up to 4%" to the multi-hop cycle and scatter-gather features.

**What remains.** IBM's Graph Feature Preprocessor ships in `snapml` under a Proprietary licence and
is excluded by this project's commercial-OK-only rule. The mechanism that matters is architectural
rather than legal: GFP computes per-transaction subgraph patterns over a graph updated as each
transaction arrives, while this pipeline computes per-account scalar aggregates. That is
implementable without `snapml`; it is simply larger than renaming aggregates.

**One defect found in passing, worth fixing on its own merits.** Aggregates are fitted over the whole
training window, so a training row's feature includes its own future - the median transaction has
only 46.4% of its out-degree accrued by its own timestamp - while test rows receive the same table as
a stale past. A causal, prior-at-time computation removes the asymmetry and measured **1.45x AUPRC**
over the shipped approach at matched feature count.

Reported as measured; not adjusted to close the gap.

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
