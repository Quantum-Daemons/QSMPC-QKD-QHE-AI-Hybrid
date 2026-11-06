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
- **Published ceiling**: 0.7581 - Multi-FraudGT, the best AML Small HI row of Table 16 ('Minority class F1 scores (%) for six AML datasets') in Bilgi and Atasu, 'MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes', TMLR (08/2026), arXiv:2412.00241v3: 75.81 +/- 0.75. In the same column the paper's own MEGA-PNA scores 74.01 +/- 1.55 and MEGA-GenAgg 74.48 +/- 0.84, and Multi-PNA 67.35 +/- 2.89; Small HI is the one AML dataset of the six where Multi-FraudGT is ahead of MEGA-PNA.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

This reaches minority-class F1 well below the published 0.6323 (GFP+XGBoost, AMLworld HI-Small, NeurIPS 2023 D&B). The cause is a feature ARCHITECTURE difference, not a tuning one: the gap is unreachable at any threshold, because measured AUPRC is far below what that F1 requires. IBM's Graph Feature Preprocessor computes per-transaction subgraph patterns over a dynamically updated graph; this project computes per-account scalar aggregates. A degree is one number per account, whereas a GFP feature is a pattern match in that transaction's own neighbourhood at the moment it arrives. GFP ships in `snapml` under a Proprietary licence and is excluded by this project's commercial-OK-only rule, but the licence is not itself the cause - the architecture is, and it is reimplementable without `snapml`, simply at more cost than renaming aggregates. Reachability is computed to 4 hops against injected CYCLE attempts of up to 10; that was previously recorded as a main cause and is NOT one - the GFP paper (arXiv:2402.08593) decomposes its own contribution, and its Table 5 (HI-Small, XGBoost, batch 128) gives fan-in/fan-out, which is reimplemented here, 37 F1 points (19.75 to 56.88) and multi-hop cycle and scatter-gather patterns 1.7 more (to 58.60; 'up to 4%' in its text is F1 points, not a relative gain). The evaluation is strictly inductive and temporally split. Per-account aggregates are causal (each row sees only edges strictly before it); the structural block - cycle closure, scatter-gather, bank diversity, burst degree - is still fitted on the training window, because recomputing sparse reachability at every timestep is not tractable here. Reported as measured; not adjusted to close the gap. Full workings: references/finance-gap-investigation.md.

## Calibration map

The demo page reads this model's score \\(q\\) as a risk through this map before it compares it with a threshold probability, as its decision curve does. Fitted on the validation split and scored on the held-out test split.

$$p = \frac{1}{1 + \exp\!\big(a\,\operatorname{logit}\big(\operatorname{clip}(q,\ \varepsilon,\ 1-\varepsilon)\big) + b\big)}$$

$$\operatorname{logit}(q) = \log\frac{q}{1-q},\qquad \varepsilon = 10^{-12}$$

| quantity | value |
|---|---|
| `a` | -0.5853491088584304 |
| `b` | 1.6417462922842536 |
| fitted on | 710969 validation rows, 801 positive |
| scored on | 1523504 test rows, 2321 positive |
| Brier score, raw to calibrated | 0.00150259 to 0.00142199 |
| log-loss, raw to calibrated | 0.00965987 to 0.00818964 |
| CORP miscalibration (MCB), raw to calibrated | 0.00009238 to 0.00001178 |
| base-rate Brier score (CORP UNC) | 0.00152114 |

Brier skill against the base rate, \\(1 - \mathrm{BS}/\mathrm{UNC}\\), is 0.0652.

Why Platt and not isotonic: MEASURED Round 49 on this split, not argued: isotonic Brier 0.00142218 / log-loss 0.00820025 against 0.00142199 / 0.00818964 for the same fit_platt fed logit(s) - within noise on both proper scoring rules, so ICML 2005's 'as good as, or better than' holds here in the 'as good as' form. Platt is kept on the tie-break: isotonic is only WEAKLY monotone, collapsed 1,388,669 distinct scores to 138, and cost AUROC 0.00031674 and AUPRC 0.01121590 (-9.09%), while both Platt arms moved neither to ten decimals; it also emitted exactly 0 on 464,176 test rows, one a true positive, and 138 breakpoints are not two floats a browser can apply. ONE CORRECTION TO THE OLD REASONING SURVIVES HERE: the calibration set carries 801 events, not 2,321 (that is the test split). The second - that a/b were fitted on a PROBABILITY rather than a log-odds, making the map return 0.00090148 for nine of ten deciles - is FIXED, not outstanding: on_logit is true above and the map published here is the logit(s) arm those numbers describe. Workings: references/finance-calibration-isotonic-vs-platt.md.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists. THE TEST WINDOW RUNS PAST THE DATA: HI-Small holds ten days, September 1-10, 2022 (IBM's dataset card), and the last 30 percent of its transactions in time order run to September 18. The 1,108 rows dated September 11-18 are there, the dataset's creator explains, so that laundering patterns begun earlier can complete; 655 of them are laundering, 28.2% of the test window's 2,321 positives. Without them the window's rate is 0.109 percent rather than 0.152, the teacher's minority-class F1 0.1387 rather than 0.1519 and its AUPRC 0.1030 rather than 0.1233, and the CKKS student's F1 0.0473 rather than 0.0614. The published 60/20/20 split's test set carries the same 1,108 rows (docs/measurements/finance-test-window-tail-2026-11-06.json).

## Notes and limitations

The browser-tier finance model: it scores one feature row per transaction with no message passing at scoring time. That row is not stateless - its per-account and structural features are computed from the account's earlier transactions - so a deployment needs that history to score a new transaction. It is not evaluated under encryption: the public page encrypts finance_aml.student, distilled from this model, on the 8,192 test-window rows of this model's sample, and a four-feature PaySim surrogate for its one-row CKKS step.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
