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

# IBM Multi-PNA+EU, causal protocol

PNA with reverse message passing, multigraph port numbering, ego IDs and edge updates (IBM/Multi-GNN at 252b025, Apache-2.0), 2 layers of width 20

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is **IBM's official Multi-PNA+EU** (IBM/Multi-GNN at commit 252b025, Apache-2.0; Egressy et al., 'Provably Powerful
Graph Neural Networks for Directed Multigraphs', AAAI 2024, arXiv:2306.11586), trained here from IBM's code on the
`finance_aml` page's windows under the causal protocol, which reads nothing past the horizon the page's own models give a transfer, and exported to ONNX at opset 18. The page runs it in
plaintext in onnxruntime-web on 16 held-out transfers with the neighbourhood its protocol sampled for them
(`eval_sample.json`); nothing of it runs under encryption. This project's changes to IBM's code are the patches in
`scripts/train/ibm_multignn/` of the repository, each stated in that folder's README as Apache-2.0 section 4 asks of
modified files; IBM's repository ships a LICENSE and no NOTICE file, and its LICENSE is copied here as
`LICENSE-IBM-Multi-GNN.txt`. IBM's loop never calls `model.eval()`, so a checkpoint it saves carries BatchNorm
statistics last set by the test pass. This model's were recomputed on the fit window, and the table below is the
exported model's, in eval mode at the threshold tuned on validation; `logged_test_f1` is the figure IBM's own loop
logs at the epoch it keeps, in training mode, which is how the paper's figures are made.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.998943 |
| `auprc` | 0.578913 |
| `auprc_baseline` | 0.00152435 |
| `auroc` | 0.981609 |
| `decision_threshold` | 0.431105 |
| `kept_epoch` | 16 |
| `logged_test_f1` | 0.5474 |
| `logged_validation_f1` | 0.6258 |
| `minority_class_f1` | 0.625204 |
| `n_test` | 1522612 |
| `n_validation` | 710969 |
| `selected_seed` | 1 |

## Five seeds

IBM's loop keeps an epoch when its validation F1 beats every earlier epoch's; the logged figures are its own, in
training mode. The exported way is eval mode with the BatchNorm statistics recomputed on the fit window, at the
threshold that maximises validation F1. The seed exported is the one with the best logged validation F1.

| seed | kept epoch | logged validation F1 | logged test F1 | test F1, exported way | test AUPRC | test F1, plain batches of 2,048 |
|---|---|---|---|---|---|---|
| 1 | 16 | 0.6258 | 0.5474 | 0.625204 | 0.578913 | 0.458121 |
| 2 | 28 | 0.6178 | 0.6025 | 0.578565 | 0.498434 | 0.400265 |
| 3 | 34 | 0.6171 | 0.5737 | 0.597337 | 0.522967 | 0.409632 |
| 4 | 33 | 0.6115 | 0.5718 | 0.658933 | 0.59071 | 0.463759 |
| 5 | 14 | 0.6152 | 0.5188 | 0.643999 | 0.584393 | 0.451194 |
| mean | | 0.61748 (sample sd 0.005254) | 0.56284 (sample sd 0.031421) | 0.620808 (sample sd 0.032962) | 0.555083 (sample sd 0.041644) | 0.436594 (sample sd 0.029416) |

## The time input under the causal protocol

IBM's model reads each transfer's timestamp as an edge feature. Under this protocol every window is scaled by the
fit window's statistics, and the test window's timestamps then lay outside every value training saw (\\(z\\) of 2.9
to 9.7, against at most 2.0 on the fit window). This model's time input (`QSH_CAUSAL_TIME=lag` in
`scripts/train/ibm_multignn/causal.patch`) is instead each transfer's time before the end of its batch, over the fit window's span \\(s\\) (443,520 s):

$$\ell = \max\left(-1,\ \frac{t - t_{\mathrm{end}}}{s}\right) \in [-1, 0],$$

where \\(t_{\mathrm{end}}\\) is the latest time among the scored transfers of the batch. Temporal GNNs likewise read
time relative to the event rather than absolute, through an encoding of the difference (TGAT, arXiv:2002.07962;
GraphMixer, arXiv:2302.11636); this model reads the scaled difference itself.

Both replacements were trained, five seeds each, and the one exported has the better mean logged validation
F1 (`lag` 0.61748 against `drop` 0.5977); the other's seeds are below, with the runs on the absolute timestamp before it was replaced.

Under the protocol's batches every scored transfer lies at most 1,020 s before its batch's end, as in
the fit window, so \\(\ell \ge -0.0023\\) on every window. Under the plain batches of 2,048 of the
last column, 1,352 test transfers, 654 of
them laundering (of the test window's 2,321), lie further back than any fit-window
transfer, down to the clip at \\(-1\\). Over five seeds the plain batches' mean test F1 lies below the exported way's by 0.184 for `lag` and 0.060 for `drop`.

## Five seeds with the time input `drop`

The same protocol and columns, with the time input set to zero, so time only orders and cuts the batches.

| seed | kept epoch | logged validation F1 | logged test F1 | test F1, exported way | test AUPRC | test F1, plain batches of 2,048 |
|---|---|---|---|---|---|---|
| 1 | 30 | 0.5989 | 0.5403 | 0.525489 | 0.531327 | 0.487408 |
| 2 | 34 | 0.6109 | 0.593 | 0.615631 | 0.590427 | 0.546947 |
| 3 | 25 | 0.5842 | 0.5156 | 0.55168 | 0.526011 | 0.508799 |
| 4 | 23 | 0.592 | 0.4966 | 0.546944 | 0.53775 | 0.462697 |
| 5 | 46 | 0.6025 | 0.4875 | 0.566507 | 0.516206 | 0.50088 |
| mean | | 0.5977 (sample sd 0.010167) | 0.5266 (sample sd 0.042268) | 0.56125 (sample sd 0.033764) | 0.540344 (sample sd 0.029088) | 0.501346 (sample sd 0.030916) |

## The absolute timestamp, before it was replaced

IBM's absolute timestamp, scaled by the fit window's statistics; stopped after these seeds, its test inputs lying outside every value training saw (the owner's choice of 2026-11-16).

| seed | kept epoch | logged validation F1 | logged test F1 | test F1, exported way | test AUPRC | test F1, plain batches of 2,048 |
|---|---|---|---|---|---|---|
| 1 | 20 | 0.6192 | 0.4125 | 0.399484 | 0.337865 | 0.351995 |
| 2 | 24 | 0.6165 | 0.4307 | 0.354615 | 0.265073 | 0.299001 |
| 3 | 25 | 0.6155 | 0.5544 | 0.588787 | 0.515613 | 0.520668 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6816 - Multi-PNA+EU on AML Small HI: 68.16 +/- 2.65 minority-class F1, five runs on the paper's own split (Egressy et al., AAAI 2024, arXiv:2306.11586, Table 2), under IBM's protocol. This model reads less than that protocol does, nothing past a transfer's horizon, so the figure is a reference for it rather than a like-for-like comparison.
- **Published ceiling**: 0.7581 - Multi-FraudGT, the best AML Small HI row of Table 16 ('Minority class F1 scores (%) for six AML datasets') in Bilgi and Atasu, 'MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes', TMLR (08/2026), arXiv:2412.00241v3: 75.81 +/- 0.75. In the same column the paper's own MEGA-PNA scores 74.01 +/- 1.55 and MEGA-GenAgg 74.48 +/- 0.84, and Multi-PNA 67.35 +/- 2.89; Small HI is the one AML dataset of the six where Multi-FraudGT is ahead of MEGA-PNA.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data. No real laundering-labelled corpus under a permissive licence was found (2026-11-22): Elliptic and Elliptic2 are CC BY-NC-ND 4.0, and the one real bank set found is GPL-3.0 with differentially private PCA features (arXiv:2507.08835). THE TEST WINDOW RUNS PAST THE DATA: HI-Small holds ten days, September 1-10, 2022 (IBM's dataset card), and the last 30 percent of its transactions in time order run to September 18. The 1,108 rows dated September 11-18 are there, the dataset's creator explains, so that laundering patterns begun earlier can complete; 655 of them are laundering, 28.2% of the test window's 2,321 positives. Without them the window's rate is 0.109 percent rather than 0.152, the teacher's minority-class F1 0.5499 rather than 0.6161 and its AUPRC 0.5408 rather than 0.6428, and the CKKS student's F1 0.2083 rather than 0.3088. The published 60/20/20 split's test set carries the same 1,108 rows (docs/measurements/finance-test-window-tail-2026-11-11.json).

## Notes and limitations

IBM's Multi-PNA+EU under the causal protocol (scripts/train/ibm_multignn/causal.patch on top of the split patch, the owner's choices of 2026-11-15): each mini-batch is one batch of 2,048 transfers in time order, split where it spans more than 17 minutes, sampled with IBM's shared sampler, with every sampled transfer later than that batch dropped, and every window scaled by the fit window's statistics. Scaled that way, IBM's absolute timestamp input lay outside every value training saw on the test window, so this model reads in its place each transfer's time before the end of its batch over the fit window's span, clipped to [-1, 0] (QSH_CAUSAL_TIME=lag, the owner's choice of 2026-11-16); the input set to zero (drop) was trained the same way, and lag has the better mean logged validation F1 over five seeds each. Trained with early stopping after 20 epochs without a better validation F1 and four loader processes; the kept epoch is IBM's rule and the exported seed the best logged validation F1 of lag's five. Nothing it reads lies past the horizon the page's own models give a transfer, the end of its batch of 2,048 and at most 17 minutes after it, and its BatchNorm statistics are recomputed on the fit window, so its normalisation does not read the test window either. The record also gives the same checkpoints' test F1 under the plain batches of 2,048, under which lag's input lies outside the range training saw for part of the test window, as the card states. It runs in plaintext in this tab, beside its twin finance_aml.gnn. Message passing over the whole graph needs every account's neighbourhood at scoring time, which the browser holds here only for these 16 transfers. Bilgi et al. (arXiv:2610.18054v2, preprint) run an edge-centric GNN under CKKS on the IBM AML benchmark at 5.31 s per queried transaction (1.65 s amortized over a batch of 16) with HEonGPU on one RTX 5090, with the graph held in plaintext by the server - server-side evaluation only, since those figures exclude the client's encryption and decryption, key generation and network communication - and its encrypted and plaintext predictions agree on 99.25% of transactions. It is not the best published figure on this split: the published version of MEGA (TMLR, see the ceiling) re-runs Multi-PNA at 67.35 +/- 2.89 on Small HI, below FraudGT, Multi-FraudGT and every MEGA variant in its table.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
