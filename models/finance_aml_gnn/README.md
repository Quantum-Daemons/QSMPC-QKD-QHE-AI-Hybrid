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

# IBM Multi-PNA+EU, IBM's protocol

PNA with reverse message passing, multigraph port numbering, ego IDs and edge updates (IBM/Multi-GNN at 252b025, Apache-2.0), 2 layers of width 20

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is **IBM's official Multi-PNA+EU** (IBM/Multi-GNN at commit 252b025, Apache-2.0; Egressy et al., 'Provably Powerful
Graph Neural Networks for Directed Multigraphs', AAAI 2024, arXiv:2306.11586), trained here from IBM's code on the
`finance_aml` page's windows under IBM's protocol, which samples each transfer's neighbours from its window's whole graph, and exported to ONNX at opset 18. The page runs it in
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
| `accuracy` | 0.999139 |
| `auprc` | 0.631829 |
| `auprc_baseline` | 0.00152434 |
| `auroc` | 0.985447 |
| `decision_threshold` | 0.637421 |
| `kept_epoch` | 33 |
| `logged_test_f1` | 0.6751 |
| `logged_validation_f1` | 0.6208 |
| `minority_class_f1` | 0.682182 |
| `n_test` | 1522629 |
| `n_validation` | 710969 |
| `selected_seed` | 4 |

## Five seeds

IBM's loop keeps an epoch when its validation F1 beats every earlier epoch's; the logged figures are its own, in
training mode. The exported way is eval mode with the BatchNorm statistics recomputed on the fit window, at the
threshold that maximises validation F1. The seed exported is the one with the best logged validation F1.

| seed | kept epoch | logged validation F1 | logged test F1 | test F1, exported way | test AUPRC |
|---|---|---|---|---|---|
| 1 | 36 | 0.6177 | 0.6616 | 0.673119 | 0.639675 |
| 2 | 38 | 0.6089 | 0.6506 | 0.664573 | 0.651 |
| 3 | 54 | 0.5928 | 0.6254 | 0.639289 | 0.596986 |
| 4 | 33 | 0.6208 | 0.6751 | 0.682182 | 0.631829 |
| 5 | 39 | 0.5732 | 0.6309 | 0.629219 | 0.579145 |
| mean | | | 0.64872 (sample sd 0.020777) | 0.657676 (sample sd 0.022553) | 0.619727 (sample sd 0.030358) |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6816 - Multi-PNA+EU itself on AML Small HI: 68.16 +/- 2.65 minority-class F1, five runs on the paper's own split (Egressy et al., 'Provably Powerful Graph Neural Networks for Directed Multigraphs', AAAI 2024, arXiv:2306.11586, Table 2). Run here from IBM's code on this project's windows over five seeds, each listed with the test F1 IBM's loop logs at its kept epoch, the paper's measure, beside the exported model's. GFP+XGBoost's 63.23 is the baseline of the gradient-boosting model this demo ships, not of this architecture.
- **Published ceiling**: 0.7581 - Multi-FraudGT, the best AML Small HI row of Table 16 ('Minority class F1 scores (%) for six AML datasets') in Bilgi and Atasu, 'MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes', TMLR (08/2026), arXiv:2412.00241v3: 75.81 +/- 0.75. In the same column the paper's own MEGA-PNA scores 74.01 +/- 1.55 and MEGA-GenAgg 74.48 +/- 0.84, and Multi-PNA 67.35 +/- 2.89; Small HI is the one AML dataset of the six where Multi-FraudGT is ahead of MEGA-PNA.
- **Companion metric shown alongside**: `auprc` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists. THE TEST WINDOW RUNS PAST THE DATA: HI-Small holds ten days, September 1-10, 2022 (IBM's dataset card), and the last 30 percent of its transactions in time order run to September 18. The 1,108 rows dated September 11-18 are there, the dataset's creator explains, so that laundering patterns begun earlier can complete; 655 of them are laundering, 28.2% of the test window's 2,321 positives. Without them the window's rate is 0.109 percent rather than 0.152, the teacher's minority-class F1 0.5499 rather than 0.6161 and its AUPRC 0.5408 rather than 0.6428, and the CKKS student's F1 0.2083 rather than 0.3088. The published 60/20/20 split's test set carries the same 1,108 rows (docs/measurements/finance-test-window-tail-2026-11-11.json).

## Notes and limitations

IBM's official Multi-PNA+EU, trained here from IBM's code (IBM/Multi-GNN at 252b025, Apache-2.0, with scripts/train/ibm_multignn/project_split.patch for this project's windows) for IBM's 100 epochs, five seeds. The kept epoch is IBM's own rule and the exported seed is the one with the best logged validation F1. IBM's loop never calls model.eval(), so a checkpoint it saves carries BatchNorm statistics last set by the test pass; the exported one has them recomputed on the fit window (precise_bn.py, the owner's choice of 2026-11-16), and its figures are scored in eval mode at the threshold tuned on validation. The page runs it in plaintext: onnxruntime-web scores its 16 held-out transfers with the neighbourhood IBM's loader sampled for them, and nothing of it runs under encryption. IBM's protocol reads each window's whole graph, so a transfer's neighbourhood can include later transfers, which the tab counts; its causal twin is finance_aml.gnn_causal. The compact GNN and the MLP are distilled from this model, seed 1. Message passing over the whole graph needs every account's neighbourhood at scoring time, which the browser holds here only for these 16 transfers. Bilgi et al. (arXiv:2610.18054v2, preprint) run an edge-centric GNN under CKKS on the IBM AML benchmark at 5.31 s per queried transaction (1.65 s amortized over a batch of 16) with HEonGPU on one RTX 5090, with the graph held in plaintext by the server - server-side evaluation only, since those figures exclude the client's encryption and decryption, key generation and network communication - and its encrypted and plaintext predictions agree on 99.25% of transactions. It is not the best published figure on this split: the published version of MEGA (TMLR, see the ceiling) re-runs Multi-PNA at 67.35 +/- 2.89 on Small HI, below FraudGT, Multi-FraudGT and every MEGA variant in its table.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
