---
title: QSH Hosted Inference
emoji: 🔐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Full-precision industry models for the QSH quantum-safe demo (T3 tier)
models:
  - Daemons-Q/qsh-medical-breastmnist-resnet18
  - Daemons-Q/qsh-finance-aml-gbm
---

# QSH hosted inference (T3 tier)

Serves the **plaintext** per-industry models at full precision for
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid).

The demo has three model tiers. The encrypted tier runs under real CKKS and is small because
homomorphic evaluation is cost-bounded — the largest model ever demonstrated end-to-end under CKKS is
8B parameters at roughly a 10^4x runtime gap (IACR ePrint 2026/935). The browser tier runs int8 ONNX
under 12 MB. This Space is the third: models too large for either.

Results are labelled **REMOTE** in the demo. The latency reported here includes network and queue
time and is not comparable with the in-browser measurements.

## Honest scope

Metrics are what the project measured, shown with the published baseline — including where the
measurement falls short of it. The finance model reaches minority-class F1 0.081 against a published
0.632; the model card explains why (the published baseline uses IBM's Graph Feature Preprocessor,
which ships under a Proprietary licence this project cannot adopt).

## Quota

ZeroGPU quota is per visitor, not per Space owner: 5 minutes/day signed in, 2 minutes anonymous.
Calls here are short by design.
