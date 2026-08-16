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
homomorphic evaluation is cost-bounded - the largest model ever demonstrated end-to-end under CKKS is
8B parameters at roughly a 10^4x runtime gap (IACR ePrint 2026/935). The browser tier runs int8 ONNX
where a quantised graph exists (medical ResNet-18, 11.3 MB); the two tree-ensemble models serve **fp32**,
because dynamic int8 quantisation has nothing to quantise when the parameters live in node attributes
rather than weight tensors. This Space is the third tier: models too large for either.

> **Not currently wired up.** Nothing in `web/src` calls this Space - searching the frontend for
> `qsh-inference`, `.hf.space` or `zerogpu` returns no hits, and although `ProvenanceBadge.tsx` declares a
> `remote` tier, no code path ever sets it. The Space is complete and works standalone, but the **REMOTE**
> label described below is unreachable in the deployed demo. Treat this as a staged component awaiting a
> caller, not as a live tier.

When a caller exists, results are labelled **REMOTE** in the demo. The latency reported there includes
network and queue time and is not comparable with the in-browser measurements.

## Honest scope

Metrics are what the project measured, shown with the published baseline - including where the
measurement falls short of it. The finance model reaches minority-class F1 0.151948 against a
published 0.632; the model card explains why (the published baseline uses IBM's Graph Feature
Preprocessor, which ships under a Proprietary licence this project cannot adopt).

This read 0.081 until Round 49, in BOTH this file and `app.py` - a pre-improvement figure left behind
by a retrain, on the one surface in this project that is published outside the repository. The
current value is what `models/metrics.json` records for `finance_aml.gbm`.

## Quota

ZeroGPU quota is per visitor, not per Space owner: 5 minutes/day signed in, 2 minutes anonymous.
Calls here are short by design. **These figures are uncited and Hugging Face changes them** - check
the current ZeroGPU documentation before relying on them for capacity planning.
