---
title: QSH Hosted Inference
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
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid).

The demo has three model tiers. The encrypted tier runs under real CKKS and is small because
homomorphic evaluation is costly at this operating point; REFERENCES.md carries the published larger runs. The browser
tier runs int8 ONNX
where a quantised graph exists (medical ResNet-18, 11.3 MB); the two tree-ensemble models serve **fp32**,
because dynamic int8 quantisation has nothing to quantise when the parameters live in node attributes
rather than weight tensors. This Space is the third tier: models too large for either.

> **Not currently wired up.** Nothing in `web/src` calls this Space - searching the frontend for
> `qsh-inference`, `.hf.space` or `zerogpu` returns no hits, and the decision not to wire it is recorded in
> `docs/findings.md`. Treat this as a component with no caller, not as a tier of the demo.

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
