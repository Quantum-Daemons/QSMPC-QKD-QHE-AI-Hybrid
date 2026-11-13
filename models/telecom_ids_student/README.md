---
license: apache-2.0
library_name: onnx
pipeline_tag: tabular-classification
tags:
  - quantum-safe
  - privacy-preserving
  - telecom-ids
  - onnx
  - onnxruntime-web
---

# CKKS-linear IDS student (8 classes)

multinomial logistic regression, 8x7 + 8, distilled from the forest on soft targets

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **encrypted-tier** model for the `telecom_ids` use case: a
distilled student small enough to evaluate under CKKS. Whether the public demo
encrypts THIS model differs by use case; the notes below say what actually runs.
It is not the plaintext teacher, and its numbers should be read against that
teacher rather than against the published literature. Its agreement with the
teacher was measured offline, in plaintext.

## Measured performance

| metric | value |
|---|---|
| `aggregate_accuracy` | 0.988333 |
| `agreement` | 0.995 |
| `macro_f1` | 0.988329 |
| `n_classes` | 8 |
| `n_classes_predicted` | 8 |
| `n_eval` | 600 |
| `n_parameters` | 64 |

## Published baselines this is measured against

- **Target metric**: macro F1 (synthetic QKD-attack telemetry)
- **Baseline**: Architecture showcase, not a performance claim. On THIS project's synthetic telemetry neither quantum model beats the classical random forest (RF macro F1 0.987 vs variational QNN 0.467). Published results are MIXED, not uniformly below a random forest -- an earlier revision of this note claimed they were, and the very paper cited here contradicts it: Al-Kuwari et al. (arXiv:2509.14282) Table VIII reports a hybrid QLSTM at 94.7% accuracy / 94.7% F1 against their own random-forest baseline on the same generated data, at 81.5 to 82.6% accuracy by tree count (read at the source, 2026-11-12). This project's forest reaches macro F1 0.987 on its own generator; the metrics and the simulations differ, so the figures rank nothing, but this generator's classes separate more easily than theirs. What remains well-founded is caution about evaluation quality rather than a universal ranking: arXiv:2512.05069 was withdrawn by its own authors for evaluation that was 'insufficient to fully support the conclusions'. That single withdrawal is thin support for a field-wide claim, so the anchor is now a survey: Kaissar, Bou Nassif & Bouridane, Future Internet 18(5):234 (2026-04-27, CC BY 4.0, DOI 10.3390/fi18050234), systematic to January 2026, finds that 'most current approaches rely on simulated quantum environments and legacy datasets, while evaluation practices remain inconsistent across studies' - active but methodologically early, which is a different and better-founded statement than 'worse than a random forest'. The Al-Kuwari paper is now peer-reviewed: IET Quantum Communication 7(1), DOI 10.1049/qtc2.70028 (2026-02-23). And the nearest strong 2026 result is CLASSICAL, which corroborates the framing here - but it is a DIFFERENT problem and Round 46 stopped calling it the same one. IACR ePrint 2026/1282 reaches 96.9% recall / 97.6% precision with physics-aware temporal features plus XGBoost against 17.3% for static QBER thresholding - but its abstract, read at source, is entanglement-based BBM92, a 24-dimensional feature space over 30-second sliding windows, binary detection of blended sub-threshold attacks on simulated FSO telemetry. This model classifies 8 BB84 attack TYPES from per-session physical observables. Neither the protocol, the label space nor the feature space matches, so it is an anchor, not a baseline.

## Training data

- **Dataset**: QKD attack telemetry (synthetic, signatures of this project's own design)
- **Licence**: Apache-2.0
- **Source**: https://arxiv.org/abs/2509.14282 (licence read 2026-11-01)

8 classes = 7 attack types + normal, per Al-Kuwari et al. (IET Quantum Communication 7(1), e70028, 2026). The class list follows the paper; the 7 features and their signature values are this project's own design: illustrative, not fitted to any measurement. No public real QKD-attack capture was found (arXiv:2508.12749 reports anomalies recorded on an experimental QKD receiver and states no data release); this is labelled synthetic everywhere it is reported. The rows are written by this repository's generator (qsh.qml_ids.attacks), and they and the weights trained on them are released under the repository's licence, Apache-2.0.

## Notes and limitations

Distilled from the random forest published as telecom_ids.qml, from its PROBABILITIES, not its argmax: on 8 classes the argmax discards exactly the minority-class signal macro F1 is computed from, and hard-label distillation is a defect this project has already made once. Read n_classes_predicted before macro_f1 - a linear student that stops emitting a class scores 0 on it by construction, which reads as poor discrimination rather than an absent output. Under CKKS the page evaluates the 8 dot products on the encrypted row, then decrypts all 8 scores and takes the argmax in the clear, since CKKS has no native comparison; with these public weights that exposes nothing, but a private linear model's raw scores would let the key holder recover it with about as many queries as it has parameters per class (Tramer et al., USENIX Security 2016). The ONNX graph is built from weights.json (qsh.models.linear_onnx), not refitted, so the page's plaintext re-score and its encrypted tier evaluate the same model.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
