---
license: other
license_name: cc-by-4.0
library_name: onnx
pipeline_tag: image-classification
tags:
  - quantum-safe
  - privacy-preserving
  - medical-fl
  - onnx
  - onnxruntime-web
---

# ResNet-18 on BreastMNIST

ResNet-18 @28px

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo. This is the **plaintext** model for the
`medical_fl` use case; the encrypted path runs a distilled student, not this model.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.8525641025641025 |
| `auc` | 0.8779239766081871 |
| `best_val_auc` | 0.945698 |
| `macro_f1` | 0.790310326690433 |
| `n_parameters` | 11171266 |
| `resolution` | 28 |
| `wall_clock_s` | 7.6 |

## Published baselines this is measured against

- **Target metric**: AUC
- **Baseline to beat**: 0.901 - ResNet-18 @28px, MedMNIST v2, Yang et al., Scientific Data 10:41 (2023) (AUC 0.901 / ACC 0.863)
- **Published ceiling**: 0.919 - Google AutoML Vision, MedMNIST v2 (AUC 0.919)
- **Companion metric shown alongside**: `accuracy` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: MedMNIST v2 BreastMNIST
- **Licence**: CC-BY-4.0
- **Source**: https://medmnist.com/ (licence read 2026-08-03)

780 images at 28x28. The tiny tier; also the source of the FHE-feasible T1 student.

## Notes and limitations

The small tier: fits the browser comfortably and has a clean published baseline.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
