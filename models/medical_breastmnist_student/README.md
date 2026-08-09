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

# FHE-feasible diagnostic CNN

1 conv block (1->4ch, 3x3), avg-pool, dense head, 14x14 input

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo. This is the **plaintext** model for the
`medical_fl` use case and it runs under real CKKS encryption.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.730769 |
| `auc` | 0.609231 |
| `disagreement_on_positives` | 0.157895 |
| `majority_class_rate` | 0.730769 |
| `metric_delta` | 0.166667 |
| `metric_delta_pp` | 16.6667 |
| `n_classes_predicted` | 2 |
| `n_eval` | 156 |
| `student_metric` | 0.685897 |
| `student_params` | 3226 |
| `teacher_metric` | 0.852564 |
| `teacher_params` | 11171266 |

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

Genuine encrypted convolution: im2col + CKKS dot_plain, ReLU/max-pool under MPC.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
