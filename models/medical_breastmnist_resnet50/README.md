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

# ResNet-50 on BreastMNIST (published comparison)

ResNet-50 @28px, 1-channel stem, trained from random initialisation

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo. This is the **plaintext** model for the
`medical_fl` use case; the encrypted path runs a distilled student, not this model.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.782051282051282 |
| `auc` | 0.7987677527151211 |
| `best_val_auc` | 0.935673 |
| `macro_f1` | 0.6492063492063492 |
| `n_parameters` | 23505858 |
| `resolution` | 28 |
| `wall_clock_s` | 12.9 |

## Published baselines this is measured against

- **Target metric**: AUC
- **Baseline to beat**: 0.857 - ResNet-50 @28px, MedMNIST v2, Yang et al., Scientific Data 10:41 (2023) (AUC 0.857 / ACC 0.812). Note this is BELOW the ResNet-18 baseline of 0.901 on the same corpus and the same 100-epoch Adam regime: on 780 images the deeper network loses. That is the published finding, and it is why this model exists here - to reproduce it rather than to win.
- **Published ceiling**: 0.919 - Google AutoML Vision, MedMNIST v2 (AUC 0.919)
- **Companion metric shown alongside**: `accuracy` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: MedMNIST v2 BreastMNIST
- **Licence**: CC-BY-4.0
- **Source**: https://medmnist.com/ (licence read 2026-08-03)

780 images at 28x28. The tiny tier; also the source of the FHE-feasible T1 student.

## Notes and limitations

Trained to test a published claim rather than to win one. MedMNIST v2 reports ResNet-50 at AUC 0.857 against ResNet-18's 0.901 on this corpus, so the expected outcome is that the larger model scores LOWER, and it is reported either way. It replaces a DenseNet-121 entry that named a dataset with no loader, no fetch command and no artefact directory, and linked a Hugging Face repo that did not exist. Architectures benchmarked only ImageNet-pretrained (AlexNet, VGG16, DenseNet-121, EfficientNet-B4 in MedMNIST+) are deliberately absent: this project dropped ImageNet weights over redistribution terms, so their published numbers are not comparable to a from-scratch run.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
