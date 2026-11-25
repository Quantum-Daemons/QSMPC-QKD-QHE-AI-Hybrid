---
license: cc-by-3.0
pipeline_tag: image-classification
tags:
  - quantum-safe
  - privacy-preserving
  - medical-fl
---

# Fully encrypted LIDC-IDRI 3D nodule CNN student (square activation)

1 conv block (1->4ch, 3x3x3, batch norm folded), x^2, 2x2x2 avg-pool, dense 4*14^3->2, 28x28x28 input (a 28 mm cube at 1 mm)

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **3D encrypted-tier** model for the `medical_fl` use case: a network small enough to
evaluate end to end under CKKS, distilled with soft labels from an out-of-fold teacher that is not published.
The medical page runs it in a Web Worker on twelve held-out test crops that ship with the site, from the
site's own copy of these weights, and draws each crop as three orthogonal planes; its fidelity under CKKS on
all 347 test crops was measured offline, by the page's own module under node-seal. The notes below give the
pre-registered comparison and the parameter sets.

## Measured performance

| metric | value |
|---|---|
| `auc` | 0.892568 |
| `auc_ci95_high` | 0.933242 |
| `auc_ci95_low` | 0.847758 |
| `balanced_accuracy` | 0.846003 |
| `bootstrap_resamples` | 2000 |
| `export_seed` | 5 |
| `majority_class_rate` | 0.685879 |
| `n_eval` | 347 |
| `student_params` | 22066 |
| `val_auc` | 0.838577 |

## Published baselines this is measured against

- **Target metric**: AUC, malignant against benign nodules, on the 347 test nodules of 148 held-out patients
- **Baseline**: none on this split. NoduleMNIST3D's official-split test AUCs (MedMNIST v2, Yang et al., Scientific Data 10:41, 2023: ResNet-18 3D 0.863, ResNet-50 3D 0.875, auto-sklearn 0.914) are cited for context and not compared: that split shares scans across its parts and labels differently
- **Companion metric shown alongside**: `balanced_accuracy` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: LIDC-IDRI (The Cancer Imaging Archive)
- **Licence**: CC-BY-3.0
- **Source**: https://www.cancerimagingarchive.net/collection/lidc-idri/ (licence read 2026-11-25)
- **Attribution (CC-BY-3.0)**: Armato III, S. G., et al. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. doi:10.7937/K9/TCIA.2015.LO9QL9SX; Armato et al., 'The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)', Medical Physics 38(2), 915-931 (2011), doi:10.1118/1.3528204. The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.

Thoracic CT of 1,010 patients with lesions marked by up to four radiologists; TCIA states CC BY 3.0 for the images and for the radiologists' annotations, and asks for the data citation with its DOI, the publication citation and the NCI and FNIH acknowledgement (its collection page, read 2026-11-25). This project fetched the 883 CT series of the 875 patients whose scans hold at least one nodule cluster (215,912 files) through TCIA's REST interface, every file checked against TCIA's MD5 and every series against its image count, the training patients' series twice (`docs/measurements/lidc-stream-2026-11-24.json`). One 28 mm crop at 1 mm per pylidc cluster of the readers' marks (Hancock and Magnan, Journal of Medical Imaging 3(4), 044504, 2016); the label is the readers' median malignancy rating, above 3 malignant and below 3 benign, exactly 3 left out: a rating, not a diagnosis. The split is by patient, drawn once from the records before any image was fetched (`docs/measurements/medical-nodule3d-split.json`).

## Notes and limitations

End to end under CKKS, with no layer decrypted in between: the client encrypts a crop's 27 convolution taps, the evaluator convolves, squares and applies the pooled dense layer on the ciphertexts, and the client decrypts two logits; multiplicative depth 3, under N=8192 with 48+34+34+34+52 bits (A2) or N=16384 with 60+40+40+40+60 (B), both within the HomomorphicEncryption.org standard's 128-bit post-quantum bound. Distilled with soft labels from the best of three pre-registered teacher candidates, each scored out of fold over the training patients (a 3D ResNet-18 trained from scratch 0.874, TotalSegmentator's nodule size 0.786, CT-FM's linear probe 0.641; the ResNet-18 taught; no teacher is published). Pre-registered criteria, five seeds: test AUC 0.907 (sd 0.009) against 0.882 (sd 0.027) for the same network with a ReLU, the margin of criterion 1 being the control's mean minus 0.03. The exported seed, 5, chosen by validation AUC as pre-registered, scores 0.893 on test (95 % interval over test patients 0.848 to 0.933), the lowest of the five square seeds there. Under CKKS on all 347 test crops the decrypted logits are within 3.2e-5 (A2) and 1.6e-5 (B) of the plaintext ones, with no decision changed. The labels are the readers' median malignancy rating, not a diagnosis. For research and demonstration only, not for clinical use.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
