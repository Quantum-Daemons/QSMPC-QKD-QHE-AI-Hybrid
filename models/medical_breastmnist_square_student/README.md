---
license: other
license_name: cc-by-4.0-asserted-by-medmnist
license_link: https://creativecommons.org/licenses/by/4.0/
library_name: onnx
pipeline_tag: image-classification
tags:
  - quantum-safe
  - privacy-preserving
  - medical-fl
  - onnx
  - onnxruntime-web
---

# Fully encrypted BreastMNIST CNN student (square activation)

1 conv block (1->4ch, 3x3, batch norm folded), x^2, avg-pool, dense 196->2, 14x14 input

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **encrypted-tier** model for the `medical_fl` use case: a
distilled student small enough to evaluate under CKKS. Whether the public demo
encrypts THIS model differs by use case; the notes below say what actually runs.
It is not the plaintext teacher, and its numbers should be read against that
teacher rather than against the published literature. Its agreement with the
teacher was measured offline, in plaintext.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.75 |
| `auc` | 0.819758 |
| `disagreement_on_positives` | 0.22807 |
| `export_seed` | 2 |
| `majority_class_rate` | 0.730769 |
| `metric_delta` | 0.160257 |
| `metric_delta_pp` | 16.0257 |
| `n_classes_predicted` | 2 |
| `n_eval` | 156 |
| `student_metric` | 0.705128 |
| `student_params` | 434 |
| `teacher_metric` | 0.865385 |
| `teacher_params` | 11171266 |

## Published baselines this is measured against

- **Target metric**: AUC
- **Baseline to beat**: 0.901 - ResNet-18 @28px, MedMNIST v2, Yang et al., Scientific Data 10:41 (2023) (AUC 0.901 / ACC 0.863)
- **Published ceiling**: 0.919 - Google AutoML Vision, MedMNIST v2 (AUC 0.919)

## Training data

- **Dataset**: MedMNIST v2 BreastMNIST
- **Licence**: CC-BY-4.0 asserted by MedMNIST; the BUSI authors publish no licence
- **Source**: https://medmnist.com/ (licence read 2026-10-26)
- **Attribution (CC-BY-4.0 asserted by MedMNIST; the BUSI authors publish no licence)**: Yang, Shi, Wei, Liu, Zhao, Ke, Pfister & Ni, 'MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification', Scientific Data 10:41 (2023), and Yang, Shi & Ni, 'MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis', IEEE ISBI 2021; CC BY 4.0. BreastMNIST is derived from Al-Dhabyani, Gomaa, Khaled & Fahmy, 'Dataset of breast ultrasound images', Data in Brief 28:104863 (2020).

780 images at 28x28. The tiny tier; also the source of the FHE-feasible T1 student. TWO problems, both upstream of MedMNIST, both checked 2026-08-10. LICENCE: MedMNIST licenses ITSELF CC BY 4.0, but BreastMNIST is a resize of BUSI (Al-Dhabyani et al., Data in Brief 28:104863, 2020) and the CC BY there attaches to the ARTICLE - PMC carries 'This is an open access article under the CC BY license' and states no separate licence for the data, while the dataset host (scholar.cu.edu.eg) asks only 'If you use this dataset, please cite'. A citation request is not a grant. CHECKED AGAIN 2026-10-26 at every source the authors publish: the host page asks the same in every capture from 2021 to 2026; the article's supplements hold the Baheya Hospital ethics committee's approval to publish the dataset publicly and the waiver of consent, and no licence; the authors made no repository deposit; and every Kaggle mirror carries a licence its uploader chose, none of them an author. So the BUSI authors publish no licence: the data was released publicly with no terms, and the only licence anyone states is MedMNIST's CC BY 4.0 for its derived dataset. The three HF repos stay public on that basis, by decision (2026-10-26). That differs from CIC-IDS2017, declined, and ImageNet, dropped: their providers state conflicting or restrictive terms, and BUSI's state none. CONTAMINATION: Pawlowska, Karwat & Zolek (Letter to the Editor, Data in Brief 48:109247, 2023-05-19; PMC10293973, read 2026-10-31) report that 155 images are copies of other images in the collection, 19% of its 780 (the duplicated cases their Appendix B lists come to 235 items), that eight lesion images appear in both the benign and the malignant sets, and that 70 images, more than 8%, 'show other structures than the breast' (their Table B3 lists them as suspected axilla images); they warn of train/test leakage. MEASURED HERE on the actual npz rather than assumed: at 28x28 exactly ONE pair survives as a pixel-exact duplicate - train[115] label=0 (MALIGNANT) and test[76] label=1 (BENIGN). The same image, trained as malignant and scored as benign. That is 1 of 156 test images, 0.64%, which the model cannot get right and which no amount of training fixes. Loosening to RMSE<=0.05 finds 216 near-duplicate pairs, 120 crossing a split boundary and 6 both crossing AND contradicting - but those are threshold-dependent and the exact pair is the only hard number. So the letter's 19% does NOT wholesale survive downsampling; a real floor on this benchmark does.

## Notes and limitations

End to end under CKKS, with no layer decrypted in between: the client encrypts the image's nine convolution taps, the evaluator multiplies by the kernel taps, adds the bias, squares, and takes one inner product per class with the dense weights averaged over each pooling window, and the client decrypts two logits. That is multiplicative depth 3, so it runs in its own CKKS context rather than the page's (60+40+40+60 allows two rescales): N=8192 with 48+34+34+34+52 bits (A2) or N=16384 with 60+40+40+40+60 (B), both within the HomomorphicEncryption.org standard's 128-bit post-quantum bound. Pre-registered criteria, five seeds: test AUC 0.8435 (sd 0.0198) against 0.8182 (sd 0.0045) for the published student's architecture trained by the same recipe; the exported seed, chosen by validation AUC, scores 0.8198. Under CKKS on all 156 test images the decrypted logits are within 3.1e-4 (A2) and 6.1e-5 (B) of the plaintext ones, with no decision changed; a first N=8192 set with a 30-bit scale missed the 1e-3 limit on every image and is not offered. Its operating point is the argmax, as pre-registered: it flags 66 of the 156 test images as malignant and catches 31 of the 42 malignant ones (recall 0.738, specificity 0.693), so its accuracy, 0.705, is below the 0.731 of always answering benign; the CKKS student's argmax flags 42 (recall 0.571, specificity 0.842). MedMNIST states that its dataset is NOT intended for clinical use, and neither is this model.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
