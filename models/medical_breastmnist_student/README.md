---
license: other
license_name: cc-by-4.0-asserted-by-medmnist
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
a quantum-safe orchestration demo.
This is the **encrypted-tier** model for the `medical_fl` use case: it is a
distilled student small enough to evaluate under real CKKS, and the demo runs it
that way. It is not the plaintext teacher, and its numbers should be read against
that teacher rather than against the published literature.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.826923 |
| `auc` | 0.788221 |
| `disagreement_on_positives` | 0.149123 |
| `majority_class_rate` | 0.730769 |
| `metric_delta` | 0.096154 |
| `metric_delta_pp` | 9.6154 |
| `n_classes_predicted` | 2 |
| `n_eval` | 156 |
| `student_metric` | 0.769231 |
| `student_params` | 3226 |
| `teacher_metric` | 0.865385 |
| `teacher_params` | 11171266 |

## Published baselines this is measured against

- **Target metric**: AUC
- **Baseline to beat**: 0.901 - ResNet-18 @28px, MedMNIST v2, Yang et al., Scientific Data 10:41 (2023) (AUC 0.901 / ACC 0.863)
- **Published ceiling**: 0.919 - Google AutoML Vision, MedMNIST v2 (AUC 0.919)
- **Companion metric shown alongside**: `accuracy` - reported together because the aggregate figure can look healthy while the class that matters is not.

### Gap to the published baseline

This scores far below the 0.901 ResNet-18 baseline. The reason is representational rather than cryptographic: 3,226 parameters over a 14x14 input is a REPRESENTATION choice, not a feasibility limit - and this note said 'a CKKS-feasibility budget' until Round 32, which overstated how forced it was. Kim et al. (arXiv:2506.15258) do HE chest-X-ray inference on a VQGAN latent at downsampling factor 8 rather than on pixels, so the alternative to downsampling 28->14 is re-representing the input. The browser already runs onnxruntime-web, so such an encoder would run locally in plaintext with no privacy loss. Recorded as a falsifiable trigger in the additional-model decision, not acted on. Round 24 found it had been WORSE than that - a constant predictor, emitting one class for all 156 test images, with accuracy 0.730769 exactly equal to the majority-class rate and AUC 0.354, below random. It was invisible because the registry declared `auc` while the record stored `accuracy`, and because this student had neither protection its teacher already had: class-weighted loss and AUC-based model selection. Both were added. Read `n_classes_predicted` and `majority_class_rate` beside the score - an accuracy equal to the majority rate is the signature of a model that learned nothing. Workings: references/medical-student-collapse-investigation.md.

## Training data

- **Dataset**: MedMNIST v2 BreastMNIST
- **Licence**: CC-BY-4.0 asserted by MedMNIST; the upstream BUSI grant is NOT verifiable
- **Source**: https://medmnist.com/ (licence read 2026-08-10)

780 images at 28x28. The tiny tier; also the source of the FHE-feasible T1 student. TWO problems, both upstream of MedMNIST, both checked 2026-08-10. LICENCE: MedMNIST licenses ITSELF CC BY 4.0, but BreastMNIST is a resize of BUSI (Al-Dhabyani et al., Data in Brief 28:104863, 2020) and the CC BY there attaches to the ARTICLE - PMC carries 'This is an open access article under the CC BY license' and states no separate licence for the data, while the dataset host (scholar.cu.edu.eg) asks only 'If you use this dataset, please cite'. A citation request is not a grant. This is the same ambiguity class that got CIC-IDS2017 declined and ImageNet dropped, and it is unresolved rather than proven bad - recorded here because publishable=True currently ships three HF repos on it. CONTAMINATION: Pawlowska, Karwat & Zolek (Letter to the Editor, Data in Brief 48:109247, 2023-05-19) found 235 duplicates (~19% of 780), >=8 images in BOTH benign and malignant, 70 axilla images that are not breast tissue, and warn of train/test leakage. MEASURED HERE on the actual npz rather than assumed: at 28x28 exactly ONE pair survives as a pixel-exact duplicate - train[115] label=0 (MALIGNANT) and test[76] label=1 (BENIGN). The same image, trained as malignant and scored as benign. That is 1 of 156 test images, 0.64%, which the model cannot get right and which no amount of training fixes. Loosening to RMSE<=0.05 finds 216 near-duplicate pairs, 120 crossing a split boundary and 6 both crossing AND contradicting - but those are threshold-dependent and the exact pair is the only hard number. So the letter's 19% does NOT wholesale survive downsampling; a real floor on this benchmark does.

## Notes and limitations

Genuine encrypted convolution: im2col + CKKS dot_plain, ReLU/max-pool under MPC.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
