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

# ResNet-18 on BreastMNIST

ResNet-18 @28px

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **plaintext** model for the `medical_fl` use case; the encrypted
path runs a distilled student, not this model.

## Measured performance

| metric | value |
|---|---|
| `accuracy` | 0.8653846153846154 |
| `auc` | 0.9146825396825395 |
| `best_val_auc` | 0.955723 |
| `macro_f1` | 0.8249158249158248 |
| `n_parameters` | 11171266 |
| `resolution` | 28 |
| `wall_clock_s` | 8.4 |

## Published baselines this is measured against

- **Target metric**: AUC
- **Baseline to beat**: 0.901 - ResNet-18 @28px, MedMNIST v2, Yang et al., Scientific Data 10:41 (2023) (AUC 0.901 / ACC 0.863)
- **Published ceiling**: 0.919 - Google AutoML Vision, MedMNIST v2 (AUC 0.919)
- **Companion metric shown alongside**: `accuracy` - reported together because the aggregate figure can look healthy while the class that matters is not.

## Training data

- **Dataset**: MedMNIST v2 BreastMNIST
- **Licence**: CC-BY-4.0 asserted by MedMNIST; the BUSI authors publish no licence
- **Source**: https://medmnist.com/ (licence read 2026-10-26)
- **Attribution (CC-BY-4.0 asserted by MedMNIST; the BUSI authors publish no licence)**: Yang, Shi, Wei, Liu, Zhao, Ke, Pfister & Ni, 'MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification', Scientific Data 10:41 (2023), and Yang, Shi & Ni, 'MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis', IEEE ISBI 2021; CC BY 4.0. BreastMNIST is derived from Al-Dhabyani, Gomaa, Khaled & Fahmy, 'Dataset of breast ultrasound images', Data in Brief 28:104863 (2020).

780 images at 28x28. The tiny tier; also the source of the FHE-feasible T1 student. TWO problems, both upstream of MedMNIST, both checked 2026-08-10. LICENCE: MedMNIST licenses ITSELF CC BY 4.0, but BreastMNIST is a resize of BUSI (Al-Dhabyani et al., Data in Brief 28:104863, 2020) and the CC BY there attaches to the ARTICLE - PMC carries 'This is an open access article under the CC BY license' and states no separate licence for the data, while the dataset host (scholar.cu.edu.eg) asks only 'If you use this dataset, please cite'. A citation request is not a grant. CHECKED AGAIN 2026-10-26 at every source the authors publish: the host page asks the same in every capture from 2021 to 2026; the article's supplements hold the Baheya Hospital ethics committee's approval to publish the dataset publicly and the waiver of consent, and no licence; the authors made no repository deposit; and every Kaggle mirror carries a licence its uploader chose, none of them an author. So the BUSI authors publish no licence: the data was released publicly with no terms, and the only licence anyone states is MedMNIST's CC BY 4.0 for its derived dataset. The three HF repos stay public on that basis, by decision (2026-10-26). That differs from CIC-IDS2017, declined, and ImageNet, dropped: their providers state conflicting or restrictive terms, and BUSI's state none. CONTAMINATION: Pawlowska, Karwat & Zolek (Letter to the Editor, Data in Brief 48:109247, 2023-05-19) found 235 duplicates (~19% of 780), >=8 images in BOTH benign and malignant, 70 axilla images that are not breast tissue, and warn of train/test leakage. MEASURED HERE on the actual npz rather than assumed: at 28x28 exactly ONE pair survives as a pixel-exact duplicate - train[115] label=0 (MALIGNANT) and test[76] label=1 (BENIGN). The same image, trained as malignant and scored as benign. That is 1 of 156 test images, 0.64%, which the model cannot get right and which no amount of training fixes. Loosening to RMSE<=0.05 finds 216 near-duplicate pairs, 120 crossing a split boundary and 6 both crossing AND contradicting - but those are threshold-dependent and the exact pair is the only hard number. So the letter's 19% does NOT wholesale survive downsampling; a real floor on this benchmark does.

## Notes and limitations

The small tier: fits the browser comfortably and has a clean published baseline. Its training regime is not the baseline's: torchvision's 7x7 stride-2 stem with max-pool, early-stopped at 39 of at most 60 epochs at lr 3e-4 (manifest.json), where MedMNIST v2's 28-px ResNets use a 3x3 stride-1 stem without max-pool and 100 epochs at lr 1e-3. Close, not like for like.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
