---
license: other
license_name: nih-chestx-ray14-terms
license_link: https://nihcc.app.box.com/v/ChestXray-NIHCC
library_name: onnx
pipeline_tag: image-classification
tags:
  - quantum-safe
  - privacy-preserving
  - medical-fl
  - onnx
  - onnxruntime-web
---

# ResNet-18 on NIH ChestX-ray14, Effusion, FedAvg over four simulated hospitals

ResNet-18 @224px, 1-channel stem, one logit, trained from random initialisation

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **plaintext** model for the `medical_fl` use case. No code path
evaluates this model under encryption; what each use case encrypts, if anything,
is stated in the notes below.

## Measured performance

| metric | value |
|---|---|
| `auroc` | 0.80645222636351 |
| `auroc_ap` | 0.774081931867184 |
| `auroc_int8` | 0.8062635260132024 |
| `auroc_pa` | 0.8428006953243197 |
| `features_int8_head_auroc` | 0.8062589191862829 |
| `int8_decisions_differing_at_operating_point` | 134 |
| `n_parameters` | 11170753 |
| `n_test_images` | 25589 |
| `n_test_patients` | 2797 |
| `onnx_vs_torch_max_abs_logit` | 1.2159347534179688e-05 |
| `operating_point_logit` | -2.2849953174591064 |
| `sensitivity_at_operating_point` | 0.882946735395189 |
| `specificity_at_operating_point` | 0.5492762623608656 |
| `test_positives` | 4656 |
| `wall_clock_s` | 7513.3 |

## Published baselines this is measured against

- **Target metric**: AUROC, Effusion, official test split
- **Baseline to beat**: 0.7585 - Wang et al., ChestX-ray8, CVPR 2017 (arXiv:1705.02315 v5, Table 17): Effusion AUROC 0.7585 on the published split, a ResNet-50 from ImageNet with only its last layers trained, for all 14 findings
- **Companion metric shown alongside**: `auroc_ap` - the AUROC on AP films alone, the view the model does worse on, shown beside the overall figure because the test split holds a larger share of AP films than training did.

### Gap to the published baseline

0.806 on the official split, above Wang et al.'s 0.7585 (a ResNet-50 from ImageNet with only its last layers trained, for all 14 findings at once) and below Baltruschat et al.'s 0.818 to 0.822 (ResNet-38 to -101 at 448 px with age, sex and view as inputs, Sci Rep 2019, Table 5). This model reads 224 px, one finding, from random initialisation, under FedAvg over four simulated hospitals: protocols differ, so the comparison places it, and does not rank methods.

## Calibration map

The demo page shows this model's logits and its decision, not a probability, and reads no score through this map. It states what the score would mean as a risk, for a reader who needs one. Fitted on the four hospitals' validation images and scored on the official test split.

$$p = \frac{1}{1 + \exp\!\big(a\,\operatorname{logit}\big(\operatorname{clip}(q,\ \varepsilon,\ 1-\varepsilon)\big) + b\big)}$$

$$\operatorname{logit}(q) = \log\frac{q}{1-q},\qquad \varepsilon = 10^{-12}$$

Here \\(q\\) is the graph's softmax probability of effusion, the sigmoid of z.

| quantity | value |
|---|---|
| `a` | -0.915056249677083 |
| `b` | 0.052483029719335425 |
| fitted on | 8710 validation rows, 841 positive |
| scored on | 25589 test rows, 4656 positive |
| Brier score, raw to calibrated | 0.120466 to 0.12017376 |
| log-loss, raw to calibrated | 0.38164418 to 0.3798849 |
| CORP miscalibration (MCB), raw to calibrated | 0.00100401 to 0.00071176 |
| base-rate Brier score (CORP UNC) | 0.14884622 |

Brier skill against the base rate, \\(1 - \mathrm{BS}/\mathrm{UNC}\\), is 0.1926.

Raw is the sigmoid of z, from a network trained with an unweighted cross-entropy on hospitals where 8 to 14% of images show effusion; on the test split, where 18.2% do, its mean is 18.4%.

## Training data

- **Dataset**: NIH ChestX-ray14
- **Licence**: NIH ChestX-ray14 terms; no restrictions on use, attribution requested
- **Source**: https://nihcc.app.box.com/v/ChestXray-NIHCC (licence read 2026-11-14)
- **Attribution (NIH ChestX-ray14 terms; no restrictions on use, attribution requested)**: NIH Clinical Center, ChestX-ray14, https://nihcc.app.box.com/v/ChestXray-NIHCC. Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR 2017. The data is provided by the NIH Clinical Center.

112,120 frontal chest radiographs of 30,805 patients from the NIH Clinical Center, as 1024 px PNGs, with 14 findings mined from the radiology reports by natural-language processing (Wang et al., CVPR 2017). NIH's FAQ, question 4, read in the Box viewer on 2026-11-14: the usage of the data set is unrestricted, and a link to the download site, an acknowledgement of the NIH Clinical Center and a citation of the CVPR 2017 paper are asked; the README asks for the citation. The optional MAPLEZ labels in the same folder are CC BY-NC 4.0 and are not used. LABELS: mined from the reports, not read from the images; a radiologist re-reading 50 images per finding put Effusion's positive predictive value at 70 to 74% (Oakden-Rayner, Academic Radiology 27:106, 2020). DUPLICATES: 25 groups of pixel-identical images, three of them across the official split, among them the ten pairs NIH's change log of 2026-10-09 lists; training drops 26 images. VIEW: AP films, typically taken at the bedside, carry more effusion than PA films, and the test split more than train_val (AP 20.0% and PA 15.8% against 12.6% and 8.6%).

## Notes and limitations

The medical page's chest X-ray federation, recorded offline (`scripts/train/medical_cxr_fl.py`): NIH's train_val patients placed in four simulated hospitals by how well each patient's AP share matches a hospital's target (realised AP shares 11.9% to 77.4%), then 12 FedAvg rounds of one local epoch, AdamW at 1e-3, batch 128, at 224 px from random initialisation. This is the first of three seeds, fixed before training; the three score 0.803 to 0.806 on the official test split. The hospitals are simulated from one archive, so the federation shows what a split by view does, not what four institutions would. What it learned besides effusion: among films without effusion its score runs higher on AP films (the score alone tells the views apart with AUROC 0.64), and a linear probe on its pooled features tells AP from PA with AUROC 0.991, though an untrained network's features already reach 0.990: the view is in any such network's features, not something this training added. Its class-activation maps put their maximum inside NIH's 153 Effusion boxes 22% of the time, against 3% for a map that always points at the centre. Inputs must be prepared as the set was: the 1024 px PNG resized to 224 px by Pillow's bilinear filter, then pixel/255 (the graph normalises). With Pillow's five other filters, or bilinear sampling without a low-pass (what PyTorch's interpolate without antialias, or a GPU texture lookup without mipmaps, computes), the test AUROC stays between 0.805 and 0.806, but 182 to 620 of the 25,589 decisions at the operating point change (the repository's `docs/measurements/cxr-preupload-2026-11-14.json`). The operating point is the logit at which the hospitals' validation images first reach a sensitivity of 0.80. The labels were mined from reports (Effusion's positive predictive value 70 to 74% on a radiologist's re-read). `onnx/features.onnx` and its int8 twin are the same network up to its 512 pooled features, for a head trained in a browser or one evaluated under encryption; the network's own last layer on the int8 features scores AUROC 0.806 on the test split (`features_report.json`). `eval_sample.json` holds 16 films of the official test split at 224 px, four for each class and view, from distinct patients, under NIH's terms with its credit; the int8 graph classifies 10 of them correctly at the operating point, which shows the model running and not how well it ranks. For research and demonstration only: not for clinical use, and not a medical device.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
