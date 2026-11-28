# Pre-registration: the published BreastMNIST models on BUS-BRA, an external test

Written and committed on 2026-11-28, on the owner's choices of that day (download BUS-BRA; pre-register its check;
freeze it), and frozen at its SHA-256 in the script before any of the models below scores any BUS-BRA image. Before
it was frozen, the CKKS stage below was tried on 40 images of BreastMNIST's test split (not BUS-BRA) to check that it
runs: it agreed with the float64 networks to within 1.1e-5 and changed no decision.

## What is known, said first

The four published BreastMNIST models were trained and tested on BreastMNIST, a 28x28 resize of BUSI (one site in
Egypt), whose source carries 155 duplicate images (Pawlowska et al., Data in Brief 48:109247, 2023; `docs/findings.md`),
so its test figures may be optimistic. Each model's figure is on BreastMNIST's official test split: ResNet-18,
ResNet-50, the CKKS student and the square student (the registry's records). Nothing has measured them on images from
another site. The EDA of 2026-11-28 read only BUS-BRA's metadata and image sizes: 1,875 images of 1,064
patients from Brazil, biopsy-proven (benign 1,268, malignant 607), BI-RADS 2 to 5, four devices (GE Logiq 7: 902,
GE Logiq 5: 809, Toshiba Aplio 300: 139, U-Systems: 25), each with a tumour box and a mask. Its images are 240 to 579
pixels wide (median 286) and 268 to 478 tall (median 383), where MedMNIST describes BUSI's as 500 x 500 ("The source images of 1x500x500 are resized into 1x28x28", MedMNIST v2,
arXiv:2110.14795v2, Methods): the two are
framed differently, so a fall in accuracy may come from the framing as much as from the site.

## The question

How do the four published models score on an external site, and does each student's CKKS path still agree with its
plaintext network on images it was not built for?

## The data

BUS-BRA (Zenodo 8231412, CC BY 4.0 on the record; the archive's LICENSE.txt, in MIT-style terms, requires that work
using it cite Gomez-Flores, Gregorio-Calas and Pereira, "BUS-BRA: A breast ultrasound dataset for assessing
computer-aided diagnosis systems", Medical Physics 51(4):3110-3123, doi:10.1002/mp.16812), every image, with its
`bus_data.csv` row: the patient is `Case`, the label `Pathology`, the device `Device`. Each image read as 8-bit
grayscale (Pillow's mode L) and resized whole to 28x28 by cubic spline interpolation, the method MedMNIST v2
states for all its datasets ("cubic spline interpolation operation for image resizing", arXiv:2110.14795v2, Design
Principles; read 2026-11-28, the journal version not checked): `skimage.transform.resize` to 28x28 with `order=3`,
`anti_aliasing=True` (its default for a downscale, with its default width; MedMNIST does not state whether it smoothed
first) and `preserve_range=True`, then rounded to the nearest integer and clipped to 0-255 as MedMNIST stores 8-bit
images; then the models' preprocessing as their
feature specs state it (pixel / 255, then (x - 0.5) / 0.5; the students then area-resized to 14x14 as their trainers
do). Labels: malignant is BreastMNIST's class 0, benign its class 1 ("normal or benign"). No image is excluded, and
no model is retrained, recalibrated or thresholded on BUS-BRA.

## The models

Each model is scored with ONNX Runtime on the CPU from a published graph whose SHA-256 is the one the shipped config
pins for it (`web/src/lib/config.defaults.json`), or nothing runs: ResNet-18 from both its published graphs, the int8
build the page prefers and the fp32 graph its recorded figure comes from (the fp32 copy on this machine was found not to
be the pinned file and was replaced by the pinned one, on the owner's approval, before this was frozen); ResNet-50 from
its only published graph, fp32; each student from its fp32 graph at 14x14. Each graph is also scored by the same code on its own published
evaluation sample (BreastMNIST's official test split, all 156 images, in `models/<model>/eval_sample.json`), so each
model's BUS-BRA figures stand beside the same graph's BreastMNIST figures; the registry's recorded figure is printed
beside them. For each student, the float64 network read from its published `weights.json` is checked against its
graph on every BUS-BRA image, and the largest logit difference is reported.

## What is measured, per model

- AUC and AUPRC (average precision, scikit-learn's) of the malignant class, from the model's own score (the malignant
  logit minus the benign one), with a 95 percent bootstrap interval resampled by patient (2,000 resamples, every image
  of a drawn patient, `numpy.random.default_rng(0)`; a resample without both classes adds nothing), beside the same
  graph's BreastMNIST test figure.
- Accuracy and the confusion matrix at the model's own decision rule (the larger logit; a tie is malignant, as the
  page's rule and `numpy.argmax` both take it).
- Per device, the same figures where a device has at least 100 images (so U-Systems is reported as counts only).
- For the CKKS student and the square student: every image through the page's own CKKS module in node, under the
  parameter sets the page offers: the CKKS student by `encryptedStudentInference` (`web/src/lib/clientFHE.ts`, the
  page's context: N = 8,192, moduli of 60+40+40+60 bits, scale 2^40), the square student by `runSquareStudent`
  (`web/src/lib/clientFHESquare.ts`) under each set its published weights list, A2 and B; against the float64
  network on the same input, the largest difference between a decrypted and a plaintext logit, and the decisions that
  differ, on every image. The images are run in chunks of 250, one node process per chunk.

## Stated before any figure

- No criterion decides adoption: nothing on the page or the Hub changes because of a result. Whether a card shows it
  is the owner's choice after the result.
- Expected: lower figures than on BreastMNIST, the size of the fall not predicted.
- The CKKS agreement is held to the page's own rule: the largest logit difference at most 1e-3 and no decision changed
  on an image whose plaintext margin exceeds twice it. A failure is reported as found.

## Not done here

No fine-tuning, no domain adaptation, no threshold moved, no use of the masks or boxes (an image is scored whole, as
BreastMNIST's are), no image published, and no per-patient figure published.

## Compute

CPU throughout: ONNX Runtime for the graphs, NumPy for the students' float64 networks, node for the CKKS stage. Script
`scripts/check_busbra_external.py`; the record is `docs/measurements/medical-busbra-external-<the run's date>.json`,
aggregate figures only, with the deposit's DOI, the archive's SHA-256 and each graph's.
