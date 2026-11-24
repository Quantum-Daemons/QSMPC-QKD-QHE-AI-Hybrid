# 3D nodule student (square activation) - LIDC-IDRI chest CT

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. A browser-side network for the medical page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid), served from the site
(`web/public/data/medical/nodule3d_square_student.json`). Pre-registered before any image was fetched
(`references/medical-nodule3d-square-student-preregistration.md`) with an addendum before any model was trained, and
trained, scored and exported by `scripts/train/medical_nodule3d_student.py` and `scripts/export_nodule3d_student.py`.
For research and demonstration only, not for clinical use.

## What it is

One 3 x 3 x 3 convolution with four filters (batch norm folded in), the square $x^2$ as its activation, a 2 x 2 x 2
average pool and a dense layer from $4 \cdot 14^3$ to two logits, on a 28 mm cube at 1 mm around a nodule's centre.
The square makes the whole network a polynomial of depth 3, so it runs end to end under CKKS with no layer decrypted
in between (`web/src/lib/clientFHESquare3d.ts`). It was distilled with soft labels from the best of three pre-registered
teacher candidates, each scored out of fold over the training patients; the decision is Platt scaling of the logit
difference and the threshold that maximises balanced accuracy on validation, both fitted on the validation part.

## Data and split

LIDC-IDRI (TCIA, CC BY 3.0; Armato III et al. 2015, doi:10.7937/K9/TCIA.2015.LO9QL9SX; Armato et al., Medical Physics
38(2), 915-931, 2011, doi:10.1118/1.3528204), every file checked against TCIA's MD5 and every series against its image
count (`docs/measurements/lidc-stream-2026-11-24.json`). One crop per pylidc cluster of the readers' marks (Hancock and
Magnan, Journal of Medical Imaging 3(4), 044504, 2016): 2,651 crops of 875 patients, HU clipped to [-1000, 400] and
scaled to uint8. The label is the readers' median malignancy rating, above 3 malignant and below 3 benign, exactly 3 left
out. The split is by patient, stratified, drawn once from the records before any image was fetched
(`docs/measurements/medical-nodule3d-split.json`): 1,178 labelled training nodules of 518 patients, 170 validation of
74, 347 test of 148; 31.4 % of the test nodules are malignant.

## Measured performance

The exported seed's figures as `weights.json` records them, from the record
`docs/measurements/medical-nodule3d-square-student.json`, on the 347 test nodules.

| quantity | value |
|---|---|
| test AUC, the exported seed's folded float64 network | 0.892568 |
| its 95 % interval over test patients, low (2,000 resamples, seed 0) | 0.847758 |
| its 95 % interval, high | 0.933242 |
| validation AUC (which chose the seed) | 0.838577 |
| balanced accuracy at the validation threshold | 0.846003 |
| mean test AUC of the five square seeds | 0.907301 |
| mean test AUC of the five control seeds (ReLU, same recipe) | 0.882052 |
| majority-class rate on test | 0.685879 |

**Seed position:** the exported seed, 5, has the best validation AUC of the five, which is the pre-registration's rule,
fixed before any run; on test it is the lowest of the five square seeds (0.893 against 0.906 to 0.915). No seed was
picked on test figures.

## The pre-registered comparison

| | test AUC, five seeds | criterion |
|---|---|---|
| square student | $0.907 \pm 0.009$ | 1, accuracy: at least the control's mean minus 0.03 - met |
| control (ReLU) | $0.882 \pm 0.027$ | 2, no collapse: every square seed at least 0.60 and predicting both classes - met |

The teacher candidates, each scored out of fold over the training patients: the 3D ResNet-18 trained from scratch
0.874, TotalSegmentator's nodule size 0.786, and CT-FM's linear probe 0.641; the ResNet-18 taught. CT-FM's probe reads
its features averaged over the whole 64 mm cube, of which the nodule is a small part; that is the pre-registered probe,
reported as measured, not tuned after the fact. NoduleMNIST3D's official-split figures (test AUC: ResNet-18 3D 0.863,
ResNet-50 3D 0.875, auto-sklearn 0.914) are cited for context and not compared: that split shares scans across its
parts and labels differently.

## How it runs on the page

The card on the medical page decodes one of twelve test crops (the addendum's section 4: the first six labelled
malignant and the first six benign, in the records' order, fixed before any model was trained; no patient or series
identifier is shipped), draws it as three orthogonal planes with the exact contribution map, and runs it under CKKS in
a Web Worker, under the parameter sets that met the pre-registered limit.

## Under CKKS, every test crop (the record)

`scripts/train/medical_nodule3d_student.py ckks` ran the page's own module under node-seal on all 347 labelled test
crops, under both pre-registered sets, in node on this project's machine (load average about 1).

| parameter set | $N$ | modulus bits | scale | largest logit error | decisions changed | median time a crop | ciphertexts up, back | bytes up, back |
|---|---|---|---|---|---|---|---|---|
| A2 | 8,192 | 48+34+34+34+52 (202) | $2^{34}$ | $3.2 \times 10^{-5}$ | 0 of 347 | 1.7 s | 162, 2 | 85.0 MB, 262 kB |
| B | 16,384 | 60+40+40+40+60 (240) | $2^{40}$ | $1.6 \times 10^{-5}$ | 0 of 347 | 2.2 s | 81, 2 | 84.9 MB, 525 kB |

Both sets met criterion 3 (every decrypted logit within the pre-registered $10^{-3}$ of the plaintext one, no
decision changed) and both are offered; both are within the HomomorphicEncryption.org standard's 128-bit post-quantum
bound for a ternary secret (202 bits at $N = 8192$, 411 at $N = 16384$). The input travels as the 27 taps of the
convolution, packed over $28^3 = 21{,}952$ voxels in 6 blocks under A2 and 3 under B.

Retrain: `scripts/data/lidc_stream.py` (fetch, verify, cut), then `scripts/train/medical_nodule3d_student.py teachers`,
the CT-FM and TotalSegmentator candidates' own scripts, `students` and `ckks` (the 3D ResNet-18 and the students on the
GPU), then `scripts/export_nodule3d_student.py`.
