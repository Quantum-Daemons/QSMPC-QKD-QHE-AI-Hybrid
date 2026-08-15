# Variational QCNN - medical, 9 wires

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `medical_fl` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It carries no
`hf_repo` and no pinned revision.

## Why this width is 9 and not 7

Medical carries **8 features**, so 7 wires is not available - the trainer refuses it
(`8 features exceeds 7 wires`). Telecom could be narrowed because it carries 7 features on 8 wires;
this use case can only be widened. One spare, unencoded wire is the same question from the other
side.

## The answer, and seed 0 gets it backwards

| accuracy (headline) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | median | range |
|---|---|---|---|---|---|---|---|
| 8 wires (shipped) | 0.7326 | 0.9186 | 0.9070 | 0.8023 | 0.9535 | **0.9070** | [0.7326, 0.9535] |
| 9 wires (this)    | 0.8140 | 0.9186 | 0.8721 | 0.7907 | 0.9070 | **0.8721** | [0.7907, 0.9186] |

**At seed 0 the 9-wire circuit leads by 0.081. At the median it trails by 0.035.** The reason the
single-seed reading flatters the wider circuit is recorded on the 8-wire card and was measured in
Round 42: **seed 0 is the worst of five for the 8-wire model**. Comparing seed 0 to seed 0 therefore
compares this width's ordinary draw against the other width's unluckiest one.

The 9-wire spread is *narrower* ([0.7907, 0.9186] against [0.7326, 0.9535]), which is the only
difference this experiment separates. It is one observation of a variance, not a finding about
variance, and it is not offered as one.

**8 and 9 wires are indistinguishable here at five seeds.**

## Measured, seed 0

| | value |
|---|---|
| accuracy (headline) | **0.813953** |
| minority-class F1 | 0.771429 |
| classical LR accuracy, same split | 0.953488 |
| classical LR minority-class F1 | 0.939394 |
| majority-class accuracy floor | 0.593023 |
| all-positive F1 floor | 0.578512 |
| test prevalence | 0.406977 |
| n_train / n_test | 398 / 171 |

Accuracy leads here because the split is close to balanced (prevalence 0.407), which is the reverse
of the finance card. **This circuit beats its floors and loses to classical logistic regression**,
the same verdict the 8-wire model carries.

**Seed position:** seed 0, sitting between the median and the low end of this width's own five
(0.813953 against a median of 0.8721, range [0.7907, 0.9186]). It is **not** the worst of five, which
the 8-wire model's seed 0 is - and that asymmetry is the whole reason the width comparison above is
stated at the median rather than at the seed.

## Data and licence

Wisconsin Diagnostic Breast Cancer (UCI, CC-BY-4.0), 8 of 30 features. **Not the corpus the page's
published model trains on** - the registered medical models train on MedMNIST v2 BreastMNIST
ultrasound, a different modality and a different cohort. This circuit is a demonstration on tabular
morphometry and its numbers are not comparable to the ResNet figures on the same page.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase medical --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 9

That command is recorded inside `weights.json` and **writes this directory**. Until Round 43 the
trainer wrote `models/qcnn_<usecase>` at every width, so a recorded `--qubits` command reproduced the
weights and not the artifact. The output directory is now a function of the width.

Wall clock: 6.4 s on the reference host (i5-13600K, single thread, state-vector simulation at
2^9 = 512 amplitudes).
