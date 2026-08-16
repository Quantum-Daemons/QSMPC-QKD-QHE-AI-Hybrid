# Variational QCNN - medical, 9 wires

> **Round 47 retrained this artifact after two circuit corrections.** Current headline `accuracy` =
> **0.860465**. The card this replaces contradicted itself: its banner gave the accuracy as 0.918605
> while its own table gave 0.813953, and it printed 0.918605 as the MINORITY-CLASS F1. A rewrite had
> written one field into the other, and both figures belonged to the Round 46 circuit in any case.
> The current minority-class F1 is 0.8125.
>
> Round 46's banner is superseded rather than deleted: it re-measured this artifact after the
> hardware-efficient ansatz was replaced with a real QCNN, and its figures now belong to a retired
> circuit.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `medical_fl` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It carries no
`hf_repo` and no pinned revision.

## Why this width is 9 and not 7

Medical carries **8 features**, so 7 wires is not available - the trainer refuses it
(`8 features exceeds 7 wires`). Telecom could be narrowed because it carries 7 features on 8 wires;
this use case can only be widened. One spare, unencoded wire is the same question from the other
side.

## The answer, on the corrected circuit: no, and now it is exact

**Every recorded metric is identical between 8 and 9 wires.** Not close - identical, to all six
digits both artifacts record: accuracy 0.860465, minority-class F1 0.8125, and the same floors,
prevalence and split sizes. The finance pair says the same thing at the same two widths, so this is
now measured twice on two corpora.

The mechanism is the shared filter. One two-qubit filter is applied at every position, so no wire has
angles of its own and a pad held in |0> is capacity the filter cannot allocate. That is a property of
the architecture rather than of this corpus, which is why it reproduces exactly rather than
approximately.

The trained parameter vectors are not bit-identical: 4 of 77 angles differ by more than 1e-4. All
four are angles the `<Z0>` readout cannot see - three are the final RZ on the wire each pooling stage
discards, one is the FC RZ on the measured survivor - and moving all four by 1.234 radians through
the trainer's own `circuit()` changes `<Z0>` by under 1e-06 on a 64-row batch, against 0.83 for the
FC RY beside them. They carry no gradient signal and drift on numerical noise. No published number
moves with them.

**The five-seed width comparison this card used to publish is SUPERSEDED.** It reported 8 wires at
median 0.9070 and 9 wires at median 0.8721 on accuracy, with seed 0 flattering the wider circuit
because seed 0 was the worst of five for the 8-wire model, and every one of those ten runs was on the
hardware-efficient ansatz Round 46 replaced. That ansatz gave each wire its own trainable angles,
which is precisely why a data-free wire could move the result there and cannot here. Those numbers
stay recorded in [`references/qcnn-width-ablation.md`](../../references/qcnn-width-ablation.md),
marked superseded rather than deleted, because they are the evidence that the conclusion moved.

## Measured, seed 0

| quantity | value |
|---|---|
| `accuracy` (headline) | 0.860465 |
| logistic regression accuracy, same split | 0.953488 |
| majority-class accuracy floor | 0.593023 |
| minority-class F1 (companion, not the headline) | 0.8125 |
| logistic regression minority-class F1 | 0.939394 |
| all-positive F1 floor | 0.578512 |
| test prevalence | 0.406977 |
| n_train / n_test | 398 / 171 |

Accuracy leads here because the split is close to balanced (prevalence 0.406977), which is the
reverse of the finance card. **This circuit beats both of its floors and loses to classical logistic
regression**, by 0.093023 of accuracy and 0.126894 of minority-class F1 - the same verdict the 8-wire
model carries, on numerically the same run.

**Seed position:** unmeasured for this circuit. The five-seed spread that used to place this card's
seed - "between the median and the low end", 0.813953 against a median of 0.8721 - was measured on
the retired ansatz and does not transfer, and no sweep has been run on the corrected circuit. Read
the headline as one draw. The seed is deliberately not re-picked: choosing one by test performance is
test-set selection, and `references/qcnn-entanglement-ablation.md` refuses it.

## Data and licence

Wisconsin Diagnostic Breast Cancer (UCI, CC-BY-4.0), 8 of 30 features. **Not the corpus the page's
published model trains on** - the registered medical models train on MedMNIST v2 BreastMNIST
ultrasound, a different modality and a different cohort. This circuit is a demonstration on tabular
morphometry and its numbers are not comparable to the ResNet figures on the same page, and none of
them is a claim of quantum advantage: classical logistic regression on this same split wins.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase medical --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 9

That command is recorded inside `weights.json` and **writes this directory**. Until Round 43 the
trainer wrote `models/qcnn_<usecase>` at every width, so a recorded `--qubits` command reproduced the
weights and not the artifact. The output directory is now a function of the width.

Wall clock: 6.4 s on the reference host (i5-13600K, single thread, state-vector simulation at
2^9 = 512 amplitudes).
