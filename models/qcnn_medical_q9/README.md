# Variational QCNN - medical, 9 wires

> **Round 47 retrained this artifact a SECOND time, after finding that 4 of its 77 parameters could
> not move the readout at any input.** Current headline `accuracy` = **0.883721**, against 0.860465
> on the card this replaces; the current minority-class F1 is **0.84375**, against 0.8125, so
> **+0.031** on the companion metric. This is one of the three artifacts the fix improved -
> `qcnn_finance` regressed by 0.105 under the identical change, so the result across the set is
> mixed and neither this card nor any of the other five presents it as a clean win.
>
> The four dead angles were `theta[76]`, the fully-connected block's RZ applied AFTER its RY and so
> commuting with the `Z` it is measured in, and `theta[23]`, `theta[47]` and `theta[71]`, the 8th
> conv angle of each stage's final layer round - an `rz` on the wire that stage's pooling is about to
> discard. **The fix was ORDERING, not deletion**: RZ before RY, so the phases RZ writes are mixed
> back into amplitudes by the RY that follows. `theta` is still 77 scalars and all 77 are now
> reachable.
>
> Two earlier banners are superseded rather than deleted. The one before last contradicted itself -
> it gave the accuracy as 0.918605 while its own table gave 0.813953, and printed 0.918605 as the
> MINORITY-CLASS F1 - and both of those figures belong to the Round 46 circuit. 0.860465 belongs to
> this circuit with four of its angles inert.

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
digits both artifacts record: accuracy 0.883721, minority-class F1 0.84375, and the same floors,
prevalence and split sizes. The finance pair says the same thing at the same two widths, so this is
measured twice on two corpora - and it has now survived two retrains that moved the values
underneath it, which is a stronger statement than surviving one.

The mechanism is the shared filter. One two-qubit filter is applied at every position, so no wire has
angles of its own and a pad held in |0> is capacity the filter cannot allocate. That is a property of
the architecture rather than of this corpus, which is why it reproduces exactly rather than
approximately.

**The parameter vectors converged too, and the dead-parameter fix is why.** Before it, 4 of 77 angles
differed between the widths by more than 1e-4 - `theta[23]`, `theta[47]`, `theta[71]` and
`theta[76]`, exactly the four the `<Z0>` readout could not see. Carrying no gradient signal, they
drifted on numerical noise while no published number moved with them. Now that the ordering makes
them reachable, **zero of 77 angles differ by more than 1e-4**: the largest disagreement anywhere in
the vector is 4.2e-06, `scale` is bit-identical between the widths and `bias` agrees to 1.8e-07.

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
| `accuracy` (headline) | 0.883721 |
| logistic regression accuracy, same split | 0.953488 |
| majority-class accuracy floor | 0.593023 |
| minority-class F1 (companion, not the headline) | 0.84375 |
| logistic regression minority-class F1 | 0.939394 |
| all-positive F1 floor | 0.578512 |
| test prevalence | 0.406977 |
| n_train / n_test | 398 / 171 |

Accuracy leads here because the split is close to balanced (prevalence 0.406977), which is the
reverse of the finance card. **This circuit beats both of its floors and loses to classical logistic
regression**, by 0.069767 of accuracy and 0.095644 of minority-class F1 - the same verdict the 8-wire
model carries, on numerically the same run. The gap narrowed with this retrain; it did not close, and
it did not change sign.

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
