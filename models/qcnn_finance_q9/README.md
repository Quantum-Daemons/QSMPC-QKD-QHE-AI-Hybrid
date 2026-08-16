# Variational QCNN - finance, 9 wires

> **Round 47 retrained this artifact a SECOND time, after finding that 4 of its 77 parameters could
> not move the readout at any input - and this is the use case where the fix made the headline
> WORSE.** Current headline `minority_class_f1` = **0.735294**, against 0.84058 on the card this
> replaces: **-0.105**. Three of the four QCNN circuits gained from the identical change and this
> one regressed; the honest reading, and the reason it is not being re-seeded, is on
> [`models/qcnn_finance/README.md`](../qcnn_finance/README.md).
>
> The four dead angles were `theta[76]`, the fully-connected block's RZ applied AFTER its RY, and
> `theta[23]`, `theta[47]` and `theta[71]`, the 8th conv angle of each stage's final layer round -
> an `rz` on the wire that stage's pooling is about to discard. **The fix was ORDERING, not
> deletion**: RZ before RY, so the phases RZ writes are mixed back into amplitudes by the RY that
> follows. `theta` is still 77 scalars and all 77 are now reachable.
>
> Two earlier banners are superseded rather than deleted. The one before last contradicted itself -
> it gave the minority-class F1 as 0.976667 while its own table gave 0.813559, and 0.976667 was that
> table's ACCURACY - and both of those figures belong to the Round 46 circuit. 0.84058 belongs to
> this circuit with four of its angles inert.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `finance_aml` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It carries no
`hf_repo` and no pinned revision.

## Why a second width exists here, and why it is 9 and not 7

Round 40 gave telecom a second width so the browser's **QCNN qubits** control could reach a real
circuit instead of silently falling back to the 8-wire file. Telecom carries **7 features**, so 7 was
available to it and the 8-wire model's spare wire was the thing worth measuring.

Finance carries **8 features**, so 7 wires is not a choice - the trainer refuses it outright
(`8 features exceeds 7 wires`). The only direction available is wider, and one spare wire is the same
question telecom answered from the other side: **is unencoded capacity worth anything?**

## The answer, on the corrected circuit: no, and now it is exact

**Every recorded metric is identical between 8 and 9 wires.** Not close - identical, to all six
digits both artifacts record: minority-class F1 0.735294, accuracy 0.97, and the same floors,
prevalence and split sizes. The medical pair says the same thing at the same two widths, so this is
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
the vector is 1.1e-05, and `scale` and `bias` agree to 4.8e-07 and 6.0e-07.

**The five-seed width comparison this card used to publish is SUPERSEDED.** It reported 8 wires at
median 0.7143 and 9 wires at median 0.6800 on minority-class F1, with seed 0 flattering the wider
circuit, and every one of those ten runs was on the hardware-efficient ansatz Round 46 replaced. The
old ansatz gave each wire its own trainable angles, which is precisely why a data-free wire could
move the result there and cannot here. Those numbers stay recorded in
[`references/qcnn-width-ablation.md`](../../references/qcnn-width-ablation.md), marked superseded
rather than deleted, because they are the evidence that the conclusion moved.

## Measured, seed 0

| quantity | value |
|---|---|
| `minority_class_f1` (headline) | 0.735294 |
| logistic regression, same split, minority-class F1 | 0.865672 |
| all-positive F1 floor | 0.098257 |
| accuracy (companion, not the headline) | 0.97 |
| logistic regression accuracy | 0.985 |
| majority-class accuracy floor | 0.948333 |
| test prevalence | 0.051667 |
| n_train / n_test | 2800 / 1200 |

Accuracy is not the headline and is shown only so the floor beside it can be read: at a prevalence of
0.051667, predicting "legitimate" for every row scores 0.948333. **This circuit beats its F1 floor and
loses to classical logistic regression on the same split**, by 0.130378 of minority-class F1 and by
0.015 of accuracy - the same verdict the 8-wire model carries and the same direction it carried
before either retrain. The size of the gap grew; the direction did not change.

**Seed position:** unmeasured for this circuit. The five-seed spread that used to place this card's
seed - "best of five", 0.813559 against a median of 0.6800 - was measured on the retired ansatz and
does not transfer, and no sweep has been run on the corrected circuit. Read the headline as one draw.
The seed is deliberately not re-picked: choosing one by test performance is test-set selection, and
`references/qcnn-entanglement-ablation.md` refuses it.

## Data and licence

PaySim simulator output (Kaggle `ealaxi/paysim1`, CC-BY-SA-4.0; Lopez-Rojas et al. 2016) - SYNTHETIC
transactions from aggregated real mobile-money logs, 8 AML features, the same 8 the browser pipeline
feeds. Nothing here is a claim about a live payment system, and nothing here is a claim of quantum
advantage: classical logistic regression on the same split wins.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase finance --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 9

That command is recorded inside `weights.json` and **writes this directory**. Until Round 43 the
trainer wrote `models/qcnn_<usecase>` at every width, so a recorded `--qubits` command reproduced the
weights and not the artifact - running the telecom 7-wire line destroyed the 8-wire file the demo
serves. The output directory is now a function of the width.

Wall clock: 6.6 s on the reference host (i5-13600K, single thread, state-vector simulation at
2^9 = 512 amplitudes).
