# Variational QCNN - finance, 9 wires

> **Round 47 retrained this artifact after two circuit corrections.** Current headline
> `minority_class_f1` = **0.84058**. The card this replaces contradicted itself: its banner gave the
> minority-class F1 as 0.976667 while its own table gave 0.813559, and 0.976667 is the number that
> table printed as the ACCURACY. A rewrite had written accuracy into the F1 field. Both figures
> belonged to the Round 46 circuit in any case, and neither is this artifact's.
>
> Round 46's banner is superseded rather than deleted: it re-measured this artifact after the
> hardware-efficient ansatz was replaced with a real QCNN, and its figures now belong to a retired
> circuit.

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
digits both artifacts record: minority-class F1 0.84058, accuracy 0.981667, and the same floors,
prevalence and split sizes. The medical pair says the same thing at the same two widths, so this is
now measured twice on two corpora.

The mechanism is the shared filter. One two-qubit filter is applied at every position, so no wire has
angles of its own and a pad held in |0> is capacity the filter cannot allocate. That is a property of
the architecture rather than of this corpus, which is why it reproduces exactly rather than
approximately.

The trained parameter vectors are not bit-identical: 4 of 77 angles differ by more than 1e-4. All
four are angles the `<Z0>` readout cannot see - three are the final RZ on the wire each pooling stage
discards, one is the FC RZ on the measured survivor - and moving all four by 1.234 radians through
the trainer's own `circuit()` changes `<Z0>` by 3.6e-07 on a 64-row batch, against 0.37 for a
neighbouring conv angle. They carry no gradient signal and drift on numerical noise. No published
number moves with them.

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
| `minority_class_f1` (headline) | 0.84058 |
| logistic regression, same split, minority-class F1 | 0.865672 |
| all-positive F1 floor | 0.098257 |
| accuracy (companion, not the headline) | 0.981667 |
| logistic regression accuracy | 0.985 |
| majority-class accuracy floor | 0.948333 |
| test prevalence | 0.051667 |
| n_train / n_test | 2800 / 1200 |

Accuracy is not the headline and is shown only so the floor beside it can be read: at a prevalence of
0.051667, predicting "legitimate" for every row scores 0.948333. **This circuit beats its F1 floor and
loses to classical logistic regression on the same split**, by 0.025092 of minority-class F1, which is
the same verdict the 8-wire model carries and the same one it carried before the retrain.

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
