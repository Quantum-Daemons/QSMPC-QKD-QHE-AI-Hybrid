# Variational QCNN - finance

> **Round 47 retrained this artifact a SECOND time, after finding that 4 of its 77 parameters could
> not move the readout at any input - and this is the use case where the fix made the headline
> WORSE.** Current headline `minority_class_f1` = **0.735294**, against 0.84058 on the card this
> replaces: **-0.105**. Three of the four QCNN artifacts gained from the same change and this one
> regressed. It is not presented as a clean win anywhere on these six cards.
>
> The four dead angles were `theta[76]`, the fully-connected block's RZ, applied AFTER the RY with
> only a CNOT ring and `|amplitude|^2` between it and the readout - a diagonal gate applied last
> cannot move a computational-basis readout - and `theta[23]`, `theta[47]` and `theta[71]`, the 8th
> conv angle of each of the three stages' final layer round, each an `rz` on the wire that stage's
> pooling is about to discard. **The fix was ORDERING, not deletion**: RZ before RY in both places,
> so the phases RZ writes are mixed back into amplitudes by the RY that follows. `theta` is still 77
> scalars and all 77 are now reachable.
> `tests/test_the_qcnn_has_the_structure_the_name_claims.py::test_every_trainable_parameter_reaches_the_readout`
> perturbs every parameter at five (width, readout, layers) configurations and fails if one cannot
> move the output.
>
> Two earlier figures on this card are superseded rather than deleted: 0.714286 belongs to the
> retired hardware-efficient ansatz Round 46 replaced, and 0.84058 to this circuit with four of its
> angles inert.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `finance_aml` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It is not
one of the tiered published models, so it carries no `hf_repo` and no pinned revision. This file
exists because Round 28 found that these weights drive the class surfaced on a use-case page while
appearing in no model card and no registry entry - an artefact with real influence and no
documentation.

## What it is

3 conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RZ,RY) per stage, with pooling by a
rotation on the survivor conditioned on its partner, on 8 qubits, reducing the register
8 -> 4 -> 2 -> 1. A fully-connected block then acts on what survives, which here is the single wire
0. Data is encoded as `RY(tanh(z) * pi)` and the circuit is read out as the expectation of Z on qubit
0, mapped through a learned scale and bias.

`theta` is **77** scalars: 72 conv (3 stages x 3 rounds x 8 angles in the shared filter), 3 pool
angles, and 2 in the fully-connected block. It was 76 until Round 47, when the fully-connected block
stopped being one shared scalar and became a per-survivor RZ+RY. Cong, Choi & Lukin specify "a
unitary F on the remaining qubits"; weight sharing is what makes a CONVOLUTION a convolution and
belongs to the conv layers only. The count stays O(log N) in the register width because pooling stops
at a bounded survivor count, not because F is shared.

**Within the filter and within F, RZ precedes RY, and that ordering is load-bearing.** Applied after
the RY, an RZ on the measured survivor is a diagonal gate applied last and commutes with the `Z` it
is measured in; an RZ on the wire pooling is about to discard is followed only by that wire's use as
a CRY control, which is diagonal in its basis, and then by the readout ignoring it. Measured under
the old order, 4 of these 77 angles could not move the readout at any input.

Trained by backpropagation through a dense state vector, then reimplemented independently in plain
numpy (`src/qsh/qhe/qcnn_trained.py`). Re-measured on the weights shipped here, the two agree to
**1.1e-06**, stable at that value for every row count tried (8, 16, 24, 64, 256) - float32 noise.
The often-quoted **2.4e-07** is HISTORICAL: it was measured on the pre-Round-47 circuit, and it is
how a crossed CNOT ring in the trainer was found. The gate re-measures the agreement on every run
at a 1e-4 tolerance, which is why the drift across two retrains was not caught by the suite.

The prediction path applies NO QOTP mask. Undoing a Pauli frame through Clifford+T is not
implemented in this repository, so hiding and prediction are kept separate rather than claiming a
correction that does not exist.

## Measured performance

Held-out split, **600 test rows** against 2800 training rows, prevalence 0.051667. The headline
metric is **minority_class_f1**, chosen by the corpus rather than by preference. 4000 rows split
70 / 15 / 15 gives 2800 train, 600 validation, 600 test; every figure below is computed on those
600 test rows only. An earlier revision of this card said 1200, which was validation PLUS test: the
trainer recorded `n_test` as everything after the training cut. The metrics were always computed on
the 600-row slice, so only the row count was wrong.

| quantity | value |
|---|---|
| `minority_class_f1` (this circuit) | 0.735294 |
| logistic regression, identical split, minority-class F1 | 0.865672 |
| all-positive F1 floor | 0.098257 |
| accuracy (companion, not the headline) | 0.97 |
| logistic regression accuracy | 0.985 |
| majority-class accuracy floor | 0.948333 |
| test prevalence | 0.051667 |
| n_train / n_val / n_test | 2800 / 600 / 600 |

**The classical baseline is ahead**, by 0.130378 of minority-class F1 and by 0.015 of accuracy.
That is reported here rather than omitted, and the browser log line says the same thing on the page.
Randomly initialised QCNNs are known to operate only on low-bodyness measurements of their input and
to be efficiently classically simulable (arXiv:2408.12739), so a classical model winning on tabular
features of this size is the expected result, not an anomaly. Accuracy is shown only so its floor can
be read beside it: at a prevalence of 0.051667, predicting "legitimate" for every row scores
0.948333, which is why accuracy is not the headline here.

The checkpoint was selected on a validation fold, never on the test rows, and the run is seeded.

**Seed position:** unmeasured for this circuit. The five-seed spread this card used to publish -
0.714286 / 0.324324 / 0.800000 / 0.461538 / 0.786885, median 0.7143 - was measured in Round 42 on the
hardware-efficient ansatz that Round 46 replaced, and is SUPERSEDED. It has not been re-run on the
corrected circuit, so the figure above is one draw and not a median, and the wide spread that
measurement found is a reason to distrust any single draw from this circuit rather than a range that
still applies. Do not re-pick the seed to close a gap: selecting on test performance is test-set
selection, refused in
[`references/qcnn-entanglement-ablation.md`](../../references/qcnn-entanglement-ablation.md).

## This is the artifact the dead-parameter fix made worse

Four parameters that no gradient could reach became reachable, and the minority-class F1 fell from
0.84058 to 0.735294. Across the four distinct circuits the same one-line reordering produced:

| artifact | headline metric | before | after | change |
|---|---|---|---|---|
| `qcnn_telecom` | macro F1 | 0.458324 | 0.510963 | +0.053 |
| `qcnn_telecom_q7` | macro F1 | 0.500452 | 0.541894 | +0.041 |
| `qcnn_medical` | minority-class F1 | 0.8125 | 0.84375 | +0.031 |
| **`qcnn_finance`** | **minority-class F1** | **0.84058** | **0.735294** | **-0.105** |

Three improved and one regressed, and the one that regressed moved further than any of the three that
improved. **This is not a clean win and is not presented as one.**

What is established is narrow and worth stating exactly: four angles that could not affect the output
at any input can now affect it, which is a defect fixed regardless of which way the metric went. What
is NOT established is that the change improves this architecture, because a fix that alters
`torch.randn`'s consumption order also moves the initialisation, and this circuit's retired five-seed
ablation spanned 0.32 to 0.80 on exactly this metric. A single-draw move of 0.105 sits comfortably
inside that width. **Reading either sign of this row as an architectural result would need a seed
sweep that has not been run**, and re-picking the seed to recover the old figure would be test-set
selection.

The Round 47 MULTICLASS readout correction - binning the class index over surviving wires rather than
over wires 0..m-1 - still does not reach this use case at all: `readout.kind` is `expval_z0`, one
expectation through one scalar scale and bias, binary by construction.

## Data and licence

PaySim simulator output (Kaggle ealaxi/paysim1, CC-BY-SA-4.0; Lopez-Rojas et al. 2016) - SYNTHETIC
transactions from aggregated real mobile-money logs, 8 AML features, the same 8 the browser
pipeline feeds

**Licence: CC-BY-SA-4.0 (ShareAlike).** PaySim simulator output (Kaggle ealaxi/paysim1).
ShareAlike propagates to this derivative, so these weights are offered under CC BY-SA 4.0 and NOT
under the repository's Apache-2.0. See DATA_LICENSES.md section 2b for why PaySim is the training
corpus here despite being ruled out for the published models.

## Scope

Simulated state-vector evaluation, not quantum hardware. The circuit is a demonstration that the
private-inference selector surfaces a class from something with trained parameters; it is not
offered as a competitive classifier for this task, and it loses to logistic regression on the split
it was measured on.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase finance --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 8

Recorded inside `weights.json` and, since Round 43, it writes THIS directory: the trainer's output
path is now a function of the width, so a `--qubits` run cannot overwrite a sibling artifact. Until
then every width wrote `models/qcnn_<usecase>`, which is why running the telecom 7-wire line would
have destroyed the 8-wire file the demo serves.

## A second width ships, and it is now provably inert

`models/qcnn_finance_q9` carries a 9-wire circuit - one wire more than the 8 features, so one
unencoded pad. **Every recorded metric is identical between the two widths**: minority-class F1
0.735294, accuracy 0.97, and the same floors, prevalence and split sizes, to all six digits the
artifacts record. The same is true of the medical pair, so the pad-wire result is now measured twice
and in the same direction - and it has now survived two retrains that moved the values, which is a
stronger statement than surviving one.

The mechanism is the shared filter. One two-qubit filter is applied at every position, so there are
no per-wire angles for a data-free wire to claim; a pad in |0> contributes nothing the filter can
allocate capacity to. The pre-Round-46 ansatz gave every wire its own trainable angles, which is
exactly why the same comparison produced a spread worth ablating back then - it was measuring the
ansatz's spurious per-wire capacity rather than a property of the data.

**The trained parameter vectors are now near-identical too, and that is a consequence of the
dead-parameter fix.** Before it, 4 of 77 angles differed between the two widths by more than 1e-4 -
`theta[23]`, `theta[47]`, `theta[71]` and `theta[76]`, which are exactly the four the `<Z0>` readout
could not see. They carried no gradient signal, so they drifted on numerical noise while no published
number moved with them. With the ordering corrected those four are reachable and no longer drift:
**zero of 77 angles now differ by more than 1e-4**, the largest disagreement anywhere in the vector
is 1.1e-05, and `scale` and `bias` agree to 4.8e-07 and 6.0e-07. A pad wire changes nothing the
optimiser can use, and now it changes almost nothing at all.

7 wires is not available here: this use case encodes 8 features and `qcnn_variational.py` refuses a
run with fewer wires than features. Telecom can be narrowed because it carries 7. See
[`references/qcnn-width-ablation.md`](../../references/qcnn-width-ablation.md) for all three, and
note that its telecom finding is now superseded in the opposite direction.
