# Variational QCNN - medical

> **Round 47 retrained this artifact a SECOND time, after finding that 4 of its 77 parameters could
> not move the readout at any input.** Current headline `accuracy` = **0.883721**, against 0.860465
> on the card this replaces; the companion minority-class F1 went 0.8125 to **0.84375**, **+0.031**.
> This is one of the three artifacts the fix improved. The fourth, `qcnn_finance`, regressed by
> 0.105 under the identical change, so the result across the set is mixed and is reported that way.
>
> The four dead angles were `theta[76]`, the fully-connected block's RZ, applied AFTER the RY and
> therefore a diagonal gate commuting with the `Z` it is measured in, and `theta[23]`, `theta[47]`
> and `theta[71]`, the 8th conv angle of each of the three stages' final layer round - an `rz` on
> the wire that stage's pooling is about to discard, whose only later interaction is as a CRY
> control. **The fix was ORDERING, not deletion**: RZ before RY in both places, so the phases RZ
> writes are mixed back into amplitudes by the RY that follows. `theta` is still 77 scalars and all
> 77 are now reachable.
> `tests/test_the_qcnn_has_the_structure_the_name_claims.py::test_every_trainable_parameter_reaches_the_readout`
> perturbs every parameter at five (width, readout, layers) configurations and fails if one cannot
> move the output.
>
> Two earlier figures on this card are superseded rather than deleted: 0.732558 belongs to the
> retired hardware-efficient ansatz Round 46 replaced, and 0.860465 to this circuit with four of its
> angles inert.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `medical` page of
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
numpy (`src/qsh/qhe/qcnn_trained.py`); the two agree to 2.4e-07, which is how a crossed CNOT ring
in the trainer was found.

The prediction path applies NO QOTP mask. Undoing a Pauli frame through Clifford+T is not
implemented in this repository, so hiding and prediction are kept separate rather than claiming a
correction that does not exist.

## Measured performance

Held-out split, 171 test rows against 398 training rows, prevalence 0.406977. The headline metric
is **accuracy**, chosen by the corpus rather than by preference: this split is close to balanced.

| quantity | value |
|---|---|
| `accuracy` (this circuit) | 0.883721 |
| logistic regression accuracy, identical split | 0.953488 |
| majority-class accuracy floor | 0.593023 |
| minority-class F1 (companion, not the headline) | 0.84375 |
| logistic regression minority-class F1 | 0.939394 |
| all-positive F1 floor | 0.578512 |
| test prevalence | 0.406977 |
| n_train / n_test | 398 / 171 |

**The classical baseline is ahead**, by 0.069767 of accuracy and by 0.095644 of minority-class F1.
That is reported here rather than omitted, and the browser log line says the same thing on the page.
Randomly initialised QCNNs are known to operate only on low-bodyness measurements of their input and
to be efficiently classically simulable (arXiv:2408.12739), so a classical model winning on tabular
features of this size is the expected result, not an anomaly. The circuit does clear both of its
floors, which is all the aggregate establishes.

The checkpoint was selected on a validation fold, never on the test rows. An earlier revision
selected and reported on the SAME rows and was unseeded, so its figure was a draw from a
distribution rather than a measurement. Those reruns, and the five-seed spread that replaced them,
are recorded below as superseded rather than deleted.

**Seed position:** unmeasured for this circuit. The five-seed spread this card used to publish -
minority-class F1 0.622951 / 0.892308 / 0.870968 / 0.738462 / 0.933333, median 0.8710, with seed 0
the WORST of the five - was measured in Round 42 on the hardware-efficient ansatz that Round 46
replaced, and is SUPERSEDED. So is the unseeded reread spread that preceded it (0.9181 / 0.9064 /
0.8947 / 0.8480). Neither has been re-run on the corrected circuit, so the figure above is one draw
and not a median, and the width of that old spread is a reason to distrust any single draw from this
circuit rather than a range that still applies. The seed is deliberately NOT re-picked: choosing one
by test-set performance is test-set selection, which the validation-fold checkpoint selection exists
to prevent, and
[`references/qcnn-entanglement-ablation.md`](../../references/qcnn-entanglement-ablation.md) refuses
it.

## Where this artifact sits in the dead-parameter result, which is mixed

Four parameters that no gradient could reach became reachable. Across the four distinct circuits the
same one-line reordering produced:

| artifact | headline metric | before | after | change |
|---|---|---|---|---|
| `qcnn_telecom` | macro F1 | 0.458324 | 0.510963 | +0.053 |
| `qcnn_telecom_q7` | macro F1 | 0.500452 | 0.541894 | +0.041 |
| **`qcnn_medical`** | **minority-class F1** | **0.8125** | **0.84375** | **+0.031** |
| `qcnn_finance` | minority-class F1 | 0.84058 | 0.735294 | -0.105 |

Three improved and one regressed, and the regression is larger than any of the three gains. **The
result is mixed and is not presented as a clean win**, on this card or on any of the other five.

What is established is that four angles which could not affect the output at any input can now affect
it - a defect fixed regardless of which way each metric went. What is NOT established is that the
change improves this architecture: reordering the gates also changes how `torch.randn` is consumed,
so the initialisation moved as well, and this circuit's retired five-seed ablation spanned 0.62 to
0.93 on minority-class F1. A single-draw move of 0.031 is well inside that width. Reading this row as
an architectural gain would need a seed sweep that has not been run.

The Round 47 MULTICLASS readout correction - binning the class index over surviving wires rather than
over wires 0..m-1 - still does not reach this use case at all: `readout.kind` is `expval_z0`, one
expectation through one scalar scale and bias, binary by construction.

## Data and licence

Wisconsin Diagnostic Breast Cancer (UCI, CC-BY-4.0), 8 of 30 features

**Licence: CC-BY-4.0.** Wisconsin Diagnostic Breast Cancer (UCI dataset 17). Attribution required;
commercial use permitted.

## Scope

Simulated state-vector evaluation, not quantum hardware. The circuit is a demonstration that the
private-inference selector surfaces a class from something with trained parameters; it is not
offered as a competitive classifier for this task, and it loses to logistic regression on the split
it was measured on.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase medical --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 8

Recorded inside `weights.json` and, since Round 43, it writes THIS directory: the trainer's output
path is now a function of the width, so a `--qubits` run cannot overwrite a sibling artifact. Until
then every width wrote `models/qcnn_<usecase>`, which is why running the telecom 7-wire line would
have destroyed the 8-wire file the demo serves.

## A second width ships, and it is now provably inert

`models/qcnn_medical_q9` carries a 9-wire circuit - one wire more than the 8 features, so one
unencoded pad. **Every recorded metric is identical between the two widths**: accuracy 0.883721,
minority-class F1 0.84375, and the same floors, prevalence and split sizes, to all six digits the
artifacts record. The same is true of the finance pair, so the pad-wire result is measured twice and
in the same direction - and it has now survived two retrains that moved the values underneath it,
which is a stronger statement than surviving one.

The mechanism is the shared filter. One two-qubit filter is applied at every position, so there are
no per-wire angles for a data-free wire to claim; a pad in |0> contributes nothing the filter can
allocate capacity to. The pre-Round-46 ansatz gave every wire its own trainable angles, which is
exactly why the same comparison produced a spread worth ablating back then - it was measuring the
ansatz's spurious per-wire capacity rather than a property of the data. The five-seed medians this
card used to quote for the width comparison (8 wires 0.9070, 9 wires 0.8721 on accuracy) belong to
that ansatz and are superseded.

**The trained parameter vectors are now near-identical too, and the dead-parameter fix is why.**
Before it, 4 of 77 angles differed between the two widths by more than 1e-4 - `theta[23]`,
`theta[47]`, `theta[71]` and `theta[76]`, exactly the four the `<Z0>` readout could not see. Carrying
no gradient signal, they drifted on numerical noise while no published number moved with them. With
the ordering corrected they are reachable and no longer drift: **zero of 77 angles now differ by more
than 1e-4**, the largest disagreement anywhere in the vector is 4.2e-06, `scale` is bit-identical
between the widths and `bias` agrees to 1.8e-07.

7 wires is not available here: this use case encodes 8 features and `qcnn_variational.py` refuses a
run with fewer wires than features. Telecom can be narrowed because it carries 7. See
[`references/qcnn-width-ablation.md`](../../references/qcnn-width-ablation.md) for all three, and
note that its telecom finding is now superseded in the opposite direction.
