# Variational QCNN - finance

> **Round 47 retrained this artifact after two circuit corrections.** Current headline
> `minority_class_f1` = **0.84058**, against **0.714286** on the card this replaces - a figure that
> predates the Round 46 rewrite of the circuit and was never updated for it, so this card was stating
> a retired ansatz's number while the browser served a different one. Only the second correction
> reaches this use case, and the section below says why the number moved at all.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `finance_aml` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It is not
one of the tiered published models, so it carries no `hf_repo` and no pinned revision. This file
exists because Round 28 found that these weights drive the class surfaced on a use-case page while
appearing in no model card and no registry entry - an artefact with real influence and no
documentation.

## What it is

3 conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RY,RZ) per stage, with pooling by a
rotation on the survivor conditioned on its partner, on 8 qubits, reducing the register
8 -> 4 -> 2 -> 1. A fully-connected block then acts on what survives, which here is the single wire
0. Data is encoded as `RY(tanh(z) * pi)` and the circuit is read out as the expectation of Z on qubit
0, mapped through a learned scale and bias.

`theta` is **77** scalars: 72 conv (3 stages x 3 rounds x 8 angles in the shared filter), 3 pool
angles, and 2 in the fully-connected block. It was 76 until Round 47, when the fully-connected block
stopped being one shared scalar and became a per-survivor RY+RZ. Cong, Choi & Lukin specify "a
unitary F on the remaining qubits"; weight sharing is what makes a CONVOLUTION a convolution and
belongs to the conv layers only. The count stays O(log N) in the register width because pooling stops
at a bounded survivor count, not because F is shared.

Trained by backpropagation through a dense state vector, then reimplemented independently in plain
numpy (`src/qsh/qhe/qcnn_trained.py`); the two agree to 2.4e-07, which is how a crossed CNOT ring
in the trainer was found.

The prediction path applies NO QOTP mask. Undoing a Pauli frame through Clifford+T is not
implemented in this repository, so hiding and prediction are kept separate rather than claiming a
correction that does not exist.

## Measured performance

Held-out split, 1200 test rows against 2800 training rows, prevalence 0.051667. The headline
metric is **minority_class_f1**, chosen by the corpus rather than by preference.

| quantity | value |
|---|---|
| `minority_class_f1` (this circuit) | 0.84058 |
| logistic regression, identical split, minority-class F1 | 0.865672 |
| all-positive F1 floor | 0.098257 |
| accuracy (companion, not the headline) | 0.981667 |
| logistic regression accuracy | 0.985 |
| majority-class accuracy floor | 0.948333 |
| test prevalence | 0.051667 |
| n_train / n_test | 2800 / 1200 |

**The classical baseline is ahead**, by 0.025092 of minority-class F1 and by 0.003333 of accuracy.
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

## Why this number moved when the readout fix was not about this circuit

The first Round 47 correction is to the MULTICLASS readout, which binned the class index over wires
0..m-1 - a list that includes wires pooling had discarded. This circuit does not use that readout:
`readout.kind` is `expval_z0`, one expectation through one scalar scale and bias, binary by
construction. The correction cannot reach it.

The second correction does reach it, and not through capacity. The new fully-connected block adds an
RZ on the survivor, and RZ commutes with the Z it is measured in, so that angle is invisible to the
readout: moving it by 1.234 radians through the trainer's own `circuit()` changes `<Z0>` by 1.8e-07
on a 64-row batch, while moving the neighbouring FC RY by the same amount changes it by 0.78. What it
does change is `param_count`, 76 to 77 - and `torch.randn(77)` is not `torch.randn(76)` with one
extra draw, so the whole initialisation vector moved and the run landed in a different basin. **The
movement on this use case is a re-initialisation effect, not a gain in expressive power**, and it
should be read together with the wide seed spread recorded above rather than as an improvement the
architecture earned.

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
0.84058, accuracy 0.981667, and the same floors, prevalence and split sizes, to all six digits the
artifacts record. The same is true of the medical pair, so the pad-wire result is now measured twice
and in the same direction.

The mechanism is the shared filter. One two-qubit filter is applied at every position, so there are
no per-wire angles for a data-free wire to claim; a pad in |0> contributes nothing the filter can
allocate capacity to. The pre-Round-46 ansatz gave every wire its own trainable angles, which is
exactly why the same comparison produced a spread worth ablating back then - it was measuring the
ansatz's spurious per-wire capacity rather than a property of the data.

The trained parameter vectors are not bit-identical: 4 of 77 angles differ by more than 1e-4 between
the two widths. All four are angles the `<Z0>` readout cannot see - three are the final RZ on the
wire each pooling stage discards, one is the FC RZ on the measured survivor - and moving all four by
1.234 radians changes `<Z0>` by 3.6e-07, against 0.37 for a neighbouring conv angle. They carry no
gradient signal and drift on numerical noise, which is why the angles differ while no published
number does.

7 wires is not available here: this use case encodes 8 features and `qcnn_variational.py` refuses a
run with fewer wires than features. Telecom can be narrowed because it carries 7. See
[`references/qcnn-width-ablation.md`](../../references/qcnn-width-ablation.md) for all three, and
note that its telecom finding is now superseded in the opposite direction.
