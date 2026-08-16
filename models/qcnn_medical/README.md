# Variational QCNN - medical

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `medical` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It is not
one of the tiered published models, so it carries no `hf_repo` and no pinned revision. This file
exists because Round 28 found that these weights drive the class surfaced on a use-case page while
appearing in no model card and no registry entry - an artefact with real influence and no
documentation.

## What it is

3 conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RY,RZ) with pooling by a rotation on the survivor conditioned on its partner, on 8 qubits, reducing the register 8 -> 4 -> 2 -> 1, then a fully-connected block on the survivors (76 SHARED parameters, O(log N) in the register width), data encoded as `RY(tanh(z) *
pi)`, read out as the expectation of Z on qubit 0 and mapped through a learned scale and bias.
Trained by backpropagation through a dense state vector, then reimplemented independently in plain
numpy (`src/qsh/qhe/qcnn_trained.py`); the two agree to 2.4e-07, which is how a crossed CNOT ring
in the trainer was found.

The prediction path applies NO QOTP mask. Undoing a Pauli frame through Clifford+T is not
implemented in this repository, so hiding and prediction are kept separate rather than claiming a
correction that does not exist.

## Measured performance

Held-out split, 171 test rows against 398 training rows, prevalence 0.406977. The headline metric
is **accuracy**, chosen by the corpus rather than by preference.

| quantity | value |
|---|---|
| `accuracy` (this circuit) | 0.732558 |
| logistic regression, identical split | 0.953488 |
| majority-class rate | 0.593023 |

**The classical baseline is ahead.** That is reported here rather than omitted, and the browser
log line says the same thing on the page. Randomly initialised QCNNs are known to operate only on
low-bodyness measurements of their input and to be efficiently classically simulable
(arXiv:2408.12739), so a classical model winning on tabular features of this size is the expected
result, not an anomaly.

The checkpoint was selected on a validation fold, never on the test rows. An earlier revision
selected and reported on the SAME rows and was unseeded, so its figure was a draw from a
distribution rather than a measurement: reruns on identical inputs returned 0.9181 / 0.9064 /
0.8947 / 0.8480. Those numbers are higher than the one above and should not be compared with it.

Seeding made the figure REPRODUCIBLE. It did not make it typical, and that distinction was measured
on 2026-08-13 rather than assumed. The same command at seeds 0 to 4 returns minority-class F1
0.622951 / 0.892308 / 0.870968 / 0.738462 / 0.933333 - **median 0.8710, range [0.6230, 0.9333]** -
and the seed this card reports, seed 0, is **the worst of the five**. Finance's seed 0, by contrast,
lands exactly on its own median. So the number above is exactly what it says it is and is also an
unlucky draw, and a reader comparing it against the classical baseline is comparing against the
low end of a wide spread.

The seed is deliberately NOT re-picked. Choosing one by test-set performance is test-set selection,
which is the thing the validation-fold checkpoint selection two paragraphs up exists to prevent;
publishing the spread is the defensible response and this is it. Workings, including an
entanglement ablation run over the same seeds:
[`references/qcnn-entanglement-ablation.md`](../../references/qcnn-entanglement-ablation.md).

**Seed position:** seed 0, the WORST of five on minority-class F1 (0.622951 against a median
of 0.8710, range [0.6230, 0.9333]). Read every figure on this card as the low end of that spread.

## Data and licence

Wisconsin Diagnostic Breast Cancer (UCI, CC-BY-4.0), 8 of 30 features

**Licence: CC-BY-4.0.** Wisconsin Diagnostic Breast Cancer (UCI dataset 17). Attribution required;
commercial use permitted.

## Scope

Simulated state-vector evaluation, not quantum hardware. The circuit is a demonstration that the
private-inference selector surfaces a class from something with trained parameters; it is not
offered as a competitive classifier for this task.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase medical --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 8

Recorded inside `weights.json` and, since Round 43, it writes THIS directory: the trainer's output
path is now a function of the width, so a `--qubits` run cannot overwrite a sibling artifact. Until
then every width wrote `models/qcnn_<usecase>`, which is why running the telecom 7-wire line would
have destroyed the 8-wire file the demo serves.

## A second width ships, and it does not win

`models/qcnn_medical_q9` carries a 9-wire circuit - one wire more than the 8 features, so one unencoded
pad, the same question telecom answered from the narrow side. **At seed 0 the 9-wire model leads; at
the five-seed median it trails.** 8 wires median 0.9070, 9 wires median 0.8721 on accuracy; the seed-0 comparison flatters 9 wires because seed 0 is the WORST of five for the 8-wire model, which this card records above.

7 wires is not available here: this use case encodes 8 features and `qcnn_variational.py` refuses a
run with fewer wires than features. Telecom can be narrowed because it carries 7. See
[`references/qcnn-width-ablation.md`](../../references/qcnn-width-ablation.md) for all three.
