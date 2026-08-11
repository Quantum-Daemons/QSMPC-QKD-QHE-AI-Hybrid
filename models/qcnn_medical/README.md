# Variational QCNN - medical

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `medical` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It is not
one of the tiered published models, so it carries no `hf_repo` and no pinned revision. This file
exists because Round 28 found that these weights drive the class surfaced on a use-case page while
appearing in no model card and no registry entry - an artefact with real influence and no
documentation.

## What it is

3 layers of trainable RY rotations plus a CNOT ring on 8 qubits, data encoded as `RY(tanh(z) *
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

## Data and licence

Wisconsin Diagnostic Breast Cancer (UCI, CC-BY-4.0), 8 of 30 features

**Licence: CC-BY-4.0.** Wisconsin Diagnostic Breast Cancer (UCI dataset 17). Attribution required;
commercial use permitted.

## Scope

Simulated state-vector evaluation, not quantum hardware. The circuit is a demonstration that the
private-inference selector surfaces a class from something with trained parameters; it is not
offered as a competitive classifier for this task.
