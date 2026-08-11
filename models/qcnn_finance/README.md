# Variational QCNN - finance

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `finance` page of
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

Held-out split, 1200 test rows against 2800 training rows, prevalence 0.051667. The headline
metric is **minority_class_f1**, chosen by the corpus rather than by preference.

| quantity | value |
|---|---|
| `minority_class_f1` (this circuit) | 0.714286 |
| logistic regression, identical split | 0.865672 |
| all-positive F1 | 0.098257 |

**The classical baseline is ahead.** That is reported here rather than omitted, and the browser
log line says the same thing on the page. Randomly initialised QCNNs are known to operate only on
low-bodyness measurements of their input and to be efficiently classically simulable
(arXiv:2408.12739), so a classical model winning on tabular features of this size is the expected
result, not an anomaly.

The checkpoint was selected on a validation fold, never on the test rows, and the run is seeded.
An earlier revision of the trainer selected the checkpoint on the reported rows, so figures
produced before the validation split was introduced are not comparable with the one above.

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
offered as a competitive classifier for this task.
