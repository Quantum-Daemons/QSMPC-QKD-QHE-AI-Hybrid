# Variational QCNN - medical

> **Round 84 retrained this artifact on the same tree circuit at a new encoding scale, and changed
> the units it is fed.**
>
> * **The encoding** is RY(c * pi * tanh(z)) with c = 0.5, chosen by a pre-registered five-seed
>   experiment (2026-10-30). At c = 1 the two tails of every feature fold onto one state - RY(-pi)|0>
>   and RY(pi)|0> are the same ray - and on WDBC that is not a corner case: the fidelity between a
>   feature's 1st and 99th percentile encodings was 0.94 to 0.996 on all eight features, so a very
>   small and a very large tumour encoded almost alike. At c = 0.5 it is at most 0.015. Over five
>   seeds c = 0.5 gained +0.042 accuracy, better on 5 of 5.
> * **The units** are raw WDBC values (`input_space: "raw"`), standardised by every inference path
>   with this file's own `feature_mean` / `feature_std`. It was `zscore`, fed the bundled sample's
>   z-scores as they came, and those were made with another scaler (the unshuffled first 70 percent of
>   the cohort) than the trainer's shuffled training split: 2 of the 86 held-out rows got a different
>   class in the browser than in the trainer. Now the browser and `qcnn_trained.py` give the trainer's
>   class on all 86.
>
> Current headline `accuracy` = **0.953488**, against 0.883721 at c = 1. Logistic regression on the
> identical split ALSO scores **0.953488**: at this seed the two make the same prediction on every one
> of the 86 held-out rows. Over five seeds logistic regression is ahead (below). The 9-wire sibling,
> `qcnn_medical_q9`, was removed in this round.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `medical` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid). It is not
one of the tiered published models, so it carries no `hf_repo` and no pinned revision. This file
exists because Round 28 found that these weights drive the class surfaced on a use-case page while
appearing in no model card and no registry entry - an artefact with real influence and no
documentation.

## What it is

**`circuit.name`: `tree`.** 3 conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RZ,RY) per
stage on the DISJOINT pairs of the active register, with pooling by a CRY on the survivor
conditioned on its partner, on 8 qubits, reducing the register 8 -> 4 -> 2 -> 1. A fully-connected
block then acts on what survives, the single wire 0. Data is encoded as `RY(0.5 * pi * tanh(z))`
with z the raw value standardised by the training split's moments shipped here, and the circuit is
read out as the expectation of Z on qubit 0 through a learned scale and bias.

`theta` is **77** scalars: 72 conv (3 stages x 3 rounds x 8 angles in the shared filter), 3 pool
angles, and 2 in the fully-connected block (RZ then RY: an RZ applied last would commute with the
`Z` it is measured in, which is how 4 of these angles were dead until Round 47). All 77 reach the
readout.

**A tree of two-qubit blocks, and therefore classically simulable in linear time at any width**:
the filter touches disjoint pairs only and a pooled wire is never touched again, so after every stage
the survivors are a product of one-qubit states. The pre-registered overlap experiment covered this
use case too: coupling neighbouring pairs with the same filter gained +0.005 accuracy and Hur et
al.'s SU(4) block +0.023 to +0.028, neither significant over five seeds, so medical keeps the tree.
It is simulated exactly on 2^8 amplitudes, unencrypted, with no QOTP mask: undoing a Pauli frame
through Clifford+T is not implemented in this repository, so hiding and prediction are kept separate
rather than claiming a correction that does not exist.

Three implementations compute it - the torch trainer, `web/src/lib/quantumSim.ts` and the numpy
`src/qsh/qhe/qcnn_trained.py` - and on the 86 held-out rows, fed raw, the browser and the numpy path
agree with the trainer to 1.5e-06 and give its class on every row.

## Measured performance

Held-out split, **86 test rows** against 398 training rows, prevalence 0.406977, seed 0. The headline
metric is **accuracy**, chosen by the corpus rather than by preference: this split is close to
balanced. 569 rows split 70 / 15 / 15 gives 398 train, 85 validation, 86 test; every figure below is
computed on those 86 rows only.

| quantity | value |
|---|---|
| `accuracy` (this circuit) | 0.953488 |
| logistic regression accuracy, identical split | 0.953488 |
| majority-class accuracy floor | 0.593023 |
| minority-class F1 (companion, not the headline) | 0.939394 |
| logistic regression minority-class F1 | 0.939394 |
| all-positive F1 floor | 0.578512 |
| test prevalence | 0.406977 |
| encoding scale c | 0.5 |
| checkpoint epoch, chosen on the validation rows | 60 |
| n_train / n_val / n_test | 398 / 85 / 86 |

## Against logistic regression

**At this seed the circuit and logistic regression are level, and not by coincidence of totals**:
they make the same prediction on all 86 held-out rows (82 right, 4 wrong each; exact McNemar on the
discordant rows has nothing to test). The page computes the comparison from these two numbers rather
than asserting a winner.

**Over five seeds logistic regression is ahead.** Circuit minus LR: -0.0186 accuracy on average,
worse on 4 seeds and level on 1, two-level bootstrap 95% interval [-0.0395, -0.0023]. So "the
classical model is ahead" is true of this circuit and false of this seed, and a sentence that quotes
seed 0 must say which.

**Seed position:** seed 0 scores 0.9535, second highest of the five seeds the pre-registered
experiment ran on this exact configuration (0.9535, 0.9419, 0.9186, 0.8953, 0.9651; mean 0.9349, sd
0.0280). It is a favourable draw, and the five-seed mean is the figure to compare. The seed was fixed
before the experiment and is deliberately not re-picked: choosing one by test performance is test-set
selection, which
[`references/qcnn-entanglement-ablation.md`](../../references/qcnn-entanglement-ablation.md) refuses.

## Five seeds, from the pre-registered experiment (not this file)

Seed 0 of the bold row IS this artifact (theta, scale and bias equal to max |diff| 0). Not recorded in
`weights.json`, so not checked by the model-card gate.

| configuration, accuracy | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| tree, c = 1 (shipped until Round 84) | 0.8837 | 0.9186 | 0.8953 | 0.8837 | 0.8837 | 0.8930 |
| tree, c = 0.75 (exploratory arm) | 0.9302 | 0.9302 | 0.9419 | 0.9070 | 0.9535 | 0.9326 |
| **tree, c = 0.5 (this artifact)** | **0.9535** | 0.9419 | 0.9186 | 0.8953 | 0.9651 | **0.9349** |
| logistic regression | 0.9535 | 0.9651 | 0.9651 | 0.9070 | 0.9767 | 0.9535 |

c = 0.5 minus c = 1: +0.0419, better on 5 of 5 seeds, two-level 95% interval [+0.0070, +0.0791] -
the pre-registered switch criterion. It closes about 69 percent of the five-seed gap to logistic
regression and not the rest.

## The units it is fed, and why they changed

The bundled sample for this page ships z-scores, and the trainer standardises with the moments of its
own shuffled training split. Same units, different constants, so feeding the sample's z-scores
straight to a `zscore` model put each feature a little off where training put it; on the held-out
rows, 2 of 86 crossed the decision boundary. Shipping a copy of the sample's scaler in this file would
only have moved the problem - the sample is rebuilt for reasons of its own - so the model now takes
raw WDBC values and standardises them with its own moments, exactly as finance does. The page must
hand it raw rows: `tests/test_the_encoder_receives_the_space_the_artifact_declares.py` refuses a
sample that does not ship rows in the space this artifact declares, and the Python pipeline must pass
the eight raw columns this model names in `feature_names`.

## Superseded

* **0.883721** (c = 1, Round 47 to Round 84): the same circuit at the full angle; minority-class F1
  0.84375. Before that, 0.860465 with four of its angles inert, and 0.732558 on the hardware-efficient
  ansatz Round 46 replaced. The four dead angles were `theta[76]`, the fully-connected RZ applied
  after the RY, and `theta[23]`, `theta[47]`, `theta[71]`, an RZ on each stage's discarded wire; the
  fix was ordering, not deletion.
* **The 9-wire sibling** (`models/qcnn_medical_q9`, removed in Round 84). Its ninth wire carried no
  feature and was folded into wire 0 without being convolved, so it gave the same class as this
  circuit on every row and identical metrics at five of five seeds. An option that changes nothing
  was removed rather than explained. 7 wires is not available: this use case encodes 8 features and
  the trainer refuses fewer wires than features.

## Data and licence

Wisconsin Diagnostic Breast Cancer (UCI, CC-BY-4.0), 8 of 30 features

**Licence: CC-BY-4.0.** Wisconsin Diagnostic Breast Cancer (UCI dataset 17). Attribution required;
commercial use permitted.

## Scope

Simulated state-vector evaluation, not quantum hardware, and not a claim of quantum advantage. The
circuit demonstrates that the inference selector surfaces a class from something with trained
parameters; it is not offered as a competitive classifier, it ties logistic regression at this seed
and trails it over five.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase medical --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 8 --arch tree --encoding-scale 0.5

Recorded inside `weights.json` with the device it ran on (`training.device` cuda); it writes THIS
directory. On CUDA it regenerates `theta`, `scale`, `bias` and every metric bit for bit: the
pre-registered experiment's run and the retrain that wrote this file agree to max |diff| 0.
