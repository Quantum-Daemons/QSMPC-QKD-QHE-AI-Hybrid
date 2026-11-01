# Variational QCNN - finance

> **Round 84 retrained this artifact on the four columns the finance linear model reads, at four
> wires.** It read all eight columns of the bundled PaySim sample until then, and two of them,
> `device_risk` and `geo_mismatch`, are computed from PaySim's balance columns, which the dataset's
> description says must not be used for fraud detection because fraudulent transactions are
> cancelled. Without them the task is much harder for both models on this split: current headline
> `minority_class_f1` = **0.228571**, and logistic regression on the identical split scores 0.439024.
> The eight-column circuit's 0.735294, against logistic regression's 0.865672, is superseded (below):
> those figures measured the leak as much as either model.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `finance_aml` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It is not
one of the tiered published models, so it carries no `hf_repo` and no pinned revision. This file
exists because Round 28 found that these weights drive the class surfaced on a use-case page while
appearing in no model card and no registry entry - an artefact with real influence and no
documentation.

## What it is

**`circuit.name`: `tree`.** 3 conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RZ,RY) per
stage on the DISJOINT pairs of the active register, with pooling by a CRY on the survivor
conditioned on its partner, on 4 qubits: the first stage pools four wires to two and the second two
to one. A fully-connected block then acts on the single survivor, wire 0. Data is encoded as
`RY(tanh(z) * pi)` (encoding scale c = 1), with `z` the raw value standardised by the training
split's moments shipped here, and the circuit is read out as the expectation of Z on qubit
0 through a learned scale and bias.

**Its inputs, one per wire in this order (`feature_names`)**, are the four columns
`registry.PAYSIM_MODEL_FEATURES` defines: `amount` (log(1 + amount)), `cross_border` (1 if the
PaySim transaction type is TRANSFER or CASH_OUT; PaySim records no geography), `night_tx` (1 if the
step falls in hours 0 to 5 of its day) and `new_payee` (1 if the row is the first in the PaySim file
to name its destination account). The page feeds it the same four columns of the same transaction,
in raw units, from `samples_by_space.raw` of `web/public/data/finance/paysim_sample.json`.

**Why four wires.** The trainer encodes feature `j` on wire `j`, refuses fewer wires than
features and leaves every further wire unencoded, so the feature count is the narrowest width it
accepts and a wider register adds only wires that carry no feature. Four is also a power of two, so
the tree pools it with no orphan wire.

`theta` is **52** scalars: 48 conv (2 stages x 3 rounds x 8 angles in the shared
filter), 2 pool angles and 2 in the fully-connected block (RZ then RY). All 52 reach
the readout. A tree of two-qubit blocks, and therefore classically simulable in linear time at any
width; here it is simulated exactly on 2^4 = 16 amplitudes, unencrypted, with no QOTP mask:
undoing a Pauli frame through Clifford+T is not implemented in this repository, so hiding and
prediction are kept separate rather than claiming a correction that does not exist.

Three implementations compute it - the torch trainer, `web/src/lib/quantumSim.ts` and the numpy
`src/qsh/qhe/qcnn_trained.py`. On the 600 held-out rows, fed raw, the browser and the numpy path
give the trainer's class on every row, and on 8 to 256 synthetic rows the numpy path agrees with
the trainer to at most 1.1e-06, float32 noise.

## Measured performance

Held-out split, **600 test rows** against 2800 training rows, prevalence 0.051667, seed 0. The
headline metric is **minority_class_f1**, chosen by the corpus rather than by preference: at this
prevalence, predicting "legitimate" for every row scores 0.948333 accuracy. 4000 rows split
70 / 15 / 15 gives 2800 train, 600 validation, 600 test; the checkpoint was chosen on the validation
rows and every figure below is computed on the 600 test rows only.

| quantity | value |
|---|---|
| `minority_class_f1` (this circuit) | 0.228571 |
| logistic regression, identical split, minority-class F1 | 0.439024 |
| all-positive F1 floor | 0.098257 |
| accuracy (companion, not the headline) | 0.955 |
| logistic regression accuracy | 0.961667 |
| majority-class accuracy floor | 0.948333 |
| test prevalence | 0.051667 |
| encoding scale c | 1 |
| checkpoint epoch, chosen on the validation rows | 30 |
| n_train / n_val / n_test | 2800 / 600 / 600 |

## Against logistic regression

**At this seed logistic regression is ahead, and on very few rows.** Of the 31 fraud rows among the
600, the circuit flags 4 and logistic regression 9; the circuit flags no legitimate row and logistic
regression 1. The two disagree on 6 of the 600 rows, logistic regression right on 5 of them and the
circuit on 1, so the exact McNemar test on the discordant rows gives p = 0.21875: this split does
not separate them. The page computes its comparison from the two recorded numbers rather than
asserting a winner.

**Over five seeds neither is ahead.** Each seed draws its own split and its own initialisation, and
the circuit leads logistic regression on 2 of the 5 and trails it on 3: circuit minus logistic
regression is +0.0426 minority-class F1 on average, with the five differences spread by 0.2088
(sample sd); paired t(4) = 0.46, p = 0.67, and the exact Wilcoxon signed-rank p is 0.81.

**Seed position:** seed 0 scores 0.2286, the lowest of the five seeds run on this exact
configuration (0.2286, 0.4878, 0.6190, 0.2353, 0.5385; mean 0.4218, sample sd 0.1796). It is an
unfavourable draw, and the five-seed mean is the figure to compare. The seed was fixed before the
runs and is deliberately not re-picked: choosing one by test performance is test-set selection,
which [`references/qcnn-entanglement-ablation.md`](../../references/qcnn-entanglement-ablation.md)
refuses.

## Five seeds, run for this card (not in weights.json)

Seed 0 of the bold row IS this artifact. Seeds 1 to 4 ran the same command with `--out` pointed
outside `models/`, so they did not overwrite it; they are not shipped, so these figures are not
checked by the model-card gate.

| configuration, minority-class F1 | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| **tree, four columns, 4 wires, c = 1 (this artifact)** | **0.2286** | 0.4878 | 0.6190 | 0.2353 | 0.5385 | **0.4218** |
| logistic regression, the same split | 0.4390 | 0.2353 | 0.3529 | 0.2778 | 0.5909 | 0.3792 |

## The encoding scale and the circuit, and why they did not change

The scale stays c = 1. The half-angle experiment of 2026-10-30 tested
`RY(c * pi * tanh(z))` at c = 0.5 and c = 0.75 here as on medical and telecom, on the eight-column
input: at c = 0.5 finance did not gain (-0.002 minority-class F1 over five seeds),
and the c = 0.75 arm (+0.083, exploratory) failed its pre-registered confirmatory run on fresh
seeds 5 to 9 (+0.035, interval [-0.009, +0.080]). The four-column retrain keeps that decision rather
than re-choosing the scale on its own test rows, so whether c = 0.5 helps on four columns is
unmeasured. The circuit stays the tree for the same reason: the overlap experiment, also on the
eight-column input, found coupling neighbouring pairs worse here (-0.053, worse on 5 of 5 seeds).

## Data and licence

PaySim simulator output (Kaggle ealaxi/paysim1, CC-BY-SA-4.0; Lopez-Rojas et al. 2016) - SYNTHETIC
transactions from aggregated real mobile-money logs; the four columns the finance linear model reads
(amount, cross_border, night_tx, new_payee), none built from PaySim's balance columns

The bundled TSV's other four columns are not inputs, for the reasons
`registry.PAYSIM_EXCLUDED_FEATURES` gives: `device_risk` and `geo_mismatch` are computed from the
balance columns (the description's warning is quoted in `registry.PAYSIM_BALANCE_WARNING`),
`amount_zscore` is an affine copy of `amount`, and `velocity_24h` is non-zero in 2 of the 4,000 rows.

**Licence: CC-BY-SA-4.0 (ShareAlike).** PaySim simulator output (Kaggle ealaxi/paysim1).
ShareAlike propagates to this derivative, so these weights are offered under CC BY-SA 4.0 and NOT
under the repository's Apache-2.0. See DATA_LICENSES.md section 2b for why PaySim is the training
corpus here despite being ruled out for the published models.

## Scope

Simulated state-vector evaluation, not quantum hardware, and not a claim of quantum advantage. The
circuit demonstrates that the inference selector surfaces a class from something with trained
parameters; it is not offered as a competitive classifier for this task, and it loses to logistic
regression on the split it was measured on.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase finance --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 4 --arch tree --encoding-scale 1

Recorded inside `weights.json`, and it writes THIS directory: the trainer's unsuffixed directory for
finance is its shipped width, four wires (`SHIPPED_WIDTH` in the trainer). On a CUDA device two
independent runs gave byte-identical files, in about 8 seconds each. `scripts/ship_qcnn_weights.py`
then copies it to `web/public/data/finance/qcnn_weights.json`, the copy the page serves.

## Superseded

* **0.735294** (eight columns at 8 wires, Round 47 to Round 84): minority-class F1, against
  logistic regression's 0.865672 on the same split, accuracy 0.97 against 0.985. Two of its eight
  inputs, `device_risk` and `geo_mismatch`, were built from PaySim's balance columns, so the figures
  measured that as much as the circuit; `amount_zscore` was an affine copy of `amount`, and
  `velocity_24h` was constant in the bundled rows. Before it, 0.84058 on the same circuit with four
  of its angles inert, and 0.714286 on the hardware-efficient ansatz Round 46 replaced.
* **The 9-wire sibling** (`models/qcnn_finance_q9`, removed in Round 84). Its ninth wire carried no
  feature and was folded into wire 0 without being convolved, so it gave identical metrics at five
  of five seeds. An option that changes nothing was removed rather than explained.
