# Variational QCNN - finance, 9 wires

> **Round 46 re-measured this artifact.** The circuit was replaced with a real QCNN (shared two-qubit conv filter, pooling by conditioned rotation, register reduction); the previous figures came from a hardware-efficient ansatz that was labelled a QCNN and was not one. Current headline `minority_class_f1` = **0.976667**.

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

## The answer, and it reverses when you stop using one seed

Seed 0 says the spare wire helps by a wide margin. Five seeds say it does not.

| minority-class F1 | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | median | range |
|---|---|---|---|---|---|---|---|
| 8 wires (shipped) | 0.7143 | 0.3243 | 0.8000 | 0.4615 | 0.7869 | **0.7143** | [0.3243, 0.8000] |
| 9 wires (this)    | 0.8136 | 0.6800 | 0.4444 | 0.7937 | 0.6800 | **0.6800** | [0.4444, 0.8136] |

**At seed 0 the 9-wire circuit leads by 0.099. At the median it trails by 0.034.** A single-seed
width comparison would have reported the spare wire as a clear gain on this use case, and it is not
one. Both spreads are wide enough that neither width is separated from the other by this experiment;
the honest statement is that **8 and 9 wires are indistinguishable here at five seeds**, not that
either wins.

This is exactly what the medical card's Round-42 warning predicted, and it is why `QC20` asks every
card to place its seed rather than print a number.

## Measured, seed 0

| | value |
|---|---|
| minority-class F1 (headline) | **0.813559** |
| accuracy | 0.976667 |
| classical LR, same split, minority-class F1 | 0.865672 |
| classical LR accuracy | 0.985 |
| all-positive F1 floor | 0.098257 |
| majority-class accuracy floor | 0.948333 |
| test prevalence | 0.051667 |
| n_train / n_test | 2800 / 1200 |

Accuracy is not the headline and is shown only so the floor beside it can be read: at a prevalence of
0.0517, predicting "legitimate" for every row scores 0.9483. **This circuit beats its F1 floor and
loses to classical logistic regression on the same split**, which is the same verdict the 8-wire
model carries.

**Seed position:** seed 0, and it is the **best of five** on this width (0.813559 against a median of
0.6800, range [0.4444, 0.8136]). Read the headline above as the high end of a wide spread - the
opposite of the medical 8-wire card, whose seed 0 is the worst of its five, and a reminder that a
seed is not re-picked in either direction. Selecting on test performance is test-set selection and is
refused in `references/qcnn-entanglement-ablation.md`.

## Data and licence

PaySim simulator output (Kaggle `ealaxi/paysim1`, CC-BY-SA-4.0; Lopez-Rojas et al. 2016) - SYNTHETIC
transactions from aggregated real mobile-money logs, 8 AML features, the same 8 the browser pipeline
feeds. Nothing here is a claim about a live payment system.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase finance --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 9

That command is recorded inside `weights.json` and **writes this directory**. Until Round 43 the
trainer wrote `models/qcnn_<usecase>` at every width, so a recorded `--qubits` command reproduced the
weights and not the artifact - running the telecom 7-wire line destroyed the 8-wire file the demo
serves. The output directory is now a function of the width.

Wall clock: 6.6 s on the reference host (i5-13600K, single thread, state-vector simulation at
2^9 = 512 amplitudes).
