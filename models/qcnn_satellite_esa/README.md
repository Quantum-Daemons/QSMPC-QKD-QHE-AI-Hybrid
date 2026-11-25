# Variational QCNN - satellite, ESA-ADB Mission1

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. A browser-side circuit for the satellite page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid), served from the site
(`web/public/data/satellite/satellite_qcnn_esa.json`), second to the OPS-SAT circuit on that page. Trained, scored and
shipped as `references/satellite-qcnn-tn-preregistration.md` fixed before any training (2026-11-24, section B and its
addendum 2).

## What it is

The same circuit as `models/qcnn_satellite_opssat`: the `tree` circuit on 8 wires, 3 filter rounds a stage, RY(0.5 pi
tanh(z)) encoding of raw features standardised by the moments shipped here, <Z> on wire 0 through a learned scale and
bias. A tree tensor network, simulated classically and exactly; not quantum hardware, and not a claim of quantum
advantage.

## Data and split

ESA Anomalies Dataset, Mission1 (Kotowski et al., ESA-ADB; Zenodo 12528696, CC BY 3.0 IGO), channels 41 to 46, on the
shipped Telemanom student's grid (ESA-ADB's zero-order hold at 30 s, a z-score per channel from the unlabelled training
rows before 2006-10-01), cut into non-overlapping 64-step windows (32 minutes); a window is positive when an Anomaly
interval overlaps it (not rare nominal events or communication gaps). Eight features a window: each channel's standard
deviation, the largest one-step change, the largest absolute value (`scripts/data/esa_qcnn_windows.py`). ESA-ADB's own
split: fit before 2006-10-01 (all 712 positive windows and four negatives for each, seed 0: 3,560), validation to
2007-01-01 (4,139 windows, 2 positive), test from 2007-01-01 (115,064 windows, 1,077 positive). By addendum 2 the
circuit keeps its last epoch (`--keep-last-epoch`), as ESA-ADB's own runs do, and average precision leads.

## Measured performance

The trainer's own figures, recorded in `weights.json`, on the ESA-ADB test windows (115,064 rows), at the circuit's own decision rule. The pre-registered figures, below, are scored differently (average precision, and a threshold chosen on validation).

| quantity | value |
|---|---|
| `minority_class_f1` (this circuit, the headline), at its own threshold (a positive logit) | 0.715854 |
| accuracy (companion, not the headline) | 0.994377 |
| logistic regression minority-class F1, identical split | 0.716418 |
| logistic regression accuracy, identical split | 0.994386 |
| classical surrogate minority-class F1 (6,561 products) | 0.361964 |
| classical surrogate accuracy | 0.968044 |
| all-positive F1 floor | 0.018546 |
| majority-class accuracy floor | 0.99064 |
| test prevalence | 0.00936 |
| encoding scale c | 0.5 |
| checkpoint epoch | 119 |
| n_train / n_val / n_test | 3560 / 4139 / 115064 |

## Pre-registered figures (test windows, the record `docs/measurements/satellite-qcnn-2026-11-24.json`)

| Model, same split and features | Average precision | Minority F1 at the validation threshold | Balanced accuracy |
|---|---|---|---|
| This QCNN (seed 0) | 0.854 | 0.109 | 0.529 |
| QCNN, five seeds | 0.864 +- 0.006 (0.854 to 0.870) | 0.364 +- 0.234 | 0.621 +- 0.090 |
| Logistic regression on the eight z-scored inputs | 0.857 | 0.824 | 0.850 |
| Classical surrogate: logistic regression on the circuit's 6,561 sine-cosine products | 0.892 | 0.472 | 0.655 |

**The classical surrogate leads** on average precision, and the circuit is close to plain logistic regression. The
thresholded columns rest on the two positive validation windows that choose the threshold, which is why they swing
from seed to seed; addendum 2 said so before any model was trained. At the circuit's own decision rule, a positive
logit, seed 0 scores a minority F1 of 0.716 on the test windows, a rule the pre-registration did not choose and which is
reported, not ranked.

**Seed position:** seed 0 is the lowest of five by average precision (0.854; the five range from 0.854 to 0.870), shipped
because the pre-registration fixed seed 0 before any run; no seed was picked on test figures.

## How it runs on the page

As for the OPS-SAT circuit: in the clear; under CKKS one row at a time (the Pauli-form path); under CKKS one row a slot
(`clientQcnnTreeCkks.ts`, N = 16384, 360 bits), whose measurement on all 115,064 test windows is in
`docs/measurements/satellite-qcnn-tree-ckks-2026-11-24.json`; and the same contraction as an ONNX graph, which equals
the state vector on every test window within 1e-5 with no decision changed. The repository commits no `.onnx`:
`scripts/export_qcnn_tree_onnx.py` rebuilds it from `weights.json` byte for byte, and `tree.json` records its SHA-256 and
its check. The page ships the first 8,192 test windows, one CKKS block.

Retrain: `python scripts/data/esa_qcnn_windows.py`, then `python scripts/train/qcnn_variational.py --usecase
satellite_esa --layers 3 --epochs 120 --seed 0 --threads 1 --qubits 8 --arch tree --encoding-scale 0.5
--keep-last-epoch` (on the CPU).
