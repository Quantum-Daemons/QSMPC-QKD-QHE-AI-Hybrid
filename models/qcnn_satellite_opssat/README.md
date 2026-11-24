# Variational QCNN - satellite, OPS-SAT-AD

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. A browser-side circuit for the satellite page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid), served from the site
(`web/public/data/satellite/satellite_qcnn_opssat.json`). Trained, scored and shipped as
`references/satellite-qcnn-tn-preregistration.md` fixed before any training (2026-11-24, section A).

## What it is

**`circuit.name`: `tree`.** Per stage, 3 rounds of one shared two-qubit filter (RY, RZ - CNOT - RZ, RY) on the
disjoint pairs of the active wires, then a CRY pool per pair, on 8 wires, 8 -> 4 -> 2 -> 1; an RZ, RY block on the
last wire; the readout is <Z> on wire 0 through a learned scale and bias. Each feature is encoded as
RY(0.5 pi tanh(z)), z the raw value standardised by the training rows' moments shipped here (`input_space: "raw"`).
The circuit is a tree tensor network and is simulated classically, exactly (`src/qsh/qhe/qcnn_tree.py`, equal to the
state vector to 1e-15); nothing here ran on quantum hardware, and nothing is a claim of quantum advantage.

## Data and split

OPS-SAT-AD (Ruszczak, Kotowski, Evans and Nalepa, Scientific Data 12, 710, 2025; Zenodo 12588359, CC BY 4.0):
2,123 telemetry segments of the ESA OPS-SAT mission, the benchmark's own partition. Eight of its nineteen features,
chosen on the training partition alone by how far each one's AUC is from 0.5: n_peaks, kurtosis, smooth10_n_peaks,
len, diff2_peaks, var_div_len, var_div_duration, sampling. Validation is a stratified 15 percent of the training
partition (seed 0): 1,354 rows fit, 240 validate, and the 529 test rows (113 anomalous) are the benchmark's.

## Figures (test rows, `docs/measurements/satellite-qcnn-2026-11-24.json`)

| Model, same split and features | Average precision | Minority F1 at the validation threshold | Balanced accuracy |
|---|---|---|---|
| This QCNN (seed 0) | 0.933 | 0.893 | 0.918 |
| QCNN, five seeds | 0.931 +- 0.003 (0.929 to 0.935) | 0.882 +- 0.015 | 0.911 +- 0.014 |
| Logistic regression on the eight z-scored inputs | 0.881 | 0.764 | 0.882 |
| Classical surrogate: logistic regression on the circuit's 6,561 sine-cosine products | 0.943 | 0.911 | 0.926 |

**The classical surrogate leads.** A classical model with the circuit's own hypothesis class (Schreiber, Eisert and
Meyer, PRL 131, 100803, 2023) beats the trained circuit on every metric; the circuit beats plain logistic regression
on the same eight inputs. The shipped OPS-SAT models use all nineteen features and score an average precision of
0.968 (teacher and student): not like for like.

**Seed position:** seed 0 is the second of five by average precision (0.933; the five range from 0.929 to 0.935), shipped
because the pre-registration fixed seed 0 before any run; no seed was picked on test figures.

## How it runs on the page

- In the clear: the state vector, or its tree contraction.
- Under CKKS, one row at a time: the existing Pauli-form path (`clientQcnnCkks.ts`), one inner product over the 6,561
  products.
- Under CKKS, one row a slot: the tree network stage by stage (`clientQcnnTreeCkks.ts`; N = 16384, moduli 60 + 6 x 40 +
  60 = 360 bits, inside the HomomorphicEncryption.org standard's post-quantum bound at that N),
  `docs/measurements/satellite-qcnn-tree-ckks-2026-11-24.json`.
- `tree.onnx`: the same contraction as an ONNX graph, written only because it equals the state vector on every test row
  within 1e-5 with no decision changed (`tree.json`).

Retrain: `python scripts/train/qcnn_variational.py --usecase satellite_opssat --layers 3 --epochs 120 --seed 0 --threads 1
--qubits 8 --arch tree --encoding-scale 0.5` (on the CPU; CPU and CUDA kernels round differently).
