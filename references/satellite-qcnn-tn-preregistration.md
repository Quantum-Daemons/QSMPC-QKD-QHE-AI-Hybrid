# Pre-registration: a satellite QCNN on real telemetry, its tree-network surrogate in ONNX, and the tree network under CKKS (2026-11-24)

Written and committed before any satellite QCNN is trained, on the owner's choices of 2026-11-24: a QCNN for the
satellite use case on both OPS-SAT-AD and ESA-ADB's Mission1, rather than synthetic rows; the classical tree
tensor network the QCNN is, exported to ONNX; and a prototype that evaluates that network under CKKS. Nothing below
is changed after a figure is seen; a change is an addendum with its date and reason, as for the medical students.

## Why the tree circuit

The trainer's `tree` circuit applies one shared two-qubit filter to disjoint pairs and pools each pair with a
controlled-RY, so after every pooling stage the survivors are independent single-qubit states: the circuit is a tree
tensor network (Grant et al., npj Quantum Information 4, 65, 2018) and is simulated classically in time linear in the
width. QCNNs of this kind are classically simulable in general (Bermejo et al., PRX Quantum 7, 020304, 2026), so
nothing here is a claim of quantum advantage: the circuit is a classifier whose structure is a tree network, and the
pages say so. Its classical form is what makes an ONNX export and an encrypted evaluation possible.

## A. OPS-SAT-AD

- **Data.** OPS-SAT-AD (Ruszczak et al.; Zenodo 12588359, CC BY 4.0), its 2,123 segments, with the benchmark's own
  partition (the `train` column: 1,594 training, 529 test), as the shipped OPS-SAT models use it.
- **Features: eight of the nineteen**, chosen on the training partition only: each of `OPSSAT_FEATURES` alone is
  scored by $`|\mathrm{AUC} - 0.5|`$ against the label on the training partition, and the eight highest are kept,
  ties broken by name. The eight and their scores are recorded. No test row is read to choose them.
- **Validation.** 15 percent of the training partition, stratified, seed 0; the circuit is fitted on the rest.
- **Encoding.** z-scored with the fitted rows' mean and deviation; RY($`c \pi \tanh z`$) with $`c = 0.5`$, the scale
  the 2026-10-30 pre-registration chose on medical and telecom.
- **Circuit and training.** The medical artifact's recipe: `--arch tree --layers 3 --epochs 120 --qubits 8
  --encoding-scale 0.5`, seeds 0 to 4, on the CPU with one thread (the GPU is in use for the medical series; a CPU
  run is recorded as such). Seed 0's artifact ships; the five seeds give the spread.

## B. ESA-ADB, Mission1, the lightweight channels

- **Data.** ESA Anomalies Dataset, Mission1 (Kotowski et al.; Zenodo 12528696, CC BY 3.0 IGO), channels 41 to 46,
  already on this machine; positives are the annotations whose category is Anomaly, as ESA-ADB's lightweight score
  counts them (rare nominal events and communication gaps are not positives).
- **Grid and split.** The shipped Telemanom student's preparation (`prepare` in
  `scripts/train/satellite_telemanom.py`): ESA-ADB's zero-order hold on a 30 s grid, and a z-score per channel from the
  unlabelled training rows before the validation months; ESA-ADB's own split: fit before 2006-10-01, validation from
  2006-10-01 to 2007-01-01, test from 2007-01-01. A window that crosses either boundary is dropped.
- **Windows and label.** Non-overlapping windows of 64 grid steps (32 minutes); a window is positive when it overlaps
  an Anomaly interval on any of the six channels.
- **Features (eight a window).** The standard deviation of each of the six channels in the window; the largest
  one-step change of any channel; the largest absolute value of any channel. All on the z-scored grid.
- **Imbalance.** The fitted set keeps every positive window and a seeded sample (seed 0) of four negatives for each;
  validation and test keep every window, at the mission's own prevalence.
- **Encoding, circuit, training.** As in A.

## Metrics and comparators (A and B)

- **Metrics on the test rows**: average precision (AUC-PR), the minority-class F1 at the threshold that maximises F1
  on validation, and balanced accuracy at that threshold. The trainer's own selection rule (F1 when the minority is
  under a quarter) is kept.
- **Comparators on the identical split and features**: logistic regression on the eight z-scored inputs, and the
  classical surrogate on the circuit's own feature map (Schreiber, Eisert and Meyer, PRL 131, 100803, 2023), as for the
  shipped QCNNs. For A, the shipped OPS-SAT teacher and student are listed beside them as reference rows: nineteen
  features against eight, so not like for like.
- **Reading.** If the QCNN loses to a comparator, the figure that ships says so. The five-seed spread is shown with
  every QCNN figure.

## C. The tree network in ONNX

- The trained tree circuit is written as its classical contraction: each wire's encoded state, each pair's filter and
  pool as a fixed map on the pair's single-qubit states, then the readout head, as an ONNX graph (opset 17) that ONNX
  Runtime Web runs on the page.
- **Criterion.** On every test row of A and B, the ONNX readout equals the state-vector simulation's within
  $`10^{-5}`$, and no decision differs. Otherwise the export does not ship.

## D. The tree network under CKKS (prototype)

- **What is encrypted.** The client encodes each feature, forms each wire's single-qubit state in the clear, encrypts
  the states (one row a slot, so a ciphertext holds a block of rows), and receives the encrypted readout of the
  network; it decrypts and applies the readout head and threshold. The server evaluates every pooling stage as a
  bilinear map on the two input states with the trained coefficients in plaintext: one ciphertext product per pair of
  components and plaintext scalars, three stages for eight wires.
- **Parameters.** node-seal (Microsoft SEAL in WebAssembly, the library the medical students use) in the browser; a set inside the HomomorphicEncryption.org standard's
  128-bit post-quantum bound for its ring dimension (Table 2), chosen before the first run by the depth the three stages
  need, with the modulus and scale recorded, and listed in the CBOM, which `tests/test_r86_cbom_he_parameters.py`
  checks.
- **Criteria.** On every test row of A and B: the decrypted readout within $`10^{-3}`$ of the plaintext network's, and
  no decision changed. Times and sizes are recorded with the machine. A failure is reported as one, with the set that
  failed.
- **What it is not.** The model's weights are public, the client holds the key, and the server learns nothing it could
  decrypt; it is the same threat model as the medical page's CKKS students, not a protection of the model.

## Not decided here

Which of the A and B artifacts the satellite page shows by default, and whether the encrypted tree network ships on
the page or stays a recorded prototype, are the owner's choices after the figures.

## Addendum, 2026-11-24, before any window or model was computed

Section B first described the grid as "resampled to 30 s by mean" with a z-score "of the rows before 2007-01-01". The
shipped Telemanom student's preparation, which B says it uses, is ESA-ADB's zero-order hold and a z-score from the
unlabelled training rows before 2006-10-01; B now says so, and adds that windows crossing a boundary are dropped. Found
while writing `scripts/data/esa_qcnn_windows.py`, before it ran.
