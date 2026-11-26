# Pre-registration: the finance QCNN on AMLworld (2026-11-25)

Written and committed before any AMLworld input was scored for selection and before any circuit was trained, on the
owner's choices of 2026-11-25: retrain the finance QCNN on AMLworld after a pre-registration, at eight wires with four
wires reported beside it, under the criteria below. Nothing here is changed after a figure is seen; a change is an
addendum with its date and reason.

## Why

Since #94 (merged as f690f1e) every model on the finance page reads IBM's AMLworld HI-Small except the QCNN, which
still encodes four PaySim columns (amount, cross_border, night_tx, new_payee) and reads a PaySim row of its own; the
run says so. The question is whether a tree QCNN on a few of the 114 AMLworld inputs carries enough signal to take the
PaySim QCNN's place, read beside classical comparators of the same inputs. It is not a claim of quantum advantage: a
tree QCNN is a tree tensor network (Grant et al., npj Quantum Information 4, 65, 2018), and QCNNs of this kind are
classically simulable (Bermejo et al., PRX Quantum 7, 020304, 2026).

## Data

As F7b (`references/finance-square-student-page-preregistration.md`): AMLworld HI-Small (CDLA-Sharing-1.0, synthetic),
the 114 inputs of `finance_aml.model_inputs`, on the published MLP's temporal windows (fit 2,843,872 transactions,
validation 710,969, test 1,523,504 with 2,321 laundering), standardised by the fit window's mean and standard deviation
and clipped at 10 in absolute value. Nothing is chosen on the test window.

## Inputs: k of the 114, chosen on the fit window only

- Each input alone is scored by $`|\mathrm{AUC} - 0.5|`$ against the label on the fit window, and the k highest are
  kept, ties broken by name (the satellite pre-registration's rule of 2026-11-24). The inputs and their scores are
  recorded. No validation or test row is read to choose them.
- **k = 8** (eight wires, the medical and satellite width) is the primary arm; **k = 4** (four wires, the finance
  page's present width) is reported beside it. The four are the first four of the same ranking.

## Fit set, validation, test

- Fit: every laundering transaction of the fit window and a seeded sample (seed 0) of four non-laundering ones for each.
- Validation and test: every transaction of the window, at its own prevalence.

## Encoding, circuit, training

- One input a wire, encoded RY($`c \pi \tanh z`$) on the standardised value with $`c = 0.5`$.
- The trainer's corpus `finance_amlworld`, `--arch tree --layers 3 --epochs 120 --encoding-scale 0.5`, seeds 0 to 4, at
  each k. The trainer's own selection rule is kept: the epoch with the best validation F1 when the minority is under a
  quarter.
- Device: the CPU, threads recorded; a GPU run only on the owner's approval of that job, and recorded as such.

## Metrics and comparators

- On the test window: average precision (AUC-PR, the finance models' metric), the minority-class F1 at the threshold
  that maximises F1 on validation, and balanced accuracy at that threshold; the five-seed spread with every figure.
- Comparators on the identical split and inputs: logistic regression on the k standardised inputs, and the classical
  surrogate on the circuit's own feature map ($`3^k`$ Pauli products: 81 at k = 4, 6,561 at k = 8; Schreiber, Eisert
  and Meyer, PRL 131, 100803, 2023). Reference rows, not like for like (114 inputs against k): F7b's square network
  (mean test AUPRC 0.500 over seeds 5 to 9) and the published linear student (0.184).
- If the QCNN loses to a comparator, the figure that ships says so.

## Criteria for the page change

1. Signal: at k = 8, every seed's test AUPRC is at least twice the test window's prevalence,
   $`2 \times 2{,}321 / 1{,}523{,}504 \approx 0.0030`$.
2. Export: the page's clear contraction of the shipped seed (`clearReadout` in `web/src/lib/clientQcnnTreeCkks.ts`)
   equals the state-vector readout within $`10^{-5}`$ on every test row, and no decision differs.
3. Encrypted, only if the tree-under-CKKS path is offered on the finance page as on the satellite page: on one block of
   the shipped test rows the decrypted readout is within $`10^{-3}`$ of the clear one and no decision changes, with the
   time and size per block recorded in a browser Web Worker.

If 1 or 2 fails, the PaySim QCNN stays, and the result is recorded and reported. The k = 4 arm is reported under the
same three checks and does not decide the page change.

## Not decided here

Which seed ships, whether the page shows the AMLworld QCNN by default, and any Hub upload are the owner's choices after
the figures.
