# Pre-registration: a CKKS student for the 5G-NIDD detector (2026-11-25)

Written and committed before any student below was trained, on the owner's choice of 2026-11-25 to give the telecom
page's real-data detector an encrypted path (`references/telecom-real-data-2026-11.md`, option 1, without the other
testbeds, which are a later step). Nothing here is changed after a figure is seen; a change is an addendum with its
date and reason.

## Why

The telecom page's encrypted path runs only on synthetic QKD telemetry. Its real-data detector, the gradient-boosted
trees on 5G-NIDD (`telecom_ids.nidd`, CC BY 4.0; macro-F1 0.997 on its test split), runs in the clear only. The
repository's own plaintext probe, a multinomial logistic regression on the 36 numeric features, reaches macro-F1 0.969
on the same split (`models/metrics.json`, `telecom_ids.nidd.linear_baseline`), so a linear student, which CKKS evaluates
as one dot product per class, is plausible. A 2026 pre-registered study found a distilled student no better calibrated
than a temperature-scaled student trained directly (arXiv:2609.31141), so both are trained and compared.

## Data

As the teacher: 5G-NIDD's `BTS1_BTS2_fields_preserved.csv` after the repository's removal of the Benign copies of the
flows published twice (934,361 rows, 9 classes), the teacher's 75/25 split stratified at seed 42 (test 233,591 rows).
Inputs: the 36 numeric features, standardised by the training rows' mean and deviation; the ten ordinal codes are left
out, because their integer order is nominal and a linear model would read it as a magnitude (the probe's reasoning).
A stratified 10 percent of the training rows, seed 0, is the validation set; nothing is chosen on the test split.

## The arms

- **Distilled:** a multinomial logistic regression trained on the teacher's class probabilities for the training rows
  (cross-entropy to the soft targets), L2 penalty chosen on validation from $`10^{-6}`$ to $`10^{-1}`$ by decades.
- **Direct:** the same model trained on the labels, the same penalty grid, then one temperature fitted on validation by
  log loss.
- The teacher is reported beside them as the reference.

## Metrics

On the test split: macro-F1, per-class F1, accuracy, the expected calibration error (15 equal-width bins of the top
probability), and the share of rows on which the student's class is the teacher's.

## Criteria for the page

1. The student the page shows is the arm with the higher validation macro-F1; its test macro-F1 is at least 0.95.
2. Under CKKS in the browser, on the held-out rows the page ships (a stratified sample of 2,048 test rows, CC BY 4.0
   with its credit and the change notes), every decision on a row whose plaintext top-two margin exceeds twice the
   largest CKKS deviation equals the plaintext one, and the largest deviation is below $`10^{-3}`$.

If 1 fails, nothing ships and the result is recorded. If 2 fails, the encrypted path does not ship.

## Not decided here

A Hub upload, and whether the page leads with the encrypted student or the trees, are the owner's choices after the
figures.

## Addendum, 2026-11-25, before any student was trained or any 5G-NIDD row read for it

Two choices the text above left open, fixed while writing `scripts/train/telecom_nidd_student.py` and before it ran.
**The penalty** of each arm is the one with the lowest validation log loss against that arm's own target (the
teacher's probabilities for the distilled arm, the labels for the direct one); the arm the page shows is still chosen
by validation macro-F1, as criterion 1 says. **The input range:** after standardisation every input is clipped to
$`[-20, 20]`$, in training and on the page alike, because a flow's byte and packet counts reach tens of thousands of
deviations and CKKS needs a bounded input. **The teacher** is the published graph at its pinned revision
(`models/hub_file_hashes.json`), checked by its SHA-256 before use.
