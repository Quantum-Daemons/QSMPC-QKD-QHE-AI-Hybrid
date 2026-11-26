# Pre-registration: federated learning across 5G-NIDD's two base stations (2026-11-26)

Written and committed before any model below was trained, on the owner's choice of 2026-11-26 to federate the telecom
page's real-data detector across the two base stations its corpus was captured at. Nothing here is changed after a
figure is seen; a change is an addendum with its date and reason. Before writing it, the label counts of each base
station's rows in each split were read (no feature was summarised and no model fitted); they are in "Data".

## Why

5G-NIDD (Siriwardhana et al., IEEE Data Descriptions, doi:10.1109/IEEEDATA.2025.3592888; a 5G test network at the
University of Oulu; CC BY 4.0) was captured at two base stations, and
`BTS1_BTS2_fields_preserved.csv` concatenates their two files. The telecom page's detector and its CKKS student
(`references/telecom-nidd-ckks-student-preregistration.md`) are trained on one pool of flows from both. The repository
measured that the gradient-boosted teacher trained on the second station's file scores macro-F1 0.902 on the first's,
with SYNFlood at F1 0.223, against 0.995 the other way (`models/metrics.json`, `telecom_ids.nidd.cross_base_station`).
Two operators who cannot pool their flows could federate instead, and the page already has the secure aggregation for
it: pairwise masks from seeds each pair agrees by X25519 (`web/src/lib/clientQFL.ts`, its twin
`src/qsh/ai/federated.py`). The questions: does a model the two stations train together beat each station's own, how
close does it come to pooling the flows, and does it hold under CKKS on the page.

## Data

As the student: `D.load_5g_nidd()`, 934,361 rows and 9 classes after the removal of the Benign copies of the flows
published twice. The first 446,787 rows are the first base station's file (`D.NIDD_SECOND_FILE_STARTS_AT_CLEANED`), the
remaining 487,574 the second's. The split is the student's, exactly: 75/25 stratified by class at seed 42, then a
stratified 10 percent of the training part at seed 0 as validation. Each station's fit, validation and test rows are the
rows of those three sets that come from its file:

| Station | Fit | Validation | Test | Fewest test rows of a class |
|---|---:|---:|---:|---|
| 1 (first file) | 301,496 | 33,522 | 111,769 | ICMPFlood, 154 |
| 2 (second file) | 329,197 | 36,555 | 121,822 | ICMPFlood, 135 |

Every class is present in every split at both stations. So the pooled arm below is the student itself, and the union
of the two stations' test rows is the student's test split: each of the 2,048 held-out rows the page ships
(`web/public/data/telecom_nidd/student_heldout.json`) comes from one station, and the record names which.

Inputs are the student's 36 numeric features, standardised and clipped to $`[-20, 20]`$. A station alone standardises
by its own training rows (fit and validation), which is all it has alone. For the federated model the two stations
exchange their training-row count and, for each feature, their mean and population deviation, and each combines them
into the pooled mean and deviation by the parallel formula; these 73 numbers per station are the only per-station
figures exchanged in the clear. They are not masked because the masks' ring holds magnitudes up to 1,024 at
$`2^{-20}`$ resolution and the raw features' second moments are far larger. Each station computes its figures as the
student computes its own, over the loader's single-precision rows, so the combination must equal the student's
standardisation to a relative $`10^{-5}`$, which the run checks; the record gives the largest difference this makes to
a shipped held-out row's standardised value, since the page scores both models on the rows the student's statistics
standardised.

## The arms

The model is the student's: a multinomial logistic regression on the 36 inputs, 333 parameters, fitted from zeros to
the mean cross-entropy of the fit rows plus $`\lambda \lVert W \rVert^2`$, with $`\lambda`$ from $`10^{-6}`$ to
$`10^{-1}`$ by decades, then one temperature fitted on validation by log loss, searched over $`\log T \in [-4, 4]`$.
The grid is the student's, kept for comparability although the student chose its smallest value.

- **Each station alone (two models).** The student's procedure on one station's rows: L-BFGS-B to at most 1,000
  iterations for each $`\lambda`$, the one with the lowest validation log loss on that station's validation rows (ties
  to the smaller), the temperature on the same rows.
- **Pooled.** The same procedure on both stations' rows: the student. Its test macro-F1 must reproduce the student's
  record, 0.974439, to $`10^{-6}`$, or the run stops and nothing ships.
- **Federated.** FedAvg (McMahan et al., AISTATS 2017) under the page's secure aggregation, for each $`\lambda`$ of the
  grid. From zeros, 40 rounds; in a round each station runs 20 iterations of L-BFGS-B on its own fit rows' objective,
  starting from the current global model, and sends its parameters weighted by its share of the fit rows,
  $`w_k = n_k / (n_1 + n_2)`$, quantised to $`2^{-20}`$ and masked (`SecureAggregator`, the page's construction byte for
  byte). The aggregator sums the masked shares modulo $`2^{32}`$; the sum is the next global model. The 40 rounds of 20
  iterations are fixed here, not tuned: 800 local iterations per station, within the 1,000 one pooled fit may take.
  The penalty is chosen by the pooled validation log loss of each run's final model (ties to the smaller), which the
  aggregator learns as the masked sum of each station's mean validation loss weighted by its share of the validation
  rows; the temperature is fitted by the same bounded search, each evaluation one such masked sum. Every masked message (each round of each run,
  each loss report) takes its own index from one counter under one key agreement, so no mask is used twice. A share
  outside the ring's range stops the run.
- **The teacher** is a reference row from its own record (`models/metrics.json`), not run here.

## Metrics

For every model, on each station's test rows: macro-F1 over the nine classes, per-class F1, accuracy, and the expected
calibration error (15 equal-width bins of the top probability, after the model's temperature). The headline is the
mean of the two stations' test macro-F1, each station weighted equally, as the medical page averages FLamby's
hospitals ("FLamby's metric" below). Also: macro-F1 on the union of the two test sets (the student's test split); the
2-by-2 table of each station's own model on each station's test rows; and, for the federated arm, the pooled training
objective at its final model minus that of the pooled fit at the same $`\lambda`$. As evaluation outside the protocol,
the record holds the federated global model's macro-F1 on each station's validation and test rows after every round of
the chosen run; no choice reads them.

## Criteria for the page

1. The federated model's FLamby metric is above that of each station's own model. The page states the comparison
   with its figures either way.
2. The page says the federated model comes within 0.01 of pooling only if its FLamby metric is at most 0.01 below the
   pooled one; otherwise it states the gap.
3. Under CKKS in the browser, on the 2,048 held-out rows the page ships, every decision on a row whose plaintext top-two
   margin exceeds twice the largest CKKS deviation equals the plaintext one, and the largest deviation is below
   $`10^{-3}`$, as the student's criterion 2. If it fails, the federated model's encrypted path does not ship.

The comparison ships whatever criteria 1 and 2 find: it is a measurement. The page also replays the last round's
aggregation in the tab: fresh X25519 keys for two notional stations, their last local models masked and summed as the
aggregator would, and the sum checked against the shipped global model bit for bit in the fixed point. If it does not
reproduce it, the card says so.

## What the page says it is not

With two stations, each can compute the other's update from the global model it receives and its own, so the masks
hide the updates from the aggregator only. The standardisation statistics are exchanged in the clear. The mask seeds
come from X25519 alone, which is not post-quantum, and the masked updates are not sealed. One process plays both
stations and the aggregator, offline, once, on the CPU. No differential privacy is applied. The split is the student's
random stratified one, so a station's test rows share attack sessions with its training rows; a station's model on the
other station's rows is the cross-station reading.

## Not decided here

A Hub upload of the federated model, and whether the page leads with it, are the owner's choices after the figures.
