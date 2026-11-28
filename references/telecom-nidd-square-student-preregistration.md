# Pre-registration: a square network for the telecom page's encrypted 5G-NIDD path

Written on 2026-11-27 for the owner's reading. On the owner's choice of 2026-11-28 (freeze it, then run on the GPU), it
is committed, and then frozen at its SHA-256 in the training script, before any network below is trained. The owner
chose on 2026-11-27 to pre-register this question and approved the GPU job on 2026-11-28.

One correction was made between the draft and this commit: the draft named the validation FLamby metric as the rule
that keeps the square arm's epoch while calling the re-run's recipe unchanged, and that recipe (the re-run's
pre-registration, arm 4, and `mlp_arm` in `scripts/train/telecom_nidd_clean_protocols.py`) keeps the epoch by
validation macro-F1; this file names that rule. The exported seed is still chosen by the validation FLamby metric, as
drafted.

## What is already known, said first

The pre-registered re-run of batch 9 (`references/telecom-nidd-clean-protocols-preregistration.md`, arm 4, recorded in
`docs/measurements/telecom-nidd-clean-protocols-2026-11-27.json`) trained this recipe on seeds 0 to 4, on the 30 inputs
without LEAF's six columns: 0.9902 (sd 0.0003) on the time split in each capture file, 0.9883 trained on station 1 and
tested on station 2, and 0.9642 the other way. The linear CKKS student on the same inputs scored 0.9632, 0.9734 and
0.9455. The comparison is therefore not an open question and the margins in criterion 1 were chosen after these
numbers were known. What was not measured is whether the network is fit for the page: its fidelity and its time and
memory under CKKS in a Web Worker, and whether new seeds reproduce its accuracy.

## The question

Should the telecom page's encrypted 5G-NIDD path offer the square network beside, or in place of, the linear student?

## The data

As the re-run: 5G-NIDD (CC BY 4.0; `DATA_LICENSES.md`), the shipped rows with the contradictory Benign copies removed,
the 30 numeric inputs without `sHops`, `dHops`, `sVid`, `dVid`, `SrcTCPBase` and `DstTCPBase` (the re-run's F30), under
its protocol P2: in each (capture file, class) group in file order, the first 70 percent to fit, the next 15 to
validation, the last 15 to test. Standardised by the fit part's mean and standard deviation and clipped at 20 in
absolute value, as the linear student. Nothing is chosen on the test part.

## The arms

- **Square:** the re-run's arm-4 network and recipe unchanged (30 inputs, a dense layer to 64 units, the square
  $`x^2`$, a dense layer to nine outputs; Adam at $`10^{-3}`$, batch 512, unweighted cross-entropy, 30 epochs, the
  epoch kept by validation macro-F1), on new seeds 5 to 9, so that no seed the re-run trained is reused. The
  exported seed is the one with the best validation FLamby metric among them.
- **Control:** the linear student the re-run fitted under P2 on F30 (deterministic), unchanged.

## The criteria

1. **Accuracy** (margins chosen after batch 9, as said above). The square arm's mean test FLamby metric over seeds 5
   to 9 is at least the control's plus 0.015, and every seed's is at least the control's plus 0.010.
2. **No collapse.** Every seed's test macro-F1 is at least 0.90 and no class present in the test part has an F1 of 0.
3. **Fidelity under CKKS.** The exported seed, on 2,048 flows of the test part drawn as the shipped student's held-out
   rows were (each class in its share, `numpy.random.default_rng(0)`), by the page's own square-network module under
   node-seal with one flow a slot, under each parameter set below: the largest absolute difference between a decrypted
   logit and the plaintext one is at most $`10^{-3}`$, and the argmax equals the plaintext one on every flow whose two
   largest plaintext logits differ by more than twice that difference.
4. **Fit for the page.** Under each parameter set, the module in a Web Worker in headless Chrome evaluates the 2,048
   flows in at most 30 s from the start of key generation to the last decryption, in each of three runs; the worker's
   WebAssembly memory is at most 1.5 GB after each run; and every decision is as in criterion 3. The runs are on this
   machine with no other job running; the record states the browser's version and the load average before each run.
5. **Reported whatever the outcome.** Per seed: test and validation FLamby metric, macro-F1, each class's F1, the epoch
   kept. For the exported seed: a 95 percent bootstrap interval of its test FLamby metric over test flows (2,000
   resamples, seed 0). Per parameter set: the largest logit error and the decisions that differ. In the browser: each
   run's time (key generation, encryption, evaluation, decryption) and the worker's memory.

If criterion 1 or 2 fails, or criterion 3 or 4 fails under both parameter sets, the page does not change and the result
is recorded as a negative finding. If criterion 3 or 4 fails under one set only, that set is not offered.

## The parameter sets

The finance square network's two sets, unchanged (three rescales: a dense layer, the square, a dense layer):

| set | $`N`$ | coefficient moduli (bits) | total | scale |
|---|---|---|---|---|
| A2 | 8,192 | 48, 34, 34, 34, 52 | 202 | $`2^{34}`$ |
| B | 16,384 | 60, 40, 40, 40, 60 | 240 | $`2^{40}`$ |

## The page and the Hub

If the network is adopted, the telecom page runs the exported seed under CKKS in a Web Worker on the 2,048 flows it then
ships for this split, one flow a slot, beside the record's figures for the whole test part. Whether it replaces the
linear student or sits beside it is the owner's choice after the result. Publishing the network on the Hub waits for the
owner's approval, as every publication does.

## Not done here

No hyperparameter is retuned; no input is added; LEAF's six columns and the shipped random split are not used; no
other split is trained for export (the re-run's cross-station figures for this recipe stand as recorded).

## Compute

Training on the GPU only on the owner's approval of that job (given on 2026-11-28), otherwise on the CPU; the device is
recorded and changes no rule above. The record is `docs/measurements/telecom-nidd-square-student-<the run's date>.json`.
