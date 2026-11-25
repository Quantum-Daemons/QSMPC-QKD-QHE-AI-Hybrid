# Pre-registration: a fully encrypted finance student with a square activation (F7)

Written on 2026-11-25, before any network of this design was trained on this corpus. The training script
(`scripts/train/finance_square_student.py`) refuses to run unless this file's SHA-256 is the one it records. The
design's three open choices were the owner's that day: one hidden layer 128 wide with a ReLU control, the CKKS check on
every test row, and training on the GPU once the 3D audit had finished.

## The question

The finance page runs two private paths. The published linear student (`finance_aml.student`, 115 parameters) is
evaluated by CKKS alone, and its minority-class F1 on the test window is 0.309. The published MLP
(`finance_aml.mlp_large`, 114 -> 128 -> 64 -> 2 with ReLU) reaches test AUPRC 0.643 and minority-class F1 0.628, but its
ReLUs are evaluated in garbled circuits between the parties, so it needs the two-party half. The question here: does a
network whose only non-linearity is a square, so that CKKS evaluates it end to end with no layer decrypted and no
garbled circuit, keep the accuracy of the same network with a ReLU?

A fixed degree-2 feature map was tried on this use case before and rejected (`scripts/train/finance_aml.py`,
`--student-degree`: on the 32 features of that time it raised agreement with the teacher and lowered minority-class F1,
0.0185 to 0.0115). That expanded the inputs into every product and fitted a linear model on them. This is a different
model: a learned rank-128 quadratic form of the 114 inputs, trained end to end on the labels. The earlier result does not
decide this one, and is cited so that it is not repeated as a new finding.

## The data

IBM AMLworld HI-Small (CDLA-Sharing-1.0, synthetic; `DATA_LICENSES.md`), the 114 inputs of
`finance_aml.model_inputs` (32 per-account aggregates, 79 per-transaction window features, 3 pair features), on the
published MLP's temporal windows: fit 2,843,872 transactions, validation 710,969, test 1,523,504 (2,321 laundering,
a base rate of 0.0015). The inputs are standardised by the fit window's mean and standard deviation, the published MLP's
choice on validation, and clipped to +-10, as there. Nothing is chosen on the test window.

## The arms

- **Square:** a dense layer 114 -> 128, batch normalisation, the square $`x^2`$, a dense layer 128 -> 2. For inference the
  batch normalisation is folded into the first dense layer (exact in evaluation mode), so the network is a polynomial
  of degree 2 in the inputs and multiplicative depth 3 under CKKS: a plaintext-weight product, a ciphertext square, a
  plaintext-weight product.
- **Control:** the same network with a ReLU in place of the square.

Both are trained by the published MLP's recipe (`scripts/train/finance_aml_mlp.py`, `fit_one`): class-weighted
cross-entropy on the labels, AdamW at learning rate $`3 \times 10^{-4}`$ with weight decay $`10^{-4}`$, batches of 16,384,
at most 60 epochs, the epoch kept by validation AUPRC with patience 8, seeds 0 to 4. Each seed's decision threshold is
the one that maximises minority-class F1 on validation (`finance_aml._best_threshold`), folded into the last bias, so
the decision is the argmax of the two logits, the form an encrypted pass decrypts. The exported seed is the square seed
with the best validation AUPRC.

## The criteria

1. **Accuracy.** The square arm's mean test AUPRC over the five seeds is at least the control's mean minus 0.02. With
   2,321 test positives, the binomial standard error of a proportion near 0.6 is about 0.010, so the margin is about two
   such errors; it was fixed before any arm was trained.
2. **No collapse.** Every square seed's test AUPRC is at least 0.15 (a hundred times the base rate), and each seed
   predicts both classes on the test window at its validation threshold.
3. **Fidelity under CKKS.** The exported seed, on every test row (all 1,523,504), under each parameter set below, by the
   page's own module under node-seal with one row a slot: the largest absolute difference between a decrypted logit and
   the plaintext one is at most $`10^{-3}`$, and the decision (the argmax of the two logits) equals the plaintext one on
   every row.
4. **Reported whatever the outcome.**
   - Per seed, both arms: test and validation AUPRC, minority-class F1 at the validation threshold, and the epoch kept.
   - For the exported seed: a 95 % bootstrap interval of its test AUPRC over test rows (2,000 resamples, seed 0).
   - Per parameter set: the largest logit error, the decisions that differ, the time per block of rows, and the
     ciphertexts and bytes in and out per block.
   - For reference, not as criteria: the published MLP's five recorded seeds, the published linear student and the
     published gradient-boosted teacher on the same test window.

If criterion 1 or 2 fails, the candidate is not adopted and the page does not change; the result is recorded as a
negative finding. If criterion 3 fails under one parameter set only, that set is not offered.

## The parameter sets

The medical square students' two sets, each within the HomomorphicEncryption.org standard's 128-bit post-quantum bound
for a ternary secret (202 bits at $`N = 8192`$, 411 at $`N = 16384`$):

| set | $`N`$ | coefficient moduli (bits) | total | scale |
|---|---|---|---|---|
| A2 | 8,192 | 48, 34, 34, 34, 52 | 202 | $`2^{34}`$ |
| B | 16,384 | 60, 40, 40, 40, 60 | 240 | $`2^{40}`$ |

One row a slot: each of the 114 inputs travels as its own ciphertext over a block of rows (4,096 under A2, 8,192 under
B), and the logits come back as two ciphertexts for the block. A value's magnitude is bounded by the last prime and the
scale ($`2^{13}`$ under A2, $`2^{19}`$ under B); a row whose logits exceed that bound decrypts wrongly, and criterion 3
on every test row is what would show it.

## The page

If the candidate is adopted, the finance page runs the exported seed under CKKS in a Web Worker on the 8,192 test rows it
already ships (`web/public/data/finance/student_heldout.json`, CDLA-Sharing-1.0), one row a slot, and shows the record's
figures for the whole test window beside the run. No new data is shipped.
