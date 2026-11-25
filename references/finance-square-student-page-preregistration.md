# Pre-registration: the square network on the finance page's fully encrypted path (F7b)

Written on 2026-11-25, after F7's result and before any network was trained on the seeds below. The training script
will refuse to run unless this file's SHA-256 is the one it records. The owner chose that day to pre-register this
question in the next batch, and approved the margins and the budget below.

## What is already known, said first

F7 (`references/finance-square-student-preregistration.md`, recorded in
`docs/measurements/finance-square-student-2026-11-25.json`) compared the square network with the same network with a
ReLU, and did not adopt it: mean test AUPRC 0.500 against the control's 0.549, where its criterion 1 asked for at least
0.529. Under both parameter sets every test row decrypted within $`3.1 \times 10^{-6}`$ of the plaintext logits, and no
decision changed. Its record also gives, as a reference, the published linear student, the model the finance page runs
fully encrypted today: test AUPRC 0.184, against the square seeds' 0.475 to 0.521. That comparison is therefore not an
open question, and this text does not treat it as one: the accuracy margins in criterion 1 were chosen after F7's
numbers were known, and are stated as such.

## The question

Should the finance page's fully encrypted path run the square network in place of the linear student? F7 gives its
accuracy against the linear student. What F7 did not measure is whether the network is fit for the page: its time and
memory under CKKS in a Web Worker in a browser, and whether new seeds reproduce F7's accuracy.

## The data

As F7: IBM AMLworld HI-Small (CDLA-Sharing-1.0, synthetic; `DATA_LICENSES.md`), the 114 inputs of
`finance_aml.model_inputs`, on the published MLP's temporal windows (fit 2,843,872 transactions, validation 710,969,
test 1,523,504 with 2,321 laundering), standardised by the fit window's mean and standard deviation and clipped at 10 in
absolute value. Nothing is chosen on the test window.

## The arms

- **Square:** F7's network and recipe unchanged (a dense layer 114 -> 128, batch normalisation folded for inference,
  the square $`x^2`$, a dense layer 128 -> 2; the published MLP's recipe; each seed's threshold folded into the last
  bias), on new seeds 5 to 9, so that no seed F7 trained is reused. The exported seed is the one with the best
  validation AUPRC among them.
- **Control:** the published linear student (`finance_aml.student`), unchanged, on the same test window.

## The criteria

1. **Accuracy** (margins chosen after F7, as said above). The square arm's mean test AUPRC over seeds 5 to 9 is at least
   the linear student's plus 0.15, and every seed's is at least the linear student's plus 0.10.
2. **No collapse.** As F7: every seed's test AUPRC is at least 0.15, and each seed predicts both classes on the test
   window at its validation threshold.
3. **Fidelity under CKKS.** As F7: the exported seed, on every test row, under each parameter set below, by the page's
   own module under node-seal with one row a slot: the largest absolute difference between a decrypted logit and the
   plaintext one is at most $`10^{-3}`$, and the decision equals the plaintext one on every row.
4. **Fit for the page** (new). Under each parameter set, the page's module in a Web Worker in headless Chrome evaluates
   one block of the rows the page ships (`web/public/data/finance/student_heldout.json`: the first 4,096 under A2, all
   8,192 under B) in at most 30 s from the start of key generation to the last decryption, in each of three runs; the
   worker's WebAssembly memory is at most 1.5 GB after each run; and every decrypted decision on the block equals the
   plaintext one. The runs are on the machine F7's record names, with no other job running; the record states the
   browser's version and the load average before each run.
5. **Reported whatever the outcome.**
   - Per seed: test and validation AUPRC, minority-class F1 at the validation threshold, and the epoch kept.
   - For the exported seed: a 95 % bootstrap interval of its test AUPRC over test rows (2,000 resamples, seed 0).
   - Per parameter set: the largest logit error and the decisions that differ on every test row; in the browser, each
     run's time (key generation, encryption, evaluation, decryption) and the worker's memory.

If criterion 1 or 2 fails, or criterion 3 or 4 fails under both parameter sets, the page does not change and the
result is recorded as a negative finding. If criterion 3 or 4 fails under one set only, that set is not offered.

## The parameter sets

F7's two sets, unchanged:

| set | $`N`$ | coefficient moduli (bits) | total | scale |
|---|---|---|---|---|
| A2 | 8,192 | 48, 34, 34, 34, 52 | 202 | $`2^{34}`$ |
| B | 16,384 | 60, 40, 40, 40, 60 | 240 | $`2^{40}`$ |

## The page

If the network is adopted, the page runs the exported seed under CKKS in a Web Worker on the rows it already ships, one
row a slot, and shows the record's figures for the whole test window beside the run; no new data is shipped. Whether it
then replaces the linear student on that path or shows the two side by side is the owner's choice after the result;
this text does not fix it.
