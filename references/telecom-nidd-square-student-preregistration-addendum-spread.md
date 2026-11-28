# Addendum to the telecom square-network pre-registration: criterion 3 from run to run, and the page's set

Written and committed on 2026-11-28, after the pre-registered run and the owner's choice of that day to put the network
in place of the linear student on the telecom page's encrypted 5G-NIDD path, and before the page shipped with it. The
pre-registration (`references/telecom-nidd-square-student-preregistration.md`) is unchanged, and so is its record
(`docs/measurements/telecom-nidd-square-student-2026-11-28.json`).

## What was found

Criterion 3 holds a set to the largest logit error at most $`10^{-3}`$ on the 2,048 test flows with no decision changed.
The run measured it once per set, one encryption each: $`4.1 \times 10^{-4}`$ under A2 and $`2.2 \times 10^{-5}`$ under
B. The page encrypts afresh on every run, with new keys and new noise, and holds each run to the same rule. A full test
suite on 2026-11-28 found one run of the page's module under A2 at $`1.4 \times 10^{-3}`$. Repeated runs of that module on
the flows the page ships then gave, under A2, $`3.2 \times 10^{-4}`$ to $`1.4 \times 10^{-3}`$ over 27 runs (median
$`5.1 \times 10^{-4}`$, two above $`10^{-3}`$), and under B $`1.3 \times 10^{-5}`$ to $`2.3 \times 10^{-5}`$ over 15
runs. No decision changed in any run, and every run scored the accuracy the record states. So under A2 the page's rule
fails in a share of runs that one run could not show; under B it held with room to spare in every run.

## The owner's choice

On 2026-11-28 the owner chose B as the page's set. A2 stays in the page file as a set the criteria offered. The finance
page's square network keeps A2: its largest logit errors under A2 are about $`2 \times 10^{-6}`$ in its record.

## What is measured

`scripts/measure_telecom_square_ckks_spread.py`, committed after this addendum with its SHA-256 frozen in it, runs the
page's own module (`runSquareTabular` in `web/src/lib/clientFHESquareTabular.ts`, in node with the real node-seal) 30
times under each set on the 2,048 flows of `web/public/data/telecom_nidd/square_network.json`, scaled as the page scales
them, each run with fresh keys. Per run: the largest logit error, the decisions that differ, and the accuracy. Per set:
the minimum, median and maximum of the largest error, the runs above $`10^{-3}`$, and the runs with a decision changed.
The record is `docs/measurements/telecom-nidd-square-ckks-spread-2026-11-28.json`.

## Stated before it runs

- Expected: B holds the rule in every run; A2 exceeds it in some.
- Nothing in the pre-registration's verdict changes: the record keeps the run's figures, measured as it states them.
- If B exceeds $`10^{-3}`$ in any run, or changes any decision, the record says so and the page does not ship with B
  before the owner is asked.
