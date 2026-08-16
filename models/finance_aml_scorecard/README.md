# Integer risk scorecard - finance AML, evaluated under BFV

> **Nothing in this repository reads this file.** `scorecard.json` is written by
> `scripts/train/finance_scorecard.py` and, searched across `src/`, `web/src/`, `api/`, `scripts/`
> and the generated config, it is opened by nothing at run time and by no test. The four references
> to it anywhere are the trainer that writes it, the two card gates under `tests/`, and this card.
> Even `tests/test_scorecard_bfv_exact.py`, which the metrics record cites, fits a **fresh**
> scorecard on synthetic integers rather than loading the one shipped here. Stated rather than
> papered over: this is a tracked artifact with no consumer, kept as the record of a fitted model,
> and a reader should not assume that editing it changes anything downstream.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. There is no `ModelSpec` for it, and
`finance_aml.scorecard` appears in `models/metrics.json` as a deliberate non-registry record - the
reason is written down in `tests/test_model_artifact_inventory.py::NON_REGISTRY_METRICS`, which
requires every such record to declare itself rather than sit there looking like a leftover. There is
nothing to publish in any case: the weights are a coordinate search over a bounded integer grid and
are reproducible in less time than a download.

## What it is, and why this form

Every other model in this project is fitted to a surrogate loss and then given an operating point
afterwards - two approximations to the quantity actually cared about. This one makes the fit itself
the metric: maximise **net benefit** across a RANGE of decision thresholds, over integer
coefficients, with a sparsity cap (Chi and Birbil, arXiv:2604.04241).

Finance is the natural home for it. Integer scorecards are the industry form for credit and AML
risk - they are what a model-risk function and a regulator actually read - and at roughly one
laundering transaction in 981 the threshold choice dominates, which is exactly what net benefit
prices.

**The cryptographic reason it exists.** The output is a short vector of integers, and BFV evaluates
integer arithmetic exactly, so an encrypted score equals the plaintext score bit for bit rather than
to a precision bound as a CKKS student can only promise. That is a stronger claim about the
encrypted tier than the other students can make. It needs no scheme switching; see
`references/he-scheme-selection.md` for why a CKKS/BFV hybrid is not the route taken.

**What has and has not been verified about that.** `tests/test_scorecard_bfv_exact.py` asserts
`got == plain` under BFV and `approx != plain` under CKKS for the same model - so the CONSTRUCTION is
pinned. It does that on a scorecard it fits in the test, not on this artifact, and it is marked slow
and needs `tenseal`. The `exact_under_bfv` flag in the metrics record should be read as a property of
the model class, demonstrated, rather than as a measurement taken on the file in this directory.

## What this file records

| quantity | value |
|---|---|
| `n_bins` | 10 |
| `b`, the intercept | -75 |
| `cut`, the decision cut | -1 |

Around those three scalars the file carries the parts that are lists rather than numbers, and they
are the model: `features`, 32 names in the order everything else is indexed by; `w`, one integer
coefficient per feature, most of them zero; `bin_edges`, one quantile-edge array per feature; and
`thresholds`, the five decision thresholds the fit integrated over. `method` is `coordinate-search`,
and it is recorded as a label rather than a claim of optimality - the exact integer program over all
sample-by-threshold indicators is not tractable at this row count, and reporting a heuristic's result
as an optimum would be a different claim.

**The input is integer BIN INDICES, not floats.** `input` states it: bin each feature by `bin_edges`
with a right-sided search, dtype `int64`, one index per feature in `features` order. Ten quantile
bins per feature is the scorecard convention and it keeps the score range small, which matters
because BFV's plaintext modulus has to hold the score without wrapping. The edges were fitted on the
training window only and are published with the model, so a client reproduces the preprocessing
exactly instead of guessing it.

**There are no metrics in this file.** It records the model. Everything in the next section came
from `models/metrics.json`, and is labelled as such.

## What the run scored, recorded in models/metrics.json and not here

Under the key `finance_aml.scorecard`, on the temporally-split AMLworld HI-Small test window:

| figure | value |
|---|---|
| minority-class F1 (laundering) | 0.037211 |
| AUPRC | 0.036529 |
| net benefit at the reported threshold | 0.00030763 |
| net benefit of treating everybody, same threshold | -0.00856216 |
| reported threshold, the median of the fitted range | 0.01 |
| area under the net-benefit curve, on the fit split | 0.00018612 |
| non-zero coefficients | 12 |
| features available | 32 |
| coefficient range, so coefficients live in -8 to +8 | 8 |
| decision cut | -1 |
| test rows | 1523504 |
| wall clock, seconds | 272.3 |

The operating point was chosen on the VALIDATION split, never on test, and at a threshold from the
declared range. Evaluating net benefit at 0.5 would price one false positive at one true positive,
which at this base rate no model wins and which is not the exchange rate the scorecard was fitted
for.

**This loses, and the card says so plainly.** Minority-class F1 of 0.037211 sits below the CKKS
linear student on the same use case (0.061394) and far below the XGBoost teacher (0.151948), which
in turn is far below the published 0.6323 for GFP+XGBoost on this corpus. What the scorecard buys is
not accuracy: it is a decision-theoretic fit, a model a human can read off twelve rows, and exact
evaluation under BFV. The net-benefit comparison is the honest one for this form - it clears
treat-everybody, which is negative at this threshold - and that is a much narrower claim than
detection quality. Those comparison figures belong to other models' records, which is why they are
in prose here rather than in a table of this artifact's numbers.

## Reproducibility

    export QSH_DATA_ROOT=$PWD/data-full
    python scripts/train/finance_scorecard.py

Defaults: seed 42, coefficients bounded to a range of 8 either side of zero, at most 12 features,
5 rounds of coordinate search. This rewrites `scorecard.json` here and the `finance_aml.scorecard`
record in `models/metrics.json`.

## Data and licence

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
  (licence read 2026-08-03)

5M transactions, 515K accounts, roughly one laundering transaction in 981. NeurIPS 2023 D&B.
SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states that everything is
synthetic; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen
for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real
data - no commercial-OK corpus of real laundering transactions exists.

## Scope

A research proof of concept, not a production alerting system. The figures above are what this
repository measured on the split described, with the code in `scripts/train/`. Where a figure is
carried from another model's record or from the literature, it is labelled as such.
