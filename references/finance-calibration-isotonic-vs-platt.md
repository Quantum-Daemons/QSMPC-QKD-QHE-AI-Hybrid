# Isotonic against Platt on the finance AML split, and whether cases or events bind

> **DEADLINE EXECUTED Round 49. THE OPEN QUESTION IS ANSWERED AND THE ANSWER IS NEITHER "KEEP" NOR
> "SWAP".** `src/qsh/models/calibration.py` conditioned this comparison on **Round 48**; Round 48
> shipped nothing, so it is discharged here one round late.
>
> The measurement found a third thing that neither side of the argument had considered:
> **`fit_platt` is being handed a probability, not a log-odds.** Platt's sigmoid is linear in its
> argument, 90% of this teacher's scores lie below 1e-4, and the fitted map therefore returns the
> single constant **0.00090148** for nine of ten deciles. The shipped arm is not Platt scaling
> losing to isotonic; it is Platt scaling applied to the wrong argument.
>
> Give `fit_platt` `logit(s)` and it matches isotonic on both proper scoring rules (Brier
> **0.00142199** against **0.00142218**, log-loss **0.00818964** against **0.00820025**), keeps the
> ranking to the last recorded digit, and never emits a probability of exactly 0. Isotonic costs
> **9.09% of AUPRC**.
>
> **Recommendation: do not switch to isotonic. Change the argument passed to `fit_platt`.**
>
> **APPLIED AND RE-VERIFIED SINCE. The recommendation above was carried out.** `fit_platt` grew an
> `on_logit=True` option that carries the transform, the clip and the published formula string with
> the two floats; `scripts/train/finance_aml.py` passes it; the teacher was retrained and
> `models/metrics.json` and `web/src/lib/config.defaults.json` regenerated from it. **The retrain
> reproduced this file's predictions exactly** - `a = -0.5853491088584304`, `b = 1.6417462922842536`
> against the `a = -0.58535`, `b = 1.64175` measured below; Brier `0.00142199`, log-loss
> `0.00818964`, CORP MCB `1.178e-05`; `brier_raw`, `log_loss_raw`, `corp_raw` and the DSC/UNC terms
> all bit-identical, and AUROC `0.9525582593503528` / AUPRC `0.12334101789973353` unchanged to every
> recorded digit, which is the ranking claim this file makes and now verifies from a second
> direction. Nothing measured below is edited: every superseded figure stays where it was measured,
> and the `platt_prob` arm is retained in the harness as the control that makes the comparison
> legible.

Measured 2026-08-16, Round 49. Script: `scripts/train/finance_calibration_ablation.py`.
Raw output: `data-full/artifacts/finance_calibration_ablation.json` (untracked, regenerate with the
script). Reproduce with:

    python scripts/train/finance_calibration_ablation.py --refit --repeats 20 --bins 20

`--refit` matters and did not appear here at first: the cached score file is keyed on the SEED
ALONE, so a run after any change to the teacher, the features or the split is served the old scores
with no warning. Discarding the cache is the only guard, and it is manual.

## Provenance: this is the shipped split, not a new one

The booster is retrained from `scripts/train/finance_aml.py`'s own functions at its own seed (42),
`TEST_FRACTION` 0.30 and `VAL_FRACTION` 0.20 over the full 5,078,345-row HI-Small corpus. Three
recorded numbers reproduce to every digit `models/metrics.json` carries:

| quantity | `models/metrics.json` | this run |
|---|---|---|
| AUROC | 0.9525582593503528 | 0.9525582593503528 |
| AUPRC | 0.12334101789973353 | 0.12334101789973353 |
| Brier, uncalibrated | 0.00150259 | 0.0015025935 |

## The record named the wrong partition's event count

| partition | cases | positives | base rate |
|---|---|---|---|
| calibration (validation window) | 710,969 | **801** | 1 in 888 |
| scoring (test window) | 1,523,504 | 2,321 | 1 in 656 |

`calibration.py` and the register argue that "the binding quantity is plausibly the **2,321**
positive events". 2,321 is the TEST split's event count. The calibration set the whole argument is
about carries **801**. `models/metrics.json` already records `positives_calibration: 801` on the
line above the sentence that says 2,321, so the two disagree inside one JSON object. This is the
same class of error Round 46 corrected - a count taken from the wrong denominator - committed
inside the correction.

## 1-3. The arms, fitted on the calibration partition, scored on the held-out partition

`platt_prob` is what ships. `platt_logit` is the identical `fit_platt` applied to `logit(s)`.
ECE is over 20 equal-frequency bins; `ECE_pos` is the same per-bin gap weighted by each bin's
positive count instead of its row count.

| arm | Brier | log loss | ECE(20) | ECE_pos(20) | Brier, y=1 only | log loss, y=1 only | AUROC | AUPRC | distinct outputs |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.00150259 | 0.00965987 | 0.00064837 | 0.003586 | 0.862178 | 5.745480 | 0.95255826 | 0.123341 | 1,388,669 |
| `platt_prob` (shipped) | 0.00145433 | 0.01049835 | 0.00183223 | 0.013111 | 0.943890 | 6.256003 | 0.95255826 | 0.123341 | 1,388,669 |
| `platt_logit` | **0.00142199** | **0.00818964** | 0.00078864 | 0.006953 | 0.912352 | **4.926949** | 0.95255826 | 0.123341 | 1,388,669 |
| isotonic | 0.00142218 | 0.00820025 | **0.00077500** | 0.007348 | **0.910687** | 4.935637 | 0.95224152 | 0.112125 | **138** |

Numbers that do not support a clean story are in that table on purpose:

- **raw has the best count-weighted ECE of the four** (0.00064837). No calibrator improves it. The
  uncalibrated booster is already well ordered and only mildly over-confident in the top decile.
- **raw has the best positive-class Brier** (0.862178), for the uninteresting reason that predicting
  near zero everywhere is cheap on a squared error and every calibrator raises the floor.
- **the shipped Platt is worse than doing nothing on three of the six calibration metrics**: log
  loss (+8.68%), ECE (2.83x raw's), ECE_pos (3.66x raw's).

### ECE is not sensitive to the bin count, which is worth stating because it could have been

count-weighted / event-weighted, equal-frequency:

| bins | raw | `platt_prob` | `platt_logit` | isotonic |
|---|---|---|---|---|
| 10 | 6.484e-04 / 0.0036 | 1.832e-03 / 0.0091 | 7.837e-04 / 0.0053 | 7.746e-04 / 0.0053 |
| 20 | 6.484e-04 / 0.0036 | 1.832e-03 / 0.0131 | 7.886e-04 / 0.0070 | 7.750e-04 / 0.0073 |
| 50 | 7.054e-04 / 0.0030 | 1.835e-03 / 0.0195 | 7.911e-04 / 0.0095 | 7.786e-04 / 0.0106 |
| 100 | 9.828e-04 / 0.0106 | 1.840e-03 / 0.0239 | 7.982e-04 / 0.0109 | 7.822e-04 / 0.0141 |

The ORDERING is identical at every bin count. The event-weighted column climbs with bin count for
every arm, which is what happens when 2,321 events are spread over more bins, and is a property of
the estimator rather than of any arm.

### CORP, which needs no bin count at all

| arm | MCB | DSC | UNC |
|---|---|---|---|
| raw | 9.2383e-05 | 1.1093e-04 | 1.5211e-03 |
| `platt_prob` | 4.4116e-05 | 1.1093e-04 | 1.5211e-03 |
| `platt_logit` | 1.1782e-05 | 1.1093e-04 | 1.5211e-03 |
| isotonic | **9.3363e-06** | 1.0829e-04 | 1.5211e-03 |

MCB is the term calibration exists to move, and every arm moves it the right way. Isotonic moves it
furthest, `platt_logit` is 1.26x isotonic's, and the shipped arm is **4.7x** isotonic's. Note
isotonic's DSC: 1.0829e-04 against 1.1093e-04 for the other three. **Discrimination fell.** A
monotone map is not supposed to be able to do that, which is section 3.

### Where the shipped map actually goes wrong: the reliability table

10 equal-frequency bins on the test split. `pred` is the mean predicted probability in the bin,
`obs` the observed frequency.

| bin | n | positives | raw pred | `platt_prob` pred | `platt_logit` pred | isotonic pred | observed |
|---|---|---|---|---|---|---|---|
| 1 | 152,351 | 1 | 0.00000002 | 0.00090148 | 0.00000524 | 0.00000000 | 0.00000656 |
| 2 | 152,351 | 0 | 0.00000006 | 0.00090148 | 0.00001135 | 0.00000000 | 0.00000000 |
| 3 | 152,351 | 0 | 0.00000013 | 0.00090148 | 0.00001769 | 0.00000000 | 0.00000000 |
| 4 | 152,351 | 5 | 0.00000023 | 0.00090148 | 0.00002532 | 0.00000773 | 0.00003282 |
| 5 | 152,350 | 2 | 0.00000041 | 0.00090148 | 0.00003532 | 0.00000811 | 0.00001313 |
| 6 | 152,350 | 8 | 0.00000074 | 0.00090148 | 0.00004982 | 0.00002463 | 0.00005251 |
| 7 | 152,350 | 12 | 0.00000149 | 0.00090149 | 0.00007481 | 0.00003139 | 0.00007877 |
| 8 | 152,350 | 63 | 0.00000420 | 0.00090151 | 0.00013565 | 0.00014296 | 0.00041352 |
| 9 | 152,350 | 285 | 0.00003383 | 0.00090171 | 0.00043969 | 0.00052864 | 0.00187069 |
| 10 | 152,350 | 1,945 | 0.00871024 | 0.00202781 | 0.00670535 | 0.00674554 | 0.01276666 |

The isotonic column's bins are cut on ITS output, so its rows are not the same 152,351 transactions
as the raw column's - the last bin holds 1,958 positives rather than 1,945. The other three arms are
strictly monotone in the raw score and therefore bin identically.

`platt_prob` returns **0.00090148 for deciles 1 through 9**, seven significant figures apart at most.
Its full output support is **[0.00090148, 0.695715]**: it cannot express a risk below 1 in 1,109 for
any input whatsoever, on a corpus whose base rate is 1 in 656 and whose first decile's observed rate
is 1 in 152,351. That is the mechanism, and it follows directly from the form of the map. Platt fits
$`p = 1/(1 + e^{a s + b})`$, linear in $`s`$. With $`a = -7.8378`$, $`b = 7.0106`$ and $`s < 10^{-4}`$ across
nine deciles, `a*s` is below 1e-3 and the map is a constant to four decimals. Fitted on `logit(s)`
instead, `a = -0.58535`, `b = 1.64175` and the same two floats spread deciles 1 to 9 over
5.2e-06 to 4.4e-04.

**The unit tests could not have caught this.** `tests/test_calibration.py` names its fixture
variable `logit` and builds it as `3.0 * (x - 0.5)`, so every test exercises `fit_platt` on a
log-odds. The only production caller, `scripts/train/finance_aml.py`, passed
`teacher.predict_proba(Xval)[:, 1]`. The suite and the pipeline disagreed about what the argument
is, and both were internally consistent. (No line number is given because the one first recorded
here went stale within the round: that call now reads `fit_platt(..., on_logit=True)`, so a pointer
would land a reader on a line that contradicts the sentence pointing at it. The hole is closed from
the other side too - `test_a_probability_score_is_not_fitted_into_a_near_constant` asserts the
fitted map is not near-constant across the bottom nine deciles, and fails at `1.000399x` against
the probability-fed call.)

## 3. Calibration and ranking: true of Platt, FALSE of isotonic, measured

`calibration.py` states that calibration "cannot fix ranking, so it moves neither AUC nor AUC-PR",
and `finance_aml.py` records `does_not_move_ranking` beside the numbers. On this split:

| arm | AUROC | change | AUPRC | change |
|---|---|---|---|---|
| raw | 0.95255826 | - | 0.12334102 | - |
| `platt_prob` | 0.95255826 | 0 to 10 dp | 0.12334102 | 0 to 10 dp |
| `platt_logit` | 0.95255826 | 0 to 10 dp | 0.12334102 | 0 to 10 dp |
| isotonic | 0.95224152 | **-0.00031674** | 0.11212512 | **-0.01121590 (-9.09%)** |

The claim is exact for both Platt arms and wrong for isotonic. Isotonic regression is WEAKLY
monotone: it is a step function, and here it collapses 1,388,669 distinct scores to **138** distinct
values. Ties are not a ranking; every pair inside a step becomes a coin flip, and average precision
pays for that. AUPRC is this project's stated companion to the headline minority-class F1, so 9.09%
of it is not a rounding difference.

## 3b. Isotonic emits exact zeros and exact ones

| arm | output support | distinct | rows at exactly 0 | true positives among them |
|---|---|---|---|---|
| raw | [4.407e-11, 0.999971] | 1,388,669 | 0 | 0 |
| `platt_prob` | [9.015e-04, 0.695715] | 1,388,669 | 0 | 0 |
| `platt_logit` | [1.680e-07, 0.988628] | 1,388,669 | 0 | 0 |
| isotonic | [0, 1] | 138 | **464,176 (30.5%)** | **1** |

Isotonic assigns probability exactly 0 to 464,176 test transactions and exactly 1 to 10. One real
laundering transaction landed on an exact zero. `log_loss` clips at 1e-15, so the run scores; the
scored value is a property of the clip, not of the model. Measured directly, that single row
contributes 0.28% of isotonic's whole log loss (2.267e-05 of 0.00820025), and re-clipping at
$`1/(2n) = 7.033\times10^{-7}`$ moves the total only to 0.00818709. So the hazard is real and, on this draw,
small. It is a hazard rather than a result because one event is one event.

Two consequences beyond the metric. A probability of exactly 0 is an assertion no amount of AML
evidence supports. And `Platt.as_dict()` ships **two floats and a formula string** so the browser
can apply the identical transform from the weights JSON; an isotonic map is 138 breakpoints, so
switching also changes what has to be published and re-verified client-side.

## 4. The point of contention, measured: cases or events?

Base rate is events divided by cases, so only two of the three quantities are free. **No sweep can
hold two of them and vary the third.** Three sweeps therefore pin one quantity each, 20 subsample
draws per cell (1 for the full-set cell, which has no randomness), each draw fitted on the subsample
and scored on the same untouched 1,523,504-row test split.

Reported as **Platt minus isotonic** on the Brier score. Positive means isotonic is ahead.
`sd` is the standard deviation across the 20 draws.

### Sweep A: base rate pinned at 1.13e-03, cases and events scale together

| events | cases | isotonic Brier | `platt_prob` - iso | sd | `platt_logit` - iso | sd | isotonic AUPRC |
|---|---|---|---|---|---|---|---|
| 3 | 2,663 | 0.00149143 | +8.173e-06 | 4.2e-05 | -4.011e-05 | 6.5e-05 | 0.053950 |
| 6 | 5,326 | 0.00145948 | +1.686e-05 | 3.4e-05 | -2.013e-05 | 3.4e-05 | 0.075536 |
| 12 | 10,651 | 0.00145237 | +1.670e-05 | 2.1e-05 | -2.198e-05 | 2.4e-05 | 0.086150 |
| 25 | 22,190 | 0.00143718 | +2.402e-05 | 9.2e-06 | -9.030e-06 | 1.3e-05 | 0.093290 |
| 50 | 44,380 | 0.00143292 | +2.597e-05 | 8.4e-06 | -8.812e-06 | 8.5e-06 | 0.101354 |
| 100 | 88,760 | 0.00142691 | +2.835e-05 | 2.6e-06 | -3.690e-06 | 3.0e-06 | 0.103640 |
| 200 | 177,520 | 0.00142414 | +3.071e-05 | 2.0e-06 | -1.484e-06 | 1.3e-06 | 0.107924 |
| 400 | 355,041 | 0.00142300 | +3.140e-05 | 9.8e-07 | -5.497e-07 | 7.1e-07 | 0.110532 |
| 801 | 710,969 | 0.00142218 | +3.214e-05 | - | -1.919e-07 | - | 0.112125 |

**No crossover anywhere.** Isotonic is ahead of the shipped Platt at every size down to 3 events and
2,663 cases; below about 25 events the lead stops being separable from the draw-to-draw noise
(+1.670e-05 against sd 2.1e-05 at 12 events) but it never reverses. `platt_logit` is nominally ahead
of isotonic at all nine sizes and never by as much as 1 sd; the evidence there is the consistency of
the sign across nine independent grids, not any one cell.

**The published crossover cannot be reached at this base rate.** ICML's regime is under 1,000
calibration cases and UAI's under 2,000. At 1 in 888, a 1,000-case calibration set holds ONE
positive and `fit_platt` refuses a one-class set by design. So at the deployment base rate, cases
and events cannot be separated at the small end at all - that is a limit of the corpus, not a result.

### Sweep B: cases pinned at the full 710,969, events thinned

| events | cases | base rate | isotonic Brier | `platt_prob` - iso | sd | `platt_logit` - iso | sd |
|---|---|---|---|---|---|---|---|
| 3 | 710,171 | 4.22e-06 | 0.00152014 | +3.019e-06 | 3.4e-06 | -4.404e-05 | 3.3e-06 |
| 6 | 710,174 | 8.45e-06 | 0.00151774 | +5.413e-06 | 4.5e-06 | -4.063e-05 | 4.5e-06 |
| 12 | 710,180 | 1.69e-05 | 0.00151379 | +9.356e-06 | 4.8e-06 | -3.457e-05 | 4.3e-06 |
| 25 | 710,193 | 3.52e-05 | 0.00150517 | +1.794e-05 | 5.4e-06 | -2.316e-05 | 5.9e-06 |
| 50 | 710,218 | 7.04e-05 | 0.00149449 | +2.855e-05 | 5.5e-06 | -1.025e-05 | 6.2e-06 |
| 100 | 710,268 | 1.41e-04 | 0.00147867 | +4.422e-05 | 6.3e-06 | -2.976e-06 | 8.8e-06 |
| 200 | 710,368 | 2.82e-04 | 0.00145978 | +6.278e-05 | 3.2e-06 | **+5.069e-06** | 3.6e-06 |
| 400 | 710,568 | 5.63e-04 | 0.00143907 | +1.759e-05 | 2.1e-06 | **+2.890e-06** | 2.9e-06 |
| 801 | 710,969 | 1.13e-03 | 0.00142218 | +3.214e-05 | - | -1.919e-07 | - |

Against the **shipped** Platt: no crossover. Cutting events by 267x, from 801 to 3, with the case
count held at 710,969, never flips the sign.

Against the **correct** Platt: there IS a crossover, and it is on the event axis. `platt_logit` beats
isotonic by 13 sd at 3 events (-4.404e-05, sd 3.3e-06) and by 3.9 sd at 25 (-2.316e-05, sd 5.9e-06);
the sign turns over at 200-400 events (+5.069e-06 at 1.4 sd, +2.890e-06 at 1.0 sd) and returns to a
dead heat at 801 (-1.919e-07). So for a correctly applied Platt, the quantity that governs is the
event count, and the threshold is roughly **100-200 events** - two orders of magnitude below the
CASE thresholds the papers state, and already passed by this calibration set's 801.

### Sweep C: events pinned at 100, cases varied by thinning negatives

| events | cases | base rate | isotonic Brier | `platt_prob` - iso | sd | `platt_logit` - iso | sd |
|---|---|---|---|---|---|---|---|
| 100 | 500 | 2.00e-01 | 0.02279629 | **-5.935e-03** | 3.0e-03 | -2.129e-03 | 2.1e-03 |
| 100 | 1,000 | 1.00e-01 | 0.01072297 | **-4.576e-03** | 1.7e-03 | -7.896e-04 | 8.6e-04 |
| 100 | 3,000 | 3.33e-02 | 0.00373975 | **-1.298e-03** | 2.7e-04 | -1.485e-04 | 1.2e-04 |
| 100 | 10,000 | 1.00e-02 | 0.00183834 | **-1.853e-04** | 6.0e-05 | -8.508e-05 | 5.2e-05 |
| 100 | 30,000 | 3.33e-03 | 0.00146227 | +3.368e-05 | 1.3e-05 | -2.129e-05 | 2.0e-05 |
| 100 | 100,000 | 1.00e-03 | 0.00142750 | +2.806e-05 | 3.6e-06 | -1.338e-06 | 1.5e-06 |
| 100 | 300,000 | 3.33e-04 | 0.00145459 | +6.772e-05 | 3.8e-06 | +5.498e-06 | 4.4e-06 |
| 100 | 710,268 | 1.41e-04 | 0.00147774 | +4.516e-05 | 5.3e-06 | -2.159e-06 | 7.3e-06 |

**The sign flips between 10,000 and 30,000 cases**, at a fixed 100 events, and it flips by margins of
3 sd and up. This is the only cut in which the shipped Platt ever beats isotonic.

Read that with its confound stated: thinning negatives at fixed events also raises the base rate,
from 1.41e-04 to 2.00e-01. At 500 cases the calibration set is 20% positive and the test split is
0.15%, so both arms are badly mis-located and the smooth one degrades more gracefully. The isotonic
Brier column is not monotone in cases for the same reason - it is best at 100,000 cases (base rate
1.00e-03, nearest the test split's 1.52e-03) and worse on either side. That column is measuring base
rate mismatch, not sample size.

### The answer

**Against the estimator that ships, cases bind and events do not.** Holding cases at 710,969 and
cutting events 267-fold never flips the comparison (sweep B); holding events at 100 and cutting cases
70-fold does flip it (sweep C). The retained-Platt argument - that at a 1-in-981 base rate the
binding quantity is the positive-event count - is not supported by the measurement in the form it
was written.

**Against a correctly applied Platt, events bind.** Sweep B's crossover for `platt_logit` sits at
100-200 events with cases pinned at maximum. The intuition behind the retained-Platt argument was
therefore right in kind and wrong in scale: the threshold is around 150 events, not 2,000, and this
calibration set's 801 is past it either way.

**Both readings converge on the same instruction.** ICML 2005's sentence - "when there are 1000 or
more points in the calibration set, isotonic yields performance as good as, or better than, Platt" -
holds here, at 710,969 cases, in the exact form it is written: **as good as**. The measured gap
between isotonic and a correct Platt at full size is 1.9e-07 of Brier, 1.06e-05 of log loss, and the
tie-break is that isotonic costs 9.09% of AUPRC and Platt costs none.

## Supplementary: the only 2,321-event calibration set available

The register's argument names 2,321 events. No calibration set with 2,321 events exists in the
shipped configuration, because that count belongs to the test split. Swapping the two partitions -
fit on test, score on validation - produces one, and both partitions are out-of-sample for the
booster (it is fitted only on the first window), so the swap is sound as a measurement even though
it is not the shipped arrangement. The structural graph features differ slightly between the two
windows (`fit_on=fit` against `fit_on=tr`), which is why this is reported as supplementary.

Calibration set 1,523,504 cases / 2,321 events; scored on 710,969 cases / 801 events:

| arm | Brier | log loss | ECE(20) | ECE_pos(20) | Brier, y=1 | log loss, y=1 | AUROC | AUPRC | distinct |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.00114486 | 0.00635503 | 0.00081842 | 0.008983 | 0.722692 | 4.234174 | 0.95836322 | 0.197493 | 685,000 |
| `platt_prob` | 0.00104814 | 0.00722038 | 0.00151102 | 0.006742 | 0.838169 | 5.001447 | 0.95836322 | 0.197493 | 685,000 |
| `platt_logit` | 0.00100574 | **0.00587767** | 0.00108245 | 0.011028 | 0.783752 | 3.327292 | 0.95836322 | 0.197493 | 685,000 |
| isotonic | **0.00100033** | 0.00604684 | 0.00113179 | 0.010824 | 0.793681 | **3.320963** | 0.95799723 | 0.186739 | **97** |

At 2,321 events the two split the two proper scoring rules - isotonic takes Brier by 5.4e-06,
`platt_logit` takes log loss by 1.7e-05 - and isotonic again costs AUPRC, 0.197493 to 0.186739
(-5.45%). Tripling the events from 801 to 2,321 does not separate them.

## What this does NOT establish

- **One seed, one corpus, one booster.** Every number is from `random_state=42` on HI-Small with the
  32-feature GFP reimplementation. The subsample sweeps have 20 draws each and report their spread;
  the teacher itself was trained once. A second teacher seed is not measured here.
- **The temporal split is not exchangeable, and the subsampling is.** Drawing rows uniformly out of
  the validation window destroys the time ordering within it, so the sweeps measure sample size and
  not the drift a genuinely smaller, earlier calibration window would also carry. The full-set row
  of each sweep is the only one free of that.
- **Nothing here is measured on medical, satellite or telecom.** Those splits carry about 42 and 113
  events and no calibrator is fitted on them at all. The floor for a flexible reliability CURVE
  (Van Calster et al. 2016: 200 events and 200 non-events) still binds there and this file does not
  touch it.
- ~~**The `platt_logit` arm is not shipped.** It is a measurement of a one-line change, not the
  change.~~ **SUPERSEDED - it ships now.** Kept rather than deleted because it was a true and
  load-bearing caveat when this table was written: everything else in this section still limits what
  the measurement establishes, and this line limited what had been ACTED on. The retrain closed it
  and reproduced the numbers above to every recorded digit, so the caveat that remains is the one
  directly beneath: one seed, one corpus, one booster.

## Sites carrying prose this measurement contradicts

Recorded rather than all corrected, because two of them are generated and hand-patching a generated
artifact is a defect with a timer on it.

| site | status |
|---|---|
| `src/qsh/models/calibration.py` docstring and `why_not_isotonic` | corrected in this round (it is the generator) |
| `web/src/components/ModelPanel.tsx` | corrected in this round; it told visitors the calibration set "is inside the range where the published comparison puts Platt ahead", which Round 46 had already refuted and which never reached this file |
| `models/metrics.json` -> `finance_aml.gbm.calibration.why_not_isotonic` | NOT hand-edited. REGENERATED SINCE, by the `scripts/train/finance_aml.py` run that applied the fix; the block now carries `on_logit: true`, `logit_clip_eps`, the log-odds `formula` string and the new `a`/`b` |
| `web/src/lib/config.defaults.json` -> `models/finance_aml[1].achieved.calibration` | NOT hand-edited. Derived from `models/metrics.json` via `scripts/export_config.py`, re-run after the retrain. Checked: no allowlist sits between the two, so `on_logit` and `logit_clip_eps` do reach the browser |
| `web/src/components/ModelPanel.tsx` map row | CORRECTED SINCE, and it was the one live "landed in one of three sites" defect: the row rendered `p = 1 / (1 + exp(a*s + b))` unconditionally, so after the retrain it would have printed a score-domain formula beside logit-fitted floats. It now branches on `on_logit` |
| `tests/test_calibration.py` | CORRECTED SINCE. The `imbalanced` fixture claimed a 1-in-650 base rate and 90% of scores below 1e-4 and delivered 1-in-16 and 34.8%, drawing its labels from the score itself - so the defect was not reproduced and a shape assertion would have passed on the broken arm. Rebuilt, and the missing near-constant test added. A second test now pins the ARTIFACT (`on_logit`, the formula string, the clip, and that the map can express a risk an order of magnitude below the base rate), because nothing in CI read that field and a revert of the trainer's one argument left every gate green |
| `docs/browser-verification-checklist.md` PP10 | CORRECTED SINCE. Its worked example was "measured on finance: Brier and MCB improved while log-loss got WORSE", which the fix makes false. The rule is kept; the example is re-labelled as the pre-fix state |
| `docs/deliverable-verification.md` Round 46 stamp | CORRECTED SINCE. Still asserted in the present tense that `fit_platt` "is fed a probability"; every sibling site had a supersession note and this one did not |
| `report/technical-report.tex`, `README.md` pin counts | ADJACENT, corrected where measured. The retrain unpinned both finance models (`hf_revision` -> `hf_revision_invalidated_by_retrain`), emptying the finance browser tier until a re-publish; README's "10 of 13" was wrong before and after, and is now stated from the registry (11 of 11 published specs pinned, 4 of 15 not published) |
