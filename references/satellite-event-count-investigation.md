# The satellite metric is computed over three events (Round 24, 2026-08-09)

> **THE MODEL MEASURED HERE WAS RETRAINED IN ROUND 71 (2026-10-26).** Its weights had been trained with
> a z-score fitted on the whole series, test window included, a leak the trainer lost in Round 34
> without the model being retrained. The retrained model still flags 2 points of 441,600, and neither
> falls inside one of the 3 scored events: event-wise F0.5 **0** (tp=0 fp=2 fn=3), point-adjusted F1
> 0. The figures below are the leaky runs'. The argument is unchanged and the new run is one more
> instance of it: with 3 events the metric moves in steps a single flag can take.

> **THE COMPARATOR THIS INVESTIGATION ARGUES AGAINST WAS SUPERSEDED IN ROUND 24, AND THIS FILE STILL
> OPENS ON IT. Banner added Round 55.**
>
> Everything below is framed against ESA-ADB's **0.786** from Table 2. That is the LENIENT row and the
> wrong comparator: Table 2 scores all events excluding communication gaps, i.e. anomalies PLUS
> annotated rare *nominal* events, while this project scores anomalies alone. The like-for-like row is
> Supplementary Table 9, Telemanom-ESA-Pruned, **corrected event-wise F0.5 0.968**, and that is what
> the registry, the README, both model cards, `config.defaults.json`, `report/build_data.py` and the
> trainer have all carried since Round 24.
>
> `README.md` links here in the present tense, so a reader arrives at a page whose opening sentence
> contradicts every other surface. **The denominator argument below is unaffected and is the reason
> this file exists** - three scored events against ESA-ADB's 29 is the point, and it holds against
> either comparator. Only the number in the opening paragraph is retired. 0.786 legitimately survives
> in the registry as the recorded field `published_lightweight_f0_5`.
>
> Following the convention of [`qcnn-vs-published-wdbc.md`](qcnn-vs-published-wdbc.md), which carries
> the same kind of banner for the same kind of reason.

`satellite_comms.telemanom` records event-wise F0.5 **0.4545** against ESA-ADB's published
**0.786**, and the demo has been rendering those two numbers side by side as if they were
comparable. They are not, and the reason is the denominator.

## What was actually being divided by

The metric's denominator is labelled anomaly **events**, not samples. The record beside it says
`n_anomaly_intervals: 320`, which is the whole timeline. Counted directly from the corpus at the
trained cadence:

| | |
|---|---|
| resampled rows (300s) | 1,472,832 |
| labelled intervals, whole timeline | 320 |
| scored test window (last 30%, minus the 250-step warmup) | 441,600 points |
| **labelled anomaly events INSIDE the scored window** | **3** |

The three: 630 points (52.5 h), 1 point (0.1 h), 6 points (0.5 h). The annotated anomalies are
concentrated earlier in the mission, so a temporal split leaves almost nothing to score. The
reconstruction reproduces the recorded `n_eval` of 307,601 exactly, so these are the same numbers
the training run used.

## What the recorded scores decompose to

On 3 events each F0.5 has exactly one integer solution, so no inference is needed:

| recorded run | decomposition |
|---|---|
| 0.714286 (earlier round) | tp=1, fp=0, fn=2 |
| 0.454545 (current)       | tp=1, fp=1, fn=2 |

**Same single detection, same two misses, both times. The entire 0.26 swing between them is one
extra false-alarm run.** From that one detection the score reads:

| false-alarm runs | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| event-wise F0.5 | 0.714 | 0.455 | 0.333 | 0.263 | 0.217 | 0.185 |

This closes an item the model card carried as **STILL OPEN** - "event-wise F0.5 does not reproduce
across identical-config retrains". It reproduces exactly as well as a 3-event denominator permits.
The instability was real and was correctly attributed to overlap-based metrics in general
(arXiv:2607.11969, 2026-07); the operative fact turned out to be simpler and local.

Point-adjusted F1 reproduces across the same retrains (0.9937 vs 0.9945) for the same reason it is
inflated: it is computed over 441,600 points rather than 3 events.

**The teacher flags 2 points out of 441,600.** The dynamic threshold is that conservative, so the
0.4545 is one hit out of two flags. A metric that can be moved a quarter of its range by a single
flagged sample is not a measurement of detector quality.

## The knock-on: a fidelity claim built on 2 events

The distillation slice holds **2** of those 3 events. On it the teacher scores **0.0** - it detects
neither - and the 49-parameter CKKS student scores 0.003234, which decomposes uniquely to tp=1,
**fp=385**, fn=1.

So `metric_delta` came out at **-0.0032**, and the demo rendered that with no caveat: a negative
delta reads as the encrypted 49-parameter student outperforming an 80,486-parameter LSTM. What it
actually represents is one detection bought with 385 false alarms, set against a teacher that caught
nothing.

A guard existed for this and did not fire. `both_scored_zero` was written after a real incident in
an earlier round, but its condition required BOTH models to score zero, and here the student scored
0.0032. **A guard whose condition is narrower than the thing it guards against gets walked around.**
The condition is now `teacher_scored_zero`: if the teacher scored nothing on the slice there is no
fidelity claim to make, whatever the student did.

## What changed

1. `event_wise_counts()` returns `n_true_events` / tp / fp / fn beside the score.
2. The trainer records `n_events_scored` and the tp/fp/fn, and prints a warning under 10 events.
3. `FidelityReport` carries `n_scoring_units` + `scoring_unit`, because `n_eval` counts samples and
   for this metric that is wrong by five orders of magnitude (307,601 samples, 2 events).
4. The guard generalised from `both_scored_zero` to `teacher_scored_zero`.
5. The UI prints the event count on the same row as the baseline - the row where the comparison
   actually gets made - and says "too few to compare with the baseline" under 10.

## What was deliberately NOT done

**The split was not moved to capture more events.** Choosing a split because it puts more anomalies
in the test set is fitting the evaluation to the answer, and the temporal split is the honest one
for a forecasting detector.

**The score was not reweighted, smoothed, or replaced with the reproducible companion.**
`point_adjusted_f1` is stable at 0.99 and is the metric TSB-AD (NeurIPS 2024) showed a uniform
random score beats; promoting it because it reproduces would trade a noisy honest number for a
stable dishonest one.

**The number stands as measured, with its sample size attached.** The remaining honest options are
to score more channels - the published full-channel figure is 0.061, so the lightweight subset was
already the favourable choice - or to accept that this corpus's temporal tail does not support a
benchmark-comparable event-wise measurement, and to say so where the number is displayed. Round 24
did the latter.
