# The satellite metric is computed over three events (Round 24, 2026-08-09)

> **SUPERSEDED IN ROUND 86 (2026-11-07): THE MODEL IS NOW TRAINED AND SCORED AS ESA-ADB DOES, OVER ITS
> 29 ANOMALIES.** Everything below this banner is the history of the earlier models and is kept as it
> was written.
>
> The three-event denominator came from this project's own 70/30 cut by row. Since Round 86,
> `scripts/train/satellite_telemanom.py` follows ESA-ADB's Telemanom-ESA recipe, read from
> kplabs-pl/ESA-ADB at commit 67194d3:
>
> - **Data.** ESA-ADB's Mission1 split: train before 2007-01-01, stop early on the months from
>   2006-10-01, test on the half after, which holds **29 anomalies** and 36 rare nominal events on
>   channels 41-46. The 30 s zero-order hold; a z-score from the unlabelled training rows; training
>   only on windows that touch no label.
> - **Model.** A 10-step forecast scored on its first step, from Keras's initialisers; early
>   stopping with patience 20, keeping the last epoch's weights.
> - **Thresholding.** Telemanom's thresholding as ESA-ADB's TelemanomThresholding applies it. Run
>   locally against that class, 480 comparisons over 210 random series flagged the same rows.
> - **Scoring.** A reimplementation of ESAScores, checked against ESA-ADB's own test cases.
>
> One model, two input series, three error floors, scored over the test half (corrected event-wise
> F0.5; the record's `esa_scores` holds every cell):
>
> | input series | floor | anomalies only | tp / fp / fn | all events |
> |---|---|---|---|---|
> | plain zero-order hold (**this project's result**) | 0.0105, chosen on the validation months | **0.950** | 23 / 0 / 6 | 0.745 |
> | plain zero-order hold | 0.007, the paper's | 0.950 | 23 / 0 / 6 | 0.769 |
> | plain zero-order hold | none | 0.094 | 26 / 309 / 3 | 0.185 |
> | ESA-ADB's prepared series (replication) | 0.007, the paper's | **0.968** | 25 / 0 / 4 | 0.791 |
> | ESA-ADB's prepared series | none | 0.101 | 28 / 308 / 1 | 0.191 |
> | published, Telemanom-ESA-Pruned (Supp. Table 9 / Table 2) | 0.05/7 | 0.968 (P 0.999, R 0.862) | | 0.786 |
> | published, plain Telemanom-ESA | none | 0.090 | | 0.178 |
>
> **The two arms differ by one step of ESA-ADB's data preparation.** That step reads the labels: in
> every 30 s bin whose last raw sample is nominal, it writes the bin's last Anomaly or Rare Event
> sample back onto the grid (`restore_annotated`, checked against a transcription of its loop). It
> writes 10 samples on these channels, and with them the forecaster detects two more anomalies.
> A deployed detector has no labels, so this project's own result does not use the step. The
> replication applies it, and it reproduces the published figure exactly, precision and recall
> included.
>
> **The floor decides most of the score.**
>
> - Without a floor, both arms land near the benchmark's plain Telemanom-ESA, and with one, near
>   Pruned.
> - The paper's 0.05/7 is not shown to have been chosen without the test half. This project picks
>   its floor on the validation months instead. Those months hold only 2 anomalies, so the choice is
>   weakly determined: every floor from 0.0045 to 0.016 scores the same there. The paper's 0.007
>   lies inside that range and gives the same detections on the test half.
> - One seed.
>
> **The all-events reading** selects ESA-ADB's Table 2 events. The published recall 0.424 is no whole
> number of events out of this selection's 65, so ESA-ADB's all-events set differs from it by at
> least one event. The anomalies-only reading, the like-for-like row, matches 25 of 29 exactly.
>
> **Methods surveyed (2025-2026) and tested.**
>
> - A web survey found other options: a rolling-IQR threshold, robust quantiles, EVT/POT, newer
>   forecasters and cross-channel agreement. None is needed to reach the benchmark, and most have a
>   stated problem: future data in a centred window, copyleft implementations, or no code.
> - The cross-channel agreement rule was tested on this model at the owner's request: a step is
>   flagged only when at least k channels flag it (a scratch probe on this checkpoint, not a recorded
>   run). Two results:
>   - **On top of a floor it collapses recall.** At either floor, k = 2 or 3 keeps 4 to 6 of the 29
>     anomalies, where any one channel keeps 23 to 25.
>   - **In place of a floor it is a real alternative.** With no floor, k = 3 gives 0.912 on the plain
>     arm (25 / 2 / 4) and 0.905 on the replication arm (27 / 3 / 2). On all events it gives 0.900 and
>     0.897, where the floored rule gives 0.745 and 0.791, because it keeps rare events the floor drops.
>     With k = 2 there are 10 false alarms and 0.739 / 0.762.
>
>   Both k values were read off the test half, which is the very objection made to the paper's floor
>   above. So the owner chose to make it a third reading with k chosen on the validation months, and
>   the trainer now records it per arm:
>
>   | input series | k (from the validation months) | anomalies | tp / fp / fn | all events |
>   |---|---|---|---|---|
>   | plain zero-order hold | 6 | 0.909 | 22 / 1 / 7 | 0.845 |
>   | ESA-ADB's prepared series | 6 | 0.930 | 24 / 1 / 5 | 0.859 |
>
>   On those months every k from 2 to 6 finds both of their anomalies with no false alarm. The
>   time-level TNR alone picks 6, which flags the least nominal time, so the choice is as weakly
>   determined as the floor's. Against the floor the reading scores lower on anomalies (0.909
>   against 0.950) and higher on all events (0.845 against 0.745). The like-for-like headline,
>   anomalies alone, stays with the floor.
>
> **The CKKS student is replaced by a linear next-step forecaster.** Against this teacher, the
> 49-parameter logistic student scored 0 on its evaluation slice, where the teacher scored 0.961: a
> threshold on a forecast error is not linear in the window. The forecast itself nearly is. So the new
> student (`scripts/train/satellite_student_forecaster.py`) is six dot products over the last 8 rows
> (294 parameters), ridge-fitted to the teacher's own forecasts on the teacher's unlabelled training
> windows. Over the whole test half, at the teacher's floor:
>
> | pipeline | anomalies | tp / fp / fn | all events | forecast MAE, unlabelled windows |
> |---|---|---|---|---|
> | teacher (LSTM) | 0.950 | 23 / 0 / 6 | 0.745 | 0.304 |
> | student (linear, distilled) | 0.940 | 22 / 0 / 7 | 0.719 | 0.276 |
> | persistence | | | | 0.351 |
>
> The same model fitted to the truth instead of the teacher scores the same events. The page
> evaluates the student on a published three-day stretch:
> - the forecast is summed as the generator summed it;
> - the thresholding is a TypeScript port that flags the same rows as the Python on random series and
>   on the stretch;
> - every forecast runs again under CKKS (8,640 rows in 135 ciphertexts, max error 1.5e-9), and the
>   decrypted forecast changes no flag.


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
> `docs/findings.md` (the README until Round 80) links here in the present tense, so a reader arrives at a page whose opening sentence
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
