# The finance AML gap: what actually causes it, and what got fixed (Round 24, 2026-08-09)

`finance_aml.gbm` scores minority-class F1 **0.0808** against the published **0.6323**
(GFP+XGBoost, AMLworld HI-Small, NeurIPS 2023 D&B). The repository has carried an explanation for
several rounds. This is the measured investigation of whether that explanation is right.

Short answer: **partly wrong, and it hid a real defect that is worth fixing.**

## The recorded explanation, and what survives

`models/metrics.json` and `models/finance_aml_gbm/README.md` gave two causes:

1. IBM's Graph Feature Preprocessor is Proprietary (`snapml`) and excluded by the commercial-OK rule.
2. The corpus injects CYCLE attempts up to 10 hops while reachability here is computed to 4.

**Cause 2 cannot carry the weight it was given.** The GFP paper (arXiv:2402.08593) decomposes its own
contribution: fan-in/fan-out features improve minority-class F1 by "more than 30%" over basic
transaction features, and multi-hop cycle and scatter-gather features add "up to 4%" on top. Four
percent relative cannot explain a 7.8x gap, and this project already implements fan-in/fan-out.

**Cause 1 is real but was never specific.** It named a licence, not a mechanism.

## What was ruled out, with the measurement

| Hypothesis | Test | Verdict |
|---|---|---|
| The decision threshold (0.928426) is badly chosen | F1 = 0.6323 needs precision ~0.63 held to recall ~0.63, i.e. AUPRC >= ~0.40. Measured AUPRC is **0.0732** | **Ruled out.** Unreachable at ANY threshold - it is a ranking gap, not a tuning gap |
| Accounts unseen in training get zeroed graph features | Measured over the real temporal split: source unseen **0.24%**, destination unseen **0.17%**, either side **0.39%** | **Ruled out.** Too rare to matter |
| 10-hop cycles truncated to 4 hops | GFP paper caps multi-hop contribution at ~4% relative | **Ruled out as a main cause** |

## What was found instead: a train/test feature asymmetry

`scripts/train/finance_aml.py:82-127` computes per-account aggregates with a `groupby` over the
**whole training window**, then reindexes them onto every row. The comment at `:45` explains the
design as leak-avoidance - "Recomputing them over the full timeline would leak future edges into
past rows" - which is true of the naive alternative but not of the correct one.

The consequence is an asymmetry:

- **Training rows** receive aggregates that span the entire training window, so a row's feature
  includes activity that happened *after* it. Measured: the median transaction has only **46.4%** of
  its whole-window out-degree accrued by its own timestamp. The feature mostly describes the
  account's future.
- **Test rows** receive the same table, which for them is strictly in the past, and stale.

So the booster is fitted on a leaky feature and evaluated on a non-leaky one. The correct fix is not
a stale snapshot but a **causal** computation: for each transaction use only edges strictly before
it. That leaks nothing, needs no `fit_on`, and produces no unseen-account zeros.

## Measured, three matched arms

Same booster, same temporal split (3,554,841 train / 1,523,504 test, 5,177 positives), same 9-10
features, so only the aggregation rule differs.

| Arm | AUPRC | AUROC | best-F1 |
|---|---|---|---|
| whole-window static - **leaks**, spans train and test | 0.1731 | 0.9378 | 0.2365 |
| **causal, prior-at-time** - no leak | **0.0627** | 0.8964 | 0.1358 |
| train-fitted static - the shipped approach | 0.0432 | 0.8499 | 0.1104 |

Two things follow.

**Causal aggregation beats the shipped approach by 1.45x AUPRC (+45% relative)** at matched feature
count, with no leakage. That is a real, honest improvement and it is worth applying to the full
32-feature set.

**Leakage buys 2.76x over causal.** That is the number to keep in mind whenever a graph-feature
pipeline reports a good result: the whole-window arm looks by far the best and is not achievable
honestly. This is why the shipped pipeline's more conservative choice, though it costs accuracy,
was not a mistake in intent.

## A correction to my own first conclusion

The first comparison run here put causal (0.0627) against the leaky whole-window arm (0.1731) and
concluded that causal aggregation *hurt*. That was wrong: the comparison was against an arm that
cannot be used. The correct comparator is the shipped approach (0.0432), which causal beats. Recorded
because the mistake is easy to repeat - a leaky baseline makes an honest change look like a
regression.

## What this does NOT explain

Causal aggregation lifts AUPRC from roughly 0.043 to 0.063 in this harness. Even applied to the full
feature set, that scale of gain does not approach 0.6323. The residual is the difference the licence
note was pointing at without naming it: **GFP computes per-transaction subgraph patterns over a
dynamically updated graph, while this project computes per-account scalar aggregates.** A per-account
degree is one number per account; a GFP feature is a pattern match in that transaction's own
neighbourhood at the moment it arrives. That is an architectural difference, not a licence one, and
it is implementable without `snapml` - it is simply a larger piece of work than renaming aggregates.

## Recommendation

1. Correct the recorded note: drop the 10-hop/4-hop claim as a main cause, keep the GFP exclusion but
   state the mechanism (per-transaction subgraph patterns vs per-account aggregates).
2. Apply causal aggregation in `scripts/train/finance_aml.py` and re-measure. It is honest, it is a
   measured improvement, and it removes a train/test asymmetry that should not be there regardless of
   what it scores.
3. Do **not** retrain `finance_aml.student` to chase the baseline. Its teacher is limited by the
   feature architecture above, and distillation cannot exceed it.

---

## Outcome: the asymmetry was fixed, and the gain held at full scale

The matched-harness experiment above used 9-10 features and predicted a 1.45x AUPRC gain. Applied to
the real 32-feature trainer and retrained end to end on the full 5,078,345-row corpus, the gain was
larger:

| | whole-window (shipped before) | causal prior-at-time | change |
|---|---|---|---|
| minority-class F1 | 0.0808 | **0.1519** | **+88%** |
| AUPRC | 0.0732 | **0.1233** | **+68%** |
| AUROC | 0.8980 | **0.9526** | +6.1 pp |
| decision threshold | 0.928426 | 0.386869 | - |

**Leakage was disproved, not argued.** `causal_account_features` was run over the full frame and again
over a truncated copy; all 11 features are bit-identical on the overlapping rows. A feature that read
the future could not survive that test. The single inclusive quantity - the running maximum - includes
only the current transaction's own amount, which is knowable at scoring time and is already a feature.

**Scope of the change, stated so it is not over-read.** Eleven per-account SCALAR aggregates became
causal. The structural features - cycle closures, scatter-gather, bank diversity, burst degree, format
risk - remain fitted on the training window, because recomputing a sparse-matrix reachability power at
every timestep is not tractable here. So the train/test asymmetry is removed for the aggregates the
experiment isolated and remains for the structural block. That is a smaller claim than "the pipeline
is now causal", and it is the accurate one.

**The distilled student did not follow.** Teacher minority-F1 rose 0.0808 -> 0.1519 while the student
moved 0.0178 -> 0.0181, so the teacher-student delta widened from +6.30 pp to +13.38 pp and
disagreement on positives rose from 0.643 to 0.850. That is the expected shape: a 33-parameter linear
model distilled for CKKS feasibility cannot represent what the booster gained from better features.
Worth stating plainly rather than reporting the teacher's improvement alone - the encrypted path did
not get better, and the honest summary is that the gap between what the server can do and what the
browser can do under encryption grew.

