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

---

## The student: fidelity improved, discrimination did not, and a probe of mine over-promised

The student is the model that runs under CKKS, so it is the only one whose improvement reaches the
encrypted path. Two defects were found in how it was distilled:

1. **Hard labels.** `distil_linear` was fed `teacher.predict(...)` - the teacher's argmax - so the
   confidence information distillation exists to transfer was discarded before fitting.
2. **An untuned decision threshold.** The teacher's cut was tuned on validation; the student was
   scored at sklearn's default 0.5. At a 1-in-981 positive rate 0.5 is nowhere near optimal, and the
   encrypted path cannot apply a tuned cut downstream because it decides by argmax over two logits -
   which IS a 0.5 threshold. The fix is to fold the tuned threshold into the bias, since
   $`p \ge q`$ is $`w \cdot x + b - \text{logit}(q) \ge 0`$; verified exact against $`\log(q/(1-q))`$.

Both were fixed. The result is honest but mixed, and the mixed half needs stating first.

| | before Round 24 | after |
|---|---|---|
| agreement with teacher | 0.8841 | **0.9471** |
| disagreement on positives | 0.6433 | **0.3262** |
| minority-class F1 | 0.0178 | 0.0185 |

**Fidelity improved substantially - disagreement on positives halved - and discrimination did not
move.** Those measure different things, and only the first is what a distilled substitute is for: it
says the encrypted model now reproduces the plaintext model's decisions on the rare class twice as
often. The F1 is unchanged because the student's *ranking* is the binding constraint, not its
operating point.

**RETRACTED, later the same round: the ceiling was measured on a badly-fitted student.** This
section previously reported a measured ceiling of 0.0262 over any threshold and concluded that
capacity was the binding constraint. It also recorded that a probe of mine had "over-promised" by
predicting 0.0503 (hard targets) / 0.0587 (soft) and failing to reproduce. Both conclusions were
wrong, and the probe was right.

The trainer was passing the teacher `scaler.inverse_transform(scaler.transform(Xtr))` - a
mathematical no-op that is not a numerical one - while the student saw `scaler.transform(Xtr)`. When
degree-2 expansion was added the feeding was reorganised, and fixing it moved the shipped student
from 0.0185 to **0.0614** minority-class F1, above the 0.0262 "ceiling" this section had asserted.

| | before | after |
|---|---|---|
| minority-class F1 | 0.0185 | **0.0614** |
| agreement with teacher | 0.9471 | **0.9829** |
| disagreement on positives | 0.3262 | 0.3619 |

0.0614 lands next to the probe's 0.0587 prediction, so the probe reproduced and the trainer was the
outlier. **The lesson inverts the one recorded before.** I had written that "the trainer's number is
the one that counts because it is the one that ships" - true as far as it goes, and it stopped me
looking for the defect that made the two disagree. A probe and a pipeline that disagree by 3x is
evidence of a bug in one of them, and "the shipped one is authoritative" is a reason to investigate,
not to close.

The capacity claim is not fully retracted: 0.0614 against the teacher's 0.1519 is still a large gap,
and a 32-feature linear model under CKKS remains the constraint. What is retracted is the specific
number, the conclusion that tuning was exhausted, and the characterisation of the probe.

**What would actually move the student**, in cost order: more capacity than 32 linear features
(a small MLP is not CKKS-feasible at this budget, which is the real constraint); or teacher features
the student can use, since the student inherits the same 32 columns and cannot exceed what a linear
function of them supports.

---

## Windowing the aggregates does not close the gap (Round 24f, 2026-08-10)

Taking the architecture difference recorded above - per-transaction neighbourhood patterns against
per-account cumulative aggregates - one reading of it is **recency**: a cumulative degree cannot separate an account that has
been busy for a year from one that was silent for a year and then sent forty transactions this
morning, and the second is what a mule looks like.

That reading was implemented and measured. `windowed_causal_features` adds six columns - prior count
and prior amount inside a 24-hour sliding window, per side, plus the share of lifetime activity
falling inside it - computed by binary search within each account's own sorted block, so they are
strictly prior by construction and cost 14.9s over 5.08M rows.

| arm | features | minority-F1 | AUPRC | AUROC |
|---|---|---|---|---|
| A cumulative causal (shipped) | 32 | **0.1519** | **0.1233** | 0.9526 |
| B + windowed recent-history | 38 | 0.1308 | 0.1152 | **0.9606** |
| delta | +6 | **-0.0211** | **-0.0081** | +0.0080 |

**It made the model worse on both metrics that matter, and better on the one that does not.** AUROC
rose while AUPRC and minority-class F1 fell, which is the textbook signature of a change that helps
the majority class at a 1-in-981 positive rate - exactly the reason this project reports AUPRC and
minority-F1 as the headline and keeps AUROC as a companion. Had AUROC been the headline this would
have shipped as an improvement.

**So recency is not the missing piece, and the windowed features are not shipped.** The function is
retained because the negative result is worth keeping executable - anyone re-reading the gap note
will have the same idea, and it costs 15 seconds to re-run rather than a day to re-derive.

What remains untested is the other reading of the same sentence: genuine per-transaction MOTIF
membership - does this specific transaction participate in a detected scatter-gather or cycle
pattern within its own neighbourhood - rather than an account-level statistic evaluated over any
window. That is a materially larger piece of work than re-aggregating existing quantities, and this
result is evidence that the cheaper interpretation has been exhausted.
