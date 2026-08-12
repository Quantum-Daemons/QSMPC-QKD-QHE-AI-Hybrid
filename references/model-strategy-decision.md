# Should the encrypted tier get a different model? Download, fine-tune, RL, or train from scratch
### Round 24d, 2026-08-10. A decision with the measurements and citations behind it.

The recurring question is whether the WebUI needs a better ML model - fetched from Hugging Face,
fine-tuned, transfer-learned, trained with reinforcement learning, or trained from scratch. This
records the answer per option, with what was measured rather than assumed.

## The constraint, stated precisely

The encrypted tier evaluates 20-3,226 parameters. That is not a preference: CKKS at
`poly_modulus_degree` 8192 gives 4096 slots, and a ciphertext-plaintext dot product is one
multiplicative level. Anything deeper needs bootstrapping, which this project does not do in a
browser tab.

**But "small" and "weak" are different claims, and conflating them cost two rounds.** The medical
student was 3,226 parameters and scored AUC 0.5984, which was written into three documents as a
capacity limit. It was a shuffled-label bug; the same architecture on aligned labels scores 0.7882.
Checklist item S3 exists because of that: *before concluding the encrypted tier is small so it is
weak, train the same small model properly once and see what it can do.*

## Option 1: download a different model from Hugging Face - NO

No pretrained model exists at 20-3,226 parameters for these tasks; the smallest useful published
models are five to six orders of magnitude larger. A downloaded model would have to be distilled
down anyway, which is what already happens, and the teachers here already sit at or above their
published baselines (resnet18 0.9147 vs 0.901; opssat 0.9681 vs 0.979; resnet50 0.7988 vs 0.857).

## Option 2: reinforcement learning - NO, and it is a category error at this scale

Checked because it was asked, and the 2026 literature is clear in the opposite direction.

**The paper that looks like a counterexample is not one.** arXiv:2602.04118 ("Learning to Reason in
13 Parameters", 2026-02-04) trains **13 parameters** by RL to 91% on GSM8K. Those 13 parameters are
LoRA deltas on a **frozen 8-billion-parameter backbone**. The capacity is in the 8B; RL worked
because there was an enormous pretrained function to steer. It is evidence against, not for,
standalone tiny-model RL.

**Direct evidence of fragility at scales far above ours.** arXiv:2607.25091 (2026-07-27) tested RL
on Pythia 70M/160M/410M and SmolLM2 135M/360M and documents three reproducible failure modes: silent
LoRA parameter freezing, PPO importance-ratio overflow under bfloat16, and catastrophic policy
collapse from reward-model error. At **70M** parameters RL was already unstable. This project's
students are four to six orders of magnitude smaller than that.

**Where RL is legitimate nearby, and why it still does not apply.** arXiv:2508.08646 (P-CAFE) uses
RL for per-patient, budget-constrained, sequential feature acquisition. The value comes from the
sequential budget, which this project does not have - features arrive together. And the closest
published treatment of the actual problem, arXiv:2605.03289 ("Imbalanced Classification under
Capacity Constraints", 2026-05-05), solves maximise-sensitivity-subject-to-a-cap with **prior
reweighting and post-hoc thresholding, explicitly not RL**.

A 33-parameter convex problem does not need policy gradients. It has a global optimum reachable by
convex optimisation, and the operating point is a one-dimensional sweep.

## Option 3: train from scratch - ALREADY WHAT HAPPENS, and it is now the default

Every student here is trained from scratch by distillation. The teachers are too, since Round 24c:
ImageNet initialisation was made opt-in because ImageNet's terms are uncleared for redistribution
and the published artefact had been recording `pretrained: ImageNet1K_V1` while two documents said
random initialisation.

## Option 4: change the FEATURE MAP, not the model - YES, and it is measured

This is the one that paid, and the framing that unlocked it was that **4096 slots, not parameter
count, is the real budget**. A polynomial expansion is a function of the plaintext input, so the
client computes it before encrypting; the encrypted step remains one dot product at unchanged depth.

| model | linear | quadratic | verdict |
|---|---|---|---|
| `satellite_comms.opssat_student` | F1 0.8649, AUC-PR 0.9275 | **0.9043 / 0.9681** | shipped |
| `finance_aml.student` | F1 0.0614 | 0.0115 | rejected |

**And it survives a bootstrap, which was worth checking before believing it.** A 0.04 AUC-PR delta
on a 529-row imbalanced test set is exactly the size that turns out to be resampling noise. Paired
bootstrap, 4,000 resamples, both models scored on identical rows in each draw:

| delta | mean | 95% CI | P(quadratic > linear) |
|---|---|---|---|
| AUC-PR | +0.0460 | **[+0.0089, +0.0853]** | 0.991 |
| minority-class F1 | +0.0528 | **[+0.0154, +0.0929]** | 0.999 |

Both intervals exclude zero, so the gain is real at this sample size. The interval is wide - the
lower bound is a fifth of the point estimate - so the honest claim is "a real improvement of at
least about one AUC-PR point", not "+0.046".

**Not novel, and cited as such.** The mathematics is a known *explicit feature map*
(arXiv:2203.03636, which reports +12.04% balanced accuracy / +12.58% F1 for a logistic classifier in
the aEFM domain). Placing feature construction in client-side plaintext specifically to bound HE
depth is published system design: arXiv:2605.16647 (2026-05-15) does tokenisation, embedding lookup,
projection and clipping client-side then encrypts, reaching depth 8 / ring 32768 against depth 10 /
ring 65536 for attention; Safhire (arXiv:2509.01253) offloads non-linearities to the client for a
1.5-10.5x latency improvement. This project applies a known technique; it does not introduce one.

**Two costs that are NOT zero and were initially overlooked.** Depth is unchanged, but slot-summation
rotations rise from ceil(log2 19) = 5 to ceil(log2 209) = **8 key-switches**, which is more noise
growth and more runtime. And the client must know the 209-term feature map, which **discloses model
structure** (not weights) - Safhire treats exactly that as the problem it solves. Neither is fatal
here, and both belong in the record rather than in a footnote nobody reads.

## Option 5: better targets - measured per model, and the answer differs

S3 applied to the finance student, identical architecture, three target sources:

| arm | minority-class F1 | AUPRC |
|---|---|---|
| A teacher SOFT probabilities (ships) | **0.0614** | 0.0241 |
| B teacher HARD argmax | 0.0537 | 0.0243 |
| C GROUND TRUTH | 0.0484 | 0.0219 |

**The shipped student beats the same architecture trained on ground truth.** So distillation is
adding value rather than costing it, and for finance the capacity claim survives S3 - now
demonstrated rather than asserted, which is the difference from the medical case where the identical
probe disproved it.

## Decision

No new model. No RL. The remaining levers, in cost order, are the feature map (done where it pays) and
better targets (checked, already optimal for finance).

**The third lever - fitting the model to the objective it is scored on - is CLOSED and implemented, and
this section called it "the one genuinely open option" for several rounds after it shipped (corrected
Round 33).** The integer scoring systems of Chi and Birbil (arXiv:2604.04241), which optimise net
benefit directly rather than a surrogate loss, are in the tree end to end:

| where | what is there |
|---|---|
| `src/qsh/models/scorecard.py` | the implementation; its docstring writes out the sparse integer program |
| `scripts/train/finance_scorecard.py` | the trainer |
| `tests/test_scorecard_bfv_exact.py` | asserts equality under BFV **and** inequality under CKKS, so a change that made CKKS look exact would fail |
| checklist row `H4` | 60 of 60 rows exact under BFV, against **-1.999998397974359** for an exact -2 under CKKS |
| `report/technical-report.tex` | cites it |
| `references/he-scheme-selection.md` | the scheme argument and the net-benefit comparison |

**And as of 2026-08-12 the numbers are a machine record rather than hand-written prose.** Until this
round `models/metrics.json` had no `finance_aml.scorecard` key at all while `he-scheme-selection.md`
published its figures by hand - which is how the new generated-prose gate found it. The trainer was run
(CPU, 272.3 s) and the record now reads **net_benefit 0.00030763** on **1,523,504** test rows,
**n_nonzero_coefficients 12**, **exact_under_bfv true**, with `net_benefit_threshold` and
`net_benefit_treat_all` giving the decision curve real inputs. So the hand-written +0.000308 in
`he-scheme-selection.md` is confirmed by an artifact rather than resting on a transcription.

The finding this closed on is not the accuracy: the integer scorecard **loses** to the CKKS student on
net benefit, and that was measured in Round 24h. What it buys is **exactness** - the encrypted score
equals the plaintext score bit for bit under BFV, which no CKKS student can offer - and interpretability,
which is the reason the credit-scoring literature uses the form at all. It ships as a second,
independent BFV context alongside CKKS, with no scheme conversion anywhere.

**So there is no genuinely open option left on this list.** If a later round wants one, it has to be
found rather than inherited from this paragraph.
