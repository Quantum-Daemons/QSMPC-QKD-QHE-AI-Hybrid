# Verification results log

What each round actually measured against the deployed demo. Split out of
the project's browser verification checklist in Round 24, when it had reached 1478 lines - a checklist you
cannot execute in one pass has stopped being a checklist.

These are **records, not instructions**. The checklist says what to check; this says what was found
when it was checked, with the bundle hash and date so a claim can be traced to a specific deploy.
Nothing here is deleted when it ages - a superseded measurement is evidence about how a number
moved, which is exactly what the next round needs when a figure looks wrong.

**Since Round 56 the deploy walks, with their bundles and measured values, are written in the round
register, [`deliverable-verification.md`](deliverable-verification.md), and this file holds what
each round taught.** Noted 2026-10-30, when the technical report's pointer was brought in line with
that practice.

## Ordering: it is not chronological, and reading it as chronological will mislead you

**Stated explicitly in Round 33, because the file had drifted into three different orders and nothing
warned the reader.** Round 28 sits at the top because it was the newest entry when this file was split
out and was written as the lead; Rounds 21 through 24b then run **oldest to newest** beneath it; and
Rounds 29 onward are **appended at the end**. So the top of the file is neither the newest nor the
oldest entry, and scrolling down crosses the round sequence twice.

**The convention from Round 32 onward: append new entries at the END, in ascending round order.** Nothing
above is reordered, because every entry names its own round, date and bundle, and rewriting the file to
fix the sequence would rewrite records whose value is that they have not been touched since they were
written. The index is the fix instead:

| entry | where |
|---|---|
| Round 21 | below Round 28 |
| Round 22 | below Round 21 |
| Round 23b | below Round 22 |
| Round 24, Round 24 part 2, Round 24b | below Round 23b |
| **Round 28** | **at the TOP of the file** |
| Rounds 29, 30, 31 | appended, reconstructed in Round 32 |
| Round 32 | appended, reconstructed in Round 33 |
| Round 33, Round 34 | appended, at the end |
| Round 35 | appended, measured in Round 36 |
| Round 36 | appended, at the end |
| Round 37 | appended, at the end |

Rounds 25, 26 and 27 have no entry here and never did. That is a real gap in the record rather than an
ordering artefact, and it is not reconstructable now: it predates the point where this file existed as
the recorded home for gate-4 results, and the measurements were not kept anywhere else.

## Verified in Round 28, deploy 2f0d550, bundle index-BET6lubQ.js, 2026-08-11

Deployed `origin/main` at `2f0d550` after clone-verifying that the commit stands alone. Built with
both `-p qsh-demo` and the behind-proxy override; container recreated and confirmed on the front
proxy's network as well as its own; front proxy restarted rather than reloaded.

**The bundle changed**, which is the check that proves the image was rebuilt rather than reused:
`index-BPtZC3yo.js` (1,076,461 B) became `index-BET6lubQ.js`. The old bundle contained `Kotowski
et al` and no `Ruszczak`; the new one is the reverse.

| what was checked | measured |
|---|---|
| `crossOriginIsolated` | **true** - so the timing figures the demo prints are meaningful |
| `/api/` requests in demo mode | **0** of 12 total requests; **0** cross-origin hosts |
| console errors across five use-case pages | none |
| satellite overlay | **529** scatter symbols, and the page renders the CORRECTED citation |
| medical overlay | **569** symbols, labelled as a different cohort from the model's corpus |
| finance overlay | **8192** symbols, CDLA share-alike terms shown, log axis noted |
| genomics overlay | **64** symbols - the gnomAD allele-frequency slice |
| telecom overlay | **0**, as intended: closed as impossible, not pending |
| medical QCNN weights served | 19 keys with `headline_metric`, accuracy 0.732558 vs LR 0.953488 |
| finance QCNN weights served | present as `application/json` (they 404'd as HTML before this deploy) |

**The medical accuracy in that row is no longer the artifact's, and the row is left as measured.**
0.732558 is what the deploy served on 2026-08-11 and the record of a deploy is not editable
afterwards. The circuit has been replaced once (Round 46, a hardware-efficient ansatz relabelled as
a real QCNN) and corrected TWICE in Round 47 - once for the fully-connected block and the multiclass
readout, and once for two classes of analytically dead parameter. That gave accuracy
**0.883721** with minority-class F1 **0.84375** against the same logistic regression's **0.953488** /
**0.939394**, so the direction of the comparison held and only its size moved. Round 84's retrain at
c = 0.5 then brought seed 0 level, 0.953488 each, with logistic regression ahead over five seeds
(`models/qcnn_medical/README.md`). The
intermediate figures from the first Round 47 retrain, accuracy 0.860465 with minority-class F1
0.8125, are themselves SUPERSEDED and are recorded here only so the sequence is followable.
`models/qcnn_medical/weights.json` is authoritative; anything quoting 0.732558, or 0.860465, as
current is quoting a retired circuit.

**The licence defect is cleared on the live site.** `/data/satellite/real_2d.json` had been
serving `Ruszczak et al., 'Machine learning detects terminal singularities'` - an
algebraic-geometry paper's title on an OPS-SAT dataset. OPS-SAT-AD is CC BY 4.0, so that was a
licence condition being breached in public, not a typo. It now serves the correct title and the
full author list, confirmed both in the JSON and in the rendered page text.

**Content-type was the discriminator throughout, never the status code.** Every probe was paired
with a nonsense-path control in the same session. The control returns **200 with `text/html`, 645
bytes** - the SPA shell - so a file that was never deployed looks present to a status-code check.
Absent assets matched the control byte-for-byte; present ones returned `application/json`.

**One defect is confirmed still live, and it is fixed in an unmerged branch.** On the genomics
page, selecting `Inference = QCNN` produces a page that contradicts itself. The run log says,
correctly:

> QCNN selected but no trained weights ship for genomics: the untrained Clifford+T circuit is a
> hiding and T-count demonstration (56 T-gates on 8 qubits), NOT a classifier

while the explanation panel says "the class this page surfaces comes from the quantum circuit" and
the diagram bar says "QCNN surfaced as primary". The surfaced class was 0, from CKKS. Both panels
read the control value rather than the artifact that records which backend ran. Reproduced
deliberately here so the record shows the defect was observed on the deployed site rather than
only reasoned about.

## Verified in Round 21

Observed on `https://<demo-host>`, recorded rather than ticked.

Environment, browser A: `crossOriginIsolated` true, zero `/api` requests, WebGPU adapter present but
software (SwiftShader, no `shader-f16`), WebNN API present and deliberately not offered to ORT.
Browser B: same page, `requestAdapter()` returns null, so the wasm build is used. **The adapter is
recorded with every latency below, because it changes them by two orders of magnitude.**

Published model running inside the orchestration, per use case:

| Use case | Model | Adapter | Execution provider | Latency | Input |
|---|---|---|---|---|---|
| finance_aml | finance_aml.gbm | B | onnx-wasm | 1.42 ms | 1x32 |
| medical_fl | medical_fl.resnet18 | A software | onnx-webgpu | 288 ms | 1x1x28x28 |
| medical_fl | medical_fl.resnet50 | A software | onnx-webgpu | 12842 ms | 1x1x28x28 |
| medical_fl | medical_fl.resnet50 | B none | onnx-wasm | **42 ms** | 1x1x28x28 |
| genomics | none registered, and the run says so | | | | |
| telecom_ids | telecom_ids.qml | B | onnx-wasm | 5.94 ms | 1x7 |
| satellite_comms | satellite_comms.telemanom | A software | onnx-webgpu | 151 ms | 1x250x6 |
| satellite_comms | satellite_comms.opssat | B | onnx-wasm | 4.36 ms | 1x19 |
| satellite_comms | satellite_comms.opssat_student | B | onnx-wasm | 5.98 ms | 1x19 |

The two ResNet-50 rows are the finding: 12842 ms on a software WebGPU adapter against 42 ms on wasm,
same model and same row. `hasWebGPU()` now declines software adapters, so the 12842 row is no longer
reachable - it is kept here as the evidence for why.

Routes swept through the SPA router: all ten clean - zero page overflow, zero elements wider than the
viewport, zero SVG labels outside their viewBox, zero `NaN`/`undefined`/`Infinity`/`[object Object]`.
`/agentic-ai` redirects to `/usecases`; `/usecase/<bad-id>` renders "No such use case".

Diagram labels, measured as `bbox.x >= 0 && bbox.x + bbox.width <= viewBoxWidth` after a run, in the
state producing each one's longest label: finance 81 chars (172/167 px margins), medical ResNet-50 91
chars (147/143), satellite OPS-SAT student 95 chars (133/129). All fit. Before the `local-wide` fix
the satellite label lost its first 17 characters off the left edge.

The published-model select is disabled while `state === "running"` and re-enabled at `done`, sampled
at 200 ms intervals from the click.

Dynamic parameter reaching all three surfaces, medical page, `qkd_distance_km` at two values:

| Distance | Diagram | qber | skr | secure_key_bits |
|---|---|---|---|---|
| 50 km | BB84 decoy-state . QBER 0.02 | 0.02 | 0.00906 | 1811 |
| 250 km | BB84 decoy-state . QBER 0.5 | 0.5 | 0 | 0 |

Diagram and metrics move together, and the key dies where the channel model says it should. Note that
QBER is already pinned at its 0.5 ceiling by 250 km, earlier than the "around 400 km" this file
records under numeric edges - that figure was approximate and the saturation point is nearer 250 at
the default 0.2 dB/km.

The config-to-bundle half of the same check, run against the override layer: `config.yaml` at 0.2
gives a bundle carrying `fiber_loss_db_per_km:.2`, at 0.9 gives `.9`, and restoring returns `.2`.

Three CKKS students now run and reproduce, each against the `expected` block of its own
`models/*/eval_sample.json`. **The three figures were written out here and in `README.md` and went
stale in both** - Round 50 corrected them and removed the copy rather than correcting it twice, since
one sentence duplicated across two files is two places to forget. The artifact is the record; read
`expected.accuracy` and `expected.minority_class_f1` there. `satellite_comms.student` still reports
`DID NOT RUN` with its reason - it is scored on event-wise F0.5, a sequence metric the per-row
evaluator cannot compute.

What the stale copies said, kept because a number that moved is only checkable against the record
that moved it: medical accuracy 0.730769 against the true 0.769231 - and 0.730769 is that artifact's
own `majority_baseline`, so the README was printing the trivial predictor as the model's score, the
exact signature this project elsewhere calls "a model that learned nothing". OPS-SAT rare-class F1
0.864865 against 0.904348, and finance 0.333709 against 0.376130, both superseded by retrains.

Reproduction check: ResNet-18 scored 85.9 percent in the browser against 85.9 percent measured
offline on the same 156 rows, majority baseline 73.1 percent, 11.3 MB transferred. The medical
student scores 0.7308 against a majority baseline of 0.7308 - it reproduces its published figure and
does not beat guessing, and the page says both.

Benchmark rows after the Round 21 sweep, medical page, all three naming what they are:

| path | model | accuracy | security |
|---|---|---|---|
| MPC-FHE | FHE-feasible diagnostic CNN (3226 params, measured) | em dash | 68 |
| QSMPC-FHE | QOTP + 8-qubit QCNN circuit (no trained weights) | em dash | 96 |
| plaintext | medical_fl.resnet18 | em dash | em dash |

Every accuracy is an em dash and every one is correct: MPC-FHE evaluates no labelled set, QSMPC-FHE
has no trained weights, and the plaintext row is a single inference. Before this sweep the middle row
read 1.0.

Inference provenance, same run, both inferences named:

    Pipeline model, on this page's own data    MEASURED (plain JS) - js-linear (bundled real model)
    Published model, on its own corpus         MEASURED (local WASM) - onnx-wasm

## Verified in Round 22 (2026-08-08, public URL, bundle index-Bjfk0ThY.js)

Worked against `https://<demo-host>/usecase/finance_aml`, cache-busted, with
the served bundle grepped first. `crossOriginIsolated` true, `SharedArrayBuffer` a function.

**The test the retired score failed, now passing.** Swapping the KEM and re-running moves the
reported facts, and the movement checks against the standard:

| Control | Before | After | Checks against |
|---|---|---|---|
| KEM select | ml-kem-768+x25519, NIST Category 3 | ml-kem-1024+x25519, NIST Category 5 | FIPS 203 |
| KEM ciphertext | 1088 bytes | 1568 bytes | FIPS 203 |
| Signature (unchanged control) | ML-DSA-65, 3309 bytes | ML-DSA-65, 3309 bytes | FIPS 204 |

The facts do NOT move when the selector changes without a re-run, which is correct - they are pinned
to the run that produced them.

**Eavesdropper, both directions.** `pp-qkd_eve` 0 -> QBER 0.01, 1915 secure bits, SKR 0.00958/pulse,
step ticked in the normal colour. `pp-qkd_eve` 0.35 -> QBER 0.1, **0 secure bits**, SKR 0, and both
the table and the diagram report the failure. The old score gave this run 100/100.

**Advantage raw value.** Page printed `raw 10.65 (clamped to 0-10)`; recomputing from the page's own
printed ratios (comm 6145.8, latency 3396.03, security 1.5) against the config weights
(0.6/0.2/0.2, log_scale 3.2, log_offset 1.0) gives 10.65. Reproduced exactly.

**Accessibility.** 41 controls on the use-case page, **zero** without an accessible name (4.1.2).
Zero genuine contrast failures (1.4.3) once alpha is composited; the Art. 50 disclosure measures
12.42:1. Parameters disclosure: `aria-expanded` false -> true, `aria-controls="pipeline-parameters"`
resolves, 15 fields appear.

**Every parameter field against its own constraints.** No violations. `pp-seed` now satisfies its own
`min` - the Round 21 half-fix is closed. Two selects hold `""`, and both are legitimate sentinels
("(use case default)" and "CKKS (FHE) - default"); `config.inference || "ckks"` confirms the second
label is honest for every use case.

**Agentic AI panel, exercised end to end** (genomics, distilGPT-2 via transformers-wasm, 2430 ms):
tool `pool_allele_counts` - the per-use-case tool, not a generic one - `authenticated (ML-DSA-65)`
true and `tampered message rejected` true. The Combiner select genuinely changes the derived key:
**hkdf -> d9c4f1c2, its_toeplitz -> 3152292a**. A combiner that left the fingerprint unchanged would
mean the selection never reached the key derivation.

**Export buttons** are all enabled after a run and carry a reason before one. Round 22 found the
Agentic AI panel's "Download results (JSON)" was the exception - disabled with no `title`, so it read
as broken rather than as not-yet-applicable. Fixed. When auditing this, read `disabled` AND `title`
together; `disabled` alone tells you nothing about whether the user can understand why.

**Found and fixed this round:** the diagram ticked a zero-key BB84 step. See the panel-consistency
section above.

One caution for the next round, learned twice here: a synthetic `.click()` on a React control does
not update `aria-expanded` within the same tick, and reading it immediately makes a working button
look inert. Read after a frame.

## Verified in Round 23b (2026-08-09, public URL)

Public-page claims re-checked against primary sources this round, all **confirmed**:

| Page claim | Source fetched | Verdict |
|---|---|---|
| AI Omnibus = Regulation (EU) 2026/1744, in force 27 Jul 2026 | EUR-Lex | correct, and so are both deferral dates it cites |
| Annex III to 2 Dec 2027, Annex I embedded to 2 Aug 2028 | EUR-Lex / Commission | correct |
| Art. 50 transparency applicable 2 Aug 2026 | Commission guidelines | correct |
| EO 14412, 22 Jun 2026, 91 FR 38483, PQC by 2030, signatures by 2031 | Federal Register | correct on all four details |
| RSA-2048 factoring: ~1/20 the qubits of the 2019 estimate, runtime ~20x higher | arXiv:2505.15917 (Gidney 2025) | correct: 20M -> under 1M qubits, 8 hours -> under a week |
| ETSI GS QKD 014 V1.1.1 (2019-02) still the only published edition | ETSI | correct - and an edition that has NOT moved is itself worth recording |
| liboqs 0.16.0 current | GitHub releases | correct (2026-07-09) |

Registry and artefact checks, all clean after the fixes in this round: 10 model cards agree with
`models/metrics.json`; 10 declared HF repos return 200; every `ARTIFACT_DIRS` entry resolves; all 10
ONNX input shapes match what the run reports in the browser.

Per-page: no retired composite on any of the five use-case pages or the index; compact heading reads
"Security posture & quantum advantage" on telecom and satellite, full heading elsewhere; telecom run
end to end gives QBER 0.01, macro-F1 0.96, `published telecom_ids.qml @ f0d754313210 - onnx-wasm -
1x7`, matching that model's ONNX input of `[?, 7]`.

## Verified in Round 24 (2026-08-09, public URL, bundle index-CHGs2rHz.js)

**XAI on all five use cases**, each attribution checked against something outside itself:

| Use case | Top attribution | External check |
|---|---|---|
| telecom_ids | `qber_mean` **+4.692**, 8.5x the next feature, for a predicted `intercept_resend` | correct - intercept-resend is the attack that raises QBER, verified here at exactly 25% against Gottesman-Lo |
| satellite_comms | `E-6[t7]` -0.427, then `E-6[t6]`, `E-6[t5]` | correct shape - one channel, latest timesteps, which is what a windowed detector should key on |
| genomics | `rs3803662` | a real TOX3-locus breast-cancer PRS SNP |
| finance_aml / medical_fl | reference point reported as `sample mean E[x] over the shipped rows` | the artifact records WHICH reference point, so the claim is checkable |

**Run status indicator**: rendered `position: fixed`, in viewport at 3655 px scrolled, `role="status"`
with `aria-live="polite"`, text "running". Absent on idle and after a completed run has been read.

**Served bundle contained every new literal** before the browser was trusted: `role:"status"`,
`aria-live`, `ai_generated`, `browser_override_reason`, `Article 50(2)`. (This list said
`is_ai_generated` until Round 40; the emitted key is `ai_generated`.)

**One false negative worth recording.** The status indicator first appeared not to render. It was
rendering correctly - the run had simply finished before the check ran, and the indicator hides on
`done` by design. The lesson is the one this document already carries in another form: when a control
is conditional on a transient state, sample it DURING that state, not after.

**Facts re-derived rather than trusted:**

- QSDC abort threshold: $`C_s = 1 - 2h(e)`$ reaches zero at **0.110028**, identical to Shor-Preskill
  because it is the same expression. At the GLLP zero-key point 0.098106 QSDC still has Cs = +0.074.
- Finance gap: AUPRC 0.0732 makes F1 0.6323 unreachable at any threshold; unseen accounts affect
  0.39% of test rows; causal aggregation measures 1.45x the shipped approach, leakage 2.76x causal.

## Round 24, part 2 (2026-08-09) - causal aggregates shipped

Retrained `finance_aml.gbm` on the full 5,078,345-row corpus with prior-at-time aggregates:

| | before | after |
|---|---|---|
| minority-class F1 | 0.0808 | **0.1519** |
| AUPRC | 0.0732 | **0.1233** |
| AUROC | 0.8980 | **0.9526** |
| decision threshold | 0.928426 | 0.386869 |

Leakage disproved by truncation: all 11 causal features bit-identical whether or not later rows
exist. Published as `cf950f8c1e8d` (gbm) and `60686ed0139e` (student), both repos PUBLIC before and
after. Verified live: dropdown lists both models, and the run reports `published finance_aml.gbm @
cf950f8c1e8d - onnx-wasm - 1x32 - 5.13 ms`.

The distilled student did not follow the teacher: 0.0178 -> 0.0181 while the teacher nearly doubled,
so the delta widened to +13.38 pp and disagreement on positives rose to 0.850. The encrypted path did
not improve.

## Round 24b - 2026-08-09 - four defects found by following a "cosmetic" registry mismatch

Verified on https://<demo-host> after each deploy, not locally.

| # | Finding | Where it was visible | Status |
|---|---|---|---|
| 1 | Finance gap note still blamed 4-hop truncation, which the source paper caps at ~4% | live on the public demo | corrected at the trainer and propagated |
| 2 | Satellite headline F0.5 computed over **3** labelled events, shown beside a published 0.786 | model card + panel | event count recorded and rendered |
| 3 | `both_scored_zero` guard too narrow; negative delta read as the encrypted student beating its teacher | satellite student panel | condition generalised to `teacher_scored_zero` |
| 4 | Medical CKKS student was a **constant predictor** (AUC 0.354, below random) | hidden by accuracy 0.7308 | class weighting + AUC selection + folded cut |
| 5 | BreastMNIST class labels **inverted** in 4 places; demo showed the opposite diagnosis | published weights + browser | read from MedMNIST metadata |

Browser-verified after deploy:

- finance: gap note renders the corrected architecture explanation, no 4-hop claim
- satellite: `event_wise_f0_5 0.4545 baseline 0.786 . over 3 labelled events - too few to compare`
- satellite student: the teacher-scored-zero caveat renders and says a negative delta does NOT mean
  the student outperformed its teacher
- medical student: `auc 0.6092 baseline 0.901`, `n_classes_predicted 2`, `majority_class_rate 0.7308`,
  and the metric-mismatch caveat
- medical run: "Predicted normal, benign (true normal, benign)" - correct label, matching truth
- all 5 use-case routes 200; all 9 published models pinned and PUBLIC before and after

Numbers that moved:

| model | before | after | note |
|---|---|---|---|
| medical student AUC | 0.3544 | **0.7882** | was below random; 0.6092 was an intermediate value, superseded twice since |
| medical student malignant recall | 1/42 | **24/42** | |
| medical student accuracy | 0.7308 | **0.7692** | now ABOVE the 0.7308 majority rate |
| finance student agreement | 0.8841 | 0.9471 | fidelity, not discrimination |
| finance student disagreement on positives | 0.6433 | 0.3262 | |
| resnet50 AUC | 0.7988 | 0.7988 | unchanged - see the retraction below |
| resnet18 AUC | 0.9147 | 0.9147 | unchanged - see the retraction below |

### Retraction within the same round: those two AUCs did not move, and I moved them

Mid-round I reported resnet18 falling 0.9147 -> 0.8779 and resnet50 rising 0.7988 -> 0.8876 on
reruns "with no functional change", concluded it was seed variance, measured a 6-seed spread
(sd 0.0198) and wrote that spread into the README, the model card and this log.

**The initialisation changed, not the seed.** `scripts/train/medical_breastmnist.py` defaults to
ImageNet1K_V1 and takes `--from-scratch` to disable it. I reran without the flag. Re-running WITH it
reproduces the documented figures to four decimal places - 0.914683 and 0.798768 against 0.9147 and
0.7988 - so the documented numbers were right the whole time and the "variance" was my own change of
experiment.

Three things this invalidated, all now reverted: the README's claim that the ResNet-50/ResNet-18
ordering is unresolvable (from-scratch reproduces the published ordering cleanly), the same claim in
`deliverable-verification.md`, and the seed-spread figure quoted as if it described the shipped
model. The 6-seed measurement was real but was taken on the ImageNet-pretrained variant, which is
not what ships.

**It also exposed a live licence divergence that predates this round.** DATA_LICENSES.md and the
README both state `medical_fl.resnet18` is trained from random initialisation, *because* ImageNet's
terms are uncleared for redistribution here. The artefact published on the Hub recorded
`pretrained: ImageNet1K_V1` - and so did the previous revision, `c660dc564078`, from before this
session. So the project had been publishing ImageNet-derived weights while documenting that it does
not, and nothing checked. Both teachers are now retrained from random initialisation and
republished, and `scripts/publish_hf.py` refuses to upload any publishable-dataset model whose
manifest records a pretrained source - verified to fire on an ImageNet manifest and pass on a
from-scratch one.

The lesson is narrower than "check your flags". A default that contradicts a licence commitment will
eventually be taken, and the only durable fix is a machine check at the point of publication.


## Rounds 29-31, added in Round 32 - and why they were missing

`deliverable-verification.md` states that gate-4 results are recorded here. They were not: the last
entry above was Round 28, so **three rounds of browser verification existed only in commit messages and
the ledger** while a published document claimed to hold them. The register of results needs the same
staleness check as any other document, which is the lesson this file's own Round 24b entry draws about a
different file.

Written from the commits and the measurements taken at the time, and labelled as reconstructed rather
than presented as if logged live.

### Round 29 (merged `932a3fb`) - two run-integrity defects, found by clicking rather than reading

- **Restart mid-run overwrote the new run with the old one.** The superseded run's async callbacks kept
  writing after Restart, so the metrics table showed the previous run's numbers under the new run's
  header. Fixed with a run token that makes a superseded run inert; `usePipelineRun` now discards events
  whose token is stale.
- **`config.usecase` was read but never emitted.** `clientCompute` chose its channel model with
  `config?.usecase === "satellite_comms"`, and `buildConfig` never put `usecase` on the object - so the
  key was permanently `undefined`, the ternary permanently took the fibre branch, and the satellite page
  printed a decoy-state tagged fraction of **7.26** computed over 50 km of FIBRE, against roughly 125 for
  its own free-space downlink, labelled `measured`. `tests/test_run_config_keys.py` now compares the keys
  the run reads against the keys it emits.

### Round 30 (merged `69314f4`) - a licence condition only the Hub could fix

- The Hub served a **retracted attribution** at the pinned revision: `eval_sample.json` for
  `qsh-satellite-opssat-ad` said `Kotowski et al` where the corrected local card said `Ruszczak`, and
  `clientCompute` interpolated that Hub-fetched string verbatim into the run log. OPS-SAT-AD is CC BY 4.0,
  so correct attribution is a licence condition and **no local fix could reach that surface**. All ten
  repos were re-uploaded and re-pinned; verified afterwards on the live site, which printed `Ruszczak`.
- The novelty claim was restated as **the composition** after an adversarial prior-art search found
  browser MPC has prior art (webSPDZ) while browser CKKS *evaluation* did not.
- WS4(c) was claimed with a Clifford demo that had been in the tree, uncited, for several rounds.

### Round 31 (merged `cc4a151`) - three defects found by walking the deploy

Each of these was invisible in the source and visible in the browser, which is the argument for gate 4
existing at all.

- **The genomics baseline had nowhere to render.** The `MetricSpec` rename to `auroc_eur` was correct and
  no achieved record emits that key, because a reference panel has no phenotype - so the published 0.633
  appeared only inside a provenance paragraph. The number a visitor is most likely to quote was the
  hardest one on the page to find. Now its own row, labelled "published baseline, NOT measured here".
- **Satellite's anomaly verdict and its whole attribution panel were unreachable.** Both sat inside
  `if (!abort)` in the `qsdc_telemetry` stage, and QSDC essentially never delivers: measured secrecy
  capacity **-51.58 bit/carrier at the default 800 km** (24.5 dB, $`g = 280`$) and still **-2.15 at 200 km**
  (12.4 dB, $`g = 17.5`$), the shortest downlink the panel offers. A page advertised a published model whose
  output no user action could produce. The abort is correct physics and is unchanged; the false dependency
  is gone.
- **The genomics 2D overlay still described "the 77 variants of PGS000001, 64 resolved"** after the page
  had moved to PGS000004. A stale overlay renders perfectly. Now 307 of 313, by GRCh38 coordinate.

Also verified live that round: telecom's RandomForest attribution renders with `dark_count_rate` drawn as
no measured effect ($`+0.0001 \pm 0.0010`$); the XAI paragraph says "no encrypted inference stage" on telecom
and satellite and keeps the original caveat on medical; `crossOriginIsolated === true`; zero `/api`
requests against a 404 `text/plain` control.

## Round 32 (merged `4ad9b4e`), added in Round 33 - and the round that fixed this file left a fourth gap

**Round 32 reconstructed Rounds 29-31 here because this file was three rounds stale, and then recorded
no entry of its own.** Grepping this file for "Round 32" before this entry matched exactly one line: the
title of the block above, which is about Rounds 29-31. So the round whose finding was "the register of
results needs the same staleness check as any other document" **demonstrated the finding on itself in the
same commit** - it closed a three-round gap and opened a one-round gap, and the one-round gap is the same
defect at an earlier stage.

Reconstructed from the commits and the measurements taken at the time, and **labelled as reconstructed**
rather than presented as if logged live - same convention as the block above.

**The measurement this file is the recorded home for, and which existed only in a commit message.**
`check:pqc` was broken on purpose five ways after being promoted to a gate, and the result is stated in
commit `3de8599`:

| case | exit | note |
|---|---|---|
| clean vendored tree | **0** | 2,595 malformed keys, all rejected |
| empty vector file | **1** | the hole this closed: it used to exit 0 with "0 malformed keys, 0 rejected, 0 accepted OK" |
| truncated gzip | **1** | |
| tampered gzip | **1** | caught by the SHA-256, which runs before `gunzipSync` |
| missing vector file | **1** | "This is NOT a pass. The conformance check did not run." |

**The hash fires first on the vendored path, so the count assertion is the second line of defence rather
than the first.** That ordering is why the *earlier* record of "779 rejected, 1 accepted when a
well-formed key is smuggled into the malformed set" can no longer be produced: any modification to the
vendored gzip now exits at the hash check with a different message. That correction is written up in
`deliverable-verification.md`; the five-way result is transcribed here because a measurement that lives
only in a commit message is not a record. **From the tree, a commit describing work that happened and a
commit describing work that did not look identical.**

**Two run-level defects found by running rather than reading:**

- **The PRS313 switch had missed the E2E pipeline**, so Python scored 77 rsID-keyed variants while the
  browser, trainer, registry and payload generator scored 313 position-keyed ones. `prs_name =
  config.get("prs", "breast_cancer")` with nothing setting that key - a literal default that cannot fail
  and is invisible to the config dump, to per-run override and to the browser comparison. Third defect of
  that exact shape here. `tests/test_usecases.py` had a docstring naming PGS000004/313 and assertions
  asserting 77, and it passed, because the assertions agreed with the pipeline's stale default.
- **The tree-attribution sentence rendered on the four pages that do not have the attribution block, and
  never on the one that does.** It sat inside the `publishedModel` clause and telecom is precisely the page
  where `publishedModel` is falsy. The first fix nested it under `hasTreeAttribution` while leaving it
  inside `publishedModel`, which made it render **nowhere** - no longer false and no longer useful. Found
  by loading the page twice; a diff review passed both attempts.

**Two measurements that changed a claim rather than confirming one:**

- **A `node-seal` SEAL bump is available**, after several rounds of recording node-seal as "the
  constraint" without anyone trying it. The fork's divergence from SEAL 4.1.2 is purely additive - about
  1,005 lines of Emscripten embind and CMake glue, no crypto source touched. SEAL v4.4.3 plus that binding
  layer built to WASM with **zero errors**, producing `libseal-4.4.a`. The blocker on earlier attempts was
  a deprecated emcc macro (`defines.h` tests bare `EMSCRIPTEN`; emcc 6.0.6 defines only `__EMSCRIPTEN__`),
  not SEAL. **Not verified and so not claimed:** a CKKS round trip through the linked module.
- **SEAL 4.4.0 is upstream-labelled a CRITICAL security update**, not the "Major security update, no
  further detail" that the README and the ledger both carried. That is true of the GitHub release body and
  false of `CHANGES.md`, read at the primary source. Both documents were understating a critical advisory
  in a dependency the browser demo ships, while the technical report - the document an audit had flagged
  as overclaiming - had it right. **A release body is not a changelog.**

## Round 33 (`66df224`, `df7c847` on this branch) - three gates that could not see what they assert

Recorded from the two commits on this branch, which are done and committed. **Where this entry describes a
browser observation rather than a gate run it says so**, because most of this round's findings were found
by running gates and by reading generated artifacts, not by walking a deploy.

**1. The report contradicted its own title page.** The title page, the abstract and a section pointer said
"eight defect classes" while `\ndefectclasses{}` and the generated findings table say **19**; the abstract
said "three" retracted conclusions while the Summary and the section itself say **four**. Round 32's final
commit was literally "rebuild the report", so a regenerated PDF shipped over a title page disagreeing with
its own contents. **All counts now come from the macro**, and "and four others" became "among others" so no
arithmetic is hardcoded to go stale. The generated body was right every time; the hand-written front
matter was wrong - which is the argument for the next item.

**2. A new CI gate, 3/9, that reads the generator rather than the artifact.**
`scripts/check_generated_prose.py` parses each trainer's prose literal **out of the AST** - no model runs,
so it costs seconds - and diffs it against `models/metrics.json`. The reason it had to exist: the
config-parity gate (2/9) compares two **derived** artifacts to each other, so it is blind by construction
when both agree and neither matches the code that writes them. It reports statically-unreadable fields as
a printed count rather than hiding them, so a note that moves into an f-string leaves coverage visibly.

Three instances on its first run:

| found | what it was |
|---|---|
| `medical_fl.student.baseline_gap_note` | Round 32 judged it an overclaim and corrected it **by hand** in both `models/metrics.json` and `config.defaults.json`, leaving `scripts/train/medical_breastmnist.py` emitting the original. The next retrain would have reverted the correction silently, `publish_hf.py` would have pushed the reverted sentence to a public model card, and the ledger entry recording the correction would have become false with nobody editing it |
| `finance_aml.scorecard` | `finance_scorecard.py` emitted the key and `metrics.json` had no such record, while `references/he-scheme-selection.md` published its numbers **by hand**. Trainer run on CPU, 272.3 s: **net_benefit 0.00030763** on 1,523,504 test rows, 12 non-zero coefficients, `exact_under_bfv` true - confirming the hand-written +0.000308 from a machine record for the first time, and giving the decision curve real inputs via `net_benefit_threshold` and `net_benefit_treat_all` |
| `satellite_comms.telemanom` | the trainer emitted a `baseline_gap_note` no artifact carried, so a careful explanation of why `n_events_scored` is 3 rather than 320 reached no reader. Copied via AST, so the two are byte-identical by construction rather than by transcription - which is exactly how the medical note diverged |

**3. A Japanese-named local file in the repo root passed every confidentiality gate - measured, not
hypothesised.** Copied in and walked through all three: not matched by any ignore pattern, so
not ignored, staged by `git add` without complaint, `check-secrets` printed OK, and the no-Japanese
content gate scanned a **directory list that excluded root-level `.md` other than README**. Fixed:
`check-secrets` gains a CJK-filename check and the checklist path, and the content gate now derives its
scope from `git ls-files`, which covers the root, covers any new directory without editing the line, and
makes "public" mean exactly "tracked". **A first attempt at that failed silently** because `git ls-files`
octal-escapes non-ASCII paths, so the codepoints never reached `grep` - validated in both directions
afterwards. Also corrected: `.gitignore`'s claim that `*.docx` is "strictly stronger: it catches every one
of them" is false, because three of those notes are `.md`.

**4. `hasTreeAttribution` was described as one shared predicate and was not.** Round 32's comment said
"ONE predicate, shared by" the renderer and the pointer sentence. The predicate tested only **permutation
importance** while `AttributionRows` required **both** tree measures, so a record carrying one would print
"does carry attribution" above a block that rendered nothing. **That is the Round 31 defect coming back
through a different door, inside the fix written to remove it** - the second time in two rounds that
closing an item is where the next defect was made. Now a single `recordHasTreeAttribution` used by both.

**5. The PQC market figure, and the result inverted the expectation.** The **unsourced** README figure was
the correct one - one house's current edition - and the **sourced** Value-page row was a blend. Its base
was one house's 2026 value, its endpoint spliced that house's 2033 figure to a second house's 2034 one,
and its growth-rate range took its high end from **a third house the row does not name, whose base for
the same market was a quarter of the one the same row quoted.** [The figures were removed from this
entry on 2026-10-24, with the market table and the README sentence it describes: this project's public
material carries no market-size figures.] The
footnote two rows below already describes this exact error ("producing a range no single analyst
projects") about a *different* row: the lesson had been learned for one row and not applied to the row
above it. The PQC row now lists each house separately, states the 4x scope spread rather than averaging
it, and the README points at the table instead of carrying a second copy. (Both removed 2026-10-24.)

**Also in `df7c847`, each the same class - a published surface asserting something the tree contradicts:**
the README described a serving ZeroGPU third tier "awaiting a caller" whose Space returns HTTP 401 and
whose `remote` badge state was deleted from the type in Round 24d, against a ledger that records the tier
as DECIDED against; `threat-model.md` said the provenance badge has five states twenty lines after the same
document says four; `DATA_LICENSES.md` listed a gitignored file under "committed"; three `REFERENCES.md`
entries named `models/registry.py` as citing IACR 2026/935, which Round 31 had removed; and the finance
chart rendered **two different base rates in one figcaption** - a computed 0.020020 beside a hardcoded
0.0173 with a 13.3x lift. Both now derive from one `_base_rate()` helper, and the true lift is **11.5x**.

**Verified live, 2026-08-12.** The line here previously said nothing in this entry was a browser
measurement. That was true when written and is superseded for the rows below; the remainder of the entry
is still local-only and the distinction is kept rather than smoothed over.

Deployed from git on the demo host - branch pulled, rebuilt with the behind-proxy override and the demo
project name - with CI green on the same commit first. The served bundle moved from `index-Bmsp8Rv3.js`
to **`index-Chd2NwZu.js`** (1,102,074 bytes), which is how the deploy was confirmed fresh rather than by
trusting the container's own reported status.

| check | observed |
|---|---|
| Telecom encrypted tier runs in the tab | `encrypted tier (REAL CKKS, 64 params, 8 dot products): class "intercept_resend" in 112.2 ms` |
| Top-1 mismatch count (arXiv:2605.22237 convention) | **0 of 1** scored inputs |
| Precision in bits (the OpenFHE and HElib unit) | max abs error **4.294e-9**, about **27 bits**, labelled MAX not mean |
| The argmax disclosure fires | `the ARGMAX is NOT encrypted: all 8 class scores are decrypted and the maximum taken in the clear` |
| No silent fallback | no "skipped" or "weights unavailable" line present |
| `crossOriginIsolated` | `true` |
| Server compute | **zero** `/api/` requests; the page contacted only its own origin |
| Bundled student weights served | `data/telecom/student_weights.json`, HTTP 200, `application/json`, 3,482 bytes, W 8x7, b 8, 64 parameters |
| The withdrawn TSB-AD figure | **0** occurrences of "26th of 32" in the served bundle |
| Registry-driven labels before any run | the "runs under CKKS" tag and "CKKS-linear IDS student" both render |

**A verification method that does not work, found by walking into it.** `curl` returning
`200 text/html` for a path proves nothing about whether that route exists. The SPA fallback serves
`200 text/html` for **every** path, so the content-type discriminator this file relies on separates an
absent API from a present SPA and does **not** separate a valid route from an invalid one. Measured:
`/use-cases/telecom_ids` returned `200 text/html` to `curl` and rendered **"Page not found"** in the
browser - the real route is `/usecase/<id>`, singular. Only rendered content tells the two apart, so a
route check that never reads the DOM is not a route check. `RS2`'s nonsense-path control remains correct
for what it tests; this is a second and different confusion, and it needs its own row.

## Round 34 - OpenFHE bootstrapping was verified nowhere, behind two green signals

Not a browser observation: everything below was measured by running the code on the EL9 dev host
(Linux 5.14.0-611.5.1.el9_7, Python 3.12 in `.venv`, OpenFHE 1.5.1, 20 cores) and by running the
collector CI runs.

**The shape of it.** `OpenFHEBackend(use_bootstrap=True)` exited **139 (SIGSEGV, core dumped)**
while the same constructor without the flag exited 0. Two separate signals both read green over
that: locally the only test of it, `tests/test_round2.py::test_openfhe_bootstrap`, could skip for
two different reasons, and on CI it never ran at all. So the feature README line 514 advertises had
**no passing execution anywhere** - and the crash was not even reachable by the skip logic, because
a SIGSEGV is not an exception and `except ImportError` never sees it.

**The remedy named in the source was already applied, and was not the problem.**
`requirements-fhe.txt:12` says to fix this with `bash env/build_openfhe_el9.sh`. Checked before
running it: the installed extension's highest required symbol is **`GLIBCXX_3.4.29`** (EL9's
ceiling - the stock Ubuntu-24.04 wheel needs `3.4.32`), and `openfhe.cpython-312-*.so` plus
`lib/libOPENFHE*.so.1.5.1` are dated **13:56** against the wheel's own `__init__.py` at **13:46**.
That is the build script's output, already in place. `import openfhe` and a full CKKS
encrypt/multiply/decrypt both worked before any change. **The libstdc++ story was true history and
the wrong diagnosis for this bug** - re-running the rebuild would have changed nothing, and
"the documented fix is already applied" is the thing worth writing down.

**The actual cause: two crypto-context parameters, both of which must be set before
`GenCryptoContext`.** Located by `faulthandler`, which put the crash on the
`cc.EvalBootstrapSetup([3, 3])` line, after `Enable(FHE)` had already succeeded.

| parameter | what was set | what bootstrapping needs | symptom when wrong |
|---|---|---|---|
| multiplicative depth | `4` (`config/defaults.py:152`) | `GetBootstrapDepth([3,3], UNIFORM_TERNARY)` = **20**, plus the levels you want afterwards | **SIGSEGV** - walks off the modulus chain |
| scaling technique | CKKS default (`FIXEDAUTO`) | **`FLEXIBLEAUTO`** | `RuntimeError: Degree [10] must be less than or equal to the correction factor [9]` |

`FLEXIBLEAUTOEXT` fails the same way as the default. Depth is now computed as
`mult_depth + GetBootstrapDepth(...)` = **24**, and both settings are applied only on the
`use_bootstrap=True` path, so the plain CKKS context is byte-for-byte what it was.

**What a passing run now measures**, at ring dimension **65536** (OpenFHE's standard 128-bit
security level, not a reduced toy parameter set): context and keys in **3.0 s**, one
`EvalBootstrap` in **7.7-9.1 s**. A ciphertext is driven down to level **22** by repeated
multiplication by 1.0 - which spends a level and leaves the plaintext alone - and decrypts there as
`0.4999999999923404`. After bootstrapping it sits at level **19** (**3 levels regained**) and
decrypts as `0.5023873687017761`: bootstrapping is approximate, and that ~2-decimal error is the
evidence it actually ran rather than returning its input. One further multiply then succeeds,
which is the part that proves the level budget was really restored and not just relabelled.

**Bootstrapping only pays above level 20 here, and cannot run at all at level 23.** `EvalBootstrap`
returns level 19 whatever it is handed, so on a fresher ciphertext it is an expensive no-op; and on
the last RNS tower it raises `DropLastElement(): Removing last element of DCRTPoly renders it
invalid`. Both are now in the `bootstrap()` docstring, because both look like bugs from the outside.

**What CI was actually doing - measured, not inferred.** `.github/workflows/ci.yml` runs
`pytest -m "not slow" -q`, and the test carries `@pytest.mark.slow`. For `tests/test_round2.py`:

```
10/11 tests collected (1 deselected) in 0.21s
```

`test_openfhe_bootstrap` is absent from the collected node ids. It was **deselected, never skipped**
- CI has not executed a line of it. A new CI step now runs exactly that test with
`QSH_REQUIRE_OPENFHE_BOOTSTRAP=1`, and the old step carries a comment saying what it does not cover,
so a green tick on the main suite is no longer readable as FHE coverage.

**Three guards, each verified by breaking it on purpose.** A test that can report success without
running is the defect class here, so none of these was accepted on the strength of its code:

| guard | how it was proven | observed |
|---|---|---|
| Native crash must be a red test, not a dead suite | reintroduced the depth bug (`+ boot_depth * 0`) and ran the suite | `Failed: ... crashed the interpreter: SIGSEGV (-11)`, pytest exit **1** - previously exit **139**, which took the other 290 tests with it |
| A skip must not cover a working host | a stub test skipping with the old `"bootstrap not available"` text, on this host where `openfhe` imports | banner `UNJUSTIFIED SKIP - failing the session`, exit **1** despite the summary reading `1 passed, 2 skipped`; an unrelated skip in the same run was left alone |
| Where bootstrapping is expected, even a legitimate skip is a failure | a stub `openfhe` module raising the real `GLIBCXX_3.4.32 not found` text | without the env var: skipped, quoting that error verbatim, exit 0. With `QSH_REQUIRE_OPENFHE_BOOTSTRAP=1`: **failed**, exit 1 |

The skip that remains names the host's actual error and points at `env/build_openfhe_el9.sh`; the
old one guessed ("needs Ubuntu-24.04-class libstdc++") and, on this host, guessed wrong.

**The backend no longer degrades silently.** `use_bootstrap=True` used to wrap its setup in
`except Exception: self.bootstrap_enabled = False`, so a hard failure and an absent feature produced
the identical state - and the test read that state and skipped. It now raises. `bootstrap_enabled`
is False only when bootstrapping was never asked for, which is the same no-silent-fallback rule
`get_backend` already follows.

## Round 35 (`6907585` on its branch) - four checks defined and none of them run

**Written in Round 36, from measurements taken against the Round 35 deploy.** Round 35 added `QK4`
through `QK7` to the checklist and merged without executing any of them, and without an entry here.
That is the same gap as Rounds 29-31, with the same cause: the round that writes the checks is not
the round that runs them, and nothing forces the second half.

`QK4`, `QK5` and `QK7` were run against the deployed Round 35 bundle and all three pass. The
readings, and why each is the sharp form of its claim:

| setting | row | reading |
|---|---|---|
| finance_aml, eve 1.0, misalignment 0 | QK4 | `qber` **0.25**, `secure_key_bits` 0 |
| finance_aml, misalignment 0.09 | QK5 | `qber` 0.09, `secure_key_bits` **176** |
| finance_aml, misalignment 0.10 | QK5 | `qber` 0.10, `secure_key_bits` **0** |
| satellite, 800 km | QK7 | `channel_loss_db` **24.5**, g 280.43 |
| satellite, 1 km | QK7 | `channel_loss_db` **8.0**, g 6.31 |

`QK4` is Gottesman-Lo's intercept-resend figure verbatim - 25%, not a value near it. `QK5` is the
one that needed both halves: a key at 0.09 and none at 0.10 places the zero-key point inside
(0.09, 0.10], which is the GLLP root 0.098106 and NOT the textbook 0.11, where a key would still
have been alive at 0.10. Measuring only the dead side would have been consistent with both. `QK7`'s
second row is the one that distinguishes the models: at 1 km the loss does not fall towards zero,
because the atmospheric and pointing terms do not scale with range.

**`QK6` could not be run at all**, and that is the round's finding rather than an oversight in
executing it. It asked for monotonicity in distance AND in mu; `mu` is `qkdCfg().mu_signal`, a config
constant with no control on any page. Half the check had no way to be exercised by a person. It moved
into `npm run check:qkd` in Round 36, where both axes move.

## Round 36 (`4b32b62` on its branch, bundle `index-C0as5iW-.js`) - three pages stated a security property the run did not provide

The round asked whether the checklist covers each use case's SECURITY COMMUNICATION path. The
measured answer was no, and not narrowly: `pqc_handshake`, `qkd_key_agreement`, `satellite_qkd`,
`qsmpc_compute`, `federated_training` and `_sessionKey` had **zero occurrences each** in the whole
3,584-line file. Not "no numbered rows" - the words were absent.

Going to look at the unchecked stages found three false claims on screen.

| what the page said | what the run did |
|---|---|
| `federated train + pairwise masking (demo keys)` | plaintext size-weighted mean; no masks, no seeds, no key material |
| `QSDC direct telemetry - secrecy 0 bit/carrier` with a done tick, at eve 1.0 | telemetry withheld; the log said "QSDC NOT AVAILABLE on this link" |
| `BB84 decoy-state (w/ eavesdropper)` and a party box named `Remote site / Eve` | `qkd_eve` defaults to 0.0 |

Measured on the deploy after fixing each:

| setting | reading |
|---|---|
| finance, defaults | `size-weighted FedAvg + 6 pairwise masks (Bonawitz CCS 2017), fixed-point mod 2^32 - acc 0.56 untrained -> 0.68 on 80 held-out rows` |
| medical, defaults | `acc 0.44 untrained -> 0.88 on 80 held-out rows` |
| satellite, defaults | `qsdc_secrecy_capacity` **-51.58** (was clamped to 0); step renders NOT AVAILABLE, link-budget limited |
| satellite, 1 km, 2.0+1.0 dB, eve 0 | `C_s` **+0.49**, delivered - QSDC closes on this page for the first time |
| satellite, 1 km, 2.0+1.0 dB, eve 0.40 | `C_s` **-0.63**, `eavesdrop_detected` **1** - the ABORTED branch fired for the first time |
| telecom, `qkd_eve` 0.0 / 0.35 | `BB84 decoy-state - QBER 0.01` / `BB84 decoy-state (w/ eavesdropper 0.35) - QBER 0.1` |
| finance, `aml_transactions` 4000 / 1000 | `psi_set_size` **260** / **65** - the `perParty` formula both times |
| finance, eve 1.0, misalignment 0 | `qber` **0.25**, `secure_key_bits` **0** - QK4, unchanged across both deploys |

**The clamp was hiding a model disagreement, not just a sign.** Reporting `qsdc_secrecy_capacity`
signed exposed that the same nominal 800 km downlink was 24.48 dB in the browser and 28.0 dB in the
reference implementation, which still applied `distance_km * 0.035` - the linear law the browser's
own comment records as rejected. Both sides had been reporting "0" for what were really -51.58 and
-138.93 bit/carrier.

**The federated metric was worse than doing nothing.** The bundled samples shipped 16 and 12 rows, so
the accuracy was a 4-row and a 3-row score; at five seeds each, finance read untrained 0.75 to
trained 0.25 and medical untrained 1.00 to trained 0.00, identical on all six rounds. Two causes, and
only one was size: `amount` spans 0 to 17.69 unstandardised while the feature map is
`tanh(mean sin(x + i/2))`, so it wraps `sin` nearly three times. Both files are now 400 rows,
standardised, with an 80-row holdout and a declared EDA block.

**Three of this round's own mistakes are in the checklist rather than here**, because they are
instructions for the next round: an over-broad browser assertion that reported FAIL on a clean arrow
and thereby found a second unconditional claim; a report rebuild run with its output redirected and
its exit status dropped, which is the `DP2` failure applied to this project's own generator; and a
measurement table written with row IDs in the first column three sections after writing the warning
against exactly that.

## Round 37 (`ac5d89f` on its branch, bundle `index-CLUy7CWv.js`) - private inference had never been drawn, and was measured over one row

The axis was the standing 19-bullet request, and the finding is why it keeps arriving:
`docs/deliverable-verification.md` holds a register whose own rule is that an item deferred three
rounds must be done or reclassified, and it had **no sections for Rounds 34, 35 or 36** and no entry
for any of the four longest-carried asks. The register that stops items rolling forward had stopped
being written.

**Every encrypted-versus-plaintext figure this project reported was over a single row.**
`encrypted_scores_compared` counted CLASSES; `encrypted_top1_mismatches` could only be 0 or 1. The
fix costs no homomorphic work: `encryptedLinear` padded one 7-feature input into 4096 slots and the
rotate-and-add reduction sums a window of `sumWidth`, so packing rows at that stride leaves each
row's dot product in its own slot with the loop unchanged.

Measured on the deploy, telecom:

| quantity | one row | 512 rows |
|---|---|---|
| wall clock | 107.23 ms | **110.92 ms** |
| class scores compared | 8 | **4096** |
| top-1 mismatches | 0 of 1 | **0 of 512** |
| precision | 26 bits | 24 bits |

512 of **600** held-out rows; the 88 that did not fit are disclosed in the metrics, the log and the
caption. The chart renders 512 symbols with the `y = x` reference line, zero page overflow.

Unchanged and re-confirmed on this bundle: satellite's signed `qsdc_secrecy_capacity` **-51.58** and
its `QSDC NOT AVAILABLE - link-budget limited` step; medical's federated masking at
`0.44 untrained -> 0.88 on 80 held-out rows` with 6 pairwise masks.

**The QCNN literature check, declined since Round 25, was falsifiable all along.** The stated reason -
"no published QCNN reports this circuit on this data" - is true of the circuit and false of the data:
the medical circuit trains on WDBC. Against Sci Rep 14 (2024), DOI 10.1038/s41598-024-74778-7
(QCCNN 0.96155 validation over 30 features, no test set), this project's five-seed median is 0.9070
over 8 features on a held-out test split. **The finding is not the gap**: this project's plain
logistic regression on 8 features scores 0.9651, above the paper's quantum model on 30 and well above
the paper's own logistic regression (0.91184), so the published quantum margin rests on a classical
baseline about 5 points below what the dataset supports.

Five self-inflicted defects, all found and fixed inside the round, and two of them are the interesting
ones:

- the batch scored 512 of 600 rows silently, while the function's own comment claimed "a cap is
  visible, and the caller reports how many rows it asked for". It was not, and it could not.
- the documentation-linkage check **passed locally and failed in CI**, because its haystack included
  the gitignored checklist - so a link nobody who clones the repository can follow counted as a link.
  It also searched only markdown, so two investigations reachable from `quantumSim.ts` and from model
  cards were reported as orphans. Measured properly there was **one** genuine orphan, not three.
- the agreement chart rendered with a NULL accessible name, because `<title>` was written as a child
  of `ScatterChart` rather than passed as the `title` prop every other chart in the file uses - the
  same regression Round 34 recorded. Its axis ticks also rendered as sixteen-digit floats. Neither is
  visible to tsc, the build, or any assertion in the suite; both were found by looking at the page.

  **Both fixes were re-verified on a fresh deploy rather than assumed**, because the defects were
  found on bundle `index-J5Enwme_.js` and the fix landed two commits later: the box was still serving
  the broken build when the fix was committed. On `index-CLUy7CWv.js` the figure's accessible name
  reads *"Encrypted score against plaintext score, 512 held-out rows"* and the ticks read
  `3.73 / 8.73 / 13.73 / 21.79`. Claiming a UI fix without redeploying is the same error as reading a
  metric before the state settles.

## Round 38 - the page encrypted one model and explained another, and the plan's own metric was 50x wrong

Host: Chrome 149.0.0.0, Linux x86_64, 20 logical cores, `crossOriginIsolated` true. Measured on a
local production build before deploy, then re-verified on the deployed bundle.

### What was encrypted, before and after

Finance and genomics reported a class from the bundled real model, drew SHAP bars labelled "real
bundled model", and computed every `encrypted_*` / `fhe_*` figure from the demo pattern
`((i % 3) - 1) * 0.15`. `realClientHybrid` took no model argument, so it could not have done
otherwise - while `privateInference`'s own comment three lines away asserted the real model "is what
the SHAP attribution and the CKKS evaluation both consume".

| finance | deployed `main` (before) | local build (after) |
|---|---|---|
| `fhe_model` | absent | bundled real model |
| `encrypted_scores_compared` | 2 (one row) | 400 rows scored |
| `fhe_max_abs_error` | 6.20e-13 | 1.93e-9 |
| `encrypted_precision_bits` | 40 | 28 |
| `encrypted_agreement` | 1 over 1 row | 1 over 400 rows |

The precision drop is expected and is the honest cost: the real surrogate's weights are orders of
magnitude larger than the demo pattern's, so the encoded norm is larger. The old numbers described a
weight vector that appeared nowhere else on the page.

Genomics: 30 rows, `encrypted_agreement` ranging 0.83-0.97 across runs - and the corpus does NOT
vary. `realInputCorpus("genomics")` returns all 30 `dosage_samples` deterministically and the model
is the fixed top-8 PGS weights, so the same page runs the same 30 rows against the same model every
time. **Seven of those 30 individuals carry no effect allele at any of the eight loci**, so their
plaintext score is exactly 0 on both classes and each flips on ~1e-9 of encryption noise,
independently, per run. The headline rate is therefore 23% coin toss. Counted from the shipped
`pgs000004_prs313.json`, not inferred.

### Three defects found by verifying that change

- **`encrypted_agreement` was written twice into one stage.** The batched figure emitted a rate over
  30 rows; the single-row path then emitted `mism === 0 ? 1 : 0`, and won. Finance hid it - both read
  1 - and genomics exposed it, printing `encrypted_agreement 1` beside
  `encrypted_top1_mismatches_batch 3` of 30. `metric()` now records any key rewritten with a
  different value as `metric_overwritten`; **seeding the defect back produced
  `encrypted_agreement: 0.83 -> 1`**, so the guard is known to fire.
- **A mismatch count without a margin points the wrong way.** Genomics' model is the published PGS
  weights as `[-w/2, +w/2]` with zero bias, so its two logits are exact negatives and the decision
  margin IS the polygenic score. `encrypted_flip_margin_max` measured **exactly 0** at the flipped
  rows: those samples carry no effect allele at any of the top-8 loci, so the plaintext model's own
  top-1 comes from argmax tie-breaking and any deviation flips it. "3 of 30 mismatches" beside a
  7.2e-10 deviation reads as CKKS being unreliable, which is the opposite of the measurement.
- **A caption whose scope was narrower than its sentence.** "worst absolute deviation 1.80e-8"
  printed immediately after "(800 class scores)" was a maximum over the 400 plotted winning-class
  scores; the metric row beside it read 2.47e-8 over all 800. Both correct, 37% apart.

### Per-rotation cost, which is the comparable axis, and the plan's error

`fhe_linear_ms` is the measurement multiplied by `benchScale()`; `fhe_dense_rotations` is a raw
count. The division this round's plan proposed reads **159.9 ms per rotation on genomics against a
measured 3.18**. `rotateVector` is now timed directly.

| page | rotations | rotate ms | ms/rotation |
|---|---|---|---|
| finance | 3 | 9.29 | 3.10 |
| medical, bundled linear head | 3 | 13.17 | 4.39 |
| medical, student convolution | 16 | 67.81 | 4.24 |
| medical, student dense layers | 136 | 421.40 | 3.10 |
| telecom, 512-row batch | 24 | - | 3.17 |
| genomics | 6 | 19.07 | 3.18 |

Published comparator, fetched and read directly: node-seal `BENCHMARK.md` (2025-03-27, M1 Pro),
CKKS `EvaluateRotate` at n=8192/log(q)=218 reads **6830 us in Chrome**. Same order, this host
roughly 2.2x faster. The n=16384 row is at log(q)=438 against 200 bits here and is recorded as NOT
comparable.

**A claim first refuted and then restored.** The round reported rotations at 57% of the dense layer
and 10% of the convolution, and rewrote three comments around it. The 10% used the whole student's
wall-clock as the convolution's denominator. Re-measured with the scopes separated:

| stage | rotations | rotate ms | stage ms | share |
|---|---|---|---|---|
| bundled linear model | 3 | 13.17 | 23.2 | 57% |
| student, convolution | 16 | 67.81 | 96.60 | **70%** |
| student, dense layers | 136 | 421.40 | 490.23 | **86%** |

96.60 + 490.23 = 586.83, the student's total, with nothing unattributed. Key-switching dominates
every path; the original comments were right and are restored.

### Ring degree 16384: measured and reverted

| | 8192 | 16384 |
|---|---|---|
| keygen (first click) | 255.26 ms | 476.71 ms | **UNCONTROLLED PAIR - superseded.** These two came from different sessions. Re-measured back to back: 253.41/255.46 at 8192 and 490.59/492.15 at 16384, under 1% inside each ring and 1.93x between them |
| ms per rotation | 3.098 | 6.725 |
| linear layer, measured | 19.25 ms | 34.41 ms |
| batch, 400 rows | 20.59 ms | 36.16 ms |
| batch precision | 25 bits | 24 bits |
| rows per ciphertext | 512 | 1024 |
| medical conv patches packed | 196 (one image) | **196 (one image)** |

Every published FHE figure roughly doubled; the realised benefit was zero. The convolution still
packed one image because no caller hands it two, and the extra batch rows went unused because finance
offers 400 and genomics 30. Reverted at the user's direction; the config wiring stayed.

**The first attempt at the change was inert and reported success.** Editing
`src/qsh/config/defaults.py` alone changed nothing, because `load_config()` overlays `config.yaml`.
The export script rewrote the JSON with the old value and exited 0, `report/build_data.py`
regenerated the crypto table with the old value and exited 0, and the bundle rebuilt with the old
value. Caught by reading the generated output, not by any signal.

### Operational window

`client_compute_ms` reading 0.07 ms directly under "total: 700 ms / inference", labelled "client
compute (wall-clock)", timed only the plaintext reference call. `encrypted_tier_measured_ms` is the
measured sum: **finance 700 ms**, **medical 1,269 ms**. The medical figure is the whole encrypted student, not the convolution - 579 ms was `fhe_encrypted_student_ms`, and this sentence called it the convolution until Round 39. The per-stage split, measured after the scopes were separated, is in the table below: convolution 84.35 ms, dense layers 506.69 ms.
Both inside the 1-10 s interactive-batch window, and the page now states the verdict rather than
leaving two numbers to be compared.

### WS2

`ids_classes_scored` renders (8), the headline sentence quotes it instead of a literal, and
`ids_metric_roles` marks which rows lead - this run shows binary F1 1.00 beside a worst class of
0.87, which is the gap the marker exists for. `quantum_beats_classical` reaches the DOM for the first
time, derived from the macro F1 values in the same table and cross-checked against the stored flag:
*"the classical baseline (0.9866) is ahead of the best quantum macro F1 here (0.8316)"*.

### Confirmed on the deployed site, not only locally

Deployed from GitHub - fetch, Docker build (exit status captured, 0), `up -d` with the shared-proxy
override - and re-verified in Chrome on the public URL. Container age 6 s at first read; bundle
`index-pejoV-ed.js`, changed from Round 37's `index-CLUy7CWv.js`.

| page | what the deployed run shows |
|---|---|
| finance | `fhe_model` bundled real model; 400 rows scored; `encrypted_agreement` 1 over 400; 3.227 ms/rotation; `encrypted_tier_measured_ms` 682 ms |
| genomics | `encrypted_agreement` **0.83** over 30 rows with 5 mismatches - the value the old double-write would have replaced with 1; `encrypted_flip_margin_max` exactly 0; caption states the ties |
| medical | `fhe_conv_rotation_share` **0.10 - WITHDRAWN**, see the correction below: that share divided the convolution's rotation time by the whole student's wall-clock. Measured with the scopes separated it is 0.67-0.70. Tier 1,280 ms; 3.04 ms/rotation |
| telecom | `ids_classes_scored` 8; `ids_metric_roles` renders; `quantum_beats_classical` reads *"the classical baseline (0.9866) is ahead of the best quantum macro F1 here (0.8316)"*; 3.146 ms/rotation over 512 rows |
| satellite | unchanged, no collision |

`PF5` checked on the live page by running twice in one tab: `fhe_keygen_ms` **absent** on the second
run, with per-rotation 3.227 then 3.213 - 0.4% apart, which is the stability the comparison rests on.
Across the four deployed encrypted pages: **3.04, 3.058, 3.146, 3.227 ms/rotation**.

The keygen figure on that deployed run read 470.39 ms at ring 8192, against 242-255 ms measured
locally at the same ring. Those are not comparable: the deployed reading was taken with a build, a
preview server and a dozen tabs competing for the machine. Because the ring experiment's keygen
delta rested on two numbers from different sessions, it was **re-measured back to back under one set
of conditions**: ring 8192 gave 253.41 and 255.46 ms, ring 16384 gave 490.59 and 492.15 ms - within
1% inside each ring and **1.93x between them**, so the delta is real and the earlier pairing was the
sloppy part, not the conclusion.

No `metric_overwritten` on any page. No `NaN`, `undefined` or `[object Object]`. No horizontal
overflow.

### Re-verified on the deployed bundle after the adversarial review

Redeployed from GitHub at `d8bf7ac` (build exit 0, container age 6 s, bundle `index-BhIF87WC.js`).

Genomics, live: `encrypted_agreement` **0.87** this run against **0.83** on the previous deployed
run - the same 30 fixed rows and the same model, moving by chance - while
`encrypted_agreement_over_decidable` reads **1.00 over 23 rows** on both. That contrast is the
finding: the headline is 23% coin toss and the decidable rate is stable and perfect.

Medical, live, with the scopes separated:

| stage | rotations | rotate ms | stage ms | share |
|---|---|---|---|---|
| student, convolution | 16 | 56.28 | 84.35 | **0.67** |
| student, dense layers | 136 | 440.34 | 506.69 | **0.87** |

84.35 + 506.69 = 591.04, which is `fhe_encrypted_student_ms` exactly. Compare the figure this round
published before the review: a 10% convolution rotation share, computed by dividing the first row's
rotate time by the last row's total.

No `metric_overwritten`, no `NaN`/`undefined`/`[object Object]`, no horizontal overflow.

### Quantum resistance per use case, measured because the standing request asks for it

The security-communication rows (`SC1`-`SC14`) establish what produces key material and what does
not. None of them asserts the STRENGTH in force on the wire. Measured on the deployed site, all five
pages: `ml-kem-768+x25519 - NIST Category 3` and `ML-DSA-65 - NIST Category 3`, with
`kem_ciphertext_bytes` 1088, `signature_bytes` 3309, `classical_public_key_bytes` 32 and
`signature_verified` 1 - identical on finance, medical, genomics, telecom and satellite.

**The crypto-agility selection genuinely reaches the algorithm.** Switching genomics to
`ml-kem-1024+x25519` and `ML-DSA-87` moves the ciphertext to **1568 bytes** and the signature to
**4627 bytes**, with `signature_verified` still 1 - the FIPS 203 and FIPS 204 values for those
parameter sets, so the larger key was generated, used and verified rather than relabelled. This
project has shipped a control whose label moved while nothing behind it did, so the check is
worth having and the result is worth recording: `QS1`-`QS4`. The FIPS sizes themselves are asserted by `npm run check:pqc` (`PQ1`-`PQ3`) and are deliberately not restated as browser rows.

Scope, stated: `QS2` was exercised on genomics only. The handshake stage is shared across all five,
which is a reason to expect the same behaviour elsewhere and not a measurement of it.

### The corrections were themselves reviewed, and four of them were wrong

A second adversarial pass, scoped to the correction commits only. Four survived, all mine.

**The run log still printed the withdrawn attribution.** The `10%` fix changed the `metric()` calls
and left the `emit("log")` forty lines below untouched, so the medical page's event log read
"16 rotations and 4 multiply-plain in 591.04 ms" - the conv block's rotation count over the whole
student's wall-clock, 36.9 ms per rotation - while the metrics table on the same page read 84.35 ms
and 3.7. It also labelled `real.fheMs - real.conv.convMs`, which is this page's own 8-feature bundled
linear demo at ~18 ms, as "dense layer", while the student's dense stack cost ~499. Every term is now
scoped: measured after the fix, the log reads 75.05 ms for the convolution and 498.59 ms for the
student's 136 dense-layer rotations, and 75.05 + 498.59 = 573.64 is `fhe_encrypted_student_ms`
exactly.

**`fhe_conv_max_abs_error` was the student's final-logit error on the published path** and the conv
block's on the synthetic path - one key, two meanings, decided by whether a Hub fetch succeeded, under
a comment claiming the two errors were reported separately "so a regression in one cannot hide behind
the other". Now 4.25e-9 for the convolution and 6.87e-9 for the student's logits, as two rows.

**The replacement tie test could not fail either.** `margin / scoreRange <= 1e-6` looked independent
of the deviation and is not: a flip forces $`\text{margin} \le 2E`$, so $`\text{rel} \le 2E/\text{range}`$, and with $`E \sim 10^{-9}`$
against a range of order 1 that is ~1e-9 - three orders below the threshold, on every run. It would
take a 500x precision collapse to reach the other branch, and that case is already reported by
`encrypted_batch_precision_bits`. **And the experiment that "proved" it reachable forced
`flipMarginRel` to a literal**, which exercised the rendering branch rather than the predicate - the
wrong thing, convincingly. The boolean is removed rather than re-tuned; `encrypted_undecidable_rows`
carries what it pretended to and does vary (7 of 30 on genomics, 0 of 512 on telecom).

**"3.2-4.2 ms throughout" excluded both of its own endpoints.** The stages it summarised measure 4.39,
4.24 and 3.10; the interval contained neither extreme, and the same string sat in two code comments.
Now 3.10-4.39.

#### Confirmed on the deploy, and one thing splitting the error key revealed

Redeployed at `2393c79`. Medical, live: the event log reads "16 rotations and 4 multiply-plain in
**77.67 ms**; the student's 136 dense-layer rotations take **494.04 ms** more. This page's own bundled
linear model adds 18.9 ms" - and the metrics table reads `fhe_conv_ms` 77.67,
`fhe_student_dense_ms` 494.04, `fhe_encrypted_student_ms` 571.71, with 77.67 + 494.04 = 571.71
exactly. Log and table now answer the same question the same way.

`encrypted_flips_are_ties` and `encrypted_flip_margin_relative` are gone from every page.

**Splitting the error key showed something the single key hid**: `fhe_conv_max_abs_error` 4.65e-9
against `fhe_student_logit_max_abs_error` 3.46e-9 - the convolution's error is LARGER than the
student's final one. Average pooling over 2x2 divides the deviation by about two before the dense
layers see it, so the end-to-end figure understates the convolution. Under one key the larger of the
two was invisible, which is the concrete form of what that metric's own comment had claimed to
prevent.

### The disclosure had to be shipped four times, so the fifth was an assertion

Checking the second-pass fixes myself turned up two more instances of the same defect, both mine:
telecom's batch LOG had no undecidable sentence while its metrics and caption did, and the caption
was gated on `undecidableRows > 0` - so telecom's **zero**, the value that makes its 1.00 a different
claim from genomics' 1.00, rendered nothing at all.

That is four mis-scoped shipments of one disclosure: one of two batch paths, then metrics without the
caption, then telecom's log, then suppressed at zero. Fixing instances was not working, so
`tests/test_agreement_disclosure_parity.py` asserts the invariant: every path emitting
`encrypted_agreement` must emit `encrypted_agreement_rows`, `encrypted_undecidable_rows`,
`encrypted_decidable_rows` and `encrypted_agreement_over_decidable` from the same block, and the
caption must have a zero branch rather than a `> 0` gate.

**It failed on its first run**, on a third `encrypted_agreement` call site nobody had counted - the
no-corpus path, publishing a 0/1 over a single row with no denominator, no tie count and no decidable
rate. Three adversarial review passes and the author had all walked past it; a count-parity
assertion found it in one run. Both assertions were then confirmed able to fail by seeding the two
defect shapes they exist for.

Confirmed on the deploy at `b1437f4`, telecom, all three surfaces: the log reads *"0 of the 512 rows
are scored identically across classes by the plaintext model itself; over the 512 it does decide,
agreement is 1"*, the caption reads *"None of these 512 rows is a tie in the plaintext model"*, and
the metrics read 0 / 512 / 1.00.

### Client-side-only inference, re-read against the rows that already assert it

The standing request asks for confirmation that ML inference is client-side only. `CS1`, `CS2`,
`CS4`, `CS9`, `CS10` and `RS30` already assert it, so **no row was added** - the increment is the
reading.

Measured on the deployed medical page by wrapping `fetch` and `XMLHttpRequest.open` before clicking
Run, so the record is of the run itself rather than of page load. Ten requests:

- **Zero `/api/` calls** (`CS1`).
- Own origin: `/data/medical/breast_cancer_sample.json`, `/assets/ort-wasm-simd-threaded-*.wasm`,
  `/assets/seal_throws-*.wasm` - the bundled sample and the two WASM runtimes (`CS9`).
- One external origin, `huggingface.co`, and all seven of its URLs are
  `/<repo>/resolve/<40-char commit sha>/...` - `eval_sample.json`, `onnx/model*.onnx`,
  `weights.json`. No `api-inference` host, no per-token or per-request pattern.

`CS10`'s discriminator is the one that matters: a hosted inference API would look identical in the
UI and produce a request per inference. Every Hub URL here is a pinned-revision artefact download,
and each revision is a full commit SHA rather than a branch name, so the pin is exact. Inference
itself runs on onnxruntime-web and node-seal inside the tab.

### The third pass was not dry: removing one disclosure took another with it

Three independent lenses found the same thing. The commit that removed the tie verdict from the
agreement caption also deleted the **ciphertext-cap** sentence sitting in the same JSX expression -
the removed block's closing `) : null}` merged with the tie block's - so `rowsOffered` was left
declared on `EncryptedAgreementPlot`, set by four producers, and read by nothing.

`tsc` passes on that state. An unread optional interface field is not a type error, which is why
nothing in the build, the gate or the suite noticed.

The effect was on telecom, the one page where the cap bites: 512 of 600 held-out rows are scored.
The figure read *"one of 512 held-out rows ... Top-1 mismatches: 0 of 512 ... None of these 512 rows
is a tie"* with nothing saying the split held 600 and 88 were never scored - silently reverting
`5d20d88`, which added that sentence for exactly this reading, and making checklist row `PI2` a row
that could only fail.

Restored and confirmed live: *"The split holds 600 rows; one ciphertext fits 512 at this feature
width, so 88 were not scored and the agreement above is over 512, not over the whole split."*

**`MS9` was too narrow to catch it**, which is the more useful half. It asserted the undecidable trio
travels with the agreement rate and said nothing about any other field. `MS10` asserts the general
form - no field the plot type declares may be set by a producer and rendered by nothing - and
re-seeding the deletion makes it fail naming `rowsOffered` while `tsc` stays clean.

An orphaned comment went out with the same commit: the three lines justifying the deleted 1e-6
threshold survived into the middle of an object literal, arguing for a test the file retracts thirty
lines above. Removed.

### The fact-check sweep found no new literature, and one real defect in the old literature

Five modalities swept for anything dated 2026-08-10 or later: CKKS/FHE, QKD/QSDC, standards and
regulation, MPC and federated learning, QML. **Nothing published in that window changes a claim this
project makes.** The negative is checkable rather than an absence of effort - the FHE sweep
enumerated the full arXiv cs.CR monthly listing rather than keyword-matching, and probed ePrint
numbers directly (2026/1644 exists at 2026-08-09; 2026/1648 returns 404) to establish the boundary.

The finding came from reading a standard the project ALREADY CITES against the code.

**NIST SP 800-227 (September 2025, final) section 4.6.3** defines the IND-CCA-preserving combiner as

    KeyCombineCCA_H:  H(K1, K2, c1, c2, ek1, ek2, domain_sep)

and states, of the reference it is drawn from, that it "does not incorporate encapsulation keys into
the combiner, **as this is not needed to achieve the IND-CCA-preserving property**". So the
ciphertexts are load-bearing and the encapsulation keys are optional - useful for binding the secret
to party identities.

This project bound `ek1 || ek2 || c2` and **omitted `c1`, the ML-KEM ciphertext**: the optional half
included, one of the two that deliver the property dropped. Three files shared that shape -
`src/qsh/pqc/hybrid_kem.py` (encapsulate and decapsulate), `web/src/lib/crypto.ts` and
`web/src/lib/clientCompute.ts`. A fourth, `web/src/lib/clientMcp.ts`, bound **nothing at all**: IKM
`K1 || K2` under a bare label, which is expression (16) of the same document - the combiner it names
as not preserving IND-CCA - while its own comment claimed it matched the backend.

**Demonstrated, not argued.** With the old transcript, a `c1` mauled by one bit that decapsulates to
the same `K1` derived a byte-identical session key; with `c1` bound it derives a different one:

    old transcript contains c1: False        new transcript contains c1: True
    OLD: mauled c1, same K1 -> identical session key: True
    NEW: mauled c1, same K1 -> identical session key: False

That is the failure direction a hybrid exists to hedge: ML-KEM broken, X25519 sound, and the
composite key unchanged anyway.

**Scoped honestly**: this runs in one process with no network transport and no exposed decapsulation
oracle, so it was a conformance and claim-accuracy defect rather than an exploitable break in the
shipped artefact. It is fixed because standards-conformance is the claim.

All four sites changed together - they must, or backend and browser derive different keys - and the
domain separator moved to `v2`, because deriving a different key under an unchanged label is what
domain separation exists to prevent. The deployed page now reports
*info="qsh/hybrid-kem/v2", transcript c1||c2||ek1||ek2 per NIST SP 800-227 s4.6.3*.
**And the fix's own first version violated the same section it cites, twice.** Both were caught by
re-reading section 4.6.3 against what had just been written rather than against what it replaced:

- The MCP path was given `qsh/hybrid-kem/v2`, the handshake's separator - but it is a different
  composite scheme (static-static X25519, no ephemeral `c2`, transcript `c1 || ek1 || xA || xB`).
  Two constructions under one separator is what a separator exists to prevent. It now has
  `qsh/mcp-agent-kem/v1 ML-KEM-768+X25519-static`.
- The separator named `ML-KEM-768` as a fixed literal while `HybridKEMRecipient` takes a `kem_name`
  and the browser's `kemImpl()` genuinely switches to ML-KEM-1024 on the category-5 path - measured
  live, ciphertext 1088 bytes to 1568. So a category-5 run derived its key under a separator
  asserting the category-3 parameter set. Section 4.6.3 requires domain_sep to identify the
  "choice of parameter set" explicitly. It is now derived: selecting category 5 on the deployed page
  produces `info="qsh/hybrid-kem/v2 ML-KEM-1024+X25519"` beside `kem_ciphertext_bytes 1568`.

That also removes a concatenation ambiguity: the transcript joins fixed-width fields with no length
prefixes, which is unambiguous only within one parameter set, and the set is now in the separator.

`references/stress-testing-fact-check.md:56` proposed a test for "the ciphertext-binding combiner";
that description became true of the code with this change and was not before, and the file now says
so.

### A fifth combiner site nobody had listed, and what its severity actually is

The combiner change was reviewed by four lenses with three refuters each: **all 29 refutations were
sustained against the four sites** - nothing survived. The completeness critic then found a fifth
site, and two defects in the fix itself.

**`clientAliceBob` derived its session key as `enc.sharedSecret.slice(0, 32)`** - the raw ML-KEM
shared secret, truncated. No X25519, no KDF, no transcript, no separator. Not expression (16), which
is at least `KDF(K1, K2)`; this was `K1` alone, in a function whose header calls it a mirror of
`src/qsh/benchmark/alice_bob_e2e.py`, whose Phase 3 goes through the combiner and then `_mix_qkd`.
It also reported `qkd_mixed: qkd.secureKeyBits >= 256` - a PRECONDITION for mixing, stated as though
the mix had happened, in a function that mixed nothing.

**Severity, measured rather than assumed.** `clientAliceBob` is imported nowhere; `api.aliceBob`
posts to a full-mode endpoint and is called nowhere; and `qkd_mix_note`, a string this fix adds,
appears **0 times in every built bundle** - the function is tree-shaken out of the production build.
So this was never rendered and no visitor ever saw the claim. The review's framing ("Open the
Alice<->Bob page") describes a page the demo does not reach. It is fixed rather than deleted because
it is exported and documented as a mirror, so it would ship the claim the moment anything imported
it - but it is recorded as dead code, not as something the site published.

It now runs the hybrid: X25519 ephemeral-static beside the ML-KEM half, the same v2 combiner over
c1 || c2 || ek1 || ek2, and `qkd_mixed: false` with a note saying why - the browser's `bb84()`
returns bit counts and no key material by design (`SC2`), so there is nothing to fold in.

### Two defects in the fix, and one in the test written to prevent them

**The MCP file's lead comment still claimed it matched the handshake** - the exact false claim the
transcript change removed, left standing at the top of the block that removes it, contradicted 25
lines below. A maintainer told to make the two agree would have re-merged two constructions under
one separator.

**The output length was absent from the domain separator.** HKDF-Expand for L bytes is a prefix of
the expansion for any larger L', and `length` genuinely varies across callers - 32 for transport
sessions, `max(32, (2n+7)//8)` for the QOTP pad - so a long pad began with exactly the bytes a short
session key over the same material would. Not reachable, because every `encapsulate` draws fresh
material, but it is the same class of gap as the parameter set and the docstring had just claimed
"no two parameter sets share a derivation". Verified after the fix: the 32-byte derivation is no
longer a prefix of the 51-byte one.

**And the test written to enforce c1-binding did not enforce it.** Its first version built
`c1 + c2 + ek1 + ek2` itself and asserted that mauling `c1` changed `_combine`'s output - testing
`_combine` given a transcript handed to it, not what `encapsulate` binds. Reverting the production
combiner left it green. That is the defect this round spent its length finding elsewhere, committed
in the guard written to prevent it, and caught only by running the revert.

Rewritten to pin the production path: derive the key both ways from the real ciphertext and assert
which one `encapsulate` agrees with, plus a `decapsulate` test that holds K1 fixed so c1 is the only
variable. The comparison that matters, on a revert of the c1 binding alone:

    tests/test_pqc.py                      0 failures
    tests/test_hybrid_combiner_binding.py  2 failures

which is exactly the review's point: the property had nothing enforcing it.

---

## Round 43

Round 43's own records are the pointer section directly below and its sub-block on the rounds it
reconstructed; this heading exists so that a search for the round finds them (added 2026-11-13).

## Rounds 39-42, added in Round 43 - and this is a POINTER, not four entries

**This file's last round section was Round 38.** Rounds 39, 40, 41 and 42 have no entry, which is the
third instance of the gap this file already documents twice: Rounds 25-27 were never reconstructable,
and Rounds 29-31 were reconstructed in Round 32 by a round that then recorded nothing of its own.
The cause has not changed and is stated at `:364`: *"the round that writes the checks is not the round
that runs them, and nothing forces the second half."*

**What this block deliberately is not.** It is not four reconstructed entries. Each of those rounds
did real browser verification and the measurements exist - in
[`deliverable-verification.md`](deliverable-verification.md), which carries a full `## Round 39`,
`## Round 40`, `## Round 41` and `## Round 42` section, and in the branch commit messages. Writing
them out here from those sources would produce four entries that read as though someone sat at a
browser and took them, and **a reconstruction presented as a record is worse than an absent record**,
because the absence is visible and the imitation is not. Round 32's block says the same thing about
its own reconstruction: *"labelled as reconstructed rather than presented as if logged live."*

So this is a pointer. Where the measurements for those four rounds actually live:

| Round | Subject | Where the measurements are |
|---|---|---|
| 39 | the previous round's outputs disagreed with each other; backend and browser derived different session keys | `deliverable-verification.md` "Round 39", plus `tests/test_hybrid_combiner_parity.py` |
| 40 | claims that move numbers - `security_model` is parsed to raise two displayed scores | `deliverable-verification.md` "Round 40"; `tests/test_security_model_prose_is_not_a_silent_score_input.py` |
| 41 | every gate names a boundary and checks one side of it | `deliverable-verification.md` "Round 41" |
| 42 | four gates that enforce nothing in CI, and a number that is wrong | `deliverable-verification.md` "Round 42" |

**What Round 43 measured on the deployed site**, recorded here because it was taken here and belongs
in this file rather than in the register:

- **Client-side compute, on the wire.** A full `medical_fl` run captured at bundle
  `index-DWKgg_IL.js`, 2026-08-15: **zero POST requests**, and exactly two origins contacted - the
  demo origin for static assets (`ort-wasm-simd-threaded-*.wasm`, `seal_throws-*.wasm`,
  `/data/*/real_2d.json`) and `huggingface.co` / `us.aws.cdn.hf.co` for model weights pinned by
  40-hex revision. **No request to the demo origin matched `/api/`, `/run`, `/infer`, `/predict` or
  `/compute`.** This is the strongest evidence `CS1`-`CS11` have had; every prior round asserted the
  property from the source.
- **All ten pages**, same bundle: 0 empty cards, 0 single-cell table rows, no `NaN`/`undefined`, no
  horizontal overflow, no console output from the deployed bundle. The skip link is the first
  focusable element and activation leaves `document.activeElement === document.getElementById("main")`.
  The DAG carries `[running]`/`[done]` text and a 10-item `sr-only` stage list. The agreement scatter
  draws 27 circles and 3 crosses with real paths. `skr (bit/s)` renders `n/a` beside `key budget`
  `1,436 bit`. Agreement reads `100.0% of the time`.
- **A limit, stated.** The browser harness could not deliver synthetic `Tab` or `Enter` to the page -
  focus did not move off the element it started on - so the skip link's tab-ORDER was not walked. Its
  position as first focusable establishes it structurally and not empirically.

## Round 44, added in Round 45 - a POINTER, and the fourth time this gap has opened

**Round 44 wrote no section here.** That is the fourth recurrence of the gap this file already
documents twice - Rounds 25-27 were never reconstructable, Rounds 29-31 were reconstructed in Round
32 by a round that recorded nothing of its own, Rounds 39-42 became the pointer above - and it
recurred *in the round immediately after the one that wrote that pointer*. The root cause stated at
`:364` still holds: **the round that writes the checks is not the round that runs them, and nothing
forces the second half.**

Round 45 added what does force it: `tests/test_the_log_has_an_entry_for_the_current_round.py`. A
round that raises the register's heading without leaving a section here now fails a gate rather than
relying on the next round to notice.

This is a pointer and not a reconstruction, for the reason `:1241` gives: *a reconstruction presented
as a record is worse than an absent record, because the absence is visible and the imitation is not.*
Round 44 **did** verify on the deployed site; it wrote the measurements into the register instead of
here.

| Round | Where its deployed-site measurements live |
|---|---|
| 44 | `docs/deliverable-verification.md`, "Round 44" section - `skr_per_pulse` 0.00958 x `source_rate_hz` 1e8 = `skr_bit_s` 957,659; `channel_time_s` 0.002; `stored_key_count` 4 = floor(1436/352), restored from a permanent 0 |

**One line above is now known to be wrong, and it is left standing with this correction beside it.**
The Round 43 entry records `skr (bit/s)` rendering `n/a` beside `key budget 1,436 bit`. It does not
record the `0` sitting between them - `stored_key_count`, permanently zero from Round 42 to Round 44.
Round 44's register entry already says so. Repeating it here because this file is where a reader
looks for what was seen on the page, and the entry that missed it is in this file.

**And the figure Round 44 recorded is no longer the one the demo shows.** Its 957,659 bit/s was the
non-decoy GLLP fraction multiplied by an assumed clock, while three surfaces called it a decoy-state
bound; Round 45 measured that gap at 4.07x and replaced the model. The Round 44 numbers above are
correct as a record of what was on the page that day, and are not the current expected values.

## Round 45 (`b914e8f` on its branch, bundle `index-DKpFP7vh.js`) - the rate was 4.07x the bound the page named

Verified in Chrome against the public deploy, 2026-08-15. Every figure below was read off the page,
not computed from the source.

**The decoy bound, and it reconciles in both directions.**

| read on the page | value |
|---|---|
| `skr_per_pulse` | **0.00235** |
| `skr_bit_s` | **235,479** |
| `channel_time_s` | **0.002** |
| `source_rate_hz` | **100,000,000** |
| `qkd_pulses` | **200,000** |
| `secure_key_bits` | **470** |
| panel `skr` / `eskr` | **235,000** / **176,000** |
| `key budget` | **352 bit** |
| `stored_key_count` | **1** = floor(352/352) |

`0.002355 x 1e8 = 235,479`. `200,000 / 1e8 = 0.002 s`. `235,479 x 0.002 = 470.96`, floored to the
**470** the page prints. The browser figure equals `src/qsh/qkd/bb84.py::_decoy_secure_fraction` to
better than 1e-9, which is asserted by a gate rather than eyeballed here.

**Before this round the same panel showed ~958,000 bit/s**, from the non-decoy GLLP fraction on the
observed QBER, while three surfaces called it a decoy-state bound.

**The Source rate control is real, which is the check no source gate can make.** Moving it from 100
to 250 MHz, on the deployed page:

| | 100 MHz | 250 MHz | expected |
|---|---|---|---|
| `skr_bit_s` | 235,479 | **588,697** | x2.5 = 588,698, to rounding |
| `channel_time_s` | 0.002 | **0.0008** | /2.5 |
| `skr_per_pulse` | 0.00235 | **0.00235** | unchanged - the bound is a property of the channel, not the clock |
| panel `skr` | 235,000 | **587,500** | scales |

Two rendered strings promised this control before it existed. The third row is the one that matters:
a control that moved the per-pulse figure too would mean the clock had leaked into the physics.

**`/validations` shows a number again.** `secret key rate (per pulse)` reads **0.002355** where it
had rendered an empty cell since Round 44; `secure key bits` **94** = floor(0.002355 x 40,000), the
`demo_alice_bob_pulses` count. QBER 0.0124, eavesdropper false.

**`/orchestration?usecase=` resolves or explains itself.**
`?usecase=nonsense_value` renders *"There is no use case called nonsense_value. Showing Finance —
cross-bank AML / PSI instead - pick another from the selector below."* and the selector reads
`finance_aml`. `?usecase=satellite_comms` selects satellite with no notice. Both directions walked.

**The two source-rate controls are per channel, not shared.** The satellite page renders
`sat_source_rate_mhz` and NOT `qkd_source_rate_mhz`; the fibre pages render the reverse. That mirrors
`sat_distance_km` / `qkd_distance_km`, whose conflation once had the satellite page evaluating a
free-space model at fibre parameters.

**All ten pages walked**: `/`, `/value`, `/usecases`, `/orchestration`, `/validations` and the five
use-case pages. Zero occurrences of `NaN`, `undefined` or `Infinity` in rendered text on any of them,
correct `h1` on each, and **no console errors or warnings** from this bundle across the sweep.

**The E91 label, walked rather than inferred.** This was going to be recorded as a stated limit -
verified by gate and by source reading only - and it was cheap enough to close instead. Selecting
`e91` in the Parameters panel (`pp-proto`) and running telecom_ids on the deployed page:

| row | rendered |
|---|---|
| QKD key material | `1597 secure bits` - measured, `E91 / CHSH run, GLLP secure fraction` |
| Secret-key rate | `0.00799 per pulse` - measured, **`GLLP secure fraction on the observed QBER; E91 has no decoy states`** |
| `chsh_s` | **2.7** (above the classical bound of 2, so a key exists) |

Before this round that second row read *"GLLP bound over the decoy-state estimate"* on this page,
because it was emitted outside the protocol branch three lines above it. E91 is entanglement-based
and has no attenuated laser to put decoy states on, so the page was attributing its key to a
technique the protocol cannot use. The word "decoy" still appears in the row - now negating.

## Round 46 (`f5c1d0a`-era branch, bundle `index-CtNKHnlK.js`) - both languages, one model, and a block size that can carry a key

Verified in Chrome against the public deploy, 2026-08-15. Every figure read off the page.

**The block size, and the counterfactual that was inert until this round.**

| read on the page | value |
|---|---|
| `qkd_pulses` (control) | **2,000,000** |
| `skr_per_pulse` | **0.002355** |
| `skr_bit_s` | **235,479** |
| `channel_time_s` | **0.02** |
| `secure_key_bits` | **4,709** |
| `skr_per_pulse_no_decoy` | **0** |
| panel `skr` / `eskr` | **235,500** / **176,650** |
| `key budget` | **3,532 bit** |
| `stored_key_count` | **10** = floor(3532/352) |

`0.002355 x 2e6 = 4,710`, floored to the 4,709 shown. `2e6 / 1e8 = 0.02 s`. The bit/s figure is
UNCHANGED by the pulse increase, correctly - a rate is per-pulse times clock, and only the totals
scale. `stored_key_count` was 1 at 200,000 pulses.

**`skr_per_pulse_no_decoy` renders 0**, and that is the point of emitting it. Round 45 added the
field with a JSDoc saying "Reported beside `secretKeyRate`" and nothing read it. It is zero because
the tagged fraction is 7.26 at $`\mu = 0.5`$: without decoy states this link certifies no detection as
single-photon and yields nothing at all. The counterfactual is now visible rather than asserted.

**The 0-255 band, walked at two distances.** At 95 km the run gives 499 secure bits and no warning -
correct, it is above the threshold. At 115 km:

    secure_key_bits   141
    stored_key_count  0
    "... below the 256 a session key needs, so no QKD key is derived and the session rests on the
     PQC hybrid alone. The bits are real; there are not enough of them."

Before this round that band printed "no secure key bits" beside a positive count, and fired no
warning at all because the test was `keyBits === 0` while the behaviour switches at 256. Raising the
pulse count also moved the entry to the band from ~63 km to ~110 km, so the cliff is no longer three
clicks from the default.

**The QCNN runs in the browser**, on `/usecase/telecom_ids` with Inference set to QCNN:
`qcnn_qubits 8`, `qcnn_classes_available 8`, `qcnn_inference_ms 1.92`, class rendered. This is the
ported architecture - shared conv filter, pooling by conditioned rotation, register reduction - not
the ansatz it replaced, and it is the same circuit the trainer and `qsh/qhe/qcnn_trained.py` run.

**The provenance strings are protocol-correct**: `BB84 decoy-state run, two-intensity secure
fraction` and `two-intensity decoy-state bound, Ma, Qi, Zhao & Lo, Phys. Rev. A 72, 012326 (2005)`.

Zero occurrences of `NaN`, `undefined` or `Infinity` in rendered text on any page walked.

**A limit, stated.** The ten-page sweep for this round covered `/usecase/finance_aml` and
`/usecase/telecom_ids` in depth plus the pages reached during the Round 45 sweep on an earlier
bundle. The remaining seven were not re-walked on `index-CtNKHnlK.js`, so their cleanliness is
inherited from the previous bundle rather than measured on this one.

## Round 47 (`3c63ea8`, branch `round47-the-gate-was-a-file-list`) - a POINTER, and the absence is recorded on the day it happened

**No deployed-site walk has been recorded for Round 47.** The bundle carrying this round's changes
has not been walked in Chrome, and no figure below was read off a page. This entry is written the way
the Rounds 39-42 and Round 44 entries are - as a pointer that is honest about what is missing - and
for the reason `:1241` gives: *a reconstruction presented as a record is worse than an absent record,
because the absence is visible and the imitation is not.* Round 44 is the counter-example this file
keeps: it verified on the deploy and wrote nowhere, and the gap was found a round later. Here the
gap is stated in the round that has it.

**Where Round 47's evidence lives, all of it in-repo:**

| what it establishes | where |
|---|---|
| the round as a whole | `docs/deliverable-verification.md`, "Round 47" and "Round 47, continued" sections |
| the finite-key threshold, computed rather than cited, and checked against the paper's Table I | `src/qsh/qkd/finite_key.py`, `tests/test_finite_key_reproduces_the_published_table.py` |
| the block size every config key must clear, and the two declared residuals | `tests/test_every_block_size_clears_the_band_the_config_names.py` |
| the browser and the trainer computing the same QCNN, numerically, on the shipped artifacts | `tests/test_the_qcnn_agrees_across_languages.py` |
| the readout binning over surviving wires | `tests/test_qcnn_multiclass_readout.py` (which checks the trainer's `readout_wires()` against an independent re-derivation of the survivors, not the artifacts), and `readout.wires` = `[0, 2, 4]` in the TWO MULTICLASS artifacts - `models/qcnn_telecom/weights.json` (8 wires) and `models/qcnn_telecom_q7/weights.json` (7 wires). The other four QCNN artifacts carry `readout.kind` = `expval_z0` with no `wires` key at all: a single `<Z0>` expectation bins nothing, so it has no binning to correct. This row said "every `models/*/weights.json`" for one round, and a reviewer executing it on any binary artifact found nothing. |
| no trainable parameter being analytically dead, at five configurations | `tests/test_the_qcnn_has_the_structure_the_name_claims.py::test_every_trainable_parameter_reaches_the_readout` |
| the published QCNN metrics, per artifact | `models/*/weights.json` and the cards gated against them |
| the browser and Python agreeing on the decoy bound, including under an eavesdropper | `tests/test_the_browser_computes_the_bound_the_page_names.py` |
| those cross-language gates actually RUNNING rather than skipping | `.github/workflows/ci.yml`: Node install plus `QSH_REQUIRE_BROWSER_GATES=1` |

**What that leaves unestablished, stated rather than implied.** These are gate results and artifact
contents, which is a different thing from a rendered page. Every one of the numbers below is a
quantity a visitor can see, and each has moved since the last bundle that was walked:

- `/usecase/telecom_ids` with Inference = QCNN now runs a **per-survivor** fully-connected block and
  reads the class index off the **surviving** wires. Before the fix the browser and the trainer
  disagreed on the argmax for 39 of 64 telecom rows, so the rendered CLASS is the thing to look at,
  not merely the absence of `NaN`.
- **All six QCNN artifacts were retrained a second time** after two classes of parameter turned out
  to be analytically dead, so every rendered QCNN figure on every use-case page has moved again:
  telecom macro F1 0.458324 -> **0.510963**, telecom-q7 0.500452 -> **0.541894**, medical accuracy
  0.860465 -> **0.883721**, finance minority-class F1 0.84058 -> **0.735294**. The finance number
  went DOWN. The classical logistic regression rendered beside each of them is unchanged, because it
  never sees the circuit, and it is still ahead on all three use cases.
- **The Eve intercept control now does something**, and what a visitor sees on the QKD panel when
  they move it is the single largest untested change in this round. At `eve_intercept_fraction` =
  1.0 the page used to print 0.001976547 bit/pulse and 395 secure bits at QBER 0.2598 - the same
  figures as eve = 0. Measured in-repo after the fix, at 200,000 pulses and identical in both
  languages: eve 0.2 gives 0.000058995 and 11 secure bits, eve 0.3 gives zero from the bound itself,
  and from eve 0.35 the `pp_threshold` abort fires. **Whether the page RENDERS that abort sensibly -
  rather than a blank, a `NaN` or a silent zero - has not been observed.**
- The QKD panel's per-pulse figure moves in two directions at once and neither has been read off a
  page: the intensity change to the finite-key optimum costs **16.5%** of the asymptotic rate
  (0.002368 -> 0.001977 bit/pulse, `src/qsh/config/defaults.py`), while the decoy-bound error-term
  fix RAISES the fibre figure by **0.57%** and the 800 km downlink by **18.9%**.
- `qkd_surrogate_skr_default`, rendered in the same panel as `skr_per_pulse`, was computed by a
  FOURTH implementation of the decoy algebra that was left on the old error term. It read 0.498% low
  on fibre and 13.664% low on the 800 km downlink, so the two figures in that one panel disagreed
  with each other. Now 0.000% on both, and unobserved on a page.
- `qkd.pulses` and `demo_alice_bob_pulses` are both **20,000,000**, ten times the block the Round 46
  sweep saw, so `channel_time_s`, `secure_key_bits`, `key budget` and `stored_key_count` all scale
  and the `/validations` count is a different number from the 94 recorded in the Round 45 entry.

**And the gates that stand in for a walk were not all running.** Every browser-side cross-language
check guards on `if not ESBUILD.exists(): pytest.skip(...)`, `web/node_modules` is gitignored, and
the GitHub CI Python job installed no Node - so in CI they SKIPPED. Measured on a clone of the tip
with no `web/node_modules`: **21 collected, 11 passed, 10 skipped** - 6 in
`tests/test_the_browser_computes_the_bound_the_page_names.py` and 4 in
`tests/test_the_qcnn_agrees_across_languages.py`. This paragraph said 20 / 11 / 9 for one round,
which was the count one commit before it was written; the commit that published it added a 21st
gate, and that gate skips too. For the rounds in which this file's entry has been a pointer
to in-repo evidence, part of that evidence was not being produced by CI at all; it was being produced
only by `scripts/ci_local.sh` on one machine. `.github/workflows/ci.yml` now installs Node and sets
`QSH_REQUIRE_BROWSER_GATES=1`, which turns the skip into a failure.

A walk of the deployed bundle therefore remains owed, and this paragraph is the record that it is
owed rather than done.

## Round 48 (`e192214`, branch `round48-the-page-was-serving-a-crippled-model`) - WRITTEN IN ROUND 49, because Round 48 wrote nothing

**This entry is late by one round, and that is the most important thing in it.** Round 48 shipped
three commits (`2d83b62`, `5aa444b`, `75b4cbf`), merged as PR #39, and deployed - and wrote no
`## Round 48` section in `docs/deliverable-verification.md`, no entry here, and no checklist rows for
any of its six changes. Not a thin record: none.

The convention this file already had for a missing record is the Rounds 39-42 and Round 44 pointer,
and `:1241` states the principle those entries rest on - *a reconstruction presented as a record is
worse than an absent record, because the absence is visible and the imitation is not.* So this entry
is dated as what it is: written in Round 49, about Round 48, from the commits and the artifacts
rather than from anything observed at the time.

**What made this different from Round 44's identical gap is that nothing could have noticed.** Both
round-obligation gates derive "which round it is" from the register's newest `## Round N` heading -
`tests/test_register_deadlines.py::_current_round()` and a second copy of the same function in
`tests/test_the_log_has_an_entry_for_the_current_round.py`, whose own comment explains that reading
the same file means the two "cannot disagree". They cannot. Neither can disagree with a register that
stopped being written. Measured on this tip: `_current_round()` returned **47**, so the log gate asked
whether the log named round 47, it did, and both went green with an entire round missing. Three
obligations conditioned `by Round 48` stayed pending for the same reason.

Round 49 adds `tests/test_the_register_records_every_round_that_shipped.py`, which takes the round
from git merge subjects instead - `roundN-` branch names, which no amount of forgetting to write prose
can change - and fails when the register lags. It is red at this commit, where history said 48 and the
register said 47.

**Where Round 48's evidence lives:**

| what it establishes | where |
|---|---|
| the round as a whole | `docs/deliverable-verification.md`, "Round 48" section, written in Round 49 |
| the finance QCNN receiving the feature space its artifact declares | `tests/test_the_encoder_receives_the_space_the_artifact_declares.py` |
| the raw-space rows the fix required, and the EDA that found the degenerate columns | `scripts/data/build_federated_samples.py`, `samples_by_space` and `eda.duplicates` |
| the browser selecting that space by name, with no default branch | `web/src/lib/realData.ts` (`rowBySpace`), `web/src/lib/clientCompute.ts` |
| an artifact shorter than its declared circuit failing loudly | the theta-length guard in `web/src/lib/quantumSim.ts` |

**The defect it fixed was visitor-facing and real.** The finance page fed the QCNN z-scored values
while its artifact declared `input_space: "raw"`, so the encoder received a space the weights were not
trained on; correcting it raised the encoded spread on wire 0 by about **17,000x** on the live site.

**What Round 48 got wrong, found in Round 49 and fixed there.** Both are the same defect class this
project has now recorded seven rounds running - a fix reaching one of the places that states the thing:

- `skr_per_pulse` has FOUR emitters, and Round 48's "one panel, two roundings" fix reached three.
  `src/qsh/orchestration/stages.py` kept five decimals nine lines below its own satellite twin using
  six.
- The corrected federated figures (25.25% / 0.9125) landed in ONE of three copies. The commit that
  said "this commit uses the right one" left the superseded 17.25% in two other files.

**No deployed-site walk was recorded for Round 48**, and unlike Round 47 - where the absence was
written down in the round that had it - here the absence was itself unrecorded until now. A walk of
this bundle is owed and was never performed.

## Round 49 (branch `round49-the-round-that-left-no-record`) - a POINTER, and the deploy walk is deferred by an explicit decision

**No deployed-site walk has been recorded for Round 49.** This is a deferral, not an oversight: with
the round's three conditioned obligations all landing in one session, the choice was between finishing
the record and walking the bundle, and the record won. Round 48 is the reason - a round that ships
without its record disables every gate that depends on knowing which round it is, and this round
exists mostly to fix that. Walking the site and writing nothing down is the failure this file already
records twice (Round 44, Round 48); writing the record and owing the walk is the smaller debt, and it
is stated here rather than left to be discovered.

**What the round was for.** Round 48 shipped and wrote no record, and no gate noticed, because both
round-obligation gates take the round number from the register a forgotten round fails to write. The
fix is a second, independent source: git merge subjects name `roundN-` branches.

| what it establishes | where |
|---|---|
| the register cannot silently lag the merged history | `tests/test_the_register_records_every_round_that_shipped.py` - red at `e192214`, where history said 48 and the register said 47, and it asserts on `__code__.co_names` that its own source is not the register |
| one metric has one rounding, across both languages | `tests/test_one_metric_has_one_rounding.py` - scoped to metric channels, so the `mean`/`sd` over-fire was fixed by narrowing the scope rather than by widening an exception list |
| every directory that ships an artifact ships a card | `tests/test_every_shipped_artifact_has_a_card.py` - enumerates ARTIFACTS, which is the one thing a card-globbing gate cannot do |
| the 8-class readout is identifiable at every shipped width | `tests/test_qcnn_multiclass_readout.py`, rewritten - the version it replaces could not fail |
| the three conditioned obligations | `references/qcnn-width-ablation.md`, `references/finance-calibration-isotonic-vs-platt.md`, and the 5G-NIDD sections of `docs/deliverable-verification.md` |

**The three obligations, all discharged.**

1. **The width ablation, re-run at five seeds.** Finance and medical give BYTE-IDENTICAL metrics at 8
   and 9 wires at every seed on both corpora - the pad-wire finding is now exact where it was one
   seed. Telecom 7-vs-8 is the honest half: 7 wins 4 of 5 seeds and the mean gap is +0.0463, LARGER
   than the single-seed figure it replaces, but the ranges overlap and neither test separates them
   (paired t p = 0.059, Wilcoxon p = 0.125). Reported as a direction that survives and a magnitude
   that does not. Seed 0 reproduces all four shipped artifacts to every recorded digit, which is what
   makes the sweep about the current circuit rather than some other one.
2. **Isotonic against Platt, and the answer was neither.** The shipped `fit_platt` is fed a
   PROBABILITY where its map expects a log-odds, so it returns one constant (0.00090148) for deciles
   1 through 9 and cannot express a risk below 1 in 1,109 on a corpus whose base rate is 1 in 656.
   That is the source of the +8.7% log-loss the record has carried for three rounds as a base-rate
   effect. The unit tests could not catch it: their fixture passes a logit and the trainer passes a
   probability. Isotonic wins nothing a corrected Platt does not also win and costs 9.09% of AUPRC,
   collapsing 1,388,669 distinct scores to 138 and emitting exact zeros on 30.5% of test rows.
   Recommendation recorded: keep Platt, fix what it is fed. **THE FIX WAS APPLIED LATER IN THIS
   ROUND, on this branch.** This sentence first read "THE FIX IS NOT APPLIED - it retrains and
   rewrites `models/`, and this was scoped as a measurement", which was true when written and
   stopped being true without the paragraph changing - the defect class this log exists to catch.
   What landed: `fit_platt` grew `on_logit=True`, carrying the log-odds transform, a 1e-12 clip and
   the published formula string WITH the two floats rather than leaving the transform at the call
   site; `scripts/train/finance_aml.py` passes it; the teacher was retrained and
   `models/metrics.json` and `web/src/lib/config.defaults.json` regenerated from it. Verified
   against the prediction rather than assumed: `a = -0.5853491088584304`, `b = 1.6417462922842536`
   against the ablation's `-0.58535` / `1.64175`; Brier 0.00145433 -> **0.00142199**; log-loss
   0.01049835 -> **0.00818964** (the arm that had been shipping was WORSE than not calibrating at
   all, 0.00965987); CORP MCB 4.412e-05 -> **1.178e-05**; AUROC and AUPRC bit-identical at
   0.9525582593503528 / 0.12334101789973353, which is this round's ranking claim measured from a
   second direction. Two things the fix needed beyond the one line, both of the "landed in one of
   three sites" class this round was already tracking: `web/src/components/ModelPanel.tsx` rendered
   $`p = 1/(1 + e^{a s + b})`$ unconditionally and would have printed a score-domain formula beside
   logit-fitted floats, and the `imbalanced` test fixture did not reproduce the defect it was written
   for - it claimed a 1-in-650 base rate and 90% of scores below 1e-4 and delivered 1-in-16 and
   34.8%, drawing its labels from the score itself - so the regression test would have been built on
   a fixture that could not fail. **A THIRD CONSEQUENCE, WHICH WAS NEARLY FILED AS A PRE-EXISTING
   FAILURE AND WAS NOT ONE.** The retrain moved `hf_revision` into
   `hf_revision_invalidated_by_retrain` for `finance_aml.gbm` and `finance_aml.student` - correct
   behaviour, since the recorded sha now pointed at weights the metrics no longer described - which
   turned `tests/test_qcnn_reachable.py::test_every_published_model_is_pinned_to_a_revision` RED. It
   was GREEN at `ece9830`. `onnxInference.ts` then skips both, so the finance use case had zero
   browser-runnable models: the flagship panel, including the calibration row this round fixed,
   would have rendered a stale-pin explanation instead of a model. Resolved by re-publishing both to
   the Hub - `finance_aml.gbm` at `91562e197d93340397d62b2d8a7e8f6350134e6c` (new bytes),
   `finance_aml.student` at `89e2de895db41591b480eb744ddb6761ed208f75` (unchanged bytes, so the Hub
   returned the existing commit rather than an empty one) - and re-running `scripts/export_config.py`.
   The gate is green again. Recorded because "a retrain unpins what it retrains" is a standing
   consequence of every retrain in this repository, not a one-off, and the round that discovered it
   is the round that should say so.
3. **5G-NIDD adopted and published**, `Daemons-Q/qsh-telecom-5g-nidd-gbm` at revision `b00628c5`.

**A record correction found while measuring.** The retained-Platt argument named 2,321 positive
events as the binding quantity. That is the TEST split's event count; the calibration partition has
**801**, and `models/metrics.json` already recorded `positives_calibration: 801` one line above the
sentence that said 2,321.

**Defects this round found in its own past work.** Two from Round 48, both the same class - a fix
reaching one of the places that states the thing: `skr_per_pulse` had a fourth emitter the "one
panel, two roundings" fix missed, and the corrected federated figures landed in one of three copies.
A seventh instance was found in `qcnn_variational.py`, where the docstring still stated the readout
convention Round 47 replaced forty lines below it - documentation drift only, since all three
implementations bin by survivors and the browser throws when an artifact disagrees.

**Stale figures corrected, each re-derived rather than transcribed.** `docs/crypto-agility.md` was
wrong on two axes at once - its figures reproduce exactly at the retired $`\mu = 0.5`$ / $`\nu = 0.1`$, AND it
called the bound's input "intrinsic QBER" when the two differ by the dark-count term. Re-deriving
mattered: the corrected triple carried into this round from the plan turned out to mix the two
quantities, so transcribing it would have shipped a wrong set. `spaces/qsh-inference` published a
pre-improvement finance F1 (0.081 against the recorded 0.151948) in BOTH its README and its `app.py`,
on the one surface this project publishes outside the repository. `DATA_LICENSES.md` said "three of
the six" reference papers where the table has seven rows and two of them are arXiv papers.
`references/REFERENCES.md` said "15 bundled PDFs" where 7 are committed and 11 are on a working
checkout, and its "zero hits" HCI grep now returns exactly one - itself, by self-reference, which is
recorded rather than edited to a number that would read as a real source appearing.

**The checklist was two rounds behind, not one.** Rounds 47 AND 48 both added nothing; the highest
round it named was 46. Twenty rows added. Zero duplicate IDs, re-verified - an earlier scan in this
round reported two, which was a defect in the scan's own regex truncating the `RS2b`/`SC8b` suffix
convention, not in the file.

A walk of the deployed bundle is owed for Rounds 48 and 49 both.

### Round 49's own defect, found after the PR was opened: a docstring that asserted what the code did not do

The `imbalanced` fixture added in `da7cd89` carried three figures in its docstring - a base rate near
1 in 650, ninety percent of scores below 1e-4, and the claim that it reproduces the probability-fed
collapse. **None of the three was measured, and all three were false.** Re-run on the committed
draw: base rate **1 in 15.9**, **34.83%** of scores below 1e-4, and a probability-fed spread of
**2.48x** across the bottom nine deciles - so the fixture written to expose the collapse did not
produce it. `y = uniform < s` draws the labels from the score itself, which makes the teacher
perfectly calibrated by construction; the on-logit fit then comes back near the identity and there is
no defect left to catch.

The tests passed, which is the point. They asserted that the published formula reproduces the fitted
map and that the clip keeps a saturated score finite - both true, and neither sensitive to whether
the fixture had the shape its own docstring described.

This is the same failure as the one this round corrected in `docs/crypto-agility.md`, committed in
the same round by the correction pass. That paragraph was wrong because a figure was transcribed
instead of derived; this docstring was wrong because a figure was *asserted* without ever being
computed. The log entry above says every figure was "re-derived rather than transcribed", and for the
documents it names that is true - it was not true of the test fixture written alongside them.

The replacement builds the score from a log-odds with a long left tail and the labels from a
DIFFERENT map of that log-odds, so the teacher is miscalibrated the way a real booster is. Every
figure in its docstring was re-measured against the committed draw and reproduces exactly: 34
positives in 24,000 (1 in 705.9), 89.91% below 1e-4, span 6.975e-08 to 0.9998613, probability-fed
spread 1.000399x against on-logit 90.6954x.

It also carries the assertion the suite never had.
`test_a_probability_score_is_not_fitted_into_a_near_constant` checks the SHAPE of the fitted map
rather than a metric, which is what every existing test missed: fed a probability, `fit_platt`
returned a map emitting one number for nine of ten deciles and Brier IMPROVED, so a metric-only suite
waves it through. It did, for three rounds.

## Round 50 (branch `round50-one-column-explained-the-gap`) - the first walk in three rounds, on the deploy

**A deployed-site walk was performed and this entry records it**, which Rounds 48 and 49 both owed and
neither did. `DP7` is discharged for the first time in sixteen rounds.

**Measured before touching anything.** The served bundle was `index-CvOb630D.js`: it contained Round
48's `input_space` (2 hits) and `samples_by_space` (1), and **zero** hits for `telecom_ids.nidd`,
`qsh-telecom-5g-nidd-gbm` or `5G-NIDD`. So Round 49 had never reached the box, and the model it
published to Hugging Face was invisible to every visitor for a full round.

After `scripts/deploy_demo.sh --return-to-main`: box on `main` at `f555a89`, bundle
`index-Dxu788k6.js`, public URL 200, and the same greps return 1 each.

**All ten pages walked.** No `NaN`, no `undefined`, no `[object Object]`, no duplicate headings on any
page. Genomics renders its orphaned baseline as an explanation rather than a bare number.

**The 5G-NIDD model runs for a visitor, and reproduces its own recorded figure:**

> PASS telecom_ids.nidd: re-scored all 16379 published held-out rows in this tab and got macro_f1 0.93
> against the recorded 0.93 (|delta| 0.00e+0)

on `onnx-wasm`, 1x39, 6.3 ms, predicted class 7 against true 7, with the full IEEE Data Descriptions
citation and CC BY 4.0 attribution rendered. Backend is wasm rather than WebGPU because the graph is
`ai.onnx.ml` and the WebGPU build does not register `TreeEnsembleClassifier` - correct, not a
regression. Independently corroborated this round (corrected 2026-11-05: the trial was planned for M147-M149 and
disabled on 2026-03-27, and WebNN is behind a flag in Chrome 156, per blink-dev and chromestatus): WebNN is still Chrome Origin Trial M147-M149 with
no Safari or Firefox implementation, so the code's decision to probe it and never offer it to ORT is
right for 2026.

**Two defects found on the deploy, both fixed in this round.** The companion caption rendered four
times identically as "companion - shown so the gap is visible", including on the row where it asserts
the opposite of the truth. And `export.opset` renders as **0** to a visitor, which is not a valid
opset - recorded here as observed; it is a display of an absent value rather than a wrong computation.

**What the walk could not cover.** The finance page was run and its CKKS agreement plot verified
(`encrypted_agreement 1`, 400 of 400 rows, 0 top-1 mismatches, 7 charts rendered), but the medical,
genomics and satellite pipelines were not run to completion, so their charts were confirmed present
rather than confirmed correct. The 5G-NIDD figures a visitor now sees are the RETRAINED ones
(0.982897 / 0.995019 at pin `a3cbe059`), which post-date this walk - the walk observed the 0.7609
model. **A second walk is owed on the redeploy** and is not claimed here.

## Round 51 (branch `round51-the-caption-swapped-one-wrong-claim-for-another`) - the second walk, and it found the fix from last round

**The walk Round 50 recorded as owed was performed**, on `index-CYDapC47.js` at `62e6b79` - the
content that merged as `9adcdf0`, so the served bundle is the merged tree even though the box's branch
pointer still needs returning. All five use-case pipelines were RUN to completion this time, which is
what Round 50 could not claim: it ran finance only and confirmed the other four as present.

| use case | published models | encrypted agreement | charts | NaN / dup headings |
|---|---|---|---|---|
| finance_aml | PASS gbm, PASS student | 1 (400/400) | 7 | none |
| medical_fl | PASS resnet18, resnet50, student | 1 | 1 | none |
| genomics | correctly reports no published model | **0.8** - see below | 6 | none |
| telecom_ids | PASS qml, PASS nidd (16,379 rows) | n/a | - | none |
| satellite_comms | PASS telemanom, opssat, opssat_student | none - structural | 2 | none |

**Genomics `encrypted_agreement` 0.8 is not a defect, and the run log already says why.** Five of
thirty rows flip, and the page reports that seven of the thirty are scored IDENTICALLY across classes
by the plaintext model itself, so their outcome is decided by tie-breaking; over the 23 rows the model
actually decides, agreement is 1. The widest plaintext margin among the five flipped rows is
**0.000e+0** - they are exact ties - and max absolute CKKS error is 8.277e-10, about 30 bits. The
encryption did not disagree with anything; it broke ties differently.

**But the metric row renders a bare 0.8** beside finance and medical at 1.0, and the sentence that
makes it readable is in the run log rather than beside the number. That is the same defect class as
`metric_note` rendering nowhere: a figure shipped without the caveat that makes it interpretable.
Recorded, not yet fixed.

### Round 50's own fix was wrong, and the deployed site is what showed it

Round 50 replaced a hardcoded companion caption - "shown so the gap is visible", which asserts the
companion is always the flattering number - with one DERIVED from the values. Deriving the direction
was right. Attaching an interpretation to it was not.

Measured on the deploy: `medical_fl.resnet18` renders `auc` 0.9147 as its declared metric and
`accuracy` 0.8654 as the companion, and the caption announced the accuracy as **"the pessimistic
figure, not the flattering one"**. AUC is rank-based and accuracy threshold-based; on an imbalanced
split AUC sits above accuracy by construction, and neither is pessimistic about the other. That
model's own `metric_note` - made visible by the same round - says *"Do not read the accuracy against
the AUC baseline"*. So the new caption contradicted the caveat it had been written to reveal.

Flattering-versus-pessimistic only means anything when both rows estimate the same quantity on the
same scale, and comparability is not decidable from the key names. The caption now states the
RELATION and stops: "above `auc` here", "below `macro_f1` here", "equal to `macro_f1` here", and it
points to the note where one exists. The relation is a fact; the reading is a judgement, and the
judgement now has `metric_note` to live in.

Three rounds running, this caption has been the same defect in three forms: a hardcoded claim, a
derived claim that over-reached, and now a stated relation. The pattern is asserting an
interpretation the code has not checked.

**`DP7` is NOT discharged this round.** The return-to-main deploy was blocked by a local permission
control, so the box remains on the Round 50 branch. The served content is the merged tree, so no
visitor sees anything stale; what is wrong is the branch pointer, and it is recorded here rather than
left to be discovered.

### Round 51, continued - the sweep items, each with what was actually established

**QCNN reproduction, verified rather than asserted.** All 33 QCNN gates pass with
`QSH_REQUIRE_BROWSER_GATES=1` set, so `test_the_qcnn_agrees_across_languages.py` RAN rather than
skipping - the browser's `qotpQcnnInfer` and the trainer's `circuit()` agree numerically on the
shipped artifacts. Readout identifiability holds at all five shipped widths, and no trainable
parameter is analytically dead. The per-artifact figures against their classical baselines, all
recorded on the pages: finance accuracy 0.97 against LR 0.985; medical 0.883721 against 0.953488;
telecom macro F1 0.510963 (8 wires) and 0.541894 (7 wires) against LR 0.976447. **The classical
baseline beats the circuit on every use case**, which is what the pages say and is the honest result.

**2D plots on the E2E page, measured.** `/orchestration` renders two: the encrypted-vs-plaintext
agreement scatter and the real-corpus 2D scatter, with `encrypted_agreement` 1 on the demo flow and no
`NaN`. It does NOT render `VerificationChart`, so the calibration diagram added this round appears on
use-case pages only - a scope fact, not a defect, and stated here so the next walk does not look for
it in the wrong place.

**The calibration diagram is NOT yet verified on the deploy.** It typechecks, its data is computed,
both call sites exist and the dead-component gate passes - but the deployed bundle predates it,
because this branch is unmerged and `DP7` is blocked. Recorded as locally-verified and
deploy-unverified rather than counted as done.

**JupyterLite / marimo / pyvista-wasm: the existing declination stands, and now has outside
corroboration.** `DC8` declines all three with a four-part reopening condition, recording that the
crypto stack does not exist in Pyodide - no SEAL/CKKS, no TenSEAL, no MPC runtime, no ONNX among 354
packages - and that `pyvista-wasm` does not exist as a real package. An independent 2026 search
agrees on both counts: JupyterLite and marimo are real and Pyodide-backed, and no credible
pyvista-to-WASM/WebGPU path surfaced. **No adoption work is warranted.** The WebGPU-3D half was
genuinely absent from the checklist - the literal string `3D` occurred zero times before Round 50 -
and is now `DC9`.

**Client-side-only inference, re-confirmed against the rows that assert it.** 14 `CS` rows cover the
six technologies the standing request names. WebNN is probed and never offered to ORT, which external
evidence supports (corrected 2026-11-05: that trial was disabled on 2026-03-27 before it ran, and WebNN stays behind a flag): it remains a Chrome Origin Trial (M147-M149) at W3C Candidate Recommendation with
no Safari or Firefox implementation. `ai.onnx.ml` forces the wasm build because the WebGPU build does
not register `TreeEnsembleClassifier`, so both telecom models report `onnx-wasm` by design.

**The round-trip workflow the standing request names is covered**: `RT0`-`RT11`, with `RT9` closing
the loop back to `RT0` on any public-site failure and `RT11` (added Round 50) requiring the local
implementation check before the walk. Host identity is deliberately absent from every row.

## Round 52 (branch `round52-the-credit-reached-the-artifact`, bundle `index-BrgAhT5P.js`) - walked, and the credit was observed rather than inferred

`DP7` discharged first: box returned to `main` at `e433371`, then the round branch deployed at
`e96a270`. All ten pages walked on the deployed site.

**Clean across every page**: zero `NaN`, zero `undefined`, zero `[object Object]`, zero duplicate
headings. Charts render on all five use-case pages (genomics 2, others 1 before a run, 8 on finance
after one).

**The licence fix was verified where it binds, in three places rather than one.**

| surface | observed |
|---|---|
| Hub card at the new pin | 1 attribution line on each of the four republished repos, fetched back from huggingface.co |
| satellite page | 4 Attribution lines - ESA-ADB twice, OPS-SAT-AD twice |
| finance page | 3 Attribution lines - AMLworld |

Fetching the card back from the Hub rather than reading the local copy is the point: they are
different objects, and assuming otherwise is the whole defect.

**Both republished finance models ran at their NEW pins and reproduced their recorded figures:**

> PASS finance_aml.gbm: re-scored all 8192 published held-out rows in this tab, minority_class_f1
> 0.54 against the recorded 0.54
> PASS finance_aml.student (not the selected model): same, 8192 rows

with `pinned to b35b4cbb2ad3` and `pinned to 475b160d9b42` rendered on the page - so the browser is
fetching the republished artifacts, not cached older ones.

**`metric_note` renders** on the telecom page, and the `/usecases` index still shows `vs published`
for the three models that have a baseline.

**What this walk did NOT cover, stated rather than implied.** Only the finance pipeline was run to
completion; medical, genomics, satellite and telecom were confirmed to render but not driven through
a full run, so their charts are confirmed PRESENT and not confirmed CORRECT. The satellite page's
missing encrypted-agreement plot is structural and unchanged.

**CORRECTION, made in Round 53 against the same deploy.** The sentence that stood here said
"`export.opset` still renders 0 on the telecom page - recorded in Round 50, still not fixed". That was
written from the Round 50 note rather than from the page, and it is false. Re-read on the deployed
site: the telecom page renders `export.opset 17` for `telecom_ids.qml` and **no opset row at all** for
`telecom_ids.nidd`; the finance page renders no opset row either. Round 50's observation of `0` was of
the PRE-RETRAIN 5G-NIDD record, whose `export.opset` was literally `0`; the Round 50 refit re-exported
it and the field is now `None`, which the panel omits rather than printing.

That is the correct behaviour and worth saying why: both are `ai.onnx.ml` tree ensembles, and such a
graph carries no default-domain (`ai.onnx`) opset, so there is no number to show. `0` was a wrong
answer, `None` omitted is the right one, and the item is CLOSED - not by anyone fixing it, but as a
side effect of the retrain. Recorded that way rather than claimed as a fix.

## Round 53 (branch `round53-the-pipelines-were-driven-to-completion`, bundle `index-BrgAhT5P.js`) - the four pipelines Round 52 only rendered

Round 52 recorded that only finance had been driven to completion and that the other four were
"confirmed to RENDER and not confirmed to be CORRECT". All five have now been run end to end on the
deployed site.

| use case | published models re-scored in-tab | charts | result |
|---|---|---|---|
| finance_aml | `gbm` 8192 rows, `student` 8192 rows | 8 | PASS both, at the new pins `b35b4cbb` / `475b160d` |
| medical_fl | `resnet18` 156, `resnet50` 156, `student` 156 | 8 | PASS all three |
| genomics | none - it registers no browser model | 6 | correct by construction, and the page says so |
| satellite_comms | `telemanom` 128, `opssat` 529, `opssat_student` 529 | 2 | PASS all three |
| telecom_ids | `qml` 600, `nidd` 16,379 | 4 | PASS both |

Zero `NaN`, zero `undefined`, zero `[object Object]`, zero duplicate headings on any of them.
`/orchestration` renders its 10-node DAG clean.

**The QRNG fix is visible on the deploy**, on every use-case page, and it reports the source rather
than the extractor:

> PASS per SP 800-90B 4.4 (repetition-count 11/23, adaptive-proportion 529/612); SP 800-22 reported
> not gating: pass

**`satellite_comms.student` is absent from the satellite row above and that is the recorded state**,
not an omission: it is scored on event-wise F0.5, a sequence metric the per-row evaluator cannot
compute, so it reports DID NOT RUN with its reason rather than substituting a figure.

## Round 55 (branch `round55-the-reference-line-could-never-draw`, bundle `index-BYWN9-KK.js`) - a feature that rendered nowhere, found by looking

Round 54 merged, box returned to `main` at `770b533`, new bundle deployed and walked.

**The calibration diagram works, and this is the first time anyone has seen it.** Round 51 added it;
it had never been deployed. Confirmed on the live site from the rendered figure captions:

> Calibration of `finance_aml.gbm`: mean p(positive) against the frequency...
> Calibration of `telecom_ids.qml`: mean winning-class confidence against the frequency...

Both branches, both correct: binary bins p(positive), multi-class bins winning-class confidence.

**And the baseline reference line added the same round rendered NOWHERE.** Queried the SVG text on
finance and telecom after full runs; the only `published` string on either page is the model-run
label. The cause is structural: every model with a baseline plots a PR curve, every model plotting
per-class F1 has a null baseline, and the intersection is empty. The Round 54 gate passed because it
asserted which FIELD was read, never that the result appears.

Replaced with `MeasuredAgainstPublished` - one axis, both bars the same metric, the metric named on
the figure. **Seven models can render it**: `finance_aml.gbm`, three medical, three satellite. The
gate now asserts reachability, so an unreachable comparison fails.

Pipelines re-run on this bundle: finance (8 charts) and telecom (4 charts) both to completion, zero
`NaN`, zero `[object Object]`.

**What this walk did not cover.** Medical, genomics and satellite were not re-run on THIS bundle -
they were driven to completion on the previous one in Round 53 and nothing between the two rounds
touched their pipelines. Stated rather than implied, because "re-verified" and "verified last round
and unchanged since" are different claims.

### Round 55, continued: QCNN on the deploy, and what "the same footing as CKKS" actually means

Driven on the live site: finance with `Inference = QCNN`, Parameters panel opened, `#pp-inf` set to
`qcnn`, full run.

**It runs and reports its structure**, which matches the Cong/Hur architecture the project cites:
`<Z0> = -0.6622`, **77 shared-filter angles over 8 wires, 3 conv rounds per stage**, state-vector
simulation. Advantage 10/10, class 0, no `NaN`.

**And it reports a comparison, which is honest and is not what `CS13` asks for.** Verbatim:

> Held-out minority-class F1 0.7353 against a baseline of 0.0983 and logistic regression at 0.8657 on
> the same split - the classical model is ahead, and this reports what was measured rather than what
> would flatter.

That is a SAME-SPLIT classical comparison. `CS13` asks for QCNN "against literature on the same
footing as CKKS", and the published QCNN figure this project cites (`QC18`, Sci Rep 14 (2024), QCCNN
0.96155) comes from a different corpus and a different task. Putting it beside a PaySim
minority-class F1 would be the cross-corpus category error `CS14` forbids, and the same error Round 55
just removed from the chart layer.

So the asymmetry is real and is arguably the RIGHT way round: QCNN is compared against a classical
model on the data actually in front of it, which is more informative than a number from another
dataset. `CS13` is annotated rather than ticked, because the row asks for a footing the evidence says
should not be adopted.

### Round 55: the full-page audit, all ten pages on the deploy

Systematic pass over every route, checking what the standing request names - content, terminology,
numbers, buttons, results, duplicates - rather than the `NaN` spot-check earlier rounds settled for.

| page | chars | headings | buttons | unnamed | dup headings | bad tokens |
|---|---|---|---|---|---|---|
| `/` | 3,802 | 7 | 0 | 0 | none | none |
| `/value` | 7,615 | 14 | 0 | 0 | none | none |
| `/validations` | 1,793 | 9 | 6 | 0 | none | none |
| `/usecases` | 5,050 | 6 | 0 | 0 | none | none |
| `/orchestration` | 2,332 | 4 | 15 | 0 | none | none (10 DAG nodes) |
| five `/usecase/*` | - | - | 14 each | 0 | none | none |

"Bad tokens" scans for `NaN`, `undefined`, `[object Object]` and `Infinity`. Zero across all ten.
Zero duplicate headings anywhere. Zero buttons without an accessible name.

**Two things that looked like defects and are not.**

`/usecase/nope` did not match the wording this log recorded in Round 49 ("There is no use case called
X. Showing Finance"). It has been REPLACED by something better and the old record is what is stale:
the page now renders `h1` = "No such use case", lists all five valid ids with their titles, and offers
a route back to the index. It no longer silently substitutes Finance, which is the behaviour Round 28
recorded as a defect. Correcting the record rather than the page.

A checkbox on the finance page carries no `id`, which an automated scan flags. It is wrapped in a
`<label>` reading "mix QKD-domain key", so the association is valid and the control is named. Recorded
because an `id`-based a11y check would report it as a fault and it is not one.

### Round 55, continued: three flags re-verified as wrong, and a chart that committed the error it was built to prevent

**The eight open fact-check items were re-verified at source before any of them was applied, and three
of them were themselves wrong.** That ratio is the argument for the step. The GNN-under-FHE compile
failure stands - its own abstract says quantization and pruning "were used to render the GNN
FHE-compatible", which reads like success, while section 4.4 says no circuit was ever produced; the
abstract is the trap and the line citing it was right. ZKBoost was described correctly - "proving
training occurred" is what a zero-knowledge proof of training is. And the 8-class taxonomy grounding
is confirmed, with all eight classes listed in the paper's section III-C.

**Of the five that held, one broke a second claim as it fell.** "Exactly two published
BreastMNIST-under-encryption baselines" was wrong: PriMed (*Expert Systems* 42(1):e13283) is
chronologically FIRST, three years before the two that were listed - and it performs a genuine
homomorphic convolution. So the sentence resting on the count, "neither performs a genuine homomorphic
convolution, this project does", fell with it. A claim of the form "nobody else does X" is only ever as
good as the search behind it, and this one had a search behind it that stopped at two. What replaces it
is a cost claim, which is checkable rather than exhaustive: PriMed pays 294.2 MB of encrypted input and
a mid-inference re-encryption round-trip under BFV, against CKKS at $`N = 8192`$ with no round-trip. A fourth,
Gecko (arXiv:2608.02378), is listed with the caveat that has to travel with it - its 413 ms never
encrypts the image, only the features.

**`MeasuredAgainstPublished` was built to stop a category error and committed one.** Its caption
asserted "both bars are the same metric on the same axis" for every model. For
`satellite_comms.telemanom` that is false, and the registry says so in three separate fields - three
scored events against ESA-ADB's 29, a different split, and a time-level TNR factor applied there and
not here. Its own `sample_size_note` reads "NOT comparable to a published benchmark computed over the
full event set". The chart drew 0.4545 beside 0.968 anyway.

The first fix for that was wrong and worth recording. Keying the caveat off "does the model carry a
`sample_size_note`" looked structural, and it is not: `telecom_ids.nidd` carries one too and its note is
a REASSURANCE - stratified 75/25, every class at corpus proportion. One field name, two opposite
meanings, which is the same trap `tests/test_two_baselines_are_not_one.py` exists for. The caveat is now
an explicit per-model registry field.

**And `sample_size_note` rendered nowhere at all.** Round 50 found `metric_note` exported to the browser
and displayed on no page, and fixed it. That fix reached ONE field: `sample_size_note` and
`classical_rf_attribution_note` were in exactly the same state five rounds later. The new gate
enumerates note fields FROM THE CONFIG rather than from a list - a hard-coded list would need updating
by the same person who forgot to render the field. It also strips comments before matching, because the
first version passed a seeded revert: the comment in `ModelPanel` explaining why the field must be
rendered NAMES the field, so deleting every render site left the identifier in the file and the gate
green. That is the QRNG gate's failure repeated, and catching it in the same hour is the only reason it
is not a tenth entry in the register.

### Round 55, continued: QSDC delivered on a link it had no power to judge

**118 times in 200.** The abort was decided on the point estimate of 29 check carriers. On the shipped
default channel the true QBER is 0.01520 against a tolerance of 0.00241 - the link provably cannot carry
QSDC - and an 11-byte message delivered 59% of the time, because `0.9848^29 = 0.64` makes a zero-error
sample the common case. Not a wrong threshold: a threshold applied to a number that does not mean what
the comparison assumes.

It now runs on a one-sided **Serfling** bound at `eps = 1e-9`, sampling without replacement because the
check carriers come from the same finite block. **Recorded as this project's own conservative choice
rather than as a citation**, because the QSDC literature has no finite-size parameter-estimation
procedure and says so in print - Pan et al., Photonics Res. 8, 1522 (2020) call it "an interesting
direction for future research", the IEEE COMST survey calls it "currently under investigation", and
Long's own group in Entropy 27(4), 332 (2025) adheres to "the asymptotic limit, assuming an infinite
block length". The two papers that do carry finite-size QSDC analysis are about the wiretap CODE and
about DECOY INTENSITIES; citing either here would misrepresent them.

Aborts attribute three ways now. `statistics_limited` says the sample could not tell, which is not a
statement about the channel, and reporting it as an eavesdropper or a link-budget failure would be a
claim the data does not support.

**Separately, Eq. (8) takes `h(e_x + e_z)` and this used a pooled rate.** The paper's own arithmetic
settles it: `I(A:E) = g*Q_Bob*h(e_x+e_z) = 9.1e-4` at `g = 2.57`, `Q_Bob = 0.3%`, `e_x = e_z = 0.8%`,
and only `h(0.016)` reproduces it at 9.1248e-4 against 4.08e-4 for the pooled `h(0.006)`. Pooling
understates Eve by 2.24x and overstates the secrecy capacity by 26%. **That is the third permissive
correction to one formula**, after the 0.11 threshold and the omitted detector efficiency, and all three
erred in the same direction - a simplification that makes a demo work is not a neutral coin flip.

**The fix then missed a call site inside the file it was being applied to.** `clientCompute.ts`
evaluates the capacity twice - once for the measured channel, once with Eve removed to attribute the
abort - three hundred lines and a `switch` arm apart, and the commit whose comment said "fixed in BOTH
languages in one change" reached one of them. Both now call one exported `qsdcSecrecyCapacity`, pinned
by a gate that also reproduces the paper's own number, because two implementations can agree with each
other and both be wrong, which is exactly what had been happening.

**What the bound costs, stated rather than tuned around.** A 2 km fibre link at misalignment 0.001 needs
a **32 KB** block before a certificate is possible - 8 KB gives $`C_s = -0.485`$. The satellite downlink is
reachable only at zero atmospheric AND zero pointing loss with a **4 MB** block. The demo sends **74
bytes**. Qi et al.'s own experiment runs blocks of 1,312 x 830 = 1,088,960 pulses, so the requirement is
not an artefact of strictness - it is the regime the published experiments already occupy.

The satellite test's previous docstring, written earlier in this same round, said the fixture sat "far
enough from the boundary that finite-sample fluctuation cannot flip the verdict". That treats finite
sampling as noise to steer around. It is a certification limit and no distance from the boundary escapes
it.

### Round 55, continued: two claims about the outside world that were wrong

**`pyvista-wasm` exists.** It was dismissed here as a "third-party API lookalike in a personal
namespace". It is version 0.12.0 as of 2026-08-21, Apache-2.0, by `tkoyama010` - the #5 all-time
contributor to `pyvista/pyvista` - wrapping Kitware's own `@kitware/vtk-wasm`. A personal namespace was
read as evidence of illegitimacy when it was only evidence of a namespace. It stays declined, on grounds
that survive checking: pre-alpha, two CDN origins by default, a 22 MB payload pinned two majors stale,
and no three-dimensional data in this project to render. All five `real_2d.json` are `[x, y, class]`
with a binary label in the third slot.

The Pyodide half of that decision is stronger than before, not weaker: 356 packages in the 314.0.5
lockfile, none of TenSEAL/OpenFHE/SEAL/concrete-ml/onnxruntime present, and none of them has ever
published an emscripten wheel. What did change is that JupyterLite and marimo were being dismissed
together when only marimo's CDN blocker is unfixable - JupyterLite supports self-hosting three ways.

**All three MPC dependencies are archived upstream**, last pushed mid-2025. `threat-model.md` had
already been corrected once, replacing "JIFF is unmaintained and the PSE MPC Framework sunset" with
these three as the maintained replacements - and the replacements have since been sunset in turn. A
correction goes stale exactly like the claim it fixed. `RS21` runs this query for submodules and nothing
ran it for npm packages, which is why a year passed unnoticed; `CS15` now covers it.

### Round 55, continued: the deploy walk, and the outcome check Round 54 did not do

Deployed `main` at `6b94278`, bundle **`index-D9K2fM4m.js`** (previously `index-BYWN9-KK.js`), box on
`main`, public URL 200.

**`MeasuredAgainstPublished` renders, and it discriminates per model.** Round 54 asserted that a
published baseline was read from the right field and shipped a feature that rendered nowhere. The
difference this time is that the chart was observed on the deployed site, per model, with the
caveat branch checked against the non-caveat branch:

| model | caption | x-axis | bars | caveat footnote |
|---|---|---|---|---|
| `satellite_comms.telemanom` | "share a metric **name** and are **not** the same measurement" | `this project`, **`published*`** | 2 | **yes** |
| `satellite_comms.opssat` | "both bars are the same metric on the same axis" | `this project`, `published`, `ceiling` | 3 | no |
| `medical_fl.resnet18` | "both bars are the same metric on the same axis" | `this project`, `published`, `ceiling` | 3 | no |
| `finance_aml.gbm` | "both bars are the same metric on the same axis" | `this project`, `published`, `ceiling` | 3 | no |

The star and the footnote appear on exactly one model and the ceiling bar appears on exactly the
three that carry one. **A feature that renders identically everywhere is not evidence that it is
reading anything** - two branches observed to differ is.

**The chart is gated behind a RUN, which is why "it renders" needed checking at all.** On page load
`MeasuredAgainstPublished` returns null: it lives inside the `artifacts.verification` block, so
nothing draws until a pipeline run completes and a published model has been selected. A verifier who
loads the page and looks will see zero charts and conclude the feature is dead. That is the shape of
`RI8` - a panel that exists and no user action reaches - and here the action exists but is not
obvious. Recorded so the next walk does not repeat the confusion.

Also observed on the deploy: `Sample size:` renders on `satellite_comms.telemanom` and correctly does
NOT render on `medical_fl.resnet18`, which carries no such note - the field-presence conditional
works rather than always-firing. `Reading these metrics:` renders on medical. `Gap to baseline:`
renders on finance. And the QSDC stage emits
`asymptotic (infinite block; no finite-size certificate)`.

### Round 55, continued: an exhaustive audit, and four gates that could not fail

A 77-agent audit across eight areas of the standing request, every finding put to an independent
adversarial verifier told to refute it. **51 findings survived; 17 were refuted.** Each of the six
acted on below was then re-verified by hand, by seeding the defect and watching the gate stay green.

**The PQC conformance gate could not detect a totally broken ML-KEM.** Its ML-KEM half read one
outcome - did `encapsulate` throw - and never encapsulated to a WELL-FORMED key. Seeded by making
`encapsulate` throw unconditionally, it printed `780 malformed keys, 780 rejected, 0 accepted  OK`
and exited **0**. The ML-DSA block eight lines below already had a keygen-sign-verify round trip, and
its own comment says why: a noble `0.2.x` to `0.6.x` argument-order change once broke every signature
while TypeScript and the production build stayed green. `web/package.json` pins `^0.6.1`, so the same
class of break is in range for ML-KEM. The lesson reached one of the two primitives in one file.

**The SP 800-90B adaptive-proportion cutoff was one too high in the browser, at every entropy.**
`1 + c` where the loop already held the answer; the `1 +` belongs to the Python copy, where
`binom.isf` returns one BELOW the cutoff. Measured across the grid: 991/990, 793/792, 626/625,
590/589, 325/324, 106/105 - **permissive**, because a higher cutoff fails fewer windows, and
visitor-facing, because this is the client-side-only demo and `healthPassed` is computed from it.

The reason nothing caught it is the part worth keeping. There WAS an assertion:
`repetitionCountCutoff(2.0, 2^-20) === 11`, which is correct, is SP 800-90B section 4.4.1's own
worked example - and is on the SIBLING function. **One point on one of two functions cannot see a
uniform offset in the other.** The fix is a 13-entropy grid for both functions, exported from Python
by `scripts/export_qrng_cutoffs.py`, regenerated in CI, and compared entry by entry.

**Two CI gates reported success for work that did not happen.** Gate 2 ran `export_config.py` with
`>/dev/null 2>&1` and its status discarded, then diffed the artefact. Both generators write their
output LAST, so an exception leaves the committed file byte-identical and the diff returns 0: seeded
by raising at the top of the generator, exit 1, verdict **PASS "in sync"**. Gate 15 piped pytest into
`grep -c` wrapped in `|| true`, so only the skip count survived: seeded by deleting a gate file,
pytest exits 4, the skip count is 0, verdict **PASS "its four gates ran here"**. These were the only
two commands in a 206-line script whose status was neither tested nor logged - which made `DP2`,
"the build's exit status is captured and printed, never redirected to /dev/null", broken inside the
script that enforces it. Gate 15 now also requires a minimum RUN COUNT, because a deleted file
reports zero skips exactly like a clean run.

**A fabricated `accuracy=1.0` on the row the full deployment serves.** `run_quantum` returned it from
a `correct += 1` that was unconditional and never read. There is nothing to be accurate about - the
QCNN has no trained weights, and `private_inference.py` says in its own docstring "Nothing is
decrypted and nothing is compared". The browser twin had already been corrected to `accuracy: null`,
with a comment recording the three rounds it had been wrong; **that fix reached one language**, and
`usePipelineRun.ts` PREFERS the Python artifact, so the identical row of the identical table printed
1.0 through one transport and an em dash through the other. The flattering side of the contradiction
was the one that shipped.

**The BreastMNIST licence downgrade reached one of four copies.** On 2026-08-10 it was checked and
downgraded - MedMNIST licenses ITSELF CC BY 4.0, but the upstream BUSI grant attaches to the ARTICLE,
and a citation request is not a grant. `registry.py` and one row of `DATA_LICENSES.md` carried it.
The row whose own text reads "the same corpus as the ResNet-18 row above" said a flat "CC BY 4.0 /
Yes". So did the redistributed-eval-sample paragraph. And `scripts/prep_model_inputs.py` hardcoded
the flat string into the `provenance` of every published `eval_sample.json` - **the surface where the
claim actually binds**, because that file ships inside three Hugging Face repos. The word
"unverifiable" appeared exactly once in a 428-line licence document.

**The first version of the licence gate cried wolf and had to be narrowed.** Scanning every line
that mentions the corpus fired on three that merely REFER to BreastMNIST while stating a different
corpus's licence - the PneumoniaMNIST row saying it is "a different subset from BreastMNIST", the
WDBC overlay row, and the paragraph where the corpus and its qualifier sit on adjacent wrapped lines.
A gate that fires on correct text gets disabled, so it now reads the licence CELL of the row whose
DATASET cell names the corpus.

**And the new gate 15 immediately caught my own mistake**: two prefixes added to the checklist without
adding them to the legend. That is the gate doing exactly what the old one could not.

### Round 55, continued: three visitor-facing labels, and the guard the checklist said it had

**A badge said WASM for a loop that is plain JS.** Telecom emitted `measured-wasm` for
`Math.exp`/`Math.max` arithmetic over a JS LCG, and the page rendered "MEASURED (local WASM) /
Computed in this browser tab on the WebAssembly runtime". The twin had been corrected for this exact
reason and carries the reason in its own comment - "Claiming WASM here made every pipeline run on
every page render 'MEASURED (local WASM)' without a wasm module ever being instantiated" - with
`ProvenanceBadge.tsx` adding "the claim was wrong in the direction that flatters". The fix reached one
of two sites, and telecom has no `private_inference` stage, so nothing downstream overwrote it.

**A latency was printed under the name of a runtime that did not produce it.** The FHE stage logged
`ONNX ${clientMs} ms`, where `clientMs` times `privateInference` - which has had no ONNX tier since
Round 17 and says so in its own comments. The genuine ONNX latency lands in the SAME stage log, so
one line carried two ONNX-labelled numbers differing by orders of magnitude. The sibling emit fifteen
lines below already said "client compute".

**Selecting PQC-only on satellite deleted four panels.** `break` inside a `switch` case exits the
whole case, so the PQC-only guard skipped not just QSDC but the anomaly verdict, the SHAP panel and
the published-model row - and `usePipelineRun` offers that profile on every use case. It contradicted
the comment on the block it was skipping, which argues at length that the step "scores a window of the
BUNDLED corpus... so it has no dependency on whether QSDC aborted". Round 31 moved that block out of
`if (!abort)` for exactly that reason and left it inside the PQC-only return: the same fix, one of two
call sites.

**And the duplicate-ID guard the document claims to have does not exist.** The legend's history
paragraph ends: `RG1`-`RG4` were two families at once, "fixed by renaming the newer one to `AR`, and
**by making the build fail rather than trusting anyone to notice**". Nothing implemented that. No
test, no script, no Makefile target scanned for duplicate row IDs. Round 55 alone produced four
collisions - `CS13`/`CS14`, `MN7`, `QS9`-`QS11`, `RG12`/`RG14` - every one caught by a manual
`sort | uniq -d` and none by a gate.

That is precisely the failure `test_checklist_prefix_legend.py`'s own docstring names: "the rule is
right, greppable, and has no executor." It was written about the register deadline, `DP7` and `CI1`'s
step count, and was true of the paragraph three lines above it.

**A correction to something I asserted mid-round.** I reported a *pre-existing* `RG12` collision in
this document. There was none - the duplicate was my own new row, and I had misread my own scan
output. The gate now says so unambiguously by printing both line numbers.

### Round 55, continued: the audit fixes verified on the deploy

`main` at `cc9a1c6`, bundle **`index-B8CifKLO.js`**, box on `main`, public 200. Each fix checked by
observing the thing it was supposed to change, and the thing it was supposed to leave alone.

**The provenance badge, and the discrimination is the evidence.** The telecom page now carries TWO
badges saying different things, each correct:

    MEASURED (plain JS) . in-browser softmax training (clientIDS), synthetic telemetry
    MEASURED (local WASM) . onnx-wasm

The JS training loop reads plain JS; the published ONNX model, which genuinely does run on the wasm
runtime, still reads local WASM. "WebAssembly runtime" no longer appears for the JS path. A single
badge changing would not have shown that the downgrade was targeted rather than blanket.

**PQC-only on satellite now skips one stage and keeps the page.** With that profile selected: the
QSDC skip message renders, including its new sentence that "the telemetry anomaly verdict and its
explanation still run below"; the anomaly verdict renders; the SHAP panel renders; the
published-model row renders; three figures draw; and `qsdc_secrecy_capacity` is correctly ABSENT.
Before this, all four of those panels vanished with no explanation.

**The ONNX label is gone from the client timer.** On genomics the FHE line renders and reads
`plaintext reference <ms>`; `, ONNX <n>` no longer appears anywhere on the page. That line used to
carry two ONNX-labelled numbers differing by orders of magnitude, one of which was two plain-JS dot
products.

### Round 55, continued: corrections that had already been made somewhere else

Every item here is a claim this project had **already corrected in one place** and left standing in
another. That is the round's dominant finding at document granularity rather than at code granularity.

**An un-re-verified figure was being published to the Hub.** `DATA_LICENSES.md` records that this
project has carried two different spans for ESA-ADB - 14 years there, 17.5 in the registry - and that
neither is re-verifiable: the Zenodo record, arXiv:2406.17826 and the upstream README all decline to
state it. That file **withheld** the number. `registry.py` kept asserting it, and `publish_hf.py`
copies `DatasetSpec.notes` verbatim into the model card, so the withheld figure appears in
`models/satellite_telemanom/README.md` and `models/satellite_telemanom_student/README.md`. **A figure
withheld everywhere except the copy that leaves the repository is not withheld.**

**Two README copies of a generalisation `DATA_LICENSES.md` explicitly withdraws as WRONG.** That file
records: the sentence "no commercial-OK real network-attack corpus exists" generalised two true facts
about UNSW-NB15 and CIC-IDS into a claim about the field, and **5G-NIDD falsifies it** - CC BY 4.0,
and this repository trains on it. The README said it twice, and `5G-NIDD` appeared nowhere in the
README at all.

**The checklist's orientation paragraph said telecom runs no CKKS.** The MPC half is right; the CKKS
half is false, and four other passages in the same file already carry the correction. **Measured on
the deployed site** at bundle `index-B8CifKLO.js`: the telecom page emits `encrypted_student_ms`,
`encrypted_batch_ms` and `fhe_ms_per_rotation` from real node-seal. A correction landing in the
detailed rows does not reach the paragraph a reader orients from, and the orientation paragraph is
the one that misleads. The same paragraph called BB84/E91 "state-vector"; both are analytic channel
models with a Gaussian-sampled QBER, and the adjective belongs to the QRNG named beside them.

**"Clifford+T QCNN" and "T-gate count" for a circuit with zero T gates.** `qcnn.py` builds RY/RZ
rotations and CNOT entanglers; `t_count()` is a Solovay-Kitaev BUDGET - `ceil(3*log2(1/eps))` per
rotation times the parameter count - which its own docstring calls "rough" and "-ish", returning 630
at 8 qubits. `report/build_data.py` records this correction as *fixed* and
`docs/deliverable-verification.md` names README and architecture.md as the two files corrected. This
module was the third copy. `BenchmarkPanel` also asserted "counted from the circuit" for both rows of
one table, which is true of the browser row (56, genuinely counted) and false of the Python one.

**A deleted `def` line merged an anti-tautology test into its neighbour.**
`test_the_comparison_would_notice_a_divergence` lost its header but not its body, so collection read
10 instead of 11 and `pytest -k` on that name matched nothing, while
`docs/deliverable-verification.md` went on asserting the test exists. Detection was never lost - the
body still ran - but a check nobody can ask for by name is one nobody will notice losing.

**`DATA_LICENSES.md` said 5G-NIDD was "adopted, NOT yet implemented", that "no code path loads it
yet", and that it has "52 columns".** All three false: `datasets.py` fetches it,
`scripts/train/telecom_nidd.py` trains on all 1,215,890 flows, and the file has **113 columns**, 66
dropped, **47 features**. The attribution obligation itself was discharged and renders in
`ModelPanel`; only the prose went stale.

**`references/satellite-event-count-investigation.md` still opens on the retired comparator.**
Everything in it is framed against ESA-ADB's lenient 0.786 from Table 2, which Round 24 replaced with
Supplementary Table 9's like-for-like 0.968 across the registry, README, both model cards, config,
`build_data.py` and the trainer - and `README.md` links here in the present tense. Banner added,
following this project's own convention. The denominator argument is unaffected and is why the file
exists; three scored events against ESA-ADB's 29 holds against either comparator.

**And the MathJax audit measured a proxy for the property.** Round 54 concluded "the conversion is
largely DONE" from a grep for **backticked** `x = y`. Bare prose mathematics has no backticks and is
invisible to that pattern by construction, so the probe could only ever return the already-marked-up
cases - mechanism rather than outcome, again. An outcome-shaped scan finds four bare sites the grep
cannot, each of them within three lines of its own converted twin. Converted in the house
`` $`...`$ `` form; the first attempt used plain `$...$`, which is the delimiter this file's own
style note reserves for display math.

### Round 55, continued: a config constant wearing a measurement's name, and a workaround whose reopen condition had quietly been met

**`fhe_max_abs_error` was emitted from `config.defaults.json` on one branch and from a real
decryption on the other.** The fallback published the literal `4.6e-07` under the identical key the
measured path fills with `6.20e-13` / `1.93e-9`. The only distinguisher was the log line beside it,
and `downloadMetricsCsv` exports `{metric, value}` pairs with no log - so the CSV a reviewer opens
showed a config default indistinguishable from a measurement, under a key `docs/threat-model.md`
describes as the computed CKKS approximation error.

Worth recording that the direction cuts AGAINST the usual pattern: `4.6e-07` is three to four orders
WORSE than measured, so it understated precision. It is still a constant wearing a measurement's
name. Renamed `fhe_max_abs_error_representative`, with an `encrypted_tier` status beside it following
the `published_model_status: "failed"` convention already in the file. The latent twin on the bench
row - which set the same constant UNCONDITIONALLY, measured branch included, and reaches no component
today - is renamed too, because "nothing renders it" is a property of the current UI and not of the
value.

**A gate written earlier this same round had two defects.** `test_qsdc_agrees_across_languages.py`
used a bare `pytest.skip`, making it the only one of the three cross-language browser gates not
escalated by `QSH_REQUIRE_BROWSER_GATES=1`: measured with the variable set and esbuild absent, the
two twins FAIL and it SKIPs. And its skip message claimed "this skip is why `check:qkd` is also a CI
gate" - `qsdcSecrecyCapacity` appears **zero** times in `web/scripts`. Nothing in the web job covers
that identity. **Citing a compensating control without checking it is the same move as citing an
authority for a number that is not in it**, which `RI11` exists for, and I made it while writing a
gate about one-language fixes.

`scripts/ci_local.sh` step 1 now exports the variable too. This machine has esbuild, so it costs
nothing and turns a silent skip - from a pruned `node_modules`, a failed `npm ci`, a renamed binary -
into a failure.

**And the OpenFHE EL9 workaround has met its own reopen condition.** It was recorded as a deliberate,
permanent, host-specific workaround for a SIGSEGV inside `EvalBootstrapSetup` on AlmaLinux 9.7, with
the condition written down: "reopens if the crash stops reproducing on the pinned OpenFHE version on
EL9". Nothing watched for it. Measured on this EL9 host with `QSH_REQUIRE_OPENFHE_BOOTSTRAP=1`, three
consecutive runs: **16.85 s, 16.05 s, 16.82 s, all exit 0.** The crash does not reproduce.

That also closes a gap the script's own header claimed away. `ci.yml` runs that test as a blocking
job; step 1 here deselects it with `-m "not slow"`, so "runs every blocking CI gate" was false for
exactly one job - and **nothing anywhere compared the two step sets**. It is step 17 now, kept
separate rather than folded into step 1, so a regression names which platform changed.

A condition with no executor is an intention. That is the third form of the same finding this round,
after `DP7` and the duplicate-ID guard the checklist said it had.

### Round 55, closing: the Hub still carried both corrections' pre-fix text

**The audit said it never inspected Hugging Face, and that turned out to be the gap that mattered.**
Its own coverage note reads: "Hugging Face Hub state was not inspected... the findings assert what
*would be* published from this tree." Checked against the live Hub: both satellite cards still carried
**"17.5 years"**, the span `DATA_LICENSES.md` withheld as un-re-verifiable, and the three medical
cards still stated a flat **CC BY 4.0** for BreastMNIST, the grant downgraded on 2026-08-10.

**Fixing the registry does not fix the published artifact**, and that is the whole point of the
licence finding: `eval_sample.json` ships inside three model repos, so the string a third party
receives is the one that binds. Round 51 recorded this same lesson in its own plan and it had to be
learned again on a different field.

Republished five models. Verified by fetching the cards back:

    qsh-medical-breastmnist-resnet18        licence hedged, YAML license_name cc-by-4.0-asserted-by-medmnist
    qsh-medical-breastmnist-resnet50        licence hedged
    qsh-medical-breastmnist-ckks-student    licence hedged
    qsh-satellite-telemanom-esa             span withheld, "17.5 years" gone
    qsh-satellite-telemanom-ckks-student    span withheld, "17.5 years" gone

All five return 200 to an anonymous fetch, and the five new `hf_revision` pins resolve 200 with the
ONNX weights reachable at the pinned revision.

**`--public` was mandatory and checking that first mattered.** `publish_hf.py` calls
`update_repo_settings(private=private)` explicitly, precisely because `create_repo(exist_ok=True)`
does NOT change visibility - a fix recorded in the script's own comment. Without the flag it would
have flipped five currently-public repositories to private, and the demo fetches its weights from
them, so the live site would have broken while every local gate stayed green.

**A third false positive, recorded like the other two.** The probe reported the medical cards as
still carrying a flat licence. It was matching the YAML slug
`license_name: cc-by-4.0-asserted-by-medmnist` - hyphenated, so a regex looking for `CC BY 4.0` not
followed by " as asserted" hit it. The card is correct in four places: the YAML field, the Licence
line, the Attribution parenthetical and the full argument. Three false positives this round - an
unanchored `ARR` matching "carried", `encrypted_tier` matching `encrypted_tier_measured_ms`, and this
- every one caught by reading what actually matched instead of trusting the boolean.

### Round 55, closing: the unresolved lead resolved, and it cleared the file it was filed against

The audit left one lead open: `src/qsh/models/datasets.py` states that "the same method scores
event-wise F0.5 of 0.786 here and 0.061 on all channels", and the auditor could not tell whether
0.061 comes from ESA-ADB's Table 2 or its Supplementary Table 9. Neither reported nor cleared.

**Resolved by reading arXiv:2406.17826v2, and the sentence is CORRECT.** Both figures are Table 2,
same panel, same event set, differing only in channel scope:

    Table 2 (all events excluding communication gaps), Telemanom-ESA-Pruned, Mission1
        lightweight subset, channels 41-46 ....... 0.786
        full set of channels ..................... 0.061

    Supplementary Table 9 (anomalies alone) ...... 0.968 lightweight / 0.039 full

`0.061` occurs three times in the whole PDF, all inside Table 2, and appears nowhere in Supplementary
Table 9. So the pairing is within-table and there is no category error. **And swapping 0.786 for the
Round 24 comparator 0.968 would CREATE one**, because 0.968's full-channel partner is 0.039.

**The defect was next door, and it was latent.** `scripts/train/satellite_telemanom.py` read the
lightweight half from `spec.metric.baseline` - which Round 24 correctly moved to Supplementary Table
9's **0.968** - while keeping `0.061` as a literal. That pairs two tables and two event sets under a
label whose entire meaning is that only the channel scope differs. It had not fired because the
shipped `models/metrics.json` still holds the consistent Table 2 pair: the trainer has not run since
the baseline moved. The next run would have written the mixed one.

**The shape is worth more than the instance.** Replacing a literal with a registry read looks like
removing duplication. Here it silently swapped which table one half of a matched pair came from, and
it did so at the moment the registry value was being corrected - so a fix to one number broke a
different number that had been reading it. Three prose sites carried the same implication and are
corrected; `datasets.py` needed no change and is the one place that stayed consistent, precisely
because it kept 0.786.

**The duplicate-ID gate caught its third collision of the round**, again before it shipped. It has
now fired on `CS13`/`CS14`, `MN7`, `QS9`-`QS11`, `RG12`/`RG14`, `CI9`/`CI10` and `MS9` - every one my
own, and every one invisible to the previous four rounds of manual scanning.

### Round 55, closing: a republish is not a deploy

Measured immediately after republishing five models, before rebuilding the site: the repository
carried **11 `hf_revision` pins and the served bundle contained 6 of them.** The five republished
revisions were absent.

That gap is invisible from every direction that was being watched. The Hub cards were correct, the
repository was correct, all seventeen local gates were green, and the deployed page fetched weights at
the OLD revisions - which is the artefact a visitor actually receives. Nothing on the page said so,
because nothing on the page knows which revision it wanted.

`HF15` records the check as a set difference rather than a reminder: extract the 40-hex strings from
the deployed `index-*.js`, subtract the `hf_revision` values in `config.defaults.json`, and require
zero missing. It is the Hub half of the lesson `DP4` already carries for source changes - a change
that has not reached the served bundle has not reached anyone - and the two halves had never been
stated together.

**The duplicate-ID gate fired for the fourth time this round**, on `HF14`. Every one of the four has
been my own new row, and every one was caught before it shipped. Four rounds of manual scanning had
never caught one.

### Round 55, closing: the combiner branch every run takes had no vectors, and the reason was mechanical

`transport._mix_qkd` reads `crypto.hybrid_combiner_mode`, which **defaults to `hkdf`**. The committed
vector file pinned three hybrid-KEM cases and two ITS cases - and **zero** for the default transport
branch. The non-default path was gated and the default path was not.

**The reason is mechanical rather than a lapse of judgement, which is what makes it worth recording.**
`itsCombine` is `export`ed from `clientMcp.ts`; `deriveAeadKey` was module-private. A checker can only
reach what is exported, so the sibling that was reachable is the sibling that got a gate. Nobody chose
to leave the default unpinned; the module boundary chose for them.

Four parameters are hand-copied between `src/qsh/pqc/transport.py` and `web/src/lib/clientMcp.ts`: the
info string `qsh/transport/pqc+qkd`, the salt/ikm order, 32 output bytes and SHA-256. Verified equal
before changing anything - both derive `3782c865514641d3ed3e05fb4f834e74928470b788ca2c54e9cd78652062e7b9`
from the same inputs - which is exactly why a drift would have been silent.

**The seeding found something the vector design nearly missed.** Transposing salt and ikm in the
browser makes it derive **identical bytes for both QKD-key lengths**, because with the session key as
both salt and IKM the QKD key stops contributing at all. That is not a wrong key, it is a key with no
quantum contribution - and a single-length vector would have reported it as merely different rather
than as degenerate. Two lengths, because the salt is the variable-length input.

Adjacent and verified CLEAN, so it is not re-checked: the **SDN control plane** twin
(`src/qsh/qkd/sdn_controller.py` / `web/src/lib/clientSdnController.ts`) IS covered across languages.
`tests/test_sdn_fields_carry_the_dimension_the_spec_says.py` and
`tests/test_a_field_emptied_on_purpose_has_no_arithmetic_reader.py` both read the TypeScript and
assert DIMENSIONAL correctness - bits against bit/s - in both files, and their docstrings record the
Round 42/44 instance where a fix reached the browser and left Python budgeting in the wrong unit for
two rounds. The shared literals between those two files are arXiv IDs and ETSI clause numbers, not a
shared numeric derivation, so there is no silent-drift surface there.

## Round 56 (branch `round56-qrng-crossed-the-language-boundary`)

### I merged on cancelled CI, and the wait loop I wrote is why

PR #50 merged at 13:35:37 while CI on its head was **in progress**, and the two runs before it had
concluded **cancelled**. The loop was:

    [ "$n" -ge 2 ] && [ "$n" = "$c" ] && break     # c counts status == "completed"

and **`cancelled` IS `completed`**. Pushing a follow-up commit cancels the in-flight runs, so the loop
saw two completed runs, exited, and the next line merged.

The head_sha discipline established earlier in this session was performed and passed. The commit was
the right one; **the verdict was never read.** That is this project's second signature defect - a
check reading the MECHANISM (did the run finish) and not the OUTCOME (did it pass) - committed in my
own tooling, in the same session that found four instances of it elsewhere.

The remedy is `scripts/wait_for_ci.sh`, which requires `conclusion == "success"` on every run for the
sha, is bounded so a sha with no runs cannot hang forever ("no verdict is not a pass"), and carries
the incident in its header. Exercised both ways: it refuses the cancelled commit `6ea8bb6` with
"A cancelled run is COMPLETED and is NOT a pass. Do not merge.", and accepts `bf22678`, the last
genuinely green one.

Everything that merged passes all 17 gates locally and CI on `main` was re-run after the merge. The
process was wrong even though the artefact was fine, and the process is what was supposed to be
carrying the guarantee.

### The Round 51 QRNG fix reached one language, and the browser is the one visitors run

`health_suite` was moved onto the raw stream in Python in Round 51, with
`tests/test_health_tests_measure_the_noise_source.py` written to pin it. `clientQRNG.ts` bound `raw`
and then computed `hMin`, both SP 800-90B continuous tests and the reported `minEntropyPerBit` from
`bits` - the post-extractor output.

Measured inside that module on a P(1) = 0.90 stream: **raw min-entropy 0.1510, conditioned 0.9749.**
Repetition-count cutoff 134 against 22; adaptive-proportion 965 against 597. An extractor produces
near-uniform output by construction, so the figure described the Toeplitz stage under a label reading
"entropy source" - and the demo is client-side only, so that was the only figure anyone saw.

**Why it survived in both languages:** on the SHIPPED near-fair source the two readings differ in the
third decimal - 0.9057 against 0.9605. The Python test's docstring records exactly this, and it was
true of the browser for five more rounds.

`healthPassed` is deliberately unchanged. Sections 4.4.1 and 4.4.2 catch a source that has gone
STUCK, not one that is merely skewed, so the verdict was correct either way; claiming otherwise would
assert the standard says something it does not.

**And my first gate for it was the wrong shape.** It checked that `minEntropyMcv` separates a biased
stream from its conditioned output - which the estimator always did - and **passed with the call site
reverted.** Mechanism instead of outcome, inside the gate written to catch that. No output-based
check can discriminate here, because the shipped source is fair by construction; that is why the
Python twin asserts on the call with an `ast` parse. The browser check now does the same, with
comments stripped first - the block explaining why `raw` is required contains the string
`minEntropyMcv(raw)`, so a plain grep matches its own documentation.

### Round 56, continued: two labels that claimed more than their evidence, and my own gate that reached three of four

**The badge reported an adapter probe as an execution provider.** `status.backend` is
`gpu ? "onnx-webgpu" : "onnx-wasm"`, where `gpu` is `hasWebGPU()`. The comment above it said "the
provider that RAN" - and **that same comment records fixing this exact class once already**, for
WebNN: *"a label describing something the code does not do, which is the exact defect this project
keeps finding."* The fix reached the WebNN half of one ternary and not the WebGPU half.

What the probe establishes is real and worth reporting: a non-software adapter exists (rasterisers
are declined by name, on a measured 300x) and the WebGPU build loaded. What it does not establish is
node placement - ORT falls back **per node** to the WASM floor for unregistered operators, which is
the same mechanism the `ai.onnx.ml` note two sentences earlier depends on. `README.md` asserted both
facts three lines apart: that `ai.onnx.ml` models take WASM whatever the adapter says, and that the
reported backend is "never a false webgpu that silently ran on WASM".

**What I did NOT establish, stated because the audit asserted it and I could not reproduce it.** The
audit named specific unregistered operators per model. I confirmed the graphs - int8 is PREFERRED,
and `medical_breastmnist_resnet18`'s int8 build is 20x `ConvInteger` + 18x `DynamicQuantizeLinear`,
`satellite_telemanom`'s is Slice/Concat/Transpose/Shape - but ORT 1.27.0's WebGPU op registry is not
enumerable from the shipped bundle, so **which of those fall back is unmeasured here.** The comments
now state the mechanism and stop short of the per-model verdict rather than repeating a claim I
cannot support. Closing it needs `endProfiling()` and a third badge state, because two cannot express
a partial answer.

**And the licence gate I wrote this session reached three of four surfaces.** `README.md:863` read
`MedMNIST v2 **BreastMNIST** (CC-BY-4.0)` - a proper PREFIX of the registry string, cut exactly where
the hedge begins - and the file carried zero occurrences of BUSI, unverifiable or asserted-by. The
gate written to stop a downgrade reaching only some copies was hardcoded to `DATA_LICENSES.md` and
`prep_model_inputs.py`, so it could not see the file most readers open first. **The same shape as the
defect it exists for, one level up:** a fix that covers the places the fixer happened to be looking
at. Extended to README's table and seeded - reverting the truncation now fails with the line number.

### Round 56, continued: a caption its own payload retracts, and a figure that was unfalsifiable rather than wrong

**The separability caption asserted a verdict the same figcaption contradicts four lines later.** It
appended *"so the model's decisions are being drawn on data that is genuinely separable - chosen by
scoring every feature pair, not by eye."* Both halves fail on payloads that render it:

- "Genuinely separable" is contradicted by `note`, printed by the SAME figcaption. Finance's note
  says 0.2302 against a 0.02002 base rate *"is a 11.5x lift and still a LOW number... two features do
  not solve AML"*. Telecom's says the lift is 1.13x.
- "Not by eye" is false for finance, and the generator says so in its own words: the AUC-PR winner
  `format_risk x src_bank_diversity` scores **0.3195** and was REJECTED as unreadable - *"A tree does
  not care how values are spaced; an eye does."* The shipped pair was chosen on readability among the
  separable ones. A defensible choice, honestly recorded, and the opposite of what the caption said.

Written for satellite when it was the only payload carrying the field, then inherited by two more.
**Third instance of this defect in this one component**, the first two recorded in its own comments,
and the remedy is the one it used twice: derived numbers in the caption, interpretation in `note`.

**And the audit's claim about `separability_auc_pr` was wrong - as was my first check of it.** The
audit reported the finance figure as not reproducing on the plotted rows. My first probe agreed:
best single-feature rank AUC-PR is 0.0566 against a stated 0.2302. **Both of us measured a single
feature; the stated figure is a JOINT two-feature score.** Measured on the shipped points:

    single-feature rank AUC-PR   0.0566
    RF 2D, 5-fold cross-validated 0.2016     <- what the literal reports
    RF 2D, in-sample              0.4378
    stated                        0.2302

So the number reproduces and the audit is refuted. What survives is narrower and real: the field is a
hardcoded literal and **no payload stated the estimator**, so an unstated choice moves finance's
answer by a factor of eight, and both alternatives are what a re-deriver tries first. The number was
never wrong - it was **unfalsifiable**, which for this project is its own defect.

The estimator is now recorded in the generator and pinned. **The gate's band is itself checked to
EXCLUDE the in-sample and single-feature readings**, because a tolerance admitting all three would
pass whichever estimator produced the literal - the wider-tolerance-than-the-drift shape already on
record here from a decoy-state band. Telecom is SKIPPED with its reason stated: its four readings are
0.9798 / 0.9915 / 0.9925 / 1.0000, all inside any usable band, so the estimator is genuinely
undetermined there and no tolerance fixes that.

**Two audit claims refuted by checking, in two turns.** That ratio is the argument for verifying
before acting, and it cuts both ways: the same discipline that refuted these produced the narrower
findings that replaced them.

### Round 56, continued: the frontier fact-check, and a Round 55 withdrawal that reached one of two copies

A sweep of everything load-bearing that could have moved since 2026-08. **Four of five areas
UNCHANGED**, which is the expected answer for a three-week window and is recorded as an answer rather
than as an absence of findings.

**The one that changes an argument.** `docs/browser-verification-checklist.md` still asserted that
fidelity quantum kernels "beat that regime only at **15 qubits or more** (arXiv:2605.31449)", used to
explain the telecom quantum-kernel shortfall. Round 55 read that paper, found the reading inverted,
and withdrew it - **in `references/model-choice-vs-2026-practice.md` only.** The checklist copy went
on asserting it. A Round 55 fix, reaching one of two copies, found in Round 56.

15 is a CEILING on the fidelity kernel, not a floor: *"For n >= 15, the test accuracy of the FQK
drops sharply, approaching a random guesser"*, and concentration worsens with qubit count, so 4 and 8
are the LEAST-concentrated regime. The paper is a proposed FIX for concentration, not an account of
why a fidelity kernel underperforms at small n. The honest state is **"quantum lost, cause
unknown"**, and a citation that says the opposite is worse than no citation because it stops anyone
looking.

**ITU-T, re-fetched per recommendation rather than taken on report.** Y.3835 reads "In force" alone
at 9,796 B; Y.3836, Y.3837 and Y.3838 read "In force" AND **"prepublished"** at 9,647 / 9,590 /
9,515 B - so `REFERENCES.md` naming only Y.3836 as prepublished was wrong about the other two. The
ceiling holds: **Y.3839 and Y.3840 return 200 with a zero-length body** (Y.3839 followed, approved 2026-10-19; this is the entry as measured). Worth recording the method
trap: a browser User-Agent - the thing that defeats ETSI's bot block - collapses *every* one of these
ITU pages to a 245 B stub, destroying the very signal that discriminates them.

**`@noble/post-quantum` 0.6.1 -> 0.7.0.** `^0.6.1` does not admit 0.7.0 (a caret on 0.x is
`>=0.6.1 <0.7.0`), so nothing would have moved without a deliberate bump. Verified by running
`check:pqc` **unmodified** against 0.7.0: identical rejection counts, identical sizes, exit 0. That
evidence only became meaningful earlier this same round - before the round trip was added, that gate
passed a library whose `encapsulate` threw unconditionally.

**And a caution recorded against my own future self.** 0.7.0's changelog line about a `MultiplyNTTs`
intermediate reaching $`2^{35}`$ reads exactly like a security fix. It is not: 0.7.0's own comment says the
wide intermediate forces "slow float fmod", and 0.6.1 reduces before truncating, so 0.6.1 is not
numerically wrong. Written down because the shape of that line is what gets transcribed as "upgraded
for a security fix" by someone reading only the changelog.

**Unchanged and verified, so they are not re-checked:** FIPS 203/204/205 (no new publication; FIPS
206 and 207 IPDs return 404); every ETSI GS QKD draft status dated 2026-06-09 or earlier with the TB
approval wave still ahead; SEAL 4.4.3, OpenFHE 1.5.1, node-seal 7.0.0, TFHE-rs 1.7.0, MP-SPDZ 0.4.3
all current; **onnxruntime-web is still 1.27.0 on npm** although the core repo tagged 1.28/1.29 -
reading only GitHub releases would wrongly call the pin stale; and `ConvInteger`,
`DynamicQuantizeLinear` and `MatMulInteger` are absent from the WebGPU operator list at 1.27.0, 1.29.0
AND main, so `CS17`'s open question would still be open on the newest release.

**TFHE-rs' WASM limitation holds verbatim**, which is the ground `CS11` declines Rust on: *"it does
not support FHE computations"*, last touched 2026-06-19.

### Round 56, continued: mobile was unmeasurable with one tool and trivial with another

`UX12` recorded mobile layout as UNMEASURED, with the reason: the browser automation resizes the OS
window without moving the viewport - `innerWidth` stayed 1280 and `outerWidth` read **0**. That was
honest and it was also a tooling limit, not a fact about the site. **CDP device emulation moves the
viewport**, and the measurement took minutes once the right tool was used.

Measured on the deployed page at a 500px viewport: **204px of horizontal document overflow.**

**The naive probe blamed 40 elements and was wrong about 39 of them.** Almost all were nav links
inside `.nav { overflow-x: auto }` - a deliberate scrolling strip, `scrollWidth` 1285 against
`clientWidth` 453, whose children legitimately extend past the viewport. Counting an element that
sits inside a scroller as "overflowing the page" is the same false-positive shape as the four earlier
probe errors this round. Discounting any element with a scrolling ancestor leaves **exactly one
genuine cause**.

**And the cause is a correct decision whose enabling condition was dropped one file away.**
`TwoPartyDiagram`'s SVG sets `min-width: 680px` deliberately: a 780-unit viewBox in a 360px column
renders 11-unit step labels at about 4.5px, and its comment argues "illegible is worse than
off-screen, because off-screen can be scrolled to". That is right. The comment then says the
scrolling is "what `.card { overflow-x: auto }` already provides" - and `.card` does carry that rule,
with its own comment recording the audit that added it: at 390px the document was 1326px wide, 936px
of scroll, "from two diagram SVGs (990px, 930px)" among others.

**`UseCasePage` renders the diagram in a hand-styled div that copies the card's background, radius
and padding - and not its `overflow-x`.** The look was reproduced and the one property that mattered
was dropped. A fix that reached the CSS class and not the element that imitates it.

**A correction to something I asserted in `UX12`.** I wrote that "no element carries a `min-width`
above 0/auto". That was read off a TRUNCATED probe output - the SVG carries `min-width: 680px` and
was in the list I never saw the end of. The claim that mobile was unmeasured was right; the
supporting evidence I offered for it was not.

**And the gate for this needed comment-stripping, for the third time this session.** The comment
added to `UseCasePage` explaining why the wrapper must scroll QUOTES `minWidth: 680`, so the raw-text
scan classified `UseCasePage` itself as a wide component. Prose about a property is exactly what a
file accumulates when someone fixes that property. The gate also had to move from the file that SETS
the width to the RENDER SITES, because the scroller lives in a different file - the first version
flagged the component it had just fixed.

### Round 58: two comparisons that measured different things, and the E91 page described a protocol it was not running

**Context.** An independent check of the public text (2026-10-06) and a live survey of the deployed demo
on 2026-10-23. The full findings table, with a primary source per row, is the Round 58 section of
`docs/deliverable-verification.md`; this entry records what was surprising.

**The E91 relation reproduces exactly - from a number the page did not show.** On `/orchestration` with
E91 selected the three scenarios read QBER 0.03 / S 2.66 / positive key (misalignment 0.015, 50 km),
0.27 / 1.29 / zero key (Eve = 1) and 0.02 / 2.74 (no misalignment, 1 km). $`S = 2\sqrt{2}\,(1 - 2Q)`$ holds
for all three only with the MODEL error rate e0; from the displayed QBER, which is a sampled draw
rounded to two places, two of the three give 1.30 and 2.72. It is a self-consistency check of a
Werner-state model, not an independent validation. The model rate is now emitted as
`e91_model_qber` at both E91 sites so the relation can be checked from the page.

**And the same page captioned the E91 QBER with BB84's decoy-state bound.** The protocol test had been
hoisted into a constant whose own comment said a shared constant "is what stops the next row from
being the next omission" - declared inside a branch, while the QBER row sat outside it. Fixed by
moving the declaration to function scope; the gate runs the real browser module for both protocols.

**The ratio card and the advantage score were the same mistake twice.** Each divided a simulated
state vector's bytes (or qubits) by a CKKS byte ratio at under 1% slot use. One more thing the
retirement had to reach: the score was also fed to the in-browser agent as run context, so a model
was being asked to reason from it.

**The security-posture table called every QKD row "measured".** Under "what this run measured", with
"The bits are real; there are not enough of them." The browser's `bb84()` returns a count and never key
bytes, which a checklist row (SC2) already recorded. The rows are now kind "simulated", and
the banner every page renders says the quantum channel is simulated.

**Three attributions the sources did not support, found by reading them rather than their
summaries.** BSI TR-02102-1 v2026-01 was credited with the QKD add-on; its section 2.2 hybrid is
quantum-safe plus classical and does not mention QKD. A sentence quoted from "the NSA FAQ" is on a
different NSA page. And "a FAR rule putting contractors on the 2030 date" describes a rule that EO
14412 only directs to be PROPOSED.

**And the deployed branch showed one more, which my own gate had written down as correct.** After the
QBER row was fixed, an E91 run on the live page still listed "Decoy-state necessity" with the warning
that "the decoy states are what make this link work at all" - on a protocol with no decoy states. The
test written earlier the same day asserted FOUR QKD rows for both protocols, so it did not miss the
defect; it specified it. It now expects three on E91 and checks the whole table, and seeding the row
back fails two tests. The same walk found three words run together by JSX line breaks and two agent
prompts that asked the in-browser model to defend claims the page itself rejects.

**One copy of several, again.** The novelty sentence had a third copy in the technical report; the
market figures had five, two of them in this file and the register, quoted as history. Both gates
enumerate the copies rather than naming them.

### Round 58, continued: composing Eve's error, and what the model curves showed

**The composition was the smallest change with the widest reach.** Two flips on one bit cancel, so
intercept-resend's 0.25 f composes with the channel error as a + b - 2ab; the code added them. It took
four sites in two languages at once - the abort reads one and the decoy bound the other, and a
cross-language test compares the two rates at every Eve fraction - and it moved nothing that has no
eavesdropper. E91 at 50 km under a full attack now reads S 1.33, not 1.29, and $`S = 2\sqrt{2}\,(1 - 2Q)`$
still holds exactly, because the state is a Werner/intercept-resend mixture whose CHSH value in the
measured plane equals a Werner state's at the same error rate. The caption had called that mixture a
Werner state; it is separable at a full attack.

**The caption's two thresholds came from different error-correction assumptions.** "S > 2.424, i.e.
Q < 0.0715" is the device-independent point at $`f = 1`$; the zero-key point beside it, 0.0981, is at
$`f = 1.16`$. At one f the device-independent threshold is S > 2.456, Q < 0.0658, and both are now
computed rather than quoted.

**Plotting a quantity across its range found two things a single run hides.** Without decoy states the
BB84 bound on the telecom page is zero at every distance shown - the multi-photon fraction exceeds the
detection probability everywhere - so the decoy states are not improving the link, they are the link.
And QSDC on the satellite page closes only when the fixed atmospheric and pointing losses are set near
zero, out to about 129 km at 0 + 0 dB, which no LEO slant range is; at the default budget it fails at
every range, 0.1 km included. Four tooltip figures saying otherwise dated from before two formula
corrections; they are now computed by the function the stage runs.

**The redesign corrected more than it rearranged.** Moving text into tables meant re-reading each
sentence against the code: the Overview pill row put QKD on a key path it never joins, the agility
paragraph offered two options the demo does not have, the Validations page filed a real PQC handshake
under "Quantum simulations", the browser's benchmark row named the Python row's MPC mechanism, and four
"above/below" pointers pointed the wrong way. The one 3D view with content - the QCNN's reduced Bloch
vectors shrinking and leaving the encoding plane - reads the real circuit through a hook that leaves
the measured path unchanged; the two illustrations say on the figure that they add nothing.

**The network layer, added as math rather than tunnels.** A browser has no sockets, so the companion
network project's WireGuard and IPsec work could only come in as key schedules run between two in-tab
peers - and each was held to an official vector where one exists: NIST ACVP for ML-KEM-768 (10/10,
including two modified ciphertexts that must fall through to implicit rejection), the cacophony vector
for the Noise_IKpsk2 core, NIST CAVP SP 800-135 for the IKEv2 derivation. Two layers have no vectors
anywhere - the RFC 9370 intermediate exchange and the RFC 8784 PPK - and two of twelve seeded defects,
both composition errors in exactly those layers, pass the check. The page labels those rows "computed
per spec - no official vectors" rather than letting a green row imply more. One check was added
because a test showed a gap: dropping WireGuard's identifier string from the handshake passed every
other assertion, so both peers' initial state is now compared with constants an independent
implementation hardcodes, labelled non-authoritative.

### Round 57: one sentence in two files, and why counting quotes could not tell a correction from a claim

**Context.** Finished on its own branch after Round 58 merged. Found by reading the OPS-SAT student's
record against the prose that described it: the distiller's docstring and the trainer's comment both
said the student beat the published baseline and its teacher, and the record says the opposite on the
first count and a different metric on the second.

**The interesting failure was the gate's, not the prose's.** A withdrawal has to restate the claim it
withdraws, so a scanner for "beats 0.979" sees the correction as the defect. The first discriminator
counted double quotes from the start of the comment block - and Python source is full of quote
characters, so a single apostrophe or string literal upstream inverted every verdict below it. An
explicit marker pair replaces the heuristic: what is quoted between them is blanked before the scan,
the markers must balance, and each span must quote something. Rewording my own refutation was also
needed - "the beats-the-baseline claim is false" repeats the boast word, and under a strict convention a
refutation that is not inside the markers should not reuse it.

### Round 59: an audit count that depended on a flag, and a timeout that was the download, not the upgrade

**Context.** A separate branch for `npm audit`, after Round 57 merged. The question for each advisory
was not whether it exists but whether the public demo, which is a static bundle served by Caddy,
reaches it.

**Two counts, and which one is the claim.** The same lockfile flags 9 packages with `--omit=dev` and
16 without it. Quoting either number alone would mislead in one direction or the
other, so the register carries both, before and after: 9 to 4 for production dependencies, 16 to 6
overall. The two survivors only the full count shows - vite and esbuild - are dev-server advisories,
and that was checked in the installed vite rather than assumed: its `preview()` mounts `sirv` and an
HTML fallback and none of the open-in-editor, raw-fs or transform middlewares the advisories name.

**The first timing said regression, and it was wrong.** After the upgrade the in-browser agent ran
into the panel's 120 s timeout on its first run. The weights cache then showed why: the run had fetched
`onnx/model.onnx`, 313 MiB, which finished at 277 s, long after the timeout fired. Reproducing that on
main's build required the same origin, because the browser's weights cache is per origin, and main
timed out too. On the public demo the same file later arrived in 19.6 s and the first run timed out
anyway, so the download explains the local timeout but not the defect. Wall-clock times from the panel could not settle it either way, since it samples and
each run stops at a different length, so a small page timed both library builds with greedy decoding
of a fixed 32 tokens, A-B-A. The upgrade is faster on WebGPU (795 against 1136 ms per token) and level
on WASM.

**What the controlled timing found instead.** WASM runs the same model at about 9 ms per token; WebGPU
at 750 to 1140 ms per token, on a workstation-class NVIDIA adapter. The panel prefers WebGPU whenever an
adapter exists, so on this machine it picks the path that is 80 times slower and twice as large to
download, and its label says "~80 MB" for a model whose smallest loaded weights are 226 MiB. Only the
WASM path had ever been timed. Recorded, not fixed in the dependency change: the fix decides which
backend the panel uses.

### Round 60: the default that was wrong for the machine it was measured on, and a limit that timed the network

**Context.** The fix Round 59 recorded and left open. Both in-browser model tiers chose WebGPU because
an adapter existed, and on the one machine measured that was the slow path for both.

**What the change is, and is not.** The default moved to WASM, and WebGPU became a selection in each
place a run starts: the agent panel's Backend and the pipeline's Model backend. It is not a claim that
WASM is faster everywhere. One machine was measured, the record says so, and WebGPU stayed
selectable for exactly that reason. The selection is sent only when made, like the published-model
choice beside it, so a payload never records a choice nobody made.

**The second defect sat inside the first.** The agent's 120 s limit was a limit on total time, so on
a slow connection a first run's download could use it up on its own while every byte was still
arriving. With WASM as the default that would have been the next timeout report. The limit now
measures time without progress. The download reports progress, which a visitor can now watch, and
each event resets the clock.

**Labels that compute rather than assert.** The old size was typed beside the model id and was off by
a factor of three for the file actually fetched. Recording the bytes per backend fixed the label only
together with two other changes: pinning the revision, so the file cannot change under the number,
and passing `dtype` explicitly, so the library's per-device default cannot pick a different file. The
gate checks all three, because any one alone leaves the label free to drift.

### Round 61: a major upgrade where the newest major was the wrong one

**Context.** The first of the two major upgrades decided after Round 59, one PR each.

**Which major.** npm's `latest` for react-router is 8.5.0, and an upgrade that took `latest` would have
required React 19 and a newer Node than this project's image is pinned to - a far larger change than
the advisory asked for. The advisories are fixed from 7.18.0, so the target is the `version-7` tag,
7.18.4. The difference between "the fixed version" and "the newest version" is the whole size of this
change.

**What was checked rather than trusted.** The v6 to v7 notes list behaviour changes, each of which was
compared with the app instead of assumed not to apply: relative paths in splat routes (no relative
link targets exist, and the only splat is the catch-all), state updates wrapped in startTransition
(navigation was exercised by clicking, not by loading URLs, so a transition that never committed would
have shown), and the removal of data-router APIs (none used). The gate written for the advisories is
kept, because it holds a property and not a version.

### Round 62: upgrading the tool the advisories were about, and checking that tool

**Context.** The second major upgrade decided after Round 59. vite's advisories concern its dev
server, which the public demo never runs, so checking only the production bundle would have tested
the part that did not need fixing. The dev server was started and exercised too: it served the app
with the cross-origin isolation headers the browser tiers depend on, and refused a request for a file
outside the project.

**The asset paths were the risk in the bundle.** Three runtime paths depend on how vite rewrites
asset references: node-seal's wasm through a `?url` import, ONNX Runtime's wasm through
`import.meta.url`, and the GIF worker through another `?url`. Each was confirmed by what the browser
fetched or what the build emitted, not by the build exiting cleanly: a mis-rewritten wasm path is
served `index.html` by the SPA fallback and fails only when the module is instantiated.

### Round 63: a prototype measured beside the demo, and a toolchain the host could not supply

**Context.** The Polytune prototype decided after Round 60's research, built after the two major
upgrades and kept out of the production build.

**The host route failed for a reason a version number shows.** The wasm32 standard library was added
to the host for this round, and the build would still have failed: Polytune declares Rust 1.89 and the
system compiler is 1.88. The container is not a preference here; it is the only route that builds the
crate on this machine, and the documentation says which route works for which crate rather than
offering both as equivalent.

**The measurement had to count what the tab cannot show.** Both implementations run their parties in
one tab, where a message costs a function call, so wall-clock alone flatters whichever sends more. The
in-memory channel therefore counts messages as well as bytes, and emp-wasm's counts were taken by
replicating its call against the same installed packages. That is what turns "six times faster" into
"six times faster here, and five times the traffic".

**Browser note.** The connected Chrome could no longer reach this machine's loopback, so the prototype
and the emp-wasm baseline were both measured in the app's built-in browser, which keeps the two on one
engine.

### Round 64: a link that was shaped correctly and a payload that was not, and an answer that depends on N

**Context.** The network measurement decided after Round 63, before any replacement.

**Choosing the shaper by what it shapes.** The user-space proxy that was considered first terminates TCP
at both ends, so the browsers would have measured a loopback connection with a proxy's buffer behind
it. Congestion control - the thing the request asked to be represented - would not have run at all.
netem on each relay's own interface shapes the segments the browsers' TCP actually sends, which is why
the relays' TCP_INFO shows windows growing and RTT inflating under load.

**The smoke test lied, and the qdisc counters caught it.** 297 Mbit/s through a 10 Mbit/s shaper is
impossible, and the counters on the shaped interface showed why without guessing: almost nothing had
crossed it. Compression, negotiated by default, had removed the payload. A calibration that measures a
link has to send what the protocols send - random bytes - and a relay in a measurement must not
transform what it forwards.

**The headline needed its condition.** "Polytune is faster" is true at 2 classes on every link and false
at 10. The demo only ever runs 2, which was checked in the models rather than assumed.

### Round 65: replacing a dependency tier, and the gate that had to change its mind

**Context.** The replacement decided after the measurements of Rounds 63 and 64.

**A gate written to forbid exactly this.** Round 63's gate said the web image may not invoke a Rust
toolchain, because a prototype was not to ship by accident. Round 65 ships Rust on purpose, and the
honest edit was to keep the property that mattered - nothing from prototypes/ ships - and drop the
proxy for it, rather than delete the test. Its new form names what the image may build.

**Tests that run where the artefact is made.** The argmax's correctness test runs inside the image's
build stage, so the module that reaches visitors is one that passed it minutes earlier on the same
toolchain. The local and GitHub CI run the same stage, not a parallel recipe that could drift from it.

**Notices quote, they do not summarise.** A licence identifier says which terms apply, but MIT and BSD
attach to the copyright line in each crate's own file, so the notices copy those files. The generator
refuses rather than warns, because a build that ships a binary under unreviewed terms is not one to
finish quietly.

### Round 66: a mask is not an input, and a fix that must not move anything else

**Context.** One of the follow-ups chosen after Round 65's report, before the dependency audit gate.

**The demo's generator was doing two jobs.** A seeded generator is right for what a page should
reproduce: which row is encrypted, the parties' secrets, the simulated channel. It is wrong for a mask,
whose whole purpose is that the other party cannot predict it. The two jobs had shared one call, `u()`,
since the masks were written. They were separated at the two places where nothing a run reports
depends on the mask.

**The fix was designed so that a comparison could prove it.** Taking the masks out of the stream would
have shifted every later draw. Genomics would then encrypt a different row for the same seed, and a
before-and-after run could no longer tell the change apart from noise. So the secure sum still
advances the generator once per secret. The proof was a genomics run on each build: the seeded
figures (the sum, the PRS, the class, the top driver) agree exactly. What differs is the CKKS noise on
the tied rows, and it differs as much between two runs of one build.

**A language that panics where a CPU wraps.** The obvious form, `a + b` in 32 bits, would have
compiled and then panicked on most shares, because a uniform mask makes most sums cross $`2^{32}`$. The
compiler source settled it, and the crate's test now uses masks near `u32::MAX` so that the wrap is
what it exercises.

**The twin was found by asking where else.** Round 40 fixed the wording of one seeded mask, and the
argmax's mask went unexamined because its line says `a[k]`, not "mask". The gate therefore scans for
share material drawn from any seeded call, and every exception it allows must keep its disclosure, or
the exception lapses.

### Round 67: a count that was true on the day it was measured

**Context.** The second follow-up chosen after Round 65's report.

**Zero was a measurement, not a property.** Seven rounds recorded npm audit counts, and every one of
them was true on its day. None of them says anything about the next day, because what changes the
answer, a newly published advisory, happens outside the repository. A check that runs on push cannot
see it. The only form that can is one that runs on a clock.

**The first run against the Rust crates was the first audit they ever had.** It found one advisory,
and the right response was neither to silence the class of advisory nor to leave the check red. The
policy ignores that one advisory by its ID, and states the reason. The test repeats the ID with the
condition that reopens it. A new advisory, including a vulnerability in the same crate, still fails.

**Each failure path was made to fire before the success path was trusted.** That meant a removed
ignore, a disallowed git source, and a scratch project locking a version with a known critical
advisory. A green audit is evidence only once the same audit has been seen to go red.

### Round 68: the library held up; the code around it did not

**Context.** The follow-up chosen after Rounds 65 to 67: a security review of how the demo uses
Polytune.

**The library was read, not trusted.** The questions were the ones the demo's claim depends on:
- whether a trusted dealer could be in the path;
- where the randomness comes from;
- what the statistical parameter is;
- what is open upstream.
Each answer is a line of Polytune's source at the pinned commit. The test pins that commit, so the
answers stop being trusted the moment it moves.

**The defect was in the failure path, again.** A fallback that printed a cost model under measured
names had been fixed once, for one key. The fix and its gate named the key rather than the rule. The
same branch kept doing it for three more keys, and it misattributed the failure as well. The gate now
treats the cost model's row as what it is, and the old code fails it on exactly those lines.

**A self-test the deployment could never have.** Two hosts could not compare the circuit's answer
with the clear one, because neither holds both shares. This tab does, so it does. The check costs one
loop, and a broken integration can no longer report a confident wrong index.

**Failure paths were exercised by breaking the build, not by reading the code.** Removing one wasm
file, then the other, showed what a visitor would see in each case, including that the missing file
came back as HTML.

### Round 69: three open items, each closed by a primary source, one of them in a direction nobody expected

**Context.** The follow-up chosen after Round 68: the items carried as UNVERIFIED across rounds.

**The question was smaller than the defect.** It asked which annex the medical model falls under. The
answer, Annex I and 2 August 2028, showed that the panel's sentence was wrong on every page, not only
on the medical one. Two of the pages describe systems the Act does not treat as high-risk at all.

**The copies again.** The panel's claim had four siblings: two README lines and two Python docstrings.
The gate scans tracked text for the family, which is a high-risk claim about AML, or an Art. 13 date
without its annex, rather than the panel's exact words.

**A figure that dissolved on reading.** The "about 15x" was carried for rounds as a number awaiting
confirmation. In the paper, the only 15 is a coefficient in a gate count. Keeping it UNVERIFIED, and not
quoting it, was the right default, because the alternative would have published a number with no
source.

### Round 70: re-checking by age

**Context.** The standing request's full sweep, ordered by how long ago each area was last verified,
oldest first. It is stacked on Round 69, which stays open by decision.

**Most of what was old was still right.** Seven areas were re-run on the demo, and their sources
re-read at the latest versions. The code needed no change. What had gone stale was prose:
- a description that named the full deployment's method for the browser's;
- a gain that read as comparable with the literature's when its baseline was not;
- a fact-check sentence that a standards body had contradicted in public;
- a checklist row still quoting a circuit retired 24 rounds earlier, with a conclusion that the
  retirement had reversed.

**A conclusion can outlive its numbers.** QC18's finding was correct when it was written and false
now. Nothing tracked had repeated it, because it lived only in the checklist, which is the reason the
checklist has to be re-read and not only re-run.

### Round 71: a console message led to a sample, and the sample to the model

**Context.** Fixing the two satellite console messages Round 70 had recorded as proposals.

**The messages were symptoms, one level up.** The 404 asked for a sample the model publishes none of
by design, and the shape warning came from an export traced on one row. Both fixes were small. What
mattered was where the samples came from. Reading the generator to fix the second message showed
that three samples were drawn from the whole corpus, while the log called every sample held-out.

**A generator that copies the trainer drifts from it.** Both drifted copies were written as
mirrors, with comments saying they followed the training script step for step. They are now calls
into the trainer, plus two checks against artefacts the trainer had already saved: the export
fixtures, and the published constants or the recorded metric. "The same preprocessing" became a
check that can fail, and it did fail once, which is how the satellite leak was found.

**A fix to the code is not a fix to the artefact.** Round 34 corrected the trainer's z-score. The
model on the Hub was trained three days earlier and never retrained, so every figure recorded for it
came from the leaky pipeline the trainer no longer ran. The same shape appeared twice more in this
round:
- the medical licence downgrade reached the registry, but no sample was rebuilt;
- a withheld int8 graph was deleted locally and kept on the Hub.

Each time the repository was right and the published copy was not.

**The honest retrain scored zero.** It caught none of the three scored events, where the leaky runs
had caught one. A second seed might catch one again, and choosing between seeds after seeing the
results would be choosing a number. The run is recorded as measured, with its counts.

### Round 72: the carried items, each taken to its source

**Context.** The items carried as UNVERIFIED, some of them since August. Three web questions went to
parallel research agents, and two were settled locally. Every claim an agent made about code was then
re-read in the source this build compiles before anything was written.

**Most of them closed in the direction already stated.** Garble's cast keeps the low bits, as Round 66
had read; wasm-opt would buy almost nothing on the wire; QC2 holds because the function takes no seed.

**Two changed what the project says.**
- **The BUSI images have no licence from their authors.** "Unverifiable" had meant that a grant might
  exist somewhere unread. It does not exist anywhere the authors publish. The data was released
  publicly with a hospital's approval and no terms, and the project now says that and stays public by
  decision.
- **The satellite AI Act sentence named the right test and missed why.** Space is not digital
  infrastructure in the directive the Act points to.

**One opened.** Reading the code behind #63 showed the OT-extension check's challenge fixed before the
matrix it checks. Polytune's own history shows the ordering changed to save a round trip. Whether that
matters is a cryptographic question this project cannot settle. So the public text stops short of
claiming what it would need answered, and the question goes to the people who can answer it.

### Round 73: a style that nothing enforced, and a duplicate that was mine

**Context.** The oldest-first audit, continued: the mathematics notation and the documentation
duplicate scan, last measured in Round 55.

**The drift started in my own recent rounds.** The register sections I wrote for Rounds 70 and 72
used plain inline math, against a house style recorded in the same file. A convention that lives only
in prose is followed by whoever happens to remember it. It is now a gate.

**The duplicate was also mine.** The Round 70 re-check entered a survey that `REFERENCES.md` already
cited, because the search before adding it looked for the claim and not for the paper.

### Round 74: a count that looked at the wrong code, and a fix that made a failure

**Context.** The oldest-first audit, continued: the accessibility rows, last measured on 2026-08-15.

**The Escape handler was there all along.** The checklist had counted handlers in this repository and
found none, so it recorded the tooltip as not dismissible. The handler lives in recharts. A count of
the code that is ours answers a question about our code; the criterion is about what the page does.
Pressing the key on the deployed page answered it in one step, and in the same step showed the defect
nobody had looked for: the tooltip stays open after Tab moves on.

**A scripted focus is not a key press.** `element.focus()` did not open a chart's tooltip, and a real
ArrowRight did. The first reading, made with scripted events, would have recorded the keyboard path
as broken.

**My contrast fix measured one neighbour and broke another.** The histogram's negative class was
lightened until it passed against the card, and that took it to 1.10:1 against the stacked segment it
sits on. I found it only because the record needed the numbers against every adjacent colour. The
gate now checks stacked segments against each other.

**The heading item had been held open by the fixes that were considered.** Both changed visual
weight. Hidden section headings change nothing on screen, and were available the whole time.

**A caption typo led to the round's real finding.** "0.644against" was a JSX line break, and it was
worth a gate on its own. On the medical page the same caption said "predicted -5.471", which no
probability can be. The chart had been binning logits, and it should not have existed on any binary
page, by a rule this project wrote down in three places and enforced in none. A rule recorded as
prose was correct for two months while a chart contradicted it on every visit. The fix that mattered
was the enforcement, not the conversion.

**The citation was the second layer of the same thing.** The rule was right and its source was not:
Riley et al. is the paper that says the rule of thumb can be too small, not the paper that sets it.
It had been marked METADATA-ONLY from the start, which is accurate, and nobody had gone back.

**One of my checks was wrong before the model was.** A first pass suggested the medical student
predicted one class for everything. It did not: my script applied the ResNet preprocessing to the
student, which takes no preprocessing. With each sample's own steps, all three accuracies reproduce
their records exactly. Reading the preprocessing from the sample, as the browser does, is the only
way to run these graphs.

**I committed a fix for the wrong cause, and the probe that disproved it found the real one.** A
console 404 after a run looked like a debugger fetching a source map the build names and does not
ship. I fixed that and deployed. A fresh tab then showed no error at all: the 404 was an old entry in
a long-used tab, the exact trap checklist row RS16 describes. Checking whether the map path could
404 showed that nothing on the site could - every missing file came back as the app shell with a 200.
The lesson is the order: reproduce in a clean context before explaining, and test the explanation's
own premise before shipping it.

### Round 75: parameters at their sources, and an MPC tier that was one party

**Context.** By the user's choice: every channel and protocol parameter read at the source it cites.

**Two wrong values can hide each other.** The satellite page's 5.5 kbit/s looked plausible for
Micius. It was the product of a loss budget missing about 8 dB and a noise floor 200 times too high.
Each value checked alone would have been caught; the total checked against intuition was not. The
gate now quotes Liao's sentence beside each value, and the model reproduces a Micius measurement it
was not fitted to.

**A test that adapts to its input cannot detect a failing input.** The QRNG health tests computed
their cutoffs from the entropy of the stream they were testing, so a biased source raised its own bar
and passed. The standard says "assessed", and the word was the whole defect.

**The bound used was not the protocol's.** The fibre rate used the vacuum-yield estimator for a
protocol that never measures the vacuum. The docstring disclosed it and called the result "tighter",
without the size. The size was 40 percent at 120 km.

**The MPC claim was true of the library and false of the call.** MPyC is a real Shamir
implementation. Run with its command line cleared, it was one party, and one party's share is the
value. Nothing looked wrong because nothing was measured: the communication figure was a constant
multiplied out. Making it three parties made the bytes real and exposed two harness defects at once,
a port drawn from the range outgoing connections use and a wait loop that stopped watching after the
first live party. The hang was reproduced before the cause was written down: 5 of 300, every one the
same port parity.

**Then the security claim beside it.** Reading the architecture document against the code showed a
sentence the code had never made true: the client "never holds a plaintext intermediate". It held
every one, because the conversions decrypted and the MPC opened its results to everyone. The user
chose masking now over correcting the sentence. The masks are measured rather than asserted: each
run reports its own statistical bound, and a test checks that what the key holder decrypts is a
masked value and that the parties' shares add up to the real one.

**The page found what the gates did not.** Both walk defects were consistent everywhere a test could
see: one rounding at every site, one formula in both languages. The first was a precision chosen for
one magnitude and wrong at another, and the second a label that was right until this round changed
what its g leaves out. Reading the panel beside the metric it prints is still the check that finds
these.

### Round 76: a set nobody had sourced, and the term it was missing

**Context.** The fibre detector values, which the config had labelled modelling choices since Round 35.

**A label is not a source.** "Modelling choice" was honest and still left the model 3x a measured
link's rate. The comparison that showed it needed a paper this file already cited for a different
number, cited since Round 47 for its block size; the measurement had been in the file for nearly
thirty rounds.

**The right source was the one already implemented.** Of five papers read, the one whose equations
`finite_key.py` runs states a complete parameter set, and a test was already evaluating the code at
exactly those values. Adopting it made the shipped channel and the validated channel the same thing.

**A clock is part of the detector model.** Moving to 1 GHz without a dead time would have multiplied
the bit/s by ten for free. The paper's own dead-time term cut the 50 km detections by a third, and the
optimiser had to learn it too, or it would have chased brighter pulses into saturated detectors.

**The literal was still in Python.** The browser read config; Python's channel defaulted two of its
five values from literals, and the fibre stage relied on them. The same defect class as the satellite
literals of Round 75, one dataclass over, and found only because this round changed the values.

### Round 77: a check that counted differently, and a horizon the model had no answer to

**Context.** The satellite atmosphere, then the minimum elevation, the satellite's own source, link
templates, and what carries a key past the horizon - each by the user's choice.

**A validation can agree for the wrong reason.** Round 75's "within 10 percent of Liao at 645 km"
counted every pulse as a signal; Liao counted signals only, which are half of them. The agreement was
the factor of two cancelling a model that was 2.9 dB low there. The tell was available all along: the
paper's own ratio of sifted to detected bits, 0.47057, is the signal-only figure to four places.

**A recommendation inherits the error in its evidence.** The 8 dB anchor was recommended on that
comparison and chosen on it. Corrected, the evidence pointed elsewhere, so the question went back to
the user with the new numbers rather than the old choice being kept as settled.

**Two measured passes are worth more than one fitted one.** Neither anchor fits both Liao's pass
total and its slope, and a second, independent Micius pass (Zvenigorod) shows the same 2 dB shortfall
in slope. That makes it a property of the model - a missing elevation-dependent term - rather than of
one night's weather, and it is stated as such instead of being absorbed into a fitted constant.

**A template is only as good as its weakest value.** The loader refuses a number with neither a
quote nor a stated assumption. Two candidates failed that standard in substance rather than form:
Jinan-1 would have needed its pass fitted to check its pass, and a 2026 atmosphere paper's axis did
not mean what its label said.

**Moving code moves what the gates can see.** Moving the satellite link builder out of the pipeline
broke a test that scans the pipeline for the keys it reads, and a new component looked unrendered to
a test that lists only tracked files. Both gates were right about what they could see; each was
widened to see the new place rather than silenced.

### Round 78: a convention changed for the user, and a recommendation withdrawn on the evidence

**Context.** Three user choices: rates per pulse sent; a low-elevation loss term; Jinan-1 as a
template, the last two only after verification.

**A convention change finds what the old convention hid.** Halving every rate moved every pin, and
re-deriving them surfaced a panel that had named one bound for three different protocols, and a
ratio printed as a literal two detector sets ago.

**I recommended a term the evidence did not support.** Round 77 framed the 2 dB residual as a
missing elevation term and asked the user which term to add. The verification showed the clearest
residual, Zvenigorod's, sits on one half of the pass only, and that the model's centre against all
23 of Liao's nights is within a few hundredths of a dB with 2.4 dB of weather around it. The term
went in as an option, by the user's choice, labelled with the one value it has to assume.

**A calibration is honest when it names what it leaves to check.** Jinan-1 could not be a template
without one fitted constant. Fitting it to a pass's level leaves that pass's shape untouched, and the
shape is what matched; the template file says which value was fitted and to what.

### Round 79: a bound built for the satellite found two errors in the fibre's

**Context.** Two user choices: a finite-key length for the downlink with Lim et al.'s vacuum+weak
analysis, and a block chosen by the published practice between seconds at the range and a whole pass.

**Writing a sibling is an audit of the first.** Lim et al.'s equations sit beside Rusca et al.'s, and
reading them side by side exposed two errors in the 1-decoy code three rounds of checks had passed: a
constant taken from a PDF text extraction, where a superscript 2 on 21 reads as "212", and an
unstated choice made the looser way. Both were on the safe side, so every claim survived; the
figures moved, and the rendered page, not the extracted text, is now where a constant is read.

**A ratio cancels what a level cannot.** Against Rusca's table both bounds read about half the
published rates, the same factor for both, so it belongs to the channel model they share. The ratio of
the two protocols cancels it and agrees with the paper within 15 percent; that is the check that says
the new bound is the paper's.

**The number that motivated the work did not survive it.** The case for the vacuum bound began as "the
1-decoy bound finds no key on Liao's pass". After the two corrections it finds 152,598 bits, and the
vacuum adds 8 percent. The documents now say that rather than the first version.

**Changing what a number means moves everything that reads it.** The key budget became a pass's, and
the control plane, which divides spent bits by channel time, went on dividing by the sampled fifth of
a second; found by reading its consumers, not by a test. The cross-language test then caught a source
the asymptotic bound had accepted since Round 77: the vacuum switched on where the profile sends no
vacuum pulses.

**Research decided the block; the user asked it to.** Every flown experiment computes its finite key
over a pass or over blocks inside one orbit; none on seconds of a low-orbit pass. The default follows
that, the alternatives stay, and the seconds option says how long a key would need rather than
pretending a fifth of a second could carry one.

### Round 80: a check for the new default found an old error in the old tests

**Context.** Two user choices: every fibre key a composable finite key, at every site; its block
decided by the published practice.

**Put the change where every caller already looks.** The fibre key had five readers in two
languages. Changing each would have invited the one-site-short defect this project has recorded
many times; changing the model they all call - `run_bb84`, `bb84()` - left them nothing to miss, and
the cross-language test holds the two models to the bit.

**The research's check was more useful than its recommendation.** It recommended the block every
fibre experiment uses, which was expected. It also handed back the paper's own block TIMES as a
check, and ours came out twice as long at every row. A rate that is twice as slow everywhere is a
channel, not a bound: the tests had applied the detector on top of an attenuation the paper defines
as including it. Thirty-three rounds, from the test's first commit in Round 47, had recorded the
resulting factor as this code's conservatism.

**An inference earns its keep when a later check can break it.** Round 79 chose the decoy's errors
for (A16) from one sentence of the paper. On the paper's own channel and dead time that choice
reproduces Table I's 26 dB row to a few parts in a thousand, and the other choice reads 13 percent low.

**A rounding is part of a metric's meaning.** The satellite's block time at one decimal was fine for
minutes of pass; the fibre's blocks run from milliseconds, and the rounding gate allows one rounding
per name. Significant figures serve both.

### Round 80, continued: a screenshot is a reading, and a move is an edit to every pointer

**Context.** The user held the merge until the README was reorganised: a summary with diagrams and a
screenshot of every page, the long records moved to split files and a short appendix in the report.

**An image taken for a document is a check nobody had run.** Three defects were on the deploy and
no gate looked for them: six of ten pipeline nodes undrawn after a run, a 199-character label wider
than its diagram, and every "log" label in the event log drawn as a box over the line above. Each
was found by opening an image to see whether it was fit to publish. Each test checked what a page
says; none checked what it shows. The screenshots now get regenerated with the pages and read like
any other output.

**A class built from data shares a namespace with the layout.** The event type became a CSS class,
and one type, "log", was also the name of the panel it is printed in. Nothing about either name was
wrong; their meeting was. A prefix keeps data out of the layout's names.

**Moving text is when a pointer's truth gets tested.** Twelve places said "the README" for text that
left it, and two of them had been wrong before anything moved: one named tools the README never
mentioned, one a citation it never held. A pointer nobody follows is never checked; a move forces
it to be followed.

**A gate that reads a file for a claim passes when the claim leaves the file.** Three tests read the
README for content; after the move they would have kept passing on a README that no longer held it.
Three more enumerate with `git ls-files` and could not see the new files until they were staged.
Moving text means asking each gate where it looks.

**A number copied from a summary carries the summary's error.** The log entry above said seven
rounds; the report's draft appendix took the figure from it. The first commit of the test says
thirty-three. When a record states a duration, check the commit that started it, not the sentence
that remembered it.

### Round 80, continued again: drawing the flowcharts was an audit

**Context.** The user held the merge for diagrams of the verification environment, the workflow and
each use case, in the report and the README, and for the pages' diagrams, which misbehaved during a
run.

**Look during the run, not after it.** Every defect the user saw showed while a run was in flight:
dimmed steps that read as empty, a status chip over the diagram, placeholder rows marked as running.
The after-the-run screenshots this round had already been taken and read showed none of it.

**Fixing that a thing shows is not fixing its parts.** The explicit node size of the first
continuation made the pipeline graph's nodes draw again; its edges had gone with the same cause and
nobody counted them. On the deploy the graph had no edges at all. A check that names the parts - nine
edges, at load, mid-run and after - is the one that would have caught it.

**A diagram of what implements each stage has to be read out of the code, both codes.** Asking, stage
by stage, what runs this in the browser and what in Python found a genomics sum over stand-ins under
a page that said allele counts, a Python pipeline that ignored two settings the page offers, and a
diagram written this very round that put a key into the session where the browser puts none.
Drawing the flowchart was the audit; the drawing came last.

**A width is measured in the font that draws it.** Wrapping by character count guesses at a font the
page does not choose. Measuring with the page's own font, and keeping room for the mark a step gains,
is what keeps a label inside its box in a browser the author never opened.

**Two copies of a figure need a test, or they are two figures.** The user chose TikZ and Mermaid over
a shared renderer; the test pairs them by key and compares every label, so either copy changing alone
fails.

### Round 80, continued a third time: a warning nobody reads, and a font nobody tried

**Context.** The user held the merge for text over lines in the report's Figure 1 and a label at the
edge of its box in Figure 2. The user asked for the same faults to be looked for everywhere, and for
the look to become a re-check.

**A warning nobody reads gates nothing.** The LaTeX log said "Overfull \hbox" nine times, a table 393
pt off the page among them, for as long as the report has been built. The build exited zero and its
output went to /dev/null. A warning becomes a check only when something fails on it, and
`make report` now fails on this one.

**A figure no command draws is a figure from the day someone last drew it.** The baselines figure
carried August's numbers for two months because only a hand ran its script. Now the build draws it,
and its inputs are in the staleness gate. Current figures are something the build guarantees, not
something a reader assumes.

**The first font is a sample of one.** The scan passed every page in the font this workstation
resolves `system-ui` to. In DejaVu Sans, the default on the CI runner and on many Linux desktops,
three labels failed. One was on a chart fixed earlier this same round by widening an axis. Widening a
box to fit one font moves the failure to the next font. Taking the unit off the tick, and keeping
margins sized for the widest common font, removed the dependence. Running the gate in two fonts,
locally and on the runner, is cheaper than guessing the visitor's.

**A check with no rule for a shape cannot see text over it.** The scan compared text with text and
text with its box. A title running into a badge dot was neither. Measuring the boxes' spare room
directly found it, and the scan did not. The scan gained the rule, and was then run on the build that
had the fault, where it reported it. When a measurement outside a gate finds what the gate missed,
the gate gets the rule and is seeded with that fault.

**A seed that never reaches the rule tests a different rule.** The first seeds for the Mermaid
diamond check shrank whole shapes, so the bounding-box rule fired first and the diamond rule never
ran. The CSS transform also replaced each diamond's own translate. Only narrowing a diamond by its
points, its text still inside the bounding box, exercised the rule the seed was for.

**A new rule is run first on the fault that prompted it.** The box-clearance rule passed every page,
including the label it was written for. It read a computed stroke width of "1.6px" with a unary plus,
got NaN, and so treated every box as unstroked. A rule that finds nothing on its own motivating case
is broken until shown otherwise.

**A failure path makes claims too.** When the request for a sample failed, the code returned the
value it used for "the Hub has none", and the page then reported the Hub's answer to a question it
never received. Scanning with the network cut was meant to keep CI offline, and it was the first run
that exercised that path. Run the pages once without the network: every message they print then
should be about the failed request, and none about the thing it failed to fetch.

### Round 80, continued a fourth time: a drawing of where things run is a claim about the code

**Context.** The owner asked for the E2E graph's kind of animation on the five use-case pages, and
chose swimlanes, the run's real parties, and replacing the sequence diagram.

**Drawing where a stage runs is an audit of its direction.** The sequence diagram had drawn the
satellite's key and telemetry going up for as long as it existed. Nobody misread the code; the
drawing never asked which way the link goes. A canvas has to put the source on one side, so it had to
ask, and the answer was in the satellite module's first line. A figure that only lists steps can
hide a fact that one placing them cannot.

**A rule written for one graph reaches every graph.** The global `.react-flow__node` rule padded the
E2E graph's nodes, and silently padded every node of the new canvas. Every lane ended inside its
boxes and every arrowhead was hidden. The first explanation, a measurement taken at the wrong scale,
was plausible and wrong. Measuring the node, the box and the handle took one probe and named the
cause. Measure before fixing a layout fault: a plausible mechanism can stand in for the real one.

**An attribute is not a style.** SVG markers take their colour as an attribute, where a CSS variable
does not resolve. The theme's colours had to be written as literals where they become attributes.

### Round 80, continued a fifth time: read every image the README shows

**Context.** The owner held the merge again: the README's screenshots looked stale, and the owner
asked for them to be taken after a walk and checked for bugs.

**An image is a test result only when someone reads it.** The eleven images had been regenerated,
committed and served for two rounds. Reading them one by one found five faults in pages that every
gate had passed: an unthemed control, an irregular axis, a line through six labels, two empty cards,
and a wrong word. Each was visible at a glance, and none had been looked for.

**A rule that sees geometry must see what is drawn.** The first version of the line rule reported a
curve over a tick label where the curve is clipped and never drawn. A second version accepted any
halo, and passed a build where the line still showed between two words. A rule earns its place only
when it is right on the cases it was written for, and seeding both ways is how that is shown.


### Round 81: a legend is a claim about the figure above it

**Context.** Round 80 merged and was walked on `main`. The owner chose tabs for the E2E page's two
views and a Party A and Party B canvas for its generic demo.

**One text for five figures is five claims, and two were false.** The canvas's legend and caption
were written once, for the use cases that compute together, and shown under all five:
- the telecom and satellite canvases explained lanes they do not draw;
- their captions described an inference their runs do not perform;
- the satellite's caption put it on a WAN.

Every check had asked whether the drawing was right; none asked whether the words under it were.
Building the legend and the caption from the rows the canvas draws makes them follow the drawing,
and the check now compares them with it.

**A node fixed at the start of a run is a guess about the rest of it.** The GIF recorder held the
element it was given when the run began. With the figure in tabs, a switch of view detached that
element, and every later frame would have been of nothing. The recorder now looks up its element at
each frame, and pads frames of different sizes, so a run that changed views still encodes.

### Round 82: a test of an inference, and a third reading of one equation

**Context.** Round 79 read the same fraction on two flown passes, 0.55 and 0.56, as the gap being
the analysis. The owner chose to test it with a Chernoff-type bound.

**A recommendation can be right for one case and wrong for the next.** The first measurement, on the
satellite, showed Chernoff far tighter, and the owner chose it for both protocols on that evidence.
Measured on the fibre, it was looser: its advantage depends on a count's share of its basis, which
the satellite's small decoy and vacuum have and the fibre's counts do not. The question went back
with the numbers, and the answer was the tighter of the two. A choice made on one case's evidence is
worth re-asking when the second case disagrees. [2026-10-30: re-asked once more, on the finding that
the union bound loses to Chernoff alone on the satellite blocks: Chernoff on the satellite and the
tighter of the two on the fibre. Register, Round 83.]

**A coincidence is a weak test of a cause.** The same fraction twice looked like one cause. Changing
the cause it named moved both fractions, by different amounts, and they parted. The statistics
closed 45 percent of one gap and 29 percent of the other, and the rest is not shared. (This said
"half of the gap" until the fix commit: a band wide enough to hold both numbers had been asserted
instead of either.)

**A table rounded to the third place can hide the claim it is quoted for.** "The tightest of the
three on every block measured" sat above a table whose satellite rows read 0.751 and 0.751; unrounded,
Chernoff alone was 17 bits ahead on Liao's pass. The fix gave the satellite its own default. And a
cross-language test only checks what both sides are asked: its harness called the bound without
naming the statistics, so it compared the fibre's default to the satellite's until it was told.

**A walk of the controls finds what a walk of the text cannot.** The fix's deploy passed every
content check, and a headless pass that pressed every button found five faults the text walks had read
past: an export that drops columns because its header comes from the first row, a 3D view rebuilt on
every render, and three disabled-or-enabled states that say nothing visible. Pressing is a different test
from reading, and each needs its own row.

**An equation read three times found three errors.** Round 79 read Rusca et al.'s appendix and found
a constant transcribed as 2 to the 21st and an unnamed choice of errors. Reading (A16) again at 300 dpi
for this round found tau0 on a term the paper prints outside it. That one erred in the unsafe
direction, and it had matched the paper's own table better than the printed form does. Agreement with
a published number is not evidence that a formula is the published one.

### Round 83: a label is a claim about the code under it (2026-10-30 to 2026-10-31)

**Context.** Five page fact-checks, a review of private inference against current research, a QCNN
review and a documents audit found public text that claimed more than the code does. The register's
Round 83 entry lists each correction with its evidence, and the walk of the deployed branch on
2026-10-31; this entry keeps what they taught.

**A name carries a property, and the property has to be checked in the code.** "Secure
aggregation", "private set intersection", "information-theoretic", "QHE" and "held-out" each named a
property the construction under it does not have: masks derived from a key the aggregator holds, a
hash both parties can compute, a seed drawn from the extractor's own input, a circuit that is never
encrypted, a model's own fitting rows. None of them was found by reading the label again. Each was
found by reading the code beneath it.

**A test that reads one slot proves one slot.** The masked conversion's test decrypted slot 0 and
passed while the other slots held the weights. When a property is about a whole ciphertext, the test
has to decode the whole ciphertext.

**A union bound is safe, not best.** Taking the tighter of two bounds, each at half the failure
probability, never fails more often than allowed, but it can give less key than the better of the
two alone at the full probability, and on the satellite blocks it did. Round 82 called it the tightest of the
three without measuring that on every block.

**A record can overclaim as easily as a page.** This file and the register said the statistics
explained about half of each gap. They explained 45 percent of one and 29 percent of the other. A
fraction that fits one case is not a summary of two.

**A text the browser fetches is checked where the browser fetches it.** The generator's sentence
claiming an FHE/MPC pipeline on the satellite page was corrected in the repository, and the walk
still found it in the satellite run log: the page prints the evaluation sample's provenance from the
Hub, at the pinned revision, and that sample had not been republished. A correction reaches a
visitor only through the artefact the page loads.

**A count copied is a count to recompute.** The fix commit wrote Liao's pass as 226,073 bits and the
default pass as 139,201, and the two figures were copied into six places, each one bit high. Both
languages compute 226,072.49 and 139,200, and the page showed 226,072 and 139,200 throughout. The
copies agreed with each other, which is why no copy looked wrong.

### Round 84: a licence, a citation and a comparison each have to be read where they apply (2026-10-31 to 2026-11-01)

**Context.** The defects drafted on 2026-10-30, a re-sweep of every page against post-August-2026
sources, the owner's three decisions (a two-basis satellite key, the genomics score's licence, both
fibre intensity pairs selectable) and a documents consolidation. The register's Round 84 entry lists
each change with its evidence; this entry keeps what they taught.

**A licence is read where the data sits, not where it is indexed.** The PGS Catalog lists PGS000004
openly, and the Catalog's own terms are permissive. Its weights, though, appear only in the
supplementary data of an article licensed CC BY-NC-ND 4.0, and that is the licence that travels with
them. The score the page serves now is one whose source article is CC BY 4.0.

**A comparison with a published figure compares the conventions first.** The satellite pass's finite
key read 0.55 to 0.75 of Liao et al.'s, and Rounds 82 and 83 explained part of the gap with the
statistics. The flown keys were counted on both bases; this model counted one. Under the
satellite's default statistics, 95 percent of the drop from the asymptotic key was that convention.
Counted the same way, the model's key is 1.50 of the published one.

**A feature can carry the label through the data's own process.** Two of the finance model's eight
columns were built from PaySim's balance fields, and the dataset's description says that
transactions detected as fraud are cancelled, so those fields "must not be used" for fraud
detection: their values record what happened to a fraud. The score was partly reading the answer.
Nothing in the model's metrics showed it; the dataset's documentation did.

**A citation to a moving identifier changes meaning under it.** The OPS-SAT-AD concept DOI resolved
to version 1 when it was cited and to version 2, with different files, by this round. A citation
names the version the files came from.

**A control reaches a run only if it reaches every reader of the run.** The intensity pair is put on
the run's channel, where the stage, the bound, the security facts, the distance chart and OptiQKD's
baseline all read it, rather than passed to the stage alone. The test runs both languages' stages
at both pairs, and it was seeded with a browser stage that ignored the choice to show that it fails.

**A figure rounded is not the figure a run prints.** The second pair's key had been quoted as
4,977,637; the bound gives 4,977,636.9 and the run reports whole bits, 4,977,636. The same shape as
Round 83's copied counts: the quoted figure agreed with itself everywhere it was copied.

**A fact told in seven documents has seven copies to correct.** What the browser does not protect
was stated in seven files, so every correction had to find all seven, and some did not. Each such
fact now has one owner holding the full statement; the other copies are a clause and a link, and a
test checks the links resolve and the owner holds the fact.

**A gate that enumerates one list checks that list.** The attribution gate read the datasets attached
to a model. PaySim is attached to none, though its rows ship in the bundled demo sample under CC BY-SA
4.0, so its empty credit line passed from the round that wrote the gate. The gate now reads every
dataset the registry defines.

**A deferral is a promise with a list, and the list has to be read again.** Round 83 deferred eight
changes of behaviour to this round. The drafted defects round and the sweep did three of them and part
of a fourth. The rest (the retired score in the full deployment, the explanation flag and the signed
record's backends, the pairwise mask keys, the OPRF-based PSI, and the licence fields of the shipped
samples) were found still open only by reading Round 83's list again near the end of this one.

**A key both parties hold cannot keep them apart.** The PSI's HMAC key was the session key, which
both banks hold, so it hid each list from third parties and from neither bank. The OPRF's key is
held by Bank B alone, and the session key now does the one job a shared key can: it seals the
transcript.

### Round 85: a walk has to run the way a visitor runs, in one tab, for a dozen runs (2026-11-01)

**Context.** The owner reported finance's inference verification failing and asked for every page to
be walked again with selections other than the defaults, every option and every pair. The register's
Round 85 entry lists the changes and their evidence; this entry keeps what they taught.

**A check that starts from a fresh page cannot see what accumulates.** Every earlier walk ran each
combination once, often in a new tab, and every run passed. Run one after another in one tab, as a
visitor pressing Start does, the finance page stopped responding at about its fourteenth run, with
nothing else changed. The defect was not in any one run: each run left over 300 MB in a WebAssembly
heap that never shrinks.

**Memory outside the JavaScript heap has no collector.** node-seal's ciphertexts and plaintexts live
in a WebAssembly heap, and ONNX Runtime Web's sessions keep their weights there: neither is freed
when the JavaScript object that names it is collected. A comment in the CKKS code said node-seal's
finalizers would free its objects eventually; node, with the collector forced between runs, measured
the heap growing by the same amount every run. The test now measures the heap, not the code.

**A heap that grows in steps hides a small leak behind a large one.** Run after finance in one
process, telecom's and genomics' leaks fitted into the room finance's had reserved, and the first
version of the heap test passed them on the base commit. Each use case now runs in a process of its
own.

**A browser without a GPU walks a different page.** Headless Chrome offers only a software adapter,
so the page disables its WebGPU option there, and every headless walk ran WebAssembly only. A path
that needs hardware is walked on hardware.

**An option that can only fail should not be offered as a choice.** The satellite page offered a
model that publishes no evaluation sample by design, so selecting it could only end in "did NOT
run". The Models panel already explained it; the selector did not.

**The word on a mark is a claim too.** A step the model withholds by design was badged "failed", in
red, on 33 of the satellite page's 38 combinations. The badge now says "withheld" or "no key"; a real
error keeps "failed".

### Round 86: a release is a claim about every copy of its version, and a tag publishes its whole tree (2026-11-02)

**Context.** The owner chose to release 0.85.0 as a tag with release notes only, and to keep the
technical report private, with a new summary of the verification gates beside it. The register's
Round 86 entry lists the changes and their evidence; this entry keeps what they taught.

**A version written in five places is five versions.** `pyproject.toml`, `qsh.__version__`, two
literals in `api/main.py` and `web/package.json` each said 0.1.0, and none read another. A bump would
have shipped whichever copies it reached. There is one literal now, and the copies npm has to carry
are compared with it.

**"No assets" does not mean "no report".** GitHub builds a tag's source archive from the tagged tree,
so a report still tracked at the tagged commit would be published inside the archive. The order is
fixed by that: the report leaves the tree, then the tag is cut.

**An ignored directory is no home for another checkout.** git overwrites or deletes an ignored file
when it checks out a commit that tracks it, and `.gitignore` already records a round of the
checklist lost that way. The private checkout went beside this repository instead of into `report/`.

**A rule written for one reason fires on a change made for another.** The dropped-PDF rule exists for
third-party PDFs without a redistribution grant; moving the project's own report tripped it. The
exception names four paths rather than a pattern, so it cannot widen into the case the rule is for.

**A check that spans two repositories should read the owner's list, not copy it.** The report's tests
load the public tests' retired wording and flowchart nodes from the public files; a copy would have
been a third version of each list, waiting to drift.

**A layout rule is a gate only where it reads the output.** The summary's size rule is checked on the
font sizes the PDF's own content streams set. That reading caught microtype's font expansion scaling
glyphs through the text matrix, which changes no size but made the reading impure, and it is why the
rule could be seeded at all: a seven-page PDF in small type without page numbers fails it three ways.

**A status in a checklist is a claim with a date.** Seven rows read "FAILS LIVE" for rounds after the
defects round fixed them, so a reader of the checklist would have planned work already done. The audit
re-measured each on the deployed site, and the rows now carry what was measured and when.

**An outline is not a colour.** The tick style set `stroke`, and to the eye the labels looked light; the
text's fill was recharts' #666, which is what contrast is computed on and what a magnifier or a
high-contrast mode sees. The fix sets the fill and drops the outline.

**A measurement has to be taken where the defect lives, from the start.** The layout shift existed only at
desktop width: zero at mobile, where overlay scrollbars and a narrower card wrap differently. The app's own
small browser pane read 0.095 from its 317-pixel viewport, and an observer registered after the load read
zero; only an observer installed before the page's scripts, at a full desktop size, read the real 0.28.
And the audit's first pass capped its samples at eight a page, which the dominant tick failure filled:
the legend and badge failures surfaced only once the ticks were fixed and the scan reported everything.


**A fix that reaches one twin leaves the leak where the claim is made.** Round 84 masked every slot of
the hybrid's dense layer and convolution, and the threat model then said every slot the key holder
decodes is masked. The linear inference that produces the full deployment's reported class decrypted
its rotate-and-sum whole, as both twins had: it was the copy nobody had read for that defect, because
its output looked like a different thing, a class rather than a layer. The fix went into the function
the four callers share, and the test names every server decryption, so the next twin is an AST failure
rather than a review.

**A leak detector needs a seeded positive beside its negative.** The masking test asks whether any
linear combination of the other slots predicts a weighted product; a detector that never fires passes
that too. The same detector is run on the unmasked reference path and required to recover the products
exactly, so a broken detector fails.

**A comparison is a check only if NaN fails it.** Each precision check read `err >= 1`, which NaN makes
false, so a NaN feature decrypted to the bias and reported nothing. A check written `!(err < 1)`, with
the value refused before it is encoded, cannot pass what it never measured.

**A self-test that shares the computation's quantisation cannot see the quantisation.** The argmax's
self-test ran the same argmax in the clear on the same 1/64 steps, so a tie the steps decided agreed
with itself. The float argmax is the reference that can disagree.

**A recommendation is checked against the code's own record before it becomes a plan.** The review
proposed a quadratic finance student on the strength of an OPS-SAT gain; the finance trainer records
that variant as measured and rejected, and the OPS-SAT "before" figure is recorded nowhere, as the
trainer's withdrawn-claim note says. The plan carries the trainer's measurement, not the proposal.

**An expected effect is a measurement not yet taken.** The masked logit's error was written down as
about 2e-6 against 1e-7 unmasked, the masks' noise added on top; measured over 200 queries it is 6.3e-7
against 5.4e-7, because the dot product's rescale dominates both. The sentence was corrected before it
was committed.

**A history rewrite is finished only when every copy of the old history is accounted for.** The new
repository was clean within the hour, and the old history then turned up in other clones: this
machine's (its branches, and twelve scratch worktrees whose detached heads kept it alive) and the demo
host's, one of them holding uncommitted work that no other copy had. A push from any of them would have
restored the report to the public history. Each was checked against the backup before it was removed,
and the one thing found nowhere else was saved first.

**A style enforced against the wrong delimiters still lets a formula through with none.** The house
form was tested for single-dollar math, so "2^30" in plain prose passed for 30 rounds. The new check
reads paragraphs, not lines, because code spans, inline math and display blocks cross line breaks, and
it leaves quotations alone, because a source's own typing is part of the quotation; its first catch was
in the entry that introduced it.

**A label is a claim about a format.** "MCP-style" and "JSON-RPC envelope" hid that the messages were
notifications by shape: no id, so nothing could answer them. Validating against the upstream schema took
minutes and would have shown it at any point. Saying "depends on none of that" in the references was true
only because the messages were not what they were called.

**A comment saying two implementations agree is not a check, again.** "The exact same bytes" sat beside two
serialisers that differ by a space. It had no consequence, because `body` travels as the signed string, which
is exactly why nothing ever caught it.

**The exception in a rule is part of the rule.** The gap analysis quoted the very-short-text exception as if it
covered marking in general; it is written into the watermark sub-measure alone. The file then repeated the
reasoning about itself, in a note saying a file cannot carry metadata.

**A superseded figure needs its date on every copy.** The model card moved to 0.953488 in Round 84 and was gated, so it stayed right.
Two references and a log note kept "the shipped figure is now 0.883721" for two rounds, because a gate holds the card and a sentence
saying "now" has no owner. The same shape produced "the teachers already sit at or above their published baselines", true of one of five.

**Reading the PDF settled what a note could not.** The registry already said 0.11 was the recall; `metrics.py` two directories away still
quoted it as the F1. Table II of Engelen et al. shows both on one row.

**A dependency's default is a decision someone else made.** Transformers.js chose a CDN and WebLLM chose a repository with no licence file, and both
were visible only on the wire or in the library's source. "Every request goes to the demo or the Hub" had been measured before either engine was
run in a capture, on a page that fetched no model.

**A build can publish its builder.** The libraries were tested, hashed and ready before a look at their strings showed the local home directory
221 times. Nothing in a functional test sees that. The script now refuses its own output if the directory appears.

**Testing the fix found the next two defects.** Running every model for real, rather than one, found the wrong size on a label, and that a lost request
ended a 2 GB download. Cancel was only testable with a library present: without one, the local preview answers a missing `.wasm` with `index.html` and a 200.

**A checklist row is a check only once it is run on this project's own build.** G1, list the GPU architectures compiled into a shared
library, was written for third-party wheels. The HEonGPU module built here carried only Turing's for a Blackwell GPU, and every test
passed, because the driver recompiles PTX at load. The build script runs the check now.

**A cause is established by varying it.** OpenFHE's 594 ms was put down to other load, and an idle machine then read 577. Varying the
thread count, not the load, moved it: 81 to 577 ms at 20 threads, 41 to 54 at 6.

**A fallback written for a missing capability outlives it.** `_hybrid_bench_row` sent every backend it did not know to TenSEAL, which
was right while HEonGPU could not run. The day it could, the same line split one run across two backends under one selection, and
nothing failed.

**A reader's list is checked against the code before it becomes work.** Of twelve functional findings written down on
2026-10-30, three were fixed outright and three in part by the time the readers re-read them on 2026-11-05, and the two
that a reader marked unsure were both still live. Each item was re-read on main before anything was changed.

**Two spellings of one key are two twins.** `student_params` and `n_parameters` both hold a student's size, and the
readers asked for one; `event_wise_f0_5_full_channels` and `published_full_channel_f0_5` both name one figure, and the
card printed the one no record has. A check that reads the record, not the spec, found the second kind on five cards at once.

**A threshold has one owner.** The QKD stage aborts at `qkd.pp_threshold`, and four readers of the same idea took it from
the retired score's config, equal to it only by value. Removing the score would have left them pointing at nothing; moving
them first is what made the removal a deletion and not a breakage.

**A decision made and only half done reads as done.** The owner chose a decision curve for finance in Round 29; the
citation landed and the curve lived in the technical report, so when the report left the repository the page had
none, and nothing listed it as owed. Re-reading the orders against the page is what found it.

**An enriched sample is right about rates and wrong about the rate.** The finance sample draws each class on its own,
so its true- and false-positive rates are fair and its 2 percent is not. Weighted by that 2 percent, net benefit at
t = 0.01 reads 95.4 per 10,000 where the whole window gives 5.44; at the window's own rate the sample reads 6.26, with
an interval that holds the window's value at 40 of 43 thresholds. Measuring against the window, not only reasoning
about it, is what turned the correction from plausible into checked.

**An open issue is not a missing feature.** The 2026-11-01 re-check kept marimo declined on its offline issue #575.
The release notes of 0.25.0 had shipped `--offline` 39 days earlier, and a probe with every other host unresolvable
ran an exported cell. Test the capability, not the tracker.

**A row marked pending is a claim about the past.** Thirty-nine checklist rows said "PENDING (defects round)" for a walk
on 2026-10-30. Re-read against main a week later, twenty-six were already fixed and nobody had said so, and two of the
rest were in files published on the Hub, which no test of the tree reads.

**A failed build can leave a result behind.** hpmpc's Makefile went on after a compiler error and kept the previous
executable, so the first GPU benchmark timed the matrix-multiplication test built before it: flat at 0.2 s from $`N = 128`$
to 4096, with almost nothing sent. The sent volume, beside the CPU run's at the same N, is what gave it away; the script
now checks the build log, the executable's age and the run's output before it records a time.

**A benchmark that checks nothing runs without the hardware.** With the GPU hidden, function 36 still finished. The
correctness test is the control: it passes with the GPU visible and its matrix multiplication fails with it hidden.

**A credit has one owner too.** The OPS-SAT dataset record was appended by two payload builders and absent from the
registry, so every file built from the registry, the cards and the evaluation samples, credited the article alone.

**A field a script adds to a record is carried forward onto the next one.** The student's calibration map was first
fitted by a script of its own, on the published weights. `record_metrics` carries any field a new payload lacks from
the old record, so the next retrain, writing only the fidelity block, would have left the old weights' map beside new
weights, and the page would have applied it to their logits. The map is now fitted in the run that fits the student,
and a test fails if the trainer's call leaves it out.

**A map is a function of the argument it was fitted on, and every surface must hand it that argument.** Fitted on the
student's decision logit, the map was right for the private-inference figure and wrong for the model panel, which hands
every published graph's probability to the same function once the config names the student there. Refitted on the
probability with `on_logit` set, as the teacher's is, the two surfaces read 4.57 per 10,000 at $`t = 0.01`$ from two
different routes, the decrypted logit and the graph's float32 output, and so does the measurement.

**A record states what was measured on its own model.** `Platt.as_dict()` publishes the teacher's Round 49 isotonic
comparison as its retention argument, so the student's record would have carried the teacher's numbers. Measured on the
student, isotonic wins on both proper scoring rules, and the record says that, with the reason Platt is kept, in words
derived from the numbers so a retrain cannot leave the sentence saying the opposite.

**A published page is read where it is published, in that site's dialect.** The calibration card passed every local
check, its numbers against the record included, and on the Hub its inline formulas printed as raw text with their
backslashes, because the Hub reads `\\(...\\)` and not single dollars, and its display formula ran past a column that
does not scroll. Both were found by reading the published card and measuring it, and both are now tests of the
generator; the measurement that closes each is on the Hub, not in the tree.

**A computation with no test gets one before it is timed.** hpmpc checks its matrix multiplication and not its
convolution, and a benchmark checks no result, so a GPU convolution figure could have timed anything that finished. The
script now runs fixed integers through the layer on every path against a plain convolution and NumPy, requires the GPU
paths to fail it with the GPU hidden, and removes the check before timing. A build flag chosen to get past an upstream
defect, `PRINT_TIMINGS=0`, is recorded with the defect, and the claim that it was needed was checked by building without
it.

**A build that finishes can still install the wrong thing.** The OpenFHE 1.6.1 build compiled, linked and installed,
and the venv went on importing the old binary: CMake had built for another Python, and the install step took the first
`openfhe.cpython-*.so` it found. The import failed with the error the build exists to fix, which is the only reason it
was noticed. The script now names the file by the venv's own ABI suffix and stops when that file is not there.

### Round 86, PRs #7 to #12 on the deployed site (2026-11-07 and 2026-11-08)

The demo host was returned to `main` after each merge batch and read in a browser on the public URL. Measured there:

| page | what was read |
|---|---|
| satellite | The student's verification runs in a Web Worker: 3 long tasks, 381 ms blocked over a 30 s run, CKKS 21.05 s, maximum error $`1.95\times10^{-9}`$, 0 of 8,640 flags changed |
| satellite, OPS-SAT | The sample reproduces the card's accuracy, 0.9660, and no reproduced-figure note appears |
| satellite, Network tab | Khmelev et al.'s session is counted from 28 degrees (227 s in the model against their roughly 220 s), 415,669 and 829,658 bits against 629,000, 1.32 times |
| medical | The malignant curve at 30 of 42 (0.714); the timing row split into the student and the regression (145.67 and 34.94 ms); the int8 reason on the card; the IID caveat in the log |
| all pages | Every verification passed and the console showed no error |

`main` was then `2d9210d`, bundle `index-DaZW2Hat.js`. After the kernel and glibc update of 2026-11-08 (5.14.0-687.66.1, glibc 2.34-284), `scripts/ci_local.sh` passed all 25 gates on that commit.

### Round 86, PR #14 on the deployed site and the student cards republished (2026-11-08)

`main` was `d1c8474`, bundle `index-BmS-bF8M.js`, read in a browser on the public URL.

| what | read |
|---|---|
| medical page, `#sel-mpccnn` ticked by default | One held-out image through the student under MPC: class "normal/benign", "matches the clear network"; 800 ReLUs in 15 batches, 69,167 AND gates, 174.4 MB in 5,235 messages, 28 rotations and 6 conjugations; 18.5 s in a Web Worker (garbled circuits 16.5 s, compiling them 1.4 s, CKKS 0.2 s); statistical distance at most $`2^{-22.7}`$; largest logit error $`1.89\times10^{-4}`$ |
| medical page, unticked | No MPC card and the verification passes |
| finance page | The verification passes and the page offers no MPC control |
| console | No error on either page |

Republished on the owner's approval, each file fetched back and compared by hash with the local file (the only change to `weights.json` in each is its `note`, a JSON comparison with the old revision showed): `qsh-medical-breastmnist-ckks-student` at `b582844a`, `qsh-finance-aml-ckks-student` at `e822202d`, and `qsh-satellite-opssat-ckks-student` at `58450bb4`. The page reads these pins from the next pull request on.

**An interval is part of an AUC on 156 images.** The served ResNet-18 graph scores AUC 0.9146 on the official test split (the trainer's own figure, in PyTorch, is 0.9147: float32 export moves a pair or two of images). DeLong's 95% interval is 0.858 to 0.971 and a stratified bootstrap gives 0.853 to 0.964, so the published 0.901 lies inside it and the +0.014 is not a result. The card now prints the interval beside the AUC, and for ResNet-18 the spread over six earlier runs differing only in the seed.

### Round 86, PR #15 on the deployed site and three card-only republishes (2026-11-08)

`main` was `6a3c1b7`, bundle `index-r3dPuWGu.js`, read in a browser on the public URL: the five static routes, the five use-case pages, and the three negative paths.

| what | read |
|---|---|
| `/` | The BSI sentence reads "read 2026-11-08" and names RFC 10024. No horizontal overflow |
| `/value` | Both IR 8547 rows read `announced`; the rest of the clock is unchanged |
| `/validations` | All eight Run buttons complete, none FAIL; Download results enables after them; the diagram reads "ML-KEM-768 (draft) + PPK (RFC 8784)" |
| `/orchestration` | Start, Pause (`pausing`, then `paused`), Step, Resume to `done`; the use-case select disabled during the run; exports disabled before it and enabled after |
| `/agentic-ai`, `/usecase/nope`, `/no-such-page` | Redirect to `/usecases`, "No such use case", "Page not found" |
| finance | PASS. The student's agreement row prints the 0.9995 reference and the counts 759 and 26,151. Brier skill +0.0652 and -0.0124. The curve is below zero at 5 of 43 thresholds, and says what that means. The SHAP bar reads `transfer_or_cash_out` |
| medical | PASS. Each `auc` row prints its 95% interval; ResNet-18 also the spread over six seeds. CNN under MPC ran and matched the clear network |
| genomics | Done. The visible first sentence of the baseline carries the covariates and the 162 of 187 SNPs |
| telecom | PASS. The corpus is named "signatures of this project's own design"; "No public real QKD-attack capture was found"; ML-KEM-1024 and ML-DSA-87 with QKD off give a 4,627-byte signature and "protocol: none". Events, Metrics, PNG and GIF each produced a file; the agent ran and returned its envelope |
| satellite | PASS. The metric reads `qsdc_eq8_bound` = -80.82 |
| console | No error on any page |

Found there and fixed on the next branch: the Value page's prior-art row still described only the decrypted path; the clock's legend did not explain `announced`; at scale 1 the measured and scaled MPC rows printed the same numbers; and 26 flattened `calibration.*` rows pushed each finance card's headline metric off the screen. Mobile width (375 px) was read on the local build, because the Chrome window could not be narrowed: no horizontal overflow on the finance page after a run or on `/value`.

Republished on the owner's approval, each file fetched back and compared with the local file: `qsh-finance-aml-gbm` at `eb15e2a9` (card only), `qsh-telecom-5g-nidd-gbm` at `1c198d29` (card only), and `qsh-telecom-qml-ids` at `78ef4189` (card, and the sample's provenance sentence: a JSON comparison with the old revision found no other field changed). The finance teacher's and student's decision-curve records were re-measured at their new revisions, and their results are identical.

### Round 86, PR #16 on the deployed site (2026-11-08)

`main` was `8d4ac02`, bundle `index-CDtBtyom.js`, read in a browser on the public URL.

| what | read |
|---|---|
| finance, three banks (the default) | 3 accounts held by all, matches the clear intersection; 4,344 AND gates, 30.5 MB in 1,262 messages, 2.9 s in a Web Worker |
| finance, five banks | 8,440 AND gates, 191.6 MB in 6,884 messages, 14.7 s in a Web Worker; matches the clear intersection |
| finance, unticked | No PSI card, no `mpsi_*` metric, no lane; the bank count is disabled; the verification passes |
| WebM | `finance_aml-simulator.webm`, 260,486 bytes, `video/webm`, decodes to 3.56 s at 1418 by 2530 |
| the four fixes from the previous walk | The prior-art row, the clock's legend, no repeated measured MPC rows at scale 1, no flattened `calibration.*` rows |
| pins | The cards show `eb15e2a93616`, `e822202dacbc`, `78ef41898e3d` and `1c198d29f934` |
| medical, telecom, satellite | Each passes; no horizontal overflow; no console error on any page |

### Round 86, PR #17 on the deployed site (2026-11-09)

`main` was `786bb13`, bundle `index-BFh2pMO-.js`, read in a browser on the public URL.

| what | read |
|---|---|
| the CBOM card on the Validations page | 21 rows, and the JSON file it links to parses |
| the type scale | body line height 24 px on 16 px text, card headings 16 px |
| every page | no horizontal overflow and no console error |

### Round 86, PR #18 on the deployed site, and what each page uses in the browser (2026-11-09)

`main` was `12e0a6a`, bundle `index-DoS3Lbnq.js`, read in a browser on the public URL with no other load on the client.

| what | read |
|---|---|
| the lending page | 5 checks, all matching the clear check; the margin from both shares within 0.020 stable units; 4 Web Workers, 3.6 cores' worth of work, 91% busy, 1,325 ms |
| the Use Cases page | six cards, the table row reads "no model: a margin under CKKS, its sign in 2PC" and "none" for the quantum channel |
| the E2E page | the selector offers the lending page; on the satellite page the quantum controls are still there |
| every page | no console error |

One run per page, on a machine with 20 logical cores, cross-origin isolated, with `navigator.gpu` and `navigator.ml` present. "Long tasks" are the main thread's, from a `PerformanceObserver`.

| page | run | long tasks on the page's thread | WebAssembly | Web Workers |
|---|---|---|---|---|
| finance | 10.9 s | 12, 2.7 s in all, longest 414 ms | ONNX Runtime (threaded), SEAL, Polytune | the PSI among banks |
| medical | 26.5 s | 12, 3.1 s in all, longest 746 ms | ONNX Runtime (threaded), SEAL, Polytune | the CNN under MPC |
| genomics | 2.1 s | 9, 1.6 s in all, longest 291 ms | SEAL, Polytune | none |
| telecom | 4.2 s | 11, 1.5 s in all, longest 230 ms | ONNX Runtime (threaded), SEAL | none |
| satellite | 28.6 s | 10, 1.1 s in all, longest 189 ms | ONNX Runtime (threaded), SEAL | the forecaster student |
| DeFi lending | 1.3 s for the checks | 6, 0.6 s in all, longest 191 ms | SEAL and Polytune, inside the workers | four, for the sweep and for the checks |

WebGPU is optional and off by default: the published models use WASM unless it is selected, and the agent panel's models can use it. SEAL and Polytune have no GPU path. ONNX Runtime takes half the logical cores up to four threads.

### Round 86, PR #19 on the deployed site (2026-11-09)

`main` was `87fceeb`, bundle `index-Cet92AlO.js`, read in a browser on the public URL.

| what | read |
|---|---|
| the default run, a separate key holder | 5 checks, every bit matching the clear check; every signed bit verified; one submission from Bob; 4 Web Workers |
| the same page, seed 3 | 9 checks |
| the borrower holding the key | 9 submissions, one per check |
| every page | no console error |

### Round 86, PR #20 on the deployed site (2026-11-09)

`main` was `e2a0fb3`, bundle `index-ARJpglst.js`, read in a browser on the public URL.

| what | read |
|---|---|
| the telecom page, 5G-NIDD selected | the pipeline names `telecom_ids.nidd @ 5852153ee411`; the verification passes, 0.9976 on all 16,379 rows of the sample |
| the card | the table of the rows as published and the rows used: accuracy 0.7616 and 0.9979, macro F1 0.9275 and 0.9972, ceiling 0.7685 and 1.0000, across base stations 0.4199 and 0.6059, and 0.9973 and 0.9877 |
| the calibration chart | withheld, "34 wrong predictions and 16,345 right ones", under the 200 events it needs |
| the 2D plot | the rows this model was scored on, 4,096 of them, AUC-PR 0.9956 against a base rate of 0.8052 |
| every page | no console error |

### PR #22 on the deployed site (2026-11-09)

Deployed from merged `main`, bundle `index-BuktvS8o.js`, read in Chrome on the public site.

| page | read |
|---|---|
| the telecom page, 5G-NIDD selected | the card's link, "Check the two copies yourself", opens `/walkthrough/` in a new tab |
| `/walkthrough/`, loading | cross-origin isolated under `require-corp`; all 250 resources came from the site itself; no Python error |
| `/walkthrough/`, the result | 105 of 113 columns identical over all 200 pairs, and the eight that differ are the three labels and five columns that place a row in its file; the ceiling 0.7685 |
| `/walkthrough/`, the slider | moving it re-ran the table: `Unnamed: 0` went from 235186 to 236603 between pairs 2 and 3 |

### PR #23 on the deployed site (2026-11-09)

Deployed from merged `main`, bundle `index-CeQi94Vk.js`, read in Chrome on the public site at 0, 5, 10, 15 and 27 s.

| where | read |
|---|---|
| the satellite page, before a run | the Downlink tab draws three curves, four published points and ticks from 10 to 60 dB |
| the run, in order | QSDC NOT AVAILABLE at $`C_s/Q_{\mathrm{Bob}} = -80.8180`$ and $`g = 532.7`$, then quasi-QSDC delivered the 74-byte telemetry; the word "falls back" does not appear |
| the table | delivered, QBER 1.12%, Eq. 3 bound $`3.36 \times 10^{-5}`$, 98,512,944 signal pulses, 1.97 s, 98,512,944 bits reserved, 11,847 spent, 3,306 regenerated |
| each published point, on hover | Pan et al. at 23.55, 29.71 and 34.45 dB and Sun et al. at 31.32 dB, each beside the bounds at that loss |
| the page | no console error and no horizontal overflow |

### PR #25 on the deployed site (2026-11-09)

Deployed from merged `main`, bundle `index-lyVicXRc.js`, read in Chrome on the public site.

| where | read |
|---|---|
| the finance page, published model | the run's `published_model_sha256` equals the value recorded for the file it fetched |
| the network, one run on each of five pipeline pages | finance 37 requests, genomics 10, telecom 23, satellite 32, DeFi 15; every one a GET, to the site or to the model host and its storage |

### PR #26 on the deployed site (2026-11-10)

Deployed from merged `main`, bundle `index-BUsEwUaY.js`, read in Chrome on the public site with Inference set to "QCNN (trained, under CKKS)".

| page | products encrypted | evaluation | up, down | wall-clock | largest difference from the circuit in the clear |
|---|---|---|---|---|---|
| finance | 81 in 1 ciphertext | 1 inner product, 12 rotations | 0.39 MB, 0.39 MB | 307 ms | $`3.1 \times 10^{-10}`$ |
| medical | 6,561 in 2 ciphertexts | 1 inner product, 12 rotations | 0.79 MB, 0.39 MB | 594 ms | $`1.3 \times 10^{-9}`$ |
| telecom | 2,187 in 1 ciphertext | 8 inner products, 96 rotations | 0.39 MB, 3.15 MB | 709 ms | $`5.7 \times 10^{-10}`$ |

On each page the run recorded `primary_inference_backend` as `qcnn-ckks`, the decrypted class equalled the class of the same circuit in the clear, and the panel listed the held-out figures beside the classical comparators (finance minority-class F1 0.2286, logistic regression 0.4390, the 81 products 0.4889; medical accuracy 0.9535, 0.9535, 0.9419; telecom macro F1 0.9444, 0.9764, 0.9723). The circuit card drew finance's 90 gates, medical's 206 and telecom's 895, the last in a box that scrolls; no page scrolled sideways and the console held no error.


### PR #27 on the deployed site (2026-11-10)

Deployed from merged `main`, bundle `index-DrzwzzZ0.js`, read in Chrome on the public site at 0, 5, 10 and 15 s and after the run.

| where | read |
|---|---|
| the finance page, `MLP under MPC` on | class laundering on held-out row 1 of 16, the clear network's; 192 ReLUs, 16,879 AND gates, 42.66 MB of garbled-circuit traffic, 5.8 s in a Web Worker; statistical distance at most $`2^{-21.5}`$; logits $`-4.349, -1.256`$ recorded and $`-4.350, -1.256`$ reconstructed |
| `finance_aml.mlp_large` as the published model | ran on `onnx-wasm` at revision `3bddcf07034f`; the SHA-256 the run reports equals the entry in `models/hub_file_hashes.json` |
| the page | no console error |

### PR #28 on the deployed site, with every disclosure open (2026-11-10)

Deployed from merged `main`, bundle `index-DgDuTCPd.js`, read in Chrome at 1280 px after a run on each pipeline page, with every `<details>` opened and every SVG label's box compared with the others and with its container.

| where | read |
|---|---|
| finance, the transaction graphs | six laundering shapes, letters only; the edge numbers of the fan-out, fan-in and bipartite shapes overprinted where the edges meet (16 pairs) |
| medical | the occlusion map under the MPC panel after a run; the QCNN wire label "mean smoothn..." 1 px past the SVG's left edge |
| genomics, the locus plot | 183 points and five names; "rs11785598" over "rs10979327", and "rs6039942" past the right edge |
| telecom, the confusion matrix | 593 of 600; the diagonal counts in muted grey on full-strength blue, 1.08:1 against the 4.5:1 SC 1.4.3 asks |
| every pipeline page and the E2E page | the private-against-plaintext scatter printed its lower bound on both axes at the corner, the two labels meeting by up to $`6.2 \times 3.0`$ px; "CNOT ring" 1 px above its SVG |
| DeFi | the market and volatility charts' corner ticks met, by $`1.2 \times 3.6`$ and $`5.4 \times 4.1`$ px |
| Validations, Use Cases | clean |

The layout scan had never opened a disclosure, so none of the first four was measured; it opens them now, and the locus plot and the contrast failure are what it found on the next build, before the fixes.

### PR #29 on the deployed site (2026-11-10)

Deployed from merged `main`, bundle `index-CgSfj_T8.js`, read in Chrome with every `<details>` open, after a run on each pipeline page at 1280 px and in a 375 px same-origin frame.

| where | read |
|---|---|
| finance | the transaction graphs' edge numbers sit near the leaves and clear of each other; after a run, 16 cards and no label over another or past its box |
| genomics | the locus plot's 183 points and five names, none overlapping and none past the plot |
| telecom | the confusion matrix 593 of 600, its lowest cell contrast 6.3:1 |
| medical | the circuit card's wire labels and "CNOT ring" inside their SVGs |
| DeFi | the market and volatility charts' corner ticks clear of each other |
| 375 px | finance, genomics, telecom and medical with every disclosure open: no overlap, no clipped label, no sideways page scroll |

### PR #31 on the full local deployment (2026-11-10)

Not the public demo, which has no monitoring. The compose stack (`docker/docker-compose.yml`: API, web, Prometheus, Grafana,
node-exporter) started from this branch, then one finance run and one benchmark, read on Grafana and in Prometheus's API.

| where | read |
|---|---|
| Prometheus targets | `qsh-api` and `node` up |
| before any run | 14 stage series at count 0; `qsh_qber NaN`; the QBER panel empty rather than 0 |
| Stage latency p95 | the first run drawn, eight stages: PQC handshake 24 ms, QKD key agreement 975 ms, QSMPC 242 ms, each interpolated inside a bucket about 2.5x wide; the six stages that did not run hidden. Before the fix every stage read 10 |
| Benchmark latency | one line per backend, equal to the rows: TenSEAL 411.2 ms, OpenFHE 818.5 ms, QSMPC-FHE 3.7 ms. Before the fix MPC-FHE read the 1,000 ms bucket bound for two rows that were neither |
| QBER | 0.011 after the run; a gap, not a 0, between the API restart and the run |

A compose project keeps the bind mounts of the checkout it was first started from: rebuilding only the API from another
checkout left Grafana on the first checkout's dashboard. Start every service from one checkout before reading a panel.

### PR #32 on the deployed site (2026-11-10)

Deployed from merged `main`, bundle `index-BXeYuQrU.js`, read in Chrome.

| where | read |
|---|---|
| the circuit card's `Re-score on the held-out rows` | telecom macro F1 0.9444 over 480 rows in 147 ms, medical accuracy 0.9535 over 86 rows in 9 ms, finance minority-class F1 0.2286 over 600 rows in 7 ms; each `the value the weights file records` |
| telecom, 7 wires, `QCNN (trained, under CKKS)` | the log: the circuit's figure re-scored in this tab, 0.9679, the value the weights file records, and logistic regression's recorded at training time; `qcnn_rescored_rows` 480, `qcnn_rescored_headline` 0.967932, `qcnn_rescore_matches_record` 1; the CKKS panel's row the same |
| requests | `qcnn_heldout.json` and the weights from the site itself, by GET; no console error |

### The four finance models republished (2026-11-11)

The four AMLworld models went to the Hub on 2026-11-11 and were read back there; a second publish the same day carried
what the first had left stale, with the weights and graphs unchanged (each digest equal to the first's).

| model | revision | what changed | read back |
|---|---|---|---|
| `finance_aml.gbm` | `3393a270` | the card's calibration notes, until then Round 49's text about an earlier teacher, measured on the published graph | graph SHA-256 `d69b6532...` as before; the card holds the new notes and no Round 49 figure |
| `finance_aml.student` | `1bae2180` | the card: 115 parameters and 256 ciphertexts of 32 rows, not 33 and 64 | weights `dd58191c...` as before |
| `finance_aml.mlp_large` | `65283a88` | the card, and the sample's split, which still said "with the 79 per-transaction features added" | weights and graph as before; the sample states the registry's split |
| `finance_aml.gnn_compact` | `1cd8f653` | the card, which named `transformers` as its library, and a new `eval_sample.json`: the 16 rows its weights carry as fixtures | weights as before; the sample's logits reproduced from the weights in Python to 9.5e-7, and through the browser's MPC path to 4.5e-5 |

Three things the second pass found. The teacher's Platt map lowers the Brier score on the validation rows it was fitted
on and raises it on the test window, whose laundering rate is 0.152 percent against 0.113; the record now says so.
The published graph emits exactly 0.0 on 63,274 test rows, 2 of them laundering, so its raw log-loss is 0.00475425 where
the training run's booster scored 0.00473171; both are in the record. And the cards had counted the student's
parameters from a literal in the registry's notes, which had outlived the retrain on the 114 inputs.

A third, card-only publish on 2026-11-12, with the owner's approval, after the fact-check below: every card names GFP + XGBoost's
batch of 128 and the paper's $`64.77 \pm 0.47`$ at the batch of 2,048 these features use; the MLP's note says the teacher
reads the same 114 inputs, where it had said "on the 32 aggregates"; and the MLP's and GNN's calibration notes name their
model, re-measured with the published maps passed in, so `a` and `b` are unchanged. Every other file at the new revision
equals the previous pin's bytes, and each served card equals the local one.

| model | revision |
|---|---|
| `finance_aml.gbm` | `f3373960` |
| `finance_aml.student` | `67cdb41e` |
| `finance_aml.mlp_large` | `2733e357` |
| `finance_aml.gnn_compact` | `006a55c6` |

### The QCNN artifacts regenerate byte for byte (2026-11-11)

Run on the CUDA machine, each artifact's recorded `training.command` into a fresh directory, compared with `cmp`:

| artifact | before | after |
|---|---|---|
| `models/qcnn_finance/weights.json` | same values, `classical_surrogate` in another place | byte-identical |
| `models/qcnn_medical/weights.json`, `qcnn_telecom`, `qcnn_telecom_q7` | the same, and a `training.note` older than the trainer's | byte-identical |

`qcnn_classical_surrogate.py` had put `classical_surrogate` last before `note`, without retraining, while the trainer wrote
it after `classical_lr_accuracy`; the trainer now writes it where the files hold it. The three non-finance artifacts also
carried the note from before the trainer reworded it on 2026-11-01 for the finance retrain; that one string is updated in
them and their browser copies. No weight or metric changed. `tests/test_r86_qcnn_artifact_key_order.py` holds the order
without a training run, since CI does not run the slow regeneration test; it fails on all four against the old trainer.

### The finance figures the cards and docs quote, read at their sources (2026-11-11)

Read from each paper's own HTML or record, the tables by their captions; the dataset card and the TMLR record in Chrome.

| claim | source | read |
|---|---|---|
| GFP + XGBoost on HI-Small, $`63.23 \pm 0.17`$ minority-class F1 | Altman et al., arXiv:2306.16424, Table 2 | as quoted |
| fan-in and fan-out take XGBoost from 19.75 to 56.88, multi-hop patterns to 58.60, at batch size 128 | Blanusa et al., arXiv:2402.08593, Table 5 | as quoted; at batch 2,048 the column reads 19.75, 59.71, 61.01 |
| GFP + XGBoost at batch size 2,048 | the same paper, Table 4 | $`64.77 \pm 0.47`$, not quoted before: these features use batches of 2,048, so the docs now name it beside 0.6323 |
| Multi-GIN+EU $`64.79 \pm 1.22`$, Multi-PNA+EU $`68.16 \pm 2.65`$ on AML Small HI | Egressy et al., arXiv:2306.11586, AML results table | as quoted |
| Multi-FraudGT 75.81, MEGA-PNA 74.01, MEGA-GenAgg 74.48, Multi-PNA 67.35, TMLR 08/2026 | Bilgi and Atasu, arXiv:2412.00241v3, Table 16; OpenReview: accepted by TMLR, published 29 Aug 2026 | as quoted |
| HI-Small: September 1-10, 2022, 515K accounts, 5M transactions, 5.1K laundering, 1 in 981 | IBM's dataset card | as quoted; the raw file has 5,078,345 rows and 5,177 laundering |
| the rows after September 10 | the dataset card says they "are all laundering" | the raw file has 1,108 such rows and 655 laundering, as the cards say; the card's sentence overstates it |
| an edge-centric GNN under CKKS: F1 $`63.72 \pm 1.86`$, 5.31 s per query, 1.65 s amortised at batch 16, 99.25% agreement, HEonGPU on an RTX 5090, timings excluding the client's encryption, decryption, key generation and network | Bilgi et al., arXiv:2610.18054v2 | as quoted |
| Pfeiffer and Gail, Biometrical Journal 62(3):764-776; Van Calster et al., J Clin Epidemiol 74:167-176 (2016); Riley et al., Stat Med 40(19):4230-4251 (2021); Dimitriadis, Gneiting and Jordan, PNAS 118(8) e2016191118 | Crossref | as cited |

### PR #33 on the deployed site (2026-11-13)

Deployed from merged `main` (961a3b7) with `scripts/deploy_demo.sh --return-to-main`; the box, the public URL and the
browser all serve `index-DDyVXW9b.js`, which carries the four new finance pins and neither old one. Read in Chrome, two
full finance runs of all 12 stages, the second after a reload with the console recorded.

| where | read |
|---|---|
| requests | every Hub file at the new pins: gbm `f3373960`, student `67cdb41e`, MLP `2733e357`, GNN `006a55c6`; no console error or warning |
| verification | PASS for `finance_aml.gbm` and `finance_aml.student`, each re-scored on all 8,192 rows of its sample |
| the compact GNN under MPC | one held-out transaction, 16 of 16 slots filled; class laundering from the 2PC argmax, matching the clear network; 608 ReLUs in 13 batches, 52,655 AND gates; logits recorded -2.483, 0.722 and reconstructed the same, largest error 1.43e-4; 6.8 s in a Web Worker; the neighbourhood figure on its 720-wide canvas with no label past its edge |
| the MLP under MPC | the class from the 2PC argmax matches the clear network; the 114 features and the batch of 2,048 named |
| the CKKS student | 8,192 rows in 256 ciphertexts of up to 32 rows, 8,435 ms of encrypted work |
| decision curves | student 9.65 per 10,000 at $`t = 0.01`$ (166 at the sample's rate), operating point a calibrated risk of 0.079; teacher 10.91 (148) and 0.309; both above treat-none at every threshold drawn |
| Models panel | the compact GNN reads "Verified during the pipeline run: press Start with GNN under MPC ticked..." where it read "Cannot be run here" |

One defect found and left for the next PR: the `finance_aml.gnn` reference card says "Cannot be run here: never trained
in this build" under notes that now say IBM's Multi-PNA+EU was run here as a teacher.

### Three cards republished, card only (2026-11-13)

With the owner's approval, after the per-page audit of the published models. Each baseline line was read again at its
source before the publish.

| model | revision | what the card said | what it says |
|---|---|---|---|
| `finance_aml.gnn_compact` | `10104e89` | GFP + XGBoost's 0.6323 as the baseline to beat, a gradient booster's figure | Multi-PNA+EU, the architecture it is distilled from: $`68.16 \pm 2.65`$ on AML Small HI, averaged over five runs (Egressy et al., arXiv:2306.11586, Table 2), and 0.6616 and 0.6878 in this project's two runs of it; the capped 16-edge neighbourhood named as the difference |
| `medical_fl.resnet18` | `638b7a4a` | one run's 0.9147 against MedMNIST's 0.901 | the same, and that 0.901 is "the mean value of at least 3 trials" (Yang et al.), while this model's six seeds average $`0.8905 \pm 0.0198`$ |
| `telecom_ids.qml` | `ce545929` | the paper's hybrid QLSTM at 94.7% against its own random forest | the same, with that forest's 81.5 to 82.6% accuracy by tree count (Al-Kuwari et al., arXiv:2509.14282, Table VIII), and that its metrics and simulations differ from this project's |

Every other file at each new revision equals the previous pin's, compared by LFS digest or blob id; each served card equals
the local one; `models/hub_file_hashes.json` changed only in the three revisions.

### Every page's encrypted path, on the deployed site (2026-11-13)

The owner asked whether each page runs its models under encryption as well as in plaintext, as finance does. Read in
Chrome on the deployed `main` (961a3b7, `index-DDyVXW9b.js`), one default run per page; finance is the entry above. No
console error on any page.

| page | in plaintext, re-scored on the published sample | under encryption |
|---|---|---|
| medical | `resnet18`, `resnet50` and `student`: PASS on all 156 test images | the bundled WDBC regression on 171 held-out rows, 1 ciphertext, 0 top-1 mismatches; the published CNN student under CKKS, logits within 1.85e-9 of plaintext; with "CNN under MPC" ticked, as it is by default, held-out image 1 of 156 through CKKS and garbled circuits, 800 ReLUs, 69,167 AND gates, 174.38 MB, 16.6 s, class equal to the clear network's |
| genomics | no trained model: PGS000072's published weights | the 183-variant score in one ciphertext, equal to plaintext within 5.7e-10; the 30-person panel in 2 ciphertexts, 0 mismatches; the garbled-circuit argmax |
| telecom | `qml` PASS on 600 rows, `nidd` PASS on 16,379 | `telecom_ids.student`, 64 parameters, on 600 held-out rows in 2 ciphertexts, 0 mismatches, accuracy 0.988 against a majority baseline of 0.125; the argmax decrypted, as the page says |
| satellite | `telemanom` PASS on 128 windows, `student` on 8,640, `opssat` and `opssat_student` on 529 | the ESA student's forecast on all 8,640 rows in 135 ciphertexts, 0 of 8,640 flags changed; the OPS-SAT student on 529 held-out rows in 34 ciphertexts, 0 mismatches; each decision read after decryption |
| DeFi | no trained model | the margin under CKKS and its sign in a garbled circuit, 5 checks, each equal to the check in the clear, 1,835 AND gates |

One finding, fixed on the branch: in the default medical run the CNN student's encrypted input was the page's tabular row
tiled to fill the image, which the page said, so its class meant nothing. It now reads a held-out image of its published
evaluation sample, the one the CNN-under-MPC card takes for the same seed, and names its true class; without that sample
the student does not run, as without its weights. On a local client-only build of the branch: image 1 of 156, true class 1,
the CKKS student's class 1 and the MPC card's class 1, no console error. Telecom's student reads weights bundled with the page, not fetched from
a pin; that and the other gate gaps of the audit are the next PR's.

### The medical federated stage and BreastMNIST's duplicates, from the full deployment (2026-11-13)

Asked by the owner whether the medical page's federated learning matches the published cases (COVID-19 chest X-ray
CNNs across hospitals, FeTS, NVIDIA FLARE, federated fine-tuning of foundation models), the stage was run in Python and
its log read. The browser's federated stage is tabular (WDBC, four hospitals, six rounds, a logistic head on three
closed-form features). The full deployment's CNN path, `{"dataset": "medmnist", "model": "cnn"}`:

| run | before | after |
|---|---|---|
| seeds 0, 1 and 2 | 0.7282 every time, the 103 evaluation rows' majority class: one class for every image; logged as "global model accuracy = 0.728" | seed 0: 0.738 against a majority of 0.728, balanced accuracy 0.619, AUC 0.685; each hospital alone 0.500, 0.500, 0.620, 0.638 (AUC 0.641, 0.500, 0.695, 0.717). Seed 1: still one class, balanced accuracy 0.500, AUC 0.656, and the log says so |
| default tabular run | "global model accuracy = 0.956" | the same, against a majority of 0.772, balanced accuracy 0.944, AUC 0.979 |

Settings tried on seeds 0 to 2 before choosing: SGD at 0.05 and 0.01, Adam at 0.05, 0.01 and 0.003, with and without
each hospital's own class weights, one or three local epochs, 6 or 20 rounds, IID or label-skewed. SGD without
weights never left the majority; no setting passed an AUC of 0.72. The cases the owner named, read at their sources the
same day: Dayan et al. (Nature Medicine 2021, EXAM), 20 institutes and 16,148 cases, local-test AUC 0.795 to 0.920 and
generalisability 0.667 to 0.920 at 24 hours; Diniz et al. (npj Digital Medicine, 14 February 2026), 160 articles and
8,149 comparisons, decentralised learning above local in 82 percent of AUROC comparisons and centralised above
decentralised in 51 percent of AUROC and 78 percent of accuracy and Dice comparisons. On PneumoniaMNIST (Kermany et al.'s
chest X-rays, CC BY 4.0 at Mendeley), with the CKKS student's shape and the same aggregation, three seeds outside the
repository: AUC 0.911 for each hospital alone, 0.917 federated, 0.939 pooled.

BreastMNIST, measured on MedMNIST's splits: one test image byte-identical to a training image with the other label;
24 of 156 test images within 3/255 mean absolute difference per pixel of a training image, as within the training split
(91 of 546), consistent with the 155 duplicates Pawlowska et al. found in BUSI (Data in Brief 48:109247, 2023). The
served ResNet-18: AUC 0.9146 on all 156, 0.9281 without those 24, 0.9107 without the 43 within 5/255.

### PR #34 on the deployed site (2026-11-13)

Deployed from merged `main` (65faffd) with `scripts/deploy_demo.sh --return-to-main`; the box, the public URL and the
browser all serve `index-MC40jGBa.js`, which carries the three new pins and none of the old ones. Read in Chrome:

| where | read |
|---|---|
| medical, a default run | `fhe_student_input` "held-out image 1 of 156, the official BreastMNIST test split at 14x14", true class 1, the CKKS student's class 1, logits within 1.34e-9 of plaintext; the CNN under MPC on the same image, class 1, as the clear network; `medical_fl.resnet18` re-scored PASS, its graph fetched at `638b7a4a` |
| telecom | `telecom_ids.qml` fetched at `ce545929`, graph SHA-256 `505c91b8...` as recorded, re-scored PASS on 600 rows; the student's 600 held-out rows under CKKS, 0 mismatches |
| finance | the compact GNN's weights fetched at `10104e89`, GNN under MPC class 1 as the clear network, 608 ReLUs, largest logit error 1.6e-4; `finance_aml.gbm` and `finance_aml.student` re-scored PASS; the `finance_aml.gnn` entry reads "no trained artifact in this build" |

No console error on any of the three pages.

### Every published sample carries a per-row record, and the page compares every row (2026-11-13)

The owner chose exact per-row verification, with every sample republished, over the aggregate tolerance of 0.05 the
page had applied. The records were added to the PUBLISHED samples, not regenerated ones: each file, without its record,
was checked byte for byte against its pinned digest before anything was written, and the metric each record implies was
checked against the one the file already stated (accuracy and F1 to 1e-12, MAE to 1e-6) before it was written
(`scripts/add_per_row_records.py`). At each new revision only `eval_sample.json` changed, and each served file equals
the local one. The page's own `evaluatePublishedModel`, run in node with ONNX Runtime Web's WASM build on one thread
against the pinned bytes (`tests/test_r86_published_rescore.py`):

| model | revision | rows | decided as recorded | largest margin difference |
|---|---|---|---|---|
| `finance_aml.gbm` | `b74d16f3` | 8,192 | all | 4.8e-7 |
| `finance_aml.student` | `c305a104` | 8,192 | all | 3.1e-6 |
| `finance_aml.mlp_large` | `65926d2e` | 8,192 | all | 5.3e-6 |
| `medical_fl.resnet18` | `b24642c6` | 156 | all | 1.1e-6 |
| `medical_fl.resnet50` | `d11ec391` | 156 | all | 5.7e-5 |
| `medical_fl.student` | `1de47e04` | 156 | all | 1.1e-6 |
| `satellite_comms.telemanom` | `6fbeb225` | 128 windows | (forecast values) | 6.9e-7 |
| `satellite_comms.opssat` | `95955c8d` | 529 | all | 4.7e-7 |
| `satellite_comms.opssat_student` | `9175fa16` | 529 | all | 1.1e-6 |
| `telecom_ids.qml` | `cd149476` | 600 | all | 4.6e-7 |
| `telecom_ids.nidd` | `9fb901e3` | 16,379 | all (one row within 1e-3 of the boundary) | 1.6e-6 |
| `telecom_ids.student` | `c351a335` (new) | 600 | all | 2.5e-6 |

The tolerance is 1e-3: seventeen times the largest difference measured, for the page's threaded and WebGPU runs, and far
below what a different graph or preprocessing moves. Seeded on the telecom QML sample as served: one decision flipped
fails the verification, and one margin moved by 0.01 fails it. The compact GNN and the forecaster student already carried
row-level records (logits, and a digest of the flags), so every published model is now verified row by row.

Also in the republish:

- **`telecom_ids.student` published** as `Daemons-Q/qsh-telecom-ids-ckks-student` (Apache-2.0): its graph is built from its
  weights file (`qsh.models.linear_onnx`, 566 bytes, the same bytes on every build), its sample is the QML forest's 600
  held-out rows read from that sample, and the card says the argmax is taken in the clear after decryption. Its four
  files on the Hub equal the local ones; macro F1 0.988329 on the 600 rows.
- **Four change notices corrected.** `finance_aml.gbm`, `satellite_comms.opssat`, `satellite_comms.opssat_student` and
  `telecom_ids.qml` said their values were "rounded to 6 decimals" while their builders have written each value as the
  shortest decimal that reads back as the same float32 since Round 86 (checked on the files: no sample of the four holds
  6-decimal values). Only the `modifications` field changed in each.
- **The CKKS tier's bundled student copies were checked against nothing.** Finance's and telecom's are byte-identical to
  their pinned `weights.json`; satellite's is a derived form with the scaler its graph applies moved into `input`.
  `tests/test_r86_bundled_students_are_the_pinned_ones.py` now holds each to its pinned file.
- **The finance decision curves re-measured at `b74d16f3` and `c305a104`:** the same figures; the records' revisions and
  sample blobs moved with the pins.

### The medical page federates four real hospitals (2026-11-13)

The owner chose, among the published medical cases, to federate real hospitals on the tabular side first. FLamby's
Fed-Heart-Disease was read at its source (`dataset.py`, `common.py`, `metric.py`, the benchmark configuration and its
five result files) and reproduced:

| step | read | here |
|---|---|---|
| data | four UCI files, the MD5s FLamby's download script checks | the same MD5s; 303, 294, 123 and 200 raw rows |
| preparation | slope, ca and thal dropped, missing rows removed, columns 2 and 6 dummy-coded with the first level dropped, the label binarised, a 66/34 split per centre at random_state 43 (stratified where each class has more than two rows), per-centre standardisation on training rows with the sample standard deviation | the same; 199/104, 172/89, 30/16 and 85/45 rows, FLamby's counts; 13 features |
| training | `nn.Linear(13, 1)`; Adam at 0.001, batches of 4, 50 epochs for pooled; FedAvg with SGD at 0.001, 15 rounds of 100 local steps | the same; each hospital alone with the pooled settings |
| metric | accuracy at 0.5, averaged over the four test sets; AUC is commented out, FLamby's comment says because centre 2 (Switzerland) has only y = 1 | the same; of the 46 complete Swiss rows one has no disease (raw file: 8 of 123), and it falls in the test split, so the 30 Swiss training rows hold one class |
| results, five seeds (sample standard deviation) | FLamby, seeds 42 to 46, the four client test sets (each file also scores a pooled test set, left out): alone 0.722, 0.629, 0.567, 0.637; FedAvg $`0.685 \pm 0.028`$; FedYogi $`0.731 \pm 0.013`$; pooled 0.735 | seeds 0 to 4: alone 0.735, 0.636, 0.601, 0.644; FedAvg $`0.712 \pm 0.017`$; pooled 0.715 |

The browser module and its Python twin agree on every figure of every arm for seeds 0 to 2, and on mulberry32's
stream, value for value (`tests/test_r86_fed_heart_twin.py`); the page's federated stage, run in node through
`runClientPipeline`, reports the twin's figures for its seed (`tests/test_r84_pairwise_masks.py`). The UCI files are
CC BY 4.0; the payload carries the credit, the DOI and the list of changes, and `DATA_LICENSES.md` has its row.

FLamby's metric scores one model over the four hospitals' test rows. On its own test rows the comparison runs the other
way: in each of the five seeds a hospital's own model scores above the federated one at three or four of the four
hospitals, with no ties (means over the seeds, own model against federated: 0.750 against 0.717 at Cleveland, 0.751
against 0.735 at Budapest, 0.863 against 0.738 at Zurich and Basel, 0.711 against 0.658 at Long Beach). The Swiss model
trains on 30 rows that all have disease and scores 15 of 16 by calling every patient diseased; the federated model's
gain is on the other hospitals' patients, where a model trained at one hospital falls to 0.337 in seed 0. The panel's
first draft, never committed, put each hospital's own-test figure in an "Alone" column beside the federated model and
averaged that column under FLamby's name, so the table showed the federation losing in every row while its caption
said it won. The committed panel is the whole matrix, models by test sets, the own-test cells underlined, and its
caption and the run's log count both comparisons from the run instead of asserting either. Diniz et al. was read
again for the figures the page quotes: decentralised learning (federated in 557 of its 710 models) above local in 83
percent of accuracy comparisons and 82 percent of AUROC ones, centralised above decentralised in 78 percent of
accuracy comparisons and 51 percent of AUROC ones.

### PR #35 on the deployed site (2026-11-13)

Merged as `eb53b00` and deployed from `main`; the bundle `index-C8kQEvZF.js` is the same name and the same SHA-256 in the
deploy log, on the public page and in a local build of `eb53b00` with `VITE_CLIENT_ONLY=1`. The served bundle carries
all fourteen pins of `config.defaults.json` and none of the three it replaced. Walked in Chrome, one run per page:

| page | models re-scored | rows, as recorded | largest difference |
|---|---|---|---|
| telecom | the QML forest, 5G-NIDD, the CKKS student (now in the model selector) | 600, 16,379, 600 | 3.25e-7, 7.16e-7, 2.50e-6 |
| medical, WASM | ResNet-18, ResNet-50, the CKKS student | 156 each | 1.14e-6, 5.68e-5, 1.06e-6 |
| medical, WebGPU (`onnx-webgpu`, an NVIDIA Blackwell adapter) | the same | 156 each | 1.14e-6, 6.51e-5, 2.37e-6 |
| finance | the GBM, the student, the large MLP | 8,192 each | 3.58e-7, 3.13e-6, 5.26e-6 |
| satellite | Telemanom (each of 128 forecasts), OPS-SAT, its student; the forecaster student by its flag digest | 128, 529, 529; 8,640 flags | 6.89e-7, 4.65e-7, 1.08e-6 |

Every verification PASS, no console error on any page. The telecom page's encrypted tier now names its model,
`telecom_ids.student`, with 0 of 600 top-1 mismatches against plaintext; the medical CKKS student read held-out image 1
of 156 and matched its label. One display defect found: the model-backend control reads "WebGPU (no usable adapter)"
from the first render until the adapter probe resolves, so a visitor who looks first is told something not yet known; on
this machine the probe then found the Blackwell adapter and enabled the option. Fixed in #36: the option reads "checking"
until the probe answers.

### PR #36 on the deployed site (2026-11-13)

Merged as `a853fff` and deployed from `main`; the bundle `index-B5Vs94nf.js` is the same name and the same SHA-256 in the
deploy log, on the public page and in a local `VITE_CLIENT_ONLY=1` build of `a853fff`. One medical run in Chrome, at
the page's default seed 1 (the seed field starts at 1 because quantumSim's generator maps 0 and 1 to one stream): the
card is the six-by-five matrix, the four own-test cells underlined; federated 0.7005, pooled 0.7132, the hospitals
alone 0.6534 on average, the twin's figures for seed 1 to the last digit; the counts read above 3 of the 4 alone, below
pooled, the own model above the federated one at 4 of the 4 and a floor of 0.348; the canvas label and the log line as
written; no console error. The model-backend option read `WebGPU (checking for an adapter)` at load and `WebGPU` two
seconds later. The checklist's seed-0 expectation was rewritten to the page's seed.

### The medical page aggregates its federation five ways (2026-11-13)

The owner chose to put every design on the page and to run them on a button in a Web Worker. `clientFedAgg.ts` reruns
the four-hospital federation from the run's seed and swaps only the aggregation; the threshold designs run on Lattigo
v6.2.0 compiled to WebAssembly from `wasm/qsh-fedthresh` (Go 1.27.4, the image's new Go stage, `go test` first). Seed
1, 15 rounds, in Node (`npm run check:fed-agg`, which asserts the rows):

| design | accuracy | distance from exact FedAvg | sent per hospital per round | time per round |
|---|---|---|---|---|
| pairwise masks | 0.7005 | $`1.4 \times 10^{-6}`$ | 56 B | 1.3 ms |
| CKKS, one shared key (node-seal, FLARE's settings) | 0.7005 | $`1.5 \times 10^{-8}`$ | 262 kB | 14 ms |
| threshold CKKS, 16 bits ($`\sigma = 2^{30}`$, scale $`2^{60}`$) | 0.7005 | $`3.4 \times 10^{-7}`$ | 394 kB | 219 ms |
| threshold CKKS, 40 bits ($`\sigma = 2^{54}`$, scale $`2^{80}`$) | 0.7005 | $`4.7 \times 10^{-6}`$ | 394 kB | 280 ms |
| threshold BGV, 128 bits ($`\sigma = 2^{142}`$, exact) | 0.7005 | $`1.4 \times 10^{-6}`$, the masks' own | 590 kB | 134 ms |

Threshold BGV encrypts the masks' fixed-point integers (`qflQuantise`, split out of `maskedShare` for it), so every
round's decrypted sum equals the masks' to the bit and so does the model. The statistical security is
$`\log_2(\sigma / 2^{14})`$, against a noise bound of $`2^{14}`$ for the sum of four fresh encryptions, measured at
$`2^{12.3}`$ to $`2^{12.8}`$.

What was measured before choosing the settings, natively, four parties, 14 values: at the project's scale $`2^{40}`$
the $`\sigma = 2^{30}`$ that Lattigo's multiparty examples use gives an error of 0.28, enough to break FedAvg, and
$`2^{60}`$ brings it to $`1.7 \times 10^{-7}`$; for 128 bits CKKS would need a scale near $`2^{168}`$, above
Lattigo's cap of $`2^{128}`$, where the error was $`2.8 \times 10^{6}`$; BGV at 128 bits decrypted all values exactly,
with $`Q/(2t) = 2^{146}`$ eight standard deviations above the summed flooding noise.

**A defect in Lattigo's Gaussian sampler, found while choosing the 40-bit setting.** With $`\sigma = 2^{53}`$ the
decryption was wrong by $`3.4 \times 10^{16}`$ over primes of 60, 60 and 30 bits at scale $`2^{100}`$, and by
$`5.7 \times 10^{6}`$ over 60 and 40 bits at $`2^{80}`$, while scale $`2^{80}`$ over two 60-bit primes decrypts to
within $`10^{-15}`$ without the flooding. The integer branch of
`GaussianSampler.read` (taken unless $`\sigma > 2^{53}`$ and the bound exceeds $`2^{64} - 1`$) writes each sample into
every RNS limb without reducing it modulo the limb's prime: a positive sample above the prime stays unreduced, a
negative one wraps modulo $`2^{64}`$. A self-contained reproducer: at $`\sigma = 2^{41}`$ over 60- and 40-bit primes,
5069 residues at or above their prime and 2517 of 8192 coefficients outside the bound, and a two-party key switch off by
$`1.35 \times 10^{8}`$, against $`3.3 \times 10^{-12}`$ at $`\sigma = 2^{35}`$. `ring/sampler_gaussian.go` is the same
on `main`; no issue described it. Reported with the owner's approval as
[tuneinsight/lattigo#596](https://github.com/tuneinsight/lattigo/issues/596) (2026-11-13). Every setting here keeps the
primes above six times the deviation, `Open` refuses one that does not, and `agg_test.go` holds the settings to it.

The module is 12.6 MB uncompressed. Its notices list Go and ten modules: Lattigo's non-test packages import testify
and go-cmp, so go-spew (ISC), go-difflib and yaml.v3 are compiled in too. The SBOM job now writes
`fedthresh.cdx.json` from the same `go list` reading, and the CBOM gains threshold CKKS and threshold BGV, 23 entries.

### Differential privacy on the medical and genomics pages (2026-11-13)

The owner asked whether differential privacy is recommended for each use case and, where it is, for an implementation
on the public pages. Read at the sources (`references/differential-privacy.md`): NIST SP 800-226 (final, 6 March
2025) for the unit, $`\epsilon`$ and $`\delta`$; Kairouz, Liu and Steinke's distributed discrete Gaussian (ICML 2021,
Theorem 1 read in the PDF) for noise added in shares inside a secure sum; Canonne, Kamath and Steinke's exact sampler
(NeurIPS 2020). It fits the medical federation, per patient, and the genomics page's pooled count, per person; the other
four pages state on the page why it does not fit.

| page | design | $`\delta`$ | per-party variance | measured |
|---|---|---|---|---|
| medical | 50 full-batch steps, each patient's gradient clipped to norm 1, sums at $`2^{-16}`$, noise in shares inside fresh masks | $`10^{-6}`$ | $`\epsilon = 1`$: 1,536,822,783,572 (sd 18.9 a hospital, 37.8 in all, in gradient units) | seed 1, 20 draws: 0.721 no noise, $`0.724 \pm 0.004`$ (8), $`0.727 \pm 0.007`$ (4), $`0.726 \pm 0.015`$ (2), $`0.706 \pm 0.032`$ (1) |
| genomics | each cohort's count at $`1/16`$ plus its share, inside the secure sum over $`2^{31} - 1`$ | $`10^{-5}`$ | $`\epsilon = 1`$: 6,148 (in sixteenths) | noise sd 19.4, 9.8, 5.0, 2.6, 1.4 alleles at 0.5, 1, 2, 4, 8 |

The fixed point on the genomics count exists because in whole alleles the least variance the sampler can take, 1 per
cohort, is already $`\epsilon = 5.3`$, so the selector's 8 was unreachable. The private arm takes one full-batch step
per secure aggregation because FedAvg's local steps have no ready per-patient bound; its no-noise row (0.721, against
FedAvg's 0.7005) isolates the noise's cost. Each setting agrees fresh pair seeds, since one seed masking two messages
in a round would let the aggregator subtract them. `qflQuantise` and the new `maskIntegers` and `sumMaskedIntegers`
split the masks' arithmetic so the DP arm masks integers that already carry noise.

`tests/test_r86_dp_twin.py` holds the browser modules, bundled and run in node, to `qsh.ai.dp` and
`qsh.ai.fed_heart.run_dp` on a seeded source that is not private and that no page calls: 800 draws equal, the variances
and epsilons equal, every row of the medical run equal for seed 1, the genomics contributions equal for every epsilon
on the selector, and a chi-square of 60,000 draws at variance 9 against the discrete Gaussian's pmf.

Writing the medical page's key points against the change found two sentences PR #36 left behind: the federation still
described as FedAvg "over the page's own bundled Wisconsin sample", and "no differential-privacy budget is applied
here". Both now say what runs.

### PR #24, a pointer (merged 2026-11-09)

PR #24 merged the audit's remaining items as `38b9e76` on 2026-11-09: labelled rows on the model cards, the QCNN's tree
property as a test, a watch on Polytune upstream (`.github/workflows/audit.yml`), the GPU lock, a matched local liboqs
and the last plain-text formulas. Its record is the register's row #24 and the tests it added; this log had no entry
for it, which this pointer closes.

### PR #37 on the deployed site (2026-11-13)

Merged as `52baf07` and deployed; the bundle `index-fDVB_zFP.js` has the same name and SHA-256 in the deploy log, on the
public page and in a local `VITE_CLIENT_ONLY=1` build. The image's new Go stage ran on the demo host. In Chrome, after
a medical run (seed 1), `Compare encrypted aggregation` showed its progress and then all five rows: the masks 1.4e-6 from
the exact FedAvg, 56 bytes and 0.7 ms a round; the shared-key CKKS 1.1e-8, 262 kB, 14 ms; threshold CKKS 3.9e-7 and
6.9e-6, 394 kB, 211 and 263 ms; threshold BGV 1.4e-6, "yes, to the bit", 590 kB, 154 ms; every design at 0.7005. No
console error. `qsh_fedthresh.wasm` is served 200 as `application/wasm`, gzip-encoded, with the cross-origin
isolation headers; `/licenses/qsh-fedthresh.txt` is served with its ten sections. The extension's network log does not
record a worker's fetches, so that an ordinary run never loads the module rests on the code path: only the button
creates the worker, and only the worker fetches the module.

### PR #39: the DP noise inside threshold BGV, a published count on finance (2026-11-13)

The owner chose three additions after PR #38: the genomics sum under threshold BGV with the DP noise inside, a DP row in
the medical page's aggregation table, and a DP-noised count on finance. `clientFedHeart.trainFedHeartDp` now takes the
aggregation as a function, so the record-level arm's 50 steps can be summed by more than one design without a second
copy of the loop; `fedHeartDp` passes the masks and its rows are unchanged, which `tests/test_r86_dp_twin.py` holds
against `qsh.ai.fed_heart.run_dp` row for row. The comparison's sixth row sums each step's four noisy vectors under
threshold BGV, trains on what the shares decrypt, and compares the masks' sum of the same vectors. `npm run
check:fed-agg`, seed 1: all 50 steps' sums equal, epsilon 1.0000 at delta 1e-6, accuracy 0.7216, 590 kB a hospital a
step, 135 ms a step; the five FedAvg rows as before (0.7005). The genomics card's button calls
`clientFedAgg.sumUnderThresholdBgv` in the worker with a fresh key per press; the check sums a seeded draw at
epsilon 0.5 with a negative contribution added and gets the plain sum, 717, and refuses a value that is not a whole
number.

Finance: `clientDp.dpCount` adds one discrete Gaussian to the two banks' count; at epsilon 1 and delta 1e-8 the
least variance is 38 (standard deviation 6.2 accounts; the accountant gives 0.998), Python's `dp_count` draws the same
values on the seeded source, and a negative draw is published as zero. The banks' own answer stays exact.

Writing the medical table's note found that its four noised rows are separate releases from the same 486 patients:
published together they are epsilon 9.56, from rho 0.910, 0.254, 0.068 and 0.017 summed, where the table had let the
largest row's 8 stand for the page. `fedHeartDp` returns the composed figure and the table prints it; the twin
test holds it to Python's and to the closed form. The research behind the other pages' DP lines is in
`references/differential-privacy.md`, "The other pages".

Two math spans on main had a TAB where a backslash command began: `\times` in the threat model's aggregation section
and `\tau_0` in `docs/models-and-benchmarks.md`, each a string escape in the edit that wrote it. Both are repaired, and
`tests/test_markdown_math_uses_the_house_style.py` now fails on any control character inside a math span; run on
main's threat model, the new test fires.

### PRs #38 and #39 on the deployed site (2026-11-13)

PR #38 merged as `cb4d2fa` and deployed: `index-b1ejB7U1.js`, with the same name and SHA-256 in the deploy log, on the
public page and in a local `VITE_CLIENT_ONLY=1` build. In Chrome, a medical run (seed 1) showed the DP table under
the matrix: no noise 0.721; epsilon 8, 4, 2 and 1 with the one-hospital-short epsilons 9.40, 4.66, 2.32 and 1.16, total
noise 5.2, 9.9, 19.2 and 37.8 in gradient units, and accuracies 0.727, 0.710, 0.726 and 0.731 from that run's draws.
The genomics selector offered 0.5 to 8; at 0.5 the card released 41.81 against an exact 42 of at most 60, with noise of
standard deviation 19.40. All six pages showed their `Differential privacy:` line. No console error. The walk found
delta printed as `0.000001`, JavaScript's form, which PR #40 writes as `1e-6`.

PR #39 merged as `7e780c0` and deployed: `index-Bo5Ln4ci.js`, the same three ways. In Chrome: the finance card
published 55 against an exact 60 from lists of 260 (epsilon 1.00, delta 1e-8, standard deviation 6.16 accounts); the
genomics button decrypted 968 from the BGV sum, the masked sum's integer, 60.50 released, with 394 kB of ciphertext
and a 197 kB decryption share per cohort, and a second press gave the same; the medical DP table's note gave epsilon
9.56 for its four rows together; and `Compare encrypted aggregation` gave six rows in 20 s, the five FedAvg designs at
0.7005 and the DP row at 0.7164 with 50 of 50 steps' BGV sums equal to the masks', its note giving epsilon 9.63 with
the table's rows. No console error.

### PR #40 on the deployed site (2026-11-13)

Merged as `b7430a6` and deployed: `index-CGv08KFf.js`, the same name in the deploy log and on the public page, and the
same SHA-256 there as from a local `VITE_CLIENT_ONLY=1` build. In Chrome, on every use-case page, the Models section is
a tablist (finance: Overview and five models; medical three; genomics one; telecom four; satellite four, beside its
own Downlink and Network tabs), the arrow keys move between tabs, and the hidden panels keep their model blocks. The
comparison table is on all six pages; before a run its "this run" column says so. After a medical run (seed 1):
ResNet-18 reproduced, ResNet-50 a reference, the CKKS student not comparable, FLamby's four hospitals and pooled model
"no spread", FedAvg 0.7005 reproduced against $`0.685 \pm 0.028`$, FedYogi not run; finance's five models one not
comparable and four references; telecom's four not comparable under its note; satellite's Telemanom "no spread" and
three references; DeFi its note and no table. The verification table listed all three medical models, each PASS on
156 of 156 rows. Each download button carries its own name ("Download the four-hospital matrix as CSV"). The results
file, read in the tab without saving it: `medical_fl-results.json`, 86 kB, `build` equal to the served bundle, 159
metrics, 205 log events, the four-hospital, DP and verification cards, a 3,309-byte ML-DSA-65 signature, and no key
named like a secret. No console error. The walk found the comparison table's "this run" cell giving the sample's
accuracy, unnamed, beside an AUC row; it now names the metric and says when it is not the row's.

### PR #42: the threshold flooding drawn by the module, from a worst-case bound (2026-11-13)

`wasm/qsh-fedthresh/agg/agg.go` now leaves Lattigo's key-switching noise at its fresh level and adds, to every
coefficient of every share, an integer uniform on $`[-F, F]`$ with $`F = 2^f - 1`$, from `crypto/rand`, reduced into
every limb. $`f`$ is the least with $`NB/(2F+1) \le 2^{-\lambda}`$ over a whole share, for
$`B = k(2Nkb + b)`$, the worst case of $`k`$ fresh encryptions summed under the collective key with error bound
$`b = 19.2`$. The Go tests hold the bound at $`4(2 \cdot 8192 \cdot 4 \cdot 19.2 + 19.2)`$, the least $`f`$ for two,
four and five parties in every setting, and the draws' range, parity and size. `npm run check:fed-agg` on the rebuilt
module, seed 1: $`f`$ = 51, 75 and 163; errors $`3.6 \times 10^{-7}`$ and $`5.8 \times 10^{-6}`$ for the CKKS rows,
BGV equal to the masks to the bit; 12.09, 36.09 and 124.09 bits over the run's 15 decryptions under one key, 122.36 for
the DP row's 50; the genomics sum for five parties at $`f = 163`$ against $`B = 2^{22.91}`$. BGV moved to
$`Q = 60+60+60+20`$ bits, inside the HE standard's post-quantum 202 at $`N = 8192`$, where the four bounded floods
stay below $`Q/(2t)`$ and the decryption cannot fail; a BGV share is now 393 kB, the ciphertext 393 kB.

### PR #42 on the deployed site (2026-11-14)

Deployed from `main` at 4c8c6d0 by `scripts/deploy_demo.sh --return-to-main`: the box is on `main` and serves
`index-veWbIFx8.js`, whose SHA-256 (`4c59d231...253f`) is the same in the deploy log's bundle name, on the public page and
in a local `VITE_CLIENT_ONLY=1` build of `main`. Walked in Chrome: the medical page's encrypted-aggregation comparison
prints each threshold row's flooding, uniform on $`\pm(2^{51} - 1)`$, $`\pm(2^{75} - 1)`$ and $`\pm(2^{163} - 1)`$ for
a noise bound of $`2^{22.26}`$, with its bits per decryption and over the run's decryptions under one key: 16 and 12.1,
40 and 36.1, 128 and 124.1, and 128 and 122.4 for the DP row's 50; BGV equal to the masks to the bit. The CKKS errors on
this run were $`3.8 \times 10^{-7}`$ and $`8.7 \times 10^{-6}`$; they move with each run's flooding draw. The genomics
card's threshold sum for four cohorts: 825 against the masks' 825, with $`f = 163`$ for the bound $`2^{22.26}`$. No
console errors.

### PR #43: the DeFi committee on threshold BGV (2026-11-13)

`wasm/qsh-fedthresh/agg/committee.go` opens a committee of $`n`$ members with threshold $`t`$ on Lattigo v6.2.0 at
$`N = 16384`$, a 260-bit $`Q`$ and a 50-bit prime plaintext modulus; the borrower encrypts each balance in slot 0 of
its own ciphertext; the pool multiplies by integer weights, subtracts the settled amounts and adds a mask uniform
modulo $`t`$ and a fresh encryption of zero in slot 0; $`t`$ members re-encrypt to Carol's key by the public-key
switch, each share flooded uniformly with $`f`$ from `leastFloodBits`, and refuse a second component they have shared
for. The Go tests, on this machine: the two shares give the margin to the bit for positive, negative and zero margins
under four active sets; a second share for the same ciphertext is refused under another active set, and the next
evaluation at the same weights has a different second component and is accepted; the evaluated noise is
$`2^{35.25}`$ against the bound $`2^{44.56}`$ for the weights used, and after the switch the noise under Carol's key
is $`2^{190.00}`$, the two floods' $`f + 1`$, below the decryption limit $`2^{209.13}`$; $`f`$ = 189, 190 and 191 for
2 of 3, 3 of 5 and 4 of 7, each the least that meets 128 bits over a share; the ciphertexts survive the wire, and a cut,
an extra byte, the wrong count or a lower level is refused. The crate's tests hold the sign circuit modulo $`t`$ to the
plaintext sign at the committee's $`t`$ and at 97, with masks across $`[0, t)`$, and pin it at 648 AND gates.
`npm run check:lending-private`, seed 3: 9 checks, every bit the clear answer, the worst margin error 0.079 stable
units, every pair of members taking a turn, every other slot zero, every bit's signature verifying, and the settled
checks still exact; margins of plus and minus one stable unit get their bits. In the browser (a local preview of this
branch, the default run of five checks): all five match, the worst difference 0.045 stable units, 5.24 MB of
threshold-BGV traffic and 648 AND gates per check, 2,007 ms on one worker, of which the members' flooding was 505 ms of
the 741 ms of BGV; Carol's custody and the bilateral mode unchanged (367 AND gates, the same messages).

### PR #44: the chest X-ray federation on NIH ChestX-ray14 (2026-11-14)

`scripts/data/prepare_nih_cxr.py` read the twelve NIH archives in memory into a 224 px grey memmap (112,120 images), an
index and 25 groups of pixel-identical images, three of them across the official split; all ten pairs NIH's change log
lists are among them. `scripts/train/medical_cxr_fl.py` drops 26 duplicates (a train_val copy where a group crosses the
split), places the train_val patients in four hospitals by how well each patient's AP share matches a hospital's target,
sharpened until the realised shares came closest (k = 4.0: 11.9%, 21.1%, 51.6%, 77.4%), and trains on the GPU with the
whole set resident: 12 epochs or FedAvg rounds of one local epoch, AdamW at $`10^{-3}`$, batch 128, seeds 1 to 3, the
split fixed. On NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 620.70.06, PyTorch 2.13.0+cu130 (CUDA 13.0,
cuBLAS 12.9.3 as loaded), kernel 5.14.0-687.66.1.el9_8.x86_64: pooled 0.812 (PA 0.849, AP 0.781),
FedAvg 0.805, the hospitals alone 0.765 to 0.795, PA only 0.787 (PA 0.849, AP 0.732), ConvNeXt-atto
under FedAvg 0.732 (3,373,561 parameters, one seed); the seeds' standard deviations at most 0.006. A rescoring of
the saved models' maps by the final scoring code gave the recorded hit rates again; the untrained baselines and the centre
baseline were added the same way. `scripts/data/build_cxr_fl_summary.py` wrote `docs/measurements/cxr-fl-2026-11-14.json`
and the card's `web/public/data/medical/cxr_fl.json`, aggregate figures only, with no local path. Tests: the preparation on
a synthetic NIH folder (`tests/test_r86_cxr_prepare.py`) and the analysis's arithmetic (`tests/test_r86_cxr_fl.py`). A
local preview of the branch drew the card from the file with no console error and no table wider than its box.
### PR #43 on the deployed site (2026-11-14)

Merged as `f7e69fc` and deployed by `scripts/deploy_demo.sh --return-to-main`: the box is on `main` and serves
`index-BmeXKk2o.js`, whose SHA-256 (`29fd89ed...cc96`) is the same in the deploy log, on the public page and in a local
`VITE_CLIENT_ONLY=1` build of `main` with the wasm rebuilt by `scripts/build_wasm.sh`. The image's Go and Rust stages ran
their tests on the demo host. Walked in Chrome on the DeFi page with "A 2-of-3 committee, re-encrypting to Carol
(threshold BGV)" chosen: five checks, one liquidatable, all five matching the clear check and every bit's signature
verifying; 5.24 MB of threshold-BGV traffic and 648 AND gates per check; the pairs 1 and 2, 2 and 3, 3 and 1; 189-bit
uniform flooding against a worst case of $`2^{47.8}`$; every other slot zero at five of five checks; the committee's limits
and the canvas's "Carol, with a 2-of-3 committee". Carol's custody on the same page: five of five, 367 AND gates, 0.39 MB of
sealed CKKS traffic. No console error.

### PR #44 on the deployed site (2026-11-14)

Merged as `2799d79` and deployed: the box is on `main` and serves `index-CfC7X8aP.js`, whose SHA-256 (`d89f1eef...a51d`) is
the same in the deploy log, on the public page and in a local `VITE_CLIENT_ONLY=1` build of `main`; the served
`data/medical/cxr_fl.json` is byte for byte the file on `main`. In Chrome the medical page draws "Chest X-ray federation,
four hospitals by view" with the badge "measured offline": the four hospitals (AP 11.9% to 77.4%), the eight arms (pooled
$`0.812 \pm 0.001`$, FedAvg $`0.805 \pm 0.002`$), the view in the scores, the probe against an untrained network, the maps
against the boxes (13.9% pooled and 20.5% federated against 7.1% for a uniform map and 2.6% for the centre), NIH's credit
line and the card's CSV and JSON buttons, each with its own accessible name; no table wider than its box. No console
error.

### PR #46: the heart-sound federation on the PhysioNet/CinC Challenge 2016 data (2026-11-14)

`scripts/data/prepare_physionet2016.py` read the released zip in memory, only the members each source's
`REFERENCE-SQI.csv` names: 3,240 recordings of training-a to training-f, 20.22 hours at 2 kHz, 665 abnormal, band-passed
25 to 400 Hz and standardised. `scripts/train/medical_pcg_fl.py` holds each source out in turn and trains a
97,521-parameter CNN on 4-second log-mel windows (40 bands, 25 to 500 Hz) on the other five, pooled and under FedAvg,
12 epochs or rounds, seeds 1 to 3, on NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 620.70.06, PyTorch 2.13.0+cu130 (CUDA 13.0, cuBLAS
12.9.3 as loaded), kernel 5.14.0-687.66.1.el9_8.x86_64. A recording's score is the mean of its windows' logits. AUROC, pooled
and FedAvg: a 0.728 and 0.678; b 0.625 and 0.462; c 0.623 and 0.387; d 0.443 and 0.242; e 0.321 and 0.353; f 0.582 and 0.740; with the recordings split at random,
0.961. The seeds' standard deviations are at most 0.049. `scripts/data/build_pcg_fl_summary.py` wrote
`docs/measurements/pcg-fl-2026-11-14.json` and the card's `web/public/data/medical/pcg_fl.json`, aggregate figures only.
Tests: `tests/test_r86_pcg_fl.py` (the preparation on a synthetic zip, the filterbank, the windows, the per-recording
score, the metrics and the card's data).

Held for generalisation (the owner's choice), then the same study with `--band-mean`, which removes each mel band's mean
over the window before the window is standardised: seeds 1 to 3, same GPU, driver and PyTorch. Mean held-out AUROC 0.554
to 0.599 pooled and 0.477 to 0.530 under FedAvg; pooled, c 0.623 to 0.802 and d 0.443 to 0.659, b 0.625 to 0.602, e
0.321 to 0.252, f 0.582 to 0.551, a within 0.01; the random split 0.961 to 0.958. The builder now takes the run with
`--band-mean <dir>`: the record and the card keep it under `band_mean`, the main run's figures unchanged (checked
equal). The card names it as one remedy that helped on average, not everywhere, and its explanation of the inverted e
now follows Liu et al. 2016, section 2.6 (read here in the author version): healthy volunteers in a laboratory with a
microphone or piezoelectric sensor, patients in hospital with a Littmann. Balancing each source and label and a random
spectral tilt ran in a scratch copy only (0.541 and 0.606 pooled) and are not in the record. A literature check
(`references/heart-sound-and-ecg-prior-work.md`) found the same pattern elsewhere: Koike et al. 2021 raised every
held-out database with spectrogram standardisation, and Humayun et al. 2020 found domain-balanced batches did not help
on an unseen domain.

