# Verification results log

What each round actually measured against the deployed demo. Split out of
the project's browser verification checklist in Round 24, when it had reached 1478 lines - a checklist you
cannot execute in one pass has stopped being a checklist.

These are **records, not instructions**. The checklist says what to check; this says what was found
when it was checked, with the bundle hash and date so a claim can be traced to a specific deploy.
Nothing here is deleted when it ages - a superseded measurement is evidence about how a number
moved, which is exactly what the next round needs when a figure looks wrong.

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
readout, and once for two classes of analytically dead parameter. The shipped figure is now accuracy
**0.883721** with minority-class F1 **0.84375** against the same logistic regression's **0.953488** /
**0.939394** - so the direction of the comparison is unchanged and only its size moved. The
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
  capacity **-51.58 bit/carrier at the default 800 km** (24.5 dB, g=280) and still **-2.15 at 200 km**
  (12.4 dB, g=17.5), the shortest downlink the panel offers. A page advertised a published model whose
  output no user action could produce. The abort is correct physics and is unchanged; the false dependency
  is gone.
- **The genomics 2D overlay still described "the 77 variants of PGS000001, 64 resolved"** after the page
  had moved to PGS000004. A stale overlay renders perfectly. Now 307 of 313, by GRCh38 coordinate.

Also verified live that round: telecom's RandomForest attribution renders with `dark_count_rate` drawn as
no measured effect (+0.0001 +/- 0.0010); the XAI paragraph says "no encrypted inference stage" on telecom
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

**3. A Japanese-named confidential note in the repo root passed every confidentiality gate - measured, not
hypothesised.** Copied in and walked through all three: not matched by the private-notes or `*.docx` patterns, so
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
the correct one - Grand View's current edition - and the **sourced** Value-page row was a blend. Its
"~$2.2B" is Grand View's 2026 value, "$20-30B by 2033-34" splices Grand View's 2033 endpoint to
Precedence's 2034 one, and "~38-46%" takes its high end from **MarketsandMarkets, a house the row does not
name, which sizes this market at $0.42B in 2025 - a quarter of the base the same row quoted.** The
footnote two rows below already describes this exact error ("producing a range no single analyst
projects") about a *different* row: the lesson had been learned for one row and not applied to the row
above it. The PQC row now lists each house separately, states the 4x scope spread rather than averaging
it, and the README points at the table instead of carrying a second copy.

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
regression. Independently corroborated this round: WebNN is still Chrome Origin Trial M147-M149 with
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
evidence supports: it remains a Chrome Origin Trial (M147-M149) at W3C Candidate Recommendation with
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
