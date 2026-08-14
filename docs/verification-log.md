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

Observed on `https://qsmpc-qkd-qhe-ai-hybrid.daemons.jp`, recorded rather than ticked.

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

Three CKKS students now run and reproduce: medical accuracy 0.730769, OPS-SAT rare-class F1 0.864865,
finance rare-class F1 0.333709 on its shipped rows. `satellite_comms.student` still reports
`DID NOT RUN` with its reason - it is scored on event-wise F0.5, a sequence metric the per-row
evaluator cannot compute.

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

Worked against `https://qsmpc-qkd-qhe-ai-hybrid.daemons.jp/usecase/finance_aml`, cache-busted, with
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
`aria-live`, `is_ai_generated`, `browser_override_reason`, `Article 50(2)`.

**One false negative worth recording.** The status indicator first appeared not to render. It was
rendering correctly - the run had simply finished before the check ran, and the indicator hides on
`done` by design. The lesson is the one this document already carries in another form: when a control
is conditional on a transient state, sample it DURING that state, not after.

**Facts re-derived rather than trusted:**

- QSDC abort threshold: `Cs = 1 - 2h(e)` reaches zero at **0.110028**, identical to Shor-Preskill
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

Verified on https://qsmpc-qkd-qhe-ai-hybrid.daemons.jp after each deploy, not locally.

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
hypothesised.** Copied in and walked through all three: not matched by `MONETIZATION*.md` or `*.docx` so
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
| keygen (first click) | 255.26 ms | 476.71 ms |
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
measured sum: **finance 700 ms**, **medical 1,269 ms** (579 ms of it the encrypted convolution).
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
| medical | `fhe_conv_rotation_share` 0.10 against the dense layer's 57%; tier 1,280 ms; 3.04 ms/rotation |
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
