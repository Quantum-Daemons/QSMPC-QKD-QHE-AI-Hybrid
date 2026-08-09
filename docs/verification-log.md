# Verification results log

What each round actually measured against the deployed demo. Split out of
`browser-verification-checklist.md` in Round 24, which had reached 1478 lines - a checklist you
cannot execute in one pass has stopped being a checklist.

These are **records, not instructions**. The checklist says what to check; this says what was found
when it was checked, with the bundle hash and date so a claim can be traced to a specific deploy.
Nothing here is deleted when it ages - a superseded measurement is evidence about how a number
moved, which is exactly what the next round needs when a figure looks wrong.

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
| 50 km | BB84 decoy-state · QBER 0.02 | 0.02 | 0.00906 | 1811 |
| 250 km | BB84 decoy-state · QBER 0.5 | 0.5 | 0 | 0 |

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

