# Browser verification checklist

`deliverable-verification.md` says what "verified" means. This is the executable form of its gate 4,
the one that runs in Chrome against the deployed page.

It exists because gate 4 was being reconstructed from memory every round, and what gets reconstructed
is what someone remembers to look at. Round 20 audited controls on use-case pages and found five
inert ones; the pages it did not sweep kept theirs. This file is the sweep, written down.

## How to work it

A row passes when the observed value is written into it. A tick is not a result: the whole point of
gate 4 is that a control was set, the output was read, and the two were compared. Rows that do not
apply to a page are recorded as not-applicable rather than left blank, because a blank row and a
skipped row look identical a round later.

Run it against the public URL, not localhost. COOP/COEP, CDN reachability and cache behaviour all
differ, and this project has shipped a verification against a bundle that did not contain the fix.
Before trusting any result, grep the served bundle for a string literal from the change - bundle hash
parity proves the container matches the public URL, not that either matches source, and function
names are minified so only string literals survive.

**Then cache-bust the browser, and do it in that order.** The grep tells you what the server sends;
it says nothing about what the tab is running. Navigating to the same URL after a deploy can re-serve
the previous bundle from cache, and the page then shows the OLD text while the server has the new -
which reads exactly like a fix that did not work. That happened here: the served bundle contained the
corrected label and zero occurrences of the old one, and the tab still showed the old one. Append a
throwaway query (`?cb=` plus a timestamp) and re-read `document.querySelectorAll('script[src]')` to
confirm which bundle the tab actually loaded.

The failure is asymmetric and that is why it matters: a stale tab makes a working fix look broken,
you "fix" it again, and the second change is the one that ships unverified.

## The eight axes

| Axis | What passes |
|---|---|
| Content | Every claim is true of the running build. Nothing describes a feature that was removed. |
| Numerals | Config-traceable, measured in this session, or a cited literature value with its source. |
| Terminology | Matches the primary source. Never "experiment" for a simulation, never "information-theoretic" where it does not hold. |
| Buttons | Each does something observable. A disabled button carries a reason. |
| Open and close | Both states of every disclosure control, not only the open one. See below. |
| Selects and checkboxes | Each value produces a distinct, visible change. |
| Layout | No horizontal overflow at 1280 or at mobile width. No overlap. Measured against the viewport - see below, this line went three rounds unmeasured. |
| Data tables | Column headers match what the column renders. Baselines named. No placeholder rows. |

The data-table axis earns its place: the detailed-results table had a column headed `model` that
rendered `security_model`, so the table had no model column while appearing to have one. That
mattered the moment a second model appeared in it.

## Routes

Seven, plus two negative paths that must be checked rather than assumed.

| Route | Page | What to check beyond the axes |
|---|---|---|
| `/` | Overview | Claim list, architecture summary, any headline figure |
| `/value` | Introduction and Value | The regulatory dates. This page states obligations the product must actually meet. |
| `/validations` | Validations | Four run buttons (ML-KEM, ML-DSA, BB84, handshake), the MCP-QKD checkbox, result tables |
| `/orchestration` | Orchestration | Use-case selector with six entries, the DAG, Parameters disclosure, four export buttons |
| `/usecases` | Index | Depth badge, dataset name and licence, cross-page title consistency |
| `/usecase/:id` | Five use cases | The regions below |
| `/agentic-ai` | Redirect | Must redirect to `/usecases`. Round 12 removed the standalone page. |
| `/usecase/<bad-id>` | Negative | Must render "No such use case", never silently show Finance under another id |

## Regions on a use-case page

In DOM order. Region 10 is absent on non-inference use cases; record that rather than leaving it
blank.

| # | Region | Component |
|---|---|---|
| 1 | Title, blurb, industry and market pills | UseCasePage |
| 2 | Demo banner | demoMode |
| 3 | Crypto selects, published-model select, OptiQKD and MCP checkboxes, Parameters button | PipelineControls |
| 4 | Parameter fields, Reset to defaults | ParameterPanel |
| 5 | Two-party simulator diagram | TwoPartyDiagram |
| 6 | Four export buttons | UseCasePage |
| 7 | Event log, metrics table, signature | RunMonitor |
| 8 | Inference provenance badge and its disclosure | ProvenanceBadge |
| 9 | Per-model rows, Run buttons, result table | ModelPanel |
| 10 | SHAP attribution | ShapPanel |
| 11 | Benchmark rows and cost split | BenchmarkPanel |
| 12 | Engine, model and combiner selects, Run | AgenticAIPanel |
| 13 | Key budget, span, relay caveat | SdnControlPlanePanel |
| 14 | Footer state note | UseCasePage |

## The open and close axis

Two disclosure controls exist: the Parameters button (`PipelineControls`, with `aria-expanded` and
`aria-controls="pipeline-parameters"`) and the provenance detail toggle (`ProvenanceBadge`), which
appears more than once per page. Each is checked in six states, not one.

1. Before opening: button label, `aria-expanded="false"`, and the panel's CONTENTS not rendered -
   `hidden` set, `display: none`, `offsetParent === null`, and zero focusable fields inside it.

   This row previously demanded that the panel be "genuinely absent from the DOM rather than merely
   hidden", and that was wrong. `aria-controls` has to reference an element that exists, so the
   container must stay in the tree for the disclosure relationship to be valid at all; what must not
   exist is its content. Measured on the Orchestration page: the container is present with `hidden`,
   `display: none` and **zero** fields inside while collapsed, which is correct. The row as first
   written would have sent someone to "fix" working code and break the ARIA wiring doing it - so
   assert on the contents and the computed style, never on the container's existence.
2. The transition: no layout shift below it, no overlap with the diagram.
3. After opening: `aria-expanded="true"`, `aria-controls` resolves to the panel that appeared, and
   every field shows the value the current run would use.
4. After closing again: state restored, no orphaned DOM, field values retained rather than reset.
5. Reopened after a run: the values shown are the ones the completed run used.
6. Reopened after switching use case: this is the one that matters. Parameters survive a use-case
   change on the Orchestration page, and hidden controls used to keep reaching the run. Open on
   Finance, set QSMPC op, Inference and prime, switch to Generic demo, reopen. The panel must not
   present one use case's parameters as another's, and `buildConfig` must not ship a control the
   panel is no longer showing.

## Buttons, by group

Checked as sequences where the sequence is the point. A transport bug shows up in the transition, not
the click.

- Transport: Start, Start stepwise, Pause, Resume, Step, Restart, Stop. Run Pause then Step then
  Resume then Restart as one sequence.
- Exports: Diagram PNG, Animation GIF including its "Encoding" busy label, Events JSON, Metrics CSV,
  and the Orchestration page's CSV, JSON, PNG and hi-DPI DAG. Each must be disabled before a run with
  a stated reason, and produce a non-empty file after one.
- Reset to defaults.
- Validations: Run ML-KEM, Run ML-DSA, Run BB84, Run handshake.
- Per-model Run buttons in the Models panel, and the Agentic AI Run button.

## Numeric edges

`min` and `max` on a number input do not constrain a typed value - they style the spinner. Every
numeric field is therefore typed at its edges, not stepped to them.

| Field | Type these | Expected |
|---|---|---|
| prime | 0, 1, 2, 8, 10, 100, 3000000000 | 0 and 1 both auto; 8 and 10 both snap to 11; 3e9 clamps to 2147483647 and the run says why |
| qubits | -1, 0, 2, 14, 24 | clamped into 2 to 14, no allocation spike, `explanation` still present in the signed payload |
| n_per_class | 0, 20, 2000, 100000 | clamped into 20 to 2000; blank falls to the config default, not the panel's |
| qrng_bits | 0, 64, 1024, 16384 | the delivered count is reported; a shortfall against the request is explained |
| qrng_eps_log2 | -1, -40, -128 | moves `qrng_lhl_max_bits` on every run |
| qkd_eve, qkd_misalignment | 0, 0.1, 0.35, 1 | key dies well before the field maximum; the saturation is visible in the metrics |
| sat_distance_km | 1, 800, 1200, 5000 | saturates; QSDC aborts |
| seed | 1, 2 | different runs. 0 is no longer reachable: the generator maps it to 1 |

## The layout axis, in full

Added in Round 21 after a reader reported the published-model selector running off its card. The
axis had said "no horizontal overflow at mobile width" for three rounds and nothing had measured it.

**Measure against the VIEWPORT, not the parent.** This is the part that makes the check work at all.
Comparing an element's right edge against its containing card finds nothing, because the card grows
with the child: the 12-column benchmark table rendered 930px wide inside a 390px container and
registered as zero overflow. The first sweep written for this axis reported "0 overflowing" on a page
that had 936px of horizontal scroll.

    const vw = document.documentElement.clientWidth;
    [...document.querySelectorAll('table,svg,pre,code,select,input')]
      .filter(e => e.getBoundingClientRect().width > vw + 1)

**Check `document.documentElement.scrollWidth` against `clientWidth`** as the top-level assertion.
If they differ, something is pushing the page and the sweep above says what.

**A control that cannot shrink is the common cause.** A `<select>` sizes to its longest option and a
flex item defaults to `min-width: auto`, so neither can give way. Force the container narrow and
confirm the control follows rather than escaping:

    for (const w of [485, 430, 390, 360, 300]) { card.style.width = w + 'px'; /* measure */ }

**Check one level up after fixing.** Containing overflow in a card can create it in an ancestor:
giving `.card` a scroller pushed `.main` to 1086px against a 1265px viewport, adding 61px of page
scroll on a desktop that had none, because `.main` is a flex item that had no `min-width: 0`. Re-run
the top-level assertion after any overflow fix, at the WIDE viewport as well as the narrow one.

**Narrowest directly testable width in this environment is 485px** - the browser window will not go
below it. Behaviour below 485 is verified by forcing the container width, as above, not by emulation.
Record which method was used; they are not the same evidence.

### SVG text: neither DOM measurement works, so look at it

The diagram is SVG, and both of the measurements above are blind to it. This cost a full sweep.

`clientWidth` and `scrollWidth` are not meaningful on an SVG `<text>`. Running the element-overflow
pass over the diagram returned nine "overflowing" labels, every one of them a false positive - the
numbers describe a layout box SVG text does not use.

`getBBox()` is meaningful but measures the glyphs in USER space, against the `viewBox`. A label
centred on a party and running off the left edge still reports a bounding box that sits comfortably
inside the viewBox, because the box is measured from where the text starts, not from where the
element that should contain it starts.

What found it was a **screenshot**. The published-model step rendered as
`te_comms.opssat_student @ 82cb93255efb ...` - its first seventeen characters clipped away - because
`local-a` draws a fixed 300-unit box while that label is about 460 units wide. So:

- Take a screenshot of every diagram, in the state that produces its LONGEST label, and read it.
- Then assert in code: `bbox.x >= 0 && bbox.x + bbox.width <= viewBoxWidth` for every `<text>`.
  Both margins positive means it fits; either negative means it is clipped by the viewBox edge.
- The longest label is the one to test. For the published-model step that is a use case publishing
  several models, with the longest-named one selected - medical with ResNet-50 is the current worst
  case at 91 characters.

### A software WebGPU adapter is slower than no adapter

Not layout, but found by the same two-browser comparison and worth the same discipline: **record
which browser produced a latency, because the execution provider is not a property of the code.**

Same model, same single row, same deployed bundle:

| Adapter | Execution provider | ResNet-50 |
|---|---|---|
| SwiftShader (software rasteriser) | onnx-webgpu | 12842 ms |
| none obtainable | onnx-wasm | 42 ms |

`requestAdapter()` succeeds on SwiftShader, so "is WebGPU available" answered yes and the code took
the WebGPU build - 300x slower than the fallback it had just declined, because every shader dispatch
is emulated on the CPU while the wasm build runs SIMD kernels natively. `hasWebGPU()` now declines
software adapters by name. When recording any latency in this file, record the adapter beside it.

## During a run, not only after it

The simulator's own text changes while it runs, and a sweep that only looks at the finished state
misses it. For each use case, sample mid-flight as well as at the end:

- Metric values appear progressively. A field that renders an em dash before its stage and a number
  after is correct; one that renders `NaN`, `undefined`, `Infinity` or `null` at any point is not.
- Terminology is consistent between the diagram label, the run log and the metrics table for the
  same quantity. The same number must not be called three things.
- The published-model line names a model, a revision, an execution provider and an input shape, and
  reports either a class or an absolute error - never both, and never a class for a forecaster.
- Layout does not shift as content arrives: the log grows, the cards below it must not jump.
- Controls that must be inert during a run are disabled and say why, and the published-model select
  is one of them - changing the model mid-run would describe the run with the wrong model's name.

## Every panel must describe the run that just happened

The page performs one run and then eight panels describe it. Each was written at a different time,
and a panel that was correct when written goes quietly stale as the run grows. So after a run, read
every panel and ask the same question of each: **does this describe what the pipeline actually did,
all of it, and nothing it did not?**

Four failure shapes, all of them found on this page rather than reasoned about:

- **Describes only part of it.** "Inference provenance" read "MEASURED (plain JS) - js-linear
  (bundled real model)" on a run that had also fetched and executed a pinned ResNet-18. Both true;
  the panel named the lesser one, and the published model appeared nowhere until the Models panel far
  below. Now both, each labelled with the corpus it ran on.
- **Asserts something with nothing behind it.** The QSMPC-FHE benchmark row reported
  `accuracy: 1.0`. `qotpQcnnInfer` has no trained parameters at all - RY angles from the input, then
  seven FIXED rounds of H, CNOT and T - and the benchmark feeds it `u() * 2 - 1`, uniform noise.
  There was nothing for it to be accurate about, and the MPC-FHE row eight lines above carries a
  comment forbidding exactly that pattern. The two rows contradicted each other and the invented
  number was the flattering one.
- **A column that renders a different column.** The detailed-results table was headed `model` and
  rendered `security_model`, so it had no model column while appearing to have one.
- **Says the same thing twice, differently.** Two panels offering the same ResNet-18 run with
  different chrome; two quantum-advantage scores on one page.

Panels to read, per use case, after a run: Inference provenance · Models (including each "Run this
model in your browser" result) · Why - top features (SHAP) · Benchmark - MPC-FHE vs QSMPC-FHE ·
Detailed results · Security score and quantum advantage · Agentic AI · QKD control plane.

Cross-check them against each other and against the run log, not only against expectations:

- A model named in one panel is named the same way in all of them.
- A number appearing twice is the same number, or the two are labelled as different things.
- A panel that shows nothing after a run says why, rather than rendering empty.
- A row whose metric does not apply shows an em dash, and the em dash is correct rather than missing
  data - check which by finding what produced the row.

## The Agentic AI panel, control by control

Its own section because it is the one panel that performs work the pipeline does not, and its claims
are checkable rather than descriptive.

| Control | What to verify |
|---|---|
| Engine | Two options. WebLLM is disabled without a usable adapter, and the note above says WHICH case applies - absent, or present-and-declined. Those are different facts. |
| Model | Options change with the engine: Transformers.js lists ONNX models, WebLLM lists MLC builds. Selecting one and running reports that model id back in the result. |
| mix QKD-domain key | On and off must both reach the envelope: `KEM / QKD-mixed` flips, and the derived-key fingerprint changes with it. |
| Combiner | HKDF against Toeplitz+QLHL. Its tooltip claims "the derived-key fingerprint below changes with the choice" - so check that it does. Measured: `hkdf` gave `0b28dc66`, `its_toeplitz` gave `07224b14` on otherwise identical runs. |
| Run in-browser agent | Disabled while busy, progress text advances, and the result names the backend it used (`transformers-wasm`) and an elapsed time. |
| Download results (JSON) | Disabled BEFORE any run with a reason, enabled after, and the file contains the envelope shown on screen rather than a previous run's. |
| MCP envelope | `authenticated (ML-DSA-65) true` and `tampered message rejected true` are assertions the panel makes about work it did; both must be present after every run. |
| Staleness | Run the agent, then run the pipeline. The panel must mark its result stale rather than leaving an envelope describing superseded numbers under the new ones. |

## Simulator controls, and where each one has to show up

Every control below changes the run, so every one of them has to change something a visitor can see.
The rule from the inert-control findings applies: check the RENDERED output in all three places, not
the code. A control passes only when the diagram text, the run log and the metrics table agree with
each other and with what was selected.

| Control | Diagram | Log | Metrics / table |
|---|---|---|---|
| KEM | handshake arrow names ML-KEM-768 or 1024 | handshake line | `kem_category` 3 or 5, `kem_ciphertext_bytes` 1088 or 1568, security score moves |
| QKD (family) | quantum arrow present, or gone under pqc-only | "QKD disabled (PQC-only profile)" | `secure_key_bits`, QKD bonus in the security score |
| Protocol (bb84 / e91) | arrow reads BB84 decoy-state or E91 (entanglement / CHSH) | protocol line | `chsh_s` appears only for E91 |
| Signature | audit-record arrow names ML-DSA-65 or 87 | signing line | `signature_bytes` 3309 or 4627 |
| Published model | full-width step names model, revision, EP, input shape, latency, outcome | "published model X ran here on Y" | `published_*` metrics, plaintext benchmark row |
| Optimize QKD (OptiQKD) | OptiQKD suffix on the QKD arrow with the measured percentage | improvement and convergence lines | `qkd_skr_improvement_pct`, `qkd_convergence_evals` |
| OptiQKD samples | (via the suffix) | budget line | collapses in 8-wide buckets; saturates near 400 |
| QKD channel (pulses, distance, eve, misalignment) | QBER on the arrow | channel line | `qber`, `skr`, `secure_key_bits` |
| Free-space channel (downlink km, sat eve, sat pulses, QSDC abort threshold) | free-space arrow QBER, and the QSDC arrow must show the ABORT when it fires | "QSDC ABORTED: error X > threshold - telemetry withheld" | `qsdc_error_rate`, `qsdc_secrecy_capacity`, `qsdc_eavesdrop_detected` |
| QRNG (bits, extractor, eps) | not drawn | delivered-bits line, shortfall explained | `qrng_bits_delivered`, `qrng_lhl_max_bits`, `qrng_extraction_rate`, `qrng_health_conclusive` |
| Quantum / run (qubits, seed) | QSMPC comparison bar | T-gate line | `t_gate_count`, `qubits`; seed changes every downstream value |
| IDS (samples per class) | QML-IDS step macro-F1 | training line naming the value actually used | `ids_macro_f1`, `ids_train_per_class` |
| QSMPC op / prime | arrow kind changes: q-arrow for secure sum, c-bidir for PSI | field line, clamp line when clamped | `qsms_prime`, `secure_sum` or `psi_intersection_size` |
| Inference | QSMPC comparison bar names the surfaced backend | inference line | `primary_inference_class` |

Two traps this table exists to catch. A control can move the metrics and not the diagram - that was
five of the six inert controls. And a control can move both while a SECOND view of the same state
stays frozen, which was the sixth.

**Free-space channel is the row to work first**, because it is the only group whose whole purpose is
a threshold that has to be crossed to see anything, and it was the last group added to the panel.
Satellite only; the fibre QKD-channel group is correctly hidden there and vice versa. Set Eve to 0.5
against the default 0.11 threshold - well past the crossing near 0.32 - and check all three surfaces
agree that the transmission was refused.

That test is how the diagram-versus-log contradiction was found: the log said
`QSDC ABORTED: error 0.158 > 0.11 - telemetry withheld`, the metrics agreed, and the arrow read
`QSDC direct telemetry · secrecy 0 bit/carrier` with a done tick - delivered-but-not-secret, the
opposite of what happened, on the feature the page leads with. Observed after the fix:

    diagram   QSDC ABORTED · error 0.16 over threshold · telemetry withheld
    log       QSDC ABORTED: error 0.158 > 0.11 - telemetry withheld
    metrics   qsdc_eavesdrop_detected 1 · qsdc_secrecy_capacity 0

The same run also showed why a step must not be gated on something it does not depend on. The
published-model step sat inside the delivery branch, so an abort left it reading
`published model · —` with a done tick and no reason - even though that step runs the pinned model on
a row of the model's OWN corpus and has nothing to do with this run's channel. It now runs either
way: `published satellite_comms.telemanom @ 4507c187d0f9 · onnx-wasm · 1x250x6 · 9.59 ms ·
abs err 0.01615`.

## The source-code pass: hardcoding and fallbacks the browser cannot show you

Everything above is checked by looking at the page. This axis is not, because its failures are
invisible there: a hardcoded constant renders identically to a config-sourced one, and a swallowed
error renders as a plausible number. Run it against the diff, then confirm the fix in the browser.

**A number is config-traceable, session-measured, or a cited literature value.** Anything else is a
literal. The test that finds the real ones is not a grep for digits - it is: *change the config and
see whether the page moves.*

    # 1. edit config.yaml (the override layer - editing defaults.py alone proves nothing,
    #    because config.yaml pins the value back)
    # 2. python scripts/export_config.py && npm run build
    # 3. grep the built bundle for the key. Minified output writes 0.2 as `.2`, so match
    #    /key:([^,}]+)/ rather than the literal you typed.

Worked example, found this way in Round 21. `config.defaults.json` shipped
`qkd.fiber_loss_db_per_km` and `qkd.detector_efficiency` to the browser, and `clientCompute` spread
the `FIBER` literal from `quantumSim.ts` instead, overriding only three of five fields from the
`qkdCfg()` object it was already calling. The two agreed at 0.2 and 0.25, which is exactly why it
survived: editing `config.yaml` would have moved the Python channel model and left the browser on the
old constant, with no error anywhere. Verified after the fix: config 0.2 gives bundle `.2`, config 0.9
gives `.9`.

Still outstanding and recorded rather than fixed: `FREE_SPACE` in `quantumSim.ts` has **no** config
counterpart at all - its `lossDbPerKm` 0.035, `detectorEff` 0.4 and `misalignment` 0.025 exist only as
TS literals, so the satellite channel cannot be reconfigured without editing the source.

**A fallback must not be able to substitute a plausible answer for a failure.** Three shapes to grep
for, and the question to ask of each:

| Pattern | Ask |
|---|---|
| `catch {}` / `except Exception: pass` | If this fires, does the caller learn anything? |
| `x ?? default` / `x \|\| default` on a measured value | Is the default distinguishable from a real measurement? |
| A tier that silently degrades | Does the page say WHICH tier produced the number? |

The distinction that matters is between a fallback and a *report*. Declining a software WebGPU adapter
and using wasm is a fallback and correct - the run says which provider it used. Substituting the
bundled linear model when a published model fails to load is the same shape and is NOT correct,
because the page then shows a number for a model it never ran; that path now reports the failure and
shows nothing in its place. Two live `except Exception: pass` sites remain, both narrow and both
recorded: `usecases/medical_fl/__init__.py:168` (cosmetic SHAP feature names) and
`benchmark/gpu_mpc_adapter.py:57` (an optional GPU subprocess whose absence is handled two lines
later).

**Then confirm in the browser**, because a config value can reach the bundle and still not be read -
which is precisely the bug above. Capture each use-case page, change the parameter, capture again,
and diff the diagram text, the log and the metrics table. A value that reaches the bundle and moves
nothing on screen is the same defect as a control that renders and changes nothing.

## Navigate the way a visitor does, not the way a test does

Load each route fresh and this whole class of defect is invisible. `/usecase/:id` matches the same
route for every use case, so React Router does not remount the hook, and the Orchestration page
changes the use case without navigating at all. Anything held in that hook's state therefore crosses
a use-case boundary.

So sweep the routes through the SPA router - `history.pushState` plus a `popstate` - and only then
fall back to a fresh load to confirm what you found. Both defects of this shape were found this way
and neither was reachable from a cold load:

- Parameters (`qsmpc_op`, `prime`, `inference`) reached a run whose panel no longer showed them.
- The published-model selection survived into a use case that does not publish it, so
  `browserModels(next).find(...)` returned nothing and the page reported "no published model
  registered for this use case" - on telecom, which publishes one that runs in 6 ms. That claim is
  far stronger than the truth, because "none registered" is how a deliberate licence decision reads.

The general rule: anything keyed to the use case must be reset when the use case changes, or gated
at the point of use. Check both, because the two failures look identical from outside.

## Selects, every value

For each, set every option in turn and diff the rendered diagram, the run metrics and the log. Not
the code - the rendered output. Four of the five inert controls Round 20 found had correct-looking
code paths, and one carried a comment claiming the fix had already been made.

Audit every view that renders the same state, not one per page. The sixth inert control was found
only by repeating the sweep on the Orchestration graph, whose memo dependency array omitted the
protocol: the wiring was correct, the label function was correct, and the node still never changed.
Read the rendered node text rather than the DOM you expect - that graph renders React Flow HTML
nodes, and a sweep querying `svg text` returned nothing at all.

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

## What this checklist cannot cover

- One browser, one adapter. Every latency here is from a software WebGPU adapter; a machine with a
  discrete GPU will differ, and nothing in this repo measures that.
- The health of a corpus. The checklist verifies that the number shown is the number measured, not
  that the corpus behind it is representative.
- Anything the page does not render. A metric with no reader passes every axis here, which is why
  the audit that finds those reads the code instead.
