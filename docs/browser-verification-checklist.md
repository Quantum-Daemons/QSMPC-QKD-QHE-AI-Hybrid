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

1. Before opening: button label, `aria-expanded="false"`, and the panel genuinely absent from the DOM
   rather than merely hidden.
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

Three CKKS students now run and reproduce: medical accuracy 0.730769, OPS-SAT rare-class F1 0.864865,
finance rare-class F1 0.333709 on its shipped rows. `satellite_comms.student` still reports
`DID NOT RUN` with its reason - it is scored on event-wise F0.5, a sequence metric the per-row
evaluator cannot compute.

Reproduction check: ResNet-18 scored 85.9 percent in the browser against 85.9 percent measured
offline on the same 156 rows, majority baseline 73.1 percent, 11.3 MB transferred. The medical
student scores 0.7308 against a majority baseline of 0.7308 - it reproduces its published figure and
does not beat guessing, and the page says both.

## What this checklist cannot cover

- One browser, one adapter. Every latency here is from a software WebGPU adapter; a machine with a
  discrete GPU will differ, and nothing in this repo measures that.
- The health of a corpus. The checklist verifies that the number shown is the number measured, not
  that the corpus behind it is representative.
- Anything the page does not render. A metric with no reader passes every axis here, which is why
  the audit that finds those reads the code instead.
