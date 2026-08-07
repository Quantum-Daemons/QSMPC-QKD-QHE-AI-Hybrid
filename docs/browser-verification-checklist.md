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

Environment: `crossOriginIsolated` true, zero `/api` requests in the network panel, WebGPU adapter
present but software (SwiftShader) with `shader-f16` absent, WebNN API present and deliberately not
offered to ONNX Runtime.

Published model running inside the orchestration, per use case:

| Use case | Model | Execution provider | Latency | Input |
|---|---|---|---|---|
| finance_aml | finance_aml.gbm | onnx-wasm | 3 ms | 1x32 |
| medical_fl | medical_fl.resnet18 | onnx-webgpu | 288 ms | 1x1x28x28 |
| medical_fl | medical_fl.resnet50 | onnx-webgpu | 12842 ms | 1x1x28x28 |
| genomics | none registered, and the run says so | | | |
| telecom_ids | telecom_ids.qml | onnx-wasm | 0.44 ms | 1x7 |
| satellite_comms | satellite_comms.telemanom | onnx-webgpu | 151 ms | 1x250x6 |
| satellite_comms | satellite_comms.opssat | onnx-wasm | 4.44 ms | 1x19 |

ResNet-50 at 12.8 s against ResNet-18's 288 ms is a measurement, not a defect: it is far more work on
a software adapter. Recorded because a reader comparing the two should see why.

Selecting a CKKS student reports `DID NOT RUN (no substitute shown)` with the reason, records no
latency, and produces no plaintext benchmark row. Those four ship no evaluation sample, so there is
no input from their own corpus to run them on. The gap is visible rather than papered over.

Reproduction check: ResNet-18 scored 85.9 percent in the browser against 85.9 percent measured
offline on the same 156 rows, majority baseline 73.1 percent, 11.3 MB transferred.

## What this checklist cannot cover

- One browser, one adapter. Every latency here is from a software WebGPU adapter; a machine with a
  discrete GPU will differ, and nothing in this repo measures that.
- The health of a corpus. The checklist verifies that the number shown is the number measured, not
  that the corpus behind it is representative.
- Anything the page does not render. A metric with no reader passes every axis here, which is why
  the audit that finds those reads the code instead.
