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

## The nine axes

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
| UX and accessibility | A first-time reader can find the control, understand the output, and complete the task. Contrast, focus, keyboard, labels. See below. |

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

## Fact-check the numbers against current sources

A number can be internally consistent and still describe a world that has moved. So each round,
re-verify the load-bearing external claims and record the date:

- Every baseline the pages print beside a measured figure has an entry in `references/` with a
  resolvable identifier. An audit in Round 21 found two entries with none, and ten citations used as
  authority in registry docstrings or the SPA's baked config with no entry at all.
- Regulatory dates are the ones most likely to move and the most costly to get wrong, because the
  product asserts compliance against them. EU AI Act Art. 50 became applicable during Round 21 while
  the panel it governs carried no disclosure.
- Standards versions and publication dates: ETSI GS QKD 014 / 015 / 020, NIST FIPS 203/204/205, and
  whichever instrument is currently binding. Round 21 confirmed GS QKD 020 at V1.1.1, 2026-06-29, and
  found `pqc-roadmap.md` more precise than the secondary sources - it also records that
  `pqc/transport.py` targets 014 only.
- When a doc turns out to be ahead of the sources you find, say so in the commit rather than
  "correcting" it downward to match a summary.

## Reproduce each panel's headline from what the page itself shows

The fact-check above asks whether an external claim is still true. This asks something stricter and
entirely local: **can a reader recompute the number from the inputs printed beside it and the formula
in `references/`?** A figure that cannot be recomputed from the page is an assertion, however well
sourced.

Two things have to hold. The panel prints its own inputs, and the formula is traceable to a citation
or to config rather than to a literal in the component.

| Panel | Headline | Recompute from | Where the formula lives |
|---|---|---|---|
| Security score and quantum advantage | advantage `n/10` | the ratios the card prints | `quantum_advantage` weights in config; mirror of `benchmark/quantum_advantage.py` |
| Security score | `68` / `96` | KEM category, QKD key bits, QBER | `security_scoring` weights in config; `benchmark/security.py` |
| Why - top features (SHAP) | per-feature phi | `W`, `b`, `x`, baseline | exact linear SHAP, `phi_i = w_eff_i (x_i - baseline_i)`, `w_eff = W[1]-W[0]`; mirror of `ai/explainability.py` |
| Benchmark - MPC-FHE vs QSMPC-FHE | latency split | FHE-linear + MPC-non-linear ms | measured, so check the parts sum to the whole |
| QKD control plane, GS QKD 015 | key budget, span | `secure_key_bits`, distance | field names must match the spec's own (`qkdn_id`, `qkdl_id`, `skr`, `eskr`) |
| Quantum-kernel SVC + variational QNN | macro F1 | published per-class scores | `registry.py` metric spec + its baseline citation |
| Inference provenance | badge state | the backend the run reported | the five-state enum in `ProvenanceBadge` |
| Agentic AI, MCP | derived-key fingerprint | combiner choice | changes with the combiner - verified, not assumed |

Worked example, run in Round 21. The advantage card printed
`communication ×6145.8 · latency ×3608.79 · security ×1.5` beside a score of `10/10`. Taking the
`quantum_advantage` weights from config rather than retyping them - `w_comm 0.6`, `w_latency 0.2`,
`w_security 0.2`, `log_scale 3.2`, `log_offset 1.0` - gives
`6145.8^0.6 x 3608.79^0.2 x 1.5^0.2`, then `log10(composite + 1) x 3.2 + 1.0`, which lands on 10 and
matches. The score is checkable in place by anyone reading the page.

The security score reproduces the same way, and doing so found something: `68 -> 63` and `96 -> 91`
are correct, but the second pair only in the BROWSER. `secFromConfig` derives both rows' key bits
from the run, while the Python quantum row pins `qkd_key_bits=256` and never moves. Reproducing a
number is how you discover which deployment it describes.

**Two failure modes to look for, and they are opposite.** A headline whose inputs are not printed
cannot be checked at all. And a headline whose inputs ARE printed but which does not recompute is
worse, because it looks checkable. When a recomputation disagrees, suspect the reader first - a first
attempt here missed the inputs entirely because it grepped for the metrics-table format while the
card renders them as prose, and nearly reported a working panel as unverifiable.

**A third, quieter one: check whether the headline is SATURATED.** Re-running the advantage
recomputation later in Round 21 reproduced again - 10.0000 against a shown 10 - but on different
inputs (`latency x1265.67` where the earlier run had `x3608.79`, because this round's work changed the
latencies). It reproduced both times because the raw score is 10.37 and 10.66 respectively, and the
formula clamps to 10.

So the panel has been sitting ABOVE its ceiling. It reproduces, and it is correct, and it carries
almost no information: the score only becomes discriminating once the communication ratio drops
below about 6000 or the latency advantage disappears.

    comm 6145.8  lat 1265.67  -> raw 10.37  shown 10
    comm 6145.8  lat    1.00  -> raw  8.39  shown 8.39
    comm    100  lat    1.00  -> raw  5.03  shown 5.03

A number pinned at its clamp is the numeric cousin of a control that renders and changes nothing: it
passes every check and tells the reader nothing. Record the RAW value alongside the clamped one when
recomputing, so the saturation is visible rather than hidden by the ceiling.

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

**Closed in the same round it was recorded.** `FREE_SPACE` in `quantumSim.ts` had no config
counterpart at all - `lossDbPerKm` 0.035, `detectorEff` 0.4, `misalignment` 0.025 and `darkCount`
existed only as TS literals, so four of the satellite channel's six physical constants were
unreachable from configuration while every call site spread the object and overrode just distance and
Eve. `QKDConfig` now carries a `free_space_*` block and BOTH channels are built from `qkdCfg()` at
module load, so neither has a literal left. Values were moved, not retuned.

Verified by the config-to-bundle test above rather than by reading the diff: config `0.035` emits
`free_space_loss_db_per_km:.035`, config `0.9` emits `.9`, and restoring returns `.035`. Note the
minification - the bundle writes `.035`, never `0.035`, which is why the test matches on
`key:([^,}]+)` and not on the literal you typed. The config also lands in its OWN chunk, not the main
`index-*.js`, so grep every asset rather than the one you expect.

**A fallback must not be able to substitute a plausible answer for a failure.** Three shapes to grep
for, and the question to ask of each:

| Pattern | Ask |
|---|---|
| `catch {}` / `except Exception: pass` | If this fires, does the caller learn anything? |
| `x ?? default` / `x \|\| default` on a measured value | Is the default distinguishable from a real measurement? |
| A tier that silently degrades | Does the page say WHICH tier produced the number? |

The grep that finds the first shape:

    grep -rn "catch" web/src/lib/*.ts | grep -vE "status\.error|console|throw|emit|return \{|note:|error:"

Round 21 found one that way, and it had been shipping. Both encrypted paths did
`try { ctBytes = saveToArray(...).length } catch { ctBytes = 0 }`, and `clientCompute` divides that by
`featDim * 8` to produce the ciphertext/plaintext ratio - so a failed size read printed **ct/pt 0**
beside rows carrying real ratios. Zero is not a small ratio, it is an impossible one, since a
ciphertext is always larger than its plaintext. It appeared under a MEASURED label with nothing
thrown and nothing warned. `ctBytes` is nullable now, the ratio is withheld rather than zeroed, and
the cost model's literature figure is deliberately NOT substituted in its place - borrowing an
estimate into a measured row is the same error in better clothes.

**Not every substitution is a defect, and the checklist should not imply it.** A default that comes
from config is a default. `demoMode`'s query-parameter handling, `config.ts` returning early from
`refreshConfigFromApi` in demo mode, and `clientMcp`'s `catch { tamperRejected = true }` - which is
the EXPECTED outcome of a deliberate tamper test - are all correct. The question is only ever whether
the substituted value is later reported as though it had been obtained honestly.

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

Two points, not one, and predict the second before running it. The free-space fix was confirmed by
`800 km -> 28 dB` and `2000 km -> 70 dB`, both exactly `distance x 0.035`, with the diagram arrow
reading `free-space BB84 decoy-state (28 dB) · QBER 0.04`. One point cannot distinguish "reads the
config" from "happens to agree with the literal it replaced" - which is the failure mode this whole
axis exists for.

**Assert the field took before reading the result.** These are React-controlled inputs: setting
`.value` and dispatching `input` is not synchronous with the state the next run will read, and a
prior script that errored mid-flight can leave a value behind. This produced a false alarm here - a
run reported 70 dB where 28 was expected, which looks exactly like a broken fix, and the field was
still holding 2000 from an earlier aborted call. Read `el.value` back and throw if it is not what you
set, before clicking Start. The arithmetic is what exonerated the code: 70 is 2000 x 0.035, so the
config value was being read correctly all along.

## Reproduce the formulas against the papers they cite

The axes above ask whether a number is measured, config-traceable and consistently displayed. None of
them asks whether it is **right** - whether the closed form in the code is the closed form in the
cited source. A correctly-plumbed, config-driven, consistently-rendered wrong formula passes every
other check in this file.

Three levels, cheapest first, and all three are needed:

**1. Re-derive the closed form independently.** Write the textbook expression in Python, not in
TypeScript, and compare. Writing it in the same language invites copying the implementation.

**2. Check the limits and the bounds the theory imposes.** These are the assertions a paper makes
that an implementation can violate silently - a maximum that can be exceeded, a threshold in the
wrong place, a quantity that should be monotone and is not.

**3. Cross-check two independently-reported metrics that the formula relates.** If the run reports
both `skr` and `secure_key_bits`, their definitions constrain each other; if they disagree, one of
them is not being computed the way it is described.

Verified in Round 21, all against `references/`:

| Quantity | Source form | Implementation | Check |
|---|---|---|---|
| Binary entropy `h2` | Shannon | `-x log2 x - (1-x) log2(1-x)` | `h2(0.5) = 1.000000` at the maximum |
| Key fraction | GLLP / Shor-Preskill `R = 1 - h2(e_ph) - f h2(e_bit)` | `1 - h2(q) - 1.16 h2(q)`, so `e_ph = e_bit = q` | zero-key QBER **0.098106**; closed form `h2(q) = 1/(1+1.16) = 0.462963` gives the same, and the Parameters sweep independently measured the key dying near 0.097-0.098 |
| CHSH | Werner state, `S = 2 sqrt(2) V` | same | Tsirelson bound `2.828427` reached only at `V = 1` and never exceeded, since `V <= 1`; violation threshold `V = 1/sqrt(2) = 0.707107` |
| Leftover hash lemma | `l <= H_inf(X) - 2 log2(1/eps) + 2` | `minH - 2(-eps_log2) + 2` | identical once `log2(1/eps) = -eps_log2`; monotone decreasing in `|eps|`, as the lemma requires |
| Free-space link budget | `loss_dB = distance x loss_per_km` | config-sourced 0.035 dB/km | 800 km gives 28 dB, 2000 km gives 70 dB, both exact |
| Linear SHAP | `phi_i = w_eff_i (x_i - base_i)`, `w_eff = W1 - W0` | same | completeness, dummy and linearity all hold over 2000 random linear models, re-derived in Python |
| Security score | `benchmark/security.py` | `securityScore` in `clientCompute.ts` | two independent implementations agree: MPC-FHE **68**, QSMPC-FHE **96** on the same configuration |

The security-score row is the strongest kind of check available here, because the browser mirror and
the Python original were written separately and can be run against each other. Do that rather than
reading either one: `compute_security_score` with the call sites' own arguments gives 68 and 96, and
the deployed benchmark table prints 68 and 96.

**An axiom that holds by construction still has to be computed.** `linearShap` returned
`efficiency_gap: 0` as a literal. It is algebraically correct - the attributions telescope to the
margin exactly - but a field named after a measurement and set to a constant cannot fail, so if the
attribution rule ever stopped satisfying completeness the payload would go on asserting that it did.
It is the residual now. This is the same defect class as a control that renders and changes nothing,
moved into a signed artefact: the value was right, and it was not evidence of anything.

Level-3 cross-check, measured live on telecom at QBER 0.01: `skr` 0.00958 and `secure_key_bits` 1915
against 200000 pulses. `skr x pulses = 1916`, which agrees with the reported bit count to within the
floor the implementation applies - so the two metrics are computed from one quantity rather than two.
The GLLP fraction at that QBER is 0.825487, implying 2320 sifted bits and a sift ratio of 0.0116.

**What this axis cannot do.** It checks the formula, never the physics: every QBER and SKR here comes
from an analytic channel model, and no amount of internal consistency makes a simulated number a
measured one. It also cannot check a formula the code does not implement in closed form - the
decoy-state bound the optimiser searches over is verified by its behaviour, not by re-derivation.

## Check the MODEL, not only the number

A figure can match the literature at one operating point and come from the wrong physics. That is
harder to see than a wrong number, because the one point you check agrees.

Round 21's case: the satellite downlink reported 28 dB at 800 km, comfortably beside the 27 dB
Micius publishes - and it was computed as `distance_km * 0.035`, a FIBRE attenuation law. A satellite
link is not attenuated per kilometre. Its dominant term is diffraction, which scales as 1/r^2, and
the atmosphere contributes a few fixed dB because a downlink crosses it once near the ground rather
than along the whole path. So the model was right where it had been calibrated and wrong everywhere
else:

    km      linear law    diffraction
    800        28 dB         24.5 dB
    1200       42 dB         28.0 dB
    2000       70 dB         32.4 dB
    5000      175 dB         40.4 dB

The Downlink control has no maximum and steps by 50, so a visitor reached meaningless numbers a short
drag past the default - and the QBER saturating at 0.5 looked like a channel result rather than an
artefact of the wrong law.

Verified on the deployed page after the fix, two points, each predicted before it was run:

    800 km    24.5 dB   QBER 0.03    (Micius publishes 27 dB total)
    2000 km   32.4 dB   QBER 0.04    (was 70 dB, QBER 0.5 saturated)

Both match the diffraction calculation exactly, and the QBER no longer pins at 0.5 for a link a real
mission would close.

So for any physical quantity the page reports:

- **Evaluate it at two or three points, not one**, and check the SHAPE against the literature -
  linear, inverse-square, exponential. A single agreeing point proves calibration, not correctness.
- **Ask what the dominant loss term actually is** for the regime being modelled. Fibre attenuation and
  free-space diffraction are different physics wearing the same units.
- **Check the reported metric and the computed one are the same quantity.** Fixing this nearly
  shipped a second defect: `bb84()` and `e91()` recomputed the loss for their result object with the
  old linear product, so with the new model selected the page would have REPORTED 0 dB while the
  transmittance used the real figure.

## Capture every control's value against its OWN constraints

Distinct from the numeric-edges pass, which types values in. This one types nothing: it dumps every
control on the page with its current value and its own `min` / `max` / `step`, and asks whether the
two are consistent. One script per page, and read the output rather than skimming it.

    const tog = [...document.querySelectorAll('button')].find(b => /Parameters/.test(b.innerText));
    if (tog?.getAttribute('aria-expanded') === 'false') tog.click();
    // then dump:
    //   selects     -> id, value, and EVERY option as `value = text`
    //   numbers     -> id, value, min, max, step
    //   checkboxes  -> id, checked, and the label text they sit inside

What it catches that nothing else does:

- **A default outside its own field's range.** `pp-seed` shipped `value = 0` with `min = 1`. Round 21
  raised the minimum, because the generator does `(seed >>> 0) || 1` and maps 0 and 1 to one stream,
  and left `makeDefaultParams` on 0. So every page rendered a field below its own minimum and "Reset
  to defaults" restored an out-of-range value. Reading either the min or the default alone shows
  nothing; only the pair does.
- **An option whose `value` and `text` disagree about what it means** - the `pp-inf` empty option
  read "(use case default)" while resolving to `ckks`, naming a mechanism that does not exist.
- **A control with no `id`**, which cannot be targeted by `label[for]` or by a checklist script. The
  Agentic panel's "mix QKD-domain key" checkbox is the one remaining case; it is wrapped in its label,
  so it is correct for assistive technology, but it is the only control the sweep cannot address by
  name.

Run it on every use-case page, not one: the control set differs per page, and the pages that hide a
group are exactly where a stale default survives unnoticed.

## Every control, against a published reference value

The simulator-controls table says which controls exist and where each must show up. It does not say
what the resulting NUMBER should be, so a control could move the display and still be wrong. This
section gives each control a **reference value or a formula from outside this project**, so the check
is falsifiable rather than "the number changed".

Set the control, run, read the result, and compare against the right-hand column. Record the observed
value, not a tick.

### KEM (`sel-kem`)

| Selection | Fact | Reference |
|---|---|---|
| ml-kem-768+x25519 | NIST Category 3, ciphertext **1088 bytes** | FIPS 203 |
| ml-kem-1024+x25519 | NIST Category 5, ciphertext **1568 bytes** | FIPS 203 |

Both sizes are specified by FIPS 203, so a reader verifies them without trusting this project. The
facts are pinned to the completed run: changing the selector without re-running must NOT change them.

### Signature (`sel-sig`)

| Selection | Fact | Reference |
|---|---|---|
| ml-dsa-65 | NIST Category 3, signature **3309 bytes** | FIPS 204 |
| ml-dsa-87 | NIST Category 5, signature **4627 bytes** | FIPS 204 |

### QKD protocol (`pp-proto`)

**BB84 (decoy)** - the secure fraction comes from the GLLP bound over the decoy-state estimate. The
verdict must be taken from `secure_key_bits > 0`, NOT from a QBER threshold: this repository's
measured zero-key point is **0.098106**, below the textbook 0.11, so a QBER in [0.098, 0.11) passes
the textbook test on a channel that yields nothing.

**E91 (CHSH)** - three independent checks, all of which must hold together:

    2 < S <= 2*sqrt(2)             classical bound < S <= Tsirelson bound (2.8284)
    S = 2*sqrt(2) * (1 - 2*QBER)   Werner-state relation between the two displayed numbers

The second is the strong one, because it ties two separately-rendered metrics together. Measured
2026-08-08: QBER 0.0247 predicts S = 2.6887; the page reported **S = 2.7**, and the diagram read
"E91 (entanglement / CHSH)". An S above 2.8284 is unphysical and is a bug, not a good result.

### QRNG extractor (`pp-ext`) and LHL epsilon (`pp-qrng_eps_log2`)

| Selection | Expected behaviour | Reference |
|---|---|---|
| von Neumann | extraction rate **<= 0.25**; no LHL bound reported | von Neumann debiasing yields at most p(1-p) per pair, max 1/4 at p = 1/2 |
| Toeplitz (LHL) | higher extraction rate, far lower throughput, and `qrng_lhl_max_bits` appears | Toeplitz hashing approaches the min-entropy rate but costs a matrix product |
| seedless (2-source) | distinct from both | two-source extractor |

Measured: von Neumann 0.2 with 3500.9 kbps; Toeplitz 0.5 with 28.2 kbps. The rate rises and the
throughput collapses, which is the correct trade and the wrong-way-round would be the bug.

**The LHL bound is exactly reproducible**, and this is the best single check in this section:

    m <= H_min - 2*log2(1/eps)

At `pp-qrng_eps_log2` = -40 the term is 80 bits; at -80 it is 160. So halving epsilon's exponent must
drop `qrng_lhl_max_bits` by **exactly 80**. Measured: 7309 -> 7229, delta exactly 80. Any other delta
means the extractor is not applying the Leftover Hash Lemma it claims.

### QKD channel (`pp-qkd_eve`, `pp-qkd_misalignment`, `pp-qkd_distance_km`, `pp-qkd_pulses`)

**The textbook anchor, and it costs one run.** Set `pp-qkd_eve` to 1.0 and `pp-qkd_misalignment` to
0. Full intercept-resend on BB84 gives **exactly 25%** - Gottesman and Lo, IEEE Trans. Inf. Theory
49, 457 (2003), verbatim: "For BB84, the intercept and resend strategy gives an error rate of 25%."
Measured 2026-08-08: QBER **0.25**, 0 secure key bits, step marked failed. Any other value means the
channel model is wrong, and this check is cheaper than every other one in this document.

Get the thresholds right when quoting them. From Gottesman-Lo Table 1: BB84 is proven secure below
**11.0%** one-way (Shor-Preskill) and 18.9% two-way. **14.6% is an INSECURITY bound**, not a security
threshold, and is widely misquoted as one. 12.6% and 12.9% are not canonical figures at all.


Eavesdropper 0 -> QBER 0.01, 1915 secure bits, SKR 0.00958/pulse, step ticked normally. Eavesdropper
0.35 -> QBER 0.1, **0 secure bits**, SKR 0, and BOTH the posture table and the diagram report the
failure. Both directions are required; testing only the failing one cannot catch a predicate that
always fails.

### Decoy-state necessity - what the decoy states are actually buying

The runtime QKD model is the ASYMPTOTIC decoy-state one, which already assumes the single-photon
contribution has been bounded. That assumption is invisible on screen until the counterfactual is
computed, so the posture table reports the GLLP tagged fraction over the run's own channel:

    Delta = P(n >= 2) / Q_mu,   Q_mu = Y_0 + 1 - exp(-eta*mu),   P(n >= 2) = 1 - exp(-mu)(1 + mu)

`Delta >= 1` means the worst-case bound on Eve-known detections exceeds the total detection
probability, so not one detection could be certified single-photon and a non-decoy link yields
nothing. Measured on the default fibre link (eta 0.025 at 50 km, mu 0.5): **Delta = 7.26** - the
decoy states are carrying the entire link, not merely improving it.

Recompute it from the config to check the row:

    eta   = detector_efficiency * 10^(-fiber_loss_db_per_km * distance_km / 10)
    Delta = (1 - exp(-mu)(1 + mu)) / (dark_count_prob + 1 - exp(-eta*mu))

Raising Distance must raise Delta; raising mu must raise it too. Both are one-line checks.

**The reference model is asserted against the papers in `tests/test_decoy_reference.py`** - 13
tests, none of whose numbers originate in this repository. The anchors, from Ma et al. PRA 72,
012326 (2005) with the GYS parameter set, and GLLP QIC 4, 325 (2004):

| Assertion | Published | Measured |
|---|---|---|
| tagged fraction at mu = 0.1, zero distance | > 1 | 1.0417 |
| decoy max distance | over 140 km | 142.2 km |
| non-decoy max distance | about 42 km | 39.8 km |
| decoy R/eta (flat, linear in eta) | 0.11 | 0.1136 |
| non-decoy R/eta^2 (flat, quadratic) | 0.15 | 0.1510 |
| decoy optimal mu | about 0.5 | 0.3 - 0.8 |

**The trap in reproducing those figures, and it is a quiet one.** The published curves are plotted
for the efficient BB84 of Lo-Chau-Ardehali, where the sifting factor q tends to 1. This project runs
standard BB84 with q = 1/2. Reproduce the paper with q = 1/2 and every value comes out at exactly
half the published one - 0.0568 against 0.11, 0.0755 against 0.15. That is close enough to look like
a fitting or units discrepancy and wrong enough to make the comparison meaningless. `q_sift` is an
explicit field for this reason, and a test asserts the factor of two so it cannot drift.

### Free-space channel (satellite: `pp-sat_distance_km`, `pp-sat_eve`, `pp-sat_pulses`, `pp-qsdc_threshold`)

Loss must rise with distance and the reported dB must appear in the diagram label. Measured at Eve
0.4: 24.5 dB, QBER 0.13, 0 secure bits, step marked failed; QSDC then aborted with "error 0.13 over
threshold - telemetry withheld". The abort keeps its tick, correctly - aborting IS the protocol
working, and the label says ABORTED unambiguously.

Open question recorded rather than silently changed: `pp-qsdc_threshold` defaults to **0.11**, the
textbook BB84 figure, which sits above the GLLP zero-key point. QSDC is a different protocol with its
own security analysis, so 0.11 is not automatically wrong there - but it has not been traced to a
QSDC source, and it should be.

### Published model (`sel-published-model`)

Each option must produce a run whose diagram names the model AND its commit hash, so the artefact is
identifiable. Measured: "published finance_aml.gbm @ c46157c27e64 - onnx-wasm - 1x32 - 0.34 ms -
class 0". Check the hash against the registry, and check that selecting a different option changes
it. Note that telecom and genomics ship no selector because they publish a single model; confirm
that is still true rather than assuming it.

### OptiQKD (`sel-optiqkd`) and the key-mixing checkbox

Ticking OptiQKD must emit `qkd_skr_improvement_pct` and a convergence count, and the diagram arrow
must carry the measured improvement rather than merely "OptiQKD on". Unticked, neither appears.
The "mix QKD-domain key" checkbox controls whether QKD material enters the session key at all;
unticking it must drive the posture table's QKD row to zero the same way `pqc-only` does.

## Every panel, against its cited standard

Four of these panels had ZERO mentions in this checklist before Round 22 while being the panels a
technical reader looks at hardest. Each row below is a fact checkable against a specification or
recomputable from numbers the panel itself prints.

### Security posture & quantum advantage / Benchmark - MPC-FHE vs QSMPC-FHE

The same component under two headings: the compact form (telecom, satellite - no private inference)
and the full form. **The compact heading read "Security score & quantum advantage" until Round 22
and was stale the moment the composite was retired** - it advertised a number the panel no longer
showed. Whenever a metric is retired, grep for its name in headings, legends and index pages, not
only at the computation site.

- Posture rows: see the per-control section above (FIPS 203/204 sizes, NIST categories, GLLP).
- Quantum advantage: a weighted geometric mean, tiered 0-10, with the **uncapped raw value printed
  beside it when the run is past the ceiling**. Recompute from the panel's own printed ratios:

      raw = log10(comm^0.6 * lat^0.2 * sec^0.2 + 1) * 3.2 + 1.0

  Measured 2026-08-08: comm 6145.8, latency 3396.03, security 1.5 -> raw 10.65, and the page printed
  `raw 10.65 (clamped to 0-10)`. Exact.
- The legacy 0-100 composite is retained below, and must be visibly demoted and labelled retired.

### QKD control plane - ETSI GS QKD 015

A display panel with no controls, so its verification is terminology and internal consistency.

- Three planes must stay separated, and the Agentic-AI panel must be described as a **SAE**, out of
  scope of the control interface per GS QKD 015 clause 1. If the agent panel ever appears to drive
  the quantum layer, that is a modelling error, not a feature.
- Key association link (clause 4.5): `qkdl_type` PHYS for a direct channel, and the trust note must
  say no intermediate node handles the key. `eskr <= skr`.
- **`skr` must equal the run's `secure_key_bits`.** Measured: both 1915.
- Key delivery (GS QKD 014): `key_size` **352**, quoted from the specification's own Status example
  where it is defined as "Default size of key the KME can deliver to the SAE (in bit)". And
  `stored_key_count` must equal `floor(eskr / key_size)` - measured floor(1436/352) = 4, displayed 4.
- `max_key_per_request` is **8** here against the specification example's 128. That is legitimate -
  a KME advertises its own limit - but it must be labelled as this simulator's choice. Round 22 found
  it, and `key_size`, as bare uncited literals; `key_size` was written twice, so editing one copy
  would have made the field and the divisor silently disagree.
- The panel must name the edition it targets: **GS QKD 014 V1.1.1 (2019-02)**, still the only
  published edition.

### Inference provenance

Names which model produced the reported class, from a five-state enum. With two inferences running
(bundled student + published model) the badge must name BOTH and say which one the reported class
came from. The SHAP panel below it must name the model it explains and the reference point it
explains against - see the XAI section.

### Why - top features (SHAP)

- Exact linear SHAP: `phi_i = w_eff_i * (x_i - baseline_i)` with `w_eff = W[1] - W[0]`.
- **Completeness holds for ANY baseline**, so `efficiency_gap` cannot detect a wrong reference point.
  Check the baseline separately: it must be E[x] over the dataset, not the zero vector. The panel
  must state which reference point it used.
- The published model has no attribution, and the panel must say so and why: exact linear SHAP
  applies to the linear student, not to a tree ensemble, a ResNet or a fixed circuit.
- With Inference set to QCNN the panel must gate or relabel rather than silently explaining the CKKS
  model for a class the page no longer reports.

### Agentic AI - quantum-safe MCP

Three selects (Engine, Model, Combiner), each of whose options must produce an observably different
run - see the per-control section. The Combiner choice (`hkdf` vs `its_toeplitz`) must change the
derived key fingerprint. The panel is a SAE and must not appear to control the quantum layer.

### Quantum-kernel SVC + variational QNN

Report the kernel, the qubit count and the T-gate count from the actual circuit, not from a
description. Qubit count must track `pp-qubits`. Accuracy must be reported against a named baseline.

## A completion marker is not a verdict

The diagram ticks a step when its stage finishes. That tick renders immediately after a measured
value, and a reader cannot tell "this stage ran" from "this value is fine".

Round 22 caught the live case. With the eavesdropper parameter at 0.35 the diagram drew

    BB84 decoy-state · QBER 0.1 ✓

while the posture table on the same page, for the same run, reported **0 secure bits**, an SKR of
**0 per pulse**, and "no secure key at this error rate". One run, two panels, opposite conclusions -
and the reassuring one was the bigger and more looked-at.

So the check is: **after every run, read the diagram labels and the tables against each other**, not
each on its own.

    [...document.querySelectorAll('svg text')].map(t => t.textContent.trim()).filter(t => /✓|✗/.test(t))

For each ticked step, find the number it sits beside and find where else the page reports the same
quantity. If the two disagree, the tick is the bug. Steps now declare an outcome (`ok`) separately
from completion, and a failed one is drawn with `✗` in a warning colour - but the axis outlives this
particular fix, because any status glyph next to any measured value invites the same misreading.

Note the trap that made this hard to see: `body.innerText` flattens SVG into the surrounding text, so
the diagram's tick appears to belong to the table row above it. Read `svg text` nodes separately from
table cells, or you will attribute a finding to the wrong panel - as happened here first.

## The UX axis, and the accessibility half nobody was checking

The layout axis asks whether anything overflows or overlaps. It never asked whether a visitor can
use the page. Those are different questions and only the first was being answered: a grep for `WCAG`
across this repository returned **zero hits** before Round 22, despite an earlier round having
scheduled a WCAG 2.2 phase.

Work this against the public demo, not a local build, and against **a reader who has not seen the
project** - that is the whole difficulty of the axis.

**Comprehensibility.** For each panel headline, can a reader say what the number means and what would
move it, from the page alone? The two failures this project has shipped are both of this kind: a
composite score with no printed inputs, and an attribution panel that did not name which of two
models it explained. State the unit, the reference point, and the source.

**Information hierarchy.** The most important number on a panel should be the largest thing on it. If
a legacy metric is retained for comparison, it must be visibly demoted rather than sitting at equal
weight beside the current one.

**Task flow.** Run each use case as a visitor would - land, run, read - and note every point where
you had to know something the page did not tell you. Disabled controls must carry their reason at
the point of disablement, not in a paragraph further down.

**Accessibility, measured rather than assumed** (WCAG 2.2 AA, the four that this UI can actually
fail):

- **1.4.3 contrast 4.5:1** for body text, 3:1 for large text and UI-component boundaries. The muted
  greys and the coloured card borders are the risk. Measure, do not eyeball - but measure correctly:

  **Composite the alpha, or you will invent failures.** This UI tints its notes with `rgba(246, 183,
  60, 0.08)` - an 8 percent amber over a dark card. Read that as an opaque background and near-white
  text scores 1.51:1, which is a loud failure on the EU AI Act Art. 50 disclosure. It is not real.
  Composite the whole ancestor stack onto the page base and the same text measures **12.42:1**.

  So walk every ancestor, keep each `backgroundColor` whose alpha is above zero, and blend them
  back-to-front onto the page background before computing luminance:

      over = (fg, bg) => fg.rgb.map((v, i) => v * fg.a + bg[i] * (1 - fg.a))

  Skip elements with children (you would measure a container, not a text run) and anything whose
  `offsetParent` is null. Round 22 measured **zero** genuine 1.4.3 failures on the use-case page this
  way, after the naive version reported two.

- **2.1.1 keyboard**: tab through every route. Every control reachable, every disclosure operable
  with Enter/Space, no trap in the parameter panel.
- **2.4.7 focus visible** and **2.4.11 focus not obscured**: the ring must survive the dark theme and
  not be hidden behind a sticky header.
- **4.1.2 name, role, value**: every control has an accessible name. Audit with

      [...document.querySelectorAll('button,select,input')]
        .filter(e => !(e.getAttribute('aria-label') || e.labels?.length || e.textContent?.trim()))

  and expect that list to be empty. The `aria-expanded` / `aria-controls` pairing on the Parameters
  disclosure is already correct and is the pattern to match.

Record the measured ratio or the failing selector, never a tick. An accessibility row that says
"pass" without a number is the same unverified claim this checklist exists to prevent.

## Enabling WebGPU and WebNN, and what to expect when you do

The checklist mandates a two-browser comparison and never said how to reach one, so `chrome://`
appeared nowhere in it and the flags lived only in the README. A reader working this document could
not get to the state it requires.

    chrome://flags/#enable-unsafe-webgpu              WebGPU
    chrome://flags/#enable-vulkan                     also required on Linux
    chrome://flags/#web-machine-learning-neural-network   WebNN
    # or, as launch flags:
    --enable-unsafe-webgpu --enable-features=Vulkan,WebMachineLearningNeuralNetwork

Then confirm what you actually got, because enabling the flag is not the same as having the hardware:

    const a = await navigator.gpu?.requestAdapter();
    ({ adapter: !!a, info: a?.info, shaderF16: a?.features?.has?.("shader-f16"), webnn: !!navigator.ml })

**Expect the honest outcome on a host with no GPU.** The flag succeeds, an adapter is returned, and
it is SwiftShader - a CPU rasteriser. `hasWebGPU()` declines it by name, because it measured about
300x slower than the wasm build it would displace (ResNet-50: 12842 ms against 42 ms). So "WebGPU
enabled" and "WebGPU used" are different facts and the run reports the second. WebNN is enabled and
deliberately not offered to ORT on this project's own measurement of 162.7 s against 5.1 s.

Record which of the three states you are in - no adapter, software adapter, hardware adapter - beside
every latency, because it moves them by two orders of magnitude.

## Submodules: check they are still maintained, and that the licence is written down

A submodule is a dependency whose licence and maintenance status travel with the repository, and
nothing was checking either. Round 22 found both of this project's submodules absent from
`DATA_LICENSES.md` entirely.

Per round, per submodule, against the GitHub API rather than by assumption:

    git submodule status                     # what we pin
    gh api repos/<owner>/<repo> \
      --jq '"\(.pushed_at) \(.license.spdx_id) archived=\(.archived)"'
    gh api repos/<owner>/<repo>/commits?per_page=1 \
      --jq '"\(.[0].sha[0:12]) \(.[0].commit.committer.date)"'   # upstream HEAD

Use `gh api`, not bare `curl`. The unauthenticated API allows 60 requests an hour per IP, and it
does not fail loudly - it returns HTTP 200 with a JSON body containing `message: API rate limit
exceeded`, so a script that indexes straight into the response dies on a `KeyError` for a field
that looks missing. That reads as "the repository has no licence" rather than "you were throttled".

Four questions, and the third is the one that gets skipped:

- Is the licence permissive, and is it recorded in `DATA_LICENSES.md`?
- Is the upstream archived?
- **Is the pin the current head of the branch we track, or have we drifted?**
- Is "actively maintained" true of the branch we track, or only of the repository? These differ:
  hpmpc's last push is 2026-07-25 while its default branch has not moved since 2026-03-02, so the
  activity is on other branches. Record the distinction rather than the flattering half.

Act when an upstream is archived, a licence changes, or the tracked branch has not moved in a year.

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

**Found and fixed this round:** the diagram ticked a zero-key BB84 step. See the panel-consistency
section above.

One caution for the next round, learned twice here: a synthetic `.click()` on a React control does
not update `aria-expanded` within the same tick, and reading it immediately makes a working button
look inert. Read after a frame.

## What this checklist cannot cover

- One browser, one adapter. Every latency here is from a software WebGPU adapter; a machine with a
  discrete GPU will differ, and nothing in this repo measures that.
- The health of a corpus. The checklist verifies that the number shown is the number measured, not
  that the corpus behind it is representative.
- Anything the page does not render. A metric with no reader passes every axis here, which is why
  the audit that finds those reads the code instead.
