# Browser verification checklist

`deliverable-verification.md` says what "verified" means. This is the executable form of its gate 4,
the one that runs in Chrome against the deployed page.

It exists because gate 4 was being reconstructed from memory every round, and what gets reconstructed
is what someone remembers to look at. Round 20 audited controls on use-case pages and found five
inert ones; the pages it did not sweep kept theirs. This file is the sweep, written down.

## Topic index - the same concern, in more than one place

This file DOES contain duplicated text, and an earlier version of this line claimed it did not -
"zero repeated check-rows and zero repeated sentences". That was false and load-bearing, because
it was the stated reason for indexing rather than merging. An audit found nine duplicated
passages, eleven ID pairs stating the same check twice, and seven incidents narrated three or four
times. Check-row IDs are unique (149 of 149); sentences are not.
What it does have is **scattering** - one concern addressed in several sections, each written when a
different incident exposed a different face of it. That is deliberate, because each section carries
the incident that produced its rows and that context is what stops a row being deleted later as
noise. It is also the thing that makes the file hard to work top-to-bottom.

So the fix is an index rather than a merge. If you are chasing one concern, start here; if you are
running a round, use the Run sheet below instead.

| Concern | Sections | Where |
|---|---|---|
| **Deploy, CI, and what the browser is actually served** | 10 | [Run sheet](#run-sheet---the-whole-round-in-one-page) . [How to work it](#how-to-work-it) . [The layout axis](#the-layout-axis-in-full) . [Reproduce each panel's headline from...](#reproduce-each-panels-headline-from-what-the-page-itself-shows) . [Check the MODEL](#check-the-model-not-only-the-number) . [Anything outside web/ and src/ still...](#anything-outside-web-and-src-still-ships-and-nothing-was-checking-it) . [Local-green is not CI-green](#local-green-is-not-ci-green-and-the-deploy-will-not-tell-you) . [Bootstrap a claimed improvement before...](#bootstrap-a-claimed-improvement-before-shipping-it) . [The deploy step itself needs verifying](#the-deploy-step-itself-needs-verifying-and-a-failed-build-looks-like-a-successful-one) . [An investigation is not landed until...](#an-investigation-is-not-landed-until-the-string-the-ui-renders-says-so) |
| **Is the number measuring what it claims to** | 4 | [A shuffled DataLoader is correct for...](#a-shuffled-dataloader-is-correct-for-training-and-silently-wrong-for-harvesting) . [Bootstrap a claimed improvement before...](#bootstrap-a-claimed-improvement-before-shipping-it) . [A model can score respectably and have...](#a-model-can-score-respectably-and-have-learned-nothing---check-that-it-predicts-more-than-one-class) . [Ask what the metric was divided by](#ask-what-the-metric-was-divided-by-not-just-how-many-samples-there-were) |
| **Publication, the Hub, and what may be redistributed** | 8 | [Run sheet](#run-sheet---the-whole-round-in-one-page) . [Routes](#routes) . [Retraining a published model is a...](#retraining-a-published-model-is-a-five-step-cycle-and-two-of-the-steps-bite) . [PQC library conformance](#pqc-library-conformance-run-nists-negative-vectors-not-just-the-happy-path) . [Submodules](#submodules-check-they-are-still-maintained-and-that-the-licence-is-written-down) . [Navigate the way a visitor does](#navigate-the-way-a-visitor-does-not-the-way-a-test-does) . [A default that contradicts a licence...](#a-default-that-contradicts-a-licence-commitment-will-eventually-be-taken) . [Retired and orphaned Hugging Face repos](#retired-and-orphaned-hugging-face-repos) |
| **Claims about the outside world** | 4 | [The nine axes](#the-nine-axes) . [Search outward](#search-outward-not-just-inward---and-fetch-the-source-never-the-summary) . [Reproduce each panel's headline from...](#reproduce-each-panels-headline-from-what-the-page-itself-shows) . [Every control](#every-control-against-a-published-reference-value) |
| **Coverage of the page itself** | 7 | [Run sheet](#run-sheet---the-whole-round-in-one-page) . [Routes](#routes) . [Regions on a use-case page](#regions-on-a-use-case-page) . [Every panel must describe the run that...](#every-panel-must-describe-the-run-that-just-happened) . [Simulator controls](#simulator-controls-and-where-each-one-has-to-show-up) . [Every control](#every-control-against-a-published-reference-value) . [The UX axis](#the-ux-axis-and-the-accessibility-half-nobody-was-checking) |

**Every check row has a unique ID.** The prefix says which section it belongs to: `RS` run sheet,
`RG` page region, `CI` continuous integration, `DP` deploy, `DN` denominator/sample size, `HF`
Hugging Face, `B` baseline row, `C` constant predictor, `L` licence default, `N` landed-in-the-UI,
`R` bootstrap, `S` sampler alignment, `E` eval-sample staleness, `G` GPU/accelerator claims, `H`
HE context, `LK` label leak, `M` use-case motivation, `P` published-model parity, `QC` QCNN
inference, `RD` real-data overlay, `T` training corpus, `TR` training environment, `V` vendored
crypto. Two collisions were fixed in Round 24e: `DP` and `DN` were both
`D`, and the run sheet and the region table both used bare integers, so "step 3" and "region 3" and
"check D3" were three different things sharing two names.

## Run sheet - the whole round in one page

This document is about 2400 lines because each row carries the incident that produced it, and that
context
is what stops a row being deleted later as noise. But a checklist you cannot execute in one sitting
has stopped being a checklist, so this is the executable index. Work down it; follow a link only when
a row fails or you do not recognise it.

**Off the browser first** - these are commands, and they either pass or they do not:

    .venv/bin/python -m pytest -q -m "not slow"        # 208 expected (measured 2026-08-10)
    cd web && npx tsc --noEmit && npm run build
    cd web && npm run check:pqc                        # FIPS 203 s7.2 + FIPS 204 sizes/round-trip
    .venv/bin/python scripts/export_config.py          # if the registry or config changed
    git ls-files '*.md' '*.ts' '*.tsx' '*.py' | xargs grep -lP '[\x{3040}-\x{9faf}]'   # expect empty

**Then the sweep, in order.** Each line names what to do and where the detail lives.

| # | Do this | Detail |
|---|---|---|
| RS1 | Deploy from git, grep the served bundle for a new string literal, **then** cache-bust the tab | [How to work it](#how-to-work-it) |
| RS2 | Walk all 7 routes plus the two negative paths | [Routes](#routes) |
| RS3 | On each use-case page, walk the 14 regions in DOM order | [Regions on a use-case page](#regions-on-a-use-case-page) |
| RS4 | Open AND close every disclosure; check all six states | [The open and close axis](#the-open-and-close-axis) |
| RS5 | Run the transport sequence: Start, Pause, Step, Resume, Restart, Stop | [Buttons, by group](#buttons-by-group) |
| RS6 | Every select, every option; every checkbox, both states | [Selects, every value](#selects-every-value) |
| RS7 | Capture every control's value against its OWN min/max | [Capture every control's value](#capture-every-controls-value-against-its-own-constraints) |
| RS8 | Set each control, re-run, confirm the result MOVES and matches its reference value | [Every control, against a published reference value](#every-control-against-a-published-reference-value) |
| RS9 | After a parameter change, grep the page for the OLD value, not just the new one | [Every panel must describe the run that just happened](#every-panel-must-describe-the-run-that-just-happened) |
| RS10 | Recompute each panel headline from the inputs printed beside it | [Reproduce each panel's headline](#reproduce-each-panels-headline-from-what-the-page-itself-shows) |
| RS11 | Check each panel against its cited standard | [Every panel, against its cited standard](#every-panel-against-its-cited-standard) |
| RS12 | Check every XAI attribution against its own physics, not only its axioms | [XAI covers all five use cases](#xai-now-covers-all-five-use-cases---check-each-against-its-own-physics) |
| RS13 | Read diagram labels and tables against each other - a tick is not a verdict | [A completion marker is not a verdict](#a-completion-marker-is-not-a-verdict) |
| RS14 | Layout at 1280 and mobile; no horizontal overflow, no overlap | [The layout axis, in full](#the-layout-axis-in-full) |
| RS15 | UX and WCAG 2.2 AA - composite the alpha before judging contrast | [The UX axis](#the-ux-axis-and-the-accessibility-half-nobody-was-checking) |
| RS16 | Console clean on the CURRENT bundle - old errors persist in history | [During a run, not only after it](#during-a-run-not-only-after-it) |

**Then the checks that reach outside the browser:**

| # | Do this | Detail |
|---|---|---|
| RS17 | Source-code pass: hardcoding, dead guards, fallbacks the browser cannot show | [The source-code pass](#the-source-code-pass-hardcoding-and-fallbacks-the-browser-cannot-show-you) |
| RS18 | Every declared HF repo resolves; every artifact dir exists; every card matches metrics.json | [The model registry claims things about the outside world](#the-model-registry-claims-things-about-the-outside-world---check-them) |
| RS19 | `models/`, `monitoring/`, `spaces/` - they ship too, and nothing else checks them | [Anything outside web/ and src/](#anything-outside-web-and-src-still-ships-and-nothing-was-checking-it) |
| RS20 | Re-measure every pinned dependency, and watch which "upstream" a sentence means | [Dependency claims decay](#dependency-claims-decay-so-re-measure-them-and-watch-the-wording) |
| RS21 | Submodules: pin vs upstream HEAD, licence, archived, branch drift. Use `gh api` | [Submodules](#submodules-check-they-are-still-maintained-and-that-the-licence-is-written-down) |
| RS22 | Search OUTWARD - what changed outside that the project has not noticed | [Search outward, not just inward](#search-outward-not-just-inward---and-fetch-the-source-never-the-summary) |
| RS23 | Retired a metric? Grep its name AND a characteristic value across every doc | [Retiring a metric is not done until the checklist stops asking](#retiring-a-metric-is-not-done-until-the-checklist-stops-asking-for-it) |
| RS24 | Re-check the dated items - they expire whether or not anyone looks | see below |
| RS25 | Any threshold constant: is it the LOSS-FREE idealisation of a loss-dependent bound? | see below |

**Dated checks, because a deadline does not wait for a convenient round:**

    curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/spaces/Daemons-Q/qsh-inference
    # 401 -> still blocked, condition stands.  200 -> decide NOW: wire the Space, or delete it.
    #        (unlock was due ~2026-09-01). The `remote` ProvenanceBadge tier this used to say to
    #        delete alongside it was ALREADY removed in Round 24d - ProvenanceBadge.tsx now exports
    #        only measured-webgpu | measured-wasm | measured-js | simulated. Half of this
    #        instruction had already been carried out and the note had not noticed.

EU AI Act Art. 50(2), machine-readable marking of synthetic text: **CLOSED as not-applicable**,
not as done. The Agentic AI panel emits 96 tokens (Transformers.js) or 160 (WebLLM), both under the
Code of Practice's 200-token "very short text" exception, so no marking layer is owed and the
C2PA-versus-in-band choice never opens. What stays live is the trip-wire, not the deadline - re-check
if the panel exceeds **200 tokens**, if generated text ships inside a file container, or if C2PA text
bindings become implementable (spec 2.4 defines them; no conforming product implements them yet).
Workings in [`ai-act-art50-gap-analysis.md`](ai-act-art50-gap-analysis.md).

Record what you observed in [`verification-log.md`](verification-log.md), with the date and the
bundle hash.

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

In DOM order. Region 10 (SHAP attribution) used to be absent on non-inference use cases and this
line said to record that; since Round 24 **all five render one**, so an absence is now a defect
rather than an expected not-applicable. Record the value rather than leaving it
blank.

| # | Region | Component |
|---|---|---|
| RG1 | Title, blurb, industry and market pills | UseCasePage |
| RG2 | Demo banner | demoMode |
| RG3 | Crypto selects, published-model select, OptiQKD and MCP checkboxes, Parameters button | PipelineControls |
| RG4 | Parameter fields, Reset to defaults | ParameterPanel |
| RG5 | Two-party simulator diagram | TwoPartyDiagram |
| RG6 | Four export buttons | UseCasePage |
| RG7 | Event log, metrics table, signature | RunMonitor |
| RG8 | Inference provenance badge and its disclosure | ProvenanceBadge |
| RG9 | Per-model rows, Run buttons, result table | ModelPanel |
| RG10 | SHAP attribution | ShapPanel |
| RG11 | Benchmark rows and cost split | BenchmarkPanel |
| RG12 | Engine, model and combiner selects, Run | AgenticAIPanel |
| RG13 | Key budget, span, relay caveat | SdnControlPlanePanel |
| RG14 | Footer state note | UseCasePage |

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

## Search outward, not just inward - and fetch the source, never the summary

Every other section here asks whether what the project already says is still true. This one asks the
opposite question: **what changed outside that the project has not noticed?** It is the axis that
produced most of Rounds 22-23's real findings, and none of them would have surfaced from reading this
repository.

| Found by searching outward | What it was |
|---|---|
| `@noble/post-quantum` 0.2.1 | accepted **all 2,595** malformed ML-KEM keys; FIPS 203 section 7.2 check was a no-op |
| ePrint 2026/203 | CPAD **impossibility** for the class covering CKKS; the threat model said nothing about the notion |
| EU AI Act Art. 50(2) | **found** as an apparently unmet obligation with a 2 December 2026 deadline - later **CLOSED as not-applicable** (the panel emits 96-160 tokens, under the Code of Practice's 200-token "very short text" exception). Kept in this table because the *search* is what the row illustrates; the obligation itself is not open |
| Microsoft SEAL 4.4.3 | six releases past the bundled 4.1.2, one self-described "major security update" |
| QERS, CARS, and three vendor sources | candidate metrics that would have been adopted on the strength of a search summary |

### The discipline, in the order that matters

1. **Fetch the primary source. Do not act on a search summary.** This is the whole rule. A summary
   of QERS reads as a calibrated framework; the paper says the weights are author-selected across
   three scenarios with no calibration methodology, and normalises against an ESP32 Wi-Fi testbed.
   A summary of the ACVP vectors implies they test the modulus check; the ML-KEM-512 negatives are
   1216-byte keys against a correct 800, i.e. length checks, and a library can score 10/10 there
   while failing every modulus vector.
2. **Ask what the source actually prescribes, then whether it is implementable here.** ISO/IEC
   23837-2 endorses discretised parameter sweeps and then states it "does not specify values for the
   thresholds"; 21 of its 24 activities need an optical bench. Endorsing a method is not supplying
   one.
3. **Prefer a reference value you can assert in a test.** A source is most useful when it yields a
   number this repository can be checked against - Ma et al.'s decoy curves became 13 assertions,
   NIST and C2SP vectors became `npm run check:pqc`. A source that yields only prose yields only
   prose.
4. **Decline in writing, with the reason.** Adopting a framework with invented weights because it
   has a citation is worse than having no framework, because the citation implies a rigour that did
   not transfer. `references/security-scoring-fact-check.md` and
   `references/stress-testing-fact-check.md` exist so the same five sources are not re-evaluated
   every round.
5. **Label vendor material as vendor material.** Sponsored articles, product pages and project blogs
   surface high in searches and carry no methodology. Recorded separately in both fact-check files.

### What to sweep each round

- The standards the product cites by number - editions change, and an edition that has NOT changed is
  itself a finding worth recording (ETSI GS QKD 014 is still V1.1.1, 2019-02).
- Regulatory dates the UI asserts, and any obligation the product has not yet met.
- Every pinned crypto/FHE dependency, plus its upstream - see the dependency section below.
- Attacks against the primitives in use. CKKS, ML-KEM and QKD all have active attack literature, and
  an impossibility result against a class the project belongs to is the kind of thing that never
  appears in a release note.

### One caution

A claim being new is not a reason to adopt it. A 2026 preprint is a preprint. Weigh venue, whether
the numbers are reproducible, and whether the paper's setting is this project's setting - QERS was
scoped to "resource-constrained IoT/IIoT" and said applicability beyond that "remains unaddressed",
which is disqualifying regardless of how recent it is.

## Reproduce each panel's headline from what the page itself shows

The fact-check above asks whether an external claim is still true. This asks something stricter and
entirely local: **can a reader recompute the number from the inputs printed beside it and the formula
in `references/`?** A figure that cannot be recomputed from the page is an assertion, however well
sourced.

Two things have to hold. The panel prints its own inputs, and the formula is traceable to a citation
or to config rather than to a literal in the component.

| Panel | Headline | Recompute from | Where the formula lives |
|---|---|---|---|
| Security posture and quantum advantage | advantage `n/10`, plus `raw` when clamped | the ratios the card prints | `quantum_advantage` weights in config; mirror of `benchmark/quantum_advantage.py` |
| Security posture table | per-row facts, no composite | FIPS 203/204 sizes; the run's `secure_key_bits`, `skr`, `qber` | `securityFacts` in `clientCompute.ts`; each row names its own source |
| Why - top features (SHAP) | per-feature phi | `W`, `b`, `x`, baseline | exact linear SHAP, `phi_i = w_eff_i (x_i - baseline_i)`, `w_eff = W[1]-W[0]`; mirror of `ai/explainability.py` |
| Benchmark - MPC-FHE vs QSMPC-FHE | latency split | FHE-linear + MPC-non-linear ms | measured, so check the parts sum to the whole |
| QKD control plane, GS QKD 015 | key budget, span | `secure_key_bits`, distance | field names must match the spec's own (`qkdn_id`, `qkdl_id`, `skr`, `eskr`) |
| Quantum-kernel SVC + variational QNN | macro F1 | published per-class scores | `registry.py` metric spec + its baseline citation |
| Inference provenance | badge state | the backend the run reported | the FOUR-state enum in `ProvenanceBadge` - `measured-webgpu \| measured-wasm \| measured-js \| simulated`. The `remote` tier was removed in Round 24d; two rows here said five for several rounds afterwards |
| Agentic AI, MCP | derived-key fingerprint | combiner choice | changes with the combiner - verified, not assumed |

Worked example, run in Round 21. The advantage card printed
`communication x6145.8 . latency x3608.79 . security x1.5` beside a score of `10/10`. Taking the
`quantum_advantage` weights from config rather than retyping them - `w_comm 0.6`, `w_latency 0.2`,
`w_security 0.2`, `log_scale 3.2`, `log_offset 1.0` - gives
`6145.8^0.6 x 3608.79^0.2 x 1.5^0.2`, then `log10(composite + 1) x 3.2 + 1.0`, which lands on 10 and
matches. The score is checkable in place by anyone reading the page.

That method - reproduce the number from its printed inputs - is what retired the 0-100 security
score. Recomputing it showed `68 -> 63` and `96 -> 91` held only in the BROWSER, because the Python
quantum row pinned `qkd_key_bits=256` and never moved; then the same method showed that 16 of the 28
points separating 68 from 96 came from which arguments two call sites happened to supply. The score
is gone, but keep the habit: **reproducing a number is how you discover which deployment it
describes, and sometimes that it describes nothing.**

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
- **A column that renders a different column** (told in full under "The nine axes" above, with
  the consequence). The detailed-results table was headed `model` and
  rendered `security_model`, so it had no model column while appearing to have one.
- **Says the same thing twice, differently.** Two panels offering the same ResNet-18 run with
  different chrome; two quantum-advantage scores on one page.

Panels to read, per use case, after a run: Inference provenance . Models (including each "Run this
model in your browser" result) . Why - top features (SHAP) . Benchmark - MPC-FHE vs QSMPC-FHE .
Detailed results . Security posture and quantum advantage . Agentic AI . QKD control plane.

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
| KEM | handshake arrow names ML-KEM-768 or 1024 | handshake line | `kem_category` 3 or 5, `kem_ciphertext_bytes` 1088 or 1568; the posture table's NIST category row moves |
| QKD (family) | quantum arrow present, or gone under pqc-only | "QKD disabled (PQC-only profile)" | `secure_key_bits`; the posture table's QKD row reports none under pqc-only |
| Protocol (bb84 / e91) | arrow reads BB84 decoy-state or E91 (entanglement / CHSH) | protocol line | `chsh_s` appears only for E91 |
| Signature | audit-record arrow names ML-DSA-65 or 87 | signing line | `signature_bytes` 3309 or 4627 |
| Published model | full-width step names model, revision, EP, input shape, latency, outcome | "published model X ran here on Y" | `published_*` metrics, plaintext benchmark row |
| Optimize QKD (OptiQKD) | OptiQKD suffix on the QKD arrow with the measured percentage | improvement and convergence lines | `qkd_skr_improvement_pct`, `qkd_convergence_evals` |
| OptiQKD samples | (via the suffix) | budget line | collapses in 8-wide buckets; saturates near 400 |
| QKD channel (pulses, distance, eve, misalignment) | QBER on the arrow | channel line | `qber`, `skr`, `secure_key_bits` |
| Free-space channel (downlink km, sat eve, sat pulses) | free-space arrow QBER, and the QSDC arrow must show the ABORT when it fires | "QSDC NOT AVAILABLE on this link - secrecy capacity ... The limit is the link budget, not an attack" (loss-limited, which is the default state) or "QSDC ABORTED: secrecy capacity ... the loss is attributable" (Eve-limited) | `qsdc_error_rate`, `qsdc_secrecy_capacity`, `qsdc_eve_gain_g`, `qsdc_link_budget_limited`, `qsdc_eavesdrop_detected` |
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
`QSDC direct telemetry . secrecy 0 bit/carrier` with a done tick - delivered-but-not-secret, the
opposite of what happened, on the feature the page leads with. Observed after the fix:

    diagram   QSDC ABORTED . error 0.16 over threshold . telemetry withheld
    log       QSDC ABORTED: error 0.158 > 0.11 - telemetry withheld
    metrics   qsdc_link_budget_limited 1 . qsdc_eavesdrop_detected 0 . qsdc_secrecy_capacity 0
    #         NOT `qsdc_eavesdrop_detected 1`. This block previously said 1, which is the
    #         behaviour Round 24m identified as the DEFECT: the page reported an eavesdropper on
    #         a clean channel on every run. See the free-space section below - this link is
    #         loss-limited at every setting, so the abort is never attributable to Eve.

The same run also showed why a step must not be gated on something it does not depend on. The
published-model step sat inside the delivery branch, so an abort left it reading
`published model . -` with a done tick and no reason - even though that step runs the pinned model on
a row of the model's OWN corpus and has nothing to do with this run's channel. It now runs either
way: `published satellite_comms.telemanom @ 4507c187d0f9 . onnx-wasm . 1x250x6 . 9.59 ms .
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
shows nothing in its place. **Three** live `except Exception: pass` sites remain, not two, and the third is not cosmetic:
`usecases/medical_fl/__init__.py:213` (cosmetic SHAP feature names - this was recorded as :168, which
is a different `except` returning 1), `benchmark/gpu_mpc_adapter.py:57` (an optional GPU subprocess
whose absence is handled two lines later), and the one that was never recorded,
`config/__init__.py:66` - `pass  # malformed config => fall back to defaults`. That last one
swallows a malformed `config.yaml` and silently returns defaults, which would make the
config-to-bundle procedure prescribed in this same section report "the page did not move" when the
real cause is that the config never parsed.

**Then confirm in the browser**, because a config value can reach the bundle and still not be read -
which is precisely the bug above. Capture each use-case page, change the parameter, capture again,
and diff the diagram text, the log and the metrics table. A value that reaches the bundle and moves
nothing on screen is the same defect as a control that renders and changes nothing.

Two points, not one, and predict the second before running it. The free-space fix was confirmed by
`800 km -> 28 dB` and `2000 km -> 70 dB`, both exactly `distance x 0.035`, with the diagram arrow
reading `free-space BB84 decoy-state (28 dB) . QBER 0.04`. One point cannot distinguish "reads the
config" from "happens to agree with the literal it replaced" - which is the failure mode this whole
axis exists for.

**Assert the field took before reading the result.** These are React-controlled inputs: setting
`.value` and dispatching `input` is not synchronous with the state the next run will read, and a
prior script that errored mid-flight can leave a value behind. This produced a false alarm here - a
run reported 70 dB where 28 was expected, which looks exactly like a broken fix, and the field was
still holding 2000 from an earlier aborted call. Read `el.value` back and throw if it is not what you
set, before clicking Start. The arithmetic exonerated the code of MISREADING the config: 70 is
2000 x 0.035, so the value was being read correctly. It did NOT establish that the formula was
right, and it was not - `distance x 0.035` is a FIBRE attenuation law and was replaced by the
diffraction model. See "Check the MODEL, not only the number" below; the numbers in this
paragraph are superseded (2000 km is 32.4 dB, not 70). The
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
| Free-space link budget | **SUPERSEDED** - was `loss_dB = distance x loss_per_km`, a fibre law. Now diffraction: geometric + atmospheric + pointing | config-sourced beam divergence / aperture / atmospheric / pointing | 800 km gives **24.5 dB** (was 28), 2000 km gives **32.4 dB** (was 70). The old row's "both exact" was exact arithmetic on the wrong formula |
| Linear SHAP | `phi_i = w_eff_i (x_i - base_i)`, `w_eff = W1 - W0` | same | completeness, dummy and linearity all hold over 2000 random linear models, re-derived in Python |
| Decoy-state reference model | `qkd/decoy_reference.py` | `taggedFraction` in `quantumSim.ts` | same GLLP definition; the Python side additionally asserts published values from Ma et al. PRA 72, 012326 |

**The parity check is the strongest kind available here**, because the browser mirror and the Python
original were written separately and can be run against each other. This table used to carry a
security-score row as its headline example, on the grounds that two independent implementations both
produced 68 and 96. (Told in full, with both external assertions named, under "Retiring a metric is
not done until the checklist stops asking" below - this is the short version.) They did - and the
number was still meaningless, because both implementations
faithfully reproduced the same argument-passing artefact. **Agreement between two mirrors proves they
match each other, not that either is right.** Pair every parity check with at least one assertion
against a source outside this repository; `tests/test_decoy_reference.py` and `npm run check:pqc` are
the two that do.

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
tests, none of whose numbers originate in this repository, drawn from Ma et al. PRA 72, 012326
(2005) with the GYS parameter set and GLLP QIC 4, 325 (2004). Those anchors are NOT restated here.
They are executable assertions in that test file and a table in `deliverable-verification.md`; a
third handwritten copy in this document would be the copy that silently goes stale, and none of them
is browser-checkable anyway - `npm run` cannot reach them, `pytest` can.

What IS browser-checkable is the tagged fraction the posture table prints, above.

**The trap in reproducing those figures, and it is a quiet one.** The published curves are plotted
for the efficient BB84 of Lo-Chau-Ardehali, where the sifting factor q tends to 1. This project runs
standard BB84 with q = 1/2. Reproduce the paper with q = 1/2 and every value comes out at exactly
half the published one - 0.0568 against 0.11, 0.0755 against 0.15. That is close enough to look like
a fitting or units discrepancy and wrong enough to make the comparison meaningless. `q_sift` is an
explicit field for this reason, and a test asserts the factor of two so it cannot drift.

### Free-space channel (satellite: `pp-sat_distance_km`, `pp-sat_eve`, `pp-sat_pulses`)

Loss must rise with distance and the reported dB must appear in the diagram label. The abort keeps
its tick, correctly - aborting IS the protocol working, and the label says so unambiguously.

**`pp-qsdc_threshold` is gone, and the open question about it is closed.** It defaulted to 0.11, the
loss-free root of the secrecy-capacity expression, which is why it coincided exactly with
Shor-Preskill and looked derived. The browser was corrected to abort on `Cs = 1 - h(e) - g h(e)`
with `g = 10^(loss_dB/10)`, at which point **nothing read `config.qsdc_threshold` any more and the
control became inert** - it sat in ParameterPanel moving no surface for several rounds. It has been
removed; under the corrected criterion there is no QBER threshold to set. Round 24m also finished
the other half: `src/qsh/qkd/qsdc.py` had kept the old fixed-QBER criterion the whole time, so the
Python and browser paths disagreed about whether the same link delivers.

**What to expect now, and it is not a bug:** this downlink is loss-limited at EVERY setting.
Measured on the deployed page, secrecy capacity -51.58 at 24.5 dB with g=280.4 and no eavesdropper.
(The loss table below gives g=281.8 at "24.5 dB". Both are right: 280.43 is g at the ACTUAL
24.4782 dB the run computes, 281.84 is g at the DISPLAYED, rounded 24.5. A rounded input carried
into an exponential is worth labelling rather than reconciling.)
The fixed 5 dB atmospheric + 3 dB pointing budget alone gives g = 6.31, where `Cs > 0` needs QBER
below ~0.019 against a 0.025 misalignment floor - so `sat_distance_km` cannot produce a delivery at
any position. Check that the log says NOT AVAILABLE / link budget, and that
`qsdc_eavesdrop_detected` is **0** while `qsdc_link_budget_limited` is 1. The old behaviour reported
an eavesdropper on a clean channel on every single run.

### Every panel that restates a run parameter must restate the RUN's value

A parameter can be wired into the pipeline and still be wrong on screen, because a second panel may
be reading the CONFIG DEFAULT for the same quantity. Both of this project's instances of that were
found by setting a parameter away from its default and then searching the whole page for the old
number.

Measured on satellite, Downlink 800 -> 2000 km: the QKD stage, the channel loss and the diagram all
moved, while the SDN control-plane panel kept printing "a single 800 km span". Two panels, one run,
two geometries. Now 2000 km in both.

So after changing any parameter, grep the rendered page for the DEFAULT value, not just for the new
one:

    document.body.innerText.match(/\d+(\.\d+)?\s*km span[^\n]{0,60}/g)

Note the sequencing that makes this class hard to see: a previous fix had already corrected this
panel from reading the FIBRE distance to reading the free-space one, which was right and which is
why the line looked reviewed. Wrong-channel and wrong-value are separate defects, and fixing the
first does not fix the second.

### Published model (`sel-published-model`)

Each option must produce a run whose diagram names the model AND its commit hash, so the artefact is
identifiable. Measured: "published finance_aml.gbm @ c46157c27e64 - onnx-wasm - 1x32 - 0.34 ms -
class 0". Check the hash against the registry, and check that selecting a different option changes
it. Note that telecom and genomics ship no selector, but for DIFFERENT reasons, and the reason matters:
telecom publishes exactly one model, while **genomics publishes none at all**. `genomics.prs` carries
`hf_repo=None` (registry.py, "publication blocked pending the per-score licence audit"), and
`onnxInference.ts` drops any spec without an `hf_repo_id`, so `browserModels("genomics")` returns an
empty list - which is also why the in-run verification chart never fires there and the ancestry
strata chart exists instead. A verifier told the reason is "publishes a single model" will confirm
the wrong fact about genomics. `PipelineControls.tsx` states it correctly in a comment: "telecom
publishes one, genomics none".

### OptiQKD (`sel-optiqkd`) and the key-mixing checkbox

Ticking OptiQKD must emit `qkd_skr_improvement_pct` and a convergence count, and the diagram arrow
must carry the measured improvement rather than merely "OptiQKD on". Unticked, neither appears.
The "mix QKD-domain key" checkbox controls whether QKD material enters the session key at all;
unticking it must drive the posture table's QKD row to zero the same way `pqc-only` does.

## XAI now covers all five use cases - check each against its own physics

Until Round 24 only finance, medical and genomics had an attribution. Satellite computed one in
Python that nothing rendered; telecom had none anywhere. All five now render one, and the method is
the same everywhere - exact linear SHAP over a genuinely linear model - because in every case a
linear model was already there. **No new attribution method was invented for any of them**, which is
the bar to hold if this is extended again.

| Use case | Model explained | Reference point | Check the top feature against |
|---|---|---|---|
| finance_aml | bundled PaySim-derived LR | E[x] over the shipped rows | plausibility only |
| medical_fl | bundled Wisconsin LR | E[x] - note the features are z-scored, so E[x] IS 0 here | plausibility only |
| genomics | real PGS000001 effect weights | population mean dosage | the SNP should be a known locus for the trait |
| satellite_comms | bundled NASA telemetry LR (2xN margin) | mean of the bundled windows | the driving channel and timestep |
| telecom_ids | the in-browser 8-class softmax, one-vs-rest | zero vector, because clientIDS z-scores | **the attack physics** |

**Telecom is the one that can be checked hardest, so check it.** The classifier names an attack, and
the attribution says which telemetry drove that name. Measured 2026-08-09: predicted
`intercept_resend`, top contribution `qber_mean` **+4.692**, about 8.5x the next feature. That is
correct - intercept-resend is the attack that raises QBER, and this repository separately verifies
the textbook 25% figure against Gottesman-Lo. An attribution that named `sift_ratio` or
`dark_count_rate` as the dominant driver of an intercept-resend verdict would be wrong even if every
axiom held.

**Satellite is checkable the same way.** Measured: top contributions `E-6[t7]` -0.427, `E-6[t6]`,
`E-6[t5]` - clustered on one channel at the LATEST timesteps, which is what a windowed anomaly
detector should key on. Contributions negative, anomaly score 0.4656, verdict nominal: internally
coherent. Attributions spread evenly across all six channels and all eight timesteps would suggest
the model is keying on nothing.

**Do not let the panel conflate two models.** The bars explain the bundled linear model. The class
the page reports comes from the encrypted CKKS+MPC stage, which evaluates a *different* model - a
fixed demo pattern on finance and genomics, the published student on medical's convolution branch.
The panel said the bars described "the CKKS-linear model" until Round 24, which merged the two. When
reading the panel, confirm it still distinguishes them.

**The gate matters too.** `ShapPanel` was gated on `meta.inference`, a flag meaning "has a
classical-vs-quantum private-inference path" - a different question from "is there an attribution to
show". Satellite answered no to the first and yes to the second, so its explanation was computed and
discarded for six rounds. Gate on the artifact existing.

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
- The legacy 0-100 composite is **not rendered at all** - do not go looking for something to
  confirm is demoted. `BenchmarkPanel.tsx` prints "The 0-100 composite that used to sit here is
  retired and no longer rendered" and "There is no composite", and the detailed-results table renders
  a fixed column list that excludes it. `securityScore()` survives in `clientCompute.ts` only to
  build benchmark rows; no component reads it. What to check is that the composite is still ABSENT,
  which is the opposite of what this line used to ask for. It also contradicted this document's own
  "The score is gone" a few hundred lines earlier.

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

Names which model produced the reported class, from a FOUR-state enum (`remote` was removed in
Round 24d). With two inferences running
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

    BB84 decoy-state . QBER 0.1 yes

while the posture table on the same page, for the same run, reported **0 secure bits**, an SKR of
**0 per pulse**, and "no secure key at this error rate". One run, two panels, opposite conclusions -
and the reassuring one was the bigger and more looked-at.

So the check is: **after every run, read the diagram labels and the tables against each other**, not
each on its own.

    [...document.querySelectorAll('svg text')].map(t => t.textContent.trim()).filter(t => /yes|no/.test(t))

For each ticked step, find the number it sits beside and find where else the page reports the same
quantity. If the two disagree, the tick is the bug. Steps now declare an outcome (`ok`) separately
from completion, and a failed one is drawn with `no` in a warning colour - but the axis outlives this
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

## Retiring a metric is not done until the checklist stops asking for it

When a metric is removed, three kinds of place still refer to it, and they fail differently:

1. **The UI** - headings, legends, index pages. Caught in Round 22 by grepping the metric's NAME.
2. **Published result tables in the docs.** README carried the retired 0-100 score as a live
   **Security** column - `68 / 100` and `96 / 100` - for a full round after the metric was removed
   from the product. A reader takes a README table as current results.
3. **This checklist.** Six rows still told a verifier to recompute the score. That is the worst of
   the three, because a checklist is executed by a person: the row either wastes their time or
   produces a false failure against a panel that is behaving correctly.

So the retirement procedure is:

    git ls-files '*.md' | xargs grep -n "<metric name>"        # prose and tables
    git ls-files '*.md' | xargs grep -n "<a characteristic value>"   # e.g. 68 / 100
    grep -rn "<metric name>" web/src/ src/                     # headings, legends, computation

and every hit must end up in one of three states: **removed**, **rewritten to the replacement**, or
**explicitly marked historical**. A hit with none of those is a claim the project no longer stands
behind.

**Recompute the replacement's own numbers at the same time.** Retiring the score left the
quantum-advantage figures unexamined next to it, and one of them - "9.2 on the finance E2E run" -
was stale; the run reports 10/10, and 10 is a CLAMP over a raw 10.65, not a maximum.

### The parity trap this uncovered

The cross-implementation parity section used the security score as its strongest example: two
independently written implementations, Python and TypeScript, both produced 68 and 96. They did.
The number was still meaningless, because both faithfully reproduced the same argument-passing
artefact.

**Agreement between two mirrors proves they match each other, not that either is right.** Pair every
parity check with at least one assertion against a source OUTSIDE this repository. Two now qualify:
`tests/test_decoy_reference.py` asserts published values from Ma et al. PRA 72, 012326, and
`npm run check:pqc` asserts against NIST and C2SP vectors.

## Anything outside web/ and src/ still ships, and nothing was checking it

The browser checks cover `web/`. The verification gate covers `src/`. Three directories publish
artefacts and were audited for the first time in Round 23b: `models/`, `monitoring/`, `spaces/`.

**Assert model cards against `models/metrics.json`, not against memory.** All ten cards agreed when
checked properly - but note the trap in checking: a naive comparison that rounds
(`str(round(v, 4))`) reports false mismatches, because the cards carry full precision
(`0.08077786088257292`). Compare the repr and a range of fixed-point forms, and treat booleans as
flags rather than metrics, or `isinstance(v, (int, float))` will flag every `False` as missing.

**A hosted Space duplicates numbers and shapes that live in the registry, and it drifts.** Two real
divergences in `spaces/qsh-inference/app.py`:

- `"shape": [1, 26]` for the finance model, where the exported graph's input is `[N, 32]`. Verified
  with `onnx.load` against `models/finance_aml_gbm/onnx/model.onnx`. A 26-length vector is rejected
  by the session, so the Space's finance model would have failed on its first real request. **This
  is a runtime bug, not a documentation defect** - it survived because the Space is not yet hosted.
- `"metric": "AUC 0.888"` where `metrics.json` records **0.9147**.

So per round, for anything that restates a registry value outside the registry:

    python -c "import onnx,pathlib; m=onnx.load('models/<name>/onnx/model.onnx'); \
      print([[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim] for i in m.graph.input])"

and diff every hardcoded metric string against `models/metrics.json`. **A directory that is not
deployed yet is exactly where this rots**, because nothing exercises it.

## A constant that reproduces a familiar number is not thereby right

Round 24 defended `qsdc_threshold = 0.11` by deriving it: the wiretap secrecy capacity
`Cs = 1 - 2h(e)` has its root at 0.110028, identical to Shor-Preskill, so the two coincided by
identity rather than by transcription. The derivation was correct and the conclusion was wrong,
because `Cs = 1 - 2h(e)` is the **loss-free** case. The published criterion carries a loss term:

    Cs = Q_Bob [ 1 - h(e) - g h(e_chk) ],   g = Q_Eve / Q_Bob = 10^(loss_dB / 10)

| back-channel loss | g | threshold |
|---|---|---|
| 0 dB | 1.00 | 0.1100 |
| 4.1 dB (Qi et al. 2019, measured on fibre) | 2.57 | 0.0485 |
| 24.5 dB (this project's own free-space downlink) | 281.8 | **0.0003** |

Wrong by three orders of magnitude on the page that uses it, and wrong in the **permissive**
direction - it delivered over channels whose secrecy capacity was already zero.

**The check, for any threshold constant in this repository:** ask whether the published bound has a
parameter the constant silently fixes. Loss, block size, basis bias and detector efficiency are the
usual ones. A constant is safe only where that parameter is genuinely pinned; otherwise compute the
bound from the run's own measurements, and report the parameter alongside it so the number is
checkable rather than asserted.

**And the meta-lesson, which is why this section exists rather than a one-line fix:** I stopped
checking when the arithmetic reproduced the number already in the file. Agreement with the status
quo is the weakest possible evidence, because it is exactly what a wrong constant also produces.

## Retraining a published model is a five-step cycle, and two of the steps bite

Round 24 retrained `finance_aml.gbm` and nearly broke the demo twice. Both failures are silent, so
they belong here rather than in anyone's memory.

    1. train            .venv/bin/python scripts/train/finance_aml.py
    2. check the pin    metrics.json hf_revision is now null  <- THE FIRST TRAP
    3. publish          scripts/publish_hf.py --key <k> --public   <- THE SECOND TRAP
    4. re-export        .venv/bin/python scripts/export_config.py
    5. verify in browser: the dropdown still lists it, and the run names the NEW revision

**Trap 1: training clears `hf_revision`.** `browserModels()` requires BOTH `hf_repo_id` and
`hf_revision`; with the revision null the model silently drops out of the published-model dropdown
and stops running. Nothing errors - the control just disappears. Publishing writes the pin back.

**Trap 2: publishing without `--public` makes an already-public repo PRIVATE.** `publish_hf.py`
calls `update_repo_settings` explicitly, so the default (private) is applied to existing repos too.
The browser fetches over public URLs, so this takes the demo down while every local check still
passes. Confirm visibility before and after:

    curl -s https://huggingface.co/api/models/<org>/<repo> \
      | python3 -c "import sys,json;d=json.load(sys.stdin);print('private' if d.get('private') else 'PUBLIC', d.get('sha','')[:12])"

**Then verify the URL the browser will actually use**, not just that the repo exists - the resolve
endpoint 302s, so follow it:

    curl -sL -o /dev/null -w "%{http_code} %{size_download}\n" \
      "https://huggingface.co/<org>/<repo>/resolve/<revision>/onnx/model.onnx"

**And confirm the run names the new revision.** Measured 2026-08-09: `published finance_aml.gbm @
cf950f8c1e8d - onnx-wasm - 1x32 - 5.13 ms`. A stale revision here means the browser is running the
old weights while every local number describes the new ones.

**One licence check that must not be skipped**: `publish_hf.py` refuses a public push when the
dataset is not marked publishable, and excludes `fixtures.npy` by deny list. Read the excluded list
in the output rather than assuming - it is what keeps corpus rows off the Hub.

## The model registry claims things about the outside world - check them

`hf_repo` is not a label, it is an assertion that a repository exists and is reachable. The registry
documents its own convention on the field: **`None` means not published.** So a non-None value that
does not resolve is the registry stating a falsehood, and it is surfaced through `hf_repo_id` to both
the CLI and `frontend_config`.

Per round:

    # every declared repo must resolve
    python -c "from qsh.models.registry import _SPECS; \
      print('\n'.join(s.hf_repo_id for s in _SPECS if s.hf_repo))" \
      | while read r; do echo "$r $(curl -s -o /dev/null -w '%{http_code}' \
          https://huggingface.co/api/models/$r)"; done

    # every artifact dir must exist
    python -c "from qsh.models.registry import ARTIFACT_DIRS; import pathlib; \
      print([(k,d) for k,d in ARTIFACT_DIRS.items() if not pathlib.Path(d).is_dir()])"

    # every published spec should have a model card
    ls models/*/README.md

Round 23b found `finance_aml.gnn` declaring `hf_repo="qsh-finance-aml-gnn"` while
`Daemons-Q/qsh-finance-aml-gnn` returned **HTTP 401** and its ten siblings returned 200. There was no
`models/finance_aml_gnn` directory, no entry in `models/metrics.json`, and no model card. The spec
described a model that had never been trained. Its `notes` explained why the browser ships the GBM
instead, which read as an explanation and hid the fact that nothing existed at all.

**The 401-versus-200 contrast is what makes this checkable.** A single 401 could be a private repo or
a token problem; ten 200s beside one 401 cannot be.

After editing the registry, regenerate the baked frontend config - `python scripts/export_config.py`
- or `tests/test_api_config.py::test_baked_json_matches_backend_config` fails. That test is doing its
job; do not silence it.

## Dependency claims decay, so re-measure them and watch the wording

A version claim is only true on the day it is written, and the dangerous ones are the claims that
are true of one thing and read as true of another.

Per round, for each pinned crypto or FHE dependency:

    gh api repos/<owner>/<repo>/releases/latest --jq '"\(.tag_name) \(.published_at)"'
    gh api repos/<owner>/<repo>/releases --jq '.[] | "\(.tag_name) \(.published_at[0:10]) \(.name)"'
    npm view <package> version
    gh api graphql -f query='{securityVulnerabilities(first:10, ecosystem:NPM, package:"<pkg>"){nodes{advisory{ghsaId summary}}}}'

**Ask which "upstream" a sentence means.** This repository carried "node-seal is on SEAL 4.1.2, and
that is upstream's newest" for several rounds. True of node-seal, false of SEAL, and the sentence
does not say which. Measured 2026-08-09: node-seal 7.0.0 is indeed still newest, while Microsoft
SEAL had shipped **six releases** since 4.1.2 and reached 4.4.3 four days earlier. The claim was not
a lie and was not stale in the way a version number goes stale - it was ambiguous in the flattering
direction, which is harder to catch and is why this row exists.

**Distinguish "no advisory" from "no problem."** SEAL 4.4.0's release note is the single line "Major
security update", with no CVE, no GHSA and no affected-version range. The honest report is the
version history plus the absence of an advisory, not an inference in either direction. Where a fix
is described (4.4.3: null-output-pointer dereferences and C++ exceptions crossing the **C API**
boundary), record whether it can even reach this project's build - a WASM/Emscripten build may not
use the C API at all, and "unverified" is a legitimate answer to write down.

## PQC library conformance: run NIST's negative vectors, not just the happy path

Every run on every page exercises ML-KEM and ML-DSA, and every one of them takes the happy path.
Nothing in a normal run tells you whether the library REJECTS what the standard says it must.

FIPS 203 section 7.2 is normative on this: "ML-KEM.Encaps shall not be run with an encapsulation key
that has not been checked as above." A library that skips the check encapsulates to a malformed key
and returns a shared secret, silently.

    cd web && npm run check:pqc

Round 22 found this was not hypothetical. The pinned `@noble/post-quantum` **0.2.1 accepted all 2,595
malformed keys** in the C2SP/CCTV modulus set - 775 / 780 / 1040 across the three parameter sets,
zero rejected. Its polynomial coder used the compression path for d=12 where FIPS 203 specifies
plain ByteEncode12/ByteDecode12, so the round-trip that detects an out-of-range coefficient was a
no-op. **0.6.1 rejects all 2,595.**

**Validate the checker against a known-bad build before trusting it.** A conformance script that
always passes is worse than none, and this one is one `try/catch` away from being exactly that.
Install the old version in a scratch directory, point the script at it, and confirm it exits
non-zero. Done for this script: 3 FAILs and exit 1 on 0.2.1, all-OK and exit 0 on 0.6.1.

Where the vectors come from, and why more than one source is needed - they do not overlap:

| Source | Covers | Licence |
|---|---|---|
| C2SP/CCTV `modulus/` | every value in [q, 2^12-1] at every key position - the deep section 7.2 set | CC0-1.0 |
| NIST ACVP `ML-KEM-encapDecap-FIPS203-tr1` | the section 7.3 decapsulation-key hash check | US Gov public domain |
| C2SP/wycheproof | type and length checks only | Apache-2.0 |

Two traps found while wiring this up, both of which produce a falsely reassuring result:

- **The ACVP `encapsulationKeyCheck` negatives are wrong-LENGTH keys**, not modulus violations - the
  invalid ML-KEM-512 entries are 1216 bytes against the correct 800. Any library with a length check
  scores 10/10 there while failing every modulus vector. Passing ACVP's key-check group is not
  evidence the section 7.2 check exists.
- **That ACVP file's `decapsulationKeyCheck` group carries no key material at all** - the tests
  contain only a `tcId`. A harness that reads a `dk` field gets `undefined`, throws, and records a
  "rejection", which scores as a pass. Assert that the vector field you are reading actually exists.

Scope this honestly when reporting it: it tests one normative input check in one library. noble is
self-audited with no side-channel protection, and liboqs's own README still says in capitals that it
should not be relied on in production.

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
the code - the rendered output. Four of the SIX inert controls Round 20 found had correct-looking
code paths, and one carried a comment claiming the fix had already been made.

Audit every view that renders the same state, not one per page. The sixth inert control was found
only by repeating the sweep on the Orchestration graph, whose memo dependency array omitted the
protocol: the wiring was correct, the label function was correct, and the node still never changed.
Read the rendered node text rather than the DOM you expect - that graph renders React Flow HTML
nodes, and a sweep querying `svg text` returned nothing at all.

## Local-green is not CI-green, and the deploy will not tell you

For an entire round the local gate was `pytest -m "not slow"` + `tsc --noEmit` + `npm run build`,
all three green, and **CI was RED on every single push**. The failure was one `ruff E702` in a file
the blocking lint scope covers. Nine commits went out and the demo deployed correctly from every one
of them, because the VPS pulls the git branch - not a CI artifact - so a red build deploys exactly
as smoothly as a green one. Nothing anywhere says otherwise unless you look.

The trap is that the local gate is a SUBSET of CI and the subset is invisible. CI also runs config
parity, two lint scopes, a secrets check, a Japanese-character check, and a committed-artifact check.
None of those is in `pytest`.

| # | Check | Pass condition |
|---|---|---|
| CI1 | `make ci-local` before every push | exits 0; it runs all seven blocking gates in CI's own order |
| CI2 | After pushing: `gh run list --limit 3 --branch <branch>` | conclusion is `success`, not `failure` |
| CI3 | If red: `gh run view <id> --log-failed` | read it before deploying, not after |
| CI4 | When adding a CI step, add it to `scripts/ci_local.sh` too | the two lists match |
| CI5 | Never report a round "green" on local gates alone | the claim names which gates ran |

**CI2 is the one that was missing.** A local gate can only ever prove what it covers; only CI proves
CI. `make ci-local` narrows the gap but does not close it - the runner installs from
`requirements-*.txt` on a clean ubuntu-24.04 with `HF_HUB_OFFLINE=1`, so a dependency present in the
local venv and absent from the requirements files still passes locally and fails there.

**Deploying from a red commit is not blocked by anything.** That is a deliberate property of the
deploy path, not a bug: the VPS does `git reset --hard origin/<branch>`, which knows nothing about
CI. So the check has to be a habit at CI2, because no machine will stop you.

## A shuffled DataLoader is correct for training and silently wrong for harvesting

The medical CKKS student was distilled on teacher labels attached to the WRONG IMAGES, for two
rounds, and every signal said it was fine.

`predict_loader(model, tr_dl)[1].argmax(1)` returns scores in the SAMPLER's order. `tr_dl` is built
with `shuffle=True`, correctly, because it is also the training loader. The result was paired with
`to14(xtr)` in the dataset's original order, so each teacher label landed on a different image.

Measured: labels harvested that way matched ground truth on **0.6227** of training rows, where a
sequential pass matched on **1.0000**. The student scored **AUC 0.5984**; the same architecture on
aligned labels scores **0.8250**. Fixing the alignment moved the shipped model 0.5984 -> **0.7882**
and its validation Youden's J from 0.1028 -> 0.5388.

**Nothing downstream can detect this.** Shapes match. The loss decreases. Early stopping runs to
completion. Agreement-with-teacher is *high*, because the student faithfully learns the marginal
distribution the shuffled labels describe. A plausible number reaches the model card.

| # | Check | Pass condition |
|---|---|---|
| S1 | Every `predict_loader` / `model.predict` whose OUTPUT is paired with a dataset by position | the loader is sequential |
| S2 | Harvested teacher labels vs ground truth on the TRAINING split, **as minority-class F1, not accuracy** | far above the permutation null - see below |
| S3 | Compare that figure against `p^2 + (1-p)^2` for the ACTUAL positive rate | the measured value must clear it by a wide margin, not merely be high |
| S4 | A student that is much worse than the same architecture trained on ground truth | suspect the targets, not the capacity |
| S5 | The guard is machine-enforced, not remembered | `predict_loader` raises on a `RandomSampler` |

**S2 is the cheap decisive one, but "near 1.0" is the wrong pass condition and would have missed
this bug on the most imbalanced use case here.** Under a random permutation of binary labels with
positive rate p, expected agreement is `p^2 + (1-p)^2`. That is a property of the class balance, not
of the model:

| use case | positive rate | agreement a FULLY SHUFFLED harvest scores |
|---|---|---|
| medical BreastMNIST | 0.2692 | 0.6065 |
| OPS-SAT segments | 0.2136 | 0.6640 |
| finance AMLworld | 0.0010 | **0.9980** |

So at 1-in-981 a completely broken pairing scores 0.9980 and sails past "near 1.0". The medical bug
was caught because 0.6227 looked obviously wrong at a 73/27 split - the same defect in finance would
have looked perfect. **Use minority-class F1 of teacher-vs-truth instead, and state the permutation
null beside it.** Confirmed on the medical case: 4,000 simulated permutations give a 99% interval of
[0.5641, 0.6484], and the observed 0.6227 falls inside it, so that harvest was statistically
indistinguishable from random - which is exactly what it was.

This is the same failure shape as `both_scored_zero`: a guard written for a real incident, with a
condition narrow enough that the next instance walks around it.

**Run on finance, which is the case S2-as-written could not have protected:**

```
  positive rate 0.000723   (n = 2,843,872 training rows)
  accuracy    teacher-vs-truth  0.9984   permutation chance 0.9986   <- BELOW chance, and passes "near 1.0"
  minority-F1 teacher-vs-truth  0.4746   permutation      0.0012     <- separates by ~400x
```

The accuracy figure is *below* its own permutation null and still looks like a pass. The
minority-class F1 is the statistic that discriminates. Finance is confirmed correctly aligned, which
is the point: the check has to be able to return "clean" for a reason, not by accident.

S3 is the diagnostic habit worth keeping. I first read this student's weakness as a CKKS capacity
limit and wrote that into three documents. A probe of the identical architecture on aligned labels
scoring 0.23 AUC higher is what disproved it - **before concluding "the encrypted tier is small, so
it is weak", train the same small model properly once and see what it can actually do.**

## Two HE contexts is not a hybrid, and the hybrid is not available

Asked whether to adopt a CKKS + BFV hybrid. The answer is no, and the reason is stronger than "not
implemented": Eldefrawy, Genise and Manohar (PQCrypto 2023, ePrint 2023/988) give strong evidence
that switching between BGV/BFV and CKKS is **as hard as bootstrapping**. CHIMERA specified BFV
bridges and never implemented them. Everything that works - PEGASUS, OpenFHE's `EvalCKKStoFHEW` -
switches CKKS to FHEW/TFHE, not BFV; searching OpenFHE for `CKKStoBFV` returns zero hits. And none
of it is browser-reachable: SEAL has no scheme switching, and `openfhe-wasm` binds `pke` and `core`
but not `binfhe`.

**The goal did not need switching.** An integer scorecard runs in its own BFV context beside the
CKKS models, and no ciphertext is ever converted.

| # | Check | Pass condition |
|---|---|---|
| H1 | Before adopting a "hybrid", confirm whether it needs CONVERSION or just two contexts | two contexts is nearly always the answer, and it is free |
| H2 | For any HE capability, verify it exists in the library the BROWSER can load | SEAL != OpenFHE; `openfhe-wasm` != OpenFHE |
| H3 | Check the specific scheme PAIR, not the feature name | "scheme switching" exists; CKKS-to-BFV does not |
| H4 | An integer model under BFV must reproduce plaintext EXACTLY | assert equality, not a tolerance |

H4 is the property worth testing rather than assuming, and it is what distinguishes the two schemes
in this project. Measured: a fitted scorecard evaluated over BFV matched its plaintext score on
60/60 rows exactly, while CKKS returned -1.999998397974359 for an exact -2. Both are correct
behaviour for their scheme; only one supports the claim "the encrypted result IS the plaintext
result". `tests/test_scorecard_bfv_exact.py` asserts equality for BFV and asserts *in*equality for
CKKS, so a future change that silently makes CKKS look exact would fail.

Workings: [`references/he-scheme-selection.md`](../references/he-scheme-selection.md).

## An ABSENT licence is more restrictive than a bad one, and no scanner reports it

`submodules/hpmpc` is MIT and declares two submodules of its own. One of them,
`chart21/pygeon`, has **no licence file at all**, which by default reserves all rights - a stronger
restriction than GPL, not a weaker one. It is not populated in this checkout, so nothing unlicensed
is present; the exposure is that `git submodule update --init --recursive` would fetch it silently,
into a project whose rule is commercial-OK-only.

Licence scanners look for a restrictive SPDX identifier. Absence returns nothing, which reads as
clean.

| # | Check | Pass condition |
|---|---|---|
| T1 | For each submodule, read ITS `.gitmodules` too | transitive dependencies are enumerated, not assumed absent |
| T2 | For every transitive repo, query the licence via the API, not the file listing | `license: null` is a finding, not a pass |
| T3 | Confirm whether the nested module is actually populated | present-and-unlicensed is a live problem; declared-but-absent is a latent one |
| T4 | Initialise submodules with `--init`, never `--init --recursive` | the recursive form is what pulls the unlicensed tree |

T2 is the one an automated check gets wrong. `license: null` from the GitHub API means all rights
reserved, and a naive check that only fails on a deny-list of SPDX identifiers passes it.

## The QCNN inference backend returned a coin flip and the page called it a prediction

Selecting **Inference = QCNN (QHE)** routed the surfaced class through `qotpQcnnInfer` and reported it
as `primary_inference_class`, beside the CKKS prediction. It carried no information about the input.

Measured over 24 random feature vectors x 8 run seeds each:

| quantity | value |
|---|---|
| vectors whose class FLIPS with the seed | **24 / 24 = 100%** |
| mean \|<Z0>\| with the QOTP mask | 0.0761 |
| mean \|<Z0>\| with the mask DISABLED | 0.0748 |
| correlation, unmasked class vs mean(feature) | **-0.070** |

The QOTP mask is applied and never undone, so an X on qubit 0 flips the sign - but that is the
smaller cause. Disabling the mask barely moves the numbers, because seven FIXED rounds of H, a CNOT
ring and T on every qubit drive the state toward uniform. The sign of <Z0> was numerical noise.

**Why nothing caught it.** The project already knew the neighbouring benchmark row was empty -
`clientCompute.ts` labels it "no trained weights", states "There is nothing here to be accurate
ABOUT", and this document records that an invented `accuracy: 1.0` was removed from it. That
scrutiny stopped at the benchmark row. The selector inherited the code and not the caveat, and every
word of its log line was true: the T-gates are counted, the simulation is a real state-vector
simulation, the class is genuinely computed. It just was not a prediction.

| # | Check | Pass condition |
|---|---|---|
| QC1 | A surfaced class comes from something with trained parameters | a fixed circuit cannot classify |
| QC2 | Class is STABLE across run seeds for a fixed input | flipping with the seed means it is noise |
| QC3 | \|<Z0>\| is not pinned near zero across all inputs | ~0.07 for every input is a scrambled state |
| QC4 | The model is trained in the space the browser actually SUPPLIES | z-scores in, z-scores trained |
| QC5 | Circuit parity is checked against an independent reimplementation | before any weights are shipped |
| QC6 | The classical baseline is reported beside the quantum number | every time, including when it loses |
| QC7 | The epoch is selected on a fold that is NOT the one reported | selecting and reporting on the same rows is leakage |
| QC8 | The run is seeded and the seed ships with the weights | unseeded reruns moved this headline by 0.07 |
| QC9 | The printed metric NAME is the metric that was computed | a mislabelled number is worse than a missing one |

**QC4 and QC5 each caught a fault that would have made the fix worthless.**

QC5: the training circuit and an independent numpy reimplementation disagreed by up to **0.73** on a
value bounded in [-1, 1]. The trainer applied CNOTs with two chained `movedim` calls and used the
ORIGINAL axis index for the second, so control and target were crossed on every gate. It trained
perfectly well - the circuit was self-consistent, just not the circuit documented - and only a parity
check exposed it. After rewriting with flat index arithmetic the two agree to **2.4e-07**.

QC4: the first trained model min-max scaled its inputs against RAW WDBC ranges (area 143..2501). The
bundled sample the browser feeds is **StandardScaler output** - verified at L1 distance 0.0002 - so
seven of eight features pinned to -1 and <Z0> came out at **0.0098**, a degenerate state that made
the TRAINED path exactly as meaningless as the untrained one it replaced. Retrained in z-space with
a bounded `RY(tanh(z)*pi)` encoding, since z-scores span [-3.11, 6.05] and would otherwise wrap the
Bloch sphere: <Z0> moved to **0.1669**.

**What the result is, stated without varnish.** Held-out accuracy **0.7326** on 171 test rows,
against a majority baseline of **0.5930** and logistic regression at **0.9535** on the identical
split. The QCNN learns and the classical model beats it comfortably, and the log line on the page
says so in those words rather than implying an advantage. That is consistent with this project's
other quantum results - the telecom QNN loses to a random forest - and with the concentration
literature.

An earlier revision of this section recorded **0.8947** here, and that figure should not be trusted
in preference to the one above. Three things were wrong with how it was obtained. The run was
unseeded, and reruns on identical inputs returned 0.9181 / 0.9064 / 0.8947 / 0.8480 - a 0.07 spread,
so the recorded value was a draw from a distribution rather than a measurement. The epoch was
selected on the same rows the headline was then read off, which is leakage. And there was no
validation fold at all, so the split it refers to no longer exists. After seeding, a 70/15/15 split
and epoch selection on the VALIDATION fold, the shipped figure is 0.7326. The fall is the
correction; QC7, QC8 and QC9 above exist so it is not repeated.

**QOTP hiding and QCNN prediction are now separate.** Undoing a Pauli frame through Clifford+T is
not a detail that can be waved through: `T X T-dagger` is not Pauli, which is the difficulty Ortega &
Martin-Delgado's scheme exists to solve, and this repository does not implement it. So
`qotpQcnnInfer` remains the hiding and T-count demonstration - both of which it genuinely supports -
and the prediction path is a separate unmasked trained circuit. A use case with no trained weights
falls back to CKKS and says why, rather than passing off a coin flip.

Workings: [`references/qcnn-inference-investigation.md`](../references/qcnn-inference-investigation.md).

## A reproducible number measured on contaminated rows is still contaminated

The finance evaluation sample carried a near-perfect label leak, and every gate this project has was
green the whole time. The payload was well-formed. The browser re-scored it and matched the recorded
expectation. `make ci-local` passed. None of that is evidence about the ROWS.

What was wrong: the sample was built from `nrows=400_000` of AMLworld HI-Small. That corpus is
time-sorted for background traffic, but IBM's injected laundering rows carry timestamps spanning the
whole period while sitting early in the file. So the first 400k rows are 99.97% hours 0-2
(344,209 / 33,144 / 22,472) with roughly 175 rows across hours 3-23, a large share of them
laundering. Sampling negatives uniformly from that head reproduces its shape exactly - predicted
6,924 / 667 / 452 against observed 6,912 / 661 / 470 - so in the shipped sample **negatives held 3
distinct hour values and positives held 22**, and `hour_of_day >= 3` implied laundering at 100%
precision.

Measured, not asserted. `hour_of_day` alone scored AUC-PR 0.7104 against a 0.0192 base rate, a 37x
lift. On the shipped ONNX the in-browser figure was 0.5100, falling to 0.4398 permuted and 0.4479
zeroed - so **13.8% of the verification number came from the truncation**.

**The source corpus is innocent, and checking that mattered.** Across all 5,078,345 rows both classes
span all 24 hours and the laundering rate moves only 0.00028 to 0.00174. Had the leak been in
AMLworld it would be a dataset caveat; being in the sampling made it this project's defect.

**Scope, stated precisely because it is easy to overstate.** This contaminated the BROWSER
VERIFICATION figure, not the headline result. `models/metrics.json` records minority-class F1 0.1519
on the real temporal test split of 1,523,504 rows at a 0.00102 positive rate, and that number never
touched the leaked sample.

| # | Check | Pass condition |
|---|---|---|
| LK1 | No single feature separates an evaluation sample on its own | per-column ROC AUC < 0.90; the generator warns |
| LK2 | A sample is never drawn from the HEAD of a sorted corpus | read the whole file, or sample across it |
| LK3 | A leak found in a sample is checked against the SOURCE before blaming the dataset | here the source was clean and the generator was not |
| LK4 | Stratified prevalence is a stated target, not a side effect of an expression | `want // 4` silently became 25% once the truncation was lifted |
| LK5 | Fixing the generator is not fixing the demo | the browser fetches `eval_sample.json` from HF at a pinned revision |
| LK6 | The guard's own arithmetic is self-tested before its output is believed | plant a leak, plant pure noise, check both |
| LK7 | `feature_names` length equals the sample's column count | a mismatch makes the guard skip the payload SILENTLY |

**LK6 exists because this guard was wrong twice**, and both versions produced confident output.
Version one asked whether positives sat on values no negative took - sound for an hour index,
meaningless for a continuous float, and it flagged `amount_paid_log` at 99.7% because distinct
doubles rarely collide. Version two used a binary rank-AUC formula on multi-class labels, where
`y == 1` picks one class of eight while the ranks span every row; it printed **AUC 5.1292**. An
impossible number is at least loud. A plausible wrong one would have been believed. It now runs
one-vs-rest and is self-tested against a planted leak and against pure noise.

**LK7 is how telecom escaped checking entirely.** Its payload wrote the 8 CLASS names under
`feature_names` for a 7-column matrix - `generate()` returns `(X, y, label_names)` and the caller
unpacked the third value into a variable named `feature_names`. Nothing rendered it so nothing
complained, but the guard skips any payload whose column count disagrees with its name count, so the
one flat-feature use case that most needed checking was never checked.

**And what checking it found is a caveat, not a bug: telecom is SEPARABLE BY CONSTRUCTION.**
`sift_ratio` and `click_asymmetry` each separate `detector_blinding` from the other seven classes at
**one-vs-rest AUC 1.0000**; `qber_mean` reaches 0.9785 on `intercept_resend` and `multiphoton_rate`
0.9703 on `photon_number_splitting`. That is the generator's `_SIGNATURES` table doing its job - each
attack is given a distinctive value on purpose. The corpus was already labelled synthetic everywhere
it is reported; what was missing is that its 0.9867 accuracy is close to tautological. It measures
the simulator, not the world, and the payload's provenance now says so in those words.


**LK4 is the trap inside the fix.** Removing the truncation made 5,177 positives reachable, so
`take_pos = want // 4` jumped from 149 (1.82%, capped by scarcity) to 2048 - a 25%-laundering
sample. The expression had produced a defensible balance by accident and stopped doing so the moment
its input changed. It is now an explicit 2% target.

**LK1's first implementation was wrong and is worth recording.** It tested whether positives occupied
values no negative took. That is sound for a low-cardinality index like an hour and nonsense for a
continuous float - it flagged `amount_paid_log` at 99.7% purely because distinct doubles rarely
collide. Rewritten as per-column ROC AUC, which does not care about cardinality.

**LK5 is why this round did not close the issue.** `onnxInference.ts` derives the sample URL from the
model URL, so the browser reads it from Hugging Face at the pinned revision. The generator is fixed
and the local artefact is regenerated, but the deployed demo keeps serving the leaked sample until
the model repos are re-uploaded and re-pinned.

## Real data next to the model's own numbers, and the licence that decides whether it can ship

A score plotted alone cannot distinguish a good model on easy data from a mediocre model on hard
data - the reader has no way to see how separable the classes were to begin with. `RealDataScatter`
draws the actual corpus so that question is answerable by looking. What decides whether a use case
gets one is **redistribution licence**, which is a stricter test than "publicly downloadable" and
knocks out most obvious candidates.

Checked 2026-08-10, per use case:

| use case | training corpus | licence | ships as real 2D? |
|---|---|---|---|
| satellite | OPS-SAT-AD | CC-BY-4.0 | **yes, and it IS the training corpus** |
| finance | AMLworld | CDLA-Sharing-1.0 | yes, but share-alike propagates to the slice |
| telecom | generated here | n/a | **NO - not implemented.** `VerificationChart.tsx` registers four use cases and telecom is not among them. This row said "yes" while no `web/public/data/telecom/real_2d.json` exists. The corpus is also separable by construction, so a plot of it would measure the simulator |
| medical | BreastMNIST / BUSI | CC BY on the ARTICLE, not the data | **no** - WDBC substitutes |
| genomics | 1000 Genomes | IGSR grants no blanket licence | **no** |

| # | Check | Pass condition |
|---|---|---|
| RD1 | A bundled corpus has a licence that survives REDISTRIBUTION in a public bundle | "publicly downloadable" is not the test |
| RD2 | `is_training_corpus` is true only when it is literally the corpus the model trained on | satellite yes; medical/genomics substitutes must say so |
| RD3 | A substitute corpus is never drawn on the model's own axes | no shared feature space = no shared axes |
| RD4 | The axis pair is chosen by measurement, not by eye | satellite: all 171 pairs scored, `n_peaks` x `diff2_peaks` won at AUC-PR 0.8492 vs 0.2136 base rate |
| RD5 | Licence + attribution travel INSIDE the data file | so they cannot drift from the points they describe |
| RD6 | The chart is confirmed to RENDER, not merely to typecheck | count `.recharts-scatter-symbol` nodes in the browser |
| RD7 | The axis pair is chosen for separability AND readability, which are different criteria | look at it; a metric cannot tell you a chart is a blob |
| RD8 | A heavy-tailed axis is log-scaled, and the scale is decided by the GENERATOR | whether compression is warranted is a property of the data |
| RD9 | Local browser checks run in the SAME mode production runs | `?demo=client`, or a stale local service will answer for it |
| RD10 | A payload field that carries the finding is actually RENDERED | `note` was written, computed, shipped, and displayed nowhere |

**RD9 nearly produced a false bug report, which is the point of writing it down.** Verifying the
genomics overlay locally, the page showed "the European-to-African AUROC drop is the finding" - a
sentence the registry had explicitly corrected to "a distribution shift, NOT an AUROC drop ... no
AUROC is computable". It looked like a stale duplicate contradicting its own correction on a live
page.

It was not in the source. Not in `config.defaults.json`. Not in the built bundle - `grep` over
`web/dist/assets/*.js` returned zero. Not anywhere in the repository. It survived a hard reload with
the cache cleared.

The cause was a **uvicorn left running on :8000 from a previous session**, days old.
`refreshConfigFromApi()` does `CONFIG = { ...defaults, ...(await r.json()) }`, so a stale API's
`notes` silently overwrite the correct baked ones. The bundle was right and the page was wrong, and
nothing in the repository could have told me that.

Production is unaffected and that was checked rather than assumed: the public demo returns **404 on
`/api/config`**, so the fetch fails and the baked defaults stand. Re-verified with `?demo=client`,
which takes the same path production takes - corrected text present, stale text absent.

The transferable rule: a local browser check is only as trustworthy as the services answering it.
Pin the mode, or confirm what is actually serving each response.

**RD10** is smaller and duller: the `note` field carrying each chart's actual finding - "8 of the 32
larger-effect variants differ by at least 0.20 in allele frequency" - was computed by the generator,
shipped in the payload, and rendered by nothing. Writing a finding is not publishing it.



**RD7 cost a full iteration and is the most transferable lesson here.** Ranking finance's 496 feature
pairs by AUC-PR put `format_risk` x `src_bank_diversity` first at 0.3195, so that shipped - and
rendered as a few vertical bands hard against the origin plus one lone outlier. `format_risk` has
**six distinct values**, five bunched near zero and one at 0.0075. A gradient-boosted tree does not
care how values are spaced along an axis; an eye does. Every gate passed and the chart was useless.

The measured alternatives, which is what made the choice defensible rather than aesthetic:

| pair | AUC-PR | readable? |
|---|---|---|
| `format_risk` x `src_bank_diversity` | 0.3195 | no - 6 values, uneven |
| `src_fan_out_ratio` x `src_burst_fanout` | **0.2302** | **yes, with a log y-axis** |
| `src_fan_out_ratio` x `dst_fan_in_ratio` | 0.0764 | yes, but barely separates |

Shipping the middle one trades 0.09 of AUC-PR for a chart that shows something, and 0.2302 against a
0.0173 base rate is still a 13.3x lift. The log axis then earned its place immediately: at linear
scale the high-burst-fan-out laundering cluster sat flat on the baseline, and compressing the axis is
what made it visible as a distinct group - which is the structuring signature the feature exists to
capture.



**RD6 exists because of a specific incident.** Round 24j shipped a chart that typechecked, built, and
rendered nothing - the payload was attached at 1 of 4 return sites. `tsc --noEmit` and `npm run build`
both pass on a component that never mounts. The only evidence that a chart works is a node count from
a real page:

    document.querySelectorAll('.recharts-scatter-symbol').length     // expect 529 on satellite
    document.querySelectorAll('figure.chart-figure').length          // expect >= 1

**RD3 is the one with an ethical edge.** WDBC is breast-cancer data and so is BreastMNIST, which makes
overlaying them feel natural. They share a disease and nothing else - WDBC is cell-nucleus morphometry
from fine-needle aspirates, BreastMNIST is 28x28 ultrasound. Putting them on one axis pair would
assert a correspondence that does not exist, in a clinical context. They are drawn separately and the
caption leads with "A DIFFERENT cohort and modality".

**And a finding that fell out of RD1:** the satellite page bundles SMAP/MSL for its anomaly demo,
which is NOT redistributable - 17 U.S.C. 105 removes US copyright domestically, which is not a grant
and is territorial - while the corpus it actually trains on, OPS-SAT-AD, IS CC-BY-4.0. The
redistributable one was not the one being shipped. Swapping it is not a one-line change because
`telemetry.json` also feeds the SHAP explanation baseline and the in-browser linear anomaly model, so
it is recorded here rather than done in passing.

## The training environment, measured rather than assumed

Every number here was measured on this host on 2026-08-10, not read off a spec sheet. The reason to
measure is in the first row: **`torch.cuda.is_available()` returned `True` on an interpreter that
could not launch a single kernel.**

| what | measured value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, **sm_120**, 102 GB (100.6 GB free) |
| CPU / RAM | 13th Gen Intel Core i5-13600K, 20 threads / 125 GB |
| CUDA toolkit | **13.0** (`nvcc` V13.0.88) |
| project interpreter | `.venv/bin/python` = **Python 3.12.11** |
| torch | **2.13.0+cu130**, arch list includes `sm_120` |
| fp32 matmul | **33.0 TFLOP/s** (4096^3, 20 iters) |
| autograd + bf16 autocast | OK |

**The footgun, and why the 3.12 venv rule exists.** The system interpreter is Python 3.9.23 carrying a
stale `torch 2.1.2+cu121` built for `sm_50`-`sm_90` only. On that interpreter:

    torch.cuda.is_available()  ->  True          # <-- false green
    a @ a                      ->  RuntimeError: no kernel image is available for execution on the device

`~/.local/lib/python3.9/site-packages` also holds a corrupted `-orch` distribution from a failed
install, which makes pip print warnings that look like the real problem and are not. Any script run
with bare `python3` instead of `.venv/bin/python` inherits all of this. I asserted "the 96 GB card is
unusable" from exactly this mistake before checking which interpreter I was on.

| # | Check | Pass condition |
|---|---|---|
| TR1 | GPU probes run under `.venv/bin/python`, never bare `python3` | `python3 -c "import torch"` on this host is the WRONG environment |
| TR2 | Capability is proven by a real kernel launch, not `is_available()` | `(a @ a).sum()` must return a number; `is_available()` lies here |
| TR3 | `torch.cuda.get_arch_list()` contains the device's `sm_XX` | `sm_120` for this card; a missing arch is the silent failure mode |
| TR4 | GPU is used because it was measured faster, not because it exists | see the reversal below. Same rule as **G4** below |
| TR5 | Every training dependency is commercial-use and actively maintained | licence checked from installed metadata, not from memory. Same check as **G3** below, stated for the training environment rather than the accelerator |
| TR6 | ONNX export path stays CPU-reproducible | `onnxruntime` here has no CUDA provider, and the browser runs WASM anyway |

**TR4 is the one that bites.** The card is not automatically a win at this project's model sizes:

| workload | GPU | CPU | verdict |
|---|---|---|---|
| XGBoost 200k x 32, 200 trees | **0.68 s** | 28.58 s | **42x - use the GPU** |
| torch 50x gram matrix, 200k x 32 | 0.064 s | **0.048 s** | **GPU slower** - launch + transfer dominates |

So the honest guidance is: **GPU for the tabular gradient-boosted teachers (finance AML, telecom IDS),
where it is a 42x difference; CPU is fine and sometimes faster for the small linear and MLP students
this project actually publishes.** Reaching for `.cuda()` on a 4-layer student is cargo cult.

**Dependency licences, read from installed metadata on this host:**

| package | version | licence | commercial |
|---|---|---|---|
| torch | 2.13.0+cu130 | Apache-2.0 (with LLVM exception) | yes |
| torchvision | 0.28.0+cu130 | BSD | yes |
| scikit-learn | 1.9.0 | BSD-3-Clause | yes |
| xgboost | 3.3.0 | Apache-2.0 | yes |
| onnx / onnxruntime | 1.22.0 / 1.28.0 | Apache-2.0 / MIT | yes |
| skl2onnx | 1.20.0 | Apache-2.0 | yes |
| transformers | 5.14.1 | Apache-2.0 | yes |
| datasets | 5.0.1 | Apache-2.0 | yes |
| numpy / scipy / pandas | 2.5.1 / 1.18.0 / 3.0.5 | BSD-3-Clause / BSD / BSD | yes |

No copyleft and no non-commercial terms in the training path. Note this covers the **tools**; dataset
licences are a separate and stricter question tracked in `DATA_LICENSES.md`.

Not installed, and deliberately so until something needs them: `lightgbm`, `catboost`, `optuna`,
`imbalanced-learn`, `peft`, `trl`, `accelerate`, `onnxruntime-gpu`, RAPIDS `cuml-cu13`. Adding a
dependency to this list requires a measured reason, which is what TR4 is for.

## The most tempting motivation for a use case is the one most likely to be false

The satellite use case had no stated problem, only a stated mechanism. Checking what the real
motivation is turned up the reverse of the obvious one: **export control does not block moving
satellite telemetry, it explicitly decontrols it.** 22 CFR 121.1 USML XV Note 3 names housekeeping
data - temperatures, pressures, currents, battery charges - and says encrypting it "does not, in and
of itself, cause the telemetry data to become subject to the ITAR".

"ITAR blocks telemetry" is what a reasonable person assumes, it is quotable, it sounds authoritative,
and a reader who knows Category XV stops reading at it. This project had never made the claim; the
value of checking was in learning not to.

| # | Check | Pass condition |
|---|---|---|
| M1 | Every use case states a PROBLEM, not only a mechanism | "QKD over a LEO downlink" is a mechanism |
| M2 | The motivation cites a primary source that says it in its own words | an operator, regulator or benchmark author, quoted |
| M3 | The most obvious-sounding justification is checked against the regulation TEXT | not against what everyone says it says |
| M4 | Any novelty claim survives an adversarial search for prior art | search for the thing, then search for who already did it |
| M5 | A cited dollar figure comes from the contract system of record, not a press release | USASpending obligated value, not a vendor ceiling |

M5 came out of checking defence-space programs this project does not cite and now knows not to: the
widely repeated "$640M Digital Bloodhound contract" is a contractor-stated ceiling against
**$128.6M actually obligated** (USASpending PIID 47QFMA24F0023), and a Space Systems Command article
citing its own "$17 million" zero-trust award sits against **$5.15M obligated** (PIID FA880223C0002).
Both figures are real numbers about real contracts and both are roughly 5x and 3x the money that
exists. An agency's own press office is not a primary source for its own spending.

M4 separated two claims that felt equally safe. Encrypted inference on housekeeping telemetry for
anomaly detection: zero verified prior work, a real gap. Encrypted **conjunction assessment**: a
decade-deep field with an active ESA contract (ShareSat, Cybernetica, Sharemind MPC, TRL 4) - a
novelty claim there would have collided with prior art immediately. Neither intuition would have
told them apart.

Workings: [`references/satellite-motivation-fact-check.md`](../references/satellite-motivation-fact-check.md).

## Check whether the model protects the thing that actually leaks

Five verticals were checked against 2026 privacy-preserving-inference literature. Four came back
defensible. The fifth returned a finding no accuracy metric could have surfaced: **genomics protects
the computation, and the computation is not where the leak is.**

Pooled allele counts are the canonical genomic privacy failure - Homer et al. (2008) is the paper
that caused NIH to withdraw public aggregate GWAS data - and encrypting their computation is
orthogonal to that. Released allele frequencies support genome reconstruction at 70% F1 against
53.4% for binary responses, so publishing the counts raises risk over publishing mere presence.

| # | Check | Pass condition |
|---|---|---|
| P1 | Name the adversary each crypto layer defeats | written down per use case, not implied by "encrypted" |
| P2 | Ask what is RELEASED, not only what is computed | an encrypted computation with a plaintext output protects the input, not the output |
| P3 | Check the literature for attacks on that released artefact | membership inference, reconstruction, linkage |
| P4 | If output perturbation is absent, say so | "no DP on the released aggregate" is a statement, silence is not |
| P5 | Check quantum components against their own published operating regime | a parameter below the threshold its literature states is a reason, not a mystery |

P5 came out of the telecom check. The quantum-kernel SVC here scores macro F1 0.8316 against
RandomForest's 0.9866, and the reason is quantified: fidelity quantum kernels exponentially
concentrate and beat that regime only at **15 qubits or more** (arXiv:2605.31449, May 2026), while
this project runs 4 and 8. The number was already reported honestly with
`quantum_beats_classical: false`; what was missing was that it underperforms for a published reason
at a qubit count that IS the explanation. A measured shortfall with a citation is a finding; the
same shortfall without one invites the reader to assume a bug.

P2 is the general form. This project spent several rounds checking that metrics measure what they
claim, and this is the same question one level up: does the CRYPTOGRAPHY defend what the threat model
says is at risk. Full treatment in [`docs/threat-model.md`](threat-model.md) and
[`references/model-choice-vs-2026-practice.md`](../references/model-choice-vs-2026-practice.md).

## A published sample can go stale against the weights it scores, and only a run will say so

Verifying every published model during the run - rather than only the selected one - found a real
divergence on its first execution: `medical_fl.student` scored **accuracy 0.77** on its own published
held-out sample while that sample recorded **0.69**. The weights had been retrained and republished;
the `eval_sample.json` beside them had not been regenerated, so the artefact and its expectation
described two different models at the same pinned revision.

Auditing all nine by modification time, **six samples were older than the weights they score.**
Older does not always mean wrong - a retrain that reproduces its weights leaves the sample valid -
but it is the only cheap signal available before running them, and one of the six was confirmed
wrong by the run.

| # | Check | Pass condition |
|---|---|---|
| E1 | After ANY retrain, regenerate `eval_sample.json` before publishing | sample mtime newer than the weights it scores |
| E2 | Run the pipeline once per use case and read every verification line | all PASS, and the count matches the number of published models |
| E3 | A FAIL is a divergence, not a lower score | investigate before republishing anything |
| E4 | Verify EVERY published model, not only the selected one | medical publishes three, satellite four |

E4 is what turned this from invisible into a log line. The retrain-and-publish cycle documented
above already warned that training clears `hf_revision` and that publishing without `--public` flips
a repo private; this is the third trap in the same cycle - **republishing weights without
regenerating the sample beside them** - and it is the one that produces a confidently wrong number
rather than a broken link.

## Check the binary, not the version string, before trusting GPU support

Before training anything destined for the Hub, confirm the local stack actually targets this
machine. The trap is that version numbers and release notes are a proxy for what shipped, and the
proxy is sometimes wrong in the direction that costs a rebuild.

Measured here 2026-08-10 on an RTX PRO 6000 Blackwell (compute capability **sm_120**), CUDA **13.0**,
PyTorch 2.13.0+cu130, 20 cores, 125 GB RAM, Python 3.12:

| # | Check | Pass condition |
|---|---|---|
| G1 | List the SASS architectures actually compiled into the shared library | `cuobjdump --list-elf <lib>.so` includes this GPU's capability |
| G2 | If absent, check for PTX at a lower arch | `--list-ptx` - PTX JITs forward, at a startup cost, and is not the same as native SASS |
| G3 | Confirm the licence permits commercial use AND the project is maintained | Apache-2.0/MIT/BSD, with a release inside the last year |
| G4 | Check whether the accelerator addresses the MEASURED bottleneck | profile first; do not install a GPU library for a stage that is 5% of runtime |
| G5 | Check for arch-specific correctness bugs, not only support | see below - "supported" and "correct" are different claims |

**G1 caught a false recommendation.** Research advised upgrading XGBoost to 3.4.0 "for native sm_120".
Inspecting the installed 3.3.0 binary:

```
cuobjdump --list-elf libxgboost.so   ->  SASS archs 75, 80, 90, 100, 120
cuobjdump --list-ptx libxgboost.so   ->  PTX 120
```

sm_120 was already there. The upgrade was unnecessary, and no release note would have told us -
only the binary did.

**G4 is what stops a plausible optimisation from being a waste.** Profiling the finance pipeline gave
graph feature construction 52.7s of 69.1s (76.4%) and the XGBoost GPU fit 3.9s (5.6%). Installing a
GPU dataframe stack to accelerate a 69-second pipeline optimises something that is not a problem.

**G5 is the one with teeth.** cuDF PR #23229 (merged 2026-07-21, released in 26.08) fixes
*"incorrect DECIMAL128 groupby sums on Blackwell with CUDA 13.0"* - nvcc miscompiles a 128-bit
`atomicCAS` on sm_100 and newer. **That is exactly this machine's configuration**, and the failure
mode is wrong numbers with no error. A library can support an architecture and still compute
incorrectly on it. Pin RAPIDS >= 26.08 if it is ever introduced, or avoid DECIMAL128.

Verified commercial-OK and Blackwell-capable at this revision: XGBoost 3.3.0+ (Apache-2.0, sm_120
native), LightGBM 4.7.0 (MIT, sm_120 in its arch list, but **pip wheels are CPU-only** - CUDA needs
a source build). **CatBoost has no sm_120 in its build configuration** and its default arch list
includes architectures CUDA 13 no longer accepts. Neither vendored submodule accelerates *training*:
HEonGPU (Apache-2.0) and hpmpc (MIT) accelerate homomorphic and MPC *evaluation*.

## A vendored crypto library has no CVE to find, so nothing will tell you it is old

`node-seal` bundles Microsoft SEAL as a WASM build. The pinned dependency resolves to node-seal
7.0.0, whose vendored `CMakeLists.txt` declares **SEAL 4.1.2, released 2024-07-11**. Upstream SEAL
shipped **4.4.0 on 2026-07-25** carrying the vendor's own words: *"Critical security update: all
users should upgrade to Microsoft SEAL 4.4.0 as soon as possible"*, hardening untrusted
deserialization, followed by 4.4.1 and 4.4.3 in the first week of August 2026.

**No CVE or GHSA exists for either package**, so `npm audit` reports nothing, Dependabot opens no
pull request, and the lockfile is perfectly current against its own registry. The gap is invisible
to every automated signal and is only findable by reading the upstream release notes.

| # | Check | Pass condition |
|---|---|---|
| V1 | For every vendored/transitive crypto library, find the UPSTREAM version it embeds | recorded, with its release date |
| V2 | Compare against upstream's latest release notes, not against the registry | any security-labelled release since is named |
| V3 | If the wrapper cannot be upgraded, state whether the fixed surface EXISTS in this deployment | a reasoned yes/no, not silence |
| V4 | Re-check per round | the version is dated, and the date is recent |

**V3 is what turns this from alarm into engineering.** Checked here: the 4.4.0 fix hardens
deserialization of untrusted input. This demo generates its keys, encrypts, evaluates and decrypts
**entirely in the visitor's own tab** with zero server compute, and deserialises no externally
supplied ciphertext - so the hardened surface is not reachable in this deployment. That is a real
mitigation, not a dismissal, and it stops being true the moment any server-side component accepts a
ciphertext from a client.

The same reasoning applies to a second upstream statement. SEAL's `SECURITY.md` was rewritten in
July 2026 to say in writing that its CKKS targets IND-CPA and **not** IND-CPA-D, and that noise
flooding "must be implemented by the application before a decrypted CKKS value crosses a trust
boundary". Here the secret key and the decryption live in the same tab as the plaintext result, so
no such boundary is crossed. Record the reasoning, because the architecture is what makes it safe -
not the cipher.

## Bootstrap a claimed improvement before shipping it

The OPS-SAT student's AUC-PR moved 0.9225 to 0.9681 when its feature map changed. On a 529-row test
set at a 21 percent positive rate, a delta of that size is exactly the magnitude that turns out to
be resampling noise, and nothing in the pipeline would have said so.

| # | Check | Pass condition |
|---|---|---|
| R1 | Any shipped improvement has a paired bootstrap CI on the delta | interval excludes zero |
| R2 | Paired, not independent - both models scored on the SAME resampled rows | shared row-difficulty variance is removed |
| R3 | Quote the interval, not just the point estimate | "at least +0.009", not "+0.046" |
| R4 | Degenerate resamples dropped, not silently scored | draws with one class present are skipped |

Measured here: 4,000 resamples give AUC-PR delta 95% CI **[+0.0089, +0.0853]** and minority-F1
**[+0.0154, +0.0929]**, so the change survives. Note how wide that is - the lower bound is a fifth of
the point estimate, so the defensible claim is "a real improvement of at least about one AUC-PR
point", and quoting +0.046 alone would overstate what 529 rows can support.

R1 pairs with the small-sample rule above. The satellite event-wise metric is computed over 3
events, which is the case where a bootstrap does not rescue you - there is nothing to resample.
Reporting the count and reporting an interval are the same discipline applied at two different
sample sizes.

## A default that contradicts a licence commitment will eventually be taken

DATA_LICENSES.md and the README both stated that `medical_fl.resnet18` is trained from random
initialisation, *because* ImageNet's terms are uncleared for redistribution here. The trainer
implemented that as `--from-scratch`. It was not the default. The artefact published on the Hub
recorded `pretrained: ImageNet1K_V1`, and so did the revision before this round - so the project had
been publishing ImageNet-derived weights while documenting that it does not, across several rounds,
with nothing checking.

The trainer's own docstring argued for the wrong default: *"training from scratch is the wrong move
and transfer learning is the right one"*. That is true about accuracy and irrelevant to the actual
constraint, which is redistribution.

| # | Check | Pass condition |
|---|---|---|
| L1 | Every licence claim in prose names the artefact field that proves it | e.g. "random init" -> `manifest.pretrained` |
| L2 | Read that field from the PUBLISHED artefact, not the local one | `curl .../resolve/main/manifest.json` |
| L3 | The licence-safe option is the DEFAULT, not a flag | opt IN to the restricted path, never out |
| L4 | A machine refuses the unsafe publish | verified to fire, not just present |
| L5 | Docstrings that argue for a default are checked when the default changes | they are the reason the wrong one was chosen |

L3 and L4 are the durable pair. `publish_hf.py` now refuses to upload any publishable-dataset model
whose manifest records a pretrained source, and random initialisation is the default - tested both
ways, so the refusal is known to fire rather than assumed to.

**How this was found, which is the transferable part.** Not by auditing licences. I retrained the
model without the flag, saw two AUCs move, and wrote it up as seed variance - measuring a six-seed
spread to support the claim. Re-running WITH the flag reproduced the documented figures to four
decimal places, which is what exposed both my wrong explanation and the underlying divergence. **An
unexplained number that moves is worth chasing to the actual cause**, because "it is just variance"
is available for any change and explains nothing.

## Check the baseline is the right ROW, not just the right paper

The satellite baseline was recorded as 0.786 with a precise citation: ESA-ADB Table 2, Mission1
lightweight channels 41-46, best event-wise F0.5. Round 24c fetched the paper. **The number is real
and correctly attributed** - Telemanom-ESA-Pruned, and it is the maximum of that row. The citation
was not wrong.

It was the wrong row. Table 2 scores **all events excluding communication gaps**, which is anomalies
PLUS annotated rare nominal events. This project scores anomalies only, and ESA-ADB publishes that
separately in **Supplementary Table 9**, where the same method scores **0.968**. So the project had
been comparing itself against the lenient row of the correct paper for several rounds.

Verifying a citation resolves is not verifying it is the right comparator.

| # | Check | Pass condition |
|---|---|---|
| B1 | The cited number exists in the cited table | fetch the paper, find the row |
| B2 | The cited row scores the SAME TARGET the project scores | anomalies-only vs all-events, per-class vs macro, etc. |
| B3 | The cited metric is computed the SAME WAY | see below - this one was also wrong |
| B4 | The cited split matches | ESA-ADB halves each mission; this project tests on the last 30 percent |
| B5 | The cited sample size is recorded next to the project's | 65 events there against 3 here |

**B3 caught a second mismatch in the same check.** ESA-ADB's published figure is the *corrected*
event-wise F0.5: event precision multiplied by the TIME-LEVEL true-negative rate,
`Prec_corr = TP_e/(TP_e+FP_e) * TNR_t`. That correction is why most of its baselines read < 0.001.
This project computed the UNCORRECTED value and set it beside a corrected baseline - two different
quantities, and the uncorrected one is the flattering direction. Both are now computed and recorded.

A near-miss worth knowing about: a July 2026 study on the same corpus (arXiv:2607.07335, Table III)
reports **0.7809** for GAT on the FULL channel set with UNCORRECTED precision. That is numerically
almost identical to ESA-ADB's 0.786 and is an unrelated number from a different paper, protocol and
channel subset. Two plausible-looking 0.78s that mean different things is exactly the shape of error
B1-B5 exists to catch.

## The deploy step itself needs verifying, and a failed build looks like a successful one

The section above covers CI on the PUSH side. This one covers what happens after, on the box, and it
is a different failure surface: CI can be green and the deploy still not land.

The deploy this project uses is `git reset --hard origin/<branch>` then `docker compose build` then
`up -d`. Three things can go wrong quietly:

**A failed build leaves the previous image serving.** If `build` is chained with `&&` and its output
is redirected - which is exactly how this was being run, `build web >/dev/null 2>&1 && ... up -d` -
then on failure the `&&` short-circuits, `up -d` never runs, nothing is printed, and the OLD
container keeps answering 200 with the OLD bundle. Every external signal says healthy. Capture the
build's exit status and print PASS/FAIL explicitly instead of hiding it.

**`docker ps` saying Up is not evidence of anything.** It reports the container, not the code in it,
not the network it joined, and not the commit it was built from. This project has already had a
container report `Up` while sitting on no network and 502-ing behind the shared proxy.

**The VPS silently drifts from origin.** A docs-only commit gets deployed with `git reset` and no
rebuild - reasonable at the time - and then the next web change is pushed while the box is two
commits back. Checked 2026-08-09: the VPS was at `3f44e89` while origin was at `89d2fca`.

| # | Check | Pass condition |
|---|---|---|
| DP1 | `git rev-parse --short HEAD` on the VPS vs `origin/<branch>` locally | identical strings |
| DP2 | The build's exit status is captured and printed, not redirected to `/dev/null` | an explicit OK/FAILED line |
| DP3 | Container age after deploy | seconds, not hours - a stale age means `up -d` reused the old container |
| DP4 | If web sources changed, the served bundle FILENAME changed | new `index-<hash>.js`; unchanged hash after a web change means the old image is serving |
| DP5 | If web sources did NOT change, the bundle hash SHOULD be identical | an unexpectedly new hash means something non-deterministic is in the build |
| DP6 | Site returns 200 AND the page renders, not just a proxy 200 | fetch a use-case route, not only `/` |

DP4 and DP5 are the same check read in both directions, and the second one is the one people skip. A
changed hash where nothing changed is as informative as an unchanged hash where something did.

Deploying from a red CI commit is not blocked by anything - see CI2 above - so DP1 only proves the box
matches origin, never that origin is good.

## Retired and orphaned Hugging Face repos

Weights live on the Hub at a pinned revision, so the Hub accumulates state the repository does not.
Two directions to check, and they fail differently:

| # | Check | Pass condition |
|---|---|---|
| HF1 | Every registry `hf_repo` exists on the Hub | no 404 - a missing repo breaks the model dropdown |
| HF2 | Every Hub repo under the account is referenced by the registry | no orphans, or the orphan is deleted |
| HF3 | Every published repo is PUBLIC | `private: false` - a private repo 401s for visitors while working for you |
| HF4 | Every model has a non-empty `hf_revision` | training clears it; see the retrain cycle above |
| HF5 | Retired models are DELETED from the Hub, not left dangling | an unreferenced public repo is a claim nobody maintains |
| HF6 | No file exists on the Hub that is absent from the local model dir | zero - a stale weights file from an earlier round still downloads for anyone browsing `main`, even though the demo pins a revision |

One command for HF2, which is the direction nothing else checks:

    curl -s "https://huggingface.co/api/models?author=<account>" \
      | python3 -c "import sys,json;[print(m['id']) for m in json.load(sys.stdin)]"

then diff against the registry's `hf_repo` values. **Checked 2026-08-09: 10 Hub repos, 10 registry
references, zero orphans, all public** - so nothing needed deleting this round. The check is recorded
anyway, because "nothing to delete" is a result and an unchecked Hub is not the same thing.

HF5 matters because a stale public model is worse than a missing one. Someone can download it, and
it carries this project's name while reflecting a version of the code that no longer exists.

## A model can score respectably and have learned nothing - check that it predicts more than one class

`medical_fl.student`, the CNN the encrypted medical path runs, emitted the SAME class for all 156
test images. Its accuracy of 0.730769 is exactly 114/156 - the majority-class rate - and its AUC was
0.354, below random. It shipped like that for several rounds, and the demo displayed its output as a
diagnosis.

Nothing looked wrong. Accuracy 0.7308 against a published 0.901 reads as "behind but plausible".
Agreement with the teacher was 0.75. The registry declared `auc` while the record stored `accuracy`,
which was on the plan as a cosmetic tidy-up item and was in fact the thing hiding the failure - AUC
would have shown 0.354 and been unmissable.

| # | Check | Pass condition |
|---|---|---|
| C1 | Count distinct classes the model predicts on the test set | more than one |
| C2 | Compare accuracy against the majority-class rate | strictly better, or the model is the majority class |
| C3 | Check the record contains the metric the REGISTRY declares, measured as that metric | present, or a note saying why it cannot be |
| C4 | For an imbalanced task, read recall on the minority class, not accuracy | reported explicitly |
| C5 | Verify class-label ORDER against the dataset's own metadata | matches; the browser renders `classes[argmax]` |

C2 is the cheapest and would have caught this instantly: accuracy equal to the majority-class rate,
to six decimal places, is not a coincidence. `n_classes_predicted` and `majority_class_rate` are now
recorded and rendered so the check is visible rather than remembered.

C5 caught a separate defect in the same pass. MedMNIST's own metadata says
`{0: malignant, 1: "normal, benign"}`; four places in this repo asserted the opposite, so the demo
showed the OPPOSITE diagnosis to what the model meant. The repo even contradicted itself - two
`eval_sample.json` files written by different code paths made opposite claims about which class is
cancer. **Never hardcode which index is the positive class.** The two MedMNIST corpora used here
disagree with each other (PneumoniaMNIST is `{0: normal, 1: pneumonia}`), so a constant that is
correct for one is silently wrong for the other; read it from the dataset's metadata.

**Accuracy going DOWN can be the fix.** After correction the student scored 0.6859, below the
majority rate, because it stopped answering with the majority class. On an imbalanced screening task
accuracy is maximised by not screening, so a drop there alongside a rise in minority-class recall
(1/42 to 8/42) and AUC (0.354 to 0.609) is the improvement, not a regression.

**Threshold tuning can degenerate the same way.** Tuning the cut on minority-class F1 picked the
extreme grid point and predicted the positive class for everything, because F1 is maximised by
calling everything positive when discrimination is weak. Use a criterion that cannot be gamed like
that - Youden's J gives all-positive a score of 0 - and reject any threshold that collapses to a
single class whatever it scored.

Workings: [`references/medical-student-collapse-investigation.md`](../references/medical-student-collapse-investigation.md).

## Ask what the metric was divided by, not just how many samples there were

The satellite card showed event-wise F0.5 0.4545 beside a published baseline of 0.786 for several
rounds. The scored window holds **3 labelled anomaly events**. The record beside it says
`n_anomaly_intervals: 320`, which counts the whole timeline, and `n_eval: 307601`, which counts
samples - so both numbers on the card invited reading the evaluation as large when its denominator
was three. One extra false-alarm run moves that score from 0.714 to 0.455.

Sample count is not sample size. For an event-wise, group-wise, or session-wise metric the
denominator is events, groups or sessions, and it can be smaller than the sample count by orders of
magnitude.

| # | Check | Pass condition |
|---|---|---|
| DN1 | For every headline metric, find what its denominator counts - samples, events, groups, patients | written down, not inferred |
| DN2 | Read that count from the record, not from `n_eval` / `n_train` / an interval count | the count is recorded explicitly |
| DN3 | If it is under ~10, the score is NOT comparable to a published benchmark | the UI says so on the row that shows the baseline |
| DN4 | Perturb by one unit and recompute | if one event moves it by tenths, the caveat is mandatory |
| DN5 | A metric that reproduces across retrains while the headline does not | suspect the stable one is stable because it is inflated |

DN5 is the one that pays. Point-adjusted F1 reproduced across retrains (0.9937 vs 0.9945) while
event-wise F0.5 swung 0.714 to 0.455. That is not evidence the point-adjusted figure is better - it
is the same fact from the other side, since it is computed over 441,600 points rather than 3 events,
and it is the metric TSB-AD showed a random score beats. Reaching for the reproducible companion
because it is stable trades a noisy honest number for a stable dishonest one.

**Guards: check the condition, not the existence.** `both_scored_zero` was written for exactly this
family of bug in an earlier round, and it did not fire here because it required BOTH teacher and
student to score zero, while the student scored 0.003234. The delta then went negative and rendered
as the 49-parameter encrypted student beating an 80,486-parameter LSTM. When a guard exists, verify
the condition covers every shape of the thing it guards against - a narrower condition will be
walked around eventually, and it fails silently because a guard that does not fire looks identical
to a guard that had nothing to catch.

Workings: [`references/satellite-event-count-investigation.md`](../references/satellite-event-count-investigation.md).

## An investigation is not landed until the string the UI renders says so

Round 24 disproved the recorded cause of the finance baseline gap, wrote the workings into
`references/finance-gap-investigation.md`, and left the note the browser actually renders unchanged.
It stayed live on the public demo for the rest of the round. The investigation was right and
invisible, which is the same as wrong for anyone reading the site.

The mechanism is worth naming because it generalises: **explanatory prose is generated by the
trainer, baked into `models/metrics.json` at training time, copied into the model card, exported
into `web/src/lib/config.defaults.json` and rendered from there.** Correcting the analysis document
touches none of those. Nothing fails; every test stays green; the demo keeps explaining itself with
the retracted reason.

| # | Check | Pass condition |
|---|---|---|
| N1 | For every claim retired or corrected this round, grep the SERVED bundle for a distinctive phrase from the OLD wording | zero hits |
| N2 | Grep the served bundle for a distinctive phrase from the NEW wording | at least one hit |
| N3 | Read the claim in the browser, on the page that renders it, after a cache-bust | the rendered text is the corrected one |
| N4 | If the prose is generated by a trainer, correct it THERE and propagate | trainer, `metrics.json`, model card, baked config and README all agree |

N1 is the one that catches this. A correction verified by reading the file you just edited proves
only that you edited it.

**Propagating into `models/metrics.json` is a surgical string replacement, never a JSON re-dump.**
A `json.dump(..., sort_keys=True)` once reordered 508 lines to change one sentence, which buries the
actual edit and makes the diff unreviewable. Replace `json.dumps(old_string)` with
`json.dumps(new_string)` in the raw text and assert the file still parses.

**Changing a model card means republishing.** The card is a file in the HF repo, so a corrected card
that is never pushed leaves the published artifact contradicting the demo. Republishing then moves
the revision, which means re-export, rebuild and redeploy - see the five-step retrain cycle above,
including its two traps.

## What this checklist cannot cover

- One browser, one adapter. Every latency here is from a software WebGPU adapter; a machine with a
  discrete GPU will differ, and nothing in this repo measures that.
- The health of a corpus. The checklist verifies that the number shown is the number measured, not
  that the corpus behind it is representative.
- Anything the page does not render. A metric with no reader passes every axis here, which is why
  the audit that finds those reads the code instead.

## Results from previous rounds

Moved to [`verification-log.md`](verification-log.md) in Round 24. Those sections were records of
what past rounds measured, not instructions for the next one, and they were 157 lines of a document
whose whole purpose is being executable in one sitting.

Read them when a number looks wrong and you want to know what it was last time.
