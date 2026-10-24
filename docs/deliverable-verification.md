# Deliverable verification

An independent research engineer's problem is not building the thing. It is being able to say, to
someone who has no reason to trust you, exactly which of your claims are checkable and how. This
document maps the acceptance ladder this project already runs on its code onto the deliverables the
work is meant to produce.

It exists because the methodology was strong and the mapping was missing. Twenty rounds produced a
detailed engineering gate and no statement of what "verified" means for a proposal, a benchmark
figure quoted to a funder, or a published model someone else might build on.

## The rule everything else derives from

> A result labelled MEASURED that produced no network or compute activity is the most serious
> defect class in this project.

Two corollaries, both of which have caught real defects:

- **A control that renders but changes nothing is a defect** - and so is one that changes the run
  while the architecture diagram above it stays still. Round 20 found five: a QKD protocol selector
  that switched the simulation while the diagram kept saying BB84; a satellite page whose four
  channel parameters had never been rendered at all, so the eavesdrop check it advertised could not
  fire; an OptiQKD checkbox that measured a 48.8% surrogate SKR gain and showed nothing; a Workload
  selector that ran a real encrypted convolution and drew the same arrow as a dot product; and an
  Inference selector with no visible effect anywhere.

  The method that found them is worth keeping: **set each control to each value and diff the
  rendered output**, rather than reading the code and reasoning about it. Four of the five had
  correct-looking code paths. The QKD one even carried a comment claiming the fix had been made -
  the `e91` branch existed and was unreachable, because the diagram was handed the crypto-agility
  family rather than the protocol.

  A **seventh** was found in Round 21, and it is the one this rule was written for: `qrng_eps_log2`
  had been *restored* by Round 20 on the stated grounds that it "moves the extracted bit count". It
  does not, at any value the spinner can reach. It moves the leftover-hash-lemma ceiling, and the
  delivered output is min(ceiling, requested) with requested almost always the smaller - so at 1024
  bits every eps from -1 to -128 returns byte-identical output. A comment asserting the control works
  is not evidence that it does, and this one had survived a round that specifically re-examined it.
  The ceiling is now reported, which is the observable the parameter actually moves.

  Fixing it surfaced a worse defect of the same family: at 64 bits and eps -128 the run delivers ONE
  bit and reported health PASS with min-entropy 0, because both SP 800-90B continuous tests return
  "passed" when handed too little data. A check that passes because there is nothing left to test is
  the MEASURED-with-no-work rule in its purest form. INCONCLUSIVE is now a third outcome.

  A sixth was found only by repeating the sweep on a **second view of the same state**. The
  Orchestration page's graph is memoised, and its `useMemo` dependency array listed `kem`, `qkd` and
  `sig` but not `qkdProtocol` - so the wiring was correct, the label function was correct, and the
  node still never changed. Code review cannot catch that: both halves read right. Two lessons
  carried forward: **audit every view that renders the same state**, not one per page, and **read
  the rendered node text, not the DOM you expect** - the first sweep queried `svg text` and returned
  nothing at all, because that graph renders React Flow HTML nodes rather than SVG.
- **A number must be either traceable to config or produced by a measurement in this session.**
  Anything else is a literal, and gets moved into `defaults.py` or deleted.

## The ladder

Eight gates. Steps 1-3 and 5-8 are pass/fail; step 4 is a loop that runs until clean.

1. **Static** - `pytest -m "not slow"`, `scripts/export_config.py --check` parity,
   `npx tsc --noEmit`, `npm run build`.
2. **Artefact** - for anything published: the int8 graph must agree with fp32 on at least 0.99 of
   the confident set and the fp32 graph must match its source model to 1e-4. A graph that misses the
   bar is withheld, not shipped with a relaxed gate.
3. **Environment** - `self.crossOriginIsolated === true` on the deployed page, or every timing the
   demo prints is coarsened and invalid. Zero `/api` calls in demo mode, verified in the network
   panel rather than by reading code.
4. **Content, in Chrome, every page** - four axes, iterated until clean. The executable form of this
   gate is the browser verification checklist - maintained privately, not in this repository (results recorded in `verification-log.md`, which is published): six route patterns over ten landable pages plus three negative paths (recounted 2026-08-13 against `App.tsx` and the deployed site; the previous figure disagreed with the checklist's own route table, and `/usecase/demo` - counted as a page by one round - renders "No such use case"), fourteen
   regions per use-case page, eight axes, and the numeric edges to type into every field. It is a
   file rather than a habit because what gets reconstructed from memory each round is whatever
   someone remembers to look at.
   - *Wording* matches the primary source. Never "experiment" for a simulation, never
     "information-theoretic" where it does not hold.
   - *Numbers* are config-traceable or session-measured.
   - *Controls* each produce an observable, distinct change, captured by diffing run metrics before
     and after.
   - *Provenance* is labelled, and a MEASURED label is backed by observable work.
5. **Container** - build both stacks, then re-run step 4 against the containerised build, because it
   can differ from the dev server.
6. **Live deploy** - re-run step 4 against the public URL. COOP/COEP, CDN reachability and cache
   behaviour all differ from localhost. Check the bundle hash agrees across the container, the
   public URL and `document.querySelectorAll('script[src]')` before trusting any browser result.
7. **Git-only deployment** - commit, push, then pull and build on the server. Never rsync.
8. **Final greps** - no new literals, no Japanese in public files, CI green on the head commit.

   As of Round 21 there is **no** Japanese in any tracked text file - the rule holds with no
   exceptions. The private business notes are Japanese by design and untracked; several live one
   directory above the repository and have never been inside it. They are deliberately not named
   here - a public file that lists the filenames of confidential ones has leaked something, even
   when the files themselves are absent.

   `.gitignore` used to be the exception, carrying five Japanese-named `.docx` patterns. They were
   guarding a path that does not exist, so they are gone and `*.[dD][oO][cC][xX]` replaced them:
   strictly stronger, since it catches those files plus any future report whatever it is called.
   Case-folded by hand because gitignore matching is case-SENSITIVE on Linux - a plain `*.docx` lets
   `Report.DOCX` straight through, which `git check-ignore` confirms. Verified against all five real
   filenames and four casings.

   Run the grep over TEXT files only (`grep -I`). Every research PDF under `references/` matches the
   CJK byte ranges, the way arbitrary binary does; a scan that includes them reports twelve
   violations and not one is real. A check that cries wolf twelve times is a check nobody runs twice.

## Mapping the ladder onto deliverables

### A published model

The claim is "this model scores X on Y". Verified when:

- the corpus licence was read at its URL on a recorded date, and permits redistributing derived
  weights (`DatasetSpec.licence_checked`);
- the metric is the specific one, not the flattering aggregate, and the aggregate is printed beside
  it so the gap is visible;
- the published baseline is named with its source, and the comparison is like for like - same
  resolution, same split, same training regime. Round 20 rejected four architectures on precisely
  this ground: AlexNet, VGG16, DenseNet-121 and EfficientNet-B4 are benchmarked on BreastMNIST only
  ImageNet-pretrained, which this project dropped over redistribution terms, so their numbers cannot
  be compared with a from-scratch run;

  **SqueezeNet is declined too, and for a stronger reason - recorded 2026-08-13, having been the one
  item left open across 29 ORDER files.** The request was conditional: add SqueezeNet, DenseNet-121
  or AlexNet *only if they are recommended as benchmarks*. It slipped because the four-architecture
  sentence above enumerates MedMNIST+'s benchmark set, and SqueezeNet is not in it - so the item
  never matched a search for a decision that had been made about the others.

  Checked at the source rather than inherited: **MedMNIST+ (arXiv:2404.15786) evaluates ten models -
  VGG16, AlexNet, ResNet-18, DenseNet-121, EfficientNet-B4, and five ViT-B/16 variants - and
  SqueezeNet is not among them.** Two independently framed reads of the paper's HTML agree, and its
  own sentence names the convolutional set as "VGG16, AlexNet, ResNet-18, DenseNet-121, and
  EfficientNet-B4, all of which were pretrained on the ImageNet1k dataset". An internal planning
  note had recorded SqueezeNet as being in that set; it is not.

  So the four rejected architectures have a published BreastMNIST comparator this project cannot use
  (wrong training regime). SqueezeNet has **no published BreastMNIST comparator at all** - none in
  MedMNIST+, and none cited anywhere in this repository's references. Training it from scratch would
  produce a number with nothing to compare it against, which is the definition of a figure that is
  not a deliverable here. The ORDER's condition is therefore not met, and its own fallback applies:
  keep the ResNet-50 plan. **Decision: declined, no repository change.**

  Not re-verified today: whether MedMNIST v2's own baseline table contains SqueezeNet. Its abstract
  names only "2D / 3D neural networks and open-source / commercial AutoML tools" without
  enumerating them. What IS established is that no SqueezeNet BreastMNIST figure is cited anywhere
  in this project, so there is nothing here to compare against either way;
- the artefact passes gate 2 and is pinned to a commit sha, never a branch;
- a browser reproduces the offline figure on the same rows, and says so when it does not.

**A negative result is a deliverable.** ResNet-50 scoring 0.7988 against ResNet-18's 0.9147 is a
reproduction of a published finding, not a failure. So is `quantum_beats_classical: false`.

### A label on a diagram

A diagram is the most-looked-at part of a page and the least likely to be tested. Verified when
every control that alters the run alters what the diagram says, checked by diffing rendered text
across each control's values - and when a figure it shows names the model that produced it. Round 20
found `qkd_optimized_skr` printed beside `skr` where the two came from *different models*: the
optimiser's analytic surrogate against the decoy-state simulation. The pair read as the optimiser
making things worse, 0.0035 against 0.00906, while the same panel claimed +48.8%. Both ends of the
ratio are now emitted under names that say which model produced them, so the percentage is checkable
in place.

### A model that runs under encryption

Verified when the encrypted path evaluates the **published weights**, not an architecture of the
same shape, and when the run says which of the two it did. Round 20 closed this for medical: the
convolution ran on the student's shape with a stated kernel pattern until the student began
publishing a structural `weights.json` its ONNX could not supply.

Two preprocessing traps surfaced doing it, both of the same family - a missing step that produces
plausible output rather than an error:

- The exporter's first version inferred activations from layer types and dropped the ReLU between
  the conv and the pool. The result reproduced **every argmax** on the 156-image test split while
  its logits were off by 1.5e-01. Activations are now read from the trained module order, and an
  independent re-implementation built from the JSON alone agrees with torch to 4.4e-08.
- The linear students published coefficients but not the `StandardScaler` they were fit on, so
  their graphs accept raw features and answer confidently wrong. That is why they had no evaluation
  sample: generating one naively would have shipped the wrong number instead of an error.

### A benchmark figure quoted to a third party

Verified when the number is reproducible from a stated command on stated data, its baseline is
named, and its unit is what it claims. A latency that is a measured unit cost multiplied by a
modelled op count is an estimate, and must say so. Round 20 removed one: the CNN workload multiplied
a dot product's latency by about 401 and presented the result as measured; it now performs a real
encrypted convolution and reports the rotation count it actually executed.

### A protocol implementation

Verified when the construction's own constraints are checked, not merely quoted. QSMS is defined
over $`GF(d)`$ with $`d`$ prime and $`n \le d \le 2n`$; the shipped config pinned $`d = 4099`$ for $`n = 7`$, which
satisfied "prime" and "exceeds n" and nothing else, and silently disabled the exact statevector
simulation the UI described. Where two of a construction's requirements genuinely conflict - here,
the field bound against the sum not wrapping - the run records which trade it made rather than one
being imposed invisibly.

### A funding or grant claim

Every figure traces to a gate above, and every limitation the gates exposed is stated in the same
document as the claim. The asymmetry is the point: one unsupported number invalidates the reader's
trust in the supported ones, which is expensive precisely because most of them are good.

## The backlog gate

The deferred list has grown for three consecutive rounds: `results/` went 953 files to 1133,
`REFERENCES.md` provenance columns were promised in three plans before landing, and Prometheus went
from "add a dashboard" to "charts nothing" to "zero call sites".

So a round is not complete when its new work passes. It is complete when:

- every item it defers is recorded with the reason and the condition that would resolve it;
- no item has been deferred three rounds running without either being done or being reclassified as
  a decision. Round 20 reclassified two: `finance_aml.gnn` and `telecom_ids.transformer` are not
  backlog, they are documented decisions - the finance GNN is the weaker model on AMLworld and its
  FHE path is published as non-compiling, and the telecom corpus has unresolved redistribution
  terms;
- the count of deferred items did not increase.

**Round 32 failed the first and third clauses at once - three items closed, at least six opened with no
condition - and the audit that repaired it is in "Round 33" below.** Read that section before adding
anything to a deferred list here, because it is where the conditions live and where four items were
reclassified for failing `QC16`.

**Round 21 recorded two failures against this gate and then closed both, rather than carrying them
into a fourth round.** They are kept here as the evidence for why the gate exists, in the past tense
they now belong in.

`results/` had gone 953 to 1133 to 1421 files across three rounds with no retention mechanism
anywhere in the tree. It is gitignored, so this was local and VPS disk rather than repository bloat -
which is exactly why it stayed easy to defer rather than easy to notice. Closed:
`orchestration.keep_runs` (default 200, 0 disables) prunes on persist, grouped by run stem so a run
is kept or removed whole. Measured after: 600 files, 200 runs.

Prometheus had zero call sites in its third consecutive round. `monitoring/exporter.py` defined four
metrics and three recorder helpers with no caller outside that file, and `BENCH_LATENCY` had no
reader at all, not even inside it - so `/metrics` served a valid, permanently empty scrape and the
Grafana dashboard charted it. The gate's own escalation had run "add a dashboard" to "charts nothing"
to "zero call sites". Closed: the recorders are called from `orchestration/pipeline.py` per stage and
per run, and from `benchmark/runner.py` for the path label `BENCH_LATENCY` was always going to need.
Measured: the scrape goes from 1 series to 39 after one run, and `tests/test_monitoring_wired.py`
asserts that production code calls them rather than that the helpers exist.

Also closed in Round 21: the EU AI Act Art. 50 disclosure, which had become a live obligation five
days before the work started, and the liboqs version mismatch, which turned out to be worse than
recorded - the Dockerfile cloned the default branch, so two builds of the same commit of this repo
could ship different cryptography with nothing recording which. Pinned to 0.16.0.

## Decisions, so they stop being backlog

The gate says an item deferred three rounds must be done **or reclassified**. Round 22 audited the
ledger and found three that had been carried without either, plus one that silently left the list.
Reclassifying is not a way of avoiding work - it is the honest label when the answer is "no", and
these four had answers.

**WebGL: declined, and it will not be implemented.** Six ORDER files asked for the demo to compute
client-side "using JavaScript, WebAssembly, WebGL/WebGPU", and Round 21 explicitly scheduled this
written decline and then did not write it - so it survived six requests and one plan that named it.
The reason it was never built is that it cannot be: **ONNX Runtime Web ships WASM and WebGPU
execution providers, not a WebGL one**, and Transformers.js is the same. The instruction listed WebGL
and WebGPU as alternatives, and this project took the one that exists. WebGPU is implemented, probed
for a genuine hardware adapter, and declined when the adapter is a software rasteriser. Nothing is
missing; the list simply named a technology the runtime does not offer.
*Reopens if* ONNX Runtime Web ships a WebGL execution provider - a fact about the dependency, so it
is checked by reading that release's provider list, not by re-arguing the decision. "It will not be
implemented" is a statement about the runtime's capabilities and expires the moment they change.

**ZeroGPU Space: blocked on a date, not on work.** Four consecutive rounds, the longest-carried item
here. `spaces/qsh-inference/` exists, and `ProvenanceBadge` declares a `remote` tier no code path
sets. **BLOCKED UNTIL 2026-09-01**, a Hugging Face account-age unlock. **DEADLINE EXECUTED Round 43**
- and the discharge is that the decision was already taken and never stamped here. "ZeroGPU: DECIDED
2026-08-10, and the decision is not to wire it" is recorded further down this file; this paragraph
kept its blocker unstamped, so `tests/test_register_deadlines.py` would have gone red on 2026-09-01
on wall-clock alone, with no code change and nothing left to decide. Seventeen days of warning,
found by reading the gate rather than by waiting for it. That is a condition, not a
task, so it is recorded as one: **when the account unlocks, wire it or delete the Space and the
`remote` tier together.** It must not appear in a fifth deferred list unqualified.

Round 39 added the marker so the date is enforced rather than remembered:
`tests/test_register_deadlines.py` fails from 2026-09-01 with no code change, which is the only way
a gate can fire on an item whose defining property is that nobody is editing it. Four rounds of
"still blocked" happened because the blocker was a sentence. **Do not move the date** when it goes
red - wire the Space or delete it and the `remote` tier, then stamp this paragraph. Moving the date
is how the arXiv deadline in item 11 reached Round 39.

**OpenFHE `EvalBootstrapSetup` SIGSEGV: THE REOPEN CONDITION HAS BEEN MET. Round 55.**
Measured on this EL9 host with `QSH_REQUIRE_OPENFHE_BOOTSTRAP=1`, three consecutive runs:
**16.85 s, 16.05 s, 16.82 s, all exit 0.** The crash does not reproduce. The paragraph below
states its own reopen condition as "if the crash stops reproducing on the pinned OpenFHE
version on EL9", and nothing was watching for that - a condition with no executor, which is
the shape this round kept finding. `scripts/ci_local.sh` now runs the test as step 17, so the
condition is checked every round rather than remembered.

This also closes the gap that the script's header claimed away: ci.yml runs this test as a
blocking job and step 1 here deselects it with `-m "not slow"`, so "runs every blocking CI
gate" was false for exactly one job, and nothing compared the two step sets. The original
decision is kept below because it was right when made and the failure mode is host-specific -
if EL9 starts segfaulting again, step 17 names which platform changed.

**OpenFHE `EvalBootstrapSetup` SIGSEGV: permanently worked around, on purpose.** Three rounds. The
segfault is inside OpenFHE on AlmaLinux 9.7, and `fhe/backends.py` guards an `ImportError`, which a
segfault is not. `make test` deselects it with `-m "not slow"`; `make test-all` still dies. Fixing an
upstream crash on one host distribution is not this project's work, and the workaround is the
outcome rather than a placeholder for one. Recorded as a decision.
*Reopens if* the crash stops reproducing on the pinned OpenFHE version on EL9 - which CI already has
standing evidence about, since the workflow runs that one deselected test on purpose with
`QSH_REQUIRE_OPENFHE_BOOTSTRAP=1` on ubuntu-24.04. Two platforms, one of which passes, is the whole
reason this is a host-specific workaround rather than a missing feature; if the EL9 host starts
passing, the deselection is dead weight and `make test-all` should stop dying.

**node-seal is on SEAL 4.1.2, and node-seal is the constraint, not us.** Promised in Round 19,
tasked again in Round 21, never written, and then it dropped off the list without resolution. The
answer: `web/package.json` pins `node-seal ^7.0.0`, 7.0.0 is installed, and node-seal 7.0.0's own
package metadata states SEAL 4.1.2. **Re-checked 2026-08-09** - node-seal 7.0.0 (2025-11-08) is
still the newest node-seal on npm, so this is not a stale pin here.

The wording this file carried until Round 23 - "and that is upstream's newest" - was ambiguous in a
way that flattered the position. It is true of node-seal and false of SEAL. Measured: Microsoft SEAL
has shipped **six releases since 4.1.2** and reached **4.4.3 on 2026-08-04**.

**Round 32 correction, and it was wrong in the direction that matters.** Until now this paragraph said
4.4.0 was "described only as 'Major security update', with no further detail". That is true of the
GitHub *release body* and false of the changelog. `CHANGES.md` on `main`, read at the primary source
2026-08-11, gives six bullets and the first is stronger than "Major":

> **Critical security update:** all users should upgrade to Microsoft SEAL 4.4.0 as soon as possible.
> Hardened untrusted deserialization. Restored loading of objects serialized by earlier Microsoft SEAL
> 4.x minor releases; the wire format is unchanged. Fixed native memory-safety and concurrency defects
> across ciphertext evaluation and memory management. Hardened the C ABI and .NET wrapper against
> various failures. Strengthened cryptographic input validation and release-build/dependency security
> controls.

So this register **under-stated a critical advisory in a dependency the browser demo ships**, which is
the worse direction to be wrong in, and it did so while `report/technical-report.tex` described it
correctly - the one place that had it right was the one an audit flagged as overclaiming. The lesson is
narrow and practical: **a GitHub release body is not a changelog.** Read `CHANGES.md`.

The correction also sharpens the WASM question rather than leaving it open. 4.4.1 fixes a MinGW build
and 4.4.3 fixes null-output-pointer dereferences and C++ exceptions leaking across the **C API**
boundary (issues #753-#755) - and the hedge "whether C API fixes reach an Emscripten/WASM build is
unverified" is correct for those. But **untrusted deserialization and ciphertext-evaluation memory
safety are core C++**, not the C API, so 4.4.0's two most serious items **do** reach a WASM build. The
hedge applies to 4.4.3, not to 4.4.0, and applying it to both was how the more serious release came to
read as the less consequential one.

What this does and does not mean for the demo is in `docs/threat-model.md`: `clientFHE.ts` deserialises
only ciphertexts it produced in the same tab, so the hostile-input precondition for the deserialization
hardening is absent here. Round 19's instruction was to document the gap rather than claim a
vulnerability, and that still holds - a GitHub Security Advisory query for SEAL returns **no
advisories** and no CVE exists. The version and release history are recorded as facts, the gap is
recorded as widening, and no vulnerability is claimed.

**Re-check this on any round that touches the FHE path.** Round 31 touched `requirements-fhe.txt` and
did not honour that trigger, which is how a three-week-old critical advisory went unread; the cadence
since April 2026 is roughly monthly.

### Is a bump available? Measured in Round 32, and the answer is yes

Round 31 declined FIDESlib on a build outcome rather than a README claim. The same method applied here,
because "node-seal is the constraint" had been asserted for several rounds without anyone trying it.

What node-seal actually is: `.gitmodules` points `submodules/SEAL` at **`s0l0ist/SEAL`**, a fork with
zero tags and one branch, pinned at commit `dd3c6548`. Diffed against `microsoft/SEAL` v4.1.2, the fork's
divergence is **purely additive** - `native/src/seal/bindings.cpp` (1,005 lines of Emscripten embind),
`base64.h`, CMake glue, and bumped bundled ZLIB/Zstd. **It modifies no SEAL crypto source.**

So the test was: take `microsoft/SEAL` v4.4.3, add the fork's binding layer, build to WASM.

| step | result |
|---|---|
| configure (emcmake, mirroring node-seal's own `seal-cmake.sh`) | exit 0 |
| build (emmake, emcc 6.0.6) | exit 0, **0 errors** |
| `bindings.cpp` object | `build_final/.../bindings.cpp.o` produced |
| library | `libseal-4.4.a`, 4.9 MB |

**The embind layer compiles unchanged against SEAL 4.4.3's public API.** That is the question that
mattered, and it is answered.

**The one blocker found is not SEAL's, and it is a build flag.** The first attempts died on
`native/src/seal/util/defines.h:84: error: "Unsupported system"`. That guard tests for the bare
`EMSCRIPTEN` macro, and it is **byte-identical in 4.1.2 and 4.4.3** - so this is not a regression. The
cause is upstream Emscripten: emcc 6.0.6 defines `__EMSCRIPTEN__` and **not** bare `EMSCRIPTEN` (verified
with `emcc -dM -E`). Passing `-DEMSCRIPTEN` fixes it. Two consequences worth separating:

- A node-seal bump to 4.4.3 is a **build-configuration change plus a submodule re-point**, not a porting
  exercise.
- node-seal's own build would fail the same way on a current emsdk **at 4.1.2**, because it does not pass
  that define and relies on a deprecated macro. Its nine months without a release is not only a
  maintenance signal; the build is pinned to an emsdk era.

**What was NOT verified, so it is not claimed:** a CKKS encrypt/evaluate/decrypt round trip through the
linked WASM module. This built the static library and the bindings object; it did not produce and exercise
the full `node-seal` JS package. That is the remaining step before any bump ships, and it is the step that
would catch a behavioural difference the compiler cannot see.

Reopening condition, now concrete rather than "wait for node-seal": either upstream node-seal cuts a
release on SEAL 4.4.x, or this project carries the three-line delta (fork re-point, `-DEMSCRIPTEN`,
rebuild) and passes a round-trip test. The second is now known to be reachable.

**Genomics has no label, so it has no QCNN: declined, and not for the reason recorded for two
rounds.** The deferral condition on file was the per-score licence audit. Round 31 checked what would
actually happen if that audit came back clean, and the answer is nothing: `scripts/train/qcnn_variational.py`
is supervised end to end - a labelled loss (`BCEWithLogitsLoss` on two classes, `CrossEntropyLoss`
on more), best epoch chosen on a validation split, and a report block comparing against whichever
baseline the regime warrants (`all_positive_f1`, `majority_baseline`, or the macro-F1 floors) - and
**1000 Genomes is a reference panel with no breast-cancer case/control status**. `registry.py` already says so, and
`genomics_prs.py` records `auroc=null` rather than inventing one. There is no `y` to train against.

Two ways to manufacture one, both rejected on their merits rather than on effort:

- **Regress the PRS itself.** The target would be a weighted dot product of the input dosages, which
  is precisely what the CKKS path already computes exactly. The classical logistic-regression
  baseline in the same script would win by construction, and would be right to. Fitting a
  variational circuit to reproduce a linear function of its own inputs measures nothing.
- **Dichotomise at the high-risk cut.** That derives the label from the model's own output, so the
  circuit would be scored on agreeing with the thing it was given. It also inherits the finding this
  use case exists to show - that one globally-set cut flags a far larger share of AFR than EUR - and
  would relabel a known ancestry artefact as ground truth.

So the honest scope is: finance and medical have trained QCNN weights because they have labelled
corpora; genomics does not and will not until a labelled cohort with usable terms exists, which is a
data-access question and not a modelling one. That is a stronger trigger than the licence audit,
because a licence to redistribute a reference panel still would not put a phenotype in it.
`tests/test_qcnn_usecases.py` continues to compare the availability set against the files on disk in
both directions, so if weights ever do appear the claim cannot go unnoticed.

**WS4(f), hiding the circuit from the client: not claimable on this machinery.** Moved here from the
deferred list, because it is not a backlog item - it is in tension with the mechanism. Under plain
QOTP the client decrypts by applying the circuit's action to its own Pauli frame, so "the client
updates its frame" and "the client does not learn the circuit" are close to mutually exclusive.
`qcnn.py` emits `hides_circuit_from_client: False` and `tests/test_qhe_fhe.py` asserts it is False, so
claim, artifact and test agree. Earning it would mean implementing the Ortega and Martin-Delgado
construction, which this repository does not. WS4(c) is met for Clifford circuits and is claimed
separately, so (f) no longer rides along with it. See also "What this project cannot verify".

**Rust and tfhe-rs in the client: DECLINED, and it should have been recorded three rounds ago.** Asked
for explicitly in one ORDER file ("if tfhe-rs is implementable on WebGPU, introduce it; Rust is
acceptable as well as JavaScript for client-side processing") and then listed among the required
client-side technologies in two more. It was never done and never declined - the repository has no
`Cargo.toml`, `tfhe-rs` appears only in licence tables, and the private checklist contained the
whole-word `rust` **zero** times in 2,748 lines. That is the "items that vanish without a decision"
failure this gate exists to catch, and it is worse than a stale deferral because there was nothing on
any list to go stale.

Two independent reasons, either sufficient:

- **It cannot do the thing it would be adopted for.** Zama's own TFHE-rs WASM documentation states the
  build supports key generation, encryption and decryption but **not FHE computation**, and IACR ePrint
  2026/515 defines "client-side operations" the same way while the server evaluates. Adopting Rust to
  get in-browser FHE evaluation would produce a client that encrypts and decrypts - which is what the
  demo's PQC path already does in TypeScript, and less than what its CKKS path already does through
  `node-seal`. There is no capability to gain.
- **The licence reason this project had recorded was imprecise, and the accurate one is sharper.**
  `references/REFERENCES.md` said "the commercial terms for the tfhe-rs family require a paid licence".
  Read at the primary source, tfhe-rs is **BSD 3-Clause Clear**, whose distinguishing clause is "NO
  EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE". The copyright
  grant is permissive; the **patent grant is expressly withheld**. That is a stronger objection for a
  project whose rule is commercial-OK dependencies, and it is checkable in one file rather than being a
  claim about a vendor's pricing.

So Rust is not missing from the client-side stack, it is declined, and WASM is present without it -
`node-seal` (SEAL through Emscripten) and `emp-wasm` are both WebAssembly. The condition that would
reopen this: Zama shipping FHE **evaluation** in the WASM target, or any Rust FHE library doing so under
a licence that grants patent rights.

### Additional models: DECIDED, re-stated 2026-08-15 (Round 43) - no additional model

**Moved here from `docs/browser-verification-checklist.md` in Round 34, and the move is the point.**
That file is deliberately untracked and gitignored, so a deferred item whose resolving condition was
"add a sixth row to the trigger table" could not be verified from the repository at all. A condition
that lives outside the artifact it gates is decoration, by this project's own `QC16` test. Worse, the
`medical_fl.student.baseline_gap_note` in `models/metrics.json` asserted that a latent-encoder trigger
had been "recorded as a falsifiable trigger in the additional-model decision" when the table had five
rows and none concerned latent-space encoding - a claim appealing to a record that did not contain it,
made MORE specific in Round 33 and then filed by Round 33 as a defect against itself. The decision now
lives in a tracked file, so the appeal is checkable and the sixth trigger below is real.

Not an HF download, not fine-tuning, not transfer learning, not RL, not training something different
from scratch. The binding constraint is cryptographic rather than statistical: CKKS at ring degree
8192 gives 4096 slots and one multiplicative level per ciphertext-plaintext dot product, and anything
deeper bootstraps or switches scheme. The 2026 literature holds the model class fixed and moves the
protocol, which is the same choice this makes.

`AM1` in the checklist requires this to be re-stated with its date every round. It read "2026-08-11
(Round 31)" through Rounds 32 and 33 - two rounds during which a decision taken once read as current
because nothing made the staleness visible. Keeping it in a tracked file is what lets a reviewer see
the date go stale.

**The six triggers that would flip it**, each with the measurement that currently blocks it:

| trigger | current state |
|---|---|
| A bootstrapped CKKS CNN cheap enough for WASM at N about 8192 | cheapest bootstrapped ImageNet CNN is N = 2^15 |
| A real, labelled, commercial-OK QKD attack corpus | closest is QF-LOG: synthetic, four classes, subscription-gated |
| A fidelity quantum kernel beating a random forest at 8 qubits or fewer | **NO LONGER BLOCKED - Round 55.** The blocker read "concentration bound says >= 15 qubits"; arXiv:2605.31449 says 15 is where the fidelity kernel FAILS and that concentration worsens with qubit count, so 8 qubits is the favourable regime. This falsifier is live and unblocked |
| An encrypted transformer or tabular foundation model on any finance tabular benchmark | none published |
| Anyone publishing an encrypted BreastMNIST result above this project's | two exist, both simpler architectures |
| **A latent-space encoder whose latent fits 4096 CKKS slots at multiplicative depth 1 AND runs under `onnxruntime-web`** | Kim et al. (arXiv:2506.15258) encode chest X-rays with a VQGAN latent at downsampling factor 8, but f=8 on this project's 28x28 gives 3.5x3.5, so the transfer is the open measurement rather than a citation. Added Round 34 |

The sixth is the one `metrics.json` was already claiming existed. It is now here, with the condition
stated as a measurement someone could take rather than as a direction someone could agree with.

## Round 28: a claim retracted, and the work it would take to earn it back

**SUPERSEDED IN PART BY ROUND 30, marked in Round 33, and this is the one place in the project where the
DOCUMENTATION was weaker than the IMPLEMENTATION.** Everywhere else the risk here runs the other way, so
an unmarked supersession in this direction is worth naming rather than quietly editing. Round 30 earned
**WS4(c) back for Clifford circuits** and `README.md` has claimed it since; the retraction below, and
the standing prohibition at the end of this section, were left unmarked for three rounds. A reader
checking whether this project may cite `qotp.py` for circuit privacy therefore found a rule forbidding
exactly what the README already does, and the rule looked current because nothing said otherwise.

**Current status, so it is not reconstructed from the history below.** WS4(c) is **MET for Clifford
circuits**: a random Clifford word over {H, S, CNOT} is evaluated on the QOTP-encrypted state, the
client tracks the Pauli frame through the same word, and the plaintext result is recovered at fidelity
1.0, verified to depth 30 across seeds - `qhe/qcnn.py::qotp_clifford_random_circuit_demo`, with the frame
rules in `qhe/qotp.py::update_key_h/s/cnot`. WS4(f) remains **NOT met**, and is a decision rather than a
backlog item, because it is in tension with the mechanism instead of merely unimplemented - see the
bullet in "What this project cannot verify". The Round 28 text is kept in full below, because it is the
evidence for why the claim was retracted in the first place and for what the tautology looked like.

**Circuit privacy (WS4 items c and f) is NOT demonstrated, and the function that was cited for it
applies no circuit.** `qhe/qotp.py` carried `pauli_concealment_demo`, whose docstring said it
returned "fidelities proving the result (and hence the server's circuit effect) is hidden".
Reading the six lines of the body: `server_out = encrypt(psi, a, b)`. That is the INPUT state
under a mask. No gate, no unitary, therefore no circuit effect to conceal. `README.md` cited it as
the WS4(c)/(f) evidence and a passing test asserted the verdict, so the claim had three
independent-looking supports and no measurement underneath any of them.

The second half is worse than the first. `fidelity_correct_key` compared `decrypt(encrypt(psi, a,
b), a, b)` against `psi`, which is 1.0 for every input and every key by construction - and it was
half of the `circuit_private` predicate, so half the verdict was carried by a term that could not
fail. `tests/test_round2.py` asserted both the 1.0 and the verdict. This is the same tautology
this project found in `qhe/private_inference.py`, where `plain_probability` was set to the same
value it was compared against; that one was fixed in Round 25 and this was the last instance.
**When a tautology is found in one place, the remaining instances are not hypothetical - they
should be searched for by shape rather than waited for.**

What is retracted: the function is renamed `pauli_output_concealment_demo`, the tautological term
is excluded from the verdict and renamed `roundtrip_fidelity_selfcheck`, the payload carries
`applies_a_circuit: False`, and the README row now says PARTIAL and names what is missing. QOTP
input hiding is untouched and remains real.

**The condition for earning the claim back**, recorded so this does not become an item that stops
being mentioned: apply an actual circuit on the server side and propagate the Pauli frame through
it. That is well defined for **Clifford** circuits, where a Pauli conjugated by a Clifford is a
Pauli, so the client can update its frame and decrypt exactly. It breaks at **T**, since `T X
T-dagger` is not Pauli - which is precisely the difficulty Ortega and Martin-Delgado's scheme
addresses and which this repository does not implement. A Clifford-only demonstration would be
honest and worth having; until it exists, no document here may cite `qotp.py` as evidence of
circuit privacy.

**That prohibition was SATISFIED, not broken, and it is lifted for the Clifford case as of Round 30.**
The Clifford-only demonstration this paragraph asked for is what Round 30 built, so `README.md` row (c)
citing `qhe/qotp.py::update_key_h/s/cnot` alongside `qhe/qcnn.py::qotp_clifford_random_circuit_demo` is
the permitted case and not a violation. The prohibition **stands unchanged for anything containing T**,
for the reason this paragraph already gives: `T X T-dagger` is not Pauli, the frame does not propagate,
and no document here may cite `qotp.py` as evidence of circuit privacy for a non-Clifford circuit. It
was left unmarked for three rounds - Round 33 marked it.

**Two adjacent labels corrected at the same time**, both of the same kind - a true statement about
one artefact applied to a different one. `README.md` and `docs/architecture.md` called
`qhe/qcnn.py` a "Clifford+T QCNN"; that module contains no T gate at all (`AngleEmbedding` then
CNOT/RY/RZ) and its `t_count()` is a labelled Solovay-Kitaev estimate of a fault-tolerant
deployment's cost. The label is accurate for the BROWSER circuit in `web/src/lib/quantumSim.ts`,
which is built from H, CNOT and T and counts 56 of them. And `README.md` published `1.0` and
`1.2769` for what read as one ct/pt ratio; they measure qubits and bytes respectively, and the
note now says so where the number appears.

**A genomics control that announced a result it did not produce.** Selecting `Inference = QCNN` on
the genomics page rendered "the class this page surfaces comes from the quantum circuit" while no
trained weights ship for genomics, so CKKS had produced it. Two panels read the control value
instead of `artifacts.primary_inference.backend`. Fixed, and the option is now hidden where no
weights exist, since this project's rule is that such a control is removed or made real.
**Genomics QCNN training was deferred with a condition**: revisit alongside the per-score licence
audit that already blocks `genomics.prs` publication, because both turn on the same corpus
questions. `tests/test_qcnn_usecases.py` compares the availability set against the files on disk
in both directions, so the deferral cannot quietly become a stale list.

**Round 31 reclassified it, because that condition was not the binding one.** See "Genomics has no
label, so it has no QCNN" below: the licence audit could be satisfied in full and the item still
could not proceed.

## Round 29: an ignored file is not a protected file, and a staged score

**A round's worth of edits to the verification checklist was lost, and the commit message
describing them was already in the repository.** Round 28 untracked
`docs/browser-verification-checklist.md` and gitignored it, then continued editing it: a
round-trip section, four run-sheet index rows, four QCNN rows, nine consolidated duplicates and
two legend fixes. Because the file was already untracked, none of that was committed - while the
commit message for that work describes it as done. The generated report macros are the only
surviving evidence it existed (191 rows, 2,522 lines, against the 175 and 2,443 of the last
tracked copy).

The mechanism, reproduced rather than guessed: `main` at `2f0d550` still TRACKED the file, so `git
checkout main` restored the tracked copy over the ignored working one, and `git pull` past
`9d35501` - the merge that untracked it - then deleted the result. **Ignoring a file protects it
from being committed, not from being overwritten.** Any branch switch through a commit where it
was still tracked will clobber it.

Recovered to the last tracked state and the structural edits redone. A copy now lives outside the
repository and `.gitignore` carries the hazard beside the rule that caused it. The nine duplicate
consolidations were the part not yet redone at the time this was written.

**They were done in Round 30** (commit `dd9c9b0`), and the re-audit found **twelve** pairs rather than
nine. This paragraph went on asserting them as outstanding, with the superseded count, for two rounds -
which is the defect this same file diagnoses elsewhere as "a register of open items needs the same
staleness check as any other document". Round 31 found the mirror image of this in the checklist, where
a paragraph claimed a consolidation that had **not** happened, and did not check the ledger for the
inverse. Both directions are the same failure: a status line that nobody re-reads because it reads as
settled.

**Two commit messages have now described checklist work that was not in the commit.** That is the
same class as a stale document asserting a fixed item is open: the artefact and the description of
the artefact drift apart, and the description is the thing people read.

**PGS000004 / PRS313 was staged, not switched - and this paragraph went on saying so for two rounds
after it stopped being true (corrected in Round 33).** What it said, in the present tense: the
313-variant instrument and its position-keyed loader path were in place while "both the Python pipeline
and the browser still score PGS000001 / PRS77", and the switch was being taken in one step because a use
case scoring PRS313 in Python while displaying PRS77 in the browser "would be worse than either alone".

**The switch completed in Round 31 for four surfaces and in Round 32 for the fifth.** PGS000004 /
PRS313 is ACTIVE for the pipeline, the browser payload, the trainer, the registry and the generated
payload; the score in force is the config field `DataConfig.genomics_prs_score` rather than a default
written into any call site, and `tests/test_prs_pipeline_parity.py` runs the use case and compares its
output against the payload the registry points at. PGS000001 / PRS77 is RETIRED and kept only because
its dosage file is the one small enough to commit in full, so it is what CI exercises for the
rsID-keyed code path.

**Two things are worth keeping about how this paragraph aged, because they are different failures.**
For one round the shipped state was the exact INVERSE of the warning - PRS313 in the browser, PRS77 in
Python - and the sentence arguing against that split was still sitting here while the split was live.
Then Round 32 fixed this same sentence in `src/qsh/data/loaders.py`, wrote a comment block recording
that it had been wrong, and did not check whether any other copy existed. **A correction that knows it
is correcting a sentence should grep for that sentence**, which is the cheapest possible check and was
not run. That is the third instance in this file of a stale open-items entry, after the two in "CLOSED
in Round 29" and the consolidation count in Round 29 itself.

What the migration required, because it is the interesting part: PGS000004 publishes **no rsIDs at
all** - every one of its 313 harmonised rows has an empty `hm_rsID` with
`hm_source=Author-reported`. Reverse-mapping was measured on 40 positions against Ensembl GRCh37
before any code was written: 70% carry exactly one rsID, 28% carry several, allele matching
resolves about half of those, and roughly 15% would have had to be guessed - which scores a
different variant's genotypes while every number still looks plausible. Keying on `chr:pos` with
an allele match against the phase-3 VCF matched **313 of 313, none unmatched**, indels included.
The remaining steps, so they are not rediscovered: a gnomAD allele-frequency fetch by coordinate
for the ancestry overlay, a regenerated browser payload, and a `MetricSpec` change from C-index
0.628 (PPM017270) to **AUROC 0.633 [0.624, 0.641]** (PPM001647) - the latter measured on 400,812
Europeans **with covariates** (family history, genotyping array) that this repository's bare dot
product does not have, and with **8% of variants missing** in that cohort. It is a citation, not a
like-for-like target, and must be recorded as one.

The licence posture does not change: PGS000004 carries the same EMBL-EBI default terms as
PGS000001, so the migration neither unblocks nor further blocks publication, and `hf_repo` stays
None.

## Round 30: a licence condition only Hugging Face could fix, and a claim the project hid from itself

**The Hub was serving a retracted attribution and no local change could reach it.** The corrected
OPS-SAT citation had been in the tree and the baked config for two rounds. The browser fetches
each model's `eval_sample.json` from the Hub at its pinned revision and interpolates that file's
own `provenance` into the run log, so the live satellite page printed both attributions at once -
Ruszczak three times from the baked config, "Kotowski et al" once from the Hub. OPS-SAT-AD is CC
BY 4.0, so that was a licence condition being breached in public. All ten repos re-uploaded and
re-pinned; verified at each new revision by fetching the Hub rather than trusting the push log.

The lesson generalises past this incident: **the repository had corrected itself everywhere it
could see, and the surface it could not see was the one still wrong.** `HF6` checked for files on
the Hub absent locally and nothing checked the reverse. `HF7`-`HF9` now do, including the specific
case that a provenance string the browser fetches must be verified on the Hub rather than in the
repo.

**WS4(c) was already met, and the project had been carrying it as unmet.** `qcnn.py` contained a
working, tested Clifford homomorphic evaluation - real circuit on the encrypted state, Pauli frame
propagated, fidelity 1.0, maximally-mixed ciphertext - cited in no README and no doc. After
several rounds of finding overclaims this is the inverse, and it is worth noticing that the same
process produces both: nobody had read that module against the WS4 table.

The honest gap was narrow: one fixed circuit is not arbitrary circuits. Closing it needed the S
frame rule, verified numerically over 40 random states before use because a sign error is
invisible in a single fixed circuit, and a CNOT applier not hardcoded to wires 0 and 1. The demo
now draws random Clifford words and recovers the result at fidelity 1.0 to depth 30. **WS4(f) is
split out and NOT claimed** - the client must know the circuit to update its frame, so "the client
updates its frame" and "the client does not learn the circuit" are close to mutually exclusive
under plain QOTP. One row covering both could only ever read PARTIAL, which is how a met claim
stayed invisible.

**The novelty claim was restated after applying M4 to it.** Browser MPC has prior art: webSPDZ
(IACR 2025/487) compiles MP-SPDZ to WASM and names JIFF and MPyC-Web. Browser CKKS *evaluation* is
where the search came back empty - IACR 2026/515 defines "client-side operations" as
encrypt/decrypt while users "offload large computations to servers", and Zama's TFHE-rs WASM API
documents that it does not evaluate in the browser. The claim is now the composition, stated
identically in the landing page, the value page, README and the technical report, with both halves
of the asymmetry and the two caveats that travel with it: the MPC parties are notional, and the
quantum channel is simulated.

**My own Round 29 fix was half a fix, found by verifying the deploy rather than the code.** The
satellite decoy figure moved from 7.26 to 2.89 when it should have moved to 124.77: the channel
MODEL switched to free-space and the parameters fed to it still read the fibre page's keys. The
intermediate value was LOWER than the wrong original, because free-space at 50 km is less lossy
than 50 km of fibre - so a half-corrected number looked more conservative than an uncorrected one.
**A number that moves is easy to accept as fixed.** Verified this time by computing all three
values from the same model and matching the live page against them.

**The checklist could not verify the architecture the demo is built on.** No row asserted that
anything runs client-side - every check verified what was displayed, never where it ran.
`CS1`-`CS9` close that, and `AM1`-`AM3` turn "no additional model" into a dated decision with its
five falsifiable triggers rather than inertia. Telecom's overlay moves from "not implemented" to
CLOSED as impossible.

## Round 31: four deferred items closed, and two lessons about how they hid

**The backlog gate was being violated and the ledger said so.** Telecom's RandomForest attribution had
been deferred for FOUR rounds against a gate that allows three. `check:pqc` was recorded here as "a gap
rather than a decision" for the same span. Both are done. Genomics QCNN and WS4(f) are reclassified as
decisions with their real reasons. That is the deferred list shrinking by four rather than growing,
which is what the gate exists to force.

**Lesson one: a check row can ratify the thing it should catch.** The row covering telecom's
attribution read "the published model has no attribution, and the panel must say so and why". A
verifier working it confirmed the absence and PASSED. Four rounds of conscientious verification could
not have surfaced this, because the instrument had been written to agree with the defect. The same
shape appeared twice more this round: the checklist's XAI paragraph asserted for five use cases what is
true of three, so it could not catch the panel making the identical error; and the index paragraph
claimed a consolidation that had not happened, which is the hardest kind of duplicate to find because
it tells the next reader the work is done.

Worth generalising: **when an item survives several rounds of a working process, suspect the process
before the effort.** The question to ask of a long-deferred item is not "why has nobody done this" but
"what is telling everybody it is fine".

**Lesson two: the deferral's stated condition may not be the binding one.** Genomics QCNN was recorded
as blocked on a per-score licence audit. Checking what would happen if that audit came back clean:
nothing. `qcnn_variational.py` is supervised end to end and 1000 Genomes is a reference panel with no
phenotype, so there is no `y`. A licence to redistribute a reference panel still would not put a label
in it. `QC16` now asks this of every deferral - if satisfying the stated condition tomorrow would leave
the item blocked, the condition is decoration.

**Three defects found by walking the deployed site rather than reading the code**, which is the reason
that step is in the round trip:

- The genomics baseline had nowhere to render. The `MetricSpec` rename to `auroc_eur` was correct, and
  no achieved record emits that key because no AUROC is computable from a reference panel - so the
  published 0.633 appeared only inside a provenance paragraph. The number a visitor is most likely to
  quote was the hardest one on the page to find.
- Satellite's anomaly verdict and its whole attribution panel were unreachable. Both sat inside
  `if (!abort)`, and QSDC essentially never delivers: measured secrecy capacity is -51.58 bit/carrier at
  the default 800 km and still -2.15 at 200 km, the shortest downlink the panel offers. The page
  advertised a published model whose output no user action could produce.
- The genomics 2D overlay still described "the 77 variants of PGS000001, 64 resolved" after the page had
  moved to PGS000004. A stale overlay renders perfectly.

**And one found by retraining.** `generate()` returns label names and says so; the telecom trainer
unpacked them as `feature_names` and wrote them into `feature_spec.json` as `"features"`, so the
published spec advertised eight features named after the eight attack classes for a seven-feature model.
The committed JSON had been corrected BY HAND while the generator kept emitting the wrong value, under a
note reading "Generated from the source, not written by hand" - so the first retrain reintroduced it.
A hand-patched generated artifact is a defect with a timer on it.

## Round 32: a switch that reached four of five surfaces, and a gate that passed on nothing

**Three of the four most serious findings were defects the previous round introduced or left**, which is
the useful thing about this round rather than an embarrassment to bury. Round 31 closed four long-deferred
items and, in doing so, created two new ones and left a third unfinished. The lesson is not "be more
careful"; it is that **the work of closing an item is where the next defect gets made**, so a round's own
changes deserve the audit it gives everything else.

**1. The PRS313 switch reached four surfaces and missed the fifth.** The E2E pipeline read
`config.get("prs", "breast_cancer")` with nothing anywhere setting that key, so Python scored 77
rsID-keyed variants while the browser, trainer, registry and payload scored 313 position-keyed ones. And
`data/loaders.py` carried a sentence arguing that this exact split "would be worse than either alone" -
by then describing the shipped state rather than warning against it. `config.get(key, literal)` cannot
fail: an absent key silently yields the default and the result is complete and plausible. **Third
occurrence of that shape** here, after `config.usecase` read but never emitted and the free-space channel
reading the fibre page's distance keys. The common factor every time is a default no mechanism compares
against anything, so the score is now a config FIELD and `tests/test_prs_pipeline_parity.py` runs the use
case and compares its output to the payload the registry points at.

**2. The newly-gated `check:pqc` exited 0 on an empty vector file.** It asserted `accepted === 0` and
never asserted how many keys it had tested, so testing nothing reported exactly what testing 2,595 and
finding no acceptances reports. **Vendoring introduced this**: a fetch either returns the file or fails,
but a file on disk can be present and empty. It is the project's own SP 800-90B lesson - a health test
reporting PASS because it was handed too little data to fail - inside the check promoted to a gate one
round earlier. The three re-validations recorded at the time were the failure modes of the design being
replaced. Generalised as `RI9`: **after promoting any check to a gate, break it on purpose.**

**3. A citation to an authority that did not contain it.** Three source files cited "DATA_LICENSES
section 4 justifies about 30,800 genotype cells"; section 4 stated a shape and made no cell-budget
argument at all. The code invented the authority it appealed to, and the appeal reads exactly like a
checked one. Section 4 now contains the argument, tabulates all four tracked genotype files, and separates
tracked-in-git from served-to-the-public - which turned out to matter, because a retired payload was still
being served for no demo benefit. `RI11` makes "grep the cited section for the cited figure" a row.

**Two measurements that changed a claim rather than confirming one.**

`node-seal` had been recorded as "the constraint" for several rounds without anyone trying it. It pins a
FORK of SEAL whose divergence from v4.1.2 is **purely additive** - 1,005 lines of Emscripten embind and
CMake glue, no crypto source touched. Taking `microsoft/SEAL` v4.4.3, adding that binding layer and
building to WASM: **zero errors**, `libseal-4.4.a` produced. The blocker that stopped the first attempts
was not SEAL's - `defines.h` tests the bare `EMSCRIPTEN` macro, byte-identical in both versions, and emcc
6.0.6 defines only `__EMSCRIPTEN__`. So a bump is a build flag plus a submodule re-point, and node-seal's
own build would fail the same way at 4.1.2 on a current emsdk. Not verified and so not claimed: a CKKS
round trip through the linked module.

And the HE-genomics literature falsifies this project's own framing in the direction of understating it.
`registry.py` explained T1 smallness by homomorphic cost; Knight et al. run polygenic scoring under HE
over **110,000 SNPs on a single CPU**. So `genomics.prs` is small because PGS000004 has 313 variants -
Mavaddat 2018's property, not CKKS's. The cost argument survives where the model has depth and was
over-generalised to a linear score, which read as rigour.

**What Round 31 got right and is worth keeping:** every one of these was found by running or deploying,
not by reading. The pipeline split needed the loader run to see it; the empty-snapshot hole needed the
file emptied; the tree-attribution sentence needed the page loaded twice, because the first fix rendered
it nowhere and a diff review would have passed it.

## Round 33: the count went up, which is the one thing the gate forbids

**Round 32 closed three items and opened at least six with no resolving condition.** The gate above has
three clauses and this failed the two that are supposed to be the easy ones: every deferral records its
condition, and the count does not increase. It did not even fail quietly - the Round 32 plan named two
of the new deferrals itself, wrote "give each a condition or reclassify", and then did neither. **An
instruction a plan issues to itself and does not execute is the same defect as a commit message
describing work that is not in the commit**, which this file records twice about the checklist. The
plan is read once and the artefact is read forever, so the artefact is where the instruction has to
land.

Every item below now carries either a condition that is externally checkable or a reclassification as a
decision with a reopening condition. Each was put through `QC16` first: ask what would happen if the
condition were satisfied tomorrow, and if the answer is "still blocked", the condition is decoration.
**Four failed that test and are reclassified rather than conditioned, and three collapsed into one
item**, which is how the count comes down instead of being renumbered.

### The condition-free items opened in Round 32

**1. The medical latent-space encoder, and an artifact citing a record that does not contain it.**
`models/metrics.json` (`medical_fl.student.baseline_gap_note`) and its mirror in
`web/src/lib/config.defaults.json` both say Kim et al. (arXiv:2506.15258) is "recorded as a falsifiable
trigger in the additional-model decision". It is not. That decision's trigger table in
`docs/browser-verification-checklist.md` has **five rows** - bootstrapped CKKS CNN, labelled QKD attack
corpus, quantum kernel at 8 qubits, encrypted transformer on finance tabular, an encrypted BreastMNIST
result above this one - and none of them concerns latent-space encoding or a VQGAN. `references/REFERENCES.md`
repeats the same claim. So three artifacts assert a record, and the record does not exist. **This is
the Round 32 finding about DATA_LICENSES section 4 in a second place**: the appeal to an authority reads
exactly like a checked one, and nothing checks that the authority contains the thing appealed to.

The checklist is owned by another worker this round, so the fix is recorded rather than made.
**Condition, in two parts, both greppable:** a sixth row is added to that trigger table naming the
latent-encoder route with the measurement that currently blocks it, and the `baseline_gap_note` stops
asserting the record until the row exists - "not recorded anywhere yet" is a true sentence and costs
nothing. The row's own content when it is written: *an encoder whose latent fits 4096 CKKS slots at
multiplicative depth 1 and which runs under `onnxruntime-web`* - the paper's own regime is chest X-ray
at downsampling factor 8, and 28x28 at factor 8 is 3.5x3.5, so whether the technique transfers to this
input size at all is the open measurement and must be stated as unmeasured rather than assumed either
way.

**2. Tailor, the no-MPC exact-ReLU design - RECLASSIFIED as a decision.** IACR ePrint 2026/1551 is
recorded in `docs/architecture.md` as "an alternative design with its trade-off" and appears on no
list, so it was carried without being carried. `QC16` retires the condition anyone would write for it:
if Tailor were adopted tomorrow the demo could not load it. Its backends are **TFHE and FINAL**, and
the only HE library this project's browser tier can load is `node-seal`, which exposes `bfv`, `bgv` and
`ckks` and no TFHE at all. So the blocker is not the 222 s server measurement against a page-interaction
budget, which is what the architecture note leads with - it is that there is no browser artefact to
evaluate. **This is the Rust/tfhe-rs decline above, applied to a second artefact**, and it should carry
the same reopening condition: *a TFHE or FINAL implementation that loads in a browser tab and
EVALUATES - not merely keygen, encrypt and decrypt - and evaluates one exact ReLU inside a page
interaction.* Recorded as a decision so the architecture note stops being the only place it lives.

**3. Quantum crosstalk in the SDN controller - CONDITIONED, and it is in scope.** `src/qsh/qkd/sdn_controller.py`
contains no crosstalk term (measured: zero occurrences of the string), while Konteli et al.
(arXiv:2607.24299) publish all-to-all quantum-level crosstalk on a **commercial** 16x16 MEMS switch with
the impact expressed directly in decoy-state BB84 terms. The tempting decline is that this project's
quantum layer is simulated, so a physical-layer penalty is out of scope. That is wrong, and checking it
is what makes the condition writable: `qkd/decoy_reference.py` already carries a **background-count**
term and reproduces Ma et al. against it. Crosstalk enters as a background-count contribution, so this
is a parameter with a measured external source, not a new mechanism. **Condition:** the per-link budget
in `sdn_controller.py` gains a crosstalk-sourced background contribution taken from arXiv:2607.24299,
and the item is resolved when a switched multi-user allocation reports a **different** secure key rate
from an unswitched one - a term that changes no output is the defect this project treats as serious, so
the test is the delta and not the presence of the code.

**4 and 10 and part of 9 are one item, and that is the finding.** The two 2026 QKD proofs on the
deferred list are a matched pair - Sixto et al. (arXiv:2606.29943) transmitter side, Navarrete et al.
(arXiv:2608.09674) receiver side, overlapping author sets, and the receiver paper published within days
of the already-cited 2608.09793 - and both supersede the same Ma, Qi, Zhao and Lo (2005) baseline that
`qkd/bb84.py::_decoy_secure_fraction` and `qkd/decoy_reference.py` rest on. `references/REFERENCES.md`
ends the Sixto entry with "the honest statement is that the demo has not adopted it", which is honest
and is not a condition. Carrying them as two items, plus "hardware QKD field test" as a third, is
double and triple counting, because **all three are gated on the same missing thing**.

`QC16` finds it: if either proof were "adopted" tomorrow, nothing on the page could change, because
their inputs are **device characterisation parameters** - state-preparation flaw magnitude, side-channel
state distinguishability, detection-efficiency mismatch - and this repository has no device. Adopting
the proof would mean inventing the numbers the proof exists to bound, which is the move this project
refuses everywhere else. So it is one item, gated on characterisation, and **it is reclassified as a
decision with a condition that does not need a device**, because a version is reachable without one:

> **Condition:** publish, beside the Ma et al. secure-key figure, what the 2026 bounds do to it at
> **one stated parameter point** - one assumed state-preparation flaw magnitude, one assumed mismatch -
> labelled *modelled, not measured*, so the reader learns the direction and rough size of the penalty
> rather than being told a 2005 number is current. If a characterised hardware link ever exists, adopt
> Sixto and Navarrete together in place of Ma et al. in that same round, because that is the round in
> which the inputs exist.

That condition can be satisfied in a day and it cannot be satisfied by decoration, which is the test.

**The `trojan_horse` / `wavelength_trojan` half of item 4 is a separate and smaller thing, and the
reference entry states it slightly too strongly.** Those are attack CLASSES in a synthetic IDS corpus -
five-tuples in `src/qsh/qml_ids/attacks.py`, mirrored in `web/src/lib/clientIDS.ts` - and a synthetic
label generator does not need a security proof. What it needs is a stated provenance for its numbers.
**Condition:** each class's five-tuple names the source its magnitudes come from, and where the
magnitude is this project's own choice the class says so in the same line. Greppable, and it closes the
real gap, which is unattributed numbers rather than a missing proof.

**5. The second same-day quantum Executive Order - RECLASSIFIED, with a dated reopening condition.**
Recorded in `references/REFERENCES.md` as "cited nowhere here", with the reason given as the appearance
problem: a roadmap citing one of two same-day quantum EOs looks like it missed one. Read at the primary
source rather than left as an appearance question. It is **EO 14413, "Ushering in the Next Frontier of
Quantum Innovation", signed 2026-06-22, published 2026-06-25 at 91 FR 38487**, five pages, eleven
sections: national strategy, quantum computing for scientific discovery, sensors and networks, supply
chains, technology protection, workforce, international engagement, reports.

It is **not** a cryptographic-migration instrument - that is EO 14412 at 91 FR 38483, which this
repository already cites correctly and whose number the secondary sources get wrong. The one place
14413 touches this at all is **section 4(f)**, which directs an intelligence assessment of the national
security implications of commercial quantum computers increasing in scale and performance, "such as the
implications for the migration to post-quantum cryptography". That is a reporting directive to named
officials, and it imposes no obligation on any system. **So: declined for scope as a roadmap instrument, and it should be cited in
one line for exactly that reason** - the citation's job is to record that it was read and found not to
apply, which is the opposite of missing it. `docs/pqc-roadmap.md` is owned elsewhere this round, so the
line is recorded here and not written there.

**Reopening condition, and it is a date rather than a hope:** section 3(a) directs an update to the
National Quantum Strategy within **180 days**, so about **2026-12-19**. If that update, or section 5's
quantum-network work, carries a migration or deployment directive, 14413 becomes a roadmap instrument
and must be cited as one. Re-check on that date.

**6. `RI9`'s five-way re-validation - and the premise that it was never run is WRONG. It was run and
never recorded, which is a different and worse defect.** The Round 32 commit `3de8599` states it
outright: *"Five ways, measured: clean 0, empty 1, truncated 1, tampered 1, missing 1. The hash fires
first on the vendored path, so the count is the second line rather than the first, and the comment says
so."* Meanwhile `RI9` in the checklist still reads as an instruction to re-validate five ways,
`docs/verification-log.md` carries nothing, and the record below in "Verification assets added in Rounds
22-23" still lists **three**. **So this is the THIRD time a commit message here has described work that
no artifact records** - after the two checklist cases in Round 29 - and it is the first time the work
was actually done. The lesson sharpens rather than repeats: the previous two were commits describing
work that had not happened, and this one is a commit describing work that had. **From the tree, those
two look identical.** A measurement that exists only in a commit message is not a record, because
nothing reads commit messages when checking whether a claim holds.

The outstanding requirement is therefore not to run the five - it is to put them where the register says
gate results live, and to reconcile three documents that disagree about whether it happened.

The mechanism is worth writing down because it is a good example of a fix invalidating a test rather
than a claim. Round 32 added a SHA-256 assertion on the vendored gzip, and it runs **inside
`loadVectors`, before `gunzipSync`**. The third recorded re-validation was a well-formed key smuggled
into the malformed set, reported as exit 1 with "779 rejected, 1 accepted" - and 779 + 1 = 780, which is
exactly the ML-KEM-768 count, so that test was constructed to keep the count unchanged. That is
precisely the case the count assertion cannot catch and the hash can. **The hash now fires first and
prints a hash mismatch, so the "779 rejected, 1 accepted" line cannot be produced on the vendored path
at all.** The script's own comment says the hash catches empty, truncated and tampered files alike, and
that is measured; nobody then went back to the record that quotes the superseded output.

**Condition:** all five re-validations - clean exit 0, empty exit 1, truncated exit 1, tampered exit 1,
missing exit 1 - recorded in `docs/verification-log.md` with each exit code and each printed message.
Round 32 ran them and recorded them only in commit `3de8599`, so the condition is a transcription rather
than a measurement, and it is written into the Round 32 entry of that file in Round 33.

**One thing does still need measuring, and it is the case the five-way list quietly replaced.** The
"smuggled well-formed key" test and the "tampered" test are not the same test. Tampering is caught by the
hash, which is why it exits 1; a smuggled key was the test of whether the CHECK ITSELF can distinguish
779 rejections from 780, and that is now unreachable on the vendored path. To exercise it at all, the
record has to state **how it got past the hash** - either re-vendor with the recomputed hash, so the
count assertion is the line of defence under test, or use `--live`, where there is no hash to check.
Naming the technique is part of the record, because a five-way list that silently substitutes an easier
case for a harder one is exactly the shape `RI9` exists to catch.

### The older items that never had a condition

**7. Malicious-secure MPC - and the premise that this is the longest-carried deferral is WRONG in the
direction this project cares about.** "Deferred since Round 1, no condition in any document" is what the
ledger says, and the ledger is the last place still saying it. The blanket sentence in "What this
project cannot verify" - "The security model is semi-honest. Malicious security was designated in Round
1 and never built" - is **false about the browser's non-linear layer**, and it has been false for
several rounds while three other documents described the real state correctly.

`web/src/lib/clientMPC.ts` runs `mpc-framework` over `emp-wasm-engine`, and `emp-wasm` is a WASM build
of **`emp-toolkit/emp-ag2pc` and `emp-agmpc`** - authenticated garbling, Wang, Ranellucci and Katz, CCS
2017, IACR ePrint 2017/030, whose title is literally "Authenticated Garbling and Efficient **Maliciously
Secure** Two-Party Computation". `docs/architecture.md`, `docs/crypto-agility.md` and
`docs/threat-model.md` all call it authenticated garbling; this register generalised "semi-honest" over
the whole project from the paths where it is true. **Same shape as the SEAL 4.4.0 advisory: the register
was wrong in the understating direction, while the documents an audit would have flagged as
overclaiming were right.**

What is genuinely semi-honest, and is now scoped rather than generalised: MPyC / Shamir in
`src/qsh/mpc/secret_inference.py` (the server tier), the keyed-hash PSI, and `src/qsh/qsmpc/qot.py`.
And the honest caveat on the malicious-secure path is not about the protocol: both parties share one JS
heap, so the party separation is notional and this is a demonstration of a malicious-secure protocol
rather than two mutually distrusting hosts. That caveat is already at `clientMPC.ts:10-12`.

The residual is what `README.md` says it is, and the README's parenthetical was **more precise than
this register's sentence**: PQ-OPRF PSI and **identifiable** abort. Authenticated garbling gives
security with abort and does not identify the aborting party. **Condition:** a post-quantum OPRF with a
browser-loadable implementation, and a protocol providing identifiable abort - each named with the
library that would supply it, at the point where it is claimed. Until one of those exists the residual
is two named cryptographic primitives that are absent, which is a decision, not a task.

**8. ETSI GS QKD 014 mutual TLS - RECLASSIFIED, because the stated condition was never the binding
one.** This is a clean `QC16` case. If client certificates were provisioned tomorrow the item would not
move, because **there is nothing to attach them to**: `src/qsh/qkd/kme.py` is an in-process Python
object, `api/` exposes no KME route at all, and the module's own `authentication_note()` says so - "this
in-process KME has no transport to attach that to". Mutual TLS is a transport property and this KME has
no transport. **Reopening condition:** if the KME is ever exposed over HTTP, mTLS with the SAE identity
taken from the client certificate goes in **the same change** - the two must not be separable, because
a reachable KME whose SAE identity is a parameter is worse than an in-process one that says so.
Greppable: a KME route in `api/` without mTLS is the failure.

**9. The three README residual gaps - and only one of the three is an independent item.** `README.md`
listed "hardware QKD field test (vs the BB84 simulator); larger fault-tolerant QHE; production
bootstrapping throughput" with no conditions and no dates, next to "malicious-secure MPC (PQ-OPRF PSI +
identifiable abort)". Listing them as four independent gaps inflates the count, because one is a boundary
of the deliverable, one is downstream of a decision already recorded above, and only the third stands on
its own:

- *hardware QKD field test* is the same gate as items 4 and 10 above - no characterised device - and this
  file already carries the decision in "What this project cannot verify": no QKD hardware exists here
  and there will be no browser API for it. It is a **boundary of the deliverable, not a gap in it**.
  Definition of done if it ever happens, so it is not an open-ended wish: the first thing published is
  the measured QBER against `decoy_reference.py`'s prediction for the same link budget.
- *production bootstrapping throughput* is downstream of the OpenFHE `EvalBootstrapSetup` SIGSEGV
  decision above, which is already recorded as a permanent workaround on this host distribution. A
  throughput figure needs the bootstrap path to run. **Condition:** publish the figure when `make
  test-all` completes without the segfault, or when the HEonGPU collective-bootstrap path is benchmarked
  end to end - one item, gated on a thing already decided, not a second independent item.
- *larger fault-tolerant QHE* is the only one of the three that is genuinely its own item. Today's
  `t_count()` is a labelled Solovay-Kitaev estimate of a fault-tolerant deployment's cost, not an
  executed count. **Condition:** replace the estimate with an executed count when a backend exposing
  logical, error-corrected qubits is addressable from PennyLane. Externally checkable against a provider
  backend list, and if it were satisfied tomorrow the item would move.

**11. The four metadata-only arXiv entries - CONDITIONED with a deadline, because that is what they were
missing.** arXiv:2604.04783, 2602.11470, 2512.11269 and 2602.22229 are labelled honestly in
`references/REFERENCES.md` - "abstracts not individually opened, so no number from these four should be
quoted" - and that label is doing real work. What they lacked is any way to stop being in that state.
**Condition:** an entry is promoted only when a number from it is quoted **with the section or page it
came from**, and any entry not promoted by **Round 35** is removed. Both halves are greppable, and the
deadline is the point: an honest label with no deadline is a permanent resident. The Julia decision in
`README.md`, taken in this same round, is what that looks like after twenty-nine rounds - deferred with
no condition attached, so it could not even fail the three-round rule.

**DEADLINE EXECUTED Round 39.** None of the four was ever promoted, so all four were removed from
`references/REFERENCES.md`; the arXiv IDs survive in one sentence there for anyone re-opening the
search. It ran three rounds late, and the reason is the finding: the rule was correct, greppable and
had **no executor**, so it was discharged only when someone happened to read the paragraph. That is
the same shape as `DP7` - a checklist row that was right and ignored for five consecutive rounds -
and as the disclosure that shipped four times with the wrong scope. A rule nobody is obliged to run
is an intention, not a control. `tests/test_register_deadlines.py` now parses this file for its own
`by **Round N**` sentences and fails once one passes undischarged, so the next deadline is enforced
by the suite rather than by memory. The stamp above is the only thing that discharges one; the first
draft of that test exempted paragraphs containing "promoted" or "removed", which is what a
conditioning sentence says, so it passed while this very deadline was three rounds overdue.

### Two corrections to this file's own register, found while writing the above

**The staged-PRS paragraph was describing a state that ended in Round 31**, and Round 32 fixed the
identical sentence in `src/qsh/data/loaders.py` while leaving this copy. Corrected in place, in the Round
29 section above. This is the third instance of this file's own lesson - a register of open items needs
the same staleness check as any other document - and the third instance is the one that should end the
argument that it is bad luck.

**The Round 28 circuit-privacy retraction and its standing prohibition were unmarked as superseded for
three rounds**, and this was the one place where the DOCUMENTATION was weaker than the implementation.
Everywhere else in this project the risk runs the other way. Marked in place in the Round 28 section
above, with the Round 28 text kept as history.

### Two decisions taken outside this file in the same round, recorded here so the register is complete

- **Quantum Risk / Readiness Indices: DECLINED**, in
  [`references/security-scoring-fact-check.md`](../references/security-scoring-fact-check.md). Named in the
  instruction that drove the security-scoring work, then absent from the whole repository while the other
  four candidates got written verdicts. Several citable instruments do exist under that name - a national
  agency self-assessment, a central-bank sector index, two vendor indices, a maturity model, and Mosca's
  x + y > z inequality - and every one of them scores an **organisation's migration programme**, where
  this project's surface is one run of a cryptographic pipeline. Reopening condition recorded there.
- **Julia: DECLINED**, in [`README.md`](../README.md). No in-browser path exists and no numerical gap
  exists for it to close. Reopening condition recorded there, and it requires both halves so it can
  actually fire.

## Rounds 34-36, and the three rounds this register was not written in

**This section exists because the register stopped.** The rule stated above - an item deferred three
rounds must be done or reclassified - is enforced by writing it down, and Rounds 34, 35 and 36 were
not written down. The standing per-round request then arrived four times asking for items this
document already had answers for.

### Round 34 - OpenFHE bootstrapping, and a backend that degraded silently

Recorded in `docs/verification-log.md`. The finding that belongs here: `use_bootstrap=True` wrapped
its setup in `except Exception: self.bootstrap_enabled = False`, so a hard failure and an absent
feature produced the identical state, and the test read that state and skipped. It now raises.

### Round 35 - four checks defined and none of them run

`QK4`-`QK7` were added to the checklist and merged without being executed, and without a verification
log entry. `QK4`, `QK5` and `QK7` were measured in Round 36 and pass. **`QK6` could not be run at
all**: it asked for monotonicity in `mu`, which has no control on any page, so half of it was never
exercisable by a person. It moved into `npm run check:qkd`.

The generalisable form, and it is the reason this section is late: **the round that writes a check is
not the round that runs it, and nothing forced the second half.** `RT9` closes that loop for content
defects; nothing closed it for newly-added rows.

### Round 36 - three pages stated a security property the run did not provide

The axis was per-use-case security communication. Measured: `pqc_handshake`, `qkd_key_agreement`,
`satellite_qkd`, `qsmpc_compute`, `federated_training` and `_sessionKey` had **zero occurrences each**
in the whole checklist. Three false claims followed from nobody having looked:

- the diagram drew "pairwise masking (demo keys)" over a plaintext weighted mean, with
  `secureAgg: true` a literal on the return line. Now implemented (Bonawitz, CCS 2017) and gated by
  `npm run check:qfl`.
- the QSDC eavesdrop check was unreachable - `link_budget_limited` was 1 at every setting - so a
  full-strength eavesdropper drew "QSDC direct telemetry" with a completion tick. The budget terms
  are now controls and the check fires at 1 km / 3 dB / Eve 0.40.
- the telecom QKD arrow and its party box both claimed an eavesdropper while `qkd_eve` defaults to
  0.0.

Unclamping the QSDC secrecy capacity exposed what the clamp had hidden: **the same nominal 800 km
downlink was 24.48 dB in the browser and 28.0 dB in the reference implementation**, which still
applied the linear law the browser had rejected. Both sides had been reporting "0".

## Decisions, part two: the four standing asks

Same format as the register above - the decision, why, and what would reopen it. These four arrive in
every ORDER file and had no entry here, which is why they arrive in every ORDER file.

**Published models versus current MPC-FHE research: ANSWERED, and the answer is stable.**
`references/model-choice-vs-2026-practice.md` checks all five verticals against 2025-2026 encrypted
inference literature with a paper-count table. Finance is the sharp case: across verified papers the
architecture distribution is ~10 tree ensembles, ~5 linear/SVM, ~2 small MLPs, and **zero**
transformers or tabular foundation models on a finance benchmark. This project publishes an XGBoost
teacher and a 33-parameter logistic student under CKKS - squarely what the field evaluates.
*Reopens if* a verified paper benchmarks encrypted inference on a finance tabular task with a
different architecture class.

**An additional ML model: DECLINED per option, with measurements.**
`references/model-strategy-decision.md`. The encrypted tier evaluates 20-3,226 parameters because
CKKS at `poly_modulus_degree` 8192 gives one multiplicative level; no pretrained model exists at that
size, and the teachers already sit at or above their published baselines. *Reopens if* bootstrapping
lands in the browser tier, which would change the parameter budget rather than the preference.

**Metric choice against 2026 practice: ANSWERED, and two of the five choices are
defensible-but-not-best-practice rather than correct.**
[`references/2026-metric-practice-fact-check.md`](../references/2026-metric-practice-fact-check.md)
resolves every arXiv ID it cites and drops the ones that did not resolve. Two findings this project
acts on: tuning a screening threshold on F1 degenerates to all-positive when discrimination is weak,
which is a **2014 theorem** (Lipton, Elkan & Narayanaswamy, arXiv:1402.1892, ECML-PKDD 2014) rather
than a local surprise - the sweep that hit it reproduced it exactly. And **Youden's J, which replaced
F1 here, is not what current practice recommends**: Van Calster et al. (STRATOS TG6, Lancet Digital
Health 2025;7(12)) find all classification measures improper at thresholds other than 0.5 or the
prevalence. *Reopens if* a decision-curve or net-benefit readout is added to a use-case page, which
is the direction that document points.

**This entry exists because the document was an ORPHAN.** No tracked file pointed at it, so the
question it answers - "are these metric choices current?" - could only be re-asked. It was linked
from the gitignored checklist, which does not ship, and a planning sweep reported the set as clean
because the check written for it could not fail. `DC1` now runs over tracked files only.

**QCNN against literature numbers: RECLASSIFIED from declined to done, Round 37.** The stated reason
- "no published QCNN reports this circuit on this data" - was true of the circuit and false of the
data. `QC18`-`QC20` and `references/qcnn-vs-published-wdbc.md` carry the comparison against Sci Rep
14 (2024) DOI 10.1038/s41598-024-74778-7 with the three differences that must travel with it.
*Reopens if* a published QCNN result on WDBC states its qubit count, depth, encoding and feature
subset on a held-out test split, which would allow a matched comparison instead of an annotated one.

**2D plots of private inference against test data: DONE, Round 37.** The five scatters plotted the
INPUT corpus and never an inference result, and the one file that is its model's training corpus
belongs to the only use case that runs no private inference. `EncryptedAgreementChart` now draws the
encrypted score against the plaintext score per held-out row. *Reopens if* a use case gains a private
inference path whose output is not linear, since the batching that makes it affordable is specific to
the packed linear layer.

## Round 38: the round measured its own plan and found it wrong twice

Two of this round's own premises did not survive being measured. Recording that is the point of this
register.

**Per-rotation cost was to be "one division away".** The plan proposed
`fhe_linear_ms / fhe_dense_rotations`, both of which the page already prints beside each other.
`fhe_linear_ms` is the measured wall-clock multiplied by `benchScale()` for the selected model size,
while the rotation count is raw, so on genomics that division reads 159.9 ms per rotation against a
measured 3.18. `rotateVector` is now timed directly, and `fhe_linear_measured_ms` and
`fhe_bench_scale` render beside `fhe_linear_ms` so the projection is decomposable. Rows `PF1`-`PF7`.

**No comparable published figure was expected to exist.** The plan intended to record the absence as
the finding. One does exist: node-seal's own `BENCHMARK.md` publishes CKKS rotation timings in
Chrome, Firefox and Safari at both ring degrees. The measurement here is 3.10-3.24 ms per rotation
against its 6.83 ms Chrome figure at ring 8192, a comparison the two parameter sets support; the
16384 rows are NOT comparable, because the published modulus is 438 bits against 200 here, and that
is stated rather than quietly averaged over. Both comparators were fetched and read directly -
[`encrypted-inference-performance.md`](../references/encrypted-inference-performance.md) quotes the
rotation rows verbatim.

**A claim this round refuted, and then had to un-refute.** The round reported that "key-switching
dominates CKKS cost" held for the dense layer at 57% and failed for the convolution at 10%, and
rewrote three comments accordingly. The 10% divided the convolution block's rotation time by the
WHOLE student's wall-clock, because `encryptedStudentInference` returns `{...conv, fheMs}` - the
spread carries the conv block's rotation figures and `fheMs` is overridden with the total. Measured
with the scopes separated: **57% dense, 70% convolution, 86% the student's dense layers**, summing
to the total exactly. Key-switching dominates everywhere; the comments are restored and now carry
the measurement. `PF7` and the scope-named metrics keep the two apart.

This was caught by an adversarial review of the round's own diff, not by a test or by looking at the
page - every figure rendered plausibly. The round existed to ask "over what set is this number
computed, and does the sentence beside it describe that set"; it had asked that of the code it was
fixing and not of the code it was adding.

**The ring degree: raised to 16384, measured, reverted.** Every FHE figure roughly doubled - keygen
255 to 477 ms on the first click, per-rotation 3.10 to 6.73, precision 25 to 24 bits - and the
realised benefit measured zero: the medical convolution still packed one image because nothing asks
it for two, and the linear batch's 512 to 1024 rows went unused because finance offers 400 and
genomics 30. **Decided** at the user's direction: value back to 8192, wiring kept. *Reopens* when
batched convolution exists to spend the slots on, at which point it is one line in `config.yaml` and
one in `defaults.py`.

**The medical batched-convolution figure: DEFERRED, with the arithmetic that decides it.** This was
the item the ring change existed to serve, and it is recorded separately rather than as a clause
inside that decision, because a reader asking "what happened to it?" should not have to find it
there. The student's encrypted convolution needs 3,136 slots per image - a 14x14 image padded to
16x16, a 3x3 kernel giving 196 patches at a 16-slot block - so ring 8192 holds one image and ring
16384 would hold two. Two points is not a figure, and medical already carries a 400-row linear
agreement chart, so the marginal reader gains nothing while every FHE number on the page doubles.
Building it properly needs explicit chunking across ciphertexts, which pays linear homomorphic cost
per chunk and has to report that cost rather than hide it behind one number - the same reason
`encryptedConv2d` throws today instead of tiling silently.

*Reopens if* either of two things becomes true: the encrypted convolution is batched across
ciphertexts with the per-chunk cost surfaced, at which point the ring raise pays for itself and both
land together; or medical's linear agreement chart is removed, which would make even a two-point
convolution figure the only encrypted-vs-plaintext evidence that page has.

**The wiring is the durable result.** `FHEConfig.poly_modulus_degree` had no reader anywhere in
`src/`; the value the TenSEAL backend ran came from its own default arguments, the browser's came
from a literal in `clientFHE.ts`, and the report PRINTED a literal inside `build_data.py`. Four
copies, one dead field, agreeing only because nobody had changed one. Rows `PS1`-`PS4` and
`tests/test_fhe_parameters_single_source.py`.

**And the trap that caught the author, which is why `PS4` exists.** Raising the ring in
`defaults.py` alone changed nothing: `load_config()` overlays `config.yaml`, so the export script
rewrote the JSON with the old value and exited 0, the report regenerated with the old value and
exited 0, and the bundle rebuilt with the old value. An inert edit with three successful exit
statuses behind it - the `DP2` failure mode with no failure to hide.

**Encrypting the model the page explains.** Finance and genomics reported a class from the bundled
real model, drew SHAP bars labelled "real bundled model", and computed every `encrypted_*` figure
from a demo weight pattern that appeared nowhere else on the page. Verifying the fix surfaced three
more defects in numbers already being published - a metric key written twice with different values,
a mismatch count with no margin beside it, and a caption whose scope was narrower than its sentence.
Rows `MS1`-`MS5`.

## Round 39: the round that reconciled Round 38 with itself

Round 38 fixed a cross-language crypto defect across five sites and, in the same round, **introduced
a new one**. It also left no trace of any of that work here: when Round 39 opened, `grep -c "SP
800-227\|combiner"` over this file returned **0**. The round's largest finding was absent from the
artefact whose entire purpose is that the standing per-round asks need not be re-derived, because
the register is written last, when the round is already long.

**The hybrid-KEM combiner: backend and browser derived DIFFERENT session keys for two rounds.**
Round 38 hardened the combiner toward NIST SP 800-227 section 4.6.3, which requires the ciphertexts
in the derivation for the IND-CCA-preserving property, and separately added an output-length tag so
a 32-byte session key cannot be a prefix of a longer QOTP pad - HKDF-Expand for `L` bytes is a
prefix of its own expansion for any `L' > L`. The tag went into `src/qsh/pqc/hybrid_kem.py` and into
**none** of the three browser sites. Measured, not argued:

```
python info : qsh/hybrid-kem/v2 ML-KEM-768+X25519/L32|<transcript>
browser info: qsh/hybrid-kem/v2 ML-KEM-768+X25519|<transcript>
same session key?  False
```

Three shipped strings asserted the opposite, one of them rendered in the Validations panel. The
commit that introduced the divergence is the one **immediately after** a commit message reading
*"All four sites change together - they must, or backend and browser derive different keys."* The
existing binding test was Python-only, so nothing could catch it.

*What was actually wrong* was not the missing tag - it was that the same value had four definitions.
Round 39 gave it **one**: `web/src/lib/hybridCombiner.ts` holds the browser's single definition, and
`web/scripts/check_hybrid_combiner.mjs` reproduces committed Python vectors **through the real
browser module** - not through a re-implementation, which would have agreed with itself while the
browser drifted. Its fifth check greps the tree and fails if the scheme literal appears in more than
one module, so the four-definition state cannot return. `scripts/export_combiner_vectors.py`
generates the vectors from the production Python function, and CI diffs them, so a Python-side
change that is not regenerated turns the parity gate red instead of leaving it green and wrong.

**The backend logged a QKD mix that never happened.** `standard_stages()` ran `PQCHandshakeStage`
**before** `QKDKeyAgreementStage`; the handshake read `ctx.data.get("qkd_key")`, which is `None` at
that point, and the QKD stage then stored a key with no reader anywhere in `src/`, `web/src` or
`api/` while logging *"mixed into session"*. `SatelliteQKDStage` had the same defect with a quieter
symptom. **Scoped honestly:** live in the full deployment, and **not** page-visible in the
client-only public demo, whose `securityFacts` reports secure bits as a measurement and claims no
mix. Fixed by folding the derived key into the live channels rather than by reordering the stages,
so a run that yields under 256 secure bits still degrades to PQC-only exactly as before - the fold
only **adds** the mix on runs where QKD succeeded. `tests/test_qkd_is_mixed_in.py` asserts on the
key bytes and not on the log line, parameterised over both pipelines, because a test that greps the
log passes as soon as the wording is fixed.

**The expired rule, and why a rule needs an executor.** See item 11 above: a deadline this register
set on itself ran three rounds past its date because nothing was obliged to check it.
`tests/test_register_deadlines.py` now enforces every `by **Round N**` sentence in this file. Its
own first draft **passed on the overdue deadline**, because it exempted any paragraph containing
"promoted" or "removed" and that is what a conditioning sentence says. That is this project's
tautology class - a check whose predicate cannot fail - for the fourth time, and the first time it
was caught before the commit rather than a round later.

**Documents that disagreed with the code and with each other.** Round 38 published corrected numbers
without retracting the superseded ones in the same breath, so both stood. The scope error was in the
document that exists to correct scope errors: *"1,269 ms (of which 579 ms is the encrypted
convolution)"*, where 579 ms is the whole student and the convolution measured 84 ms. The withdrawn
`fhe_conv_rotation_share 0.10` was still tabulated under "what the deployed run shows", 26 lines
above its own retraction. The retracted uncontrolled keygen pair survived in two files beside the
controlled one. `docs/crypto-agility.md` described the combiner as *"ML-KEM XOR X25519"* - never
true of this code, in the canonical document for the KEM registry, sixteen rounds stale. Each is now
corrected **in place with its provenance**, and where the file is a log the superseded figure is
marked rather than deleted, because a log that edits its own history is not a log.

**Three claim-accuracy defects found by inventory, recorded as conditioned decisions rather than
half-fixed.** A 44-site sweep of key derivation and crypto claims produced seven candidates. Two were
live and are fixed above; one was refuted by a runtime probe (`stages.py`'s zero-key default is
unreachable, because the handshake always precedes the stage and a failed stage aborts the run - it
was hardened anyway, since this round moved key material between stages and the cost is nothing); one
is the ITS combiner. The remaining three are each a sentence claiming more than the code does, in
paths the public demo either does not reach or does not depend on, and each needs its own
justification rather than a blanket sweep:

- **`src/qsh/qkd/kme.py` says keys come from "this project's QRNG + extractor path"; `get_key` calls
  `secrets.token_bytes`.** The claim appears twice - the module docstring and a user-visible
  `status_extension` note - and the project genuinely has the path being claimed, under
  `src/qsh/qrng/`. So this is not a wording slip to paper over: it is a choice between wiring the KME
  to the QRNG path or saying it uses the OS CSPRNG. **Deferred, not declined**, because changing key
  generation is a behavioural change and no API route constructs a `KeyManagementEntity` today -
  reachability is tests only, plus a browser mirror of the status string. *Reopens the moment any
  route or page reaches it*, since a false provenance sentence in a rendered status body is a
  different severity from one in a docstring.
- **`src/qsh/qsmpc/qot.py` binds its commitment to `_` and never opens it**, while the module
  docstring cites "a SHA-256 commitment" as a source of obliviousness. Unreachable from both the demo
  and the orchestration - the stage imports only the PSI functions - so nothing renders this claim.
  *Reopens if* `quantum_oblivious_transfer` gains any caller outside `tests/`.
- **`web/src/lib/quantumSim.ts` calls a mulberry32 frame a "QOTP perfect-hiding mask"**, keyed on the
  user-visible run seed. Reachable from the public demo. "Perfect hiding" is an information-theoretic
  term and a 32-bit-state non-cryptographic PRG keyed on a value the visitor can type does not
  provide it. This is the smallest and most clearly wrong of the three, and it is recorded rather
  than reworded in the same pass **because a wording change here needs to be made once, correctly,
  against what the demonstration actually supports** - the surrounding code already documents that
  this path is a QOTP-hiding and T-count demonstration rather than a predictor. *Reopens next round
  as a wording fix, unconditionally* - it is page-reachable and no measurement is needed to settle it.
  **DEADLINE EXECUTED Round 40**, and not as the sentence above expected: reading the line changed
  the fix, because the quantum one-time pad genuinely IS perfectly hiding as a scheme. What was
  missing was the disclosure that the demo draws its mask from mulberry32 seeded by a value on the
  page, and that is what `quantumSim.ts:404-414` now carries. See "The claim that was true, and the
  fix that would have been wrong" in Round 40. Left visible rather than deleted because a promise
  that was kept in a different shape than it was made is worth more than a promise that vanished.

Recording all three with conditions rather than fixing two and forgetting the third is the point:
this round's own evidence is that an unconditioned item survives indefinitely.

**A gate added, never watched run, and unable to pass.** The "Combiner vectors are regenerated" step
that closes the stale-vector direction was put in the **web** job, which sets up Node and nothing
else. It failed with `ModuleNotFoundError: No module named 'oqs'` the first time it ever executed -
a round after being written, on the first push that ran it. `scripts/ci_local.sh` could not have
caught it and never could: the virtualenv is always on the path locally, so the identical command
passes. That is the local-green/CI-red asymmetry the script's own header exists to warn about,
reappearing inside the script written to close it.

Recorded rather than quietly moved, because the interesting part is not the misplacement. The step
was correct, its command was correct, and reading the diff it looked right - what was wrong was the
job it sat in, which is the one property a step's own text cannot show.
`tests/test_ci_steps_run_in_a_job_that_can_run_them.py` now parses the workflow and fails when a step
invokes an interpreter its job never installed; run against the commit CI rejected, it names that
step and that job. The local script gained the same check as its own last gate, so the two lists
agree again - which `CI4` has required all along and which nothing enforced.

*This sentence named "gate 13/13" until Round 40.* It was true when written and false one commit
later, because the same round added a fourteenth gate and renumbered the script. The renumbering was
caught by `tests/test_ci_local_step_numbering.py` - which guards the script, not prose about the
script - so a count in a document describing a count in code drifted with every gate green. That is
the third instance of a hand-written count going stale in two rounds, and the reason none is written
by hand any more: the sentence now names the position, not the number.

**A gate nobody has watched run has an unknown result, not a good one.** Three of this round's
findings are that sentence in different costumes: this one, the register deadline with no executor,
and the prefix legend under an instruction to regenerate.

**A fact-check that found one thing worth watching, and it is not a change.** Thirty searches over
2026-08-01 to 2026-08-14 returned the expected null against every claim here, verified at primary
sources rather than at coverage - SP 800-227's document history reads Draft 2025-01-07 and Final
2025-09-18 and nothing else. The exception is ePrint 2026/1591, a **preliminary draft** claiming a
polynomial-time quantum algorithm for the Dihedral Coset Problem, which composed with known
reductions would reach SVP and LWE. It is unrefereed, four of its proofs are marked "(Sketch)", its
final corollary rests on two "Personal Communication" citations, and a full-text search for any
deployed parameter set returns nothing.

It is recorded anyway, with the part that is uncomfortable rather than the part that is reassuring:
**the ML-KEM + X25519 hybrid is not a hedge against this risk.** X25519 already falls to Shor, so
ML-KEM is the only quantum-resistant leg, and a confirmed result reaching ML-KEM's concrete
parameters would not be survived by the hybrid. Neither condition holds today, no claim changes, and
`references/REFERENCES.md` carries the reopening conditions. Writing down the risk the design does
NOT cover is the reason this entry exists; a watch item that only lists reasons to relax is a
reassurance, not a record.

**And the round shipped its own class again, which is the entry that matters most here.** Putting the
QCNN on the same cost axis as CKKS meant timing `qcnnInferTrained`. It has **two** call sites -
`private_inference` for finance and medical, and `qml_ids` for telecom, which exists precisely
because telecom has no `private_inference` stage. Only the first was instrumented, so the telecom
page reported `primary_inference_backend = qcnn-trained` with no latency beside it, while the two
checklist rows written in the same commit asserted the latency renders on every such page. **Rows
that can only fail, added by the round whose stated job was removing rows that can only fail.**

Nothing available could have caught it: `tsc` was green, the Python suite was green, and all
thirteen local gates were green, because both branches are individually well-formed and the defect is
the RELATION between them. It was found by running the deployed telecom page and reading "inference
not run" - the fourth time this project has been saved by looking at the live site after every gate
agreed.

This is the fourth instance of one-fix-to-one-of-two-twins recorded here: the combiner length tag
(Python, not the browser), the ITS combiner PRG (SHAKE-256 against a SHA-256 chain), the federated
mask round index (TypeScript, not Python), and now this. So the response is the structural check
rather than a fifth careful read: `web/scripts/check_qcnn_metrics_at_every_site.mjs` finds every site
that CLAIMS the backend and requires the same metric set at each, keyed on the claim rather than on
the function name - because a call made only for comparison carries no such obligation. Seeded with
the real defect, it names the exact line.

**The generalisation this round is willing to defend.** When the same defect class appears three
times, stop fixing instances and write the invariant. Round 38's disclosure defect was shipped four
times with the scope wrong; the parity test written instead of a fifth fix **failed on its first
run** against a call site nobody had counted. The combiner gate did the same thing: it was written
to stop a known divergence and immediately proved the "one definition" property was not held.

## Round 40: claims that move numbers, and controls that move nothing

**The claim that was true, and the fix that would have been wrong.** Round 39 listed
`quantumSim.ts`'s "QOTP perfect-hiding mask" as a wording fix due this round. Reading it changed the
fix. The quantum one-time pad **is** perfectly hiding as a scheme, and ct/pt = 1 **is** its genuine
consequence, so deleting the phrase would have corrected a true statement. What was false is
narrower: nothing beside the label said the demo draws its mask from mulberry32 seeded by a value on
the page - the run seed on the inference path, a loop index on the benchmark path - so
"information-theoretic" read as a property of the run that had just happened. Disclosed instead, in
the form `SecureAggregator.__post_init__` already uses for its pairwise keys.

The `security_model` string is deliberately **unchanged**, and that is the half worth recording. It
is parsed in three places - `clientCompute.ts:302`, `:342`, and `security.py:48` - to select the
information-theoretic score branch and the quantum-advantage multiplier. Editing that sentence for
readability moves two displayed numbers with every gate green. Measured: `clientCompute.ts` carries
no diff, so no score moved. `tests/test_security_model_prose_is_not_a_silent_score_input.py` pins the
mapping rather than refactoring two languages' scorers onto an enum in the round whose own plan warns
that scoring changes are where numbers move by accident; seeded with the edit this round nearly made,
it fails.

**A hand-written count drifted for the third and fourth time.** `CI1`'s gate total, the prefix
legend, and then the workstream index - each corrected by someone fixing the previous one. The WS
index was the sharpest: Round 39 wrote *"the count above is now derived from the row IDs beside it,
so the two cannot disagree"* having derived only WS2's, leaving WS1 at 20 against a spec expanding to
29 and WS4 at 27 against 35, plus "all 346 rows" against 380 and "Three rows" two lines under the 6
that replaced it. `tests/test_checklist_ws_index.py` expands each spec and requires the count to
equal it and every named row to exist. **It found a further inconsistency on its first run** - WS2's
cell said 6 beside eight backticked IDs - which is the third gate in three rounds to fail
immediately on something nobody had counted.

**The sixth inert control, and the first one measured out of existence.** The "QCNN qubits" field did
not reach the trained circuit: the requested width fed the benchmark row and the untrained fallback
while `loadQcnnWeights` returned a fixed 8-wire file. Making it real required a second width to
exist, so telecom - which encodes 7 features on 8 wires - was trained at both. The padding wire is
**worth 0.079 macro F1** and two of eight classes are better without it; the per-class table is in
`references/qcnn-width-ablation.md` because a single macro number hides that. The refactor was proven
inert before the comparison was trusted, by reproducing the shipped model's `theta` bit for bit.

*Reopens if* a third width is trained, which would make "is 8 the peak" answerable rather than open.

**DESILO FHE: DECLINED, and the answer was available for thirty-nine rounds.** ORDER 1, 2 and 4 asked
for it - ORDER 4 explicitly saying to check commercial usability at the vendor's licence page and
adopt an alternative if it failed. It appears nowhere in this repository, and neither did the verdict.
Fetched 2026-08-14: the DESILO FHE Library Binary Software License Agreement grants a
*"non-exclusive, non-transferable, royalty-free license to use the Software solely for Non-commercial
Use"*, and *"for any Commercial Use, you must obtain a separate license by contacting DESILO Inc."*
**That fails this project's standing constraint of commercially-usable OSS only**, so the absence was
correct and only the record was missing. *Reopens if* DESILO publishes commercial terms that do not
require a per-user negotiation.

**The other mandates that vanished without a decision.** Recorded here because `references/` is where
this project keeps what it declined, and an ask that stops appearing is indistinguishable from one
that was answered:

- **SHARK and RHOMBUS** (ORDER 1, as the GPU performance path alongside HEonGPU and PIGEON/hpmpc).
  Absent, undocumented. *Reopens if* either publishes a browser-reachable or CPU-reachable build,
  since neither helps a demo whose whole compute budget is a visitor's tab.
- **The `submodules/` populate mandate** (ORDER 1-7). Reversed in practice when MPyC became a pip
  dependency in Round 17. *Reopens if* a dependency needs a source-level patch, which is the only
  thing a submodule buys over a pinned package.
- **Pause / Resume / Step to spec** (ORDER 2-7, six consecutive files, then never again). The buttons
  exist in `PipelineControls.tsx` and **no checklist row covers them** - the longest-standing
  coverage gap found this round, and the one most likely to hide a live defect, because a control
  that half-works looks like one that works.
- **`sequence-toolbox/SeQUeNCe`** - `docs/pqc-roadmap.md:59` says "GitHub reports the licence as
  NOASSERTION, so confirm before depending on it", and nothing depends on it, so the confirmation was
  never needed. *Reopens the moment anything imports it.*
- **The EU NIS Cooperation Group PQC roadmap** (`references/REFERENCES.md`) - the promise was
  "Next round: find the Cooperation Group publication itself", written in Round 33. **DEADLINE
  EXECUTED Round 42**: the primary source was loaded, its section 4 timeline is quoted in
  `docs/pqc-roadmap.md`, and the lead turned out to overstate that source twice. The reason it took
  nine rounds is recorded above and is now closed on both axes - `test_register_deadlines.py` reads
  every tracked markdown file under `docs/` and `references/`, and reads `Next round:` alongside
  `by **Round N**` and `BLOCKED UNTIL`.

**WebNN: still declined, and now with the condition it always needed.** `README.md` records the
measurement - probed by building and executing a 2x2 matmul, three cheaper checks having lied, then
left unused because ORT's WebNN provider cannot complete a session for the medical ResNet - and `CS8`
checks it. What it lacked was a reopening condition, exactly as WebGL did before Round 39. The
context moved this round: Chrome 146 promoted WebNN to an **origin trial** and the W3C specification
reached **Candidate Recommendation in January 2026**, while it remains unavailable cross-browser.
Neither fact touches the reason for declining. *Reopens if* ONNX Runtime Web's WebNN execution
provider completes a session for the medical ResNet - a fact about the dependency, checked by running
the probe that already exists, not by re-reading the spec's maturity.

**Round 43: the condition was right and it is now stronger, because on the path where WebNN would
help it cannot be requested at all.** Read off the installed package, `onnxruntime-web` **1.27.0**
(pinned in `package-lock.json`; npm `latest` is still 1.27.0, so nothing newer is being declined):

- `dist/ort.wasm.js:3659` - `if (false) { registerBackend("webnn", wasmBackend2, 5); }`
- `dist/ort.webgpu.js:4466` - `if (true)  { registerBackend("webnn", wasmBackend2, 5); }`

and upstream's own error text, in the shipped bundle: *"WebNN EP requires either JSEP or WebGPU EP to
be enabled."* That is a structural dependency, not a build-flag accident.

`onnxInference.ts` imports `onnxruntime-web/webgpu` when a real adapter exists and
`onnxruntime-web/wasm` otherwise. **So on the no-adapter machine - exactly where WebNN would be the
useful fallback - no WebNN backend is registered and none can be requested.** `status.webnnAvailable`
reports a BROWSER capability the LOADED BUNDLE has no path to consume, and the comment calling the
provider "probed, not used" is understated: on that path it could not be used.

The condition therefore has two halves and **both must be false** before it reopens: (1) the EP still
requires JSEP or WebGPU, so the wasm path can never offer it; (2) on the webgpu path, where it IS
registered, it still could not complete a session for the medical ResNet when measured (162.7 s
against 5.1 s on wasm). Spec maturity - Chrome's origin trial, the W3C Candidate Recommendation of
January 2026 - is not part of this condition and never was.

**Also recorded, no defect measured, so no change made:** two ONNX Runtimes ship. `onnxruntime-web`
1.27.0 is pinned for this project's inference while `@huggingface/transformers` pulls its own nested
`onnxruntime-web@1.26.0-dev.20260416`. Nothing is broken by it; a real artifact with influence and no
record is the shape this project keeps finding, so it is written down rather than acted on.

## Round 41: every gate names a boundary and checks one side of it

Round 40 wrote five checks. **Each names a boundary and verifies one side of it**, and eight findings
fall out of that single fact - the row but not the prose beneath it, TypeScript but not Python, the
demo transport but not the backend, the deploy script's return but not its outgoing half. Each gate
file's own header cites four or five previous one-of-two-twins defects; each was then scoped to one
twin. The three-strikes rule for writing an invariant applies here at eight.

**Two were worse than stale.** `check:qcnn-metrics` could be switched off by a single comment: it
tested `!block.includes(metric)` over raw text, so deleting both `metric()` calls and leaving
`// TODO: restore qcnn_inference_ms and qcnn_qubits here` made it print OK with the metrics gone -
demonstrated with a negative control before the fix. And **Pause was dead in the full deployment**:
`events.py` had only `pipeline_pause`, emitted from inside the gate - semantically the effect, under
the browser's name for the request - so a paused backend run reached `"pausing"` and stayed there,
with Resume and Step disabled forever. Rows `OR7`/`OR8` assert exactly that behaviour and were walked
on the deployed page, which is client-only: the one transport where both events exist. They now say
which side each was verified on, which is the smallest form of the fix this whole round is about.

**The DAG was wrong for every use case on the backend.** `explainability` had no node on four flows,
`qkd_optimize` none on two, `telemetry_anomaly` none on satellite, while finance drew
`federated_training` that `standard_stages()` never runs. Demo mode agreed, so the
"diagram shows a stage the pipeline does not run" defect this file was written to fix was live the
whole time on the transport nobody looked at. `BACKEND_EXTRA` carries them, each labelled with the
deployment that lights it, because a node dim for a good reason and a node dim because the diagram is
wrong look identical.

**Writing the prose-count gate reproduced its own subject, twice.** The first version matched digits
and missed `at six` - one of the two instances it existed for - and its quoted-history exemption was
per LINE, so a live claim sharing a sentence with a quotation was exempted. Both are recorded in the
test rather than quietly fixed, because the pattern is the point: a gate covering one notation and
not the other is the same defect one level up.

**`\gitsha` could never name its own commit**, and that is a fact to state rather than a bug to fix:
it reads `HEAD` at generation time, which is the parent of the commit that ships the output. It was
four behind and asserting a row count three short of the tree. It now uses `git describe --dirty`, so
a working tree that differs from `HEAD` says so instead of naming a clean commit the numbers do not
describe. Separately, `report/build_data.py` rewrote four tracked files **on import** - an audit that
imported it to reuse one function modified the repository. Verified fixed by measurement: the md5 of
`report/generated` before and after an import is identical.

**A signed record named two backends that did not run.** The demo replaces the FHE/MPC selects with a
badge reading "node-seal CKKS + secret-sharing MPC", while the state kept the server defaults
`tenseal-cpu` and `mpyc-shamir` - read nowhere except the ML-DSA-signed payload, under a comment
claiming it records every selection that shaped the run. A signed record of the wrong thing is worse
than an unsigned one, because the signature is what invites a reader to trust it.

**Four areas were re-fact-checked and found sound**, recorded so the next round does not re-derive
them: OptiQKD's `+14-25%` is correctly attributed to AML-QKD (arXiv:2603.04192) with the v1 -> v2
downward revision noted and the repo's own retracted 48.8% marked as retracted; the QRNG reproduces
SP 800-90B section 4.4.1's worked example and gates on the continuous tests rather than SP 800-22;
`references/model-choice-vs-2026-practice.md` carries a dated re-check from 2026-08-13; and the BUSI
licence is genuinely unresolvable - an independent attempt this round reproduced exactly the
"a citation request is not a grant" conclusion `DATA_LICENSES.md` already states.

**A correction to Round 40's own register entry.** It said SHARK and RHOMBUS "appear nowhere". True
of the repository; `submodules/shark/` exists on the deploy box, untracked, along with nine other
trees against the two the repository declares. `.dockerignore` keeps them out of the image. The
sentence was written unqualified about a scope that had only been measured in one place - which is,
again, this round's subject.

## Round 42: four gates that enforce nothing in CI, and a number that is wrong

**The largest finding of the last four rounds is about the gates those rounds added.**
`docs/browser-verification-checklist.md` is gitignored by design - it must not be published - and
four tests read it: the prefix legend, the workstream index, `CI1`'s step numbering, and the prose
counts. Measured against a fresh `git archive` checkout, not argued: **all four SKIP**, and
`pytest -q` still prints PASS. Every clone, and every CI run, has been reporting green for eleven
tests that examined nothing.

Rounds 39, 40 and 41 each wrote one of those gates and verified it here, where the file exists. The
scope was measured on one machine. That is this project's most-repeated defect - a check whose
reported scope is wider than the scope it has - applied to its own enforcement.

**What was done, given the file cannot become public.** The four skip messages now say what is
unenforced rather than that a file is absent, and `scripts/ci_local.sh` gains gate 15: it fails when
the checklist is missing, and fails when the four gates skip *on this machine*. That does not make
them CI gates. It makes this machine the declared enforcement point and removes the one way it could
silently stop being one. **A local-only gate is worth having; a local-only gate that reads as a CI
gate is not, and that is what these were.**

Two consequences worth recording: `report/build_data.py`'s `\nchecklistrows` and `\nchecklistlines`
have the same input, so on any clone they are reused from the committed value rather than computed -
`\nchecklistlines` is currently 4,178 against 4,186 lines and nothing can catch it. And every row
this round added or repaired lives in a file no CI job can read.

**Round 43 correction to the sentence above.** Both numbers agreed again, because `c64dbfe`
regenerated the macros at the end of Round 42. The paragraph describing them was not updated, so this
file went on reporting a drift that its own round had closed - **the number was fixed and the
sentence about the number was left**, which is the same shape as everything else in this section,
committed inside the paragraph that names it.

**Round 45: and the correction ITSELF went stale, which is the third turn of the same screw.** That
sentence was written as "both numbers are now 4,240"; by Round 44 they were 4,244 and by Round 45
they are 4,256 across 407 rows. A correction that pins a literal to a growing quantity is a stale
sentence with a later date on it. The figures are therefore **removed from this paragraph rather than
updated**: `report/generated/macros.tex` holds the current values, `make report` regenerates them,
and prose that needs the count should say where it lives instead of copying it. The lesson this
section keeps re-learning is not "update the number" - it is "do not transcribe a number that has a
generator".

**A promise that outlived nine rounds, and the gate that could not see it.**
`references/REFERENCES.md` ended a metadata-only entry with *"Next round: find the Cooperation Group
publication itself before writing any of it down as a date."* (**DEADLINE EXECUTED Round 42** - the
quotation is history, and the stamp is what says so; without it this paragraph reads as a fifth
pending promise, which is the gate working.) It was written in **Round 33**. It was
still there in Round 42. Nothing malfunctioned: `test_register_deadlines.py` read one file,
`docs/deliverable-verification.md`, and two phrasings, `by **Round N**` and `BLOCKED UNTIL <date>`.
The promise was in another file, in a third phrasing. **The test's own docstring stated that hole**
- "someone can still write 'blocked until September' in prose and no gate will see it" - and stating
a hole is not covering it. This is the round's subject arriving inside the test written to stop
obligations from being forgotten.

Both axes are now widened. The file set is every **tracked** markdown file under `docs/` and
`references/`, defined by `git ls-files` so it is what a clone contains rather than what one machine
has - the same definition, for the same reason, that the checklist finding above turns on. The
phrasing set gains `Next round:` and `Reopens next round`, which are treated as **due on sight**: a
promise relative to a round it never names cannot be waited for, so the only ways to satisfy it are
to do the thing or to rewrite it as `by **Round N**`.

The phrasing match is **case-sensitive**, and that is its whole precision. These files are dense with
prose *about* future rounds that promises nothing - "so the next round does not re-derive", "one
caution for the next round", "instructions for the next round:". Measured before the regex was
written: the case-sensitive form matched 5 lines, all 5 genuine; the case-insensitive form added 3,
all 3 false. A gate that fires on commentary gets an exemption bolted onto it within two rounds, and
this project's tautology class is made of exemptions.

**Four promises were found and all four discharged**, none by deletion. They are listed as separate
items on purpose: the stamp discharges the item it sits in, so a section-wide stamp would exempt any
promise a later round writes into this section - the discharge has to be as narrow as the obligation.

- **The EU lead itself**, in `references/REFERENCES.md`. Done at the primary source this round.
  DEADLINE EXECUTED Round 42.
- **Its quotation in `docs/pqc-roadmap.md`**, which recorded the lead while replacing the "Verified
  absent" paragraph. DEADLINE EXECUTED Round 42.
- **Its entry in this register's own reopening list**, which named the hole precisely - "it carries
  no `by **Round N**` phrasing, so `test_register_deadlines.py` cannot see it" - and then waited for
  someone to act on the observation. DEADLINE EXECUTED Round 42.
- **`quantumSim.ts`'s wording fix**, promised as *"Reopens next round as a wording fix,
  unconditionally"* - and this one was the surprise. It had **already been kept, in Round 40**, and
  was still reading as pending two rounds later. A promise kept in a different shape than it was made
  is invisible to everyone except the person who kept it. DEADLINE EXECUTED Round 40, stamped this
  round with the round that actually did it, not the round that noticed.

**The gate broke on its own stamp, immediately.** Two of the four `DEADLINE EXECUTED Round <N>`
stamps landed near the 100-column margin, the wrap split `DEADLINE` from `EXECUTED`, and the gate
reported both items as pending while they read as discharged to any human. A marker that stops
matching when a paragraph is reflowed would have been "fixed" by unwrapping prose to satisfy a
parser. It now tolerates one line break between its words and refuses to span a blank line, so a
stamp cannot be assembled out of two different paragraphs. Both directions are pinned by a test.

**And the seed went on the side that was added**, which is the discipline this project keeps
relearning: a seeded `by **Round N**` in the register would have proved nothing, because that file
and that phrasing were already covered. The seed is a `Next round:` sentence in `references/`, both
new axes at once. Its first version was itself defective - it asserted a non-empty result, and went
green off a promise that was already in the file and not yet discharged, so it would have passed with
the seeding line deleted. It now diffs against the unseeded file.

**Round 41's defect, and it is mine.** `Orchestration.tsx` wrapped `RealDataScatter` in an
unconditional `<div className="card">` and the comment on the lines immediately above it said the
component *"renders null where no overlay ships, which covers the generic `demo` entry"*. It does.
The wrapper did not, so `/orchestration?usecase=demo` drew an **empty bordered box after every run**,
while `OR12` - written in the same commit - promised "It renders nothing for the generic `demo`
entry". The component's behaviour was checked and the wrapper's was not: this round's subject, one
level in, in code written during the round that named it.

The frame is now a `card` prop drawn after the component's own `return null`, because the caller
cannot answer the question - `demo` has no `REAL_2D_PATH` entry and even a mapped use case can fetch
a payload with no points, an answer that arrives in a `useEffect` after render. A call-site condition
would have been a second and weaker copy of a decision the component already makes.
`tests/test_a_frame_is_not_drawn_around_nothing.py` asserts the class, not the instance: no card in
`web/src` may contain a lone component that can `return null`.

**That gate caught nothing on its first run, and the reason was the comment explaining the fix.**
The comment quotes `<div className="card">`, the card regex matched inside it, and `.*?</div>` then
ran past the seeded defect twenty lines below. A source gate that does not blank comments is at its
blindest exactly where someone took the trouble to write down what went wrong - the QCNN gate was
defeated by one comment in an earlier round and was repaired the same way. Its second failure was in
the same family: the body slice was bounded by `// The genomics model`, a comment the blanking had
just removed, so the slice ran to end of file and the ordering assertion would have compared this
component's `return null` against a different component's frame.

**`QC20` could not be passed by anything, and was rewritten rather than reported FAIL again.** It
asked for "the five-seed spread reported beside any shipped QCNN figure". Two of the four shipped
circuits have no spread - `qcnn_telecom` states so outright, one run being 6 layers x 400 epochs, and
`qcnn_telecom_q7` inherits a single seed - and `grep spread web/src` finds the spread rendered on no
page for any of the four, so the browser half was unsatisfiable everywhere. **A row nothing can pass
is not a standard; it is a permanent FAIL, and a column of permanent FAILs is a column readers learn
to skip.** The row now asks for the property the spread was FOR - can a reader tell whether this
number is typical - which is satisfiable two honest ways: give the seed's position in a measured
spread, or say no spread was measured.

**And the gate for it failed its own negative control.** Written as a prose match over the phrasings
the cards use, it went GREEN when the disclosure was deleted from `qcnn_telecom_q7` - because that
card cross-references the MEDICAL card's spread, and a regex cannot tell a card's statement about
itself from its statement about a sibling. A card could satisfy a check about its own numbers by
mentioning somebody else's. The answer was the one this register keeps arriving at: an explicit
marker nobody writes by accident. `BLOCKED UNTIL`, `DEADLINE EXECUTED`, and now `**Seed position:**`
at the start of a line, on all four cards. **Both of this round's new gates were wrong on their first
run and both were caught by seeding rather than by reading them.**

**The accessibility sweep, ordered by whether a visitor is MISLED rather than inconvenienced.**

*Misled.* The private-inference agreement scatter told `agrees` from `argmax differs` by **fill
alone** - three rounds after the same sweep gave a second channel to `RealDataScatter`, the
histogram, the forecast lines and the benchmark bars. It is the chart where colour carries the most:
red means the ENCRYPTED path disagreed with plaintext, the one fact the chart exists to show. It was
missed because the charts were enumerated from memory, and the round's own plan had named that as the
likeliest way this item would ship a defect. `tests/test_chart_accessibility.py` now enumerates
SERIES from the source. The DAG had the same shape one level up: `.node-active` is a blue ring and
`.node-done` changes `border-color`, so "which stage is running" was **colour only** - and React Flow
renders it as a class on a text-free `<div>`, so it was unavailable in text as well. Both channels
added: `[running]` / `[done]` on the node label, and an `sr-only` ordered list built from the same
`retitle()` and `stateOf()` calls the nodes use, so it cannot describe a different graph. Agreement
printed as a bare **0.98** - a dimensionless decimal that reads as a score, a probability or a ratio,
where `UX5`'s own worked example is "agrees 98.3% of the time"; now a percentage, at one decimal,
because `toFixed(2)` on a fraction gave two significant figures and 0.9825 and 0.9849 both printed
0.98.

*Blocked rather than misled.* `RunMonitor`'s empty row had one cell where every other row has two, so
it announced as a row with a column missing. Six export buttons across two pages put their reason in
a `title` on a `disabled` control - and `disabled` removes a button from the tab order, so the reason
was reachable by hovering a mouse and by nothing else; one visible sentence now says it to everyone.
`/orchestration` had **no live region** while `/usecase/<id>` has had one since Round 39 - the page
whose whole subject is watching a pipeline advance was the page that did not announce it. And **2.4.1
was failing outright**: the sidebar's eleven links were the first eleven tab stops on every page, on
every navigation, with no way past. A skip link now targets `<main id="main" tabIndex={-1}>`, and the
`tabIndex` is the half that is easy to ship broken - without it the browser scrolls and leaves focus
on the link, so the block is skipped for the eye and not for the keyboard.

**Three criteria had no row at all** - 4.1.3 Status Messages, 1.3.1 Info and Relationships, 2.4.1
Bypass Blocks - and two of the three were failing when a row was finally written for them. That is
the same finding this file already recorded about scoping the accessibility list: *"a checklist that
names the criteria it will check is asserting that the rest cannot fail."* It named four, then seven,
and the criteria outside the list kept being the ones that failed.

**Two open items are now open by DECLARATION rather than by omission**, which is the only difference
that lets them be counted. **1.4.11 Non-text Contrast**: added as a bullet in Round 34, no
measurement recorded anywhere since, so nothing said it was outstanding - dated unmeasured as of
2026-08-15. **1.4.13 Content on Hover or Focus**: re-measured, and it has grown - **10** `<Tooltip>`
sites now, 3 in `BenchmarkPanel` and 7 in `VerificationChart`, still zero `trigger` props, zero
Escape handlers, zero keyboard paths. Recharts' default tooltip is neither dismissible nor
persistent, so closing it means a custom tooltip in every chart; that belongs to a round that chooses
it. **Heading levels skip** `h1` to `h3` on three pages because the app's convention is h1-for-page
and h3-for-card in the page files AND the components; half-fixing the page files would leave the
component headings skipping, so it is recorded with a date instead of rushed. None of the three
misleads a reader, which is why they are below the ones that do.

**And the standing UX bullet cannot be answered by `references/`, which nobody has said in nine
rounds of being asked.** Measured rather than asserted: a grep of all 16 markdown files for
`usability`, `accessibility`, `WCAG`, `human-computer`, `human factors`, `affordance`, `user
experience`, `HCI`, `cognitive load` and `eye-tracking` returns **zero** hits; widening to `interface
design`, `visualis/visualiz`, `readability`, `legibility` and `colour-blind` returns two, neither a
source; all 15 bundled PDFs are quantum, cryptography or business-of-quantum papers. There is no HCI
source in the reference set at all. So the UX rows rest on the **WCAG 2.2 criteria themselves** and
on the checklist's own measurement procedures, and `references/REFERENCES.md` now says so. It is
deliberately NOT resolved by adding one HCI citation to make the sentence true, because a citation no
row derives from is the decoration-condition failure `QC16` exists to name.

**The accessibility fix shipped a blank chart, and only the deployed page said so.** Adding
`shape="circle"` / `shape="cross"` to the agreement scatter moved recharts off its default renderer
onto `<Symbols>`, whose `size` is an **animated** prop. With animation left on, all 30 symbols sat in
the DOM, correctly positioned, rendering `d="M0,0"`: a legend naming two shapes above an **empty plot
area**, which is worse than the colour-only version it replaced. `RealDataScatter` in the same file
has carried `shape` AND `isAnimationActive={false}` together since Round 34; half of that pattern was
copied - the half the checklist row named.

Every gate was green. They read the source, found the `shape` prop they asked for, and have no way to
see a path of length zero. The pairing is now its own invariant, which is what "write the invariant
at the boundary" means when the boundary is between a prop being PRESENT and it having an EFFECT.
**This is the third gate this round that was wrong on its first run**, and the only one of the three
that no seeding would have caught, because the seed and the subject were both in the source and the
defect was in the render. `bash scripts/ci_local.sh` cannot replace opening the page.

**Two findings from the Chrome pass that are NOT this round's subject**, recorded with conditions so
they cannot become permanent residents:

- **Switching use case after a run leaves the previous run's model in the DAG caption.** Measured
  2026-08-15 on the deployed site: run `finance_aml`, switch the selector to `demo`, and the QHE/FHE
  node still reads `(finance_aml.gbm on wasm)`. A fresh load with `demo` selected reads
  `Private Inference`, so the caption is correct until a run has happened and stale afterwards. It is
  this round's own class - a caption naming something other than what ran - and it is left for a
  round that can change the run-state lifecycle rather than bolted on at the end of this one.
  **Resolve or condition by **Round 44**. DEADLINE EXECUTED Round 43** - and the discharge was
  written a round before the stamp. Round 43 rewrote `usePipelineRun.ts`'s `[usecase]` effect to
  call `reset()` and to cancel the in-flight run, and verified it on the deployed site across four
  surfaces: the DAG caption went from `(finance_aml.gbm on wasm)` to `Private Inference`, the badge
  from `done` to `idle`, the exports from enabled to disabled, and the agreement card from present
  to absent. The work was done and the sentence describing it still read as pending, which is the
  same shape as the count sentence two sections up.
- **`/orchestration?usecase=demo` ignores the query parameter** and renders `finance_aml`. Nothing
  links to that form, so nobody has hit it, but a URL that names a use case and shows a different one
  is a link that lies. **Resolve or condition by **Round 44**. CONDITIONED Round 44, by **Round 46**.
  DEADLINE EXECUTED Round 45.** Round 44 spent its length on the rate fields and the two defects
  behind them, and this is a separate surface: `Orchestration.tsx` held the use case in `useState`
  and never read `useSearchParams`, so resolving it meant deciding what an unknown or misspelled
  value does - fall back visibly, or refuse - and that is a UX decision rather than a one-line read.

  **Round 45 resolved it, and the reason it did rather than re-dating it a third time is worth
  recording.** This item was written "resolve **or condition**", was conditioned in Round 44, and
  said so in its own text - and it would still have gone red the moment `## Round 45` was written,
  because the gate accepts only a `DEADLINE EXECUTED` stamp. That is deliberate: the gate's own seed
  says the conditioning-sentence exemption "is exactly the bug this test was rewritten to remove",
  since otherwise "conditioned" is an unbounded deferral device, which is how one sentence in
  `references/REFERENCES.md` reached its ninth round. **The gate was right and the register was
  wrong.** An item on its third date with a one-line read behind it is not a scheduling problem.

  What shipped: `usecaseFromQuery()` validates `?usecase=` against the canonical `USECASES` list;
  a known id is selected, an unknown one falls back to `finance_aml` **and the page names the
  rejected value** ("There is no use case called X. Showing Y instead"). Read once as the initial
  value, so a visitor changing the selector is not reverted by a stale query string. Silence would
  have reproduced the original behaviour with intent added, which is why the notice is the half that
  makes it a fix. Guarded by
  `tests/test_a_page_renders_what_the_path_it_runs_produces.py::test_a_url_that_names_a_use_case_shows_it_or_says_why_not`,
  which asserts all three of: the parameter is read, it is validated against `USECASES`, and a
  rejection is reported.

**A stated limit on this round's keyboard verification.** The browser harness could not deliver
synthetic `Tab` or `Enter` to the page - focus did not move off the element it started on - so the
skip link's tab-order behaviour was NOT verified by walking it. What WAS verified on the deployed
site: it is the first focusable element in the document (`querySelector` of all focusables returns
it), it becomes visible on focus (`left` goes from `-9999px` to `0px` with a real bounding box), and
activating it moves focus to `<main id="main" tabIndex={-1}>` rather than only scrolling - checked by
reading `document.activeElement` after activation. The unverified step is that a physical Tab reaches
it first, which its position as first focusable establishes structurally but not empirically. Said
here rather than reported as a clean keyboard walk, because an over-broad browser assertion that
reported a false FAIL is already in `docs/verification-log.md` as this project's own warning.

## Round 43: the licence column truncated the caveat, and the record stopped being kept

**The published deliverable overclaimed by truncation.** `report/build_data.py` rendered the Data
licence column as `esc(s.dataset.licence.split(";")[0][:22])` - two truncations in one expression,
both cutting from the end. Of the **seven** licences that reach the report, **four are altered and
three lose their qualifier**:

    CC-BY-4.0 asserted by MedMNIST; the upstream BUSI grant is NOT verifiable  ->  "CC-BY-4.0 asserted by "
    per-score; EMBL-EBI Terms of Use by default                                ->  "per-score"
    provider's own pages state conflicting terms                               ->  "provider's own pages s"

"CC-BY-4.0 asserted by " tells a reader the corpus is CC-BY-4.0; the sentence exists to say the
upstream grant behind that assertion is not verifiable. **Truncation is biased toward overclaiming,
because a caveat comes last.** The licences now print whole in their own table and the models table
names the corpus. The gate asserts no rendered cell is a **proper prefix** of a source licence, which
catches `split(";")` and `[:22]` with one predicate - a length check would have caught the second and
sailed past the first.

**An earlier draft of this round's plan said "6 of 11 altered, 5 losing a qualifier".** Four of the
eleven `Dataset` objects in `registry.py` are attached to no `ModelSpec` and never reach the table.
An overstatement of a defect about overstatement, caught by counting rather than trusting the count.

**The record stopped being kept.** `docs/verification-log.md` ended at Round 38; Rounds 39-42 have no
entry, the third instance of a gap that file already documents twice. It now carries a **pointer**,
not four reconstructed entries - the measurements exist in this file's own per-round sections, and a
reconstruction presented as a record is worse than an absent record, because the absence is visible
and the imitation is not.

**Three claims this audit made about itself were wrong**, and correcting them is the point:

- **`RI9` was not outstanding.** Its five-way table is transcribed at `verification-log.md:434-444`.
  Grepping that file for `RI9` returns zero hits because the measurement sits under its own heading
  and does not name the row. **A search for the label is not a search for the thing.** Only the
  smuggled-well-formed-key case remains, conditioned `by **Round 45**`. **DEADLINE EXECUTED Round
  45** - run as a bad re-vendor (truncate to 770 vectors, recompute the SHA so the hash gate passes)
  and the count assertion caught it, exit 1. Full output and the scope limit are recorded with the
  RI9 item later in this file.
- **`PF7` was not violated.** 84.35 + 506.69 does not reconstruct 1,269 because 1,269 is
  `encrypted_tier_measured_ms`, the whole tier, while `fhe_encrypted_student_ms` is 579. The identity
  holds by construction in `clientFHE.ts`. The real defect is one sentence calling 1,269 "the whole
  encrypted student" and then naming 579 as that field - two names for one quantity.
- **The finance agreement pair was already past tense** at both sites. What it lacked was a round
  anchor, so `0.947` read as current where `models/metrics.json` says `0.982943`. Both now say
  Round 24 and name the current value.

**A use-case change cleared one field of the run.** `usePipelineRun.ts` carried the rule - "anything
keyed to the use case has to be reset when the use case changes" - and implemented `publishedModel`.
The nine fields a run writes survived, so switching use case after a run left the previous model in
the DAG caption, its events under the new use case's export filename, and its agreement chart under
the new selection. `reset()` was itself incomplete at seven of nine.

**The QCNN width control now offers only widths that ship, and training one revealed two more
things.** 7 wires is impossible for finance and medical - both encode 8 features and the trainer
refuses a narrower run - so the spare-wire question runs the other way, 8 to 9. The trainer wrote
`models/qcnn_<usecase>` at every width, so the reproduce command recorded in `qcnn_telecom_q7` would
have **destroyed the 8-wire artifact**; the output path is now a function of the width. And **the
width result reverses at five seeds**: seed 0 says +0.099 (finance) and +0.081 (medical) for the
wider circuit, the median says -0.034 and -0.035. The sign flips on both, for the reason Round 42
recorded - medical's seed 0 is the worst of five at 8 wires.

**The WebNN condition is stronger than it was written.** In the installed `onnxruntime-web` 1.27.0,
`ort.wasm.js:3659` has `if (false) { registerBackend("webnn", ...) }` against `ort.webgpu.js:4466`'s
`if (true)`, and the bundle carries upstream's own *"WebNN EP requires either JSEP or WebGPU EP to be
enabled."* On the no-adapter path - exactly where WebNN would help - it cannot be requested at all.

**BUSINESS DOC RECONCILED Round 44.** The confidential business-recommendations document - its
filename is Japanese and is therefore not written here, which is why this paragraph describes it
instead - lives at the parent directory, **outside this repository**, and 21 consecutive ORDER files
(18-38) name it in the documentation-merge bullet. Its own title says it covers rounds 17-20. Every document sweep for
twenty-three rounds walked past it, because the sweeps walk the repo and it is not in the repo -
the same scope defect this project keeps finding, applied to the one file it was told about by name.
It is confidential-class like `MONETIZATION*`, so it stays out of the repository and **its content is
not recorded here or anywhere tracked**. `scripts/ci_local.sh` gate 16 now fails when this stamp is
older than the register's latest round. Local-only by construction, as gate 15 is.

**`docs/deployment.md` had zero inbound links** while being the only document carrying the demo's
port-collision workaround, and the technical report had no `make` target. Both fixed, and the orphan
question is now asked of `docs/` and `report/` - the same question `test_model_artifact_inventory.py`
asks of `references/`, whose scope was one folder.

**Fact-check, 2026-08-15.** 55 agents, six lanes, refuted-by-default: **6 claims survived, 42 were
refuted, 61 negatives recorded.** The Federal Register API returns `"count":0` for post-quantum
cryptography since 2026-08-01; arXiv cs.CR's newest is 2608.13465 (2026-08-13) and IACR 2026/1645 is
404, so **2026-08-14 and 2026-08-15 are structurally empty** - a sweep of those days is a sweep of an
empty set. One correction against our own interest: NIST CSRC has **two** August 2026 items, not one;
the added item is a sector CSF profile, so the conclusion holds and the word "only" was wrong.

**Where this round shipped its own defect.** Three, all caught by running rather than reading:

- The licence gate's scope was `models.tex`; when the licences moved to `licences.tex` **as part of
  the fix**, it went red on all seven and reported correct output as broken.
- The seed-position gate enumerated `qcnn_*/README.md`, so a new artifact with **no card at all**
  satisfied "every card documents its seed". It now enumerates `weights.json`.
- The new `PM5` row collided with an existing `PM5`, and `report/build_data.py::_count_rows` refused
  to build - the duplicate-ID guard a previous round added, catching this round's row on the first
  rebuild after it was written. Renamed `PM6`. **The guard was worth more than the care that skipped
  it**, which is the argument for having it.

And gate 6 caught gate 16: the first version put the business document's Japanese filename into both
`ci_local.sh` and this file, and the no-Japanese-in-public-files rule failed on the check written to
enforce a different rule. The register now describes that file rather than naming it.

## Round 44: the fix changed one language, and the field it emptied is still being read

**Round 42's own fix did both things it was written to prevent, and its guard caught neither.**

**It changed TypeScript only.** `src/qsh/qkd/sdn_controller.py` went on budgeting a bit COUNT against
`eskr` and accumulating granted BITS into `expected_consumption`, both documented `#: bit/s` - the
exact operation Round 42 removed from the browser. The guard written alongside,
`test_sdn_fields_carry_the_dimension_the_spec_says.py`, asserted the bit-denominated ledger against
the TS file and read the PY file **only for the unit comment**. That file is the dimensional
reference the gate cites when judging the browser, so the reference was the thing that was wrong.

**And emptying `eskr` broke its reader, silently.** `stored_key_count` still computed
`(link?.eskr ?? 0) / 352`, so the GS QKD 014 table advertised a key management entity holding
**exactly zero keys, on every run, for two rounds**. `null ?? 0` is 0; nothing raised, nothing
logged, no test failed. `tests/test_sdn_qkd.py` does assert `stored_key_count > 0` - against the
**Python** `KeyManagementEntity`, a different implementation that was never changed. **Round 43
walked this exact panel**, recorded the `n/a` and the `1,436 bit` beside it, and did not record the
`0` between them. Reading a panel for the thing you changed is not reading the panel.

**The rate fields now carry what their names promise.** The gap between a per-pulse key rate and
bit/s was exactly one scalar, and it is not measurable here: a source repetition rate is a property
of a transmitter, stated in a paper's experimental section rather than derived from one. So it is an
explicit operator input - `source_rate_hz`, defaulting to the 100 MHz of arXiv:2607.24118v1 (CC BY
4.0, no journal DOI), now promoted into `references/REFERENCES.md` because a shipped default may not
rest on a code comment. Two keys, not one: on the satellite page that clock is the anchor's own
measured configuration, and on fibre it is the same number carried to different apparatus. One key
could carry only one of those sentences.

**`eskr` has no storage in either language.** It is `linkEskr()` in the browser and a read-only
property in Python, so `allocate()` cannot decrement it - a field that does not exist cannot be
spent. A comment asking nobody to spend a rate is weaker than removing the thing to be spent.

**Round 42's stated conclusion was broader than its evidence**, and this is the correction rather
than a reversal. It surveyed three libraries and rejected them on three DIFFERENT grounds -
per-pulse, GPL licence before the physics was examined, "would still require asserting a clock" -
then generalised to *"No library supplies a repetition rate because none can."* Round 44's sweep
found that **SeQUeNCe does model a source clock**, with a default of 8e7 Hz. A default is not a
measurement, so the operative conclusion survives; the sentence did not. The narrow claim is that a
repetition rate is an INPUT to key-rate analysis, which is the argument for asking openly.

**Two things that must not be written**, recorded because the research pass had to correct them
before they entered a document: *"QuTiP's time axis is dimensionless"* is false - setting hbar = 1
identifies energy with inverse time rather than removing the unit, and qutip-qip documents hardware
parameters in GHz, making that axis nanoseconds. The defensible statement is that qutip core exposes
no source-clock, repetition-rate, channel or link-budget abstraction. And *"Boaron et al. reach
6.5 bit/s over 421 km"* couples the wrong rate to the wrong distance: 6.5 bps is at **405 km**, and
the 421.1 km row is **0.25 bps**. A 26x error in a reference point is this project's own defect class.

**An uncited number was removed rather than promoted.** *"2.5 GHz in the 64 Mbps fibre
demonstration"* had a single occurrence in `clientSdnController.ts`, no source anywhere in the
repository, and was doing argumentative work inside the rationale for not computing a rate at all.

**The deadline gate was blind to its own line wrap.** `by\n**Round 45**` at `:1992` was invisible to
`_DEADLINE`, which required a literal space - while `_EXECUTED`, twelve lines away in the same file,
had been made wrap-tolerant in Round 42 for exactly this reason. All three markers now share `_WRAP`.
The lesson was learned on one regex and not carried to its sibling.

**Where this round shipped its own defect.** The research sweep, reading the live tree, found that
`src/qsh/orchestration/stages.py` put a per-pulse fraction into a metric named `skr` - **after** I had
fixed that collision in the browser and written a commit message about Round 42 having changed one
language. Three Python sites renamed together. The gate that asserts the emptied fields are nullable
also fired on my own change, and its premise was restated rather than deleted; its module docstring
said the same stale thing and was corrected in the same edit.

**Verified on the deployed site**, which no source gate can do: `skr_per_pulse` 0.00958 x
`source_rate_hz` 1e8 = `skr_bit_s` 957,659; `channel_time_s` 0.002 = 200,000 / 1e8;
957,659 x 0.002 = 1915 = `secure_key_bits`; panel `skr` 958,000 and `eskr` 718,500 =
958,000 - 479 bits / 0.002 s; `key budget` 1,436 bit; **`stored_key_count` 4 = floor(1436/352)**,
restored from a permanent 0. No NaN, no Infinity.

## Round 45: the number got a unit, and the unit made it wrong in public

Round 44 gave the secret-key rate a genuine bit/s and put ~958 kbit/s on the public demo. It did not
ask whether the per-pulse fraction underneath was the quantity the page said it was. It was not.

**BUSINESS DOC RECONCILED Round 45.**

### The rate was not the bound the page named, and the gap was 4.07x

`web/src/lib/quantumSim.ts` computed $`1 - h_2(q) - f\,h_2(q)`$ on the **observed** QBER - the
non-decoy GLLP form - while seven surfaces called it a decoy-state bound. `taggedFraction()` had
existed since Round 29, was called exactly once, and its only consumers were a display string and a
warning: the decoy estimate was computed, printed beside the rate, and never fed it.

The gap was not a coefficient. Two departures from physics rode along, and **both were stronger than
a decoy bound rather than approximations of one**, so the label named a technique that would have
*lowered* the figure:

- `nSift` used the transmittance $`\eta`$ as the per-pulse detection probability. For a weak coherent
  source that is $`1 - e^{-\eta\mu}`$, so using $`\eta`$ asserts $`\mu \approx 1`$ photon/pulse. Real
  decoy systems run $`\mu \approx 0.4`$-$`0.5`$.
- every sifted bit was credited as single-photon, $`p_1 = 1`$.

Measured at the demo's own operating point, 50 km on the shipped fibre defaults, against this
repository's own `src/qsh/qkd/bb84.py::_decoy_secure_fraction`:

| model | bit/pulse | bit/s at 100 MHz |
|---|---|---|
| displayed before | 0.009434 | 943,371 (958,000 live, seed-dependent) |
| two-intensity decoy, Ma et al. 2005 | **0.002355** | **235,479** |
| non-decoy GLLP counterfactual | 0.000000 | 0 |

`bb84()` now calls `decoySecureFraction()`, a transcription of the Python function rather than a
re-derivation - the repository already owned this algebra twice, and a third unchecked copy is the
recurring defect here, not a safeguard. Detections come from the gain
$`Q_\mu = Y_0 + 1 - e^{-\eta\mu}`$, which halves the sifted count at the defaults; that is the
correction, not a regression, since the old figure counted one detection per transmitted photon.

**The external check.** Yuan et al., *IEEE/OSA J. Lightwave Technol.* **36**, 3427 (2018): the T12
protocol gives over 1 Mb/s at 1 GHz across 50 km of fibre, $`>0.001`$ bit/pulse, and the same paper
says T12 sifts about twice as efficiently as the standard BB84 modelled here. The old display was
~9.6x a real deployed link. The new one is ~2.4x it, which the optimistic detector efficiency (0.25)
and the asymptotic treatment account for. Cited, not bundled: its arXiv licence grants nothing.

**The bound is no longer seed-dependent.** It takes the intrinsic error rate, so it is a property of
the channel; the measured QBER still varies. Five seeds previously produced five different "bounds",
0.008992 to 0.009577, which is why the public figure moved between page loads.

**The counterfactual is zero, not merely smaller.** At $`\mu = 0.5`$ the tagged fraction is 7.26, so
no detection can be certified single-photon - without decoy states this link yields nothing at all.

**E91 was left alone, because GLLP on the observed QBER is correct there.** Its defect was the label:
the secret-key-rate row was emitted OUTSIDE the protocol branch three lines above it, so the E91 page
attributed its rate to decoy states - a countermeasure for the multi-photon pulses of an attenuated
laser, which entanglement-based QKD does not have. One shared `isE91` constant now feeds both rows.

### The first numeric cross-language gate, and why it had to be new

Every existing cross-language QKD check is **textual**: one greps the Python file for a unit comment,
and `web/scripts/check_qkd.mjs` deliberately asserts only plumbing and identities inside TypeScript,
because a mirror written in TypeScript would copy a TypeScript bug. Both are right about what they
do. Neither can see that two languages compute different physics.
`tests/test_the_browser_computes_the_bound_the_page_names.py` executes the browser through esbuild +
node and compares against the Python function evaluated live.

**Its first version asserted the wrong direction and the model corrected it.** It required the
non-decoy counterfactual to be LARGER, reasoning that the browser's old number was larger and was
therefore the non-decoy one. It was neither: crediting $`p_1 = 1`$ is an unbounded estimate above
both. Recorded in the test rather than quietly swapped - mistaking "the number that was displayed"
for "the number without the certification" is how this survived seven surfaces.

### Three defects Round 44 shipped, all the same shape

- **`/validations` rendered a blank cell on the public demo.** The client branch of `demoQkd` still
  returned `skr` while the page read `skr_per_pulse`, and the demo builds `CLIENT_ONLY`, so that
  branch is the only one it executes. Three producers and the reader were renamed; the one that ships
  was not. `SW3` could not catch it - an `undefined` React child renders as nothing, not as the token
  `undefined`. The comment above the broken line still called the rename "conditioned rather than
  done here", a sentence falsified in the same commit that broke the page.
- **Two rendered strings promised a "Source rate control in the Parameters panel" that did not
  exist.** Six INERT controls have been found here - rendered, read by nothing. This was the inverse
  and worse: a visitor could not even go and look. The control now exists, one per channel
  (`qkd_source_rate_mhz`, `sat_source_rate_mhz`), separate for the same reason `qkd_distance_km` and
  `sat_distance_km` are.
- **`satellite_comms/__init__.py`** wrote a per-pulse fraction into `ctx.metrics["skr"]` three lines
  below its own comment saying that renaming one write and not the other is the asymmetry Round 44
  had just found. It repeated it under its own warning.

### A gate that failed on the change that introduced it

`test_every_rendered_control_is_read_somewhere` asserts the inert direction, which nothing had ever
checked. The first `sourceRateHz` took a config object and a KEY NAME and read `config?.[key]`
internally, so both controls added this round were read at runtime and invisible to static analysis -
and the new gate reported both as inert. It was right about what it could see. The call sites now
pass `config?.qkd_source_rate_mhz` directly.

### Every deadline due was discharged, and one was on its third date

- **`/orchestration?usecase=`** now reads, validates against the canonical list, and **names a
  rejected value**. Silence would have reproduced the original behaviour with intent added. The item
  said "resolve **or condition**", was conditioned in Round 44, and would still have gone red the
  moment `## Round 45` was written - the gate accepts only a `DEADLINE EXECUTED` stamp, by design,
  because otherwise "conditioned" is an unbounded deferral device. **The gate was right and the
  register was wrong.**
- **`_DEADLINE` was blind to a bold span. DEADLINE EXECUTED Round 45**, and this bullet is why the
  marker is now emphasis-aware rather than adjacency-based: **the gate fired on this sentence**.
  Quoting the pattern it had just fixed put two live `by Round 45**` matches inside a narrative
  item, indistinguishable from an obligation, and both would have gone red the moment `## Round 46`
  was written. Round 46 made the marker case-sensitive, which excludes narrative without excluding
  this. `references/qcnn-width-ablation.md:127` writes
  `**Re-run it by Round 45**`, so the literal text is `by Round 45**` with no `**` before `Round`.
  That file returned **zero** matches at every value of `now` while carrying an obligation due this
  round. Round 44 taught the marker to tolerate a line wrap and never asked what else could sit
  between the words. The closing `**` stays required, measured rather than assumed: dropping it too
  sweeps in four narrative sentences that are not obligations. `_EXECUTED` and `_BLOCKED_UNTIL`
  contain no asterisk and cannot be hidden this way - stated rather than left silent, because "fixed
  one regex of a sibling set" is the defect being repaired.
- **Telecom's pad wire, re-run at five seeds, and it behaves unlike the other two.** Seed 0
  reproduced the card's 0.8098 exactly. The five-seed median gap is **+0.0786** against seed 0's
  **+0.0786** - identical to four decimals, because seed 0 is the median draw for both widths.
  Finance went +0.099 to -0.034 and medical +0.081 to -0.035; this one did not move. It still does
  not separate the widths: the ranges overlap and the best single result in the whole experiment is a
  **seven**-wire draw, 0.8616, above all five 8-wire runs.
  **SUPERSEDED IN ROUND 47, AND THE SIGN FLIPPED.** Every figure in this bullet was measured on the
  hardware-efficient ansatz Round 46 replaced, through a readout Round 47 found was binning the class
  index over a wire pooling had discarded. On the READOUT-CORRECTED circuit, same seed and same
  command, telecom was **0.500452 at 7 wires against 0.458324 at 8** - the narrower one led by
  0.042128. **Those two figures are THEMSELVES SUPERSEDED by the continuation section below; the
  reversal is not.** Current, read from `models/qcnn_telecom_q7/weights.json` and
  `models/qcnn_telecom/weights.json` after the analytically dead parameters were made live:
  **0.541894 at 7 wires against 0.510963 at 8 - the narrower one still leads, by 0.030931**, a margin
  that NARROWED by 0.011197 rather than opening. The five-seed spread was not re-run, so both sides
  of the reversal are single-seed. `0.8098`, `+0.0786`, `0.8616`, `0.500452`, `0.458324` and
  `0.042128` are all retained as what was measured then and must not be quoted as current; see the
  Round 47 section and `references/qcnn-width-ablation.md`.
- **RI9's smuggled well-formed key**, run as a bad re-vendor. Full output with the RI9 item above.

### `references/` PDFs: adjudicated per file, and the old rule was inverted

`.gitignore` excluded on **size** and one copyright call, never on licence, and the outcome was an
inversion - the two most permissively licensed papers in the project, both CC BY 4.0, were dropped
for being ~17 MB each while a CC BY-NC-ND camera-ready shipped to every clone. Seven kept, four
removed, each verified at primary source. Full table in `DATA_LICENSES.md` section 8, which is also
where the file gained a **fourth preamble bucket**: its taxonomy had three and none covered "a
third-party document redistributed verbatim", so 14.5 MB - about 45% of everything tracked - had
never been in scope for the file that owns licence findings.

**The trap worth recording.** arXiv's Atom API (`export.arxiv.org/api/query`) carries no licence
field at all and reports "not present" even for a CC0 paper. An adjudication run through it reads the
entire corpus as unlicensed. OAI-PMH returns a real `<license>` element; three of the seven kept
papers are kept because that distinction was checked.

### Mathematical notation: GitHub markdown math, adopted where it is safe

Display equations use `$$...$$` and inline uses `` $`...`$ `` - the code-span form, which degrades to
visible backticks in a non-rendering viewer and cannot be captured by a stray `$`. GitHub documents
four delimiter forms and no per-surface difference between them.

**The currency risk was measured, not assumed.** `README.md` and `docs/verification-log.md` use `$`
for dollar amounts. GitHub's server-rendered HTML for `main` contains both as literal text with zero
math markers, against positive controls that do show them - so those lines are safe as *observed
output*, not as derived from a rule. The widely repeated tokenizer heuristic ("the closing `$` must
not be followed by a digit") is **NOT FOUND** in any GitHub primary source, and `github/cmark-gfm`
contains no math extension at all, so there is no specification to cite.

**This paragraph cited line numbers and both had drifted.** It named `docs/verification-log.md:531`,
which carries no `$` at all; the dollar amounts had moved to `:543` and `:545` by Round 49 and to
`:552` and `:554` by Round 50, twice in two rounds, because every insertion above them moves them
again. So the evidence for a claim about `$` was a line containing no `$`, and the re-scan condition
pointed at the wrong place.

Corrected Round 50 by citing the CONTENT instead of the line: the risky configuration is a line
carrying **two `$` on one line**. Find them with `grep -nP '\$[^$]*\$'` rather than by line number.
There is exactly one such CURRENCY line in each file - the PQC market sentence in `README.md` and the
market-spread sentence in `docs/verification-log.md`. Re-scan when either is edited. [Round 58,
2026-10-24: both sentences were removed, and so were the figures this paragraph quoted from them - the
project's public material carries no market-size figures - so neither file has a currency line now.]

**That grep also matches deliberate math, and saying "exactly one per file" without qualifying it was
wrong when first written here.** `docs/verification-log.md` returns two hits: the currency sentence,
and the QSDC secrecy-capacity inline converted in this same round, which is a `` $`...`$ `` code span
and is *supposed* to carry two `$`. The two cases are told apart by the backticks, not by the count.
A reader running the grep will see both, so the distinction belongs here rather than being discovered
as a surprise - which is how the line-number pointer failed in the first place.

A line number is a pointer into a file that changes; the sentence is a pointer into a file's meaning,
which is what the claim is actually about.

**THE REMAINING BACKLOG IS MUCH SMALLER THAN THE COUNT SUGGESTS, measured Round 54.** A grep for
backticked `x = y` across `README.md`, `docs/` and `references/` returns 325 hits, and earlier audits
put "~46 formulas remaining". Sampled, almost all of them are CODE rather than mathematics -
`use_bootstrap=True`, `shape="circle"`, `on_logit=True`, `auroc=null`, `d="M0,0"`, `now = 45`. In this
file, 31 hits contained **three** genuine physics quantities, and two of those were already converted.

So the honest statement is that the conversion is largely DONE for the things the house style is
about, and the residual count is an artefact of a pattern that cannot tell `p_1 = 1` from
`hm_source=Author-reported`. Converting by count would turn config keys into mathematics, which is
worse than leaving them.

**AND THAT MEASUREMENT WAS TAKEN WITH A PROBE THAT COULD NOT SEE THE REMAINING CASES. Round 55.**
The Round 54 sweep greps for **backticked** `x = y`. Bare prose mathematics has no backticks, so it
is invisible to that pattern by construction, and "largely DONE" was concluded from a search that
could only ever return the already-marked-up cases. This is the mechanism-versus-outcome shape again:
the probe measured a proxy for the property rather than the property.

Re-run with an outcome-shaped probe - scan for Greek-letter names, entropy and ratio expressions
OUTSIDE code fences and outside `$...$` - it finds bare sites the backtick grep cannot: `mu = 0.5 /
nu = 0.1` at :2402 (whose converted twin sits in `verification-log.md`), `R/eta ~ 0.11` at :3937
three lines from its own $`R/\eta`$, `q = 1/2` at :3141 on the line after a converted $`q = 1/2`$,
and `(g = 2.57)` in `REFERENCES.md` three lines under its own `$$` definition. All four converted.

The "do not blanket-convert" list below still stands and grew: `mu`, `nu` and `eta` at :2627 are
real Python parameter names in `decoy_reference.py`, :3929 is sigma-as-a-unit, :2618 is a table
header, and :3939 and :2360 are verbatim quotations. Four sites converted by hand, five deliberately
left.

**A bulk edit here damaged four already-correct sites before it was caught.** Replacing
`` `p_1 = 1` `` with `` $`p_1 = 1`$ `` double-wrapped the sites that already had the delimiters,
producing `` $$`p_1 = 1`$$ `` - a DISPLAY-math delimiter around a code span, which renders wrong.
Caught by reading the result rather than the diff stat, reverted, and recorded because "convert every
match" is exactly the shape of instruction that produces it.

**Ported, not derived.** `report/technical-report.tex` already typesets five of these correctly; the
QSDC secrecy capacity existed as **three** ASCII restatements across three files against one correct
display equation, so that is where the conversion started.

### JupyterLite / marimo / pyvista-wasm: declined, with a reopening condition

Decisive, and not a matter of taste: **the crypto stack does not exist in Pyodide.** Re-verified
2026-08-22 against the authoritative `pyodide-lock.json` for **314.0.5 (2026-08-15)**: **356**
packages, of which `tenseal`, `openfhe`, `seal`, `concrete*`, `onnxruntime` and `onnx` are **all
absent**, and the only crypto present is `cryptography`, `pycryptodome` and `pynacl`. Checked at PyPI,
none of TenSEAL 0.3.17, openfhe 1.5.1, concrete-ml 1.9.0 or onnxruntime 1.29.0 has **ever** published
an emscripten wheel, and no issue requesting one exists in `pyodide/pyodide` or
`pyodide/pyodide-recipes`. A notebook that runs this project's actual pipeline is **not
implementable**, not merely inadvisable - it could only narrate the pipeline or re-implement it, and
two implementations of one thing is this project's named defect class.

Two further blockers. Both tools fetch their runtime from the **jsDelivr CDN** by default, which
breaks the no-third-party-origin claim the demo rests on - **but not equally, and lumping them
together was imprecise.** JupyterLite supports self-hosting through `pyodideUrl` in
`jupyter-lite.json`, the `JUPYTERLITE_PYODIDE_URL` environment variable, and
`jupyter lite build --pyodide <local tarball>`. marimo **hardcodes** `cdn.jsdelivr.net` in
`frontend/src/core/wasm/worker/bootstrap.ts` as a bare template literal with no config key, no
environment variable and no flag; its own bug #575 has been open since 2024-01-15 and was last touched
2026-05-19. So the CDN objection is *fixable for JupyterLite and unfixable for marimo*. And
accessibility is unassessed upstream: no JupyterLab/JupyterLite issue cites SC 1.4.1, 1.4.11, 1.4.13
or 4.1.3, and marimo has no accessibility documentation at all - absence of reports, not conformance.

**CORRECTION, Round 55: `pyvista-wasm` exists, and the reasoning that said otherwise is withdrawn.**
This section claimed the PyPI packages of that name are "third-party API lookalikes in a personal
namespace, not PyVista artifacts". Checked at source: `pyvista-wasm` **0.12.0 was published
2026-08-21**, Apache-2.0, the fourteenth release since 2026-03-25, by **Tetsuo Koyama
(`tkoyama010`)** - the **#5 all-time contributor to `pyvista/pyvista` with 683 contributions**, a core
maintainer rather than an impostor. It wraps `@kitware/vtk-wasm`, which is Kitware's own Apache-2.0
WebAssembly port of VTK, at v3.0.0 (2026-08-20). "Personal namespace" was read as evidence of
illegitimacy when it was only evidence of a namespace.

It is still declined, on grounds that are checkable rather than dismissive: pre-alpha with 6 stars and
not under the `pyvista` org; it fetches from **two** third-party origins by default (`unpkg` for the
UMD bundle, `jsDelivr` for a **22 MB** binary payload) and pins `vtk-wasm@1.7.4`, two majors behind
Kitware's current release; and, decisively, **this project has no three-dimensional data**. All five
`web/public/data/*/real_2d.json` are `[x, y, class]` where the third element is a binary label, not a
coordinate. The agreement scatter and the calibration plot are 2-D *by necessity* - their meaning is
distance from `y = x`, and a third axis destroys the read rather than enriching it.

*The notebook walkthrough reopens only when all four hold:* a Pyodide wheel exists for at least one of
CKKS/TFHE or an MPC runtime, so the notebook can execute the real pipeline; the runtime is self-hosted
from this project's own origin with a CI check that the kernel's package set matches the app's
lockfile; cold start is measured in-house, because **no citable benchmark exists** - the universally
repeated "6.4 MB, 4-5 seconds" is byte-identical boilerplate across the 2022 and current Pyodide
roadmaps and is not a measurement of any shipped version; and the four WCAG criteria above are tested
in-house. **For marimo specifically, a fifth:** it ships a supported way to move the runtime off
`cdn.jsdelivr.net`, which as of 2026-08-22 it does not.

*The 3-D path reopens only when this project acquires data that is genuinely three-dimensional.* If it
ever does, the dependency to reach for is **`@kitware/vtk-wasm` direct from npm and vendored**, not
`pyvista-wasm` - whose only contribution is a Python code-generation layer this project has no
Python-in-browser tier to use, and whose defaults reintroduce the two third-party origins.

**What comparable projects actually do, which is the evidence this decision was missing.** Hugging
Face's `webml-community` org publishes 74 Spaces of which **72 are `sdk=static`** and zero are Gradio;
`Xenova` publishes 55 of which 52 are static; `transformers.js-examples` has 60 example directories
and **zero notebooks**. **No FHE or MPC project anywhere ships JupyterLite or marimo** - Zama's six
public Spaces are all `sdk=gradio`, i.e. server-side Python, and all were last modified in 2024. Where
JupyterLite genuinely is used - SymPy Live, the NumPy/SciPy/scikit-learn docs - the payload is always
pure Python and never a native crypto extension. The convergence is on bespoke static JS/wasm apps,
which is what this project already is.

**And an independent re-confirmation of `CS11`.** Zama's own documentation states that TFHE-rs's WASM
API "includes functionality for key generation, encryption, and decryption. **However, it does not
support FHE computations**" - reached from the opposite direction, by surveying live demos rather than
by reading the licence, and landing on the same conclusion.

### ETSI horizon, recorded because it has a date for the first time

**GS QKD 015 ed3 is V2.1.2 Stable draft (2026-05-27), TB approval 2026-12-02**, and GS QKD 014 ed2
(V1.3.1, Stable draft) carries the same approval date. **The "V3.1.1 draft" figure is refuted** -
zero hits for "3.1.1" on the work-item page and no v3 tag of any kind in the forge, whose in-progress
branch is named v2.1.2. Recorded so a later round does not re-import it from a search summary. No
current claim changes: `docs/pqc-roadmap.md`, `references/REFERENCES.md`, `src/qsh/qkd/kme.py` and
`web/src/lib/clientSdnController.ts` all correctly state V1.1.1 as the only published GS QKD 014
edition.

## Round 46: the fix changed one language again, and the gate was built so it could not see it

**BUSINESS DOC RECONCILED Round 46.**

### The spine

`_decoy_secure_fraction` is reached only when `qkd_params` is supplied, and in the whole repository
only `src/qsh/qkd/optimize.py` ever supplied it. Every production Python surface called `run_bb84`
without it and went on computing the figure Round 45 condemned, removed from the browser, and wrote
three documents about:

| path | bit/pulse | secure bits |
|---|---|---|
| browser, after Round 45 | 0.002355 | 470 |
| **Python, all production call sites** | **0.008936** | **1,787** |

And `orchestration/crypto_agility.py` registered that path under the name **`"bb84-decoy"`**. The
name was the claim and the claim was false.

**Round 45's own gate was built so it could not see it.** It imported `_decoy_secure_fraction` and
called it directly - a private helper no production caller invokes. Pointed at what production runs,
resolved through the registry exactly as `QKDStage` resolves it, it is **red by 3.795x** on the
unmodified tree. Round 44's title was "the fix changed one language"; Round 45 did it again and
certified itself against the half it had already fixed.

Fixed at the registry, so `stages.py` inherits it, plus the three surfaces that bypass the registry.
A scope gate pins the class: no production call site may reach the legacy branch by omission.

### The block size turned out to be a security parameter

**THE THRESHOLD IN THIS SUBSECTION WAS WRONG AND THE CITATION UNDER IT WAS NOT IN THE PAPER.
CORRECTED IN ROUND 47, IN PLACE, because the numbers below were carried into five other files and a
silent edit would leave the propagation unexplained.** What Round 46 wrote: *"the positive-key
threshold for these device parameters is ~5e3 to 1.1e4 sifted bits (Rusca, Boaron, Grunenfelder,
Martin & Zbinden, Appl. Phys. Lett. 112, 171104, 2018)"*. Two failures, and the second is the worse
one.

**The attribution.** That paper states no positivity threshold anywhere. It compares the 1-decoy
against the 2-decoy protocol for privacy-amplification block sizes $`n_Z`$ from $`10^5`$ to $`10^{11}`$, and its own
sentence is that the 1-decoy approach wins *"unless a really big (> 10^11) or really small (< 10^5)
PA block size is applied"*. A number was attributed to a paper that does not contain it, and the
citation is what stopped anyone checking.

**The number.** `src/qsh/qkd/finite_key.py` now implements the paper's equations, and on this
project's own 50 km channel they put the threshold at **~1e5 sifted bits, about 4e7 pulses** - an
order of magnitude above the band that was published here. At the shipped 2,000,000 pulses the
single-photon lower bound `s^l_{Z,1}` is **NEGATIVE**, so no composable key exists at that block size
under any setting of the intensities, their probabilities or the basis bias; that was searched over
the paper's own variable set, not assumed.

**The refutation was already on the page.** Lucamarini et al., *Opt. Express* **21**, 24550 (2013),
Fig. 2: *"1.4x10^5 counts ... is the smallest sample size providing a positive rate"* at 50 km. Round
46 printed that one line below a band 13 to 28 times smaller and called it a corroborating primary
datum. It never corroborated that band. It agrees with the corrected figure, and it did so in the
next sentence for three rounds.

**How accurate our own number is, quoted wherever the threshold is quoted.** Validated against Rusca
et al.'s published Table I (SKR at $`n_Z = 10^7`$ for 26 / 46 / 56 / 64 dB), this implementation is
CONSERVATIVE by roughly 2-3x at moderate loss and by more at high loss, even after optimising the
intensities and the basis bias as the paper describes. Its threshold is therefore an UPPER bound on
what is required and is stated as an order of magnitude, never to two figures. What survives that
uncertainty: Round 46's 2,000,000-pulse block carried no composable key, since a factor of 2-3 cannot
rescue a negative count; the Round 46 band was far too small; and the citation was wrong independent
of any numerics.

**What Round 46 did, which stands on narrower ground than it claimed.** `qkd.pulses` 200,000 ->
**2,000,000** (1,244 -> 12,442 sifted bits), with `demo_alice_bob_pulses` and `sat_pulses` raised in
step - the satellite downlink had fallen to **223** secure bits, below the 256 needed to derive a
key, so satellite had silently stopped mixing QKD at all. That last consequence was real and is
unaffected. The finite-key argument for the raise was not: 2,000,000 does not clear the corrected
threshold either, so the raise landed in the right direction for a reason that did not survive
checking. Round 47 sized the fibre block against the COMPUTED threshold instead, and against a margin
rather than against the threshold itself, because a 2-3x conservative bound should not be taken at
its edge - `src/qsh/config/defaults.py` records the current value, the intensity re-optimisation that
lowers the pulse count needed, and the margin in composable bits. What the page publishes at 50 km is
still the ASYMPTOTIC decoy bound, labelled as one on every surface; the composable length is a second
quantity shown beside it, not a relabelling of the first.

**The honest residual, and it is bigger than Round 46 recorded:** at 10e6 the 800 km downlink yields
**~2,440 sifted bits** - a 5-seed mean of 2,442, spread 2,397 to 2,470, measured at the intensities
this round shipped (mu = 0.3416, nu = 0.1417). **This entry said ~3,660 for one round**, which is the
figure at the retired $`\mu = 0.5`$ / $`\nu = 0.1`$ (5-seed mean 3,582); this subsection was rewritten in the
same round that moved the intensities and the line was carried across without re-measurement. Round
46 measured its own figure against the refuted band and concluded ~30e6 pulses and ~1.5 GB of Python
array would clear it; both figures fall with the band, and the residual is larger still at the
corrected intensities. Against the corrected fibre threshold the shortfall is well over an order of
magnitude, and the free-space threshold itself is
not quoted because it has not been evaluated - only its direction is safe, and it is the unhelpful
one, since 24.48 dB of loss means lower gain per pulse and a higher QBER at equal statistics. The
reason is physical rather than a setting. The free-space rate is asymptotic-only in a stronger sense
than the fibre one.

### Round 45's own caveat was wrong in both directions

It read "roughly 1,900 sifted bits" - a figure matching no computed quantity anywhere; it is 1,244.
And it called the result "a modelled upper bound, ~2.4x a real deployed link". Refuted: Li et al.,
*Nat. Photon.* **17**, 416-421 (2023), arXiv:2307.02364, measured **22.2 +/- 0.8 Mb/s at 50 km** on a
2.5 GHz clock = **8.88e-3 bit/pulse** with full composable finite-key security - **3.8x HIGHER per
pulse than this project's figure**. It is not an upper bound on real hardware; it is the N->infinity
limit of a lower bound, and real hardware beats it.

### A real QCNN, in all three places the circuit lived

Everything named `qcnn_*` was a hardware-efficient ansatz - RY on every wire, a CNOT ring - scoring
**zero of five** Cong, Choi & Lukin structural elements. The model cards already described it
correctly; only the name was false, which is the harder version.

Now conv (one SHARED filter at every position) -> pool (rotation on the survivor conditioned on its
partner) -> register reduction -> fully-connected block. Parameters **51 / 76 / 101** at 4 / 8 / 16
qubits: increments of 25 and 25, a CONSTANT per doubling where the old ansatz doubled.

**The circuit existed in THREE places** - the trainer, the browser's own state-vector copy, and a
third Python inference path - and all three are ported in the same change. "Two implementations of
one thing" is this project's named recurring defect; here it was three, and the third surfaced only
because a test raised `IndexError` on the new weight shape.

Two measurements changed the design. A 2-angle shared filter underfits where the readout needs
several wires, so the filter is the expressive 8-angle two-qubit block. And odd registers were broken
outright: `layer_plan` computed `active //= 2` (9 -> 4) while the circuit did `active[0::2]` (9 -> 5),
leaving one wire that carried an encoded feature to the readout having never been convolved or
pooled. That collapsed finance at 9 qubits to accuracy **0.0517** on a 5% positive set and medical at
9 qubits to minority-F1 **0.0000**.

Retrained: medical 0.9186, finance 0.9767, telecom macro F1 0.6146, telecom-q7 0.5104. **8 and 9
qubits now agree exactly**, and that is a result rather than a coincidence - a shared filter cannot
allocate capacity to a wire carrying no feature, where the old ansatz gave it independent trainable
angles. Telecom is down from 0.8098 and that is reported: an 8-class readout needs 3 wires, so only
one pool stage is possible at 8 qubits and the architecture's depth cannot be used.

**EVERY NUMBER IN THAT PARAGRAPH WAS RE-MEASURED IN ROUND 47 AND ALL FOUR MOVED**, because the
multiclass readout was binning the class index over a discarded wire and the fully-connected block
was one shared scalar rather than a per-survivor unitary. **THEN ALL FOUR MOVED AGAIN IN THE SAME
ROUND**, when two classes of parameter turned out to be analytically dead. The intermediate figures -
medical accuracy 0.860465 (minority F1 0.8125), finance minority F1 0.84058 (accuracy 0.981667),
telecom macro F1 0.458324, telecom-q7 macro F1 0.500452 - are **SUPERSEDED** and belong to the
circuit that carried five dead angles; see "Round 47, continued" below. Current, from
`models/*/weights.json`: medical accuracy **0.883721** (minority F1 0.84375), finance minority F1
**0.735294** (accuracy 0.97), telecom macro F1 **0.510963**, telecom-q7 macro F1 **0.541894**.
Classical logistic regression on the identical splits is ahead on all three use cases, before and
after both retrains: 0.953488, 0.865672 and 0.976447 respectively.

The claim that 8 and 9 qubits agree exactly SURVIVES both retrains and is now measured on three
successive circuits - the two widths are **identical on every recorded metric** on both binary
corpora. That phrasing is deliberate and replaces "byte-identical", which was wrong of the
ARTIFACTS: the recorded metrics match exactly, but `theta` does not (max |difference| 1.1e-05 on
finance, 4.2e-06 on medical), `bias` differs on both and `scale` differs on finance. See the
continuation section below, which states the same pair of facts. And the
claim that telecom fell survives too, further than Round 46 knew: part of 0.614583 was the discarded
wire's bit and was never the QCNN's to claim. The telecom-q7 figure has also crossed its 8-wire
sibling, which reverses the width finding the Round 45 entry above restates. See the Round 47
sections.

### Round 45's six gate defects, each the class it was fixing

`secretKeyRateNonDecoy` was an inert FIELD shipped in the same commit as the gate against inert
CONTROLS. That gate could be satisfied by a COMMENT, because it did not call `_code_only()` which the
same file defines - and `clientCompute.ts` carried the satisfying string inside the JSDoc Round 45
wrote to explain wiring it. `quantumSim.ts` documented the RETRACTED direction of an inequality its
own test asserts the other way. The log gate was self-perpetuating: `## Round 44, added in Round 45`
satisfied `now = 45`, so writing the PREVIOUS round's pointer discharged the CURRENT round's
obligation, forever. `_DEADLINE` was fixed at the opening `**` and left at the closing, so a clause
bolded as a whole still returned zero matches. And the `skr_per_pulse` rounding pair split.

**The deadline gate fired on Round 45's own changelog**, which quoted the pattern it had just fixed.
Making the marker case-sensitive is what lets narrative and obligation be told apart.

### Records corrected, each re-verified at primary source

Two of the research pass's own proposed corrections were refuted in its adversarial stage, so none
was applied on trust. IACR ePrint 2026/1282 is **BBM92**, 24-dimensional, binary, on simulated FSO
telemetry - not "this exact problem". PGS scoring files carry **no** `#license` header: PGS000004
downloaded from the EBI FTP has 14 `#` lines and none is a licence, which
`scripts/train/genomics_prs.py` has said correctly all along while two prose sites contradicted it.
`decoy_reference.py` had **four wrong equation numbers** against arXiv:quant-ph/0503005v2 - the key
rate is Eq. (1) not (33), `Y_i` is (7) not (6), `e_i` is (9) not (7), and $`e_0 = 1/2`$ is prose - in
the module whose docstring says it "exists to be CHECKED".

**Calibration: the denominator was wrong, and the correction is not a swap.** `calibration.py`
compared positive-EVENT counts against thresholds both papers state in calibration CASES;
`n_calibration` is 710,969, which is 355x the UAI figure. Read at source, ICML 2005 says the opposite
of the conclusion drawn from it: *"When there are 1000 or more points in the calibration set,
Isotonic Regression always yields performance as good as, or better than, Platt Scaling."* Platt is
retained anyway, because those papers benchmark balanced problems and at a 1-in-981 base rate the
binding quantity is plausibly the 2,321 positive events, which they do not address. Switching on a
sentence about case counts would repeat the same error in the other direction. Conditioned
`by **Round 48**`. **DEADLINE EXECUTED Round 49** - measured, and the paragraph above is wrong in
two places: the calibration set carries **801** events, not 2,321 (2,321 is the TEST split), and the
shipped `fit_platt` WAS fed a probability rather than a log-odds, which MADE its map return one
constant for nine of ten deciles - **FIXED LATER IN THE SAME ROUND**: `fit_platt` gained
`on_logit=True`, which carries the transform, the 1e-12 clip and the published formula string with
the two floats; the teacher was retrained and the shipped map is now the logit arm
(`a = -0.5853491088584304`, `b = 1.6417462922842536`, `on_logit: true`). Isotonic and a correctly-fed Platt score within noise. Result and
recommendation: Round 49's section below, workings in
`references/finance-calibration-isotonic-vs-platt.md`.

### 5G-NIDD: the rejection is withdrawn and the adoption is NOT a drop-in

`DATA_LICENSES.md` rejected 5G-NIDD for carrying no licence string. Overturned at the authors'
repository of record - Fairdata/Metax, DOI 10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361 - which
returns **CC BY 4.0**, `access_type: Open`, `restriction_grounds: []`, verified today. IEEE DataPort's
"Subscription Required" is a mirror, not the repository of record.

**But it does not fill the gap it was expected to.** The telecom model classifies QKD-layer attacks
(`intercept_resend`, `photon_number_splitting`, `trojan_horse`, ...) from QKD physical observables
(`qber_mean`, `sift_ratio`, `multiphoton_rate`, ...). 5G-NIDD classifies network-layer attacks
(UDPFlood, SYNScan, ...) from 52 columns of flow statistics. **Measured overlap: zero in the label
space and zero in the feature space.** The reason telecom has no real dataset is that QKD attack
classification needs a physical QKD testbed under attack, and 5G-NIDD is a different problem rather
than a substitute. It is therefore adopted as a SEPARATE network-layer model, conditioned
`by **Round 48**`.

**DEADLINE EXECUTED Round 49.** Adopted as a separate network-layer model and published:
`telecom_ids.nidd`, `Daemons-Q/qsh-telecom-5g-nidd-gbm`, pinned at revision `b00628c5`. Corpus
fetched from the Fairdata repository of record (the token dance is a two-step POST to
`etsin.fairdata.fi/api/download/authorize`, NOT the 404 `download.fairdata.fi/authorize`, and the JWT
expires - so `NIDD_FETCH` is one pipeline). 1,215,890 rows, 9 classes, 39 usable features after
dropping 3 labels, 3 row identifiers, 5 bit-identical duration columns and 2 single-valued columns.
CC BY 4.0 attribution reaches a visitor on three surfaces, and the `attribution` field was backfilled
for ALL FOUR CC BY corpora in the same change rather than added for this one.

**The result worth publishing is not the model, it is the contrast.** On the 39 usable features:
macro F1 **0.9264**, accuracy **0.7609**, all 9 classes predicted. Restore the three integer row
identifiers - `Unnamed: 0`, `Seq`, `Offset` - and the identical configuration scores **0.9995 /
0.9997**, reproducing the 99.6-99.7% accuracies published on this corpus and costing **23.9 accuracy
points** when removed. Same shape as the Engelen et al. result on CIC-IDS2017 that this repository
already cites as its reason for reporting per-class F1 first.

Two things stated because they weaken it. **It is not a refutation of the published work**: those
papers' feature sets are unstated, so this establishes that the identifiers carry the signal here,
not that anyone used them. And the leak check fires on **21 of 39** features, not the 2 the design
predicted - top `Rate` 0.9941, `AckDat` 0.9928, `sTtl` 0.9920 - because each class was generated by a
different tool against a different service. The identifier contrast is therefore narrower than a
clean "three counters versus honest features" reading suggests, and `single_feature_leak_check` in
`models/metrics.json` records the full list rather than the design's prediction.

**The metric inversion is the opposite of every other model here** and the card says so: macro F1
0.9264 is the OPTIMISTIC figure and accuracy 0.7609 the pessimistic one, because Benign and UDPFlood
(39.3% and 37.6% of rows) mutually confuse - 23.7% of the test set - while the seven minority classes
score 0.98 to 1.00. `baseline` is deliberately `None`: the literature figures are accuracies and this
model is judged on macro F1, so putting them there would render an accuracy beside a macro F1.

## Round 47: the fix changed one language a fifth time, and the fifth one is this round's

**BUSINESS DOC RECONCILED Round 47.**

Round 44's title was "the fix changed one language". Round 45 repeated it, Round 46 repeated it
again, and this round opened by finding the FOURTH instance inside Round 46's own commit - then
committed the FIFTH itself, in the QCNN, where a gate written earlier the same day caught it. A
SIXTH, in the third implementation of the same circuit, was found after that commit and is recorded
below with its status rather than folded into the narrative as though it had been caught in time.

### The fourth instance, and the gate that should have caught it was a file list

`demo_alice_bob_pulses` was raised 40,000 -> 400,000 in Round 46 **in the browser only**. Two Python
call sites of the same protocol kept the literal:

| surface | block size after Round 46 | consequence |
|---|---|---|
| `web/src/lib/config.defaults.json` | 400,000 | page reports a QKD-mixed session |
| `api/inference_service.py` | **40,000** | same panel, different block |
| `src/qsh/benchmark/alice_bob_e2e.py` | **40,000** | benchmark scored 73.0 with **`qkd_mixed: False`** |

At 40,000 the decoy bound yields **94** secure bits, under the **256** `derive_shared_key` needs, so
two demonstrations of one protocol disagreed about whether the quantum half contributed at all. Both
Python sites now read `get_config().web.demo_alice_bob_pulses`; neither spells a number.

**The gate that should have caught it iterated a hardcoded tuple of three paths**, under a docstring
reading *"stated as a property rather than a file list, because a gate whose scope names files is
this project's most-repeated defect"*. It was a file list. The scope is now derived from
`git ls-files` and parsed with `ast`, and no call site may spell a block size as a literal at a
`run_bb84` call at all - which is the stronger property, because a literal that agrees today is the
thing that stops agreeing later.

`tests/test_every_block_size_clears_the_band_the_config_names.py` adds the value side of the same
question: every integer config field whose name contains `pulses` must clear the computed threshold,
by introspection rather than by naming the two keys known to matter. A block that cannot clear it
must be DECLARED with a reason - `demo_sat_pulses` (24.48 dB at 800 km, physics rather than a
setting) and `optimize_n_pulses` (a per-evaluation sample, not a published block) are the two, and
"it would be slow" is explicitly not an admissible reason.

### The finite-key threshold, corrected in place in the Round 46 section above

The correction itself is written where the wrong number was published, at *"The block size turned out
to be a security parameter"*, because that band was carried into five other files and a silent edit
would leave the propagation unexplained. What this round BUILT, recorded here:

- **`src/qsh/qkd/finite_key.py`** implements Rusca et al.'s equations (1), (3), (4), (A16), (A19),
  (A21), (A22), (A23) of arXiv:1801.03443v2, each cited by number, so the repository computes the
  threshold it used to cite a band for.
- **It is checked against the paper's own Table I** - secret key rate at $`n_Z = 10^7`$ for 26 / 46 / 56 /
  64 dB - in `tests/test_finite_key_reproduces_the_published_table.py`. It comes out CONSERVATIVE by
  roughly 2-3x at moderate loss and more at high loss, so the threshold it computes is an upper bound
  on what is required. It is therefore quoted everywhere as an order of magnitude, **~1e5 sifted
  bits, about 4e7 pulses**, and never as the 1.436e5 / 3.84e7 the code returns.
- **The intensities moved to the finite-key optimum**, by differential evolution over exactly the
  variables the paper says it optimises:

| | mu | nu | asymptotic bit/pulse | positive-key threshold |
|---|---|---|---|---|
| shipped until Round 47 (asymptotic optimum) | 0.5 | 0.1 | 0.002368 | 3.84e7 pulses, ~2.0 GB of array |
| shipped now (intensities only, from the finite-key optimum) | **0.3416** | **0.1417** | **0.001977** | **1.41e7 pulses, ~0.74 GB** |

  It COSTS 16.5% of the headline asymptotic rate and buys a **2.7x** smaller block. The two optima
  are in different places, which is the whole point: mu ~ 0.5 maximises a rate that assumes infinite
  statistics.

  **AND ONLY TWO OF THE FOUR OPTIMISED VARIABLES SHIPPED.** `optimise_parameters` searches mu, nu,
  `p_signal` and `p_z`, and its full solution also sets `p_signal` = 0.3764 and `p_z` = 0.5042, at
  which the threshold is 1.31e7 pulses. Neither has a shipped counterpart - there is no `p_signal`
  config field and `QKDConfig.basis_prob` is 0.5 - so the threshold that applies to this build is the
  **14,055,773** pulses the gate's own `_threshold()` computes, and both rows of the table above are
  at `p_signal` = `p_z` = 0.5 and therefore comparable. **This table said 1.31e7 / 2.9x for one
  round**, which credited the config with two variables it does not carry;
  `src/qsh/config/defaults.py` records the same correction.

- **`qkd.pulses` and `demo_alice_bob_pulses` are both 20,000,000**, past the 1.41e7 threshold with a
  margin rather than at its edge: **1,626** bits of composable key, while 1.4e7 sits just BELOW the
  threshold and yields **no composable key at all** (-12.0 bits). That is a factor of 1.42, which is
  thin given this implementation's own 2-3x uncertainty. (The "about 1,800 bits, against 217 at
  1.4e7" this bullet carried for one round are the optimiser's full-solution figures, not this
  build's; at the shipped `p` the 1.4e7 comparison point has no key.)
- What the page publishes at 50 km is still the **asymptotic** decoy bound, labelled as one on every
  surface. The composable length is a second quantity shown beside it, not a relabelling of the
  first.

### The QCNN multiclass readout was measuring a wire the circuit had discarded

The class index was binned by `i & (k-1)`, which reads wires 0..m-1. After pooling the survivors are
`[0, 2, 4, 6]`, so **wire 1** - the pooling control of the pair (0, 1), conditioned into wire 0 and
never touched again - supplied one bit of every class index while frozen at its PRE-pooling value.
`layer_plan`'s own docstring justifies stopping the register reduction early on the premise that the
readout wires ARE the survivors; the readout did not implement that premise. It now bins over the
survivors and records them as `readout.wires` = `[0, 2, 4]` in the **two MULTICLASS artifacts** -
`qcnn_telecom` (8 wires) and `qcnn_telecom_q7` (7 wires), 2 of the 6 QCNN artifacts. The four binary
artifacts are untouched by this fix and deliberately carry NO `wires` key: the trainer emits
`readout.kind` = `expval_z0` for them, a single `<Z0>` expectation that reads one surviving wire and
bins nothing. An earlier revision said "each artifact", and `docs/verification-log.md` said "every
`models/*/weights.json`" - a reviewer executing that on any of the four binary artifacts finds
nothing.

**Telecom's macro F1 falls 0.5692 -> 0.458324**, and the previous number was inflated by the defect.
**The published headline appeared to fall 0.614583 -> 0.458324, and that pairing crosses metrics**:
0.614583 was the previous run's ACCURACY - `git show 4ee258e:models/qcnn_telecom/weights.json`
records `test_macro_f1` = 0.5692 beside it - so the defect-attributable macro-F1 loss is **0.110876**,
not the 0.156259 this entry asserted for one round. The remaining 0.045 is the card mislabel that
this round's own new gate, `tests/test_a_model_card_states_what_its_weights_state.py`, exists to
catch; it is not the readout. The accuracy fall on its own is 0.614583 -> 0.502083 = 0.1125, so
0.156259 is only ever the cross-metric subtraction. Wire 1's bit encoded the register BEFORE pooling
mixed it away, which is information the architecture is supposed to have discarded. **0.458324 is
itself SUPERSEDED later in this same round by 0.510963**, when the dead parameters below were made
live; the fall it records is unaffected, and
every figure in the rest of this subsection is the pre-dead-parameter one. Four configurations were
run to see whether the loss was capacity rather than the fix, and **none of them recovered it** (macro
F1, corrected readout throughout):

| configuration | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 8 wires, 6 layers | 0.4547 | 0.4567 | 0.4826 |
| 8 wires, 10 layers | 0.4448 | 0.4614 | 0.4732 |
| 12 wires, 6 layers | 0.4362 | - | - |
| 8 wires, 6 layers, per-survivor FC (SHIPPED) | **0.4583** | 0.4185 | 0.4553 |

Ten runs across four configurations did not find the missing **0.110876** - macro F1 against macro
F1, measured from the previous circuit's own 0.5692 and not from the 0.614583 the row above records
as an accuracy mislabelled as an F1. That is recorded rather than retried until something flattered,
and it is the evidence that the drop was the readout defect and not a capacity limit. The shipped
seed is the HIGHEST of its three, so the headline is the top of a spread and not a median; the seed
is deliberately not re-picked, because choosing one on test performance is test-set selection.

**Every published QCNN metric, as the artifacts stated them AT THIS POINT IN THE ROUND. The QCNN
column below is SUPERSEDED** by the retrain recorded in "Round 47, continued"; the classical and
floor columns are unchanged, because neither depends on the circuit. The table is kept because the
supersession is only checkable against what it replaced:

| artifact | headline | QCNN (SUPERSEDED) | classical LR | majority |
|---|---|---|---|---|
| `qcnn_telecom` | macro F1 | 0.458324 (acc 0.502083) | **0.976447** (acc 0.977083) | 0.152083 |
| `qcnn_telecom_q7` | macro F1 | 0.500452 (acc 0.541667) | **0.976447** | 0.152083 |
| `qcnn_finance` | minority F1 | 0.84058 (acc 0.981667) | **0.865672** (acc 0.985) | 0.948333 |
| `qcnn_finance_q9` | minority F1 | identical to `qcnn_finance` on every recorded metric | | |
| `qcnn_medical` | accuracy | 0.860465 (minority F1 0.8125) | **0.953488** (minority F1 0.939394) | 0.593023 |
| `qcnn_medical_q9` | accuracy | identical to `qcnn_medical` on every recorded metric | | |

**The classical baseline wins on all three use cases, as it did before the retrain.** Nothing here is
a quantum-advantage result and the direction of the comparison is unchanged; only its size moved.
That is still true of the current figures, which are tabulated in the continuation section below and
read from the same `models/*/weights.json` that
`tests/test_a_model_card_states_what_its_weights_state.py` checks each card against.

**The width conclusion REVERSED.** `references/qcnn-width-ablation.md` concluded that 8 wires beat 7
on telecom by +0.0786 macro F1, and Round 45 restated that as a five-seed median. With the corrected
readout, 7 wires scores 0.500452 and 8 wires 0.458324 - the narrower circuit leads by 0.042128.
**Those two figures are superseded by the continuation section; the reversal is not.** The mechanism
is consistent with the defect: at 7 wires pooling leaves exactly three survivors `[0, 2, 4]` and all
three are read, so there was never a spare pre-pooling bit to leak. Both figures are single-seed, on
both sides of the reversal, so this retires the old direction rather than establishing the new one.
The ablation carries a supersession banner and nothing in it is deleted.

### The fifth instance, committed by this round and caught by its own morning's gate

The fully-connected block became a **per-survivor** RY+RZ in the trainer (reordered to RZ+RY later in
the round - see "Round 47, continued"), per Cong, Choi & Lukin's
*"a unitary F on the remaining qubits"* - one shared scalar makes F a single global rotation, which
cannot distinguish the wires the class index is read from. The browser was left on the shared scalar.
Measured before it was caught: **maximum readout probability difference 0.45**, and the argmax class
disagreed on **39 of 64 telecom rows**. The deployed page would have served confident, wrong labels
from weights it could not interpret, while every structural test stayed green and both cards printed
the trained metrics.

`tests/test_the_qcnn_agrees_across_languages.py` is the numeric check that was missing: it executes
the browser's own `qcnnInfer` through esbuild + node on the artifact the browser actually loads and
compares class probabilities against the trainer's `circuit()` on the same inputs. It is green on
arrival, so it is **seeded** - a companion test perturbs one implementation's parameter layout and
asserts the comparison rejects it, because a gate never observed to fail is a gate nobody has
checked. The preprocessing contract is pinned in the same file, having been got wrong twice in ten
minutes while the comparison was being built.

Round 46 implemented this circuit in three places and asserted only that they had the same STRUCTURE.
They happened to agree, so the omission cost nothing and was invisible.

**AND A SIXTH INSTANCE, IN THE THIRD IMPLEMENTATION, FOUND AFTER THE COMMIT THIS ENTRY DESCRIBES.**
Stated with its status: it is in the working tree and NOT in `3c63ea8`, which is why it is written
here as a finding rather than as a fix. `src/qsh/qhe/qcnn_trained.py` - the numpy inference path, the
third of the three places the circuit lives - was still reading
`fc = float(theta[n_conv + stages])` and applying that one angle to every survivor. The trainer and
`quantumSim.ts` were both moved to the per-survivor block and this one was not, in the round named
for that defect, in the language that had already been fixed twice in it.

**It was not producing a wrong number, and that is the part worth recording.** Both artifacts this
module serves are BINARY, so they pool all the way to a single survivor; `theta[n_conv + stages]` is
then exactly that survivor's RY angle. **The second coincidence is HISTORICAL and must be read as
such:** under the RY-then-RZ ordering in force while the old form was live, the RZ it dropped was the
LAST gate before the readout and so commuted with the `<Z0>`. That argument was invalidated LATER IN
THIS SAME ROUND, when the fc block was reordered to RZ-then-RY - see "Two classes of parameter were
analytically dead" below - so there is no longer a trailing diagonal to commute. Measured against the
trainer of the revision that carried the old form, on that revision's weights and the 24 rows the
gate uses, the gap was **6.9e-07** on finance and **1.4e-06** on medical - float32 noise. Both
figures belong to that revision, not to the current tree.

Correct by two coincidences rather than by construction, and one artifact away from breaking: at
`readout_qubits = 2` on 8 wires the register pools to TWO survivors and the old form diverges from
the trainer by **0.0335** in class probability (0.0946 summed per row). That is the figure the repo's
own seeded gate produces -
`tests/test_the_qcnn_agrees_across_languages.py::test_the_old_fc_block_diverges_once_there_are_two_survivors`,
torch seed 1, 16 rows - re-measured on the current tree. **This entry said 0.2701 for one round**,
which is about 8x the reproducible value and matches no draw the repo pins; the divergence is
seed-dependent, so only the pinned seed's number is quotable, and the gate's own assertion is
`> 1e-2`, which is why nothing caught the overstatement. A shape check now raises rather than
applying whatever angles the artifact happens to carry.

### The decoy bound was handed the wrong error term, in three languages at once

`_decoy_secure_fraction` was passed `intrinsic_qber`, which is `misalignment + 0.5 * p_dark / eta`,
where Ma et al. want `e_det` - the probability that a detected SIGNAL photon lands in the wrong
detector. The dark-count contribution is the separate `e0 * y0` term inside the same function, so the
background was charged twice and the bound came out LOW - by how much depends on the intensities, and
an earlier revision of this sentence gave "0.57% and 18.9%" without naming them. **18.9% is not a
"low by" figure at any intensity pair**: it is the same gap with the WRONG bound as denominator, i.e.
how much the fix RAISES the number, which is the convention `docs/verification-log.md` correctly uses
for it. Both conventions, at both pairs:

| intensities | fibre 50 km | 800 km downlink |
|---|---|---|
| mu 0.5 / nu 0.1, in force while the double charge shipped | 0.5673% low (0.5705% raise) | **15.9152% low** (18.9276% raise) |
| mu 0.3416 / nu 0.1417, **shipped now** | 0.4984% low (0.5009% raise) | **13.6638% low** (15.8262% raise) |

The fibre figure rounds to 0.57% under both conventions, which is exactly what let the (0.57, 18.9)
pair be copied out of a "RAISES" sentence into a "LOW by" one without the mismatch showing. eta is
17.5x smaller on the downlink and the dark-count term correspondingly larger. Safe direction, wrong
number, and worst exactly where the link is hardest. Fixed in `src/qsh/qkd/bb84.py`,
`web/src/lib/quantumSim.ts` and `web/scripts/check_qkd.mjs` together, which is
the discipline the four previous instances exist to teach. `intrinsic_qber` stays correct for the
OBSERVED QBER, which is the quantity it names.

**THAT FIX WAS HALF RIGHT AND HAD TO COME BACK IN THE SAME ROUND, AND THE HALF IT GOT WRONG PUBLISHED
A KEY UNDER A FULL ATTACK.** It replaced `intrinsic_qber` with the BARE MISALIGNMENT, which removed
the double charge and silently unhooked the eavesdropper - `intrinsic_qber` was the only path by
which `eve_intercept_fraction` reached the bound. It also named THREE implementations of the decoy
algebra when there are four. Both are recorded in "Round 47, continued" below, with the measurements.

### Supporting, each one an instance of a class this register already names

- **`QKDParams` reads its intensities from config** instead of spelling `mu_signal = 0.5` and
  `mu_decoy = 0.1` as dataclass literals. Every production surface constructs a bare `QKDParams()`,
  so moving the shipped intensities in `QKDConfig` would have reached the browser through
  `config.defaults.json` and left every Python caller on the old pair - the same defect as the block
  size, in the same round, found by the cross-language rate identity rather than by reading.
- **`config.yaml` shadowing is gated rather than commented.**
  `tests/test_config_yaml_does_not_shadow_a_stale_value.py`: a value repeated in the YAML overrides
  the dataclass silently, so editing the default moves nothing.
- **Shipped QCNN weights are copied by `scripts/ship_qcnn_weights.py`**, not by hand. Six artifacts
  reach `web/public/data/` and a hand copy that misses one is exactly the split this round is named
  for.
- **The 5G-NIDD licence rejection is withdrawn in all three files that carried it**, not only in the
  fourth that recorded the overturn. Round 46 corrected the finding and left the rejections standing.
- **Every model card is gated against its own `weights.json`**
  (`tests/test_a_model_card_states_what_its_weights_state.py`), because a card is the surface a
  reader trusts and it had drifted from the artifact twice.

### The figures this round moved, and where they were corrected

The retrain changes numbers that six rounds of documents had propagated. Corrected as part of this
round's records pass, in place and with the superseded values left visible:

| file | figure | treatment |
|---|---|---|
| `docs/deliverable-verification.md` (Round 45 entry) | 0.8098, +0.0786 | superseded note beside the bullet |
| `docs/deliverable-verification.md` (Round 46 entry) | 0.9186 / 0.9767 / 0.6146 / 0.5104, 0.8098 | superseded note beside the paragraph |
| `docs/verification-log.md` (Round 28 entry) | medical accuracy 0.732558 | correction note under the table, row left as measured |
| `references/REFERENCES.md` | 0.7326, 0.7143, 0.8098, 0.809846, 0.714286 | current values inserted; ablation figures labelled as the retired circuit's |
| `references/qcnn-entanglement-ablation.md` | 0.714286, 0.809846, and a deleted circuit in the present tense | supersession banner, its first |
| `references/qcnn-vs-published-wdbc.md` | 0.7326 as the shipped accuracy | supersession banner, its first |
| `references/qcnn-width-ablation.md` | +0.0786 | banner written with the code change |

**Two of those files had never received a banner at all**, while their width-ablation sibling got one
in Round 46 - the entanglement ablation and the WDBC comparison both still described the pre-Round-46
ansatz, in the present tense, as the circuit this repository ships.

**And the entanglement ablation is no longer exact on the current circuit**, which is recorded there
rather than left for a reader to discover: `--no-entangle` gates the conv filter's CNOT and the
survivor ring in the fully-connected block, but the pooling stage is a CRY - a controlled rotation,
which entangles - and it is NOT gated. The file's own claim that "the ablation is exact rather than
approximate" was true of the ring-only ansatz it was measured on and is false of the QCNN.

## Round 47, continued: parameters the optimiser carried and could not move, and a key published under a full attack

This section is a continuation of the one above rather than a new round. Everything in it was found
after the entry above was written, by continuing to check the things that entry had just fixed - which
is how the fourth, fifth and sixth instances of the one-language defect were found too.

### Two classes of parameter were analytically dead

Dead in the strict sense: carried by the optimiser, counted in the parameter count, and **unable to
move the readout at any input**. There were two of them and they had different mechanisms.

**1. The fully-connected block's per-survivor RZ.** It was applied AFTER the RY, and the only
operations after that block are a CNOT ring - a basis permutation - and `|amplitude|^2`, which
discards phase. A diagonal gate applied last cannot move a computational-basis readout, and on the
two binary artifacts it commutes with the `<Z0>` it is measured in.

RE-MEASURED, WITH THE RECIPE, BECAUSE THE FIRST SET OF FIGURES HERE HAD NEITHER. An earlier revision
of this paragraph read "moving every fc RZ by 1.234 rad changed the telecom readout by 5.96e-08 and
the finance readout by 8.94e-08 - float32 noise - against 2.6e-01 and 5.9e-01 for the RY angles
beside them", with no seed, no inputs and no command, on weights that a later retrain in this same
round replaced. Those figures are WITHDRAWN. The procedure that replaces them: reconstruct the
pre-fix order from `git show 8017538^:scripts/train/qcnn_variational.py`, take each artifact's
`theta` from `models/qcnn_*/weights.json`, feed `x = tanh(randn(8, n_qubits) * 0.7)` at
`torch.Generator().manual_seed(0)`, add 1.234 rad to every fc RZ angle - and separately to every fc
RY angle - and take `max |delta|` of `circuit()`:

| artifact | fc RZ, pre-fix order | fc RY, pre-fix order | fc RZ, shipped order |
|---|---|---|---|
| `qcnn_telecom` | **5.96e-08** | 1.77e-01 | 2.28e-01 |
| `qcnn_telecom_q7` | **1.19e-07** | 3.64e-01 | 1.59e-01 |
| `qcnn_finance` | **5.96e-08** | 9.89e-01 | 4.82e-02 |
| `qcnn_medical` | **8.94e-08** | 5.12e-01 | 7.16e-02 |

The three pre-fix RZ figures are float32 ULPs - $`2^{-24}`$, $`2^{-23}`$ and $`1.5 \times 2^{-24}`$ - so the angles were
inert rather than weak, which is the claim. `8.94e-08` belongs to MEDICAL under this recipe, not to
finance as the withdrawn sentence had it. These angles were added earlier in this same round, for
faithfulness to Cong, Choi & Lukin's "a unitary F on the remaining qubits". Half of what was added
was decoration.

**2. The 8th conv angle of each stage's final layer round.** An `rz` on the wire pooling is about to
discard. Its only later interaction is as the CONTROL of a CRY, which is diagonal in that wire's
basis, and then marginalisation at readout. That was 3 of 72 conv angles on the binary models and 1
of 48 on telecom, **carried since Round 46**.

**THE FIX WAS ORDERING, NOT DELETION.** RZ before RY in both places, so the phases RZ writes are
mixed back into amplitudes by the RY that follows. No parameter was removed and no count changed;
what changed is how many of them the readout can see. Measured on the same perturbation after the
change, per artifact: **4.82e-02 to 2.28e-01** - the right-hand column of the table above. A single
`1.4e-01` was quoted here for one rewrite; it matches no artifact under the stated recipe and is
withdrawn with the rest of that set.

Counted per artifact, by perturbing every angle under the pre-fix ordering:

| artifact | angles | dead before the fix | which |
|---|---|---|---|
| `qcnn_telecom` | 57 | **5** | 4 fc RZ, one per survivor, plus `theta[47]` |
| `qcnn_telecom_q7` | 55 | **4** | 3 fc RZ, one per survivor, plus `theta[47]` |
| `qcnn_finance`, `qcnn_finance_q9` | 77 | **4** | `theta[76]` plus `theta[23]`, `theta[47]`, `theta[71]` |
| `qcnn_medical`, `qcnn_medical_q9` | 77 | **4** | the same four indices |

**The gate is a property over ALL parameters, not a check for this mistake.**
`test_every_trainable_parameter_reaches_the_readout` in
`tests/test_the_qcnn_has_the_structure_the_name_claims.py` perturbs EVERY parameter at five
(width, readout, layers) configurations and fails if any one of them cannot move the output. The
specific error is not the interesting one: a circuit that ends in a diagonal gate, that pools a wire
away before its angle is used, or that indexes past the end of a block all produce the same
condition, and all of them are invisible to accuracy - the model simply trains the parameters that
work - and to a parameter count, which counts angles rather than asking whether they do anything.

### The measured effect of making them live is MIXED, and is not presented as a win

| artifact | headline metric | before | after | change |
|---|---|---|---|---|
| `qcnn_telecom` | macro F1 | 0.458324 | **0.510963** | +0.053 |
| `qcnn_telecom_q7` | macro F1 | 0.500452 | **0.541894** | +0.041 |
| `qcnn_medical` | minority-class F1 | 0.8125 | **0.84375** | +0.031 |
| `qcnn_finance` | minority-class F1 | 0.84058 | **0.735294** | **-0.105** |

**Three improved, one regressed, and the regression is larger than any of the three gains.** What is
established is that four or five angles per circuit that could not affect the output at any input can
now affect it - a defect fixed whichever way each metric went. What is NOT established is that the
change improves this architecture: reordering the gates also changes how `torch.randn` is consumed,
so every initialisation moved as well, and the retired five-seed ablations on these corpora spanned
0.32 to 0.80 (finance) and 0.62 to 0.93 (medical) on exactly these metrics. Single-draw moves of this
size sit inside those widths. **A seed sweep on the current circuit has not been run**, and the seed
is deliberately not re-picked to recover the finance figure, because choosing one on test performance
is test-set selection.

### Every published QCNN metric, current, with its classical baseline and its floors

Read from `models/*/weights.json`. `tests/test_a_model_card_states_what_its_weights_state.py` checks
each card against the file in its own directory, and was RED on all six cards until they were
rewritten to these values.

| artifact | headline | QCNN | classical LR, identical split | floor |
|---|---|---|---|---|
| `qcnn_telecom` | macro F1 | 0.510963 (acc 0.554167) | **0.976447** (acc 0.977083) | uniform 0.125, constant 0.027778, majority acc 0.152083 |
| `qcnn_telecom_q7` | macro F1 | 0.541894 (acc 0.585417) | **0.976447** (acc 0.977083) | as above |
| `qcnn_finance` | minority F1 | 0.735294 (acc 0.97) | **0.865672** (acc 0.985) | all-positive F1 0.098257, majority acc 0.948333 |
| `qcnn_finance_q9` | minority F1 | identical to `qcnn_finance` on every recorded metric | | |
| `qcnn_medical` | accuracy | 0.883721 (minority F1 0.84375) | **0.953488** (minority F1 0.939394) | all-positive F1 0.578512, majority acc 0.593023 |
| `qcnn_medical_q9` | accuracy | identical to `qcnn_medical` on every recorded metric | | |

**THE CLASSICAL BASELINE STILL WINS ON ALL THREE USE CASES.** It won before the Round 46 rewrite,
before the readout correction, and before this one. Nothing in this round is a quantum-advantage
result; the direction of the comparison has never changed and only its size has moved. Every circuit
does clear its floors, which is the only thing the aggregate numbers establish.

### The telecom headline has now moved four times, and this is all four

Stated once, plainly, because the number is quoted in several documents and three of the four are
wrong as a baseline for a different reason each.

| figure | what it was measured on |
|---|---|
| 0.809846 | the RETIRED hardware-efficient ansatz replaced in Round 46. A different circuit. **Never this one's baseline.** |
| 0.614583 | published as macro F1 by an earlier card rewrite. It was the ACCURACY of that run, not an F1. |
| 0.458324 | the real QCNN with the readout corrected to bin over surviving wires - the first true macro F1 of this architecture. |
| **0.510963** | **current**, after the dead parameters were made live. |

The first row is the one to watch for: it is the largest, it belongs to a deleted circuit, and it is
what an older document offers a reader asking what telecom scored.

### The width reversal holds and the gap NARROWED (0.042128 -> 0.030931); the pad wire is still inert

| telecom macro F1 | 8 wires | 7 wires | delta |
|---|---|---|---|
| superseded, Round 40/45 finding | 0.8098 | 0.7312 | +0.0786 for 8 wires |
| superseded, corrected readout and FC block | 0.458324 | 0.500452 | -0.042128, 7 wins |
| **current** | **0.510963** | **0.541894** | **-0.030931, 7 WINS** |

Two independent retrains now agree on the sign, having moved both sides. Both remain single-seed, so
this retires the published direction rather than settling the opposite one; `QC22` still applies.
`references/qcnn-width-ablation.md` carries the updated banner and nothing in it is deleted.

**Pad-wire inertness holds on both binary corpora**: finance 8 against 9 wires and medical 8 against
9 wires are identical on every recorded metric, for the third successive circuit. And the one place
the widths did differ has closed. Before this fix, 4 of 77 trained angles differed between the widths
by more than 1e-4 - exactly the four the `<Z0>` readout could not see, drifting on numerical noise
because no gradient reached them. Now **zero of 77 differ by more than 1e-4**: the largest
disagreement anywhere is 1.1e-05 on finance and 4.2e-06 on medical, with `scale` bit-identical
between the medical widths.

### The browser published a full-strength key under a complete intercept-resend attack

**MEASURED before the fix: at `eve_intercept_fraction` = 1.0 the browser reported 0.001976547
bit/pulse and 395 secure bits at a QBER of 0.2598 - unchanged, to every printed digit, from eve = 0.**
A page with a control labelled "Eve intercept fraction" moved the displayed QBER and nothing else.

Two causes, and the first is this round's own doing. The entry above replaced `intrinsic_qber` with
the BARE MISALIGNMENT in `_decoy_secure_fraction`, which removed a genuine double-charge of the dark
counts and, in the same line, **removed the only path by which `eve_intercept_fraction` reached the
bound**: `transmittance` does not depend on it and `E_mu` is modelled rather than measured, so with
the eavesdropper gone from `e_det` the decoy bound became independent of the attack. Second, the
browser had no counterpart to Python's `pp_threshold` abort, so even a QBER far above the
post-processing limit produced a key rather than a refusal. Python survived only because that abort
fired; the browser had nothing to fall back on.

Fixed as `e_det = misalignment + 0.25 * eve_intercept_fraction` in both languages - Ma et al.'s
`e_det` is the error on DETECTED SIGNAL photons, which is exactly what intercept-resend causes, at
the textbook 0.25 rate, while the dark-count term stays where it belongs as `e0 * y0` inside the same
function - plus porting the `pp_threshold` abort into `web/src/lib/quantumSim.ts`. After, at 200,000
pulses on the default 50 km fibre, **identical in both languages to every digit printed**:

| `eve_intercept_fraction` | bit/pulse | secure bits | what stops it |
|---|---|---|---|
| 0 | 0.001976547 | 395 | nothing; this is the published figure |
| 0.2 | 0.000058995 | 11 | the decoy bound, degraded 33x |
| 0.3 | 0 | 0 | the decoy bound reaches zero |
| 0.35 and above | 0 | 0 | `pp_threshold` = 0.11 exceeded; post-processing ABORTS |

The eve = 0 row is what the page used to print at eve = 1.0. **A protocol that does not abort has not
checked for Eve; it has only drawn her a QBER.**

### `clientOptiQKD.ts` was a FOURTH implementation of the decoy algebra, and the commit that fixed three did not name it

The entry above says the error term was fixed "in three languages at once" and lists
`src/qsh/qkd/bb84.py`, `web/src/lib/quantumSim.ts` and `web/scripts/check_qkd.mjs`. A grep for
`e0 * y0` returns FOUR files. `web/src/lib/clientOptiQKD.ts` computes the same bound for the
surrogate-model baseline and was left on `intrinsicQber`.

**Measured gap while it was behind**: 0.498% low on the 50 km fibre (0.001966695848615933 against
`bb84()`'s 0.001976547318720314) and **13.664% low on the 800 km free-space downlink**
(4.793756674e-5 against 5.552428392e-5). Both numbers are rendered in the SAME panel -
`qkd_surrogate_skr_default` sits beside `skr_per_pulse` - so a visitor could read two figures for one
channel and see them disagree. Now 0.000% on both.

This is the SEVENTH instance of the one-language defect, counting from the fourth named in the entry
above, and the sharpest: the commit whose subject line is that a fix reached one language enumerated
three call sites of four.

### Every browser-side cross-language gate was SKIPPING in GitHub CI

The gates that execute the browser's own TypeScript through esbuild and node, and compare its numbers
against Python's, are the only checks in this repository that can see a change reaching one language
and not the other - which is this round's entire subject. They guard on
`if not ESBUILD.exists(): pytest.skip(...)`. `web/node_modules` is gitignored and the Python CI job
installed no Node, so on a fresh checkout they reported SKIP and the workflow went green.

**MEASURED on a clone of the tip with no `web/node_modules`: 21 collected, 11 passed, 10 SKIPPED** -
6 in `tests/test_the_browser_computes_the_bound_the_page_names.py` and 4 in
`tests/test_the_qcnn_agrees_across_languages.py`. This entry said 20 / 11 / 9 for one round; that was
the count one commit earlier, and the commit that published it added a 21st gate which skips too.

Proven consequence, with the seeding procedure stated: replace the per-survivor `fcAt` pair in
`quantumSim.ts` with `for (const q of active) RY(q, w.theta[nConv + stages]);` - the pre-Round-47
shared scalar - and `tests/test_the_qcnn_agrees_across_languages.py` fails FOUR gates when esbuild is
present, at maximum readout differences **0.399170** (telecom-q7), **0.331032** (telecom),
**0.016233** (medical) and **0.008699** (finance), while the entire workflow passes when esbuild is
absent. **The single 0.585 this entry quoted is HISTORICAL and was stale in its own commit:** it
reproduces exactly - 0.585404 on telecom - against the PRE-RETRAIN weights of `f5861c9`, and the
commit that published the figure retrained all six artifacts in the same change. The local runner
`scripts/ci_local.sh` was therefore strictly STRONGER than CI, which is the reverse of what a CI gate
is for.

`.github/workflows/ci.yml` now installs Node 22, runs `npm ci` in `web/`, and sets
`QSH_REQUIRE_BROWSER_GATES=1`, which turns the skip into a failure. A gate that cannot run is not a
gate that passes; it is a gate nobody is running.

### The figures this continuation moved, and where they were corrected

| file | treatment |
|---|---|
| all six `models/qcnn_*/README.md` | rewritten to their own `weights.json`, per-class tables regenerated from `per_class_f1`; previous headlines marked superseded, not deleted |
| `references/qcnn-width-ablation.md` | Round 47 banner updated with the new width figures and the mixed-effect table; both superseded rows kept |
| `docs/deliverable-verification.md`, Round 46 section | the four intermediate figures marked SUPERSEDED in place, current values added beside them |
| `docs/deliverable-verification.md`, Round 47 section | 0.458324 / 0.500452 / 0.84058 / 0.8125 / 0.860465 marked superseded where they appear as current; the four-configuration sweep kept as capacity evidence |
| `references/REFERENCES.md`, `references/qcnn-entanglement-ablation.md`, `references/qcnn-vs-published-wdbc.md` | current values inserted where the previous retrain's figures were being presented as current |
| `docs/verification-log.md`, Round 47 entry | the list of quantities a visitor can see, and which have moved since the last walked bundle, extended with the eavesdropper and QCNN changes |

### Every Round 47 figure an adversarial re-measurement found wrong, and what it is now

The round above wrote a large number of quantitative claims. They were re-measured against this tree,
one at a time, and the ones below did not hold. Each row states what was CLAIMED, what is MEASURED,
and by what. Nothing here changes a conclusion; the directions all survive. What changed is that a
figure was quoted from a superseded intensity setting, from a pre-retrain artifact, from a different
metric, or with no recipe at all - which is the same class of defect as the one-language fix this
round is named for, in the numbers rather than in the code.

| claimed | measured | how |
|---|---|---|
| held-out sizes 171 / 1200 / 960 on the six cards | **86 / 600 / 480** | the trainer recorded validation PLUS test as `n_test`; the metrics were always on the smaller slice |
| telecom "the missing 0.156" | **0.110876** | 0.5692 (the previous artifact's macro F1) minus 0.458324; 0.156259 subtracted a macro F1 from an ACCURACY |
| "the two agree to 2.4e-07" (finance, medical cards) | **1.1e-06** finance, **4.7e-07** medical at 24 rows | the repo's own parity harness on the CURRENT weights; 2.4e-07 is the pre-Round-47 circuit |
| "0.57% low on fibre, 18.9% low on the 800 km downlink" | **0.4984% / 13.6638% low** at the shipped intensities; 0.5673% / 15.9152% low at mu 0.5 / nu 0.1 | 18.9% is the RAISE convention, not "low by", at any pair |
| "20 collected, 11 passed, 9 skipped" | **21 collected, 11 passed, 10 skipped** | fresh clone of the tip with no `web/node_modules` |
| "0.2701 divergence at readout_qubits=2" | **0.0335** | the repo's own seeded gate for that claim, torch seed 1, 16 rows |
| "fails these gates by a readout difference of 0.585" | **0.399170 / 0.331032 / 0.016233 / 0.008699**, four gates | seeding the old shared scalar at the tip; 0.585404 reproduces only against `f5861c9`'s pre-retrain weights |
| "`readout.wires` in every `models/*/weights.json`" | **the two multiclass artifacts**, 2 of 6 | the four binary ones carry `readout.kind` = `expval_z0` and no `wires` key |
| "at 10e6 the 800 km downlink yields ~3,660 sifted bits" | **~2,440** (5-seed mean 2,442) | `run_bb84` on the shipped free-space channel at the shipped intensities |
| "77.922% at every one" for the OptiQKD improvement | **113.429%** at seed 0, 106.354 to 113.431 across seeds 0-4 | `optimize_qkd`; the INVARIANCE across pulse counts is real and is what the exemption rests on |
| "at 40_000 the decoy bound yields 94 secure bits" | **79** | `run_bb84(40_000, 50 km)`; 94 is the mu 0.5 / nu 0.1 figure |
| Round 45 note giving 0.500452 / 0.458324 as the corrected circuit's answer | **0.541894 / 0.510963** | `models/qcnn_telecom{,_q7}/weights.json`; the older pair is now marked superseded in place |
| the dropped RZ "commutes with the `<Z0>`" | true of the RY-then-RZ ordering only | the fc block was reordered to RZ-then-RY later in the same round |
| 8-vs-9 wires "byte-identical" | **identical on every recorded metric**; `theta` differs by 1.1e-05 / 4.2e-06 | key-by-key diff of the two artifacts |
| dead-parameter figures 5.96e-08 / 8.94e-08 / 2.6e-01 / 5.9e-01 / 1.4e-01 | re-measured per artifact with the recipe stated - see the table in "Two classes of parameter were analytically dead" | pre-fix circuit from `8017538^`, shipped theta, torch seed 0 |
| "the largest of the four effects" (+0.053, telecom card) | the largest of the three GAINS; the finance regression of -0.105286 is twice as large | the four before/after pairs read from the artifacts at each commit |
| "the reversal holds and WIDENS" / "the gap GREW" | the gap **NARROWED**, 0.042128 -> 0.030931 | the two macro-F1 pairs the same sentences quote |
| finance_q9 / medical_q9: "both of those figures belong to the Round 46 circuit" | only one does; 0.813559 / 0.813953 are the RETIRED ansatz's stored metrics | `git show e94ea90:models/qcnn_*_q9/weights.json` |
| width ablation: Round 46's 0.976667 "made 0.84058" | an accuracy against a minority-class F1; finance ROSE on both metrics over that step | the artifacts at `4ee258e` and `3c63ea8` |
| "1.31e7 threshold, ~1,800 bits at 2e7, 217 bits at 1.4e7" | **1.406e7**, **1,626** bits, and **no key at all** (-12.0 bits) at 1.4e7 | the shipped config ships two of the optimiser's four variables; `p_signal` and `p_z` are not config keys |

One correction in this pass was itself wrong, and it is recorded because that is the same failure
one level up. The "0.45 max readout difference, argmax disagreeing on 39 of 64 telecom rows" in
`web/src/lib/quantumSim.ts` was withdrawn as unrecoverable, on the grounds that two retrains had
replaced the artifacts. It is recoverable: checking out `3c63ea8`, using ITS trainer and ITS
`models/qcnn_telecom/weights.json`, and collapsing only the fully-connected block to the shared
scalar reproduces **0.585404 with 8 of 8 argmax disagreements at n=8**, and **0.598319 with 62 of
64 at n=64**, on rows built as `tanh(standardise(randn(n, 7) * 0.7))` at torch seed 0.

The published figures therefore UNDERSTATED the defect rather than overstating it - the only claim
in this round's record that erred in that direction - and 39 of 64 is reachable only by feeding the
trainer unstandardised rows, which violates the preprocessing contract the round's own test file
documents. A claim withdrawn for being unmeasurable, when it was measurable all along, is the same
defect as a claim asserted without measurement.

## Verification assets added in Rounds 22-23

Two checks now exist that assert against sources OUTSIDE this repository. That is the point of both:
every other test in this suite asserts that the code does what the code was written to do, which
cannot catch a shared misunderstanding.

**`tests/test_decoy_reference.py` (13 tests) against `src/qsh/qkd/decoy_reference.py`.** Reproduces
Ma, Qi, Zhao and Lo, Phys. Rev. A 72, 012326 (2005) with the GYS parameter set, and the GLLP
tagged-fraction bound (QIC 4, 325). None of the asserted numbers originated here:

| Assertion | Published | Measured |
|---|---|---|
| tagged fraction at $`\mu = 0.1`$, zero distance | > 1 | 1.0417 |
| decoy max distance | over 140 km | 142.2 km |
| non-decoy max distance | about 42 km | 39.8 km |
| decoy $`R/\eta`$, flat, linear in $`\eta`$ | 0.11 | 0.1136 |
| non-decoy $`R/\eta^2`$, flat, quadratic | 0.15 | 0.1510 |

The trap this closed is worth keeping: the published curves are plotted for efficient BB84 where the
sifting factor $`q \to 1`$, while this project runs standard BB84 at $`q = 1/2`$. Reproducing them with
$`q = 1/2`$ gives values at exactly HALF the published ones - close enough to read as a fitting
discrepancy, wrong enough to void the comparison. `q_sift` is explicit and a test pins the factor.

**`npm run check:pqc` (`web/scripts/check_mlkem_modulus_vectors.mjs`).** FIPS 203 section 7.2
negative vectors from C2SP/CCTV, plus a FIPS 204 size and round-trip check. It found the pinned
`@noble/post-quantum` 0.2.1 accepting **all 2,595** malformed encapsulation keys; 0.6.1 rejects all
of them. The checker was itself validated against the known-bad build - 3 FAILs and exit 1 on 0.2.1,
all-OK on 0.6.1 - because a conformance script that always passes is worse than none.

**`check:pqc` now gates, as of Round 31.** It is step **8/8** of `scripts/ci_local.sh` and a step in
CI's `web` job, and `make check-pqc` runs it alone. The blocker recorded here for four rounds was
that it fetched its vectors; the resolution was not an offline-*tolerant* mode, because tolerance was
the trap. The vectors are CC0-1.0, so a 56 KB snapshot is vendored at `web/scripts/vectors/`, pinned
to an upstream commit with per-file SHA-256 in `PROVENANCE.txt`, and `--live` still checks the current
published set. There is deliberately no fallback in either direction: a missing snapshot exits 1
saying the check did not run, and `--live` exits 1 if the network fails. A skip-on-error version of
this particular check would have reported the known-bad 0.2.1 build as clean, which is the single
outcome it exists to prevent.

**The re-validation record here was wrong in two ways and is corrected in Round 33.** It said: "exit 0
with 2,595 rejected on a clean tree; exit 1 on a missing vector file; exit 1 and '779 rejected, 1
accepted' when a well-formed key is smuggled into the malformed set." Three re-validations, against a
checklist row (`RI9`) that requires **five** - clean 0, empty 1, truncated 1, tampered 1, missing 1 - and
the third of the three **can no longer be produced by the code that ships**.

Round 32 added a SHA-256 assertion on the vendored gzip, and it runs inside `loadVectors`, **before**
`gunzipSync`. The smuggle test kept the vector count at 780, which is exactly ML-KEM-768's count, so it
was built to slip past a count assertion - and that is precisely the case the hash catches. Any
modification to the vendored gzip now exits at the hash check printing "vendored snapshot does not match
its recorded SHA-256", so "779 rejected, 1 accepted" is unreachable on the vendored path. The script's
own comment states that the hash fires first and catches empty, truncated and tampered files alike, and
records it as measured; nobody then re-read the record quoting the superseded output. **A fix can
invalidate a test as easily as it can invalidate a claim, and only the claim gets re-read.**

**Outstanding, NARROWED in Round 43 - and this paragraph was overstating it.** It said the five-way
re-validation was outstanding. It is not: the five-row table, with each exit code and each printed
message, is transcribed at `docs/verification-log.md:434-444` under "The measurement this file is the
recorded home for, and which existed only in a commit message". Round 33 did what the condition above
asked for, and this sentence was never reduced to what remained.

Round 43 found it by grepping this repository for `RI9` in that file, getting **zero hits**, and
concluding the transcription had never happened. The grep matched the ROW ID; the measurement is
recorded under its own heading and does not name the row. **A search for the label is not a search for
the thing** - which is the same defect as everything else in this section, committed while auditing it.

What is genuinely outstanding is **one case of the five**: the smuggled well-formed key. It cannot be
run as written, because the SHA-256 fires before `gunzipSync`, so any record of it has to state **how
it got past the hash** - either re-vendor with the recomputed hash, so the count assertion is the line
of defence under test, or use `--live`, where there is no hash at all. **Resolve or condition by
**Round 45**. DEADLINE EXECUTED Round 45**, by the first route, and here is what it produced.

The case was run as a **bad re-vendor**, which is the one situation the file's own comment says the
count check exists for: `ML-KEM-512.txt.gz` was truncated from 775 vectors to 770, re-gzipped, and its
SHA-256 recomputed and written into `SETS` - so the hash gate is satisfied by a file that is genuinely
short. Measured output:

```
  ML-KEM-512: expected 775 vectors, read 770.
  This is NOT a pass. A truncated or empty snapshot tests nothing, and testing
  nothing reports zero acceptances - which is what a clean run also reports.
PQC conformance: 1 check(s) FAILED.          (exit 1)
```

So the count assertion **is** the line of defence once the hash is made to agree, and the failure text
names the confusion it prevents: zero acceptances from an empty file reads identically to zero
acceptances from a clean run. The vectors and the checker were restored immediately and re-verified
green (775 / 780 / 1040, 0 accepted). What this does NOT establish is a case where a *well-formed but
wrong* key is accepted - no such vector was constructed, and the honest scope of this discharge is the
truncation route, which is the route the code was written to survive.

## What this project cannot verify, and says so

- **No QKD hardware exists here.** Every SKR and QBER comes from an analytic channel model. There is
  no browser API for quantum communication and there will not be one.
- **The security model is semi-honest ON THE PATHS WHERE IT IS - and this bullet was a blanket claim
  that is false about the browser's non-linear layer. Corrected in Round 33.** It read "The security
  model is semi-honest. Malicious security was designated in Round 1 and never built", which made this
  the longest-carried item in the project. `web/src/lib/clientMPC.ts` runs `mpc-framework` over
  `emp-wasm-engine`, and `emp-wasm` is a WASM build of `emp-toolkit/emp-ag2pc` and `emp-agmpc` -
  **authenticated garbling**, Wang, Ranellucci and Katz, CCS 2017 (IACR ePrint 2017/030), titled
  "Authenticated Garbling and Efficient **Maliciously Secure** Two-Party Computation". Three other
  documents here describe it correctly as authenticated garbling; this register generalised from the
  paths where semi-honest is true. Same shape as the SEAL 4.4.0 advisory: **the register was wrong in
  the understating direction while the documents an audit would flag as overclaiming were right.**
  Semi-honest, scoped: MPyC / Shamir in `mpc/secret_inference.py`, the keyed-hash PSI, and
  `qsmpc/qot.py`. The caveat on the malicious-secure path is not about the protocol - both parties share
  one JS heap, so party separation is notional and this demonstrates a malicious-secure protocol rather
  than two mutually distrusting hosts (`clientMPC.ts:10-12`). **The residual is two named primitives,
  and `README.md` stated it more precisely than this bullet did:** a PQ-OPRF for the PSI, and
  **identifiable** abort, which authenticated garbling does not provide. Condition: each named with the
  browser-loadable library that would supply it, at the point where it is claimed.
- **Simulated quantum layers are simulated.** The ETSI control plane is standards-faithful over a
  simulated quantum layer, and every payload says so in-band.
- **GS QKD 014 mutual TLS is not implemented**, so SAE identity is a parameter rather than an
  authenticated fact. **Reclassified as a decision in Round 33, because the condition anyone would write
  for it is decoration.** Provisioning client certificates tomorrow would move nothing: `qkd/kme.py` is
  an in-process Python object, `api/` exposes no KME route, and mutual TLS is a transport property. The
  module's own `authentication_note()` already says it - "this in-process KME has no transport to attach
  that to". **Reopening condition: if the KME is ever exposed over HTTP, mTLS with the SAE identity taken
  from the client certificate goes in the SAME change.** The two must not be separable, because a
  reachable KME whose SAE identity is an unauthenticated parameter is worse than an in-process one that
  says so. Checkable by grep: a KME route in `api/` without mTLS is the failure.
- **Trusted relays are an assumption**, not a mechanism: a relay holds the key in plaintext and no
  cryptography here changes that.
- **Circuit privacy against the client (WS4(f)) is not achievable with the QOTP scheme implemented
  here**, and this is a property of the mechanism rather than a missing feature: the client can only
  decrypt by applying the circuit's action to its Pauli frame. Circuit privacy against the *server*
  (WS4(c)) is met for Clifford circuits and demonstrated; the two were split so one could be earned
  without the other being implied.
- **No phenotype exists in the genomics corpus.** 1000 Genomes is a reference panel, so nothing on
  that page can report a discrimination metric, and no supervised model can be trained from it. The
  page reports a distribution shift instead, which is what the data supports.

## CLOSED in Round 29, and this section was itself the thing that rolled forward (Round 23b)

Two items were recorded here as "known, verified and NOT fixed", in the present tense, in the section
whose stated purpose is that nothing deferred rolls forward silently. **Both had been fixed, and the
register was the last place still asserting them.** One of its sentences was flatly false against the
tree: it said `federated` is "absent from `web/src/lib/config.defaults.json`" when the key is present
and carries the browser overrides explicitly. That is worse than a stale number, because a reader
checking whether an item is open would have trusted it and stopped.

The lesson is narrow and worth keeping: a register of open items needs the same staleness check as any
other document. It is not self-maintaining just because maintaining it is its purpose.

**1. Browser federated hyperparameters - CLOSED.** `FederatedConfig` is exported by
`frontend_config()` and reaches `web/src/lib/config.defaults.json`, which carries the server values
(8 rounds, lr 0.05) alongside explicit `browser_rounds: 6`, `browser_lr: 0.15` and a
`browser_override_reason` string. The browser reads them from there rather than from literals, and CI
fails on a config-parity diff. The DIVERGENCE is unchanged and deliberate - a federated round has to
finish inside a page interaction - but it is now a declared override read from one source rather than
two numbers that happened to differ.

The residual this entry recorded - "`browser_rounds` and `browser_lr` are literals inside
`frontend_config()` rather than fields on `FederatedConfig`" - was **closed in Round 32**, and it is
worth noting why it took three rounds. It was recorded with a reason and **no condition that would
resolve it**, which fails this gate's first clause as well as its three-round clause, so nothing was
ever going to trigger it: an item with no condition cannot become due. The fix was two dataclass fields
and two lookups, smaller than the paragraph that deferred it.

**2. `qsdc_threshold` - CLOSED, by deletion rather than by justification.** The constant no longer
exists. `quantum_secure_direct_comm()` takes no threshold parameter and aborts on a loss-aware
secrecy capacity $`C_s = 1 - h(e) - g\,h(e_{\text{chk}})`$ with $`g = 10^{\,\text{loss\_dB}/10}`$, after Qi et al., *Light Sci.
Appl.* 8, 22 (2019), attributing the abort to the link budget where that is the cause. The 0.11 that
remains is `pp_threshold`, scoped to BB84/E91 post-processing where the Shor-Preskill bound is the
right reference, and the inert UI control that used to expose it was removed in Round 27.

**A third instance of the same 0.11, found in Round 29 and fixed there.**
`security_scoring.qber_secure_threshold = 0.11` awarded full eavesdropper-detection marks in both
scoring engines for any QBER below it - including the band [0.098106, 0.11) where this project's own
GLLP model returns zero secure key. The browser's fact display had been guarded against exactly this
and the score component had not, and `references/security-scoring-fact-check.md` already listed it
under defects fixed. Both engines now refuse full marks when the measured key is dead.

## Dated and dropped items, checked in Round 24

**ZeroGPU: DECIDED 2026-08-10, and the decision is not to wire it.** Round 22 set the condition as
"when the account unlocks, wire it or delete the Space and the `remote` tier together", and it must
not appear in a fifth deferred list unqualified. It appeared in five. The unlock is still pending -
the Space returns **HTTP 401** and the org space listing is empty - so waiting would produce a sixth.

The condition was framed around availability and the real question is architectural. **This demo's
stated property is that it computes privacy-preserving inference in the browser with zero VPS
compute.** A remote GPU inference tier does not extend that; it contradicts it. The number a REMOTE
badge would carry - "computed off this machine, timing includes network latency" - is the one claim
the public demo exists to avoid making. So the unlock date was never the blocker, and treating it as
one is what let this ride for five rounds.

Executed: the `remote` member is deleted from `ProvenanceBadge` (nothing set it, so no behaviour
changes), and `t3_large` is retained but documented as an intentionally empty taxonomy slot, because
the in-browser LLM genuinely is T3-shaped and simply is not a registry entry.
`spaces/qsh-inference/` is KEPT as reference source and is explicitly not on any roadmap - it is not
deleted because it is real work that predates this analysis and costs nothing to retain.

**Reopening condition, stated so this is a decision and not an omission:** wire it only if the demo
acquires a workload that genuinely cannot run client-side, and re-add the `remote` badge in the same
commit that first sets it.

The previous entry, kept for the record: the org page returned HTTP 404 and the Space HTTP 401 on
2026-08-09, the account-age unlock (~2026-09-01) had not happened, and the condition was recorded as
one command to re-check:

    curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/spaces/Daemons-Q/qsh-inference

**A 401 is the blocker; a 200 is the decision point.** On 200, either wire the Space or delete it
together with the `remote` tier - a type member nothing sets is the same defect as a metric with no
reader, which this project treats as serious.

**EU AI Act Art. 50(2) - CLOSED as not-applicable. The date was right; my correction to it was
wrong, and that correction is retracted here.**

Round 24c rewrote this entry to say the recorded 2026-12-02 was "wrong" because Article 50 became
applicable 2 August 2026. Checked against the Official Journal text of Regulation (EU) 2026/1744:
**2026-12-02 is correct for this system.** The recital is explicit - "it is appropriate to introduce
a transitional period of four months for providers who have already placed their systems on the
market before the 2 August 2026" - and that transition applies specifically to the Article 50(2)
MARKING obligations. This demo predates 2 August 2026, so four months from that date is its
applicable deadline. Article 50 generally applies from 2026-08-02; 50(2) marking, for pre-existing
systems, from 2026-12-02. Both statements are true and the earlier entry had the right one.

What IS correct in the Round 24c rewrite: the Annex III deferral to 2027-12-02 does not touch
Article 50, and the C2PA-versus-in-band choice is not open because neither is needed. The obligation
was analysed and found **not to apply** regardless of date: the panel
emits 96 tokens (Transformers.js) or 160 (WebLLM), both under the Code of Practice's 200-token "very
short text" exception, so no marking layer is required and therefore no detection duty is triggered.
The full quoted chain is in [`ai-act-art50-gap-analysis.md`](ai-act-art50-gap-analysis.md), along
with the three standing constraints that would reopen it - exceeding 200 tokens, shipping generated
text inside a file container, or C2PA text bindings becoming implementable. C2PA is not a viable
route today: spec 2.4 defines text bindings but all 157 entries on the Conforming Products List are
still 2.2.

**Two items that fell off the backlog without a decision**, found by reading every plan file. Both
appeared in five consecutive plans and then simply vanished, which is the failure mode Round 22's
backlog gate was written to catch and did not:

- *The Zhang et al. QOT optics figure.* Deferred from Round 4 to Round 16, absent thereafter. What
  was built instead is `TwoPartyDiagram.tsx`, a data-driven sequence diagram that fills from live
  metrics. **Declined**: a static reproduction of a paper's optical bench would be a picture, while
  the diagram that exists shows the run that actually happened. The right citation lives in
  `references/`, not in a redrawn figure.
  *Reopens if* the QOT path ever runs against real optics rather than a simulation, because then the
  bench layout stops being someone else's picture and starts describing this project's own apparatus;
  or if `TwoPartyDiagram.tsx` stops filling from live metrics, since the whole argument for declining
  is that the diagram which exists is the measured one.
- *GitHub / Google Drive MCP interop.* Deferred with the reason "needs OAuth creds and outward-facing
  accounts that cannot be provisioned in an automated build", which is still true. **Declined**:
  filesystem MCP interop already demonstrates that the quantum-safe channel carries a real third-party
  MCP server, and a second connector would demonstrate the same property at the cost of credentials
  this project should not hold.
  *Reopens if* a third-party MCP server can be reached with **no credential this project should not
  hold** - an unauthenticated or read-only-public server - because the stated cost then goes to zero
  and the demonstration is strictly wider. It does **not** reopen because OAuth became easier; the
  objection is to holding the credentials, not to the effort of obtaining them.

Recorded as decisions so neither reappears as a silent omission, and both now carry the condition
that lets them fail the three-round rule. A decision with no reopening condition cannot be wrong -
it can only be permanent, which is the property this register exists to deny.

## Round 48: the finance page ran a model with its best feature switched off - written in Round 49

**BUSINESS DOC RECONCILED Round 48.**

**This section is late, and the lateness is the first thing in it.** Round 48 shipped three commits
(`2d83b62`, `5aa444b`, `75b4cbf`), merged as `e192214`, and deployed - and wrote no register section,
no verification-log entry, and no checklist rows. The entire record layer was skipped. It is written
here in Round 49, marked as such in the heading, following the convention the log already uses to
separate the round a section is ABOUT from the round it was WRITTEN IN.

What that omission cost is in the Round 49 section below: every gate that could have noticed reads
the round number from this file, so a round that does not write here becomes invisible to all of
them at once.

### The finance QCNN was fed the wrong feature space

`web/public/data/finance/qcnn_weights.json` declares `input_space: "raw"` - its moments were fitted
by the trainer on the raw TSV - and the page fed it `paysim_sample.json`, whose rows are
StandardScaler output and say so in that file's own `scaling` field. The browser standardised
already-standardised data a second time, with a DIFFERENT scaler.

Measured through the browser's own `qcnnInferTrained`, comparing the array the page fed against the
array the artifact declares:

| | fed `samples` (as deployed) | fed the declared space |
|---|---|---|
| wire 0 (`amount`) tanh sd | **3.3e-05** | **6.35e-01** |
| range | [-0.99999992, -0.99982701] | [-0.99997923, 0.99150921] |
| saturated `\|tanh\| > 0.999` | 556 of 3200 | - |
| accuracy on the bundled labels | 0.8200 | 0.9125 |
| minority-class F1 | 0.7857 | 0.9041 |
| rows given a different class | - | **101 of 400 (25.25%)** |

The model's most informative input carried no information at all. Medical is correctly paired
(`zscore` declared, z-scores fed) and telecom is correct (`raw` declared, `clientIDS.ts` hands over a
genuinely raw row). Finance was the only mismatched pair, and nothing anywhere compared the two.

**Two different comparisons exist here and only one describes what a visitor saw.** The figures above
compare two arrays through one encoder. A second measurement - 69 of 400, 17.25%, accuracy 0.9875 -
compares `raw` against `zscore` on the SAME array, one array through two encoders, and describes a
different fix ("stop standardising") that the trainer's raw-fitted moments rule out. Both are true of
what they measure. Round 48's own commit said so and then wrote the superseded pair into two of the
three places it appears; Round 49 corrected those.

Neither figure is a generalisation estimate: the 400 rows are drawn from the artifact's own training
corpus, and the held-out numbers are on the card.

### The fix, and why the first version of it was the same defect

The payload now declares `feature_space` for the array it ships and `samples_by_space` for every
other, and consumers ask for the space they were trained on **by name**. The first version read
`input_space === "raw" ? rawRow : feats` - a silent default branch that would send an artifact
retrained into a third space to whichever array the `else` names, which is exactly how the defect
arose. A miss is now a refusal, not a fallback.

`samples` keeps its bytes: the distilled CKKS-linear model and the exact-linear SHAP panel are fitted
in that space. Verified rather than assumed - after regeneration `samples`, `labels`, `model`,
`provenance` and `names` are byte-identical.

### Three more things the page told a visitor

- **A short artifact returned "normal" instead of failing.** `quantumSim.ts` indexed `w.theta` with
  no length guard: `undefined` becomes NaN through cos/sin, `NaN > 0` is false, the argmax loop never
  advances, and the readout returns class 0 - which for telecom is the class named `normal`. A
  retrain changing the parameter count without changing the survivor set produces exactly it. It now
  throws, naming both counts.
- **A retired bound was cited as the source of a QBER fact** - "GLLP zero-key point 0.098106",
  removed from both languages in Rounds 45-46. The decoy bound's zero-key point is channel-dependent,
  so no constant replaces it; the fact names the mechanism instead.
- **One panel, two roundings** - `skr_per_pulse` at five decimals from Python and six from the
  browser. That fix reached three of four sites; the fourth is corrected in Round 49.

### A bound true only on an input range nobody wrote down

Two docstrings claimed the untrained QCNN's `predict_proba` "never exceeds 0.241, so it never crosses
the 0.5 boundary and returns class 0 for every input". Measured, 200 draws per cell, fixed circuit at
seed 0:

| inputs | nq=4 max | nq=8 max | crossings of 0.5 |
|---|---|---|---|
| uniform[0,1) | 0.2392 | 0.2304 | 0 / 200 |
| uniform[-1,1) | 0.2280 | **0.2737** | 0 / 200 |
| normal(0,1) | **0.9834** | **0.9122** | 18 / 200 and 20 / 200 |

It reproduces on uniform[0,1) - presumably what was drawn, and never stated - and is not robust even
inside bounded inputs. The conclusion survives; the recipe is now recorded beside it.

### Gates added

- `tests/test_the_encoder_receives_the_space_the_artifact_declares.py` measures the SYMPTOM rather
  than comparing labels: a wire carrying variance in the source must still carry it after the
  declared preprocessing. Scope globbed from `web/public/data/*/qcnn_weights*.json`.
- Its duplicate half was too weak on the first pass - it accepted a column merely being NAMED in
  `eda.columns`, which names every column by construction, so it could not have failed on anything.
  Recorded in the file rather than quietly tightened.
- `amount_zscore` equals `amount` to 1.0e-05 and nothing declared it, while the SHAP panel split that
  feature's attribution across both bars. `_eda` now computes duplicate pairs.
- The cross-language QCNN gate covered 4 of 6 artifacts from a hand-written list, on a branch named
  for file lists. Both omitted widths are visitor-selectable. Now globbed: ten comparisons.

## Round 49: the round that left no record, and the machinery that could not notice

**BUSINESS DOC RECONCILED Round 49.**

Round 48 shipped and wrote nothing here. That is an omission; what makes it a defect is that **every
gate able to notice reads the round number out of this file.**

`tests/test_register_deadlines.py::_current_round()` takes `max()` over this register's `## Round N`
headings. `tests/test_the_log_has_an_entry_for_the_current_round.py` defines a second copy of that
function, reading the same file, and its own comment explains why: *"Deriving the round from the same
place both files do means this cannot disagree with the deadline gate about which round it is."*

That is true, and it is the blind spot. They cannot disagree with each other, and neither can
disagree with a register that stopped being written. Measured on the tip that shipped Round 48:
`_current_round()` returned **47**, so the log gate asked "does the log name round 47?" - it did -
and both went green with a whole round missing. The three obligations conditioned `by Round 48`
stayed pending for the same reason, since the deadline comparison is against that same number.

**The round is recoverable from git, which forgetting to write prose cannot change.** Every round
merges a branch named `roundN-...` and the merge subject records it.
`tests/test_the_register_records_every_round_that_shipped.py` compares the merged history against
this file and fails when this file lags. Verified against the actual failure rather than a
hypothetical: at `e192214` the history said **48** and this register said **47**, and the gate is red
there. It is seeded on the code, too - `_shipped_round` is asserted not to read the register, so a
refactor cannot quietly turn it into a third copy of the same blind spot.

A claim drafted for this round and withdrawn before it was written, recorded because it is the
failure mode the round is about: that the deadline scan reads only this register, leaving an
obligation in `references/` invisible. It does not. There are two deadline tests and the second scans
`_watched()` - tracked markdown under `docs/` AND `references/`. The scope was already right; only
the number it compares against came from a file a forgotten round fails to write.

### Round 48's fixes each reached one site short

Two instances, both found by re-measuring rather than re-reading, and both inside the commit that
described the defect class.

- **`skr_per_pulse` had a fourth emitter.** Round 48's "one panel, two roundings" fix moved
  `api/inference_service.py` from five decimals to six and reached three of four sites.
  `orchestration/stages.py` kept five - nine lines below its own satellite twin, which already used
  six - so the fibre orchestration path published 0.00198 for the quantity every other surface
  published as 0.001977. `tests/test_one_metric_has_one_rounding.py` now compares every metric name
  emitted from more than one site and requires one precision per name.
- **The corrected finance figures landed in one of three copies.** Round 48's commit stated that
  25.25% is the comparison describing what a visitor saw and that 17.25% describes a different fix -
  and then left 17.25% in `build_federated_samples.py` and `clientCompute.ts`, correcting only
  `realData.ts`.

The rounding gate over-fired on its first pass, matching `mean` and `sd` - per-column keys in two
unrelated EDA blocks that share an English word. The fix was to scope it to names published through a
metric channel, not to add them to the exception list: an exception list that absorbs a scoping error
is how it becomes a list of things people got tired of fixing.

### Calibration: the open measurement is closed, and the estimator was never the variable

**DEADLINE EXECUTED Round 49.** Round 46 conditioned the isotonic comparison and Round 48 did not run
it. It is run: `scripts/train/finance_calibration_ablation.py`, on the shipped split, reproducing
`models/metrics.json`'s AUROC (0.9525582593503528), AUPRC (0.12334101789973353) and uncalibrated
Brier (0.00150259) to every recorded digit. Full workings, every cell of every sweep including the
ones that do not support a clean reading: `references/finance-calibration-isotonic-vs-platt.md`.

**Two facts in the Round 46 paragraph are wrong, and the second one is not about the argument.**

The calibration set carries **801** positive events, not 2,321. 2,321 is the TEST split's count -
`models/metrics.json` records `positives_calibration: 801` one line above the sentence that says
2,321. Correcting a wrong denominator with a wrong numerator is the same defect at one remove.

And `fit_platt` is handed a **probability**, not a log-odds. Its map $`p = 1/(1 + e^{a s + b})`$ is linear
in `s`; 90% of the teacher's scores are below 1e-4, so with `a = -7.8378` the map is constant to four
decimals across nine of ten deciles and returns **0.00090148** for all of them. Its entire output
support is [0.00090148, 0.695715] - it cannot express a risk below 1 in 1,109 for any input, on a
corpus whose base rate is 1 in 656. **That is where the +8.7% log-loss the record already reported
comes from**, and it was read for three rounds as a property of the base rate. The measured ECE says
the shipped calibrator is 2.83x WORSE than not calibrating at all.

`tests/test_calibration.py` could not have caught it: its fixture is a logit and the trainer's
argument is a probability, so the suite and the pipeline are each internally consistent about
different things.

**The comparison the register actually asked for.** Fitted on 710,969 cases / 801 events, scored on
the held-out 1,523,504 / 2,321:

| arm | Brier | log-loss | ECE (20 equal-frequency bins) | AUPRC |
|---|---|---|---|---|
| uncalibrated | 0.00150259 | 0.00965987 | **0.00064837** | 0.123341 |
| Platt as shipped | 0.00145433 | 0.01049835 | 0.00183223 | 0.123341 |
| Platt on `logit(s)` | **0.00142199** | **0.00818964** | 0.00078864 | 0.123341 |
| isotonic | 0.00142218 | 0.00820025 | 0.00077500 | **0.112125** |

ICML 2005's sentence holds at 710,969 cases in the exact form it is written - **as good as**. The
gap between isotonic and a correct Platt is 1.9e-07 of Brier. Note the column nobody predicted: the
UNCALIBRATED score has the best ECE of the four, and no calibrator improves it.

**Cases or events: the two sweeps disagree, and reporting only the one that answers cleanly would be
the failure this register keeps documenting.** Base rate is events over cases, so no sweep can pin
two quantities and vary the third; three sweeps pin one each, 20 draws per cell. Against the SHIPPED
arm the case count binds - pinning cases at 710,969 and cutting events 267-fold (801 to 3) never
flips the sign, while pinning events at 100 and cutting cases flips it between 10,000 and 30,000.
Against the CORRECT arm the event count binds, crossing over at **100-200 events**, two orders of
magnitude below the papers' CASE thresholds and already passed by 801. At the deployment base rate a
calibration set small enough for the published Platt regime holds one positive, and `fit_platt`
refuses a one-class set - so at 1 in 888 the two quantities are not separable at the small end at
all, which is a limit of the corpus rather than a finding.

**The recommendation is neither of the two on offer: keep Platt, and change what it is fed.**
Isotonic wins nothing that a corrected Platt does not also win, and it loses three things that are
not proper scoring rules. It is only WEAKLY monotone, so it collapsed 1,388,669 distinct scores to
138 and cost AUROC 0.00031674 and AUPRC 0.01121590 - **9.09% of the metric this project prints beside
its headline**, on a transform the record claims cannot move ranking at all. It emitted probability
exactly 0 on 464,176 test rows, one of them a real laundering transaction. And `Platt.as_dict()`
publishes two floats and a formula so the browser can reapply the identical map; 138 breakpoints are
not that.

**What is NOT done here, said rather than left to be discovered.** ~~The one-line argument fix is not
applied: it retrains the teacher and rewrites `models/`, and this round was scoped as a measurement.
So `models/metrics.json` and the `config.defaults.json` derived from it still carry the superseded
`why_not_isotonic` sentence~~ - **SUPERSEDED later in this same round: the fix WAS applied, the
teacher retrained, and both derived artifacts regenerated.** They now carry `on_logit: true`, the
log-odds `formula` string, `logit_clip_eps`, and `a = -0.5853491088584304` /
`b = 1.6417462922842536`, reproducing the ablation's predicted `-0.58535` / `1.64175` to five
figures. The struck text stands rather than being deleted because the doctrine in the rest of this
sentence is still right and was followed to the letter: the derived copies were never hand-patched,
the generator was corrected and the artifacts re-derived from it. They are deliberately NOT
hand-patched - the generator (`src/qsh/models/calibration.py`) is corrected instead, because a
hand-edited generated artifact is a defect with a timer on it. **ONE CLAUSE OF THAT DOCTRINE IS
WRONG AND WAS MEASURED WRONG WHILE APPLYING IT:** `scripts/check_generated_prose.py` does NOT exist
for exactly that reason in this case, because it cannot see this field at all - it does not descend
into nested dicts, the calibration block reaches `record_metrics` through a `**splat` whose AST key
node is `None`, and its trainer glob never reads `calibration.py`. The gate exited 0 throughout
while `models/metrics.json` carried a `does_not_move_ranking` its generator no longer emitted. The
config-parity gate is no help either - it compares the two DERIVED copies to each other, so it stays
green while both are stale together. **Nothing in CI would have caught a skipped retrain.** One
visitor-facing copy WAS corrected, because it was false on its own terms rather than merely stale:
`web/src/components/ModelPanel.tsx` still told visitors this calibration set "is inside the range
where the published comparison puts Platt ahead" - the claim Round 46 refuted, in a file Round 46's
correction never reached. That is Round 48's "landed in one of three copies" defect, found again in
the paragraph that documented it.
## Round 50: one column explained the gap, and the round's own claims needed correcting three times

**BUSINESS DOC RECONCILED Round 50.**

Round 49 shipped a model and a claim about it. The model was invisible - it had never been deployed -
and the claim was wrong about someone else's work. Both are fixed, and the corrections came from
measurement rather than from re-reading.

### DP7 discharged after sixteen rounds, and the walk found what it was for

The demo box had been left on the Round-48 branch, so Round 49's entire output - including a published
model - reached no visitor. `scripts/deploy_demo.sh --return-to-main` puts it on `main` at `f555a89`.
Measured before and after from the served bundle: `index-CvOb630D.js` carried Round 48's `input_space`
and zero occurrences of `telecom_ids.nidd`; `index-Dxu788k6.js` carries the model, the pin and the
attribution. All ten pages walked - no `NaN`, no `undefined`, no duplicate headings - and the 5G-NIDD
model reproduced its recorded metric client-side across all 16,379 published rows at `|delta| 0.00e+0`.

### The 5G-NIDD claim was overstated, and the cause is one column

Round 49 wrote that the published 99.6-99.7% figures are "an anchor rather than a target" and that "it
is not established that those papers dropped `Unnamed: 0`". Neither sentence had been checked, and
leakage-aware work on this corpus exists - three papers, now cited in
`references/5g-nidd-metric-profile.md`.

Measured instead of argued. Binary benign-vs-attack scores **0.7630** against the 9-class **0.7608**,
so the multiclass structure costs nothing and the whole error budget is one pair: Benign/UDPFlood,
23.7% of the test split, with all seven other classes contributing 0.23%. The cause is **`IdleTime`**,
a flow statistic the release's `Combined.csv` export does not carry and its
`BTS1_BTS2_fields_preserved.csv` does. Refitted on the wide export with every identifier and host
fingerprint dropped: **accuracy 0.982897, macro F1 0.995019**, Benign recall 0.5345 -> **1.0000**,
residual error 1.71% and all of it in one direction. Republished at revision `a3cbe059`.

So the published range is reachable **without leakage of any kind**, and Round 49 read a 24-point gap
as evidence about other people's methods when the larger part of it was its own choice of export. The
identifier finding stands on its own - four row counters still move the configuration to 0.9997 - but
it does not explain anyone else's numbers.

### Three of this round's own claims were wrong, and each was caught by checking rather than by review

Recorded because the pattern matters more than the individual figures:

1. **The leak reproduced itself inside the investigation.** The first wide fit returned 0.9997 and was
   briefly taken as confirmation. The drop list named `SrcId`, `Rank`, `Seq` and `Offset` but not
   `Unnamed: 0`. Caught only because 0.9997 was suspiciously identical to Round 49's counterfactual,
   then confirmed with a deliberate control.
2. **"The other five new columns each score AUC 0.5000" was false.** `Dir` scores 0.9811 and `TcpOpt`
   0.9433, both above this project's own leak threshold. The true statement is that they are
   *redundant*, not empty - dropping all six costs 0.2219 against 0.2213 for `IdleTime` alone. That
   claim had reached five places before it was corrected.
3. **The proposed explanation for the surviving metric inversion was wrong.** It was written that a
   small class drags the macro mean down; ICMPFlood scores 1.0000 and *lifts* it. The mechanism is
   unchanged from before the refit - Benign and UDPFlood carry 77% of rows, so accuracy is dominated
   by the only pair that confuses while the macro is carried by seven classes at 0.9999 or better.

The inversion itself **persists** and the record says so: macro F1 0.9950 above accuracy 0.9829. What
changed is its size, from 0.1655 to 0.0121. It was not fixed and is not described as fixed.

### Six models explained their own metrics to nobody

`metric_note` reached the browser inside `config.defaults.json` and had no render path in `web/src/`.
Six records carry one. The consequence was live: the companion row's caption was the hardcoded string
"companion - shown so the gap is visible", which asserts the aggregate is the flattering figure -
false of `telecom_ids.nidd`, whose macro F1 sits above its accuracy. The one model that needed the
caveat was the one whose caption contradicted it, while the note explaining it was invisible. The
caption is now derived from the two values it captions.

### Stale figures, and one that was printing the trivial predictor as a score

`README.md` reported medical student accuracy as 0.730769. That is the artifact's own
`majority_baseline`; the accuracy is 0.769231. Two more were superseded retrains (0.864865 ->
0.904348, 0.333709 -> 0.376130), and two derived figures moved with them - the all-positive ratio is
9.6x not 9.3x, and "19x" described the prevalence ratio, not the F1 ratio, which is 6.1x. The whole
sentence was duplicated verbatim in `docs/verification-log.md` and stale in both; the copy is removed
rather than corrected twice.

`DATA_LICENSES.md` claimed "the rules now state a licence reason per file". Four of eight reference-PDF
rules still exclude on size, and `.gitignore`'s own comments say so - two tracked files contradicting
each other about which rule is in force. Also corrected: "every PDF under `references/` is handed to
anyone who clones" is false of a working checkout, where 7 of 11 are tracked.

### The checklist had a duplicate this project's own row forbids

`DP8`, added in Round 49, restated `DP7`'s second clause with the same command and the same failure
history - in the same table as `RT10`, "No two checklist rows assert the same thing in different
words". Removed. `CS12`'s rationale claimed `CS1` reads the source when `CS1` says "read the network
panel, not the code"; `QK3` wrote the bare `skr` this file retracts twice. Fourteen rows added for
gaps the standing request names and the file genuinely lacked, each verified absent first. Recount
with the file's own regex: **449 rows, 53 prefixes, zero duplicate IDs** - and Round 49's report of
"377 rows, 40 prefixes" was wrong, from a narrower regex than the file documents.

## Round 51: the caption swapped one wrong claim for another, and the plot could not be drawn honestly

**BUSINESS DOC RECONCILED Round 51.**

### The second walk, performed

Round 50 recorded a walk as owed because its own walk observed the pre-retrain model. Done, on
`index-CYDapC47.js` at `62e6b79` - the content that merged as `9adcdf0` - with **all five pipelines
RUN to completion**, which Round 50 could not claim. Every published model reproduced its recorded
metric in-tab: finance gbm and student, medical resnet18/resnet50/student, satellite telemanom/opssat/
opssat_student, telecom qml and nidd. No `NaN`, no duplicate headings, no `[object Object]` anywhere.

`encrypted_agreement` is 1 on finance (400/400) and medical, and **0.8 on genomics** - which is not a
defect and the run log says why: five of thirty rows flip, seven of the thirty are scored IDENTICALLY
across classes by the plaintext model, the widest plaintext margin among the flipped rows is
**0.000e+0**, and over the 23 rows the model actually decides agreement is 1. Max CKKS error 8.277e-10.
The encryption broke ties; it did not disagree. **Recorded as still-open**: the metric row renders a
bare 0.8 beside 1.0 elsewhere and the sentence that makes it readable lives in the log, which is the
same defect class as `metric_note` rendering nowhere.

### Round 50's fix was wrong in a new way, and the deploy is what showed it

Round 50 replaced a hardcoded companion caption - "shown so the gap is visible", asserting the
companion is always the flattering figure - with one derived from the values. Deriving the direction
was right; attaching an interpretation to it was not.

`medical_fl.resnet18` renders `auc` 0.9147 as its declared metric and `accuracy` 0.8654 as the
companion, and the caption announced the accuracy as "the pessimistic figure, not the flattering one".
AUC is rank-based, accuracy threshold-based; on an imbalanced split AUC sits above accuracy by
construction. That model's own `metric_note` - made visible by the same round - says *"Do not read the
accuracy against the AUC baseline"*, so the caption contradicted the caveat it existed to reveal.

The caption now states the relation and stops. **Three rounds, three forms of one defect**: a
hardcoded claim, a derived claim that over-reached, a stated relation. What recurred was asserting an
interpretation the code had not checked.

### The requested prior-research plot cannot be drawn honestly, and that is the finding

The standing request asks for prior-research accuracy on the private-inference plots, and Round 50
wrote `RD14` demanding one. Measured before building: **not one published model has a baseline whose
metric lives on its chart's axis.** Medical's baseline is an AUC against a per-class-F1 chart; finance,
satellite and medical-student are F1 / F0.5 / AUC-PR against a precision axis. The single pairing that
would map - `telecom_ids.nidd`, macro F1 on a macro-F1 chart - has `baseline: null` by design.

So the line would be wrong on all five that have a baseline and correct only on the one that does not.
0.901 on an F1 axis is the right numeric range and the wrong quantity, and a chart gives the reader no
column header to check it against. Not built. `RD14` is rewritten from "must be plotted" to "shown
beside the metric it measures, and NOT drawn on a foreign axis", and a gate pins the mapping so a
changed metric or plot pairing forces the question to be re-answered rather than inherited. It also
asserts on the code that `VerificationChart` never reads `metric.baseline` - two fields share that
name, the registry's being prior research and `ModelEvalResult`'s the trivial predictor.

### Opset 0 is not an opset

`graph_opset` returned 0 for "no default-domain opset", which is the tree-ensemble case by
construction, so `finance_aml.gbm` and `telecom_ids.nidd` both published `export.opset: 0` to
visitors. `None` now means absent and the row disappears. Re-derived from the shipped graphs rather
than retrained; re-serialising `metrics.json` changed exactly the two intended lines.

### DP7 not discharged

The return-to-main deploy was blocked by a local permission control. The served content is the merged
tree, so nothing a visitor sees is stale - what is wrong is the box's branch pointer. Recorded here
rather than left to be discovered, and it carries to Round 52.

### Four published models were distributing work whose licence conditions were unmet

Found while checking dataset commercial-use eligibility, which is a different question and is how it
surfaced at all. Two corpora sit under licences that REQUIRE attribution and carried an empty credit
line:

| corpus | licence | published models affected |
|---|---|---|
| ESA Anomaly Dataset (ESA-ADB) | **CC-BY-3.0-IGO** | `satellite_comms.telemanom`, `satellite_comms.student` |
| IBM AMLworld HI-Small | **CDLA-Sharing-1.0** | `finance_aml.gbm`, `finance_aml.student` |

All four are live on the Hub, so this is an unmet condition on distributed artifacts rather than an
untidy record. **The two clauses are not the same thing and the fix should not blur them**: CC BY 3.0
IGO is an author-credit requirement; CDLA-Sharing-1.0 section 3.1 is a notices-must-travel
requirement. Both warrant the credit line; only the first is credit in the CC sense, and describing
CDLA as an attribution licence would be over-claiming in the opposite direction from the original
defect.

**This is the EIGHTH instance of one defect class, and the seventh was the fix that caused it.**
Round 49 added `DatasetSpec.attribution` and backfilled what it called "all four CC BY corpora in the
same commit", explicitly reasoning that adding it for one corpus is this project's most-repeated
mistake. It matched on the literal string `CC-BY-4.0`. The two corpora whose licences are SPELLED
differently - `CC-BY-3.0-IGO` and `CDLA-Sharing-1.0` - are exactly the two it missed. They then stayed
missed for two rounds because every other corpus was correct, so no page looked wrong and no gate
fired.

`tests/test_every_attribution_licence_carries_its_credit.py` matches the licence FAMILY (`CC-BY`,
`CC BY`, `CDLA-SHARING`) instead of a literal, so a corpus added tomorrow under `CC-BY-SA-4.0` or any
IGO port is in scope the day it lands. CC0 and public-domain are explicitly waived, so the gate's
silence about them is a recorded decision rather than an oversight. Seeded both ways.

**A second finding, which explains how the ESA-ADB gap was possible.** The CC BY 3.0 IGO term appears
on the Zenodo landing page and is ABSENT from Zenodo's REST metadata, which returns no licence field
at all. An API-only adjudication reads this corpus as unlicensed - the same failure shape as Round
47's 5G-NIDD error, where a mirror's silence was read as the source's silence. Recorded on the model
card so the next reader does not repeat it.

**Why each licence bites, read at source rather than summarised - and the two reasons are different.**

*CC BY 3.0 IGO* requires keeping copyright notices intact and providing "any attributions that the
Licensor indicates", the title, and the URI where practicable. It reaches adaptations, so
`satellite_comms`'s `eval_sample.json` - real ESA-ADB channels resampled to 60s and z-scored - is
covered as an adaptation of the Work.

*CDLA-Sharing-1.0* is narrower and the narrowness is the point. Section 3.1(c) requires preserving
"all credit or attribution to the Data Provider(s)" and 3.3 requires the licence name or a hyperlink
to travel - but **section 3.5 exempts Results, the outputs of computational analysis**. Trained
weights are a Result. So the obligation does NOT attach to `finance_aml.gbm` because a model was
trained on AMLworld; it attaches because its `eval_sample.json` ships **8,192 row-level records** of
the corpus, described in its own provenance as "unmodified rows of the published corpus, carrying the
engineered features". Row-level derived columns are Enhanced Data under 3.1, not a Result under 3.5.

That distinction is worth holding precisely in both directions. Overstating it - calling CDLA an
author-credit licence - would be the mirror of the original defect. Understating it, by reading 3.5 as
covering anything a model touches, would have concluded no credit was owed at all and left the
condition unmet.

**The registry fix alone does not discharge the condition.** Hub cards are generated, so the credit
reaches the published artifact only on a republish. Fixing the record and not republishing would leave
four uncredited artifacts and a repository that believes otherwise - which is the failure this round
is about, one layer up.

## Round 52: the credit reached the artifact, and two gates were wider than the drifts they watched

**BUSINESS DOC RECONCILED Round 52.**

Round 51 put the missing attribution in the registry and the browser. That did not discharge either
licence, because Hub model cards are GENERATED - the credit reaches the published artifact only on a
republish, and the published artifact is what a recipient receives. Four models republished and each
verified by fetching its card back from the Hub at the new pin rather than reading the local file:
`satellite_comms.telemanom` `22b4cd0e`, `satellite_comms.student` `fb7542b9`, `finance_aml.gbm`
`b35b4cbb`, `finance_aml.student` `475b160d`. One attribution line each.

### Two fact-check findings closed, and both were mis-stated by the sweep that raised them

**The QRNG entropy figure was measuring the extractor.** `simulator.py` passed post-Toeplitz output
to `health_suite` while `raw` sat two lines above. The reported symptom - "a broken source passes" -
is FALSE and re-deriving it is what showed that: `passed` correctly stays True for a merely biased
source either way, because SP 800-90B's continuous tests catch a stuck source, not a skewed one.
The real defect is the estimate. A p(1)=0.90 source has raw min-entropy **0.1472** bit/bit and read
**0.9806** after conditioning, against **0.9892** for a fair source - a badly biased source and a
healthy one a hundredth of a bit apart, because an extractor produces near-uniform output by
construction. That is also why it survived: on the data this project generates the two readings are
indistinguishable, so nothing looked wrong.

**`FiniteKeyParams` still carried the retired intensities**, 0.5 / 0.1 against the shipped
0.3416 / 0.1417 - the sibling of a fix `QKDParams` took in Round 47, unfollowed for four rounds. The
reported downstream consequences were re-derived and **the repository was already right**: the
positive-key threshold measures 3.84e7 pulses at the retired pair and 1.40e7 at the shipped one,
matching what Round 47 recorded. The sweep's "42,448 against 143,645" is a sifted-bit figure, a
different quantity from the pulse counts the docs quote. Nothing downstream needed changing, which is
precisely why re-deriving before correcting is the rule - a plausible correction to a correct number
is still a regression.

### The pattern worth keeping: two gates with tolerances wider than the drift

The refuted-band test constructs `FiniteKeyParams()` bare and asserts the positive-key threshold
falls in 5e6 to 2e8. **Both intensity pairs land inside that band.** So the gate sitting exactly where
the drift happens was green on the defect and green on the fix alike. The replacement asserts the
SOURCE of the default - an identity, with no tolerance to hide in - and checks `default_factory`
rather than the value, so a literal that happens to equal the config today still fails.

**And the first gate written for the QRNG fix had the same shape of hole.** It checked
`"raw=raw" in inspect.getsource(...)`. Seeded by reverting the call site, it stayed GREEN - because
the comment explaining why `raw=raw` is load-bearing sits directly above the call and contains the
string. A gate matching its own explanatory comment passes for the wrong reason. It now parses the
call with `ast` and checks the keyword set; re-seeded with the comment deliberately left in place, it
fires.

### Licence precision, read at source

CC BY 3.0 IGO reaches adaptations, so the satellite `eval_sample.json` - real ESA-ADB channels
resampled to 60s and z-scored - is covered. CDLA-Sharing-1.0 **section 3.5 exempts Results**, so
trained weights alone would owe nothing; the obligation attaches because the finance
`eval_sample.json` ships 8,192 row-level records its own provenance calls "unmodified rows of the
published corpus", which is Enhanced Data under 3.1. Read at cdla.dev rather than summarised, because
calling CDLA an author-credit licence would be the mirror of the original defect.

Also recorded: ESA-ADB's CC BY 3.0 IGO term is on its Zenodo landing page and ABSENT from Zenodo's
REST metadata, so an API-only adjudication reads the corpus as unlicensed. That is how it went
uncredited, and it is the same shape as reading a mirror's silence as the source's silence.

## Rounds 53-54: the four pipelines, and a fix that reached the one language that does not ship

**BUSINESS DOC RECONCILED Round 54.**

Round 52 recorded that only finance had been driven to completion. All five pipelines have now been
run end to end on the deployed site with every published model re-scored in-tab and passing. Four
fact-check findings closed and three left open with reasons.

### E91 confused a visibility with an error probability, in FOUR places

`v_eff = visibility * (1 - e)`. A channel that flips a bit with probability `e` has visibility
`(1 - 2e)`: visibility spans [-1, 1] of correlation, an error probability spans [0, 1/2] of QBER, and
the factor of two between them was missing. QBER understated by **24.8%**, always in the demo's
favour. At the shipped 50 km channel the corrected form gives 0.029744 and independent composition
`q_V + e - 2*q_V*e` gives 0.029744 - agreeing to **0.000%**, exactly rather than to first order,
which is what makes it a correction and not a choice between models.

**Round 53 fixed `qkd/e91.py` and nothing else. This demo is CLIENT-SIDE ONLY, so the copy it fixed
is the one no visitor executes and the two it missed are the ones that render** - ninth instance of
one fix reaching one language, committed by the round that was closing a fact-check about that exact
defect class.

**And the fourth copy was in the checker.** `web/scripts/check_qkd.mjs` asserts the CHSH-Werner
identity as `2*sqrt(2)*V*(1-e)`. It carried the same confusion, so it was ASSERTING the bug, and it
stayed green for as long as every implementation was wrong in the same way. It only went red once the
implementations were right - which is the useful diagnostic and the reason a checker that shares a
derivation with the thing it checks is worth suspecting.

### The cross-language gate had collected the evidence and never looked at it

`tests/test_the_browser_computes_the_bound_the_page_names.py` already exported `e91Qbers` from the
browser and asserted nothing on them; its docstring said "E91 is left alone", which was true of the
secure-fraction FORM and had quietly come to cover the QBER too.

Closing it needed one more decision. The browser SAMPLES its reported `qber` (`e0 + std * gauss(u)`)
while Python returns the analytic value, so comparing them is comparing two different quantities. At
this demo's ~2,500 sifted pairs the shot noise is sigma ~ 0.0034 against a bug of 0.0074 - **a
5-sigma statistical gate would have put the bug at -2.17 sigma and passed it.** So `e91()` now
returns `e0`, the model error rate the bound already uses, and the gate asserts an identity at 1e-9
with no statistical slack. Seeded by restoring the bug in both browser copies: it fires with the exact
figures, 0.022372 against 0.029744.

### Also closed, and two corrections to this project's own record

`decoy_reference.py` asserted its own outputs as the paper's: $`R/\eta \approx 0.11`$ and $`R/\eta^2 \approx 0.15`$ appear
nowhere in Ma et al.'s prose, and the module reached them by passing `q_sift=1.0` while the paper
introduces the figure those curves come from with "where q = 1/2 with standard BB84". Deliberately NOT
resolved - deciding it needs values read off a plotted figure - and the docstring now carries that
status instead of the claim.

`export.opset` does NOT still render 0; that sentence was written in the Round 52 log from a Round 50
note rather than from the page. It renders 17 for `telecom_ids.qml` and is correctly OMITTED for the
two tree ensembles, which carry no default-domain opset. Closed as a side effect of the Round 50
retrain, recorded as such rather than claimed as a fix.

`runs_encrypted=True` is on **six** specs and `model-choice-vs-2026-practice.md` said five - while its
next paragraph says the sixth was being added in that same round, so the document contradicted itself.
Annotated rather than rewritten, because the pre-closure count is the finding.

## Round 55: the reference line could never draw, and the gate checked the mechanism not the outcome

**BUSINESS DOC RECONCILED Round 55.**

Round 54 drew `MetricSpec.baseline` as a reference line on the per-class-F1 chart and gated it: the
value must come from `spec.metric.baseline`, never from `res.baseline` (the trivial predictor). Both
assertions were true. **Measured on the deployed site, the line rendered nowhere.**

The reason is structural rather than a wiring slip. Every model carrying a published baseline plots
as a **PR curve** - finance 0.6323, medical 0.901/0.857, satellite 0.968/0.979 - and every model that
plots **per-class F1** is a telecom multiclass model whose baseline is deliberately `null`. The
intersection of "has a baseline" and "draws that chart" was **empty**, and nothing noticed because the
gate tested the MECHANISM (which field is read) and not the OUTCOME (whether anything appears).

**Moving it to the PR-curve branch would have been worse.** That y-axis is PRECISION, and a
`minority_class_f1` of 0.6323 is not a precision value. A horizontal line there compares an F1 against
a precision - the same category error as drawing the trivial predictor as prior research, which the
gate for that field exists to prevent. The fix was not to relocate the line but to stop drawing a
line at all.

`MeasuredAgainstPublished` gives the comparison its own axis, where both bars are the same quantity by
construction and the metric is named on the figure. **Seven models render it**: finance gbm, three
medical, and three satellite. The gate now asserts REACHABILITY - at least one shipped model must have
both a published baseline and a measured value under the same declared metric - so a comparison
nobody can see fails rather than passes.

### The calibration diagram was verified on the deploy, and it works

Round 51 added it and it had never been seen by a visitor. Confirmed rendering on both branches:
"Calibration of `finance_aml.gbm`: mean p(positive) against the frequency..." and "Calibration of
`telecom_ids.qml`: mean winning-class confidence against the frequency...". Binary bins p(positive),
multi-class bins winning-class confidence, and the forecast branch has none because a regression has
no probability to calibrate.

Also settled by reading rather than building: the per-row prediction-vs-label scatter is not missing
work. It was considered and declined, with the reason recorded in the type - "a prediction-versus-label
scatter for a classifier is a confusion matrix drawn badly, the y-axis is categorical". The reliability
diagram is the instrument that answers the same question honestly.

### A range heading loses its upper bound

`## Rounds 53-54` parses as **53** under `^##+ Rounds? (\d+)`, which both round-readers use. With
Round 54 merged, the register read 53 against a git history of 54 and
`test_the_register_records_every_round_that_shipped` went red - correctly, and for a reason that has
nothing to do with the round's content. Recorded because the range form is used elsewhere in this file
and will do it again.


## Round 56: the standing requirements, each with its evidence and its gap

The standing request asks for a list of what is done, what is pending, and what is a concern. This is
that list, built from measurement rather than from recollection. Row counts are from the gitignored
checklist; every "verified" line names how it was checked.

### Covered, with the evidence

| requirement | covering rows | evidence |
|---|---|---|
| The workflow loop: local -> push -> CI -> box pull -> Docker rebuild -> public URL -> repeat | **`RT0`-`RT9`**, an ORDERED loop; `RT9` is the branch back | `RT1` local gate before push, `RT3` CI green on the PUSHED head, `RT5` image REBUILT not reused, `RT6` bundle filename changed, `RT7` public URL with cache-bust |
| Client-side-only inference | 17 `CS` rows | `CS1` zero `/api/` requests, `CS2` `/api/*` returns 404 `text/plain` (measured), `CS4` `crossOriginIsolated`, `CS12` zero POSTs on the wire, `CS15` npm crypto deps checked for archival |
| 2D plots per vertical | 21 `RD` rows | all five payloads served, 8192/569/307/3200/529 points, each with licence, attribution, provenance and named axes |
| QCNN | 36 `QC`/`QR` rows | cross-language parity to 2.4e-07; `QC18` carries the WDBC literature comparison |
| Published HF models | 32 `HF`/`PM`/`MS` rows | `HF15` added this round: the DEPLOYED bundle must carry the new pins |
| Page content, terminology, numbers, buttons, duplicates | 43 `RG`/`SW`/`UX`/`MN` rows | ten pages walked; zero bad tokens, zero duplicate headings, zero unnamed buttons, zero CJK |
| Dataset licence and commercial use | 15 `L`/`DL`/`CP`/`T` rows | 7 tracked reference PDFs each with an explicit redistribution grant, adjudicated per file at primary source; 4 excluded and gitignored; gate derives its corpus from `git ls-files` |

### Open, with what is actually missing

| item | state | what would close it |
|---|---|---|
| **Mobile layout** | **UNMEASURED.** The automation here resizes the OS window without moving the viewport - `innerWidth` stays 1280, `outerWidth` reads 0 | CDP device emulation, then assert zero horizontal overflow and WCAG 2.2 AA 2.5.8 tap targets at 24x24 CSS px. `UX12` |
| **WebGPU node placement** | **UNMEASURED.** The badge reports an accepted adapter, not per-node assignment; ORT 1.27.0's op registry is not enumerable from the shipped bundle | `endProfiling()` or `logSeverityLevel: 1`, plus a THIRD badge state - two cannot express a partial answer. `CS17` |
| **Telecom's separability estimator** | **UNDETERMINED and recorded as such.** Its four readings are 0.9798 / 0.9915 / 0.9925 / 1.0000; every plausible estimator lands inside any usable band | Nothing can fix this by tolerance. Either state the estimator in the payload or accept it. `RD22` skips with the reason |
| **QSDC finite-size certification** | Working as designed and the design is now honest: at the demo's 74-byte telemetry no block can be certified | Nothing to fix. `QS13` records the crossovers - 32 KB on fibre, 4 MB on the downlink |
| **JupyterLite / marimo / pyvista-wasm** | DECLINED with reopening conditions, re-verified 2026-08-22 against the Pyodide 314.0.5 lockfile (356 packages, no CKKS/MPC/ONNX) | `DC8`; the `pyvista-wasm` "does not exist" reasoning was WITHDRAWN this round - it exists and is by a PyVista core maintainer |

### The concern worth stating above the others

Every defect this round was **one copy of several**: a formula corrected in one of two call sites in
one file, a lesson applied to one of two primitives in one gate, a licence downgrade reaching three
of four surfaces, a health fix reaching one of two languages, a caption written for one payload and
inherited by two more. **Three of those were fixes I made in this same session**, which is the part
that matters: the defect is not carelessness about the first copy, it is that nothing enumerates the
copies. The gates added this round enumerate rather than list - note fields from the config, row IDs
from the file, combiner branches from the exporter - because a hand-written list is maintained by the
same person who forgot the second copy.

BUSINESS DOC RECONCILED Round 56. The out-of-repo document was at R16-R18 and is now brought level
with what has been measured since. Its content is not quoted here, in the checklist, or in any commit
- only the fact of the reconciliation, which is what gate 16 reads.

## Round 58: two comparisons that measured different things, and public text that claimed more than the demo does

An independent check of the public text (2026-10-06) listed ten issues and changed nothing. A survey
of all ten routes of the deployed demo on 2026-10-23 confirmed them and found more. Every item below
was re-read at its primary source before it was changed; copies of the sources were kept for the
session. Nothing was changed on the strength of a secondary write-up.

### Retired from every surface

| what | why it was not a measurement | where it reached a visitor |
|---|---|---|
| Overview card "~1.32 Rate-1 QFHE ciphertext/plaintext ratio vs ~1483x for classical CKKS", and the `qsmpc_ct_pt_ratio` constant behind it | 1.32 divided bytes of a CLASSICALLY SIMULATED $2^n$-amplitude state vector; a quantum ciphertext has no size in bytes. It is not in the paper it was attributed to: Chardouvelis, Doettling and Malavolta (TCC 2021) report no concrete figure, and their "rate 1" is an asymptotic limit under quantum LWE. 1483 was CKKS bytes over about 34 inputs, i.e. slot utilisation (about 12x at full packing; Cheon, Kim, Kim and Song, ASIACRYPT 2017). Classical FHE reaches rate 1 as well (Gentry and Halevi, TCC 2019; Brakerski, Doettling, Garg and Malavolta, TCC 2019) | the landing page |
| The 0-10 "quantum advantage" score, its tier, interpretation and log line | it divided the same mismatched byte and qubit ratios, divided a real CKKS plus garbled-circuit latency by an 8-qubit simulation, and multiplied by 1.5 for a security label the run does not achieve. Three runs printed 5794x, 8163x and 9610x "faster" | /orchestration log, every inference page's panel and log, a diagram label, the in-browser agent's prompt context, /usecases, README, and the Python stage metric. The Python module is kept and called from nowhere, as the composite was in Round 22 |
| The QSMPC-FHE row's ct/pt cell | the last place the two ratios sat side by side | the benchmark table, now a dash |

### Corrected

| item | was | is | source |
|---|---|---|---|
| E91 security facts | the QBER row said "two-intensity decoy-state bound ... aborts above 0.11" on an E91 run, because the protocol test was declared inside a branch the row could not see | Werner-model error rate; device-dependent rate reaches zero at 0.0981 (f = 1.16); a device-independent analysis would need S > 2.424, i.e. Q < 0.0715 | Acin et al., Phys. Rev. Lett. 98, 230501 (2007); Ma, Fung and Lo, Phys. Rev. A 76, 012307 (2007) |
| E91 model error rate | not exposed, so the relation S = 2 sqrt(2) (1 - 2 e0) could only be checked against the 2-dp SAMPLED QBER, from which two of three reported rows do not reproduce | `e91_model_qber` at both E91 sites (fibre and satellite) | the simulator's own model |
| QKD rows of the security-posture table | "measured", under "what this run measured", with "The bits are real" | kind "simulated"; the browser's `bb84()` returns a count, never key bytes (checklist SC2) | code |
| Demo banner | silent about QKD; the simulated label sat at the bottom of two pages and in hover text | every page says the quantum channel (QKD, QSDC) is simulated | code |
| "information-theoretic" | unqualified | "as a scheme" (quantum one-time pad, given a uniform pad); the demo seeds the pad, so no run here is information-theoretically secure | Ambainis, Mosca, Tapp and de Wolf, FOCS 2000 |
| Agent prompt on SPA navigation | `useState(defaultPrompt)` read its prop once: Finance then Medical in the nav kept the AML question | reset on use-case change | reproduced by a nav click, not by a fresh URL |
| Validations "mix QKD key" | described a fixed 32-byte constant as a QKD key | "mix fixed QKD-domain constant" | `clientMcp.ts`, `mcp_demo` |
| Novelty sentence, three copies | "then QKD-derived key material" | "then a simulated QKD link (it yields a key length, not key bytes)", identical in README, Value and the technical report | code |
| Market-size figures | README top, a Value-page table and notes, a line on every use-case page, two quotes in the verification records | removed; the project's public material carries none | policy |
| Overview subtitle | "validated for finance, healthcare, genomics and telecom/defence" | "demonstrated on ... and satellite use cases, with the quantum channel simulated" | the page's own "simulated, not hardware-validated" note |
| BSI attribution | QKD as a hybrid add-on "per BSI", "on BSI's authority" | the add-on is this project's choice. TR-02102-1 v2026-01 section 2.1 (p.29) does not recommend QKD; section 2.2 recommends a quantum-safe plus CLASSICAL hybrid and does not mention QKD | BSI TR-02102-1 v2026-01 |
| NSA | a "2027 procurement mandate"; "QKD non-recommended standalone"; a quote attributed to the FAQ | national security systems only; CNSSP 15: new acquisitions 2027-01-01, phase-out 2030-12-31, all use 2031-12-31; the FAQ answers "No" to QKD for NSS; "does not support the usage of QKD or QC" is from the Post-Quantum Cybersecurity Resources page | CNSA 2.0 and Quantum Computing FAQ Ver. 2.2 (October 2026); CNSSP 15 |
| CNSA 2.0 eligibility | "position bids for CNSA 2.0 / FIPS-aligned procurement"; a sidebar badge; "be compliant before bids" | the defaults are not CNSA 2.0 parameter sets (ML-KEM-1024 and ML-DSA-87 are selectable); nothing here is FIPS 140-3 or NIAP validated | FAQ Ver. 2.2 |
| Cloudflare / Google 2029 | "Cloudflare commits"; Google's the "weaker" commitment | both are targets; neither post ranks itself against the other | Cloudflare blog 2026-04-07; Google blog 2026-03-25 |
| EO 14412 | high-value assets only; "a FAR rule putting contractors on the 2030 date" | high-value assets AND high impact systems (section 4(b)); section 6(c) directs a PROPOSED FAR rule within 180 days, none in the Federal Register as of 2026-10-23 | 91 FR 38483 |
| NIST IR 8547 (report) | "deprecates RSA-2048 and ECC-256 by 2030" | initial public draft; 112-bit strength deprecated after 2030, P-256 only disallowed after 2035 | NIST IR 8547 ipd, Table 2 |
| FIPS 140-2, M-26-15 | future tense | past: certificates historical since 2026-09-21; agency plans were due about 2026-10-22 | CMVP; M-26-15 |
| QCNN simulability citation | arXiv preprint | PRX Quantum 7, 020304 (2026) | arXiv journal reference |
| E91 decoy-state row (found on the deployed branch) | an E91 run showed "Decoy-state necessity" and its warning that "the decoy states are what make this link work at all" - and the gate written earlier this round ASSERTED four QKD rows for both protocols, so it codified the defect | BB84 only; the E91 test now checks the whole table, not three selected rows | Ma, Fung and Lo, Phys. Rev. A 76, 012307 (2007): an entangled source needs no decoy states |
| Agent prompts (found on the deployed branch) | medical asked why HIPAA needs a "(PQC + QKD)" channel; satellite asked why QSDC beats classical encryption on a link where this page finds QSDC cannot close | neutral questions | HIPAA names no PQC or QKD requirement; the page's own QSDC finding |
| Three missing spaces (found on the deployed branch) | "foundation -demonstrated", "recommendation:BSI", "thequantum_comparison": a JSX line break before an inline tag or a comment drops the space | `{" "}` at each | rendered text |
| References | arXiv:2604.23490 called nothing; arXiv:2604.19256 had its v1 title | 2604.23490 v3 carries the authors' comment that its conclusion contains mistakes (NOT withdrawn); 2604.19256 has its v2 title | arXiv API |

### Proposals: four adopted in the same PR, two still open

The first four were recorded as proposals and then adopted by the user's decision on 2026-10-24, in
this same PR:

| item | adopted as |
|---|---|
| Eve's error was added linearly to the channel error | composed as independent errors, a + b - 2ab, at all four sites (`intrinsic_qber` / `e_det_for_bound` and their browser mirrors) through one helper per language. Eve-free figures unchanged; E91 at 50 km with a full attack moves from S 1.29 / e0 0.2722 to S 1.33 / e0 0.2649 (displayed QBER 0.27 either way). The relation S = 2 sqrt(2)(1 - 2Q) still holds exactly |
| The S > 2 gate never binds | stated in the caption and the code; the device-independent threshold is now computed from the configured f: S > 2.456, Q < 0.0658 at f = 1.16 (the caption had compared an f = 1 threshold, S > 2.424, with an f = 1.16 zero-key point) |
| "Werner state" for Eve > 0 | the caption names the state a mixture of Werner and intercept-resend pairs, separable once V'(3 - 2f) <= 1 |
| `/orchestration` log lines | "simulated" in the QKD and QSDC log lines, on the graph's quantum nodes and in the diagram legend |

Still open:

| item | why it is a judgement | proposal |
|---|---|---|
| EU AI Act date on the medical line | it said Art. 13 applies from 2 Dec 2027 (Annex III); a diagnostic model is more likely an Annex I product (MDR), 2 Aug 2028 | removed with the market line; UNVERIFIED - needs confirmation which annex applies before restating it anywhere |
| Honest perfectly-secure QHE expansion | about 15x per Ortega and Martin-Delgado | UNVERIFIED - needs confirmation against the paper before it is quoted |

### The concern worth stating above the others

The defect class is the same as every recent round: one copy of several. The novelty sentence had a
third copy, in the technical report, that the plan did not name. The market figures sat in five places,
two of them in the verification records. The E91 protocol test existed and was scoped one branch too
narrowly for the row that needed it - and after the fix, the deployed page showed a fourth BB84-only
row on E91 runs, which a gate I wrote the same day had asserted should be there: a test that selects
the rows it checks cannot see the row it did not select. The new gates enumerate the copies instead of naming one:
`test_public_text_claims_only_what_the_demo_does.py` scans every public text file for market figures,
the three novelty copies and twelve corrected phrases; `test_e91_facts_describe_the_e91_model.py` runs
the real browser module for both protocols and both key-count bands; every check was seeded and fires.

### The readability layer, and what it surfaced (same PR)

The user asked for every page to lead with its key points and fold the rest, with diagrams and 2D/3D
models where prose stood. Doing it surfaced more corrections than it made layout changes:

| page or component | change | what the change exposed |
|---|---|---|
| Overview | an architecture diagram of the stages as they run, each badged real / notional parties / simulated / seeded pad; workstreams and crypto-agility as tables; the Research PoC note moved up unchanged | the pill row drew QRNG as its own stage (it runs inside the PQC handshake), coloured real CKKS as a scaffold and put QKD on the key path; the agility paragraph listed the PQC+QKD combiner and TenSEAL/OpenFHE as demo run options (neither is) |
| Value | the novelty sentence, verbatim, directly under the heading (it was about 1,300 px down; measured at 248-440 px in a 1280x621 window after); a regulatory clock - timeline by jurisdiction plus a table - replacing a 191-word paragraph | every date now carries who it binds and how firm it is; the closing "preparation, not a present-day mandate" sentence stays visible |
| Validations | the real PQC hybrid handshake moved out of a "Quantum simulations" heading; QBER and secure bits against the eavesdropper fraction as a chart | the subtitle dropped "simulated" in demo mode |
| Use-case pages | model curves: BB84 rate with and without decoy states and E91's CHSH value against fibre length (telecom); QSDC secrecy capacity against downlink range (satellite) | without decoy states the bound is zero at EVERY distance shown, so the decoy states carry the whole link; QSDC closes only with the fixed losses near zero (out to about 129 km at 0 + 0 dB), never at a LEO slant range |
| Benchmark, agreement chart, SDN panel, model panel | tables and a stat strip in place of paragraphs; long generated registry prose shows its first sentence with the rest folded at render time (it cannot be edited: CI regenerates it) | four wrong references: "Press Run" for a Start button, "table above", "model panel below", "benchmark below" |
| 3D (three.js, loaded only where drawn) | the QCNN's per-qubit reduced Bloch vectors through the real circuit (a snapshot hook leaves the measured path unchanged); the one-time pad and the downlink geometry as illustrations | the two illustrations are captioned as adding no information: the pad's states lie on one circle, and the channel model has no altitude or elevation |
| Browser benchmark row | security model "computational (RLWE) + garbled-circuit 2PC" | it said "secret-sharing MPC" - the Python row's mechanism - on the row of a browser that runs a garbled-circuit 2PC |
| Satellite tooltips and log | QSDC figures computed at render time by the function the stage runs (`qsdcLink`) | four quoted figures dated from before two formula corrections; at 1 km and 2 + 1 dB the tooltip promised a delivery the current model refuses even with no eavesdropper; the NOT AVAILABLE log quoted the superseded pooled criterion |
| Agent prompts | neutral questions on the medical and satellite pages | they asked the model why HIPAA needs PQC + QKD (it names neither) and why QSDC beats classical encryption on a link the page finds it cannot close over |

Gates added for this part: the Eve composition at all four sites with browser parity, the DI threshold
recomputed independently, no QSDC literal in a tooltip, the decoy row absent on E91, and the existing
accessibility, wide-content and dead-component gates extended to the new components by construction.

### Network-layer key schedules, computed in the tab (same PR)

The user asked for the useful parts of the companion network project (WireGuard, IPsec) to be added
and verified. A browser has no sockets and no kernel tunnel, so what can be added is the key-schedule
MATH, run for real between two in-tab peers, and each piece is held to an official vector where one
exists:

| computation | validated against | computed per spec only |
|---|---|---|
| ML-KEM-768 known-answer test | NIST ACVP-Server (commit 975de31): keyGen 3, encapsulation 3, decapsulation 4 (two valid, two modified ciphertexts) - 10/10 in the browser | - |
| WireGuard handshake (Noise_IKpsk2), PSK from an in-tab ML-KEM-768 + X25519 hybrid through the session key's combiner | the Noise core: cacophony vector (commit 8ee9d41, public domain) - both messages, four transport messages, handshake hash | WireGuard framing (148 / 92 bytes, mac1, TAI64N, indices) per the whitepaper - no authoritative implementation publishes a fixed transcript; two initial-state constants cross-checked against boringtun, labelled non-authoritative |
| IKEv2 key schedule | RFC 7296 prf / prf+ / SKEYSEED / key stream: NIST CAVP SP 800-135 (6 cases); ECP-256 shared secret: RFC 5903 section 8.1 | the RFC 9370 ML-KEM-768 intermediate exchange and the RFC 8784 PPK layer - no official vectors; ML-KEM in IKEv2 is draft-ietf-ipsecme-ikev2-mlkem-09, in the RFC Editor queue (IANA method 36) |

Ten of twelve seeded defects were caught by the check; the two that pass are composition errors in the
RFC 9370 and RFC 8784 layers, which nothing can catch without vectors - stated, not hidden. Sources,
commits and hashes: `references/net-key-schedules.md`; licences: `DATA_LICENSES.md` section 6b. No
performance figure was imported: the companion repository commits none. A positioning diagram shows
the two layers and links the companion demo with its caveat that its QKD is simulated too.

### Walked on the deployed branch (ab105a7)

All ten routes and both not-found paths were walked in a browser on the public demo before the PR:
the architecture diagram, the novelty sentence at 248 px, the regulatory clock (18 rows), the use-case
table, the E91 scenarios on `/orchestration` (0.03 / 2.66 / 0.007286 and e0 0.0297; Eve 1: 0.27 / 1.33 /
0 and e0 0.2649; 0.02 / 2.74 / 0.09037 and e0 0.015), the three key-schedule cards (ACVP 10/10, the
WireGuard checks, CAVP 7/7 and RFC 5903) with their chunks fetched only on click, the posture table and
3D view on an inference page, the E91 caption under an eavesdropper, and the computed QSDC tooltips
(+0.1150, -64.7390). At a 390 px emulated phone viewport every route measured zero horizontal page
overflow. One defect was found and fixed in this PR: the diagram's QSDC label paired the 24.5 dB
channel loss with a g that includes the detector's 4 dB (g = 701 is 28.5 dB), while the log printed the
same g at 28.5 dB; both now say which loss they name.

BUSINESS DOC RECONCILED Round 58. The out-of-repo document is brought level with this round. Its content
is not quoted here, in the checklist, or in any commit - only the fact of the reconciliation, which is
what gate 16 reads.

## Round 57: a docstring that beat a baseline it did not, and a gate that tells a withdrawal from a claim

Started before Round 58 and finished after it, on its own branch. Two source files - the distiller's
docstring and the OPS-SAT trainer's comment - carried one identical sentence: the CKKS student's
minority-class F1 and AUC-PR "beat both the teacher (0.9189) and the published baseline (0.979)".
Against `models/metrics.json`:

| quantity | claimed | recorded |
|---|---|---|
| student auc_pr | 0.9810 | 0.968055 - below the published 0.979 |
| student minority-class F1 | 0.9391 | 0.904348 (`student_metric`, metric_name minority_class_f1) |
| teacher | "0.9189" | that is the teacher's F1, read against the student's AUC-PR; the teacher's auc_pr is 0.968107 |
| degree-1 "before" figures | 0.8485 / 0.9235 | recorded in no committed artefact |

Both copies are corrected together, and say what the record does support: at 209 expanded features
(210 parameters with the bias) and one encrypted dot product, the student TRACKS its teacher on the
declared metric - 0.968055 against 0.968107 - which is a fidelity-at-low-cost result, not a win.

**The gate.** `tests/test_prose_does_not_beat_a_baseline_the_record_does_not.py` fails when a source file
says a model beats, passes, exceeds or outperforms a published baseline (named within three lines) that
the record puts it below. A correction must quote what it withdraws, so the corrected text contains the
sentence the gate looks for; the first version told the two apart by counting double quotes back to the
start of the comment block, which misfired on the very correction it protected - Python source is full
of quote characters. It is now an explicit convention: a retracted claim is quoted between
`WITHDRAWN-CLAIM-BEGIN` and `WITHDRAWN-CLAIM-END`, those spans are blanked before the scan, and nothing
else is exempt. The markers must balance and each span must hold a paired quotation, so the exemption
cannot become a loophole. Seeded four ways - an unmarked boast, a dangling BEGIN, a span with its
opening quote removed, and a quoted but unmarked boast - and all four fire.

The commit that started this branch was authored with the wrong identity and is re-authored here, as
this repository's commits now are, with the account's noreply address.

## Round 59: dependency advisories, which of them reach the public demo, and an agent path nobody had timed

`npm audit` on main flagged 9 production packages, and 16 with the development tooling included; it
counts packages, and several share one advisory (react-router-dom is flagged for react-router's,
mpc-framework for zod's). This round updates what can be updated inside the declared ranges, assesses the rest
against what the public demo actually ships, and gates the one assessment that depends on application
code rather than on packaging.

| packages flagged | main (e2508b3) | this branch |
|---|---|---|
| production only (`npm audit --omit=dev`) | 9: 3 moderate, 6 high | 4: 2 moderate, 2 high |
| including development tooling | 16: 5 moderate, 11 high | 6: 3 moderate, 3 high |

**What moved.** `npm audit fix` without `--force`, so no declared range was widened: 51 lockfile
versions changed, 2 packages were added and 7 removed. The ones that matter are
`@huggingface/transformers` 4.2.0 to 4.3.1, which brings its own `onnxruntime-web` (1.26.0-dev to
1.31.0-dev), `onnxruntime-node` (1.24.3 to 1.30.0, and with it `adm-zip` 0.5.17 to 0.6.1) and `sharp`
(0.34.5 to 0.35.5); `protobufjs` 7.6.4 to 7.6.6, which both copies of `onnxruntime-web` use;
`react-router` and `react-router-dom` 6.30.4 to 6.30.6; and, development-only, `postcss`, `browserslist`, `nanoid`, `source-map-js` and
`baseline-browser-mapping`. The `react-router-dom` range now starts at the patched release (`~6.30.6`,
was `^6.27.0`), because 6.30.2 to 6.30.5 carry GHSA-jjmj-jmhj-qwj2 and main resolved to 6.30.4. That
advisory needs an application with an open redirect, which this one does not have (see the gate below),
so the public demo was not exposed through it; the floor is raised anyway so that a fresh resolve cannot
land in that range.

**What remains, and whether the public demo reaches it.**

| package | advisory | fixed in | reaches the public demo? |
|---|---|---|---|
| react-router, react-router-dom (moderate) | GHSA-wrjc-x8rr-h8h6: open redirect through a backslash in a `<Link>` or `useNavigate` target built from attacker input | react-router 7.18.0, a major upgrade | No. Nothing calls `useNavigate`, and every link target is a literal or comes from the static NAV and USECASES lists; the one query parameter the app reads, `?usecase=`, is checked against USECASES and never becomes a link. Held by the gate below. |
| react-router (moderate) | GHSA-337j-9hxr-rhxg: constructor injection through `deserializeErrors()` in SSR hydration | 7.18.0 | No. The advisory says it does not affect Declarative Mode; the app renders one `<BrowserRouter>` and does no server rendering or hydration. |
| zod, through mpc-framework (high) | GHSA-mrhx-c7xc-m3w5: memory exhaustion while validating an oversized array | zod 4.6.6; no mpc-framework release | No. mpc-framework imports zod only in its PlaintextEngine and errorToString helpers, and none of the built chunks carries zod code. mpc-framework is archived upstream (threat model, Round 55 caveat), so no fix will come from there. |
| vite (8 advisories, 2 high) and esbuild (moderate) | dev-server issues: `server.fs` bypasses, a `.map` path traversal in optimized deps, launch-editor command handling (mostly Windows), esbuild serve-mode CORS | vite 6.4.4, a major upgrade | No. The web image builds with Node and serves the static bundle from `caddy:2-alpine`; no Vite or esbuild process runs in production. `npm run preview` binds every interface, but the preview server carries none of the dev-server code paths these advisories name. `npm run dev` binds localhost only, and the stated conditions (network exposure with `--host`, Windows, middleware mode, or a Vim or Emacs editor plus an attacker-created file name starting with a plus or minus sign) are not this project's setup. That makes them a developer-machine risk, not a zero one. |

**The gate.** `tests/test_navigation_targets_are_not_built_from_input.py` holds the router assessment,
because it is a property of the code and code changes: no `useNavigate` or `redirect(` anywhere in
`web/src`; every `to={...}` expression is on a two-entry allowlist, each entry from a static list; the
`?usecase=` parameter stays validated and never feeds a link; and an anti-tautology check fails if the
scan stops seeing the targets it allows. The value of `to={...}` is read with balanced braces, because a
first regex stopped at the `}` of a template literal's `${}` and then swallowed the next prop. Seeded
three ways - a `useNavigate` import, a `<Link to={searchParams.get(...)}>`, and the NAV targets replaced
by a literal - and each fires the check it should; the restored tree passes.

**The upgraded inference library, timed against the one it replaces.** Of everything that moved, the
inference library is the change a visitor would feel. Measured on one machine (an NVIDIA RTX PRO 6000
Blackwell adapter, Chrome 152 on Linux), both builds served from the same local origin so they share
one weights cache, greedy decoding of exactly 32 tokens, run order A-B-A:

| build | WebGPU, fp32 weights | WASM, q8 weights |
|---|---|---|
| 4.2.0 (main) | 41.3 s and 36.3 s; 1136 ms per token | 0.44 s and 0.33 s |
| 4.3.1 (this branch) | 30.1 s and 23.9 s, then 33.2 s and 25.4 s; 748 and 795 ms per token | 0.36 s and 0.28 s; 96 tokens in 0.87 s |

Both versions emit the same tokens. In the app itself, on the branch build, the agent answered on WebGPU
in 74.4 s and 18.2 s on two runs with the weights cached (sampling, so the lengths differ), and on every
run the quantum-safe MCP envelope authenticated and the tampered copy was rejected. The upgrade is not a
regression on either backend.

**A defect the timing exposed, recorded and not fixed here.** On this machine the WebGPU path, which the
agent prefers whenever a GPU adapter exists, is about 80 times slower per token than WASM, and no first
run finished inside the panel's 120 s limit. Locally the 313 MiB of fp32 weights it fetches
(`onnx/model.onnx`) took 277 s on their own, and main's build timed out the same way; on the public
demo the same file arrived in 19.6 s and the first run timed out anyway, because generation itself runs
at 0.75 to 1.1 s per token for up to 96 tokens. Warm runs took 18 to 117 s, depending on the sampled
length. The model's label in
`src/qsh/config/defaults.py` reads "tiny, ~80 MB", a figure that matches the old
`decoder_model_merged_quantized.onnx`, which neither version loads; WASM loads the 226 MiB
`model_quantized.onnx`. The panel's second Transformers.js model is further off: by the Hub's file
listing, `onnx-community/Qwen2.5-0.5B-Instruct` would fetch a 1901 MiB `model.onnx` on WebGPU and a
488 MiB `model_quantized.onnx` on WASM (sizes from the listing, not timed here). Only the WASM path had
been exercised before (the verification log records transformers-wasm at 2430 ms). Both defects are on main and on the public demo today. The fix changes
which backend and weights the panel uses and the text that describes them, so it is left for its own
change. UNVERIFIED - needs confirmation: whether WebGPU is this slow on other GPU and OS combinations.

**Walked on the public demo with this branch deployed** (57f42b2, serving `index-B6Uw15BG.js`, the
file name the local build produced). All ten pages render their headings with no error boundary and
zero horizontal overflow at desktop width. Links clicked inside the app keep the same document, so
client-side routing works on 6.30.6; an unknown path renders the not-found page with only the static
targets; `/orchestration?usecase=nonsense_value` shows its fallback notice and no link carries the
value. The medical pipeline ran all eight stages and signed its summary (ML-DSA-65, verified true). The
agent timed out on its first run and answered on the second, on WebGPU, in 85.0 s; the envelope
authenticated and rejected the tampered copy. Two observations on the same WebGPU path, neither caused
by this change: the published medical ResNet-18's `session.run` alone took 18216 ms on `onnx-webgpu`
(the README records a gap of the same order, 12842 ms against 42 ms on WASM, and attributes it to a
software adapter; this adapter is hardware), and the inference tier's ONNX Runtime (1.27.0, not moved
here) logs at error level that some nodes were not assigned to the WebGPU provider.

BUSINESS DOC RECONCILED Round 59. The out-of-repo document is brought level with this round. Its content
is not quoted here, in the checklist, or in any commit - only the fact of the reconciliation, which is
what gate 16 reads.

## Round 60: the small models run on WASM unless the visitor selects WebGPU

**Round 59 closed first.** PR #55 merged as fdcd939, the demo was redeployed from main and serves the
bundle the branch walk saw (`index-B6Uw15BG.js`), and `npm audit --omit=dev` on merged main reports
the branch's 4 (checklist NS11).

**What changed.** Round 59 found that both in-browser model tiers took WebGPU whenever a hardware
adapter existed, and that on the machine measured that path was the slow one by one to two orders of
magnitude. The fix chosen: WASM by default, WebGPU one selection away, and sizes that match the
download.

- Defaults in the single source of truth: `WebConfig.tfjs_default_device` and
  `WebConfig.onnx_default_backend`, both `wasm`, exported to `config.defaults.json`.
- The agent: a Backend select (WASM by default; WebGPU disabled without a usable adapter).
  `runTransformersJs` decides the device from the request first and the adapter check second, and a
  WebGPU request on a machine without an adapter runs WASM and says so in the result.
- The ONNX tier: `loadModel` takes the requested backend, its session cache is keyed on model and
  requested backend, and `OnnxStatus.requested` records what was asked beside `backend`, what was
  granted. The pipeline sends `onnx_backend` only when WebGPU is chosen (a Model backend select beside
  Published model); the single-row run and the whole-sample verification both use it, and the log says
  when WebGPU was selected but no usable adapter existed.
- Sizes: each Transformers.js model is pinned to a commit and records the dtype and the bytes of the
  weights file per backend, from the Hub's file listing at that commit. The loader passes `dtype` and
  `revision`, so the size the picker computes describes the file that is fetched: distilGPT-2 is 237 MB
  of q8 weights on WASM and 328 MB of fp32 on WebGPU, Qwen2.5-0.5B 512 MB and 1994 MB.
- The agent's 120 s limit now measures a stall, not the total. A first run reports its download
  ("downloading onnx/model_quantized.onnx 49% of 237 MB"), each event resets the limit, and a download
  that is still arriving is no longer reported as a timeout.
- Public text: three README passages, `docs/architecture.md`, the panel note that called WASM "the
  slower" backend, and the timeout note that advised "a WebGPU-capable browser". Both phrases are
  retired in `test_public_text_claims_only_what_the_demo_does.py`.

**Measured locally, on the machine Round 59 measured** (Chrome 152, NVIDIA RTX PRO 6000):

| path | before (Round 59) | now |
|---|---|---|
| agent, first run | timed out at 120 s, on WebGPU | 13.7 s on WASM, of which 10.8 s was the 237 MB download |
| agent, warm | 18 to 117 s, on WebGPU | 1.6 s, on WASM |
| agent, WebGPU selected | - | 99.9 s including the 328 MB download; answered, envelope verified |
| medical ResNet-18, one `session.run` | 18216 ms on onnx-webgpu (public demo) | 37.43 ms on onnx-wasm; 7649.87 ms with WebGPU selected |
| medical pipeline, all stages | 63.6 s (public demo) | 16 s; all three published models re-verified PASS on 156 rows, on both backends |

**The gate.** `tests/test_small_models_default_to_wasm.py`: WASM is the default in both configs;
neither tier derives its backend from the adapter's existence alone; every Transformers.js model is
pinned and records dtype and bytes for both backends, and its label carries no size; the loader passes
`dtype` and `revision`; the picker computes its size from the bytes; and the agent's limit measures a
stall. Seeded nine ways - the default set to webgpu, a typed size in a label, the revision removed, the
device decided by the adapter alone in each tier, `dtype` not passed, the panel showing the typed
label, a total-time limit, and no progress reported - and each fires the check it should.

**What this does not establish.** One machine. UNVERIFIED - needs confirmation: whether WebGPU is this
slow on other GPU, driver and OS combinations; and how much of the ResNet's WebGPU time is ONNX
Runtime's per-node fallback for the int8 operators (the console warns that some nodes were not
assigned to the WebGPU provider). The runtime's profiler would answer the second, and was not run.

**The archived garbled-circuit tier: what could replace it (researched, nothing changed).**
`mpc-framework`, `emp-wasm-engine` and `summon-ts` are archived, and `mpc-framework` pulls the zod that
carries GHSA-mrhx-c7xc-m3w5. Checked at the GitHub API on 2026-10-24:

| candidate | licence | state | protocol |
|---|---|---|---|
| `sine-fdn/polytune` with `sine-fdn/garble-lang` | MIT | not archived; last commit 2026-09-30; latest release v0.2.0-alpha.4 (2025-12-29), a pre-release | WRK17b multi-party authenticated garbling, "secure against malicious adversaries that can control up to all-but-one of the parties" (its README) |
| `tlsnotary/mpz2` (formerly `privacy-scaling-explorations/mpz`) | GitHub detects no licence file | not archived; pushed 2026-10-16; no release | half-gates, semi-honest |

Polytune is the closer fit: the same protocol family as the tier today, maintained, and its browser
example runs every party from one page (through an HTTP broker; an in-memory channel would have to be
written). The cost is a Rust and wasm-pack step in the web build and an alpha API. mpz would mean a
downgrade to semi-honest security and a threat-model change. Nothing was built or run. UNVERIFIED -
needs confirmation: that two Polytune parties complete over an in-memory channel on WASM's single
thread; what the argmax costs in time and bytes; whether the Summon-built circuit can be expressed in
Garble; and the state of the other candidates looked at (swanky, emp-ag2pc, webSPDZ, MPyC-Web, JIFF),
none of which was confirmed here to run both parties in one tab without a server.

BUSINESS DOC RECONCILED Round 60. The out-of-repo document is brought level with this round. Its content
is not quoted here, in the checklist, or in any commit - only the fact of the reconciliation, which is
what gate 16 reads.
