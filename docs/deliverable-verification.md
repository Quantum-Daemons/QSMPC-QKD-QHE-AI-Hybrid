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
over GF(d) with d prime and n <= d <= 2n; the shipped config pinned d = 4099 for n = 7, which
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
| A fidelity quantum kernel beating a random forest at 8 qubits or fewer | concentration bound says >= 15 qubits |
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
them: OptiQKD's `+14-25%` is correctly attributed to AML-QKD (arXiv:2603.04192) with the v1→v2
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

**The currency risk was measured, not assumed.** `README.md:43` and `docs/verification-log.md:531`
use `$` for dollar amounts. GitHub's server-rendered HTML for `main` contains both as literal text
with zero math markers, against positive controls that do show them - so those lines are safe as
*observed output*, not as derived from a rule. The widely repeated tokenizer heuristic ("the closing
`$` must not be followed by a digit") is **NOT FOUND** in any GitHub primary source, and
`github/cmark-gfm` contains no math extension at all, so there is no specification to cite. Re-scan
if line 43 is edited.

**Ported, not derived.** `report/technical-report.tex` already typesets five of these correctly; the
QSDC secrecy capacity existed as **three** ASCII restatements across three files against one correct
display equation, so that is where the conversion started.

### JupyterLite / marimo / pyvista-wasm: declined, with a reopening condition

Decisive, and not a matter of taste: **the crypto stack does not exist in Pyodide.** No SEAL/CKKS, no
TenSEAL, no MPC runtime and no ONNX among the 354 distribution packages; `onnxruntime` publishes no
emscripten wheel. A notebook that runs this project's actual pipeline is **not implementable**, not
merely inadvisable - it could only narrate the pipeline or re-implement it, and two implementations
of one thing is this project's named defect class.

Two further blockers, each independently sufficient. Both tools fetch their runtime from the
**jsDelivr CDN** by default (marimo's own bug #575, open since 2024-01-15), which breaks the
no-third-party-origin claim the demo rests on. And accessibility is unassessed upstream: no
JupyterLab/JupyterLite issue cites SC 1.4.1, 1.4.11, 1.4.13 or 4.1.3, and marimo has no accessibility
documentation at all - absence of reports, not conformance.

**`pyvista-wasm` does not exist.** No `vtk`, `pyvista`, `trame` or `itk` in the Pyodide lockfile; the
PyPI packages of that name are third-party API lookalikes in a personal namespace, not PyVista
artifacts.

*Reopens only when all four hold:* a Pyodide wheel exists for at least one of CKKS/TFHE or an MPC
runtime, so the notebook can execute the real pipeline; the runtime is self-hosted from this
project's own origin with a CI check that the kernel's package set matches the app's lockfile; cold
start is measured in-house, because **no citable benchmark exists** - the universally repeated
"6.4 MB, 4-5 seconds" is byte-identical boilerplate across the 2022 and current Pyodide roadmaps and
is not a measurement of any shipped version; and the four WCAG criteria above are tested in-house.

### ETSI horizon, recorded because it has a date for the first time

**GS QKD 015 ed3 is V2.1.2 Stable draft (2026-05-27), TB approval 2026-12-02**, and GS QKD 014 ed2
(V1.3.1, Stable draft) carries the same approval date. **The "V3.1.1 draft" figure is refuted** -
zero hits for "3.1.1" on the work-item page and no v3 tag of any kind in the forge, whose in-progress
branch is named v2.1.2. Recorded so a later round does not re-import it from a search summary. No
current claim changes: `docs/pqc-roadmap.md`, `references/REFERENCES.md`, `src/qsh/qkd/kme.py` and
`web/src/lib/clientSdnController.ts` all correctly state V1.1.1 as the only published GS QKD 014
edition.

## Verification assets added in Rounds 22-23

Two checks now exist that assert against sources OUTSIDE this repository. That is the point of both:
every other test in this suite asserts that the code does what the code was written to do, which
cannot catch a shared misunderstanding.

**`tests/test_decoy_reference.py` (13 tests) against `src/qsh/qkd/decoy_reference.py`.** Reproduces
Ma, Qi, Zhao and Lo, Phys. Rev. A 72, 012326 (2005) with the GYS parameter set, and the GLLP
tagged-fraction bound (QIC 4, 325). None of the asserted numbers originated here:

| Assertion | Published | Measured |
|---|---|---|
| tagged fraction at mu = 0.1, zero distance | > 1 | 1.0417 |
| decoy max distance | over 140 km | 142.2 km |
| non-decoy max distance | about 42 km | 39.8 km |
| decoy R/eta, flat, linear in eta | 0.11 | 0.1136 |
| non-decoy R/eta^2, flat, quadratic | 0.15 | 0.1510 |

The trap this closed is worth keeping: the published curves are plotted for efficient BB84 where the
sifting factor q tends to 1, while this project runs standard BB84 at q = 1/2. Reproducing them with
q = 1/2 gives values at exactly HALF the published ones - close enough to read as a fitting
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
