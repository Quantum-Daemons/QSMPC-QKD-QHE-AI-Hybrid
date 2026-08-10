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
   gate is `browser-verification-checklist.md` (results recorded in `verification-log.md`): seven routes including two negative paths, fourteen
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

**ZeroGPU Space: blocked on a date, not on work.** Four consecutive rounds, the longest-carried item
here. `spaces/qsh-inference/` exists, and `ProvenanceBadge` declares a `remote` tier no code path
sets. The blocker is a Hugging Face account-age unlock around 2026-09-01. That is a condition, not a
task, so it is recorded as one: **when the account unlocks, wire it or delete the Space and the
`remote` tier together.** It must not appear in a fifth deferred list unqualified.

**OpenFHE `EvalBootstrapSetup` SIGSEGV: permanently worked around, on purpose.** Three rounds. The
segfault is inside OpenFHE on AlmaLinux 9.7, and `fhe/backends.py` guards an `ImportError`, which a
segfault is not. `make test` deselects it with `-m "not slow"`; `make test-all` still dies. Fixing an
upstream crash on one host distribution is not this project's work, and the workaround is the
outcome rather than a placeholder for one. Recorded as a decision.

**node-seal is on SEAL 4.1.2, and node-seal is the constraint, not us.** Promised in Round 19,
tasked again in Round 21, never written, and then it dropped off the list without resolution. The
answer: `web/package.json` pins `node-seal ^7.0.0`, 7.0.0 is installed, and node-seal 7.0.0's own
package metadata states SEAL 4.1.2. **Re-checked 2026-08-09** - node-seal 7.0.0 (2025-11-08) is
still the newest node-seal on npm, so this is not a stale pin here.

The wording this file carried until Round 23 - "and that is upstream's newest" - was ambiguous in a
way that flattered the position. It is true of node-seal and false of SEAL. Measured: Microsoft SEAL
has shipped **six releases since 4.1.2** and reached **4.4.3 on 2026-08-04**. 4.4.0 (2026-07-25) is
described only as "Major security update", with no further detail; 4.4.1 fixes a MinGW build; 4.4.3
fixes null-output-pointer dereferences and C++ exceptions leaking across the **C API** boundary
(issues #753-#755). Whether C API fixes reach an Emscripten/WASM build at all is **unverified** and
should not be assumed either way.

Round 19's instruction was to document the gap rather than claim a vulnerability, and that still
holds: a GitHub Security Advisory query for SEAL returns **no advisories**, and no CVE exists. So
the version and the release history are recorded as facts, the gap is recorded as widening, and no
vulnerability is claimed. Re-check this on any round that touches the FHE path; the release cadence
since April 2026 is roughly monthly.

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

Both are cheap and neither is wired into `make test` yet, which is a gap rather than a decision:
`check:pqc` fetches its vectors over the network, so it needs an offline-tolerant mode before it can
gate CI.

## What this project cannot verify, and says so

- **No QKD hardware exists here.** Every SKR and QBER comes from an analytic channel model. There is
  no browser API for quantum communication and there will not be one.
- **The security model is semi-honest.** Malicious security was designated in Round 1 and never
  built.
- **Simulated quantum layers are simulated.** The ETSI control plane is standards-faithful over a
  simulated quantum layer, and every payload says so in-band.
- **GS QKD 014 mutual TLS is not implemented**, so SAE identity is a parameter rather than an
  authenticated fact.
- **Trusted relays are an assumption**, not a mechanism: a relay holds the key in plaintext and no
  cryptography here changes that.

## Open, with numbers, so they do not roll forward silently (Round 23b)

Two items are known, verified and NOT fixed. Recorded here rather than left implicit, which is this
project's standing rule for anything deferred.

**1. The browser's federated hyperparameters differ from `config.federated`, and cannot read it.**

| | Python config | Browser |
|---|---|---|
| rounds | 8 | 6 |
| learning rate | 0.05 | 0.15 |
| clients | (config) | 4 |

`FederatedConfig` is **not exported to the baked frontend config at all** - `federated` is absent
from `web/src/lib/config.defaults.json` - so the browser could not read these even if it wanted to.
The divergence is defensible (a federated round has to finish inside a page interaction) and is now
stated at the call site in `clientCompute.ts`, but it is a divergence. Fixing it means exporting the
dataclass first; changing the browser numbers alone would move every reported federated accuracy
without changing what the server does.

**2. `qsdc_threshold` defaults to 0.11 and is still not traced to a QSDC source.**

0.11 is the textbook BB84 one-way security threshold (Shor-Preskill). QSDC is a different protocol
with its own security analysis, so the value is not automatically wrong - but nothing in
`references/` justifies it for QSDC, and it sits above this repository's own measured GLLP zero-key
point of 0.098106. Either find the QSDC source or replace the default with one that has one.

Neither is user-visible today. Both are the kind of item that becomes a finding two rounds later if
it is not written down.

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
- *GitHub / Google Drive MCP interop.* Deferred with the reason "needs OAuth creds and outward-facing
  accounts that cannot be provisioned in an automated build", which is still true. **Declined**:
  filesystem MCP interop already demonstrates that the quantum-safe channel carries a real third-party
  MCP server, and a second connector would demonstrate the same property at the cost of credentials
  this project should not hold.

Recorded as decisions so neither reappears as a silent omission.
