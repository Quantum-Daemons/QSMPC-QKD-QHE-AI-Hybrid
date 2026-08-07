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
   gate is `browser-verification-checklist.md`: seven routes including two negative paths, fourteen
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

   Three files are exceptions, and the third is the interesting one. A private workflow note
   and the private business-strategy reports are Japanese by design and untracked. `.gitignore` is
   tracked and contains Japanese because its patterns name those confidential `.docx` files exactly -
   and that is a confidentiality control, not prose. Rewriting them in ASCII to satisfy this
   convention would weaken the one rule whose job is to stop a leak, which is the wrong trade; a
   plain `*.docx` line sits beside them so the control holds by extension alone even if a filename
   changes.

   Run the grep over TEXT files only. Every research PDF under `references/` matches the CJK byte
   ranges, because arbitrary binary does; a scan that includes them reports twelve violations and
   none of them are real.

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
