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
4. **Content, in Chrome, every page** - four axes, iterated until clean:
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
