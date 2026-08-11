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
   gate is the browser verification checklist - maintained privately, not in this repository (results recorded in `verification-log.md`, which is published): seven routes including two negative paths, fourteen
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

**Genomics has no label, so it has no QCNN: declined, and not for the reason recorded for two
rounds.** The deferral condition on file was the per-score licence audit. Round 31 checked what would
actually happen if that audit came back clean, and the answer is nothing: `scripts/train/qcnn_variational.py`
is supervised end to end - `BCEWithLogitsLoss`, best epoch chosen on a validation split, and a report
block comparing against `all_positive_f1` and `majority_baseline` - and **1000 Genomes is a reference
panel with no breast-cancer case/control status**. `registry.py` already says so, and
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

## Round 28: a claim retracted, and the work it would take to earn it back

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
consolidations are the part not yet redone, and are recorded here rather than assumed.

**Two commit messages have now described checklist work that was not in the commit.** That is the
same class as a stale document asserting a fixed item is open: the artefact and the description of
the artefact drift apart, and the description is the thing people read.

**PGS000004 / PRS313 is staged, not switched.** The 313-variant clinical instrument, its
position-keyed dosage fetcher and the loader path are in place and tested; both the Python
pipeline and the browser still score PGS000001 / PRS77. The switch is deliberately one step rather
than done halfway, because the browser payload and the gnomAD ancestry overlay are generated
separately and a use case scoring PRS313 in Python while displaying PRS77 in the browser would be
worse than either alone.

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
outcome it exists to prevent. Re-validated at the point of gating: exit 0 with 2,595 rejected on a
clean tree; exit 1 on a missing vector file; exit 1 and "779 rejected, 1 accepted" when a well-formed
key is smuggled into the malformed set.

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
two numbers that happened to differ. Residual, recorded rather than fixed: `browser_rounds` and
`browser_lr` are literals inside `frontend_config()` rather than fields on `FederatedConfig`, so the
override cannot be tuned the way the values it overrides can.

**2. `qsdc_threshold` - CLOSED, by deletion rather than by justification.** The constant no longer
exists. `quantum_secure_direct_comm()` takes no threshold parameter and aborts on a loss-aware
secrecy capacity `Cs = 1 - h(e) - g*h(e_chk)` with `g = 10^(loss_dB/10)`, after Qi et al., *Light Sci.
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
- *GitHub / Google Drive MCP interop.* Deferred with the reason "needs OAuth creds and outward-facing
  accounts that cannot be provisioned in an automated build", which is still true. **Declined**:
  filesystem MCP interop already demonstrates that the quantum-safe channel carries a real third-party
  MCP server, and a second connector would demonstrate the same property at the cost of credentials
  this project should not hold.

Recorded as decisions so neither reappears as a silent omission.
