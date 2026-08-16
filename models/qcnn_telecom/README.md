# Variational QCNN - telecom

> **Round 47 retrained this artifact a SECOND time, after finding two classes of parameter that
> could not move the readout at any input.** Both were carried by the optimiser and counted in the
> parameter count. The fully-connected block's per-survivor RZ was applied AFTER the RY, and the only
> operations after that block are a CNOT ring - a basis permutation - and `|amplitude|^2`, which
> discards phase; a diagonal gate applied last cannot move a computational-basis readout. The 8th
> conv angle of each stage's final layer round is an `rz` on the wire pooling is about to discard,
> whose only later interaction is as the CONTROL of a CRY - diagonal in that wire's basis - and then
> marginalisation at readout; that was 1 of this circuit's 48 conv angles, carried since Round 46.
>
> **The fix was ORDERING, not deletion.** RZ before RY in both places, so the phases RZ writes are
> mixed back into amplitudes by the RY that follows. RE-MEASURED on the weights shipped here, with
> the procedure stated because an earlier revision of this banner quoted figures with no seed and no
> inputs: reconstruct the pre-fix gate order from
> `git show 8017538^:scripts/train/qcnn_variational.py`, feed `x = tanh(randn(8, n_qubits) * 0.7)` at
> `torch.Generator().manual_seed(0)`, add 1.234 rad to every fc RZ angle and take `max |delta|` of
> `circuit()`. Under the pre-fix order that moves this circuit's readout by **5.96e-08**, which is
> 2^-24, a float32 ULP, against **1.77e-01** for the RY angles beside them. Under the shipped order
> the same RZ perturbation moves it by **2.28e-01**. The `2.6e-01` and `1.4e-01` an earlier revision
> printed here are withdrawn: they carried no recipe, and the weights they were taken on were
> replaced by this retrain.
> `tests/test_the_qcnn_has_the_structure_the_name_claims.py::test_every_trainable_parameter_reaches_the_readout`
> now perturbs EVERY parameter at five (width, readout, layers) configurations and fails if any one
> of them cannot move the output.
>
> Current headline `macro_f1` = **0.510963**. **This number has moved four times and three of the
> four earlier figures are still quoted somewhere**, so the whole trajectory is stated once, below,
> under its own heading. Every earlier banner on this card is superseded rather than deleted.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `telecom_ids` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It carries no
`hf_repo` and no pinned revision.

**This card was missing for several rounds while the finance and medical ones existed** - and this is
the most complicated of the three circuits, the only multiclass one, and the only one shipping a
readout convention as data. An artefact with real influence on a page and no documentation is the
defect Round 28 wrote the other two cards to fix; this one was not caught because the check was "do
the QCNN directories have cards" answered by looking at two of them.

## What it is

**6** conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RZ,RY), then ONE pooling stage by a
rotation on the survivor conditioned on its partner, on **8 qubits**, reducing the register 8 -> 4;
then a fully-connected block on the four survivors [0, 2, 4, 6]. 7 features are encoded as
`RY(tanh(z) * pi)` on wires 0-6. **Wire 7 carries no data**: `RY(tanh(0) * pi)` is the identity, and
the browser reaches the same state through `features[q] ?? 0`, so the two agree by construction rather
than by coincidence. `n_features: 7` records that so a test can assert it.

`theta` is **57** scalars and none of its axes is the qubit count: 48 conv (1 stage x 6 rounds x 8
angles in the shared filter), 1 pool angle, and 8 in the fully-connected block. The count did not
change in the second Round 47 retrain: the dead-parameter fix reordered gates and deleted nothing.
What changed is how many of the 57 the readout can see. **5 of 57 were analytically dead** - the four
fully-connected RZ angles, one per survivor, and `theta[47]`, the 8th angle of the last conv round -
and all 57 are reachable now. The 7-wire sibling had 4 of 55 dead, one fewer because it pools to
three survivors rather than four.

**The fully-connected block is NOT shared, and must not be.** Until Round 47 it was a single scalar
RY angle applied identically to every survivor, which makes F a global rotation that cannot
distinguish the wires the class index is read from. Cong, Choi & Lukin ask for "a unitary F on the
remaining qubits"; weight sharing is what makes a CONVOLUTION a convolution and is a property of the
conv layers only. F is now RZ+RY per survivor - 4 survivors x 2 angles = 8 parameters. This stays
O(log N) because the SURVIVOR COUNT IS BOUNDED, not because the block is shared: `layer_plan` stops
pooling while `active // 2 >= readout_qubits`, so the register reaching F holds 4 survivors at 8
qubits and 4 again at 16.

The readout is **not** the `<Z_0>` scalar the other two circuits use. It is a **basis marginal** over
three wires: `readout_qubits: 3`, `bit_order: "lsb_first"`, `wires: [0, 2, 4]`,
`class_index: "c = sum_j bit(wires[j]) << j"`.

**Those wires are the survivors, and until Round 47 they were not.** The class index was binned by
`i & (k - 1)`, which reads bits 0..m-1, and wire q is bit q - so a 3-qubit readout read wires 0, 1
and 2 while pooling had left [0, 2, 4, 6]. Wire 1 is the pooling CONTROL of the pair (0, 1): it is
conditioned into wire 0 and then never touched again, so one bit of every class index was supplied by
a wire frozen at its pre-pooling value, which the second half of the circuit does not see. The
convention ships **as data** because reading the bits the other way permutes the classes by
`[0,4,2,6,1,5,3,7]` - four fixed points, and identical aggregate metrics either way. A permuted class
map with the same macro F1 is exactly the kind of error no aggregate number can surface, so
`tests/test_qcnn_multiclass_readout.py` asserts the convention rather than trusting it.

Note that four wires survive and three are read. The fully-connected block acts on all four; wire 6
is rotated and entangled and then does not carry a bit of the class index, because 8 classes need
three bits and pooling is stopped before the register falls below that.

`scale` and `bias` are 8-element arrays here, not scalars. `src/qsh/qhe/qcnn_trained.py` reads
`float(weights["scale"])` and therefore **cannot consume this artifact** - the Python reimplementation
supports the two binary use cases only. The parity check that found a crossed CNOT ring in the trainer
does not cover this circuit.

## Measured performance

Held-out split, **480 test rows** against 2,240 training rows, 8 balanced classes. The headline
metric is **macro F1**, because an aggregate over 8 classes is what hides a broken one. 3200 rows
split 70 / 15 / 15 gives 2240 train, 480 validation, 480 test; every figure below is computed on
those 480 test rows only. An earlier revision of this card said 960, which was validation PLUS test:
the trainer recorded `n_test` as everything after the training cut. The metrics were always computed
on the 480-row slice, so only the row count was wrong.

| quantity | value |
|---|---|
| `macro_f1` (this circuit) | 0.510963 |
| accuracy (companion, not the headline) | 0.554167 |
| logistic regression, identical split, macro F1 | 0.976447 |
| logistic regression accuracy | 0.977083 |
| uniform-random macro F1 floor | 0.125 |
| constant-predictor macro F1 floor | 0.027778 |
| majority-class accuracy floor | 0.152083 |
| classes predicted | 8 |
| n_train / n_val / n_test | 2240 / 480 / 480 |

**The classical baseline is ahead by 0.465484 of macro F1 - 47 points.** Reported here, and the page
says the same thing. Randomly initialised QCNNs operate only on low-bodyness measurements of their
input and are efficiently classically simulable (Bermejo et al., PRX Quantum 7, 020304 (2026),
DOI 10.1103/8qt9-72ts; arXiv:2408.12739), so a classical model winning on 7 tabular features is the
expected result rather than an anomaly. The circuit still clears both of its floors, which is the
only thing the macro number establishes.

Per class, from this run, and the spread is the point - the aggregate is 0.510963 while one class is
above 0.90 and one is at exactly 0.0:

| class | F1 |
|---|---|
| detector_blinding | 0.903226 |
| wavelength_trojan | 0.745763 |
| photon_number_splitting | 0.605042 |
| trojan_horse | 0.591549 |
| intercept_resend | 0.571429 |
| rng_manipulation | 0.442623 |
| normal | 0.22807 |
| combined | 0.0 |

`combined` scores exactly 0.0 while `n_classes_predicted` is 8, so the label is still emitted and
never lands on a true row - the same failure the 7-wire sibling has on `wavelength_trojan`, on a
different class. An aggregate of 0.510963 with a zero underneath it is why the headline is macro F1
and why the per-class table is printed beside it.

The card two rewrites ago printed a SEVEN-row table for EIGHT classes, with `rng_manipulation` given
as "~0.80" - an approximation next to six-decimal siblings. The table above is the full eight rows of
`per_class_f1` in this directory's `weights.json`.

**Seed position:** unmeasured on this circuit. The three-seed spread this card published one rewrite
ago - 0.4583 / 0.4185 / 0.4553 at seeds 0, 1, 2, with seed 0 the highest of the three - was measured
BEFORE the dead parameters were made live and is SUPERSEDED; it is kept in the sweep table below as
the evidence that the loss was not a capacity limit, not as a range that still applies. No sweep has
been run on the current circuit, so the headline is one draw. The seed is deliberately not re-picked:
choosing one on test performance is test-set selection, which
`references/qcnn-entanglement-ablation.md` refuses.

## This headline has moved four times, and here are all four

Stated once, plainly, because three of the four are still quoted somewhere and each of them is wrong
as a baseline for a different reason.

| figure | what it was measured on |
|---|---|
| 0.809846 | the RETIRED hardware-efficient ansatz - RY on every wire plus a CNOT ring - replaced in Round 46. A different circuit. **Never this one's baseline.** |
| 0.614583 | published as this circuit's macro F1 by an earlier card rewrite. It was the ACCURACY of that run, not an F1 at all. |
| 0.458324 | the real QCNN, after the readout was corrected to bin over the SURVIVING wires. The first figure on this row that is a macro F1 of this architecture. |
| **0.510963** | **current.** The same circuit after the analytically dead parameters were made live. |

The first row is the one to watch for. It belongs to a circuit this repository deleted, it is the
largest of the four, and it is the one a reader reaching for "what did telecom score" is most likely
to find in an older document. `references/qcnn-width-ablation.md` and
`references/qcnn-entanglement-ablation.md` both carry supersession banners saying so.

## Why it fell, what the dead parameters gave back, and what never came back

The defective readout was handing the classifier a fourth feature dimension it was not entitled to.
Wire 1 was frozen at its pre-pooling value, so its bit encoded the register BEFORE pooling mixed it
away - information the architecture is supposed to have discarded. Removing it removed real
discriminative signal, and the honest reading is that part of that 0.614583 was never the QCNN's to
claim.

Making the five dead parameters live gave back **+0.053** of macro F1, from 0.458324 to 0.510963.
That is the largest of the three GAINS the same fix produced across the six artifacts, and it is not
the largest effect: the finance circuit REGRESSED by 0.105 under the identical change, twice the size
of this gain and the largest single move of the four. Measured from the artifacts at each commit:
telecom macro F1 +0.052639, telecom-q7 macro F1 +0.041442, medical minority F1 +0.031250, finance
minority F1 -0.105286. An earlier revision of this paragraph called +0.053 "the largest of the four
effects", which the finance figure in its own next clause contradicts. See the table on
`models/qcnn_finance/README.md`. What the +0.053 does establish is that the parameters were
worth something once a gradient could reach them, which is the point of the fix; what it does not
establish is that this architecture is competitive, because the classical baseline is still 47 points
ahead.

Before the dead parameters were found, four configurations were run to see whether the loss from the
readout correction could be recovered by capacity or by width. **None of them did.** Macro F1,
corrected readout throughout, and **every cell below was measured with the dead parameters still
dead** - these are superseded as current figures and kept as the capacity evidence:

| configuration | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 8 wires, 6 layers | 0.4547 | 0.4567 | 0.4826 |
| 8 wires, 10 layers | 0.4448 | 0.4614 | 0.4732 |
| 12 wires, 6 layers | 0.4362 | - | - |
| 8 wires, 6 layers, per-survivor FC (shipped at the time) | 0.4583 | 0.4185 | 0.4553 |

Ten runs across four configurations did not find the missing **0.110876**, which is the evidence that
the drop was the readout defect and not a capacity limit: more layers did not recover it, more wires
did not recover it, and a strictly more expressive fully-connected block did not recover it - the
best cell above, 0.4826, is still 0.0866 short. That gap is macro F1 against macro F1: the
pre-correction run's macro F1 was **0.5692**, recorded in
`git show 4ee258e:models/qcnn_telecom/weights.json` beside its 0.614583 accuracy, so the baseline
was never missing from the record. An
earlier revision of this card printed the gap as 0.156, which is 0.614583 minus 0.458324 - the
ACCURACY minus a macro F1, exactly the two metrics the table three paragraphs above warns are
different - and that figure is withdrawn. The dead-parameter fix then recovered 0.052639 of the
0.110876, which is nearly half rather than the third the inflated denominator implied, by making the
parameters that were already there reachable - consistent with the same reading, since it added no
capacity either.

These ten figures come from a sweep and not from this artifact, so they are not in `weights.json`
and the model-card gate does not check them. The shipped row is no longer the shipped configuration:
its seed 0 of 0.4583 is the PREVIOUS artifact's headline, superseded by the 0.510963 above.

## The width conclusion has reversed

`references/qcnn-width-ablation.md` concluded that 8 wires beat 7 on telecom by +0.0786 macro F1.
With the corrected circuit the sign has flipped, and the second retrain raised both widths and left
the sign intact while NARROWING the gap: 7 wires scores **0.541894** and 8 wires scores **0.510963**,
so the narrower circuit leads by 0.030931. At the previous retrain the same pair read 0.500452
against 0.458324, a lead of 0.042128, so the reversal has now survived a change that moved both
sides - on a margin that closed by 0.011197 rather than opening. An earlier revision of this
paragraph said the gap grew, which the two figures it quotes contradict. Both figures are still
single-seed, so this retires the old direction rather than settling the new one - see
`models/qcnn_telecom_q7/README.md` for the other side of the pair and the width ablation file for the
superseded numbers, which are marked and not deleted.

The mechanism is consistent with the readout defect: at 7 wires pooling leaves exactly three
survivors [0, 2, 4] and all three are read, so no surviving wire goes unread and no discarded wire
was ever readable. The 8-wire circuit is the one that had a spare pre-pooling bit to leak.

## Reproducibility

    python scripts/train/qcnn_variational.py --usecase telecom --layers 6 --epochs 400 --seed 0 --threads 1 --qubits 8

That command is recorded inside `weights.json` and regenerates this file: five consecutive runs gave
identical metrics and identical `theta` to nine decimal places. That is true only since the multiclass
readout stopped using `index_add_`, whose CUDA `atomicAdd` made the same seeded command land on macro
F1 0.809846 or 0.807781 depending on the run - both figures belong to a retired circuit and are kept
here, as they are in the artifact's own `training.note`, only to record why the reproducibility claim
is now safe to make.

The previous card justified the epoch and layer counts with a defaults-versus-full-command comparison
(0.4998 and 0.6443 against 0.8098). Those three figures were measured on the retired circuit and have
NOT been re-run against this one, so they are superseded and are not restated as current.

## Data and licence

**SYNTHETIC.** QKD-attack telemetry generated by `qsh.qml_ids.attacks`, class structure per Al-Kuwari
et al. (arXiv:2509.14282): 8 classes = 7 attack types + normal traffic. No commercial-OK corpus of
real QKD attack captures exists, which is why this is generated rather than collected, and why the
provenance string says so on the page.

Generated by this repository - no third-party terms attach.

## Scope

Simulated state-vector evaluation, not quantum hardware. The circuit demonstrates that the
private-inference selector surfaces a class from trained parameters on a multiclass task; it is not
offered as a competitive intrusion detector, and the logistic regression beside it is the one the page
actually reports. Nothing on this card is a claim of quantum advantage: classical logistic regression
wins on this use case by 47 points of macro F1, and it wins on the other two as well.
