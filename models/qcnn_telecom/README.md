# Variational QCNN - telecom

> **Round 47 retrained this artifact after two circuit corrections, and the headline FELL.**
> The multiclass readout had been binning the class index over wires 0..m-1, which includes a wire
> pooling discarded; it now bins over the SURVIVING wires, recorded in `weights.json` as
> `readout.wires` = [0, 2, 4]. The fully-connected block had been one shared scalar angle applied to
> every survivor; it is now a per-survivor RY+RZ, which is what Cong, Choi & Lukin specify. Current
> headline `macro_f1` = **0.458324**, against **0.614583** on the card this replaces. The drop is
> explained below and was not recovered by any of the four configurations tried.
>
> Round 46's own banner is superseded rather than deleted: it re-measured this artifact at 0.614583
> after the hardware-efficient ansatz was replaced with a real QCNN, and that figure is now itself a
> retired circuit's number.

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

**6** conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RY,RZ), then ONE pooling stage by a
rotation on the survivor conditioned on its partner, on **8 qubits**, reducing the register 8 -> 4;
then a fully-connected block on the four survivors [0, 2, 4, 6]. 7 features are encoded as
`RY(tanh(z) * pi)` on wires 0-6. **Wire 7 carries no data**: `RY(tanh(0) * pi)` is the identity, and
the browser reaches the same state through `features[q] ?? 0`, so the two agree by construction rather
than by coincidence. `n_features: 7` records that so a test can assert it.

`theta` is **57** scalars and none of its axes is the qubit count: 48 conv (1 stage x 6 rounds x 8
angles in the shared filter), 1 pool angle, and 8 in the fully-connected block.

**The fully-connected block is NOT shared, and must not be.** Until Round 47 it was a single scalar
RY angle applied identically to every survivor, which makes F a global rotation that cannot
distinguish the wires the class index is read from. Cong, Choi & Lukin ask for "a unitary F on the
remaining qubits"; weight sharing is what makes a CONVOLUTION a convolution and is a property of the
conv layers only. F is now RY+RZ per survivor - 4 survivors x 2 angles = 8 parameters. This stays
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

Held-out split, 960 test rows against 2,240 training rows, 8 balanced classes. The headline metric is
**macro F1**, because an aggregate over 8 classes is what hides a broken one.

| quantity | value |
|---|---|
| `macro_f1` (this circuit) | 0.458324 |
| accuracy (companion, not the headline) | 0.502083 |
| logistic regression, identical split, macro F1 | 0.976447 |
| logistic regression accuracy | 0.977083 |
| uniform-random macro F1 floor | 0.125 |
| constant-predictor macro F1 floor | 0.027778 |
| majority-class accuracy floor | 0.152083 |
| classes predicted | 8 |
| n_train / n_test | 2240 / 960 |

**The classical baseline is ahead by 0.518123 of macro F1 - 52 points.** Reported here, and the page
says the same thing. Randomly initialised QCNNs operate only on low-bodyness measurements of their
input and are efficiently classically simulable (Bermejo et al., PRX Quantum 7, 020304 (2026),
DOI 10.1103/8qt9-72ts; arXiv:2408.12739), so a classical model winning on 7 tabular features is the
expected result rather than an anomaly. The circuit still clears both of its floors, which is the
only thing the macro number establishes.

Per class, from this run, and the spread is the point - the aggregate is 0.458324 while two classes
are above 0.83 and one is at 0.071429:

| class | F1 |
|---|---|
| detector_blinding | 0.886076 |
| photon_number_splitting | 0.836364 |
| wavelength_trojan | 0.607407 |
| rng_manipulation | 0.483221 |
| trojan_horse | 0.285714 |
| combined | 0.25 |
| intercept_resend | 0.246377 |
| normal | 0.071429 |

The previous card printed a SEVEN-row table for EIGHT classes, from a retired circuit, with
`rng_manipulation` given as "~0.80" - an approximation next to six-decimal siblings. Both are gone:
the table above is the full eight rows of `per_class_f1` in this directory's `weights.json`.

**Seed position:** seed 0 is the HIGHEST of the three seeds measured on this exact configuration
(0.4583, 0.4185, 0.4553 at seeds 0, 1, 2 - see the sweep below), so the headline is the top of a
three-seed spread and not a median. Three seeds is fewer than the five the binary circuits carry, and
the seed is deliberately not re-picked: choosing one on test performance is test-set selection, which
`references/qcnn-entanglement-ablation.md` refuses.

## Why the number fell, and the attempt to recover it

The defective readout was handing the classifier a fourth feature dimension it was not entitled to.
Wire 1 was frozen at its pre-pooling value, so its bit encoded the register BEFORE pooling mixed it
away - information the architecture is supposed to have discarded. Removing it removed real
discriminative signal, and the honest reading is that part of 0.614583 was never the QCNN's to claim.

Four configurations were then run to see whether the loss could be recovered by capacity or by width.
**None of them recovered it.** Macro F1, corrected readout throughout:

| configuration | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 8 wires, 6 layers | 0.4547 | 0.4567 | 0.4826 |
| 8 wires, 10 layers | 0.4448 | 0.4614 | 0.4732 |
| 12 wires, 6 layers | 0.4362 | - | - |
| 8 wires, 6 layers, per-survivor FC (SHIPPED) | 0.4583 | 0.4185 | 0.4553 |

The first three rows were run before the fully-connected block was made per-survivor; the last row
adds it and is the artifact in this directory - its seed 0 is 0.458324, which is what the table above
prints to six digits. Every cell in this table is below the 0.614583 the defective readout produced,
and every cell is far below the classical logistic regression's 0.976447 on the same split. Ten runs
across four configurations did not find the missing 0.156, which is the evidence that the drop was
the readout defect and not a capacity limit: more layers did not recover it, more wires did not
recover it, and a strictly more expressive fully-connected block did not recover it.

These ten figures come from a sweep and not from this artifact, so they are not in `weights.json`
and the model-card gate does not check them; the shipped row's seed 0 is the one that is.

## The width conclusion has reversed

`references/qcnn-width-ablation.md` concluded that 8 wires beat 7 on telecom by +0.0786 macro F1.
With the corrected readout the sign has flipped: 7 wires scores **0.500452** and 8 wires scores
**0.458324**, so the NARROWER circuit now leads by 0.042128. Both are single-seed on the corrected
circuit, so this is a reversal of the recorded finding rather than a settled result in the opposite
direction - see `models/qcnn_telecom_q7/README.md` for the other side of the pair and the width
ablation file for the superseded numbers, which are marked and not deleted.

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
wins on this use case by 52 points of macro F1, and it wins on the other two as well.
