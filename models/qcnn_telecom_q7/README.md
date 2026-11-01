# Variational QCNN - telecom, 7 wires

> **Round 84 rebuilt this artifact with its 8-wire sibling, on the same circuit and encoding**:
> Hur, Kim and Park's 15-angle SU(4) block on every nearest-neighbour bond, their pooling, and
> RY(c * pi * tanh(z)) at c = 0.5 - both chosen by pre-registered five-seed experiments
> (2026-10-30) that required the change to hold at BOTH widths before either shipped. At 7 wires the
> circuit gained +0.39 macro F1 over the tree (five seeds, c = 1) and the half angle a further
> +0.051, better on 5 of 5 seeds.
>
> Current headline `macro_f1` = **0.967932**, against 0.541894 for the tree this replaces. Logistic
> regression on the identical split scores **0.976447** and is still ahead. The tree's figures are kept
> below, marked superseded.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `telecom_ids` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It carries no
`hf_repo` and no pinned revision.

## Why a second width exists

The page's **QCNN qubits** control must reach a trained circuit, and a width only exists if someone
trained it. Telecom encodes **7 features**, so the 8-wire model carries one wire with no data; this is
the same circuit at the width the data actually asks for. It is the one use case that offers two
widths, because they are different trained circuits whose classes differ - finance and medical
offered a ninth wire until Round 84 and gave the same class at 8 and 9 on every row, so that option
was removed.

## What it is

**`circuit.name`: `hur-su4`**, exactly as the 8-wire card describes, on **7 qubits**: one stage of
**6** rounds of the shared SU(4) block on every nearest-neighbour bond - the boundary bond (0, 6),
the even bonds (0,1) (2,3) (4,5), the odd bonds (1,2) (3,4) (5,6) - then one pooling stage. Pooling
pairs (0,1) (2,3) (4,5) and folds the orphan, wire 6, into wire 0 with the same two angles, leaving
the survivors [0, 2, 4]; the fully-connected block (RZ then RY per survivor, a CNOT ring) acts on
the three. All 7 features are encoded as `RY(0.5 * pi * tanh(z))` with no padding wire.

`theta` is **98** scalars - 90 conv (1 stage x 6 rounds x 15), 2 pool, 6 fully-connected (3 survivors
x 2) - plus 8 scales and 8 biases, 114 trainable. All 98 reach the readout.

**At this width every survivor is read**: pooling leaves exactly three wires and the class index
needs three bits, so [0, 2, 4] is both the survivor list and the readout list (`readout_qubits: 3`,
`bit_order: "lsb_first"`, `class_index: "c = sum_j bit(wires[j]) << j"`).

Not a tree: after pooling the three survivors are up to 0.42 in trace distance from the product of
their marginals on 24 held-out rows, where a tree of two-qubit blocks gives a product state. Simulated
exactly on 2^7 = 128 amplitudes, so classically simulable at this size (Bermejo et al., PRX Quantum
7, 020304, 2026, Sec. IV.B for classical data in a product angle encoding). The trainer and the
browser agree to 8.9e-07 on all 480 held-out rows, with the same class on every one;
`src/qsh/qhe/qcnn_trained.py` implements the binary tree only and refuses this artifact.

## Measured

Same corpus, split, seed, layers and epoch budget as the 8-wire model. Only the width differs.

| quantity | value |
|---|---|
| `macro_f1` (this circuit) | 0.967932 |
| accuracy (companion, not the headline) | 0.96875 |
| logistic regression, identical split, macro F1 | 0.976447 |
| logistic regression accuracy | 0.977083 |
| uniform-random macro F1 floor | 0.125 |
| constant-predictor macro F1 floor | 0.027778 |
| majority-class accuracy floor | 0.152083 |
| classes predicted | 8 |
| encoding scale c | 0.5 |
| checkpoint epoch, chosen on the validation rows | 380 |
| n_train / n_val / n_test | 2240 / 480 / 480 |

Per class, from this run:

| class | F1 |
|---|---|
| intercept_resend | 1.0 |
| detector_blinding | 1.0 |
| photon_number_splitting | 0.990291 |
| normal | 0.989899 |
| combined | 0.976378 |
| rng_manipulation | 0.976 |
| trojan_horse | 0.918033 |
| wavelength_trojan | 0.892857 |

**This model beats its floors and loses to logistic regression by 0.008515 of macro F1.** On this
seed that gap is small: logistic regression is right where the circuit is wrong on 4 held-out rows
and never the reverse, exact McNemar p = 0.12, so seed 0 alone does not separate the two. Five seeds
do: logistic regression wins on all five, by 0.0075 on average, two-level bootstrap 95% interval
[-0.0129, -0.0026] for the circuit minus LR. The LR figure equals the 8-wire run's because it never
sees the circuit, a useful control that the data path did not move.

**Seed position:** seed 0 scores 0.9679, second lowest of the five seeds the pre-registered experiment
ran on this configuration (0.9679, 0.9715, 0.9652, 0.9771, 0.9705; mean 0.9705, sd 0.0044). The seed
was fixed before the experiment and is not re-picked: choosing one by test performance is test-set
selection.

## The width comparison, at five seeds on this circuit

Recorded in [`references/qcnn-width-ablation.md`](../../references/qcnn-width-ablation.md), which is
where a page may take it from. Paired by seed, 7 wires minus 8 wires at c = 0.5: +0.0235, +0.0272,
+0.0039, +0.0269, +0.0021, mean +0.0167 (sd 0.0126); 7 wires wins all five, paired t p = 0.041, exact
Wilcoxon p = 0.0625, the smallest five pairs can give. So the narrower circuit is ahead on every seed
measured, by a gap smaller than the 8-wire circuit's distance to logistic regression (0.0242 on
average over the same five seeds) and about twice the 7-wire one's (0.0075).

## Superseded: the tree circuit this card described until Round 84

Until Round 84 this artifact was the disjoint-pair tree at 7 wires: 6 rounds of an 8-angle filter on
(0,1) (2,3) (4,5), a CRY pool with the orphan folded into wire 0, the same fully-connected block and
readout, `RY(tanh(z) * pi)`, 55 angles. Its headline moved three times: 0.731204 belonged to the
retired hardware-efficient ansatz, 0.500452 to the tree with half its fully-connected block inert,
and 0.541894, at five seeds 0.5309 on average, to the tree after the dead angles were made live, which
is what shipped until Round 84. Its zero-F1 class moved from `rng_manipulation` to
`wavelength_trojan` when the dead angles became live; on this circuit no class scores below 0.89.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase telecom --layers 6 --epochs 400 --seed 0 --threads 1 --qubits 7 --arch hur-su4 --encoding-scale 0.5

Recorded inside `weights.json` with the device (`training.device` cuda). On CUDA it regenerates
`theta`, `scale`, `bias` and every metric bit for bit - the pre-registered experiment's run and the
retrain that wrote this file agree to max |diff| 0. `--qubits` and the two circuit flags are in the
string because each changes the weights; a reproduction line that reproduces something else is worse
than none, because it gets checked once and trusted afterwards.

## Scope

Simulated state-vector circuit trained by backpropagation, running unencrypted; not quantum hardware
and not a claim of quantum advantage - classical logistic regression on the same split wins at both
widths and on every seed. The corpus is synthetic QKD-attack telemetry generated by
`qsh.qml_ids.attacks`, the class list following Al-Kuwari et al. (arXiv:2509.14282) and the features
this project's own design; no commercial-OK real attack corpus exists, which is why no number here is
a performance claim about a real link.
