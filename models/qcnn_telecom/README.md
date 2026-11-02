# Variational QCNN - telecom

> **Round 84 rebuilt this artifact on a different circuit and a different encoding**, both chosen by
> pre-registered five-seed experiments (2026-10-30) before any of them was run on this file:
>
> * **the circuit**: Hur, Kim and Park's 15-angle SU(4) block on EVERY nearest-neighbour bond, with
>   their pooling, replacing the 8-angle filter on disjoint pairs. On disjoint pairs each surviving
>   wire saw only its own feature pair, and that, not the block, was the limit: the same SU(4) block
>   on disjoint pairs gained +0.04 macro F1, on every bond +0.42 (five seeds, c = 1);
> * **the encoding**: RY(c * pi * tanh(z)) with c = 0.5 instead of 1. At c = 1 a feature's two tails
>   fold onto one state, RY(-pi)|0> and RY(pi)|0> being the same ray; c = 0.5 gained +0.050 macro F1
>   on top of the circuit, better on 5 of 5 seeds.
>
> Current headline `macro_f1` = **0.944439**, against 0.510963 for the tree this replaces. Logistic
> regression on the identical split still scores **0.976447** and is still ahead, on this seed and on
> every one of the five. Everything this card said about the tree is kept below, marked superseded.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `telecom_ids` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid). It carries no
`hf_repo` and no pinned revision.

## What it is

**`circuit.name`: `hur-su4`.** One convolution stage of **6** rounds on **8 qubits**. Each round
applies ONE shared two-qubit block to every nearest-neighbour bond of the register, in Hur et al.'s
order: the boundary bond (0, 7), the even bonds (0,1) (2,3) (4,5) (6,7), the odd bonds (1,2) (3,4)
(5,6). The block is their circuit 9, the Vatan-Williams form of a general two-qubit unitary: U3 on
each wire, CNOT(a, b), RY(a) and RZ(b), CNOT(b, a), RY(a), CNOT(a, b), U3 on each wire - 15 angles
and 3 CNOTs. Then ONE pooling stage, also theirs: RZ on the kept wire where its discarded partner is
1, an X on the partner, RX on the kept wire where the partner is now 1 - so RZ when it was 1 and RX
when it was 0, two shared angles - leaving the survivors [0, 2, 4, 6]. Then the project's own
fully-connected block, RZ then RY on each survivor and a CNOT ring over the four.

7 features are encoded as `RY(0.5 * pi * tanh(z))` on wires 0-6, z standardised with the training
split's moments shipped here (`input_space: "raw"`, `encoding_scale: 0.5`). **Wire 7 carries no
data**: it stays in |0>, and every inference path leaves wires past `n_features` there.

`theta` is **100** scalars - 90 conv (1 stage x 6 rounds x 15), 2 pool, 8 fully-connected (4 survivors
x 2) - plus 8 scales and 8 biases, 116 trainable in all, against 73 for the tree. None of its axes is
the qubit count: the block is shared by every bond, which is what makes it a convolution. All 100
angles reach the readout (`tests/test_r84_c2_encoding_and_circuit.py` perturbs each one).

**It is no longer a tree, and that is checked rather than asserted.** After pooling, the four
survivors of a tree of two-qubit blocks are a product of one-qubit states; here, on 24 held-out rows,
their reduced state is up to 0.57 in trace distance from the product of its marginals, against below
1e-6 for the tree's layout. What it still is: a simulation, exact on the 2^8 = 256 amplitudes it
runs on, and therefore **classically simulable at this size**. Classical data in a product angle
encoding at 8 qubits is the regime Bermejo et al. (PRX Quantum 7, 020304, 2026, Sec. IV.B) find
fully classically simulable, so nothing here suggests a quantum advantage, and nothing is claimed.

The readout is unchanged from the tree: a **basis marginal** over the first three survivors,
`readout_qubits: 3`, `bit_order: "lsb_first"`, `wires: [0, 2, 4]`,
`class_index: "c = sum_j bit(wires[j]) << j"`. Four wires survive and three are read; wire 6 is
rotated and entangled and does not carry a bit of the class index. The convention ships as data
because reading the bits the other way permutes the classes by `[0,4,2,6,1,5,3,7]` while leaving
every aggregate metric identical.

`scale` and `bias` are 8-element arrays. `src/qsh/qhe/qcnn_trained.py` implements the binary tree
only and refuses this artifact by name; the trainer and `web/src/lib/quantumSim.ts` implement this
circuit, and they agree to 1.0e-06 on all 480 held-out rows, with the same class on every one.

## Measured performance

Held-out split, **480 test rows** against 2,240 training rows, 8 balanced classes, seed 0. The
headline metric is **macro F1**, because an aggregate over 8 classes is what hides a broken one.

| quantity | value |
|---|---|
| `macro_f1` (this circuit) | 0.944439 |
| accuracy (companion, not the headline) | 0.945833 |
| logistic regression, identical split, macro F1 | 0.976447 |
| logistic regression accuracy | 0.977083 |
| uniform-random macro F1 floor | 0.125 |
| constant-predictor macro F1 floor | 0.027778 |
| majority-class accuracy floor | 0.152083 |
| classes predicted | 8 |
| encoding scale c | 0.5 |
| checkpoint epoch, chosen on the validation rows | 399 |
| n_train / n_val / n_test | 2240 / 480 / 480 |

Per class, from this run:

| class | F1 |
|---|---|
| detector_blinding | 1.0 |
| intercept_resend | 0.992126 |
| photon_number_splitting | 0.990291 |
| normal | 0.989899 |
| rng_manipulation | 0.983607 |
| combined | 0.96063 |
| trojan_horse | 0.857143 |
| wavelength_trojan | 0.781818 |

Every class is now found: the lowest is `wavelength_trojan` at 0.781818, where the tree scored
`combined` at exactly 0.0 and `normal` at 0.22807.

## Against logistic regression

**The classical baseline is ahead by 0.032008 of macro F1.** On the 480 held-out rows logistic
regression is right where this circuit is wrong on 15 rows and never the other way round (exact
McNemar p = 6.1e-05), and its per-class F1 is at least this circuit's on every class
(`trojan_horse` 0.928 and `wavelength_trojan` 0.917431 against 0.857143 and 0.781818). Over five seeds
it wins on all five, by 0.0242 on average (two-level bootstrap 95% interval [-0.0352, -0.0136] for
the circuit minus LR). The page prints both numbers side by side and computes which is ahead rather
than asserting it.

**Seed position:** seed 0 scores 0.9444, tied lowest of the five seeds the pre-registered experiment
ran on this exact configuration (0.9444, 0.9444, 0.9613, 0.9503, 0.9684; mean 0.9537, sd 0.0107).
The figure above is therefore at the low end of its spread, not a median. The seed was fixed before
the experiment and is deliberately not re-picked: choosing one by test performance is test-set
selection, which `references/qcnn-entanglement-ablation.md` refuses.

## Five seeds, from the pre-registered experiments (not this file)

These are the runs that chose this circuit and this scale; seed 0 of the bold row IS this artifact
(theta, scale and bias equal to max |diff| 0). Not recorded in `weights.json`, so not checked by the
model-card gate; the experiment's scripts and criteria are kept with the round's working files.

| configuration, 8 wires | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| tree, c = 1 (shipped until Round 84) | 0.5110 | 0.4645 | 0.4774 | 0.4835 | 0.4865 | 0.4846 |
| SU(4) on disjoint pairs only, c = 1 (control) | 0.5391 | 0.4863 | 0.5770 | 0.5304 | 0.5030 | 0.5272 |
| SU(4) on every bond, c = 1 | 0.9183 | 0.9111 | 0.8989 | 0.9052 | 0.8872 | 0.9041 |
| **SU(4) on every bond, c = 0.5 (this artifact)** | **0.9444** | 0.9444 | 0.9613 | 0.9503 | 0.9684 | **0.9537** |
| logistic regression | 0.9764 | 0.9736 | 0.9789 | 0.9837 | 0.9771 | 0.9780 |

Two separate effects, and the control is what separates them: the bigger block on the same disjoint
pairs moved the mean by +0.043, covering every bond with it moved it by +0.420, and the half angle
then added +0.050 (two-level 95% interval [+0.029, +0.072], better on 5 of 5 seeds).

## Browser cost

One inference of this circuit in `quantumSim.ts` takes **0.27 ms** median (p95 0.35 ms) in node 22 on
the development machine, against 0.07 ms for the binary tree circuits; the 7-wire sibling takes
0.12 ms. One QCNN inference runs per page run, so the page stays interactive. The page reports its own
measurement as `qcnn_inference_ms`.

## Superseded: the tree circuit this card described until Round 84

The artifact here until Round 84 was the disjoint-pair tree - 6 rounds of an 8-angle filter on (0,1)
(2,3) (4,5) (6,7), a CRY pool, the same fully-connected block and readout, `RY(tanh(z) * pi)` - and
its headline moved four times. Kept so a figure found in an older document can be placed:

| figure | what it was measured on |
|---|---|
| 0.809846 | the RETIRED hardware-efficient ansatz (RY on every wire and a CNOT ring) replaced in Round 46: a different circuit |
| 0.614583 | published as the tree's macro F1 by one card rewrite; it was the ACCURACY of that run |
| 0.458324 | the tree after the readout was corrected to bin over the SURVIVING wires |
| 0.510963 | the tree after its analytically dead parameters were made live (Round 47); shipped until Round 84 |

Two of the tree's lessons carry over unchanged. Its readout once binned over wires 0..m-1, which
included wire 1, a pooling control frozen before the fully-connected block ran; the survivor readout
above is that fix. And it once carried 5 of 57 analytically dead angles - RZ gates applied where only
a diagonal use or the readout followed - fixed by ordering, RZ before RY; the dead-parameter test now
covers this circuit's layouts too. The capacity sweep Round 47 ran on the tree (10 runs, 8 and 12
wires, 6 and 10 layers, none recovering the loss) measured the tree, and the overlap experiment is
the answer it pointed at: the loss was the disjoint-pair layout, not the parameter count.

## Reproducibility

    python scripts/train/qcnn_variational.py --usecase telecom --layers 6 --epochs 400 --seed 0 --threads 1 --qubits 8 --arch hur-su4 --encoding-scale 0.5

Recorded inside `weights.json` with the device it ran on (`training.device` cuda,
`training.device_name`). On CUDA it regenerates `theta`, `scale`, `bias` and every metric bit for
bit: two independent runs - the pre-registered experiment's and the retrain that wrote this file -
agree to max |diff| 0. On a CPU it is not guaranteed: CPU and CUDA kernels round differently, and a
400-epoch multiclass run amplifies that; the tree this replaces reproduced its metrics only on CUDA.

## Data and licence

**SYNTHETIC.** QKD-attack telemetry generated by `qsh.qml_ids.attacks`: the class list follows
Al-Kuwari et al. (arXiv:2509.14282), 8 classes = 7 attack types + normal traffic; the 7 features and
their signature values are this project's synthetic design. No commercial-OK corpus of real QKD
attack captures exists, which is why this is generated rather than collected.

Generated by this repository - no third-party terms attach.

## Scope

Simulated state-vector evaluation, not quantum hardware. The circuit demonstrates that the page's
inference selector surfaces a class from trained parameters on a multiclass task; it runs
unencrypted, it is not offered as a competitive intrusion detector, and nothing on this card is a
claim of quantum advantage: classical logistic regression wins on this use case on every seed, and on
synthetic data, so no figure here is a claim about a real link.
