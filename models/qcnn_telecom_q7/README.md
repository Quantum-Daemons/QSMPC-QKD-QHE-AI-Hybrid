# Variational QCNN - telecom, 7 wires

> **Round 47 retrained this artifact after two circuit corrections.** The multiclass readout had been
> binning the class index over wires 0..m-1, which on a wider register includes wires pooling
> discarded; it now bins over the SURVIVING wires, recorded in `weights.json` as `readout.wires` =
> [0, 2, 4]. The fully-connected block had been one shared scalar angle applied to every survivor; it
> is now a per-survivor RY+RZ, as Cong, Choi & Lukin specify. Current headline `macro_f1` =
> **0.500452**, against **0.731204** on the card this replaces.
>
> Round 46's banner is superseded rather than deleted: it re-measured this artifact after the
> hardware-efficient ansatz was replaced with a real QCNN, and its figure now belongs to a retired
> circuit.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration
circuit for the `telecom_ids` page of
[QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid). It carries no
`hf_repo` and no pinned revision.

## Why a second width exists

The browser's **QCNN qubits** control could not reach the trained circuit. `clientCompute.ts` read
the requested width and then loaded a fixed 8-wire file, so on every page shipping trained weights
the control moved the benchmark row and the untrained fallback and nothing else - an inert control,
the defect class the project's verification checklist opens with.

Making it real needed a second width to exist. Telecom is the case worth measuring: it encodes **7
features on 8 wires**, so the eighth is padded with `RY(0)` and carries no data. This model is the
same circuit trained at the width the data actually asks for.

## What it is

**6** conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RY,RZ), then ONE pooling stage by a
rotation on the survivor conditioned on its partner, on **7 qubits**, reducing the register 7 -> 3;
then a fully-connected block on the three survivors [0, 2, 4], all 7 features encoded as
`RY(tanh(z) * pi)` on wires 0-6 with no padding wire.

`theta` is **55** scalars: 48 conv (1 stage x 6 rounds x 8 angles in the shared filter), 1 pool
angle, and 6 in the fully-connected block. The fully-connected block is per-survivor RY+RZ - 3
survivors x 2 angles - and is NOT shared; weight sharing is a property of the convolution layers,
which is what makes them convolutions. It stays O(log N) because the survivor count is bounded by
where pooling stops, not because F is one angle.

Readout, class map and scaling convention are identical to the 8-wire card next door
(`readout_qubits: 3`, `bit_order: "lsb_first"`, `wires: [0, 2, 4]`,
`class_index: "c = sum_j bit(wires[j]) << j"`, 8-element `scale`/`bias`) - only the width differs,
which is the whole point of the pair.

**At this width every survivor is read.** Pooling leaves exactly three wires and the class index needs
exactly three bits, so [0, 2, 4] is both the survivor list and the readout list. That is not true at 8
wires, where four survive and three are read, and it is why the pre-Round-47 readout defect - a class
bit taken from wire 1, a pooling control frozen before the fully-connected block ran - had less to
give away here.

`src/qsh/qhe/qcnn_trained.py` reads `float(weights["scale"])` and therefore cannot consume this
artifact either, for the same reason as the 8-wire model: the Python reimplementation supports the
two binary use cases only.

## Measured

Same corpus, split, seed, layers and epoch budget as the 8-wire model. Only the width differs.

| quantity | value |
|---|---|
| `macro_f1` (this circuit) | 0.500452 |
| accuracy (companion, not the headline) | 0.541667 |
| logistic regression, identical split, macro F1 | 0.976447 |
| logistic regression accuracy | 0.977083 |
| uniform-random macro F1 floor | 0.125 |
| constant-predictor macro F1 floor | 0.027778 |
| majority-class accuracy floor | 0.152083 |
| classes predicted | 8 |
| n_train / n_test | 2240 / 960 |

**This model beats its floors and loses to classical logistic regression by 0.475995 of macro F1.**
The LR figure is identical to the 8-wire run's because it never sees the circuit, which is a useful
control: an unchanged baseline is evidence the data path did not move.

Per class, from this run:

| class | F1 |
|---|---|
| detector_blinding | 0.960526 |
| photon_number_splitting | 0.888889 |
| combined | 0.646154 |
| wavelength_trojan | 0.5 |
| trojan_horse | 0.446429 |
| normal | 0.4 |
| intercept_resend | 0.161616 |
| rng_manipulation | 0.0 |

`rng_manipulation` scores exactly 0.0 while `n_classes_predicted` is 8, so the label is still emitted
and never lands on a true row. The 8-wire circuit scores 0.483221 on that class, which is the clearest
place the two widths differ and the reason a macro number alone would mislead here.

**Seed position:** unmeasured - no seed sweep was run for the 7-wire circuit after the Round 47
corrections, so every figure on this card is one draw and not a median. The 8-wire card records a
three-seed spread for its own configuration in which seed 0 is the highest of the three, so a
same-seed comparison between the widths is being made at the 8-wire circuit's best available draw and
still loses. Do not read that as strengthening the 7-wire result; read it as a limit on both.

## The width conclusion has reversed, and this is the side that gained

`references/qcnn-width-ablation.md` concluded in Round 40, and restated at five seeds in Round 45,
that 8 wires beat 7 on telecom by +0.0786 macro F1. With the corrected readout **the sign has
flipped**: 7 wires 0.500452, 8 wires 0.458324, so this model leads by 0.042128. That is a reversal of
the recorded finding, not a restatement of it, and the ablation file now carries a supersession banner
saying so.

Both sides are single-seed on the corrected circuit. A flipped sign measured once is a reason to stop
quoting the old direction, not a licence to quote the new one as settled, and neither width is
separated from the other by this experiment. The old per-class table and the old five-seed spreads
stay in the ablation file marked superseded, because deleting them would remove the evidence that the
conclusion moved.

## Reproducing

    python scripts/train/qcnn_variational.py --usecase telecom --layers 6 --epochs 400 --seed 0 --threads 1 --qubits 7

That command is recorded inside `weights.json` and reproduces `theta` exactly. `--qubits` is in the
string because it changes the weights: the first run of this model recorded a command without it,
which reproduced the EIGHT-wire model instead - a reproduction line that does not reproduce is worse
than none, because it gets checked once and trusted afterwards.

## Scope

Simulated state-vector circuit trained by backpropagation, not quantum hardware and not a claim of
quantum advantage - classical logistic regression on the same split beats both widths by more than
0.47 of macro F1, which is far larger than the gap between them. The corpus is synthetic QKD-attack
telemetry generated by `qsh.qml_ids.attacks` with class structure per Al-Kuwari et al.
(arXiv:2509.14282); no commercial-OK real attack corpus exists, which is why no number here is a
performance claim about a real link.
