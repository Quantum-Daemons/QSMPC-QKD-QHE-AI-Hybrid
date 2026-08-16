# Variational QCNN - telecom, 7 wires

> **Round 47 retrained this artifact a SECOND time, after finding that 4 of its 55 parameters could
> not move the readout at any input.** Three were the fully-connected block's per-survivor RZ,
> applied AFTER the RY, with only a CNOT ring - a basis permutation - and `|amplitude|^2` between
> them and the readout; a diagonal gate applied last cannot move a computational-basis readout. The
> fourth is `theta[47]`, the 8th angle of the last conv round, an `rz` on the wire pooling is about
> to discard, whose only later interaction is as the CONTROL of a CRY and then marginalisation.
>
> **The fix was ORDERING, not deletion**: RZ before RY in both places, so the phases RZ writes are
> mixed back into amplitudes by the RY that follows. `theta` is still 55 scalars; what changed is
> that all 55 are now reachable from the readout.
> `tests/test_the_qcnn_has_the_structure_the_name_claims.py::test_every_trainable_parameter_reaches_the_readout`
> perturbs every parameter at five (width, readout, layers) configurations and fails if one cannot
> move the output.
>
> Current headline `macro_f1` = **0.541894**, against 0.500452 on the card this replaces - **+0.041**.
> Three of the four artifacts gained from this fix and one regressed; see
> `models/qcnn_finance/README.md` for the one that lost 0.105. Every earlier banner on this card is
> superseded rather than deleted: 0.731204 belongs to the retired hardware-efficient ansatz and
> 0.500452 to this circuit with half of its fully-connected block still inert.

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

**6** conv rounds of a SHARED two-qubit filter (RY,RZ - CNOT - RZ,RY), then ONE pooling stage by a
rotation on the survivor conditioned on its partner, on **7 qubits**, reducing the register 7 -> 3;
then a fully-connected block on the three survivors [0, 2, 4], all 7 features encoded as
`RY(tanh(z) * pi)` on wires 0-6 with no padding wire.

`theta` is **55** scalars: 48 conv (1 stage x 6 rounds x 8 angles in the shared filter), 1 pool
angle, and 6 in the fully-connected block. The fully-connected block is per-survivor RZ+RY - 3
survivors x 2 angles - and is NOT shared; weight sharing is a property of the convolution layers,
which is what makes them convolutions. It stays O(log N) because the survivor count is bounded by
where pooling stops, not because F is one angle.

**The RZ comes first, and the order is load-bearing rather than stylistic.** With RY first, RZ is a
diagonal gate applied last, and a diagonal gate applied last cannot move a computational-basis
readout - the only things between it and the measurement are a CNOT ring, which permutes basis
states, and `|amplitude|^2`, which discards phase. The same argument retires the 8th angle of the
last conv round, which sits on the wire pooling discards and is thereafter only a CRY control.
Measured under the old order, 4 of these 55 angles could not move the readout at any input.

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
| `macro_f1` (this circuit) | 0.541894 |
| accuracy (companion, not the headline) | 0.585417 |
| logistic regression, identical split, macro F1 | 0.976447 |
| logistic regression accuracy | 0.977083 |
| uniform-random macro F1 floor | 0.125 |
| constant-predictor macro F1 floor | 0.027778 |
| majority-class accuracy floor | 0.152083 |
| classes predicted | 8 |
| n_train / n_val / n_test | 2240 / 480 / 480 |

**This model beats its floors and loses to classical logistic regression by 0.434553 of macro F1.**
The LR figure is identical to the 8-wire run's because it never sees the circuit, which is a useful
control: an unchanged baseline is evidence the data path did not move.

Per class, from this run:

| class | F1 |
|---|---|
| detector_blinding | 0.993197 |
| photon_number_splitting | 0.941176 |
| combined | 0.720588 |
| rng_manipulation | 0.635135 |
| intercept_resend | 0.440678 |
| trojan_horse | 0.391608 |
| normal | 0.212766 |
| wavelength_trojan | 0.0 |

**The zero moved class.** Before the dead parameters were made live it was `rng_manipulation` that
scored exactly 0.0 here, and that class now scores 0.635135; `wavelength_trojan` took its place at
0.0. `n_classes_predicted` is 8 either way, so the label is emitted and never lands on a true row. A
class scoring zero is a property of this circuit rather than of that class, which is a thing only a
per-class table can show and no aggregate can.

**Seed position:** unmeasured - no seed sweep has been run for the 7-wire circuit on either Round 47
circuit, so every figure on this card is one draw and not a median. The 8-wire card's own three-seed
spread is superseded for the same reason: it was measured before the dead parameters were made live.
Neither side of the width comparison has a spread on the current circuit, which is a limit on both.

## The width conclusion has reversed, and this is the side that gained

`references/qcnn-width-ablation.md` concluded in Round 40, and restated at five seeds in Round 45,
that 8 wires beat 7 on telecom by +0.0786 macro F1. With the corrected circuit **the sign has
flipped**: 7 wires 0.541894, 8 wires 0.510963, so this model leads by 0.030931. That is a reversal of
the recorded finding, not a restatement of it, and the ablation file carries a supersession banner
saying so.

**The reversal survived a change that moved both sides.** At the first Round 47 retrain the pair read
7 wires 0.500452 against 8 wires 0.458324, a lead of 0.042128; making the dead parameters live raised
both widths and left the narrower one ahead, by 0.030931. Two independent retrains agreeing on the
sign is more than one, and it is still not a separation.

Both sides remain single-seed. A flipped sign measured twice without a spread is a reason to stop
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
0.43 of macro F1, which is over ten times the gap between them. The corpus is synthetic QKD-attack
telemetry generated by `qsh.qml_ids.attacks` with class structure per Al-Kuwari et al.
(arXiv:2509.14282); no commercial-OK real attack corpus exists, which is why no number here is a
performance claim about a real link.
