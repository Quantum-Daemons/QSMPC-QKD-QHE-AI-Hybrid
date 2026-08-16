# Does the entangling ring do anything? Measured, 2026-08-13

> **EVERY NUMBER IN THIS FILE WAS MEASURED ON A CIRCUIT THIS PROJECT DELETED IN ROUND 46, AND THIS
> IS THE FIRST BANNER THIS FILE HAS EVER CARRIED.**
>
> Its width-ablation sibling, `qcnn-width-ablation.md`, was banner-marked in Round 46 when the
> circuit was replaced and again in Round 47 when the readout was corrected. This file measures the
> same trainer, was superseded by the same two changes, and was marked by neither - so for two rounds
> it described a deleted circuit in the present tense as the thing the repository ships. That is the
> scope defect this project keeps finding: a rule applied to one member of a pair.
>
> **What changed under it.** Round 46 replaced the hardware-efficient ansatz - RY on every wire plus
> a ring of CNOTs, PennyLane's `BasicEntanglerLayers` - with a real QCNN: one SHARED two-qubit filter
> applied at every position, pooling by a rotation on the survivor conditioned on its partner,
> register reduction, then a fully-connected block. Round 47 corrected the multiclass readout to bin
> the class index over the SURVIVING wires rather than over wires 0..m-1, and made the
> fully-connected block a per-survivor unitary. All six artifacts were retrained across the two
> rounds.
>
> **THE ABLATION IS NO LONGER EXACT, which is a stronger objection than the numbers being old.**
> `--no-entangle` gates two things on the current circuit: the CNOT inside the shared conv filter,
> and the ring across the survivors in the fully-connected block. It does NOT gate the pooling stage,
> which is a controlled rotation (CRY) and therefore entangling and is applied whatever the flag
> says. On this circuit `--no-entangle` produces a REDUCED-entanglement circuit, not a product-state
> one, so the clean comparison this file rests on - "what remains is per-qubit rotations on a product
> state" - is not available without changing the trainer. Re-running the ablation means deciding what
> the ablation now means first.
>
> Nothing below is deleted, and none of it is re-measured. Do not quote 0.714286, 0.870968,
> 0.809846, 0.622951 or any ratio in this file as a property of the shipped circuit. The current
> published metrics are in `models/*/weights.json` and on the six model cards, which are gated
> against them: telecom macro F1 **0.510963**, telecom at 7 wires **0.541894**, finance
> minority-class F1 **0.735294**, medical accuracy **0.883721**. (Round 47 retrained twice: the
> intermediate figures 0.458324 / 0.500452 / 0.84058 / 0.860465 are superseded by a second retrain in
> the same round, after two classes of parameter turned out to be analytically dead. Do not quote
> those either.)
>
> **The one conclusion that is untouched** is the one this file shares with `REFERENCES.md`: the QCNN
> loses to logistic regression on every use case, before and after both changes. Entanglement
> mattering to a model that loses is not evidence of advantage, which is what the closing section
> already says.

Bowles, Ahmed and Schuld (arXiv:2403.07059, 12 quantum models across 160 datasets) report that
**"removing entanglement from a quantum model often results in as good or better performance,
suggesting that 'quantumness' may not be the crucial ingredient"**. That is a claim about someone
else's circuits. This document checks it on the circuit this repository actually ships.

The ablation was exact rather than approximate **on the circuit it was run against**, and that
sentence is written in the past tense because Round 47 found it had been left in the present for two
rounds after it stopped being true. As measured in Round 40, `scripts/train/qcnn_variational.py`
entangled in exactly one place - a ring of CNOTs applied after each layer's per-qubit RY rotations -
and `--no-entangle` skipped that ring and changed nothing else. Same parameter count, same encoding,
same optimiser, same seeds, same splits. What remained was per-qubit rotations on a product state: a
classical feature map wearing the same shape.

**On the circuit the repository ships today none of that paragraph holds.** There is no CNOT ring
after per-qubit RY rotations, because there are no per-wire RY layers: the trainer applies one shared
two-qubit filter, then pools, then runs a fully-connected block on the survivors. `--no-entangle`
gates the filter's CNOT and the survivor ring in that block; the pooling CRY is entangling and is not
gated. So the flag no longer produces a product state and the ablation is no longer exact. The
banner at the top of this file states the consequence: these numbers are a record, not a current
measurement.

## Result: entanglement is load-bearing here, on all three use cases

Five seeds each on the two binary models, both configurations, minority-class F1 on the held-out
split:

| use case | entangled, median | no-entangle, median | ratio | entangled wins |
|---|---|---|---|---|
| finance | **0.714286** | 0.048544 | **14.71x** | **5 of 5** |
| medical | **0.870968** | 0.833333 | 1.05x | **4 of 5** |

Ranges, because the medians alone would hide how noisy this is:

| use case | entangled range | no-entangle range |
|---|---|---|
| finance | [0.324324, 0.800000] | [0.033333, 0.077586] |
| medical | [0.622951, 0.933333] | [0.688525, 0.862069] |

Telecom, the 8-class model, at its shipped configuration (`--layers 6 --epochs 400`, seed 0),
macro F1: **0.809846 entangled against 0.467059 without**, with 8 of 8 classes predicted either way.
One seed only - a telecom run is 6 layers x 400 epochs, so a five-seed sweep there was not run and
this row should be read as a single draw.

**So the Bowles result does not reproduce on this circuit.** On finance the entangling ring is the
difference between a working classifier and a broken one; on telecom it is worth 0.34 of macro F1;
on medical it wins on four seeds of five, though by a margin (1.05x) that five seeds cannot really
resolve. Nothing here contradicts their paper - they surveyed 12 models over 160 datasets and
reported a tendency, and one repository's circuit on three tasks is not a counterexample to a
tendency. It is a statement about this circuit.

## The correction that matters more than the result

**The first version of this ablation ran ONE seed and concluded the opposite for medical.** At
seed 0, entangled scores 0.622951 and the ablation scores 0.833333 - so the single-seed reading was
"the entangling ring is actively harmful on medical, removing it improves minority F1 by 0.21".
That was about to be written down as a finding.

Five seeds show it is an artefact:

| seed | entangled | no-entangle | winner |
|---|---|---|---|
| 0 | 0.622951 | 0.833333 | ablation |
| 1 | 0.892308 | 0.862069 | entangled |
| 2 | 0.870968 | 0.771930 | entangled |
| 3 | 0.738462 | 0.688525 | entangled |
| 4 | 0.933333 | 0.833333 | entangled |

Seed 0 is the **worst of the five** entangled draws and one of the better ablated ones. It is the
single seed at which the conclusion inverts, and it is the seed the shipped model uses.

## Second finding: the shipped medical artifact is an unlucky draw

Reading the same table the other way:

| use case | shipped (seed 0) | median over 5 seeds | rank of shipped |
|---|---|---|---|
| finance | 0.714286 | 0.714286 | 3 of 5 - exactly the median |
| medical | 0.622951 | 0.870968 | **1 of 5 - the worst** |

The medical model card publishes 0.6230 for a circuit whose median over five seeds is 0.8710. That
figure is not wrong - it is what seed 0 produces, and it is reproducible - but it is not
representative, and a reader comparing it to the classical baseline (0.939394) is comparing against
the unluckiest of five draws.

**The fix is NOT to retrain on a better seed.** Choosing a seed by test-set performance is test-set
selection, which is the defect this repository refuses elsewhere; the trainer already selects its
best EPOCH on a validation split precisely to avoid that. What is defensible is reporting the
spread. A single seeded number with a [0.6230, 0.9333] range behind it should carry that range.

**The medical card no longer publishes 0.6230**, and no five-seed spread exists for what it does
publish. After the Round 46 circuit replacement and Round 47's two corrections - the
fully-connected-block rewrite and then the dead-parameter reordering - the shipped medical artifact
reports accuracy **0.883721** with minority-class F1 **0.84375** at seed 0,
against the same logistic regression's 0.953488 / 0.939394. Whether seed 0 is still an unlucky draw
is UNKNOWN on the current circuit: the seed sweep in this file was never re-run, so the argument
above stands as a method and its numbers do not stand as a measurement.

## Reproducing this

    python scripts/train/qcnn_variational.py --usecase medical --seed N            # entangled
    python scripts/train/qcnn_variational.py --usecase medical --seed N --no-entangle

`training.entangle` is recorded in every artifact, so a file produced by an ablation run cannot be
mistaken for a shipped one. Training is deterministic per seed since the multiclass readout stopped
using `index_add_` (see the note in the trainer): repeated runs of the same command reproduce every
recorded digit.

## What this does and does not license saying

- **Supported:** on this circuit, on these three tasks, the entangling ring carries a large part of
  the result, and removing it is not free.
- **Supported:** the QCNN still loses to a classical baseline on every use case. That is unchanged
  by this ablation and is the point PRX Quantum 7, 020304 (2026) makes structurally - see
  `REFERENCES.md`.
- **NOT supported:** any claim of quantum advantage. Entanglement mattering to a model that loses is
  not the same as the model being worth using.
- **NOT supported:** a general statement about QCNNs. Three tasks, one circuit, five seeds on two of
  them and one seed on the third.
