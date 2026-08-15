# Does the unencoded wire do anything? Telecom QCNN at 7 wires against 8

Measured 2026-08-14, Round 40. Companion to `qcnn-entanglement-ablation.md`, which asks the same
question of the entangling ring.

## Why this was measured at all

The browser's **QCNN qubits** control did not reach the trained circuit. `clientCompute.ts` computed
the requested width and then loaded a fixed 8-wire weights file, so on every page that ships trained
weights the control moved the QSMPC-FHE benchmark row and the untrained fallback and nothing else. A
control that silently ignores you is the inert-control defect this project's checklist opens with -
Round 20 audited the use-case pages and found five.

Making it real needed a second width to exist. Telecom is the informative case: it carries **7
features on 8 wires**, so the eighth is padded with `RY(0)` and encodes nothing. That wire is either
free capacity or dead weight, and nobody had measured which.

## Setup

Identical corpus, split, seed, layer count and epoch budget; only the wire count differs. Both
commands are recorded inside their own weights files and both reproduce `theta` exactly:

    python scripts/train/qcnn_variational.py --usecase telecom --layers 6 --epochs 400 --seed 0 --threads 1 --qubits 8
    python scripts/train/qcnn_variational.py --usecase telecom --layers 6 --epochs 400 --seed 0 --threads 1 --qubits 7

The `--qubits` argument is new in Round 40; `N_QUBITS = 8` had been a module constant. **The first
7-wire run recorded a command without it**, which therefore reproduced an 8-wire model - a
reproduction line that does not reproduce is worse than none, because it is checked once and trusted
afterwards. Any argument that changes the weights is now in that string.

The refactor was proven inert before the comparison was trusted: re-running the shipped 8-wire
command through the new code path reproduced the committed model's `theta` **bit for bit** and its
macro F1 to all six recorded digits. Without that step, a width difference and a refactor difference
would have been indistinguishable.

## Result: the padding wire is worth 0.079 macro F1

| class | 8 wires | 7 wires | delta |
|---|---|---|---|
| detector_blinding | 0.9660 | 0.9306 | +0.0354 |
| photon_number_splitting | 0.9307 | 0.8113 | +0.1194 |
| normal | 0.8911 | 0.8352 | +0.0559 |
| intercept_resend | 0.8271 | 0.6271 | +0.1999 |
| rng_manipulation | 0.8070 | 0.6718 | +0.1353 |
| combined | 0.7360 | 0.7445 | **-0.0085** |
| wavelength_trojan | 0.7107 | 0.4673 | +0.2435 |
| trojan_horse | 0.6102 | 0.7619 | **-0.1517** |
| **macro F1** | **0.8098** | **0.7312** | **+0.0786** |

Both beat their floors by a wide margin (uniform-random macro F1 0.1250, constant-predictor 0.0278)
and both lose to classical logistic regression at 0.9764, which is unchanged between the runs because
it never sees the circuit - a useful control, since an identical LR number is evidence the data path
did not move.

## Reading it honestly

**The wire is not dead weight.** It carries a trainable `RY` angle and participates in the CNOT ring,
so it adds capacity even though it encodes no feature. On a 6-layer circuit that is six extra
parameters and one more qubit of entangling structure.

**But the effect is not uniform, and two classes prefer the narrower circuit.** `trojan_horse` is
0.15 BETTER at 7 wires and `combined` marginally so. A single macro number would have hidden that;
the per-class table is why it is reported per class. With one seed and one split, a 0.15 swing on one
of eight classes is within the range a different seed could produce, so the per-class deltas are
directional rather than settled - the macro difference is the claim, and the per-class column is
evidence about its shape.

**What this does NOT show.** Not that 8 is optimal - only 7 and 8 were run, because 7 is the feature
count and 8 is what shipped. Not that more wires keep helping; nothing here rules out a peak at 8.
And not a quantum-advantage claim of any kind: classical LR is ahead of both by a margin larger than
the gap between them, which is the number that matters for anyone deciding what to deploy.

## What ships

Both. `web/public/data/telecom/qcnn_weights.json` stays the 8-wire circuit and remains the default;
`qcnn_weights_q7.json` is loaded when the visitor selects 7. A requested width nobody trained falls
back to the shipped circuit, and `qcnn_qubits` then reports the width that **actually ran** rather
than the width that was asked for - so the page is honest about a request it could not honour.
Checklist row `PF10`.

*Reopens if* a third width is trained, which would make the "peak at 8" question answerable rather
than merely open; or if the seed sensitivity of the two reversed per-class deltas is measured, which
would turn the directional reading above into a settled one.

---

# Round 43: the same question on finance and medical, and the answer reverses at the median

## Why the widths differ per use case

Telecom was narrowed from 8 wires to 7 because it encodes **7 features**, so the 8-wire model's last
wire is padding. Finance and medical encode **8 features**, so 7 is not a width they have:
`scripts/train/qcnn_variational.py` exits with `8 features exceeds 7 wires`. The only spare-wire
question available on those two is the widening one, 8 to 9.

The plan for this round assumed a 7-wire twin could be trained for all three, mirroring telecom. It
could not. The floor is the feature count, and that is a property of the data rather than a choice.

## Result: at seed 0 the extra wire helps on both. At five seeds it helps on neither.

| use case | metric | 8 wires, 5 seeds | median | 9 wires, 5 seeds | median |
|---|---|---|---|---|---|
| finance | minority-class F1 | 0.7143 / 0.3243 / 0.8000 / 0.4615 / 0.7869 | **0.7143** | 0.8136 / 0.6800 / 0.4444 / 0.7937 / 0.6800 | **0.6800** |
| medical | accuracy | 0.7326 / 0.9186 / 0.9070 / 0.8023 / 0.9535 | **0.9070** | 0.8140 / 0.9186 / 0.8721 / 0.7907 / 0.9070 | **0.8721** |

Seed 0 alone reports **finance +0.099** and **medical +0.081** for the wider circuit. The five-seed
median reports **finance -0.034** and **medical -0.035**. The sign flips on both.

## Why the single-seed reading was wrong in the flattering direction, on both

For medical the mechanism is recorded and was measured in Round 42: **seed 0 is the worst of five for
the 8-wire model** (0.7326 against a median of 0.9070). A seed-0-to-seed-0 width comparison therefore
puts the new width's ordinary draw against the old width's unluckiest one. For finance the mirror
holds - seed 0 is the **best** of five at 9 wires (0.8136 against a median of 0.6800).

Neither is evidence of a bias in the circuit. Both are evidence that a five-seed spread of width
[0.32, 0.80] cannot be read at one point, which is what `QC20` and the `**Seed position:**` marker
exist to force.

## What this does NOT show

- **Not** that 8 wires beats 9. The spreads overlap heavily on both use cases. The defensible claim
  is that **this experiment does not separate them**, and that a claim in either direction needs more
  than five seeds.
- **Not** that telecom's +0.079 for its pad wire is refuted. That measurement was also single-seed
  and is recorded as such on its card; it is now the third of three single-seed width readings and
  the only one not re-run at five seeds. **Re-run it by Round 45** or restate it as one draw.
- **Not** a statement about quantum advantage. Classical logistic regression on the same split beats
  every circuit here: finance 0.8657 against 0.8136, medical 0.9535 against 0.8140.

## What ships

Both 9-wire circuits ship, as `web/public/data/{finance,medical}/qcnn_weights_q9.json`, and the
browser's QCNN width control now offers exactly the widths that ship - {8, 9} on finance and medical,
{7, 8} on telecom - instead of 2..14 with a silent fallback to 8. Shipping a width that loses at the
median is deliberate: the control is meant to reach a real circuit, and a visitor who moves it should
see that more wires is not free.
