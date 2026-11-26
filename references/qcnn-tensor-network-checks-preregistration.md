# Pre-registration: tensor-network checks of the medical, telecom and satellite QCNNs (2026-11-26)

Written and committed before any figure below is computed, on the owner's choice of 2026-11-26 to add to each trained
QCNN outside finance what `references/finance-qcnn-amlworld-comparison-preregistration.md` adds to finance's: a check
of the circuit's classical simulability, a matrix-product-state classifier on the same inputs as a comparator, and a
statement on its card that the circuit is classically simulable and no quantum advantage is claimed. Nothing here is
changed after a figure is seen; a change is an addendum with its date and reason. No model is retrained or replaced by
anything below.

## What it is and is not

Every circuit here acts on seven or eight wires, so every state it reaches is a vector of at most 256 amplitudes that
a laptop holds exactly, and the tree circuits are tree tensor networks by construction (Grant et al., npj Quantum
Information 4, 65, 2018). QCNNs on classical data of this kind are classically simulable (Bermejo et al., PRX Quantum
7, 020304, 2026). The checks below measure how much entanglement each trained circuit actually builds and how a
classical tensor network of the same input fares; neither is a test for quantum advantage, and none is claimed.

## The artifacts

The shipped seed of each, as its weights file records it, on the split and features its trainer used
(`scripts/train/qcnn_variational.py`, `load` and `split_for`, seed 0):

| key | file | circuit | wires | classes |
|---|---|---|---|---|
| medical | `models/qcnn_medical/weights.json` | tree | 8 | 2 |
| telecom | `models/qcnn_telecom/weights.json` | Hur, Kim and Park's SU(4) | 8 (7 features) | 8 |
| telecom_q7 | `models/qcnn_telecom_q7/weights.json` | Hur, Kim and Park's SU(4) | 7 | 8 |
| satellite_opssat | `models/qcnn_satellite_opssat/weights.json` | tree | 8 | 2 |
| satellite_esa | `models/qcnn_satellite_esa/weights.json` | tree | 8 | 2 |

The weights file's `feature_mean` and `feature_std` must equal the fit rows' moments recomputed from the loader within
$`10^{-9}`$; otherwise the artifact's rows are not these rows and the check stops for that artifact.

## 1. The simulability check

For each artifact, on its test rows (all of them, or the first 1,000 of a seed-0 permutation of them where there are
more), the circuit is run with its trained angles and the state is read after the encoding, after every convolution
round, after every pooling stage and after the fully connected block. At each point the Schmidt rank of every row's
state is computed across each cut of the wire order (wires $`0..k`$ against the rest, $`k = 0, \ldots, n - 2`$),
counting singular values above $`10^{-10}`$ of the largest. Reported: the largest rank per point over rows and cuts,
beside the bound $`2^{\lfloor n / 2 \rfloor}`$ every $`n`$-wire state meets (16 at eight wires, 8 at seven). Where every
test row is used (all but `satellite_esa`), the QCNN's test figure is recomputed from the same forward pass, on the
CPU, through the artifact's own head ($`z \cdot \mathrm{scale} + \mathrm{bias}`$, then the sign or the arg-max), and
reported beside the recorded one, as a check that the circuit traced is the circuit shipped.

## 2. The MPS comparator

A matrix-product-state classifier (Stoudenmire and Schwab, NeurIPS 2016), as finance's arm 4 defines it: each input
$`z`$ mapped to $`x = \hat F(z)`$, its empirical distribution on the fit rows, and encoded as
$`[\cos(\pi x / 2), \sin(\pi x / 2)]`$; one core per input, in the feature order the circuit uses; bond dimension from
$`\{2, 4, 8\}`$; trained by Adam (learning rate 0.01, batches of 256, 50 epochs, seed 0). A binary task ends in one
logit on the logistic loss; telecom's eight classes end in eight logits on the cross-entropy loss, the last core
carrying the class index. The bond dimension is chosen on the validation rows: by average precision for a binary task,
by macro F1 for telecom.

Reported on the test rows, beside the QCNN's recorded figures and its logistic-regression comparator's: for a binary
task, the accuracy and the minority-class F1 at a logit of 0, the average precision, and the minority-class F1 at the
threshold best on validation; for telecom, the accuracy and the macro F1 at the arg-max.

## 3. The cards

Whatever the figures, each QCNN's card on its page states that the circuit is classically simulable, that no quantum
advantage is claimed, the largest Schmidt rank of section 1 beside its bound, and the MPS comparator's headline figure
beside the QCNN's and the logistic regression's, saying which is higher. Nothing ships in place of any model.

The record is `docs/measurements/qcnn-tn-checks-2026-11-26.json`, written by `scripts/train/qcnn_tn_checks.py`, with
the page's summary in `web/public/data/qcnn_tn_checks.json`. Everything runs on the CPU.
