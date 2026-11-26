# Pre-registration: the finance QCNN on AMLworld, a fair comparison (2026-11-26)

Written and committed before any run below, on the owner's choices of 2026-11-26: re-check the first pre-registration's
result (`references/finance-qcnn-amlworld-preregistration.md`), pre-register an improved QCNN and a quantum-kernel SVM
beside classical comparators of the same inputs, and check the circuits' classical simulability with tensor-network
quantities. Nothing here is changed after a figure is seen; a change is an addendum with its date and reason. What is
known when writing it: the first run's figures (`docs/measurements/finance-qcnn-amlworld-2026-11-26.json`), and the
re-check's finding that its checkpoints and seed were selected by validation F1 at a logit of 0, a threshold the
four-to-one fit sample sets, not the validation window's prevalence (0.113 percent; the test window's is 0.152).

## What it is and is not

A tree QCNN of eight wires is exactly simulable on a classical computer, and with one encoding per wire its readout is a
linear combination of the $`3^8`$ products of one, the sine and the cosine of each wire's angle (Schreiber, Eisert and Meyer, PRL 131, 100803,
2023); QCNNs on classical data of this kind are classically simulable (Bermejo et al., PRX Quantum 7, 020304, 2026).
Nothing here is a claim of quantum advantage, and the page says so.

## Data

The first pre-registration's, unchanged: the eight AMLworld inputs it chose on the fit window, standardised and clipped,
in its ranking order, input $`j`$ on wire $`j`$; the fit sample (every laundering transaction of the fit window and four
others for each, 10,275 rows); the validation and test windows whole. Nothing is chosen on the test window.

## The arms

1. **QCNN-1, selected by average precision.** The first run's circuit and recipe (`--arch tree --layers 3 --epochs 120
   --encoding-scale 0.5`, RY$`(c \pi \tanh z)`$), seeds 0 to 19, each run's checkpoint chosen by validation average
   precision rather than F1. On the GPU (the owner's approval of 2026-11-26).
2. **QCNN-2.** The same tree with each input encoded as RY$`(\pi \hat F(z))`$, $`\hat F`$ its empirical distribution on
   the fit sample, and encoded twice: again after the first convolution round, before any pooling. Seeds 0 to 19,
   checkpoints by validation average precision, on the GPU.
3. **Quantum kernels with a support vector machine**, on the CPU, on the angles $`\phi_j = b \pi \hat F(z_j)`$ with a
   bandwidth $`b`$: the fidelity kernel of the product encoding RY$`(\phi_j)`$ (no entanglement); the fidelity kernel of
   Havlicek et al.'s ZZ feature map (Nature 567, 209, 2019) on the same angles, two repetitions, each wire coupled to the
   next; and the projected kernel of that ZZ map (Huang et al., Nat. Commun. 12, 2631, 2021), $`\exp(-\gamma \sum_j
   \lVert \rho_j - \rho'_j \rVert_F^2)`$ over the wires' reduced states, $`\gamma`$ the inverse of the median of that sum
   on the fit sample. The bandwidth from $`\{0.25, 0.5, 1\}`$ and the penalty from $`\{0.1, 1, 10\}`$, chosen by
   validation average precision; the share of off-diagonal kernel values within 0.01 of their mean is reported as the
   concentration check.
4. **Classical comparators** on the same eight inputs and fit sample: logistic regression; gradient-boosted trees
   (scikit-learn's histogram booster, 300 rounds, learning rate 0.05, depth from $`\{3, 5\}`$ by validation average
   precision); an RBF support vector machine on the standardised inputs (width from $`\{0.1, 1, 10\}`$ times the inverse of the
   input count, penalty from arm 3's grid, both by validation average precision); and a matrix-product-state classifier
   on the encoding $`[\cos(\pi x / 2), \sin(\pi x / 2)]`$ of each $`x = \hat F(z)`$ (Stoudenmire and Schwab, NeurIPS 2016),
   bond dimension from $`\{2, 4, 8\}`$, trained by Adam (learning rate 0.01, batches of 256, 50 epochs, seed 0) on the
   logistic loss, the bond dimension by validation average precision. The
   logistic regression and the booster also run on the whole fit window, every one of its 2,843,872 transactions.
5. **No entanglement.** Arm 1's circuit without its entangling gates, seeds 0 to 4, on the CPU.
6. **The simulability check.** For each arm-1 seed, the largest Schmidt rank of the eight-wire state across every cut
   of the wire order, after each layer, on 1,000 test rows, beside the bound of 16 every eight-qubit state meets.

## Metrics

On the test window: average precision with a 95 percent bootstrap interval (1,000 resamples stratified by label), the
prevalence beside it; for arms 1, 2 and 5 the mean over seeds and the score of the seed ensemble (the mean of the
seeds' probabilities); the minority-class F1 at the threshold best on validation. Paired bootstrap differences between
the arm-1 seed that would ship and each other arm.

## Criteria for the page

1. The page's finance QCNN becomes the arm-1 seed with the best validation average precision if its test average
   precision is at least twice the prevalence and the first pre-registration's export criterion holds for it (the
   page's clear contraction equals the state vector within $`10^{-5}`$ on every test row, no decision differs). Arm 2
   is reported and does not ship: the page's three circuit implementations do not carry its encoding.
2. Whatever ships, the page states every arm's figure with its interval and the prevalence, says the circuit is
   classically simulable with no quantum advantage claimed, and says which comparators score above it.
