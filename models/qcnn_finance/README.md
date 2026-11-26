# Variational QCNN - finance

> **Since Round 86's batch 6 this is the AMLworld circuit of a pre-registered comparison** (the owner's choice of
> 2026-11-26): eight wires reading eight of IBM AMLworld HI-Small's 114 computed inputs, the seed best on validation of
> twenty (`references/finance-qcnn-amlworld-comparison-preregistration.md`). Headline `minority_class_f1` = **0.509493**
> on the 1,523,504 transactions of the test window, at the threshold best on the validation window; logistic regression
> on the same encoded inputs scores 0.496783 at its own. It replaces the PaySim circuit of Round 84 (four wires, PaySim's
> four columns), superseded below.

NOT PUBLISHED TO HUGGING FACE, AND NOT IN THE MODEL REGISTRY. This is a browser-side demonstration circuit for the
`finance_aml` page of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid). It is not
one of the tiered published models, so it carries no `hf_repo` and no pinned revision.

## What it is

A tree QCNN on eight wires (`--arch tree --layers 3`): each input encoded as RY of 0.5 pi tanh(z), z its value
standardised by the fit window; three stages, each of three rounds of a shared 8-angle two-qubit filter on disjoint pairs
(72 angles) and a controlled-RY pool (3); two final rotations on the last survivor (2), whose Z is read through a scale
and a bias. 77 trained angles. Its state never
exceeds a Schmidt rank of 8 of the 16 an eight-wire state can reach (the comparison's arm 6, every seed, every cut and
layer, traced in double precision), and it is classically simulable: the page evaluates it as a tree tensor network,
and the comparison found its contraction equal to the state vector within 2.1e-6 on every test transaction. Nothing here
is a claim of quantum advantage.

The eight inputs, chosen on the fit window by |AUC - 0.5| before any circuit was trained
(`references/finance-qcnn-amlworld-preregistration.md`): `format_risk`, `pair_log_same_direction`, `pair_first_contact`,
`dst_in_uniq_4320`, `dst_bank_diversity`, `src_out_uniq_60`, `src_out_mean_60`, `src_out_mean_360`.

**What the served file changes from the trained one, and what that changes.** The circuit was trained on the fit
window's standardised inputs, clipped at 10. This file carries that standardisation as its `feature_mean` and
`feature_std` and declares `input_space: "raw"`, so the page and the Python tier feed a transaction's raw values, as on
every other page; the clip is not carried, and on the test window 864 transactions have a clipped input, whose readout
it can move by at most 3.7e-9, changing no decision. The bias carries the decision threshold: the logit that maximises
minority-class F1 on the validation window, 1.441270, so a positive logit means laundering; at the trainer's own logit of
0, set by a fit sample of one laundering transaction to four others, the circuit scored minority-class F1 0.063712. The
served circuit's decisions equal the trained circuit's at that threshold on all 1,523,504 test transactions.

## Measured performance

On the test window (the published MLP's temporal split; 2,321 laundering transactions in 1,523,504, a prevalence of
0.001523), at the validation-best threshold:

| metric | value |
|---|---|
| test_minority_f1 | 0.509493 |
| test_auprc | 0.442021 |
| test_accuracy | 0.998864 |
| classical_lr_minority_f1 | 0.496783 |
| classical_lr_auprc | 0.369531 |
| majority_baseline | 0.998477 |
| all_positive_f1 | 0.003042 |

Accuracy above the majority baseline says little at this prevalence; minority-class F1 and average precision are the
figures. The classical surrogate, logistic regression on the circuit's 6,561 Pauli products, scores minority-class F1
0.262797 and average precision 0.103846 at the same rule. It is fitted on one thread: on several, the solver's sums run
in an order that follows the machine's load, and two such runs gave 0.271299 and 0.274374.

**Seed position:** first of the twenty on validation, where the rule chose it, and first on test as well: test average
precision 0.4420 against the twenty seeds' median 0.2645 and mean 0.2530 (sample sd 0.0900). It is the top of the
spread, not a typical seed.

## Against the comparison's other arms

The comparison (`docs/measurements/finance-qcnn-amlworld-comparison-2026-11-26.json`, and the finance page's comparison
card) sets this seed beside the twenty seeds' spread (average precision 0.253 +- 0.090), the twenty seeds' mean
probability, a circuit with a quantile encoding encoded twice, the circuit without its entangling gates, three quantum
kernels with a support vector machine, logistic regression, gradient-boosted trees, an RBF support vector machine and a
matrix-product-state classifier, every one on the same eight inputs and fit sample, each with a 95 percent bootstrap
interval. This seed is the one the pre-registered rule ships, the best of twenty on validation; the spread across seeds
is wide, so a seed is not the method. Beside the arms, labelled as not one of them (the owner's choice of 2026-11-26,
after the record was first written), the card sets logistic regression on the circuit's own tanh-encoded inputs, the
baseline in the table above: average precision 0.3695 (0.3474 to 0.3917), behind this seed (difference 0.0586 to 0.0857)
and ahead of the twenty seeds' mean probability, 0.2956 (0.2760 to 0.3151). The arms' logistic regression reads the
standardised inputs and scores 0.0350.

## Data and licence

IBM AMLworld HI-Small (Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*,
NeurIPS 2023 Datasets and Benchmarks; Kaggle), **CDLA-Sharing-1.0**, https://cdla.dev/sharing-1-0/. SYNTHETIC
transactions: IBM generates them with a multi-agent virtual world. Trained weights are Results under CDLA-Sharing-1.0
(sections 1.11 and 3.5), so no share-alike attaches to them. The page's held-out sample,
`web/public/data/finance/qcnn_heldout.json`, holds the circuit's eight inputs for the 8,192 test-window transactions the
page already ships, and is Enhanced Data under the same agreement.

## Scope

Simulated state-vector evaluation and its tree contraction, not quantum hardware, and not a claim of quantum advantage.
On these eight inputs the circuit's seed scores above the comparison's classical models; the published finance models,
which read all 114 inputs, score far higher (the gradient-boosted teacher's average precision is 0.6428), and they are
the page's models of record.

## Reproducing

    python scripts/data/aml_qcnn_inputs.py
    python scripts/train/finance_qcnn_comparison.py --arm qcnn1 --seeds 0-19
    python scripts/check_finance_qcnn_export.py
    python scripts/export_finance_qcnn_amlworld.py

The first writes the eight inputs and the split; the second trains arm 1's twenty seeds (`qcnn_variational.py --usecase
finance_amlworld --layers 3 --epochs 120 --qubits 8 --arch tree --encoding-scale 0.5 --select-on ap`, seed 6 here); the
third checks the page's contraction against the state vector on every test transaction; the fourth writes this file,
the page's copy, the held-out sample with its expected figures and `docs/measurements/finance-qcnn-page-export-2026-11-26.json`.

## Superseded

* **The PaySim circuit, Round 84 to Round 86's batch 6** (four wires, PaySim's four columns `amount`, `cross_border`,
  `night_tx`, `new_payee`; CC BY-SA 4.0): minority-class F1 0.228571 against logistic regression's 0.439024 on its split.
  The page's other finance models moved to AMLworld with F7b, and the QCNN read a PaySim transaction of its own until
  this circuit replaced it.
* **0.735294** (eight PaySim columns at 8 wires, Round 47 to Round 84), two of them built from PaySim's balance columns,
  which its description says must not be used for fraud detection.
