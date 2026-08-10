# Are these the models the field actually encrypts? Finance, checked against 2026 literature
### Round 24k, 2026-08-10

The question was whether this project's published models are credible representatives of
privacy-preserving inference practice, or whether the field has moved on and the demo is showing
something dated. Answered per vertical. Finance is done; the rest follow as their checks complete.

## Finance: the choices are validated, and more strongly than expected

Across verified 2025-2026 papers on encrypted inference labelled to finance, the architecture
distribution is lopsided:

| architecture class | papers found |
|---|---|
| tree ensembles (GBDT / XGBoost / random forest) | ~10 |
| logistic regression, SVM, scorecard | ~5 |
| small MLPs | ~2 |
| **transformers** | **0** |
| **sequence models over transaction streams** | **0** |
| **tabular foundation models (TabPFN line)** | **0** |

This project publishes an XGBoost teacher and a 33-parameter logistic student under CKKS. That is
squarely what the field evaluates, and the absence at the bottom of that table is not for want of
looking: encrypted-transformer machinery is mature (EncFormer arXiv:2604.09975; AEGIS
arXiv:2604.03425) and tabular foundation models are mature, but every verified encrypted-deep-model
paper benchmarks NLP or vision - GLUE, SST-2, Rotten Tomatoes, CIFAR - never a finance tabular or
transaction benchmark.

**The papers give three reasons themselves, and they reinforce each other.**

1. **Accuracy.** GBDTs still win on tabular data. IEEE S&P 2026 (arXiv:2605.26903) states it
   plainly: *"High speed and interpretability make GBDTs popular in finance and healthcare, where
   neural networks may fall short."*
2. **Cost.** A 2026 head-to-head (arXiv:2605.04858) finds FHE's benchmarked strength is regressions
   while MPC wins for CNNs - which is why this project runs CKKS for the linear tier and garbled
   circuits for the non-linearities rather than the other way round.

   **A correction to an earlier draft of this file.** It cited arXiv:2211.06675 as "6 ms encrypted
   for the tree ensemble against 296 ms for the neural network". Reading the PDF rather than the
   abstract, **the 6 ms path is not FHE at all**: it is PPXGBoost, order-preserving encryption plus
   Paillier, and OPE leaks the total order of feature values to the server. Only the 296 ms neural
   path is real FHE (CKKS via TenSEAL). Including transaction encryption the totals are 180 ms and
   306 ms, not 6 ms. The widely-repeated "6 ms encrypted fraud inference" is therefore a comparison
   between two different security models, and this project should not repeat it.

   The defensible cost anchors, all on stated hardware:

   | system | configuration | latency |
   |---|---|---|
   | Zama Concrete ML / TFHE (MSPN 2023) | 50 trees, depth 5, 6-bit | **0.95 - 81.9 s** per inference (477x - 40,969x slowdown) |
   | SilentWood (arXiv:2411.15494 v2) | 100 trees, depth 7, BFV via SEAL | **4.37 s** CPU / **1.97 s** GPU |
   | Kangaroo (NDSS 2026) | 969 trees, 411,825 nodes | **~60 ms per tree**, amortised |

   Note the Zama paper states no CPU model at all - "8 CPU cores" is the whole hardware disclosure
   for the baseline everyone benchmarks against.
3. **Regulation.** *Scientific Reports*, 2 Jan 2026 (DOI 10.1038/s41598-025-34536-9) picks logistic
   regression explicitly *"to align with widely adopted interpretable credit-risk modelling
   practices"*, and uses CKKS via TenSEAL at **poly_modulus_degree 8192** with coefficient chain
   [60,40,40,60] and scale 2^40 - the same parameters this project runs. It reports HE-federated
   accuracy ~90% against ~91% plaintext, with CKKS drift under 1%.

## Three findings that bear on decisions already taken here

**The integer scorecard was the right shape even though it lost.** Round 24h fitted one, measured it
losing to the CKKS student on net benefit, and shipped the implementation without the model. The
literature says the form is what the field uses for exactly the reason recorded there:
interpretability, not accuracy. Stronger still, IACR ePrint 2025/2188 (ALIOTH, Dec 2025) builds a
**2PC secure Weight-of-Evidence binning** protocol - the classic credit-scorecard feature transform,
and the same quantile-binning step this project's scorecard implements - transforming 3 million
samples by 100 features in under 30 minutes over WAN, and reports it *"improves both the predictive
performance of logistic regression and 2PC training efficiency"*. So the binning is not a
simplification this project invented to make BFV work; it is what the secure-computation literature
is optimising.

**BFV for the exact tier is corroborated.** SilentWood (arXiv:2411.15494, v2 Jan 2026) evaluates
gradient-boosted forests under **BFV via SEAL**, reporting 42.5x over RCC-PDTE and 27.8x over Zama's
Concrete ML XGBoost. Kangaroo (arXiv:2509.03123, NDSS 2026) also uses packed BFV. The scheme choice
recorded in `he-scheme-selection.md` matches practice.

**The AML graph-feature gap has a measured size, and the headline it is usually quoted with is on a
rigged base rate.** arXiv:2411.02926 (SPACE 2024) runs XGBoost plus IBM's Graph Feature Preprocessor
under TFHE on AMLworld HI-Small - the same corpus this project uses - and its widely-cited ">99%
accuracy and F1" holds only on a **50%-illicit balanced subset**. On its own imbalanced subset at
5.72% illicit the same pipeline gets **F1 0.31-0.39 with recall 0.19-0.25**, and the advertised
"+8% F1" is eight percentage points off a 0.31 base. This project's corpus is 1-in-981, i.e. 0.1%,
which is more adverse again - so that paper is evidence about the size of the graph-feature
contribution (single-digit points, not a factor) and **not** a target to be measured against.

Two further details worth carrying: its FHE latency is **~10.5 s per transaction at a 119,926x
slowdown**, and **its GNN pipeline failed to compile to FHE entirely** - only the XGBoost path
worked. Both are useful when anyone proposes an encrypted graph model for this use case.

## What is genuinely absent, stated as a non-goal rather than a gap

No verified paper evaluates an encrypted transformer or tabular foundation model on a finance
benchmark - IEEE-CIS, ULB, German Credit, Taiwan default, LendingClub. The components exist
separately and nobody has joined them. That is a real research opening, and it is **not** something
this project should chase: an encrypted transformer at ring degree 8192 is not a browser workload,
and the demo's claim is that its cryptography is real rather than that its models are the largest.

## Where the frontier actually is

Not architecture. The 2026 deltas are all efficiency and threat model: hiding record identities
(anonymous GBDT, IEEE S&P 2026), removing server-to-server communication (PVODTE, IEEE TDSC 2026),
amortising over very large forests (Kangaroo, 969 trees at ~60 ms/tree over WAN), making *training*
non-interactive rather than inference (PoPETs 2026, 360x speedup for XGBoost training over FHE),
proving training occurred (ZKBoost, IACR 2026/202, from JPMorgan AI Research), and securing the
preprocessing (ALIOTH). The model class is held fixed across all of them.
