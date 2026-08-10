# Are these the models the field actually encrypts? Finance, checked against 2026 literature
### Round 24k, 2026-08-10

The question was whether this project's published models are credible representatives of
privacy-preserving inference practice, or whether the field has moved on and the demo is showing
something dated. Answered per vertical. All five are now checked.

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

---

# The other four verticals

| vertical | model | verdict |
|---|---|---|
| healthcare | ResNet teacher -> 3,226-param CNN student, CKKS | **defensible, and ahead of its direct baselines** |
| genomics | linear per-SNP PRS + secure summation | **model correct, privacy framing wrong** |
| satellite | Telemanom LSTM + CKKS-linear student | **defensible as stated; the danger is misstating it** |
| telecom | 8-class QKD IDS, RF + quantum-kernel SVC + QNN | **weakest, and the quantum half is in a regime documented not to work** |

## Healthcare: ahead of the two papers that do the same thing

This was expected to be the weakest defence and is the strongest. There are exactly two published
BreastMNIST-under-encryption baselines, and **both are simpler than what this project runs**:

| baseline | architecture | result | latency | ring |
|---|---|---|---|---|
| Al Badawi & Bin Yusof, *BioData Mining* 17:33 (Sep 2024) | RBF-kernel **SVM**, CKKS/OpenFHE | F1 0.80 | 2.09 s | **N=65536** |
| Selvakumar & Senthilkumar, *Sci Rep* (Jul 2025) | 2-layer quantized **FCNN**, TFHE | 81.82% -> 81.81% encrypted | **137.68 s/sample** | - |

Neither performs a genuine homomorphic convolution. This project does, at **N=8192**, one eighth the
ring degree the SVM baseline needed. That is a legitimate efficiency position, and it means the
medical use case should be presented against *these* papers rather than against crypto-systems work
like Orion or BootNet - the latter needs N=32768 for ImageNet and is a different claim entirely.

Worth knowing: an arXiv query for `MedMNIST AND homomorphic` returns **zero** results. MedMNIST is
used in medical-informatics venues and never in the security conferences, which is why the two
baselines above are easy to miss.

## Genomics: the model is right and the threat model was pointed at the wrong thing

The linear per-SNP weighted sum is unambiguously current - *Nature Communications*, Jun 2025, finds
neural models **outperformed by linear regression** across 28 UK Biobank traits. The nearest
comparator is HEPRS (*Cell Reports Methods*, Jan 2026): CKKS via Lattigo, a 110,258-SNP schizophrenia
score, r > 0.999 against plaintext at ~4.9 s and 3.3 GB per individual.

The problem is not the model. **Pooled allele counts are the canonical genomic privacy failure**, and
encrypting their computation does nothing about their release. Full treatment now in
`docs/threat-model.md`; the short version is that released allele frequencies support genome
reconstruction at 70% F1 against 53.4% for binary responses, and a single 27-locus polygenic score
uniquely identifies 95% of individuals in a 450k cohort.

One further correction: **no verified paper applies quantum-enabled MPC to allele counts, GWAS or
PRS.** The mature secure-genomics systems are classical and much larger in scope - SF-GWAS runs
409,548 UK Biobank individuals in 6 days; SECRET-GWAS does 1M patients by 4M SNPs in 4.5 minutes -
and secure summation is a cheap sub-step of those rather than the headline.

## Satellite: defensible, and the failure mode is describing it wrongly

The paradigm is alive: on ESA-ADB's own corrected event-wise F0.5, Telemanom-ESA-Pruned is the best
algorithm on Mission1 while unsupervised baselines score below 0.001. Two traps, both already
avoided here and both worth keeping avoided:

- **Never quote Telemanom's SMAP/MSL F1.** A uniform random score reaches F1 0.931 on MSL and 0.961
  on SMAP under point adjustment (AAAI-22). Checked: this project mentions SMAP/MSL only in
  `registry.py` as a warning about exactly that inflation, and never as a result.
- **This project does not run an encrypted LSTM, and must not imply it.** Depth accumulates per
  timestep, and 2026 work either abandons CKKS for TFHE programmable bootstrapping (ePrint 2026/1091,
  5.2 s per 128-token sequence) or bounds hidden states architecturally (ePrint 2026/1502). The
  encrypted tier here is a CKKS-**linear** student and the accuracy claim rests on that. Checked: the
  existing text distinguishes the encrypted student from the plaintext LSTM teacher correctly.

Encouraging for the linear student specifically: arXiv:2602.00672 (Jan 2026) finds closed-form linear
autoregression matches or beats state-of-the-art deep time-series anomaly detectors, so a linear
encrypted tier is defensible on the merits rather than only on cost.

## Telecom: the weakest vertical, and the quantum half is below its own threshold

Two findings, both of which this project should absorb rather than argue with.

**There is no public 8-class QKD-attack dataset.** Two papers use eight classes with *incompatible*
taxonomies - only three of seven attacks overlap - both on private simulated data. This project
generates its own and labels it `synthetic, methodology-grounded`, citing arXiv:2509.14282, which is
the honest framing. But the eight-class shape should not be described as a standard, because it is
not one.

**The quantum kernel is configured below the regime the literature says works.** arXiv:2605.31449
(May 2026) documents fidelity quantum kernels **exponentially concentrating**, beating that regime
only at **15 qubits or more**. This project's configuration is `q_n_qubits = 4` and
`quantum_qubits = 8`. The measured result is exactly what concentration predicts: quantum-kernel SVC
macro F1 **0.8316** and variational QNN **0.4667** against RandomForest **0.9866**, with
`quantum_beats_classical: false` already recorded.

So the quantum components are not underperforming through a bug - they are underperforming for a
published, quantified reason, at a qubit count that is itself the explanation. That is a better
finding than "quantum lost" and it belongs beside the number.

The defensible framing for ML here is narrow and real: a conventional 11% QBER threshold catches only
**25.82%** of attacks in one 2026 study, with ML dropping the false-negative rate from 0.8477 to
0.0198. ML as an overlay for sub-threshold stealthy attacks, never as a replacement for decoy-state
accounting.

## The one architectural suggestion worth taking

For healthcare, a **frozen pretrained backbone with a CKKS-encrypted linear head** fits this
project's exact budget: a ViT CLS token is 768-D, which sits comfortably inside 4096 slots, and one
ciphertext-plaintext dot product per class is precisely depth 1. The 2025 medical instantiation
exists and is measured - arXiv:2511.20983 reports **96.12% plaintext against 90.02% encrypted** at
326 KB per aggregation round. That would let this project say "ViT-class backbone" honestly while
keeping the encrypted step exactly as cheap as it is today.

Not adopted this round, and recorded as a recommendation rather than a plan, because the backbone's
licence and redistribution terms would need the same scrutiny the ImageNet question already got.
