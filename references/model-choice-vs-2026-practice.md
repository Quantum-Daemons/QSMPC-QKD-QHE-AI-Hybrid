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
arXiv:2604.03425) and tabular foundation models are mature, but every verified encrypted **transformer or tabular-foundation-model**
paper benchmarks NLP or vision - GLUE, SST-2, Rotten Tomatoes, CIFAR - never a finance tabular or
transaction benchmark. **Scoped in Round 55, because the unrestricted version was falsified by a
paper this same file cites 120 lines below:** TGHE (arXiv:2606.26664) runs a CKKS-encrypted GNN on
DGraphFin, which is a finance fraud graph. The claim that survives is about the architecture
classes the table above actually counts, and a universal quantifier is only ever as good as the
search behind it - the second time this round that reading has been forced.

**The papers give three reasons themselves, and they reinforce each other.**

1. **Accuracy.** GBDTs still win on tabular data **outside the small-data regime**, and the
   qualifier is load-bearing: TabPFN (*Nature* 637:319-326, 8 Jan 2025) beats them *"on datasets with
   up to 10,000 samples and 500 features"*, while its own authors concede that *"for larger datasets
   and highly non-smooth regression datasets, approaches such as CatBoost, XGB or AutoGluon are
   likely to outperform TabPFN"*. This project's finance corpus is far past that threshold, so the
   GBDT choice stands - but the unqualified version of this sentence stopped being true in Jan 2025.
   IEEE S&P 2026 (arXiv:2605.26903) states the operative half plainly: *"High speed and
   interpretability make GBDTs popular in finance and healthcare, where neural networks may fall
   short."*
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

## Then why is the shipped encrypted model a 33-parameter linear student? Browser latency, and nothing else

**Stated plainly here in Round 33, because neither of the two reasons this project had on record does the
work, and leaving the question unanswered flattered the choice.** The table above establishes tree
ensembles at about ten papers against linear at about five, and the cost anchors above are anchors for
encrypting **the teacher directly**: SilentWood at **4.37 s CPU / 1.97 s GPU** under BFV via SEAL, and
Kangaroo at about **60 ms per tree** amortised over 969 trees. And `references/he-scheme-selection.md`
independently confirms that `node-seal` 7.0.0 already exposes `bfv`, so the scheme the tree work uses is
available in the browser today with no new dependency. **The gradient-boosting teacher is therefore both
stateless at inference and encryptable in seconds - the two objections that would have justified a linear
student do not apply to it.**

Round 32 corrected finance's binding constraint to **statefulness at inference**, and that correction is
right about the object it describes and does not reach the student. Statefulness is the reason there is no
GNN: message passing needs the transaction's graph neighbourhood at inference time, so `finance_aml.gnn`
carries `hf_repo=None`, has no `models/metrics.json` entry, and was never trained - it is a reference
architecture kept as the published state of the art the GBM is measured against. **A reason not to ship an
untrained GNN is not a reason to ship a 33-parameter linear model instead of the trained GBM.**

The real reason is **client-side latency**, and it is worth saying without dressing:

- Both cost anchors above are **server** measurements. SilentWood's 4.37 s is a server CPU with SEAL
  linked natively; Kangaroo's amortisation is over a 969-tree forest across a WAN. The demo's budget is a
  **single-threaded WASM tab inside one page interaction**, and "seconds on a server" and "inside a page
  interaction in a tab" are not the same budget.
- The linear student's encrypted step is **one packed ciphertext-plaintext dot product at multiplicative
  depth 1**. A degree-8192 ring gives 4096 slots, and 33 coefficients fit in one ciphertext with room to
  spare - which is why the same budget had room for the satellite student's 209-term quadratic expansion at
  unchanged depth.
- A published cost for the alternative in this project's own tier does **not** exist here. No encrypted
  tree evaluation has been benchmarked in this browser, so the comparison is between a measured in-tab dot
  product and a server-measured forest, and it must be stated as such rather than as a measured
  head-to-head.

**So the defensible claim is narrow and true: the linear student is the model that fits a page
interaction, and the field's preferred model is the teacher above it, which this project publishes and
runs in plaintext.** The literature-versus-practice question this file was written to answer therefore
comes out as: the field encrypts the tree ensemble, this demo encrypts a distillation of it, and the
difference is the runtime rather than the modelling. That is a smaller claim than the section previously
implied and it is the one that survives checking.

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

**Re-verified Round 55 and the claim STANDS, which is worth recording because the abstract contradicts
it.** The abstract says quantization and pruning "were used to render the GNN FHE-compatible", which
reads like success; section 4.4 is unambiguous that no circuit was ever produced - "the absence of
support for the ScatterElements ONNX operator in Concrete ML posed a significant challenge during the
final compilation". Read the section, not the abstract. But cite it as a **2024 result, not a standing
impossibility**: DESIGN (arXiv:2507.05649) and TGHE (arXiv:2606.26664, IEEE ICWS 2026) have since run
encrypted GNN inference. Neither is end-to-end either - TGHE transmits edge attributes in plaintext for
its correction terms.

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
proving training occurred (ZKBoost, IACR 2026/202, with JPMorgan AI Research - the author list spans CNRS/IRIF, JPMorgan and Arizona State, so "from" over-attributed it), and securing the
preprocessing (ALIOTH). The model class is held fixed across all of them.

---

# The other four verticals

| vertical | model | verdict |
|---|---|---|
| healthcare | ResNet teacher -> 3,226-param CNN student, CKKS | **defensible, and ahead of its direct baselines** |
| genomics | linear per-SNP PRS + secure summation | **model correct, privacy framing wrong** |
| satellite | Telemanom LSTM + CKKS-linear student | **defensible as stated; the danger is misstating it** |
| telecom | 8-class QKD IDS, RF + quantum-kernel SVC + QNN | **weakest, and the quantum half is in a regime documented not to work.** Separately: **no encrypted tier existed at all** - see below, it is being added this round |

## Healthcare: ahead of the two papers that do the same thing

This was expected to be the weakest defence and it holds up, but **"exactly two" was wrong and so was
the exclusivity claim that rested on it.** Round 55 searched properly and found four:

| baseline | architecture | result | latency | ring |
|---|---|---|---|---|
| **PriMed** - Gopalakrishnan et al., *Expert Systems* 42(1):e13283 (online Mar 2023) | Conv(3->8,k3) -> Square -> AvgPool -> FC, **BFV** via Pyfhel/SEAL | **87.58%** | **171.4 s** | m=4096 |
| Al Badawi & Bin Yusof, *BioData Mining* 17:33 (Sep 2024) | RBF-kernel **SVM**, CKKS/OpenFHE | F1 0.80 | 2.09 s | **N=65536** |
| Selvakumar & Senthilkumar, *Sci Rep* (Jul 2025) | 2-layer quantized **FCNN**, TFHE | 81.82% -> 81.81% encrypted | **137.68 s/sample** | - |
| **Gecko** - Wei et al., arXiv:2608.02378 (Aug 2026), preprint | frozen ResNet50v2 in **plaintext client-side**, then encrypted predictor, CKKS/Lattigo | **87.18%** vs 87.82% plaintext | **413 ms** | conjugate-invariant |

**Two corrections, and the second is the one that stings.** PriMed is not a third entry appended to a
list - it is chronologically **first**, three years before the two that were here. And it *does* perform
a genuine homomorphic convolution, so the sentence that used to stand here - "neither performs a genuine
homomorphic convolution, this project does" - was an exclusivity claim built on an incomplete search.
A claim of the form "nobody else does X" is only ever as good as the search behind it, and this one had
a search behind it that stopped at two.

What survives is a claim about COST, which is checkable rather than exhaustive: PriMed pays 294.2 MB of
encrypted input and a mid-inference client re-encryption round-trip to replenish the BFV noise budget,
where this project's CKKS path runs at N=8192 without a round-trip. So the medical use case should be
presented against *these* papers rather than against crypto-systems work like Orion or BootNet - the
latter needs N=32768 for ImageNet and is a different claim entirely.

**Gecko is listed with a caveat that has to travel with it.** Its 413 ms is not comparable to the others:
the raw image is never encrypted. The client runs the frozen backbone in plaintext and encrypts only the
compressed feature vector, so it is a hybrid, and it belongs in this table only under the "or its
features" reading. It also never spells the dataset's name - Table 5 reads "Breast", identified as
BreastMNIST structurally (ultrasound, 128x128, 2 classes, MedMNIST v2 cited), which is weaker evidence
than a verbatim mention and is recorded as such.

**The ring-degree comparison that used to follow is withdrawn (Round 33).** This section said the
convolution runs "at **N=8192**, one eighth the ring degree the SVM baseline needed. That is a legitimate
efficiency position." It does not carry, for the reason that makes ring degree the wrong axis:
**N is set by multiplicative depth and precision**, so a smaller N is a statement about the circuit, not
about efficiency at fixed capability. Comparing the two operating points honestly:

| | Al Badawi & Bin Yusof | this project's `medical_fl.student` |
|---|---|---|
| input | full **28x28** | **14x14** downsample |
| model | RBF-kernel SVM | one conv block, 3,226 parameters |
| result | **F1 0.80** | **AUC 0.788221** against the ResNet-18 baseline's 0.9146825; teacher agreement 0.826923 |
| ring | N=65536 | N=8192 |

Deeper circuit and higher precision on a larger input, by construction, needs a larger ring. **A smaller
ring on a smaller input at lower accuracy is a different operating point, not an efficiency win** - the
claim would need the same accuracy at one eighth the ring, and that is not what was measured.

This is the same demotion Round 32 already applied to the identical claim in `models/metrics.json`, where
`medical_fl.student.baseline_gap_note` stopped calling 3,226 parameters over a 14x14 input "a
CKKS-feasibility budget" and started calling it a **representation choice**. That correction did not reach
this file, so for one round the artifact was honest and the reference document arguing for it was not -
and the reference document is the one written to be quoted. **A claim demoted in one artifact should be
grepped for in the others in the same change**, which is the lesson the ledger draws about the PRS
sentence in the same round.

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

**A correction to this section, and it goes the other way from how it was written (Round 33).** This file
said arXiv:2602.00672 (Jan 2026) "finds closed-form linear autoregression matches or beats
state-of-the-art deep time-series anomaly detectors, so a linear encrypted tier is defensible on the
merits rather than only on cost". The paper is Yugay, Cui, Pei and Zaytsev, *"Strong Linear Baselines
Strike Back: Closed-Form Linear Models as Gaussian Process Conditional Density Estimators for TSAD"*,
2026-01-31, and read properly it does not support that sentence. Three things are wrong with it, and the
third is the one that matters.

- **Different estimand.** Its linear model is an autoregressive **anomaly score fitted by OLS on the
  series itself**, interpretable as a finite-history Gaussian process conditional density. It needs no
  labels and no teacher. `satellite_comms.student` is a **49-parameter logistic regression distilled from
  the Telemanom teacher** over channel-by-timestep windows. Different object, different training signal.
- **Different comparison.** The paper's linear model is compared **against** deep detectors as a
  replacement for them. This project's linear model sits **beneath** a deep detector as a student of it.
  A result about the first says nothing about the second.
- **Read correctly, the paper indicts the T2 TEACHER.** Its recommendation is that future work "should
  consistently include strong linear baselines and, more importantly, develop new benchmarks with richer
  temporal structures pinpointing the advantages of deep learning models" - i.e. an argument for
  questioning whether the LSTM is needed at all, not an argument for keeping a distilled student under
  one. Cited as encouragement for the encrypted tier, it was pointed at the wrong tier.

**And in-repo nothing demonstrates it either way.** `satellite_comms.student` scores event-wise F0.5
**0.003234** against a teacher at **0.0** over 307,601 rows, on two labelled scoring units. Agreement of
0.9987 between them is agreement about finding almost nothing, which the registry note already says.
(Round 71: retrained without the whole-series z-score, both now score 0.0 and agree on 0.9756 of rows,
which strengthens the point rather than changing it.) So
the linear tier here is not yet defensible on the merits in either direction, because no measurement on
this corpus separates the two hypotheses - and the paper is external evidence about a different setup.
The honest position is the cost one, which was the original argument and did not need improving.

The paper is still worth carrying, for the use this project can actually make of it: **it is the citation
to reach for if anyone proposes replacing the Telemanom teacher, not the one that justifies the student.**
It is cited here and in no other file, including `references/REFERENCES.md`, which is where a paper this
load-bearing should have an entry with its authors and date.

## Telecom: the weakest vertical, and the quantum half is below its own threshold

Two findings, both of which this project should absorb rather than argue with.

**There is no public 8-class QKD-attack dataset.** At least three papers use eight classes with *incompatible*
taxonomies - only three of seven attacks overlap - both on private simulated data. This project
generates its own and labels it `synthetic, methodology-grounded`, citing arXiv:2509.14282, which is
the honest framing. But the eight-class shape should not be described as a standard, because it is
not one.

**THIS PARAGRAPH READ ITS CITATION BACKWARDS AND IS WITHDRAWN. Round 55.**

It said arXiv:2605.31449 documents fidelity quantum kernels "beating that regime only at **15 qubits
or more**", concluding that `q_n_qubits = 4` and `quantum_qubits = 8` are below the regime that works
and that the weak result is therefore explained.

Read at source - Agnihotri, Krebsbach, Reiter & Wellens, *Support Vector Machine with a Scalable
Quantum Kernel*, 2026-05-29 - the paper says the opposite. Verbatim:

> "Throughout the simulations, the Hamming quantum kernel outperforms the fidelity quantum kernel
> whenever 15 or more qubits are used."

> "For n >= 15, the test accuracy of the FQK drops sharply, approaching a random guesser ... for
> n >= 25 qubits."

> "For small n, the accuracy of all three methods is close to 1."

**15 qubits is a CEILING on the fidelity kernel, not a floor**, and concentration worsens
monotonically with qubit count - corroborated independently by Thanasilp, Wang, Cerezo & Holmes,
*Nat. Commun.* **15**, 5200 (2024): exponentially concentrated *in the number of qubits*. So 4 and 8
qubits are the LEAST concentrated regime, the best case for this kernel, not a broken one. Two further
refinements: the 15-qubit crossover is Hamming-versus-fidelity specifically, and it is a shot-budget
artefact rather than a physical constant - "1024 measurement shots are insufficient to reliably
estimate the overlaps ... with more than n = 15 qubits".

**What survives is the measurement, and it now has no published explanation.** Quantum-kernel SVC
macro F1 **0.8316** and variational QNN **0.4667** against RandomForest **0.9866**, with
`quantum_beats_classical: false`. That is still an honest recorded negative. What it is NOT is
"underperforming for a published, quantified reason" - removing the inverted citation leaves *quantum
lost, cause unknown*, and saying so is the correct state of the record. Claiming a better finding than
"quantum lost" on the strength of a paper that says the reverse is worse than reporting the loss.

The relevant literature is also not empty: 2025-26 QML-IDS work operates at exactly these 4-8 qubit
counts because that is what NISQ hardware and simulators allow, and is audited as showing no advantage
rather than being unpopulated.

**The ML-overlay framing IS supported, and is now attributable** - it previously carried no citation.
arXiv:2608.04047 (2026-08-04), *Beyond the QBER Threshold*, verbatim: *"a fixed 11% QBER threshold
achieves only 25.82% accuracy with a False Negative Rate (FNR) of 0.8477, whereas the proposed
framework reduces the FNR to 0.0198."*

**But this file misdescribed it.** It said the threshold "catches only 25.82% of attacks". **25.82% is
an ACCURACY**; with FNR 0.8477 the attack catch rate is **15.23%**. That is the accuracy-versus-recall
conflation this project corrected in the finance calibration work, repeated here about someone else's
number. ML as an overlay for sub-threshold stealthy attacks, never as a replacement for decoy-state
accounting - the conclusion stands, the arithmetic behind it is corrected.

### A third finding, and it is the one this section left out: telecom had no encrypted tier at all

**Both findings above answer a MODEL-QUALITY question, and this file exists to answer a PRIVATE-INFERENCE
one. For telecom it did not answer it - it left it silently unanswered, which reads as answered
(Round 33).** A verdict of "weakest, and the quantum half is in a regime documented not to work" is a
statement about accuracy. The question the rest of this file asks of every vertical is *what runs under
encryption*, and the answer for telecom was **nothing**.

Measured rather than inferred: `runs_encrypted=True` appeared on exactly **five** `ModelSpec` entries
in `src/qsh/models/registry.py` - the finance student, the medical student, the genomics PRS, and both
satellite students.

**RE-MEASURED Round 53: it is now SIX**, and the sixth is `telecom_ids.student` - the entry the next
paragraph says was being added in the same round. So this sentence and the one below it have
contradicted each other since the day they were written, with the count frozen at its pre-closure
value while the prose described the closure. Left as "appeared" with the correction beside it rather
than silently updated, because the point of the paragraph is what the count was BEFORE the gap was
closed, and rewriting it to six would erase the finding it exists to record.

**And the student beat its teacher, which the argument below does not allow for.** Distilled from the
random forest (`distil_linear_student(rf, ...)`, `scripts/train/telecom_ids.py:285`), the 64-parameter
CKKS-linear student scores macro F1 **0.988329** against the forest's **0.986629**, at 0.995
agreement. A student exceeding its teacher is not impossible - soft-target distillation regularises,
and 0.0017 on 600 held-out rows is well inside the noise this project records elsewhere for
event-wise metrics - but it should be stated rather than assumed away. Recorded here as an
observation, not as a claim that the linear model is better than the forest. `telecom_ids` has **two T2 specs and no T1**: `telecom_ids.qml` and
`telecom_ids.transformer`, the latter itself a documented non-artifact. So on the one use case whose whole
subject matter is the security of a quantum channel, the demo ran no encrypted inference, and this file
gave it a verdict that did not mention it. **The gap was invisible because the verdict column was full.**

**It is being closed in this same round**, by an encrypted linear student for telecom: a linear multiclass
model distilled from the RandomForest teacher on soft targets, returning all eight class scores under CKKS
with the argmax taken in the clear by the client. That shape is the same one the other four verticals use
and it is the shape the 4096-slot budget supports - eight score vectors over a seven-feature input is far
inside one ciphertext at multiplicative depth 1.

**No numbers are quoted for it here, deliberately, because none have been measured at the time of
writing.** Whatever it scores is what gets recorded, and given the teacher's macro F1 of 0.9866 against a
linear student's capacity, a gap should be expected rather than treated as a defect - the medical case is
the precedent for training the small model properly once before drawing a conclusion from its size
(checklist row `S3`). What can be said now without measurement is the structural claim: after this round
every use case has an encrypted tier, and the reason telecom is last is that its encrypted tier was
missed rather than declined.

## The stated artifact class is "distilled / linear / quantised", and the third one is not instantiated

**Recorded in Round 33 because two unrelated meanings of "quantised" are allowed to blur, and the blur
flatters the project.** The artifact class this demo advertises has three members. The first two are real
and measured: every encrypted-tier model here is **distilled** from a teacher, and all of them except the
medical conv block are **linear**. The third has no instance in the private-inference path at all.

What exists is `onnx/model_int8.onnx`, and it serves the **plaintext browser tier** - `_ONNX_PAIR` in
`src/qsh/models/registry.py` marks the int8 graph `browser`, with fp32 kept as the fallback for the cases
ORT Web cannot execute. It never touches a ciphertext. And the registry says so itself, in the comment
directly above `_STUDENT_ARTIFACTS`: *"A linear model of tens of parameters gains nothing from int8"* - so
the models that DO run under encryption are precisely the ones that ship no int8 graph. The two sets are
disjoint by design.

**What "quantised" means for private inference is a different thing, and Tailor is the citation for it.**
Lin and Wu (IACR ePrint 2026/1551) make **per-layer ciphertext precision a runtime parameter under a single
keygen** and evaluate ReLU and comparisons exactly, reporting MNIST at 2-bit output / 4-bit elsewhere
reaching 96.44% in 222 s against uniform 4-bit's 96.17% in 236.1 s. That is precision **in the encrypted
domain**, chosen per layer, on TFHE and FINAL backends. It has nothing to do with an int8 ONNX graph
executed in plaintext by `onnxruntime-web`, which is a deployment-size and CPU-throughput optimisation.

So the honest statement of the artifact class is **distilled and linear, both instantiated and measured;
quantised in the plaintext-serving sense only, and not in the private-inference sense at all.** Anyone
reading "distilled / linear / quantised" alongside a CKKS claim will read the third word as the Tailor
sense, and this project has no artefact that earns it. The route to earning it is recorded as a decision
with its reopening condition in `docs/deliverable-verification.md` - it needs a TFHE or FINAL
implementation that a browser tab can load and **evaluate** in, which is the same condition that declined
Rust for the client-side stack.

## The one architectural suggestion worth taking

For healthcare, a **frozen pretrained backbone with a CKKS-encrypted linear head** fits this
project's exact budget: a ViT CLS token is 768-D, which sits comfortably inside 4096 slots, and one
ciphertext-plaintext dot product per class is precisely depth 1. The 2025 medical instantiation
is FEDERATED rather than frozen-backbone, and the difference matters here - arXiv:2511.20983 reports
**96.12% plaintext against 90.02% encrypted** at 326 KB per aggregation round, but section III-C has
each client "independently train a ViT model on its private dataset". So it validates the
CLS-token-plus-CKKS-linear-head SHAPE and says nothing about the frozen part, which is the half this
project would be relying on. That would let this project say "ViT-class backbone" honestly while
keeping the encrypted step exactly as cheap as it is today.

Not adopted this round, and recorded as a recommendation rather than a plan, because the backbone's
licence and redistribution terms would need the same scrutiny the ImageNet question already got.

## Re-check, 2026-08-13: is there anything to be judged AGAINST?

Re-run of the per-industry question with a fresh literature sweep. The finding that changes how the
other sections should be read: **for four of the five verticals there is no public
privacy-preserving benchmark to be equivalent to.** That is a verified negative, not a gap in the
search, and it means "appropriate versus the latest research" can only be answered against
*plaintext* baselines plus honestly reported crypto overhead.

The state of the field generally: a practitioner catalogue of production FHE lists exactly three
deployments (Microsoft Edge Password Monitor, Apple Live Caller ID Lookup, Apple Enhanced Visual
Search) and **none of them evaluates a neural network homomorphically** - Apple's runs the model
on-device and encrypts only the resulting embedding. There is no production FHE ML inference to be
behind.

| Vertical | Is there a benchmark to match? | What this repo should claim |
|---|---|---|
| Finance | **No.** The US-UK PETs Prize Challenge (Track A, financial crime) concluded March 2023, its SWIFT data was never released, no leaderboard was published, and no successor exists. | Anchor on the plaintext AMLworld number: GFP+XGBoost minority-class F1 **0.6323** on HI-Small (NeurIPS 2023 D&B). This repository measures **0.1519** against it and does not close the gap - see the correction below. |
| Medical | **No** privacy-preserving one. MedMNIST v2 has a real plaintext leaderboard; FeTS is federated with no cryptography; MedPerf is federated *evaluation* only. | Compare to MedMNIST plaintext. Every named production deployment is plain FL with no cryptographic guarantee, so adding CKKS puts this ahead of them rather than level. |
| Genomics | **Yes, and it is the only one.** iDASH is live - 7 Nov 2026, Dallas - with a published rubric. Track 1 is homomorphic evaluation of a pretrained decoder, scored on a correlation term with a **60-minute disqualification**. | The one vertical where an external scoring rubric exists. Note that iDASH publishes participating teams but **not scores**, so the rubric is citable and a recent iDASH *number* is not. |
| Telecom | **No.** Two 2026 surveys say so in their own words, one of them noting it "could not compare the papers it reviewed" for lack of standard protocols. | The synthetic-corpus decision looks better after this sweep, not worse - see below. |
| Satellite | **No**, and no *cryptographic* privacy work at all. The closest is plain FedAvg on ESA-ADB (Atefrad & Karami 2025, F0.5 0.86, precision 0.93, AUC 0.85), which lists differential privacy and secure aggregation as **future** work - the same category the Medical row already discounts. ESA-ADB is the credible plaintext benchmark. | Genuinely novel, which cuts both ways: nothing to be judged against, and nothing to be validated by. |

**Telecom deserves a note, because a decision recorded here as a limitation reads differently now.**
This project trains on synthetic QKD-attack telemetry because no commercial-OK real corpus exists,
and the model card says so. The 2026 literature has meanwhile turned hard against the standard
alternatives: **NSL-KDD has been withdrawn by its own publisher** (the CIC page now serves only
"this dataset is no longer available"), a EuroS&P 2024 paper excludes KDD/NSL-KDD from analysis as
"long known to be faulty", and **more than 25% of flows in CIC-IDS2017 are artefacts of a flow
terminator that closes a TCP flow on the first FIN**. The sharpest number: correcting a fifth of
CIC-IDS2017 moves aggregate accuracy **0.99 -> 0.99 -> 0.99** while per-class F1 moves 0.18 -> 0.79.
A ~99% headline on that corpus is uninformative *because* it is invariant to fixing the data.

So the honest framing for telecom is not "we used synthetic data because we had to" but "the corpora
we would have used are discredited, and a 99% number on them would have been worth less than a
disclosed synthetic one". What the synthetic corpus cannot do is support a claim about real links -
which is what the provenance string already says.

**Two things a reviewer would reach for, and what they are worth.** There is no NIDS leaderboard to
place a result on: Papers with Code, which hosted the only informal ones, was sunset in July 2025 and
now redirects to an unrelated feed; its archive is frozen at a 2025-07-28 snapshot. And the most-used
NIDS dataset appears in **only 29% of contemporary papers**, so even the popular choice is not a
standard.

**Where an equivalence claim IS available.** On tree models under TFHE the accuracy loss is exactly
zero and the entire cost is latency - one published AMLworld result reports plaintext and FHE
metrics identical at ~120,000x overhead. The same shape holds for the MPC literature: in genuine MPC
the accuracy loss is zero by construction. That is the correct frame for this repo's own numbers:
**the interesting quantity is the overhead and the fidelity of the encrypted path, not an accuracy
delta**, because a correct implementation should not have an accuracy delta. This repo's CKKS tier
reporting "0 top-1 mismatches against the plaintext reference" is measuring the right thing.

### Correction, same day, to the finance row above

The first version of this section said the published reference is 0.6323 "and this repo's finance
page prints 0.6323 - equal, on the metric the corpus warrants." That is wrong in the way that
matters. **0.6323 is the published BASELINE the page displays as a target; this repository's measured
minority-class F1 is 0.1519.** The page labels it correctly - the rendered text reads "0.6323
published baseline, NOT measured here" - and the label is what I misread.

The repository's own record was never unclear about this. `registry.py` carries `baseline=0.6323`
with `baseline_source` naming GFP+XGBoost, `metrics.json` carries `minority_class_f1` 0.1519, and
`baseline_gap_note` explains the cause at length: IBM's Graph Feature Preprocessor computes
per-transaction subgraph patterns over a dynamically updated graph, this project computes per-account
scalar aggregates, and the gap is unreachable at any threshold because measured AUPRC is far below
what that F1 requires. It closes with "Reported as measured; not adjusted to close the gap."

Recorded rather than silently edited, because the failure mode is the one this document is about: a
number that appears on a page next to a model is not automatically that model's score, and reading a
figure without its label is how a target becomes an achievement. It happened here while writing about
verification standards, which is the useful part of the example.

Verified 2026-08-13. Negatives above are search-verified rather than assumed; the two most load-bearing
(no PP-NIDS benchmark, no PP satellite work) rest on 2026 survey papers stating it directly.
