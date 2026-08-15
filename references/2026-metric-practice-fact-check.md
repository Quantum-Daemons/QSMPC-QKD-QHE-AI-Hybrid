# Fact-check: what 2026 practice actually says about the metric choices made this round
### Round 24c, 2026-08-09. Every arXiv ID below was resolved; ones that did not resolve were dropped.

Five choices were made this round on engineering judgement. This checks each against the current
literature, and two of them turn out to be defensible-but-not-best-practice rather than correct.

## 1. Tuning a screening threshold on F1 degenerates - and this was documented in 2014

My first medical threshold sweep picked the lowest grid point and predicted malignant for all 156
test images. I diagnosed it correctly from the numbers but treated it as a local surprise. It is a
known theorem.

**Lipton, Elkan & Narayanaswamy, "Thresholding Classifiers to Maximize F1 Score", arXiv:1402.1892;
published ECML-PKDD 2014, LNCS 8725, DOI 10.1007/978-3-662-44851-9_15.** Verbatim: *"if the
classifier is completely uninformative, then the optimal behavior is to classify all examples as
positive."* The paper also gives the calibrated result that the optimal threshold is half the optimal
F1. The value my sweep selected, 0.4242, is exactly the all-positive score 2p/(1+p) at prevalence
p=0.269 - so the run reproduced the theorem rather than hitting a bug.

**Label it 2014, not 2026.** It is the best citation for this and it is twelve years old.

Reaffirmed since: Nikolopoulos, arXiv:2512.00916 (2025-11-30) shows F1/AUPRC/MCC/accuracy "induce
degenerate thresholds", with the F1 coefficient (1-pi)/(2pi) diverging as prevalence falls.

## 2. Youden's J is defensible, but it is NOT what 2026 recommends

I replaced F1 with Youden's J because J cannot be gamed by predicting all-positive. That reasoning
holds, and it is also not the current recommendation.

**Van Calster et al. (STRATOS TG6), Lancet Digital Health 2025;7(12), preprint arXiv:2412.10288** -
the most authoritative source found: *all* classification measures including accuracy and F1 are
**improper** for clinically relevant thresholds other than 0.5 or the prevalence, and of 32 measures
assessed F1 was the only one possessing neither required characteristic.

**arXiv:2605.07829 (2026-05-08)**: *"The classical Youden index corresponds to a symmetric special
case and may therefore be suboptimal in realistic decision settings"*; a cost-weighted extension cut
estimated misclassification risk by up to 63%.

Real deployment evidence: Pamova et al., Frontiers in Big Data, 2025-11-18 (PMC12669008) - swapping
Youden for net benefit in AI mammography moved sensitivity 72% -> 99% (specificity 75% -> 48%).

**What this means here.** Youden's J implies a symmetric cost ratio - one missed cancer costs the
same as one false alarm - which is wrong for screening and is not a choice this project ever stated.
The 2026 answer is decision-curve analysis / net benefit, or an explicit fixed-sensitivity operating
point. Recorded as a known gap rather than silently left as "we used Youden".

## 3. Reporting fidelity separately from the task metric IS 2026 practice

**arXiv:2604.25110, "Knowledge Distillation Must Account for What It Loses" (2026-04-28)** proposes a
*"Distillation Loss Statement"* reporting "what was preserved, what was lost, and why the remaining
losses are acceptable", and argues students be judged "not only by retained task scores". That is
what `FidelityReport` does, and the finance student is the case in point: in **Round 24** agreement rose 0.884 ->
0.947 while minority-class F1 did not move.

Corroborated by arXiv:2605.31191, which finds teacher-student agreement flat while accuracy gains
vary - i.e. the two decouple empirically, so reporting one is not reporting the other.

**No 2026 source defines a "% of teacher retained" standard**, so this project should not claim one.
The nearest retention figure is 2021 (arXiv:2111.03362, 0.32-5.3 pp degradation). HEBench covers
only primitive ops; there is no MLPerf HE track and no CKKS model zoo to be benchmarked against.

## 4. Multi-seed reporting is good practice and is NOT mandated anywhere

I measured teacher AUC sd 0.0198 over 6 seeds and the student 0.0820. Checked whether any venue
requires this:

| Venue | Actual wording |
|---|---|
| NeurIPS 2026 | error-bar item is answerable **NA**, explicitly sanctioned |
| MICCAI 2026 | "strongly encouraged", not required |
| ICLR 2026 | reproducibility statement optional; does not mention seeds or error bars |
| MIDL 2026 | "not mandatory - not following those guidelines will not impact reviews" |
| TRIPOD+AI (2024) / CONSORT-AI / CLAIM | silent on training stochasticity |

So the honest framing is voluntary good practice. Independent corroboration that it matters:
**arXiv:2603.28040 (2026-03-30), Table 6, n=20 seeds** - ChestMNIST AUC .756+/-.011, RetinaMNIST
.729+/-.016, i.e. ~1.1-1.6 AUC-point seed-to-seed sd on MedMNIST. This project measured ~2.0 points
on BreastMNIST, the same order. **MedMNIST's own published tables report means with no sd.**

## 5. Degenerate-predictor checks: real tooling, no standard requires them

`n_classes_predicted` and `majority_class_rate` are a sound composition of established ideas
(scikit-learn `DummyClassifier`, Deepchecks `SimpleModelComparison` and `IsSingleValue`) but are
**not** a named practice and are **not** required by Model Cards (arXiv:1810.03993, no 2026
revision), Croissant-RAI (dataset-level only - wrong layer), ISO/IEC TS 4213:2022 (no 2026 revision),
or NIST AI RMF 1.0. CEN-CENELEC **prEN 18229-2**, the standard that would carry it, is still in
drafting; **EN 18286:2026** was published late July 2026 but is process not metrics and is not yet
cited in the Official Journal, so it grants no Article 40 presumption of conformity.

Ship these as good practice. Claiming a standard requires them would be false today.

## 6. A regulatory date this project records is now WRONG

**Regulation (EU) 2026/1744 (Digital Omnibus) entered into force 2026-07-27** and defers Annex III
standalone high-risk obligations from **2026-08-02 to 2027-12-02** (Annex I embedded: 2028-08-02).
Convergent reporting from Hunton, Orrick, Gibson Dunn and DLA Piper. Anywhere this project treats an
Annex III high-risk duty as binding now needs re-dating. Article 50 transparency duties are a
separate article and are NOT covered by that deferral - see `docs/ai-act-art50-gap-analysis.md`,
which must not be relaxed on the strength of this.

## 7. Corrections to claims worth not repeating

- **There is no 2026 State of DevOps report.** DORA's publications run to 2024; the 2025 successor
  is "State of AI-assisted Software Development", which retired the Elite/High/Medium/Low tiers.
  The genuine 2026 DORA item ("ROI of AI-assisted Software Development", v.2026.1, 2026-04-22) says
  nothing about change failure rate or deploy verification.
- **SLSA is v1.2** (approved 2025-11-24), not v1.1, and covers build/source provenance - **not**
  runtime deploy verification. OWASP CI/CD Top 10 is still v1.0 (2022).
- Docker `HEALTHCHECK` status is tracked **separately from the container's running state**, which is
  the documented mechanism behind this project's "docker ps says Up while Caddy 502s" incident.
