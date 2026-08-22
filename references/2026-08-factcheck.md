# Fact-check against sources and standards current at 2026-08-22

Every row was checked against a primary source or re-derived from this repository's own code. The
column that matters is **Verified by me**: findings arrived from a delegated sweep, and the ones
marked `spot-checked` were reproduced here before being written down. The rest are recorded as
reported, unreproduced, and must be re-measured before anything is changed on their account - this
project has twice shipped a correction that was itself wrong because a reported figure was
transcribed instead of derived.

## Contradicted, and the sharpest one is a measurement that describes the wrong thing

### 1. SP 800-90B health tests run on CONDITIONED output. spot-checked, and the reported reason was wrong

**FIXED Round 51.** `health_suite` now takes `raw=` and runs the section 4.4 continuous tests and
the min-entropy estimate on the pre-conditioning stream; SP 800-22 stays on the output, which is what
it is for. Re-derived here rather than transcribed: a p(1)=0.90 source reported **0.9806** bit/bit
before and reports **0.1472** after, against **0.9892** for a fair source - so the figure now
discriminates where it previously separated a badly biased source from a healthy one by 0.01 bit.
`tests/test_health_tests_measure_the_noise_source.py` pins it, and the call site is checked by
parsing the call's keywords rather than grepping the source. The first version of that check DID grep,
and it did not fire when the call site was reverted, because the comment explaining why `raw=raw`
matters sits above the call and contains the string.

`src/qsh/qrng/simulator.py:147` calls `health_suite(bits, ...)` where `bits` is the post-processed
output. `src/qsh/qrng/health.py`'s own opening docstring states the contract it violates: the
section 4.4 continuous tests "run on raw samples **before any conditioning**". `raw` is in scope four
lines above the call.

**The reported symptom was "a broken source passes". Reproduced here, it does not.** A p(1)=0.90
source measures min-entropy **0.1418 bit/bit** on the raw samples and `passed=True` - which is
correct SP 800-90B section 4.4 behaviour, because Repetition Count and Adaptive Proportion are tuned
to catch a source that has failed OUTRIGHT or gone stuck, not one that is merely biased.

**The real defect is narrower and still real.** Run on conditioned output the same call reports
**0.9695 bit/bit**. An extractor produces near-uniform output by construction, so that number
describes the Toeplitz stage and not the entropy source, while every label around it says entropy
source. The figure is not wrong about what it measured; it is measuring the wrong thing, and it is
the quantity a reader would use to judge the noise source.

Also reported, NOT reproduced here: `health.py:171` uses `n` where SP 800-90B section 6.3.1 uses
`(L-1)` in the MCV estimator, and the Adaptive Proportion cutoffs sit one below Table 2 at H = 0.2 /
0.6 / 0.8. Both directions are stated as conservative. **Re-derive before touching.**

### 2. `FiniteKeyParams` hardcodes the retired intensities. spot-checkable from the code

**FIXED Round 51.** `FiniteKeyParams.mu_signal` / `.mu_decoy` now use `default_factory` against
`QKDConfig`, the same wiring `QKDParams` took in Round 47. A bare `FiniteKeyParams()` returns
0.3416 / 0.1417.

The reported downstream figures were re-derived and the repository was ALREADY right about them: the
positive-key threshold at 50 km measures **3.84e7 pulses** at the retired pair and **1.40e7** at the
shipped one, which is what Round 47 recorded (3.84e7 / 1.41e7). The sweep's "42,448 against 143,645"
is a sifted-bit figure, a different quantity from the pulse counts the docs quote, so nothing
downstream needed changing - which is why the rule is to re-derive before correcting.

**Why nothing caught it, which is the part worth keeping.** The refuted-band test builds
`FiniteKeyParams()` bare and asserts the threshold lands in 5e6 to 2e8. Both pairs sit inside that
band, so it was green on the defect and on the fix alike - a gate positioned exactly where the drift
happens, with a tolerance wider than the effect. `tests/test_intensity_defaults_come_from_config.py`
asserts the SOURCE of the default instead, which has no tolerance to hide in, and checks
`default_factory` rather than the value so a literal that happens to match today still fails.

`src/qsh/qkd/finite_key.py:115-116` carries `mu=0.5, nu=0.1`; the shipped configuration is
`mu_signal=0.3416, mu_decoy=0.1417`. `src/qsh/qkd/bb84.py:141-147` fixed exactly this for `QKDParams`
and the sibling did not follow - the same one-site-short pattern this project has recorded in seven
consecutive rounds. Round 49's plan listed it as a known gap and it is still open.

Reported consequences, unreproduced: the positive-key threshold quoted as "~1e5 sifted bits" is
**42,448** at the shipped pair rather than 143,645; the "order of magnitude" phrasing at
`finite_key.py:55` is **3.86x** at shipped intensities; and Lucamarini's 1.4e5, described as agreeing
"and always did", agrees at 0.97x on the retired pair and 3.30x on the shipped one. The
refuted-band test calls with `params=None`, so it exercises the retired intensities and its 5e6-2e8
band admits both - a gate that cannot see the drift it is positioned to catch.

### 3. "Two-intensity decoy-state bound" names a protocol the cited equations are not

**RESOLVED Round 53, verified at source.** The paper was fetched and its text extracted. Section 3.4
is titled *Vacuum+Weak decoy state* and introduces Eqs. (33)-(37); the sentence that makes them tight
reads *"from vacuum decoy state, Eq. (33), Alice and Bob can estimate Y0 accurately"*. So Eqs. 34-37
are the THREE-intensity protocol. The paper's own two-intensity case is section 3.5, *One decoy
state*, which BOUNDS Y0 rather than knowing it.

This repository does neither exactly: two intensities, with `y0` supplied by the channel model. In a
simulator that is legitimate and TIGHTER than section 3.5, because the dark-count rate is known
exactly; on hardware it would not be. The name was accurate about the intensity count and misleading
about the equations, so `bb84.py` and `README.md` now say which part comes from where.

`src/qsh/qkd/bb84.py:6,86,102` and `README.md:80` call it a two-intensity bound. Ma, Qi, Zhao and Lo
section 3.4 describe Eqs. 34-37 as "a two-decoy-state protocol with a **vacuum and a weak** decoy
state" - signal plus two decoys, three intensities. The repo implements signal + one weak decoy and
supplies `Y0` analytically (`bb84.py:224`) instead of measuring a vacuum decoy, so the implementation
is self-consistent and the NAME is what misattributes it.

### 4. `decoy_reference.py` attributes two things to Ma et al. that are not there

**BOTH CONFIRMED Round 53, and the first is worse than reported.**

*`f_ec = 1.22` is not Table 1.* Table 1's columns are lambda, alpha, e_detector, Y0, eta_Bob and
**f = 2MHz / 0.1MHz - a repetition rate**. 1.22 is from the prose: *"use f(e) = 1.22, which is the
upper bound of f(e) in secure distance for this experiment"*. Four of the five defaults ARE Table 1;
the docstring said all five were. A correct number under a wrong provenance - the harder kind to
notice, because nothing computed from it is wrong.

*The q claim was backwards.* The docstring said the reproduced curves are efficient BB84 "and Ma et
al. say so in the text". The text says the reverse: Eq. (42), which is what Figure 2 plots, is
introduced with **"where q = 1/2 with standard BB84"**.

**And the reference values are not in the paper at all.** R/eta ~ 0.11 and R/eta^2 ~ 0.15 appear
nowhere in its prose - searched. `tests/test_decoy_reference.py` passes `q_sift=1.0` to reach them,
which doubles the rate; at the q the paper names for that figure this model gives 0.0568 and 0.0755.
Three possibilities and the record does not pretend to know which: the values were read off the
plotted curve and the curve is at q ~ 1 despite the caption; or they were read correctly and this
model is 2x low with `q_sift=1.0` closing the gap by coincidence; or they came from this model and
were then attributed to the paper. **NOT resolved here** - resolving it means reading values off a
plotted figure, which cannot be done from an extracted text layer. The 13 tests are left untouched
and the docstring now says the assertions are this module's own outputs until that reading is done.

Reported, and both are quotation-level claims worth re-reading at source before acting:

- `decoy_reference.py:59-67` says the published curves are plotted for the efficient BB84. The paper
  states "where q = 1/2 with standard BB84". If so, the derived `R/eta ~ 0.11` and `R/eta^2 ~ 0.15`
  are this model's own outputs at `q_sift=1.0` rather than the paper's figures.
- `decoy_reference.py:46-49` attributes `f_EC = 1.22` to Table 1. Table 1's `f` column is a
  repetition rate in MHz; 1.22 appears in the paper's prose, cited onward to its own reference.

### 5. E91 and QSDC, both permissive, neither flipping a shipped verdict

**E91 FIXED IN ALL THREE IMPLEMENTATIONS - Python in Round 53, the two browser copies in Round 54,
and the one-round gap is the finding.** Round 53 fixed `qkd/e91.py` and left
`web/src/lib/quantumSim.ts` and `web/src/lib/clientOptiQKD.ts` carrying the bug. **This demo is
client-side only**, so the copy that was fixed is the one no visitor executes and the two that were
missed are the ones that render. Ninth instance of one fix reaching one language, committed by the
round that was closing a fact-check about exactly this class of defect.

**Re-derived exactly.** The composition was a units confusion between a
VISIBILITY and an ERROR PROBABILITY: a channel flipping a bit with probability `e` has visibility
`(1 - 2e)`, not `(1 - e)`. At the shipped 50 km channel with V = 0.97 and e = 0.015200 the old form
gave **0.022372** and the corrected one gives **0.029744** - and composing the two error sources
independently, `q_V + e - 2*q_V*e`, gives **0.029744** as well, agreeing to **0.000%**. Exact rather
than first-order agreement is what makes this a correction and not a modelling preference. The
reported 24.8% understatement reproduces exactly.

**QSDC NOT addressed.** Qi et al. Eq. (8) defining `g` as back-channel loss "including the efficiency
of the detector" needs reading at source before `qsdc.py` is touched, and the reported 4.0x is
unreproduced here. Left open.

`src/qsh/qkd/e91.py:44-46` applies visibility as `V*(1-e)` and then `QBER=(1-V)/2`, which halves the
error term - reported as 0.022372 against a self-consistent 0.029744, **24.8% low**.
`src/qsh/qkd/qsdc.py:90-92` uses channel loss alone for `g`, where Qi et al. Eq. (8) define it as the
back-channel loss "including the efficiency of the detector" - reported **4.0x** understated on fibre.
Both make the demo look better than the model justifies, which is the direction that matters.

### 6. Currency

**NOT RESOLVED Round 53, and the attempt is recorded because it failed in an informative way.**
ITU's own recommendation index does not expand the Y.3800 series on the page fetched, and search could
confirm Y.3832 / Y.3833 (12/2025) and Y.3828 (04/2025) but **not** Y.3836, Y.3837 or Y.3838. The
reported "series now runs to Y.3838, approved 2026-08-13" is therefore UNVERIFIED here and
`REFERENCES.md`'s Y.3835 ceiling is left alone rather than advanced on a search summary.

The GS QKD 010 claim is worse than unverified - it is **contradicted**. The sweep reports 010 "has
never been published and sits at draft v0.6.2"; a search summary reports V1.1.1 dated 2023-04, and
both a V0.0.1 (2017-12) and a V0.4.1 (2021-06) draft are visible in ETSI's open docbox. Direct fetches
of `etsi.org/deliver` return **HTTP 403 for QKD 010, 014 AND 020** - including 014, which is certainly
published - so the 403 is a bot block and carries no information about publication status. This needs
ETSI portal access to settle; nothing in the repo was changed on it.

- **ITU-T**: `references/REFERENCES.md:962` says the series runs to Y.3835. It now runs to **Y.3838**
  (Y.3836, Y.3837 and Y.3838 approved 2026-08-13; Y.3836 is still flagged prepublished).
- **`stress-testing-fact-check.md:44,120,122`** cites ETSI GS QKD 010 clause 8 as normative. **010 has
  never been published** and sits at draft v0.6.2.
- A large ETSI revision wave targets TB approval **2026-12-02**, including 014ed2 and 015ed3 - both
  specs this project implements.

### 7. Two claims in `model-choice-vs-2026-practice.md` no longer hold

**SECOND CLAIM CONFIRMED AND CORRECTED Round 53.** `runs_encrypted=True` is on **six** specs, not
five - `finance_aml.student`, `medical_fl.student`, `genomics.prs`, `satellite_comms.student`,
`satellite_comms.opssat_student`, `telecom_ids.student`. The document contradicted itself: the
sentence after the count says the telecom gap "is being closed in this same round", which is what
made it six. Corrected by annotating rather than rewriting, since the pre-closure count is the finding.

The student-beats-teacher observation is also confirmed: `telecom_ids.student` scores macro F1
**0.988329** against the random forest it was distilled from at **0.986629**, 64 parameters, 0.995
agreement. Recorded as an observation - soft-target distillation regularises and 0.0017 on 600 rows is
inside this project's own recorded noise - not as a claim that the linear model is better.

**FIRST CLAIM NOT RESOLVED.** The 15-qubit concentration reading requires arXiv:2605.31449 at source,
and it is not settled by reasoning about the summary. Left open.

`:282-291` explains quantum-kernel underperformance by a 15-qubit concentration threshold. The cited
work (arXiv:2605.31449) reportedly says 15 qubits is where the **Hamming** kernel overtakes the
fidelity kernel, and that concentration WORSENS with qubit count - so 4 and 8 qubits are the
least-concentrated regime, and the citation is being read backwards. `:306-311` says
`runs_encrypted=True` appears on exactly five ModelSpec entries; it is now **six**, and the telecom
student's macro F1 **0.988329** exceeds its teacher's **0.986629** - the student beat the teacher,
which the surrounding argument assumes cannot happen.

## Confirmed at source - recorded because a confirmation is a result

**Standards, read from cover pages rather than from search summaries.** ETSI GS QKD 014 V1.1.1
(2019-02) is still the only edition; 015 V2.1.1 and 018 V1.1.1 both 2022-04; 020 V1.1.1 dated
2026-06-29. NIST FIPS 203/204/205 remain the 2024-08-13 finals; SP 800-90B is still January 2018 with
no revision in progress; SP 800-90C final 2025-09-25. **HQC has no published draft**, which
strengthens rather than weakens `docs/pqc-roadmap.md:168`'s decision not to wire it.

**Round 47's central correction is right at source.** The word "threshold" appears nowhere in Rusca
et al.; `eps_sec=1e-9`, `eps_cor=1e-15`, a=6, b=19 and Table I all transcribe correctly.

**Ma et al. renumbering, GYS parameters, the QLHL `+2` term, and every internal re-derivation**
checked exact: 0.002368 / 0.001977 bit/pulse, 38,410,230 -> 14,055,773, `s^l_{Z,1}` = -216.53 at 2e6,
QSDC 0.110028 / 0.048521 / 0.000267, downlink 24.478 dB, tagged fraction 1.0417.

**`model-choice-vs-2026-practice.md`'s core conclusion holds.** ResNet is still the 2026
encrypted-vision workhorse, linear PRS still what encrypted PRS work runs, Telemanom-ESA-Pruned still
best on ESA-ADB Mission1, and QSMPC's demonstrated ceiling is still an MLP on MNIST.

## The finite-key bound implemented is no longer the tightest available

Rusca et al. use Hoeffding. Since superseded by Kamin, Tupkary and Luetkenhaus (arXiv:2502.05382,
Phys. Rev. Research 2026), Mannalath, Zapatero and Curty (PRL **135**, 020803, 2025), and Mizutani et
al. (arXiv:2511.21253, via Kato's inequality). Adopting any of them lowers the threshold discussed in
finding 2. Recorded as an opportunity, not a defect: the implemented bound is valid, merely
conservative. Note that the two identifiers `README.md:550` names as the future replacement concern
device IMPERFECTIONS and leaky receivers, not tighter statistics, so they will not do this job.

## Not accessible

Wu et al., *Quantum Engineering* **1**, e26 (2019) - the DL04 5.7% figure at `defaults.py:245`.
Publisher and mirror both returned 403. Metadata confirms the citation; the text was not opened, so
the attribution is plausible and unrefuted rather than verified. CNSA 2.0 pages and the ITU AAP
record for Y.3836 were likewise unreachable.

## Disposition

Nothing above is fixed by this file. Findings 1, 2 and 3 are code-level and should be corrected
`by **Round 53**`; findings 4, 5 and 7 are attribution and prose claims that must be re-read at
source first, because two of this project's last three correction passes shipped a wrong correction
by transcribing a reported figure. Finding 6 is a documentation currency fix and carries no
measurement risk.

**DEADLINE EXECUTED Round 55.** Status of each, and the three that are NOT done say why rather than
moving:

| # | state |
|---|---|
| 1 SP 800-90B on conditioned output | **FIXED** Round 53. `health_suite` takes `raw=`; a p(1)=0.90 source now reports 0.1472 bit/bit, not 0.9806 |
| 2 `FiniteKeyParams` retired intensities | **FIXED** Round 53, `default_factory` against config. Downstream figures re-derived and the repo was ALREADY right |
| 3 two-intensity naming | **FIXED** Round 53, verified by extracting the paper: Eqs. 34-37 are section 3.4 Vacuum+Weak, three intensities, Y0 measured |
| 4 `decoy_reference.py` attributions | **BOTH CONFIRMED, one fixed.** `f_ec = 1.22` is prose not Table 1 (whose `f` column is a repetition rate) - corrected. The q claim was backwards and is corrected. The 0.11 / 0.15 reference values are NOT in the paper's prose and remain **open**: settling it needs values read off a plotted figure |
| 5 E91 and QSDC | **E91 FIXED** across all four implementations, Rounds 53-54. **QSDC open** - Qi et al. Eq. (8) needs reading at source and the reported 4.0x is unreproduced |
| 6 currency | **OPEN, and the attempt failed informatively.** Y.3836-Y.3838 unconfirmed by search; `etsi.org/deliver` returns 403 for QKD 010, 014 AND 020 alike, so the 403 carries no information about publication status. Needs ETSI portal access |
| 7 `model-choice-vs-2026-practice.md` | **Second claim FIXED** - `runs_encrypted=True` is on six specs, not five, and the document contradicted itself two paragraphs apart. **First claim open**: the 15-qubit concentration reading needs arXiv:2605.31449 at source |

Three of seven fully closed, two partly, two open with a stated obstacle. The open ones are NOT
re-conditioned to a later round, because moving a promise is not discharging it - they are recorded as
requiring something this session could not obtain (a plotted figure, a paywalled portal, a paper).
