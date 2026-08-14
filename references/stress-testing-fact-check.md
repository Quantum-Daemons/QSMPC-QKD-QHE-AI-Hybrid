# Quantum-resistance stress testing: what is prescribed, what is not (Round 22, 2026-08-08)

The question asked was whether an E2E orchestration simulator should offer a "quantum resistance
stress test", and if so what it should contain. Every source below was fetched and read, not taken
from a search summary - the same discipline that caught the QERS and CARS problems recorded in
`security-scoring-fact-check.md`.

## The headline answer

**No standard, anywhere, prescribes adversarial stress testing of a hybrid PQC + QKD + FHE + MPC
system.** There is no authority to conform to for the integrated concept.

That is not the same as the idea being invented. Four axes have genuinely prescriptive sources with
published reference values, and one standard explicitly endorses the *methodology* of parameter
sweeping. So the honest framing is not "quantum resistance stress test", which implies a score. It is
**"reproduce published attack and finite-key reference curves, and published negative-input
conformance vectors"** - every number on screen tracing to a named equation in a named paper or to a
NIST-published vector set.

## The methodology endorsement

ISO/IEC 23837-2:2023 section 5.3.3 is the one standard that blesses the shape of such a test:

> EAs for these SFRs usually involve performing tests over some pre-defined parameter spaces ...
> this document adopts the strategy of evenly discretizing the parameter spaces and performing the
> test step by step.

But section 5.3.5 withholds every number:

> this document does not specify values for the thresholds and input parameters

So it grounds the form and supplies none of the content. Of its 24 evaluation activities, 21 require
an optical bench - photon counters, OTDR, calibrated attenuators, tunable lasers. Only clause 6's
three activities and section 8.11 double-click handling are protocol logic a browser can execute.

## What each source actually prescribes

| Source | What it prescribes | Usable here |
|---|---|---|
| NIST SP 800-227 (2025) | Only CAVP/CMVP conformance, and undercuts it: validation "does not guarantee correct functioning on all inputs". Refuses to give side-channel guidance. **But section 4.6.3 is directly on point for a hybrid**: the naive combiner `KDF(K1,K2)` "does not preserve IND-CCA security, regardless of the properties of the KDF"; the approved form binds the ciphertexts and encapsulation keys. | Yes - section 4.6.3 |
| FIPS 203 section 7.2 / section 7.3 | Normative SHALL: encapsulation-key modulus round-trip check, decapsulation-key hash check. Table 1 decapsulation failure rates 2^-138.8 / 2^-164.8 / 2^-174.8. | Yes |
| NIST ACVP + C2SP/CCTV | NIST does publish negative vectors, at the algorithm-validation layer: ML-DSA sigVer's five disposition categories, and CCTV's `modulus/`, `strcmp/`, `unlucky/` sets. | Yes |
| ISO/IEC 23837-1/-2:2023 | The only QKD adversarial test procedures that exist, and they deliberately contain no thresholds. | Form only |
| ETSI GS QKD 010 | Gives the Trojan-horse relation mu_out <= mu_in x r_rel, then declines to specify the privacy amplification: the adopter "shall provide a security justification". | No |
| ETSI GS QKD 014 | Concrete negative API behaviour (400/401/503, "size shall be a multiple of 8"). Modest but real. | Yes, modestly |
| ETSI GS QKD 016 / BSI-CC-PP-0120 | Common Criteria protection profile. Contains no numeric QBER or key-rate thresholds - the place ISO said the numbers would live does not have them either. | No |
| ISO/IEC 4922-1/-2 | Taxonomy and secret-sharing mechanisms. **No testing, evaluation or conformance clause exists.** | No |
| NIST IR 8214C (2026-01) | The only authority that mentions stress testing: section 7.6.3 says a test case "should consider at least one corrupted party" and may "stress-test ... adverse conditions". Table 5 requires statistical security >= 40 bits, >= 64 suggested - with the honest caveat that this parameter "does not appear in NIST standards". | Yes |
| HomomorphicEncryption.org standard | Zero occurrences of "test", "validation" or "conformance". States CKKS "is not described in this standard" - citing it for CKKS would be a misuse. | No |
| ISO/IEC 28033 | **Not published.** Part 3 is the CKKS part (arithmetic on approximate numbers) and is at **FDIS 50.00, 2026-06-11** - one ballot from publication, so this row has a shelf life of months. Part 1 general DIS 40.60 (2025-12-11), Part 4 FDIS 50.00, Part 5 WD 20.60. Part 1 scopes out correctness: "Verification that the function itself is computed correctly is outside of the scope." Stages checked 2026-08-10. | Not yet |

## The four defensible axes, with their reference values

1. **Hybrid key-combiner robustness.** The most defensible test available, because it is the only one
   that tests the *hybrid* property this project claims. Compromise one component secret and show the
   composite survives under the ciphertext-binding combiner and fails under `KDF(K1,K2)`.
   **This became true of the code in Round 38 and was not before.** All four combiner sites
   bound the encapsulation keys and omitted `c1`, the ML-KEM ciphertext - and the MCP one
   bound nothing at all, which is expression (16) itself. Section 4.6.3 is explicit that the
   encapsulation keys are *not* what delivers the IND-CCA-preserving property: "[25] does not
   incorporate encapsulation keys into the combiner, as this is not needed to achieve the
   IND-CCA-preserving property". So the construction had included the optional half and
   dropped one of the two load-bearing ones. Demonstrated rather than argued: with the old
   transcript, a mauled `c1` that decapsulates to the same `K1` derived a byte-identical
   session key; with `c1` bound it derives a different one.
   Sources: SP 800-227 section 4.6.3; Barbosa et al., X-Wing, IACR CiC 1(1) 2024; Giacon-Heuer-Poettering,
   KEM Combiners, PKC 2018.

2. **PQC malformed-input rejection.** Binary pass/fail against vectors NIST and C2SP publish - the
   strongest falsifiability available anywhere in this list.

3. **Finite-key rate versus block size.** Two independent peer-reviewed sources ten years apart, both
   with complete parameter sets, and the recent one ships code.
   Lim, Curty, Walenta, Xu, Zbinden, PRA 89, 022307 (2014): block size 10^4 still reaches 135 km.
   Wiesemann et al., Quantum 10, 2037 (2026): N_Z = 10^7 reaches about 50 dB, ~250 km at 0.2 dB/km.

4. **Decoy versus non-decoy under photon-number splitting.** Closed form, canonical, unusually sharp
   assertions. Ma, Qi, Zhao, Lo, PRA 72, 012326 (2005) with the GYS parameter set (alpha 0.21 dB/km,
   e_det 3.3%, Y_0 1.7e-6, eta_Bob 0.045, f_EC 1.22): non-decoy R/eta^2 approximately 0.15 flat,
   decoy R/eta approximately 0.11 flat; non-decoy dies near 42 km, decoy survives past 140 km.
   The sharpest single claim: **with GYS parameters and mu = 0.1 non-decoy, the tagged fraction
   exceeds 1 at zero distance** - no key even back-to-back.

Two cheap anchors are worth having as well: intercept-resend, where BB84 gives **exactly 25%**
(Gottesman-Lo, IEEE TIT 49, 457), and double-click handling, the one ISO 23837-2 activity that is
pure protocol logic.

**Verified already passing, 2026-08-08:** setting the eavesdropper fraction to 1.0 with zero
misalignment produced QBER **0.25 exactly**, 0 secure key bits, and a failed-step marker. The
existing channel model reproduces the textbook value without modification.

## Thresholds, corrected

Three figures in common circulation are wrong or misattributed, and this project should not repeat
them. From Gottesman-Lo Table 1, read directly:

| | one-way | two-way |
|---|---|---|
| BB84 proven INSECURE above | 14.6% | 25% |
| BB84 proven secure below | **11.0%** | 18.9% |
| Six-state proven secure below | 12.7% | 26.4% |

- **11.0%** is Shor-Preskill, the point where 1 - 2H(delta) reaches zero (11.0028%).
- **14.6% is not a security threshold.** It is the one-way *insecurity* bound from optimal
  approximate cloning, and is frequently misquoted as a two-way security threshold.
- **12.6% and 12.9% are not canonical.** The nearby real numbers are 12.7% (Lo, six-state one-way)
  and 12.4% (Kraus-Gisin-Renner, BB84 one-way with noisy preprocessing).

## Declined, with reasons

**Detector blinding presented as a QBER anomaly.** The natural framing would teach the opposite of
what the primary sources establish. Gerhardt et al., Nature Communications 2, 349 (2011) report every
off-diagonal element exactly zero over 8.7M faked states and state that "inserting Eve does not alter
the rates" - QBER was 5-6% with and without Eve, i.e. **zero excess QBER**. Lydersen et al., Nature
Photonics 4, 686 (2010) never ran an end-to-end attack and report no attack QBER at all. If this is
ever modelled it must be as *silent compromise* - full key extracted, monitoring unchanged.

**ETSI GS QKD 010 Trojan-horse conformance, and the 21 optical evaluation activities of ISO
23837-2.** These need real photons. The standards decline to supply both the thresholds (section 5.3.5) and
the privacy-amplification formula (GS QKD 010 clause 8), so a browser simulation would be inventing
the inputs and the pass criteria.

**Any aggregate stress score.** This is the third time this project has been offered a composite and
the third time it should decline, and there is now an authority for it. Makarov et al., Phys. Rev.
Applied 22, 044076 (2024) - the group that actually prepared a commercial QKD system for
certification - found ISO/IEC 23837's own attack-potential scale unusable: all the vulnerabilities
they examined score as "highly resistant to attack" or beyond, so "it is then difficult to
differentiate between the vulnerabilities, as they tend to be off that scale". They substituted a
scale they themselves call empirical and adopted temporarily. If the leading QKD hacking group calls
the ISO scale unusable and their own replacement provisional, a synthesised index in a browser
simulator has no defensible calibration.

## The gap this would fill is real

Checked against primary text, not documentation claims: **SeQUeNCe** has no eavesdropper;
**QuNetSim** has a generic sniffing hook with no photonic or detector physics; **NetSquid** derives
QBER from hardware noise only; **qkdSim**'s Assumption 7 states verbatim that "no eavesdropping
strategy or attack has been considered for the simulation"; an independent peer-reviewed analysis
found **QKDNetSim** does not model the quantum layer at all. A search result claiming qkdSim
simulates backflash, Trojan-horse and blinding attacks is false and contradicted by that paper's own
assumption list.

## Vendor material identified and excluded

EE Times "Test and Measurement Strategies for QKD, PQC and Hybrid Systems" and "Five Test
Considerations to Prepare for Q-Day"; Fierce Network "Test and Verification for Quantum-Safe
Strategies" (marked sponsored); Semiconductor for You. The "physical stress testing using an emulated
network" recommendation that surfaces in search results has no published methodology behind it.
`openfhe.org/security-of-ckks/` is project blog prose with no modes, formulas or methodology - the
citable OpenFHE material is the security notes in the documentation, not that page.

## Related

`security-scoring-fact-check.md` records the five scoring frameworks declined earlier in the same
round, for the same reason: invented weights with no calibration.
