# Security scoring: sources evaluated and declined (Round 22, 2026-08-08)

Round 22 asked whether this project's 0-100 security score is a defensible, publicly grounded,
quantum-resistance-aware method or an invented formula, and named five candidate replacements.

The answer to the first question was: invented. The answer to the second was: adopt none of them.

**A sixth candidate was named in the same instruction and was neither adopted nor declined here for
eleven rounds - it is decided at the end of this file (Round 33).** "Quantum Risk / Readiness Indices"
appeared in the instruction that drove this work and then appeared nowhere in the repository: zero hits
across `references/`, `docs/` and `README.md`. Four candidates got verdicts and the fifth got silence,
which is the worse outcome of the two, because a decline is re-readable and a silence looks like an
oversight for as long as anyone keeps looking.

This file records the verdicts so a later round does not re-evaluate the same five sources and reach
the same conclusion at the same cost. Each was read directly, not summarised from search results -
which mattered, because the search summaries of the first two were considerably more flattering than
the papers.

## Why the existing score had to go

Of 17 fields in `SecurityScoringConfig`, exactly one is traceable to a citation (`leakage_low 0.1`,
Sulimany et al., PRX 15 041056). Two are traceable by name only. Fourteen are unexplained and none
carries a stated rationale. The maxima sum to exactly 30+30+20+10+10 = 100, which is the signature of
a number designed backwards from its total.

The headline 68-to-96 gap was mostly an artefact of the call sites rather than of the cryptography:
+12 for a substring match on a hand-written label, +10 because the classical call passed no QBER
while the quantum call passed a literal, +6 for the same pattern on leakage. Sixteen of the 28 points
were a difference in which arguments two functions chose to supply.

It also barely moved - 2 of 5 selectors reached it, giving 4 distinct values on a 0-100 axis whose
bottom 63 percent was unreachable - and in the Python path both rows were literal constants for every
configuration, because the caller passed neither the KEM nor the QKD key length.

## The five candidates

| Source | What it actually is | Verdict |
|---|---|---|
| QERS, arXiv:2601.13399 | Weights are author-selected across three scenarios; the paper says ML "could" optimise them and gives no calibration methodology. Normalisation bounds come from an ESP32C6 plus Raspberry Pi Wi-Fi testbed. Three of seven inputs are RSSI, packet loss and energy - none exists in a browser. Scope is resource-constrained IoT/IIoT and the paper states applicability beyond it "remains unaddressed". | Decline |
| CARS, IACR ePrint 2026/1467 | No formula published. Weights come from a two-round Delphi process with 12 engineers. "Calibrated against 43 repositories" describes measurement of codebases, with no optimisation procedure and no validation against ground truth. Scores a codebase's migration readiness - version-level and static, not per-run. | Decline |
| quantum-safe library assessment, arXiv:2605.17061 | A ternary Full/Partial/None rubric over eight dimensions, scoped to libraries. Explicitly unsuitable for scoring a single operational system's security configuration. | Decline |
| VIAVI blog | Vendor marketing. No formula, no thresholds, no acceptance criteria; promotes named test products. | Decline |
| fierce-network (sponsored), pqshield, qtonic, qodequay | The same genre - one sponsored article, two product pages, one vendor guide. The most methodological-sounding was sampled and was marketing. | Decline |

Adopting QERS or CARS would have replaced invented weights with different invented weights, and
added a citation implying a rigour that does not transfer to this setting. That is worse than the
prior state, because the prior state at least did not claim external grounding.

## What was adopted instead

One idea from arXiv:2605.17061 transfers, without any of its content: **per-dimension verdicts
sourced from specifications, rather than a synthesised number.** This project already used that shape
successfully in the provenance badge's five-state enum.

So the composite was retired in favour of reported facts, each traceable to a standard or to this
run:

- NIST security category as the headline - ML-KEM-768 is Category 3, ML-KEM-1024 is Category 5,
  FIPS 203. This is the one figure here that is standard, citable, and comparable against another
  system.
- Signature level (ML-DSA-65 / -87, FIPS 204), whether QKD key material is mixed into the session
  key, secure key bits and SKR as measured by the run, and the QBER verdict.
- The performance overheads the run already measures: FHE-linear ms, MPC-non-linear ms,
  communication KB, ciphertext/plaintext ratio, KEM ciphertext bytes, signature bytes.

Every one of these is per-use-case verifiable in the E2E simulator, which a synthesised score is not.
That was the deciding argument.

## Defects fixed regardless of presentation

- `secFromConfig` returned no `qber` field, so the consumer guard `typeof sec?.qber === "number"` was
  always false and a constant was used unconditionally - under a comment claiming the Eve slider had
  been wired through. The slider never reached the verdict; one literal had been replaced by another.
- The Python call sites passed neither `kem` nor `qkd_key_bits`, making both rows constant.
- The configured QBER threshold 0.11 sits **above** this repository's own measured GLLP zero-key
  point of 0.098106, so a channel in [0.098, 0.11) scored full marks while producing zero key.

## The sixth candidate: Quantum Risk / Readiness Indices - DECLINED 2026-08-12 (Round 33)

The first thing to say is what the honest answer is **not**. It would have been convenient to report
that no citable specification exists under that name, and that is false. Several do, they are recent,
and some are from national agencies rather than vendors. Researched at the primary sources:

| instrument | what it is | unit assessed | dimensions |
|---|---|---|---|
| Singapore CSA **Quantum Readiness Index**, v1 2026-07-16 | self-assessment questionnaire accompanying a quantum-safe migration handbook | an organisation | five domains |
| HKMA **Quantum Preparedness Index**, launched 2026-07-27 | sector instrument for banks, published with a whitepaper | a financial institution | Awareness, Planning, Pilots, Practical Preparedness |
| IBM IBV **Quantum Readiness Index 2025** | survey-derived weighted 100-point scale over 45 activities in 11 sub-categories, fitted by regression | an enterprise | strategy, technology, operations |
| IQM **State of Quantum 2026** Quantum Readiness Index | composite score with five named tiers (Aware 0-19 through Leading 80-100) | a country or organisation | Workforce, Innovation, Investment, Adoption |
| WEF **Quantum Readiness Toolkit** | principles-based assessment and prioritisation framework | an organisation | governance, risk, talent, technology |
| **QRAMM** | quantum-readiness maturity model with a free automated-scoring toolkit and compliance mapping | an enterprise | maturity levels |
| Mosca, **A Methodology for Quantum Risk Assessment** (Global Risk Institute / evolutionQ) | the x + y > z inequality: asset lifetime plus migration time against time-to-threat | an organisation's asset portfolio | three time parameters |

**The name in the instruction is plural, and that turns out to be the finding.** "Indices" is a genre,
not a specification. Its members do not agree with each other on how many dimensions there are - three,
four, four, five, and a maturity ladder - and none publishes a formula that this project could
reimplement and check. Adopting "the" quantum readiness index would mean picking one of seven and
presenting a choice as a standard, which is the failure the QERS and CARS declines already refused.

**The deciding argument is the same one that decided CARS, and it is stronger here.** Every instrument
in that table scores an **organisation's migration programme**: cryptographic asset inventory, data
shelf life, governance, workforce, investment, procurement. This project's scoring surface is a **single
run of a cryptographic pipeline** - a KEM parameter set, a signature level, whether QKD key material was
mixed into the session key, measured secure key bits, measured QBER, measured overheads. Those two have
no inputs in common. CARS was declined for being "version-level and static, not per-run"; the readiness
indices are **organisation-level and annual**, which is one step further from a per-run verdict, not
closer. There is nothing to blend, and blending them would produce a score whose inputs the demo cannot
supply and whose supplied inputs it would have to invent - three of QERS's seven inputs already failed
on exactly that test.

Mosca's inequality is the one member of the genre that is genuinely specifiable in a sentence and has a
primary source, so it is worth saying explicitly why it also does not apply: x, y and z are an asset's
confidentiality lifetime, an organisation's migration duration and a threat timeline. None of the three
is a property of a run, and a demo that asserted values for them would be publishing an opinion about
the world with a citation attached to it. That is the "different invented weights" objection again.

**Declined. Reason: every citable instrument under this name scores an organisation's migration
programme, and this project's surface is one run of a cryptographic pipeline, so the inputs do not
exist.** The NIST PQC category headline adopted above is the right shape for a per-run surface for the
same reason it was chosen originally: it is a property of the configuration that actually ran.

**Reopening condition, so this is a decision rather than a second silence.** If this project ever
publishes a **migration-programme** surface - a cryptographic inventory, a data-shelf-life claim, a
migration schedule - then adopt the **Singapore CSA Quantum Readiness Index** rather than any of the
others, and cite it by version and date. It is the one that is a published self-assessment instrument
from a national agency with a stated version, rather than a survey ranking of a population or a vendor
maturity model, so it is the one whose provenance would survive the scrutiny this file applies. Nothing
in the current demo produces such a surface, and until something does, the correct number of quantum
readiness indices in this repository is zero.

## Related

The claim in `README.md` that the security score was "the production-relevant number" was withdrawn
in the same round.
