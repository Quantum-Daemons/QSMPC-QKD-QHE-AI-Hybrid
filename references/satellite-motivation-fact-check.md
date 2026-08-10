# Why encrypted inference on satellite telemetry, and the obvious reason that is false
### Round 24m, 2026-08-10

The satellite use case had no stated motivation beyond "free-space QKD + QSDC over a LEO downlink" -
a description of the mechanism, not of the problem. This is the checked version, and the first thing
it does is remove the argument almost everyone reaches for.

## The obvious motivation is FALSE, and stating it would discredit the rest

The intuitive claim is that export control blocks sending satellite telemetry to a third party for
processing. **It does the opposite, explicitly.** 22 CFR 121.1, USML Category XV, Note 3 to
paragraph (f), verbatim:

> "Paragraph (f) and ECCNs 9E001, 9E002 and 9E515 **do not control the data transmitted to or from a
> satellite or spacecraft** ... when limited to information about the health, operational status, or
> measurements or function of, or raw sensor output from, the spacecraft ... commonly referred to as
> **'housekeeping data'** ... temperatures, pressures, power, currents, voltages, and battery charges
> ... The act of processing such telemetry data ... or **encrypting it does not, in and of itself,
> cause the telemetry data to become subject to the ITAR**."

That is precisely the data an anomaly detector consumes. The note is current - a pending amendment at
91 FR 46281 (23 Jul 2026) concerns Category I firearm suppressors, not Category XV.

**This project has never made the ITAR claim, and must not start.** Recorded because it is the
single most tempting wrong sentence available here, and a reader who knows Category XV would stop
reading at it.

What *is* still controlled, and is the honest hook: the ground-segment **software and technology**
(ECCN 9D515 / 9E515, USML XV(b) for control systems), telemetry processing equipment
(ECCN 9A515.b.2.a.1), and TT&C encryption equipment for military systems (USML XIII(b)(1)). The
*system* is controlled; the housekeeping *data* is not.

## The strongest motivation: operators treat telemetry as confidential, and it demonstrably blocks ML

Three independent primary sources, each stating it in their own words while publishing data:

**DLR German Space Operations Centre** - Schefels, Schlag & Helmsauer, *"Synthetic satellite telemetry
data for machine learning"*, CEAS Space Journal, 2025, DOI 10.1007/s12567-024-00589-1:

> "**Since satellite telemetry data are often strictly confidential**, we can share these synthetic
> data easily with our research partners."

A national space operations centre built an entire synthetic-data pipeline *because* it cannot share
the real thing. That is the cleanest possible statement of the problem.

**ESA**, publishing its own benchmark (arXiv:2406.17826, Supplementary 2.3):

> "The anonymisation had to be applied **to conform with the ESA privacy policy and to avoid any
> accidental disclosure of sensitive mission-specific information or metadata**."

ESA renamed missions, subsystems, channels, telecommands and physical units, and **scaled the
timeline by a non-disclosed factor**, purely to publish. This project uses that benchmark, so the
anonymisation is not abstract - it is why the channels here are numbered rather than named.

**The OPS-SAT benchmark team** (*Scientific Data* 2025, DOI 10.1038/s41597-025-05035-3):

> "This kind of data is **difficult and costly to obtain, often confidential**" ... "there are **no
> publicly available datasets of real satellite telemetry** accompanied with the ground-truth
> annotations"

**And an operator asking for secure computation by name-in-substance**, eleven years early. Mark
Rawlins, then Space Data Association chairman (SpaceNews, 28 May 2015):

> "We recognize some parts of the community are **never going to be able to share their data** and
> we're looking at solutions where ... **we still protect our assets without having to know what
> their assets are doing.**"

The SDA's structure corroborates it: an Isle of Man entity whose submissions to UN COPUOS (Apr 2024)
and the US Department of Commerce (Feb 2023) describe "binding legal agreements", "firewalled
processing to protect data", "retransmission to third parties prohibited", among participants who
"routinely compete as rivals". Operators will pool operational data only behind enforceable
confidentiality - which is the problem cryptography is supposed to replace with a guarantee.

## The legal hook: export law already blesses this exact architecture

ITAR 22 CFR 120.54, and its EAR twin 15 CFR 734.18:

> **120.54(b)(1)** "end-to-end encryption is defined as ... the data is not in an unencrypted form
> between an originator and an intended recipient; and **the means of decryption are not provided to
> any third party**."
>
> **120.54(c)** "**The ability to access technical data in encrypted form** ... **does not constitute
> the release or export of such technical data**."

So US export law already recognises "a third party computes on data it cannot decrypt" as a distinct
and privileged category. That applies to the ground-segment software and technology, which *is*
controlled, even though the telemetry itself is not. It is the correct legal citation for this use
case, and it is the opposite of the ITAR-as-barrier argument.

Related and real: 15 CFR 960.9(a)(1)(i) requires NOAA commercial remote-sensing licensees to be able
to apply NIST-approved encryption with **at least a 256-bit key** to TT&C communications.

## The architectural gap this fills, stated precisely

**CCSDS 355.0-B-2**, *Space Data Link Security Protocol* (Blue Book, July 2022), does provide
telemetry confidentiality, and it is worth stating that plainly rather than strawmanning it. Section
2.2.3: the protocol "provides all its functions (authentication, encryption, and authenticated
encryption) for the data in the Transfer Frame Data Field of a **TM** Transfer Frame." The crypto is
specified and current - CCSDS 352.0-B-2 mandates AES-256, GCM, HMAC-SHA-256 - and NASA ships a
working open-source implementation (CryptoLib, GSFC).

Three limits are what leave a gap, and only the third is the one this project addresses:

1. **Scope inside the frame.** Only the Transfer Frame Data Field is encrypted. SDLS "provides
   authentication for some fields in the Transfer Frame Primary Header ... **It does not provide
   encryption for these fields**", and offers nothing at all for the VC_OCF, MC_FSH or MC_OCF
   services.
2. **What operators actually turn on.** The asymmetry is uplink-first. NIST IR 8401 states that
   "**command uplinks** are typically encrypted and authenticated" and makes **no equivalent claim
   for the telemetry downlink**. CCSDS 350.0-G-3 Annex A bears this out from the other direction -
   ISS used 3DES on the uplink only.
3. **Where it stops.** SDLS applies "over a space link" and **terminates at the ground station.**
   Telemetry is plaintext from that point on, throughout the ground segment, which is where
   processing and ML actually happen.

Limit 3 is the seam. NIST IR 8401 approaches it from the other side, telling operators to "respond
to confidentiality breaches of Telemetry, Tracking, and Command (TT&C)" and that data flows to
"third-party suppliers or services such as cloud-based infrastructures" need to be "encrypted (with
independent key management)". That seam - after SDLS, inside the ground segment - is exactly where
this project's encrypted inference sits.

Two citation cautions if this section is ever expanded. The CCSDS security *architecture* document is
**351.0-M-1, a Magenta Book** (Recommended Practice), not a Green Book. And IR 8401 is keyed to
**CSF 1.1**, superseded by CSF 2.0 (NIST CSWP 29, Feb 2024), so its PR.DS-2 style subcategory
identifiers no longer map 1:1 - cite the prose, not the identifiers.

## The honest limitation, from the government, in the same breath as the novelty claim

CISA's Space Systems Critical Infrastructure Working Group assessed this exact stack - *Space Systems
Security and Resilience Landscape: Zero Trust in the Space Environment*, April 2024:

> homomorphic encryption schemes "**have yet to be applied to space systems, and their technology
> readiness level remains low**"
>
> QKD "has been tested in the space environment, but its **technology readiness level remains low**
> ... quantum systems operate on specialized equipment that is costly"

Both halves of this project's satellite story, called low-TRL by a US government body, in writing.
That is the correct citation to carry, and it should sit next to the novelty claim rather than
somewhere quieter, because it does two things at once: it independently corroborates the gap ("yet
to be applied to space systems" is CISA saying the same thing the prior-art search found), and it
sets the register - this is a research demonstrator, not something an operator could fly.

CISA attributes the QKD caution to NSA's published position. nsa.gov was not machine-retrievable
during verification, so **NSA's own wording is unverified here**; cite CISA for the assessment, not
NSA.

The two threat catalogues worth pointing at, both real and both current: ESA **SPACE-SHIELD**
(spaceshield.esa.int), an ATT&CK-style knowledge base that names "Telemetry Protocol Interception"
and "Exfiltration Over TM Channel", and Aerospace Corporation **SPARTA** v4.0 (6 August 2026, 9
tactics / 87 techniques, covers the ground segment). SPARTA is a TTP matrix, not a zero-trust
framework - do not describe it as one.

## A novelty claim that survives an adversarial search, and one that does not

**Survives:** encrypted inference on housekeeping telemetry for anomaly detection. An adversarial
search found **zero** verified work applying FHE, MPC or TEEs to satellite telemetry anomaly
detection. The single paper joining "privacy-preserving" and "satellite telemetry"
(DOI 10.56155/978-81-975670-0-1-3) uses plain FedAvg with no cryptography - "privacy-preserving"
there means data minimisation.

**Does not survive:** anything about conjunction assessment or collision avoidance. That is a
decade-deep field with an active ESA contract - ShareSat (ESA GSTP, Cybernetica, Sharemind MPC,
TRL 4, 2024-2025), Kamm & Willemson 2014, Hemenway et al. SCN 2016, and two 2025 arXiv papers
including a DLR CKKS collision-probability system (arXiv:2501.09397). Claiming novelty there would
collide with prior art immediately.

## Four motivations to avoid, and why

| tempting claim | why not |
|---|---|
| "ITAR blocks moving telemetry" | Affirmatively false; Note 3 decontrols housekeeping data explicitly |
| "SPD-5 mandates telemetry encryption" | It does not. Sec. 4(b)(i) says "authentication **or** encryption" - disjunctive - and every principle is a "should", framed as what operators "should consider, based on risk assessment and tolerance". Sec. 5 disclaims any change to existing authority. SPD-5 is real, is still in effect (EO 14369 of 18 Dec 2025 amends SPD-3, not SPD-5), and is **hortatory** |
| ESA's AI-assurance project (PINEBERRY) as precedent | Real, but about explainability, robustness and poisoning - not confidentiality |
| ESA's PPML activity as telemetry precedent | Scoped to Earth-observation imagery, not telemetry or ground segment |
| Federated learning in LEO constellations | The eight verified papers are motivated by **downlink bandwidth**, not inter-operator confidentiality |

One further caution: the multi-tenant ground-segment tension is real and vendor statements support
it - AWS Ground Station documents that it "recommend[s] that **you** follow industry best practices
and implement end-to-end encryption", which pushes confidentiality onto the tenant rather than
claiming the provider cannot see plaintext - but **no operator is on record refusing shared ground
stations over telemetry confidentiality.** That is a plausible inference, and this project should
label it as one.
