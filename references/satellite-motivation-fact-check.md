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

**CCSDS 355.0-B-2**, *Space Data Link Security Protocol* (Blue Book, July 2022), provides
"authentication and/or **data confidentiality** at the Data Link Layer" - but only "to the contents
of Transfer Frames used by Space Data Link Protocols **over a space link**."

**SDLS terminates at the ground station.** Telemetry is plaintext from that point on, throughout the
ground segment, which is where processing and ML actually happen. NIST IR 8401 addresses the same
seam from the other side, telling operators that "third-party suppliers or services such as
cloud-based infrastructures" need data flows "encrypted (with independent key management)".

That seam - after SDLS, inside the ground segment - is exactly where this project's encrypted
inference sits.

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
| ESA's AI-assurance project (PINEBERRY) as precedent | Real, but about explainability, robustness and poisoning - not confidentiality |
| ESA's PPML activity as telemetry precedent | Scoped to Earth-observation imagery, not telemetry or ground segment |
| Federated learning in LEO constellations | The eight verified papers are motivated by **downlink bandwidth**, not inter-operator confidentiality |

One further caution: the multi-tenant ground-segment tension is real and vendor statements support
it - AWS Ground Station documents that it "recommend[s] that **you** follow industry best practices
and implement end-to-end encryption", which pushes confidentiality onto the tenant rather than
claiming the provider cannot see plaintext - but **no operator is on record refusing shared ground
stations over telemetry confidentiality.** That is a plausible inference, and this project should
label it as one.
