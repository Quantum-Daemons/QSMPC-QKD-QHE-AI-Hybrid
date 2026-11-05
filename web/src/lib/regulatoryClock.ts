/**
 * The regulatory clock, as data - one row per dated obligation, each with WHO it binds and HOW firm it
 * is. It replaces a 191-word paragraph whose dates were correct and whose scopes were easy to lose:
 * "CNSA 2.0 requires ... from 2027" binds national security systems only, EO 14412's contractor rule
 * is directed but not yet proposed, NIST IR 8547 is final (October 2026, doi:10.6028/NIST.IR.8547),
 * the 2029 dates are company targets, and the EU AI Act's transparency article for high-risk systems
 * is deferred. The scope travels with every date here, so no date can be quoted without it (the
 * user's wording policy: years as timeframes with their scope).
 *
 * Every row was read at its primary source on 2026-10-23/24, and the IR 8547, FAR, FIPS 140-2 and
 * Art. 50 rows again on 2026-10-30 - see `docs/pqc-roadmap.md` and the Round 58 table in
 * `docs/deliverable-verification.md` for the documents and sections.
 *
 * Read at whitehouse.gov on 2026-10-31 (Round 84): EO 14412 sec. 4(b) does not bind agencies by
 * itself - it directs the Director of OMB to "issue guidance requiring each agency to" move HVAs and
 * high impact systems to PQC key establishment by December 31, 2030 and to PQC signatures by
 * December 31, 2031 - and OMB M-26-15 (2026-06-24) is that guidance (Phase 3, 2028-2030; Phase 4,
 * 2031), so both rows name the two. The TLS 1.3 row is M-26-15 sec. 4: "Consistent with Executive
 * Order 14306, agencies must support Transport Layer Security (TLS) protocol version 1.3 or a
 * successor version as soon as practicable, but not later than January 2, 2030." EO 14306 (June 6,
 * 2025), sec. 2(d), rewrote EO 14144 sec. 4(f): its (ii) has the Director of NSA issue that
 * requirement for national security systems and the Director of OMB for the rest, and M-26-15
 * "does not apply to national security systems".
 */
export type Firmness = "in force" | "requirement" | "ordered" | "draft" | "target" | "passed" | "deferred";

export type Lane = "NIST" | "US federal" | "US NSS" | "EU" | "Companies";

/** A primary source for a row: what the link names, and where it goes. */
export interface Source {
  label: string;
  url: string;
}

/**
 * Each instrument's primary source (checklist UX42, Round 86). The page cited every instrument below as
 * text only, so a reader who wanted to check a date had to search for it. Every URL was fetched on
 * 2026-11-02 and resolved: EUR-Lex answers a script with its challenge page (202), and NSA's media server
 * refuses scripts (403), so the CNSA 2.0 FAQ link is the one NSA's own Post-Quantum Cybersecurity
 * Resources page gives, read in a browser.
 */
export const SOURCES = {
  fips: { label: "NIST announcement", url: "https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards" },
  aiAct: { label: "AI Act, EUR-Lex", url: "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng" },
  omnibus: { label: "Regulation (EU) 2026/1744", url: "https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng" },
  cmvp: { label: "NIST CMVP", url: "https://csrc.nist.gov/projects/cryptographic-module-validation-program" },
  m2615: { label: "OMB M-26-15", url: "https://www.whitehouse.gov/wp-content/uploads/2026/06/M-26-15-Execution-of-the-Migration-to-Post-Quantum-Cryptography.pdf" },
  eo14412: { label: "EO 14412, Federal Register", url: "https://www.federalregister.gov/documents/2026/06/25/2026-12909/securing-the-nation-against-advanced-cryptographic-attacks" },
  eo14306: { label: "EO 14306", url: "https://www.whitehouse.gov/presidential-actions/2025/06/sustaining-select-efforts-to-strengthen-the-nations-cybersecurity-and-amending-executive-order-13694-and-executive-order-14144/" },
  nisRoadmap: { label: "EU roadmap, European Commission", url: "https://digital-strategy.ec.europa.eu/en/library/coordinated-implementation-roadmap-transition-post-quantum-cryptography" },
  cnsa: { label: "NSA CNSA 2.0 FAQ", url: "https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0__FAQ_v2.2.PDF" },
  ir8547: { label: "NIST IR 8547", url: "https://csrc.nist.gov/pubs/ir/8547/final" },
  cloudflare: { label: "Cloudflare", url: "https://blog.cloudflare.com/post-quantum-roadmap/" },
  google: { label: "Google", url: "https://blog.google/innovation-and-ai/technology/safety-security/cryptography-migration-timeline/" },
} as const satisfies Record<string, Source>;

export interface ClockRow {
  lane: Lane;
  /** A two-to-four-word label for the timeline marker. */
  short: string;
  /** ISO date, or a year when the source gives only a year. */
  when: string;
  /** Fractional year for the timeline axis. */
  t: number;
  instrument: string;
  binds: string;
  what: string;
  firmness: Firmness;
  /** The instrument's primary sources, linked beside it on the page. */
  sources: readonly Source[];
}

export const CLOCK: ClockRow[] = [
  { lane: "NIST", short: "FIPS 203-205 final", when: "2024-08", t: 2024.6, instrument: "NIST FIPS 203 / 204 / 205", binds: "standards",
    what: "ML-KEM, ML-DSA and SLH-DSA published as final standards", firmness: "in force", sources: [SOURCES.fips] },
  { lane: "EU", short: "AI Act Art. 50", when: "2026-08-02", t: 2026.58, instrument: "EU AI Act, Art. 50", binds: "EU: chatbots, synthetic content and deep fakes, emotion-recognition and biometric-categorisation deployers",
    what: "transparency obligations apply, the Act's general date of application; Art. 50(2) systems already on the market have until 2026-12-02 (Regulation (EU) 2026/1744); governance (Chapter VII) has applied since 2025-08-02", firmness: "in force", sources: [SOURCES.aiAct, SOURCES.omnibus] },
  { lane: "NIST", short: "FIPS 140-2 historical", when: "2026-09-22", t: 2026.72, instrument: "NIST CMVP", binds: "US and Canadian federal agencies",
    what: "FIPS 140-2 certificates active until 2026-09-21 and on the CMVP Historical List from 2026-09-22: existing systems only; FIPS 140-3 for new systems that need validation", firmness: "passed", sources: [SOURCES.cmvp] },
  { lane: "US federal", short: "agency plans due", when: "~2026-10-22", t: 2026.81, instrument: "OMB M-26-15", binds: "US federal agencies",
    what: "agency PQC migration plans due (120 days after 2026-06-24)", firmness: "passed", sources: [SOURCES.m2615] },
  { lane: "US federal", short: "FAR proposal due", when: "~2026-12-19", t: 2026.97, instrument: "EO 14412, sec. 6(c)", binds: "US federal contractors",
    what: "FAR Council ordered to publish, within 180 days of the order, a proposed rule amending the FAR to require covered contractors to comply by 2030-12-31 with NIST's FIPS, including all applicable FIPS incorporating PQC algorithms; no such proposed rule in the Federal Register as of 2026-11-05", firmness: "ordered", sources: [SOURCES.eo14412] },
  { lane: "EU", short: "EU first steps", when: "2026-12-31", t: 2027.0, instrument: "EU NIS CG PQC roadmap v1.1", binds: "EU Member States",
    what: "first steps; national roadmaps; planning and pilots initiated for high- and medium-risk use cases", firmness: "target", sources: [SOURCES.nisRoadmap] },
  { lane: "US NSS", short: "new NSS acquisitions", when: "2027-01-01", t: 2027.0, instrument: "CNSSP 15 (CNSA 2.0)", binds: "US national security systems ONLY",
    what: "CNSA 2.0 algorithms required in all new acquisitions", firmness: "requirement", sources: [SOURCES.cnsa] },
  { lane: "EU", short: "Art. 13 (deferred)", when: "2027-12-02", t: 2027.92, instrument: "EU AI Act, Art. 13 (Annex III)", binds: "EU high-risk AI systems",
    what: "transparency for high-risk systems - Chapter III Sections 1-3, which contain Art. 13, deferred by the AI Omnibus (Regulation (EU) 2026/1744, in force 27 Jul 2026)", firmness: "deferred", sources: [SOURCES.aiAct, SOURCES.omnibus] },
  { lane: "EU", short: "Annex I (deferred)", when: "2028-08-02", t: 2028.58, instrument: "EU AI Act (Annex I products)", binds: "EU high-risk AI in regulated products",
    what: "high-risk obligations for AI embedded in regulated products (deferred)", firmness: "deferred", sources: [SOURCES.aiAct, SOURCES.omnibus] },
  { lane: "Companies", short: "2029 targets", when: "2029", t: 2029.5, instrument: "Cloudflare / Google", binds: "their own products",
    what: "Cloudflare targets full PQ security including authentication; Google sets its PQC migration timeline", firmness: "target", sources: [SOURCES.cloudflare, SOURCES.google] },
  { lane: "US federal", short: "TLS 1.3 support", when: "2030-01-02", t: 2030.0, instrument: "OMB M-26-15, sec. 4, under EO 14306 (EO 14144 sec. 4(f)(ii) as amended)", binds: "US federal agencies, not NSS (for NSS, EO 14306 has NSA issue the requirement)",
    what: "agencies support TLS 1.3 or a successor version as soon as practicable, and not later than this date", firmness: "requirement", sources: [SOURCES.m2615, SOURCES.eo14306] },
  { lane: "NIST", short: "deprecate RSA/ECC", when: "after 2030", t: 2030.5, instrument: "NIST IR 8547 (final, Oct 2026), Tables 2 and 4", binds: "use under NIST standards and guidelines; NIST's announced schedule (SP 800-131A Rev. 3, which sets transition status, is still a draft), binding on federal migration plans through OMB M-26-15",
    what: "all quantum-vulnerable public-key algorithms (RSA, ECDSA and EdDSA signatures; finite-field and elliptic-curve DH and MQV; RSA key establishment) deprecated: usable at the data owner's risk. The earlier EO 14412 / OMB M-26-15 dates for high value assets and high impact systems still apply", firmness: "requirement", sources: [SOURCES.ir8547] },
  { lane: "US federal", short: "PQC key establishment", when: "2030-12-31", t: 2031.0, instrument: "EO 14412, sec. 4(b), via OMB M-26-15 (Phase 3)", binds: "US federal high-value assets and high impact systems (not NSS); OMB M-26-15 adds systems with highly sensitive data and systems an agency finds likely to be particularly vulnerable",
    what: "PQC key establishment", firmness: "requirement", sources: [SOURCES.eo14412, SOURCES.m2615] },
  { lane: "US NSS", short: "phase-out", when: "2030-12-31", t: 2031.0, instrument: "CNSSP 15 (CNSA 2.0)", binds: "US national security systems ONLY",
    what: "equipment that cannot support CNSA 2.0 phased out", firmness: "requirement", sources: [SOURCES.cnsa] },
  { lane: "EU", short: "high-risk done", when: "2030-12-31", t: 2031.0, instrument: "EU NIS CG PQC roadmap v1.1", binds: "EU Member States",
    what: "transition completed for high-risk use cases", firmness: "target", sources: [SOURCES.nisRoadmap] },
  { lane: "US federal", short: "PQC signatures", when: "2031-12-31", t: 2032.0, instrument: "EO 14412, sec. 4(b), via OMB M-26-15 (Phase 4)", binds: "US federal high-value assets and high impact systems (not NSS); OMB M-26-15 adds systems with highly sensitive data and systems an agency finds likely to be particularly vulnerable",
    what: "PQC digital signatures", firmness: "requirement", sources: [SOURCES.eo14412, SOURCES.m2615] },
  { lane: "US NSS", short: "all NSS use", when: "2031-12-31", t: 2032.0, instrument: "CNSSP 15 (CNSA 2.0)", binds: "US national security systems ONLY",
    what: "CNSA 2.0 mandatory for all use", firmness: "requirement", sources: [SOURCES.cnsa] },
  { lane: "NIST", short: "disallow RSA/ECC", when: "after 2035", t: 2035.5, instrument: "NIST IR 8547 (final, Oct 2026), Tables 2 and 4", binds: "use under NIST standards and guidelines; NIST's announced schedule (SP 800-131A Rev. 3, which sets transition status, is still a draft), binding on federal migration plans through OMB M-26-15",
    what: "all quantum-vulnerable public-key algorithms, RSA-3072 and P-256 included, disallowed", firmness: "requirement", sources: [SOURCES.ir8547] },
  { lane: "EU", short: "medium-risk done", when: "2035-12-31", t: 2036.0, instrument: "EU NIS CG PQC roadmap v1.1", binds: "EU Member States",
    what: "medium-risk completed; low-risk \"completed as much as feasible\"", firmness: "target", sources: [SOURCES.nisRoadmap] },
];
