/**
 * The regulatory clock, as data - one row per dated obligation, each with WHO it binds and HOW firm it
 * is. It replaces a 191-word paragraph whose dates were correct and whose scopes were easy to lose:
 * "CNSA 2.0 requires ... from 2027" binds national security systems only, EO 14412's contractor rule
 * is a PROPOSED rule, NIST IR 8547 is a draft, the 2029 dates are company targets, and the EU AI Act's
 * transparency article for high-risk systems is deferred. The scope travels with every date here, so
 * no date can be quoted without it (the user's wording policy: years as timeframes with their scope).
 *
 * Every row was read at its primary source on 2026-10-23/24 - see `docs/pqc-roadmap.md` and the
 * Round 58 table in `docs/deliverable-verification.md` for the documents and sections.
 */
export type Firmness = "in force" | "requirement" | "proposed" | "draft" | "target" | "passed" | "deferred";

export type Lane = "NIST" | "US federal" | "US NSS" | "EU" | "Companies";

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
}

export const CLOCK: ClockRow[] = [
  { lane: "NIST", short: "FIPS 203-205 final", when: "2024-08", t: 2024.6, instrument: "NIST FIPS 203 / 204 / 205", binds: "standards",
    what: "ML-KEM, ML-DSA and SLH-DSA published as final standards", firmness: "in force" },
  { lane: "EU", short: "AI Act Art. 50", when: "2026-08-02", t: 2026.58, instrument: "EU AI Act, Art. 50", binds: "EU: chatbots and synthetic media",
    what: "transparency obligations apply (general applicability, enforcement and governance too)", firmness: "in force" },
  { lane: "NIST", short: "FIPS 140-2 historical", when: "2026-09-21", t: 2026.72, instrument: "NIST CMVP", binds: "US federal module procurement",
    what: "FIPS 140-2 certificates moved to the Historical List: existing systems only; new systems need FIPS 140-3", firmness: "passed" },
  { lane: "US federal", short: "agency plans due", when: "~2026-10-22", t: 2026.81, instrument: "OMB M-26-15", binds: "US federal agencies",
    what: "agency PQC migration plans due (120 days after 2026-06-24)", firmness: "passed" },
  { lane: "US federal", short: "FAR rule proposed", when: "~2026-12-19", t: 2026.97, instrument: "EO 14412, sec. 6(c)", binds: "US federal contractors",
    what: "FAR Council to publish a PROPOSED rule requiring FIPS, including the PQC FIPS, by 2030-12-31 (none published as of 2026-10-23)", firmness: "proposed" },
  { lane: "EU", short: "EU first steps", when: "2026-12-31", t: 2027.0, instrument: "EU NIS CG PQC roadmap v1.1", binds: "EU Member States",
    what: "first steps; national roadmaps; planning and pilots initiated for high- and medium-risk use cases", firmness: "target" },
  { lane: "US NSS", short: "new NSS acquisitions", when: "2027-01-01", t: 2027.0, instrument: "CNSSP 15 (CNSA 2.0)", binds: "US national security systems ONLY",
    what: "CNSA 2.0 algorithms required in all new acquisitions", firmness: "requirement" },
  { lane: "EU", short: "Art. 13 (deferred)", when: "2027-12-02", t: 2027.92, instrument: "EU AI Act, Art. 13 (Annex III)", binds: "EU high-risk AI systems",
    what: "transparency for high-risk systems - Chapter III Sections 1-3, which contain Art. 13, deferred by the AI Omnibus (Regulation (EU) 2026/1744, in force 27 Jul 2026)", firmness: "deferred" },
  { lane: "EU", short: "Annex I (deferred)", when: "2028-08-02", t: 2028.58, instrument: "EU AI Act (Annex I products)", binds: "EU high-risk AI in regulated products",
    what: "high-risk obligations for AI embedded in regulated products (deferred)", firmness: "deferred" },
  { lane: "Companies", short: "2029 targets", when: "2029", t: 2029.5, instrument: "Cloudflare / Google", binds: "their own products",
    what: "Cloudflare targets full PQ security including authentication; Google sets its PQC migration timeline", firmness: "target" },
  { lane: "NIST", short: "draft: deprecate 112-bit", when: "after 2030", t: 2030.5, instrument: "NIST IR 8547 (initial public draft)", binds: "draft only",
    what: "112-bit-strength quantum-vulnerable public-key algorithms (e.g. RSA-2048) deprecated", firmness: "draft" },
  { lane: "US federal", short: "PQC key establishment", when: "2030-12-31", t: 2031.0, instrument: "EO 14412, sec. 4(b)", binds: "US federal high-value assets and high impact systems (not NSS)",
    what: "PQC key establishment", firmness: "requirement" },
  { lane: "US NSS", short: "phase-out", when: "2030-12-31", t: 2031.0, instrument: "CNSSP 15 (CNSA 2.0)", binds: "US national security systems ONLY",
    what: "equipment that cannot support CNSA 2.0 phased out", firmness: "requirement" },
  { lane: "EU", short: "high-risk done", when: "2030-12-31", t: 2031.0, instrument: "EU NIS CG PQC roadmap v1.1", binds: "EU Member States",
    what: "transition completed for high-risk use cases", firmness: "target" },
  { lane: "US federal", short: "PQC signatures", when: "2031-12-31", t: 2032.0, instrument: "EO 14412, sec. 4(b)", binds: "US federal high-value assets and high impact systems (not NSS)",
    what: "PQC digital signatures", firmness: "requirement" },
  { lane: "US NSS", short: "all NSS use", when: "2031-12-31", t: 2032.0, instrument: "CNSSP 15 (CNSA 2.0)", binds: "US national security systems ONLY",
    what: "CNSA 2.0 mandatory for all use", firmness: "requirement" },
  { lane: "NIST", short: "draft: disallow all", when: "after 2035", t: 2035.5, instrument: "NIST IR 8547 (initial public draft)", binds: "draft only",
    what: "all quantum-vulnerable public-key algorithms, P-256 included, disallowed", firmness: "draft" },
  { lane: "EU", short: "medium-risk done", when: "2035-12-31", t: 2036.0, instrument: "EU NIS CG PQC roadmap v1.1", binds: "EU Member States",
    what: "medium-risk completed; low-risk \"completed as much as feasible\"", firmness: "target" },
];
