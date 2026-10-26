/**
 * What the EU AI Act would ask of each use case's model, were it placed on the EU market for the
 * purpose its page demonstrates. The demo itself is placed on no market; this is a reading of the Act
 * for a demonstration, not legal advice.
 *
 * WHY PER USE CASE (Round 69). The SHAP panel printed one sentence on every page, giving Art. 13 the
 * Annex III date (2 Dec 2027) everywhere, and it was right for none of them.
 * - Art. 13 binds HIGH-RISK systems only. Two of the five use cases are not high-risk on the grounds
 *   the Act gives. Finance: Annex III 5(b) covers the creditworthiness of natural persons and excludes
 *   fraud detection, and money-laundering monitoring evaluates transactions, not creditworthiness.
 *   Telecom: recital 55 says a component used solely for cybersecurity is not a safety component, so
 *   Annex III 2 does not reach an intrusion detector.
 * - The two that would be high-risk reach it through Art. 6(1) and Annex I. Diagnostic software is a
 *   medical device of class IIa or higher (MDR, Annex VIII Rule 11), and polygenic risk software is an
 *   in-vitro diagnostic of class C (IVDR, Annex VIII Rule 3). A notified body assesses both, so their
 *   date is the Annex I one, 2 August 2028, not the Annex III one.
 *
 * Both dates come from the regulatory clock, which carries their source (Regulation (EU) 2026/1744),
 * so the two cannot drift apart.
 *
 * CONDITIONAL, BY DECISION (Round 69, before merge). Where the answer turns on who deploys the model
 * or why, the sentence says so rather than picking one deployment. AML is not high-risk for a bank,
 * but is under Annex III 6(e) when a law-enforcement authority uses it to profile people while
 * detecting offences. A system used exclusively for military, defence or national security purposes
 * is outside the Act (Art. 2(3)), which matters on the "Telecom / Defense" page. Recital 55 is a
 * recital, not an article, so its reading is attributed to it rather than stated as the rule. The page
 * adds `AI_ACT_CAVEAT`.
 *
 * SATELLITE, READ AGAIN IN ROUND 72 at EUR-Lex. "Depends on how it is deployed" was right and hid
 * why. Annex III 2 names critical DIGITAL infrastructure, which recital 55 ties to point 8 of the
 * CER Directive's annex, and space is a separate sector there (point 10), as it is in NIS2. So a
 * telemetry detector qualifies only if the satellite service is itself digital infrastructure, such
 * as a public electronic communications network, and only if the detector serves as a safety
 * component: Regulation (EU) 2026/1744 added Art. 6(1a), under which a system used solely for
 * performance or other non-safety purposes is not one. Art. 2(3) names military purposes too, which
 * the earlier sentences left out.
 */
import { CLOCK } from "./regulatoryClock";

function clockDate(short: string): string {
  const row = CLOCK.find((r) => r.short === short);
  if (!row) throw new Error(`regulatory clock has no row "${short}"`);
  return row.when;
}

/** Chapter III Sections 1-3 (Art. 6-27, Art. 13 among them) for Annex III systems. */
export const ANNEX_III_DATE = clockDate("Art. 13 (deferred)");
/** The same requirements for systems that are high-risk through Art. 6(1) and Annex I. */
export const ANNEX_I_DATE = clockDate("Annex I (deferred)");

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
  "October", "November", "December"];
/** "2028-08-02" -> "2 August 2028". */
export function longDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
}

/** Printed after every sentence. */
export const AI_ACT_CAVEAT = "A reading of the Act for this demonstration, not legal advice.";

export interface AiActScope {
  /** Whether the high-risk requirements, Art. 13 among them, would bind the model in the deployment the
   *  page demonstrates. Other deployments are in `text`. */
  highRisk: "yes" | "no" | "depends";
  /** The provisions that decide it. */
  basis: string;
  /** From when the requirements apply, where they do. */
  from: string | null;
  /** The sentence the page prints. */
  text: string;
}

export const AI_ACT_SCOPE: Record<string, AiActScope> = {
  medical_fl: {
    highRisk: "yes",
    basis: "Art. 6(1) and Annex I: Regulation (EU) 2017/745 (MDR), Annex VIII Rule 11",
    from: ANNEX_I_DATE,
    text: "High-risk if placed on the EU market as diagnostic software, which is a medical device of class "
      + "IIa or higher (MDR Rule 11), so Art. 6(1) and Annex I apply, and Art. 13 with them from "
      + `${longDate(ANNEX_I_DATE)}.`,
  },
  genomics: {
    highRisk: "yes",
    basis: "Art. 6(1) and Annex I: Regulation (EU) 2017/746 (IVDR), Annex VIII Rule 3",
    from: ANNEX_I_DATE,
    text: "High-risk if placed on the EU market to predict disease risk, which makes it an in-vitro "
      + "diagnostic of class C (IVDR Rule 3, human genetic testing), so Art. 6(1) and Annex I apply, and "
      + `Art. 13 with them from ${longDate(ANNEX_I_DATE)}.`,
  },
  finance_aml: {
    highRisk: "no",
    basis: "Annex III 5(b) (creditworthiness, fraud detection excepted) for a bank; Annex III 6(e) for law enforcement",
    from: null,
    text: "When a bank uses it for transaction monitoring, not high-risk: Annex III 5(b) covers the "
      + "creditworthiness of natural persons, not money-laundering monitoring, so Art. 13 does not bind it. "
      + "Used by or for a law-enforcement authority to profile people while detecting offences, it would "
      + `be high-risk (Annex III 6(e)), with Art. 13 from ${longDate(ANNEX_III_DATE)}.`,
  },
  telecom_ids: {
    highRisk: "no",
    basis: "Annex III 2 (safety components of critical digital infrastructure) read with recital 55; Art. 2(3)",
    from: null,
    text: "As a tool used solely for cybersecurity, not high-risk on the reading recital 55 gives, since "
      + "Annex III 2 covers safety components of critical digital infrastructure, so Art. 13 does not bind "
      + "it. Used exclusively for military, defence or national security purposes, it is outside the Act "
      + "altogether (Art. 2(3)).",
  },
  satellite_comms: {
    highRisk: "depends",
    basis: "no Annex I or Annex III category names telemetry anomaly detection; Annex III 2 only if the "
      + "satellite service is critical digital infrastructure (recital 55: CER Directive annex point 8; "
      + "space is point 10) and the detector a safety component (Art. 3(14); Art. 6(1a) as amended by "
      + "Regulation (EU) 2026/1744); Art. 2(3)",
    from: ANNEX_III_DATE,
    text: "Not a category the Act lists. It would be high-risk (Annex III 2) only if the satellite service "
      + "is itself critical digital infrastructure, such as a public electronic communications network - "
      + "space is a separate sector under the CER and NIS2 directives - and the detector serves as a safety "
      + "component, not solely for cybersecurity or performance. That depends on how it is deployed, and "
      + `Art. 13 would then apply from ${longDate(ANNEX_III_DATE)}. Used exclusively for military, defence `
      + "or national security purposes, it is outside the Act (Art. 2(3)).",
  },
};
