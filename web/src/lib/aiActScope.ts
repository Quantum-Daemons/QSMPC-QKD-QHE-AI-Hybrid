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

export interface AiActScope {
  /** Whether the high-risk requirements, Art. 13 among them, would bind the model. */
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
    text: "high-risk if placed on the EU market as diagnostic software, which is a medical device of class "
      + "IIa or higher (MDR Rule 11), so Art. 6(1) and Annex I apply, and Art. 13 with them from "
      + `${longDate(ANNEX_I_DATE)}`,
  },
  genomics: {
    highRisk: "yes",
    basis: "Art. 6(1) and Annex I: Regulation (EU) 2017/746 (IVDR), Annex VIII Rule 3",
    from: ANNEX_I_DATE,
    text: "high-risk if placed on the EU market to predict disease risk, which makes it an in-vitro "
      + "diagnostic of class C (IVDR Rule 3, human genetic testing), so Art. 6(1) and Annex I apply, and "
      + `Art. 13 with them from ${longDate(ANNEX_I_DATE)}`,
  },
  finance_aml: {
    highRisk: "no",
    basis: "Annex III 5(b): creditworthiness and credit scoring of natural persons, fraud detection excepted",
    from: null,
    text: "not high-risk on the Act's own terms: Annex III 5(b) covers the creditworthiness and credit "
      + "scoring of natural persons, and anti-money-laundering monitoring evaluates transactions, not "
      + "creditworthiness, so Art. 13 does not bind it",
  },
  telecom_ids: {
    highRisk: "no",
    basis: "Annex III 2 (safety components of critical digital infrastructure) and recital 55",
    from: null,
    text: "not high-risk on the Act's own terms: an intrusion detector used solely for cybersecurity is "
      + "not a safety component of critical digital infrastructure (recital 55), so Annex III 2 does not "
      + "reach it and Art. 13 does not bind it",
  },
  satellite_comms: {
    highRisk: "depends",
    basis: "no Annex I or Annex III category names telemetry anomaly detection; Annex III 2 only as a safety component",
    from: ANNEX_III_DATE,
    text: "not a category the Act lists: it would be high-risk only as a safety component of critical "
      + "digital infrastructure (Annex III 2), which depends on how it is deployed, and then Art. 13 would "
      + `apply from ${longDate(ANNEX_III_DATE)}`,
  },
};
