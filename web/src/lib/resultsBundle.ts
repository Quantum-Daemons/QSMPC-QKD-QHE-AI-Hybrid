/**
 * One run's results as one JSON file (Round 86, the owner's choice of 2026-11-13): the page, the bundle that ran, the
 * run's parameters, every metric, the run log, each card's rows (cardRows.ts, the same rows the cards' own buttons
 * download), every published model's re-scoring, and the signed record with its whole signature, so a reader can
 * verify it (`tests/test_r86_results_bundle.py` does, from the bundle alone).
 *
 * WHAT IS LEFT OUT, ON PURPOSE. A bundle is built from an allowlist, not by deleting from the run's artifacts, so a
 * new artifact stays out until someone decides it belongs: no key, no mask seed, no session key (the run's internal
 * config carries one, and the page's parameters never do, but every parameter is filtered to plain values anyway), no
 * party's own contribution (the genomics card keeps each cohort's noisy count for its threshold-BGV button; the secure
 * sum exists to keep it), and not the PSI's member list (the banks' answer; its size is in the metrics).
 */
import { financeDpRows, fedHeartDpRows, fedHeartMatrixRows, genomicsDpRows, verificationRows } from "./cardRows";

export const RESULTS_BUNDLE_SCHEMA = "qsh-results-bundle/1";

export interface ResultsBundleInput {
  usecase: string;
  params: Record<string, unknown>;
  metrics: Record<string, unknown>;
  artifacts: Record<string, any>;
  events: unknown[];
  /** The asset the page loaded, e.g. "assets/index-b1ejB7U1.js": which build produced these numbers. */
  bundleAsset: string | null;
  generatedAt?: string;
  /** The published-figure comparison's rows, the same rows the page draws in its card for this run (Round 86, batch 12). */
  publishedComparison?: Record<string, unknown>[];
}

/**
 * Every card with its own download, by its export name without the page's prefix, and where this bundle carries it: the
 * key under `cards`, or why it is left out (Round 86, batch 12: the button promised "every card's rows" while it carried
 * four kinds). `pages` names the pages a page-specific card is on; without it the card is on every page.
 * `tests/test_r86_results_bundle_names_every_card.py` holds every `<CardExport name=...>` in the source to this list.
 */
export const CARD_EXPORT_CLASSES: Record<string, { card: string; pages?: string[] } | { leftOut: string; label: string; pages?: string[] }> = {
  "four-hospitals": { card: "four_hospitals", pages: ["medical_fl"] },
  "differential-privacy": { card: "differential_privacy", pages: ["finance_aml", "medical_fl", "genomics"] },
  "verifications": { card: "verifications" },
  "published-comparison": { card: "published_comparison" },
  "encrypted-aggregation": { leftOut: "run on request by its own button, with its own download", label: "the encrypted-aggregation comparison", pages: ["medical_fl"] },
  "webgpu-beside-wasm": { leftOut: "run on request by its own button, with its own download", label: "the WebGPU and WASM comparison" },
  "murmur-dp": { leftOut: "a committed record's figures, the same on every run, which its own download carries", label: "the murmur DP figures", pages: ["medical_fl"] },
  "heart-sound-federation": { leftOut: "a committed record's figures, the same on every run, which its own download carries", label: "the heart-sound federation table", pages: ["medical_fl"] },
  "cxr-federation": { leftOut: "a committed record's figures, the same on every run, which its own download carries", label: "the chest X-ray federation table", pages: ["medical_fl"] },
  "vus-pr": { leftOut: "a committed record's figures, the same on every run, which its own download carries", label: "the VUS-PR figures", pages: ["satellite_comms"] },
  "boundary": { leftOut: "a committed record's figures, the same on every run, which its own download carries", label: "the boundary slice", pages: ["satellite_comms"] },
  "busbra-external": { leftOut: "a committed record's figures, the same on every run, which its own download carries", label: "the BUS-BRA figures", pages: ["medical_fl"] },
  "satqkd-check": { leftOut: "a committed record's figures, the same on every run, which its own download carries", label: "the SatQKD figures", pages: ["satellite_comms"] },
  "pfcp-held-out-day": { leftOut: "a committed record's figures, the same on every run, which its own download carries", label: "the PFCP figures", pages: ["telecom_ids"] },
};

/** The cards on this page that the bundle leaves out, each with its reason. */
export function cardsLeftOut(usecase: string): string[] {
  return Object.values(CARD_EXPORT_CLASSES)
    .filter((c): c is { leftOut: string; label: string; pages?: string[] } => "leftOut" in c && (!c.pages || c.pages.includes(usecase)))
    .map((c) => `${c.label}: ${c.leftOut}`);
}

const plain = (v: unknown) => v === null || ["string", "number", "boolean"].includes(typeof v);

/** The page's parameters, as plain values only, and never a key beginning with an underscore. */
export function bundleParams(params: Record<string, unknown>): Record<string, string | number | boolean | null> {
  return Object.fromEntries(Object.entries(params).filter(([k, v]) => !k.startsWith("_") && plain(v))) as
    Record<string, string | number | boolean | null>;
}

const pick = (o: Record<string, any> | undefined, keys: string[]) =>
  o ? Object.fromEntries(keys.filter((k) => o[k] !== undefined).map((k) => [k, o[k]])) : undefined;

export function buildResultsBundle(i: ResultsBundleInput) {
  const a = i.artifacts ?? {};
  const cards: Record<string, unknown> = {};
  if (a.fedHeart) cards.four_hospitals = { rows: fedHeartMatrixRows(a.fedHeart) };
  if (a.fedHeartDp) {
    cards.differential_privacy = { rows: fedHeartDpRows(a.fedHeartDp), epsilon_all_rows_together: a.fedHeartDp.epsilonAllRows,
                                   delta: a.fedHeartDp.delta };
  }
  if (a.genomicsDp) cards.differential_privacy = { rows: genomicsDpRows(a.genomicsDp) };
  if (a.financeDp) cards.differential_privacy = { rows: financeDpRows(a.financeDp) };
  if (a.multi_party_psi?.result) {
    cards.multi_party_psi = pick(a.multi_party_psi.result, ["parties", "setSize", "intersection", "matchesClear", "andGates",
                                                            "bytes", "mpcMs", "ranIn"]);
  }
  if (Array.isArray(a.verifications)) cards.verifications = { rows: verificationRows(a.verifications) };
  if (Array.isArray(i.publishedComparison) && i.publishedComparison.length) cards.published_comparison = { rows: i.publishedComparison };
  const s = a.result_signature;
  return {
    schema: RESULTS_BUNDLE_SCHEMA,
    page: i.usecase,
    generated_at: i.generatedAt ?? new Date().toISOString(),
    build: i.bundleAsset,
    params: bundleParams(i.params),
    metrics: i.metrics,
    cards,
    events: i.events,
    signature: s ? {
      algorithm: s.algorithm, public_key_hex: s.public_key_hex, signature_hex: s.signature_hex ?? null,
      signed_fields: s.signed_fields, signed_payload: s.signed_payload, verified_in_the_tab: s.verified,
      how_to_check: "verify signature_hex over the UTF-8 bytes of JSON.stringify(signed_payload) with public_key_hex; "
        + "the key pair was made for this run and discarded, so a match shows the record is unchanged since signing, "
        + "not who signed it",
    } : null,
    left_out: [
      "keys and mask seeds of every kind, and the run's session key",
      "each party's own contribution and noise (the genomics cohorts' noisy counts among them)",
      "the PSI's member list, which is the banks' answer; its size is in the metrics",
      "cards run on request (the threshold sums among them), which have their own downloads",
      ...cardsLeftOut(i.usecase),
    ],
  };
}

/** The asset the page was built into, read from its own script tags. */
export function loadedBundleAsset(): string | null {
  if (typeof document === "undefined") return null;
  const src = [...document.querySelectorAll("script[src]")].map((s) => (s as HTMLScriptElement).src)
    .find((u) => /\/assets\/index-[\w-]+\.js$/.test(u));
  return src ? src.slice(src.indexOf("assets/")) : null;
}
