// The PQC-readiness report the CBOM panel shows and downloads (Round 86, the owner's choice of 2026-11-23): each
// asset of the CBOM with NIST IR 8547's reading of it and its standing on a date, as rows, Markdown and CSV. Pure
// functions over the parsed document, so tests/test_r86_pqc_readiness.py runs them in node on the published file.
//
// The hybrid combiner gets two more rows, for its parts. They are read from the document's own dependency graph,
// not typed here: the post-quantum parts are the combiner's dependencies IR 8547 rates as post-quantum, the classical
// ones those it rates as quantum-vulnerable, and the KDF that combines them is neither.
import { IR8547, ruleFor, standingOn, type Ir8547Rule, type Standing } from "./ir8547";

export interface CbomComponent {
  "bom-ref": string;
  name: string;
  description: string;
  properties?: { name: string; value: string }[];
  cryptoProperties?: { algorithmProperties?: { primitive?: string; nistQuantumSecurityLevel?: number; certificationLevel?: string[] } };
}

export interface CbomDoc {
  bomFormat: string;
  specVersion: string;
  serialNumber: string;
  version: number;
  metadata?: { component?: { "bom-ref"?: string; name?: string; version?: string } };
  components: CbomComponent[];
  dependencies?: { ref: string; dependsOn?: string[] }[];
}

export interface ReportRow {
  ref: string;
  name: string;
  primitive: string;
  standard: string;
  usedFor: string;
  scope: string;
  cls: string;
  basis: string;
  reading: string;
  standing: Standing | "none";
  standingLabel: string;
  /** Set on the hybrid's part rows: the combiner they belong to. */
  partOf?: string;
}

export const STANDING_SUMMARY: Record<Standing, string> = {
  "post-quantum": "Post-quantum (Tables 3 and 5)",
  symmetric: "Symmetric, rated by Table 6 or 8 or by section 4.1.3",
  "approved-quantum-vulnerable": "Quantum-vulnerable, listed as approved until it is deprecated after 2030",
  "quantum-vulnerable": "Quantum-vulnerable, not named, under section 4.1.2's dates",
  deprecated: "Deprecated",
  disallowed: "Disallowed",
  "allowed-composite": "Hybrid, allowed as a composite (section 3.2.1)",
  "not-classified": "Outside IR 8547's tables",
};

const prop = (c: CbomComponent, name: string) => c.properties?.find((p) => p.name === name)?.value ?? "";

function baseRow(c: CbomComponent, r: Ir8547Rule | undefined, isoDate: string): ReportRow {
  const s = r ? standingOn(r, isoDate) : null;
  return {
    ref: c["bom-ref"], name: c.name, primitive: c.cryptoProperties?.algorithmProperties?.primitive ?? "",
    standard: prop(c, "qsh:standard"), usedFor: c.description, scope: prop(c, "qsh:scope"),
    cls: r?.cls ?? "none", basis: r?.basis ?? "none", reading: r?.reading ?? "This report has no rule for this asset.",
    standing: s?.standing ?? "none", standingLabel: s?.label ?? "No rule",
  };
}

/** The combiner's parts, by IR 8547's class of each dependency the document gives it. */
export function hybridParts(doc: CbomDoc, combinerRef: string): { pq: CbomComponent[]; classical: CbomComponent[]; other: CbomComponent[] } {
  const deps = doc.dependencies?.find((d) => d.ref === combinerRef)?.dependsOn ?? [];
  const byRef = new Map(doc.components.map((c) => [c["bom-ref"], c]));
  const out = { pq: [] as CbomComponent[], classical: [] as CbomComponent[], other: [] as CbomComponent[] };
  for (const ref of deps) {
    const c = byRef.get(ref);
    if (!c) continue;
    const cls = ruleFor(ref)?.cls;
    (cls === "post-quantum" ? out.pq : cls === "quantum-vulnerable" ? out.classical : out.other).push(c);
  }
  return out;
}

function partRows(doc: CbomDoc, combiner: CbomComponent, isoDate: string): ReportRow[] {
  const { pq, classical, other } = hybridParts(doc, combiner["bom-ref"]);
  const rows: ReportRow[] = [];
  const names = (cs: CbomComponent[]) => cs.map((c) => c.name).join(" or ");
  const kdf = other.map((c) => c.name).join(", ");
  if (pq.length) {
    const rules = pq.map((c) => ruleFor(c["bom-ref"])!);
    const cats = pq.map((c, i) => `${rules[i].category} with ${c.name}`).join(", ");
    const s = standingOn(rules[0], isoDate);
    rows.push({
      ref: `${combiner["bom-ref"]}#post-quantum-part`, name: `${names(pq)}, its post-quantum part`,
      primitive: pq[0].cryptoProperties?.algorithmProperties?.primitive ?? "", standard: prop(pq[0], "qsh:standard"),
      usedFor: combiner.description, scope: prop(combiner, "qsh:scope"), cls: "post-quantum", basis: rules[0].basis,
      reading: `Security category ${cats}. Against a quantum attacker the combination rests on this part alone.`,
      standing: s.standing, standingLabel: s.label, partOf: combiner["bom-ref"],
    });
  }
  for (const c of classical) {
    const r = ruleFor(c["bom-ref"])!;
    const s = standingOn(r, isoDate);
    rows.push({
      ref: `${combiner["bom-ref"]}#classical-part`, name: `${c.name}, its classical part`, primitive: c.cryptoProperties?.algorithmProperties?.primitive ?? "",
      standard: prop(c, "qsh:standard"), usedFor: combiner.description, scope: prop(combiner, "qsh:scope"), cls: r.cls, basis: r.basis,
      reading: `Quantum-vulnerable: deprecated after ${r.deprecatedAfter}, disallowed after ${r.disallowedAfter}. Section 3.2.1 allows the composite even when this part is disallowed, while the other input comes from an approved scheme.${kdf ? ` The two are combined by ${kdf}.` : ""}`,
      standing: s.standing, standingLabel: s.label, partOf: combiner["bom-ref"],
    });
  }
  return rows;
}

/** One row per asset, in the document's order, each hybrid followed by its parts. */
export function reportRows(doc: CbomDoc, isoDate: string): ReportRow[] {
  const rows: ReportRow[] = [];
  for (const c of doc.components) {
    const r = ruleFor(c["bom-ref"]);
    rows.push(baseRow(c, r, isoDate));
    if (r?.cls === "hybrid") rows.push(...partRows(doc, c, isoDate));
  }
  return rows;
}

/** How many of the document's assets have each standing on the date (the part rows are not counted again). */
export function reportCounts(rows: ReportRow[]): { standing: Standing | "none"; label: string; count: number }[] {
  const counts = new Map<Standing | "none", number>();
  for (const r of rows) if (!r.partOf) counts.set(r.standing, (counts.get(r.standing) ?? 0) + 1);
  const order = [...(Object.keys(STANDING_SUMMARY) as Standing[]), "none" as const];
  return order.filter((s) => counts.has(s)).map((s) => ({ standing: s, label: s === "none" ? "No rule" : STANDING_SUMMARY[s], count: counts.get(s)! }));
}

export interface Provenance {
  file: string;
  sha256: string;
  /** The digest this build recorded for the file, when known. */
  builtSha256?: string;
}

const cell = (s: string) => s.replace(/\|/g, "\\|").replace(/\s*\n\s*/g, " ");

export function reportMarkdown(doc: CbomDoc, isoDate: string, p: Provenance): string {
  const rows = reportRows(doc, isoDate);
  const app = doc.metadata?.component;
  const lines = [
    `# PQC readiness of ${app?.name ?? "the application"}'s cryptography, read against NIST IR 8547`,
    "",
    `- Document: ${p.file}, ${doc.bomFormat} ${doc.specVersion}, serial ${doc.serialNumber}, version ${doc.version}, describing ${app?.name ?? "?"} ${app?.version ?? ""}`.trimEnd(),
    `- SHA-256 of the file read: ${p.sha256}${p.builtSha256 ? (p.builtSha256 === p.sha256 ? " (the file this build was made from)" : ` (NOT the file this build was made from, ${p.builtSha256})`) : ""}`,
    `- Read against: ${IR8547.title}, ${IR8547.url}`,
    `- Standing on: ${isoDate}`,
    "- A classification by IR 8547's tables, written by the Validations page in the browser. It is not a certification, and nothing here is FIPS 140-3 validated.",
    "",
    `## Summary on ${isoDate}`,
    "",
    "| Standing | Assets |",
    "|---|---|",
    ...reportCounts(rows).map((c) => `| ${cell(c.label)} | ${c.count} |`),
    "",
    "## Assets",
    "",
    `| Algorithm | Kind | NIST IR 8547 | Reading | On ${isoDate} | Standard | Used for | Where |`,
    "|---|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${cell(r.partOf ? `- ${r.name}` : r.name)} | ${cell(r.primitive)} | ${cell(r.basis)} | ${cell(r.reading)} | ${cell(r.standingLabel)} | ${cell(r.standard)} | ${cell(r.usedFor)} | ${cell(r.scope)} |`),
    "",
  ];
  return lines.join("\n");
}

/** The CSV's rows: one per report row, the date in its own column so the file says when it was read. */
export function reportCsvRows(doc: CbomDoc, isoDate: string): Record<string, string>[] {
  return reportRows(doc, isoDate).map((r) => ({
    bom_ref: r.ref, part_of: r.partOf ?? "", name: r.name, kind: r.primitive, ir8547_basis: r.basis, ir8547_class: r.cls,
    reading: r.reading, standing_on: isoDate, standing: r.standing, standing_label: r.standingLabel, standard: r.standard,
    used_for: r.usedFor, scope: r.scope,
  }));
}

/** Hex SHA-256 of the bytes, by Web Crypto (node has it too). */
export async function sha256Hex(bytes: ArrayBuffer | Uint8Array): Promise<string> {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", view as BufferSource);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
