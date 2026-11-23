// The CBOM panel's checks as a run the reader can start, watch and replay (Round 86, the owner's request of 2026-11-23
// that the graph be visualised and verified dynamically, as the E2E simulator's is). Pure functions, so
// tests/test_r86_cbom_run.py runs them in node: the events a run emits, what the graph shows at any point of the replay,
// the standings year by year for the year slider, and the broken copies the schema check must refuse.
import type { CbomDoc } from "./cbomReport";
import { hybridParts } from "./cbomReport";
import type { SchemaCheck } from "./cbomSchemaCheck";
import { ruleFor, standingOn, type Standing } from "./ir8547";

export type CbomEventKind = "fetch" | "digest" | "schema" | "read" | "parts" | "summary";

export interface CbomEvent {
  kind: CbomEventKind;
  /** The asset the event is about, for "read" and "parts". */
  ref?: string;
  /** The assets an event draws edges to ("parts"). */
  to?: string[];
  /** true passed, false failed, null neither (a reading, not a check). */
  ok: boolean | null;
  text: string;
}

export interface RunInput {
  doc: CbomDoc;
  bytes: number;
  sha256: string;
  builtSha256: string;
  schema: SchemaCheck;
  isoDate: string;
}

/** Every event of one run of the checks, in the order the panel plays them. */
export function cbomRunEvents(x: RunInput): CbomEvent[] {
  const ev: CbomEvent[] = [];
  ev.push({ kind: "fetch", ok: true, text: `Fetched qsh.cdx.json: ${x.bytes.toLocaleString("en-US")} bytes, ${x.doc.bomFormat} ${x.doc.specVersion}, ${x.doc.components.length} assets` });
  const same = x.sha256 === x.builtSha256;
  ev.push({ kind: "digest", ok: same, text: same ? `SHA-256 ${x.sha256.slice(0, 16)}…, the file this build was made from`
    : `SHA-256 ${x.sha256.slice(0, 16)}…, NOT the file this build was made from (${x.builtSha256.slice(0, 16)}…)` });
  ev.push({ kind: "schema", ok: x.schema.valid, text: x.schema.valid ? "Valid against the vendored CycloneDX 1.7 schema"
    : `Not valid against the CycloneDX 1.7 schema: ${x.schema.error ?? "no detail"}` });
  for (const c of x.doc.components) {
    const r = ruleFor(c["bom-ref"]);
    const s = r ? standingOn(r, x.isoDate) : null;
    ev.push({ kind: "read", ref: c["bom-ref"], ok: r ? null : false,
      text: r ? `${c.name}: ${r.basis === "none" ? "outside IR 8547's tables" : r.basis}; on ${x.isoDate}, ${s!.label}` : `${c.name}: no rule for this asset` });
  }
  for (const c of x.doc.components) {
    if (ruleFor(c["bom-ref"])?.cls !== "hybrid") continue;
    const { pq, classical, other } = hybridParts(x.doc, c["bom-ref"]);
    const to = [...pq, ...classical, ...other].map((p) => p["bom-ref"]);
    ev.push({ kind: "parts", ref: c["bom-ref"], to, ok: pq.length > 0 && classical.length > 0 ? true : false,
      text: `${c.name} depends on ${[...pq, ...classical, ...other].map((p) => p.name).join(", ")}: its post-quantum part ${pq.map((p) => p.name).join(" or ") || "missing"}, its classical part ${classical.map((p) => p.name).join(", ") || "missing"}` });
  }
  const failed = ev.filter((e) => e.ok === false).length;
  ev.push({ kind: "summary", ok: failed === 0, text: failed === 0 ? `Done: every check passed, ${x.doc.components.length} assets read on ${x.isoDate}`
    : `Done: ${failed} check${failed === 1 ? "" : "s"} failed` });
  return ev;
}

export interface GraphState {
  /** Assets read so far, with their standing on the run's date. */
  read: Map<string, Standing | "none">;
  /** Edges drawn so far, as "from>to". */
  edges: Set<string>;
  /** The asset the latest event was about, if any. */
  current: string | null;
  /** Events shown so far. */
  shown: number;
  /** The date the run read the assets on. */
  date: string;
}

/** What the graph shows after the first `k` events: the replay's position. */
export function graphStateAt(events: CbomEvent[], k: number, isoDate: string): GraphState {
  const n = Math.max(0, Math.min(k, events.length));
  const read = new Map<string, Standing | "none">();
  const edges = new Set<string>();
  let current: string | null = null;
  for (const e of events.slice(0, n)) {
    if (e.kind === "read" && e.ref) {
      const r = ruleFor(e.ref);
      read.set(e.ref, r ? standingOn(r, isoDate).standing : "none");
    }
    if (e.kind === "parts" && e.ref) for (const t of e.to ?? []) edges.add(`${e.ref}>${t}`);
    current = e.ref ?? null;
  }
  return { read, edges, current, shown: n, date: isoDate };
}

/** Each year's standing for every asset, for the year slider (IR 8547's dates are "after <year>"). */
export function standingsByYear(refs: readonly string[], fromYear: number, toYear: number): { year: number; standings: Record<string, Standing | "none"> }[] {
  const out = [];
  for (let year = fromYear; year <= toYear; year += 1) {
    const standings: Record<string, Standing | "none"> = {};
    for (const ref of refs) {
      const r = ruleFor(ref);
      standings[ref] = r ? standingOn(r, `${year}-07-01`).standing : "none";
    }
    out.push({ year, standings });
  }
  return out;
}

export interface BrokenCopy {
  id: string;
  label: string;
  /** The asset the damage is in, so the graph can mark it; null for a document-level change. */
  ref: string | null;
  apply: (doc: CbomDoc) => CbomDoc;
}

type Mutable = Record<string, any>;
const clone = (d: CbomDoc): Mutable => JSON.parse(JSON.stringify(d));
/** The asset's algorithm properties, found by bom-ref; a copy that cannot find its asset fails loudly. */
const alg = (d: Mutable, ref: string) => {
  const c = d.components.find((x: Mutable) => x["bom-ref"] === ref);
  if (!c) throw new Error(`no asset ${ref} in the document`);
  return c.cryptoProperties.algorithmProperties;
};

/** Damaged copies of the document the reader can check in the tab; each must be refused, and is in the test. */
export const BROKEN_COPIES: BrokenCopy[] = [
  { id: "primitive", label: "ML-KEM-768's primitive set to one the schema does not list", ref: "crypto/ml-kem-768",
    apply: (d) => { const x = clone(d); alg(x, "crypto/ml-kem-768").primitive = "nonsense"; return x as CbomDoc; } },
  { id: "certification", label: "X25519 claiming a certification level the schema does not list", ref: "crypto/x25519",
    apply: (d) => { const x = clone(d); alg(x, "crypto/x25519").certificationLevel = ["fips140-9-l1"]; return x as CbomDoc; } },
  { id: "quantum-level", label: "AES-256-GCM given NIST quantum level 7 (the schema allows 0 to 6)", ref: "crypto/aes-256-gcm",
    apply: (d) => { const x = clone(d); alg(x, "crypto/aes-256-gcm").nistQuantumSecurityLevel = 7; return x as CbomDoc; } },
  { id: "duplicate-dependency", label: "The combiner listing ML-KEM-768 twice among its dependencies", ref: "crypto/hybrid-combiner",
    apply: (d) => { const x = clone(d); const dep = x.dependencies.find((e: Mutable) => e.ref === "crypto/hybrid-combiner"); dep.dependsOn = [dep.dependsOn[0], dep.dependsOn[0]]; return x as CbomDoc; } },
  { id: "no-format", label: "The document without its bomFormat", ref: null,
    apply: (d) => { const x = clone(d); delete x.bomFormat; return x as CbomDoc; } },
];
