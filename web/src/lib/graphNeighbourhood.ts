// What IBM's Multi-GNN read for each transfer it scored (Round 86): the counts that
// scripts/train/ibm_multignn/neighbourhood.py writes into a graph sample, re-derived here from the sample's own arrays,
// and the drawing of a scored transfer's two accounts. The definition is that file's docstring; check_graph_neighbourhood.mjs
// holds this to the same hand-counted graph as its Python test.

export interface SeedField {
  payer_out: number; payer_in: number; payee_out: number; payee_in: number;
  read: number; later_in_batch: number; beyond_batch: number; furthest_later_s: number;
}
export interface GraphView { batch_rows: number; edge_position: number[]; edge_seconds: number[]; per_seed: SeedField[] }

/** When a drawn transfer happened, against the scored one: the colour it is drawn in. */
export type When = "scored" | "earlier" | "in-batch" | "beyond";
export interface NeighbourEdge { from: string; to: string; minute: number; when: When }
export interface NeighbourGraph { nodes: string[]; edges: NeighbourEdge[]; from: string; to: string; shown: number; at_accounts: number }

const NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/** The counts for each seed. `edgeIndex` is the sample's forward edges as it stores them, the sources then the targets. */
export function seedFields(edgeIndex: ArrayLike<number>, nEdges: number, view: GraphView, seedAt: number[]): SeedField[] {
  const pos = view.edge_position, sec = view.edge_seconds, rows = view.batch_rows;
  if (pos.length !== nEdges || sec.length !== nEdges || edgeIndex.length !== 2 * nEdges) throw new Error("every edge needs a position and a time");
  const src = (e: number) => edgeIndex[e], dst = (e: number) => edgeIndex[nEdges + e];
  return seedAt.map((s) => {
    const u = src(s), v = dst(s), p = pos[s], t = sec[s], b = Math.floor(p / rows);
    const hop1 = new Set<number>();
    for (let e = 0; e < nEdges; e++) {
      if (src(e) === u || src(e) === v || dst(e) === u || dst(e) === v) { hop1.add(src(e)); hop1.add(dst(e)); }
    }
    const payerOut = new Set<number>(), payerIn = new Set<number>(), payeeOut = new Set<number>(), payeeIn = new Set<number>();
    const read = new Set<number>(), later = new Set<number>(), beyond = new Set<number>();
    let furthest = 0;
    for (let e = 0; e < nEdges; e++) {
      const q = pos[e];
      if (q === p) continue;
      if (src(e) === u) payerOut.add(q);
      if (dst(e) === u) payerIn.add(q);
      if (src(e) === v) payeeOut.add(q);
      if (dst(e) === v) payeeIn.add(q);
      if (!hop1.has(src(e)) && !hop1.has(dst(e))) continue;
      read.add(q);
      const batch = Math.floor(q / rows);
      if (sec[e] > t) { furthest = Math.max(furthest, sec[e] - t); if (batch === b) later.add(q); }
      if (batch > b) beyond.add(q);
    }
    return { payer_out: payerOut.size, payer_in: payerIn.size, payee_out: payeeOut.size, payee_in: payeeIn.size,
             read: read.size, later_in_batch: later.size, beyond_batch: beyond.size, furthest_later_s: furthest };
  });
}

/** The first seed whose counts differ from the file's, as a sentence, or null when all match. */
export function fieldsMismatch(derived: SeedField[], recorded: SeedField[]): string | null {
  if (derived.length !== recorded.length) return `the file records ${recorded.length} seeds and the sample has ${derived.length}`;
  for (let k = 0; k < derived.length; k++) {
    for (const key of Object.keys(recorded[k]) as (keyof SeedField)[]) {
      const a = derived[k][key], b = recorded[k][key];
      if (key === "furthest_later_s" ? Math.abs(a - b) > 1e-6 : a !== b) return `seed ${k + 1}: ${key} is ${a} here and ${b} in the file`;
    }
  }
  return null;
}

/** A scored transfer's two accounts and their transfers in the sample, at most `cap` besides the scored one, each
 *  marked by when it happened. Over the cap, each account keeps the half nearest the scored transfer in time (the
 *  payer's, then the payee's), and any room one leaves goes to the other's next nearest; a hub's transfers would
 *  otherwise crowd out its counterpart's. Account A pays account B; the others are lettered in time order. */
export function seedGraph(edgeIndex: ArrayLike<number>, nEdges: number, view: GraphView, seed: number, cap = 24): NeighbourGraph {
  const pos = view.edge_position, sec = view.edge_seconds, rows = view.batch_rows;
  const src = (e: number) => edgeIndex[e], dst = (e: number) => edgeIndex[nEdges + e];
  const u = src(seed), v = dst(seed), p = pos[seed], t = sec[seed], b = Math.floor(p / rows);
  const seen = new Set<number>([p]), atPayer: number[] = [], atPayee: number[] = [];
  for (let e = 0; e < nEdges; e++) {
    if (seen.has(pos[e])) continue;
    if (src(e) === u || dst(e) === u) { seen.add(pos[e]); atPayer.push(e); }
    else if (src(e) === v || dst(e) === v) { seen.add(pos[e]); atPayee.push(e); }
  }
  const nearest = (es: number[]) => [...es].sort((x, y) => Math.abs(sec[x] - t) - Math.abs(sec[y] - t) || pos[x] - pos[y]);
  const [np, nq] = [nearest(atPayer), nearest(atPayee)];
  const half = Math.ceil(cap / 2), fromPayer = Math.min(np.length, Math.max(half, cap - nq.length));
  const kept = [...np.slice(0, fromPayer), ...nq.slice(0, cap - fromPayer)].sort((x, y) => sec[x] - sec[y] || pos[x] - pos[y]);
  const at = atPayer.length + atPayee.length;
  const name = new Map<number, string>([[u, "A"]]);
  if (!name.has(v)) name.set(v, "B");
  for (const e of kept) for (const n of [src(e), dst(e)]) if (!name.has(n)) name.set(n, NAMES[name.size] ?? `${name.size + 1}`);
  const when = (e: number): When => (Math.floor(pos[e] / rows) > b ? "beyond" : sec[e] > t ? "in-batch" : "earlier");
  const edges: NeighbourEdge[] = [{ from: name.get(u)!, to: name.get(v)!, minute: 0, when: "scored" },
    ...kept.map((e) => ({ from: name.get(src(e))!, to: name.get(dst(e))!, minute: (sec[e] - t) / 60, when: when(e) }))];
  return { nodes: [...name.values()], edges, from: name.get(u)!, to: name.get(v)!, shown: kept.length, at_accounts: at };
}
