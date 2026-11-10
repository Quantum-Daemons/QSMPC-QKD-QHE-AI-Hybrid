// The layouts of the finance page's transaction graphs (Round 86), as pure functions so that a test can hold them to the box.
//
// The data is `public/data/finance/transaction_graphs.json`: labelled laundering components and the neighbourhood of each fixture row of the published MLP (`scripts/data/build_transaction_graphs.py`).

export interface TxEdge { from: string; to: string; minute: number; amount: number; currency: string; format: string; laundering: number }
export interface TxGraph { nodes: string[]; edges: TxEdge[] }
export interface Exemplar extends TxGraph { shape: string }
export interface ScoredRow extends TxGraph { row_index_in_test_window: number; label: number; from: string; to: string; scored_edge: number; edges_in_window: number }
export interface TxGraphsFile {
  provenance: string; licence: string; licence_url: string; attribution: string; modifications: string;
  batch_window: number; window_minutes: number; exemplars: Exemplar[]; scored_rows: ScoredRow[];
}
export interface Point { x: number; y: number }
export type Layout = Record<string, Point>;

/** Row spacing that keeps two 11-px nodes in one column clear of each other. */
const SPACING = 24;

const column = (names: string[], x: number, h: number, margin: number): Layout => {
  const out: Layout = {};
  names.forEach((n, i) => { out[n] = { x, y: names.length === 1 ? h / 2 : margin + (i * (h - 2 * margin)) / (names.length - 1) }; });
  return out;
};

const ring = (order: string[], w: number, h: number, margin: number): Layout => {
  const rx = w / 2 - margin, ry = h / 2 - margin, out: Layout = {};
  order.forEach((n, i) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / order.length; out[n] = { x: w / 2 + rx * Math.cos(a), y: h / 2 + ry * Math.sin(a) }; });
  return out;
};

const degree = (g: TxGraph) => {
  const out: Record<string, number> = {}, into: Record<string, number> = {};
  for (const e of g.edges) { out[e.from] = (out[e.from] ?? 0) + 1; into[e.to] = (into[e.to] ?? 0) + 1; }
  return { out, into };
};

/** Follows the edges in time order from the first edge's sender: the order a cycle or a chain visits its accounts. */
const walk = (g: TxGraph): string[] => {
  const next: Record<string, string> = {};
  for (const e of g.edges) if (!(e.from in next)) next[e.from] = e.to;
  const seen: string[] = [];
  let at: string | undefined = g.edges[0]?.from;
  while (at !== undefined && !seen.includes(at)) { seen.push(at); at = next[at]; }
  return [...seen, ...g.nodes.filter((n) => !seen.includes(n))];
};

/** Where each account of a laundering pattern goes, by the pattern's shape, and how tall the figure must be to hold them. */
export function layoutExemplar(g: Exemplar, w: number, base: number): { layout: Layout; h: number } {
  const margin = 20, { out, into } = degree(g), left = w * 0.16, right = w * 0.84;
  const senders = g.nodes.filter((n) => (out[n] ?? 0) > 0 && !(into[n] ?? 0));
  const receivers = g.nodes.filter((n) => (into[n] ?? 0) > 0 && !(out[n] ?? 0));
  const middle = g.nodes.filter((n) => (out[n] ?? 0) > 0 && (into[n] ?? 0) > 0);
  const tall = Math.max(senders.length, receivers.length, g.nodes.filter((n) => !senders.includes(n)).length, g.nodes.filter((n) => !receivers.includes(n)).length);
  const h = g.shape === "cycle" || g.shape === "chain" ? base : Math.max(base, 2 * margin + (tall - 1) * SPACING);
  let layout: Layout;
  switch (g.shape) {
    case "fan-out": layout = { ...column(senders, left, h, margin), ...column(g.nodes.filter((n) => !senders.includes(n)), right, h, margin) }; break;
    case "fan-in": layout = { ...column(g.nodes.filter((n) => !receivers.includes(n)), left, h, margin), ...column(receivers, right, h, margin) }; break;
    case "gather-scatter": layout = { ...column(senders, left, h, margin), ...column(middle, w / 2, h, margin), ...column(receivers, right, h, margin) }; break;
    case "bipartite": layout = { ...column(senders, left, h, margin), ...column(receivers, right, h, margin) }; break;
    case "chain": {
      const order = walk(g);
      layout = {};
      order.forEach((n, i) => { layout[n] = { x: margin + (i * (w - 2 * margin)) / Math.max(order.length - 1, 1), y: h / 2 }; });
      break;
    }
    default: layout = ring(walk(g), w, h, margin + 4);
  }
  return { layout, h };
}

/** The scored transaction's two accounts in the centre, their counterparties in staggered columns to the left and the right, and accounts that touch both above and below. */
export function layoutScored(g: ScoredRow, w: number, base: number): { layout: Layout; h: number } {
  const margin = 22, touchesFrom = new Set<string>(), touchesTo = new Set<string>();
  for (const e of g.edges) {
    if (e.from === g.from || e.to === g.from) { touchesFrom.add(e.from); touchesFrom.add(e.to); }
    if (e.from === g.to || e.to === g.to) { touchesTo.add(e.from); touchesTo.add(e.to); }
  }
  const others = g.nodes.filter((n) => n !== g.from && n !== g.to);
  const both = others.filter((n) => touchesFrom.has(n) && touchesTo.has(n));
  const leftOnly = others.filter((n) => touchesFrom.has(n) && !touchesTo.has(n));
  const rightOnly = others.filter((n) => !touchesFrom.has(n));
  const h = Math.max(base, 2 * margin + (Math.max(leftOnly.length, rightOnly.length) - 1) * SPACING, 2 * margin + 60 + (both.length - 1) * SPACING);
  const out: Layout = { [g.from]: { x: w * 0.38, y: h / 2 }, [g.to]: { x: w * 0.62, y: h / 2 } };
  if (g.from === g.to) out[g.from] = { x: w / 2, y: h / 2 };
  const staggered = (names: string[], x: number, dir: number) => names.forEach((n, i) => {
    out[n] = { x: x + dir * (i % 2) * 40, y: names.length === 1 ? h / 2 : margin + (i * (h - 2 * margin)) / (names.length - 1) };
  });
  staggered(leftOnly, w * 0.14, 1);
  staggered(rightOnly, w * 0.86, -1);
  both.forEach((n, i) => { out[n] = { x: w / 2, y: i % 2 === 0 ? margin + (i / 2) * SPACING : h - margin - ((i - 1) / 2) * SPACING }; });
  return { layout: out, h };
}

/** Parallel edges drawn as one line: the first of each direction, with how many there are. */
export function groupEdges(g: TxGraph): { from: string; to: string; edges: TxEdge[]; order: number; laundering: boolean }[] {
  const by = new Map<string, { from: string; to: string; edges: TxEdge[]; order: number; laundering: boolean }>();
  g.edges.forEach((e, i) => {
    const key = `${e.from}>${e.to}`;
    const hit = by.get(key);
    if (hit) { hit.edges.push(e); hit.laundering ||= e.laundering === 1; } else by.set(key, { from: e.from, to: e.to, edges: [e], order: i + 1, laundering: e.laundering === 1 });
  });
  return [...by.values()];
}
