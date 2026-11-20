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

/** Between neighbours on a fan, and the widest a fan opens (Round 86). */
const FAN_GAP = 26, FAN_SPAN = (5 * Math.PI) / 6;

/** The scored transaction's two accounts in the centre, the payer's other counterparties on a fan round the payer to the
 *  left and the payee's on one round the payee to the right, and accounts that touch both above and below.
 *
 *  A fan, an arc centred on its account (Round 86, 2026-11-18): every transfer drawn touches the payer or the payee, so
 *  every line from a fan is a ray from its centre and passes no other account on it. Two staggered columns put an inner
 *  account 1.6 px from the line to an outer one, and the IBM tab's drawing ran two lines through other accounts'
 *  letters; one column instead leaves a steep line to its last account grazing the next one. An account paying itself
 *  touches both ends with all its counterparties, which then stood in one column through it, so they go on a ring round
 *  it. Neighbours on a fan stay at least 22 px apart and every line at least 13 px from the accounts it does not join
 *  (`tests/test_r86_model_explanations.py`). */
export function layoutScored(g: Pick<ScoredRow, "nodes" | "from" | "to"> & { edges: { from: string; to: string }[] }, w: number, base: number): { layout: Layout; h: number } {
  const margin = 22, touchesFrom = new Set<string>(), touchesTo = new Set<string>();
  for (const e of g.edges) {
    if (e.from === g.from || e.to === g.from) { touchesFrom.add(e.from); touchesFrom.add(e.to); }
    if (e.from === g.to || e.to === g.to) { touchesTo.add(e.from); touchesTo.add(e.to); }
  }
  const others = g.nodes.filter((n) => n !== g.from && n !== g.to);
  if (g.from === g.to) {
    const h = Math.max(base, 2 * margin + Math.ceil((FAN_GAP * others.length) / Math.PI));
    return { layout: { ...ring(others, w, h, margin), [g.from]: { x: w / 2, y: h / 2 } }, h };
  }
  const both = others.filter((n) => touchesFrom.has(n) && touchesTo.has(n));
  const leftOnly = others.filter((n) => touchesFrom.has(n) && !touchesTo.has(n));
  const rightOnly = others.filter((n) => !touchesFrom.has(n));
  const ax = w * 0.38, bx = w * 0.62;
  // The old columns' distance where it gives room, further out (to the box's margin) where a fan needs more.
  const fan = (n: number) => {
    const r = Math.min(ax - margin, Math.max(ax - w * 0.14, (FAN_GAP * (n - 1)) / FAN_SPAN));
    return { r, span: n > 1 ? Math.min(FAN_SPAN, (FAN_GAP * (n - 1)) / r) : 0 };
  };
  const fl = fan(leftOnly.length), fr = fan(rightOnly.length);
  const reach = Math.max(fl.r * Math.sin(fl.span / 2), fr.r * Math.sin(fr.span / 2));
  const h = Math.max(base, Math.ceil(2 * (margin + reach)), 2 * margin + 60 + (both.length - 1) * SPACING);
  const out: Layout = { [g.from]: { x: ax, y: h / 2 }, [g.to]: { x: bx, y: h / 2 } };
  const place = (names: string[], cx: number, dir: number, f: { r: number; span: number }) => names.forEach((n, i) => {
    const phi = names.length === 1 ? 0 : -f.span / 2 + (f.span * i) / (names.length - 1);
    out[n] = { x: cx + dir * f.r * Math.cos(phi), y: h / 2 + f.r * Math.sin(phi) };
  });
  place(leftOnly, ax, -1, fl);
  place(rightOnly, bx, 1, fr);
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
