// What IBM's Multi-GNN read for each transfer it scored (Round 86, the owner's choices of 2026-11-15). From the graph
// sample the IBM tab scores, the scored transfer's two accounts are drawn, and the transfers the model read are
// counted, with how many lie past the transfer's horizon, the horizon of this page's own models (the end of its batch
// of 2,048 rows, and at most 17 minutes after it): the causal protocol reads none, IBM's loader samples them. The page
// derives the counts from the sample itself and reports any difference from those the export recorded
// (lib/graphNeighbourhood.ts).
import { useId, useMemo, useState } from "react";
import { fieldsMismatch, seedFields, seedGraph, type GraphView, type NeighbourEdge, type When } from "../lib/graphNeighbourhood";
import { layoutScored } from "../lib/transactionGraph";

export interface NeighbourhoodSample {
  protocol: string;
  inputs: { edge_index: number[]; n_edges: number };
  seeds: { positions_in_batch: number[]; labels: number[] };
  view?: GraphView;
}

const NODE = 11, W = 720;
const PAINT: Record<When, { colour: string; dash?: string; word: string }> = {
  scored: { colour: "var(--accent)", word: "the scored transfer" },
  earlier: { colour: "var(--muted)", word: "earlier, or in the same minute" },
  within: { colour: "var(--warn)", word: "later, within its horizon" },
  past: { colour: "var(--danger)", dash: "5 3", word: "past its horizon" },
};
const RANK: Record<When, number> = { earlier: 0, within: 1, past: 2, scored: 3 };
const halo = { paintOrder: "stroke", stroke: "var(--panel)", strokeWidth: 6 } as const;
const ROW_BUTTON = { background: "none", border: 0, padding: "2px 0", minHeight: 24, color: "var(--accent)", textDecoration: "underline",
                     cursor: "pointer", font: "inherit", textAlign: "left" } as const;

/** A count with thousands separators. */
const n = (x: number) => x.toLocaleString("en-US");
/** A span of seconds in the unit that reads best. */
const span = (s: number) => (s <= 0 ? "none" : s < 5400 ? `${Math.round(s / 60)} min` : s < 172800 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(1)} days`);
const minute = (m: number) => (m === 0 ? "the same minute" : `${Math.abs(m) < 90 ? `${Math.round(Math.abs(m))} min` : span(Math.abs(m) * 60)} ${m < 0 ? "before" : "after"}`);

function Drawing({ edges, nodes, from, to, label }: { edges: NeighbourEdge[]; nodes: string[]; from: string; to: string; label: string }) {
  const marker = useId().replace(/:/g, "");
  const { layout, h } = layoutScored({ nodes, edges, from, to }, W, 240);
  const groups = new Map<string, NeighbourEdge[]>();
  for (const e of edges) groups.set(`${e.from}>${e.to}`, [...(groups.get(`${e.from}>${e.to}`) ?? []), e]);
  const degree: Record<string, number> = {};
  for (const k of groups.keys()) { const [a, b] = k.split(">"); if (a !== b) { degree[a] = (degree[a] ?? 0) + 1; degree[b] = (degree[b] ?? 0) + 1; } }
  return (
    <div style={{ overflowX: "auto", overflowY: "hidden" }}>
    <svg role="img" aria-label={label} viewBox={`0 0 ${W} ${h}`} style={{ width: "100%", minWidth: 520, maxWidth: W, height: "auto", display: "block" }}>
      <defs>
        {(Object.keys(PAINT) as When[]).map((k) => (
          <marker key={k} id={`${marker}${k}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0L8 4L0 8z" fill={PAINT[k].colour} />
          </marker>
        ))}
      </defs>
      {[...groups.entries()].map(([key, es]) => {
        const top = es.reduce((w, e) => (RANK[e.when] > RANK[w] ? e.when : w), "earlier" as When);
        const p = PAINT[top], a = layout[es[0].from], b = layout[es[0].to];
        const title = es.map((e) => `${e.from} to ${e.to}, ${e.when === "scored" ? "the scored transfer" : `${minute(e.minute)}: ${PAINT[e.when].word}`}`).join("; ");
        if (es[0].from === es[0].to) {
          return <g key={key}><circle cx={a.x} cy={a.y - NODE - 6} r={6} fill="none" stroke={p.colour} strokeDasharray={p.dash} /><title>{title}</title></g>;
        }
        const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
        const x1 = a.x + ux * (NODE + 1), y1 = a.y + uy * (NODE + 1), x2 = b.x - ux * (NODE + 3), y2 = b.y - uy * (NODE + 3);
        const da = degree[es[0].from] ?? 0, db = degree[es[0].to] ?? 0, at = da > db ? 0.78 : db > da ? 0.22 : 0.5;
        return (
          <g key={key}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={p.colour} strokeWidth={top === "scored" ? 2.4 : 1.3} strokeDasharray={p.dash} markerEnd={`url(#${marker}${top})`} />
            {es.length > 1 && <text x={x1 + (x2 - x1) * at} y={y1 + (y2 - y1) * at + 3} textAnchor="middle" fontSize={10} fill="var(--text)" style={halo}>{`\u00d7${es.length}`}</text>}
            <title>{title}</title>
          </g>
        );
      })}
      {nodes.map((n) => {
        const q = layout[n];
        return <g key={n}><circle cx={q.x} cy={q.y} r={NODE} fill="var(--panel)" stroke="var(--border)" /><text x={q.x} y={q.y + 4} textAnchor="middle" fontSize={11} fill="var(--text)">{n}</text></g>;
      })}
    </svg>
    </div>
  );
}

export default function IbmNeighbourhood({ sample }: { sample: NeighbourhoodSample }) {
  const view = sample.view;
  const seeds = sample.seeds.positions_in_batch, labels = sample.seeds.labels;
  const [k, setK] = useState(0);
  const fields = useMemo(() => (view ? seedFields(sample.inputs.edge_index, sample.inputs.n_edges, view, seeds) : []), [sample, view, seeds]);
  const g = useMemo(() => (view ? seedGraph(sample.inputs.edge_index, sample.inputs.n_edges, view, seeds[k]) : null), [sample, view, seeds, k]);
  if (!view || !g) return null;
  const mismatch = fieldsMismatch(fields, view.per_seed);
  const reaching = fields.filter((f) => f.past_horizon > 0).length;
  const furthest = Math.max(0, ...fields.map((f) => f.furthest_later_s));
  const ibm = sample.protocol === "ibm";
  const word = (i: number) => (labels[i] === 1 ? "laundering" : "clean");
  return (
    <figure style={{ margin: "12px 0 0" }}>
      <figcaption>
        <b>What the model read for each transfer it scored.</b>{" "}
        A transfer&apos;s horizon is the one this page&apos;s own models use: the end of its batch of 2,048 rows, and at most{" "}
        {span(view.reach_s)} after it.{" "}
        {ibm
          ? `IBM's loader samples neighbours from the whole test graph: ${reaching} of the ${fields.length} transfers read at least one transfer past their horizon, the latest ${span(furthest)} after the transfer scored.`
          : `Here neighbours come only from the scored transfer's batch or earlier ones, the page's batches of 2,048 rows split where one spans more than ${span(view.reach_s)}: none of the ${fields.length} reads past its horizon, and the latest transfer read is ${span(furthest)} after the one scored.`}
      </figcaption>
      {mismatch && <p className="small" role="alert" style={{ color: "var(--danger)" }}>The counts this page derives from the sample differ from the ones the export recorded: {mismatch}.</p>}
      <p className="small" style={{ margin: "8px 0 4px" }}>
        Transfer {k + 1}, {word(k)} in the corpus, and {g.shown === g.at_accounts ? `all ${n(g.shown)}` : `${n(g.shown)} of the ${n(g.at_accounts)}`} transfers
        at its two accounts that the sample holds{g.shown === g.at_accounts ? "" : ", those nearest it in time, split between the two accounts"}.
        A pays B; the others are lettered in time order. Hover a line for when it happened.
      </p>
      <Drawing edges={g.edges} nodes={g.nodes} from={g.from} to={g.to} label={`The two accounts of transfer ${k + 1} and their transfers in the sample`} />
      <p className="small" style={{ margin: "4px 0 8px" }}>
        {(Object.keys(PAINT) as When[]).map((w, i) => (
          <span key={w} style={{ marginRight: 12, whiteSpace: "nowrap" }}>
            <svg width="22" height="8" aria-hidden="true" style={{ marginRight: 4 }}><line x1="0" y1="4" x2="22" y2="4" stroke={PAINT[w].colour} strokeWidth={w === "scored" ? 2.4 : 1.6} strokeDasharray={PAINT[w].dash} /></svg>
            {PAINT[w].word}{i < 3 ? "" : "."}
          </span>
        ))}
      </p>
      <table className="table small">
        <caption className="small" style={{ textAlign: "left", captionSide: "bottom", paddingTop: 4 }}>
          Read: the transfers at the accounts within one hop of the payer or the payee, which two rounds of message passing in
          both directions take in. Each count is of distinct transfers in this sample, the scored one excluded.
        </caption>
        <thead>
          <tr><th>Transfer</th><th>Read</th><th>Later, within its horizon</th><th>Past its horizon</th><th>Latest read</th><th>Payer out / in</th><th>Payee out / in</th></tr>
        </thead>
        <tbody>
          {fields.map((f, i) => (
            <tr key={i} style={i === k ? { background: "var(--bg-2)" } : undefined}>
              <td><button type="button" style={ROW_BUTTON} aria-pressed={i === k} onClick={() => setK(i)}>{i + 1}, {word(i)}</button></td>
              <td>{n(f.read)}</td><td>{n(f.later_within_horizon)}</td><td>{n(f.past_horizon)}</td><td>{f.furthest_later_s > 0 ? `${span(f.furthest_later_s)} after` : "none later"}</td>
              <td>{n(f.payer_out)} / {n(f.payer_in)}</td><td>{n(f.payee_out)} / {n(f.payee_in)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
