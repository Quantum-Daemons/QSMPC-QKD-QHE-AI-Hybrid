// Laundering as a graph (Round 86): the shapes the AMLworld paper models, and the accounts around the row the published MLP scored. The AMLworld paper presents its data this way, because no single row shows a pattern across accounts.
import { useEffect, useId, useMemo, useState } from "react";
import { groupEdges, layoutExemplar, layoutScored, type Layout, type ScoredRow, type TxEdge, type TxGraph, type TxGraphsFile } from "../lib/transactionGraph";

const NODE = 11;
const danger = "var(--danger)", muted = "var(--muted)", accent = "var(--accent)", edge = "var(--border)", ink = "var(--text)";
const halo = { paintOrder: "stroke", stroke: "var(--panel)", strokeWidth: 6 } as const;

const describe = (e: TxEdge) => `${e.from} to ${e.to}: ${e.amount.toLocaleString()} ${e.currency}, ${e.format}, minute ${e.minute}, ${e.laundering ? "laundering" : "clean"}`;

function GraphSvg({ g, layout, w, h, scoredEdge, numbered, label }: { g: TxGraph; layout: Layout; w: number; h: number; scoredEdge?: { from: string; to: string }; numbered: boolean; label: string }) {
  const marker = useId().replace(/:/g, "");
  const groups = useMemo(() => groupEdges(g), [g]);
  return (
    <svg role="img" aria-label={label} viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", maxWidth: w, height: "auto", display: "block" }}>
      <defs>
        {(["d", "m", "a"] as const).map((k) => (
          <marker key={k} id={`${marker}${k}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0L8 4L0 8z" fill={k === "d" ? danger : k === "a" ? accent : muted} />
          </marker>
        ))}
      </defs>
      {groups.map((grp) => {
        const a = layout[grp.from], b = layout[grp.to];
        const scored = scoredEdge && grp.from === scoredEdge.from && grp.to === scoredEdge.to;
        const color = scored ? accent : grp.laundering ? danger : muted;
        const kind = scored ? "a" : grp.laundering ? "d" : "m";
        if (grp.from === grp.to) {
          return <g key={`${grp.from}${grp.to}`}><circle cx={a.x} cy={a.y - NODE - 6} r={6} fill="none" stroke={color} /><title>{grp.edges.map(describe).join("; ")}</title></g>;
        }
        const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
        const x1 = a.x + ux * (NODE + 1), y1 = a.y + uy * (NODE + 1), x2 = b.x - ux * (NODE + 3), y2 = b.y - uy * (NODE + 3);
        return (
          <g key={`${grp.from}${grp.to}`}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={scored ? 2.4 : 1.3} markerEnd={`url(#${marker}${kind})`} />
            {(numbered || grp.edges.length > 1) && <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 + 3} textAnchor="middle" fontSize={10} fill={ink} style={halo}>{numbered ? grp.order : `×${grp.edges.length}`}</text>}
            <title>{grp.edges.map(describe).join("; ")}</title>
          </g>
        );
      })}
      {g.nodes.map((n) => {
        const p = layout[n];
        return <g key={n}><circle cx={p.x} cy={p.y} r={NODE} fill="var(--panel)" stroke={edge} /><text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={11} fill={ink}>{n}</text></g>;
      })}
    </svg>
  );
}

const SHAPE_NOTE: Record<string, string> = {
  "fan-out": "One account splits funds across many.",
  "fan-in": "Many accounts converge on one.",
  "gather-scatter": "Funds gather at one account, then scatter.",
  bipartite: "Two groups, money flowing between them.",
  cycle: "Funds return to where they started.",
  chain: "Funds hop from account to account.",
};

export default function TransactionGraphCard({ scoredRow }: { scoredRow: number | null }) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<TxGraphsFile | null>(null);
  const [failed, setFailed] = useState("");
  useEffect(() => {
    if (!open || data) return;
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/finance/transaction_graphs.json`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d: TxGraphsFile) => { if (live) setData(d); })
      .catch((e: Error) => { if (live) setFailed(e.message); });
    return () => { live = false; };
  }, [open, data]);
  const row: ScoredRow | null = data && scoredRow !== null ? data.scored_rows[scoredRow] ?? null : null;
  return (
    <details className="card details" style={{ marginTop: 14 }} id="transaction-graphs" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>What the transactions look like as a graph</summary>
      <div className="details-body">
        {failed && <p className="muted">The graphs could not be loaded: {failed}.</p>}
        {data && (<>
          <p>Laundering is a pattern across accounts, and one row does not show it, which is why the AMLworld paper draws its data this way and why the features count fan-in, fan-out and shared counterparties. Below, one example of each shape from the held-out window: every edge is labelled laundering, numbered in time order, and hovering shows the amount and format.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12, margin: "8px 0" }}>
            {data.exemplars.map((x) => (
              <figure key={x.shape} style={{ margin: 0 }}>
                <figcaption><b>{x.shape}</b>, {x.edges.length} transactions over {Math.round(x.edges[x.edges.length - 1].minute / 60)} h. {SHAPE_NOTE[x.shape]}</figcaption>
                {(() => { const { layout, h } = layoutExemplar(x, 240, 180); return <GraphSvg g={x} layout={layout} w={240} h={h} numbered label={`A ${x.shape} laundering pattern of ${x.edges.length} transactions`} />; })()}
              </figure>
            ))}
          </div>
          {row ? (
            <figure style={{ margin: "10px 0 0" }}>
              <figcaption><b>The transaction the published MLP just scored</b>, labelled {row.label ? "laundering" : "clean"} in the corpus, and the transactions at its two accounts in the three days up to the end of its batch: {row.edges.length} of {row.edges_in_window} shown (the row, then laundering-labelled edges, then the most recent).</figcaption>
              {(() => { const { layout, h } = layoutScored(row, 720, 300); return <GraphSvg g={row} layout={layout} w={720} h={h} scoredEdge={{ from: row.from, to: row.to }} numbered={false} label="The accounts around the transaction the model scored" />; })()}
              <p className="small">Blue is the scored transaction, red a laundering-labelled edge, grey a clean one. A number on a line is several transactions in that direction.</p>
            </figure>
          ) : <p className="small">Run the page with “MLP under MPC” on to draw the accounts around the transaction it scores.</p>}
          <p className="small muted">{data.provenance} {data.modifications} Licence: <a href={data.licence_url}>{data.licence}</a>. {data.attribution}</p>
        </>)}
      </div>
    </details>
  );
}
