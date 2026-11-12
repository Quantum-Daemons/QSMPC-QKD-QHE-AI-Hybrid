// The compact transaction GNN under MPC, on one held-out transaction (Round 86). Every number comes from the run or the model's record; the caveats are part of the result.
import { useId } from "react";
import { modelsFor } from "../lib/config";
import type { MpcGnnReport } from "../lib/clientCompute";
import StageBadge from "./StageBadge";

const mb = (b: number) => `${(b / 1e6).toFixed(2)} MB`;
const s = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
const short = (v: number) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : v.toFixed(0));
const halo = { paintOrder: "stroke", stroke: "var(--panel)", strokeWidth: 6 } as const;

/** The tree the circuits run on, drawn from the row's own edges. An empty slot is dashed: under encryption it is a zero vector, so the server cannot tell it from a filled one. */
function Neighbourhood({ report }: { report: MpcGnnReport }) {
  const marker = useId().replace(/:/g, "");
  // 136 px from each outer column to its edge: an edge's label, "1.8M Reinvestment", is about 125 px at 11 px in DejaVu Sans, the
  // CI runners' font, and ran past a 640-wide canvas there while it fitted in Noto (found by the two-font layout scan, Round 86).
  const D = report.perDirection, width = 720, height = 150 + 2 * 30 * D / 2 + 30;
  const left = 152, right = 568, u = { x: 276, y: height / 2 }, v = { x: 444, y: height / 2 };
  const bySlot = new Map(report.neighbourhood.map((e) => [e.slot, e]));
  const groupTop = (g: number) => (g % 2 === 0 ? 30 : height / 2 + 26);
  const place = (slot: number) => {
    const g = Math.floor(slot / D), i = slot % D;
    return { g, x: g < 2 ? left : right, y: groupTop(g) + i * 28, end: g < 2 ? u : v, into: g % 2 === 0, near: g < 2 };
  };
  const captions = [["into A", left, 16], ["out of A", left, height / 2 + 12], ["into B", right, 16], ["out of B", right, height / 2 + 12]] as const;
  return (
    <div style={{ overflowX: "auto", margin: "8px 0 10px" }}>
      <svg role="img" aria-label={`The transaction's neighbourhood: ${report.neighbourhood.length} of ${4 * D} slots filled`} viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", minWidth: 520, maxWidth: width, height: "auto", display: "block" }}>
        <defs>
          <marker id={`${marker}m`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L8 4L0 8z" fill="var(--muted)" /></marker>
          <marker id={`${marker}a`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L8 4L0 8z" fill="var(--accent)" /></marker>
        </defs>
        {Array.from({ length: 4 * D }, (_, slot) => {
          const p = place(slot), e = bySlot.get(slot);
          const a = p.into ? { x: p.x, y: p.y } : p.end, b = p.into ? p.end : { x: p.x, y: p.y };
          const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, gap = 15;
          const line = { x1: a.x + (dx / len) * gap, y1: a.y + (dy / len) * gap, x2: b.x - (dx / len) * gap, y2: b.y - (dy / len) * gap };
          if (!e) return (
            <g key={slot}>
              <line {...line} stroke="var(--border)" strokeWidth={1} strokeDasharray="3 4" />
              <circle cx={p.x} cy={p.y} r={10} fill="var(--panel)" stroke="var(--border)" strokeDasharray="3 3" />
            </g>
          );
          const partner = p.into ? e.from : e.to;
          return (
            <g key={slot}>
              <title>{`${e.from} to ${e.to}: ${e.amount.toLocaleString()} ${e.currency}, ${e.format}, ${Math.abs(e.minutes_before_scored)} minutes ${e.minutes_before_scored < 0 ? "after" : "before"} the scored transaction`}</title>
              <line {...line} stroke="var(--muted)" strokeWidth={1.2} markerEnd={`url(#${marker}m)`} />
              <circle cx={p.x} cy={p.y} r={10} fill="var(--panel)" stroke="var(--muted)" />
              <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={11} fill="var(--text)">{partner}</text>
              <text x={p.near ? p.x - 16 : p.x + 16} y={p.y + 4} textAnchor={p.near ? "end" : "start"} fontSize={11} fill="var(--muted)" style={halo}>{short(e.amount)} {e.format}</text>
            </g>
          );
        })}
        {captions.map(([t, x, y]) => <text key={t} x={x} y={y} textAnchor="middle" fontSize={11} fill="var(--muted)" style={halo}>{t}</text>)}
        <line x1={u.x + 19} y1={u.y} x2={v.x - 19} y2={v.y} stroke="var(--accent)" strokeWidth={2} markerEnd={`url(#${marker}a)`} />
        <text x={(u.x + v.x) / 2} y={u.y - 8} textAnchor="middle" fontSize={11} fill="var(--accent)" style={halo}>scored</text>
        {[["A", u], ["B", v]].map(([t, p]) => {
          const at = p as { x: number; y: number };
          return (
            <g key={t as string}>
              <circle cx={at.x} cy={at.y} r={16} fill="var(--panel)" stroke="var(--accent)" strokeWidth={2} />
              <text x={at.x} y={at.y + 5} textAnchor="middle" fontSize={13} fill="var(--text)">{t as string}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function MpcGnnPanel({ report }: { report: MpcGnnReport }) {
  const r = report.result;
  const name = (c: number) => report.classes[String(c)] ?? String(c);
  const models = modelsFor("finance_aml");
  const achieved = (key: string) => models.find((m) => m.key === key)?.achieved as Record<string, unknown> | null | undefined;
  const f1 = (key: string) => { const v = achieved(key)?.minority_class_f1; return typeof v === "number" ? v.toFixed(3) : "-"; };
  const booster = (achieved("finance_aml.mlp_large") as { xgboost_same_features?: { minority_class_f1?: number } } | null | undefined)?.xgboost_same_features?.minority_class_f1;
  const filled = report.neighbourhood.length, slots = 4 * report.perDirection;
  const reach = (achieved(report.key) as { batch_reach_minutes?: number } | null | undefined)?.batch_reach_minutes ?? "-";
  const rows: [string, string][] = [
    ["ReLUs in garbled circuits", `${r.reluCount}, in ${r.batches} batches, ${r.andGates.toLocaleString()} AND gates with the argmax`],
    ["The tree", `${r.accounts} accounts: 2 endpoints and ${slots} slots, of which ${filled} hold a transaction. Empty slots are encrypted zeros, so the server sees only the shape`],
    ["Garbled-circuit traffic", `${mb(r.gcBytes)} in ${r.gcMessages.toLocaleString()} messages between the two notional parties`],
    ["CKKS", `${r.rotations} rotations and ${r.conjugations} conjugations; ${mb(r.ckksBytes)} of ciphertexts, ${mb(r.keyBytes)} of keys`],
    ["Wall-clock", `${s(r.totalMs)} in ${report.ranIn}: garbled circuits ${s(r.gcWallMs)} on ${r.gcWorkers} worker${r.gcWorkers === 1 ? "" : "s"} (${s(r.gcMs)} of work, compiling ${s(r.gcCompileMs)}), CKKS ${s(r.ckksMs)}`],
    ["Masks", `reals uniform on +-2^${r.maskBits.dense} on the hidden layers, +-2^${r.maskBits.logit} on the logits`],
    ["Statistical distance of everything masked, this run", `at most 2^${r.statisticalDistanceLog2.toFixed(1)}`],
    ["Largest error against the clear network's logits", r.logitMaxAbsError.toExponential(2)],
    ["Logits the trainer recorded for this transaction", `${report.expectedLogits.map((v) => v.toFixed(3)).join(", ")}; reconstructed here ${r.logits.map((v) => v.toFixed(3)).join(", ")}`],
    ["Minority-class F1 over the whole held-out window", `this model ${f1(report.key)}; the published MLP ${f1("finance_aml.mlp_large")}; XGBoost on the MLP's features ${typeof booster === "number" ? booster.toFixed(3) : "-"}`],
  ];
  return (
    <div className="card" style={{ marginTop: 14 }} id="mpc-gnn">
      <h3>The compact GNN under MPC <StageBadge kind="notional" /></h3>
      <p style={{ marginTop: 4 }}>
        <code>{report.key}</code> on one held-out transaction of AMLworld&apos;s test window, index {report.rowIndex} of {report.rowCount}, labelled <b>{name(report.rowLabel)}</b>, and the {report.perDirection} most recent
        edges per direction at each of its two accounts. The client computes each slot from its own transactions and encrypts them. CKKS applies one shared map to every account at once and the server masks the result;
        garbled circuits apply the ReLUs; the sums over neighbours are done on shares, which costs no gates; and a 2PC argmax gives the class. No activation is in either party&apos;s hands.
      </p>
      <Neighbourhood report={report} />
      <p style={{ margin: "6px 0 10px" }}>
        Class from the 2PC argmax: <b>{name(r.prediction)}</b>.{" "}
        <span className={`badge-state ${r.matchesClear ? "done" : "error"}`}>{r.matchesClear ? "matches the clear network" : "DISAGREES with the clear network"}</span>
      </p>
      <table className="table">
        <tbody>
          {rows.map(([k, v]) => <tr key={k}><th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{k}</th><td>{v}</td></tr>)}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 10 }}>
        Both parties run in this tab, so the separation is notional, and only that lets the page compare the result with the clear network. The protocol is semi-honest and the masks are bounded reals, so they hide a value only
        statistically, by this project&apos;s own bound for this run. The figure&apos;s edges are the row&apos;s own, with accounts renamed by letters per row. Neighbours and features see transactions up to the end of the
        row&apos;s batch of 2,048, at most {reach} minutes past the transaction, as the Graph Feature Preprocessor paper processes them. This model reads {slots} edges of a transaction&apos;s neighbourhood; the official Multi-GNN models read a sample of 100 neighbours per hop
        from the whole graph of the test window, so the two are different settings and their figures are not a like-for-like comparison. AMLworld is synthetic by IBM&apos;s own statement.
      </p>
    </div>
  );
}
