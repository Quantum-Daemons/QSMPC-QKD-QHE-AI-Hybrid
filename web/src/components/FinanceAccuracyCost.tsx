// Accuracy against cost on the finance page (Round 86, the owner's choice of 2026-11-23). Each model's AUPRC on the
// 1,523,504-transfer test window is its record (models/metrics.json, through the page's config); its cost is this tab's
// run: the student's encrypted batch over its held-out rows divided by the rows, and one transaction each for the MLP and
// the compact GNN under CKKS and garbled circuits. The gradient-boosted model has no encrypted path here, so it is a line
// with no cost. A model whose encrypted run did not happen in this run has no point, and the caption says so.
import { useId, useMemo } from "react";
import {
  CartesianGrid, LabelList, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { ChartReadout, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import { modelsFor } from "../lib/config";

type Metrics = Record<string, unknown>;
export interface CostPoint { key: string; name: string; auprc: number; ms: number; how: string }

/** The three encrypted paths this plot prices, and where the run reports each one's time. */
export const PRICED = [
  { key: "finance_aml.student", name: "student (CKKS, batched)" },
  { key: "finance_aml.mlp_large", name: "MLP (CKKS and garbled circuits)" },
  { key: "finance_aml.gnn_compact", name: "compact GNN (CKKS and garbled circuits)" },
] as const;
export const PLAINTEXT_ONLY = { key: "finance_aml.gbm", name: "gradient-boosted model" } as const;

/** A model's AUPRC on the test window as its record states it: `auprc`, or, for the student, whose headline is F1, the
 *  raw AUPRC its calibration record scored on the same held-out test split. */
export function recordedAuprc(achieved: Record<string, unknown> | null | undefined): number | null {
  if (!achieved) return null;
  if (typeof achieved.auprc === "number") return achieved.auprc;
  const cal = achieved.calibration as { auprc_raw?: unknown } | undefined;
  return typeof cal?.auprc_raw === "number" ? cal.auprc_raw : null;
}

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Each priced model's point from a finished run's metrics, and the ones that have none, with why. */
export function costPoints(metrics: Metrics, auprcOf: (key: string) => number | null): { points: CostPoint[]; missing: string[] } {
  const points: CostPoint[] = [], missing: string[] = [];
  for (const m of PRICED) {
    const auprc = auprcOf(m.key);
    if (auprc == null) { missing.push(`${m.name}: no AUPRC in its record`); continue; }
    if (m.key === "finance_aml.student") {
      const rows = num(metrics.encrypted_rows_scored), ms = num(metrics.encrypted_batch_ms);
      if (metrics.encrypted_model !== m.key || rows == null || rows <= 0 || ms == null) {
        missing.push(`${m.name}: its encrypted batch did not run`);
        continue;
      }
      const perCt = num(metrics.encrypted_rows_per_ciphertext), cts = num(metrics.encrypted_ciphertexts);
      points.push({ key: m.key, name: m.name, auprc, ms: ms / rows,
        how: `${rows.toLocaleString("en-US")} held-out transfers${cts != null && perCt != null ? ` in ${cts} ciphertexts of ${perCt}` : ""}, ${(ms / 1000).toFixed(1)} s in all` });
      continue;
    }
    const tag = m.key === "finance_aml.mlp_large" ? "mlp" : "gnn";
    const ms = num(metrics[`mpc_${tag}_total_ms`]);
    if (metrics[`mpc_${tag}_ran`] !== 1 || metrics[`mpc_${tag}_model`] !== m.key || ms == null) {
      missing.push(`${m.name}: it did not run under MPC`);
      continue;
    }
    points.push({ key: m.key, name: m.name, auprc, ms,
      how: tag === "gnn" ? "one transfer with its neighbourhood, ReLUs and argmax in garbled circuits" : "one transfer, ReLUs and argmax in garbled circuits" });
  }
  return { points, missing };
}

const tick = { fill: "var(--muted)", fontSize: 11 };
const axisLabel = (value: string) => ({ value, position: "insideBottom" as const, offset: -8, fill: "var(--muted)", fontSize: 11 });
type AxisBox = { x?: number; y?: number; width?: number; height?: number };
const yLabel = (value: string) => ({
  value,
  content: ({ viewBox }: { viewBox?: AxisBox }) => {
    const { x = 0, y = 0, height = 0 } = viewBox ?? {};
    const cx = x + 10.5, cy = y + height / 2;
    return <text x={cx} y={cy} transform={`rotate(-90, ${cx}, ${cy})`} textAnchor="middle" dominantBaseline="central"
      fill="var(--muted)" fontSize={11}>{value}</text>;
  },
});
/** Where each name sits beside its point: the MLP and the compact GNN score within 0.03 of each other and of the GBM's
 *  line, and cost within a decade, so one name goes above and one below; the student's goes to its right. */
const PLACE: Record<string, "right" | "above" | "below"> = {
  "finance_aml.student": "right", "finance_aml.mlp_large": "below", "finance_aml.gnn_compact": "above",
};
function pointName({ cx, cy, value }: { cx?: number | string; cy?: number | string; value?: unknown }) {
  const x = Number(cx), y = Number(cy);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const [name, place] = String(value ?? "").split("|");
  const at: { x: number; y: number; anchor: "start" | "middle" } = place === "right" ? { x: x + 9, y, anchor: "start" }
    : { x, y: place === "above" ? y - 15 : y + 17, anchor: "middle" };
  return <text x={at.x} y={at.y} textAnchor={at.anchor} dominantBaseline="central" fill="var(--text)" fontSize={11}>{name}</text>;
}
/** The GBM's line names itself briefly above its left end (the caption says what it is): the MLP and the compact GNN,
 *  within 0.03 of it, sit to the right, and a longer name reached the compact GNN's at 250 px. */
function lineName({ viewBox, value }: { viewBox?: unknown; value?: unknown }) {
  const { x = 0, y = 0 } = (viewBox ?? {}) as AxisBox;
  return <text x={x + 4} y={y - 6} fill="var(--muted)" fontSize={11}>{String(value ?? "")}</text>;
}
const COLOURS: Record<string, string> = {
  "finance_aml.student": "var(--warn)", "finance_aml.mlp_large": "var(--accent-2)", "finance_aml.gnn_compact": "var(--accent)",
};
const SHORT: Record<string, string> = { "finance_aml.student": "student", "finance_aml.mlp_large": "MLP", "finance_aml.gnn_compact": "compact GNN" };

export default function FinanceAccuracyCost({ metrics }: { metrics: Metrics }) {
  const captionId = useId();
  // When the run this plot prices handed over its metrics: a finished run sets them once.
  const measuredAt = useMemo(() => new Date(), [metrics]);
  const specs = modelsFor("finance_aml");
  const auprcOf = (key: string) => recordedAuprc(specs.find((s) => s.key === key)?.achieved as Record<string, unknown> | undefined);
  const { points, missing } = costPoints(metrics, auprcOf);
  const gbm = auprcOf(PLAINTEXT_ONLY.key);
  const mlp = specs.find((s) => s.key === "finance_aml.mlp_large")?.achieved as { n_test?: number; auprc_baseline?: number } | undefined;
  const testRows = mlp?.n_test, chance = mlp?.auprc_baseline;
  if (!points.length && gbm == null) return null;
  const xs = points.map((p) => p.ms);
  // A decade's margin at least a third wide on each side, so no point sits on the frame.
  const lo = 10 ** Math.floor(Math.log10(Math.min(...xs) / 3)), hi = 10 ** Math.ceil(Math.log10(Math.max(...xs) * 3));
  const xTicks: number[] = [];
  for (let v = lo; v <= hi * 1.0001; v *= 10) xTicks.push(v);
  const when = measuredAt.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
  return (
    <div className="card" id="finance-accuracy-cost" style={{ marginTop: 14 }}>
      <h3>Accuracy against encrypted cost</h3>
      <figure className="chart-figure" style={{ margin: "6px 0 0" }}>
        <figcaption id={captionId} className="small muted">
          <b>Each model&apos;s AUPRC against what one transfer cost under encryption in this run.</b> The AUPRC is each
          record&apos;s, on the same{testRows ? ` ${testRows.toLocaleString("en-US")}-transfer` : ""} test window{chance != null
            ? <>, where a random ranking scores {chance.toFixed(4)}, the share of transfers that are laundering</> : null}. The
          times were measured in this tab, {when}, and are not
          comparable as protocols: the student&apos;s is its batch divided by the transfers it packed together, while the
          MLP&apos;s and the compact GNN&apos;s are one transfer each, its ReLUs and argmax in garbled circuits.
          {gbm != null ? <> The dashed line is the {PLAINTEXT_ONLY.name} (GBM), which has no encrypted path on this page and so no cost.</> : null}
          {missing.length ? <> No point: {missing.join("; ")}.</> : null}
        </figcaption>
        {points.length ? (
          <ChartReadout hint={HINT_POINTER}>
            <MarksHidden>
              <ResponsiveContainer width="100%" height={250}>
                <ScatterChart role="img" title="AUPRC on the test window against encrypted milliseconds per transfer, this run"
                  aria-describedby={captionId} margin={{ top: 8, right: 24, bottom: 18, left: 4 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="ms" name="ms per transfer" scale="log" domain={[lo, hi]} ticks={xTicks}
                    allowDataOverflow tick={tick} tickFormatter={(v: number) => (v >= 1 ? v.toLocaleString("en-US") : String(v))}
                    label={axisLabel("ms per transfer under encryption (log scale)")} />
                  <YAxis type="number" dataKey="auprc" name="AUPRC" domain={[0, 0.8]} ticks={[0, 0.2, 0.4, 0.6, 0.8]} tick={tick}
                    tickFormatter={(v: number) => v.toFixed(1)} width={44} label={yLabel("AUPRC, test window")} />
                  <Tooltip content={<ReadoutTooltip />}
                    formatter={(v: number, name: string, entry: { payload?: CostPoint }, i: number) =>
                      i === 0 ? [`${v < 10 ? v.toFixed(3) : v.toFixed(0)} ms (${entry.payload?.how ?? ""})`, `${entry.payload?.name ?? ""}: ${name}`]
                        : [v.toFixed(4), name]} />
                  <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                  {gbm != null && (
                    <ReferenceLine y={gbm} stroke="var(--muted)" strokeDasharray="5 4"
                      label={{ value: `GBM ${gbm.toFixed(3)}`, content: lineName }} />
                  )}
                  {points.map((p) => (
                    <Scatter key={p.key} name={p.name} data={[{ ...p, short: `${SHORT[p.key]}|${PLACE[p.key]}` }]} fill={COLOURS[p.key]} isAnimationActive={false}>
                      <LabelList dataKey="short" content={pointName} />
                    </Scatter>
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </MarksHidden>
          </ChartReadout>
        ) : null}
      </figure>
      <div style={{ overflowX: "auto" }}>
        <table className="table small" style={{ marginTop: 6 }}>
          <thead><tr><th>Model</th><th>AUPRC (record)</th><th>ms per transfer (this run)</th><th>What was timed</th></tr></thead>
          <tbody>
            {points.map((p) => <tr key={p.key}><td>{p.name}</td><td>{p.auprc.toFixed(4)}</td><td>{p.ms < 10 ? p.ms.toFixed(3) : p.ms.toFixed(0)}</td><td>{p.how}</td></tr>)}
            {gbm != null ? <tr><td>{PLAINTEXT_ONLY.name}</td><td>{gbm.toFixed(4)}</td><td>none</td><td>no encrypted path on this page</td></tr> : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
