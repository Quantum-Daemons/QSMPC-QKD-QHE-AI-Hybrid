// Privacy against utility on the medical page's DP arm, with the spread of its noise (Round 86, the owner's choice of
// 2026-11-14: the evidence plots). The band and the line are a recorded sweep of the page's own `fedHeartDp` over 20
// draws of seeded noise at the page's seed (scripts/measure_fedheart_dp_sweep.py, web/public/data/medical/
// fedheart_dp_sweep.json); the diamonds are this run, whose noise came from the tab's CSPRNG. One draw against twenty
// says whether this run sits inside what the noise usually does.
import { useEffect, useId, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Scatter, XAxis, YAxis } from "recharts";
import type { FedHeartDpResult } from "../lib/clientFedHeart";

interface SweepRow { epsilon: number | null; draws: number; mean: number; sd: number; p10: number; p50: number; p90: number }
interface Sweep { seed: number; draws: number; delta: number; metric: string; rows: SweepRow[] }

const AXIS = { fontSize: 10, fill: "#8a94a6" };
const label = (e: number | null) => (e == null ? "no noise" : `epsilon ${e}`);

export default function DpPrivacyUtility({ dp }: { dp: FedHeartDpResult }) {
  const [sweep, setSweep] = useState<Sweep | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const captionId = useId();
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/medical/fedheart_dp_sweep.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: Sweep) => { if (live) setSweep(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, []);
  if (failed) return <p className="small" style={{ color: "var(--danger)" }}>The recorded sweep did not load: {failed}.</p>;
  if (!sweep) return null;
  // Strongest privacy on the left, no noise on the right.
  const order = [...sweep.rows].sort((a, b) => (a.epsilon ?? Infinity) - (b.epsilon ?? Infinity));
  const here = (e: number | null) => dp.rows.find((x) => (Number.isFinite(x.epsilon) ? x.epsilon : null) === e)?.mean ?? null;
  const data = order.map((r) => ({ x: label(r.epsilon), band: [r.p10, r.p90], median: r.p50, run: here(r.epsilon) }));
  let lo = Infinity, hi = -Infinity;
  for (const d of data) {
    lo = Math.min(lo, d.band[0], d.run ?? Infinity);
    hi = Math.max(hi, d.band[1], d.run ?? -Infinity);
  }
  const pad = (hi - lo) * 0.1 || 0.01;
  const inside = data.filter((d) => d.run != null && d.run >= d.band[0] && d.run <= d.band[1]).length;
  const withRun = data.filter((d) => d.run != null).length;
  const strongest = order.find((r) => r.epsilon != null);
  return (
    <figure className="chart-figure" style={{ margin: "12px 0 0" }}>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} role="img" aria-describedby={captionId}
                       title={`Accuracy against epsilon over ${sweep.draws} draws of the noise, with this run`}
                       margin={{ top: 6, right: 10, bottom: 14, left: 4 }}>
          <CartesianGrid stroke="#2a3040" strokeDasharray="2 4" />
          <XAxis dataKey="x" tick={AXIS} />
          <YAxis domain={[lo - pad, hi + pad]} tick={AXIS} width={48} tickFormatter={(v: number) => v.toFixed(2)} />
          <Area dataKey="band" stroke="none" fill="#4f9dff" fillOpacity={0.18} isAnimationActive={false} />
          <Line dataKey="median" stroke="#4f9dff" strokeWidth={1.6} dot={{ r: 2.5 }} isAnimationActive={false} />
          <Scatter dataKey="run" fill="#f6b73c" shape="diamond" isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <figcaption id={captionId} className="muted small">
        <strong>Privacy against accuracy, with the noise&apos;s spread.</strong> The band holds the 10th to 90th percentile and
        the line the median of {sweep.draws} draws of the noise at each epsilon, recorded offline with the page&apos;s own code
        at its seed and seeded noise; the orange diamonds are this run, {inside} of {withRun} inside the band. The y axis is
        FLamby&apos;s metric. {strongest ? <>At {label(strongest.epsilon)} the draws run from {strongest.p10.toFixed(3)} to{" "}
        {strongest.p90.toFixed(3)} (10th to 90th percentile), so one run&apos;s figure at that epsilon is one draw from that
        spread, not the model&apos;s accuracy under DP.</> : null}
      </figcaption>
    </figure>
  );
}
