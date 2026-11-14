// The satellite segment classifier's decision boundary on the overlay's plane, as one slice (Round 86, the owner's choice
// of 2026-11-14). The published classifier reads 19 features, so a boundary on two of them is a cross-section: the other
// 17 held at one observed test segment, the test partition's medoid. scripts/measure_satellite_boundary.py drew it only
// because it passed its check (at least 20 segments near the slice, at least 90% of them given the full model's class);
// every figure here is read from web/public/data/satellite/boundary.json.
import { useEffect, useMemo, useState } from "react";
import CardExport from "./CardExport";
import KeyPoints from "./KeyPoints";
import StageBadge from "./StageBadge";
import { contour } from "../lib/contour";

interface NearPoint { x: number; y: number; label: number; model: number; slice: number }
interface Boundary {
  model: string; hf_revision: string; kind: "slice" | "surrogate"; plane: [string, string];
  x: number[]; y: number[]; p: number[][]; model_test_auc_pr: number;
  slice: { anchor: string; held_values: Record<string, number>; near: number; near_rule: string; agreement: number } | null;
  near: NearPoint[] | null;
  test_segments: number;
}

const W = 520, H = 340, L = 52, R = 14, T = 12, B = 42;
const SLATE = [0x82, 0x96, 0xb5], ORANGE = [0xe0, 0x7a, 0x3c];
const mix = (p: number) => `rgb(${SLATE.map((c, i) => Math.round(c + (ORANGE[i] - c) * p)).join(",")})`;

function ticks(lo: number, hi: number, n = 5): number[] {
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n || 1)));
  const k = [1, 2, 5, 10].find((m) => (hi - lo) / (m * step) <= n) ?? 10;
  const s = k * step, out: number[] = [];
  for (let t = Math.ceil(lo / s) * s; t <= hi + 1e-9; t += s) out.push(Number(t.toFixed(10)));
  return out;
}

export default function SatBoundaryCard() {
  const [b, setB] = useState<Boundary | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/satellite/boundary.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: Boundary) => { if (live) setB(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, []);
  const segs = useMemo(() => (b ? contour(b.x, b.y, b.p) : []), [b]);
  if (failed) {
    return <div className="card" id="sat-boundary" style={{ marginTop: 14 }}><h3>The segment classifier&apos;s boundary</h3><p className="small" style={{ color: "var(--danger)" }}>Its recorded figures did not load: {failed}.</p></div>;
  }
  if (!b || !b.slice || !b.near) return null;
  const [x0, x1, y0, y1] = [b.x[0], b.x[b.x.length - 1], b.y[0], b.y[b.y.length - 1]];
  // The data inset by PAD inside the axes, so a point on the grid's edge (a segment at the fewest peaks) shows whole.
  const PAD = 6;
  const sx = (x: number) => L + PAD + (x - x0) / (x1 - x0) * (W - L - R - 2 * PAD);
  const sy = (y: number) => H - B - PAD - (y - y0) / (y1 - y0) * (H - T - B - 2 * PAD);
  const dx = (W - L - R - 2 * PAD) / (b.x.length - 1), dy = (H - T - B - 2 * PAD) / (b.y.length - 1);
  const s = b.slice, near = b.near;
  const agree = near.filter((q) => q.model === q.slice).length;
  const anomalies = near.filter((q) => q.label === 1).length;
  const label = (k: string) => (k === "n_peaks" ? "peaks in segment" : k === "diff2_peaks" ? "peaks in 2nd difference" : k);
  return (
    <div className="card" id="sat-boundary" style={{ marginTop: 14 }}>
      <h3>Where the segment classifier decides, as one slice <StageBadge kind="offline" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        The published OPS-SAT-AD classifier reads 19 features of a telemetry segment, so its boundary has no single place on this
        page&apos;s plane of {label(b.plane[0])} against {label(b.plane[1])}. This is a cross-section: the other 17 features are held at
        one observed test segment, the test partition&apos;s medoid, and the colour is the classifier&apos;s probability of an anomaly
        there. The line is p = 0.5, its default cut-off; the AUC-PR the page reports ({b.model_test_auc_pr.toFixed(3)}) uses no
        threshold. Computed offline from the published graph at its pinned revision; nothing here ran in your tab.
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} role="img"
        aria-label={`Probability of an anomaly over ${label(b.plane[0])} and ${label(b.plane[1])}, with the 0.5 line and ${near.length} test segments near the slice`}>
        {b.p.map((row, j) => row.map((p, i) => (
          <rect key={`${i}-${j}`} x={sx(b.x[i]) - dx / 2} y={sy(b.y[j]) - dy / 2} width={dx + 0.5} height={dy + 0.5}
            fill={mix(p)} fillOpacity={0.32} />
        )))}
        {segs.map(([ax, ay, bx, by], k) => (
          <line key={k} x1={sx(ax)} y1={sy(ay)} x2={sx(bx)} y2={sy(by)} stroke="var(--text)" strokeWidth={1.6} strokeDasharray="5 3" />
        ))}
        {near.map((q, k) => (
          <circle key={k} cx={sx(q.x)} cy={sy(q.y)} r={q.label === 1 ? 4.2 : 3.2} fill={q.label === 1 ? mix(1) : mix(0)}
            stroke={q.model !== q.slice ? "var(--text)" : "var(--bg)"} strokeWidth={q.model !== q.slice ? 1.8 : 0.8} />
        ))}
        <line x1={L} y1={H - B} x2={W - R} y2={H - B} stroke="#2a3040" />
        <line x1={L} y1={T} x2={L} y2={H - B} stroke="#2a3040" />
        {ticks(x0, x1).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - B + 14} textAnchor="middle" fontSize={10} fill="#8a94a6">{t}</text>
        ))}
        {ticks(y0, y1).map((t) => (
          <text key={`y${t}`} x={L - 6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill="#8a94a6">{t}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="#8a94a6">{label(b.plane[0])}</text>
        <text x={14} y={(T + H - B) / 2} textAnchor="middle" fontSize={11} fill="#8a94a6" transform={`rotate(-90 14 ${(T + H - B) / 2})`}>{label(b.plane[1])}</text>
      </svg>
      <p className="muted small" style={{ margin: "6px 0 10px" }}>
        Points: the {near.length} of the test partition&apos;s segments whose 17 held features each lie within one standard deviation
        of the medoid&apos;s ({anomalies} of them anomalous; orange anomalous, slate nominal, by their true class). For {agree} of
        the {near.length} ({(100 * s.agreement).toFixed(1)}%) the full classifier&apos;s class at 0.5 is the class the slice gives at
        their two values; a ringed point is one where they differ.
      </p>
      <KeyPoints title="What it does not show" items={[
        <><b>The classifier&apos;s 19-dimensional boundary.</b> Another anchor gives another slice; this one speaks for segments like the
          medoid, not for the other {b.test_segments - near.length} test segments, which the overlay above draws on the same plane.</>,
        <><b>Areas with no segment.</b> The colour is extrapolated wherever no point lies; the check is over the {near.length} points only.
          Peaks in a segment are whole counts, so the colour between whole values is the classifier read where no segment can be.</>,
      ]} />
      <p className="muted small" style={{ marginTop: 6 }}>
        Held at the medoid: {Object.entries(s.held_values).map(([k, v]) => `${k} ${Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : Number(v.toPrecision(4))}`).join(", ")}.
        Data: OPS-SAT-AD, Ruszczak, Kotowski, Evans and Nalepa, Scientific Data 12, 710 (2025), and Zenodo, doi:10.5281/zenodo.12588359,
        under <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>; ESA OPS-SAT mission data.
      </p>
      <CardExport name="satellite_comms-boundary" label="the boundary slice" rows={near.map((q) => ({ ...q }))} json={b} />
    </div>
  );
}
