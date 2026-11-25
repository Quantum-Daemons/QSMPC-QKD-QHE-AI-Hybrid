// The square student's error against its time per image, for each CKKS parameter set the pre-registration tried (Round
// 86, the owner's choice of 2026-11-23). The circles and the cross are the pre-registered record, read from
// web/public/data/medical/square_student_sets.json (scripts/export_square_student_sets.py, from
// docs/measurements/medical-square-student-2026-11-19.json); the diamond is this tab's latest run. The record's times
// are TenSEAL in Python on the development machine and the tab's are node-seal in this browser, so the caption says the
// two are not one axis's worth of comparison.
import { useEffect, useId, useState } from "react";
import {
  CartesianGrid, LabelList, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { ChartReadout, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import { errorDomain } from "../lib/logAxis";

interface SetFigures {
  N: number; bits: number[]; scale_bits: number; modulus_bits: number; pq_bound_bits: number; images: number;
  max_abs_logit_error: number; median_abs_logit_error: number; median_s: number; max_s: number;
  input_bytes: number; output_bytes: number; decisions_differ: number; criterion_3: boolean;
}
interface SetsFile {
  timing: string; timed_with: string; limit: number; limit_note: string; sets: Record<string, SetFigures>;
  attribution: { dataset: string; download: string; citation: string; licence: string; licence_url: string };
  modifications: string;
}
/** One run in this tab: its largest error and its median time per image, encrypt to decrypt, keys excluded. */
export interface SquareTabPoint { set: string; worst: number; medianMs: number; images: number; ranIn: string }
interface Point { name: string; ms: number; err: number }

const tick = { fill: "var(--muted)", fontSize: 11 };
const axisLabel = (value: string) => ({ value, position: "insideBottom" as const, offset: -8, fill: "var(--muted)", fontSize: 11 });
type AxisBox = { x?: number; y?: number; width?: number; height?: number };
// Centred on the axis, as LendingPanels draws its rotated titles (Recharts' "insideLeft" runs a long title off the top).
const yLabel = (value: string) => ({
  value,
  content: ({ viewBox }: { viewBox?: AxisBox }) => {
    const { x = 0, y = 0, height = 0 } = viewBox ?? {};
    const cx = x + 10.5, cy = y + height / 2;
    return <text x={cx} y={cy} transform={`rotate(-90, ${cx}, ${cy})`} textAnchor="middle" dominantBaseline="central"
      fill="var(--muted)" fontSize={11}>{value}</text>;
  },
});
/** A point's name to its right, on one line. */
function pointName({ cx, cy, value }: { cx?: number | string; cy?: number | string; value?: unknown }) {
  const x = Number(cx), y = Number(cy);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return <text x={x + 9} y={y} dominantBaseline="central" fill="var(--text)" fontSize={11}>{String(value ?? "")}</text>;
}
export default function SquareStudentSetsChart({ run }: { run: SquareTabPoint | null }) {
  const [file, setFile] = useState<SetsFile | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const captionId = useId();
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/medical/square_student_sets.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: SetsFile) => { if (live) setFile(j); })
      .catch((e: unknown) => { if (live) setFailed(String(e)); });
    return () => { live = false; };
  }, []);
  if (failed) return <p className="small" style={{ color: "var(--danger)" }}>The parameter sets&apos; record did not load: {failed}.</p>;
  if (!file) return null;
  const names = Object.keys(file.sets).sort();
  const at = (name: string): Point => ({ name, ms: file.sets[name].median_s * 1000, err: file.sets[name].max_abs_logit_error });
  const offered = names.filter((n) => file.sets[n].criterion_3).map(at);
  const refused = names.filter((n) => !file.sets[n].criterion_3).map(at);
  const tab: Point[] = run && Number.isFinite(run.worst) && Number.isFinite(run.medianMs)
    ? [{ name: `this tab, ${run.set}`, ms: run.medianMs, err: run.worst }] : [];
  const all = [...offered, ...refused, ...tab];
  const [lo, hi] = errorDomain(all.map((p) => p.err), file.limit);
  const ticks: number[] = [];
  for (let v = lo; v <= hi * 1.0001; v *= 10) ticks.push(v);
  // Four or five round ticks: a step of 1, 2, 2.5 or 5 times a power of ten, room on the right for a point's name.
  const want = (Math.max(...all.map((p) => p.ms)) * 1.25) / 4;
  const pow = 10 ** Math.floor(Math.log10(want));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * pow >= want) ?? 10) * pow;
  const xMax = Math.ceil((want * 4) / step) * step;
  const xTicks = Array.from({ length: Math.round(xMax / step) + 1 }, (_, i) => +(i * step).toPrecision(6));
  // A plain text beside each point: LabelList's own text wraps at the symbol's width, one word a line.
  const label = <LabelList dataKey="name" content={pointName} />;
  return (
    <figure className="chart-figure" style={{ margin: "10px 0" }}>
      <figcaption id={captionId} className="small muted">
        <b>The parameter sets the pre-registration tried, error against time.</b> From the pre-registered record, on all
        156 test images: the largest distance between a decrypted logit and the plaintext one (log scale) against the
        median time to encrypt, evaluate and decrypt one image, keys excluded, timed with {file.timed_with}.
        The dashed line is the pre-registered limit, {file.limit.toExponential(0)}. {refused.map((p) => p.name).join(", ")},
        {" "}over it on every image, is drawn as a cross and not offered; the circles are the sets this card runs.
        {tab.length ? <> The diamond is this tab&apos;s latest run ({run?.images} images, node-seal in {run?.ranIn}): another
        library on another machine, so how far it sits from the circles along the time axis says nothing about the set.</> : null}
      </figcaption>
      <ChartReadout hint={HINT_POINTER}>
        <MarksHidden>
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart role="img" title="Largest decrypted-logit error against time per image, for each CKKS parameter set, with the 1e-3 limit"
              aria-describedby={captionId} margin={{ top: 8, right: 40, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="ms" name="ms per image" domain={[0, xMax]} ticks={xTicks} tick={tick}
                tickFormatter={(v: number) => v.toFixed(0)} label={axisLabel("ms per image, encrypt to decrypt")} />
              <YAxis type="number" dataKey="err" name="largest logit error" scale="log" domain={[lo, hi]} ticks={ticks}
                allowDataOverflow tick={tick} tickFormatter={(v: number) => v.toExponential(0)} width={56}
                label={yLabel("largest logit error")} />
              <Tooltip content={<ReadoutTooltip />}
                formatter={(v: number, name: string, entry: { payload?: Point }, i: number) =>
                  i === 0 ? [`${v.toFixed(1)} ms`, `${entry.payload?.name ?? ""}: ${name}`] : [v.toExponential(2), name]} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine y={file.limit} stroke="var(--danger)" strokeDasharray="4 3"
                label={{ value: `limit ${file.limit.toExponential(0)}`, fill: "var(--danger)", fontSize: 11, position: "insideBottomRight" }} />
              <Scatter name="offered (circle)" data={offered} fill="var(--accent-2)" shape="circle" isAnimationActive={false}>{label}</Scatter>
              {refused.length > 0 && <Scatter name="not offered (cross)" data={refused} fill="var(--danger)" shape="cross" isAnimationActive={false}>{label}</Scatter>}
              {tab.length > 0 && <Scatter name="this tab (diamond)" data={tab} fill="var(--warn)" shape="diamond" isAnimationActive={false}>{label}</Scatter>}
            </ScatterChart>
          </ResponsiveContainer>
        </MarksHidden>
      </ChartReadout>
      <div style={{ overflowX: "auto" }}>
        <table className="table small" style={{ marginTop: 6 }}>
          <thead><tr><th>Set</th><th>N</th><th>Modulus (bits)</th><th>Scale</th><th>Largest error</th><th>Median error</th><th>ms per image</th><th>Sent, back</th><th>Offered</th></tr></thead>
          <tbody>
            {names.map((n) => {
              const s = file.sets[n];
              return (
                <tr key={n}>
                  <td>{n}</td><td>{s.N.toLocaleString("en-US")}</td><td>{s.bits.join("+")} = {s.modulus_bits} of {s.pq_bound_bits}</td>
                  <td>2^{s.scale_bits}</td><td>{s.max_abs_logit_error.toExponential(2)}</td><td>{s.median_abs_logit_error.toExponential(2)}</td>
                  <td>{(s.median_s * 1000).toFixed(1)}</td><td>{(s.input_bytes / 1e6).toFixed(2)} MB, {(s.output_bytes / 1e3).toFixed(0)} kB</td>
                  <td>{s.criterion_3 ? "yes" : "no: over the limit"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "4px 0 0" }}>
        &quot;Of&quot; is the HomomorphicEncryption.org standard&apos;s 128-bit post-quantum bound on the modulus at that N.
        Computed from {file.attribution.dataset}. {file.attribution.citation} Licence:{" "}
        <a href={file.attribution.licence_url} target="_blank" rel="noreferrer">{file.attribution.licence}</a>. {file.modifications}.
      </p>
    </figure>
  );
}
