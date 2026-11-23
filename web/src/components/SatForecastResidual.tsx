// The satellite forecaster student's residual against time (Round 86, the owner's choice of 2026-11-23). A run scores
// every forecast of the student's published ESA-ADB stretch in the clear and under CKKS (lib/forecastStudent.ts); this
// draws each forecast's largest difference, decrypted minus plaintext, over the six channels, against its time, with
// the forecasts the plaintext flags and any flag encryption changed. 8,640 points, so they are drawn on one canvas
// (CanvasMarks), as the project's other large scatters are.
import { useId } from "react";
import { CartesianGrid, Customized, Legend, ResponsiveContainer, Scatter, ScatterChart, XAxis, YAxis } from "recharts";
import { CanvasMarks, type MarkSeries } from "./CanvasMarks";
import { ChartReadout, HINT_POINTER } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import type { ForecastResidualReport } from "../lib/clientCompute";
import { modelsFor } from "../lib/config";

const BLUE = "#4f9dff", RED = "#ff6b6b", AMBER = "#f6b73c";
const tick = { fill: "var(--muted)", fontSize: 11 };
const NO_POINTS: never[] = [];
const HOUR = 3_600_000;
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

/** The figures the card states, from the report's own arrays: they must equal the run's log line. */
export function residualSummary(r: ForecastResidualReport): { worst: number; median: number; changed: number; flagged: number } {
  const sorted = [...r.residuals].sort((a, b) => a - b);
  const n = sorted.length;
  const median = n ? (n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2) : NaN;
  return { worst: n ? sorted[n - 1] : NaN, median, changed: r.changedRows.length, flagged: r.flaggedRows.length };
}

/** Each forecast's point, in hours from the stretch's start; a residual of exactly 0 sits at the axis floor. */
export function residualSeries(r: ForecastResidualReport, floor: number): MarkSeries[] {
  const at = (t: number) => ({ x: (t * r.stepMs) / HOUR, y: Math.max(r.residuals[t], floor) });
  const flagged = new Set(r.flaggedRows);
  const series: MarkSeries[] = [
    { name: "a forecast (circle)", fill: BLUE, shape: "circle", opacity: 0.35,
      points: r.residuals.map((_, t) => t).filter((t) => !flagged.has(t)).map(at) },
    { name: "flagged in the clear (cross)", fill: RED, shape: "cross", points: r.flaggedRows.map(at) },
  ];
  if (r.changedRows.length) series.push({ name: "flag changed by encryption (diamond)", fill: AMBER, shape: "diamond", points: r.changedRows.map(at) });
  return series;
}

export default function SatForecastResidual({ report }: { report: ForecastResidualReport }) {
  const captionId = useId();
  const spec = modelsFor("satellite_comms").find((m) => m.key === report.key);
  const s = residualSummary(report);
  const positive = report.residuals.filter((v) => v > 0);
  const lo = 10 ** Math.floor(Math.log10(positive.length ? Math.min(...positive) : 1e-12));
  const hi = 10 ** Math.ceil(Math.log10(Math.max(s.worst, lo * 10)));
  const ticks: number[] = [];
  for (let v = lo; v <= hi * 1.0001; v *= 10) ticks.push(v);
  const series = residualSeries(report, lo);
  const hours = ((report.rows - 1) * report.stepMs) / HOUR;
  // Round hours, six steps at most: Recharts' own ticks on [0, 72] were 0, 20, 40 and 72.
  const step = [1, 2, 3, 6, 12, 24, 48, 96, 168].find((h) => hours / h <= 6) ?? 168;
  const xMax = Math.max(step, Math.ceil(hours / step) * step);
  const xTicks = Array.from({ length: xMax / step + 1 }, (_, i) => i * step);
  const start = new Date(report.startMs).toISOString().replace("T", " ").slice(0, 16);
  const d = spec?.dataset;
  return (
    <div className="card" id="sat-forecast-residual" style={{ marginTop: 14 }}>
      <h3>What encryption did to each forecast</h3>
      <figure className="chart-figure" style={{ margin: "6px 0 0" }}>
        <figcaption id={captionId} className="small muted">
          <b>Every forecast&apos;s largest difference, decrypted minus plaintext, over the six channels</b> (standardised
          units, log scale), against its time from the stretch&apos;s start, {start} UTC: {report.rows.toLocaleString("en-US")} forecasts
          of {report.key}, every one under CKKS in this run ({report.ciphertexts} ciphertexts, {(report.fheMs / 1000).toFixed(1)} s{report.ranIn ? ` in ${report.ranIn}` : ""}).
          The largest is {s.worst.toExponential(2)} and the median {s.median.toExponential(2)}. The red crosses are the{" "}
          {s.flagged.toLocaleString("en-US")} forecasts the plaintext forecast flags as anomalous; encryption changed{" "}
          {s.changed === 0 ? "none of the flags" : `${s.changed} of them, the amber diamonds`}.
        </figcaption>
        <ChartReadout hint={HINT_POINTER}>
          <MarksHidden>
            <ResponsiveContainer width="100%" height={240}>
              <ScatterChart role="img" title={`Decrypted minus plaintext forecast, the largest over six channels, against time, ${report.rows} forecasts`}
                aria-describedby={captionId} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" name="hours from the start" domain={[0, xMax]} ticks={xTicks} tick={tick}
                  tickFormatter={(v: number) => v.toFixed(0)}
                  label={{ value: "hours from the stretch's start", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
                <YAxis type="number" dataKey="y" name="largest |decrypted - plaintext|" scale="log" domain={[lo, hi]} ticks={ticks}
                  allowDataOverflow tick={tick} tickFormatter={(v: number) => v.toExponential(0)} width={56}
                  label={yLabel("|decrypted - plaintext|")} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                <Scatter name="a forecast (circle)" data={NO_POINTS} fill={BLUE} shape="circle" isAnimationActive={false} />
                <Scatter name="flagged in the clear (cross)" data={NO_POINTS} fill={RED} shape="cross" isAnimationActive={false} />
                {report.changedRows.length > 0 && (
                  <Scatter name="flag changed by encryption (diamond)" data={NO_POINTS} fill={AMBER} shape="diamond" isAnimationActive={false} />
                )}
                <Customized component={<CanvasMarks series={series} xName="hours" yName="|decrypted - plaintext|"
                  format={(v: number) => (v < 0.01 ? v.toExponential(2) : v.toFixed(2))} />} />
              </ScatterChart>
            </ResponsiveContainer>
          </MarksHidden>
        </ChartReadout>
      </figure>
      {d ? (
        <p className="muted small" style={{ margin: "6px 0 0" }}>
          The forecasts are this project&apos;s, from the student&apos;s published stretch of {d.name}; no telemetry value is
          drawn. {d.attribution}{" "}
          <a href={d.url} target="_blank" rel="noreferrer">Dataset</a>, licence{" "}
          <a href="https://creativecommons.org/licenses/by/3.0/igo/" target="_blank" rel="noreferrer">CC BY 3.0 IGO</a>.
          Changes: windows of the six channels, standardised, forecast by the student, and the forecasts&apos; differences.
        </p>
      ) : null}
    </div>
  );
}
