// All 16 chest X-ray films through the split (Round 86, the owner's choice of 2026-11-23): each film's decrypted logit
// against its logit in the clear, one point a film, with the model's operating point on both axes. The points come from
// the same function the single-film run calls (CxrSplitInference.tsx), so a film's point is its single run.
import { useId } from "react";
import { CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { ChartReadout, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import { logitAxis, type CxrHead, type FilmRun, filmsSummary } from "../lib/cxrSplit";

const tick = { fill: "var(--muted)", fontSize: 11 };
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

export default function CxrFilmsChart({ head, runs }: { head: CxrHead; runs: readonly FilmRun[] }) {
  const captionId = useId();
  const s = filmsSummary(head, runs);
  const point = (r: FilmRun) => ({ film: `film ${r.film + 1}`, plain: r.zPlain, enc: r.zEnc });
  const effusion = runs.filter((r) => r.label === 1).map(point), none = runs.filter((r) => r.label !== 1).map(point);
  const { lo, hi, ticks, decimals } = logitAxis(runs.flatMap((r) => [r.zPlain, r.zEnc]).concat(head.operating_point_logit));
  const t = head.operating_point_logit;
  return (
    <figure className="chart-figure" style={{ margin: "10px 0 0" }}>
      <figcaption id={captionId} className="small muted">
        <b>All {s.films} films through the split: the decrypted logit against the logit in the clear,</b> one point a film,
        each from the same steps as a single film&apos;s run. The dashed lines are the operating point, logit {t.toFixed(3)}{" "}
        (p = {head.decision_threshold.toFixed(4)}): right of the vertical line is &quot;effusion&quot; in the clear, above the
        horizontal one after decryption, so a point in the upper left or lower right would be a decision encryption changed.
        The largest difference is {s.worst.toExponential(2)}, and the decision is the same in {s.sameDecision} of {s.films}.
        {s.recorded ? <> The full int8 model&apos;s recorded decision matches the split&apos;s in {s.agreeRecorded} of {s.recorded}.</> : null}
        {s.labelled ? <> NIH&apos;s label, mined from the reports, agrees with it in {s.agreeLabel} of {s.labelled}: 16 films say nothing about accuracy.</> : null}
      </figcaption>
      <ChartReadout hint={HINT_POINTER}>
        <MarksHidden>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart role="img" title={`Decrypted against plaintext logit for the ${s.films} published chest X-ray films`}
              aria-describedby={captionId} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="plain" name="logit in the clear" domain={[lo, hi]} ticks={ticks} tick={tick}
                tickFormatter={(v: number) => v.toFixed(decimals)}
                label={{ value: "logit in the clear", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
              {/* Both axes start at the same corner; only the x axis labels it, as VerificationChart does. */}
              <YAxis type="number" dataKey="enc" name="decrypted logit" domain={[lo, hi]} ticks={ticks} tick={tick} width={48}
                tickFormatter={(v: number) => (Math.abs(v - lo) < 1e-9 ? "" : v.toFixed(decimals))} label={yLabel("decrypted logit")} />
              <Tooltip content={<ReadoutTooltip />}
                formatter={(v: number, name: string, entry: { payload?: { film?: string } }, i: number) =>
                  [v.toFixed(6), i === 0 ? `${entry.payload?.film ?? ""}: ${name}` : name]} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="var(--muted)" strokeDasharray="2 4" />
              <ReferenceLine x={t} stroke="var(--danger)" strokeDasharray="4 3" />
              <ReferenceLine y={t} stroke="var(--danger)" strokeDasharray="4 3" />
              <Scatter name="no effusion by NIH's label (circle)" data={none} fill="var(--accent-2)" shape="circle" isAnimationActive={false} />
              {effusion.length > 0 && <Scatter name="effusion by NIH's label (triangle)" data={effusion} fill="var(--danger)" shape="triangle" isAnimationActive={false} />}
            </ScatterChart>
          </ResponsiveContainer>
        </MarksHidden>
      </ChartReadout>
    </figure>
  );
}
