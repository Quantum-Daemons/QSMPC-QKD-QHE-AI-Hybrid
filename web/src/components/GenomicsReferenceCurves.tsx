// Each 1000 Genomes super-population's score distribution beside the person's score (Round 86, batch 6; the owner's choice of
// 2026-11-26). Since batch 6 the genomics page judges a person against the 90th percentile of their own super-population,
// from aggregates over all 2,504 individuals the payload carries, and keeps the pooled cut beside it as the cautionary
// one (references/genomics-real-data-2026-11.md). This draws why: each curve is a super-population's share of individuals
// at or below a score, from its shipped 1st to 99th percentiles, so the same pooled cut sits at very different heights on
// different curves. The shares above the pooled cut are the Models card's own measurement (`by_super_population`).
import { useEffect, useId, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartReadout, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import { modelsFor } from "../lib/config";
import { prsPanel, type PrsPanel } from "../lib/realData";

type Metrics = Record<string, unknown>;
const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const ORDER = ["AFR", "AMR", "EAS", "EUR", "SAS"] as const;
const COLOURS: Record<string, string> = {
  AFR: "var(--accent)", AMR: "var(--accent-2)", EAS: "var(--warn)", EUR: "var(--quantum)", SAS: "var(--muted)",
};
const tick = { fill: "var(--muted)", fontSize: 11 };
/** A score to the precision its scale needs: PGS000783's sums of beta/SE run in the thousands, PGS000072's log odds near 20. */
const fmt = (v: number) => v.toFixed(Math.abs(v) >= 100 ? 1 : 3);

/** One row a percentile: each super-population's score at it, the shape a Recharts line per population reads. */
export function curveRows(panel: Pick<PrsPanel, "reference" | "referencePercentilesAt">): Record<string, number>[] {
  return panel.referencePercentilesAt.map((p, i) => {
    const row: Record<string, number> = { share: p / 100 };
    for (const g of ORDER) if (panel.reference[g]) row[g] = panel.reference[g].percentiles[i];
    return row;
  });
}

export default function GenomicsReferenceCurves({ metrics }: { metrics: Metrics }) {
  const captionId = useId();
  const [panel, setPanel] = useState<PrsPanel | null>(null);
  useEffect(() => {
    let live = true;
    prsPanel().then((p) => { if (live) setPanel(p); }).catch(() => { if (live) setPanel(null); });
    return () => { live = false; };
  }, []);
  const card = modelsFor("genomics").find((s) => s.key === "genomics.prs")?.achieved as
    { measured?: { by_super_population?: Record<string, { frac_above_global_cut?: number }> } } | undefined;
  const above = card?.measured?.by_super_population ?? {};
  const score = num(metrics.polygenic_risk_score), own = num(metrics.prs_high_risk_cut);
  const pooled = num(metrics.prs_pooled_cut), pooledAll = num(metrics.prs_pooled_cut_all);
  const group = typeof metrics.prs_reference_super_population === "string" ? metrics.prs_reference_super_population : null;
  const rows = useMemo(() => (panel ? curveRows(panel) : []), [panel]);
  if (!panel || !rows.length || score == null || own == null || !group) return null;
  const xs = rows.flatMap((r) => ORDER.map((g) => r[g]).filter((v): v is number => typeof v === "number"));
  const lo = Math.floor(Math.min(...xs, score) - 0.5), hi = Math.ceil(Math.max(...xs, score) + 0.5);
  const shares = ORDER.filter((g) => typeof above[g]?.frac_above_global_cut === "number")
    .map((g) => `${g} ${((above[g].frac_above_global_cut as number) * 100).toFixed(1)}%`).join(", ");
  return (
    <div className="card" id="genomics-reference-curves" style={{ marginTop: 14 }}>
      <h3>The person against each super-population</h3>
      <figure className="chart-figure" style={{ margin: "6px 0 0" }}>
        <figcaption id={captionId} className="small muted">
          <b>Each curve is one 1000 Genomes super-population&apos;s share of individuals at or below a score</b>, from its 1st
          to 99th percentiles over all {panel.pooledAll.n.toLocaleString("en-US")} individuals, which the page ships as
          aggregates. The person ({group}, the population they were sampled in) scored {fmt(score)} on {panel.scoreId}; the solid
          line is that population&apos;s 90th percentile, {fmt(own)}, which decides the flag; the red line marks the person.
          {pooledAll != null ? <> The dashed line is the pooled 90th percentile over every super-population, {fmt(pooledAll)}:
            the Models card measures that it is exceeded by {shares || "very different shares of each population"}, which is why
            the page does not use it.</> : null}
          {pooled != null ? <> The dotted line is the pooled cut over the 30 bundled individuals, {fmt(pooled)}, which the
            run reports beside the decision.</> : null}
        </figcaption>
        <ChartReadout hint={HINT_POINTER}>
          <MarksHidden>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart accessibilityLayer role="img" title="Share of each super-population at or below a polygenic score" aria-describedby={captionId}
                margin={{ top: 12, right: 24, bottom: 8, left: 4 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" />
                <XAxis type="number" dataKey="x" domain={[lo, hi]} tick={tick} allowDataOverflow
                  label={{ value: `${panel.scoreId} score`, position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
                <YAxis type="number" domain={[0, 1]} tick={tick} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} width={44} />
                <Tooltip content={<ReadoutTooltip labelName="score" labelFormatter={(l) => `score ${fmt(Number(l))}`}
                  formatter={(v) => `${Math.round(Number(v) * 100)}% of the population at or below`} />} />
                <Legend verticalAlign="bottom" wrapperStyle={{ fontSize: 11, paddingTop: 14 }} />
                {ORDER.filter((g) => panel.reference[g]).map((g) => (
                  <Line key={g} data={rows.map((r) => ({ x: r[g], share: r.share }))} dataKey="share" name={g} type="monotone"
                    xAxisId={0} stroke={COLOURS[g]} strokeWidth={g === group ? 2.6 : 1.4} dot={false} isAnimationActive={false} />
                ))}
                <ReferenceLine y={0.9} stroke="var(--muted)" strokeDasharray="1 3" />
                <ReferenceLine x={own} stroke={COLOURS[group] ?? "var(--text)"} strokeWidth={1.6} />
                {pooledAll != null ? <ReferenceLine x={pooledAll} stroke="var(--text)" strokeDasharray="6 4" /> : null}
                {pooled != null ? <ReferenceLine x={pooled} stroke="var(--text)" strokeDasharray="1 4" /> : null}
                {/* The label goes on the side of the line with room: a score in the left half of the axis takes it to its right. */}
                <ReferenceLine x={score} stroke="var(--danger)" strokeWidth={2.2}
                  label={{ value: "this person", position: score < (lo + hi) / 2 ? "insideTopLeft" : "insideTopRight", fill: "var(--text)", fontSize: 11 }} />
              </LineChart>
            </ResponsiveContainer>
          </MarksHidden>
        </ChartReadout>
      </figure>
    </div>
  );
}
