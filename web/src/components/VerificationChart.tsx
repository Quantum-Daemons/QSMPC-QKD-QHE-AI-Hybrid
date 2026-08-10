// What the in-run verification actually measured, drawn rather than summarised.
//
// The run already re-scores every published model on its held-out sample and reports a single
// number. A single number is the right thing to gate on and the wrong thing to reason from: at a
// 1-in-981 positive rate, "minority-class F1 0.51" does not tell you whether the model separates the
// classes or whether the threshold happens to sit somewhere lucky. These charts answer that, from
// the same arrays the metric was computed on - nothing here is recomputed or re-fetched.
//
// Chosen per task rather than one chart for everything:
//
//   binary        precision-recall curve + score histogram split by TRUE class. PR rather than ROC
//                 because the PR baseline moves with prevalence and ROC does not, which is the whole
//                 reason this project reports AUPRC (Saito & Rehmsmeier, PLOS ONE 2015).
//   multi-class   per-class F1 bars. The aggregate is precisely what hides a broken class - Engelen
//                 et al. (WTMC 2021) measured accuracy flat at 0.99 while a per-class F1 moved
//                 0.60 -> 0.99.
//
// recharts is already a dependency (BenchmarkPanel), so this costs no bundle growth.
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import type { ModelEvalResult } from "../lib/onnxInference";

const AXIS = { fontSize: 10, stroke: "#8a94a6" };
const GRID = "#2a3040";

export function VerificationChart({ res, modelKey }: { res: ModelEvalResult; modelKey: string }) {
  const plot = res.plot;
  if (!plot) return null;

  if (plot.kind === "per-class-f1" && plot.perClass?.length) {
    const worst = plot.perClass.reduce((a, b) => (b.f1 < a.f1 ? b : a));
    return (
      <figure className="chart-figure">
        <figcaption className="muted small">
          Per-class F1 for <code>{modelKey}</code>, {res.n} held-out rows scored in this tab.
          Weakest class: <strong>{worst.label}</strong> at {worst.f1.toFixed(3)} against an aggregate
          of {res.score?.toFixed(3)}. The aggregate is what would hide it.
        </figcaption>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={plot.perClass} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis dataKey="label" tick={AXIS} interval={0} angle={-25} textAnchor="end" height={54} />
            <YAxis domain={[0, 1]} tick={AXIS} />
            <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                     formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="f1" fill="#4f9dff" name="F1" />
          </BarChart>
        </ResponsiveContainer>
      </figure>
    );
  }

  if (plot.kind === "pr-curve" && plot.pr?.length) {
    // The precision a coin-flip achieves is the prevalence, and at 1-in-981 that is 0.001. Drawing
    // it is what makes a PR curve legible - without the line, 0.5 precision looks unimpressive.
    const prevalence = plot.hist
      ? plot.hist.reduce((a, b) => a + b.positive, 0) /
        Math.max(1, plot.hist.reduce((a, b) => a + b.positive + b.negative, 0))
      : 0;
    return (
      <div className="chart-pair">
        <figure className="chart-figure">
          <figcaption className="muted small">
            Precision-recall for <code>{modelKey}</code> over {res.n} rows. The dashed line is what a
            random ranker achieves ({prevalence.toFixed(4)}) - PR rather than ROC because this
            baseline moves with prevalence and ROC's does not.
          </figcaption>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={plot.pr} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
              <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={AXIS}
                     tickFormatter={(v: number) => v.toFixed(1)} />
              <YAxis domain={[0, 1]} tick={AXIS} />
              <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                       formatter={(v: number) => v.toFixed(4)}
                       labelFormatter={(v: number) => `recall ${Number(v).toFixed(3)}`} />
              <ReferenceLine y={prevalence} stroke="#c05621" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="precision" stroke="#4f9dff" dot={false} strokeWidth={1.8}
                    isAnimationActive={false} name="precision" />
            </LineChart>
          </ResponsiveContainer>
        </figure>

        {plot.hist?.length ? (
          <figure className="chart-figure">
            <figcaption className="muted small">
              Score distribution by TRUE class. Overlap here is the reason the curve above looks the
              way it does - a metric says how well, this says why.
            </figcaption>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={plot.hist} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                <XAxis dataKey="bin" tick={AXIS} label={{ value: "score bin", position: "insideBottom",
                       offset: -2, fill: "#8a94a6", fontSize: 10 }} />
                <YAxis scale="log" domain={[0.7, "auto"]} allowDataOverflow tick={AXIS} />
                <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="negative" stackId="a" fill="#4a5568" name="true negative" />
                <Bar dataKey="positive" stackId="a" fill="#c05621" name="true positive" />
              </BarChart>
            </ResponsiveContainer>
          </figure>
        ) : null}
      </div>
    );
  }

  return null;
}
