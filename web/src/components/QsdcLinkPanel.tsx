// The satellite page's telemetry path (Round 86): two-way QSDC against one-way quasi-QSDC, on this page's link.
//
// The curves come from the channel the page's parameters describe. The measured points are the sources' own and are a
// communication rate over the clock, so they sit below the secrecy and reliability bounds for their own reasons.
import { useId, useMemo } from "react";
import { CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { qkdCfg } from "../lib/config";
import type { ChannelParams } from "../lib/quantumSim";
import { qsdcCurves, type QsdcCurveRow } from "../lib/stike";
import { ChartReadout, HINT_KEYBOARD, ReadoutTooltip } from "./ChartReadout";
import KeyPoints from "./KeyPoints";
import { MarksHidden } from "./VerificationChart";

const tick = { fill: "var(--muted)", fontSize: 11 };
const Y_TICKS = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1];
const exponent = (v: unknown, digits = 2) => (typeof v === "number" ? v.toExponential(digits) : "—");
const whole = (v: unknown) => (typeof v === "number" ? v.toLocaleString("en-US") : "—");
// Past the last positive bound there is nothing to read; the reliable rate runs on for a while, so a little of it stays.
const xMaxOf = (lastPositiveDb: number) => Math.ceil((lastPositiveDb + 5) / 5) * 5;
type Published = { label: string; loss_db: number; bit_per_pulse: number };

export default function QsdcLinkPanel({ channel, metrics }: { channel: ChannelParams; metrics: Record<string, any> }) {
  const titleId = useId();
  const published = qkdCfg().qsdc_published_points as Published[];
  const curves = useMemo(() => qsdcCurves(channel, undefined, published.map((p) => p.loss_db)), [channel, published]);
  const rows = useMemo<QsdcCurveRow[]>(() => curves.rows.map((r) => {
    const p = published.find((q) => q.loss_db === r.lossDb);
    return p ? { ...r, measured: p.bit_per_pulse, source: p.label } : r;
  }), [curves, published]);
  const xTicks = Array.from({ length: Math.floor((xMaxOf(curves.lastPositiveDb) - 10) / 10) + 1 }, (_, i) => 10 + 10 * i);
  const xMax = xMaxOf(curves.lastPositiveDb);
  const status = String(metrics.stike_status ?? "");
  return (
    <section id="qsdc-link" aria-labelledby={titleId} style={{ marginTop: 16 }}>
      <h4 id={titleId} style={{ margin: "0 0 4px" }}>Two-way QSDC and one-way quasi-QSDC on this link</h4>
      <p className="small" style={{ marginTop: 0 }}>
        Two-way QSDC (DL04) uses no key, so a lost pulse is one an eavesdropper may have kept, and Eq. 8 charges for it:
        {curves.twoWayClosesToDb != null ? ` its bound stays positive only up to ${curves.twoWayClosesToDb.toFixed(1)} dB here.` : " its bound is positive nowhere on this link."}
        {" "}One-way quasi-QSDC (STIKE) pads the message from a pre-shared key, so loss costs it rate rather than secrecy, until
        dark counts raise the error rate past what Eq. 3 allows.
        {status === "delivered" ? " In this run two-way could not close and quasi-QSDC carried the telemetry." : ""}
      </p>
      <figure className="chart-figure">
        <figcaption className="small muted">
          <b>Bits per pulse against the loss from transmitter to detector output.</b> The lines are bounds per signal pulse, on this link&apos;s source and
          error floor. The points are measured communication rates over the clock, for scale: fibre, and 1.4 km of urban air plus 10 km of fibre, so none of them is a satellite downlink.
          Each point&apos;s loss is its own channel, its receiver&apos;s optics and its detector.
        </figcaption>
        <ChartReadout hint={HINT_KEYBOARD}>
          <MarksHidden>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={rows} accessibilityLayer title="Rate against loss for two-way QSDC and one-way quasi-QSDC, with published measurements" margin={{ top: 8, right: 16, bottom: 22, left: 12 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="lossDb" type="number" domain={[10, xMax]} ticks={xTicks} tick={tick}
                  label={{ value: "loss, transmitter to detector output (dB)", position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 11 }} />
                <YAxis type="number" scale="log" domain={[1e-9, 1e-1]} ticks={Y_TICKS} allowDataOverflow tick={tick} tickFormatter={(v: number) => v.toExponential(0)} width={56}
                  label={{ value: "bit per pulse", angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 11 }} />
                <Tooltip content={<ReadoutTooltip labelName="loss (dB)" />}
                  formatter={(v: number, name: string, entry: { dataKey?: unknown; payload?: QsdcCurveRow }) => [v.toExponential(2), entry.dataKey === "measured" ? entry.payload?.source ?? name : name]} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                {Number.isFinite(curves.linkLossDb) && curves.linkLossDb <= xMax && (
                  <ReferenceLine x={curves.linkLossDb} stroke="var(--danger)" strokeDasharray="4 3"
                    label={{ value: "this link", fill: "var(--danger)", fontSize: 10, position: "insideTopRight" }} />
                )}
                <Line type="linear" dataKey="oneWayReliable" name="one-way: reliable rate (Eq. 4)" stroke="var(--muted)" dot={false} strokeWidth={1.2} isAnimationActive={false} connectNulls />
                <Line type="linear" dataKey="oneWaySecrecy" name="one-way: key rate (Eq. 3)" stroke="var(--accent)" dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls />
                <Line type="linear" dataKey="twoWay" name="two-way: Qi et al. Eq. 8" stroke="var(--accent-2)" dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls />
                <Scatter dataKey="measured" name="measured, one-way" fill="var(--text)" isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </MarksHidden>
        </ChartReadout>
      </figure>
      {status && (
        <table className="mini-table">
          <thead><tr><th scope="col">this run</th><th scope="col">value</th></tr></thead>
          <tbody>
            <tr><th scope="row">outcome of quasi-QSDC</th><td>{status.replaceAll("_", " ")}</td></tr>
            <tr><th scope="row">QBER</th><td className="num">{typeof metrics.stike_qber === "number" ? `${(100 * metrics.stike_qber).toFixed(2)}%` : "—"}</td></tr>
            <tr><th scope="row">Eq. 3 bound, bit per signal pulse</th><td className="num">{exponent(metrics.stike_secrecy_per_signal)}</td></tr>
            <tr><th scope="row">signal pulses for the telemetry</th><td className="num">{whole(metrics.stike_signal_pulses)}</td></tr>
            <tr><th scope="row">time at this source rate (s)</th><td className="num">{typeof metrics.stike_seconds === "number" ? metrics.stike_seconds.toFixed(2) : "—"}</td></tr>
            <tr><th scope="row">key reserved for the frame (bits)</th><td className="num">{whole(metrics.stike_key_in_flight_bits)}</td></tr>
            <tr><th scope="row">key spent on retained pulses (bits)</th><td className="num">{whole(metrics.stike_key_consumed_bits)}</td></tr>
            <tr><th scope="row">key regenerated, asymptotic (bits)</th><td className="num">{whole(metrics.stike_key_regenerated_bits)}</td></tr>
          </tbody>
        </table>
      )}
      <KeyPoints title="What this rests on" items={[
        <><b>Quasi, not keyless.</b> STIKE pads the message with a key both ends already hold, assumed here to be provisioned before the pass. It is not the direct communication Eq. 8 bounds.</>,
        <><b>Only the downlink is modelled.</b> Two-way QSDC would also need a return path, which adds loss, so its curve is the optimistic one.</>,
        <><b>Asymptotic.</b> Pan et al. derive their capacity in the asymptotic scenario and leave finite-length effects open. This model spreads and takes a majority where they use an LDPC code, so it spends more key per message bit.</>,
      ]} />
    </section>
  );
}
