// Jinan-1's measured pass against the page's downlink model (Round 86, batch 6; the owner's choices of 2026-11-26).
// The comparison was pre-registered (references/satellite-jinan1-pass-preregistration.md), but its windows did not hold:
// the paper's QBER column is one value per key block, so the windows the card uses were chosen after those figures were
// seen (the pre-registration's addendum 2), and the card says so. Every figure is read from
// web/public/data/satellite/jinan1_pass.json, which scripts/measure_jinan1_pass.py --post-hoc writes beside its record.
import { useEffect, useId, useMemo, useState } from "react";
import { CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { ChartReadout, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import { MarksHidden } from "./VerificationChart";
import StageBadge from "./StageBadge";

interface Fall { measured_minus_model_db: number; within_0_5_db: boolean }
interface Pass {
  post_hoc: boolean; pass: string; windows: string;
  seconds: number[]; range_km: number[]; measured_detections_per_s: number[]; model_detections_per_s: number[];
  measured_qber: number[]; model_qber: number[];
  summary: {
    background_per_s: number; profile_efficiency: number; fitted_efficiency: number; fit_minus_profile_db: number;
    residuals_after: { sd_db: number }; trend_after_db_per_air_mass: number;
    falls: { first: Fall; last: Fall };
    qber: { measured_mean: number; model_signal_mean: number };
    key: { measured_bits: number; model_pass_key_bits: number; model_finite_key_bits: number;
           model_over_measured: number; finite_over_measured: number };
    robustness?: { choices: number; mean_residual_db: [number, number]; largest_fall_difference_db: number };
  };
  preregistration: { file: string }; record: string;
  licence: string; licence_url: string; attribution: string;
}

const tick = { fill: "var(--muted)", fontSize: 11 };
const sign = (v: number, d = 2) => `${v >= 0 ? "+" : "-"}${Math.abs(v).toFixed(d)}`;

export default function SatJinanPassCard() {
  const captionId = useId();
  const [d, setD] = useState<Pass | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let on = true;
    fetch(`${import.meta.env.BASE_URL}data/satellite/jinan1_pass.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j: Pass) => { if (on) setD(j); })
      .catch((e: unknown) => { if (on) setFailed(String(e)); });
    return () => { on = false; };
  }, []);
  const rows = useMemo(() => (d ? d.seconds.map((t, i) => ({
    t, measured: d.measured_detections_per_s[i], model: d.model_detections_per_s[i], km: d.range_km[i],
  })) : []), [d]);
  const title = "Jinan-1's measured pass against this page's downlink";
  if (failed) {
    return <div className="card" id="sat-jinan-pass" style={{ marginTop: 14 }}><h3>{title}</h3>
      <p className="small" style={{ color: "var(--danger)" }}>The pass's figures could not be read ({failed}).</p></div>;
  }
  if (!d || !rows.length) return null;
  const s = d.summary, k = s.key, r = s.robustness;
  const n = rows.length;
  const lo = 10 ** Math.floor(Math.log10(Math.min(...rows.map((x) => Math.min(x.measured, x.model))))),
        hi = 10 ** Math.ceil(Math.log10(Math.max(...rows.map((x) => Math.max(x.measured, x.model)))));
  return (
    <div className="card" id="sat-jinan-pass" style={{ marginTop: 14 }}>
      <h3>{title} <StageBadge kind="offline" label={d.post_hoc ? "measured offline, post hoc" : undefined} /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        The Jinan-1 microsatellite&apos;s pass over Jinan on 25 September 2022, second by second from the paper&apos;s source
        data (its Fig. 4b and 4c), against what the page&apos;s <code>jinan1_2022_jinan</code> downlink profile predicts at
        each second&apos;s slant range. The comparison
        was pre-registered, but its windows did not hold: the paper&apos;s QBER column is one value per key block, recorded
        after the link is up and kept after it ends. So the windows here were chosen after those figures were seen, and the
        card is post hoc: the background, {s.background_per_s.toFixed(0)} detections per second, is the median of the
        file&apos;s first and last 30 seconds, and the {n} link seconds are those whose counts exceed five times it.
      </p>
      <figure className="chart-figure" style={{ margin: "6px 0 0" }}>
        <figcaption id={captionId} className="small muted">
          <b>Detections per second over the pass, measured (dots, less the background) and modelled (line)</b>, on a
          logarithmic axis; the closest range, about {Math.min(...d.range_km).toFixed(0)} km, is near the middle.
        </figcaption>
        <ChartReadout hint={HINT_POINTER}>
          <MarksHidden>
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart accessibilityLayer data={rows} role="img" aria-describedby={captionId}
                title="Detections per second over Jinan-1's 25 September 2022 pass, measured and modelled"
                margin={{ top: 10, right: 20, bottom: 8, left: 4 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" />
                <XAxis dataKey="t" type="number" domain={[0, rows[n - 1].t]} tick={tick} allowDecimals={false}
                  label={{ value: "seconds from the link's start", position: "insideBottom", offset: -6, fill: "var(--muted)", fontSize: 11 }} />
                <YAxis type="number" scale="log" domain={[lo, hi]} allowDataOverflow tick={tick} width={52}
                  tickFormatter={(v: number) => (v >= 1000 ? `${v / 1000}k` : String(v))} />
                <Tooltip content={<ReadoutTooltip labelName="second" labelFormatter={(l) => `second ${l}`}
                  formatter={(v) => `${Math.round(Number(v)).toLocaleString("en-US")} per second`} />} />
                <Legend verticalAlign="bottom" wrapperStyle={{ fontSize: 11, paddingTop: 12 }} />
                {/* Small dots: 353 seconds at the default size merge into a band that hides the line. */}
                <Scatter dataKey="measured" name="measured" fill="var(--accent)" isAnimationActive={false}
                  shape={(p: { cx?: number; cy?: number }) => <circle cx={p.cx} cy={p.cy} r={1.8} fill="var(--accent)" />} />
                <Line dataKey="model" name="the page's model" stroke="var(--warn)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </MarksHidden>
        </ChartReadout>
      </figure>
      <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        <li>The profile&apos;s one calibrated constant, the station efficiency {s.profile_efficiency} (fitted in Round 78 to a
          reading of the preprint&apos;s figure), against the source data&apos;s fit, {s.fitted_efficiency.toFixed(4)}:
          {" "}{sign(s.fit_minus_profile_db)} dB, so the constant stays.</li>
        <li>Per-second residuals after that fit: a deviation of {s.residuals_after.sd_db.toFixed(2)} dB, and a trend of
          {" "}{sign(s.trend_after_db_per_air_mass)} dB per air mass. The fall with elevation, measured against modelled,
          {" "}{sign(s.falls.first.measured_minus_model_db)} dB at the link&apos;s first ten seconds and
          {" "}{sign(s.falls.last.measured_minus_model_db)} dB at its last ten.</li>
        <li>QBER over the key blocks: {(s.qber.measured_mean * 100).toFixed(2)} percent measured, the model&apos;s signal state
          {" "}{(s.qber.model_signal_mean * 100).toFixed(2)} percent.</li>
        <li>Key: the pass made {k.measured_bits.toLocaleString("en-US")} bits. The model&apos;s finite key over the simulated
          pass is {k.finite_over_measured.toFixed(2)} times that, and its asymptotic key {k.model_over_measured.toFixed(2)}
          {" "}times. The model leaves out the paper&apos;s 800-ps gate, which keeps about 80 percent of the signal, and the
          ten percent of detections it samples for the QBER rather than the key; the pre-registered second analysis adds
          both.</li>
        {r ? <li>Robustness, also post hoc: over {r.choices} choices of the background window, the link threshold and the
          model&apos;s own background term, the mean residual stays between {sign(r.mean_residual_db[0])} and
          {" "}{sign(r.mean_residual_db[1])} dB and every fall within {r.largest_fall_difference_db.toFixed(2)} dB.</li> : null}
      </ul>
      <p className="small muted" style={{ marginTop: 6 }}>
        Pre-registration and its addenda: <code>{d.preregistration.file}</code>; record: <code>{d.record}</code>. Data:{" "}
        <a href={d.licence_url} target="_blank" rel="noreferrer">{d.licence}</a>; {d.attribution}
      </p>
    </div>
  );
}
