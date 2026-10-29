import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import { api } from "../lib/api";
import { qkdCfg, securityScoringCfg } from "../lib/config";
import { bb84, e91, FIBER, h2, qsdcLink, type ChannelParams } from "../lib/quantumSim";
import { rangeAtElevationKm } from "../lib/satNetwork";
import { ChartReadout, HINT_KEYBOARD, ReadoutTooltip } from "./ChartReadout";

/**
 * Model curves for the simulated quantum channel - what the page's single-run numbers look like across
 * the control's whole range. Every point comes from the same functions a run calls (`bb84`, `e91`,
 * `qsdcLink`, or the demo API the slider uses), so a curve cannot disagree with a run. They are
 * labelled as model output: no QKD device exists in this project.
 */

const tick = { fill: "var(--muted)", fontSize: 11 };
/** A count for an axis tick: 4.5M, 120k, 950. */
const compactCount = (v: number) =>
  Math.abs(v) >= 1e6 ? `${+(v / 1e6).toFixed(1)}M` : Math.abs(v) >= 1e3 ? `${+(v / 1e3).toFixed(0)}k` : String(v);
const grid = <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />;

/** QBER and secure key bits against the eavesdropper's intercept-resend fraction (the /validations slider). */
export function EveSweepChart({ distanceKm }: { distanceKm: number }) {
  const [rows, setRows] = useState<{ eve: number; qber: number; bits: number }[]>([]);
  useEffect(() => {
    let live = true;
    // The same call the slider makes, at every step it offers - including the server branch in a full
    // deployment - so the curve and the single readout come from one path.
    Promise.all(Array.from({ length: 11 }, (_, i) => i / 10).map(async (eve) => {
      const r = await api.demoQkd(distanceKm, eve);
      return { eve, qber: r.qber, bits: r.secure_key_bits };
    })).then((out) => { if (live) setRows(out); });
    return () => { live = false; };
  }, [distanceKm]);
  const threshold = securityScoringCfg().qber_secure_threshold;
  if (!rows.length) return null;
  return (
    <figure className="chart-figure">
      <figcaption className="small muted">
        <b>Model output (simulated BB84 at {distanceKm} km):</b> the observed QBER crosses the {threshold} abort
        threshold, and the secure key reaches zero, as the intercept-resend fraction grows. Her disturbance
        is composed with the channel's own error, not added to it.
      </figcaption>
      <ChartReadout hint={HINT_KEYBOARD}>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={rows} accessibilityLayer title="QBER and secure key bits against the eavesdropper fraction, simulated BB84" margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
            {grid}
            <XAxis dataKey="eve" tick={tick} label={{ value: "Eve intercept fraction", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
            <YAxis yAxisId="q" tick={tick} domain={[0, 0.3]} label={{ value: "QBER", angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 11 }} />
            {/* Compact ticks (Round 80): "4500000" filled the axis's 60 px, and the rotated title at its
                edge was drawn over the tick labels - seen in the README's screenshot of this page. The
                readout and the tooltip keep the full count. */}
            <YAxis yAxisId="b" orientation="right" tick={tick} tickFormatter={compactCount}
              label={{ value: "secure bits", angle: 90, position: "insideRight", fill: "var(--muted)", fontSize: 11 }} />
            <Tooltip content={<ReadoutTooltip labelName="Eve intercept fraction" />} />
            <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine yAxisId="q" y={threshold} stroke="var(--danger)" strokeDasharray="4 3" label={{ value: `abort ${threshold}`, fill: "var(--danger)", fontSize: 10, position: "insideTopRight" }} />
            <Line yAxisId="q" type="monotone" dataKey="qber" name="QBER (sampled)" stroke="var(--warn)" dot strokeWidth={2} isAnimationActive={false} />
            <Line yAxisId="b" type="monotone" dataKey="bits" name="secure key bits" stroke="var(--accent-2)" dot strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartReadout>
    </figure>
  );
}

/** Fibre-distance curves for a fibre QKD run: BB84 rate with and without decoy states, and E91's CHSH value. */
export function FibreDistanceCharts({ misalignment, eve }: { misalignment: number; eve: number }) {
  const fEC = qkdCfg().error_correction_eff;
  const { rows, sDi, sDd, noDecoyNowhere } = useMemo(() => {
    // A zero rate is drawn as a gap (null), not as a point: a log axis has no zero, and a line dropping
    // to the axis floor would read as a tiny positive rate.
    const out: { km: number; decoy: number | null; nodecoy: number | null; s: number; e91rate: number }[] = [];
    for (let km = 0; km <= 160; km += 10) {
      const ch = { ...FIBER, distanceKm: km, misalignment, eveFraction: eve };
      const b = bb84(1_000_000, ch, 1);
      const e = e91(1_000_000, ch, 1);
      const nd = b.secretKeyRateNonDecoy ?? 0;
      out.push({ km, decoy: b.secretKeyRate > 0 ? b.secretKeyRate : null, nodecoy: nd > 0 ? nd : null,
        s: e.chshS, e91rate: e.secretKeyRate });
    }
    // The two E91 thresholds at the configured error-correction efficiency, as the security facts
    // state them: device-dependent zero key, and the device-independent (Acin et al. 2007) point.
    const chsh = (q: number) => 2 * Math.SQRT2 * (1 - 2 * q);
    const solve = (r: (q: number) => number) => { let lo = 0, hi = (1 - Math.SQRT1_2) / 2;
      for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (r(m) > 0) lo = m; else hi = m; } return lo; };
    const qDd = solve((q) => 1 - (1 + fEC) * h2(q));
    const qDi = solve((q) => 1 - fEC * h2(q) - h2((1 + Math.sqrt(Math.max(0, (chsh(q) / 2) ** 2 - 1))) / 2));
    return { rows: out, sDi: chsh(qDi), sDd: chsh(qDd), noDecoyNowhere: out.every((r) => r.nodecoy == null) };
  }, [misalignment, eve, fEC]);
  return (
    <div className="chart-pair">
      <figure className="chart-figure">
        <figcaption className="small muted">
          <b>BB84 secret-key rate per pulse against fibre length (model).</b> Decoy states against the
          counterfactual without them, on the same channel: the gap is what the decoy states buy.
          {noDecoyNowhere && <> Here the counterfactual is <b>zero at every distance shown</b> - the
            multi-photon fraction exceeds the detection probability - so no second curve appears: the decoy
            states carry the whole link.</>}
        </figcaption>
        <ChartReadout hint={HINT_KEYBOARD}>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={rows} accessibilityLayer title="BB84 secret-key rate per pulse against fibre length, with and without decoy states" margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
              {grid}
              <XAxis dataKey="km" tick={tick} label={{ value: "fibre (km)", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
              <YAxis tick={tick} scale="log" domain={[1e-6, 1e-1]} allowDataOverflow tickFormatter={(v: number) => v.toExponential(0)} />
              <Tooltip content={<ReadoutTooltip labelName="fibre (km)" />} formatter={(v: number) => v.toExponential(3)} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="decoy" name="decoy-state bound" stroke="var(--accent-2)" dot={false} strokeWidth={2} isAnimationActive={false} />
              <Line type="monotone" dataKey="nodecoy" name="without decoy states" stroke="var(--warn)" dot={false} strokeWidth={2} strokeDasharray="5 4" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartReadout>
      </figure>
      <figure className="chart-figure">
        <figcaption className="small muted">
          <b>E91 CHSH value against fibre length (model).</b> S &gt; 2 only certifies entanglement; the key
          rate is already zero below S = {sDd.toFixed(3)}, and a device-independent analysis at the same
          f = {fEC} would need S &gt; {sDi.toFixed(3)}.
        </figcaption>
        <ChartReadout hint={HINT_KEYBOARD}>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={rows} accessibilityLayer title="E91 CHSH value against fibre length, with the S = 2, zero-key and device-independent thresholds" margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
              {grid}
              <XAxis dataKey="km" tick={tick} label={{ value: "fibre (km)", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
              <YAxis tick={tick} domain={[1.8, 2.9]} />
              <Tooltip content={<ReadoutTooltip labelName="fibre (km)" />} />
              <ReferenceLine y={2 * Math.SQRT2} stroke="var(--muted)" strokeDasharray="2 3" label={{ value: "Tsirelson 2.828", fill: "var(--muted)", fontSize: 10, position: "insideTopLeft" }} />
              <ReferenceLine y={sDi} stroke="var(--accent)" strokeDasharray="4 3" label={{ value: `device-independent ${sDi.toFixed(3)}`, fill: "var(--accent)", fontSize: 10, position: "insideTopLeft" }} />
              <ReferenceLine y={sDd} stroke="var(--warn)" strokeDasharray="4 3" label={{ value: `zero key ${sDd.toFixed(3)}`, fill: "var(--warn)", fontSize: 10, position: "insideTopLeft" }} />
              <ReferenceLine y={2} stroke="var(--danger)" strokeDasharray="4 3" label={{ value: "S = 2", fill: "var(--danger)", fontSize: 10, position: "insideBottomLeft" }} />
              <Line type="monotone" dataKey="s" name="CHSH S" stroke="var(--quantum)" dot={false} strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartReadout>
      </figure>
    </div>
  );
}

/** QSDC secrecy capacity against downlink range - the satellite page's headline finding as a curve. */
export function QsdcRangeChart({ channel }: { channel: ChannelParams }) {
  // The RUN's link (Round 77): its profile, budget, minimum elevation and eavesdropper, with only the
  // range swept. It took the atmosphere and pointing as numbers and rebuilt the default link around
  // them, so a non-default profile would have drawn the wrong satellite.
  const fs = channel.freeSpace!;
  const maxKm = rangeAtElevationKm(Number(fs.minElevationDeg ?? 10), Number(fs.orbitAltitudeKm ?? 500),
                                   Number(fs.stationAltitudeKm ?? 0));
  const rows = useMemo(() => {
    const out: { km: number; run: number; clean: number; zero: number }[] = [];
    const kms = [1, 25, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, Math.floor(maxKm)]
      .filter((km, i, a) => km <= maxKm && a.indexOf(km) === i);
    for (const km of kms) {
      const at = (e: number, zeroBudget = false) => qsdcLink({ ...channel, distanceKm: km, eveFraction: e,
        freeSpace: zeroBudget ? { ...fs, atmosphericDb: 0, pointingDb: 0 } : fs }).cs;
      out.push({ km, run: at(channel.eveFraction), clean: at(0), zero: at(0, true) });
    }
    return out;
  }, [channel, fs, maxKm]);
  const eve = channel.eveFraction;
  return (
    <figure className="chart-figure">
      <figcaption className="small muted">
        <b>QSDC secrecy capacity against downlink range (model).</b> Delivery needs C_s &gt; 0. With this run&apos;s link
        ({fs.atmosphericDb} dB of atmosphere at {fs.atmosphericRefKm} km, scaled by air mass, and {fs.pointingDb} dB of
        pointing) it stays negative at every range; only with both set near zero does a short link close, and no LEO
        slant range is short enough. The curve stops at {Math.floor(maxKm)} km, where the satellite sinks below
        {" "}{fs.minElevationDeg} degrees. The axis is zoomed to the zero crossing; values below -6 run off the bottom.
      </figcaption>
      <ChartReadout hint={HINT_KEYBOARD}>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={rows} accessibilityLayer title="QSDC secrecy capacity against downlink range, at this run's link and with zero atmosphere and pointing" margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
            {grid}
            <XAxis dataKey="km" tick={tick} label={{ value: "slant range (km)", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
            {/* Zoomed to the zero crossing, where the finding is: values below -6 run off the bottom. */}
            <YAxis tick={tick} domain={[-6, 0.5]} allowDataOverflow />
            <Tooltip content={<ReadoutTooltip labelName="slant range (km)" />} formatter={(v: number) => v.toFixed(4)} />
            <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine y={0} stroke="var(--danger)" strokeDasharray="4 3" label={{ value: "C_s = 0", fill: "var(--danger)", fontSize: 10, position: "insideTopRight" }} />
            {eve > 0 && <Line type="monotone" dataKey="run" name={`this link, Eve ${eve}`} stroke="var(--warn)" dot={false} strokeWidth={2} isAnimationActive={false} />}
            <Line type="monotone" dataKey="clean" name="this link, no eavesdropper" stroke="var(--accent)" dot={false} strokeWidth={2} isAnimationActive={false} />
            <Line type="monotone" dataKey="zero" name="0 dB atmosphere and pointing, no eavesdropper" stroke="var(--accent-2)" dot={false} strokeWidth={2} strokeDasharray="5 4" isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartReadout>
    </figure>
  );
}
