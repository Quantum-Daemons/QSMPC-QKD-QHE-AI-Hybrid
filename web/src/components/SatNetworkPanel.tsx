import { useMemo, useState } from "react";
import {
  CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, Tooltip,
  XAxis, YAxis,
} from "recharts";

import { linkProfiles, profileValues, qkdCfg, validationSet } from "../lib/config";
import { downlinkFromProfile } from "../lib/quantumSim";
import { randomWords } from "../lib/csprng";
import {
  blockCounts, blocksForPositiveKey, DEVIATION_LABELS, finiteKeySourceOf, keyLength, type Deviation, type FiniteKeySource,
} from "../lib/finiteKey";
import { satChannel } from "../lib/satLink";
import {
  KHMELEV_2024_SESSION, LIAO_2017_PASS, LIAO_2018_STATIONS, groundDistanceKm, islFromProfile, khmelevPass, liaoPass,
  rangeAtElevationKm, sharedViewMaxKm, siftedRateAt, simulatePass, stringOfPearls, trustedRelay, xorRelay, type IslSpec,
} from "../lib/satNetwork";
import { ChartReadout, HINT_KEYBOARD, ReadoutTooltip } from "./ChartReadout";
import { SatNetwork3D } from "./Lazy3D";
import Details from "./Details";

/**
 * The satellite card's Network tab (Round 77): what bridges the horizon a single downlink cannot.
 * Three views of one model - a pass over one station, a trusted satellite relay between two, and a
 * plane of satellites linked to their neighbours - each computed from the Downlink tab's own channel
 * and bound (`satNetwork.ts`), at the run's link profile and controls. The comparison rows are the
 * published figures, beside the model's, counted the same way.
 */

const tick = { fill: "var(--muted)", fontSize: 11 };
const fmt = (x: number, d = 0) => (Number.isFinite(x) ? x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }) : "-");
const kbit = (bits: number) => (Number.isFinite(bits) ? `${fmt(bits / 1000, 1)} kbit` : "-");
const pct = (model: number, ref: number) => `${model >= ref ? "+" : ""}${fmt((100 * (model - ref)) / ref, 1)}%`;
const ratio = (model: number, ref: number) => fmt(model / ref, 2);
const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
/** A block's composable finite key on one key basis and on both (Round 84, the user's choice): the
 *  bound's own protocol keys the Z basis and spends X on the phase error; both bases key each basis
 *  with its phase error from the other's counts, each at half of eps_sec (`finiteKey.ts::keyLength`). */
function twoKeys(segments: ReturnType<typeof simulatePass>["segments"], src: FiniteKeySource) {
  if (!segments.length) return { one: -Infinity, both: -Infinity };
  const [z, x] = blockCounts(segments, src);
  return { one: keyLength(z, x, { ...src, keyBasis: "one" }).keyBits,
           both: keyLength(z, x, { ...src, keyBasis: "both" }).keyBits };
}
const pos = (bits: number) => Math.max(0, bits);
const randomBytes = (n: number) => new Uint8Array(randomWords(Math.ceil(n / 4)).buffer).slice(0, n);

export default function SatNetworkPanel({ params }: { params: Record<string, any> }) {
  // The run's own link: its profile and every sat_* control, with the eavesdropper removed - the
  // network is planned on a clean channel.
  const link = useMemo(() => satChannel({ ...params, sat_eve: 0 }), [params]);
  const fs = link.freeSpace!;
  const orbitKm = Number(fs.orbitAltitudeKm), minEl = Number(fs.minElevationDeg);
  const liao = useMemo(() => liaoPass(link), [link]);
  // The satellite's statistics for every finite key on this tab, the relay's included: it read the
  // fibre's `fk_deviation` until Round 84, so the relay's bound named a different statistics from the
  // pass section's on one tab.
  const deviation = String(params.sat_fk_deviation ?? qkdCfg().free_space_finite_key_deviation) as Deviation;

  return (
    <div>
      <p className="small muted" style={{ marginTop: 0 }}>
        A downlink needs the satellite in view: above {minEl} degrees, which at a {orbitKm} km orbit is a
        slant range of {fmt(rangeAtElevationKm(minEl, orbitKm, Number(fs.stationAltitudeKm)))} km and a
        footprint {fmt(sharedViewMaxKm(link))} km across. Past that, two published architectures carry a
        key further. Every figure here is the Downlink tab&apos;s own model and bound, integrated over the
        geometry, for the profile and controls of this page&apos;s run. Every rate is per pulse sent: the
        asymptotic ones count the key the signal pulses carry - {fmt(100 * Number(link.pSignal ?? 0.5))}% of
        them on this profile - which is how a measured key is made, and the composable finite keys follow
        their bound&apos;s protocol, whose key draws on the detections at every intensity. Simulated: no
        satellite is involved.
      </p>
      <PassSection link={link} liao={liao} minEl={minEl}
        epsSec={Number(params.sat_eps_sec ?? qkdCfg().free_space_eps_sec)} deviation={deviation} />
      <NightsSection link={link} />
      <RelaySection link={link} defaultElevation={liao.maxElevationDeg} minEl={minEl}
        epsSec={Number(params.sat_eps_sec ?? qkdCfg().free_space_eps_sec)} deviation={deviation} />
      <ConstellationSection orbitKm={orbitKm} minEl={minEl} />
    </div>
  );
}

function PassSection({ link, liao, minEl, epsSec, deviation }: { link: ReturnType<typeof satChannel>;
                                                                 liao: ReturnType<typeof liaoPass>; minEl: number;
                                                                 epsSec: number; deviation: Deviation }) {
  const [top, setTop] = useState(Math.round(liao.maxElevationDeg * 10) / 10);
  const pass = useMemo(() => simulatePass(link, top, { stepS: 1 }), [link, top]);
  // THE PASS'S COMPOSABLE FINITE KEY (Round 79): the bound the run's source supports - Lim et al.
  // 2014 for Micius's vacuum + weak decoy - on the counts summed over the pass (`finiteKey.ts`).
  const fkSrc = useMemo(() => finiteKeySourceOf(link, epsSec, undefined, deviation), [link, epsSec, deviation]);
  const fin = useMemo(() => {
    if (!pass.segments.length) return null;
    const [z, x] = blockCounts(pass.segments, fkSrc);
    return { key: keyLength(z, x, fkSrc).keyBits, both: twoKeys(pass.segments, fkSrc).both,
             passes: blocksForPositiveKey(z, x, fkSrc) };
  }, [pass, fkSrc]);
  // Liao's rows at Liao's own failure probability, whatever the run sets, on one key basis and on both.
  const liaoFinite = useMemo(() => twoKeys(liao.segments, finiteKeySourceOf(link, 1e-9, undefined, deviation)),
    [link, liao, deviation]);
  // KHMELEV ET AL.'S SESSION (Round 84) on that link's own profile, whatever this run's: a second flown
  // pass the model is set beside, at their failure probability and this tab's statistics.
  const khmelev = useMemo(() => {
    const ch = satChannel({ sat_profile: KHMELEV_2024_SESSION.profile, sat_eve: 0 });
    const p = khmelevPass(ch);
    return { durationS: p.durationS,
             ...twoKeys(p.segments, finiteKeySourceOf(ch, KHMELEV_2024_SESSION.epsSec, undefined, deviation)) };
  }, [deviation]);
  // The step from the asymptotic row to the one-basis row, and how much of it the basis convention is.
  // The finite key also counts the decoy and vacuum detections, which the asymptotic row leaves out, so
  // on a link with large counts the both-bases key can reach the asymptotic row (Jinan-1's does).
  const gap = liaoFinite.one > 0 && liaoFinite.both > liaoFinite.one && liao.keyBits > liaoFinite.one
    ? { bothOverAsym: liaoFinite.both / liao.keyBits,
        basisShare: Math.min(1, (liaoFinite.both - liaoFinite.one) / (liao.keyBits - liaoFinite.one)) }
    : null;
  // The bound and its statistics (Round 82), so the key says which analysis it is.
  // Its two halves are named apart because one sentence below needs the bound as a noun: joined, they read
  // "by Lim et al. 2014; the tighter of Chernoff and Hoeffding per count's bound" (found live, Round 82 walk).
  const boundName = fkSrc.vacuum ? "Lim et al. 2014" : "Rusca et al. 2018";
  const statsLabel = DEVIATION_LABELS[fkSrc.deviation];
  const bound = `${boundName}; ${statsLabel}`;
  const rows = useMemo(() => pass.samples.filter((_, i) => i % 4 === 0).map((s) => ({
    t: Math.round(s.t), elevation: Math.round(s.elevationDeg * 10) / 10, rate: Math.round(s.keyRate),
  })), [pass]);
  // Time ticks at round steps (Round 80, seen in the README's screenshot). The axis was a category axis
  // over the samples, which are not evenly spaced in time, so its labels read -214, -198 ... -102,
  // -90 ... -6, 1, 9: steps of 16, then 12, then 8. A numeric axis with ticks at a round step instead.
  const timeTicks = useMemo(() => {
    if (rows.length < 2) return undefined;
    const lo = rows[0].t, hi = rows[rows.length - 1].t;
    const step = [30, 60, 100, 120, 200, 300].find((k) => (hi - lo) / k <= 10) ?? 600;
    const out: number[] = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
    return out;
  }, [rows]);
  return (
    <section className="net-section" aria-label="One pass over one station">
      <h4 style={{ margin: "4px 0" }}>One pass over one station</h4>
      <div className="net-controls">
        <label>Highest elevation of the pass (deg)
          <input type="number" min={minEl} max={90} step={1} value={top}
            onChange={(e) => setTop(Math.min(90, Math.max(minEl, Number(e.target.value) || minEl)))} />
        </label>
        <button className="btn ghost" style={{ fontSize: 12 }} onClick={() => setTop(Math.round(liao.maxElevationDeg * 10) / 10)}>
          Liao&apos;s 19 Dec 2016 pass ({fmt(liao.maxElevationDeg, 1)} deg in this model; the paper&apos;s table: 47.79)
        </button>
      </div>
      <p className="small" style={{ margin: "4px 0" }}>
        {pass.durationS > 0
          ? <>Above {minEl} degrees this pass lasts <b>{fmt(pass.durationS)} s</b>, from {fmt(pass.maxRangeKm)} km
            in to {fmt(pass.minRangeKm)} km at the top, and yields <b>{kbit(pass.keyBits)}</b> of asymptotic key.{" "}
            {fin && fin.key > 0
              ? <>As one block its composable finite key is <b>{kbit(fin.key)}</b> ({bound}, eps_sec {epsSec.toExponential(0)}),
                keyed on the Z basis as the run keys it; on both bases, <b>{kbit(fin.both)}</b>.</>
              : <>As one block it carries <b>no composable key</b> ({bound}, eps_sec {epsSec.toExponential(0)}):{" "}
                {fin?.passes ? <>{fin.passes} such passes pooled would.</> : <>not even 10,000 such passes pooled would.</>}</>}</>
          : <>A pass that never rises above {minEl} degrees makes no key.</>}
      </p>
      {pass.durationS > 0 && (
        <figure className="chart-figure">
          <figcaption className="small muted">
            <b>Elevation and key rate through the pass (model).</b> The rate is the Downlink tab&apos;s bound at each
            range; it falls as the satellite sinks, because both the range and the air the beam crosses grow. The
            dashed line is the minimum elevation.
          </figcaption>
          <ChartReadout hint={HINT_KEYBOARD}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rows} accessibilityLayer title="Elevation and key rate against time through one pass" margin={{ top: 8, right: 8, bottom: 18, left: 4 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} ticks={timeTicks} tick={tick}
                  label={{ value: "seconds from the top of the pass", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
                <YAxis yAxisId="el" tick={tick} domain={[0, 90]} />
                <YAxis yAxisId="rate" orientation="right" tick={tick} />
                <Tooltip content={<ReadoutTooltip labelName="t (s)" />} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                <ReferenceLine yAxisId="el" y={minEl} stroke="var(--danger)" strokeDasharray="4 3" />
                <Line yAxisId="el" type="monotone" dataKey="elevation" name="elevation (deg)" stroke="var(--accent-2)" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line yAxisId="rate" type="monotone" dataKey="rate" name="key rate (bit/s)" stroke="var(--accent)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartReadout>
        </figure>
      )}
      <div className="net-table">
        <table>
          <caption className="small muted" style={{ textAlign: "left", captionSide: "top", paddingBottom: 4 }}>
            Liao et al., Nature 549, 43 (2017): the pass of 19 December 2016, counted the way the paper counts it -
            the window from {LIAO_2017_PASS.maxRangeKm} km down to {LIAO_2017_PASS.minRangeKm} km and back, states
            sent 50 / 25 / 25, and sifted key from signal states only. The finite rows are {bound}, at 1e-9, on
            one key basis and on both.
          </caption>
          <thead><tr><th></th><th className="num">this model</th><th className="num">Liao 2017</th><th className="num">difference</th></tr></thead>
          <tbody>
            <tr><td>Data window</td><td className="num">{fmt(liao.durationS, 1)} s</td><td className="num">{LIAO_2017_PASS.durationS} s</td><td className="num">{pct(liao.durationS, LIAO_2017_PASS.durationS)}</td></tr>
            <tr><td>Detections</td><td className="num">{fmt(liao.detections)}</td><td className="num">{fmt(LIAO_2017_PASS.detections)}</td><td className="num">{pct(liao.detections, LIAO_2017_PASS.detections)}</td></tr>
            <tr><td>Sifted key, signal states</td><td className="num">{fmt(liao.siftedBits)}</td><td className="num">{fmt(LIAO_2017_PASS.siftedBits)}</td><td className="num">{pct(liao.siftedBits, LIAO_2017_PASS.siftedBits)}</td></tr>
            <tr><td>Secret key, asymptotic (both bases)</td><td className="num">{fmt(liao.keyBits)}</td><td className="num">{fmt(LIAO_2017_PASS.finalKeyBits)} (finite-key)</td><td className="num">{pct(liao.keyBits, LIAO_2017_PASS.finalKeyBits)}</td></tr>
            <tr><td>Secret key, finite, one basis (Z)</td><td className="num">{fmt(pos(liaoFinite.one))}</td><td className="num">{fmt(LIAO_2017_PASS.finalKeyBits)} (finite-key)</td><td className="num">{pct(pos(liaoFinite.one), LIAO_2017_PASS.finalKeyBits)}</td></tr>
            <tr><td>Secret key, finite, both bases</td><td className="num">{fmt(pos(liaoFinite.both))}</td><td className="num">{fmt(LIAO_2017_PASS.finalKeyBits)} (finite-key)</td><td className="num">{pct(pos(liaoFinite.both), LIAO_2017_PASS.finalKeyBits)}</td></tr>
          </tbody>
        </table>
      </div>
      <div className="net-table">
        <table>
          <caption className="small muted" style={{ textAlign: "left", captionSide: "top", paddingBottom: 4 }}>
            Khmelev et al., Opt. Express 32, 11964 (2024): Micius to Zvenigorod on 1 March 2022, the session from
            {" "}{KHMELEV_2024_SESSION.maxRangeKm} km to {KHMELEV_2024_SESSION.minRangeKm} km and back
            ({fmt(khmelev.durationS)} s in this model, where they received states for roughly
            {" "}{KHMELEV_2024_SESSION.receivedS} s), on that link&apos;s own profile whatever this run&apos;s, at
            their failure probability of 1e-9 and this tab&apos;s statistics. They keyed both bases.
          </caption>
          <thead><tr><th></th><th className="num">this model</th><th className="num">Khmelev 2024</th><th className="num">difference</th></tr></thead>
          <tbody>
            <tr><td>Secret key, finite, one basis (Z)</td><td className="num">{fmt(pos(khmelev.one))}</td><td className="num">{fmt(KHMELEV_2024_SESSION.finalKeyBits)} (final)</td><td className="num">{pct(pos(khmelev.one), KHMELEV_2024_SESSION.finalKeyBits)}</td></tr>
            <tr><td>Secret key, finite, both bases</td><td className="num">{fmt(pos(khmelev.both))}</td><td className="num">{fmt(KHMELEV_2024_SESSION.finalKeyBits)} (final)</td><td className="num">{pct(pos(khmelev.both), KHMELEV_2024_SESSION.finalKeyBits)}</td></tr>
          </tbody>
        </table>
        <p className="small muted" style={{ margin: "4px 0" }}>
          Their {fmt(KHMELEV_2024_SESSION.finalKeyBits)} bits are the key after an analysis of the detectors&apos;
          unequal efficiencies, which this model&apos;s detectors do not have; before it their key is
          {" "}{fmt(KHMELEV_2024_SESSION.beforeMismatchBits)} bits. Both correct errors at f_EC {KHMELEV_2024_SESSION.fEC},
          where this model uses {fmt(fkSrc.fEC, 2)}.
        </p>
      </div>
      <Details summary="How the pass is computed, and what it leaves out">
        <p>
          A circular orbit at the profile&apos;s altitude, the station a fixed angle off the ground track: t seconds from
          the top of the pass the satellite is an angle psi from the station with cos psi = cos psi0 cos(omega t), and
          the range and elevation follow from the triangle of radii. Each second runs the Downlink tab&apos;s own bound
          at that range and the rates are integrated. Earth&apos;s rotation is left out (about 0.35 km/s at Xinglong
          against the satellite&apos;s 7.6), and so are weather and pointing jitter. Liao&apos;s passes ended at 10 degrees
          and began from about 15; this model uses one threshold for both ends.
        </p>
        <p>
          The secret-key rows count the key three ways, and say which. The asymptotic row is the model&apos;s bound
          per pulse sent, which keys every sifted signal detection, in both bases, with nothing to allow for, so a
          finite key on the same counts must be smaller. The finite rows are the composable finite key of the model&apos;s
          counts over the same window, by {boundName}&apos;s bound with {statsLabel}, at Liao&apos;s failure probability
          of 1e-9. One basis keys Z and spends X on the phase-error estimate, as Lim et al.&apos;s protocol does; both
          bases keys each basis with its phase error estimated from the other&apos;s counts, each at half of eps_sec, so
          the pair keeps it. Both bases is how the published keys were counted: Liao et al. distilled theirs from
          all {fmt(LIAO_2017_PASS.siftedBits)} sifted bits, and Khmelev et al. write that &quot;both bases are used for
          key generation&quot;.{gap && (gap.bothOverAsym < 1
            ? <> Keyed on both bases the finite key is {fmt(100 * gap.bothOverAsym, 1)}% of the asymptotic row, so
              {" "}{fmt(100 * gap.basisShare)}% of the step from the asymptotic row to the one-basis row is the basis
              convention, and the rest, {fmt(100 * (1 - gap.basisShare))}%, is everything else between the two bounds:
              the statistics and the other finite-size terms, net of the decoy and vacuum detections the finite key
              also draws on.</>
            : <> Keyed on both bases the finite key is {fmt(100 * gap.bothOverAsym, 1)}% of the asymptotic row, at or
              above it because it also draws on the decoy and vacuum detections the asymptotic row leaves out, so the
              step from the asymptotic row to the one-basis row is the basis convention, all of it.</>)} Counted the
          same way, on both bases, the model&apos;s key
          is {ratio(pos(liaoFinite.both), LIAO_2017_PASS.finalKeyBits)} times Liao&apos;s, and on Khmelev et
          al.&apos;s session {ratio(pos(khmelev.both), KHMELEV_2024_SESSION.finalKeyBits)} times theirs. Liao et al. do
          not name their analysis; they allow for intensity fluctuations and correct errors with a Hamming code,
          neither modelled here, and Khmelev et al. correct errors at f_EC {KHMELEV_2024_SESSION.fEC} and subtract the
          detector mismatch, so the rows differ by the analysis as well as the model. Vergoossen et al. (Acta
          Astronautica 173, 164, 2020; arXiv:1903.07845v1, Table 3) modelled the same pass at 521,513 secret bits
          (asymptotic) against 3,926,729 detections; their sifted count, 1,963,364, is exactly half their
          detections, so it counts every state where Liao counted signals.
        </p>
      </Details>
    </section>
  );
}

/** Every measured Micius pass against the model (Round 78): Liao et al. 2017's Extended Data Table 2,
 *  each night's peak sifted rate at its shortest range, beside the model's at that range for this
 *  run's link and for the 8 dB upper-value profile. */
function NightsSection({ link }: { link: ReturnType<typeof satChannel> }) {
  const set = validationSet("liao_2017_xinglong_passes");
  const upper = useMemo(() => {
    const v = profileValues("downlink", "micius_2017_upper_atmosphere");
    return v ? downlinkFromProfile(v) : null;
  }, []);
  const { points, curve, stats } = useMemo(() => {
    if (!set) return { points: [], curve: [], stats: null };
    const col = (name: string) => set.columns.indexOf(name);
    const iD = col("date"), iKm = col("shortest_distance_km"), iPk = col("peak_sifted_khz");
    const pts = set.rows.map((r) => ({ km: Number(r[iKm]), measured: Number(r[iPk]), date: String(r[iD]) }));
    const ratios = pts.map((p) => 10 * Math.log10(p.measured / (siftedRateAt(link, p.km) / 1000)));
    const sorted = [...ratios].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const mean = ratios.reduce((a, b) => a + b, 0) / ratios.length;
    const sd = Math.sqrt(ratios.reduce((a, b) => a + (b - mean) ** 2, 0) / ratios.length);
    const c: { km: number; model: number; upper?: number }[] = [];
    for (let km = 480; km <= 1060; km += 20) {
      c.push({ km, model: siftedRateAt(link, km) / 1000, upper: upper ? siftedRateAt(upper, km) / 1000 : undefined });
    }
    return { points: pts, curve: c, stats: { median, sd, above: ratios.filter((x) => x > 0).length, n: ratios.length } };
  }, [set, link, upper]);
  if (!set || !stats) return null;
  return (
    <section className="net-section" aria-label="Every measured Micius pass against the model">
      <h4 style={{ margin: "4px 0" }}>Every measured Micius pass against the model</h4>
      <p className="small" style={{ margin: "4px 0" }}>
        Liao et al. list {stats.n} passes over Xinglong. At each one&apos;s shortest range this run&apos;s link predicts
        the peak sifted rate with a median difference of <b>{fmt(stats.median, 2)} dB</b> (measured over model), and the
        nights scatter around it by <b>{fmt(stats.sd, 1)} dB</b> (standard deviation): {stats.above} of {stats.n} measured
        peaks sit above the model. Weather moves a pass more than any loss term this model leaves out.
      </p>
      <figure className="chart-figure">
        <figcaption className="small muted">
          <b>Peak sifted rate against shortest range (log scale).</b> Points: the {stats.n} measured passes
          (Liao et al., Nature 549, 43, 2017, Extended Data Table 2). Lines: this run&apos;s link, and the 8 dB upper-value
          profile, at each range.
        </figcaption>
        <ChartReadout hint={HINT_KEYBOARD}>
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart accessibilityLayer title="Measured Micius peak sifted rates against the model's, by shortest range" margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="km" type="number" domain={[480, 1060]} tick={tick} label={{ value: "shortest range of the pass (km)", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
              <YAxis type="number" scale="log" domain={[1, 50]} allowDataOverflow tick={tick} ticks={[1, 2, 5, 10, 20, 50]} />
              <Tooltip content={<ReadoutTooltip labelName="km" />} formatter={(v: number) => `${v.toFixed(1)} kbit/s`} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
              <Line data={curve} dataKey="model" name="this run's link" stroke="var(--accent)" dot={false} strokeWidth={2} isAnimationActive={false} />
              {upper && <Line data={curve} dataKey="upper" name="8 dB upper value" stroke="var(--accent-2)" strokeDasharray="5 4" dot={false} strokeWidth={2} isAnimationActive={false} />}
              <Scatter data={points} dataKey="measured" name="measured peak, one night each" fill="var(--warn)" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartReadout>
      </figure>
    </section>
  );
}

/** The relay at the page's own composable finite key: each pass a block of its own, as the one-pass
 *  figure above is, under the run's bound and eps_sec. Exported for web/scripts/check_relay_pass_pairs.mjs. */
export function relayFiniteKey(relay: ReturnType<typeof trustedRelay>, fkSrc: ReturnType<typeof finiteKeySourceOf>) {
  const one = (pass: typeof relay.a) => {
    if (!pass.segments.length) return 0;
    const bits = keyLength(...blockCounts(pass.segments, fkSrc), fkSrc).keyBits;
    return Number.isFinite(bits) && bits > 0 ? bits : 0;
  };
  const a = one(relay.a), b = one(relay.b);
  const keyBits = Math.min(a, b);
  // Liao et al.'s "100 kB" is 800,000 bits; a pass pair adds one relayed key of the shorter length.
  return { a, b, keyBits, passPairsFor100kB: keyBits > 0 ? Math.ceil(800_000 / keyBits) : Infinity };
}

function RelaySection({ link, defaultElevation, minEl, epsSec, deviation }: {
  link: ReturnType<typeof satChannel>; defaultElevation: number; minEl: number; epsSec: number; deviation: Deviation;
}) {
  const [aId, setA] = useState("xinglong");
  const [bId, setB] = useState("graz");
  // The pass section's own elevation, to a tenth of a degree (QK49). Rounded to a whole degree, the relay simulated
  // 49 against the pass's 48.9, and its "Key with A" read 494.5 kbit beside 493.2 kbit for the same Xinglong pass.
  const [elA, setElA] = useState(Math.round(defaultElevation * 10) / 10);
  const [elB, setElB] = useState(Math.round(defaultElevation * 10) / 10);
  const [keys, setKeys] = useState<{ a: Uint8Array; b: Uint8Array } | null>(null);
  const a = LIAO_2018_STATIONS.find((s) => s.id === aId)!, b = LIAO_2018_STATIONS.find((s) => s.id === bId)!;
  const relay = useMemo(() => trustedRelay(link, a, b, elA, elB), [link, a, b, elA, elB]);
  // THE PAGE'S OWN KEY (Round 84). The pass pairs were counted at the ASYMPTOTIC key, 2 at the
  // defaults, while the pass section above gives each pass's composable finite key, 241.4 kbit:
  // 4 pairs at the figure the page itself stands behind.
  const fkSrc = useMemo(() => finiteKeySourceOf(link, epsSec, undefined, deviation), [link, epsSec, deviation]);
  const fin = useMemo(() => relayFiniteKey(relay, fkSrc), [relay, fkSrc]);
  const bound = `${fkSrc.vacuum ? "Lim et al. 2014" : "Rusca et al. 2018"}; ${DEVIATION_LABELS[fkSrc.deviation]}`;
  const demo = keys ? xorRelay(keys.a, keys.b) : null;
  const station = (value: string, set: (v: string) => void, label: string) => (
    <label>{label}
      <select value={value} onChange={(e) => set(e.target.value)}>
        {LIAO_2018_STATIONS.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
      </select>
    </label>
  );
  return (
    <section className="net-section" aria-label="Trusted satellite relay">
      <h4 style={{ margin: "4px 0" }}>Trusted satellite relay (Liao et al. 2018)</h4>
      <p className="small muted" style={{ margin: "4px 0" }}>
        Micius made a key with each station on its own pass, then &quot;performs bitwise exclusive OR operations between the
        two keys and relays the result to one of the ground stations&quot; (Liao et al., Phys. Rev. Lett. 120, 030501, 2018):
        a key across 7,600 km, Xinglong to Graz, the stations modelled here. Jinan-1 has since relayed one between China
        and South Africa, &quot;separated by over 12,900 kilometres on Earth&quot; (Li et al., Nature 640, 47, 2025). The
        satellite holds both keys, so it must be trusted.
      </p>
      <div className="net-controls">
        {station(aId, setA, "Station A")}
        <label>A&apos;s pass, highest elevation (deg)
          <input type="number" min={minEl} max={90} step={0.1} value={elA} onChange={(e) => setElA(Math.min(90, Math.max(minEl, Number(e.target.value) || minEl)))} />
        </label>
        {station(bId, setB, "Station B")}
        <label>B&apos;s pass, highest elevation (deg)
          <input type="number" min={minEl} max={90} step={0.1} value={elB} onChange={(e) => setElB(Math.min(90, Math.max(minEl, Number(e.target.value) || minEl)))} />
        </label>
      </div>
      <div className="net-table">
        <table>
          <tbody>
            <tr><th scope="row">Ground distance A to B</th><td className="num">{fmt(relay.groundKm)} km</td>
              <td className="small muted">{aId === bId ? "the same station" : relay.sharedView
                ? `within one satellite's ${fmt(sharedViewMaxKm(link))} km footprint: both can be in view at once`
                : `wider than one satellite's ${fmt(sharedViewMaxKm(link))} km footprint: the key has to be carried`}</td></tr>
            <tr><th scope="row">Key with A, one pass</th><td className="num">{kbit(relay.a.keyBits)}</td><td className="small muted">{fmt(relay.a.durationS)} s above the minimum elevation; asymptotic: the pass simulation above, run at {fmt(elA, 1)} deg and {a.name}&apos;s altitude, so at the defaults, Xinglong at the pass&apos;s own elevation, it is the one-pass figure above. As one block its composable finite key is {kbit(fin.a)}</td></tr>
            <tr><th scope="row">Key with B, one pass</th><td className="num">{kbit(relay.b.keyBits)}</td><td className="small muted">{fmt(relay.b.durationS)} s; as one block, {kbit(fin.b)}</td></tr>
            <tr><th scope="row">Relayed key A to B</th><td className="num"><b>{kbit(relay.keyBits)}</b></td><td className="small muted">the shorter of the two: one key pads the other. At the finite keys, {kbit(fin.keyBits)}</td></tr>
            <tr><th scope="row">Second key exists after</th><td className="num">{fmt(relay.waitH, 1)} h</td><td className="small muted">a sun-synchronous orbit passes each station at the same local time, so the longitude difference sets the wait</td></tr>
            <tr><th scope="row">Pass pairs for Liao&apos;s 100 kB</th><td className="num"><b>{Number.isFinite(fin.passPairsFor100kB) ? fin.passPairsFor100kB : "-"}</b></td><td className="small muted">&quot;we establish a 100 kB secure key between Xinglong and Graz&quot;: 800,000 bits over the relayed finite key per pass pair, each pass a block of its own ({bound}, eps_sec {epsSec.toExponential(0)}). The asymptotic key would need {Number.isFinite(relay.passPairsFor100kB) ? relay.passPairsFor100kB : "-"}, which no finite block distils</td></tr>
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 8 }}>
        <button className="btn ghost" style={{ fontSize: 12 }}
          onClick={() => setKeys({ a: randomBytes(16), b: randomBytes(16) })}>
          {keys ? "Draw new keys" : "Show the relay's XOR on real random keys"}
        </button>
        {demo && keys && (
          <div className="xor-demo" style={{ marginTop: 6 }} aria-live="polite">
            <div>K_A, shared by the satellite and {a.name}: {hex(keys.a)}</div>
            <div>K_B, shared by the satellite and {b.name}: {hex(keys.b)}</div>
            <div>broadcast K_A xor K_B: {hex(demo.broadcast)}</div>
            <div>{a.name} computes (K_A xor K_B) xor K_A = {hex(demo.recovered)} ({hex(demo.recovered) === hex(keys.b) ? "equals K_B" : "MISMATCH"})</div>
            <div className="muted">The broadcast is K_B padded with K_A, so a listener learns nothing from it. The satellite
              learns everything: it held both keys.</div>
          </div>
        )}
      </div>
      <Details summary="What this leaves out">
        <p>
          Both passes use this page&apos;s link profile with each station&apos;s own altitude; Liao&apos;s Graz and Nanshan
          telescopes differ from Xinglong&apos;s. Liao et al. 2018 report final keys of 61 to 398 kb per pass, with each
          station&apos;s total over its listed passes between 400 and 833 kb, and a sifted rate of &quot;~3 kb/s at ~1000 km and
          ~9 kb/s at ~600 km&quot;. Its 100 kB relayed key is more than Xinglong&apos;s listed 400 kb total, so it drew on
          passes the table does not list. Ground distances: {fmt(groundDistanceKm(LIAO_2018_STATIONS[0], LIAO_2018_STATIONS[2]))} km
          Xinglong to Graz and {fmt(groundDistanceKm(LIAO_2018_STATIONS[0], LIAO_2018_STATIONS[1]))} km Xinglong to Nanshan
          from the paper&apos;s coordinates, against its stated 7,600 and 2,500.
        </p>
      </Details>
    </section>
  );
}

function ConstellationSection({ orbitKm, minEl }: { orbitKm: number; minEl: number }) {
  const profiles = linkProfiles("isl");
  // The configured default terminal pair, not the first file alphabetically.
  const [profileId, setProfileId] = useState(String(qkdCfg().isl_profile));
  const base = useMemo(() => islFromProfile(profileId), [profileId]);
  const [aperture, setAperture] = useState(base.rxApertureM);
  const [divergence, setDivergence] = useState(base.txDivergenceUrad);
  const [n, setN] = useState(10);
  const [beta, setBeta] = useState(0);
  const [eclipseOnly, setEclipseOnly] = useState(base.eclipseOnly);
  const spec: IslSpec = { ...base, rxApertureM: aperture, txDivergenceUrad: divergence, eclipseOnly };
  const c = useMemo(() => stringOfPearls(n, orbitKm, spec, beta),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [n, orbitKm, aperture, divergence, beta, eclipseOnly, base]);
  const sweep = useMemo(() => {
    const out: { n: number; perDay: number }[] = [];
    // To 48 satellites, or twice the chosen count for a short-range terminal, about 60 points.
    const top = Math.min(200, Math.max(48, 2 * n));
    const step = Math.max(1, Math.round((top - Math.max(3, c.minSatellites)) / 60));
    for (let k = Math.max(3, c.minSatellites); k <= top; k += step) {
      const r = stringOfPearls(k, orbitKm, spec, beta);
      out.push({ n: k, perDay: Math.round(r.islKeyPerDayBits / 1000) });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orbitKm, aperture, divergence, beta, eclipseOnly, base, c.minSatellites, n]);
  const loadProfile = (id: string) => {
    setProfileId(id);
    const p = islFromProfile(id);
    setAperture(p.rxApertureM); setDivergence(p.txDivergenceUrad); setEclipseOnly(p.eclipseOnly);
  };
  const chosen = profiles[profileId];
  return (
    <section className="net-section" aria-label="Constellation with inter-satellite links">
      <h4 style={{ margin: "4px 0" }}>Constellation with inter-satellite links (Vergoossen et al. 2020)</h4>
      <p className="small muted" style={{ margin: "4px 0" }}>
        A &quot;string-of-pearls configuration of satellites in the same plane&quot; (Vergoossen et al., Acta Astronautica 173,
        164, 2020; arXiv:1903.07845),
        each linked by a quantum channel to the satellites ahead and behind, keys passed from one to the next, every
        satellite a trusted node. Operated in Earth&apos;s shadow only by default - this page&apos;s assumption for the
        inter-satellite links, since Vergoossen et al. state the eclipse condition for satellite-to-ground passes. No quantum inter-satellite
        link has flown; the terminals are a profile, so a design can be loaded and compared.
      </p>
      <div className="net-controls">
        <label style={{ flex: "2 1 220px" }}>Terminal profile
          <select value={profileId} onChange={(e) => loadProfile(e.target.value)} title={chosen ? `${chosen.source}. ${chosen.note ?? ""}` : undefined}>
            {Object.values(profiles).map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </label>
        <label>Satellites in the plane
          <input type="number" min={3} max={200} step={1} value={n} onChange={(e) => setN(Math.min(200, Math.max(3, Math.round(Number(e.target.value) || 3))))} />
        </label>
        <label>Receive aperture (m)
          <input type="number" min={0.05} max={2} step={0.05} value={aperture} onChange={(e) => setAperture(Math.min(2, Math.max(0.05, Number(e.target.value) || 0.05)))} />
        </label>
        <label>Beam divergence (urad)
          <input type="number" min={1} max={100} step={1} value={divergence} onChange={(e) => setDivergence(Math.min(100, Math.max(1, Number(e.target.value) || 1)))} />
        </label>
        <label>Sun beta angle (deg)
          <input type="number" min={0} max={90} step={5} value={beta} onChange={(e) => setBeta(Math.min(90, Math.max(0, Number(e.target.value) || 0)))} />
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={eclipseOnly} onChange={(e) => setEclipseOnly(e.target.checked)} /> in shadow only
        </label>
      </div>
      <div className="net-table">
        <table>
          <tbody>
            <tr><th scope="row">Distance between neighbours</th><td className="num">{fmt(c.neighbourKm)} km</td><td className="small muted">2 (R + h) sin(pi / N)</td></tr>
            <tr><th scope="row">Line of sight passes above the ground at</th><td className="num">{fmt(c.chordMinAltitudeKm)} km</td>
              <td className="small muted">{c.losOk ? `clear of the ${fmt(spec.clearanceKm)} km the profile requires` : `below the ${fmt(spec.clearanceKm)} km the profile requires: no link. The plane needs at least ${c.minSatellites} satellites`}</td></tr>
            <tr><th scope="row">Link loss</th><td className="num">{Number.isFinite(c.islLossDb) ? `${fmt(c.islLossDb, 1)} dB` : "-"}</td><td className="small muted">diffraction, pointing and receiver optics; no atmosphere above the clearance height</td></tr>
            <tr><th scope="row">Key rate per link</th><td className="num">{fmt(c.islBitPerS, 1)} bit/s</td><td className="small muted">the Downlink tab&apos;s bound over this channel, per pulse sent</td></tr>
            <tr><th scope="row">Time per orbit a link runs</th><td className="num">{fmt(c.islSecondsPerOrbit / 60, 1)} min</td><td className="small muted">{eclipseOnly ? `both neighbours in shadow; one satellite is in shadow ${fmt(100 * c.eclipseFraction, 1)}% of its ${fmt(c.periodS / 60, 1)} min orbit` : "always: the shadow condition is off"}</td></tr>
            <tr><th scope="row">Key per link per day</th><td className="num"><b>{kbit(c.islKeyPerDayBits)}</b></td><td className="small muted">{fmt(c.islKeyPerOrbitBits / 1000, 1)} kbit per orbit</td></tr>
            <tr><th scope="row">Trusted nodes across the ring</th><td className="num">{c.hopsAcross + 1}</td><td className="small muted">a key carried to the opposite satellite crosses {c.hopsAcross} links, and every satellite on the way holds it</td></tr>
          </tbody>
        </table>
      </div>
      <figure className="chart-figure">
        <figcaption className="small muted">
          <b>Key per link per day against the number of satellites (model).</b> More satellites mean closer neighbours,
          so less diffraction loss and a longer shared shadow; fewer than {c.minSatellites} cannot see each other over the
          Earth at a {fmt(spec.clearanceKm)} km clearance.
        </figcaption>
        <ChartReadout hint={HINT_KEYBOARD}>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={sweep} accessibilityLayer title="Key per inter-satellite link per day against satellites in the plane" margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="n" tick={tick} label={{ value: "satellites in the plane", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
              <YAxis tick={tick} />
              <Tooltip content={<ReadoutTooltip labelName="satellites" />} />
              <ReferenceLine x={n} stroke="var(--warn)" strokeDasharray="4 3" />
              <Line type="monotone" dataKey="perDay" name="kbit per link per day" stroke="var(--accent)" dot={false} strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartReadout>
      </figure>
      <Details summary="Network geometry in 3D">
        <SatNetwork3D orbitKm={orbitKm} n={n} clearanceKm={spec.clearanceKm}
          minElevationDeg={minEl} shadowHalfAngleDeg={c.eclipseFraction * 180} />
      </Details>
      <Details summary="What this leaves out">
        <p>
          One plane at the profile&apos;s altitude, equally spaced, the Sun at the beta angle above, the shadow taken as a
          cylinder. The link runs the downlink&apos;s diffraction law between two terminals with no atmosphere, so its
          loss is set by the distance, the divergence and the aperture. Dark counts are the ground station&apos;s; in orbit,
          radiation raises a detector&apos;s dark counts over a mission. Vergoossen et al. found inter-satellite links gave a
          20 to 100 percent increase in the message size their ground networks could send; this view models one link of
          such a plane, not that network.
        </p>
      </Details>
    </section>
  );
}
