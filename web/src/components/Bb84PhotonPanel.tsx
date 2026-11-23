// BB84 pulse by pulse, with an intercept-resend eavesdropper to switch on (Round 86, the owner's choice of 2026-11-22:
// a per-photon BB84 and decoy table feeding the QBER chart). The pulses are lib/bb84Photons.ts's, on the page's fibre
// channel at its configured length; each press adds its measured QBER to the chart beside the page's model.
import { useId, useState } from "react";
import { CartesianGrid, ComposedChart, ErrorBar, Line, ResponsiveContainer, Scatter, XAxis, YAxis } from "recharts";
import { type Bb84Result, bb84Expected, simulateBb84 } from "../lib/bb84Photons";
import { FIBER, sourceOf, transmittance } from "../lib/quantumSim";
import StageBadge from "./StageBadge";

const PULSES = 400_000;
const FRACTIONS = [0, 0.25, 0.5, 1];
const AXIS = { fontSize: 10, fill: "#8a94a6" };
const pct = (x: number) => `${(100 * x).toFixed(2)}%`;
const bit = (b: number | null) => (b == null ? "-" : String(b));

export default function Bb84PhotonPanel() {
  const [eve, setEve] = useState(0);
  const [last, setLast] = useState<Bb84Result | null>(null);
  const [points, setPoints] = useState<{ eve: number; qber: number; err: [number, number] }[]>([]);
  const captionId = useId();
  const src = sourceOf(FIBER);
  const eta = transmittance(FIBER);
  const run = () => {
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    const r = simulateBb84({ pulses: PULSES, eve, misalignment: FIBER.misalignment, transmittance: eta, mu: src.mu, nu: src.nu,
                             pSignal: src.pSignal, seed });
    setLast(r);
    setPoints((p) => [...p, { eve, qber: r.qber, err: [r.qber - r.ci95[0], r.ci95[1] - r.qber] }]);
  };
  const model = Array.from({ length: 21 }, (_, i) => ({ eve: i / 20, model: bb84Expected(FIBER.misalignment, i / 20) }));
  return (
    <div className="card" id="bb84-photons">
      <h3>BB84, pulse by pulse <StageBadge kind="simulated" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        Each pulse is a signal (mean {src.mu} photons) or a decoy ({src.nu}), and carries Alice&apos;s random bit in her
        random basis over {FIBER.distanceKm} km of fibre, transmittance {eta.toFixed(3)} with the detector. Bob measures in
        his own random basis; the pulses where the two bases agree are kept, and the share of those that disagree is the
        QBER. An intercept-resend eavesdropper measures a share of the pulses in a random basis and sends on what she got,
        which errs on a quarter of the kept pulses she touched. Dark counts are left out, so misalignment ({FIBER.misalignment})
        and Eve are the only errors, and the page&apos;s model gives the expected QBER exactly.
      </p>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <label className="small">Share of pulses Eve intercepts{" "}
          <select value={eve} onChange={(e) => setEve(Number(e.target.value))}>
            {FRACTIONS.map((f) => <option key={f} value={f}>{f === 0 ? "none" : f === 1 ? "all" : String(f)}</option>)}
          </select>
        </label>
        <button className="btn" onClick={run}>Send {PULSES.toLocaleString("en-US")} pulses</button>
      </div>
      {last ? (
        <>
          <table className="table" style={{ marginTop: 8 }}>
            <caption className="small" style={{ captionSide: "top", textAlign: "left" }}>
              The first {last.first.length} pulses Bob detected, out of the first {last.firstSpan.toLocaleString("en-US")} sent;
              the rest reached no detector</caption>
            <thead>
              <tr>
                <th scope="col" style={{ textAlign: "left" }}>Pulse</th><th scope="col">Photons</th>
                <th scope="col">Alice</th><th scope="col">Eve</th><th scope="col">Bob</th><th scope="col">Kept</th>
              </tr>
            </thead>
            <tbody>
              {last.first.map((p, i) => (
                <tr key={i}>
                  <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{i + 1}, {p.kind}</th>
                  <td>{p.photons}</td>
                  <td>{p.aliceBit} in {p.aliceBasis}</td>
                  <td>{p.eve ? `${p.eve.bit} in ${p.eve.basis}` : "-"}</td>
                  <td>{bit(p.bobBit)} in {p.bobBasis}</td>
                  <td>{p.sifted ? (p.error ? "kept, wrong" : "kept") : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small" style={{ marginTop: 6 }}>
            Signal pulses: {last.signal.detected.toLocaleString("en-US")} of {last.signal.sent.toLocaleString("en-US")} detected;
            decoys: {last.decoy.detected.toLocaleString("en-US")} of {last.decoy.sent.toLocaleString("en-US")}. Kept:
            {" "}{(last.signal.sifted + last.decoy.sifted).toLocaleString("en-US")}, wrong:
            {" "}{(last.signal.errors + last.decoy.errors).toLocaleString("en-US")}, QBER {pct(last.qber)} (95% interval
            {" "}{pct(last.ci95[0])} to {pct(last.ci95[1])}); the model gives {pct(last.model)}.
          </p>
        </>
      ) : null}
      {points.length ? (
        <figure className="chart-figure" style={{ margin: "10px 0 0" }}>
          <ResponsiveContainer width="100%" height={200}>
            <ComposedChart data={model} role="img" aria-describedby={captionId}
                           title="QBER against the share Eve intercepts: the model, and this tab's presses"
                           margin={{ top: 6, right: 10, bottom: 14, left: 4 }}>
              <CartesianGrid stroke="#2a3040" strokeDasharray="2 4" />
              <XAxis dataKey="eve" type="number" domain={[0, 1]} tick={AXIS} />
              <YAxis domain={[0, 0.3]} tick={AXIS} width={44} tickFormatter={(v: number) => `${Math.round(100 * v)}%`} />
              <Line dataKey="model" stroke="#4f9dff" strokeWidth={1.6} dot={false} isAnimationActive={false} />
              <Scatter data={points} dataKey="qber" fill="#f6b73c" shape="diamond" isAnimationActive={false}>
                <ErrorBar dataKey="err" direction="y" width={4} stroke="#f6b73c" />
              </Scatter>
            </ComposedChart>
          </ResponsiveContainer>
          <figcaption id={captionId} className="muted small">
            <strong>QBER against the share of pulses Eve intercepts.</strong> The line is the page&apos;s model,
            misalignment composed with a quarter of Eve&apos;s share; each diamond is one press in this tab, with its 95%
            Wilson interval. The x axis is the share Eve intercepts, from none to all.
          </figcaption>
        </figure>
      ) : null}
    </div>
  );
}
