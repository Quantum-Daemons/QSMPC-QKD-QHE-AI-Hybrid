// The QRNG's two continuous health tests, live, with a fault to inject (Round 86, the owner's choice of 2026-11-22).
// The counters are lib/qrngHealthStream.ts; the samples are simulated, a fair coin from this tab's CSPRNG standing in
// for measuring |+>, and a fault replaces them from the next sample on.
import { useEffect, useRef, useState } from "react";
import { qrngCfg } from "../lib/config";
import {
  BIASED_ONES, type Fault, failed, feed, type HealthState, newHealthState, sampleBit,
} from "../lib/qrngHealthStream";
import StageBadge from "./StageBadge";

const PER_FRAME = 4096;
const LIMIT = 1 << 20;

/** The line under the table: where the stream stands, and what a failure means. */
function summary(s: HealthState, fault: Fault): string {
  const share = s.samples ? ((100 * s.ones) / s.samples).toFixed(1) : "0.0";
  const injected = fault === "none" ? ""
    : `, with the fault "${fault === "stuck" ? "stuck at 1" : `${Math.round(BIASED_ONES * 4)} ones in 4`}" injected`;
  const end = failed(s)
    ? " A failed test takes the source offline until it is repaired and its health is shown again; no output from the"
      + " failing stretch is used."
    : s.samples >= LIMIT ? ` No alarm in ${LIMIT.toLocaleString("en-US")} samples; the stream stops there.` : "";
  return `${s.samples.toLocaleString("en-US")} samples, ${share}% ones${injected}.${end} A working source can still trip`
    + " a test, rarely: that is the false-alarm probability the cutoffs are set for.";
}

export default function QrngHealthLive() {
  const h = Number(qrngCfg()?.min_entropy_per_raw_bit ?? 0.9);
  const [view, setView] = useState<HealthState>(() => newHealthState(h));
  const [running, setRunning] = useState(false);
  const [fault, setFault] = useState<Fault>("none");
  const state = useRef<HealthState>(newHealthState(h));
  const faultRef = useRef<Fault>("none");
  const timer = useRef<number | null>(null);

  useEffect(() => () => { if (timer.current !== null) clearTimeout(timer.current); }, []);

  const tick = () => {
    const s = state.current;
    const u = new Uint32Array(PER_FRAME);
    crypto.getRandomValues(u);
    for (let i = 0; i < PER_FRAME && !failed(s) && s.samples < LIMIT; i++) feed(s, sampleBit(u[i] / 2 ** 32, faultRef.current));
    setView({ ...s });
    if (failed(s) || s.samples >= LIMIT) { setRunning(false); timer.current = null; return; }
    // A timer, not requestAnimationFrame: frames do not fire in an embedded or hidden view, and a browser slows a
    // timer in a background tab on its own.
    timer.current = window.setTimeout(tick, 16);
  };

  const start = () => {
    // A fresh start is a working source: a fault injected into the last stream does not carry over.
    faultRef.current = "none";
    setFault("none");
    state.current = newHealthState(h);
    setView({ ...state.current });
    setRunning(true);
    timer.current = window.setTimeout(tick, 16);
  };
  const inject = (f: Fault) => { faultRef.current = f; setFault(f); };

  const s = view;
  const status = (at: number | null) => (at === null ? "passing" : `FAILED at sample ${at.toLocaleString("en-US")}`);
  return (
    <div className="card" id="qrng-health-live">
      <h3>The QRNG&apos;s health tests, live <StageBadge kind="simulated" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        A noise source runs two tests on every sample and stops at the first failure (NIST SP 800-90B, section 4.4): the
        repetition count catches a source stuck on one value, the adaptive proportion a source drifting towards one. The
        cutoffs are set from the source&apos;s assessed min-entropy, {h} bits per raw bit, for a false-alarm probability
        of 2^-20. The samples here are simulated, a fair coin from this tab&apos;s CSPRNG standing in for measuring a qubit
        prepared in |+&gt;; inject a fault and watch which test catches it.
      </p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 6 }}>
        <button className="btn" onClick={start} disabled={running}>{s.samples ? "Start again" : "Start the stream"}</button>
        <button className="btn" onClick={() => inject("stuck")} disabled={!running || fault === "stuck"}>Stick at 1</button>
        <button className="btn" onClick={() => inject("biased")} disabled={!running || fault === "biased"}>
          Bias to {Math.round(BIASED_ONES * 4)} ones in 4</button>
        <button className="btn" onClick={() => inject("none")} disabled={!running || fault === "none"}>Work normally</button>
      </div>
      <table className="table" style={{ marginTop: 8 }}>
        <thead>
          <tr>
            <th scope="col" style={{ textAlign: "left" }}>Test</th>
            <th scope="col">Now</th>
            <th scope="col">Worst so far</th>
            <th scope="col">Cutoff</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>Repetition count: the current run</th>
            <td>{s.run}</td><td>{s.longestRun}</td><td>{s.rctCutoff}</td><td>{status(s.rctFailedAt)}</td>
          </tr>
          <tr>
            <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>
              Adaptive proportion: matches of the window&apos;s first sample, window of {s.window.toLocaleString("en-US")}</th>
            <td>{s.count}</td><td>{s.worstCount}</td><td>{s.aptCutoff}</td><td>{status(s.aptFailedAt)}</td>
          </tr>
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>{summary(s, fault)}</p>
    </div>
  );
}
