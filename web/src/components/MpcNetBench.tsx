import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import Details from "./Details";
import KeyPoints from "./KeyPoints";
import bench from "../lib/mpcNetBench.json";

/**
 * The secure argmax under the two garbled-circuit engines, measured between two browser tabs over a
 * shaped network (register, Round 64). RECORDED numbers, not a live run: the page shows what was
 * measured on one machine, with the date, and says what the measurement does not cover.
 *
 * The engines: emp-wasm with mpc-framework and summon, which the demo shipped until Round 65 and which
 * is archived upstream, and Polytune with Garble, which replaced it on the strength of these numbers.
 * Same function and inputs as clientMPC.ts, with the sharing it used until Round 66 (a mask bounded to
 * 2^16 and an offset the circuit subtracted). The shipped circuit has shared modulo 2^32 since then,
 * and the paragraph under the table gives its size.
 */

const tick = { fill: "var(--muted)", fontSize: 11 };
type Row = (typeof bench.configs)[number];
const ms = (v: number) => `${Math.round(v)} ms`;
/** emp-wasm time over Polytune time, from the medians: not the rounded ratio, which would round twice. */
const speed = (r: Row) => r.empWasm.medianMs / r.polytune.medianMs;

function RateChart({ rate }: { rate: number }) {
  const rows = bench.configs
    .filter((r: Row) => r.n === 2 && r.lossPct === 0 && r.rateMbit === rate)
    .map((r: Row) => ({ rtt: `${r.rttMs} ms RTT`, emp: r.empWasm.medianMs, polytune: r.polytune.medianMs }));
  const hatch = `mpcbench-hatch-${rate}`;
  return (
    <div style={{ flex: "1 1 280px", minWidth: 0 }}>
      <div className="small muted" style={{ marginBottom: 4 }}>{rate} Mbit/s each way, no loss</div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={rows} accessibilityLayer title={`Secure argmax completion time at ${rate} Mbit/s against round-trip time, emp-wasm and Polytune`} margin={{ top: 8, right: 8, bottom: 8, left: 4 }}>
          {/* WCAG 2.2 SC 1.4.1: the two engines differ by a hatch as well as by colour. */}
          <defs>
            <pattern id={hatch} patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
              <rect width="5" height="5" fill="var(--accent-2)" />
              <line x1="0" y1="0" x2="0" y2="5" stroke="var(--bg)" strokeWidth="1.4" />
            </pattern>
          </defs>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="rtt" tick={tick} />
          <YAxis tick={tick} width={56} tickFormatter={(v: number) => `${v}`} unit=" ms" />
          <Tooltip formatter={(v: number) => ms(v)} />
          <Legend />
          <Bar dataKey="emp" name="emp-wasm (replaced)" fill="var(--accent)" />
          <Bar dataKey="polytune" name="Polytune (now shipped, hatched)" fill={`url(#${hatch})`} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function MpcNetBench() {
  const two = bench.configs.filter((r: Row) => r.n === 2);
  const ten = bench.configs.filter((r: Row) => r.n === 10);
  const lo = Math.min(...two.map(speed));
  const hi = Math.max(...two.map(speed));
  const allCorrect = bench.configs.every((r: Row) => r.empWasm.correct === r.empWasm.runs && r.polytune.correct === r.polytune.runs);
  return (
    <figure className="chart-figure" style={{ margin: 0 }}>
      <KeyPoints items={[
        <>At 2 classes - what every inference page runs - Polytune finished first on all {two.length} links
          measured, {lo.toFixed(2)} to {hi.toFixed(2)} times faster, despite sending about five times the bytes.</>,
        <>At 10 classes the order reverses: emp-wasm was {(1 / speed(ten[0])).toFixed(2)} and {(1 / speed(ten[1])).toFixed(2)} times
          faster, because Polytune's traffic grows with the circuit (7.2 MB against 1.2 MB).</>,
        <>{allCorrect ? "Every run of both engines returned the right index." : "Not every run returned the right index."}{" "}
          Recorded {bench.measuredAt.slice(0, 10)} on one machine ({bench.browser}).</>,
      ]} />
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 8 }}>
        <RateChart rate={10} />
        <RateChart rate={100} />
      </div>
      <figcaption className="figure-note">
        Median completion time of three runs - the later of the two parties - for the 2-class secure argmax,
        two browser tabs connected by WebSocket relays that netem shapes. The table below adds packet loss
        and the 10-class case.
      </figcaption>
      {/* Its own scroller: ten columns are wider than the card at most widths, and a table that
          widens the card makes the whole page scroll sideways (175 px at the pane's width, measured). */}
      <div style={{ overflowX: "auto", maxWidth: "100%", marginTop: 10 }}>
      <table className="facts-table small">
        <thead>
          <tr><th>classes</th><th>RTT</th><th>Mbit/s</th><th>loss</th><th>measured RTT</th><th>measured goodput</th>
            <th>emp-wasm</th><th>Polytune</th><th>ratio</th><th>bytes emp / Polytune</th></tr>
        </thead>
        <tbody>
          {bench.configs.map((r: Row) => (
            <tr key={`${r.n}-${r.rttMs}-${r.rateMbit}-${r.lossPct}`}>
              <td>{r.n}</td><td>{r.rttMs} ms</td><td>{r.rateMbit}</td><td>{r.lossPct}%</td>
              <td>{r.measuredRttMs} ms</td><td>{r.measuredGoodputMbps} Mbit/s</td>
              <td>{ms(r.empWasm.medianMs)}</td><td>{ms(r.polytune.medianMs)}</td>
              <td>{speed(r) >= 1 ? `Polytune ${speed(r).toFixed(2)}x` : `emp-wasm ${(1 / speed(r)).toFixed(2)}x`}</td>
              <td>{(r.empWasm.bytes / 1e3).toFixed(0)} KB / {(r.polytune.bytes / 1e3).toFixed(0)} KB</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      {/* Round 65: Polytune replaced emp-wasm on the strength of these numbers. The module is built
          from Rust crates under permissive licences whose own notices must travel with it, so they
          are served beside it and linked here. */}
      <p className="small" style={{ marginTop: 10 }}>
        Since Round 65 the demo's secure argmax runs on Polytune. Since Round 66 it shares each logit
        modulo 2^32 with masks from the browser's CSPRNG, which also made the circuit smaller than the one
        measured here: 367 AND gates against 400, and 944,619 bytes per run against 1,018,898. Its
        WebAssembly module is compiled from
        Rust crates under MIT, Apache-2.0, BSD and similar licences, whose own licence files ship with the
        demo:{" "}
        <a href="/licenses/qsh-mpc.txt" target="_blank" rel="noopener noreferrer">third-party notices for
          the MPC module</a>.
      </p>
      <Details summary="How this was measured, and what it does not cover">
        <p>
          Each party ran in its own tab, on different sites (127.0.0.1 and localhost) so the browser gave
          them separate processes, as two hosts would have. They exchanged messages over WebSockets through
          two relay containers, one per direction, each shaped by netem on its own interface: one-way delay
          of half the round trip, the bandwidth cap, and the loss, with netem's default 1000-packet queue.
          The real TCP stacks therefore met the delay, the bottleneck and the loss, and the relays recorded
          the kernel's own counters - congestion windows, smoothed RTT and retransmissions - which show
          congestion control at work (for example, 1% loss cut the bulk goodput of a 100 Mbit/s, 50 ms link
          to 10.4 Mbit/s). Each link was calibrated before it was used: the measured RTT and goodput columns
          are what an echo test and a bulk transfer through the same relays achieved.
        </p>
        <p>
          Not covered: other machines, browsers and real wide-area paths; the demo as shipped, whose two
          parties share one tab and exchange messages in memory; and engines beyond these two. The 2-class
          case is what the demo computes today on the finance, medical and genomics pages.
        </p>
      </Details>
    </figure>
  );
}
