// The private set intersection among three to five banks, under MPC, as one run reports it (Round 86).
//
// Every number comes from the run. The self-test exists only because every bank is in this tab, and the card says so; the caveats are part of the result.
import StageBadge from "./StageBadge";
import type { MultiPsiReport } from "../lib/clientCompute";

export default function MultiPsiPanel({ report }: { report: MultiPsiReport }) {
  const r = report.result;
  const s = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
  const rows: [string, string][] = [
    ["Circuit", `${r.andGates.toLocaleString("en-US")} AND gates: ${r.setSize} accounts per bank, every account of the first bank against every account of the others`],
    ["Traffic", `${(r.bytes / 1e6).toFixed(1)} MB in ${r.messages.toLocaleString("en-US")} messages between the ${r.parties} notional parties`],
    ["Wall-clock", `${s(r.totalMs)} in ${report.ranIn}: the protocol ${s(r.mpcMs)}, compiling the circuit ${(r.compileMs).toFixed(0)} ms`],
    ["32-bit account hashes", `a false match between two different accounts in this run is below 2^${r.falseMatchBoundLog2.toFixed(1)}`],
  ];
  return (
    <div className="card" style={{ marginTop: 14 }} id="multi-party-psi">
      <h3>PSI among {r.parties} banks, under MPC <StageBadge kind="notional" /></h3>
      <p style={{ marginTop: 4 }}>
        Each bank brings {r.setSize} flagged accounts and is a party to one garbled circuit. Every bank receives the same output, the accounts all {r.parties} hold, and nothing
        else: unlike the two-bank OPRF above, no bank learns which accounts it shares with only some of the others.
      </p>
      <p style={{ margin: "6px 0 10px" }}>
        Accounts held by all {r.parties} banks: <b>{r.intersection}</b>.{" "}
        <span className={`badge-state ${r.matchesClear ? "done" : "error"}`}>{r.matchesClear ? "matches the intersection in the clear" : "DISAGREES with the intersection in the clear"}</span>
      </p>
      <table className="table">
        <tbody>
          {rows.map(([k, v]) => <tr key={k}><th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{k}</th><td>{v}</td></tr>)}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 10 }}>
        All {r.parties} banks run in this tab, so the separation is notional, and only that lets the page compare the result with the clear intersection. The protocol is WRK17b, designed
        against malicious parties, but it runs here in a pre-release library, and its oblivious transfers rest on a discrete-log assumption, so it is not post-quantum. A circuit compares every pair, so its
        size grows with the square of the list: the accounts here are generated, and few. Dedicated multi-party PSI goes much further, for instance five parties with 2^20 items each in 72 s, semi-honest (Kolesnikov et al., CCS 2017).
      </p>
    </div>
  );
}
