// Who holds what, and what crossed between the parties, up to the timeline's position (Round 86, the owner's choice of
// 2026-11-14). Read from the run's own `message` and `holds` events (clientCompute.ts): a size is what the run measured,
// a blank size is one it does not measure, and hex is only ever the first bytes of something sent in the clear.
import { useMemo } from "react";
import type { PipelineEvent } from "../lib/api";
import { ledgerAt, PROTECTION_NOTE, sideName, type Protection } from "../lib/runTimeline";
import type { DiagramConfig } from "../lib/useCaseFlows";
import SequenceView from "./SequenceView";
import StageBadge from "./StageBadge";

const size = (b: number | null, count?: number) =>
  b == null ? "not measured" : `${b >= 1e6 ? `${(b / 1e6).toFixed(2)} MB` : `${b.toLocaleString("en-US")} B`}${count && count > 1 ? `, ${count.toLocaleString("en-US")} messages` : ""}`;

export default function PartyLedger({ usecase, events, config }: {
  usecase: string; events: PipelineEvent[]; config: DiagramConfig;
}) {
  const { messages, holdings } = useMemo(() => ledgerAt(events), [events]);
  const parties = Object.keys(holdings).sort((x, y) => rank(x) - rank(y) || x.localeCompare(y));
  if (messages.length === 0 && parties.length === 0) return null;
  const name = (s: string) => sideName(usecase, s, config);
  const used = [...new Set(messages.map((m) => m.protection))] as Protection[];
  return (
    <div className="card" id="party-ledger" style={{ marginTop: 14 }}>
      <h3>Who holds what, and what crossed <StageBadge kind="notional" /></h3>
      <p className="muted small" style={{ marginTop: 4 }}>
        At this point of the run, read from its log. Every party runs in this tab, so the separation is notional: the bytes
        and their sizes are the run's own, and the tab could read everything. A key, a seed, a share or a set member is never
        written into the log; the first bytes in hex are shown only for what is sent in the clear.
      </p>
      <div className="grid cols-2" style={{ gap: 10 }}>
        {parties.map((p) => (
          <div key={p} className="ledger-party">
            <b>{name(p)}</b>
            <ul className="small">
              {holdings[p].map((h, i) => (
                <li key={i}>{h.kept ? <span className="tag" title="kept to itself, never sent">keeps</span> : <span className="tag">holds</span>}{" "}
                  {h.what} <span className="muted">({h.stage})</span></li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      {messages.length > 0 && (
        <div style={{ overflowX: "auto", marginTop: 8 }}>
          <table className="small">
            <thead><tr><th>Stage</th><th>From</th><th>To</th><th>What</th><th>Size</th><th>Protection</th><th>First bytes</th></tr></thead>
            <tbody>
              {messages.map((m, i) => (
                <tr key={i}>
                  <td className="muted">{m.stage}</td><td>{name(m.from)}</td><td>{name(m.to)}</td><td>{m.what}</td>
                  <td className="mono">{size(m.bytes, m.count)}</td><td>{m.protection}</td>
                  <td className="mono">{m.hex ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {used.length > 0 && (
        <ul className="muted small" style={{ marginTop: 6 }}>
          {used.map((pr) => <li key={pr}><b>{pr}</b>: {PROTECTION_NOTE[pr] ?? pr}</li>)}
        </ul>
      )}
      {messages.length > 0 && <SequenceView usecase={usecase} events={events} config={config} />}
    </div>
  );
}

/** Side A, side B, then the many parties, then anyone else in the order met. */
function rank(s: string): number {
  return s === "a" ? 0 : s === "b" ? 1 : s === "many" ? 2 : 3;
}
