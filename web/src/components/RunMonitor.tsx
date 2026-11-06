// Live event log + metrics table + ML-DSA signature note. Shared by Orchestration + use-case pages.
import { Fragment } from "react";

import type { PipelineEvent } from "../lib/api";

/** A metric's name, breakable after each underscore (Round 80). The longest, such as
 *  "encrypted_tier_within_operational_window", are 268 px of unbroken text, and with them the table
 *  was 5 px wider than its card, which scrolled sideways. `<wbr>`, not a zero-width space, so a
 *  name copied from the table is the name. */
const breakable = (k: string) => k.split("_").map((part, i, all) => (
  <Fragment key={i}>{part}{i < all.length - 1 && <>_<wbr /></>}</Fragment>
));

/** What a log line says (UX19, Round 86). The browser emits a metric with its name and value in `data` and no message,
 *  so most of a run's lines read a bare "metric <stage>". The value is written as the metrics table would hold it,
 *  numbers to six significant figures and long text cut. */
export function lineText(e: PipelineEvent): string {
  if (e.message || e.type !== "metric" || typeof e.data?.name !== "string") return e.message;
  const v = e.data.value;
  const shown = typeof v === "number" ? String(Number.isInteger(v) ? v : Number(v.toPrecision(6)))
    : typeof v === "string" ? (v.length > 120 ? `${v.slice(0, 117)}...` : v)
    : v == null ? String(v) : (JSON.stringify(v) ?? String(v)).slice(0, 120);
  return `${e.data.name} = ${shown}`;
}

export default function RunMonitor({
  events, metrics, signature, logRef,
}: {
  events: PipelineEvent[];
  metrics: Record<string, any>;
  signature: any;
  logRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <div className="grid cols-2" style={{ marginTop: 16 }}>
      <div className="card">
        <h3>Live event log</h3>
        <div className="log" ref={logRef as React.RefObject<HTMLDivElement>}>
          {events.length === 0 && <span className="muted">Press ▶ Start to stream events…</span>}
          {/* `ev-` before the event type. The bare type was the class until Round 80, and a "log"
              event's label took `.log`, this container's own rule: a bordered, padded box on an
              inline span, painted over the lines above and below it, and most of a run's events
              are "log". A prefix keeps any event type from naming a layout class. */}
          {events.map((e, i) => (
            <div key={i} className="log-line">
              <span className="t">[{e.t?.toFixed?.(3)}s]</span>{" "}
              <span className={`ev-${e.type}`}>{e.type}</span>{" "}
              <span className="muted">{e.stage}</span> {lineText(e)}
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <h3>Metrics</h3>
        {/* A key-value table whose keys are row headers (SC 1.3.1, Round 86): the audit of the deployed
            pages found this table, the largest on a use-case page, with no header cell at all. Before a
            run there is nothing to tabulate, so the empty state is a sentence rather than a one-cell
            table, which a screen reader announced as a table with a missing column. */}
        {Object.keys(metrics).length === 0 ? (
          <p className="muted">no metrics yet</p>
        ) : (
          <table>
            <tbody>
              {Object.entries(metrics).map(([k, v]) => (
                <tr key={k}><th scope="row" className="muted">{breakable(k)}</th>
                  <td className="mono" style={{ overflowWrap: "anywhere" }}>{String(v)}</td></tr>
              ))}
            </tbody>
          </table>
        )}
        {signature && (
          <div className="note" style={{ marginTop: 10 }}>
            🔏 Run summary signed with <b>{signature.algorithm}</b> — verified:{" "}
            <b style={{ color: signature.verified ? "var(--accent-2)" : "var(--danger)" }}>
              {String(signature.verified)}
            </b> ({signature.signature_bytes} B)
          </div>
        )}
      </div>
    </div>
  );
}
