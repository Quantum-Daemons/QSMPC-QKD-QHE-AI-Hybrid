// Live event log + metrics table + ML-DSA signature note. Shared by Orchestration + use-case pages.
import type { PipelineEvent } from "../lib/api";

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
          {events.map((e, i) => (
            <div key={i} className="log-line">
              <span className="t">[{e.t?.toFixed?.(3)}s]</span>{" "}
              <span className={e.type}>{e.type}</span>{" "}
              <span className="muted">{e.stage}</span> {e.message}
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <h3>Metrics</h3>
        <table>
          <tbody>
            {Object.entries(metrics).map(([k, v]) => (
              <tr key={k}><td className="muted">{k}</td><td className="mono">{String(v)}</td></tr>
            ))}
            {/* `colSpan={2}`. Every other row in this table has two cells; this one had one, so a
                screen reader announced a row with a missing second column - it reads as data that
                failed to load rather than as an empty table. SC 1.3.1: the row's structure has to
                match the structure it is announced within. */}
            {Object.keys(metrics).length === 0 && (
              <tr><td className="muted" colSpan={2}>no metrics yet</td></tr>
            )}
          </tbody>
        </table>
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
