// Every published model on the page, re-scored in this tab (Round 86, the owner's choice of 2026-11-13). The run has
// verified all of them since Round 71, and only the selected one reached the page; the rest reached the log alone.
import type { VerificationSummary } from "../lib/clientCompute";
import { verificationRows } from "../lib/cardRows";
import CardExport from "./CardExport";

const f4 = (x: number | undefined) => (x == null ? "-" : x.toFixed(4));

export default function VerificationsCard({ usecase, list }: { usecase: string; list: VerificationSummary[] }) {
  if (list.length === 0) return null;
  return (
    <div className="card" id="verifications">
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
        <h3 style={{ margin: 0 }}>Every published model, re-scored in this tab</h3>
        <CardExport name={`${usecase}-verifications`} label="every model's re-scoring" rows={verificationRows(list)}
          json={list} />
      </div>
      <p className="muted small" style={{ marginTop: 4 }}>
        Each model&apos;s published evaluation sample, scored here and compared with the figure recorded with that sample;
        row by row where the sample carries a per-row record. A sample is not the model&apos;s whole test split.
      </p>
      <table className="table">
        <thead>
          <tr>
            <th scope="col" style={{ textAlign: "left" }}>Model</th>
            <th scope="col">Outcome</th>
            <th scope="col">Metric</th>
            <th scope="col">This tab</th>
            <th scope="col">Recorded</th>
            <th scope="col">Rows (split)</th>
            <th scope="col">Rows as recorded</th>
          </tr>
        </thead>
        <tbody>
          {list.map((v) => (
            <tr key={v.modelKey}>
              <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>
                {v.modelKey}{v.selected ? " (selected)" : ""}</th>
              <td>{v.outcome}</td>
              <td>{v.metric ?? "-"}</td>
              <td>{f4(v.score)}</td>
              <td>{f4(v.expected)}</td>
              <td>{v.n != null ? `${v.n} (${v.split ?? "split not declared"})` : "-"}</td>
              <td>{v.rows != null ? `${v.rowsAsRecorded} of ${v.rows}${v.nearTies ? `, ${v.nearTies} near tie(s)` : ""}`
                : v.perRowRefused ?? v.noSampleReason ?? v.error ?? "aggregate only"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
