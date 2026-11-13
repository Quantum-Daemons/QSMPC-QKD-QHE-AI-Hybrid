// Each page's published figures beside this project's offline figure and this run (Round 86, the owner's choice of
// 2026-11-13). The rows and their verdicts come from lib/publishedComparisons.ts, where each verdict's rule is
// written; this card only draws them, and the downloads hold the same rows.
import type { ComparisonRow } from "../lib/publishedComparisons";
import CardExport from "./CardExport";

const VERDICT_CLASS: Record<ComparisonRow["verdict"], string> = {
  reproduced: "done", differs: "error", "no spread": "idle", reference: "idle", "not comparable": "idle",
};

export default function PublishedComparisonCard({ usecase, rows, note }: {
  usecase: string; rows: ComparisonRow[]; note?: string;
}) {
  return (
    <div className="card" id="published-comparison">
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
        <h3 style={{ margin: 0 }}>Published figures beside this computation</h3>
        <CardExport name={`${usecase}-published-comparison`} label="the published-figure comparison"
          rows={rows as unknown as Record<string, unknown>[]} />
      </div>
      <p className="muted small" style={{ marginTop: 4 }}>
        Each row: the figure as its source publishes it, this project&apos;s offline measurement, what this tab computed,
        and a verdict. <b>Reproduced</b> and <b>differs</b> are for the same method on the same data, judged by a
        published or measured spread; <b>no spread</b> shows the difference where neither side published one;{" "}
        <b>reference</b> is a different method or protocol on the same data, never a claim of reproduction; and{" "}
        <b>not comparable</b> says why. A model&apos;s run in this tab re-scores its published evaluation sample, which is
        not its whole test split.
      </p>
      {note && <p className="small" style={{ marginTop: 6 }}>{note}</p>}
      {rows.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th scope="col" style={{ textAlign: "left" }}>Figure</th>
                <th scope="col">Published</th>
                <th scope="col">This project, offline</th>
                <th scope="col">This run</th>
                <th scope="col">Verdict</th>
                <th scope="col" style={{ textAlign: "left" }}>Why</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.figure}>
                  <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{r.figure}<br />
                    <span className="muted small">{r.source}</span></th>
                  <td>{r.published}</td>
                  <td>{r.offline}</td>
                  <td style={{ textAlign: "left" }}>{r.thisRun}</td>
                  <td><span className={`badge-state ${VERDICT_CLASS[r.verdict]}`}>{r.verdict}</span></td>
                  <td style={{ textAlign: "left" }} className="small">{r.why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
