// F1 by the receiving account's in-degree, in SALT-GNN's bins (Round 86, the owner's choice of 2026-11-15). Computed
// offline by scripts/research/finance_degree_f1.py from IBM's runs' test and validation scores, five seeds per protocol,
// and served as a static file (web/public/data/finance/degree-f1.json, the record in docs/measurements byte for byte).
// Nothing here runs in the tab. The file is fetched when the IBM tab is opened, so a visitor who never opens it does
// not download it.
import { useEffect, useRef, useState } from "react";
import StageBadge from "./StageBadge";

type Win = "all" | "batch";
type Thr = "validation" | "0.5";
interface Bin { bin: string; f1_mean: number; f1_sd: number; f1_runs: number[]; interval_95: [number, number] }
interface ByThreshold { overall_mean: number; overall_sd: number; bins: Bin[] }
interface Family { label: string; native_window: Win; runs: { run: string; unscored: number; unscored_positives: number }[];
  by_window: Record<Win, Record<Thr, ByThreshold>> }
interface Published { source: string; caveat: string; bins: string[]; rows: { model: string; f1: [number, number][] }[] }
interface DegreeRecord { test_rows: number; test_positives: number; method: { windows: Record<Win, string> };
  bins_by_window: Record<Win, { bin: string; rows: number; positives: number }[]>; families: Record<string, Family>;
  published: Published }

const PATH = `${import.meta.env.BASE_URL}data/finance/degree-f1.json`;
const ORDER = ["pna_ibm", "pna_causal", "gin_ibm", "gin_causal"];
const SHORT: Record<string, string> = {
  pna_ibm: "PNA, IBM's protocol", pna_causal: "PNA, causal", gin_ibm: "GIN, IBM's protocol", gin_causal: "GIN, causal",
};
const f3 = (x: number) => x.toFixed(3);
const pm = (m: number, s: number | null | undefined) => (s == null ? f3(m) : `${f3(m)} ± ${f3(s)}`);

export default function DegreeF1Table() {
  const box = useRef<HTMLDivElement>(null);
  const [wanted, setWanted] = useState(false);
  const [rec, setRec] = useState<DegreeRecord | string | null>(null);
  const [thr, setThr] = useState<Thr>("validation");
  const [win, setWin] = useState<"native" | Win>("native");
  useEffect(() => {
    const el = box.current;
    if (!el || wanted) return;
    // Loaded when the IBM tab is shown, however tall the window. Its panel is mounted hidden (Tabs, keepMounted), and a
    // hidden element has no box, so the first time the element has one is the tab being opened, which a ResizeObserver
    // reports. The first version waited for an IntersectionObserver with a 6000 px margin: the table, at the foot of the
    // tab, was further than that below a 621 px tall window, and stayed on its loading line until scrolled to (the
    // deployed site, 2026-11-20), so a keyboard or screen-reader visitor who did not scroll never got it.
    if (el.getClientRects().length || typeof ResizeObserver === "undefined") { setWanted(true); return; }
    const ro = new ResizeObserver(() => { if (el.getClientRects().length) setWanted(true); });
    ro.observe(el);
    return () => ro.disconnect();
  }, [wanted]);
  useEffect(() => {
    if (!wanted) return;
    let live = true;
    fetch(PATH).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => { if (live) setRec(j as DegreeRecord); })
      .catch((e) => { if (live) setRec(String((e as Error)?.message ?? e)); });
    return () => { live = false; };
  }, [wanted]);

  const windowOf = (key: string): Win => (win === "native" ? (rec as DegreeRecord).families[key].native_window : win);
  return (
    <div ref={box}>
      <strong>F1 by the receiving account&apos;s in-degree <StageBadge kind="offline" /></strong>
      <p className="small" style={{ marginTop: 4 }}>
        Each test transfer is binned by how many transfers its receiver gets, in the bins of SALT-GNN (Losavio et al.,{" "}
        <a href="https://arxiv.org/abs/2607.10131" target="_blank" rel="noopener noreferrer">arXiv:2607.10131</a>, its
        Algorithm 1), which reports that graph models lose F1 where receivers are dense. Computed offline from IBM&apos;s
        runs&apos; scores, five seeds per protocol; nothing here runs in your tab. Under IBM&apos;s protocol a transfer is
        binned by every transfer of the file, as IBM&apos;s test graph holds them, future ones included; under the causal
        protocol, by the transfers up to the end of its batch, all that protocol lets it read.
      </p>
      {rec === null ? <p className="muted small">Loading the table...</p>
        : typeof rec === "string" ? <p className="small" style={{ color: "var(--danger)" }}>The table did not load: {rec}</p>
        : (
          <>
            <div className="small" style={{ display: "flex", flexWrap: "wrap", gap: 14, margin: "6px 0" }}>
              <label>Threshold{" "}
                <select value={thr} onChange={(e) => setThr(e.target.value as Thr)}>
                  <option value="validation">best on each run&apos;s validation scores (this page)</option>
                  <option value="0.5">0.5 (SALT-GNN&apos;s)</option>
                </select>
              </label>
              <label>Degree counted over{" "}
                <select value={win} onChange={(e) => setWin(e.target.value as "native" | Win)}>
                  <option value="native">each protocol&apos;s own window</option>
                  <option value="all">every transfer (IBM&apos;s test graph)</option>
                  <option value="batch">transfers up to the batch end (causal)</option>
                </select>
              </label>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table className="table small">
                <thead>
                  <tr><th>In-degree</th><th>Test transfers (laundering), every transfer / to the batch end</th>
                    {ORDER.map((k) => <th key={k}>{SHORT[k]}</th>)}</tr>
                </thead>
                <tbody>
                  {rec.bins_by_window.all.map((b, i) => (
                    <tr key={b.bin}>
                      <td>{b.bin}</td>
                      <td>{b.rows.toLocaleString("en-US")} ({b.positives}) / {rec.bins_by_window.batch[i].rows.toLocaleString("en-US")} ({rec.bins_by_window.batch[i].positives})</td>
                      {ORDER.map((k) => {
                        const c = rec.families[k].by_window[windowOf(k)][thr].bins[i];
                        return <td key={k} title={`95% interval ${f3(c.interval_95[0])} to ${f3(c.interval_95[1])}; runs ${c.f1_runs.map(f3).join(", ")}`}>{pm(c.f1_mean, c.f1_sd)}</td>;
                      })}
                    </tr>
                  ))}
                  <tr>
                    <td>all</td><td>{rec.test_rows.toLocaleString("en-US")} ({rec.test_positives})</td>
                    {ORDER.map((k) => { const o = rec.families[k].by_window[windowOf(k)][thr]; return <td key={k}>{pm(o.overall_mean, o.overall_sd)}</td>; })}
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="muted small" style={{ marginTop: 6 }}>
              F1 as the mean ± standard deviation over five runs; hover a cell for its 95% bootstrap interval over the
              bin&apos;s transfers and each run&apos;s value. The 100+ bin holds a single laundering transfer, so its F1 says
              nothing. Each run left 859 to 895 of the {rec.test_rows.toLocaleString("en-US")} test transfers unscored (its
              sampler&apos;s draw, at most one of them laundering); they count as not flagged.
            </p>
            <div style={{ overflowX: "auto", marginTop: 8 }}>
              <table className="table small">
                <thead>
                  <tr><th>SALT-GNN&apos;s Table 2, HI-Small, F1 at 0.5</th>{rec.published.bins.map((b) => <th key={b}>{b}</th>)}</tr>
                </thead>
                <tbody>
                  {rec.published.rows.map((r) => (
                    <tr key={r.model}><td>{r.model}</td>{r.f1.map(([m, s], i) => <td key={i}>{pm(m, s)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted small" style={{ marginTop: 6 }}>
              The paper&apos;s figures, for reference: {rec.published.caveat}. Its PNA and GIN use IBM&apos;s published
              settings (GIN 64 wide where IBM&apos;s file says 66), so set the threshold above to 0.5 to read the rows
              closest to them. Its code is not released, so no SALT model is reproduced here.
            </p>
          </>
        )}
    </div>
  );
}
