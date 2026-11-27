// Each page's published fp32 graphs on WebGPU beside WASM, in this tab (Round 86, batch 8; the owner's choice of
// 2026-11-27, checklist CS23). The rule, the tolerance and the timing protocol are lib/backendCompare.ts's; the runs are
// `compareBackends` in lib/onnxInference.ts, which opens its own two sessions and releases them, so the page's backend,
// sessions and figures are untouched. Nothing runs until the visitor asks: a comparison fetches a graph of up to about
// 100 MB and runs its whole sample twice.
import { useMemo, useState } from "react";
import { BACKEND_TOLERANCE, BATCH_REPEATS, ROW_REPEATS, fp32GraphFor, pageGraphOf, webgpuInapplicable } from "../lib/backendCompare";
import { type BackendComparison, browserModels, compareBackends } from "../lib/onnxInference";
import CardExport from "./CardExport";

const ms = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "n/a"
    : v >= 100 ? `${v.toFixed(0)} ms` : v >= 1 ? `${v.toFixed(1)} ms` : `${v.toPrecision(2)} ms`;
const sci = (v: number) => (v === 0 ? "0" : v.toExponential(2));

/** The four times of one backend, in the card's order. */
const times = (t: BackendComparison["wasm"]) =>
  t ? `${ms(t.createMs)} / ${ms(t.firstMs)} / ${ms(t.batchMs)} / ${ms(t.rowMs)}` : "n/a";

export default function BackendCompareCard({ usecase }: { usecase: string }) {
  const models = useMemo(() => browserModels(usecase).map(({ spec }) => {
    const graph = fp32GraphFor(spec.artifacts);
    const page = pageGraphOf(spec.artifacts)?.path ?? null;
    return { key: spec.key, label: spec.label, graph, page: page && page !== graph ? page : null,
             why: webgpuInapplicable((spec as { onnx_domains?: string[] }).onnx_domains, graph) };
  }), [usecase]);
  const [results, setResults] = useState<Record<string, BackendComparison>>({});
  const [busy, setBusy] = useState<string | null>(null);
  if (models.length === 0) return null;
  const runnable = models.filter((m) => !m.why);

  const run = async (keys: string[]) => {
    try {
      for (const key of keys) {
        setBusy(key);
        const r = await compareBackends(usecase, key);
        setResults((prev) => ({ ...prev, [key]: r }));
      }
    } finally {
      setBusy(null);
    }
  };
  const done = Object.values(results);
  const adapters = [...new Set(done.map((r) => r.adapter).filter((a): a is string => !!a))];
  const exportRows = done.map((r) => ({
    model: r.modelKey, graph: r.graph, page_graph: r.pageGraph, inapplicable: r.inapplicable, error: r.error,
    rows: r.rows, bytes: r.bytes, adapter: r.adapter, tolerance: r.out?.tolerance ?? BACKEND_TOLERANCE,
    max_abs_difference: r.out?.maxAbsDiff ?? null, max_abs_value: r.out?.maxAbs ?? null,
    decisions: r.out?.decisions ?? null, decisions_differ: r.out?.decisionsDiffer ?? null,
    near_ties: r.out?.nearTies ?? null, within_tolerance: r.out?.withinTolerance ?? null,
    wasm_create_ms: r.wasm?.createMs ?? null, wasm_first_ms: r.wasm?.firstMs ?? null,
    wasm_batch_ms: r.wasm?.batchMs ?? null, wasm_row_ms: r.wasm?.rowMs ?? null,
    webgpu_create_ms: r.webgpu?.createMs ?? null, webgpu_first_ms: r.webgpu?.firstMs ?? null,
    webgpu_batch_ms: r.webgpu?.batchMs ?? null, webgpu_row_ms: r.webgpu?.rowMs ?? null,
  }));

  return (
    <div className="card" id="backend-compare">
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
        <h3 style={{ margin: 0 }}>WebGPU beside WASM, on this page&apos;s fp32 graphs</h3>
        <CardExport name={`${usecase}-webgpu-beside-wasm`} label="the WebGPU and WASM comparison" rows={exportRows} />
      </div>
      <p className="muted small" style={{ marginTop: 4 }}>
        Each published fp32 graph on this page, run in this tab on WASM and then on WebGPU over every row of its published
        evaluation sample: every output value compared, and each row&apos;s decision by the sample&apos;s own rule.{" "}
        <b>Within tolerance</b> means every value within {BACKEND_TOLERANCE} of the WASM run&apos;s and no decision changed
        beyond a near tie. Times, on each backend: opening the session / the first run over every row (on WebGPU it
        includes compiling the shaders) / the median of {BATCH_REPEATS} more / the median of {ROW_REPEATS} runs on one
        row. A graph the page loads as int8 is compared as its fp32 twin, pinned at the same revision: ONNX Runtime
        Web&apos;s WebGPU provider has no kernel for the int8 graphs&apos; quantised nodes, so those graphs stay on WASM.
        Nothing here changes the backend the page runs on or any figure it shows.
      </p>
      {runnable.length > 0 ? (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", margin: "6px 0" }}>
          <button type="button" className="btn" disabled={busy != null}
            onClick={() => void run(runnable.map((m) => m.key))}>
            {busy ? `Comparing ${models.find((m) => m.key === busy)?.label ?? busy}...`
              : runnable.length === 1 ? "Compare it" : `Compare all ${runnable.length}, one after another`}
          </button>
        </div>
      ) : null}
      <div style={{ overflowX: "auto" }}>
        <table className="table">
          <thead>
            <tr>
              <th scope="col" style={{ textAlign: "left" }}>Model and graph</th>
              <th scope="col">Rows</th>
              <th scope="col">Largest |WebGPU - WASM|</th>
              <th scope="col">Decisions changed</th>
              <th scope="col">WASM</th>
              <th scope="col">WebGPU</th>
              <th scope="col">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const r = results[m.key];
              const o = r?.out;
              return (
                <tr key={m.key}>
                  <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{m.label}<br />
                    <span className="muted small">
                      {m.graph ?? "no fp32 graph"}{m.page ? ` (the page loads ${m.page}, which stays on WASM)` : ""}
                    </span></th>
                  {m.why ? (
                    <td colSpan={6} className="small" style={{ textAlign: "left" }}>Not compared: {m.why}.</td>
                  ) : !r ? (
                    <td colSpan={6} className="small" style={{ textAlign: "left" }}>
                      <button type="button" className="btn ghost" disabled={busy != null}
                        onClick={() => void run([m.key])}>{busy === m.key ? "Comparing..." : "Compare"}</button>
                    </td>
                  ) : r.inapplicable || r.error ? (
                    <td colSpan={6} className="small" style={{ textAlign: "left", color: r.error ? "var(--danger)" : undefined }}>
                      {r.error ? `It did not run: ${r.error}` : `Not compared: ${r.inapplicable}.`}
                    </td>
                  ) : (
                    <>
                      <td>{r.rows}</td>
                      <td>{o ? `${sci(o.maxAbsDiff)} (largest |value| ${o.maxAbs.toPrecision(3)})` : "n/a"}</td>
                      <td>{o ? (o.decisions ? `${o.decisionsDiffer} of ${o.decisions}${o.nearTies ? `, ${o.nearTies} near tie(s)` : ""}`
                        : "none decided (a forecast)") : "n/a"}</td>
                      <td className="small">{times(r.wasm)}</td>
                      <td className="small">{times(r.webgpu)}</td>
                      <td><span className={`badge-state ${o?.withinTolerance ? "done" : "error"}`}>
                        {o?.withinTolerance ? "within tolerance" : "OVER"}</span></td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {adapters.length ? (
        <p className="muted small" style={{ marginTop: 6 }}>WebGPU ran on {adapters.join("; ")}, as this browser names it.</p>
      ) : null}
    </div>
  );
}
