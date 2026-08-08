// Unified benchmark results panel (Round 7). It is CONTROLLED: the per-use-case page's single "Run"
// drives the live E2E pipeline and the assembled benchmark is passed in via `result` — there is no
// separate "Run benchmark" button. The MPC-FHE row is the live run's real measured FHE-linear +
// MPC-non-linear split; the QSMPC-FHE row + quantum-advantage + security are the comparison.
// `compact` (telecom/satellite, no private inference) shows the security + advantage only.
import { useRef } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { downloadCSV, downloadJSON, downloadNodePNG } from "../lib/export";

const colorFor = (path: string) => (path === "QSMPC-FHE" ? "#b072ff" : "#4f9dff");
const FHE_COLOR = "#4f9dff";   // linear → FHE (CKKS)
const MPC_COLOR = "#f5a623";   // non-linear → MPC (secret-sharing)

export default function BenchmarkPanel(
  { compact = false, result = null, clientComputeMs = null }:
  { compact?: boolean; result?: any; clientComputeMs?: number | null },
) {
  const chartRef = useRef<HTMLDivElement>(null);
  const rows: any[] = result?.results ?? [];
  const advantage = result?.advantage ?? null;
  const workload = result?.workload ?? "linear";

  // Bars are labelled by PATH, not by backend.
  //
  // `backend` is a descriptive sentence - "node-seal CKKS + emp-wasm 2PC (measured: real 4-channel
  // encrypted convolution, 16 rotations over 196 packed patches)" - and as a category label it is
  // unusable: the latency chart rendered three bars and exactly one readable axis label, the short
  // one, so two of the three bars could not be identified at all. `path` is short, distinct, and is
  // already the first column of the Detailed results table below, so the chart and the table now
  // name the same rows the same way. The backend still reaches the reader through that table.
  const ratioData = rows.filter((r) => r.ciphertext_plaintext_ratio != null)
    .map((r) => ({ name: r.path, ratio: r.ciphertext_plaintext_ratio, path: r.path }));
  const latData = rows.map((r) => ({ name: r.path, latency: r.latency_ms, path: r.path }));
  const secData = rows.filter((r) => r.security_score != null)
    .map((r) => ({ name: r.path, score: r.security_score, path: r.path }));
  // hybrid split — the MPC-FHE row carries the linear-FHE vs non-linear-MPC breakdown
  const split = rows.find((r) => r.fhe_linear_ms != null && r.mpc_nonlinear_ms != null);
  const splitData = split
    ? [{ name: "linear → FHE", ms: split.fhe_linear_ms, kind: "fhe" },
       { name: "non-linear → MPC", ms: split.mpc_nonlinear_ms, kind: "mpc" }]
    : [];

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
        <h3 style={{ marginTop: 0 }}>
          {compact ? "Security score & quantum advantage" : "Benchmark — MPC-FHE vs QSMPC-FHE (from this run)"}
        </h3>
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          {result?.live && <span className="tag" style={{ borderColor: "#b072ff", color: "#b072ff" }}>live run</span>}
          {rows.length > 0 && (
            <>
              <button className="btn ghost" onClick={() => downloadCSV(rows, `benchmark_${workload}.csv`)}>⬇ CSV</button>
              <button className="btn ghost" onClick={() => downloadJSON(result, `benchmark_${workload}.json`)}>⬇ JSON</button>
              <button className="btn ghost" onClick={() => chartRef.current && downloadNodePNG(chartRef.current, `benchmark_${workload}.png`)}>⬇ PNG</button>
            </>
          )}
        </div>
      </div>

      {rows.length === 0 && (
        <p className="muted" style={{ margin: "8px 0 0" }}>
          Press <b>▶ Run</b> above — the benchmark fills in from that live run (computation time, accuracy,
          security score{compact ? "" : ", FHE-linear / MPC-non-linear split"} and quantum advantage). One run,
          one result.
        </p>
      )}

      {advantage && (
        <div className="card" style={{ marginTop: 12, borderColor: "#b072ff" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 20, flexWrap: "wrap" }}>
            <div style={{ textAlign: "center", minWidth: 120 }}>
              <div style={{ fontSize: 38, fontWeight: 700, color: "#b072ff", lineHeight: 1 }}>
                {advantage.score}<span style={{ fontSize: 16, color: "#97a3b6" }}>/10</span>
              </div>
              <div className="tag" style={{ marginTop: 6 }}>{advantage.tier} advantage</div>
            </div>
            <div style={{ flex: 1, minWidth: 240 }}>
              <h3 style={{ marginTop: 0 }}>Quantum advantage</h3>
              <p style={{ margin: "4px 0" }}>{advantage.interpretation}</p>
              <p className="muted" style={{ fontSize: 12, margin: 0 }}>
                communication ×{advantage.communication_ratio} · latency ×{advantage.latency_ratio} ·
                security ×{advantage.security_multiplier}
              </p>
            </div>
          </div>
        </div>
      )}

      {rows.length > 0 && (() => {
        const q = rows.find((r: any) => r.path === "QSMPC-FHE");
        return (
          <p className="muted" style={{ fontSize: 11, margin: "8px 0 0", lineHeight: 1.5 }}>
            <b>How these are produced (this run):</b> the <span style={{ color: "#b072ff" }}>QSMPC-FHE</span> row is
            a genuine <b>state-vector simulation</b> of a QOTP + Clifford+T QCNN private inference run by the
            <code>quantum_comparison</code> stage — its <b>latency is measured</b> and its
            {q?.t_gate_count != null ? ` ${q.t_gate_count} T-gates / ${q.qubits} qubits are counted from the circuit` : " T-gates are counted"}
            {" "}(simulated, not quantum hardware; ct/pt = 1 is QOTP's perfect-hiding property). See the
            “QSMPC-FHE comparison” bar in the diagram above. <b>Quantum advantage</b> = a transparent weighted
            geometric mean of three ratios (communication-compactness ≈10³× from ct/pt, latency, and the
            computational→information-theoretic security jump), tiered 0–10. <b>Security score</b> (0–100) is a
            weighted sum over PQC-hybrid/HNDL, security model (information-theoretic {">"} computational), NIST
            KEM category, QBER eavesdrop-detection, and the ~0.1 bit/weight leakage bound (PRX 2025).
          </p>
        );
      })()}

      {rows.length > 0 && (
        <div ref={chartRef}>
          {!compact && (
            <div className="grid cols-2" style={{ marginTop: 14 }}>
              <div className="card">
                <h3>Ciphertext / plaintext ratio (log)</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={ratioData}>
                    <CartesianGrid stroke="#243049" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#97a3b6" }} />
                    <YAxis scale="log" domain={[0.5, "auto"]} tick={{ fontSize: 11, fill: "#97a3b6" }} allowDataOverflow />
                    <Tooltip contentStyle={{ background: "#151c2c", border: "1px solid #243049" }} />
                    <Bar dataKey="ratio">{ratioData.map((d, i) => <Cell key={i} fill={colorFor(d.path)} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="card">
                <h3>Latency per inference (ms)</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={latData}>
                    <CartesianGrid stroke="#243049" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#97a3b6" }} />
                    <YAxis tick={{ fontSize: 11, fill: "#97a3b6" }} />
                    <Tooltip contentStyle={{ background: "#151c2c", border: "1px solid #243049" }} />
                    <Legend />
                    <Bar dataKey="latency">{latData.map((d, i) => <Cell key={i} fill={colorFor(d.path)} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
          {!compact && split && (
            <div className="card" style={{ marginTop: 14, borderColor: MPC_COLOR }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
                <h3 style={{ margin: 0 }}>Hybrid split — linear → FHE · non-linear → MPC</h3>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  <span className="tag">{(split.device || "cpu").toUpperCase()}</span>
                  <span className="tag">{split.model_size}</span>
                  {split.projected && <span className="tag" style={{ borderColor: MPC_COLOR, color: MPC_COLOR }}>projected</span>}
                </div>
              </div>
              {split.model_arch && (
                <p style={{ fontSize: 12, margin: "4px 0 2px" }}>
                  Model: <b>{split.model_arch}</b>
                  {split.model_params != null ? <span className="muted"> · {split.model_params.toLocaleString()} params</span> : null}
                </p>
              )}
              <p className="muted" style={{ fontSize: 12, margin: "2px 0 10px" }}>
                Linear layers (matmul/conv) run under CKKS/FHE{split.device === "gpu" ? " (GPU)" : ""}; non-linear
                layers (ReLU/argmax) run under MPC — MPyC secret sharing on the server, and in the browser <code>emp-wasm-engine</code> garbled circuits (MPyC is pure Python and is not in the bundle). {split.backend}.
              </p>
              <div className="grid cols-2">
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={splitData}>
                    <CartesianGrid stroke="#243049" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#97a3b6" }} />
                    <YAxis tick={{ fontSize: 11, fill: "#97a3b6" }} label={{ value: "ms", angle: -90, position: "insideLeft", fill: "#97a3b6", fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: "#151c2c", border: "1px solid #243049" }} />
                    <Bar dataKey="ms">{splitData.map((d, i) => <Cell key={i} fill={d.kind === "fhe" ? FHE_COLOR : MPC_COLOR} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", gap: 10 }}>
                  <div><span style={{ color: FHE_COLOR }}>■</span> FHE-linear: <b>{split.fhe_linear_ms} ms</b></div>
                  <div><span style={{ color: MPC_COLOR }}>■</span> MPC-non-linear: <b>{split.mpc_nonlinear_ms} ms</b></div>
                  <div className="muted">MPC communication: <b>{split.mpc_comm_kb ?? "—"} KB</b></div>
                  <div className="muted">total: <b>{split.latency_ms} ms</b> / inference</div>
                  {clientComputeMs != null && (
                    <div className="muted">client compute (wall-clock): <b>{clientComputeMs} ms</b></div>
                  )}
                </div>
              </div>
            </div>
          )}
          {secData.length > 0 && (
            <div className="card" style={{ marginTop: 14 }}>
              <h3>Security score (0–100)</h3>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={secData} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid stroke="#243049" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: "#97a3b6" }} />
                  <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11, fill: "#97a3b6" }} />
                  <Tooltip contentStyle={{ background: "#151c2c", border: "1px solid #243049" }} />
                  <Bar dataKey="score">{secData.map((d, i) => <Cell key={i} fill={colorFor(d.path)} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {!compact && (
            <div className="card" style={{ marginTop: 14 }}>
              <h3>Detailed results</h3>
              <table>
                {/* The last column was headed "model" and rendered `security_model`, so the table
                    had no model column at all while appearing to have one. Both are shown now, and
                    the model one matters here: these rows are not all the same model. */}
                <thead><tr><th>path</th><th>model</th><th>backend</th><th>size</th><th>latency ms</th><th>FHE-lin ms</th><th>MPC-nl ms</th><th>comm KB</th><th>accuracy</th><th>ct/pt</th><th>security</th><th>security model</th></tr></thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i}>
                      <td style={{ color: colorFor(r.path) }}>{r.path}</td>
                      <td style={{ fontSize: 11 }}>{r.model_arch ?? "—"}</td>
                      <td>{r.backend}</td>
                      <td>{r.model_size ?? "—"}{r.projected ? "*" : ""}</td>
                      <td>{r.latency_ms}</td>
                      <td style={{ color: FHE_COLOR }}>{r.fhe_linear_ms ?? "—"}</td>
                      <td style={{ color: MPC_COLOR }}>{r.mpc_nonlinear_ms ?? "—"}</td>
                      <td>{r.mpc_comm_kb ?? "—"}</td>
                      <td>{r.accuracy ?? "—"}</td><td>{r.ciphertext_plaintext_ratio ?? "—"}</td>
                      <td><b>{r.security_score ?? "—"}</b></td><td style={{ fontSize: 11 }}>{r.security_model}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.some((r) => r.projected) && (
                <p className="muted" style={{ fontSize: 11, margin: "6px 0 0" }}>
                  * large tier is projected from measured medium-tier per-op cost (MACs / comparison counts), not run end-to-end.
                </p>
              )}
              {rows.some((r) => r.path === "plaintext") && (
                <p className="muted" style={{ fontSize: 11, margin: "6px 0 0" }}>
                  The plaintext row is a reference point, not a denominator. It is a different model
                  from the encrypted rows — the published Hugging Face model on its own corpus, against
                  the distilled student the FHE path evaluates — so dividing one latency by the other
                  does not give an FHE overhead. Compare the model column before comparing the times.
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
