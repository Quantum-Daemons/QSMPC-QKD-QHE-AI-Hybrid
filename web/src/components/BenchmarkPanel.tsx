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
/** Bar fill: the same colour as `colorFor`, plus a hatch on QSMPC-FHE so the path is not encoded by
 *  colour alone (WCAG 2.2 SC 1.4.1, Level A). `suffix` picks the per-chart pattern id, because an SVG
 *  pattern is resolved per document and two charts sharing one id is a real rendering hazard. */
const fillFor = (path: string, suffix = "") =>
  (path === "QSMPC-FHE" ? `url(#hatch-qsmpc${suffix})` : colorFor(path));
const FHE_COLOR = "#4f9dff";   // linear → FHE (CKKS)
const MPC_COLOR = "#f5a623";   // non-linear → MPC (secret-sharing)

export default function BenchmarkPanel(
  { compact = false, result = null, clientComputeMs = null }:
  { compact?: boolean; result?: any; clientComputeMs?: number | null },
) {
  const chartRef = useRef<HTMLDivElement>(null);
  const rows: any[] = result?.results ?? [];
  const facts: any[] = result?.security_facts ?? [];
  const workload = result?.workload ?? "linear";

  // Bars are labelled by PATH, not by backend.
  //
  // `backend` is a descriptive sentence - "node-seal CKKS + emp-wasm 2PC (measured: real 4-channel
  // encrypted convolution, 16 rotations over 196 packed patches)" - and as a category label it is
  // unusable: the latency chart rendered three bars and exactly one readable axis label, the short
  // one, so two of the three bars could not be identified at all. `path` is short, distinct, and is
  // already the first column of the Detailed results table below, so the chart and the table now
  // name the same rows the same way. The backend still reaches the reader through that table.
  // The QSMPC-FHE row's ct/pt is NEVER rendered, whichever transport produced it. The browser row
  // carries null; the Python row still carries a byte ratio of a SIMULATED state vector (1.2769 at
  // 8 qubits) for API compatibility. Neither is comparable with the CKKS byte ratio beside it, and
  // rendering either recreates the comparison the retired overview card and score made.
  const ctpt = (r: any) => (r.path === "QSMPC-FHE" ? null : r.ciphertext_plaintext_ratio ?? null);
  const ratioData = rows.filter((r) => ctpt(r) != null)
    .map((r) => ({ name: r.path, ratio: ctpt(r), path: r.path }));
  const latData = rows.map((r) => ({ name: r.path, latency: r.latency_ms, path: r.path }));
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
          {compact ? "Security posture" : "Benchmark — MPC-FHE and QSMPC-FHE rows (from this run)"}
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
          the security posture facts{compact ? "" : " and the FHE-linear / MPC-non-linear split"}). One run,
          one result.
        </p>
      )}

      {/* THE QUANTUM-ADVANTAGE SCORE THAT STOOD HERE IS RETIRED AND NOT RENDERED (2026-10-23).
          It printed "N/10", a tier, and "Quantum path is ~6146x more communication-compact and 5794x
          faster; dominant advantage". Its inputs did not measure the same thing on the two sides: a
          CKKS byte ratio over a QOTP qubit ratio that omits the 2n-bit key, a real CKKS + MPC
          inference over an 8-qubit simulation of an untrained circuit (the ratio moved 5794x, 8163x,
          9610x across three runs), and a 1.5x multiplier for a security label the run does not
          achieve. Same retirement, same reason, as the 0-100 composite below. */}

      {rows.length > 0 && (() => {
        const q = rows.find((r: any) => r.path === "QSMPC-FHE");
        return (
          <p className="muted" style={{ fontSize: 11, margin: "8px 0 0", lineHeight: 1.5 }}>
            <b>How these are produced (this run):</b> the <span style={{ color: "#b072ff" }}>QSMPC-FHE</span> row is
            a genuine <b>state-vector simulation</b> of a QOTP-protected parameterised RY/RZ/CNOT circuit run by the{" "}
            <code>quantum_comparison</code> stage — its <b>latency is measured</b> and its
            {/* "counted from the circuit" is true of the BROWSER row and false of the Python one.
                `qsh/qhe/qcnn.py` builds no T gates; its `t_count()` is a Solovay-Kitaev budget,
                ceil(3*log2(1/eps)) per rotation times the parameter count, returning 630 at 8 qubits.
                The browser path counts 56 from an actual gate list. One table, two meanings, and the
                caption asserted the stronger one for both. */}
            {q?.t_gate_count != null
              ? ` ${q.t_gate_count} T-gates / ${q.qubits} qubits`
                + (q.t_gate_count > 200 ? " (a Solovay-Kitaev BUDGET over the rotations, not a gate count)"
                                        : " are counted from the circuit")
              : " T-gates are estimated"}
            {" "}(simulated, not quantum hardware). <b>Its ct/pt is shown as a dash:</b> a quantum ciphertext
            has no size in bytes - the one-time pad maps n qubits to n qubits plus a 2n-bit classical key - so it
            cannot share a column with the CKKS byte ratio. <b>The
            security model named in that row is the SCHEME's, not this run's:</b> the quantum one-time pad
            is perfectly hiding when its key bits are uniform and secret, and this simulation draws them
            from a seeded non-cryptographic generator whose seed is on this page — so the mask is real and
            recomputable by anyone. The gate count, the width and the latency are measurements; the secrecy
            is not. See the
            “QSMPC-FHE comparison” bar in the diagram above. <b>Two scores that used to sit here are retired
            and no longer rendered.</b> The <b>quantum-advantage</b> score divided quantities that are not
            the same on the two sides - a ciphertext byte ratio by a qubit ratio, and a real encrypted
            inference by a simulation of an untrained circuit - and classical FHE also reaches rate 1
            (Gentry and Halevi, TCC 2019), so the gap it reported was not a quantum advantage. The 0–100
            composite: 14 of its 17 weights carried no stated rationale, it scored a zero-secure-key run
            100/100, and the 68-vs-96 gap it displayed was mostly an artefact of which arguments two functions
            happened to pass. The <b>Security posture</b> table above replaces it with per-row facts, each
            traceable to FIPS 203/204 or to this run.
          </p>
        );
      })()}

      {rows.length > 0 && (
        <div ref={chartRef}>
          {!compact && (
            <div className="grid cols-2" style={{ marginTop: 14 }}>
              <div className="card">
                <h3>CKKS ciphertext / plaintext size, this configuration (log)</h3>
                {/* Only the CKKS row is plotted. Its ratio is ciphertext bytes / (inputs x 8 bytes), so it
                    measures SLOT UTILISATION: the same N = 8192 ciphertext is about 12x against 4,096
                    packed values and about 1,500x against 33 (Cheon, Kim, Kim, Song, ASIACRYPT 2017). The
                    quantum row is left out because a quantum ciphertext has no byte size. */}
                <p className="muted" style={{ fontSize: 11, margin: "0 0 6px" }}>
                  Depends on how many of the 4,096 slots are used (about 12x when all are filled). The quantum
                  row is not plotted: a quantum ciphertext has no size in bytes.
                </p>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart accessibilityLayer title="Ciphertext to plaintext size ratio, log scale"
                            data={ratioData}>
                    <CartesianGrid stroke="#243049" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#97a3b6" }} />
                    <YAxis scale="log" domain={[0.5, "auto"]} tick={{ fontSize: 11, fill: "#97a3b6" }} allowDataOverflow />
                    <Tooltip contentStyle={{ background: "#151c2c", border: "1px solid #243049" }} />
                    {/* WCAG 2.2 SC 1.4.1 (Level A). `colorFor` encoded the ciphertext PATH by fill
                        alone, and the `<Legend />` on the latency chart covers only the single
                        `latency` series, so it explains nothing about the per-bar colours. The x-axis
                        `name` mitigates but does not cure it, because `name` is the workload rather
                        than the path. QSMPC-FHE now carries a hatch as well as its colour. */}
                    <defs>
                      <pattern id="hatch-qsmpc" patternUnits="userSpaceOnUse" width="5" height="5"
                               patternTransform="rotate(45)">
                        <rect width="5" height="5" fill="#b072ff" />
                        <line x1="0" y1="0" x2="0" y2="5" stroke="#e5d0ff" strokeWidth="1.6" />
                      </pattern>
                    </defs>
                    <Bar dataKey="ratio">{ratioData.map((d, i) => <Cell key={i} fill={fillFor(d.path)} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="card">
                <h3>Latency per inference (ms)</h3>
                {/* Kept as per-row measurements and captioned, following the recorded decision to state
                    non-comparability rather than hide a measurement. The two bars time different work. */}
                <p className="muted" style={{ fontSize: 11, margin: "0 0 6px" }}>
                  Different computations, not a speed comparison: MPC-FHE runs a real CKKS and garbled-circuit
                  inference; QSMPC-FHE simulates an untrained 8-qubit circuit on random input.
                </p>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart accessibilityLayer title="Latency per inference, milliseconds"
                            data={latData}>
                    <CartesianGrid stroke="#243049" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#97a3b6" }} />
                    <YAxis tick={{ fontSize: 11, fill: "#97a3b6" }} />
                    <Tooltip contentStyle={{ background: "#151c2c", border: "1px solid #243049" }} />
                    <Legend />
                    {/* WCAG 2.2 SC 1.4.1 (Level A). `colorFor` encoded the ciphertext PATH by fill
                        alone, and the `<Legend />` on the latency chart covers only the single
                        `latency` series, so it explains nothing about the per-bar colours. The x-axis
                        `name` mitigates but does not cure it, because `name` is the workload rather
                        than the path. QSMPC-FHE now carries a hatch as well as its colour. */}
                    <defs>
                      <pattern id="hatch-qsmpc-lat" patternUnits="userSpaceOnUse" width="5" height="5"
                               patternTransform="rotate(45)">
                        <rect width="5" height="5" fill="#b072ff" />
                        <line x1="0" y1="0" x2="0" y2="5" stroke="#e5d0ff" strokeWidth="1.6" />
                      </pattern>
                    </defs>
                    <Bar dataKey="latency">{latData.map((d, i) => <Cell key={i} fill={fillFor(d.path, "-lat")} />)}</Bar>
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
                  <BarChart accessibilityLayer title="Time split: linear layers under FHE against non-linear under MPC"
                            data={splitData}>
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
                  {/* WHAT THIS NUMBER IS, because directly under "total: 707 ms" and labelled
                      "client compute (wall-clock)" it read as the whole computation costing 0.07 ms
                      in the browser. It times the `privateInference` call only - the PLAINTEXT
                      reference this page runs so it has something to compare the encrypted result
                      against - and a linear model over 8 features is genuinely sub-millisecond. */}
                  {clientComputeMs != null && (
                    <div className="muted">
                      plaintext reference inference: <b>{clientComputeMs} ms</b>{" "}
                      <span className="small">— the unencrypted comparison run, not the total above</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          {facts.length > 0 && (
            <div className="card" style={{ marginTop: 14 }}>
              {/* "what this run measured" headed rows that come from an analytic channel model: no
                  QKD device exists in this project, so the QKD rows are now labelled "simulated". */}
              <h3>Security posture — what this run measured or simulated</h3>
              <p className="muted small" style={{ marginTop: 0 }}>
                Reported, not scored. Each row is checkable against a published standard, measured in
                this run, or simulated by this run's analytic quantum-channel model (every QKD row - no
                QKD device exists here), and says which. There is no composite: the 0–100 score this
                replaced summed five weights that were chosen rather than derived, and it awarded a
                perfect 100 to a run whose quantum channel produced zero secure key bits.
              </p>
              <table className="mini-table">
                <tbody>
                  {facts.map((f) => (
                    <tr key={f.label} className={f.warn ? undefined : "metric-companion"}>
                      <td>{f.label}</td>
                      <td className="num">{f.value}</td>
                      <td className="muted small">
                        {f.kind === "standard" ? "standard · " : f.kind === "simulated" ? "simulated · " : "measured · "}{f.source}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {facts.filter((f) => f.warn).map((f) => (
                <div key={f.label} className="note" style={{ marginTop: 10, borderColor: "var(--warn)", color: "var(--warn)" }}>
                  ⚠ <b>{f.label}:</b> {f.warn}
                </div>
              ))}
            </div>
          )}
          {!compact && (
            <div className="card" style={{ marginTop: 14 }}>
              <h3>Detailed results</h3>
              <table>
                {/* The last column was headed "model" and rendered `security_model`, so the table
                    had no model column at all while appearing to have one. Both are shown now, and
                    the model one matters here: these rows are not all the same model. */}
                <thead><tr><th>path</th><th>model</th><th>backend</th><th>size</th><th>latency ms</th><th>FHE-lin ms</th><th>MPC-nl ms</th><th>comm KB</th><th>accuracy</th><th>ct/pt</th><th>security model</th></tr></thead>
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
                      <td>{r.accuracy ?? "—"}</td><td>{ctpt(r) ?? "—"}</td>
                      <td style={{ fontSize: 11 }}>{r.security_model}</td>
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
