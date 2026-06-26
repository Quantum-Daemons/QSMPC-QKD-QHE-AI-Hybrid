// Crypto-agility selectors + advanced params + Run/Pause/Resume/Step/Stop/Restart controls.
// Shared by the generic Orchestration page and every per-use-case page (driven by usePipelineRun).
import { useState } from "react";
import { MODEL_SIZES, type usePipelineRun } from "../lib/usePipelineRun";
import { isClientDemo } from "../lib/demoMode";

type Run = ReturnType<typeof usePipelineRun>;

export default function PipelineControls({ run }: { run: Run }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  // In the public demo the FHE/MPC compute is REAL and runs entirely in the browser (node-seal CKKS +
  // local secret-sharing), so the server-side backend selectors (TenSEAL/OpenFHE/MPyC/…) don't apply.
  const demo = isClientDemo();
  const { opts, kem, setKem, qkd, setQkd, fhe, setFhe, mpc, setMpc, sig, setSig,
    modelSize, setModelSize, workload, setWorkload, qfl, setQfl, params, setParams,
    state, running, paused, start, startStepwise, pause, resume, step, stop } = run;

  const sel = (label: string, id: string, fam: string, value: string, set: (v: string) => void) => (
    <>
      <label className="muted" htmlFor={id}>{label}</label>
      <select id={id} name={id} aria-label={label} value={value} onChange={(e) => set(e.target.value)}>
        {(opts.families[fam] || []).map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </>
  );
  const num = (label: string, key: keyof typeof params, stepv = 1, min = 0) => (
    <div>
      <label className="muted" htmlFor={`p-${key}`} style={{ fontSize: 12 }}>{label}</label>
      <input id={`p-${key}`} name={String(key)} type="number" step={stepv} min={min}
        value={params[key] as number} style={{ width: "100%" }}
        onChange={(e) => setParams({ ...params, [key]: parseFloat(e.target.value) || 0 })} />
    </div>
  );

  return (
    <div className="card">
      <div className="btn-row">
        {sel("KEM", "sel-kem", "kem", kem, setKem)}
        {sel("QKD", "sel-qkd", "qkd", qkd, setQkd)}
        {/* Demo: FHE/MPC run for real in the browser → a static badge, not inert backend dropdowns.
            Full: the server-side TenSEAL/OpenFHE/MPyC selectors stay (real crypto-agility). */}
        {demo ? (
          <span className="muted" style={{ fontSize: 13 }}
            title="The demo computes FHE/MPC for real in your browser (node-seal CKKS + secret-sharing MPC) — there is no server backend to select.">
            FHE/MPC: <strong>node-seal CKKS + secret-sharing MPC</strong> · real, in-browser
          </span>
        ) : (
          <>
            {sel("FHE (linear)", "sel-fhe", "fhe", fhe, setFhe)}
            {sel("MPC (non-linear)", "sel-mpc", "mpc", mpc, setMpc)}
          </>
        )}
        {sel("Signature", "sel-sig", "sig", sig, setSig)}
        <span style={{ flex: 1 }} />
        <span className={`badge-state ${state}`}>{state}</span>
      </div>
      <div className="btn-row" style={{ marginTop: 8 }}>
        {/* model size + workload are run config: they parameterise the single Run that drives both
            the live pipeline AND the benchmark assembled from it (unified, Round 7). */}
        <label className="muted" htmlFor="sel-size">Model size</label>
        <select id="sel-size" name="sel-size" aria-label="Model size" value={modelSize}
          onChange={(e) => setModelSize(e.target.value)}>
          {MODEL_SIZES.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
        <label className="muted" htmlFor="sel-workload">Workload</label>
        <select id="sel-workload" name="sel-workload" aria-label="Workload" value={workload}
          onChange={(e) => setWorkload(e.target.value)}>
          <option value="linear">Linear</option><option value="cnn">CNN</option><option value="transformer">Transformer</option>
        </select>
        {/* Quantum Federated Learning toggle — runs a real client-side FedAvg of a quantum-classical
            model (full deployment = Flower + PennyLane HybridQNN under quantum-safe secure aggregation). */}
        <label className="muted" htmlFor="sel-qfl"
          title="Run Quantum Federated Learning this run — a hybrid quantum-classical FedAvg across simulated clients (demo: in-browser; full: Flower + PennyLane).">
          <input id="sel-qfl" name="sel-qfl" type="checkbox" checked={qfl}
            onChange={(e) => setQfl(e.target.checked)} style={{ marginRight: 4 }} />
          Federated (QFL)
        </label>
        <span style={{ flex: 1 }} />
        <button className="btn ghost" onClick={() => setShowAdvanced(!showAdvanced)}>
          {showAdvanced ? "▼" : "▶"} Advanced parameters
        </button>
      </div>
      {showAdvanced && (
        <div className="grid cols-4" style={{ marginTop: 10 }}>
          {num("QKD pulses", "qkd_pulses", 10000)}
          {num("QKD distance (km)", "qkd_distance_km", 5)}
          {num("Eve fraction (0–1)", "qkd_eve", 0.1)}
          {num("QCNN qubits", "qubits", 1, 2)}
          {num("QSMS prime (0=auto)", "prime", 2)}
          {num("RNG seed", "seed", 1)}
          <div>
            <label className="muted" htmlFor="p-op" style={{ fontSize: 12 }}>QSMPC op</label>
            <select id="p-op" name="qsmpc_op" value={params.qsmpc_op}
              onChange={(e) => setParams({ ...params, qsmpc_op: e.target.value })}>
              <option value="">(use case default)</option>
              <option value="psi">PSI</option>
              <option value="sum">Secure sum</option>
            </select>
          </div>
          <div>
            <label className="muted" htmlFor="p-inf" style={{ fontSize: 12 }}>Inference</label>
            <select id="p-inf" name="inference" value={params.inference}
              onChange={(e) => setParams({ ...params, inference: e.target.value })}>
              <option value="">(use case default)</option>
              <option value="ckks">CKKS (FHE)</option>
              <option value="qcnn">QCNN (QHE)</option>
            </select>
          </div>
        </div>
      )}
      <div className="btn-row" style={{ marginTop: 14 }}>
        <button className="btn" onClick={start} disabled={running}>▶ Start</button>
        <button className="btn ghost" onClick={startStepwise} disabled={running || paused}>⏭ Start stepwise</button>
        <button className="btn warn" onClick={pause} disabled={!running}>⏸ Pause</button>
        <button className="btn" onClick={resume} disabled={!paused}>▶ Resume</button>
        <button className="btn ghost" onClick={step} disabled={!paused}>↓ Step</button>
        <button className="btn danger" onClick={stop} disabled={state === "idle" || state === "done"}>■ Stop</button>
        <button className="btn ghost" onClick={start}>↻ Restart</button>
      </div>
    </div>
  );
}
