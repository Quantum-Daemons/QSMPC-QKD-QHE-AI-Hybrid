// Crypto-agility selectors + advanced params + Run/Pause/Resume/Step/Stop/Restart controls.
// Shared by the generic Orchestration page and every per-use-case page (driven by usePipelineRun).
import { useState } from "react";
import { type usePipelineRun } from "../lib/usePipelineRun";
import { isClientDemo } from "../lib/demoMode";
import ParameterPanel from "./ParameterPanel";

type Run = ReturnType<typeof usePipelineRun>;

export default function PipelineControls({ run }: { run: Run }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  // In the public demo the FHE/MPC compute is REAL and runs entirely in the browser (node-seal CKKS +
  // local secret-sharing), so the server-side backend selectors (TenSEAL/OpenFHE/MPyC/…) don't apply.
  const demo = isClientDemo();
  const { usecase, opts, kem, setKem, qkd, setQkd, fhe, setFhe, mpc, setMpc, sig, setSig,
    workload, setWorkload, optiqkd, setOptiqkd,
    state, running, paused, start, startStepwise, pause, resume, step, stop } = run;

  // Round 18: three controls were rendered on pages where they could not change anything a visitor
  // could see. plan18's own audit rule is that "a control that renders but changes nothing is a
  // defect", so they are shown only where they are load-bearing. The state and the pipeline code
  // behind them are untouched -- only the UI is gated.
  //
  //  * Model size scales already-measured FHE/MPC numbers, but telecom_ids and satellite_comms are
  //    not in INFERENCE_USECASES, so BenchmarkPanel renders `compact` and hides every element it
  //    affects. Its only trace there was the download filename.
  //  * Workload's cnn/transformer values are x2/x3 multipliers on MLP hidden widths. Only
  //    medical_fl + cnn selects a genuinely different (convolutional) architecture. It is also a
  //    complete no-op at Model size = Lightweight, where hidden is [] and [].map(x2) is still [].
  //  * Federated (QFL) injects a stage that runs and logs, but no diagram outside medical_fl has a
  //    federated_training node to render it.
  // Model size is gone. It selected between invented architectures -- a "tabular fraud MLP" at
  // 18/850/9,538 parameters and so on -- while the models that actually exist are an XGBoost
  // teacher and a 33-parameter distilled student. The medical tier advertised 35,458 parameters
  // for a model whose real counterpart is 11,171,266, and genomics got an MLP for a method that is
  // a 77-weight linear score with no network in it at all. The benchmark now costs the real
  // student, which is trained, published and pinned.
  //
  // Federated learning is always on rather than opt-in: it does real client-side FedAvg over the
  // page's real bundled sample, so there is no reason to make a visitor find a checkbox for it.
  const showWorkload = usecase === "medical_fl";
  // OptiQKD is kept everywhere -- it does real work -- but it cannot run without a quantum channel:
  // qkd_key_agreement returns before the optimise block when the PQC-only profile is selected.
  const optiqkdDisabled = qkd === "pqc-only";

  const sel = (label: string, id: string, fam: string, value: string, set: (v: string) => void) => (
    <>
      <label className="muted" htmlFor={id}>{label}</label>
      <select id={id} name={id} aria-label={label} value={value} onChange={(e) => set(e.target.value)}>
        {(opts.families[fam] || []).map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </>
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
        {/* Workload is run config: it parameterises the single Run that drives both the live
            pipeline AND the benchmark assembled from it (unified, Round 7). Both options now run
            REAL CKKS - a packed dot product and a genuine im2col convolution - so the run reports
            each one's rotation count, which is the cost driver the comparison is about. Until
            Round 20 the CNN option ran the dot product and multiplied its latency by a model. */}
        {showWorkload && (<>
          <label className="muted" htmlFor="sel-workload"
            title="Both run real node-seal CKKS. Linear is a packed dot product; CNN is a genuine encrypted convolution over the trained student's 14x14 input with its own 3x3 kernels. The run reports rotations for each, because key-switching dominates CKKS cost.">
            Workload
          </label>
          <select id="sel-workload" name="sel-workload" aria-label="Workload" value={workload}
            onChange={(e) => setWorkload(e.target.value)}>
            <option value="linear">Linear (encrypted dot product)</option>
            <option value="cnn">CNN (encrypted convolution)</option>
          </select>
        </>)}
        {/* Quantum Federated Learning toggle — runs a real client-side FedAvg of a quantum-classical
            model (full deployment = Flower + PennyLane HybridQNN under quantum-safe secure aggregation). */}
        {/* AI-QKD parameter optimization (OptiQKD, WS1) — measured gradient-free SKR maximization. */}
        <label className={`muted${optiqkdDisabled ? " is-disabled" : ""}`} htmlFor="sel-optiqkd"
          title={optiqkdDisabled
            ? "Unavailable with the PQC-only profile: there is no quantum channel to optimise. Select a QKD protocol to enable it."
            : "Run AI-QKD parameter optimization (OptiQKD, WS1) — a measured gradient-free search that tunes decoy/basis params to raise the secret-key rate on the simulated channel."}>
          <input id="sel-optiqkd" name="sel-optiqkd" type="checkbox" checked={optiqkd && !optiqkdDisabled}
            disabled={optiqkdDisabled}
            onChange={(e) => setOptiqkd(e.target.checked)} style={{ marginRight: 4 }} />
          Optimize QKD (OptiQKD){optiqkdDisabled ? " — needs a QKD protocol" : ""}
        </label>
        <span style={{ flex: 1 }} />
        <button className="btn ghost" onClick={() => setShowAdvanced(!showAdvanced)}>
          {showAdvanced ? "▼" : "▶"} Parameters
        </button>
      </div>
      {showAdvanced && <ParameterPanel run={run} />}
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
