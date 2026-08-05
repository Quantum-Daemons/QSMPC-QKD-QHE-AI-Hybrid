// Config-editable Parameters panel — grouped numeric/selection inputs whose defaults come from config
// (config.defaults.json / /api/config) and whose edits flow into the run config via usePipelineRun's
// buildConfig(). Editing a value + re-running demonstrably changes the run (browser-verifiable). "Reset to
// defaults" restores the config values. No hardcoded parameter values live here — only labels + UI step/min.
import { type usePipelineRun } from "../lib/usePipelineRun";

type Run = ReturnType<typeof usePipelineRun>;
// `max` exists because QCNN qubits drives a 2^n state vector evaluated several times per run and
// had no upper bound at all: typing ~24 allocates ~16.7M complex amplitudes and hangs the tab.
type Field = [label: string, key: string, step: number, min: number, title?: string, max?: number];

export default function ParameterPanel({ run }: { run: Run }) {
  const { params, setParams, resetParams, usecase } = run;
  const p = params as Record<string, number | string>;
  const setNum = (k: string, v: string) => setParams({ ...params, [k]: parseFloat(v) || 0 });

  const numField = ([label, key, step, min, title, max]: Field) => {
    // the LHL eps only affects the Toeplitz/seedless extractor — grey + disable it on the von_neumann
    // default so editing it isn't a silent no-op (it activates when QRNG extractor is switched below).
    const dim = key === "qrng_eps_log2" && p.qrng_post === "von_neumann";
    return (
      <div key={key} style={{ minWidth: 120, flex: "1 1 120px" }}>
        <label className="muted" htmlFor={`pp-${key}`} style={{ fontSize: 12, opacity: dim ? 0.45 : 1 }}
          title={dim ? "No effect on the von Neumann extractor — switch QRNG extractor to Toeplitz/seedless to use the LHL eps." : title}>{label}</label>
        <input id={`pp-${key}`} name={key} type="number" step={step} min={min} max={max} disabled={dim}
          value={p[key] as number} style={{ width: "100%", opacity: dim ? 0.45 : 1 }} onChange={(e) => setNum(key, e.target.value)} />
      </div>
    );
  };

  // Which stages a use case runs decides which fields can affect anything at all (FLOWS in
  // clientCompute.ts). A field that no stage reads is the same defect as a select that changes no
  // number: it invites a visitor to adjust something and then shows them an unchanged result.
  //
  // satellite_comms is the sharpest case. It runs `satellite_qkd`, which reads config.sat_* keys
  // that do not exist in makeDefaultParams at all, so the entire QKD-channel group was inert there
  // -- four fields, never any effect, on every run since the page was written.
  const runsFibreQkd = usecase !== "satellite_comms";
  const runsQsmpc = usecase === "finance_aml" || usecase === "genomics";
  const runsInference = usecase === "finance_aml" || usecase === "medical_fl" || usecase === "genomics";

  const groups: { title: string; fields: Field[] }[] = [];
  if (runsFibreQkd) groups.push(
    { title: "QKD channel", fields: [
      ["QKD pulses", "qkd_pulses", 10000, 1, "channel uses for the BB84 run"],
      ["Distance (km)", "qkd_distance_km", 5, 1, "fibre distance - raises loss + QBER"],
      // Both feed intrinsicQber, which saturates at 0.5, so anything past these renders identically.
      ["Eve fraction", "qkd_eve", 0.05, 0, "intercept-resend fraction (0-1) - raises QBER", 1],
      ["Misalignment", "qkd_misalignment", 0.005, 0, "baseline optical error", 0.5]] });
  groups.push(
    { title: "OptiQKD (AI-QKD)", fields: [
      ["Opt samples", "qkd_opt_samples", 8, 4,
       "evaluation budget for the coordinate-ascent search - enable 'Optimize QKD' to use it"]] },
    { title: "QRNG", fields: [
      // The Toeplitz extractor is O(n^2) and synchronous. Unbounded, this freezes the tab exactly
      // as the QCNN qubit count used to: ~4e9 operations at 32768 bits.
      ["QRNG bits", "qrng_bits", 512, 64,
       "quantum-random bits generated + health-checked per run - capped at 16384 because the "
       + "Toeplitz extractor is quadratic and runs on the main thread", 16384]] },
    { title: "Quantum / run", fields: [
      ["QCNN qubits", "qubits", 1, 2,
       "state-vector QCNN qubit count - capped at 14: the simulator is exact, so cost is 2^n", 14],
      ["RNG seed", "seed", 1, 0, "run reproducibility seed"]] },
  );
  if (usecase === "telecom_ids") groups.push({ title: "IDS",
    // 250 full-batch gradient iterations over 8 x n x 0.75 rows, synchronously: same hang class.
    fields: [["Samples / class", "n_per_class", 50, 20,
              "QML-IDS training samples per attack class - capped at 2000; training is synchronous",
              2000]] });
  if (usecase === "finance_aml") groups.push({ title: "Data",
    fields: [["AML transactions", "aml_transactions", 500, 200, "PaySim transactions used to train the fraud model"]] });

  const fieldset = (title: string, children: React.ReactNode) => (
    <fieldset key={title} style={{ flex: "1 1 210px", minWidth: 190, border: "1px solid var(--border, #33384a)",
      borderRadius: 8, padding: "4px 10px 10px" }}>
      <legend className="muted" style={{ fontSize: 12, padding: "0 4px" }}>{title}</legend>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>{children}</div>
    </fieldset>
  );

  return (
    <div style={{ marginTop: 10 }}>
      <div className="btn-row" style={{ justifyContent: "space-between", alignItems: "center" }}>
        <span className="muted" style={{ fontSize: 12 }}>Parameters — defaults from config; edits apply on the next Run.</span>
        <button className="btn ghost" onClick={resetParams} style={{ fontSize: 12 }}>Reset to defaults</button>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 8 }}>
        {groups.map((g) => fieldset(g.title, g.fields.map(numField)))}
        {fieldset("Protocol", (
          <>
            {runsQsmpc && (
            <div style={{ minWidth: 120, flex: "1 1 120px" }}>
              <label className="muted" htmlFor="pp-op" style={{ fontSize: 12 }}>QSMPC op</label>
              <select id="pp-op" name="qsmpc_op" value={p.qsmpc_op as string} style={{ width: "100%" }}
                onChange={(e) => setParams({ ...params, qsmpc_op: e.target.value })}>
                <option value="">(use case default)</option><option value="psi">PSI</option><option value="sum">Secure sum</option>
              </select>
            </div>)}
            {runsInference && (
            <div style={{ minWidth: 120, flex: "1 1 120px" }}>
              <label className="muted" htmlFor="pp-inf" style={{ fontSize: 12 }}>Inference</label>
              <select id="pp-inf" name="inference" value={p.inference as string} style={{ width: "100%" }}
                onChange={(e) => setParams({ ...params, inference: e.target.value })}>
                <option value="">(use case default)</option><option value="ckks">CKKS (FHE)</option><option value="qcnn">QCNN (QHE)</option>
              </select>
            </div>)}
            {/* Only the secure-sum path reads it, and only genomics/finance run that path. The old
                label promised a prime and accepted any integer: additive sharing over Z_100 is not
                a field, so the value is snapped to the next prime on entry rather than trusted. */}
            {runsQsmpc && (
            <div style={{ minWidth: 120, flex: "1 1 120px" }}>
              <label className="muted" htmlFor="pp-prime" style={{ fontSize: 12 }}>QSMS prime (0=auto)</label>
              <input id="pp-prime" name="prime" type="number" step={2} min={0} value={p.prime as number}
                style={{ width: "100%" }}
                title="Snapped to the next prime: secure summation shares over a prime field, and a composite modulus is not one."
                onChange={(e) => setNum("prime", e.target.value)} />
            </div>)}
            <div style={{ minWidth: 120, flex: "1 1 120px" }}>
              <label className="muted" htmlFor="pp-ext" style={{ fontSize: 12 }}>QRNG extractor</label>
              <select id="pp-ext" name="qrng_post" value={p.qrng_post as string} style={{ width: "100%" }}
                onChange={(e) => setParams({ ...params, qrng_post: e.target.value })}>
                <option value="von_neumann">von Neumann</option><option value="toeplitz">Toeplitz (LHL)</option><option value="seedless">seedless (2-source)</option>
              </select>
            </div>
            <div style={{ minWidth: 120, flex: "1 1 120px" }}>
              <label className="muted" htmlFor="pp-proto" style={{ fontSize: 12 }}>QKD protocol</label>
              <select id="pp-proto" name="qkd_protocol" value={p.qkd_protocol as string} style={{ width: "100%" }}
                onChange={(e) => setParams({ ...params, qkd_protocol: e.target.value })}>
                <option value="bb84">BB84 (decoy)</option><option value="e91">E91 (CHSH)</option>
              </select>
            </div>
          </>
        ))}
      </div>
    </div>
  );
}
