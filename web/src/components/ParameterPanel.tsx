// Config-editable Parameters panel — grouped numeric/selection inputs whose defaults come from config
// (config.defaults.json / /api/config) and whose edits flow into the run config via usePipelineRun's
// buildConfig(). Editing a value + re-running demonstrably changes the run (browser-verifiable). "Reset to
// defaults" restores the config values. No hardcoded parameter values live here — only labels + UI step/min.
import {
  runsFibreQkd, runsInference, runsQsmpc, type usePipelineRun,
} from "../lib/usePipelineRun";

type Run = ReturnType<typeof usePipelineRun>;
// `max` exists because QCNN qubits drives a 2^n state vector evaluated several times per run and
// had no upper bound at all: typing ~24 allocates ~16.7M complex amplitudes and hangs the tab.
type Field = [label: string, key: string, step: number, min: number, title?: string, max?: number];

export default function ParameterPanel({ run }: { run: Run }) {
  const { params, setParams, resetParams, usecase } = run;
  const p = params as Record<string, number | string>;
  /**
   * Commit a numeric edit, CLAMPED to that field's own min/max.
   *
   * This was `parseFloat(v) || 0`, which substituted 0 for an empty field regardless of the bounds
   * the field itself declares - and 0 is out of range for nearly all of them. The worst case was
   * `qrng_eps_log2` (min -128, max -1): clearing the box wrote 0, i.e. **eps = 2^0 = 1**, a
   * vacuous Leftover-Hash-Lemma bound. Measured on the live demo before this fix: the run then
   * reported `qrng_lhl_max_bits` 7389 against 7309 at eps 2^-40 - clearing the field GAINED 80 bits
   * of extractable key by deleting the security parameter. That is the dangerous direction, because
   * the failure makes the output look better rather than worse.
   *
   * An unparseable value falls back to `min`, which for every field here is the conservative end.
   */
  const setNum = (k: string, v: string, min: number, max?: number) => {
    const n = parseFloat(v);
    const safe = Number.isFinite(n)
      ? Math.min(max ?? Infinity, Math.max(min, n))
      : min;
    setParams({ ...params, [k]: safe });
  };

  const numField = ([label, key, step, min, title, max]: Field) => {
    // A field whose value the current selection makes unreadable is greyed out with the reason,
    // rather than left editable and silently ignored. Both cases below are real: the LHL eps is
    // only consumed by the Toeplitz/seedless extractors, and aml_transactions is only read on the
    // PSI branch of the QSMPC stage.
    const inertBecause =
      key === "qrng_eps_log2" && p.qrng_post === "von_neumann"
        ? "No effect on the von Neumann extractor - switch QRNG extractor to Toeplitz or seedless to use the LHL eps."
      : key === "aml_transactions" && p.qsmpc_op === "sum"
        ? "No effect while QSMPC op is Secure sum - only the PSI branch sizes its sets from this."
      : null;
    const dim = inertBecause !== null;
    return (
      <div key={key} style={{ minWidth: 120, flex: "1 1 120px" }}>
        <label className="muted" htmlFor={`pp-${key}`} style={{ fontSize: 12, opacity: dim ? 0.45 : 1 }}
          title={inertBecause ?? title}>{label}</label>
        <input id={`pp-${key}`} name={key} type="number" step={step} min={min} max={max} disabled={dim}
          value={p[key] as number} style={{ width: "100%", opacity: dim ? 0.45 : 1 }} onChange={(e) => setNum(key, e.target.value, min, max)} />
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
  //
  // These now ASK the flow rather than restating it as three use-case lists. The lists had already
  // drifted: FLOWS.demo runs both qsmpc_compute and private_inference and appeared in neither, so
  // the Generic demo hid three controls its own pipeline went on to read. `buildConfig` gates on the
  // same predicates, so what is shown and what is sent cannot disagree.
  const fibreQkd = runsFibreQkd(usecase);
  const qsmpc = runsQsmpc(usecase);
  const inference = runsInference(usecase);

  const groups: { title: string; fields: Field[] }[] = [];
  if (fibreQkd) groups.push(
    { title: "QKD channel", fields: [
      ["QKD pulses", "qkd_pulses", 10000, 1, "channel uses for the BB84 run"],
      ["Distance (km)", "qkd_distance_km", 5, 1, "fibre distance - raises loss + QBER"],
      // Both feed intrinsicQber, which saturates at 0.5, so anything past these renders identically.
      ["Eve fraction", "qkd_eve", 0.05, 0, "intercept-resend fraction (0-1) - raises QBER", 1],
      ["Misalignment", "qkd_misalignment", 0.005, 0, "baseline optical error", 0.5]] });
  // The satellite channel, which `satellite_qkd` and `qsdc_telemetry` have always read and nothing
  // ever rendered. Without these the QSDC abort threshold could not be reached from the UI: Eve was
  // pinned at 0, so free-space QBER stayed near 0.033 against a 0.11 abort threshold.
  if (!fibreQkd) groups.push(
    { title: "Free-space channel", fields: [
      ["Downlink (km)", "sat_distance_km", 50, 1, "ground-to-LEO slant range - raises loss + QBER"],
      ["Eve fraction", "sat_eve", 0.05, 0,
       "intercept-resend fraction (0-1). Raise it until QBER crosses the QSDC threshold and the "
       + "transmission aborts - that abort is the inline eavesdrop check this page describes", 1],
      ["Pulses", "sat_pulses", 500000, 1, "channel uses for the free-space downlink"],
      ["QSDC abort at QBER", "qsdc_threshold", 0.01, 0,
       "QSDC refuses to send when the measured QBER exceeds this", 0.5]] });
  groups.push(
    { title: "OptiQKD (AI-QKD)", fields: [
      ["Opt samples", "qkd_opt_samples", 8, 4,
       "evaluation budget for the coordinate-ascent search - enable 'Optimize QKD' to use it"]] },
    { title: "QRNG", fields: [
      // The Toeplitz extractor is O(n^2) and synchronous. Unbounded, this freezes the tab exactly
      // as the QCNN qubit count used to: ~4e9 operations at 32768 bits.
      ["QRNG bits", "qrng_bits", 512, 64,
       "quantum-random bits generated + health-checked per run - capped at 16384 because the "
       + "Toeplitz extractor is quadratic and runs on the main thread", 16384],
      // The comment here used to say this "moves the extracted bit count", and Round 20 restored the
      // field on that basis. It does not, at any setting this spinner can reach. It moves the LHL
      // BOUND (min-entropy + 2*eps_log2 + 2); delivered output is min(bound, requested), and
      // requested is the smaller one until qrng_bits drops below about 80 - which, with min 64 and
      // step 512, means 64 is the only reachable value where the delivered count changes at all.
      // Measured: at 1024 bits every eps from -1 to -128 returns byte-identical output.
      //
      // So the run now reports the bound itself as qrng_lhl_max_bits. The parameter is real - it is
      // the extractor's security level - and this is the observable it actually moves.
      ["LHL eps (log2)", "qrng_eps_log2", 1, -128,
       "extractor security 2^eps, consumed only by the Toeplitz and seedless extractors. Moves the "
       + "leftover-hash-lemma ceiling reported as qrng_lhl_max_bits; it shortens the delivered "
       + "output only when that ceiling falls below the bits you asked for", -1]] },
    { title: "Quantum / run", fields: [
      ["QCNN qubits", "qubits", 1, 2,
       "state-vector QCNN qubit count - capped at 14: the simulator is exact, so cost is 2^n", 14],
      // min 1, not 0. quantumSim's rng() does `(seed >>> 0) || 1`, so seed 0 and seed 1 produced
      // byte-identical runs everywhere: the field's own minimum was a duplicate of the value one
      // step above it, and anyone comparing "seed 0" with "seed 1" was comparing a run with itself.
      ["RNG seed", "seed", 1, 1,
       "run reproducibility seed (starts at 1: the generator maps 0 to 1, so they are not distinct)"]] },
  );
  if (usecase === "telecom_ids") groups.push({ title: "IDS",
    // 250 full-batch gradient iterations over 8 x n x 0.75 rows, synchronously: same hang class.
    fields: [["Samples / class", "n_per_class", 50, 20,
              "QML-IDS training samples per attack class - capped at 2000; training is synchronous",
              2000]] });
  // Nothing is trained here, and the earlier tooltip said this figure trained the fraud model. Its
  // only reader sizes the PSI sets (clientCompute's `perParty`), and that reader lives in the `psi`
  // branch -- so under QSMPC op = "Secure sum" the field is inert. Say both things.
  if (usecase === "finance_aml") groups.push({ title: "Data",
    fields: [["AML transactions", "aml_transactions", 500, 200,
              "sizes the per-party PSI sets (about n/20 each, 20-2000). It does not train anything: "
              + "the fraud model is the published GBM. Only read when QSMPC op is PSI"]] });

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
            {qsmpc && (
            <div style={{ minWidth: 120, flex: "1 1 120px" }}>
              <label className="muted" htmlFor="pp-op" style={{ fontSize: 12 }}>QSMPC op</label>
              <select id="pp-op" name="qsmpc_op" value={p.qsmpc_op as string} style={{ width: "100%" }}
                onChange={(e) => setParams({ ...params, qsmpc_op: e.target.value })}>
                <option value="">(use case default)</option><option value="psi">PSI</option><option value="sum">Secure sum</option>
              </select>
            </div>)}
            {inference && (
            <div style={{ minWidth: 120, flex: "1 1 120px" }}>
              <label className="muted" htmlFor="pp-inf" style={{ fontSize: 12 }}>Inference</label>
              <select id="pp-inf" name="inference" value={p.inference as string} style={{ width: "100%" }}
                onChange={(e) => setParams({ ...params, inference: e.target.value })}>
                {/* "" resolves to ckks in clientCompute - there is no per-use-case table behind it - so the
                    option says which backend it means rather than implying one. */}
                <option value="">CKKS (FHE) - default</option>
                <option value="ckks">CKKS (FHE)</option>
                <option value="qcnn">QCNN (QHE)</option>
              </select>
            </div>)}
            {/* Only the secure-sum path reads it, and only genomics/finance run that path. The old
                label promised a prime and accepted any integer: additive sharing over Z_100 is not
                a field, so the value is snapped to the next prime on entry rather than trusted. */}
            {qsmpc && (
            <div style={{ minWidth: 120, flex: "1 1 120px" }}>
              <label className="muted" htmlFor="pp-prime" style={{ fontSize: 12 }}>QSMS prime (0=auto)</label>
              <input id="pp-prime" name="prime" type="number" step={2} min={0} value={p.prime as number}
                style={{ width: "100%" }}
                title="Snapped to the next prime: secure summation shares over a prime field, and a composite modulus is not one."
                onChange={(e) => setNum("prime", e.target.value, 0)} />
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
