// Crypto-agility selectors + advanced params + Run/Pause/Resume/Step/Stop/Restart controls.
// Shared by the generic Orchestration page and every per-use-case page (driven by usePipelineRun).
import { useEffect, useState } from "react";
import { type usePipelineRun } from "../lib/usePipelineRun";
import { browserModels, hasWebGPU } from "../lib/onnxInference";
import { isClientDemo } from "../lib/demoMode";
import ParameterPanel from "./ParameterPanel";

type Run = ReturnType<typeof usePipelineRun>;

export default function PipelineControls({ run }: { run: Run }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  // In the public demo the FHE/MPC compute is REAL and runs entirely in the browser (node-seal CKKS +
  // local secret-sharing), so the server-side backend selectors (TenSEAL/OpenFHE/MPyC/…) don't apply.
  const demo = isClientDemo();
  const { usecase, opts, kem, setKem, qkd, setQkd, fhe, setFhe, mpc, setMpc, sig, setSig,
    optiqkd, setOptiqkd, publishedModel, setPublishedModel, onnxBackend, setOnnxBackend,
    state, running, paused, inFlight, start, startStepwise, pause, resume, step, stop } = run;
  // Resolved once: whether WebGPU can be offered at all. Without a usable adapter the option is shown
  // disabled with the reason, rather than offered and then silently run on WASM.
  const [gpuUsable, setGpuUsable] = useState(false);
  useEffect(() => { hasWebGPU().then(setGpuUsable); }, []);

  // The published models this use case can actually fetch and run, in registry order. Rendered only
  // when there is a genuine choice: telecom publishes one, genomics none, and offering a select with
  // a single option is a control that cannot change anything - the defect class this file already
  // gates three other controls on.
  //
  // The reason this exists: the orchestration ran browserModels(usecase)[0] and nothing else, so
  // medical's ResNet-50 and CKKS student, and three of satellite's four models, were reachable only
  // from the Models panel's own buttons and never through the E2E simulator.
  const publishedCandidates = browserModels(usecase);

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
  //    federated_training node to render it. (Removing the control fixed the symptom for the four
  //    use cases that then left the set. It came back when finance was added to FEDERATED_USECASES
  //    on the data half of the rule alone; both figures now carry the step, and the scope test
  //    asserts the render half rather than describing it in prose.)
  // Model size is gone. It selected between invented architectures -- a "tabular fraud MLP" at
  // 18/850/9,538 parameters and so on -- while the models that actually exist are an XGBoost
  // teacher and a 33-parameter distilled student. The medical tier advertised 35,458 parameters
  // for a model whose real counterpart is 11,171,266, and genomics got an MLP for a method that is
  // a 77-weight linear score with no network in it at all. The benchmark now costs the real
  // student, which is trained, published and pinned.
  //
  // Federated learning is always on rather than opt-in: it does real client-side FedAvg over the
  // page's real bundled sample, so there is no reason to make a visitor find a checkbox for it.
  // OptiQKD is kept everywhere -- it does real work -- but it cannot run without a quantum channel:
  // qkd_key_agreement returns before the optimise block when the PQC-only profile is selected.
  const optiqkdDisabled = qkd === "pqc-only";

  // Display names, kept separate from the registry VALUES the backend resolves. The QKD control
  // named a protocol it does not select: its options read "bb84-decoy" and "pqc-only" while the
  // thing that actually chooses BB84 versus E91 is `qkd_protocol` in the Parameters panel. So a
  // visitor could leave this on "bb84-decoy", set the parameter to E91, and watch E91 run under a
  // label saying BB84 - two controls that look like duplicates, one of which is lying.
  //
  // This control is a PROFILE switch: whether a quantum channel contributes key material at all.
  // Naming it that removes both the apparent duplication and the real inconsistency, and needs no
  // change to the registry, the config schema or the backend.
  const DISPLAY: Record<string, string> = {
    "bb84-decoy": "QKD + PQC hybrid",
    "pqc-only": "PQC only (no quantum channel)",
  };

  const sel = (label: string, id: string, fam: string, value: string, set: (v: string) => void,
               hint?: string) => (
    <>
      <label className="muted" htmlFor={id} title={hint}>{label}</label>
      <select id={id} name={id} aria-label={label} title={hint} value={value}
              onChange={(e) => set(e.target.value)}>
        {(opts.families[fam] || []).map((o) => (
          <option key={o} value={o}>{DISPLAY[o] ?? o}</option>
        ))}
      </select>
    </>
  );

  return (
    <div className="card">
      <div className="btn-row">
        {sel("KEM", "sel-kem", "kem", kem, setKem)}
        {/* The resolved protocol, rendered as visible text. This select is a PROFILE switch - whether a
            quantum channel contributes key material - while the thing that chooses BB84 versus E91 is
            `qkd_protocol`, inside the COLLAPSED Parameters panel. Both facts were explained at length in
            comments here and in the diagram, and on screen the distinction existed only as a `title`
            tooltip. A visitor reading "QKD + PQC hybrid" had no way to know a second, hidden control
            decides which protocol runs. */}
        {sel("Key source", "sel-qkd", "qkd", qkd, setQkd,
          "Whether a quantum channel contributes key material at all. WHICH protocol is "
          + "simulated - BB84 decoy-state or E91/CHSH - is a separate setting, "
          + "`qkd.protocol` in the Parameters panel below.")}
        {/* Demo: FHE/MPC run for real in the browser → a static badge, not inert backend dropdowns.
            Full: the server-side TenSEAL/OpenFHE/MPyC selectors stay (real crypto-agility). */}
        {demo ? (
          <span className="muted" style={{ fontSize: 13 }}
            title="The demo computes FHE/MPC for real in your browser (node-seal CKKS + an emp-wasm garbled-circuit 2PC whose two parties share this tab) — there is no server backend to select.">
            FHE/MPC: <strong>node-seal CKKS + garbled-circuit 2PC</strong> · real protocol, in-browser, notional parties
          </span>
        ) : (
          <>
            {sel("FHE (linear)", "sel-fhe", "fhe", fhe, setFhe)}
            {sel("MPC (non-linear)", "sel-mpc", "mpc", mpc, setMpc)}
          </>
        )}
        <span className="muted" style={{ fontSize: 12, alignSelf: "center" }}
          title="Set by `qkd.protocol` in the Parameters panel below, not by the Key source select.">
          protocol: <strong>{String(run.params.qkd_protocol) === "e91" ? "E91 / CHSH" : "BB84 decoy-state"}</strong>
        </span>
        {sel("Signature", "sel-sig", "sig", sig, setSig)}
        {publishedCandidates.length > 1 && (
          <>
            <label className="muted" htmlFor="sel-published-model">Published model</label>
            <select id="sel-published-model" name="published_model" aria-label="Published model"
              value={publishedModel} disabled={inFlight}
              // Names the CURRENT selection, because the closed control clips its own text on a
              // narrow viewport - the labels run to about 45 characters and the box is capped at the
              // card width. Hovering has to recover what was clipped, not repeat the generic help.
              title={`Runs on a row of this model's own corpus. Selected: ${
                publishedCandidates.find((c) => c.spec.key === publishedModel)?.spec.label
                ?? `auto (${publishedCandidates[0].spec.label})`}`}
              onChange={(e) => setPublishedModel(e.target.value)}>
              {/* Named, not blank. "" is the registry's first entry, and a bare empty option reads
                  as "none" next to a list of models. */}
              <option value="">{`auto (${publishedCandidates[0].spec.label})`}</option>
              {publishedCandidates.map((c) => (
                <option key={c.spec.key} value={c.spec.key}>{c.spec.label}</option>
              ))}
            </select>
          </>
        )}
        {publishedCandidates.length > 0 && (
          <>
            <label className="muted" htmlFor="sel-onnx-backend">Model backend</label>
            <select id="sel-onnx-backend" name="onnx_backend" aria-label="Model backend"
              value={onnxBackend} disabled={inFlight}
              title={"Where the published models run. WASM is the default: on the machine measured, one "
                + "ResNet-18 inference took 18216 ms on WebGPU with a hardware adapter, against tens of "
                + "milliseconds on WASM. Other GPUs and browsers were not measured, so WebGPU stays "
                + "selectable to try on this one."}
              onChange={(e) => setOnnxBackend(e.target.value === "webgpu" ? "webgpu" : "")}>
              <option value="">WASM (default)</option>
              <option value="webgpu" disabled={!gpuUsable}>
                {gpuUsable ? "WebGPU" : "WebGPU (no usable adapter)"}
              </option>
            </select>
          </>
        )}
        <span style={{ flex: 1 }} />
        <span className={`badge-state ${state}`}>{state}</span>
      </div>
      <div className="btn-row" style={{ marginTop: 8 }}>
        {/* The Workload selector is gone. It offered Linear against CNN, and once the CNN path
            began evaluating the PUBLISHED student's own weights end to end under CKKS, Linear was
            the only fabricated option left: an 8-feature dot product with a hand-written weight
            pattern that matched no trained model. Medical now always runs its real student. The
            internal `workload` value is retained because the benchmark cost model still reads it
            for the use cases whose students genuinely ARE linear. */}
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
        {/* A disclosure control, so it has to announce itself as one. The caret alone is visual:
            aria-expanded is what tells a screen reader the panel exists and whether it is open,
            and aria-controls points at the region it governs (WAI-ARIA disclosure pattern). */}
        <button className="btn ghost" onClick={() => setShowAdvanced(!showAdvanced)}
          aria-expanded={showAdvanced} aria-controls="pipeline-parameters"
          title={showAdvanced ? "Hide the run parameters" : "Show the run parameters"}>
          {showAdvanced ? "▼" : "▶"} Parameters
        </button>
      </div>
      <div id="pipeline-parameters" hidden={!showAdvanced}>
        {showAdvanced && <ParameterPanel run={run} />}
      </div>
      <div className="btn-row" style={{ marginTop: 14 }}>
        <button className="btn" onClick={start} disabled={inFlight}>▶ Start</button>
        <button className="btn ghost" onClick={startStepwise} disabled={inFlight}>⏭ Start stepwise</button>
        <button className="btn warn" onClick={pause} disabled={!running}>⏸ Pause</button>
        <button className="btn" onClick={resume} disabled={!paused}>▶ Resume</button>
        <button className="btn ghost" onClick={step} disabled={!paused}>↓ Step</button>
        <button className="btn danger" onClick={stop} disabled={!inFlight}>■ Stop</button>
        {/* Deliberately NOT disabled while running - superseding a run in progress is the point, and
            `Start` is disabled precisely so this is the button that does it. It is only safe because
            `launch` now takes a run token and ignores anything a superseded run emits: before that,
            the old run finished its in-flight stage and wrote its own metrics, artifacts, signature
            and benchmark over the new run's. The superseded run does keep using CPU until its current
            stage ends, because the pipeline only checks the stop flag between stages. */}
        <button className="btn ghost" onClick={start}
          title={running || paused ? "Supersedes the run in progress; its results are discarded"
                                   : "Run again from the beginning"}>↻ Restart</button>
      </div>
    </div>
  );
}
