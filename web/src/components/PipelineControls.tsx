// Crypto-agility selectors + advanced params + Run/Pause/Resume/Step/Stop/Restart controls.
// Shared by the generic Orchestration page and every per-use-case page (driven by usePipelineRun).
import { useEffect, useState } from "react";
import { runsAnyQkd, type usePipelineRun } from "../lib/usePipelineRun";
import { browserModels, hasWebGPU } from "../lib/onnxInference";
import { isClientDemo } from "../lib/demoMode";
import ParameterPanel from "./ParameterPanel";

type Run = ReturnType<typeof usePipelineRun>;

export default function PipelineControls({ run }: { run: Run }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  // In the public demo the FHE/MPC compute is REAL and runs entirely in the browser (node-seal CKKS +
  // local secret-sharing), so the server-side backend selectors (TenSEAL/OpenFHE/MPyC/…) don't apply.
  const demo = isClientDemo();
  const { usecase, opts, kem, setKem, qkd, setQkd, fhe, setFhe, mpc, sig, setSig,
    optiqkd, setOptiqkd, mpcCnn, setMpcCnn, mpcMlp, setMpcMlp, mpcGnn, setMpcGnn, multiPsi, setMultiPsi, multiPsiParties, setMultiPsiParties, publishedModel, setPublishedModel, onnxBackend, setOnnxBackend,
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
  // teacher and a distilled linear student (33 parameters then, 115 on the 114 inputs since Round 86). The medical tier advertised 35,458 parameters
  // for a model whose real counterpart is 11,171,266, and genomics got an MLP for a method that is
  // a 77-weight linear score with no network in it at all. The benchmark now costs the real
  // student, which is trained, published and pinned.
  //
  // Federated learning is always on rather than opt-in: it does real client-side FedAvg over the
  // page's real bundled sample, so there is no reason to make a visitor find a checkbox for it.
  // OptiQKD is kept everywhere -- it does real work -- but it cannot run without a quantum channel:
  // qkd_key_agreement returns before the optimise block when the PQC-only profile is selected.
  const optiqkdDisabled = qkd === "pqc-only";
  // A flow with no QKD stage has nothing for the profile switch, the protocol line or OptiQKD to change.
  const quantum = runsAnyQkd(usecase);

  // Display names, kept separate from the registry VALUES the backend resolves. The QKD control
  // named a protocol it does not select: its options read "bb84-decoy" and "pqc-only" while the
  // thing that actually chooses BB84 versus E91 is `qkd_protocol` in the Parameters panel. So a
  // visitor could leave this on "bb84-decoy", set the parameter to E91, and watch E91 run under a
  // label saying BB84 - two controls that look like duplicates, one of which is lying.
  //
  // This control is a PROFILE switch: whether the simulated quantum channel runs at all. In the
  // browser it never contributes key material either way - `bb84()` / `e91()` return a key LENGTH and
  // the session key is written by the PQC handshake alone - and in the full deployment a run that
  // clears 256 secure bits folds the simulated channel's key in. Naming it that removes both the
  // apparent duplication and the real inconsistency, and needs no change to the registry, the config
  // schema or the backend.
  // Round 86: the label read "Key source: PQC + simulated QKD stage" in this browser too, where the stage
  // contributes no key; the tooltip said so, the visible words did not. In the client demo the control is
  // named for what it switches, and the full deployment, where the simulated bits are folded in, keeps
  // "Key source".
  const DISPLAY: Record<string, string> = demo
    ? { "bb84-decoy": "on (simulated; the key comes from PQC)", "pqc-only": "off (PQC only)" }
    : { "bb84-decoy": "PQC + simulated QKD stage", "pqc-only": "PQC only (no quantum channel)" };

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
        {/* The resolved protocol, rendered as visible text. This select is a PROFILE switch - whether the
            simulated quantum channel runs - while the thing that chooses BB84 versus E91 is
            `qkd_protocol`, inside the COLLAPSED Parameters panel. Both facts were explained at length in
            comments here and in the diagram, and on screen the distinction existed only as a `title`
            tooltip. A visitor reading "QKD + PQC hybrid" had no way to know a second, hidden control
            decides which protocol runs. */}
        {quantum && sel(demo ? "Quantum channel" : "Key source", "sel-qkd", "qkd", qkd, setQkd,
          "Whether the simulated quantum channel runs at all. In this browser it contributes no key "
          + "material either way: the QKD stage yields a key length, and the session key comes from the "
          + "PQC handshake alone (the full deployment folds 256 or more simulated QKD bits in). WHICH "
          + "protocol is simulated - BB84 decoy-state or E91/CHSH - is a separate setting, "
          + "`qkd.protocol` in the Parameters panel below.")}
        {/* Demo: FHE/MPC run for real in the browser → a static badge, not inert backend dropdowns.
            Full: the FHE selector stays (each option runs). The MPC one is text since Round 86: it offered
            MP-SPDZ and hpmpc, which nothing ran, so choosing them changed nothing a run computed. */}
        {demo ? (
          <span className="muted" style={{ fontSize: 13 }}
            title="Where a page has a private-inference stage (finance, medical, genomics) or the lending page's margin check, the demo computes FHE/MPC for real in your browser (node-seal CKKS + a Polytune garbled-circuit 2PC whose two parties share this tab); telecom and satellite each run a CKKS check of a student and no MPC. There is no server backend to select.">
            FHE/MPC, where a page runs them: <strong>node-seal CKKS + garbled-circuit 2PC</strong> - real protocol, in-browser, notional parties
          </span>
        ) : (
          <>
            {sel("FHE (linear)", "sel-fhe", "fhe", fhe, setFhe)}
            <span className="muted" style={{ fontSize: 13, alignSelf: "center" }}
              title="The non-linear layers run under MPyC Shamir secret sharing among three party processes; it is the one MPC backend this deployment runs, so there is nothing to select.">
              MPC (non-linear): <strong>{mpc}</strong>
            </span>
          </>
        )}
        {quantum && <span className="muted" style={{ fontSize: 12, alignSelf: "center" }}
          title={`Set by \`qkd.protocol\` in the Parameters panel below, not by the ${demo ? "Quantum channel" : "Key source"} select.`}>
          protocol: <strong>{qkd === "pqc-only" ? "none (no QKD stage runs)" : String(run.params.qkd_protocol) === "e91" ? "E91 / CHSH" : "BB84 decoy-state"}</strong>
        </span>}
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
              {/* A model that publishes no evaluation sample by design has no row of its own to run
                  on, so selecting it could only end in "did NOT run" and a failed status (satellite's
                  CKKS student, Round 85). Offered disabled, with the reason the Models panel gives. */}
              {publishedCandidates.map((c) => (
                <option key={c.spec.key} value={c.spec.key} disabled={!!c.spec.eval_sample_reason}
                  title={c.spec.eval_sample_reason
                    ? `Cannot run here: it publishes no evaluation sample, by design - ${c.spec.eval_sample_reason}.`
                    : undefined}>
                  {c.spec.label}{c.spec.eval_sample_reason ? " (no sample, by design)" : ""}
                </option>
              ))}
            </select>
          </>
        )}
        {publishedCandidates.length > 0 && (
          <>
            <label className="muted" htmlFor="sel-onnx-backend">Model backend</label>
            <select id="sel-onnx-backend" name="onnx_backend" aria-label="Model backend"
              value={onnxBackend} disabled={inFlight}
              // Not "WebGPU is slow" (Round 84). The page loads a model's int8 graph where one ships,
              // and ONNX Runtime Web's WebGPU provider registers no ConvInteger, DynamicQuantizeLinear
              // or MatMulInteger kernel (its webgpu_execution_provider.cc at v1.27.0, v1.31.0 and v1.31.1), so
              // those nodes ran on the CPU: the 18216 ms was that mixed run, not a GPU's.
              title={"Where the published models run. WASM is the default. The page loads a model's int8 "
                + "graph where one ships, and ONNX Runtime Web's WebGPU provider has no kernel for its "
                + "ConvInteger and DynamicQuantizeLinear nodes, so with WebGPU selected those nodes run "
                + "on the CPU: on the machine measured, the medical ResNet-18 took 18216 ms that way, "
                + "against tens of milliseconds on WASM. No fp32 graph was timed on WebGPU, so WebGPU "
                + "stays selectable to try."}
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
            began evaluating the PUBLISHED student's own weights under CKKS (layer by layer, decrypted
            between layers), Linear was
            the only fabricated option left: an 8-feature dot product with a hand-written weight
            pattern that matched no trained model. Medical now always runs its real student. The
            internal `workload` value is retained because the benchmark cost model still reads it
            for the use cases whose students genuinely ARE linear. */}
        {/* Federated learning ("Quantum Federated Learning" by name) runs a real client-side FedAvg of
            a logistic head over a closed-form feature map - no circuit - with pairwise masks whose
            seeds each pair of clients agrees by X25519 (the server-side PennyLane HybridQNN, reached only by
            a direct API call, trains a real variational circuit under the same key agreement). */}
        {/* AI-QKD parameter optimization (OptiQKD, WS1): a gradient-free search of the asymptotic
            per-pulse bound, whose result is reported and not applied to the run (Round 84). */}
        {quantum && <label className={`muted${optiqkdDisabled ? " is-disabled" : ""}`} htmlFor="sel-optiqkd"
          title={optiqkdDisabled
            ? "Unavailable with the PQC-only profile: there is no quantum channel to optimise. Select a QKD protocol to enable it."
            : "Runs OptiQKD (WS1): a gradient-free coordinate ascent over the decoy intensities, the basis "
              + "probability and the abort threshold, scored on the same asymptotic decoy-state bound this "
              + "run reports as its per-pulse rate. It is a search: nothing it finds is applied to the run. "
              + "The run's composable key takes the bases at even odds, so the basis bias the search favours "
              + "has no counterpart in it, and at the tuned intensities that key can be lower; the log "
              + "reports both figures."}>
          <input id="sel-optiqkd" name="sel-optiqkd" type="checkbox" checked={optiqkd && !optiqkdDisabled}
            disabled={optiqkdDisabled}
            onChange={(e) => setOptiqkd(e.target.checked)} style={{ marginRight: 4 }} />
          Optimize QKD (OptiQKD){optiqkdDisabled ? " - needs a QKD protocol" : ""}
        </label>}
        {demo && usecase === "medical_fl" && (
          <label className="muted" htmlFor="sel-mpccnn"
            title="Runs one held-out image through the published CNN student with its ReLUs and its argmax in garbled circuits, on top of the CKKS layers, in a Web Worker so the page stays responsive. About 20 s and 170 MB of traffic between the two notional parties in this tab.">
            <input id="sel-mpccnn" name="sel-mpccnn" type="checkbox" checked={mpcCnn}
              onChange={(e) => setMpcCnn(e.target.checked)} style={{ marginRight: 4 }} />
            CNN under MPC (about 20 s)
          </label>
        )}
        {demo && usecase === "finance_aml" && (
          <label className="muted" title="Runs one held-out row through the published MLP, its linear layers under CKKS and its ReLUs and argmax in garbled circuits, in a Web Worker so the page stays responsive. Its features see transactions up to the end of their batch of 2,048.">
            <input type="checkbox" id="chk-mpcmlp" checked={mpcMlp} onChange={(e) => setMpcMlp(e.target.checked)} />{" "}
            MLP under MPC (a few seconds)
          </label>
        )}
        {demo && usecase === "finance_aml" && (
          <label className="muted" title="Runs one held-out transaction and its capped neighbourhood through the compact GNN: its linear maps under CKKS, its ReLUs and argmax in garbled circuits, in Web Workers so the page stays responsive. Its neighbours see transactions up to the end of the transaction's batch of 2,048.">
            <input type="checkbox" id="chk-mpcgnn" checked={mpcGnn} onChange={(e) => setMpcGnn(e.target.checked)} />{" "}
            GNN under MPC (ten seconds or so)
          </label>
        )}
        {demo && usecase === "finance_aml" && (
          <span className="muted" title="Runs a private set intersection among this many banks as one garbled circuit, in a Web Worker: three banks take a few seconds, five about half a minute. Each bank brings 8 generated accounts.">
            <label htmlFor="sel-mpsi">
              <input id="sel-mpsi" name="sel-mpsi" type="checkbox" checked={multiPsi}
                onChange={(e) => setMultiPsi(e.target.checked)} style={{ marginRight: 4 }} />
              PSI among
            </label>{" "}
            <select id="sel-mpsi-n" name="sel-mpsi-n" aria-label="How many banks" value={multiPsiParties} disabled={!multiPsi}
              onChange={(e) => setMultiPsiParties(Number(e.target.value))}>
              {[3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>{" "}
            banks under MPC
          </span>
        )}
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
