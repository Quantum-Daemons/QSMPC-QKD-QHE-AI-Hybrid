// Where did this number come from?
//
// This project mixes real cryptography with simulated physics and, in places, with cost models.
// That mixture is defensible only if a reader can tell, per result, which one they are looking at.
// The badge has exactly four states and every one of them carries its evidence:
//
//   MEASURED (local WebGPU)  computed in this tab on a real GPU adapter
//   MEASURED (local WASM)    computed in this tab on the WebAssembly runtime
//   REMOTE                   computed by the API or a hosted Space; latency includes the network
//   SIMULATED                no computation of the advertised kind happened
//
// SIMULATED is not a failure state and is used deliberately. QKD has no browser API and cannot have
// one, so every QKD number here comes from an analytic channel model. Zama's own documentation says
// the TFHE-rs WASM build "does not support FHE computations". No browser MPC library is maintained
// (JIFF's last commit was 2024-06; the PSE MPC Framework is sunset). Labelling those SIMULATED is
// the accurate description, and the badge exists so nothing has to be taken on trust.
import { useEffect, useState } from "react";

import { onnxStatus, webgpuInfo } from "../lib/onnxInference";

// `measured-js` exists because the other three were being asked to cover a case none of them fit.
// The bundled distilled model is real and it is measured, but it is a hand-written loop over plain
// JS numbers -- no ORT, no wasm module, no GPU. It was reporting `measured-wasm`, so a pipeline run
// that never touched WebAssembly rendered as "MEASURED (local WASM)". The claim was wrong in the
// direction that flatters, which is the direction this project cares about.
export type Provenance =
  "measured-webgpu" | "measured-wasm" | "measured-js" | "remote" | "simulated";

const LABEL: Record<Provenance, string> = {
  "measured-webgpu": "MEASURED (local WebGPU)",
  "measured-wasm": "MEASURED (local WASM)",
  "measured-js": "MEASURED (plain JS)",
  remote: "REMOTE (server)",
  simulated: "SIMULATED",
};

const TITLE: Record<Provenance, string> = {
  "measured-webgpu": "Computed in this browser tab on a real GPU adapter. Timings are wall-clock.",
  "measured-wasm": "Computed in this browser tab on the WebAssembly runtime. Timings are wall-clock.",
  "measured-js":
    "Computed in this browser tab in plain JavaScript arithmetic - a real model, really evaluated, " +
    "but with no WebAssembly or GPU involved. Timings are wall-clock.",
  remote: "Computed off this machine. The reported time includes network latency.",
  simulated:
    "No computation of the advertised kind was performed. The value comes from an analytic model " +
    "or a cost model, not from running the primitive.",
};

export type ProvenanceEvidence = Record<string, string | number | boolean | null | undefined>;

export function ProvenanceBadge({
  kind, detail, evidence, inline = false,
}: {
  kind: Provenance;
  detail?: string;
  evidence?: ProvenanceEvidence;
  inline?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [env, setEnv] = useState<ProvenanceEvidence>({});

  useEffect(() => {
    // Evidence is read from the runtime, not passed in by the caller that wants to look good.
    let alive = true;
    (async () => {
      const s = onnxStatus();
      const gpu = kind === "measured-webgpu" ? await webgpuInfo() : {};
      if (!alive) return;
      setEnv({
        crossOriginIsolated: typeof globalThis !== "undefined"
          ? (globalThis as any).crossOriginIsolated === true : false,
        wasmThreads: s.threads,
        onnxBackend: s.backend,
        modelKey: s.modelKey,
        modelBytes: s.bytes,
        servedFromCache: s.fromCache,
        loadMs: s.loadMs != null ? Math.round(s.loadMs) : null,
        onnxError: s.error,
        gpuAdapter: (gpu as any)?.info
          ? [(gpu as any).info.vendor, (gpu as any).info.architecture].filter(Boolean).join(" ")
          : undefined,
        shaderF16: (gpu as any)?.shaderF16,
      });
    })();
    return () => { alive = false; };
  }, [kind]);

  const rows = Object.entries({ ...env, ...(evidence ?? {}) })
    .filter(([, v]) => v !== undefined && v !== null && v !== "");

  return (
    <span className={`prov prov-${kind}${inline ? " prov-inline" : ""}`}>
      <button
        type="button"
        className="prov-chip"
        title={TITLE[kind]}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {LABEL[kind]}
        {detail ? <span className="prov-detail"> · {detail}</span> : null}
        <span className="prov-caret">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <span className="prov-evidence">
          {rows.length === 0
            ? <em>no runtime evidence captured</em>
            : rows.map(([k, v]) => (
                <span key={k} className="prov-kv">
                  <code>{k}</code>: {String(v)}
                </span>
              ))}
        </span>
      )}
    </span>
  );
}

/**
 * Pick the provenance for an in-browser result from what the runtime actually reports.
 *
 * Note the ordering: an ONNX load error demotes to `simulated`, because a page that could not load
 * its model but still shows a number is showing a number that came from somewhere else.
 */
export function browserProvenance(): Provenance {
  const s = onnxStatus();
  if (s.error || !s.loaded) return "simulated";
  return s.backend === "onnx-webgpu" ? "measured-webgpu" : "measured-wasm";
}
