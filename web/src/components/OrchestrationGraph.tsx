import { useMemo } from "react";
import {
  Background,
  Controls,
  type Edge,
  type Node,
  ReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// Stage sequence per use case (keys match backend stage.name values).
//
// quantum_comparison and sign_result are emitted by both the client pipeline (clientCompute FLOWS)
// and the backend, but had no node here, so the last two stages of every run lit nothing on the
// Orchestration page. The `data` node is client-only: there is no `data` stage server-side, so in
// full mode it stays dim by design.
const FLOWS: Record<string, { id: string; label: string }[]> = {
  default: [
    { id: "data", label: "Data" },
    { id: "pqc_handshake", label: "PQC Handshake\n(ML-KEM + X25519)" },
    { id: "qkd_key_agreement", label: "QKD Key" },
    { id: "qsmpc_compute", label: "QSMPC\n(PSI / secure sum)" },
    { id: "private_inference", label: "QHE/FHE\nPrivate Inference" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result\n+ artifacts" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
  // Finance is in FEDERATED_USECASES, so a `federated_training` stage is injected into its run. It
  // had no key here, so it fell to `default` - which has no such node either - and the stage lit
  // nothing. Node labels are static here (this graph has no metrics), so this one names the
  // algorithm rather than a security property: `clientQFL` aggregates plaintext updates, and
  // "(secure aggregation)" claimed a guarantee no code provided. TwoPartyDiagram carries the
  // measured `qfl_aggregation` string for the run-specific version.
  finance_aml: [
    { id: "data", label: "Banks' Data" },
    { id: "pqc_handshake", label: "PQC Handshake\n(ML-KEM + X25519)" },
    { id: "qkd_key_agreement", label: "QKD Key" },
    { id: "qsmpc_compute", label: "QSMPC\n(PSI)" },
    { id: "federated_training", label: "Federated Training\n(size-weighted FedAvg)" },
    { id: "private_inference", label: "QHE/FHE\nPrivate Inference" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result\n+ artifacts" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
  medical_fl: [
    { id: "data", label: "Hospitals' Data" },
    { id: "pqc_handshake", label: "PQC Handshake" },
    { id: "qkd_key_agreement", label: "QKD Key" },
    { id: "federated_training", label: "Federated Training\n(size-weighted FedAvg)" },
    { id: "private_inference", label: "Private Inference" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
  telecom_ids: [
    { id: "data", label: "QKD Telemetry" },
    { id: "pqc_handshake", label: "PQC Handshake" },
    { id: "qkd_key_agreement", label: "QKD Key (w/ Eve)" },
    { id: "qml_ids", label: "QML-IDS\n(attack detection)" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
  satellite_comms: [
    { id: "data", label: "Mission Telemetry" },
    { id: "pqc_handshake", label: "PQC Handshake" },
    { id: "satellite_qkd", label: "Free-space QKD" },
    { id: "qsdc_telemetry", label: "QSDC\n(direct + eavesdrop check)" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
};

export function flowFor(usecase: string) {
  return FLOWS[usecase] ?? FLOWS.default;
}

export default function OrchestrationGraph({
  usecase,
  activeStage,
  doneStages,
  config = {},
  metrics = {},
}: {
  usecase: string;
  activeStage: string | null;
  doneStages: Set<string>;
  /** Crypto-agility selections plus the Parameters-panel protocol, so the node captions name what
   *  actually ran. `qkd` is the family (is there a channel); `qkdProtocol` is the protocol. */
  config?: { kem?: string; qkd?: string; sig?: string; qkdProtocol?: string };
  /** The live run's metrics, so the inference node can name the published model that ran.
   *
   *  This graph is the SECOND view of the same run, and the checklist's own rule is to audit every
   *  view rather than one per page - which is how the frozen QKD label was found. The two-party
   *  diagram has named the published model since Round 21; this one did not, so the same run
   *  described itself two different ways depending on which page you were on. */
  metrics?: Record<string, any>;
}) {
  const stages = flowFor(usecase);
  // The FLOWS table above is static, so these captions used to keep naming ML-KEM-768/BB84/ML-DSA
  // whatever the page's selectors said. Retitle them from the live config instead.
  const m = metrics ?? {};
  const retitle = (id: string, label: string): string => {
    const head = label.split("\n")[0];
    if (id === "pqc_handshake") {
      const kem = String(config.kem ?? "");
      const pretty = kem === "ml-kem-1024+x25519" ? "ML-KEM-1024 + X25519"
        : kem === "ml-kem-768+x25519" ? "ML-KEM-768 + X25519"
        : kem ? kem.toUpperCase() : "hybrid KEM";
      return `${head}\n(${pretty})`;
    }
    // Both quantum-channel nodes. The protocol comes from the Parameters panel and the family only
    // says whether a channel exists, so reading `config.qkd` alone made the E91 branch unreachable
    // and left the satellite node captioned "BB84 downlink" whatever the visitor selected.
    if (id === "qkd_key_agreement" || id === "satellite_qkd") {
      const q = String(config.qkd ?? "");
      if (q === "pqc-only") return `${head}\n(skipped — PQC-only)`;
      const proto = String(config.qkdProtocol ?? "");
      const pretty = proto === "e91" ? "E91 / CHSH"
        : proto === "bb84" ? "BB84 decoy-state"
        : q.includes("decoy") ? "BB84 decoy-state" : q ? "BB84" : "QKD";
      return `${head}\n(${pretty}${id === "satellite_qkd" ? " downlink" : ""})`;
    }
    if (id === "sign_result") {
      return `${head}\n(${config.sig ? String(config.sig).toUpperCase() : "ML-DSA"})`;
    }
    // Whichever stage carries this use case's inference: private_inference for finance, medical and
    // genomics, qml_ids for telecom, qsdc_telemetry for satellite. Same mapping the pipeline uses.
    if (id === "private_inference" || id === "qml_ids" || id === "qsdc_telemetry") {
      const model = m.published_model;
      if (!model) return label;
      if (m.published_model_status === "failed") return `${head}\n(${model}: did not run)`;
      const be = m.published_model_backend ? String(m.published_model_backend).replace("onnx-", "") : "";
      return `${head}\n(${model}${be ? ` on ${be}` : ""})`;
    }
    return label;
  };

  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = stages.map((s, i) => {
      const cls =
        activeStage === s.id ? "node-active" : doneStages.has(s.id) ? "node-done" : "";
      return {
        id: s.id,
        position: { x: 30 + i * 230, y: 80 },
        data: { label: retitle(s.id, s.label) },
        className: cls,
        style: {
          whiteSpace: "pre-line", width: 190, textAlign: "center" as const,
          fontSize: 12, lineHeight: 1.25, padding: "10px 8px",
        },
        sourcePosition: "right" as any,
        targetPosition: "left" as any,
      };
    });
    const edges: Edge[] = stages.slice(1).map((s, i) => ({
      id: `${stages[i].id}->${s.id}`,
      source: stages[i].id,
      target: s.id,
      animated: activeStage === s.id || activeStage === stages[i].id,
    }));
    return { nodes, edges };
    // Every config field retitle() reads must be listed, or the graph keeps a stale label. This is
    // how the protocol stayed frozen: qkdProtocol was threaded in and used, but omitted here, so
    // React never rebuilt the nodes when it changed and the QKD node kept whatever it first drew.
    // Every field retitle() reads must be listed, metrics included - omitting one is precisely how
    // the protocol label stayed frozen while the wiring and the label function were both correct.
  }, [stages, activeStage, doneStages, config.kem, config.qkd, config.sig, config.qkdProtocol,
      m.published_model, m.published_model_backend, m.published_model_status]);

  return (
    <div style={{ height: 260, border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
      <ReactFlow nodes={nodes} edges={edges} fitView fitViewOptions={{ padding: 0.15 }}
        minZoom={0.3} maxZoom={2} proOptions={{ hideAttribution: true }}
        nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}
        panOnDrag zoomOnScroll>
        <Background color="#243049" gap={18} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
