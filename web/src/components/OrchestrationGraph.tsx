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
    { id: "qkd_key_agreement", label: "QKD Key\n(BB84)" },
    { id: "qsmpc_compute", label: "QSMPC\n(PSI / secure sum)" },
    { id: "private_inference", label: "QHE/FHE\nPrivate Inference" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result\n+ artifacts" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
  medical_fl: [
    { id: "data", label: "Hospitals' Data" },
    { id: "pqc_handshake", label: "PQC Handshake" },
    { id: "qkd_key_agreement", label: "QKD Key" },
    { id: "federated_training", label: "Federated Training\n(secure aggregation)" },
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
    { id: "satellite_qkd", label: "Free-space QKD\n(BB84 downlink)" },
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
}: {
  usecase: string;
  activeStage: string | null;
  doneStages: Set<string>;
}) {
  const stages = flowFor(usecase);
  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = stages.map((s, i) => {
      const cls =
        activeStage === s.id ? "node-active" : doneStages.has(s.id) ? "node-done" : "";
      return {
        id: s.id,
        position: { x: 30 + i * 230, y: 80 },
        data: { label: s.label },
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
  }, [stages, activeStage, doneStages]);

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
