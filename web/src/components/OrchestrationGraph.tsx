import { useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Controls,
  type Edge,
  MarkerType,
  type Node,
  ReactFlow,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// Stage sequence per use case (keys match backend stage.name values).
//
// quantum_comparison and sign_result are emitted by both the client pipeline (clientCompute FLOWS)
// and the backend, but had no node here, so the last two stages of every run lit nothing on the
// Orchestration page. The `data` node is client-only: there is no `data` stage server-side, so in
// full mode it stays dim by design.
// Stages the BACKEND runs that the client pipeline does not. This graph is drawn on
// `/orchestration` in BOTH modes, and `FLOWS` was reconciled against `clientFlow()` alone - so on
// the full deployment it went dark for a stage that was actually running, on every use case:
//
//   finance / genomics / demo / medical   `explainability`
//   telecom / satellite                   `qkd_optimize`
//   satellite                             `telemetry_anomaly`
//
// while drawing `federated_training` for finance, which `standard_stages()` does not contain at
// all. The "diagram shows a stage the pipeline does not run" defect this file was written to fix
// was live the whole time on the transport nobody looked at.
//
// Each label says which deployment lights it, because a node that is dim for a good reason and a
// node that is dim because the diagram is wrong look identical. `tests/test_dag_covers_both_
// transports.py` asserts this map against the real Python stage lists, resolving `stage.name` from
// the class definitions so a rename cannot slip past it.
// Each entry names the node it must sit BEFORE, because the backend runs these at a specific point
// and a diagram that draws them all just before `result` shows the run jumping backwards. Measured
// on all six use cases: `explainability` runs before `quantum_comparison` and was drawn after it;
// `qkd_optimize` runs immediately after key agreement and was drawn near the end.
//
// The first version of this map had no position and the test that guards it compared SETS, so both
// were wrong in the same direction and neither could see the other. An ordered artefact needs an
// ordered check - that is the whole finding.
const BACKEND_EXTRA: Record<string, { id: string; label: string; before: string }[]> = {
  default: [{ id: "explainability", label: "Explainability\n(full deployment)", before: "quantum_comparison" }],
  finance_aml: [{ id: "explainability", label: "Explainability\n(full deployment)", before: "quantum_comparison" }],
  genomics: [{ id: "explainability", label: "Explainability\n(full deployment)", before: "quantum_comparison" }],
  demo: [{ id: "explainability", label: "Explainability\n(full deployment)", before: "quantum_comparison" }],
  medical_fl: [{ id: "explainability", label: "Explainability\n(full deployment)", before: "quantum_comparison" }],
  telecom_ids: [{ id: "qkd_optimize", label: "OptiQKD\n(full deployment)", before: "qml_ids" }],
  satellite_comms: [
    { id: "qkd_optimize", label: "OptiQKD\n(full deployment)", before: "qsdc_telemetry" },
    { id: "telemetry_anomaly", label: "Telemetry anomaly\n(full deployment)", before: "quantum_comparison" },
  ],
};

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
    { id: "federated_training", label: "Federated Training\n(size-weighted FedAvg, demo)" },
    { id: "private_inference", label: "QHE/FHE\nPrivate Inference" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result\n+ artifacts" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
  medical_fl: [
    { id: "data", label: "Hospitals' Data" },
    { id: "pqc_handshake", label: "PQC Handshake" },
    { id: "qkd_key_agreement", label: "QKD Key" },
    { id: "federated_training", label: "Federated Training\n(size-weighted FedAvg, demo)" },
    { id: "private_inference", label: "Private Inference" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
  telecom_ids: [
    { id: "data", label: "QKD Telemetry" },
    { id: "pqc_handshake", label: "PQC Handshake" },
    // "(w/ Eve)" was here unconditionally while `qkd_eve` defaults to 0.0. Node labels in this graph
    // are static - it has no metrics - so the honest form is to name the stage and let
    // TwoPartyDiagram, which does read metrics, carry the run-specific eavesdropper tag.
    { id: "qkd_key_agreement", label: "QKD Key" },
    { id: "qml_ids", label: "QML-IDS\n(attack detection)" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
  satellite_comms: [
    { id: "data", label: "Mission Telemetry" },
    { id: "pqc_handshake", label: "PQC Handshake" },
    { id: "satellite_qkd", label: "Free-space QKD" },
    { id: "qsdc_telemetry", label: "QSDC\n(simulated; direct + eavesdrop check)" },
    { id: "quantum_comparison", label: "QSMPC-FHE\n(quantum comparison)" },
    { id: "result", label: "Result" },
    { id: "sign_result", label: "Sign Result\n(ML-DSA)" },
  ],
};

/** The nodes to draw: the client flow, plus the stages only the backend runs, before `result`.
 *
 *  Both are drawn in both modes on purpose. A visitor on the demo sees the full-deployment stages
 *  labelled as such and permanently dim, which is the honest shape - the alternative is a diagram
 *  that means something different depending on how the page was served, and this file already
 *  carries one round's worth of consequences from a flow map that described one transport.
 */
export function flowFor(usecase: string) {
  const client = FLOWS[usecase] ?? FLOWS.default;
  const extra = BACKEND_EXTRA[usecase] ?? BACKEND_EXTRA.default ?? [];
  // Each backend-only node is spliced before the node it precedes in the real pipeline, not lumped
  // in front of `result`. Lumping them there put `explainability` after `quantum_comparison` on
  // four flows and `qkd_optimize` near the end on two - so on the full deployment the run lit a
  // later node and then jumped back, which reads as a stalled or reordered pipeline.
  const out = [...client];
  for (const node of extra) {
    const at = out.findIndex((n) => n.id === node.before);
    out.splice(at >= 0 ? at : Math.max(0, out.length - 1), 0, node);
  }
  return out;
}

/** Every node's box: wide enough for a caption line at 13 px, tall enough for four lines (a number
 *  and title, a two-line parenthetical, and the state). Fixed rather than measured; see `width` /
 *  `height` below. */
const NODE_W = 176, NODE_H = 100;
const GAP_X = 40, GAP_Y = 56;
/** Space between the graph and its frame, in screen pixels. */
const PAD = 16;
/** A strip below the stages for the zoom controls. In a corner over the graph they covered half of
 *  the first stage of the second row, measured once the stages filled the frame. */
const CONTROLS_H = 34;
/** The smallest scale the stages are laid out for: 13 px captions stay at 11 px or more. */
const MIN_SCALE = 0.85;

/** Rows of stages for a frame `width` pixels wide (Round 80).
 *
 *  All stages used to sit in one row, and React Flow fitted that row to the frame: ten stages at
 *  0.36 scale on a 1280 px window, captions about 4 px tall - measured on the live site during a
 *  run. Instead the stages wrap into as few rows as keep them at MIN_SCALE or larger, balanced so
 *  the last row is not a stub, and read left to right like lines of text. */
export function stageGrid(n: number, width: number): { cols: number; rows: number } {
  const fit = width > 0 ? Math.floor((width / MIN_SCALE - 2 * PAD + GAP_X) / (NODE_W + GAP_X)) : 5;
  const rows = Math.ceil(n / Math.max(2, Math.min(n, fit)));
  return { rows, cols: Math.ceil(n / rows) };
}

/** Applies the computed viewport. Inside <ReactFlow>, because that is where its store lives; the
 *  graph sets its own scale rather than calling `fitView`, whose first fit can run before the frame
 *  has a size (the README's screenshot capture had to press the fit button for that reason). */
function ViewportSync({ x, y, zoom }: { x: number; y: number; zoom: number }) {
  const rf = useReactFlow();
  useEffect(() => { void rf.setViewport({ x, y, zoom }); }, [rf, x, y, zoom]);
  return null;
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
      // ", simulated" on the node: no QKD device exists in this project, and this graph is the one
      // place on /orchestration that names the protocol before a run.
      return `${head}\n(${pretty}${id === "satellite_qkd" ? " downlink" : ""}, simulated)`;
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

  /** The three states this graph draws, as words.
   *
   *  SC 1.4.1 Level A: state was encoded ONLY in colour. `.node-active` is a blue ring and glow and
   *  `.node-done` changes `border-color` and nothing else, so "which stage is running" and "which
   *  stages have finished" - the two questions this graph exists to answer - were unavailable to a
   *  reader who cannot separate those hues, and unavailable to every screen reader, since React Flow
   *  renders the state as a class on a `<div>` with no text.
   *
   *  Used twice below, for the visible label and for the text equivalent, so the two cannot drift.
   */
  const stateOf = (id: string) =>
    activeStage === id ? "running" : doneStages.has(id) ? "done" : "not yet run";

  // The frame's width, measured, so the rows follow it - including when the sidebar collapses.
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [frameW, setFrameW] = useState(0);
  useEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setFrameW(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const { cols, rows } = stageGrid(stages.length, frameW);
  const graphW = cols * NODE_W + (cols - 1) * GAP_X;
  const graphH = rows * NODE_H + (rows - 1) * GAP_Y;
  const zoom = frameW > 0 ? Math.min(1, (frameW - 2 * PAD) / graphW) : MIN_SCALE;
  const frameH = Math.round(graphH * zoom + 2 * PAD + CONTROLS_H);
  const view = { x: Math.max(PAD, (frameW - graphW * zoom) / 2), y: PAD, zoom };

  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = stages.map((s, i) => {
      const col = i % cols, row = Math.floor(i / cols);
      const cls =
        activeStage === s.id ? "node-active" : doneStages.has(s.id) ? "node-done" : "";
      // The state as TEXT on its own line, beside the colour rather than instead of it. `pre-line`
      // is already set below and the captions already carry parenthetical lines such as
      // "(skipped - PQC-only)", so this is the idiom the node labels use, not a new one. "not yet
      // run" is left off: it is the resting state of every node before a run, and printing it on all
      // of them at once says nothing while making each node taller.
      const st = stateOf(s.id);
      // Numbered, because once the stages wrap the order has to be readable without the arrows.
      const label = `${i + 1}. ${retitle(s.id, s.label)}` + (st === "not yet run" ? "" : `\n[${st}]`);
      return {
        id: s.id,
        position: { x: col * (NODE_W + GAP_X), y: row * (NODE_H + GAP_Y) },
        data: { label },
        className: cls,
        // EXPLICIT DIMENSIONS (Round 80). These nodes are rebuilt as new objects on every stage
        // change, and React Flow hides a node it has not measured until its size is observed again;
        // a node whose box did not change size was never re-measured, so after a run most of the
        // pipeline went invisible - on the live site, in a foreground tab, six of ten nodes. Found
        // by the README's screenshots. A node with a width and height is drawn without measuring.
        width: NODE_W,
        height: NODE_H,
        // And `measured`, for the edges. React Flow keeps a node's handle positions across a rebuilt
        // node object only when the object carries `measured`; without it they are dropped, and a
        // node with a fixed size is never measured again - so from the first stage change of a run
        // every edge vanished, all nine of them, and stayed gone after "done" (Round 80, measured in
        // the browser while checking the wrapped rows).
        measured: { width: NODE_W, height: NODE_H },
        style: {
          whiteSpace: "pre-line", width: NODE_W, height: NODE_H, textAlign: "center" as const,
          fontSize: 13, lineHeight: 1.25, padding: "8px 8px", boxSizing: "border-box" as const,
          display: "flex", alignItems: "center", justifyContent: "center",
          // A model id such as satellite_comms.telemanom has no space to break at.
          overflowWrap: "anywhere" as const,
        },
        // The last stage of a row hands on downwards and the first of the next takes it from above,
        // so the wrap reads as a line break rather than an edge across the whole graph.
        sourcePosition: (col === cols - 1 && i < stages.length - 1 ? "bottom" : "right") as any,
        targetPosition: (col === 0 && i > 0 ? "top" : "left") as any,
      };
    });
    const edges: Edge[] = stages.slice(1).map((s, i) => ({
      id: `${stages[i].id}->${s.id}`,
      source: stages[i].id,
      target: s.id,
      // Arrowheads: with the stages in rows, direction is no longer implied by position alone.
      type: (i + 1) % cols === 0 ? "smoothstep" : "straight",
      markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
      animated: activeStage === s.id || activeStage === stages[i].id,
    }));
    return { nodes, edges };
    // Every config field retitle() reads must be listed, or the graph keeps a stale label. This is
    // how the protocol stayed frozen: qkdProtocol was threaded in and used, but omitted here, so
    // React never rebuilt the nodes when it changed and the QKD node kept whatever it first drew.
    // Every field retitle() reads must be listed, metrics included - omitting one is precisely how
    // the protocol label stayed frozen while the wiring and the label function were both correct.
  }, [stages, cols, activeStage, doneStages, config.kem, config.qkd, config.sig, config.qkdProtocol,
      m.published_model, m.published_model_backend, m.published_model_status]);

  return (
    <>
    {/* TEXT EQUIVALENT, the same pattern and for the same reason as `TwoPartyDiagram`'s `sr-only`
        list. React Flow renders each stage as a `<div>` whose only state signal is a CSS class, so
        the pipeline's shape, its stage captions and its progress were all unreachable in text. The
        list is built from the SAME `retitle()` and `stateOf()` calls the nodes use, so it cannot
        describe a graph other than the one on screen - the failure mode a hand-written summary has.
        Ordered, because the order IS the pipeline. */}
    <ol className="sr-only">
      {stages.map((s) => (
        <li key={s.id}>{`${retitle(s.id, s.label).replace(/\n/g, " ")} - ${stateOf(s.id)}`}</li>
      ))}
    </ol>
    <div ref={frameRef}
         style={{ height: frameH, border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}
         role="group"
         aria-label={`${usecase} pipeline stages; the same stages and their state are listed in text immediately before this figure`}>
      <ReactFlow nodes={nodes} edges={edges} defaultViewport={view}
        minZoom={0.3} maxZoom={2} proOptions={{ hideAttribution: true }}
        nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}
        panOnDrag zoomOnScroll={false} preventScrolling={false}>
        {/* The wheel scrolls the page again: at 0.36 scale zooming was the only way to read a
            caption, and it also stopped the page scrolling whenever the pointer crossed the graph.
            Readable as laid out now; the zoom buttons remain. */}
        <ViewportSync x={view.x} y={view.y} zoom={view.zoom} />
        <Background color="#243049" gap={18} />
        <Controls showInteractive={false} position="bottom-right" orientation="horizontal" />
      </ReactFlow>
    </div>
    </>
  );
}
