// The use case's architecture, animated by the run (Round 80, the owner's choice).
//
// It replaced a two-party sequence diagram: two lifelines with one arrow per stage. That figure was
// accurate and it did not show the architecture. Which component on each side does a stage, which
// way its exchange goes, which parties take part and how the session key reaches the stages that
// use it were all left to the reader, and during a run the only sign of progress was a row's
// opacity. Here each party is a column of components, each stage a lane between the components
// that run it. The running stage's components glow and its lane flows, with a token travelling in
// the direction the data goes (both ways for an exchange). The session key runs down a bus to the
// two stages that consume it. A many-party stage draws every party the run has. Below the canvas
// the stage list carries each stage's full label and measured values, which are the sequence
// diagram's, unchanged, so nothing it showed is lost.
//
// The page is the visitor's: with prefers-reduced-motion the tokens are not drawn and the dashes
// do not move (styles.css), and the highlight alone says which stage runs.
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  BaseEdge, EdgeLabelRenderer, Handle, MarkerType, Position, ReactFlow, ReactFlowProvider,
  getStraightPath, useReactFlow, useUpdateNodeInternals, type Edge, type EdgeProps, type Node, type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import {
  USECASE_SPECS, canvasRow, kindOf, pqcOnly, qkdName, visibleSteps,
  type CanvasRow, type Channel, type DiagramConfig, type Flow, type Step,
} from "../lib/useCaseFlows";

// Layout, in canvas units. The canvas is drawn at this width and scaled to its frame.
const W = 980;
const COL_W = 300;                           // a party's column
const LEFT_X = 8, RIGHT_X = W - 8 - COL_W;   // 8 .. 308 and 672 .. 972
const COMP_X_A = LEFT_X + 30, COMP_X_B = RIGHT_X + 22;
const COMP_W = 248, COMP_H = 54;
const BUS_X_A = LEFT_X + 14, BUS_X_B = RIGHT_X + COL_W - 14;
const HEAD_H = 70;                           // the party's name and role, above its components
const ROW_GAP = 20;
const MINI_W = 172, MINI_H = 32, MINI_GAP = 8;
const WIDE_X = COMP_X_A, WIDE_W = COMP_X_B + COMP_W - COMP_X_A, WIDE_H = 46;
const LANE_MID = (COMP_X_A + COMP_W + COMP_X_B) / 2;
/** A lane's label: centred between the parties and clear of both columns' borders. */
const LABEL_W = 2 * (RIGHT_X - 12 - LANE_MID);
// Below this the frame scrolls rather than shrinking the text: at 0.8 a box's 12.5 px reads at 10 px,
// where 0.62 put it under 8 on a phone.
const MIN_SCALE = 0.8;

type Status = "idle" | "pending" | "running" | "done" | "failed";
// Literal colours, not the theme's variables: an SVG marker takes its colour as an attribute, where
// var() does not resolve, and the arrowheads drew as bars. The values are the theme's (styles.css),
// except the ciphertext's cyan, kept apart from the session key's and a finished stage's green.
const COLOR: Record<Channel, string> = {
  classical: "#4f9dff", quantum: "#b072ff", fhe: "#22d3ee", mpc: "#f5a623", none: "#97a3b6",
};
const FAILED = "#ff8f6b", PENDING = "#46506a", KEY = "#36d399";
const DASH: Partial<Record<Channel, string>> = { quantum: "7 4", mpc: "3 3" };
const MARK: Record<Status, string> = { idle: "", pending: "", running: "running", done: "done", failed: "failed" };

/** A step's label shortened for its lane: its first two parts, or its first when two run long. The
 *  stage list under the canvas carries all of it. */
const shortLabel = (full: string) => {
  const parts = full.split(" · ");
  const two = parts.slice(0, 2).join(" · ");
  return two.length <= 64 ? two : parts[0].length <= 72 ? parts[0] : `${parts[0].slice(0, 70).replace(/\s+\S*$/, "")} ...`;
};

type CompData = { text: string; status: Status; side: "a" | "b"; bus?: "src" | "dst"; fanIn?: number };
type WideData = { text: string; status: Status; variant?: "note" | "local" };
type PartyData = { name: string; sub: string; height: number };
type MiniData = { text: string; status: Status };
type LaneData = { channel: Channel; flow: Flow; status: Status; label?: string; labelY?: number; motion: boolean };
type BusData = { busX: number; status: Status };

const Comp = memo(function Comp({ data }: NodeProps<Node<CompData>>) {
  return (
    <div className={`ucc-comp ucc-${data.status}`}>
      {/* The lanes leave the side facing the other party; the key bus enters from the outer side. */}
      <Handle type="source" position={Position.Right} id="lane-a" isConnectable={false} />
      <Handle type="target" position={Position.Left} id="lane-b" isConnectable={false} />
      {/* One handle per party of a many-party stage, level with that party, so their lanes arrive
          side by side and level instead of stacking every arrowhead on one point. */}
      {Array.from({ length: data.fanIn ?? 0 }, (_, k) => (
        <Handle key={k} type="target" position={Position.Left} id={`lane-b-${k}`} isConnectable={false}
          style={{ top: k * (MINI_H + MINI_GAP) + MINI_H / 2 }} />
      ))}
      <Handle type={data.bus === "src" ? "source" : "target"} id="bus"
        position={data.side === "a" ? Position.Left : Position.Right} isConnectable={false} />
      <span className="ucc-comp-text">{data.text}</span>
      {MARK[data.status] && <span className={`ucc-badge ucc-badge-${data.status}`}>{MARK[data.status]}</span>}
    </div>
  );
});

const Wide = memo(function Wide({ data }: NodeProps<Node<WideData>>) {
  return (
    <div className={`${data.variant ? `ucc-${data.variant}` : "ucc-wide"} ucc-${data.status}`}>
      <span className="ucc-comp-text">{data.text}</span>
      {!data.variant && MARK[data.status] && <span className={`ucc-badge ucc-badge-${data.status}`}>{MARK[data.status]}</span>}
    </div>
  );
});

const Mini = memo(function Mini({ data }: NodeProps<Node<MiniData>>) {
  return (
    <div className={`ucc-mini ucc-${data.status}`}>
      <Handle type="source" position={Position.Right} id="lane-a" isConnectable={false} />
      <Handle type="target" position={Position.Left} id="bus" isConnectable={false} />
      {data.text}
    </div>
  );
});

const Party = memo(function Party({ data }: NodeProps<Node<PartyData>>) {
  return (
    <div className="ucc-party" style={{ height: data.height }}>
      <div className="ucc-party-name">{data.name}</div>
      <div className="ucc-party-sub">{data.sub}</div>
    </div>
  );
});

/** Eve on the quantum lane, drawn only when the run put her there. */
const Eve = memo(function Eve({ data }: NodeProps<Node<{ text: string }>>) {
  return <div className="ucc-eve">{data.text}</div>;
});

/** A stage's exchange: a straight lane, its arrowheads by direction, and while it runs, tokens. */
function Lane({ id, sourceX, sourceY, targetX, targetY, data, markerEnd, markerStart, style }: EdgeProps<Edge<LaneData>>) {
  const [path] = getStraightPath({ sourceX, sourceY, targetX, targetY });
  const d = data!;
  const color = d.status === "failed" ? FAILED : COLOR[d.channel];
  const dur = "1.6s";
  // A token per direction the data goes: A to B along the path, B to A against it.
  const tokens = d.status === "running" && d.motion
    ? (d.flow === "both" ? ["0;1", "1;0"] : d.flow === "ba" ? ["1;0"] : ["0;1"]) : [];
  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} markerStart={markerStart} style={style} />
      {tokens.map((kp, i) => (
        <circle key={i} className="ucc-token" r={4.5} fill={color}>
          <animateMotion dur={dur} repeatCount="indefinite" path={path} keyPoints={kp} keyTimes="0;1"
            calcMode="linear" begin={i ? "0.8s" : "0s"} />
        </circle>
      ))}
      {d.label && (
        <EdgeLabelRenderer>
          <div className={`ucc-lane-label ucc-${d.status}`}
            style={{ transform: `translate(-50%, -100%) translate(${LANE_MID}px, ${d.labelY}px)`, width: LABEL_W }}>
            {d.label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

/** The session key's way from the handshake to a stage that uses it: out to the bus, along it,
 *  and in, on the outer side of the party's column so it crosses no component. */
function Bus({ id, sourceX, sourceY, targetX, targetY, data, style }: EdgeProps<Edge<BusData>>) {
  const x = data!.busX;
  const path = `M ${sourceX} ${sourceY} H ${x} V ${targetY} H ${targetX}`;
  return <BaseEdge id={id} path={path} style={style} />;
}

const NODE_TYPES = { comp: Comp, wide: Wide, mini: Mini, party: Party, eve: Eve };
const EDGE_TYPES = { lane: Lane, bus: Bus };

/** React Flow 12 drops a node's handle positions when the node object is rebuilt without
 *  `measured`, and a node of fixed size is never measured again: the E2E graph lost every edge
 *  that way (Round 80). Every node here is rebuilt on each event, so each carries its size. */
const sized = <T extends Record<string, unknown>>(n: Omit<Node<T>, "measured" | "width" | "height">, w: number, h: number): Node<T> =>
  ({ ...n, width: w, height: h, measured: { width: w, height: h } }) as Node<T>;

interface Built { nodes: Node[]; edges: Edge[]; height: number }

function build(usecase: string, steps: Step[], status: (s: Step) => Status, metrics: Record<string, any>,
               config: DiagramConfig, motion: boolean): Built {
  const spec = USECASE_SPECS[usecase];
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  let y = HEAD_H + 16;
  let pqc: { a?: string; b?: string } = {};
  const keyUsers: { a?: string; b?: string; status: Status }[] = [];
  // Sized in canvas units, not stroke widths, so a running lane's thicker stroke keeps its arrowhead.
  const marker = (color: string) => ({ type: MarkerType.ArrowClosed, color, width: 14, height: 14,
                                       markerUnits: "userSpaceOnUse" });

  steps.forEach((step, i) => {
    const kind = kindOf(step, config);
    const row: CanvasRow = canvasRow(usecase, step, kind, metrics, config);
    const st = status(step);
    const id = `r${i}`;
    const full = step.label(metrics, config);
    if (row.wide) {
      nodes.push(sized<WideData>({ id: `${id}w`, type: "wide", position: { x: WIDE_X, y },
        data: { text: `${i + 1}. ${shortLabel(full)}`, status: st }, draggable: false, selectable: false }, WIDE_W, WIDE_H));
      y += WIDE_H + ROW_GAP;
      return;
    }
    const color = st === "failed" ? FAILED : COLOR[row.channel];
    const laneStyle = {
      stroke: st === "pending" ? PENDING : color,
      strokeWidth: st === "running" ? 3 : 2,
      strokeDasharray: st === "running" ? undefined : DASH[row.channel],
      opacity: st === "pending" ? 0.7 : 1,
    };
    const ends = (flow: Flow) => ({
      markerEnd: flow === "ba" ? undefined : marker(laneStyle.stroke),
      markerStart: flow === "ab" ? undefined : marker(laneStyle.stroke),
    });
    if (row.many) {
      // Every party the run has, one above the other, each with its own lane to the other side. Not
      // two to a line: the left one's lane would run through its neighbour.
      const n = Math.max(2, Math.min(8, row.many.count));
      const blockH = n * MINI_H + (n - 1) * MINI_GAP;
      const h = Math.max(COMP_H, blockH);
      for (let k = 0; k < n; k++) {
        nodes.push(sized<MiniData>({ id: `${id}m${k}`, type: "mini",
          position: { x: COMP_X_A, y: y + (h - blockH) / 2 + k * (MINI_H + MINI_GAP) },
          data: { text: `${row.many.noun} ${k + 1}`, status: st }, draggable: false, selectable: false }, MINI_W, MINI_H));
        edges.push({ id: `${id}e${k}`, type: "lane", source: `${id}m${k}`, sourceHandle: "lane-a",
          target: `${id}b`, targetHandle: `lane-b-${k}`, animated: st === "running", style: laneStyle,
          ...ends(row.flow ?? "ab"),
          data: { channel: row.channel, flow: row.flow ?? "ab", status: st, motion,
                  ...(k === 0 ? { label: `${i + 1}. ${shortLabel(full)}`, labelY: y - 4 } : {}) } });
      }
      // As tall as the group, so each party's lane arrives level with it.
      nodes.push(sized<CompData>({ id: `${id}b`, type: "comp", position: { x: COMP_X_B, y: y + (h - blockH) / 2 },
        data: { text: row.b ?? "", status: st, side: "b", fanIn: n }, draggable: false, selectable: false },
        COMP_W, Math.max(COMP_H, blockH)));
      // The note under the group says what each party did, once.
      nodes.push(sized<WideData>({ id: `${id}note`, type: "wide",
        position: { x: COMP_X_A, y: y + h + 4 }, data: { text: `each ${row.many.noun}: ${row.a ?? ""}`, status: st, variant: "note" },
        draggable: false, selectable: false }, COMP_W, 34));
      if (step.stageId === "federated_training") keyUsers.push({ a: `${id}m0`, b: `${id}b`, status: st });
      y += h + 38 + ROW_GAP;
      return;
    }
    if (row.a) nodes.push(sized<CompData>({ id: `${id}a`, type: "comp", position: { x: COMP_X_A, y },
      data: { text: row.a, status: st, side: "a", bus: step.stageId === "pqc_handshake" ? "src" : "dst" },
      draggable: false, selectable: false }, COMP_W, COMP_H));
    if (row.b) nodes.push(sized<CompData>({ id: `${id}b`, type: "comp", position: { x: COMP_X_B, y },
      data: { text: row.b, status: st, side: "b", bus: step.stageId === "pqc_handshake" ? "src" : "dst" },
      draggable: false, selectable: false }, COMP_W, COMP_H));
    if (row.a && row.b && row.flow) {
      edges.push({ id: `${id}e`, type: "lane", source: `${id}a`, sourceHandle: "lane-a", target: `${id}b`,
        targetHandle: "lane-b", animated: st === "running", style: laneStyle, ...ends(row.flow),
        data: { channel: row.channel, flow: row.flow, status: st, motion,
                label: `${i + 1}. ${shortLabel(full)}`, labelY: y + COMP_H / 2 - 8 } });
    } else {
      // A step one side runs alone: its label beside it, where a lane would be.
      nodes.push(sized<WideData>({ id: `${id}l`, type: "wide",
        position: { x: COMP_X_A + COMP_W + 12, y: y + 6 },
        data: { text: `${i + 1}. ${shortLabel(full)}`, status: st, variant: "local" },
        draggable: false, selectable: false }, RIGHT_X - 12 - (COMP_X_A + COMP_W + 12), COMP_H - 12));
    }
    // Eve, where the run put her: on the quantum lane, with the fraction the stage reports.
    const eve = Number(metrics["qkd_eve_fraction"]);
    if (row.channel === "quantum" && step.stageId !== "qsdc_telemetry" && Number.isFinite(eve) && eve > 0) {
      nodes.push(sized({ id: `${id}eve`, type: "eve", position: { x: LANE_MID - 105, y: y + COMP_H / 2 + 8 },
        data: { text: `Eve · intercept-resend ${eve}` }, draggable: false, selectable: false }, 210, 24));
    }
    if (step.stageId === "pqc_handshake") pqc = { a: row.a ? `${id}a` : undefined, b: row.b ? `${id}b` : undefined };
    if (step.stageId === "qsmpc_compute" && row.channel === "classical") keyUsers.push({ a: `${id}a`, b: `${id}b`, status: st });
    y += COMP_H + ROW_GAP;
  });

  // The session key, from the handshake to the stages that consume it (the PSI's HMAC key and the
  // federated masks' seed). Lit while its consumer runs.
  for (const u of keyUsers) {
    const style = { stroke: KEY, strokeWidth: u.status === "running" ? 2.4 : 1.4,
                    strokeDasharray: "2 4", opacity: u.status === "pending" || u.status === "idle" ? 0.45 : 0.95 };
    if (pqc.a && u.a) edges.push({ id: `bus-a-${u.a}`, type: "bus", source: pqc.a, sourceHandle: "bus", target: u.a,
      targetHandle: "bus", animated: u.status === "running", style, data: { busX: BUS_X_A, status: u.status } });
    if (pqc.b && u.b) edges.push({ id: `bus-b-${u.b}`, type: "bus", source: pqc.b, sourceHandle: "bus", target: u.b,
      targetHandle: "bus", animated: u.status === "running", style, data: { busX: BUS_X_B, status: u.status } });
  }

  const height = y + 4;
  nodes.unshift(
    sized<PartyData>({ id: "party-a", type: "party", position: { x: LEFT_X, y: 0 }, zIndex: -1,
      data: { name: spec.a.name, sub: spec.a.sub, height }, draggable: false, selectable: false }, COL_W, height),
    sized<PartyData>({ id: "party-b", type: "party", position: { x: RIGHT_X, y: 0 }, zIndex: -1,
      data: { name: spec.b.name, sub: spec.b.sub, height }, draggable: false, selectable: false }, COL_W, height),
  );
  return { nodes, edges, height };
}

/** Holds the viewport at the scale the frame allows, and follows it when the frame resizes.
 *
 *  And re-measures every node's handles after a change of scale. React Flow measures a handle on the
 *  screen and converts it with the scale it holds at that moment; measured while the scale was
 *  changing, every lane began and ended 11 units inside its boxes, and the boxes, drawn over the
 *  lanes, hid every arrowhead. */
function ViewportSync({ zoom, ids }: { zoom: number; ids: string[] }) {
  const { setViewport } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    void setViewport({ x: 0, y: 0, zoom });
    const t = requestAnimationFrame(() => updateNodeInternals(ids));
    return () => cancelAnimationFrame(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoom, setViewport, updateNodeInternals, ids.join(",")]);
  return null;
}

export default function UseCaseCanvas({
  usecase, activeStage, doneStages, metrics, config = {},
}: {
  usecase: string;
  activeStage: string | null;
  doneStages: Set<string>;
  metrics: Record<string, any>;
  config?: DiagramConfig;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  // The frame's width before React Flow first draws, so its first measurement is at the final scale
  // (see ViewportSync); then followed as the frame resizes.
  const [frameW, setFrameW] = useState<number | null>(null);
  useLayoutEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    setFrameW(Math.max(200, el.clientWidth || W));
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setFrameW(Math.max(200, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const motion = useMemo(() =>
    !(typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches), []);

  const spec = USECASE_SPECS[usecase];
  const steps = useMemo(() => visibleSteps(usecase, config), [usecase, config]);
  const idle = activeStage == null && doneStages.size === 0;
  const status = (s: Step): Status => {
    if (idle) return "idle";
    if (s.stageId === activeStage) return "running";
    if (doneStages.has(s.stageId)) return s.ok && !s.ok(metrics, config) ? "failed" : "done";
    return "pending";
  };
  const { nodes, edges, height } = useMemo(
    () => build(usecase, steps, status, metrics, config, motion),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [usecase, steps, activeStage, doneStages, metrics, config, motion]);
  if (!spec) return null;

  const zoom = Math.max(MIN_SCALE, Math.min(1, (frameW ?? W) / W));
  // Several rows can share a stage (private inference is the FHE lane, the MPC lane and the published
  // model at once), so the line names every row that runs.
  const activeIdx = steps.flatMap((s, i) => (s.stageId === activeStage ? [i] : []));
  const doneCount = steps.filter((s) => doneStages.has(s.stageId)).length;
  const runningLine = activeIdx.length === 0 ? ""
    : `${activeIdx.length > 1 ? `Stages ${activeIdx[0] + 1} to ${activeIdx[activeIdx.length - 1] + 1}`
      : `Stage ${activeIdx[0] + 1}`} of ${steps.length} · `
      + activeIdx.map((i) => steps[i].label(metrics, config).split(" · ")[0]).join(" + ") + " · running";
  return (
    <figure className="ucc" style={{ margin: 0 }}>
      <div className="ucc-status" role="status" aria-live="polite">
        {idle ? `${steps.length} stages · press Start to run them`
          : runningLine || `${doneCount} of ${steps.length} stages done`}
      </div>
      <div ref={frameRef} className="ucc-frame">
        <div style={{ width: W * zoom, height: height * zoom }}>
          {frameW != null && <ReactFlowProvider>
            <ReactFlow nodes={nodes} edges={edges} nodeTypes={NODE_TYPES} edgeTypes={EDGE_TYPES}
              defaultViewport={{ x: 0, y: 0, zoom }} minZoom={MIN_SCALE} maxZoom={1}
              nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}
              panOnDrag={false} zoomOnScroll={false} zoomOnPinch={false} zoomOnDoubleClick={false}
              preventScrolling={false} proOptions={{ hideAttribution: true }}
              aria-label={`${spec.a.name} and ${spec.b.name}: the use case's architecture, stage by stage`}>
              <ViewportSync zoom={zoom} ids={nodes.map((n) => n.id)} />
            </ReactFlow>
          </ReactFlowProvider>}
        </div>
      </div>
      <div className="ucc-legend" aria-label="Channels">
        <span><i className="ucc-sw ucc-sw-classical" /> classical, under the session key</span>
        {!pqcOnly(config) && <span><i className="ucc-sw ucc-sw-quantum" /> quantum, simulated ({qkdName(config)})</span>}
        <span><i className="ucc-sw ucc-sw-fhe" /> CKKS ciphertext</span>
        <span><i className="ucc-sw ucc-sw-mpc" /> MPC secret shares</span>
        <span><i className="ucc-sw ucc-sw-key" /> session key to its consumers</span>
      </div>
      {/* Every stage's label in full, with the run's values: the sequence diagram's labels, which the
          canvas shortens to fit its lanes. */}
      <ol className="ucc-stages">
        {steps.map((s, i) => {
          const st = status(s);
          return (
            <li key={i} className={`ucc-stage ucc-${st}`}>
              <span className="ucc-stage-n">{i + 1}</span>
              <span className="ucc-stage-text">{s.label(metrics, config)}</span>
              {MARK[st] && <span className={`ucc-badge ucc-badge-${st}`}>{MARK[st]}</span>}
            </li>
          );
        })}
      </ol>
      <figcaption className="figure-note">
        No trusted third party and no raw data shared: {spec.a.name} and {spec.b.name} meet over a WAN under
        the session key. MPC-FHE hybrid: linear layers under CKKS, the non-linear argmax as MPC, scored
        against QSMPC-FHE. The quantum channel is simulated, and the proof of concept runs in this tab.
      </figcaption>
    </figure>
  );
}
