// The circuits this page simulates, drawn from the code that runs them (Round 86). The block view shows the architecture a QCNN paper's figure shows; one block is then expanded gate by gate.
// Nothing here is typed in: the blocks and gates come from `recordQcnn`, which runs the simulator's own circuit function against a recording backend.
import { useEffect, useMemo, useState } from "react";
import { gateCounts, layoutBlocks, layoutGates, type PlacedBlock, type RecordedQcnn, recordQcnn } from "../lib/circuitDiagram";
import { loadQcnnWeights } from "../lib/clientCompute";
import { QOTP_COMPARISON_ROUNDS, type QcnnWeights } from "../lib/quantumSim";

const WIRE = 30, LEFT = 96, TOP = 22;
let COL = 44;
/** Wide circuits draw their columns closer, so the readout stays on screen at a desktop width. */
const setColumnWidth = (columns: number) => { COL = columns > 16 ? 36 : 44; };
const ink = "var(--text)", muted = "var(--muted)", edge = "var(--border)", accent = "var(--accent)", quantum = "var(--quantum)";
/** Wires run under their gates; the label's halo keeps its text clear of the wire. */
const halo = { paintOrder: "stroke", stroke: "var(--panel)", strokeWidth: 6 } as const;

const y = (q: number) => TOP + q * WIRE + WIRE / 2;
const x = (c: number) => LEFT + c * COL;

function BlockGlyph({ p, stageLabel }: { p: PlacedBlock; stageLabel: string }) {
  const { block, column, low, high } = p, cx = x(column) + COL / 2;
  if (block.kind === "encode") {
    return <g>{block.wires.map((q) => <g key={q}><rect x={cx - 16} y={y(q) - 11} width={32} height={22} rx={3} fill="var(--panel)" stroke={accent} /><text x={cx} y={y(q) + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>RY</text></g>)}<title>{`Encoding: RY(c pi tanh z) on each of the ${block.wires.length} feature wires`}</title></g>;
  }
  if (block.kind === "pool") {
    const d = block.discarded!, k = block.wires[1];
    return <g><line x1={cx} x2={cx} y1={y(d)} y2={y(k)} stroke={quantum} /><circle cx={cx} cy={y(d)} r={3.5} fill={quantum} /><rect x={cx - 12} y={y(k) - 11} width={24} height={22} rx={3} fill="var(--panel)" stroke={quantum} /><text x={cx} y={y(k) + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>P</text><title>{`Pooling: wire ${d} conditions wire ${k}, then is not used again (${block.gates.map((g) => g.name).join(", ")})`}</title></g>;
  }
  if (block.kind === "fc") {
    return <g>{block.wires.map((q) => <g key={q}><rect x={cx - 12} y={y(q) - 11} width={24} height={22} rx={3} fill="var(--panel)" stroke={accent} /><text x={cx} y={y(q) + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>F</text></g>)}<title>Fully-connected block: RZ then RY on each surviving wire</title></g>;
  }
  if (block.kind === "ring") {
    return <g><rect x={cx - 14} y={y(low) - 11} width={28} height={y(high) - y(low) + 22} rx={3} fill="var(--panel)" stroke={edge} /><text x={cx} y={(y(low) + y(high)) / 2 + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>CX</text><title>A ring of CNOTs over the surviving wires</title></g>;
  }
  // conv: one filter, every position
  const label = `U${(block.stage ?? 0) + 1}`;
  return <g><rect x={cx - 16} y={y(low) - 11} width={32} height={y(high) - y(low) + 22} rx={3} fill="var(--panel)" stroke={quantum} /><text x={cx} y={(y(low) + y(high)) / 2 + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>{label}</text><title>{`${stageLabel}: convolutional block, stage ${(block.stage ?? 0) + 1}, round ${(block.round ?? 0) + 1}, on wires ${block.wires.join(" and ")}. ${block.gates.length} gates, angles shared by every block of the stage.`}</title></g>;
}

function BlockDiagram({ r, weights }: { r: RecordedQcnn; weights: QcnnWeights }) {
  const { placed, columns } = useMemo(() => layoutBlocks(r.blocks, r.nQubits), [r]);
  setColumnWidth(columns);
  const lastUse = useMemo(() => { const m = new Array(r.nQubits).fill(columns); placed.forEach((p) => { if (p.block.kind === "pool") m[p.block.discarded!] = p.column; }); return m; }, [placed, columns, r.nQubits]);
  const end = x(columns) + 76, h = y(r.nQubits - 1) + WIRE;
  const name = (q: number) => (q < r.nFeatures ? (weights.feature_names?.[q] ?? `q${q}`) : "|0⟩");
  const stageNames = weights.circuit?.name === "hur-su4" ? "SU(4) block" : "two-qubit block";
  return (
    <div style={{ overflowX: "auto" }}>
      <svg role="img" aria-label={`Block diagram of the trained ${r.circuit} circuit on ${r.nQubits} wires`} width={end + 8} height={h} style={{ display: "block" }}>
        {Array.from({ length: r.nQubits }, (_, q) => (
          <g key={q}>
            <text x={LEFT - 8} y={y(q) + 4} textAnchor="end" fontSize={11} fill={muted}>{name(q).length > 12 ? `${name(q).slice(0, 11)}…` : name(q)}<title>{name(q)}</title></text>
            <line x1={LEFT} x2={x(lastUse[q]) + COL / 2} y1={y(q)} y2={y(q)} stroke={edge} />
            {lastUse[q] < columns && <line x1={x(lastUse[q]) + COL / 2} x2={end} y1={y(q)} y2={y(q)} stroke={edge} strokeDasharray="2 4" opacity={0.55} />}
          </g>
        ))}
        {placed.map((p, i) => <BlockGlyph key={i} p={p} stageLabel={stageNames} />)}
        {r.readout.kind === "z0"
          ? <g><rect x={x(columns) + 8} y={y(0) - 11} width={48} height={22} rx={3} fill="var(--panel)" stroke={accent} /><text x={x(columns) + 32} y={y(0) + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>⟨Z₀⟩</text><title>The readout: the expectation of Z on wire 0</title></g>
          : r.readout.wires.map((q) => <g key={q}><rect x={x(columns) + 8} y={y(q) - 11} width={48} height={22} rx={3} fill="var(--panel)" stroke={accent} /><text x={x(columns) + 32} y={y(q) + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>prob</text><title>{`Bit ${r.readout.kind === "marginal" ? r.readout.wires.indexOf(q) : 0} of the class index`}</title></g>)}
      </svg>
    </div>
  );
}

function GateDiagram({ r }: { r: RecordedQcnn }) {
  const block = r.blocks.find((b) => b.kind === "conv")!;
  const wires = block.wires.slice().sort((a, b) => a - b);
  const remap = (q: number) => wires.indexOf(q);
  const gates = block.gates.map((g) => ({ ...g, wires: g.wires.map(remap) }));
  const { placed, columns } = layoutGates(gates, 2);
  const cw = 40, X = (c: number) => 52 + c * cw, Y = (q: number) => 20 + q * 36, end = X(columns) + 8;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg role="img" aria-label="One convolutional block, gate by gate" width={end + 8} height={Y(1) + 24} style={{ display: "block" }}>
        {[0, 1].map((q) => <g key={q}><text x={40} y={Y(q) + 4} textAnchor="end" fontSize={11} fill={muted}>{`wire ${q === 0 ? "a" : "b"}`}</text><line x1={46} x2={end} y1={Y(q)} y2={Y(q)} stroke={edge} /></g>)}
        {placed.map(({ gate: g, column, low, high }, i) => {
          const cx = X(column) + cw / 2;
          if (g.wires.length === 1) {
            return <g key={i}><rect x={cx - 15} y={Y(g.wires[0]) - 12} width={30} height={24} rx={3} fill="var(--panel)" stroke={quantum} /><text x={cx} y={Y(g.wires[0]) + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>{g.name}</text>{g.angle !== undefined && <title>{`${g.name}(${g.angle.toFixed(4)}), a trained angle`}</title>}</g>;
          }
          const [c, t] = g.wires;
          return (
            <g key={i}>
              <line x1={cx} x2={cx} y1={Y(low)} y2={Y(high)} stroke={ink} />
              <circle cx={cx} cy={Y(c)} r={3.5} fill={ink} />
              {g.name === "CNOT"
                ? <g><circle cx={cx} cy={Y(t)} r={9} fill="var(--panel)" stroke={ink} /><line x1={cx - 9} x2={cx + 9} y1={Y(t)} y2={Y(t)} stroke={ink} /><line x1={cx} x2={cx} y1={Y(t) - 9} y2={Y(t) + 9} stroke={ink} /></g>
                : <g><rect x={cx - 15} y={Y(t) - 12} width={30} height={24} rx={3} fill="var(--panel)" stroke={quantum} /><text x={cx} y={Y(t) + 4} textAnchor="middle" fontSize={10} fill={ink} style={halo}>{g.name}</text></g>}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ComparisonDiagram({ qubits }: { qubits: number }) {
  // The first wire sits low enough that "CNOT ring" above it stays inside the SVG: at 18 its ascenders were clipped by a pixel.
  const n = Math.min(qubits, 4), X = (c: number) => 56 + c * 44, Y = (q: number) => 26 + q * 28, end = X(5) + 6;
  const box = (cx: number, q: number, t: string, s: string) => <g key={`${t}${q}`}><rect x={cx - 14} y={Y(q) - 11} width={28} height={22} rx={3} fill="var(--panel)" stroke={s} /><text x={cx} y={Y(q) + 4} textAnchor="middle" fontSize={11} fill={ink} style={halo}>{t}</text></g>;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg role="img" aria-label="One round of the untrained comparison circuit" width={end + 90} height={Y(n - 1) + 24} style={{ display: "block" }}>
        {Array.from({ length: n }, (_, q) => <g key={q}><text x={44} y={Y(q) + 4} textAnchor="end" fontSize={11} fill={muted}>{`q${q}`}</text><line x1={50} x2={end} y1={Y(q)} y2={Y(q)} stroke={edge} /></g>)}
        {Array.from({ length: n }, (_, q) => box(X(0) + 22, q, "RY", accent))}
        {Array.from({ length: n }, (_, q) => box(X(1) + 22, q, "XZ", muted))}
        {Array.from({ length: n }, (_, q) => box(X(2) + 22, q, "H", quantum))}
        <g><line x1={X(3) + 22} x2={X(3) + 22} y1={Y(0)} y2={Y(n - 1)} stroke={ink} /><text x={X(3) + 22} y={Y(0) - 8} textAnchor="middle" fontSize={10} fill={muted}>CNOT ring</text></g>
        {Array.from({ length: n }, (_, q) => box(X(4) + 22, q, "T", quantum))}
        <text x={X(5) + 12} y={(Y(0) + Y(n - 1)) / 2 + 4} fontSize={12} fill={ink}>{`× ${QOTP_COMPARISON_ROUNDS} rounds`}</text>
      </svg>
    </div>
  );
}

export default function QcnnCircuitCard({ usecase, qubits, hasComparison }: { usecase: string; qubits: number; hasComparison: boolean }) {
  const [open, setOpen] = useState(false);
  const [w, setW] = useState<QcnnWeights | null>(null);
  const [failed, setFailed] = useState("");
  useEffect(() => {
    if (!open) return;
    let live = true;
    setFailed("");
    loadQcnnWeights(usecase, qubits).then((v) => { if (live) { setW(v); if (!v) setFailed("no trained circuit ships for this page"); } })
      .catch((e: Error) => { if (live) setFailed(e.message); });
    return () => { live = false; };
  }, [open, usecase, qubits]);
  const recorded = useMemo(() => { try { return w ? recordQcnn(w) : null; } catch (e) { setFailed((e as Error).message); return null; } }, [w]);
  const counts = recorded ? gateCounts(recorded) : null;
  const total = counts ? Object.values(counts).reduce((a, b) => a + b, 0) : 0;
  return (
    <details className="card details" style={{ marginTop: 14 }} id="circuit" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>The quantum circuits on this page, drawn from the code that runs them</summary>
      <div className="details-body">
        {failed && <p className="muted">The diagram could not be drawn: {failed}.</p>}
        {recorded && w && counts && (<>
          <h3 style={{ marginTop: 8 }}>The trained QCNN</h3>
          <p>{recorded.circuit === "hur-su4" ? "Hur, Kim and Park's SU(4) circuit" : "A tree of two-qubit blocks"}, {recorded.nQubits} wires, {w.theta.length} trained angles, {total} gates ({Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(", ")}). U is one convolutional block, the same angles at every position of its stage. P conditions a kept wire on a discarded one, which is dashed afterwards. F is the final rotations on the survivors.</p>
          <BlockDiagram r={recorded} weights={w} />
          <h3 style={{ marginTop: 12 }}>One U block, gate by gate</h3>
          <GateDiagram r={recorded} />
          {w.circuit?.simulability && <p>{w.circuit.simulability}.</p>}
        </>)}
        {hasComparison && (<>
          <h3 style={{ marginTop: 12 }}>The QSMPC-FHE comparison circuit</h3>
          <p>Untrained, and it predicts nothing: encode, pad each wire with a seeded X and Z, then the rounds below, shown on the first {Math.min(qubits, 4)} of {qubits} wires. It exists to count T gates and to show the pad structure.</p>
          <ComparisonDiagram qubits={qubits} />
        </>)}
      </div>
    </details>
  );
}
