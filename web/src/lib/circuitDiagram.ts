/**
 * The trained QCNN as the page runs it, recorded from the circuit function itself rather than drawn from a description (Round 86), so a diagram cannot show gates the simulator does not apply.
 *
 * `recordQcnn` runs `applyQcnnCircuit` against a recording backend and groups the gates into the blocks the circuit announces. `layoutBlocks` then places blocks in columns: a block occupies every wire between its lowest and
 * highest, and goes in the first column where none of those wires is taken.
 */

import { applyQcnnCircuit, type CircuitMark, type GateOps, type QcnnWeights, validateQcnnWeights } from "./quantumSim";

export interface RecordedGate { name: "RY" | "RZ" | "CNOT" | "CRY" | "CRZ" | "CRX" | "X"; wires: number[]; angle?: number }
export type BlockKind = "encode" | "conv" | "pool" | "fc" | "ring";
export interface RecordedBlock {
  kind: BlockKind; wires: number[]; stage?: number; round?: number; gates: RecordedGate[];
  /** For a pooling block, the wire it discards. */
  discarded?: number;
}
export interface RecordedQcnn {
  nQubits: number; nFeatures: number; circuit: string; blocks: RecordedBlock[]; survivors: number[];
  readout: { kind: "z0" } | { kind: "marginal"; wires: number[]; classes: number };
}

export function recordQcnn(w: QcnnWeights): RecordedQcnn {
  const L = validateQcnnWeights(w);
  const nFeatures = w.n_features ?? w.n_qubits;
  const blocks: RecordedBlock[] = [{ kind: "encode", wires: Array.from({ length: nFeatures }, (_, q) => q), gates: Array.from({ length: nFeatures }, (_, q): RecordedGate => ({ name: "RY", wires: [q] })) }];
  let current: RecordedBlock | null = null;
  const start = (m: CircuitMark) => {
    current = m.kind === "conv" ? { kind: "conv", wires: [m.wires[0], m.wires[1]], stage: m.stage, round: m.round, gates: [] }
      : m.kind === "pool" ? { kind: "pool", wires: [m.discarded, m.kept], stage: m.stage, discarded: m.discarded, gates: [] }
      : { kind: m.kind, wires: m.wires.slice(), gates: [] };
    blocks.push(current);
  };
  const gate = (name: RecordedGate["name"], wires: number[], angle?: number) => {
    if (!current) throw new Error("a gate was applied before any block was announced");
    current.gates.push(angle === undefined ? { name, wires } : { name, wires, angle });
  };
  const ops: GateOps = {
    RY: (q, th) => gate("RY", [q], th), RZ: (q, th) => gate("RZ", [q], th), CNOT: (c, t) => gate("CNOT", [c, t]), X: (q) => gate("X", [q]),
    CRY: (c, t, th) => gate("CRY", [c, t], th), CRZ: (c, t, th) => gate("CRZ", [c, t], th), CRX: (c, t, th) => gate("CRX", [c, t], th),
    mark: start,
  };
  const survivors = applyQcnnCircuit(w, L, ops);
  const m = L.m;
  return {
    nQubits: L.nq, nFeatures, circuit: L.circuit, blocks, survivors,
    readout: m > 0 ? { kind: "marginal", wires: survivors.slice(0, m), classes: 1 << m } : { kind: "z0" },
  };
}

export interface PlacedBlock { block: RecordedBlock; column: number; low: number; high: number }

/** The same greedy packing for blocks and for one block's gates: `span` gives the wires an item covers. */
function pack<T>(items: T[], nWires: number, span: (i: T) => number[], place: (i: T, column: number, low: number, high: number) => void): number {
  const free = new Array(nWires).fill(0);
  let columns = 0;
  for (const item of items) {
    const wires = span(item), low = Math.min(...wires), high = Math.max(...wires);
    let column = 0;
    for (let q = low; q <= high; q++) column = Math.max(column, free[q]);
    for (let q = low; q <= high; q++) free[q] = column + 1;
    columns = Math.max(columns, column + 1);
    place(item, column, low, high);
  }
  return columns;
}

export function layoutBlocks(blocks: RecordedBlock[], nWires: number): { placed: PlacedBlock[]; columns: number } {
  const placed: PlacedBlock[] = [];
  const columns = pack(blocks, nWires, (b) => b.wires, (block, column, low, high) => placed.push({ block, column, low, high }));
  return { placed, columns };
}

export interface PlacedGate { gate: RecordedGate; column: number; low: number; high: number }

export function layoutGates(gates: RecordedGate[], nWires: number): { placed: PlacedGate[]; columns: number } {
  const placed: PlacedGate[] = [];
  const columns = pack(gates, nWires, (g) => g.wires, (gate, column, low, high) => placed.push({ gate, column, low, high }));
  return { placed, columns };
}

/** Gate totals by name, for the summary line and for the check against the simulator. */
export function gateCounts(q: RecordedQcnn): Record<string, number> {
  const out: Record<string, number> = {};
  for (const b of q.blocks) for (const g of b.gates) out[g.name] = (out[g.name] ?? 0) + 1;
  return out;
}
