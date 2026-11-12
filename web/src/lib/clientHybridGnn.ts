/**
 * The compact transaction GNN under MPC, on one held-out transaction (Round 86): CKKS for its linear maps, garbled circuits for its ReLUs and argmax.
 *
 * The neighbourhood is a tree of fixed, public shape: two endpoints and `per_direction` slots per direction at each. The client, who holds the transactions, computes every slot and endpoint vector in the
 * clear and encrypts them; the server, who holds the weights, learns neither those vectors nor which slots are empty, because an empty slot is a zero vector under encryption. The first layer's map is
 * one matrix per kind of node, applied to every node at once: the nodes sit side by side in one ciphertext, in blocks, so one evaluation per output dimension serves all the nodes. That is the SIMD a GNN
 * invites, and what the ciphertext count of Bilgi et al. and TGHE exploit. The second layer sums over shares, which costs no AND gates, so the circuits pay only for the accounts that carry an embedding.
 *
 * The protocol and its caveats are the MLP's (`clientHybridMlp.ts`, `clientHybridCnn.ts`): Gazelle's shape, semi-honest, statistical masks, both parties in this tab.
 */

import { Arena, Client, type GcPool, type GcTotals, type HybridCnnProgress, type HybridCnnResult, type Mpc, reduceBlocks, relu, reluLayer, sealed, Server, sumAbs, toRing } from "./clientHybridCnn";
import { blockFor, type CkksTally, denseUnderCkks } from "./clientHybridMlp";
import { hybridCnnCfg } from "./config";

export interface DenseLayer { weight: number[][]; bias: number[] }
export interface GnnNeighbourEdge { slot: number; from: string; to: string; amount: number; currency: string; format: string; minutes_before_scored: number }

export interface GnnWeights {
  kind: "gnn"; key: string; classes: Record<string, string>; n_parameters: number; relu_count: number;
  query: { features: string[]; scaling: { center: number[]; scale: number[]; clip: number } };
  endpoints: { features: string[]; scaling: { kind: string; center: number[]; scale: number[]; clip: number } };
  slots: { per_direction: number; order: string[]; formats: string[]; features: string[]; scaling: { scale: number[]; clip: number } };
  layers: { q1: DenseLayer; q2: DenseLayer; a_end: number[][]; a_slot: number[][]; second: { me: number[][]; incoming: number[][]; outgoing: number[][]; bias: number[] }[]; c: DenseLayer; d: DenseLayer };
  fixtures: { labels: number[]; logits: number[][]; row_index_in_test_window: number[]; x: number[][]; endpoints: number[][][]; slots: number[][][]; neighbourhoods: GnnNeighbourEdge[][] };
}

/** One row as the network reads it, after the client's scaling. */
export interface GnnRow { x: number[]; ends: number[][]; slots: number[][] }

export interface HybridGnnResult extends HybridCnnResult {
  /** The accounts that carry an embedding, and the circuits' workers: 1 when the batches ran one after another. */
  accounts: number; gcWorkers: number; gcWallMs: number;
}

const clip = (v: number, c: number) => Math.min(c, Math.max(-c, v));

/** The client's scaling of a fixture row. An empty slot's zeros stay zero, because the slot features are scaled but not centred. */
export function gnnRow(w: GnnWeights, index: number): GnnRow {
  const q = w.query.scaling, e = w.endpoints.scaling, s = w.slots.scaling;
  return {
    x: w.fixtures.x[index].map((v, i) => clip((v - q.center[i]) / q.scale[i], q.clip)),
    ends: w.fixtures.endpoints[index].map((row) => row.map((v, i) => clip((v - e.center[i]) / e.scale[i], e.clip))),
    slots: w.fixtures.slots[index].map((row) => row.map((v, i) => clip(v / s.scale[i], s.clip))),
  };
}

const dot = (m: number[][], v: ArrayLike<number>) => m.map((r) => r.reduce((a, c, i) => a + c * v[i], 0));
const plus = (a: number[], b: number[]) => a.map((v, i) => v + b[i]);
const rect = (v: number[]) => v.map((x) => Math.max(0, x));

/** The second layer as one dense map on the first layer's outputs, [q1, then 16 per account]: the part of the graph's structure the server's weights see. */
export function secondLayer(w: GnnWeights): { in: number; out: number; weight: number[][]; bias: number[] } {
  const L = w.layers, D = w.slots.per_direction, H = L.a_slot.length, Q = L.q1.weight.length;
  const accounts = 2 + 4 * D, width = Q + accounts * H, zeros = () => new Array<number>(width).fill(0);
  const weight: number[][] = L.q2.weight.map((row) => { const out = zeros(); row.forEach((v, i) => { out[i] = v; }); return out; });
  const bias = [...L.q2.bias];
  L.second.forEach((b, r) => {
    const first = 2 + 2 * r * D;
    b.bias.forEach((v, k) => {
      const out = zeros();
      for (let i = 0; i < H; i++) {
        out[Q + r * H + i] = b.me[k][i];
        for (let s = 0; s < D; s++) { out[Q + (first + s) * H + i] = b.incoming[k][i]; out[Q + (first + D + s) * H + i] = b.outgoing[k][i]; }
      }
      weight.push(out);
      bias.push(v);
    });
  });
  return { in: width, out: weight.length, weight, bias };
}

/** The same network in the clear: what the fixture's logits should be. */
export function gnnForward(w: GnnWeights, row: GnnRow): number[] {
  const L = w.layers;
  const q1 = rect(plus(dot(L.q1.weight, row.x), L.q1.bias));
  const nodes = [...row.ends.map((e) => rect(dot(L.a_end, e))), ...row.slots.map((s) => rect(dot(L.a_slot, s)))];
  const second = secondLayer(w);
  const m2 = rect(plus(dot(second.weight, [...q1, ...nodes.flat()]), second.bias));
  const head = rect(plus(dot(L.c.weight, m2), L.c.bias));
  return plus(dot(L.d.weight, head), L.d.bias);
}

export async function hybridGnnInference(w: GnnWeights, row: GnnRow, opts: { onProgress?: (p: HybridCnnProgress) => void; pool?: GcPool | null } = {}): Promise<HybridGnnResult> {
  const t0 = performance.now();
  if (w.kind !== "gnn") throw new Error(`the hybrid GNN path takes kind "gnn", and this file says ${w.kind}`);
  const L = w.layers, D = w.slots.per_direction, H = L.a_slot.length;
  const accounts = 2 + 4 * D;
  if (row.slots.length !== 4 * D || row.ends.length !== 2) throw new Error(`the tree has ${accounts} accounts and this row has ${row.ends.length + row.slots.length}`);
  const s = await sealed();
  const mpc: Mpc = await import("./clientMPC");
  await mpc.warmMPC();
  const cfg = hybridCnnCfg();
  const say = (phase: HybridCnnProgress["phase"], done: number, total: number) => opts.onProgress?.({ phase, done, total });
  const second = secondLayer(w);
  const dense: { in: number; out: number; weight: number[][]; bias: number[] }[] = [second, { in: second.out, out: L.c.weight.length, ...L.c }, { in: L.c.weight.length, out: L.d.weight.length, ...L.d }];
  const q1 = { in: L.q1.weight[0].length, out: L.q1.weight.length, ...L.q1 };
  const block = blockFor(Math.max(L.a_end[0].length, L.a_slot[0].length));
  if (accounts * block > s.slots) throw new Error(`${accounts} accounts of ${block} slots do not fit the ring's ${s.slots} slots`);
  const widest = Math.max(blockFor(q1.in), block, ...dense.map((d) => blockFor(d.in)));
  const steps = [0];                                // 0 is the conjugation
  for (let step = 1; step < widest; step <<= 1) steps.push(step);

  say("keys", 0, 1);
  const client = new Client(s, steps);
  const server = new Server(s, client.publicKey, client.galoisKeys);
  const gc: GcTotals = { and: 0, batches: 0, bytes: 0, messages: 0, compileMs: 0, mpcMs: 0 };
  const tally: CkksTally = { bytes: 0, ms: 0, statistical: 0, maxLayerError: 0, clientView: [] };
  const arena = new Arena();
  let reluCount = 0, gcWallMs = 0;
  const rx = Float64Array.from(row.x), zeros = new Float64Array(rx.length);
  try {
    // ---- First layer: the query channel as a dense layer, and the accounts side by side in one ciphertext
    const qOut = denseUnderCkks(s, client, server, arena, q1, rx, zeros, rx, cfg.dense_mask_bits, tally);
    const layout = new Float64Array(s.slots);
    [...row.ends, ...row.slots].forEach((v, j) => v.forEach((x, i) => { layout[j * block + i] = x; }));
    let t = performance.now();
    const up = client.encrypt(layout);
    tally.bytes += up.length;
    const ctNodes = server.receive(up, arena);
    tally.ms += performance.now() - t;
    const nOut = accounts * H;
    const yN = new Float64Array(nOut), rN = new Float64Array(nOut), outN = new Float64Array(nOut);
    for (let k = 0; k < H; k++) {
      t = performance.now();
      const tile = new Float64Array(s.slots);
      for (let j = 0; j < accounts; j++) (j < 2 ? L.a_end : L.a_slot)[k].forEach((v, i) => { tile[j * block + i] = v; });
      const { bytes, mask } = server.evaluate(ctNodes, tile, block, accounts, () => 0, cfg.dense_mask_bits);
      tally.bytes += bytes.length;
      const decoded = client.decrypt(bytes);
      tally.ms += performance.now() - t;
      const clear = reduceBlocks(layout.map((v, i) => v * tile[i]), block);
      tally.statistical += sumAbs(clear) / 2 ** (cfg.dense_mask_bits + 1);
      for (let j = 0; j < accounts; j++) {
        const n = j * H + k;
        yN[n] = decoded[j * block]; rN[n] = mask[j * block]; outN[n] = clear[j * block]; tally.clientView.push(Math.abs(yN[n]));
        tally.maxLayerError = Math.max(tally.maxLayerError, Math.abs(yN[n] - rN[n] - outN[n]));
      }
    }
    say("dense", 1, 4);
    type F = Float64Array<ArrayBufferLike>;
    const join = (a: F, b: F): F => { const v = new Float64Array(a.length + b.length); v.set(a); v.set(b, a.length); return v; };
    let y: F = join(qOut.y, yN), r: F = join(qOut.r, rN), out: F = join(qOut.out, outN);
    const gcLayer = async (a: F, b: F, c: F): Promise<{ share: { client: F; server: F }; clear: F }> => {
      reluCount += a.length;
      let done = 0;
      const batch = Math.min(a.length, cfg.conv_relu_batch);
      const started = performance.now();
      const shared = await reluLayer(mpc, a, b, cfg.dense_mask_bits, batch, gc, () => say("relu", (done += batch), a.length), opts.pool);
      gcWallMs += performance.now() - started;
      for (let j = 0; j < c.length; j++) tally.statistical += relu(c[j]) / 2 ** cfg.dense_mask_bits;
      return { share: shared, clear: c.map(relu) };
    };
    let { share, clear } = await gcLayer(y, r, out);

    // ---- The dense layers on shares, ending at the logits
    let yLast: F = new Float64Array(0), rLast: F = new Float64Array(0);
    for (let k = 0; k < dense.length; k++) {
      const last = k === dense.length - 1;
      const res = denseUnderCkks(s, client, server, arena, dense[k], share.client, share.server, clear, last ? cfg.logit_mask_bits : cfg.dense_mask_bits, tally);
      say("dense", k + 2, 4);
      if (last) { yLast = res.y; rLast = res.r; clear = res.out; break; }
      y = res.y; r = res.r; out = res.out;
      ({ share, clear } = await gcLayer(y, r, out));
    }

    // ---- The class: the argmax over shares of the logits
    const n = yLast.length, ring = 2 ** 32;
    const a = Uint32Array.from({ length: n }, (_, k) => toRing((yLast[k] + mpc.OFFSET) * mpc.SCALE, ring));
    const b = Uint32Array.from({ length: n }, (_, k) => toRing(-rLast[k] * mpc.SCALE, ring));
    const arg = await mpc.secureArgmaxShares(a, b);
    gc.bytes += arg.bytes; gc.messages += arg.messages; gc.mpcMs += arg.mpcMs; gc.and += arg.andOps;
    say("argmax", 1, 1);
    const logits = Array.from(yLast, (v, k) => v - rLast[k]);
    let clearPrediction = 0;
    for (let k = 1; k < n; k++) if (clear[k] > clear[clearPrediction]) clearPrediction = k;
    return {
      prediction: arg.arg, clearPrediction, matchesClear: arg.arg === clearPrediction,
      logits, logitMaxAbsError: Math.max(...logits.map((v, k) => Math.abs(v - clear[k]))),
      reluCount, andGates: gc.and, batches: gc.batches, gcBytes: gc.bytes, gcMessages: gc.messages,
      gcCompileMs: gc.compileMs, gcMs: gc.mpcMs, ckksBytes: tally.bytes, keyBytes: client.publicKey.length + client.galoisKeys.length, ckksMs: tally.ms,
      rotations: server.rotations, conjugations: server.conjugations, totalMs: performance.now() - t0,
      maskBits: { conv: cfg.dense_mask_bits, dense: cfg.dense_mask_bits, logit: cfg.logit_mask_bits },
      statisticalDistance: tally.statistical, statisticalDistanceLog2: Math.log2(tally.statistical), maxLayerError: tally.maxLayerError,
      clientViewMedianAbs: tally.clientView.sort((x, v) => x - v)[tally.clientView.length >> 1],
      accounts, gcWorkers: opts.pool?.workers ?? 1, gcWallMs,
    };
  } finally {
    arena.close(); server.close(); client.close();
  }
}

/** Off the page's thread, as the MLP's run is. The worker opens the garbling pool itself; a worker that fails is reported, never retried on the page's thread. */
export async function hybridGnnInferenceOffThread(w: GnnWeights, index: number, opts: { onProgress?: (p: HybridCnnProgress) => void } = {}): Promise<HybridGnnResult & { ranIn: string }> {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    return { ...(await hybridGnnInference(w, gnnRow(w, index), opts)), ranIn: "the calling thread" };
  }
  const worker = new Worker(new URL("./clientHybridGnn.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ progress?: HybridCnnProgress; ok?: boolean; result?: HybridGnnResult; error?: string }>) => {
        const d = e.data;
        if (d.progress) { opts.onProgress?.(d.progress); return; }
        if (d.ok && d.result) resolve({ ...d.result, ranIn: "a Web Worker" });
        else reject(new Error(`in its worker: ${d.error ?? "no result"}`));
      };
      worker.onerror = (e) => reject(new Error(`its worker failed: ${e.message || "the worker could not be loaded"}`));
      worker.onmessageerror = () => reject(new Error("its worker's result could not be read back"));
      worker.postMessage({ weights: w, index });
    });
  } finally {
    worker.terminate();
  }
}
