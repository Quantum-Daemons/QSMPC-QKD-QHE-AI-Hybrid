/**
 * A published tabular MLP under MPC, on one held-out row (Round 86, audit item B19): CKKS for its linear layers, garbled circuits for its ReLUs and argmax.
 *
 * The protocol is the medical CNN's, Gazelle's shape, with the same statistical masks and the same caveats (`clientHybridCnn.ts` states them), minus the
 * convolution: the client scales the row and encrypts it, the server evaluates each dense layer and masks every slot, a garbled circuit applies the ReLUs on
 * additive shares, and the last layer's shares go into the 2PC argmax. A layer wider than one ciphertext is split over several.
 */

import { Arena, Client, type GcTotals, type HybridCnnProgress, type HybridCnnResult, type Mpc, reduceBlocks, relu, reluLayer, sealed, Server, sumAbs, toRing } from "./clientHybridCnn";
import { hybridCnnCfg } from "./config";

export interface MlpLayer { in: number; out: number; weight: number[][]; bias: number[]; activation: "relu" | "none" }
export interface MlpWeights {
  kind: "mlp"; key: string; n_features: number; features: string[];
  scaling: { kind: string; center: number[]; scale: number[]; clip: number };
  layers: MlpLayer[]; n_parameters: number; classes: Record<string, string>;
  fixtures: { rows: number[][]; labels: number[]; logits: number[][] };
}

const blockFor = (n: number) => 1 << Math.ceil(Math.log2(Math.max(n, 2)));

/** The row as the graph sees it. This happens on the client, before anything is encrypted. */
export function scaleRow(w: MlpWeights, row: number[]): number[] {
  const { center, scale, clip } = w.scaling;
  return row.map((v, i) => Math.min(clip, Math.max(-clip, (v - center[i]) / scale[i])));
}

export async function hybridMlpInference(w: MlpWeights, row: number[], opts: { onProgress?: (p: HybridCnnProgress) => void } = {}): Promise<HybridCnnResult> {
  const t0 = performance.now();
  if (w.kind !== "mlp") throw new Error(`the hybrid MLP path takes kind "mlp", and this file says ${w.kind}`);
  if (row.length !== w.n_features) throw new Error(`the model takes ${w.n_features} features and the row has ${row.length}`);
  const s = await sealed();
  const mpc: Mpc = await import("./clientMPC");
  await mpc.warmMPC();
  const { slots } = s;
  const cfg = hybridCnnCfg();
  const say = (phase: HybridCnnProgress["phase"], done: number, total: number) => opts.onProgress?.({ phase, done, total });
  const widest = Math.max(...w.layers.map((d) => blockFor(d.in)));
  const steps = [0];                                // 0 is the conjugation
  for (let step = 1; step < widest; step <<= 1) steps.push(step);

  say("keys", 0, 1);
  const client = new Client(s, steps);
  const server = new Server(s, client.publicKey, client.galoisKeys);
  const gc: GcTotals = { and: 0, batches: 0, bytes: 0, messages: 0, compileMs: 0, mpcMs: 0 };
  let ckksBytes = 0, ckksMs = 0, statistical = 0, maxLayerError = 0, reluCount = 0;
  const clientView: number[] = [];
  const arena = new Arena();
  try {
    let clientShare: Float64Array<ArrayBufferLike> = Float64Array.from(scaleRow(w, row));
    let serverShare: Float64Array<ArrayBufferLike> = new Float64Array(clientShare.length);
    let clearVec: Float64Array<ArrayBufferLike> = clientShare.slice();
    let yLast: Float64Array<ArrayBufferLike> = new Float64Array(0), rLast: Float64Array<ArrayBufferLike> = new Float64Array(0);

    for (let k = 0; k < w.layers.length; k++) {
      const d = w.layers[k];
      if (clientShare.length !== d.in) throw new Error(`layer ${k} takes ${d.in} inputs and has ${clientShare.length}`);
      const last = k === w.layers.length - 1;
      const maskBits = last ? cfg.logit_mask_bits : cfg.dense_mask_bits;
      const b = blockFor(d.in);
      const perCiphertext = Math.floor(slots / b);
      if (perCiphertext < 1) throw new Error(`layer ${k} takes ${d.in} inputs and the ring has ${slots} slots`);
      const lay = (v: ArrayLike<number>) => {
        const out = new Float64Array(slots);
        for (let j = 0; j < perCiphertext; j++) for (let i = 0; i < d.in; i++) out[j * b + i] = v[i];
        return out;
      };
      let t = performance.now();
      const up = client.encrypt(lay(clientShare));
      ckksBytes += up.length;
      const ct = server.addShare(server.receive(up, arena), lay(serverShare), arena);
      ckksMs += performance.now() - t;

      const y = new Float64Array(d.out), r = new Float64Array(d.out), out = new Float64Array(d.out);
      for (let at = 0; at < d.out; at += perCiphertext) {
        t = performance.now();
        const count = Math.min(perCiphertext, d.out - at);
        const tile = new Float64Array(slots);
        for (let j = 0; j < count; j++) for (let i = 0; i < d.in; i++) tile[j * b + i] = d.weight[at + j][i] ?? 0;
        const { bytes, mask } = server.evaluate(ct, tile, b, count, (j) => d.bias[at + j] ?? 0, maskBits);
        ckksBytes += bytes.length;
        const decoded = client.decrypt(bytes);
        ckksMs += performance.now() - t;
        // The clear values, for the bound and the self-test only.
        const clear = reduceBlocks(lay(clearVec).map((v, i) => v * tile[i]), b);
        for (let j = 0; j < count; j++) clear[j * b] += d.bias[at + j] ?? 0;
        statistical += sumAbs(clear) / 2 ** (maskBits + 1);
        for (let j = 0; j < count; j++) {
          const n = at + j;
          y[n] = decoded[j * b]; r[n] = mask[j * b]; out[n] = clear[j * b]; clientView.push(Math.abs(y[n]));
          maxLayerError = Math.max(maxLayerError, Math.abs(y[n] - r[n] - out[n]));
        }
      }
      say("dense", k + 1, w.layers.length);
      if (last) { yLast = y; rLast = r; clearVec = out; break; }
      if (d.activation !== "relu") throw new Error(`the hybrid path expects a ReLU after layer ${k}, and the file says ${d.activation}`);
      reluCount += d.out;
      let done = 0;
      const batch = Math.min(d.out, cfg.conv_relu_batch);
      const shared = await reluLayer(mpc, y, r, cfg.dense_mask_bits, batch, gc, () => say("relu", (done += batch), d.out));
      for (let j = 0; j < d.out; j++) statistical += relu(out[j]) / 2 ** cfg.dense_mask_bits;
      clientShare = shared.client; serverShare = shared.server;
      clearVec = out.map(relu);
    }

    // ---- The class: the existing argmax, on shares of the logits
    const n = yLast.length, ring = 2 ** 32;
    const a = Uint32Array.from({ length: n }, (_, k) => toRing((yLast[k] + mpc.OFFSET) * mpc.SCALE, ring));
    const b = Uint32Array.from({ length: n }, (_, k) => toRing(-rLast[k] * mpc.SCALE, ring));
    const arg = await mpc.secureArgmaxShares(a, b);
    gc.bytes += arg.bytes; gc.messages += arg.messages; gc.mpcMs += arg.mpcMs; gc.and += arg.andOps;
    say("argmax", 1, 1);

    const logits = Array.from(yLast, (v, k) => v - rLast[k]);
    let clearPrediction = 0;
    for (let k = 1; k < n; k++) if (clearVec[k] > clearVec[clearPrediction]) clearPrediction = k;
    return {
      prediction: arg.arg, clearPrediction, matchesClear: arg.arg === clearPrediction,
      logits, logitMaxAbsError: Math.max(...logits.map((v, k) => Math.abs(v - clearVec[k]))),
      reluCount, andGates: gc.and, batches: gc.batches, gcBytes: gc.bytes, gcMessages: gc.messages,
      gcCompileMs: gc.compileMs, gcMs: gc.mpcMs, ckksBytes, keyBytes: client.publicKey.length + client.galoisKeys.length, ckksMs,
      rotations: server.rotations, conjugations: server.conjugations, totalMs: performance.now() - t0,
      maskBits: { conv: cfg.dense_mask_bits, dense: cfg.dense_mask_bits, logit: cfg.logit_mask_bits },
      statisticalDistance: statistical, statisticalDistanceLog2: Math.log2(statistical), maxLayerError,
      clientViewMedianAbs: clientView.sort((x, y) => x - y)[clientView.length >> 1],
    };
  } finally {
    arena.close(); server.close(); client.close();
  }
}

/** Off the page's thread, as the CNN's run is: a worker that fails is reported, never retried on the page's thread. */
export async function hybridMlpInferenceOffThread(w: MlpWeights, row: number[], opts: { onProgress?: (p: HybridCnnProgress) => void } = {}): Promise<HybridCnnResult & { ranIn: string }> {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    return { ...(await hybridMlpInference(w, row, opts)), ranIn: "the calling thread" };
  }
  const worker = new Worker(new URL("./clientHybridMlp.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ progress?: HybridCnnProgress; ok?: boolean; result?: HybridCnnResult; error?: string }>) => {
        const d = e.data;
        if (d.progress) { opts.onProgress?.(d.progress); return; }
        if (d.ok && d.result) resolve({ ...d.result, ranIn: "a Web Worker" });
        else reject(new Error(`in its worker: ${d.error ?? "no result"}`));
      };
      worker.onerror = (e) => reject(new Error(`its worker failed: ${e.message || "the worker could not be loaded"}`));
      worker.onmessageerror = () => reject(new Error("its worker's result could not be read back"));
      worker.postMessage({ weights: w, row });
    });
  } finally {
    worker.terminate();
  }
}
