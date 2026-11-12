/** The compact GNN's CKKS and garbled-circuit run, off the page's thread (Round 86). Its garbled batches go to a pool of further workers when the machine has the cores for them. */
import { openGcPool } from "./clientGcPool";
import { type GnnWeights, gnnRow, hybridGnnInference } from "./clientHybridGnn";
import { hybridCnnCfg } from "./config";

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<{ weights: GnnWeights; index: number }>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  const pool = openGcPool(hybridCnnCfg().gc_workers);
  try {
    const result = await hybridGnnInference(e.data.weights, gnnRow(e.data.weights, e.data.index), { onProgress: (progress) => scope.postMessage({ progress }), pool });
    scope.postMessage({ ok: true, result });
  } catch (err) {
    scope.postMessage({ ok: false, error: (err as Error)?.message ?? String(err) });
  } finally {
    pool?.close();
  }
};
