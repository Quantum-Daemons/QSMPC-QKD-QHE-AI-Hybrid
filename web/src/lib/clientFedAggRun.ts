/** The main thread's handle on clientFedAgg.worker.ts: it imports no compute, only types (Round 86). */
import type { AggregationComparison, AggregationKind } from "./clientFedAgg";

export function compareAggregationsInWorker(
  seed: number, onProgress?: (kind: AggregationKind, done: number, of: number) => void,
): Promise<AggregationComparison> {
  return new Promise((resolve, reject) => {
    // Vite bundles the worker from exactly this form: `new URL(...)` written inside `new Worker(...)`.
    const worker = new Worker(new URL("./clientFedAgg.worker.ts", import.meta.url), { type: "module" });
    const base = new URL(`${import.meta.env.BASE_URL}wasm/qsh_fedthresh/`, location.href).href;
    const dataUrl = new URL(`${import.meta.env.BASE_URL}data/medical/heart_disease_sites.json`, location.href).href;
    worker.onmessage = (e: MessageEvent) => {
      const d = e.data;
      if (d.progress) { onProgress?.(d.progress.kind, d.progress.done, d.progress.of); return; }
      worker.terminate();
      if (d.error) reject(new Error(d.error)); else resolve(d.result as AggregationComparison);
    };
    worker.onerror = (e) => { worker.terminate(); reject(new Error(e.message || "the aggregation worker failed")); };
    worker.postMessage({ seed, base, dataUrl });
  });
}
