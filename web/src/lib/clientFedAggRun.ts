/** The main thread's handle on clientFedAgg.worker.ts: it imports no compute, only types (Round 86). */
import type { AggregationComparison, AggregationKind, ThresholdSum } from "./clientFedAgg";

function inWorker<T>(message: Record<string, unknown>, onProgress?: (p: { kind: AggregationKind; done: number; of: number }) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    // Vite bundles the worker from exactly this form: `new URL(...)` written inside `new Worker(...)`.
    const worker = new Worker(new URL("./clientFedAgg.worker.ts", import.meta.url), { type: "module" });
    const base = new URL(`${import.meta.env.BASE_URL}wasm/qsh_fedthresh/`, location.href).href;
    worker.onmessage = (e: MessageEvent) => {
      const d = e.data;
      if (d.progress) { onProgress?.(d.progress); return; }
      worker.terminate();
      if (d.error) reject(new Error(d.error)); else resolve(d.result as T);
    };
    worker.onerror = (e) => { worker.terminate(); reject(new Error(e.message || "the Lattigo worker failed")); };
    worker.postMessage({ ...message, base });
  });
}

export function compareAggregationsInWorker(
  seed: number, onProgress?: (kind: AggregationKind, done: number, of: number) => void,
): Promise<AggregationComparison> {
  const dataUrl = new URL(`${import.meta.env.BASE_URL}data/medical/heart_disease_sites.json`, location.href).href;
  return inWorker<AggregationComparison>({ task: "compare", seed, dataUrl },
    onProgress && ((p) => onProgress(p.kind, p.done, p.of)));
}

/** The genomics page's noisy contributions, summed under threshold BGV; `crsLabel` is public. */
export function sumUnderThresholdBgvInWorker(values: number[], crsLabel: string): Promise<ThresholdSum> {
  return inWorker<ThresholdSum>({ task: "threshold-sum", values, crsLabel });
}
