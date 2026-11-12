/** A small pool for garbled batches (Round 86), on the pattern of `lending/pool.ts`: at most half the logical cores, and only where a worker can be started at all. */
import type { GcPool } from "./clientHybridCnn";
import type { ReluBatchResult } from "./clientMPC";

export interface OpenPool extends GcPool { close(): void }

export function gcPoolSize(cap: number, logicalCores: number): number {
  return Math.max(1, Math.min(cap, Math.floor(logicalCores / 2)));
}

/** A pool of up to `cap` workers, or null when there is no point: no `Worker` here, or only one core's worth to spare. */
export function openGcPool(cap: number): OpenPool | null {
  if (typeof Worker === "undefined" || typeof navigator === "undefined") return null;
  const n = gcPoolSize(cap, navigator.hardwareConcurrency || 1);
  if (n < 2) return null;
  const workers = Array.from({ length: n }, () => new Worker(new URL("./clientGc.worker.ts", import.meta.url), { type: "module" }));
  const idle = [...workers];
  const waiting: (() => void)[] = [];
  const results = new Map<number, { resolve: (r: ReluBatchResult) => void; reject: (e: Error) => void }>();
  let next = 0, failed: Error | null = null;
  const fail = (e: Error) => { failed = e; for (const p of results.values()) p.reject(e); results.clear(); };
  for (const w of workers) {
    w.onmessage = (e: MessageEvent<{ id: number; ok: boolean; result?: ReluBatchResult; error?: string }>) => {
      const p = results.get(e.data.id);
      results.delete(e.data.id);
      if (e.data.ok && e.data.result) p?.resolve(e.data.result);
      else p?.reject(new Error(`in a garbling worker: ${e.data.error ?? "no result"}`));
      idle.push(w);
      waiting.shift()?.();
    };
    w.onerror = (e) => fail(new Error(`a garbling worker failed: ${e.message || "it could not be loaded"}`));
    w.onmessageerror = () => fail(new Error("a garbling worker's result could not be read back"));
  }
  return {
    workers: n,
    close() { workers.forEach((w) => w.terminate()); },
    async run(y, r, s, ringBits, outBits) {
      if (failed) throw failed;
      if (!idle.length) await new Promise<void>((go) => waiting.push(go));
      if (failed) throw failed;
      const w = idle.pop() as Worker, id = next++;
      return new Promise<ReluBatchResult>((resolve, reject) => {
        results.set(id, { resolve, reject });
        w.postMessage({ id, y, r, s, ringBits, outBits });
      });
    },
  };
}
