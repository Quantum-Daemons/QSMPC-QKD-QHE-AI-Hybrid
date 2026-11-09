/**
 * A small pool of Web Workers for the lending page's tasks (Round 86). SEAL's wasm build is single-threaded and each garbled circuit runs on one thread, so one check
 * cannot use more than a core. The checks are independent, however, so several workers, each with its own wasm instances, can run them at once.
 * A failed worker is reported and the stage says so; nothing is retried on the page's thread.
 */

import { closeTaskSession, executeTask, type LendingTask, type LendingTaskResult, type TaskInit } from "./tasks";

export interface PoolRun {
  results: LendingTaskResult[];
  workers: number;
  ranIn: string;
  logicalCores: number;
  /** Per-task time the workers spent, in task order, including each worker's first-use set-up. */
  busyMs: number[];
  wallMs: number;
  /** Busy time over wall-clock: how many cores' worth of work the pool did at once. */
  speedUp: number;
  /** Speed-up per worker, from 0 to 1. */
  utilisation: number;
}

/** At most `cap`, never more than the tasks and never more than half the logical cores, so the page itself stays responsive. */
export function poolSize(cap: number, tasks: number, logicalCores: number): number {
  return Math.max(1, Math.min(cap, tasks, Math.floor(logicalCores / 2)));
}

export async function runLendingTasks(tasks: LendingTask[], init: TaskInit, cap: number, onDone?: (done: number, total: number) => void): Promise<PoolRun> {
  const logicalCores = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 1 : 1;
  const results = new Array<LendingTaskResult>(tasks.length);
  const busyMs = new Array<number>(tasks.length).fill(0);
  const t0 = performance.now();
  const finish = (workers: number, ranIn: string): PoolRun => {
    const wallMs = performance.now() - t0;
    const busy = busyMs.reduce((a, b) => a + b, 0);
    return { results, workers, ranIn, logicalCores, busyMs, wallMs, speedUp: busy / wallMs, utilisation: busy / wallMs / workers };
  };

  if (typeof window === "undefined" || typeof Worker === "undefined") {
    for (let i = 0; i < tasks.length; i++) {
      const start = performance.now();
      results[i] = await executeTask(tasks[i], init);
      busyMs[i] = performance.now() - start;
      onDone?.(i + 1, tasks.length);
    }
    closeTaskSession();
    return finish(1, "the calling thread");
  }

  const n = poolSize(cap, tasks.length, logicalCores);
  const workers = Array.from({ length: n }, () => new Worker(new URL("./lending.worker.ts", import.meta.url), { type: "module" }));
  try {
    await new Promise<void>((resolve, reject) => {
      let next = 0, done = 0;
      const give = (w: Worker) => {
        if (next >= tasks.length) return;
        const id = next++;
        w.postMessage({ id, task: tasks[id] });
      };
      for (const w of workers) {
        w.onmessage = (e: MessageEvent<{ id: number; ok?: boolean; result?: LendingTaskResult; busyMs?: number; error?: string }>) => {
          const m = e.data;
          if (!m.ok || !m.result) { reject(new Error(`in its worker: ${m.error ?? "no result"}`)); return; }
          results[m.id] = m.result;
          busyMs[m.id] = m.busyMs ?? 0;
          // Counted apart from the callback: `onDone?.(++done, ...)` skips its arguments when there is no callback, and the pool then never finishes.
          done++;
          onDone?.(done, tasks.length);
          if (done === tasks.length) resolve(); else give(w);
        };
        w.onerror = (e) => reject(new Error(`a worker failed: ${e.message || "it could not be loaded"}`));
        w.onmessageerror = () => reject(new Error("a worker's result could not be read back"));
        w.postMessage({ init });
        give(w);
      }
    });
  } finally {
    workers.forEach((w) => w.terminate());
  }
  return finish(n, n === 1 ? "one Web Worker" : `${n} Web Workers`);
}
