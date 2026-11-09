/** One worker of the lending pool (Round 86): tasks arrive by structured clone, and each result goes back with how long the worker was busy on it. */
import { closeTaskSession, executeTask, type LendingTask, type TaskInit } from "./tasks";

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<{ id?: number; task?: LendingTask; init?: TaskInit; close?: boolean }>) => void) | null;
  postMessage(message: unknown): void;
};

let init: TaskInit = {};

scope.onmessage = async (e) => {
  const { id, task, init: nextInit, close } = e.data;
  if (close) { closeTaskSession(); scope.postMessage({ closed: true }); return; }
  if (nextInit) init = nextInit;
  if (id === undefined || !task) return;
  const t0 = performance.now();
  try {
    const result = await executeTask(task, init);
    scope.postMessage({ id, ok: true, result, busyMs: performance.now() - t0 });
  } catch (err) {
    scope.postMessage({ id, ok: false, error: (err as Error)?.message ?? String(err) });
  }
};
