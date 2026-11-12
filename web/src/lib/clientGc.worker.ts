/** One garbled batch off the thread that orchestrates the run (Round 86): each worker holds its own wasm instance, and a batch depends on nothing but its inputs. */
import { secureReluBatch } from "./clientMPC";

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<{ id: number; y: Uint32Array; r: Uint32Array; s: BigUint64Array; ringBits: number; outBits: number }>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  const { id, y, r, s, ringBits, outBits } = e.data;
  try {
    scope.postMessage({ id, ok: true, result: await secureReluBatch(y, r, s, ringBits, outBits) });
  } catch (err) {
    scope.postMessage({ id, ok: false, error: (err as Error)?.message ?? String(err) });
  }
};
