/**
 * The satellite QCNN's tree network under CKKS, off the page's thread (Round 86).
 *
 * The same module as on the calling thread (`clientQcnnTreeCkks.ts`), with its own node-seal context. The network and
 * the rows arrive by structured clone and the run returns the same way; no SharedArrayBuffer, so no cross-origin
 * isolation is needed. node-seal ships one single-threaded wasm build, so this is one worker beside the page.
 */

import { treeReadoutCkks, type TreeCkksRun, type TreeNetwork } from "./clientQcnnTreeCkks";

export interface TreeCkksWorkerRequest { id: number; net: TreeNetwork; rows: number[][] }

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<TreeCkksWorkerRequest>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  const { id, net, rows } = e.data;
  try {
    const run: TreeCkksRun = await treeReadoutCkks(net, rows);
    scope.postMessage({ id, ok: true, run });
  } catch (err) {
    scope.postMessage({ id, ok: false, error: (err as Error)?.message ?? String(err) });
  }
};
