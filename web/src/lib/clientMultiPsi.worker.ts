/** The n-party PSI off the page's thread (Round 86): the lists arrive by structured clone and the result goes back the same way. */
import { multiPartyPsi } from "./clientMultiPsi";

// The project compiles against the DOM lib, where `self` is a Window; a worker's own scope needs only these two.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<{ lists: string[][] }>) => void) | null;
  postMessage(message: unknown): void;
};

scope.onmessage = async (e) => {
  try {
    scope.postMessage({ ok: true, result: await multiPartyPsi(e.data.lists) });
  } catch (err) {
    scope.postMessage({ ok: false, error: (err as Error)?.message ?? String(err) });
  }
};
