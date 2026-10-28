// The live run's state, published by the page that runs the pipeline and read by the sidebar.
//
// Round 80. Both run pages drew this status as a chip fixed to the bottom-right corner, because the
// controls card scrolls away and "only the sidebar was sticky". During a run the chip sat on top of
// the two-party diagram's lower rows, its legend, and the Metrics table - measured on the live site
// at 1280 x 677. The sidebar is sticky and holds no page content, so the status moved there: always
// visible, never over anything. A module-level store rather than a context, because the publisher
// (a page under <Routes>) and the reader (the sidebar beside it) share no parent below <App>.
import { useSyncExternalStore } from "react";

export type RunStatus = { state: string; active: string | null } | null;

let current: RunStatus = null;
const listeners = new Set<() => void>();

/** Publish the run's state, or `null` when there is nothing to report (idle, or a finished run). */
export function setRunStatus(next: RunStatus): void {
  if (next === current || (next && current && next.state === current.state && next.active === current.active)) return;
  current = next;
  for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => { listeners.delete(l); };
}

export function useRunStatus(): RunStatus {
  return useSyncExternalStore(subscribe, () => current, () => null);
}
