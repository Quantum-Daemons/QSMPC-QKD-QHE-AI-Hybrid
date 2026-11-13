/**
 * The encrypted-aggregation comparison off the main thread (Round 86): the Lattigo module is about 3 MB compressed and
 * the five federations take seconds, so the medical page runs them here, and only when its button is pressed.
 */
import { loadFedThresh, runAggregationComparison } from "./clientFedAgg";
import type { HeartSites } from "./clientFedHeart";

self.onmessage = async (e: MessageEvent<{ seed: number; base: string; dataUrl: string }>) => {
  try {
    const { seed, base, dataUrl } = e.data;
    const res = await fetch(dataUrl);
    if (!res.ok) throw new Error(`${dataUrl}: HTTP ${res.status}`);
    const sites = (await res.json()) as HeartSites;
    const api = await loadFedThresh(base);
    const result = await runAggregationComparison(sites, seed, api,
      (kind, done, of) => self.postMessage({ progress: { kind, done, of } }));
    self.postMessage({ result });
  } catch (err) {
    self.postMessage({ error: err instanceof Error ? err.message : String(err) });
  }
};
