/**
 * Lattigo off the main thread (Round 86): the module is about 3 MB compressed and the medical comparison takes seconds,
 * so the medical and genomics pages run it here, and only when a button is pressed. Two tasks: the medical page's
 * aggregation comparison, and one sum under threshold BGV for the genomics page.
 */
import { loadFedThresh, runAggregationComparison, sumUnderThresholdBgv } from "./clientFedAgg";
import type { HeartSites } from "./clientFedHeart";

type Task = { task?: "compare"; seed: number; base: string; dataUrl: string }
          | { task: "threshold-sum"; base: string; values: number[]; crsLabel: string };

self.onmessage = async (e: MessageEvent<Task>) => {
  try {
    const d = e.data;
    const api = await loadFedThresh(d.base);
    if (d.task === "threshold-sum") {
      self.postMessage({ result: sumUnderThresholdBgv(api, d.values, new TextEncoder().encode(d.crsLabel)) });
      return;
    }
    const res = await fetch(d.dataUrl);
    if (!res.ok) throw new Error(`${d.dataUrl}: HTTP ${res.status}`);
    const sites = (await res.json()) as HeartSites;
    const result = await runAggregationComparison(sites, d.seed, api,
      (kind, done, of) => self.postMessage({ progress: { kind, done, of } }));
    self.postMessage({ result });
  } catch (err) {
    self.postMessage({ error: err instanceof Error ? err.message : String(err) });
  }
};
