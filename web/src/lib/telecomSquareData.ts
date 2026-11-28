/**
 * The telecom page's square network (Round 86, batch 12), without node-seal, so that `realData.ts` can load it without
 * bundling the SEAL wasm. On the owner's choice of 2026-11-28, after its pre-registered run met every criterion under
 * both parameter sets (references/telecom-nidd-square-student-preregistration.md,
 * docs/measurements/telecom-nidd-square-student-2026-11-28.json), it replaces the linear student on the page's encrypted
 * 5G-NIDD path. The file (`data/telecom_nidd/square_network.json`, from scripts/export_telecom_square_network.py)
 * carries the exported seed's weights, the sets the criteria offered, the record's figures and the run's 2,048 test
 * flows, raw, which the client scales and clips before it encrypts them, as the network's client does.
 */
import type { SquareTabWeights } from "./squareTabularMath";

/** The page-local key the run's log and metrics name; not a registry model (the Hub waits for the owner). */
export const TELECOM_SQUARE_KEY = "telecom_ids.square_network";

export interface TelecomSquare extends SquareTabWeights {
  key: string;
  export_seed: number;
  inputs: string[];
  /** The first of the sets the criteria offered, which the page runs. */
  default_set: string;
  record: string;
  figures: {
    control_test_flamby: number; test_flamby_mean: number; test_flamby_min: number;
    export_seed_test: { flamby_metric: number; macro_f1: number; accuracy: number };
    export_bootstrap: [number, number];
    ckks: Record<string, { max_abs_error: number; decisions_differ: number; encrypted_accuracy: number }>;
    browser: Record<string, { slowest_total_ms: number; largest_wasm_memory_bytes: number }>;
    browser_name: string; test_part_rows: number;
  };
  corpus: string; split: string; fitNote: string;
  licence: string; licence_url: string; attribution: string; modifications: string;
  held_out: { raw: number[][]; labels: number[]; feature_names: string[]; class_counts: Record<string, number> };
}

export const TELECOM_SQUARE_PARAMS = (w: TelecomSquare) =>
  w.w1.length * w.w1[0].length + w.b1.length + w.w2.length * w.w2[0].length + w.b2.length;

let _loading: Promise<TelecomSquare> | null = null;

/** The site's copy of the network and its flows, fetched once a load. */
export function loadTelecomSquare(): Promise<TelecomSquare> {
  const base = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? "/";
  return (_loading ??= fetch(`${base}data/telecom_nidd/square_network.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`the telecom square network's file answered ${r.status}`);
      return r.json() as Promise<TelecomSquare>;
    })
    .catch((e) => { _loading = null; throw e; }));
}

/** The label the figure and the logs use. */
export function telecomSquareLabel(net: TelecomSquare): string {
  return `the ${TELECOM_SQUARE_PARAMS(net).toLocaleString("en-US")}-parameter square network of the 5G-NIDD detector `
    + "(CKKS end to end, site file, pre-registered)";
}
