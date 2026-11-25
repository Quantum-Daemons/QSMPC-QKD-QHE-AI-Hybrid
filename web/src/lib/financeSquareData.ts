/**
 * The finance square network's file, key and label (Round 86, F7b), without node-seal, so that `realData.ts` can load
 * it without bundling the SEAL wasm; `clientFinanceSquare.ts` holds the CKKS runs and re-exports these.
 */
import type { PlattMap } from "./decisionCurve";
import type { SquareTabWeights } from "./squareTabularMath";

/** The page-local key the config's curve names (`page_model_keys`); not a registry model. */
export const SQUARE_KEY = "finance_aml.square_network";
/** The one set F7b's criteria offered. */
export const SQUARE_SET = "A2";

export interface FinanceSquare extends SquareTabWeights {
  export_seed: number;
  /** What each of the 114 inputs is, for the SHAP bars (PP14), from the trainer's feature code. */
  feature_definitions: Record<string, string>;
  calibration: PlattMap & { method: string; corp_calibrated?: { positives?: number; n?: number };
                            fitted_after_the_preregistered_run?: boolean };
  record: string;
  figures: {
    test_auprc: number; test_minority_f1: number; test_auprc_ci95: [number, number];
    mean_test_auprc: number; sd_test_auprc: number; linear_test_auprc: number; linear_test_minority_f1: number;
    test_rows: number; test_positive: number;
    browser: Record<string, { rows: number; slowest_total_ms: number; largest_wasm_memory_bytes: number }>;
    browser_name: string; sets_not_offered: Record<string, string>;
  };
}

export const SQUARE_PARAMS = (w: FinanceSquare) =>
  w.w1.length * w.w1[0].length + w.b1.length + w.w2.length * w.w2[0].length + w.b2.length;

let _loading: Promise<FinanceSquare> | null = null;

/** The site's copy of the network, fetched once a load. */
export function loadFinanceSquare(): Promise<FinanceSquare> {
  // `import.meta.env` is Vite's; a node harness that bundles this module without it reads the site's root.
  const base = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? "/";
  return (_loading ??= fetch(`${base}data/finance/square_student_weights.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`the square network's file answered ${r.status}`);
      return r.json() as Promise<FinanceSquare>;
    })
    .catch((e) => { _loading = null; throw e; }));
}

/** What `publishedDecisionCurve` reads of a model: its key and a record with the map and the window it was scored on. */
export function squareSpec(net: FinanceSquare) {
  return { key: SQUARE_KEY, achieved: { calibration: net.calibration, n_test: net.figures.test_rows } };
}

/** The label the figure and the logs use. */
export function squareLabel(net: FinanceSquare): string {
  return `F7b's ${SQUARE_PARAMS(net).toLocaleString("en-US")}-parameter square network (CKKS end to end)`;
}

