/**
 * The satellite CKKS student's stretch, scored without encryption (Round 86): the parts a test can run under node.
 *
 * The student is a linear next-step forecaster: forecast[c] = b[c] + sum_j W[c][j] x[j] over the last K z-scored rows.
 * This module reproduces, from the published `weights.json` and the published stretch of the test half, what the
 * generator (`scripts/prep_model_inputs.py`, `build_satellite_student`) recorded:
 * - the forecast, summed term by term in the generator's order, so every double is the same;
 * - Telemanom's thresholding of |forecast - actual| in raw units (`telemanomThreshold.ts`, a port checked against
 *   `qsh.models.telemanom_threshold`), whose flags are compared with the recorded digest bit for bit;
 * - ESA-ADB's corrected event-wise F0.5 over the stretch (`esaScores.ts`).
 * It can then run the same dot products under CKKS (`encryptedLinearBatch`, unchanged) and threshold the decrypted
 * forecast, so the page shows what encryption does to the events, not only to the numbers.
 */

import { esaEventWise, type EsaLabel, type EsaScore } from "./esaScores";
import { DEFAULT_CONFIG, flag, type TelemanomConfig } from "./telemanomThreshold";

export interface ForecastEventsFile {
  task: "forecast_events";
  provenance: string;
  split?: string;
  stretch: { start_ms: number; step_ms: number; rows: number; history_rows: number };
  channels: string[];
  rows: number[][];
  labels: { id: string; start_ms: number; end_ms: number; category: string }[];
  weights: string;
  expected: {
    mae: number; persistence_mae: number; median_ae: number; persistence_median_ae: number;
    flagged: number; flagged_per_channel: number[]; flags_sha256: string;
    anomalies: { f_0_5: number; tp: number; fp: number; fn: number; n_events: number };
    all_events: { f_0_5: number; tp: number; fp: number; fn: number; n_events: number };
    note: string;
  };
}

export interface ForecasterWeights {
  kind: "linear-forecaster";
  k_rows: number;
  W: number[][];
  b: number[];
  input: { mean: number[]; std: number[] };
  thresholding: Record<string, number | boolean>;
}

export interface StretchScore {
  flags: Uint8Array;
  flagged: number;
  flaggedPerChannel: number[];
  anomalies: EsaScore;
  allEvents: EsaScore;
}

/** The feature row for target row t (0-based within the stretch): the K rows before it, oldest first, row-major. */
export function featureRows(file: ForecastEventsFile): number[][] {
  const k = file.stretch.history_rows;
  return Array.from({ length: file.stretch.rows }, (_, t) => file.rows.slice(t, t + k).flat());
}

/** The student's forecast, summed term by term in the generator's order (b first, then W[c][j] x[j] for j = 0..). */
export function plaintextForecast(w: ForecasterWeights, x: number[][]): number[][] {
  return x.map((xi) => w.W.map((wc, c) => {
    let s = w.b[c];
    for (let j = 0; j < xi.length; j++) s = s + wc[j] * xi[j];
    return s;
  }));
}

function config(w: ForecasterWeights): TelemanomConfig {
  const t = w.thresholding;
  return {
    ...DEFAULT_CONFIG,
    batchSize: Number(t.batch_size ?? DEFAULT_CONFIG.batchSize),
    smoothingWindowSize: Number(t.smoothing_window_size ?? DEFAULT_CONFIG.smoothingWindowSize),
    smoothingPerc: Number(t.smoothing_perc ?? DEFAULT_CONFIG.smoothingPerc),
    errorBuffer: Number(t.error_buffer ?? DEFAULT_CONFIG.errorBuffer),
    p: Number(t.p ?? DEFAULT_CONFIG.p),
    windowSize: Number(t.window_size ?? DEFAULT_CONFIG.windowSize),
    sdLim: Number(t.sd_lim ?? DEFAULT_CONFIG.sdLim),
    minErrorValue: Number(t.min_error_value ?? DEFAULT_CONFIG.minErrorValue),
    adjustStart: Boolean(t.adjust_start ?? DEFAULT_CONFIG.adjustStart),
  };
}

/** Threshold a forecast over the stretch and score it, as the generator did. */
export function scoreStretch(file: ForecastEventsFile, w: ForecasterWeights, forecast: number[][]): StretchScore {
  const k = file.stretch.history_rows;
  const n = file.stretch.rows;
  const nc = w.b.length;
  const errors: Float64Array[] = [], values: Float64Array[] = [];
  for (let c = 0; c < nc; c++) {
    const e = new Float64Array(n), v = new Float64Array(n);
    for (let t = 0; t < n; t++) {
      const actual = file.rows[t + k][c];
      e[t] = Math.abs(forecast[t][c] - actual) * w.input.std[c];
      v[t] = actual * w.input.std[c] + w.input.mean[c];
    }
    errors.push(e);
    values.push(v);
  }
  const { perChannel, any } = flag(errors, values, config(w));
  const times = Array.from({ length: n }, (_, t) => file.stretch.start_ms + t * file.stretch.step_ms);
  const labels: EsaLabel[] = file.labels.map((l) => ({ id: l.id, startMs: l.start_ms, endMs: l.end_ms,
                                                       category: l.category }));
  const full: [number, number] = [times[0], times[n - 1]];
  return {
    flags: any,
    flagged: any.reduce((s, v) => s + v, 0),
    flaggedPerChannel: perChannel.map((f) => f.reduce((s, v) => s + v, 0)),
    anomalies: esaEventWise(times, any, labels, ["Anomaly"], full),
    allEvents: esaEventWise(times, any, labels, ["Rare Event", "Anomaly"], full),
  };
}

/** sha256 of the flags packed as numpy's `packbits` packs them: eight per byte, the first in the high bit. */
export async function flagsDigest(flags: Uint8Array): Promise<string> {
  const bytes = new Uint8Array(Math.ceil(flags.length / 8));
  for (let i = 0; i < flags.length; i++) if (flags[i]) bytes[i >> 3] |= 0x80 >> (i & 7);
  const d = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(d), (x) => x.toString(16).padStart(2, "0")).join("");
}
