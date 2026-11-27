// Each page's published fp32 graphs on WebGPU beside WASM (Round 86, batch 8; the owner's choice of 2026-11-27,
// checklist CS23). This is the pure half: which graph a comparison runs, why a model cannot be compared, and how two
// runs of one graph on the same rows are compared. `compareBackends` in onnxInference.ts opens the two sessions.
//
// fp32 only, by the owner's choice. The page loads a model's int8 graph where one ships (`browserModels`), and ONNX
// Runtime Web's WebGPU provider registers no kernel for an int8 graph's quantised nodes (ConvInteger,
// DynamicQuantizeLinear, MatMulInteger), so those nodes would run on the CPU and the timing would describe neither
// backend. The int8 graphs stay on WASM; their fp32 twins, pinned at the same revision, are what WebGPU is given.

/** Every value within this of the WASM run's, and no decision changed beyond a near tie, is "within tolerance". Stated
 *  before any figure was measured (2026-11-27): an fp32 graph on two runtimes differs by its order of accumulation, and
 *  the samples' own per-row records allow 1e-3 on a margin for the same reason. */
export const BACKEND_TOLERANCE = 1e-3;

/** The artefact fields this reads (the registry's `artifacts`). */
export interface OnnxArtifactLike { fmt: string; path: string; precision?: string; serves?: string }

/** The graph the page itself loads, by `browserModels`' rule: the int8 graph where one ships, else the fp32 one. */
export function pageGraphOf(artifacts: readonly OnnxArtifactLike[]): OnnxArtifactLike | null {
  const onnx = artifacts.filter((a) => a.fmt === "onnx" && (a.serves === "browser" || a.serves === "both"));
  return onnx.find((a) => a.precision === "int8") ?? onnx.find((a) => a.precision === "fp32") ?? onnx[0] ?? null;
}

/** The fp32 graph a comparison runs: the page's own graph when it is fp32, else its fp32 twin (the same path without
 *  `_int8`). Null when there is no such graph, so nothing else is substituted for it. */
export function fp32GraphFor(artifacts: readonly OnnxArtifactLike[]): string | null {
  const page = pageGraphOf(artifacts);
  if (!page) return null;
  if (page.precision === "fp32") return page.path;
  const twin = page.path.replace(/_int8(\.onnx)$/, "$1");
  return artifacts.find((a) => a.fmt === "onnx" && a.precision === "fp32" && a.path === twin)?.path ?? null;
}

/** Why a model cannot be compared on WebGPU, or null when it can. */
export function webgpuInapplicable(domains: readonly string[] | undefined, fp32: string | null): string | null {
  if (!fp32) return "it ships no fp32 graph the browser can load";
  if (domains?.includes("ai.onnx.ml")) {
    return "its graph uses ai.onnx.ml operators (tree ensembles, linear classifiers), which ONNX Runtime Web's WebGPU "
      + "build does not register, so it runs on WASM only";
  }
  return null;
}

/** One row's decision and its distance from the boundary, by the sample's own rule (as `evaluatePublishedModel`
 *  decides): a threshold on p(class 1) for a two-class score with one, else the first maximum, else 0.5. */
export interface Decision { cls: number; margin: number }

/** The decision rule for a score output of `n` rows of `width` values. `prob` is the output as probabilities when its
 *  name says what it is (calibration.ts), else null; a threshold needs it. */
export function decider(data: ArrayLike<number>, width: number, threshold: number | null,
                        prob: ArrayLike<number> | null): (i: number) => Decision {
  if (threshold != null && width === 2 && !prob) {
    throw new Error("a decision threshold needs probabilities, and this output does not say it holds them");
  }
  return (i: number): Decision => {
    if (threshold != null && width === 2) {
      const m = prob![i * 2 + 1] - threshold;
      return { cls: m >= 0 ? 1 : 0, margin: m };
    }
    if (width > 1) {
      let best = 0;
      for (let c = 1; c < width; c++) if (data[i * width + c] > data[i * width + best]) best = c;
      let runner = -Infinity;
      for (let c = 0; c < width; c++) if (c !== best) runner = Math.max(runner, data[i * width + c]);
      return { cls: best, margin: data[i * width + best] - runner };
    }
    const v = (prob ? prob[i] : data[i]) - (threshold ?? 0.5);
    return { cls: v > 0 ? 1 : 0, margin: v };
  };
}

export interface OutputComparison {
  /** Values compared: rows times each row's width. */
  values: number;
  /** The largest |WebGPU - WASM| over every value. */
  maxAbsDiff: number;
  /** The largest |WASM value|, the scale the difference is read against. */
  maxAbs: number;
  /** Rows decided; zero for a forecast, whose values are compared and which has no decision. */
  decisions: number;
  /** Rows decided differently while the WASM run's margin is beyond the tolerance: a failure. */
  decisionsDiffer: number;
  /** Rows decided differently within the tolerance of the boundary: two runtimes may break a tie differently. */
  nearTies: number;
  tolerance: number;
  withinTolerance: boolean;
}

/** Compare two runs of one graph on the same `rows`: every value, and each row's decision when `decide` is given. */
export function compareOutputs(wasm: ArrayLike<number>, other: ArrayLike<number>, rows: number,
                               decide: { wasm: (i: number) => Decision; other: (i: number) => Decision } | null,
                               tolerance = BACKEND_TOLERANCE): OutputComparison {
  if (wasm.length !== other.length) {
    throw new Error(`the two runs returned ${wasm.length} and ${other.length} values for the same rows`);
  }
  let maxAbsDiff = 0, maxAbs = 0;
  for (let k = 0; k < wasm.length; k++) {
    const a = wasm[k], b = other[k];
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      throw new Error(`value ${k} is not finite (${a} on WASM, ${b} on the other backend)`);
    }
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(b - a));
    maxAbs = Math.max(maxAbs, Math.abs(a));
  }
  let decisionsDiffer = 0, nearTies = 0;
  if (decide) {
    for (let i = 0; i < rows; i++) {
      const a = decide.wasm(i), b = decide.other(i);
      if (a.cls !== b.cls) {
        if (Math.abs(a.margin) <= tolerance) nearTies++;
        else decisionsDiffer++;
      }
    }
  }
  return {
    values: wasm.length, maxAbsDiff, maxAbs, decisions: decide ? rows : 0, decisionsDiffer, nearTies, tolerance,
    withinTolerance: maxAbsDiff <= tolerance && decisionsDiffer === 0,
  };
}

/** The median of a non-empty list; NaN for an empty one. */
export function median(xs: readonly number[]): number {
  if (xs.length === 0) return NaN;
  const v = [...xs].sort((p, q) => p - q), m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/** One backend's times for one graph, in milliseconds. */
export interface BackendTimes {
  /** Opening the session: compiling the graph for the backend. */
  createMs: number;
  /** The first run over every row, which includes the backend's warm-up (shader compilation on WebGPU). */
  firstMs: number;
  /** The median of the later runs over every row. */
  batchMs: number;
  /** The median of runs on the first row alone, a visitor's single-row case; null for a graph sample, which is one
   *  graph whose rows cannot be fed alone. */
  rowMs: number | null;
}

/** Timed repeats after the first run: three over every row, five on one row. */
export const BATCH_REPEATS = 3;
export const ROW_REPEATS = 5;
