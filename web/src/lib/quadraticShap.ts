/**
 * Exact Shapley attributions for the finance square network (F7b, Round 86), kept apart from clientCompute so that
 * tests/test_r86_quadratic_shap_in_node.py runs it against the Python mirror without the pipeline's imports.
 */
// Exact Shapley values for the square network (mirror of qsh/ai/explainability.py `quadratic_shap`; F7b, Round 86):
// the margin logit1 - logit0 = c + a.z + z^T Q z with v = w2[1] - w2[0], Q = sum_k v_k w1_k w1_k^T and
// a = 2 sum_k v_k b1_k w1_k, so phi = (z - base) * (a + Q (z + base)), a pair's joint term split equally, and
// sum(phi) = margin(z) - margin(base) exactly. Same output as linearShap, so the panel renders either.
export function quadraticShap(
  w1: number[][], b1: number[], w2: number[][], b2: number[], z: ArrayLike<number>, names: string[], topK = 5,
  baseline?: ArrayLike<number>,
) {
  const n = w1[0].length, width = w1.length;
  const base = baseline ? Array.from(baseline) : new Array(n).fill(0);
  const v = w2[1].map((x, k) => x - w2[0][k]);
  const margin = (x: ArrayLike<number>) => {
    let m = b2[1] - b2[0];
    for (let k = 0; k < width; k++) {
      let h = b1[k];
      for (let j = 0; j < n; j++) h += w1[k][j] * x[j];
      m += v[k] * h * h;
    }
    return m;
  };
  // a + Q (z + base), with Q applied as sum_k v_k w1_k (w1_k . (z + base)): no n x n matrix is formed.
  const g = new Float64Array(n);
  for (let k = 0; k < width; k++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += w1[k][j] * (z[j] + base[j]);
    const c = v[k] * (s + 2 * b1[k]);
    for (let j = 0; j < n; j++) g[j] += c * w1[k][j];
  }
  const phi = Array.from({ length: n }, (_, i) => (z[i] - base[i]) * g[i]);
  const m = margin(z), baseValue = margin(base);
  const attrs = phi.map((c, i) => ({
    feature: names[i] ?? `feature_${i}`, value: Math.round(Number(z[i]) * 100) / 100, contribution: Math.round(c * 1e6) / 1e6,
  }));
  const top = [...attrs].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, topK);
  return {
    method: "exact-quadratic-shap", prediction: m >= 0 ? 1 : 0,
    base_value: Math.round(baseValue * 1e6) / 1e6, margin: Math.round(m * 1e6) / 1e6, top_features: top,
    efficiency_gap: Math.abs(phi.reduce((a, c) => a + c, 0) + baseValue - m),
  };
}

