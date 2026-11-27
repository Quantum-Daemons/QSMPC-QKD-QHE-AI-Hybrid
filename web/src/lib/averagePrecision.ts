// Average precision in the tab, for re-scores that must reproduce a figure computed offline by scikit-learn (Round 86,
// batch 8: the DeFi QCNN's sample). tests/test_r86_defi_qcnn.py holds it to `average_precision_score` on scores with ties.

/** Average precision as scikit-learn's `average_precision_score` computes it: tied scores form one threshold, and the
 *  precision at each threshold is weighted by the recall it adds. */
export function averagePrecision(scores: ArrayLike<number>, labels: ArrayLike<number>): number {
  const order = Array.from({ length: scores.length }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const positives = Array.from(labels).reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
  if (positives === 0) return NaN;
  let tp = 0, seen = 0, prevRecall = 0, ap = 0;
  for (let k = 0; k < order.length; k++) {
    const i = order[k];
    seen++;
    if (labels[i] === 1) tp++;
    if (k + 1 < order.length && scores[order[k + 1]] === scores[i]) continue;   // the rest of this tie first
    const recall = tp / positives;
    ap += (recall - prevRecall) * (tp / seen);
    prevRecall = recall;
  }
  return ap;
}
