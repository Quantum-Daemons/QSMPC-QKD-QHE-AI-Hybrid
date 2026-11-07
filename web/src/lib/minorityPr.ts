/**
 * The precision-recall curve of a binary sample's class 0, when class 0 is the minority (Round 86, fact-check C7).
 *
 * The verification's curve scores class 1. On BreastMNIST class 1 is "normal, benign", the majority (73%), so that
 * curve sits on a 0.73 baseline and flatters the model, while the class a PR curve is for, the rare one, is malignant.
 * This ranks the same score in reverse and counts class 0, and gives how many class-0 rows the page's own decision
 * rule detects. Pure, so tests/test_r86_minority_pr.py can hold it to a Python reference.
 */
export interface MinorityPr {
  pr: Array<{ recall: number; precision: number }>;
  prevalence: number;
  detected: number;
  total: number;
}

/** `positiveScore(i)` ranks class 1 high; `preds` is the page's decision per row. null when class 0 is not the minority. */
export function minorityPr(n: number, labels: ArrayLike<number>, preds: ArrayLike<number>,
                           positiveScore: (i: number) => number, stride: number): MinorityPr | null {
  let pos0 = 0;
  for (let i = 0; i < n; i++) if (labels[i] === 0) pos0++;
  if (!(pos0 > 0 && pos0 < n / 2)) return null;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => positiveScore(a) - positiveScore(b));
  const pr: Array<{ recall: number; precision: number }> = [];
  let tp = 0;
  for (let k = 0; k < n; k++) {
    if (labels[order[k]] === 0) tp++;
    if (k % stride === 0 || k === n - 1) pr.push({ recall: tp / pos0, precision: tp / (k + 1) });
  }
  let detected = 0;
  for (let i = 0; i < n; i++) if (labels[i] === 0 && preds[i] === 0) detected++;
  return { pr, prevalence: pos0 / n, detected, total: pos0 };
}
