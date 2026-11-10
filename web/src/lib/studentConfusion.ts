// The confusion matrix of the published telecom student on its held-out rows (Round 86), in the clear, from the shipped weights and rows.

export interface HeldOut { class_names: string[]; samples: number[][]; labels: number[] }
export interface LinearStudent { W: number[][]; b: number[]; input: { mean: number[]; scale: number[] } }

export function confusion(held: HeldOut, w: LinearStudent): { names: string[]; matrix: number[][]; total: number; correct: number } {
  const names = held.class_names;
  const matrix = names.map(() => names.map(() => 0));
  let correct = 0;
  held.samples.forEach((row, i) => {
    const z = row.map((v, j) => (v - w.input.mean[j]) / w.input.scale[j]);
    const scores = w.W.map((r, c) => r.reduce((a, wt, j) => a + wt * z[j], 0) + w.b[c]);
    const pred = scores.indexOf(Math.max(...scores));
    matrix[held.labels[i]][pred]++;
    if (pred === held.labels[i]) correct++;
  });
  return { names, matrix, total: held.samples.length, correct };
}
