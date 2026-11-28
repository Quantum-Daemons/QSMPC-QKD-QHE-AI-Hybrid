/**
 * A held-out corpus under CKKS through a tabular square network of any number of classes, in the shape the linear
 * student's figure takes (Round 86, batch 12): the telecom page's 5G-NIDD path, where the square network replaced the
 * linear student on the owner's choice of 2026-11-28. Finance's two-class network keeps its own function
 * (`clientFinanceSquare.ts`), which adds its decision curve; this is the same agreement for k classes, the decision the
 * first class with the largest logit, each score the winning class's, as the linear figure draws a multi-class model.
 */
import type { EncryptedAgreementPlot } from "../components/VerificationChart";
import type { FHEBatchResult, HeldOutAgreement } from "./clientFHE";
import { argmaxAt, runSquareTabularOffThread, type SquareTabBatch, type SquareTabWeights } from "./clientFHESquareTabular";
import type { HeldOutCorpus } from "./realData";

export async function squareHeldOutAgreement(held: HeldOutCorpus, net: SquareTabWeights, set: string,
                                             onBlock?: (done: number, total: number) => void): Promise<HeldOutAgreement> {
  if (!held.raw?.length) throw new Error("the held-out corpus carries no raw rows for the square network");
  const k = net.b2.length;
  const r: SquareTabBatch = await runSquareTabularOffThread(net, held.raw, set, onBlock);
  const rows = r.rows;
  const at = (a: Float64Array, i: number) => Array.from(a.subarray(k * i, k * i + k));
  const logits: number[][] = [], plainLogits: number[][] = [];
  for (let i = 0; i < rows; i++) { logits.push(at(r.logits, i)); plainLogits.push(at(r.plain, i)); }
  const encCls = logits.map((_, i) => argmaxAt(r.logits, k * i, k)), plainCls = plainLogits.map((_, i) => argmaxAt(r.plain, k * i, k));
  const margins = plainLogits.map((l) => { const s = [...l].sort((a, b) => b - a); return s[0] - s[1]; });
  const flipped = encCls.map((_, i) => i).filter((i) => encCls[i] !== plainCls[i]);
  const tieRows = margins.filter((m) => m === 0).length;
  const decidable = rows - tieRows;
  const flippedDecidable = flipped.filter((i) => margins[i] !== 0).length;
  let flipMarginMax = 0;
  for (const i of flipped) flipMarginMax = Math.max(flipMarginMax, margins[i]);
  let smallestMargin: number | null = null;
  for (const m of margins) if (m > 0 && (smallestMargin == null || m < smallestMargin)) smallestMargin = m;
  // Every class score goes through CKKS (no bias-only class), so two deviations can meet at a flip.
  const devs: number[] = [];
  logits.forEach((l, i) => l.forEach((v, c) => devs.push(Math.abs(v - plainLogits[i][c]))));
  devs.sort((x, y) => x - y);
  const medianAbsError = devs.length
    ? (devs.length % 2 ? devs[(devs.length - 1) / 2] : (devs[devs.length / 2 - 1] + devs[devs.length / 2]) / 2) : null;
  const flipBoundFactor = 2 as const;
  const flipImpossible = smallestMargin != null && smallestMargin > flipBoundFactor * r.maxAbsError;
  const labels = held.labels.slice(0, rows);
  const acc = (cls: number[]) => (labels.length ? cls.reduce((a, v, i) => a + (v === labels[i] ? 1 : 0), 0) / labels.length : null);
  const counts = new Map<number, number>();
  for (const l of labels) counts.set(l, (counts.get(l) ?? 0) + 1);
  const majorityBaseline = labels.length ? Math.max(...counts.values()) / labels.length : null;
  const p = net.ckks[set];
  const perCt = p.N / 2;
  const bitsOf = (e: number | null) => (e != null && e > 0 && e < 1 ? Math.floor(-Math.log2(e)) : null);
  const binary = k === 2;
  const batch: FHEBatchResult = {
    logits, plainLogits, rows, rowsOffered: held.raw.length, rowsPerCiphertext: perCt, ciphertexts: r.blocks * net.n_in,
    fheMs: r.totalMs, msPerCiphertext: null, ctBytes: r.per_block.bytes.input / Math.max(1, net.n_in),
    maxAbsError: r.maxAbsError, rotations: 0, multiplies: 0, rotateMs: 0,
  };
  const plot: EncryptedAgreementPlot = {
    kind: "encrypted-agreement",
    points: logits.map((l, i) => ({
      enc: binary ? l[1] - l[0] : l[encCls[i]], plain: binary ? plainLogits[i][1] - plainLogits[i][0] : plainLogits[i][encCls[i]],
      agree: encCls[i] === plainCls[i], label: labels[i], encCls: encCls[i], plainCls: plainCls[i],
    })),
    rows, rowsOffered: held.raw.length, classes: k, maxAbsError: r.maxAbsError, flipMarginMax,
    undecidableRows: tieRows, decidableRows: decidable,
    agreementOverDecidable: decidable > 0 ? 1 - flippedDecidable / decidable : null,
    heldOut: true, fitNote: held.fitNote, split: held.split, corpus: held.corpus,
    modelKey: held.modelKey, modelLabel: held.modelLabel, licence: held.licence,
    ciphertexts: r.blocks * net.n_in, rowsPerCiphertext: perCt, fheMs: r.totalMs, msPerCiphertext: null,
    workNote: `${r.blocks} block(s) of up to ${perCt.toLocaleString("en-US")} rows, one row a slot: ${net.n_in} input `
      + `ciphertexts and ${k} output ciphertexts a block, in ${r.ranIn}, under ${set} (N = ${p.N.toLocaleString("en-US")})`,
    arithmeticNote: "a plaintext-weight product, a ciphertext square and a plaintext-weight product: depth 3, no rotation, "
      + "no layer decrypted in between",
    medianAbsError, noisyScores: devs.length, smallestMargin, flipBoundFactor, scoreKind: binary ? "logit" : "winning-score",
    classNames: held.classNames, plane: null,
    accuracy: { encrypted: acc(encCls), plaintext: acc(plainCls), majority: majorityBaseline },
  };
  return {
    plot, batch, mismatches: flipped.length, tieRows, decidable, flippedDecidable, flipMarginMax, smallestMargin,
    medianAbsError, precisionBits: bitsOf(r.maxAbsError), medianPrecisionBits: bitsOf(medianAbsError), flipImpossible,
    encryptedAccuracy: acc(encCls), plaintextAccuracy: acc(plainCls), majorityBaseline,
  };
}
