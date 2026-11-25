/**
 * The finance page's fully encrypted model since F7b (Round 86): the square network in place of the linear student, on
 * the owner's choice of 2026-11-25 after F7b's pre-registered run adopted it under parameter set A2
 * (references/finance-square-student-page-preregistration.md, docs/measurements/finance-square-student-page-2026-11-25.json).
 *
 * The network ships with the site (`data/finance/square_student_weights.json`, from
 * `scripts/train/finance_square_student_page.py export`), not from the Hub: the owner's choice of the same day. CKKS
 * evaluates it end to end, one row a slot, in Web Workers (`runSquareTabularOffThread`); B is not offered, because its
 * block took 35.6 s in the browser against the pre-registered 30 s. Its Platt map was fitted on the validation window
 * after the pre-registered run, for this page's decision curve and probability, and the file says so.
 */
import { runSquareTabularOffThread, type SquareTabBatch } from "./clientFHESquareTabular";
import { type FinanceSquare, SQUARE_KEY, SQUARE_SET, squareLabel, squareSpec } from "./financeSquareData";
export { type FinanceSquare, loadFinanceSquare, SQUARE_KEY, SQUARE_PARAMS, SQUARE_SET, squareLabel, squareSpec } from "./financeSquareData";
import type { EncryptedAgreementPlot } from "../components/VerificationChart";
import type { FHEBatchResult, HeldOutAgreement } from "./clientFHE";
import { publishedDecisionCurve } from "./decisionCurve";
import type { HeldOutCorpus } from "./realData";

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/** The held-out rows under CKKS through the square network, in the shape the linear student's figure takes. */
export async function encryptedHeldOutAgreementSquare(held: HeldOutCorpus, net: FinanceSquare,
                                                      onBlock?: (done: number, total: number) => void): Promise<HeldOutAgreement> {
  if (!held.raw?.length) throw new Error("the held-out corpus carries no raw rows for the square network");
  const r: SquareTabBatch = await runSquareTabularOffThread(net, held.raw, SQUARE_SET, onBlock);
  const rows = r.rows;
  const logits: number[][] = [], plainLogits: number[][] = [];
  for (let i = 0; i < rows; i++) {
    logits.push([r.logits[2 * i], r.logits[2 * i + 1]]);
    plainLogits.push([r.plain[2 * i], r.plain[2 * i + 1]]);
  }
  const encCls = logits.map((l) => (l[1] > l[0] ? 1 : 0)), plainCls = plainLogits.map((l) => (l[1] > l[0] ? 1 : 0));
  const margins = plainLogits.map((l) => Math.abs(l[1] - l[0]));
  const flipped = encCls.map((_, i) => i).filter((i) => encCls[i] !== plainCls[i]);
  const tieRows = margins.filter((m) => m === 0).length;
  const decidable = rows - tieRows;
  const flippedDecidable = flipped.filter((i) => margins[i] !== 0).length;
  let flipMarginMax = 0;
  for (const i of flipped) flipMarginMax = Math.max(flipMarginMax, margins[i]);
  let smallestMargin: number | null = null;
  for (const m of margins) if (m > 0 && (smallestMargin == null || m < smallestMargin)) smallestMargin = m;
  // Both logits go through CKKS here (no bias-only class), so every score is a deviation and two can meet at a flip.
  const devs: number[] = [];
  logits.forEach((l, i) => { devs.push(Math.abs(l[0] - plainLogits[i][0]), Math.abs(l[1] - plainLogits[i][1])); });
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
  const p = net.ckks[SQUARE_SET];
  const perCt = p.N / 2;
  const bitsOf = (e: number | null) => (e != null && e > 0 && e < 1 ? Math.floor(-Math.log2(e)) : null);
  const batch: FHEBatchResult = {
    logits, plainLogits, rows, rowsOffered: held.raw.length, rowsPerCiphertext: perCt, ciphertexts: r.blocks * net.n_in,
    fheMs: r.totalMs, msPerCiphertext: null, ctBytes: r.per_block.bytes.input / Math.max(1, net.n_in),
    maxAbsError: r.maxAbsError, rotations: 0, multiplies: 0, rotateMs: 0,
  };
  const pts = held.plane?.points;
  const plot: EncryptedAgreementPlot = {
    kind: "encrypted-agreement",
    points: logits.map((l, i) => ({
      enc: l[1] - l[0], plain: plainLogits[i][1] - plainLogits[i][0], agree: encCls[i] === plainCls[i],
      label: labels[i], encCls: encCls[i], plainCls: plainCls[i], ...(pts?.[i] ? { x: pts[i][0], y: pts[i][1] } : {}),
    })),
    rows, rowsOffered: held.raw.length, classes: 2, maxAbsError: r.maxAbsError, flipMarginMax,
    undecidableRows: tieRows, decidableRows: decidable,
    agreementOverDecidable: decidable > 0 ? 1 - flippedDecidable / decidable : null,
    heldOut: true, fitNote: held.fitNote, split: held.split, corpus: held.corpus,
    modelKey: SQUARE_KEY, modelLabel: squareLabel(net), licence: held.licence,
    ciphertexts: r.blocks * net.n_in, rowsPerCiphertext: perCt, fheMs: r.totalMs, msPerCiphertext: null,
    workNote: `${r.blocks} block(s) of up to ${perCt.toLocaleString("en-US")} rows, one row a slot: ${net.n_in} input `
      + `ciphertexts and 2 output ciphertexts a block, in ${r.ranIn}, under ${SQUARE_SET} (N = ${p.N.toLocaleString("en-US")})`,
    arithmeticNote: "a plaintext-weight product, a ciphertext square and a plaintext-weight product: depth 3, no rotation, "
      + "no layer decrypted in between",
    medianAbsError, noisyScores: devs.length, smallestMargin, flipBoundFactor, scoreKind: "logit",
    classNames: held.classNames, plane: held.plane ? { x: held.plane.x, y: held.plane.y } : null,
    accuracy: { encrypted: acc(encCls), plaintext: acc(plainCls), majority: majorityBaseline },
  };
  // The decision curve from the DECRYPTED logits, through the map fitted on validation after the run; the threshold is
  // folded into the last bias, so the network classifies at a probability of 0.5, as the linear student did.
  const dc = publishedDecisionCurve(held.usecase, squareSpec(net), (i) => sigmoid(plot.points[i].enc), labels, rows, 0.5);
  if (dc && "curve" in dc) plot.decisionCurve = dc.curve;
  else if (dc) plot.decisionCurveWithheld = dc.withheld;
  return {
    plot, batch, mismatches: flipped.length, tieRows, decidable, flippedDecidable, flipMarginMax, smallestMargin,
    medianAbsError, precisionBits: bitsOf(r.maxAbsError), medianPrecisionBits: bitsOf(medianAbsError), flipImpossible,
    encryptedAccuracy: acc(encCls), plaintextAccuracy: acc(plainCls), majorityBaseline,
  };
}

/** One transaction under CKKS through the square network: a block holding one row, in a Web Worker. */
export async function squareOneRow(net: FinanceSquare, raw: number[]): Promise<SquareTabBatch> {
  return runSquareTabularOffThread(net, [raw], SQUARE_SET);
}
