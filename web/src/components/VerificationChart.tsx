// What the in-run verification actually measured, drawn rather than summarised.
//
// The run already re-scores every published model on its held-out sample and reports a single
// number. A single number is the right thing to gate on and the wrong thing to reason from: at a
// 1-in-981 positive rate, "minority-class F1 0.51" does not tell you whether the model separates the
// classes or whether the threshold happens to sit somewhere lucky. These charts answer that, from
// the same arrays the metric was computed on - nothing here is recomputed or re-fetched.
//
// Chosen per task rather than one chart for everything:
//
//   binary        precision-recall curve + score histogram split by TRUE class. PR rather than ROC
//                 because the PR baseline moves with prevalence and ROC does not, which is the whole
//                 reason this project reports AUPRC (Saito & Rehmsmeier, PLOS ONE 2015).
//   multi-class   per-class F1 bars. The aggregate is precisely what hides a broken class - Engelen
//                 et al. (WTMC 2021) measured accuracy flat at 0.99 while a per-class F1 moved
//                 0.60 -> 0.99.
//
// recharts is already a dependency (BenchmarkPanel), so this costs no bundle growth.
import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Customized, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";

import { agreementStats } from "../lib/agreementStats";
import type { DecisionCurve } from "../lib/decisionCurve";
import type { ModelEvalResult } from "../lib/onnxInference";
import { modelsFor } from "../lib/config";
import Details from "./Details";
import { CanvasMarks, corners, type MarkSeries, NO_MARK } from "./CanvasMarks";
import { ChartReadout, HINT_KEYBOARD, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";

// The tick text's FILL (Round 86). This set `stroke`, which outlines each glyph and leaves recharts'
// default fill, #666, inside the outline: 2.96:1 on the chart panel, measured on every verification
// chart of the deployed pages (WCAG 2.2 SC 1.4.3 asks 4.5:1 at this size). #8a94a6 is 5.6:1 there.
const AXIS = { fontSize: 10, fill: "#8a94a6" };
/** The y-axis width of the two plane charts: their ticks and the rotated title `yAxisTitle` draws at
 *  a fixed offset share it, so it is sized for the wider of the fonts `system-ui` resolves to. */
const Y_AXIS_W = 72;
/** A y-axis title centred on the axis (Round 80). Recharts anchors a rotated `insideLeft` title at
 *  its START, at the axis middle, so the title runs upward by its whole length, and on these 230 px
 *  charts every title longer than about 100 px left the chart at the top - " (log)" was cut off
 *  "source burst fan-out (log)", and a `textAnchor` in the label props is overridden. */
const yAxisTitle = (text: string, dx: number) =>
  // A render function for Recharts' `label`, not a named component: it is passed, never written as a
  // tag, and tests/test_nothing_is_computed_and_then_rendered_nowhere.py reads a capitalised
  // function nobody writes as a tag as dead.
  ({ viewBox }: { viewBox?: { x: number; y: number; height: number } }) => {
    if (!viewBox) return null;
    const x = viewBox.x + dx, y = viewBox.y + viewBox.height / 2;
    return (
      // `central`, so the rotated glyphs sit evenly either side of x: on the alphabetic baseline their
      // bodies hung 11 px to one side and the first letters' tops left the chart by 3 px.
      <text x={x} y={y} transform={`rotate(-90, ${x}, ${y})`} textAnchor="middle" dominantBaseline="central"
        fill="#8a94a6" fontSize={10}>
        {text}
      </text>
    );
  };
/** A tick on a log axis: 1, 10, 100, 1k, 10k. */
const compactTick = (v: number) =>
  v >= 1e6 ? `${+(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${+(v / 1e3).toFixed(1)}k` : String(+v.toFixed(2));
const GRID = "#2a3040";
/** The scatter marks the three canvas charts draw (`CanvasMarks`, Round 85), as constants so a
 *  series' legend `<Scatter>` and its canvas points take one colour, and the contrast gate reads it. */
const MARK_BLUE = "#4f9dff";
const MARK_RED = "#ff6b6b";
const MARK_SLATE = "#8296b5";
const MARK_AMBER = "#f5a623";
const MARK_ORANGE = "#e07a3c";
const MARK_VIOLET = "#b072ff";
/** A legend-only `<Scatter>`'s data: its points are on the canvas. */
const NO_POINTS: never[] = [];
/** The card every chart in this file is drawn on (`--panel`). Stroked around stacked segments
 *  as a separator, so each segment meets this colour rather than its neighbour. */
const CARD = "#151c2c";

/** One axis of a 2-D data plane, as the payloads name it. */
type PlaneAxis = { key: string; label: string; scale?: "log" };

/** One plotted row. `x`/`y` place it on the corpus overlay's plane where the rows have one, and the
 *  three class indices let a panel say where the encrypted decisions are right and wrong. */
export interface AgreementPoint {
  enc: number; plain: number; agree: boolean;
  x?: number; y?: number; label?: number; encCls?: number; plainCls?: number;
}

export interface EncryptedAgreementPlot {
  kind: "encrypted-agreement";
  points: AgreementPoint[];
  rows: number;
  /** Rows the split offered. Larger than `rows` means the ciphertext could not hold them all. */
  rowsOffered?: number;
  classes: number;
  /** Worst deviation over ALL `rows * classes` scores, which is a strictly larger set than `points`.
      See the caption note: one point is plotted per row, for the winning class only. */
  maxAbsError?: number;
  /** Widest PLAINTEXT top-1-to-top-2 gap among the rows that flipped. Reported as a measurement.
      There is deliberately no tie/not-tie verdict beside it: a flip occurs exactly when the
      deviation exceeds the margin, so any comparison between the two restates that identity. */
  flipMarginMax?: number;
  /** Rows the plaintext model scores identically across classes, so their encrypted outcome is a
      coin toss and they carry no verification signal - and the agreement rate over the rest.

      In the caption UNCONDITIONALLY, because the count is a property of the corpus and the model,
      not of the run. The tie sentence below fires only when a flip happened, so on a run where
      every tie landed on the plaintext tie-break the figure read as a clean sweep. Genomics is the
      case: 7 of its 30 rows, 23% of the denominator. */
  undecidableRows?: number;
  decidableRows?: number;
  agreementOverDecidable?: number | null;
  /** WHAT THE ROWS ARE TO THE MODEL (Round 84). `true`: none was used to fit it, and `fitNote`
      names the rows that were. `false`: they are its fitting rows, so the figure tests the
      encryption and not generalisation. Absent: the producer did not say, and the caption claims
      neither. The caption once hardcoded "held-out" over finance's and medical's fitting rows. */
  heldOut?: boolean;
  fitNote?: string;
  /** The split, in the payload's words. */
  split?: string;
  /** For rows that are neither a fit nor a held-out split: genomics' external reference panel. */
  rowsAre?: string;
  corpus?: string;
  modelKey?: string | null;
  modelLabel?: string;
  /** The rows' licence with its link and change note: CC BY 4.0 s.3(a)(1) and CDLA-Sharing s.3.1(b)
      and s.3.3 ask for both whenever the data is shared, and this figure shares it. */
  licence?: { name: string; url: string | null; attribution: string; modifications: string };
  /** More than one when the split outgrew a ciphertext and the caller paid for every row. */
  ciphertexts?: number;
  rowsPerCiphertext?: number;
  fheMs?: number;
  msPerCiphertext?: number | null;
  /** Median deviation over the `noisyScores` class scores that were encrypted (a bias-only class is
      exact and is left out, or it would halve the median). */
  medianAbsError?: number | null;
  noisyScores?: number;
  /** Smallest non-zero plaintext margin. With `flipBoundFactor` it decides whether a flip was
      possible at all: none was when the margin exceeds the worst deviation times the factor (2 when
      every class score is encrypted, 1 when only one is). */
  smallestMargin?: number | null;
  flipBoundFactor?: 1 | 2;
  /** "logit": a two-class model's class-1-minus-class-0 logit on both axes. "winning-score": the
      encrypted top-1 class's score, the multiclass convention. */
  scoreKind?: "logit" | "winning-score";
  classNames?: string[];
  /** The corpus overlay's plane, where the points carry `x`/`y` on it. */
  plane?: { x: PlaneAxis; y: PlaneAxis } | null;
  /** Against the TRUE labels, over the scored rows. `majority` is what always calling the most common
      label scores on the same rows, the baseline an accuracy needs beside it (Round 86, review PI-12). */
  accuracy?: { encrypted: number | null; plaintext: number | null; majority?: number | null } | null;
  /** Net benefit of acting on the DECRYPTED scores across the declared thresholds, through the model's own
      calibration map (`publishedDecisionCurve`), or why there is none. Absent where the config declares no curve. */
  decisionCurve?: DecisionCurve;
  decisionCurveWithheld?: string;
}

/**
 * Takes every plotted mark out of the accessibility tree, so a chart is ONE image with one name.
 *
 * WHY THIS IS DOM WORK (UX39). Recharts 2.15 renders each scatter symbol as `<g role="img">` with no
 * name and no prop to change it (`Scatter.renderSymbolsStatically` hardcodes the role after every
 * passed prop). Finance drew 8,192 of them, each an unnamed image to a screen reader and a failure
 * of Lighthouse's `svg-img-alt`. The chart's `<svg role="img">` with its `title` is the one name, and
 * the figcaption beside it - tied by `aria-describedby` - is the text alternative.
 *
 * An observer rather than one pass: a resize gives the symbols new keys (recharts keys them on their
 * pixel position), so React mounts fresh nodes with the role back on them.
 */
export function MarksHidden({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const hide = () => {
      root.querySelectorAll(".recharts-scatter-symbol[role]").forEach((el) => {
        el.removeAttribute("role");
        el.setAttribute("aria-hidden", "true");
      });
    };
    hide();
    // childList only: the attribute writes above must not re-trigger the observer.
    const mo = new MutationObserver(hide);
    mo.observe(root, { childList: true, subtree: true });
    return () => mo.disconnect();
  }, []);
  return <div ref={ref}>{children}</div>;
}

/** The licence as CC BY 4.0 and CDLA-Sharing ask for it: name, a link to its text, the change note. */
function LicenceLine(
  { licence }: { licence: { name: string; url: string | null; attribution: string; modifications: string } },
) {
  return (
    <>
      {licence.url
        ? <a href={licence.url} target="_blank" rel="noopener noreferrer">{licence.name}</a>
        : <em>{licence.name}</em>}
      {licence.attribution ? <> - {sentence(licence.attribution)}</> : null}
      {licence.modifications ? <> Changes: {sentence(licence.modifications)}</> : null}
    </>
  );
}

const exp2 = (v: number) => v.toExponential(2);
/** A payload phrase printed as a sentence: the files end some notes with a full stop and not others. */
const sentence = (t: string) => (/[.!?]$/.test(t.trim()) ? t.trim() : `${t.trim()}.`);
/** -log2(e) floored, the bits OpenFHE and HElib print; null where an error >= 1 has none. */
const bitsOf = (e: number | null | undefined) =>
  (e != null && e > 0 && e < 1 ? Math.floor(-Math.log2(e)) : null);

/**
 * The one figure this project's central claim needs, and the one it never drew.
 *
 * Every chart beside this one shows a PLAINTEXT model's quality, or the input corpus it was scored
 * on. Neither shows what private inference produced. The claim being made throughout the page is
 * that the encrypted computation reproduces the plaintext one - and until now that claim reached the
 * visitor as a scalar taken over a SINGLE row: `encrypted_top1_mismatches` could only be 0 or 1, and
 * `encrypted_scores_compared` counted classes rather than rows.
 *
 * Each point is one row, encrypted score against plaintext score for the same row and class.
 * Agreement is the `y = x` line, so a deviation is visible as distance from it rather than as a
 * number someone has to interpret. Points are marked by whether the encrypted argmax matched the
 * plaintext argmax - the integer convention arXiv:2605.22237 Table IX reports and expects to be zero.
 *
 * WHICH ROWS, SAID BY THE PRODUCER (Round 84). The caption used to describe the rows itself, per use
 * case, and called finance's and medical's 400 rows "held-out" while the plotted model had been
 * fitted on exactly those rows. The rows' relation to the model now travels in the plot (`heldOut`,
 * `fitNote`, `rowsAre`) and the caption prints only what it is told; finance and medical now encrypt
 * rows their model was never fitted on, through `encryptedHeldOutAgreement`.
 */
export function EncryptedAgreementChart({ plot }: { plot: EncryptedAgreementPlot }) {
  const captionId = useId();
  // Every row, on the canvas (`CanvasMarks`): the plaintext score across, the decrypted one up.
  const marks = useMemo<MarkSeries[]>(() => {
    const pts = plot?.points ?? [];
    const differs = pts.filter((p) => !p.agree).map((p) => ({ x: p.plain, y: p.enc }));
    return [{ name: "agrees (circle)", fill: MARK_BLUE, shape: "circle",
              points: pts.filter((p) => p.agree).map((p) => ({ x: p.plain, y: p.enc })) },
            ...(differs.length ? [{ name: "argmax differs (cross)", fill: MARK_RED, shape: "cross" as const,
                                    points: differs }] : [])];
  }, [plot]);
  if (!plot?.points?.length) return null;
  let mismatches = 0, worst = 0, lo = Infinity, hi = -Infinity;
  // Loops, not `Math.max(...)` spreads: finance hands over 8,192 points.
  for (const p of plot.points) {
    if (!p.agree) mismatches++;
    worst = Math.max(worst, Math.abs(p.enc - p.plain));
    lo = Math.min(lo, p.enc, p.plain);
    hi = Math.max(hi, p.enc, p.plain);
  }
  const logit = plot.scoreKind === "logit";
  const who = plot.modelKey ? <code>{plot.modelKey}</code> : <>{plot.modelLabel ?? "the model"}</>;
  const name = `Encrypted score against plaintext score, ${plot.rows}`
    + `${plot.heldOut ? " held-out" : ""} rows`;
  const factor = plot.flipBoundFactor ?? 2;
  const noFlipPossible = plot.smallestMargin != null && typeof plot.maxAbsError === "number"
    && plot.smallestMargin > factor * plot.maxAbsError;
  const medianBits = bitsOf(plot.medianAbsError), worstBits = bitsOf(plot.maxAbsError ?? worst);
  return (
    <figure className="chart-figure">
      {/* A strip of the numbers the caption below qualifies, for a reader scanning the page. The
          caption keeps every denominator and caveat beside the figure (checklist UX6); only the
          meaning of a flipped row's margin is folded. Round 58. A description list since Round 86:
          it was one table row of alternating label and value cells, which no header cell can
          describe (a row header heads its whole row), and the audit counted it as a table with none. */}
      <dl className="facts-strip small">
        <div><dt>rows scored</dt><dd>{plot.rows}{plot.rowsOffered != null && plot.rowsOffered > plot.rows ? ` of ${plot.rowsOffered}` : ""}</dd></div>
        {plot.ciphertexts != null ? <div><dt>ciphertexts</dt><dd>{plot.ciphertexts}</dd></div> : null}
        <div><dt>top-1 mismatches</dt><dd>{mismatches}</dd></div>
        <div><dt>plaintext ties</dt><dd>{plot.undecidableRows ?? "not reported"}</dd></div>
        <div><dt>worst deviation</dt><dd>{exp2(plot.maxAbsError ?? worst)}</dd></div>
        {plot.medianAbsError != null ? <div><dt>median deviation</dt><dd>{exp2(plot.medianAbsError)}</dd></div> : null}
        {plot.smallestMargin != null ? <div><dt>smallest margin</dt><dd>{exp2(plot.smallestMargin)}</dd></div> : null}
      </dl>
      <figcaption id={captionId} className="muted small">
        <strong>Private inference against the plaintext reference.</strong>{" "}
        {plot.heldOut === true ? (
          <>
            Each point is one of <strong>{plot.rows} held-out rows</strong>
            {plot.corpus ? <> of {plot.corpus}</> : null}, scored under CKKS in this tab with{" "}
            {plot.modelLabel ?? "the model"}{plot.modelKey ? <> (<code>{plot.modelKey}</code>)</> : null}{" "}
            and plotted against the same row&apos;s plaintext score. <strong>None of these rows was
            used to fit the plotted model</strong>{plot.fitNote ? <>: {sentence(plot.fitNote)}</> : "."}
            {plot.split ? <> Split: {sentence(plot.split)}</> : null}
          </>
        ) : plot.heldOut === false ? (
          <>
            Each point is one of {plot.rows} rows <strong>{who} was fitted on</strong>, scored under
            CKKS in this tab against the same row&apos;s plaintext score, so the figure tests the
            encryption, not generalisation.
          </>
        ) : plot.rowsAre ? (
          <>
            Each point is one of {plot.rows} {plot.rowsAre}, scored under CKKS in this tab and plotted
            against the same row&apos;s plaintext score.
          </>
        ) : (
          <>
            Each point is one of {plot.rows} rows scored under CKKS in this tab, plotted against the
            same row&apos;s plaintext score.
          </>
        )}{" "}
        {logit
          ? <>Both axes are the decision logit, class 1 minus class 0; the diagonal is exact agreement
              and a flip would sit where the two signs differ.</>
          : <>Both axes are the score of the class the encrypted model ranks first; the diagonal is
              exact agreement.</>}{" "}
        Top-1 mismatches: <strong>{mismatches}</strong> of {plot.rows}.{" "}
        {plot.ciphertexts != null && plot.ciphertexts > 1 ? (
          <>
            The rows travelled in <strong>{plot.ciphertexts} ciphertexts</strong> of up to{" "}
            {plot.rowsPerCiphertext ?? "?"} rows each{typeof plot.fheMs === "number"
              ? <>, {plot.fheMs.toFixed(0)} ms of encrypted work in all</> : null}{plot.msPerCiphertext != null
              ? <> ({plot.msPerCiphertext.toFixed(1)} ms per ciphertext)</> : null}: scoring every row
            of the split costs one ciphertext per {plot.rowsPerCiphertext ?? "?"} rows.
          </>
        ) : (
          <>All {plot.rows} rows travelled in a single ciphertext, so they cost the homomorphic work of one.</>
        )}
        {/* THE CIPHERTEXT CAP, restored.
            `5d20d88` put this in the caption precisely so a reader could not take the agreement for
            the whole split, and `2393c79` - the commit that removed the tie verdict from this same
            sentence - deleted it as collateral: the removed block's closing `) : null}` was merged
            with the tie block's, so the cap clause went out with the verdict. `rowsOffered` was left
            declared and set by four producers and read by nothing, which no type check can see. */}
        {plot.rowsOffered != null && plot.rowsOffered > plot.rows ? (
          <>
            {" "}The split holds <strong>{plot.rowsOffered}</strong> rows; one ciphertext fits{" "}
            {plot.rows} at this feature width, so {plot.rowsOffered - plot.rows} were not scored and
            the agreement above is over {plot.rows}, not over the whole split.
          </>
        ) : null}
        {/* AT ZERO TOO. Gated on `> 0` first, which put the disclosure on genomics and hid it on
            telecom - and telecom's zero is the informative case: it is what makes that page's 1.00
            a different claim from genomics' 1.00. Suppressing a count because it is zero is the
            same conditional-disclosure defect this sentence exists to fix, one level down. */}
        {plot.undecidableRows != null ? (
          plot.undecidableRows > 0 ? (
            <>
              {" "}<strong>{plot.undecidableRows} of those {plot.rows} rows are scored identically
              across classes by the plaintext model itself</strong>, so its own top-1 there comes
              from argmax tie-breaking and the encrypted outcome is decided by encryption noise -
              they carry no signal either way, and the count above moves between runs because of
              them. Over the {plot.decidableRows} rows the model does decide, the encrypted score
              agrees with the plaintext one{" "}
              {/* A PROPORTION NEEDS ITS DENOMINATOR NAMED (UX5), at one decimal place: `toFixed(2)`
                  on a fraction printed 0.9825 and 0.9849 both as 0.98. */}
              <strong>{plot.agreementOverDecidable != null
                ? `${(plot.agreementOverDecidable * 100).toFixed(1)}% of the time`
                : "by an amount that was not computed"}</strong>.
            </>
          ) : (
            <>
              {" "}<strong>None</strong> of these {plot.rows} rows is a tie in the plaintext model -
              it separates every one of them - so the figure above is over rows that carry a real
              decision, and the agreement is not diluted by any coin toss.
            </>
          )
        ) : null}
        {/* SCOPE, stated because the first version of this caption did not state it and was read
            wrong as a result. One point is plotted per row, so `worst` is a maximum over `rows`
            deviations; `maxAbsError` is the maximum over `rows * classes`, a strictly larger set.
            On finance those two once read 1.80e-8 and 2.47e-8: both correct, 37% apart. */}
        {" "}Worst deviation among the {plot.rows} plotted scores: {exp2(worst)}
        {/* Counted from the scores CKKS evaluated (`noisyScores`), not rows x classes. Until Round 86 this
            read "across all 16384 class scores" on finance, whose one-logit student gets a zero class-0
            row that is returned as its bias and never encrypted (`realData.ts`, `clientFHE.ts`); so it
            doubled the count and promised unplotted scores that do not exist. */}
        {typeof plot.maxAbsError === "number" ? (
          (plot.noisyScores ?? plot.rows * plot.classes) > plot.rows
            ? <>; across all {plot.noisyScores ?? plot.rows * plot.classes} encrypted class scores, including the
                ones this figure does not plot: {exp2(plot.maxAbsError)}</>
            : <>; that is every score CKKS evaluated here</>
        ) : null}
        {worstBits != null ? <> (about {worstBits} bits)</> : null}.
        {plot.medianAbsError != null ? (
          <> Median over the {plot.noisyScores ?? "encrypted"} class scores that went through CKKS:{" "}
            {exp2(plot.medianAbsError)}{medianBits != null ? <> (about {medianBits} bits)</> : null}.</>
        ) : null}
        {/* THE NO-FLIP SENTENCE ONLY WHEN IT IS TRUE. A flip at a margin below twice the deviation
            is always possible, so the sentence is stated only when the inequality excludes it, and
            otherwise the closest row is reported as within reach. Factor 1 where only one class
            score is encrypted (the bias-only class 0 of finance and medical is exact). */}
        {plot.smallestMargin != null && typeof plot.maxAbsError === "number" ? (
          noFlipPossible ? (
            <> No decided row could have flipped: the smallest plaintext margin, {exp2(plot.smallestMargin)},
              is above {factor === 2 ? "twice " : ""}the worst deviation.</>
          ) : (
            <> The smallest plaintext margin, {exp2(plot.smallestMargin)}, is within{" "}
              {factor === 2 ? "twice " : ""}the worst deviation, so the closest rows could have flipped.</>
          )
        ) : null}
        {/* What a mismatch MEANS - stated, not adjudicated. Two earlier versions computed a
            tie/not-tie verdict here; both restated the theorem that a flip happens exactly when the
            noise exceeds the margin. The count of rows the model scores IDENTICALLY is the number
            that varies, and it is reported above, unconditionally. */}
        {mismatches > 0 && typeof plot.flipMarginMax === "number" ? (
          <>
            {" "}Widest margin the plaintext model held at those {mismatches} flipped rows:{" "}
            {plot.flipMarginMax === 0
              ? <><strong>exactly zero</strong> - it does not distinguish those samples at all.</>
              : <>{exp2(plot.flipMarginMax)}.</>}
          </>
        ) : null}
        {/* THE PUBLISHED SCALE, AS A RELATION AND NOT A NUMBER TO BEAT (the owner's choice, Round 36).
            Lee et al. is the nearest primary source that reports this figure's quantity, the share of
            inputs on which the encrypted model's top class equals the plaintext model's. Every figure
            here encrypts one linear layer (one plaintext multiply and its rotations), so the two are
            not like for like and the sentence says which way the difference should point. */}
        {/* The peer-reviewed version since Round 86: IEEE Access 10 (2022) reports 98.43% and 92.43% +/- 2.65%
            accuracy, where the 2021 preprint said 98.67%. And "one multiplicative level" counted one of the
            two plaintext multiplications a layer makes here (the weights, then the 0/1 output mask). */}
        {" "}For scale only: a 20-layer ResNet under RNS-CKKS, with bootstrapping and approximated ReLU,
        reported 98.43% identical results to its plaintext model on CIFAR-10 (Lee et al., IEEE Access 10,
        30039-30054, 2022; the 2021 preprint, arXiv:2106.07229, reported 98.67%). This model is one linear
        layer, two plaintext multiplications (the weights, then a 0/1 output mask) with no rescale, no
        bootstrapping and no approximated activation, so it does far less homomorphic arithmetic and should
        agree more often. Neither figure ranks the other.
        {plot.licence ? <> Rows: <LicenceLine licence={plot.licence} /></> : null}
      </figcaption>
      {mismatches > 0 && typeof plot.flipMarginMax === "number" && plot.flipMarginMax > 0 ? (
        <Details summary="What a flipped row's margin means">
          A flip happens precisely when the deviation exceeds the margin, so the margin is always below
          twice the worst deviation; it is reported as a measurement, not as a verdict on the encryption.
        </Details>
      ) : null}
      <ChartReadout hint={HINT_POINTER}>
        <MarksHidden>
        <ResponsiveContainer width="100%" height={200}>
          {/* `title` as a PROP, the way every other chart in this file sets it: recharts does not
              forward a `<title>` child, so the figure rendered with a null accessible name. And
              `role="img"`, NOT `accessibilityLayer`: on a ScatterChart that sets role="application"
              without a keyboard path, because recharts steps an activeIndex along a categorical
              axis and two numeric axes have none (`KB6`). The figcaption is the text alternative. */}
          <ScatterChart role="img" title={name} aria-describedby={captionId}
                        margin={{ top: 6, right: 10, bottom: 14, left: 4 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            {/* Formatted, because the raw ticks rendered as `3.7326450091317454` - sixteen digits of a
                logit, on both axes. Both axes TITLED (PI9): the y axis is the decrypted one. */}
            <XAxis type="number" dataKey="plain" name="plaintext" domain={[lo, hi]} tick={AXIS}
                   tickFormatter={(v: number) => v.toFixed(2)}
                   label={{ value: logit ? "plaintext logit" : "plaintext score", position: "insideBottom",
                            offset: -10, fill: "#8a94a6", fontSize: 12 }} />
            {/* The corner is the lower bound of both axes, and both labelled it: the two "-24.03" met at the origin. The x axis keeps it. */}
            <YAxis type="number" dataKey="enc" name="encrypted" domain={[lo, hi]} tick={AXIS}
                   tickFormatter={(v: number) => (Math.abs(v - lo) < 1e-9 ? "" : v.toFixed(2))}
                   label={yAxisTitle(logit ? "encrypted logit (decrypted)" : "encrypted score (decrypted)", 14)} />
            <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#8a94a6"
                           strokeDasharray="4 4" ifOverflow="extendDomain" />
            {/* SHAPE as well as fill (SC 1.4.1): red here means the ENCRYPTED path disagreed with
                plaintext, and a reader who cannot separate red from blue must still see which points
                are the disagreements. The shape is named in the series name, so the legend, the
                readout and the figcaption all say it.
                `isAnimationActive={false}` IS LOAD-BEARING: `shape` moves recharts onto `<Symbols>`,
                whose `size` animates, and with animation on every symbol stayed at `d="M0,0"` -
                present, positioned, invisible. Found only on the deployed page.
                The points themselves are on the canvas (Round 85); these keep the legend. */}
            <Scatter name="agrees (circle)" data={NO_POINTS}
                     fill={MARK_BLUE} shape="circle" isAnimationActive={false} />
            {mismatches > 0 && (
              <Scatter name="argmax differs (cross)" data={NO_POINTS}
                       fill={MARK_RED} shape="cross" isAnimationActive={false} />
            )}
            {mismatches > 0 && <Legend wrapperStyle={{ fontSize: 10 }} />}
            <Customized component={<CanvasMarks series={marks} xName="plaintext" yName="encrypted"
                                                format={(v: number) => v.toFixed(6)} />} />
          </ScatterChart>
        </ResponsiveContainer>
        </MarksHidden>
      </ChartReadout>
      <AgreementEvidence plot={plot} />
      {plot.plane ? <DecisionPlane plot={plot} plane={plot.plane} /> : null}
      {plot.decisionCurve ? (
        <DecisionCurveChart curve={plot.decisionCurve} modelKey={plot.modelKey ?? "the model"} n={plot.rows}
                            scored="decrypted" />
      ) : plot.decisionCurveWithheld ? (
        <p className="muted small">No decision curve: {plot.decisionCurveWithheld}.</p>
      ) : null}
    </figure>
  );
}

/**
 * Two readings of the scatter's pairs (Round 86, the owner's choice of 2026-11-14: the evidence plots). Left, Bland
 * and Altman's difference against mean, with the bias and the limits of agreement; right, each row's precision in bits
 * as Lattigo counts it. Computed by `lib/agreementStats.ts` from the same points the scatter draws, one per row. The
 * reference lines carry no label on the chart, where a label would sit on its line; the caption names them.
 */
function AgreementEvidence({ plot }: { plot: EncryptedAgreementPlot }) {
  const captionId = useId();
  const st = useMemo(() => agreementStats(plot.points ?? []), [plot]);
  const marks = useMemo<MarkSeries[]>(() => (st ? [{ name: "a row (circle)", fill: MARK_BLUE, shape: "circle", points: st.points }] : []), [st]);
  if (!st || st.n < 2) return null;
  let xlo = Infinity, xhi = -Infinity, ylo = Math.min(st.limits[0], 0), yhi = Math.max(st.limits[1], 0);
  for (const q of st.points) { xlo = Math.min(xlo, q.x); xhi = Math.max(xhi, q.x); ylo = Math.min(ylo, q.y); yhi = Math.max(yhi, q.y); }
  const pad = (yhi - ylo) * 0.08 || 1e-12;
  const logit = plot.scoreKind === "logit";
  const e = (v: number) => v.toExponential(2);
  const bars = st.bits.map((b) => ({ bits: String(b.bits), count: b.count }));
  return (
    <div className="agreement-evidence" style={{ marginTop: 10 }}>
      <div className="grid cols-2" style={{ gap: 10 }}>
        <ChartReadout hint={HINT_POINTER}>
          <MarksHidden>
            <ResponsiveContainer width="100%" height={210}>
              <ScatterChart role="img" title={`Bland-Altman: encrypted minus plaintext against their mean, ${st.n} rows`}
                            aria-describedby={captionId} margin={{ top: 6, right: 10, bottom: 14, left: 4 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                <XAxis type="number" dataKey="x" domain={[xlo, xhi]} tick={AXIS} tickFormatter={(v: number) => v.toFixed(2)}
                       label={{ value: "mean", position: "insideBottom",
                                offset: -10, fill: "#8a94a6", fontSize: 12 }} />
                <YAxis type="number" dataKey="y" domain={[ylo - pad, yhi + pad]} tick={AXIS} width={Y_AXIS_W}
                       tickFormatter={(v: number) => v.toExponential(1)} label={yAxisTitle("encrypted minus plaintext", 14)} />
                <ReferenceLine y={st.bias} stroke="#e6edf6" strokeWidth={1.2} ifOverflow="extendDomain" />
                <ReferenceLine y={st.limits[0]} stroke="#8a94a6" strokeDasharray="5 4" ifOverflow="extendDomain" />
                <ReferenceLine y={st.limits[1]} stroke="#8a94a6" strokeDasharray="5 4" ifOverflow="extendDomain" />
                <Scatter name="a row (circle)" data={NO_POINTS} fill={MARK_BLUE} shape="circle" isAnimationActive={false} />
                <Customized component={<CanvasMarks series={marks} xName="mean" yName="difference"
                                                    format={(v: number) => v.toExponential(3)} />} />
              </ScatterChart>
            </ResponsiveContainer>
          </MarksHidden>
        </ChartReadout>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={bars} accessibilityLayer title={`Precision of the ${st.n - st.exact} rows with a difference, in bits`}
                    aria-describedby={captionId} margin={{ top: 6, right: 10, bottom: 14, left: 4 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis dataKey="bits" tick={AXIS} label={{ value: "precision, bits", position: "insideBottom",
                                                       offset: -10, fill: "#8a94a6", fontSize: 12 }} />
            <YAxis allowDecimals={false} tick={AXIS} width={Y_AXIS_W} label={yAxisTitle("rows", 14)} />
            <Bar dataKey="count" fill={MARK_BLUE} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p id={captionId} className="muted small" style={{ marginTop: 4 }}>
        <strong>The same pairs, two more ways.</strong> Left, Bland and Altman&apos;s plot (Lancet, 1986): each of the {st.n} rows&apos;
        difference, encrypted minus plaintext, against the mean of the two {logit ? "logits" : "scores"}. The solid line is the mean difference, the bias, {e(st.bias)};
        the dashed lines are the limits of agreement, the bias plus and minus 1.96 standard deviations of the differences,{" "}
        {e(st.limits[0])} and {e(st.limits[1])}, and {st.within} of the {st.n} rows lie between them. The two methods are one
        model in the clear and under CKKS, so the bias says whether the encryption shifts the scores and the limits how far a
        row&apos;s score moves. Right, each row&apos;s precision, -log2 |encrypted - plaintext|, as Lattigo&apos;s GetPrecisionStats
        counts it, in bins of whole bits{st.medianBits != null && st.minBits != null ? <>: median {st.medianBits.toFixed(1)} bits, lowest {st.minBits.toFixed(1)}</> : null}
        {st.exact > 0 ? <>; {st.exact} {st.exact === 1 ? "row came" : "rows came"} back exactly equal and {st.exact === 1 ? "is" : "are"} not binned</> : null}.
        One point per row, the class the scatter plots, so these are over {st.n} scores rather than every class score CKKS
        evaluated.
      </p>
    </div>
  );
}

/** The decision plane's outcome series, in the order the legend keys them, flips last and on top.
 *  Built from the same filters the counts table uses, so the canvas draws what the table counts. */
function decisionMarks(plot: EncryptedAgreementPlot, plane: { x: PlaneAxis; y: PlaneAxis }): MarkSeries[] {
  const logY = plane.y.scale === "log";
  const pts = plot.points.filter((p) => typeof p.x === "number" && typeof p.y === "number"
    && typeof p.label === "number" && typeof p.encCls === "number");
  // A row at or below zero has no place on a log axis; the caption counts the ones left out.
  const drawn = logY ? pts.filter((p) => (p.y as number) > 0) : pts;
  const xy = (p: AgreementPoint) => ({ x: p.x as number, y: p.y as number });
  const names = plot.classNames ?? [];
  const nameOf = (c: number) => names[c] ?? `class ${c}`;
  const cell = (t: number, d: number) => drawn.filter((p) => p.label === t && p.encCls === d).map(xy);
  const out: MarkSeries[] = plot.classes === 2
    ? [{ name: `${nameOf(0)}, called ${nameOf(0)} (circle)`, fill: MARK_SLATE, shape: "circle", points: cell(0, 0) },
       { name: `${nameOf(0)}, called ${nameOf(1)} (diamond)`, fill: MARK_AMBER, shape: "diamond", points: cell(0, 1) },
       { name: `${nameOf(1)}, called ${nameOf(1)} (triangle)`, fill: MARK_ORANGE, shape: "triangle", points: cell(1, 1) },
       { name: `${nameOf(1)}, called ${nameOf(0)} (square)`, fill: MARK_VIOLET, shape: "square", points: cell(1, 0) }]
    : [{ name: "class right (circle)", fill: MARK_SLATE, shape: "circle",
         points: drawn.filter((p) => p.label === p.encCls).map(xy) },
       { name: "class wrong (triangle)", fill: MARK_ORANGE, shape: "triangle",
         points: drawn.filter((p) => p.label !== p.encCls).map(xy) }];
  const flips = drawn.filter((p) => !p.agree).map(xy);
  if (flips.length) out.push({ name: "encryption changed the decision (cross)", fill: MARK_RED, shape: "cross", points: flips });
  return out;
}

/**
 * Where on the corpus the ENCRYPTED decisions fall, on the overlay's own plane (Round 84).
 *
 * The agreement scatter above says whether encryption changed a decision; it cannot say whether the
 * decision was right, or where the wrong ones sit. This draws the same held-out rows on the two
 * features the corpus overlay uses, each marked by its outcome against the TRUE class - so the
 * overlay's "how separable is this corpus" and the model's "where does it fail" can be read against
 * each other. The counts table below it is the text alternative.
 *
 * One series per outcome, each its own shape and colour, so no outcome is told apart by colour alone
 * (SC 1.4.1). Flips are drawn last, on top.
 */
function DecisionPlane({ plot, plane }: { plot: EncryptedAgreementPlot; plane: { x: PlaneAxis; y: PlaneAxis } }) {
  const captionId = useId(), tableId = useId();
  const marks = useMemo(() => decisionMarks(plot, plane), [plot, plane]);
  const pts = plot.points.filter((p) => typeof p.x === "number" && typeof p.y === "number"
    && typeof p.label === "number" && typeof p.encCls === "number");
  if (!pts.length) return null;
  const names = plot.classNames ?? [];
  const nameOf = (c: number) => names[c] ?? `class ${c}`;
  const logY = plane.y.scale === "log";
  // A row at or below zero has no place on a log axis. Counted and said, not silently clipped.
  const offAxis = logY ? pts.filter((p) => (p.y as number) <= 0).length : 0;
  const flips = pts.filter((p) => !p.agree);
  const binary = plot.classes === 2;
  const count = (f: (p: AgreementPoint) => boolean) => pts.filter(f).length;
  return (
    <figure className="chart-figure" style={{ marginTop: 10 }}>
      <figcaption id={captionId} className="muted small">
        <strong>Where the encrypted decisions fall.</strong> The same {pts.length} rows on{" "}
        {plane.x.label} against {plane.y.label}, the plane of this use case&apos;s corpus overlay, each
        marked by the ENCRYPTED decision against its true class.{plot.heldOut
          ? " A correct decision on a row the model never saw is evidence about the model." : ""} A
        cross is a row where encryption changed the decision ({flips.length} here).{offAxis > 0 ? <> {offAxis} rows with a {plane.y.label} of 0 cannot
          sit on the log axis; they are in the counts below but not drawn.</> : null}
        {!binary && plot.accuracy?.encrypted != null ? (
          <> Encrypted top-1 matches the true class on {(plot.accuracy.encrypted * 100).toFixed(1)}% of
            the rows{plot.accuracy.plaintext != null
              ? <>, plaintext on {(plot.accuracy.plaintext * 100).toFixed(1)}%</> : null}{plot.accuracy.majority != null
              ? <>; always calling the most common class would score {(plot.accuracy.majority * 100).toFixed(1)}%</> : null}.</>
        ) : null}
      </figcaption>
      <table id={tableId} className="facts-table small" style={{ marginBottom: 6 }}>
        {binary ? (
          <tbody>
            <tr><th scope="col">true class</th><th scope="col">called {nameOf(0)}</th>
                <th scope="col">called {nameOf(1)}</th></tr>
            {[0, 1].map((t) => (
              <tr key={t}><th scope="row">{nameOf(t)}</th>
                <td>{count((p) => p.label === t && p.encCls === 0)}</td>
                <td>{count((p) => p.label === t && p.encCls === 1)}</td></tr>
            ))}
          </tbody>
        ) : (
          // Column headers over one row of counts (Round 86). It was one row of alternating label and
          // value cells, which no header cell can describe, and the audit found it with none.
          <tbody>
            <tr><th scope="col">class right</th><th scope="col">class wrong</th>
                <th scope="col">encryption changed the class</th></tr>
            <tr><td>{count((p) => p.label === p.encCls)}</td><td>{count((p) => p.label !== p.encCls)}</td>
                <td>{flips.length}</td></tr>
          </tbody>
        )}
      </table>
      <ChartReadout hint={HINT_POINTER}>
        <MarksHidden>
        <ResponsiveContainer width="100%" height={230}>
          <ScatterChart role="img" title={`Encrypted decisions on ${plane.y.label} against ${plane.x.label}`}
            aria-describedby={`${captionId} ${tableId}`} margin={{ top: 6, right: 10, bottom: 30, left: -14 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis type="number" dataKey="x" name={plane.x.label} tick={AXIS}
                   label={{ value: plane.x.label, position: "insideBottom", offset: -20,
                            fill: "#8a94a6", fontSize: 12 }} />
            {/* 72 px, not Recharts' 60 (Round 84): in DejaVu Sans, the CI runners' and many Linux
                visitors' system-ui, the held-out telecom rows' five-character ticks (0.075, 0.225)
                reached the rotated title at its fixed offset; in Noto they did not, so local scans passed. */}
            <YAxis type="number" dataKey="y" name={plane.y.label} tick={AXIS} width={Y_AXIS_W}
                   tickFormatter={logY ? compactTick : undefined}
                   scale={logY ? "log" : "auto"} domain={logY ? [0.9, "auto"] : undefined}
                   allowDataOverflow={logY}
                   label={yAxisTitle(plane.y.label + (logY ? " (log)" : ""), 24)} />
            <Legend verticalAlign="top" align="right" height={30} wrapperStyle={{ fontSize: 10 }} />
            {/* Fills from top-level constants, one per outcome: the contrast gate resolves a colour
                only from a literal or a top-level constant, and an unresolvable mark fails it. The
                points are on the canvas (Round 85); these keep the legend, and the corners keep the
                automatic domain. */}
            {binary ? (
              <>
                <Scatter name={`${nameOf(0)}, called ${nameOf(0)} (circle)`} data={NO_POINTS}
                         fill={MARK_SLATE} shape="circle" isAnimationActive={false} />
                <Scatter name={`${nameOf(0)}, called ${nameOf(1)} (diamond)`} data={NO_POINTS}
                         fill={MARK_AMBER} shape="diamond" isAnimationActive={false} />
                <Scatter name={`${nameOf(1)}, called ${nameOf(1)} (triangle)`} data={NO_POINTS}
                         fill={MARK_ORANGE} shape="triangle" isAnimationActive={false} />
                <Scatter name={`${nameOf(1)}, called ${nameOf(0)} (square)`} data={NO_POINTS}
                         fill={MARK_VIOLET} shape="square" isAnimationActive={false} />
              </>
            ) : (
              <>
                <Scatter name="class right (circle)" data={NO_POINTS}
                         fill={MARK_SLATE} shape="circle" isAnimationActive={false} />
                <Scatter name="class wrong (triangle)" data={NO_POINTS}
                         fill={MARK_ORANGE} shape="triangle" isAnimationActive={false} />
              </>
            )}
            {flips.length > 0 && (
              <Scatter name="encryption changed the decision (cross)" data={NO_POINTS}
                       fill={MARK_RED} shape="cross" isAnimationActive={false} />
            )}
            <Scatter data={corners(marks)} fill={MARK_SLATE} shape={NO_MARK} legendType="none"
                     isAnimationActive={false} />
            <Customized component={<CanvasMarks series={marks} xName={plane.x.label} yName={plane.y.label}
                                                format={(v: number) => v.toLocaleString()} />} />
          </ScatterChart>
        </ResponsiveContainer>
        </MarksHidden>
      </ChartReadout>
    </figure>
  );
}

// The real corpus, drawn underneath the model's own numbers.
//
// A score on its own cannot distinguish a good model on easy data from a mediocre model on hard
// data - the reader has no way to see how separable the classes were to begin with. This draws the
// actual corpus so that question is answerable by looking. It draws TRUE classes only: no model's
// decision is on it, and the caption no longer says otherwise (it called itself "a genuine
// train-versus-inference comparison" while drawing no inference). Where a page encrypts these rows,
// the private-inference card draws the encrypted decisions on this same plane.
//
// What the payload states, rather than the caption guessing:
//
//   1. `is_training_corpus`: whether this is the corpus the Models card's published models train on
//      (finance, satellite, telecom) or a substitute (medical's WDBC beside BreastMNIST models,
//      genomics' gnomAD frequencies). `models` names which models, so "this model" is never vague.
//   2. `split_note`: which rows these are relative to those models' fits - held out, or the whole
//      corpus. Telecom's overlay was all 3,200 generated rows, 70 percent of them a QCNN's training
//      rows, under a caption reading "held-out split".
//   3. `licence`, `licence_url`, `modifications`: CC BY 4.0 s.3(a)(1) and CDLA-Sharing s.3.1(b) and
//      s.3.3 ask for the licence's link and a note of the changes, not only its name (RD23).
//
// Regenerated by scripts/prep_real_data_2d.py; licence and attribution travel inside the payload
// so they cannot drift away from the points they describe.
type Real2D = {
  usecase: string;
  is_training_corpus: boolean;
  /** Whether the rows were SIMULATED rather than measured. Carried rather than inferred from the
   *  provenance prose, because the caption below used to hardcode the word "real" for all five
   *  payloads and two of them are simulator output - AMLworld by IBM's own statement, and the
   *  telecom corpus by this repository. The finance page said "8192 real points" about simulated
   *  transactions until telecom made the wording visible. */
  corpus_is_synthetic?: boolean;
  provenance: string;
  licence: string;
  /** Link to the licence text; null where nothing is licensed (the generated telecom corpus). */
  licence_url?: string | null;
  attribution: string;
  /** What was changed from the source, as CC BY 4.0 and CDLA-Sharing ask to be noted. */
  modifications?: string;
  /** Registry keys of the published models this corpus belongs to; empty for a substitute. */
  models?: string[];
  /** Which rows these are relative to those models' fits. */
  split_note?: string;
  x: PlaneAxis;
  y: PlaneAxis;
  class_names: [string, string];
  separability_auc_pr?: number;
  /** The estimator behind `separability_auc_pr`, named per payload. */
  separability_method?: string;
  base_rate: number;
  note: string;
  points: Array<[number, number, number]>;
};

const REAL_2D_PATH: Record<string, string> = {
  satellite_comms: "data/satellite/real_2d.json",
  medical_fl: "data/medical/real_2d.json",
  finance_aml: "data/finance/real_2d.json",
  genomics: "data/genomics/real_2d.json",
  // Added 2026-08-13. Telecom was the only use case with no entry here, and the reason was not
  // licence - `prep_real_data_2d.py` records that its corpus "is generated by this repository, so
  // there is no third party to ask". The use case with the fewest obstacles was the one with no
  // chart, which is the shape of an oversight rather than a decision.
  telecom_ids: "data/telecom/real_2d.json",
};

/**
 * `card` asks this component to draw its own bordered frame.
 *
 * WHY THE FRAME IS A PROP AND NOT THE CALLER'S `<div>`. Round 41 added this chart to the
 * orchestration page inside an unconditional `<div className="card">`, with a comment stating that
 * the component "renders null where no overlay ships, which covers the generic `demo` entry". The
 * component does. The wrapper did not, so `/orchestration?usecase=demo` drew an empty bordered box
 * after every run - while the checklist row written in that same commit promises "It renders nothing
 * for the generic `demo` entry".
 *
 * Whether anything renders is known HERE and only here: `demo` has no `REAL_2D_PATH` entry, and even
 * a mapped use case can fetch a payload with no points. A caller cannot condition on an answer that
 * arrives in a `useEffect`, so a caller-side frame is a copy of a decision the caller cannot make.
 */
/** Overlays beyond a use case's own, by name: the telecom page has two corpora, one per model family (Round 86). */
const REAL_2D_EXTRA: Record<string, string> = { telecom_nidd: "data/telecom_nidd/real_2d.json" };

export function RealDataScatter({ usecase, card = false, overlay }: { usecase: string; card?: boolean; overlay?: string }) {
  const [data, setData] = useState<Real2D | null>(null);
  const captionId = useId();
  // The two classes, on the canvas (`CanvasMarks`, Round 85), at the opacities the SVG symbols had.
  const marks = useMemo<MarkSeries[]>(() => (data?.points?.length ? [0, 1].map((c) => ({
    name: data.class_names[c], fill: c ? MARK_ORANGE : MARK_SLATE, shape: c ? "triangle" as const : "circle" as const,
    opacity: c ? 0.85 : 0.75,
    points: data.points.filter((p) => p[2] === c).map((p) => ({ x: p[0], y: p[1] })),
  })) : []), [data]);
  useEffect(() => {
    const rel = overlay ? REAL_2D_EXTRA[overlay] : REAL_2D_PATH[usecase];
    if (!rel) { setData(null); return; }
    let live = true;
    // Missing file is not an error worth shouting about - the generator skips a use case whose
    // corpus is not present locally, and a demo without the overlay is still a correct demo.
    // `r.ok` is not the test: this is an SPA, so the host answers 200 with index.html for a path
    // that does not exist. Content-type is the discriminator; without it the branch reads as
    // handling the absent case while the `catch` on HTML actually does the work.
    fetch(`${import.meta.env.BASE_URL}${rel}`)
      .then((r) => (r.ok && (r.headers.get("content-type") ?? "").includes("application/json")
        ? r.json() : null))
      .then((j) => { if (live) setData(j); })
      .catch(() => { if (live) setData(null); });
    return () => { live = false; };
  }, [usecase, overlay]);

  // The single decision point. Everything below - including the frame, when `card` is set - is
  // downstream of it, so there is no second place that can answer this question differently.
  if (!data?.points?.length) return null;
  const logY = data.y.scale === "log";
  // A zero has no place on a log axis; finance has 26. Said, not silently clipped.
  const offAxis = logY ? data.points.filter((p) => p[1] <= 0).length : 0;
  const kind = data.corpus_is_synthetic ? "simulated" : "real";
  const models = data.models?.length
    ? <>{data.models.map((m, i) => <span key={m}>{i ? ", " : ""}<code>{m}</code></span>)}</>
    : null;

  // Built as a value and wrapped at the end, NOT with a wrapper component declared here: a component
  // defined during render has a new identity on every render, so React would unmount and remount the
  // whole chart each time and Recharts would re-measure from zero. The empty-card fix must not cost
  // the thing it frames.
  const figure = (
    <figure className="chart-figure">
      <figcaption id={captionId} className="muted small">
        {data.is_training_corpus ? (
          <>
            {/* "The corpus X trains on: N points" read as if these rows were the training rows; they are
                drawn from that corpus and, where `split_note` says so, from rows no model was fitted on. */}
            <strong>From the corpus {models ?? "this page's published models"}{" "}
            {data.models && data.models.length > 1 ? "train" : "trains"} on</strong>:{" "}
            {data.points.length} {kind} points, coloured by TRUE class - no model&apos;s decision is
            drawn here{data.corpus_is_synthetic
              ? ", and the rows are simulator output, so nothing here is a claim about a live system"
              : ""}. {data.provenance}
          </>
        ) : (
          <>
            {/* The specific reason a substitute was needed differs per use case and lives in the
                payload. Medical's substitute is a different MODALITY (morphometry vs ultrasound);
                genomics' is the same modality from a different COHORT. Hardcoding one phrase here
                made the genomics caption claim a modality difference that does not exist. */}
            <strong>NOT the published models&apos; training corpus</strong>: {data.points.length}{" "}
            {kind} points, coloured by TRUE class. {data.provenance}
          </>
        )}{" "}
        {data.split_note ? <>{data.split_note}{" "}</> : null}
        {/* THE NUMBERS, NOT A VERDICT ON THEM. This used to append "so the model's decisions are
            being drawn on data that is genuinely separable - chosen by scoring every feature pair,
            not by eye", and both halves failed on a payload that rendered it: finance's own note calls
            its figure a LOW number, and finance's pair was chosen for readability among the
            separable ones. The estimator is named per payload, because telecom's figure once came
            from a different one under the same sentence. */}
        {data.separability_auc_pr != null && (
          <>The two classes separate in this projection at AUC-PR {data.separability_auc_pr.toFixed(4)}
          {data.separability_method ? <> ({data.separability_method})</> : null} against a
          base rate of {data.base_rate.toFixed(4)} - a {(data.separability_auc_pr / data.base_rate).toFixed(2)}x
          lift. Read the note below for what that means on this corpus. </>
        )}
        {offAxis > 0 ? <>{offAxis} points with a {data.y.label} of 0 cannot sit on the log axis and
          are not drawn. </> : null}
        {/* `note` carries the finding - what the chart actually shows, computed by the generator
            from the same points. It was written and then not rendered, which is the quietest way to
            lose the most useful sentence in the payload. */}
        {data.note && <><br /><strong>What it shows:</strong> {data.note}<br /></>}
        <LicenceLine licence={{ name: data.licence, url: data.licence_url ?? null,
                                attribution: data.attribution, modifications: data.modifications ?? "" }} />
      </figcaption>
      {/* Axis labels sit INSIDE the plot margin and the legend is given its own reserved height -
          at the default the x-axis title and the legend row landed on the same pixels. */}
      <ChartReadout hint={HINT_POINTER}>
        <MarksHidden>
        <ResponsiveContainer width="100%" height={230}>
          {/* `role="img"` with a title, NOT `accessibilityLayer`: on a scatter the layer makes the
              surface focusable without making its points keyboard reachable, because recharts walks
              an activeIndex along a categorical axis and two numeric axes have none (measured on the
              deployed page). The figcaption above, tied by `aria-describedby`, is the text
              alternative: what is drawn, how many, its licence, link and changes. */}
          <ScatterChart role="img"
            title={`Scatter plot: ${data.y.label} against ${data.x.label}`}
            aria-describedby={captionId}
            margin={{ top: 6, right: 10, bottom: 30, left: -14 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis type="number" dataKey="x" name={data.x.label} tick={AXIS}
                   label={{ value: data.x.label, position: "insideBottom", offset: -20,
                            fill: "#8a94a6", fontSize: 12 }} />
            {/* Log scale is carried in the payload, not inferred here. A heavy-tailed axis - finance's
                burst fan-out runs to 12,540 with a 95th percentile of 7,801 - renders every point on
                the baseline unless it is compressed, and whether that is warranted is a property of
                the DATA, so the generator that measured it is the thing that should decide. */}
            {/* Compact log ticks (Round 80): "10000" ran under the rotated axis title. */}
            <YAxis type="number" dataKey="y" name={data.y.label} tick={AXIS} width={Y_AXIS_W}
                   tickFormatter={logY ? compactTick : undefined}
                   scale={logY ? "log" : "auto"}
                   domain={logY ? [0.9, "auto"] : undefined}
                   allowDataOverflow={logY}
                   label={yAxisTitle(data.y.label + (logY ? " (log)" : ""), 24)} />
            <Legend verticalAlign="top" align="right" height={18} wrapperStyle={{ fontSize: 10 }} />
            {/* The points are on the canvas (Round 85); these keep the legend, the corners the domain. */}
            <Scatter name={data.class_names[0]} data={NO_POINTS} fill={MARK_SLATE} fillOpacity={0.75}
                     shape="circle" isAnimationActive={false} />
            <Scatter name={data.class_names[1]} data={NO_POINTS} fill={MARK_ORANGE} fillOpacity={0.85}
                     shape="triangle" isAnimationActive={false} />
            <Scatter data={corners(marks)} fill={MARK_SLATE} shape={NO_MARK} legendType="none"
                     isAnimationActive={false} />
            <Customized component={<CanvasMarks series={marks} xName={data.x.label} yName={data.y.label}
                                                format={(v: number) => v.toLocaleString()} />} />
          </ScatterChart>
        </ResponsiveContainer>
        </MarksHidden>
      </ChartReadout>
    </figure>
  );

  return card ? <div className="card" style={{ marginTop: 16 }}>{figure}</div> : figure;
}

// The genomics model is the one published model with no browser-runnable ONNX graph - it is a dot
// product against published effect weights, so nothing above fires for it and the page rendered its
// result as a flat table of ~20 `by_super_population.*` rows. The registry note for that model says
// "Read those rows, not this sentence", and twenty rows is exactly where a finding goes to hide.
//
// What the rows say: ONE globally-set high-risk cut (the cohort P90, so it is built to flag 10%)
// flags a wildly different share of each genetic ancestry group. That is the PRS portability
// problem - Martin et al., Nat Genet 51:584 (2019) - and it is a distribution shift, NOT an accuracy
// drop: 1000 Genomes is a reference panel with no case/control status, so no AUROC exists here and
// the trainer records auroc=null rather than inventing one.
//
// The reference line is the POOLED rate recomputed from the group counts, not a literal 0.10. The
// cut is defined as a percentile, so the pooled share is what it actually flagged on this cohort;
// hardcoding the nominal figure would let the line drift away from the bars it is drawn against.
type Strata = { n: number; mean: number; sd: number; p90: number; frac_above_global_cut: number };

export function AncestryStrataChart({ achieved }: { achieved: Record<string, unknown> | null }) {
  if (!achieved) return null;
  const measured = (achieved.measured ?? achieved) as Record<string, unknown>;
  const groups = measured.by_super_population as Record<string, Strata> | undefined;
  if (!groups || Object.keys(groups).length < 2) return null;

  const rows = Object.entries(groups)
    .map(([label, v]) => ({ label, frac: v.frac_above_global_cut, mean: v.mean, n: v.n }))
    .sort((a, b) => b.frac - a.frac);
  const totalN = rows.reduce((a, r) => a + r.n, 0);
  if (!totalN) return null;
  const pooled = rows.reduce((a, r) => a + r.n * r.frac, 0) / totalN;
  const hi = rows[0], lo = rows[rows.length - 1];
  const ratio = lo.frac > 0 ? hi.frac / lo.frac : null;
  const cut = typeof measured.global_high_risk_cut === "number" ? measured.global_high_risk_cut : null;

  return (
    <figure className="chart-figure">
      <figcaption className="muted small">
        Share of each genetic-ancestry group flagged high-risk by <strong>one global cut</strong>
        {cut != null ? <> (PRS &ge; {cut.toFixed(3)})</> : null}, over {totalN} samples. The dashed
        line is what that cut flagged pooled ({(pooled * 100).toFixed(1)}%) - every bar above it is a
        group over-flagged relative to the threshold's own intent.{" "}
        {ratio ? <><strong>{hi.label}</strong> is flagged at {(hi.frac * 100).toFixed(1)}% against{" "}
          <strong>{lo.label}</strong> at {(lo.frac * 100).toFixed(1)}%, a {ratio.toFixed(1)}x gap</> : null}
        {/* "driven by allele-frequency differences rather than by risk" until Round 86: plausible, and
            measured nowhere here. Frequency and LD differences can produce such a gap on their own
            (Martin et al., Nat Genet 51:584, 2019); with no case/control status the panel cannot split it. */}
        {ratio ? ". " : " "}Differences in allele frequency and linkage disequilibrium between groups can
        produce such a gap on their own (Martin et al., Nat Genet 2019); with no case/control status on this
        panel, this page cannot say how much of it is risk. This is a distribution shift, not an accuracy
        drop, and no AUROC is computable here for the same reason.
      </figcaption>
      <ChartReadout hint={HINT_KEYBOARD}>
        <ResponsiveContainer width="100%" height={190}>
          <BarChart accessibilityLayer title="Share flagged high-risk, by genetic-ancestry group"
            data={rows} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis dataKey="label" tick={AXIS} interval={0} />
            <YAxis tick={AXIS} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip content={<ReadoutTooltip />} formatter={(v: number, name: string) =>
                       name === "flagged" ? `${(v * 100).toFixed(2)}%` : v.toFixed(4)}
                     labelFormatter={(l: string) =>
                       `${l} — n=${rows.find((r) => r.label === l)?.n ?? 0}, `
                       + `mean PRS ${rows.find((r) => r.label === l)?.mean.toFixed(4) ?? ""}`} />
            <ReferenceLine y={pooled} stroke="#c05621" strokeDasharray="4 4" />
            <Bar dataKey="frac" fill="#4f9dff" name="flagged" />
          </BarChart>
        </ResponsiveContainer>
      </ChartReadout>
    </figure>
  );
}

/** Predicted probability against observed frequency. Both axes are probabilities, so `y = x` is
 *  perfect calibration and the deviation is readable without a legend.
 *
 *  This is the one chart here that compares a PREDICTION against the TEST data on a shared axis. The
 *  others answer different questions: `pr` is ranking, `hist` is separability, `perClass` is where
 *  the aggregate hides a weak class. None of them can see whether a probability means what it says,
 *  and that is a property this project has already been caught getting wrong - Round 50 found the
 *  finance calibrator emitting one constant across nine deciles while Brier IMPROVED, so every
 *  metric on the page waved it through.
 *
 *  Deliberately NOT drawn as a prediction-versus-label scatter, which is what the request's wording
 *  suggests: for a classifier the label axis is categorical, so that plot is a confusion matrix drawn
 *  badly. Probability against frequency is the same comparison in the form the data actually takes.
 */
function ReliabilityChart(
  { data, of, from, modelKey }: {
    data: Array<{ p: number; observed: number; n: number }>; of: string;
    from?: "probabilities" | "logits"; modelKey: string;
  },
) {
  const captionId = useId();          // before the early return: a hook must run on every render
  if (data.length < 2) return null;   // one bin is a point, and a point has no calibration to show
  const worst = data.reduce((a, b) =>
    Math.abs(b.observed - b.p) > Math.abs(a.observed - a.p) ? b : a);
  return (
    <figure className="chart-figure">
      <figcaption id={captionId} className="muted small">
        Calibration of <code>{modelKey}</code>: mean {of} against the frequency actually observed in
        the same bin, over {res_n(data)} rows. The grey diagonal is perfect calibration. Widest
        deviation {Math.abs(worst.observed - worst.p).toFixed(3)} at predicted {worst.p.toFixed(3)}{" "}
        {/* `{" "}` above: a line break after a value is dropped, and this read "0.644against" on the
            deployed page until Round 74 (web/scripts/check_jsx_text_spaces.mjs). */}
        against observed {worst.observed.toFixed(3)} ({worst.n} row{worst.n === 1 ? "" : "s"}). Bins
        are equal-WIDTH, so a calibrator that collapses its output onto one value shows up as a single
        tall bin rather than being spread across the axis.
        {/* Said, because the model does not emit these numbers itself: the medical pages binned the
            raw logits as probabilities until Round 74 (lib/calibration.ts). */}
        {from === "logits" && <> The graph emits logits, so the probabilities binned here are their
          softmax.</>}
      </figcaption>
      <ChartReadout hint={HINT_POINTER}>
        <MarksHidden>
        <ResponsiveContainer width="100%" height={180}>
          {/* `role="img"` and NOT `accessibilityLayer`, which is what the other scatters here do and
              what `tests/test_chart_accessibility.py` enforces: on a ScatterChart the accessibility
              layer sets role="application" with tabindex=0 and no keyboard path, because recharts
              walks an activeIndex along a categorical axis and this chart has none. That takes a
              screen-reader user out of browse mode and hands them a widget that ignores every key.
              The figcaption above is the text alternative, and it carries the numbers. */}
          <ScatterChart role="img" title={`Calibration of ${modelKey}`} aria-describedby={captionId}
            margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis type="number" dataKey="p" domain={[0, 1]} tick={AXIS} name="predicted"
                   tickFormatter={(v: number) => v.toFixed(1)} />
            <YAxis type="number" dataKey="observed" domain={[0, 1]} tick={AXIS} name="observed" />
            <Tooltip content={<ReadoutTooltip />} formatter={(v: number) => Number(v).toFixed(4)} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#8a94a6"
                           strokeDasharray="4 4" />
            <Scatter data={data} fill="#4f9dff" isAnimationActive={false} name="bin" />
          </ScatterChart>
        </ResponsiveContainer>
        </MarksHidden>
      </ChartReadout>
    </figure>
  );
}

const res_n = (d: Array<{ n: number }>) => d.reduce((a, b) => a + b.n, 0);

/** Net benefit across the declared thresholds, against treat-all and treat-none (Vickers and Elkin 2006; Round 86).
 *
 *  `lib/decisionCurve.ts` computes everything drawn and quoted here. Per 10,000 transactions, because at finance's
 *  1-in-656 window rate a whole-row net benefit is a number like 0.0004, whose leading zeros are the hardest part of
 *  it to read. The y axis stops a little below zero and treat-all falls past it, as in Vickers' own figures: above the
 *  prevalence, acting on everyone has a net benefit far below zero, and an axis stretched to hold it would flatten
 *  the model's curve into the zero line. The readout still gives treat-all's value. */
function DecisionCurveChart({ curve, modelKey, n, scored = "graph" }: {
  curve: DecisionCurve; modelKey: string; n: number;
  /** What was scored: the published graph's own output on its evaluation sample, or the logits decrypted above. */
  scored?: "graph" | "decrypted";
}) {
  const captionId = useId();
  const PER = 10_000;
  const data = curve.points.map((p) => ({
    t: p.t, model: p.model * PER, lo: p.lo * PER, hi: p.hi * PER, treatAll: p.treatAll * PER,
  }));
  const ceiling = curve.prevalence * PER;
  // Down to minus the ceiling, so treat-all is seen leaving the axis rather than not drawn at all: at a rate below
  // every threshold here it starts under zero and falls fast.
  const yMin = Math.min(-ceiling, ...data.map((d) => d.lo)) - 0.05 * ceiling;
  const yMax = Math.max(ceiling, ...data.map((d) => d.hi)) + 0.05 * ceiling;
  const fmtT = (t: number) => (t < 0.01 ? t.toFixed(3) : t.toFixed(2));
  const count = (x: number) => x.toLocaleString("en-US");
  const per = (x: number) => (x * PER).toFixed(2);
  const pct = (x: number) => `${(100 * x).toFixed(x < 0.01 ? 3 : 1)}%`;
  const cov = Math.round(curve.coverage * 100);
  // `of them` once the first clause has named the thresholds, so the caption does not say the same phrase twice.
  const where = (rows: typeof data, again = false) => {
    const span = rows.length > 0 && rows.length < data.length
      ? `, between t = ${fmtT(rows[0].t)} and ${fmtT(rows[rows.length - 1].t)}` : "";
    if (rows.length === data.length) return again ? "at all of them" : "at every threshold drawn";
    if (rows.length === 0) return again ? "at none of them" : "at no threshold drawn";
    return (again ? `at ${rows.length} of them` : `at ${rows.length} of the ${data.length} thresholds drawn`) + span;
  };
  const above = data.filter((d) => d.model > 0), clear = data.filter((d) => d.lo > 0), below = data.filter((d) => d.model < 0);
  const allPays = data.filter((d) => d.treatAll > 0);
  const m = curve.atMedian, op = curve.operating;
  const tMax = curve.thresholds[curve.thresholds.length - 1];
  // Ticks at whole steps of a round size: recharts' own, on this hand-set domain, came out at -2, 3, 8 and 16.
  const step = [1, 2, 5, 10, 20, 50].find((s) => (yMax - yMin) / s <= 7) ?? 100;
  const gone = data.find((d) => d.treatAll < yMin);
  const yTicks: number[] = [];
  for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) yTicks.push(v);
  return (
    <figure className="chart-figure">
      <figcaption id={captionId} className="muted small">
        Decision curve for <code>{modelKey}</code> over the {count(n)}{" "}
        {scored === "decrypted" ? "rows above, from their decrypted logits" : "rows of its evaluation sample"}: net
        benefit per 10,000 transactions of acting when the calibrated risk is at least t, each false alarm priced at
        t/(1 &minus; t) of a caught case (Vickers and Elkin 2006). The model is above treat-none, the zero
        line, {where(above)}, and its {cov}% interval (dotted) clears zero {where(clear, true)}.{below.length
          ? ` It is below zero ${where(below, true)}: there the flagged rows' laundering rate is under t, so acting on them costs more than it catches.`
          : ""} Treat-all
        (dashed) is {allPays.length ? `above zero ${where(allPays)}` : "below zero across the range"}{gone
          ? `, and leaves the axis by t = ${gone.t.toFixed(4)}` : ""}: acting
        on every transaction pays only at a threshold under the prevalence. Read a value against its
        threshold, not as one number.
      </figcaption>
      <ChartReadout hint={HINT_KEYBOARD}>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart accessibilityLayer title={`Decision curve for ${modelKey}`} aria-describedby={captionId}
            data={data} margin={{ top: 6, right: 10, bottom: 14, left: 4 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis dataKey="t" type="number" scale="log" domain={[data[0].t, data[data.length - 1].t]}
                   ticks={curve.thresholds} tick={AXIS} tickFormatter={fmtT}
                   label={{ value: "threshold probability t (log scale)", position: "insideBottom",
                            offset: -10, fill: "#8a94a6", fontSize: 12 }} />
            <YAxis type="number" domain={[yMin, yMax]} allowDataOverflow tick={AXIS} width={Y_AXIS_W}
                   ticks={yTicks} tickFormatter={(v: number) => String(v)}
                   label={yAxisTitle("net benefit per 10,000", 14)} />
            <Tooltip content={<ReadoutTooltip />} formatter={(v: number) => Number(v).toFixed(2)}
                     labelFormatter={(v: number) => `t = ${Number(v).toFixed(4)}`} />
            {/* At the top: at the bottom it sat on the x axis's title. */}
            <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: 10 }} />
            <ReferenceLine y={0} stroke="#8a94a6" strokeDasharray="4 4" />
            {/* Dash pattern as well as colour (SC 1.4.1), named in each legend entry. */}
            <Line dataKey="treatAll" stroke={MARK_AMBER} strokeDasharray="6 3" strokeWidth={1.4} dot={false}
                  isAnimationActive={false} name="treat all (dashed)" />
            <Line dataKey="lo" stroke={MARK_BLUE} strokeDasharray="1 3" strokeWidth={1} dot={false}
                  isAnimationActive={false} name={`${cov}% interval (dotted)`} />
            <Line dataKey="hi" stroke={MARK_BLUE} strokeDasharray="1 3" strokeWidth={1} dot={false}
                  isAnimationActive={false} legendType="none" name={`${cov}% interval, upper`} />
            <Line dataKey="model" stroke={MARK_BLUE} strokeWidth={1.8} dot={false} isAnimationActive={false}
                  name="model" />
          </LineChart>
        </ResponsiveContainer>
      </ChartReadout>
      <p className="muted small">
        Why the window&apos;s rate: the {count(curve.events)} laundering and {count(curve.nonEvents)} other rows were drawn
        from each class of the test window on its own, so their true- and false-positive rates are unbiased and
        their {pct(curve.events / n)} is not. The curve weights them by the window&apos;s own{" "}
        {count(curve.windowEvents)} of {count(curve.windowRows)} ({pct(curve.prevalence)}), the case-control estimator of Pfeiffer and Gail (2020,
        eq. 11): at t = {m.t} it reads {per(m.corrected)} per 10,000, where the sample&apos;s own rate would
        read {per(m.naive)}. The interval is pointwise, for the rows drawn. Scores pass through the
        record&apos;s Platt map first, because t is a risk{op ? (scored === "decrypted"
          ? <>; the encrypted path classifies where the decrypted logit is 0 (the trainer folded its tuned
              threshold into the bias), a calibrated risk of {op.risk.toFixed(3)}{op.risk > tMax ? ", above this range" : ""}</>
          : <>; this page classifies at {op.raw.toFixed(4)} on the graph&apos;s own output, a calibrated risk
              of {op.risk.toFixed(3)}{op.risk > tMax ? ", above this range" : ""}</>)
          : null}. A perfect model would score {ceiling.toFixed(2)}, the window&apos;s rate.
        {/* "in the model panel" until Round 86, which /orchestration does not have: this figure renders on
            both surfaces, and the teacher's curve is drawn on the finance use-case page only. */}
        {scored === "decrypted" ? <> The teacher&apos;s curve, in the model panel of the finance use-case page, is
          drawn over the same thresholds at the same rate, so the two differ by the model and its map, not by
          the decision context.</> : null}
      </p>
    </figure>
  );
}

/** Why there is no encrypted-vs-plaintext chart for this run.
 *
 *  ONE COPY, used by both surfaces. `/orchestration` carried this explanation and the satellite
 *  USE-CASE page rendered `null`, so a visitor comparing satellite against finance saw a missing
 *  chart on one page and a reason on the other. An absence that is explained in one place and silent
 *  in another is worse than either, because the silent page reads as an oversight.
 */
export function NoEncryptedComparison() {
  return (
    <div className="card">
      <h3 style={{ margin: "0 0 4px" }}>Private inference against plaintext, this run</h3>
      <p className="muted" style={{ fontSize: 12, margin: 0 }}>
        This flow produced no encrypted-vs-plaintext comparison. The chart is drawn from
        <code> encrypted_agreement_plot</code>, which only a stage that scores rows under CKKS emits
        {" "}- this use case either has no such stage, or, like the generic demo, has no bundled
        corpus for it to score - and a stopped or failed run has not reached one. Nothing is missing; there is nothing to
        plot.
      </p>
    </div>
  );
}

/** Measured against published, on ONE metric and ONE axis.
 *
 *  WHY THIS IS A SEPARATE CHART AND NOT A REFERENCE LINE. Round 54 drew `MetricSpec.baseline` as a
 *  line on the per-class-F1 chart. Measured afterwards on the deployed site, it rendered NOWHERE, and
 *  the reason is structural rather than a wiring slip: every model carrying a published baseline
 *  (finance 0.6323, medical 0.901, satellite 0.968/0.979) plots as a PR curve, and every model that
 *  plots per-class F1 is a telecom multiclass model whose baseline is deliberately `null`. The
 *  intersection of "has a baseline" and "draws that chart" is empty.
 *
 *  Moving the line to the PR-curve branch would have been worse, not better: that y-axis is
 *  PRECISION, and a `minority_class_f1` of 0.6323 is not a precision value. A horizontal line there
 *  compares an F1 against a precision - the same class of error as drawing the trivial predictor as
 *  prior research, which the gate for that field exists to prevent.
 *
 *  So the comparison gets its own axis, where both bars are the same quantity by construction and the
 *  metric is named on it - EXCEPT where the registry says otherwise, which is the `caveat` prop. */
export function MeasuredAgainstPublished(
  { measured, published, ceiling, metric, modelKey, source, caveat }:
  { measured: number | null; published: number | null; ceiling?: number | null;
    metric: string; modelKey: string; source?: string; caveat?: string | null },
) {
  if (measured == null || published == null) return null;
  // THE SAME METRIC NAME IS NOT THE SAME MEASUREMENT, and this chart shipped asserting that it is.
  //
  // The caption claimed "both bars are the same metric on the same axis" unconditionally. For
  // `satellite_comms.telemanom` that claim is FALSE, and the registry already said so in its own
  // words: the scored window holds THREE labelled events against ESA-ADB's 65, the split differs
  // (last 30% here, half the mission there), and ESA-ADB multiplies event precision by the TIME-LEVEL
  // true-negative rate. Its `sample_size_note` states outright that the figure "is NOT comparable to a
  // published benchmark computed over the full event set" - and this chart drew 0.4545 beside 0.968
  // anyway, under a caption promising the opposite. A chart built to prevent a category error
  // committed one, on the single model whose registry entry warns about it three times.
  //
  // The caveat is an EXPLICIT REGISTRY FIELD rather than a heuristic. The first draft of this used
  // "does the model carry a `sample_size_note`" as the signal, which is wrong: `telecom_ids.nidd`
  // carries one too and its note is a REASSURANCE - stratified 75/25, every class at corpus
  // proportion. One field name, two opposite meanings, which is the same two-fields-one-word trap
  // `tests/test_two_baselines_are_not_one.py` exists for.
  const who = caveat ? "published*" : "published";
  const data = [
    { who: "this project", value: measured },
    { who, value: published },
    ...(ceiling != null ? [{ who: "ceiling", value: ceiling }] : []),
  ];
  return (
    <figure className="chart-figure">
      <figcaption className="muted small">
        <code>{metric}</code> for <code>{modelKey}</code>: this project&apos;s recorded figure, from
        training rather than from this tab, against the published one.
        {caveat
          ? <> The two bars share a metric <b>name</b> and are <b>not</b> the same measurement. Read
              the note under the chart before comparing their heights.</>
          : <> Both bars are the same metric on the same axis - the comparison is only meaningful
              because of that, and this project has twice shipped a plot where it was not true. The
              splits and protocols behind the two can still differ; the baseline&apos;s source says how.</>}
        {source ? <> Baseline: {source}</> : null}
      </figcaption>
      <ChartReadout hint={HINT_KEYBOARD}>
        <ResponsiveContainer width="100%" height={150}>
          <BarChart accessibilityLayer title={`${metric} measured against published for ${modelKey}`}
            data={data} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis dataKey="who" tick={AXIS} />
            <YAxis domain={[0, 1]} tick={AXIS} />
            <Tooltip content={<ReadoutTooltip />} formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="value" name={metric} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.who}
                      fill={d.who === "this project" ? "#4f9dff"
                          : d.who.startsWith("published") ? "#e6a23c" : "#8a94a6"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartReadout>
      {caveat ? (
        <p className="muted small" style={{ marginTop: 6 }}>
          <b>* not a like-for-like comparison.</b> {caveat}
        </p>
      ) : null}
    </figure>
  );
}

export function VerificationChart(
  { res, modelKey, usecase }: { res: ModelEvalResult; modelKey: string; usecase?: string },
) {
  const plot = res.plot;

  // THE PUBLISHED BASELINE, and it is NOT `res.baseline`.
  //
  // `ModelEvalResult.baseline` is the TRIVIAL predictor - majority-class accuracy, all-positive F1,
  // persistence MAE. `MetricSpec.baseline` is prior published research. Two fields, one word, and
  // drawing the wrong one would render a published 0.901 as a chance line. They are looked up from
  // different objects here on purpose, and `tests/test_two_baselines_are_not_one.py` pins them apart.
  //
  // Drawn only when the registry's DECLARED metric is the quantity the chart plots. A macro F1
  // baseline over a per-class F1 axis is an aggregate against its own components - which is
  // informative (it shows which classes fall below the published aggregate) but must say so, so the
  // label carries the metric name rather than the word "baseline" alone.
  const spec = usecase ? modelsFor(usecase).find((m) => m.key === modelKey) : undefined;
  const prior = spec?.metric?.baseline ?? null;
  const priorName = spec?.metric?.name ?? "";
  if (!plot) return null;

  if (plot.kind === "per-class-f1" && plot.perClass?.length) {
    const worst = plot.perClass.reduce((a, b) => (b.f1 < a.f1 ? b : a));
    return (
      <div className="chart-pair">
      <figure className="chart-figure">
        <figcaption className="muted small">
          Per-class F1 for <code>{modelKey}</code>, {res.n} rows of its published evaluation sample
          ({res.split ?? "split not declared"}) scored in this tab.
          Weakest class: <strong>{worst.label}</strong> at {worst.f1.toFixed(3)} against an aggregate
          of {res.score?.toFixed(3)}. The aggregate is what would hide it.
        </figcaption>
        <ChartReadout hint={HINT_KEYBOARD}>
          {/* Horizontal bars (Round 80). The class names were x-axis ticks at -25 degrees: the longest,
              "photon_number_splitting", ran below the chart and lost its first letter, and the eight
              sat edge to edge. Laid out down the side they read level, whatever their length. */}
          <ResponsiveContainer width="100%" height={Math.max(120, 26 * plot.perClass.length + 34)}>
            <BarChart accessibilityLayer title={`Per-class F1 for ${modelKey}`} layout="vertical"
              data={plot.perClass} margin={{ top: 4, right: 12, bottom: 4, left: 4 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
              <XAxis type="number" domain={[0, 1]} tick={AXIS} />
              <YAxis type="category" dataKey="label" tick={AXIS} interval={0} width={150} />
              <Tooltip content={<ReadoutTooltip />} formatter={(v: number) => v.toFixed(4)} />
              <Bar dataKey="f1" fill="#4f9dff" name="F1" />
            </BarChart>
          </ResponsiveContainer>
        </ChartReadout>
      </figure>
      {plot.reliability?.length ? (
        <ReliabilityChart data={plot.reliability} of={plot.reliabilityOf ?? "predicted probability"}
                          from={plot.reliabilityFrom}
                          modelKey={modelKey} />
      ) : plot.reliabilityWithheld ? (
        <p className="muted small">No calibration chart: {plot.reliabilityWithheld}.</p>
      ) : null}
      </div>
    );
  }

  if (plot.kind === "pr-curve" && plot.pr?.length) {
    // The precision a coin-flip achieves is the prevalence, and at 1-in-981 that is 0.001. Drawing
    // it is what makes a PR curve legible - without the line, 0.5 precision looks unimpressive.
    const prevalence = plot.hist
      ? plot.hist.reduce((a, b) => a + b.positive, 0) /
        Math.max(1, plot.hist.reduce((a, b) => a + b.positive + b.negative, 0))
      : 0;
    // The sample's own names for the two classes, when it ships them (C5): "positive" is class 1 here, which on
    // BreastMNIST means "normal, benign".
    const [neg, pos] = plot.classLabels ?? [null, null];
    return (
      <div className="chart-pair">
        <figure className="chart-figure">
          <figcaption className="muted small">
            Precision-recall for <code>{modelKey}</code> over {res.n} rows of its published evaluation
            sample ({res.split ?? "split not declared"}).
            {pos ? <> The positive class is class 1, <strong>{pos}</strong>{neg ? <>; class 0 is {neg}</> : null}.</>
              : null} The dashed line is what a
            random ranker achieves ({prevalence.toFixed(4)}) - PR rather than ROC because this
            baseline moves with prevalence and ROC&apos;s does not.
          </figcaption>
          <ChartReadout hint={HINT_KEYBOARD}>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart accessibilityLayer title={`Precision-recall curve for ${modelKey}`}
                data={plot.pr} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={AXIS}
                       tickFormatter={(v: number) => v.toFixed(1)} />
                <YAxis domain={[0, 1]} tick={AXIS} />
                <Tooltip content={<ReadoutTooltip />} formatter={(v: number) => v.toFixed(4)}
                         labelFormatter={(v: number) => `recall ${Number(v).toFixed(3)}`} />
                <ReferenceLine y={prevalence} stroke="#c05621" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="precision" stroke="#4f9dff" dot={false} strokeWidth={1.8}
                      isAnimationActive={false} name="precision" />
              </LineChart>
            </ResponsiveContainer>
          </ChartReadout>
        </figure>

        {plot.minorityPr ? (
          <figure className="chart-figure">
            <figcaption className="muted small">
              Precision-recall with class 0{neg ? <>, <strong>{neg}</strong>,</> : null} as the positive: the minority
              class, {(100 * plot.minorityPr.prevalence).toFixed(1)}% of these rows, which is the class a PR curve is for.
              At the page&apos;s decision rule the model detects {plot.minorityPr.detected} of its{" "}
              {plot.minorityPr.total} rows, a sensitivity of{" "}
              {(plot.minorityPr.detected / plot.minorityPr.total).toFixed(3)}. The dashed line is its random ranker
              ({plot.minorityPr.prevalence.toFixed(4)}).
            </figcaption>
            <ChartReadout hint={HINT_KEYBOARD}>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart accessibilityLayer title={`Precision-recall curve for ${modelKey}, class 0 as the positive`}
                  data={plot.minorityPr.pr} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                  <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                  <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={AXIS}
                         tickFormatter={(v: number) => v.toFixed(1)} />
                  <YAxis domain={[0, 1]} tick={AXIS} />
                  <Tooltip content={<ReadoutTooltip />} formatter={(v: number) => v.toFixed(4)}
                           labelFormatter={(v: number) => `recall ${Number(v).toFixed(3)}`} />
                  <ReferenceLine y={plot.minorityPr.prevalence} stroke="#c05621" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="precision" stroke="#4f9dff" dot={false} strokeWidth={1.8}
                        isAnimationActive={false} name="precision" />
                </LineChart>
              </ResponsiveContainer>
            </ChartReadout>
          </figure>
        ) : null}

        {plot.hist?.length ? (
          <figure className="chart-figure">
            <figcaption className="muted small">
              Score distribution by TRUE class. Overlap here is the reason the curve above looks the
              way it does - a metric says how well, this says why.
            </figcaption>
            <ChartReadout hint={HINT_KEYBOARD}>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart accessibilityLayer title="Score distribution by true class"
                  data={plot.hist} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                  <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                  <XAxis dataKey="bin" tick={AXIS} label={{ value: "score bin", position: "insideBottom",
                         offset: -2, fill: "#8a94a6", fontSize: 12 }} />
                  <YAxis scale="log" domain={[0.7, "auto"]} allowDataOverflow tick={AXIS} />
                  <Tooltip content={<ReadoutTooltip labelName="score bin" />} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  {/* WCAG 2.2 SC 1.4.1 (Use of Color, Level A): these two series were separated by fill
                      alone until Round 34. Because they are STACKED they share an x position, so the axis
                      cannot disambiguate them either - colour really was the only channel, and the legend
                      swatches are colour too. A 45-degree hatch on the positive class is the second
                      channel; the fill stays, so colour and pattern are redundant rather than either one
                      being load-bearing. This is the same redundant-encoding approach the scatter chart
                      above already uses via `shape="circle"` / `shape="triangle"`, applied to a chart type
                      where Recharts offers no `shape` prop.
                      Why this survived: the checklist scoped itself to "WCAG 2.2 AA, the four that this UI
                      can actually fail" and 1.4.1 was not among the four, so no row looked for it. */}
                  <defs>
                    <pattern id="hatch-positive" patternUnits="userSpaceOnUse" width="5" height="5"
                             patternTransform="rotate(45)">
                      <rect width="5" height="5" fill="#c05621" />
                      <line x1="0" y1="0" x2="0" y2="5" stroke="#f5c199" strokeWidth="1.6" />
                    </pattern>
                  </defs>
                  {/* WCAG 2.2 SC 1.4.11 (AA), measured in Round 74 against the card (#151c2c): the
                      negative class was #4a5568 at 2.26:1, below the 3:1 a data mark needs. #717d92 is
                      4.09:1 and still reads as the muted series beside the hatched positive one.
                      The two segments are STACKED, so each also meets the other, and no grey reaches
                      3:1 against the card, the hatch base (#c05621) and the hatch lines (#f5c199) at
                      once: #717d92 is 1.10:1 against the base. The card-coloured stroke separates them,
                      the way WCAG's own pie-chart example uses a border, so each segment meets the
                      card at 3.72:1 or more. */}
                  <Bar dataKey="negative" stackId="a" fill="#717d92" stroke={CARD} strokeWidth={1}
                       name={`true class 0${neg ? `, ${neg}` : ""} (solid)`} />
                  <Bar dataKey="positive" stackId="a" fill="url(#hatch-positive)" stroke={CARD}
                       strokeWidth={1} name={`true class 1${pos ? `, ${pos}` : ""} (hatched)`} />
                </BarChart>
              </ResponsiveContainer>
            </ChartReadout>
          </figure>
        ) : null}
        {plot.reliability?.length ? (
          <ReliabilityChart data={plot.reliability} of={plot.reliabilityOf ?? "p(positive)"}
                            from={plot.reliabilityFrom}
                            modelKey={modelKey} />
        ) : plot.reliabilityWithheld ? (
          <p className="muted small">No calibration chart: {plot.reliabilityWithheld}.</p>
        ) : null}
        {plot.decisionCurve ? (
          <DecisionCurveChart curve={plot.decisionCurve} modelKey={modelKey} n={res.n} />
        ) : plot.decisionCurveWithheld ? (
          <p className="muted small">No decision curve: {plot.decisionCurveWithheld}.</p>
        ) : null}
      </div>
    );
  }

  if (plot.kind === "forecast" && plot.series?.length) {
    return (
      <figure className="chart-figure">
        <figcaption className="muted small">
          Channel 0, predicted against actual over {plot.series.length} scored windows for{" "}
          <code>{modelKey}</code>. MAE {res.score?.toFixed(4)} against a persistence baseline of{" "}
          {res.baseline?.toFixed(4)}
          {res.forecastDetail
            ? <>; median absolute error {res.forecastDetail.medianAe.toFixed(4)} against{" "}
                {res.forecastDetail.persistenceMedianAe.toFixed(4)}, and the lower mean error on{" "}
                {res.forecastDetail.windowsBetter} of {res.forecastDetail.windows} windows</>
            : null}. The error&apos;s SHAPE matters: uniform drift is a miscalibrated
          model, a few large misses are missed events.
        </figcaption>
        <ChartReadout hint={HINT_KEYBOARD}>
          <ResponsiveContainer width="100%" height={190}>
            <LineChart accessibilityLayer title="Forecast against actual, over time"
              data={plot.series} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
              <XAxis dataKey="i" tick={AXIS} />
              <YAxis tick={AXIS} />
              <Tooltip content={<ReadoutTooltip labelName="window" />} formatter={(v: number) => v.toFixed(4)} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {/* SC 1.4.1 again: these two were `stroke` colour only. A 1.4 vs 1.6 strokeWidth
                  difference is not a perceivable second channel, so `predicted` is dashed. Dashing the
                  PREDICTION rather than the actual is deliberate - the dashed line reads as the derived
                  quantity, which is what it is. */}
              <Line type="monotone" dataKey="actual" stroke="#8a94a6" dot={false} strokeWidth={1.4}
                    isAnimationActive={false} name="actual (solid)" />
              <Line type="monotone" dataKey="predicted" stroke="#4f9dff" dot={false} strokeWidth={1.6}
                    strokeDasharray="5 3" isAnimationActive={false} name="predicted (dashed)" />
            </LineChart>
          </ResponsiveContainer>
        </ChartReadout>
      </figure>
    );
  }

  return null;
}
