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
import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";

import type { ModelEvalResult } from "../lib/onnxInference";
import { modelsFor } from "../lib/config";
import Details from "./Details";

const AXIS = { fontSize: 10, stroke: "#8a94a6" };
const GRID = "#2a3040";
/** The card every chart in this file is drawn on (`--panel`). Stroked around stacked segments
 *  as a separator, so each segment meets this colour rather than its neighbour. */
const CARD = "#151c2c";

export interface EncryptedAgreementPlot {
  kind: "encrypted-agreement";
  points: { enc: number; plain: number; agree: boolean }[];
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
}

/**
 * The one figure this project's central claim needs, and the one it never drew.
 *
 * Every chart beside this one shows a PLAINTEXT model's quality, or the input corpus it was scored
 * on. Neither shows what private inference produced. The claim being made throughout the page is
 * that the encrypted computation reproduces the plaintext one - and until now that claim reached the
 * visitor as a scalar taken over a SINGLE row: `encrypted_top1_mismatches` could only be 0 or 1, and
 * `encrypted_scores_compared` counted classes rather than rows.
 *
 * Each point is one held-out row: its decrypted winning score against the plaintext score for the
 * same row and class. Agreement is the `y = x` line, so a deviation is visible as distance from it
 * rather than as a number someone has to interpret. Points are coloured by whether the encrypted
 * argmax matched the plaintext argmax - the integer convention arXiv:2605.22237 Table IX reports and
 * expects to be zero.
 *
 * The rows come from one CKKS ciphertext: `encryptedLinearBatch` packs them at the reduction's own
 * stride, so N rows cost the same homomorphic work as one.
 */
export function EncryptedAgreementChart({ plot }: { plot: EncryptedAgreementPlot }) {
  if (!plot?.points?.length) return null;
  const mismatches = plot.points.filter((p) => !p.agree).length;
  const worst = Math.max(...plot.points.map((p) => Math.abs(p.enc - p.plain)));
  const lo = Math.min(...plot.points.flatMap((p) => [p.enc, p.plain]));
  const hi = Math.max(...plot.points.flatMap((p) => [p.enc, p.plain]));
  const name = `Encrypted score against plaintext score, ${plot.rows} held-out rows`;
  return (
    <figure className="chart-figure">
      {/* A strip of the four numbers the caption below qualifies, for a reader scanning the page. The
          caption keeps every denominator and caveat beside the figure (checklist UX6); only the
          meaning of a flipped row's margin is folded. Round 58. */}
      <table className="facts-table small" style={{ marginBottom: 6 }}>
        <tbody><tr>
          <td>rows scored</td><td>{plot.rows}{plot.rowsOffered != null && plot.rowsOffered > plot.rows ? ` of ${plot.rowsOffered}` : ""}</td>
          <td>top-1 mismatches</td><td>{mismatches}</td>
          <td>plaintext ties</td><td>{plot.undecidableRows ?? "not reported"}</td>
          <td>worst deviation</td><td>{worst.toExponential(2)}</td>
        </tr></tbody>
      </table>
      <figcaption className="muted small">
        <strong>Private inference against the plaintext reference.</strong> Each point is one of{" "}
        {plot.rows} held-out rows scored under CKKS in this tab, plotted against the same row&apos;s
        plaintext score. The diagonal is exact agreement. Top-1 mismatches:{" "}
        <strong>{mismatches}</strong> of {plot.rows}. All {plot.rows} rows travelled in a single
        ciphertext, so they cost the homomorphic work of one.
        {/* THE CIPHERTEXT CAP, restored.
            `5d20d88` put this in the caption precisely so a reader could not take the agreement for
            the whole split, and `2393c79` - the commit that removed the tie verdict from this same
            sentence - deleted it as collateral: the removed block's closing `) : null}` was merged
            with the tie block's, so the cap clause went out with the verdict. `rowsOffered` was left
            declared and set by four producers and read by nothing, which no type check can see.
            Telecom is the page it matters on: 512 of 600 rows are scored, and without this the
            figure reports agreement 1.00 over a denominator it does not name. */}
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
            same conditional-disclosure defect this sentence exists to fix, one level down, and it
            is the third surface this round on which that gate had to be removed. */}
        {plot.undecidableRows != null ? (
          plot.undecidableRows > 0 ? (
            <>
              {" "}<strong>{plot.undecidableRows} of those {plot.rows} rows are scored identically
              across classes by the plaintext model itself</strong>, so its own top-1 there comes
              from argmax tie-breaking and the encrypted outcome is decided by encryption noise —
              they carry no signal either way, and the count above moves between runs because of
              them. Over the {plot.decidableRows} rows the model does decide, the encrypted score
              agrees with the plaintext one{" "}
              {/* A PROPORTION NEEDS ITS DENOMINATOR NAMED. This rendered as a bare `0.98` - the one
                  number on the page a practitioner would quote, printed as a dimensionless decimal
                  that reads equally well as a score, a probability or a ratio. `UX5`'s own worked
                  example is "the encrypted score agrees with the plaintext one 98.3% of the time",
                  and it was the sentence right beside the figure that did not say it.
                  One decimal place, not two: `toFixed(2)` on a fraction gave two significant
                  figures, so 0.9825 and 0.9849 both printed 0.98 over a few hundred rows. */}
              <strong>{plot.agreementOverDecidable != null
                ? `${(plot.agreementOverDecidable * 100).toFixed(1)}% of the time`
                : "by an amount that was not computed"}</strong>.
            </>
          ) : (
            <>
              {" "}<strong>None</strong> of these {plot.rows} rows is a tie in the plaintext model —
              it separates every one of them — so the figure above is over rows that carry a real
              decision, and the agreement is not diluted by any coin toss.
            </>
          )
        ) : null}
        {/* SCOPE, stated because the first version of this caption did not state it and was read
            wrong as a result. One point is plotted per row - the WINNING class only - so `worst`
            is a maximum over `rows` deviations. The metric row beside this figure
            (`encrypted_batch_max_abs_error`) is the maximum over `rows * classes`, a strictly
            larger set. On finance those two read 1.80e-8 and 2.47e-8: both correct, 37% apart,
            and the caption used to print the smaller one immediately after the phrase
            "(800 class scores)". A number whose scope is narrower than the sentence around it. */}
        {" "}Worst deviation among the {plot.rows} plotted winning-class scores:{" "}
        {worst.toExponential(2)}
        {typeof plot.maxAbsError === "number" ? (
          <> · across all {plot.rows * plot.classes} class scores, including the losing ones this
          figure does not plot: {plot.maxAbsError.toExponential(2)}</>
        ) : null}.
        {/* What a mismatch MEANS - stated, not adjudicated.
            Two earlier versions computed a tie/not-tie verdict here, first from the deviation and
            then from the score range. Both are unfalsifiable: a flip happens exactly when the noise
            exceeds the margin, so margin <= 2 x deviation always, and dividing by a score range of
            order 1 leaves it three orders below any threshold worth setting. The verdict restated a
            theorem while looking like a measurement.
            The number that actually varies, and that separates this page from telecom's, is the
            count of rows the plaintext model scores identically - reported above, unconditionally. */}
        {mismatches > 0 && typeof plot.flipMarginMax === "number" ? (
          <>
            {" "}Widest margin the plaintext model held at those {mismatches} flipped rows:{" "}
            {plot.flipMarginMax === 0
              ? <><strong>exactly zero</strong> — it does not distinguish those samples at all.</>
              : <>{plot.flipMarginMax.toExponential(2)}.</>}
          </>
        ) : null}
      </figcaption>
      {mismatches > 0 && typeof plot.flipMarginMax === "number" && plot.flipMarginMax > 0 ? (
        <Details summary="What a flipped row's margin means">
          A flip happens precisely when the deviation exceeds the margin, so the margin is always below
          twice the worst deviation; it is reported as a measurement, not as a verdict on the encryption.
        </Details>
      ) : null}
      <ResponsiveContainer width="100%" height={200}>
        {/* `title` as a PROP, the way every other chart in this file sets it. Written first as a
            `<title>` child, which recharts does not forward - so the figure rendered with a null
            accessible name, which is exactly the regression Round 34 recorded when six charts ended
            up with empty titles. Verified on the deployed page, not assumed.

            `role="img"`, NOT `accessibilityLayer`. The first version used the latter, twenty lines
            above the scatter that does not and the comment explaining why: `accessibilityLayer`
            makes a ScatterChart focusable and gives it `role="application"` WITHOUT making its
            points keyboard-reachable, because recharts steps an activeIndex along a categorical
            axis and two numeric axes have none. `KB6` says an application role appears only where
            the keyboard path passed, so this was an inert application role - a regression, in the
            same commit that fixed the missing title. Fixing one accessibility defect while
            introducing another is what happens when you verify the thing you changed instead of
            re-reading the row that governs the element. */}
        <ScatterChart role="img" title={name}
                      margin={{ top: 6, right: 10, bottom: 4, left: -10 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
          {/* Formatted, because the raw ticks rendered as `3.7326450091317454` - sixteen digits of a
              logit, on both axes. */}
          <XAxis type="number" dataKey="plain" name="plaintext" domain={[lo, hi]} tick={AXIS}
                 tickFormatter={(v: number) => v.toFixed(2)}
                 label={{ value: "plaintext score", position: "insideBottom", offset: -2, fill: "#8a94a6", fontSize: 10 }} />
          <YAxis type="number" dataKey="enc" name="encrypted" domain={[lo, hi]} tick={AXIS}
                 tickFormatter={(v: number) => v.toFixed(2)} />
          <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#8a94a6"
                         strokeDasharray="4 4" ifOverflow="extendDomain" />
          <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                   formatter={(v: number) => v.toFixed(6)} />
          {/* SHAPE as well as fill - SC 1.4.1, and the one chart in this file that was missed when
              the second channel was added to the others. `RealDataScatter` 170 lines below has
              carried `shape="circle"` / `shape="triangle"` since that sweep; the histogram, the
              forecast lines and the confusion bars all got a channel too. This one did not, and it
              is the chart where colour carries the most: red here means the ENCRYPTED path
              disagreed with plaintext, so a reader who cannot separate red from blue cannot see
              which points are the disagreements - the single fact the chart exists to show.
              The shape is named in the series name, so the legend, the tooltip and the figcaption
              all say it, the way the confusion bars already say "(solid)" and "(hatched)". */}
          {/* `isAnimationActive={false}` IS LOAD-BEARING, not tidiness, and leaving it off shipped
              a blank chart to the deployed site. Adding `shape` moves recharts off its default
              renderer onto `<Symbols>`, whose `size` is an ANIMATED prop; with animation left on,
              all 30 symbols stayed at their `from` state and rendered `d="M0,0"` - present in the
              DOM, correctly positioned, and zero-sized. The legend named two shapes above an empty
              plot area, which is worse than the colour-only version it replaced.
              `RealDataScatter` 170 lines below has carried `shape` AND `isAnimationActive={false}`
              together since Round 34. Half of that pattern was copied. Found only by looking at the
              deployed page - every gate was green, because a gate that reads the source sees the
              `shape` prop it asked for and cannot see a path of length zero. */}
          <Scatter name="agrees (circle)" data={plot.points.filter((p) => p.agree)}
                   fill="#4f9dff" shape="circle" isAnimationActive={false} />
          {mismatches > 0 && (
            <Scatter name="argmax differs (cross)" data={plot.points.filter((p) => !p.agree)}
                     fill="#ff6b6b" shape="cross" isAnimationActive={false} />
          )}
          {mismatches > 0 && <Legend wrapperStyle={{ fontSize: 10 }} />}
        </ScatterChart>
      </ResponsiveContainer>
    </figure>
  );
}

// The real corpus, drawn underneath the model's own numbers.
//
// A score on its own cannot distinguish a good model on easy data from a mediocre model on hard
// data - the reader has no way to see how separable the classes were to begin with. This draws the
// actual corpus so that question is answerable by looking.
//
// Two honesty constraints are built in rather than left to the caption writer:
//
//   1. `is_training_corpus` is carried in the data file and rendered differently. For satellite it
//      is TRUE - OPS-SAT-AD is genuinely what the shipped model trains on, so this is a real
//      train-versus-inference comparison. For medical it is FALSE and the panel says so loudly:
//      BreastMNIST cannot be redistributed (its CC BY attaches to the source article, not the
//      data), so WDBC stands in, and WDBC is a different cohort AND a different modality.
//   2. Substitute corpora are never drawn on the model's own axes. There is no shared feature
//      space between 28x28 ultrasound and cell-nucleus morphometry, so overlaying them would
//      assert a correspondence that does not exist.
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
  attribution: string;
  x: { key: string; label: string; scale?: "log" };
  y: { key: string; label: string; scale?: "log" };
  class_names: [string, string];
  separability_auc_pr?: number;
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
export function RealDataScatter({ usecase, card = false }: { usecase: string; card?: boolean }) {
  const [data, setData] = useState<Real2D | null>(null);
  useEffect(() => {
    const rel = REAL_2D_PATH[usecase];
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
  }, [usecase]);

  // The single decision point. Everything below - including the frame, when `card` is set - is
  // downstream of it, so there is no second place that can answer this question differently.
  if (!data?.points?.length) return null;
  const neg = data.points.filter((p) => p[2] === 0).map((p) => ({ x: p[0], y: p[1] }));
  const pos = data.points.filter((p) => p[2] === 1).map((p) => ({ x: p[0], y: p[1] }));

  // Built as a value and wrapped at the end, NOT with a wrapper component declared here: a component
  // defined during render has a new identity on every render, so React would unmount and remount the
  // whole chart each time and Recharts would re-measure from zero. The empty-card fix must not cost
  // the thing it frames.
  const figure = (
    <figure className="chart-figure">
      <figcaption className="muted small">
        {data.is_training_corpus ? (
          <>
            <strong>The corpus this model actually trains on</strong>, held-out split,{" "}
            {data.points.length} {data.corpus_is_synthetic ? "simulated" : "real"} points. This is a
            genuine train-versus-inference comparison rather than a lookalike{data.corpus_is_synthetic
              ? " - though the rows are simulator output, so nothing here is a claim about a live system"
              : ""}.
          </>
        ) : (
          <>
            {/* The specific reason a substitute was needed differs per use case and lives in the
                payload. Medical's substitute is a different MODALITY (morphometry vs ultrasound);
                genomics' is the same modality from a different COHORT. Hardcoding one phrase here
                made the genomics caption claim a modality difference that does not exist. */}
            <strong>NOT this model's training corpus</strong>, {data.points.length}{" "}
            {data.corpus_is_synthetic ? "simulated" : "real"} points - shown beside the
            model's scores, deliberately not on the same axes. {data.provenance}
          </>
        )}{" "}
        {/* THE NUMBERS, NOT A VERDICT ON THEM - and this is the third time this component has had to
            learn that; the first two are in the comments above.
            It used to append "so the model's decisions are being drawn on data that is genuinely
            separable - chosen by scoring every feature pair, not by eye." Both halves fail on at
            least one payload that renders it:
            - "genuinely separable" is contradicted four lines below by `note`, which the SAME
              figcaption prints. Finance's note says 0.2302 against a 0.02002 base rate "is a 11.5x
              lift and still a LOW number... two features do not solve AML"; telecom's says the lift
              is only 1.13x.
            - "not by eye" is false for finance and the generator says so in its own words. The
              AUC-PR winner is `format_risk x src_bank_diversity` at 0.3195 and was REJECTED as
              unreadable - "A tree does not care how values are spaced; an eye does" - so the
              shipped pair was chosen on readability among the separable ones. That is a defensible
              choice, honestly recorded, and the opposite of what the caption claimed.
            Written for satellite when it was the only payload carrying the field, then inherited by
            finance and telecom. `note` already carries the interpretation for all three, which is
            the remedy this component used twice before: keep the derived numbers here, let the
            payload speak for itself. */}
        {data.separability_auc_pr != null && (
          <>The two classes separate in this projection at AUC-PR {data.separability_auc_pr.toFixed(4)} against a
          base rate of {data.base_rate.toFixed(4)} - a {(data.separability_auc_pr / data.base_rate).toFixed(2)}x
          lift. Read the note below for what that means on this corpus. </>
        )}
        {/* `note` carries the finding - what the chart actually shows, computed by the generator
            from the same points. It was written and then not rendered, which is the quietest way to
            lose the most useful sentence in the payload. */}
        {data.note && <><br /><strong>What it shows:</strong> {data.note}<br /></>}
        <em>{data.licence}</em> - {data.attribution}
      </figcaption>
      {/* Axis labels sit INSIDE the plot margin and the legend is given its own reserved height -
          at the default the x-axis title and the legend row landed on the same pixels. */}
      <ResponsiveContainer width="100%" height={230}>
        {/* `accessibilityLayer` makes this focusable and DOES NOT make its points keyboard
            reachable. Measured on the deployed page, not inferred: focus the surface and press
            ArrowRight - the line and bar charts below open their tooltip with real values, this one
            does nothing. Recharts walks an activeIndex along a categorical axis, and a scatter with
            two numeric axes has no such index to step through, so there is nothing for the arrow
            keys to move between. Kept because focusability is still the prerequisite for any fix,
            and because removing it would hide the gap rather than close it. The text alternative
            for this chart is the figcaption above, which states what it shows, its licence and its
            attribution; that is what a screen-reader user gets today. */}
        <ScatterChart role="img"
          title={`Scatter plot: ${data.y.label} against ${data.x.label}`}
          margin={{ top: 6, right: 10, bottom: 30, left: -14 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
          <XAxis type="number" dataKey="x" name={data.x.label} tick={AXIS}
                 label={{ value: data.x.label, position: "insideBottom", offset: -20,
                          fill: "#8a94a6", fontSize: 10 }} />
          {/* Log scale is carried in the payload, not inferred here. A heavy-tailed axis - finance's
              burst fan-out runs to 12,540 with a 95th percentile of 7,801 - renders every point on
              the baseline unless it is compressed, and whether that is warranted is a property of
              the DATA, so the generator that measured it is the thing that should decide. */}
          <YAxis type="number" dataKey="y" name={data.y.label} tick={AXIS}
                 scale={data.y.scale === "log" ? "log" : "auto"}
                 domain={data.y.scale === "log" ? [0.9, "auto"] : undefined}
                 allowDataOverflow={data.y.scale === "log"}
                 label={{ value: data.y.label + (data.y.scale === "log" ? " (log)" : ""),
                          angle: -90, position: "insideLeft",
                          offset: 22, fill: "#8a94a6", fontSize: 10 }} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }}
                   contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                   formatter={(v: number) => v.toLocaleString()} />
          <Legend verticalAlign="top" align="right" height={18} wrapperStyle={{ fontSize: 10 }} />
          <Scatter name={data.class_names[0]} data={neg} fill="#8296b5" fillOpacity={0.75}
                   shape="circle" isAnimationActive={false} />
          <Scatter name={data.class_names[1]} data={pos} fill="#e07a3c" fillOpacity={0.85}
                   shape="triangle" isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
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
        {" "}driven by allele-frequency differences rather than by risk. This is a distribution shift,
        not an accuracy drop - the panel carries no case/control status, so no AUROC is computable.
      </figcaption>
      <ResponsiveContainer width="100%" height={190}>
        <BarChart accessibilityLayer title="Share flagged high-risk, by genetic-ancestry group"
          data={rows} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
          <XAxis dataKey="label" tick={AXIS} interval={0} />
          <YAxis tick={AXIS} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                   formatter={(v: number, name: string) =>
                     name === "flagged" ? `${(v * 100).toFixed(2)}%` : v.toFixed(4)}
                   labelFormatter={(l: string) =>
                     `${l} — n=${rows.find((r) => r.label === l)?.n ?? 0}, `
                     + `mean PRS ${rows.find((r) => r.label === l)?.mean.toFixed(4) ?? ""}`} />
          <ReferenceLine y={pooled} stroke="#c05621" strokeDasharray="4 4" />
          <Bar dataKey="frac" fill="#4f9dff" name="flagged" />
        </BarChart>
      </ResponsiveContainer>
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
  { data, of, modelKey }: {
    data: Array<{ p: number; observed: number; n: number }>; of: string; modelKey: string;
  },
) {
  if (data.length < 2) return null;   // one bin is a point, and a point has no calibration to show
  const worst = data.reduce((a, b) =>
    Math.abs(b.observed - b.p) > Math.abs(a.observed - a.p) ? b : a);
  return (
    <figure className="chart-figure">
      <figcaption className="muted small">
        Calibration of <code>{modelKey}</code>: mean {of} against the frequency actually observed in
        the same bin, over {res_n(data)} rows. The grey diagonal is perfect calibration. Widest
        deviation {Math.abs(worst.observed - worst.p).toFixed(3)} at predicted {worst.p.toFixed(3)}
        against observed {worst.observed.toFixed(3)} ({worst.n} rows). Bins are equal-WIDTH, so a
        calibrator that collapses its output onto one value shows up as a single tall bin rather than
        being spread across the axis.
      </figcaption>
      <ResponsiveContainer width="100%" height={180}>
        {/* `role="img"` and NOT `accessibilityLayer`, which is what the other scatters here do and
            what `tests/test_chart_accessibility.py` enforces: on a ScatterChart the accessibility
            layer sets role="application" with tabindex=0 and no keyboard path, because recharts
            walks an activeIndex along a categorical axis and this chart has none. That takes a
            screen-reader user out of browse mode and hands them a widget that ignores every key.
            The figcaption above is the text alternative, and it carries the numbers. */}
        <ScatterChart role="img" title={`Calibration of ${modelKey}`}
          margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
          <XAxis type="number" dataKey="p" domain={[0, 1]} tick={AXIS} name="predicted"
                 tickFormatter={(v: number) => v.toFixed(1)} />
          <YAxis type="number" dataKey="observed" domain={[0, 1]} tick={AXIS} name="observed" />
          <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                   formatter={(v: number) => Number(v).toFixed(4)} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#8a94a6"
                         strokeDasharray="4 4" />
          <Scatter data={data} fill="#4f9dff" isAnimationActive={false} name="bin" />
        </ScatterChart>
      </ResponsiveContainer>
    </figure>
  );
}

const res_n = (d: Array<{ n: number }>) => d.reduce((a, b) => a + b.n, 0);

/** Why there is no encrypted-vs-plaintext chart for this run.
 *
 *  ONE COPY, used by both surfaces. `/orchestration` carried this explanation and the satellite
 *  USE-CASE page rendered `null`, so a visitor comparing satellite against finance saw a missing
 *  chart on one page and a reason on the other. An absence that is explained in one place and silent
 *  in another is worse than either, because the silent page reads as an oversight.
 */
export function NoEncryptedComparison({ usecase }: { usecase?: string }) {
  return (
    <div className="card">
      <h3 style={{ margin: "0 0 4px" }}>Private inference against plaintext, this run</h3>
      <p className="muted" style={{ fontSize: 12, margin: 0 }}>
        This flow produced no encrypted-vs-plaintext comparison. The chart is drawn from
        <code> encrypted_agreement_plot</code>, which only a stage that scores rows under CKKS emits
        {usecase === "satellite_comms"
          ? <> - <code>satellite_comms</code> runs telemetry analysis and has no such stage</>
          : <> - this use case has no such stage in its flow</>}
        , and a stopped or failed run has not reached one. Nothing is missing; there is nothing to
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
        <code>{metric}</code> for <code>{modelKey}</code>: measured here against the published figure.
        {caveat
          ? <> The two bars share a metric <b>name</b> and are <b>not</b> the same measurement. Read
              the note under the chart before comparing their heights.</>
          : <> Both bars are the same metric on the same axis - the comparison is only meaningful
              because of that, and this project has twice shipped a plot where it was not true.</>}
        {source ? <> Baseline: {source}</> : null}
      </figcaption>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart accessibilityLayer title={`${metric} measured against published for ${modelKey}`}
          data={data} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
          <XAxis dataKey="who" tick={AXIS} />
          <YAxis domain={[0, 1]} tick={AXIS} />
          <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                   formatter={(v: number) => v.toFixed(4)} />
          <Bar dataKey="value" name={metric} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.who}
                    fill={d.who === "this project" ? "#4f9dff"
                        : d.who.startsWith("published") ? "#e6a23c" : "#8a94a6"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
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
        <ResponsiveContainer width="100%" height={180}>
          <BarChart accessibilityLayer title={`Per-class F1 for ${modelKey}`}
            data={plot.perClass} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis dataKey="label" tick={AXIS} interval={0} angle={-25} textAnchor="end" height={54} />
            <YAxis domain={[0, 1]} tick={AXIS} />
            <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                     formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="f1" fill="#4f9dff" name="F1" />
          </BarChart>
        </ResponsiveContainer>
      </figure>
      {plot.reliability?.length ? (
        <ReliabilityChart data={plot.reliability} of={plot.reliabilityOf ?? "predicted probability"}
                          modelKey={modelKey} />
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
    return (
      <div className="chart-pair">
        <figure className="chart-figure">
          <figcaption className="muted small">
            Precision-recall for <code>{modelKey}</code> over {res.n} rows of its published evaluation
            sample ({res.split ?? "split not declared"}). The dashed line is what a
            random ranker achieves ({prevalence.toFixed(4)}) - PR rather than ROC because this
            baseline moves with prevalence and ROC's does not.
          </figcaption>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart accessibilityLayer title={`Precision-recall curve for ${modelKey}`}
              data={plot.pr} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
              <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={AXIS}
                     tickFormatter={(v: number) => v.toFixed(1)} />
              <YAxis domain={[0, 1]} tick={AXIS} />
              <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                       formatter={(v: number) => v.toFixed(4)}
                       labelFormatter={(v: number) => `recall ${Number(v).toFixed(3)}`} />
              <ReferenceLine y={prevalence} stroke="#c05621" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="precision" stroke="#4f9dff" dot={false} strokeWidth={1.8}
                    isAnimationActive={false} name="precision" />
            </LineChart>
          </ResponsiveContainer>
        </figure>

        {plot.hist?.length ? (
          <figure className="chart-figure">
            <figcaption className="muted small">
              Score distribution by TRUE class. Overlap here is the reason the curve above looks the
              way it does - a metric says how well, this says why.
            </figcaption>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart accessibilityLayer title="Score distribution by true class"
                data={plot.hist} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                <XAxis dataKey="bin" tick={AXIS} label={{ value: "score bin", position: "insideBottom",
                       offset: -2, fill: "#8a94a6", fontSize: 10 }} />
                <YAxis scale="log" domain={[0.7, "auto"]} allowDataOverflow tick={AXIS} />
                <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }} />
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
                     name="true negative (solid)" />
                <Bar dataKey="positive" stackId="a" fill="url(#hatch-positive)" stroke={CARD}
                     strokeWidth={1} name="true positive (hatched)" />
              </BarChart>
            </ResponsiveContainer>
          </figure>
        ) : null}
        {plot.reliability?.length ? (
          <ReliabilityChart data={plot.reliability} of={plot.reliabilityOf ?? "p(positive)"}
                            modelKey={modelKey} />
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
          {res.baseline?.toFixed(4)}. The error's SHAPE matters: uniform drift is a miscalibrated
          model, a few large misses are missed events.
        </figcaption>
        <ResponsiveContainer width="100%" height={190}>
          <LineChart accessibilityLayer title="Forecast against actual, over time"
            data={plot.series} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis dataKey="i" tick={AXIS} />
            <YAxis tick={AXIS} />
            <Tooltip contentStyle={{ background: "#161b26", border: `1px solid ${GRID}`, fontSize: 11 }}
                     formatter={(v: number) => v.toFixed(4)} />
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
      </figure>
    );
  }

  return null;
}
