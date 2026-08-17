// The models behind a use case, and what they actually score.
//
// Everything here comes from the model registry (src/qsh/models/registry.py -> config.defaults.json).
// No architecture name, parameter count, metric, baseline, dataset licence or Hub URL is written in
// this file, so the page cannot drift from what the training runs produced.
//
// Two display rules that are the entire point of the panel:
//
//   * A model that has never been trained shows "not trained" — never a baseline standing in for a
//     measurement. The published baseline is what we are aiming at, not what we achieved.
//   * The honest metric is shown next to the flattering one. In four of these five industries the
//     aggregate number looks excellent while the class that matters is broken, so showing only the
//     aggregate would be true and misleading at the same time.
import { modelsFor, recordHasTreeAttribution, type ModelSpec } from "../lib/config";
import { useState } from "react";
import { browserModels, evaluatePublishedModel, type ModelEvalResult } from "../lib/onnxInference";
import { AncestryStrataChart, RealDataScatter } from "./VerificationChart";

const TIER_LABEL: Record<string, string> = {
  t1_crypto: "encrypted (FHE/MPC-feasible)",
  t2_industry: "industry model (plaintext)",
  t3_large: "large model (remote / WebGPU)",
};

/** Metrics that flatter and must never appear without their honest counterpart beside them. */
// Exact names were not enough: the set matched only whole keys, so quantum_qnn_accuracy,
// quantum_svc_accuracy and quantum_*_binary_f1 escaped the companion label and were rendered in the
// headline block — an aggregate presented as the result, which is the thing this project reports
// per-class metrics to avoid. Matched by suffix now, so a prefixed aggregate cannot slip through.
const COMPANION_METRICS = new Set([
  "accuracy", "aggregate_accuracy", "auroc", "point_adjusted_f1", "auprc_baseline",
]);
const COMPANION_SUFFIXES = ["_accuracy", "_binary_f1", "_auroc", "_point_adjusted_f1"];
const isCompanion = (k: string) =>
  COMPANION_METRICS.has(k) || COMPANION_SUFFIXES.some((sfx) => k.endsWith(sfx));

/** Run metadata that lives in the same record but is not a model metric. */
const NOT_A_METRIC = new Set([
  "wall_clock_s", "resolution", "n_eval", "n_train", "n_test", "positive_rate",
  "decision_threshold", "best_val_auc",
  // Sample sizes, not scores. These render inline on the headline row instead, next to the
  // baseline, because that is where the comparison gets made and where the count has to be read.
  "n_events_scored", "n_scoring_units", "n_anomaly_intervals",
]);

function fmt(v: unknown): string {
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString();
    // Don't let a near-miss print as a clean 1.0000. The satellite student's teacher-agreement is
    // 0.999987, which toFixed(4) rounds to "1.0000" — read as exact agreement when it is not.
    // toPrecision(3) is no better here ("1.00"), so widen the decimals until the value stops
    // rendering as an exact 1 or 0.
    for (let d = 4; d <= 8; d++) {
      const r = v.toFixed(d);
      if (Number(r) !== 1 && Number(r) !== 0) return r;
    }
    return v.toExponential(2);
  }
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

function MetricRows({ spec }: { spec: ModelSpec }) {
  const achieved = spec.achieved ?? null;
  if (!achieved) {
    return (
      <p className="muted small">
        Not trained in this build — no measured metrics. The published baseline below is the target,
        not a result.
      </p>
    );
  }
  // Flatten one level of nesting before filtering. genomics.prs keeps every measured value under
  // `measured` (n_samples, global_high_risk_cut, by_super_population...) and its only numeric
  // top-level key is wall_clock_s, which NOT_A_METRIC excludes — so the filter returned nothing and
  // the component rendered an empty <table>, hiding the ancestry stratification that the model's own
  // registry note calls "the finding, not a footnote".
  const flat: Array<[string, number]> = [];
  // `not_measured` holds published baselines and nulls for things this build did NOT measure --
  // walking into it would present someone else's 0.628 as this run's result. `licence_audit` is
  // provenance, not a metric.
  // Feature attribution is per-FEATURE, not per-model, so it does not belong in a table of model
  // metrics: flattened it would add 21 rows named `classical_rf_impurity_importance.qber_mean` to
  // the headline block, and the string that says which of the two measures to trust would be
  // dropped entirely (`walk` keeps numbers only). Rendered as its own block below instead, with
  // both measures on one row per feature so the disagreement is the thing you see.
  const ATTRIBUTION_SUBTREES = new Set([
    "classical_rf_impurity_importance",
    "classical_rf_permutation_importance",
    "classical_rf_permutation_importance_std",
  ]);
  const NOT_A_SUBTREE = new Set(["not_measured", "licence_audit", ...ATTRIBUTION_SUBTREES]);
  const walk = (obj: Record<string, unknown>, prefix: string, depth: number) => {
    for (const [k, v] of Object.entries(obj)) {
      if (NOT_A_SUBTREE.has(k)) continue;
      const key = prefix ? `${prefix}.${k}` : k;
      if (typeof v === "number" && Number.isFinite(v)) {
        if (!NOT_A_METRIC.has(k) && !NOT_A_METRIC.has(key)) flat.push([key, v]);
      } else if (v && typeof v === "object" && !Array.isArray(v) && depth < 3) {
        walk(v as Record<string, unknown>, key, depth + 1);
      }
    }
  };
  walk(achieved as Record<string, unknown>, "", 0);
  // Strip the redundant `measured.` prefix now that nothing unmeasured can reach here.
  const numeric: Array<[string, number]> = flat.map(
    ([k, v]) => [k.startsWith("measured.") ? k.slice("measured.".length) : k, v]);
  // A record that measured nothing numeric must say why rather than render an empty table.
  const notMeasured = (achieved as Record<string, unknown>).not_measured as
    | { reason?: string } | undefined;
  if (numeric.length === 0) {
    return (
      <p className="muted small">
        No numeric metric recorded for this model.
        {notMeasured?.reason ? ` ${notMeasured.reason}` : ""}
      </p>
    );
  }
  // The teacher scoring 0 on the fidelity slice voids the comparison whatever the student scored:
  // at student 0 the delta cancels to 0, and at student > 0 it goes NEGATIVE and reads as the
  // encrypted student beating its own teacher. Records from earlier rounds carry only the narrower
  // `both_scored_zero`, so fall back to reading `teacher_metric` directly rather than trusting a
  // flag written under the old condition.
  const rec = achieved as Record<string, unknown>;
  const teacherZero = "teacher_metric" in rec
    ? (rec.teacher_scored_zero as boolean | undefined) ?? (rec.teacher_metric as number) === 0
    : false;
  const studentScored = ((rec.student_metric as number) ?? 0) > 0;
  // How many things the headline was divided by. `n_events_scored` is the teacher's own count;
  // `n_scoring_units` is the distillation slice's. Neither is `n_eval`, which counts samples — for
  // an event-wise metric those differ by five orders of magnitude (307,601 samples, 2 events).
  const scoredOver = (rec.n_events_scored as number | undefined)
    ?? (rec.n_scoring_units as number | undefined) ?? null;
  const scoredUnit = (rec.scoring_unit as string | undefined) ?? "labelled events";
  // A distillation record measures a DECISION metric, because agreement compares decisions. The
  // registry's headline for the same model is often score-based (auc, auc_pr), and the panel prints
  // that baseline underneath. Naming the recorded metric is what stops a 0.7308 accuracy being read
  // against a 0.901 AUC.
  const recordedMetric = rec.metric_name as string | undefined;
  const metricMismatch = recordedMetric != null && recordedMetric !== spec.metric.name;
  const headline = numeric.filter(([k]) => !isCompanion(k));
  const companion = numeric.filter(([k]) => isCompanion(k));
  // The declared metric is not among the measured ones, so no row carries the baseline.
  //
  // This is genuinely the right state for genomics - 1000 Genomes has no phenotype, so no AUROC is
  // computable and `genomics_prs.py` records `auroc=null` rather than inventing one - but the effect
  // on the page was that the published figure appeared NOWHERE except inside a long provenance
  // paragraph. The number a visitor would quote was the hardest one on the page to find, which is
  // exactly backwards. Rendered as its own row, labelled as external, so it is visible AND cannot be
  // mistaken for something this run produced.
  const baselineOrphaned =
    spec.metric.baseline != null && !numeric.some(([k]) => k === spec.metric.name);

  // THE VERDICT, which is the one field that states what the comparison concluded and the one that
  // never reached the page. `quantum_beats_classical` is a BOOLEAN, and the walk above keeps
  // `typeof v === "number"` only, so it was dropped silently on every render. A record whose whole
  // framing is "architecture comparison, not a performance claim" published every input to the
  // comparison and not its result.
  //
  // DERIVED from the macro F1 values in this table rather than read from the flag, so a reader can
  // check the verdict against the two numbers it rests on. That is the whole value of the row.
  //
  // It is NOT an independent cross-check of the stored flag, and an earlier version of this comment
  // said it was. `scripts/train/telecom_ids.py` writes the flag as
  // `max(qnn.macro_f1, qsvc.macro_f1) > rf_metrics["macro_f1"]` and writes those same three values
  // into the same record in the same call - byte-for-byte the computation below. So agreement
  // between them is true by construction and is evidence of nothing. The disagreement branch is
  // kept because a hand-edited or externally-produced `metrics.json` can still reach it, but it is
  // not a guard against the staleness the old comment claimed, and calling it one made a decoration
  // read as a check.
  //
  // Macro F1 specifically, because that is this use case's declared headline; comparing on accuracy
  // or binary F1 would compare on exactly the aggregates the panel marks as companions.
  const num = (k: string) => (typeof rec[k] === "number" ? (rec[k] as number) : null);
  const classicalF1 = num("classical_rf_macro_f1");
  const quantumF1s = Object.entries(rec)
    .filter(([k, v]) => /^quantum_.*_macro_f1$/.test(k) && typeof v === "number")
    .map(([, v]) => v as number);
  const recordedVerdict = typeof rec.quantum_beats_classical === "boolean"
    ? (rec.quantum_beats_classical as boolean) : null;
  const verdict = classicalF1 != null && quantumF1s.length
    ? (() => {
        const best = Math.max(...quantumF1s);
        const derived = best > classicalF1;
        return { best, classical: classicalF1, derived, recorded: recordedVerdict,
                 agrees: recordedVerdict == null || recordedVerdict === derived };
      })()
    : null;

  return (
    <table className="mini-table">
      <tbody>
        {baselineOrphaned && (
          <tr className="metric-companion">
            <td><code>{spec.metric.name}</code></td>
            <td className="num">{fmt(spec.metric.baseline as number)}</td>
            <td className="muted small">
              <strong>published baseline, NOT measured here</strong>
              {spec.metric.ceiling != null ? ` · ceiling ${spec.metric.ceiling}` : ""}
              {notMeasured?.reason ? ` — ${notMeasured.reason}` : ""}
            </td>
          </tr>
        )}
        {headline.map(([k, v]) => (
          <tr key={k} className={k === spec.metric.name ? "metric-headline" : undefined}>
            {/* `student_metric` / `teacher_metric` are opaque names: they say which model, never
                which quantity. The record knows, so print it. */}
            <td><code>{k === "student_metric" || k === "teacher_metric"
              ? `${k.split("_")[0]} ${recordedMetric ?? "metric"}` : k}</code></td>
            <td className="num">{fmt(v)}</td>
            <td className="muted small">
              {k === spec.metric.name && spec.metric.baseline != null
                ? `baseline ${spec.metric.baseline}${spec.metric.ceiling != null
                    ? ` · ceiling ${spec.metric.ceiling}` : ""}`
                : ""}
              {/* The sample size belongs on the row that shows the baseline, because that row is
                  where the comparison actually gets made. An event-wise score over 3 events and a
                  benchmark over hundreds render as the same kind of decimal. */}
              {k === spec.metric.name && scoredOver != null && (
                <> · <strong>over {scoredOver} {scoredUnit}</strong>
                  {scoredOver < 10 && " — too few to compare with the baseline"}</>
              )}
            </td>
          </tr>
        ))}
        {metricMismatch && (
          <tr className="metric-companion">
            <td colSpan={3} className="muted small">
              Fidelity here is measured as <code>{recordedMetric}</code>, while the baseline quoted below
              belongs to <code>{spec.metric.name}</code>. Those are different quantities and must not be read against
              each other: fidelity compares the student's <em>decisions</em> to the teacher's, and a decision
              carries no score to rank. Where the declared metric is computable it is recorded as its own row
              above — compare the baseline to that.
            </td>
          </tr>
        )}
        {teacherZero && (
          <tr className="metric-companion">
            <td colSpan={3} className="muted small">
              <strong>The teacher scores 0 on the fidelity slice, so there is no fidelity claim here.</strong>{" "}
              {studentScored
                ? <>Because the student scored above 0, <code>metric_delta</code> comes out negative. That does
                   not mean the encrypted student outperformed its teacher — it means the teacher caught nothing
                   on this slice, and a near-zero score is being compared against zero.</>
                : <>Neither model scored, so <code>metric_delta</code> is 0 by cancellation, not because
                   encryption was free.</>}{" "}
              The slice is a later window than the one the teacher's headline metric is computed over, so
              neither number is comparable to that headline.
            </td>
          </tr>
        )}
        {companion.map(([k, v]) => {
          // The caption is DERIVED, not hardcoded. It read "shown so the gap is visible" for every
          // companion on every model, which asserts the aggregate is the flattering number being
          // disclosed. That is true of most models here and FALSE of `telecom_ids.nidd`, whose macro
          // F1 sits above its aggregate accuracy - so the one model that needed the caveat was the
          // one the caption contradicted. Compare the two numbers and say which way round they are.
          // Compare against the DECLARED metric, not against "the only headline". Most records carry
          // several headline rows - ablations, counterfactuals, linear floors - so a
          // headline.length === 1 test falls back to the uninformative caption on every model that
          // has anything to compare, which is all of them. `spec.metric.name` is the one row the
          // registry says this model is judged on, and it is what the baseline is rendered against
          // two rows up, so it is the right thing for the companion to be read against too.
          const declared = headline.find(([k]) => k === spec.metric.name);
          const h = declared ? Number(declared[1]) : NaN;
          const c = Number(v);
          const known = Number.isFinite(h) && Number.isFinite(c);
          // Three cases, not two. `telecom_ids.student` records macro_f1 and aggregate_accuracy as
          // the SAME number, and a two-branch ternary files equality under whichever branch is the
          // fallback - which would have printed "LOWER than the headline" about two identical values.
          const caption = !known
            ? "companion — the same run reported on a different metric"
            : Math.abs(c - h) < 5e-5
              ? "companion — equal to the headline here, so neither is the flattering figure"
              : c > h
                ? "companion — higher than the headline, so it is the flattering figure and the headline is the honest one"
                : "companion — LOWER than the headline here; the aggregate is the pessimistic figure, not the flattering one";
          return (
            <tr key={k} className="metric-companion">
              <td><code>{k}</code></td>
              <td className="num">{fmt(v)}</td>
              <td className="muted small">{caption}</td>
            </tr>
          );
        })}
        {verdict && (
          <tr className={verdict.agrees ? "metric-companion" : undefined}>
            <td><code>quantum_beats_classical</code></td>
            <td className="num">{String(verdict.derived)}</td>
            <td className="muted small">
              {verdict.derived
                ? <>the best quantum macro F1 here ({fmt(verdict.best)}) is <strong>above</strong> the
                  classical baseline ({fmt(verdict.classical)})</>
                : <>the classical baseline ({fmt(verdict.classical)}) is <strong>ahead</strong> of the
                  best quantum macro F1 here ({fmt(verdict.best)})</>}
              {verdict.recorded != null && !verdict.agrees ? (
                <> — <strong>and this contradicts the stored</strong>{" "}
                <code>quantum_beats_classical: {String(verdict.recorded)}</code>, which can only
                happen if the record was edited outside the training script. The verdict above is
                derived from the two numbers in this table.</>
              ) : null}
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}


/** Feature attribution for a tree-ensemble baseline: impurity and permutation, side by side.
 *
 * Why this is not the SHAP panel. Exact linear SHAP has a closed form only for a linear model; a
 * random forest has no such form, so the two standard tree measures are computed at training time
 * instead. They are shown TOGETHER because they disagree, and the disagreement carries the finding:
 * a feature the forest splits on often can still cost nothing to remove when another feature
 * separates the same class. Showing impurity alone would report such a feature as important.
 */
/** Probability calibration, where the record carries one.
 *
 * Shows the fitted Platt map and what it bought, INCLUDING any metric it cost. The "worse" branch
 * below is a GENERAL GUARD, not a finance fact: it fires off the recorded numbers whenever log-loss
 * moves the wrong way, because printing only the metrics that moved the right way is the reporting
 * failure this project documents elsewhere as pitfall P7. On finance it no longer fires.
 *
 * IT USED TO, AND WHY IT STOPPED IS THE POINT. Through Round 49 this panel rendered a map whose
 * log-loss was worse than not calibrating at all. Round 49 measured the cause and it was not the
 * base rate: `fit_platt` was being fitted on a probability rather than a log-odds, so with 90% of
 * the raw scores below 1e-4 the sigmoid was constant across nine of ten deciles and returned
 * 0.00090148 for all of them. The fix is `on_logit=True` in the trainer, and it is NOT confined to
 * the trainer - the map row below renders `a` and `b`, and those two floats mean a different
 * function depending on which argument they were fitted against, so this panel has to read
 * `c.on_logit` and say which. See references/finance-calibration-isotonic-vs-platt.md.
 *
 * No reliability diagram is drawn here even where the split supports one, because the CORP terms are
 * scalars and a curve would need a bin count this panel has no basis to choose. The record says per
 * page whether a curve would be defensible on that split, which is a different question from whether
 * this panel is the place for it.
 */
function CalibrationRows({ achieved }: { achieved: Record<string, unknown> }) {
  const c = achieved.calibration as Record<string, any> | undefined;
  if (!c || typeof c.a !== "number") return null;
  const f = (v: unknown, dp = 6) => (typeof v === "number" ? v.toFixed(dp) : "-");
  const worse = typeof c.log_loss_calibrated === "number" && typeof c.log_loss_raw === "number"
    && c.log_loss_calibrated > c.log_loss_raw;
  return (
    <div style={{ marginTop: 10 }}>
      <div className="muted small" style={{ marginBottom: 4 }}>
        <strong>Probability calibration</strong> — Platt, fitted on the validation split and scored on
        the held-out test split
      </div>
      <table className="mini-table">
        <tbody>
          {/* The two floats are meaningless without the argument they were fitted against: the map
              is linear in whatever it is handed, so the same a and b describe a different function
              over a score than over its log-odds. `on_logit` travels with them in the weights JSON
              for exactly this reason, and rendering one form for both is the defect the flag exists
              to prevent. The published `formula` string is NOT rendered instead - it is symbolic in
              a and b, and this row is the only place in the app where the fitted floats appear. */}
          <tr><td>map</td><td className="mono">{c.on_logit
            ? `p = 1 / (1 + exp(${f(c.a, 4)}·logit(clip(s, ${c.logit_clip_eps}, 1 - ${c.logit_clip_eps})) + ${f(c.b, 4)})), logit(q) = log(q / (1 - q))`
            : `p = 1 / (1 + exp(${f(c.a, 4)}·s + ${f(c.b, 4)}))`}</td></tr>
          <tr><td>Brier</td>
              <td className="mono">{f(c.brier_raw)} → {f(c.brier_calibrated)}</td></tr>
          <tr><td>CORP miscalibration (MCB)</td>
              <td className="mono">{f(c.corp_raw?.mcb, 7)} → {f(c.corp_calibrated?.mcb, 7)}</td></tr>
          <tr><td>CORP discrimination (DSC)</td>
              <td className="mono">{f(c.corp_raw?.dsc, 7)} <span className="muted">unchanged</span></td></tr>
          <tr><td>log-loss</td>
              <td className="mono" style={worse ? { color: "var(--warn)" } : undefined}>
                {f(c.log_loss_raw)} → {f(c.log_loss_calibrated)}{worse ? " (worse)" : ""}</td></tr>
        </tbody>
      </table>
      {worse && (
        <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
          Calibration moved log-loss the wrong way here. Every term is shown anyway, because
          reporting only the metrics that moved the right way is the failure this project cites
          Arp et al. for. Two causes are worth separating before this is read as a property of the
          base rate: at a low base rate log-loss is dominated by the negatives, so a sigmoid that
          lowers mean squared error can still cost on the tails — but a map fitted on the wrong
          argument produces the same signature, and on finance that is what it turned out to be.
        </p>
      )}
      <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
        Platt is <i>strictly</i> monotone, so AUC and AUC-PR are unchanged by construction —
        calibration fixes what a probability <i>means</i>, never how the rows are ordered. Platt
        rather than isotonic is a measured tie-break, not a size argument: this calibration set is
        far past the range the published comparison discusses, and on this split the two score
        within noise on Brier and log-loss. Isotonic is only <i>weakly</i> monotone, and it cost
        9.09% of AUC-PR here by collapsing 1.39M distinct scores to 138.
      </p>
    </div>
  );
}

function AttributionRows({ achieved }: { achieved: Record<string, unknown> }) {
  const num = (k: string) => (achieved[k] ?? null) as Record<string, number> | null;
  const perm = num("classical_rf_permutation_importance");
  const imp = num("classical_rf_impurity_importance");
  const std = num("classical_rf_permutation_importance_std");
  const note = achieved.classical_rf_attribution_note as string | undefined;
  // Same predicate `ShapPanel` points at, so the sentence and the block cannot disagree. Round 32
  // wrote that they shared one and they did not: this guard wanted both measures, the predicate
  // wanted one. Keeping the local `perm`/`imp` reads for the render below, but the CONDITION is
  // imported rather than restated - restating it is how the two drifted in the first place.
  if (!recordHasTreeAttribution(achieved) || !perm || !imp) return null;

  // Ranked by permutation importance, because that is the measure to read where they differ.
  const rows = Object.keys(perm).sort((a, b) => perm[b] - perm[a]);
  const max = Math.max(...rows.map((f) => Math.abs(perm[f])), 1e-9);

  return (
    <div style={{ marginTop: 10 }}>
      <div className="muted small" style={{ marginBottom: 4 }}>
        <strong>Which features this baseline needs</strong> — permutation importance on macro F1
        (held-out split), with impurity importance beside it
      </div>
      <table className="mini-table">
        <tbody>
          {rows.map((f) => {
            // Within one std of zero is not a small effect, it is no measured effect, and it has to
            // read differently from one. dark_count_rate is +0.0001 against a std of 0.0010.
            const indistinct = std ? Math.abs(perm[f]) <= std[f] : false;
            return (
              <tr key={f}>
                <td><code>{f}</code></td>
                <td className="num">
                  {perm[f] >= 0 ? "+" : ""}{perm[f].toFixed(4)}
                  {std ? <span className="muted"> ± {std[f].toFixed(4)}</span> : null}
                </td>
                <td style={{ width: 90 }}>
                  <div style={{ background: "#1a2233", borderRadius: 4, height: 10 }}>
                    <div style={{ width: `${Math.max(2, (Math.abs(perm[f]) / max) * 100)}%`,
                      height: "100%", borderRadius: 4,
                      background: indistinct ? "#5a6478" : "#4f9dff" }} />
                  </div>
                </td>
                <td className="muted small">
                  impurity {imp[f]?.toFixed(4) ?? "n/a"}
                  {indistinct && " · within one std of zero — no measured effect"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {note ? <p className="muted small" style={{ marginTop: 4 }}>{note}</p> : null}
    </div>
  );
}


/** Run one published model in the browser, or say precisely why it cannot be run.
 *
 * Every reason below is a different fact about the model and they must not be collapsed into a
 * single "unavailable": a model nobody trained, a model whose weights are licence-blocked, and a
 * model that is published but ships no held-out sample are three separate states, and flattening
 * them would hide which one applies.
 */
function ModelRunner({ usecase, spec }: { usecase: string; spec: ModelSpec }) {

  const fetchable = browserModels(usecase).some((m) => m.spec.key === spec.key);
  const blocked =
    !spec.achieved ? "never trained in this build, so there is nothing to run - the baseline above is the target, not a result"
    : !spec.hf_repo_id ? "deliberately unpublished: the source data's licence does not clearly permit redistributing derived weights"
    : spec.hf_revision_stale ? "retrained since it was last published, so the pinned artefact no longer matches the metrics above and the pin was retired rather than pointed at stale weights"
    : !spec.hf_revision ? "published but not pinned to a commit, and the browser will not fetch an unpinned model"
    : !fetchable ? "published, but it ships no browser-runnable ONNX graph at that revision"
    : null;

  if (blocked) return <p className="muted small"><em>Cannot be run here: {blocked}.</em></p>;

  // The standalone "run this model" button was REMOVED in Round 24j. It called exactly what the
  // E2E orchestration now calls on every run - evaluatePublishedModel on the whole held-out sample
  // - so it had become a second, manual path to the same check, with its own state and its own
  // chance to disagree. The run verifies every published model for this use case, not only the
  // selected one, so nothing is lost by deleting it: start the pipeline and the verification is
  // part of the result rather than a button somebody has to know to press.
  return <p className="muted small">
    Verified during the pipeline run: press <strong>Start</strong> and this model is re-scored on its
    published held-out sample in this tab, with the result reported in the run log.
  </p>;
}

/**
 * The full result of running one published model, checked against its offline expectation.
 *
 * This was `PublishedModelPanel`, a second panel mounted about forty lines below this one on every
 * use-case page. The two offered the same ResNet-18 run twice with different chrome and word-for-word
 * duplicated copy, and only one of them explained the models it could NOT run. Merged into the panel
 * that lists every model: the reason ladder above says why a model has no button, and this says what
 * happened when one did.
 *
 * The comparison is the point. `expected` was measured offline by `scripts/prep_model_inputs.py` with
 * onnxruntime on exactly these rows, so a disagreement means something differs - preprocessing,
 * quantisation, execution provider - and the honest response is to withhold the number rather than
 * render it.
 */
// `ModelEvalReport` lived here: it rendered the PASS/FAIL verdict, the mismatch warning, the sample's
// own note and the provenance line. It was defined, never referenced, and therefore tree-shaken out of
// every shipped bundle - so a verification FAIL rendered exactly like a PASS and the only trace was a
// raw metrics row. All four now render in `UseCasePage`'s "Verification, this run" card, which is the
// surface that actually appears. Deleted rather than left in place, because an unreferenced component
// that looks like it handles a case is how the case went unhandled for several rounds.

export function ModelPanel({ usecase }: { usecase: string }) {
  const specs = modelsFor(usecase);
  if (specs.length === 0) return null;

  return (
    <section className="card">
      <h3>Models</h3>
      <p className="muted small">
        The industry model runs in plaintext; the encrypted model is what homomorphic evaluation can
        actually carry. The gap between them is measured, not asserted.
      </p>

      {specs.map((spec) => (
        <div key={spec.key} className="model-block">
          <div className="model-head">
            <strong>{spec.label}</strong>
            <span className="tag">{TIER_LABEL[spec.tier] ?? spec.tier}</span>
            {spec.runs_encrypted ? <span className="tag">runs under CKKS</span> : null}
          </div>
          <p className="muted small">{spec.architecture}</p>

          <MetricRows spec={spec} />

          {/* The published RandomForest had no attribution at all for four rounds while this page
              showed exact linear SHAP over a different model trained in the tab. Renders only where
              the record carries the two tree measures, so no other model grows an empty block. */}
          {spec.achieved
            ? <AttributionRows achieved={spec.achieved as Record<string, unknown>} />
            : null}

          {/* Renders only where the record carries a fitted map, so no model grows an empty block.
              Today that is finance; the other pages have no calibration fitted yet and say nothing
              rather than showing a placeholder. */}
          {spec.achieved
            ? <CalibrationRows achieved={spec.achieved as Record<string, unknown>} />
            : null}

          {/* The one published model with no browser-runnable graph, so the in-run verification
              chart never fires for it. Its finding lives in the metric rows above and twenty rows
              hide it; this draws the same numbers. */}
          <AncestryStrataChart achieved={spec.achieved} />

          <p className="muted small">
            <strong>Data:</strong> {spec.dataset.name} — {spec.dataset.licence}{" "}
            <a href={spec.dataset.url} target="_blank" rel="noreferrer noopener">source</a>
            {!spec.dataset.publishable && " · weights not published (licence)"}
          </p>
          {/* The CC BY credit, rendered rather than merely recorded. Four corpora here carry the
              condition and this line is where a visitor meets it; the same string also goes into
              the Hugging Face model card and the evaluation sample's provenance, so all three
              surfaces a reader can reach carry the attribution instead of one of them. */}
          {spec.dataset.attribution && (
            <p className="muted small"><strong>Attribution:</strong> {spec.dataset.attribution}</p>
          )}
          {spec.metric.baseline_source && (
            <p className="muted small"><strong>Baseline:</strong> {spec.metric.baseline_source}</p>
          )}
          {typeof spec.achieved?.baseline_gap_note === "string" && (
            <p className="muted small"><strong>Gap to baseline:</strong> {spec.achieved.baseline_gap_note}</p>
          )}
          {/* `metric_note` reached the browser in config.defaults.json and was rendered NOWHERE.
              SIX records carry one - finance_aml.scorecard, medical_fl.student,
              satellite_comms.opssat_student, satellite_comms.telemanom, telecom_ids.nidd and
              telecom_ids.student - each written by a trainer, checked by the prose gate, shipped in
              the bundle, and shown to nobody. It is the field that says WHICH of two recorded numbers
              answers WHICH question, so its absence is what let the companion caption above assert a
              direction the record contradicted. Sits above `notes` because it qualifies the table
              rather than the model. */}
          {typeof spec.achieved?.metric_note === "string" && (
            <p className="muted small"><strong>Reading these metrics:</strong> {spec.achieved.metric_note}</p>
          )}
          {spec.notes && <p className="muted small">{spec.notes}</p>}
          <ModelRunner usecase={usecase} spec={spec} />
          {spec.hf_repo_id && (
            <p className="muted small">
              <strong>Weights:</strong>{" "}
              <a href={`https://huggingface.co/${spec.hf_repo_id}`} target="_blank" rel="noreferrer noopener">
                {spec.hf_repo_id}
              </a>
              {spec.hf_revision
                ? <> · pinned to <code>{spec.hf_revision.slice(0, 12)}</code></>
                : <> · <em>no pinned revision; the browser will not fetch an unpinned model</em></>}
            </p>
          )}
        </div>
      ))}

      {/* Once per use case, not once per model: this is the corpus, not any model's output. It
          answers the question a lone score cannot - whether the classes were separable at all. */}
      <RealDataScatter usecase={usecase} />
    </section>
  );
}
