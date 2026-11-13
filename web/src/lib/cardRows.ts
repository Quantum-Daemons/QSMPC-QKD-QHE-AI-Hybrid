/**
 * The rows each card's download holds (Round 86, the owner's choice of 2026-11-13). One function per card, used by the
 * card's own buttons and by the run's results bundle (resultsBundle.ts), so the two files cannot disagree about a card.
 * A row holds what the card shows and nothing it hides: no key, no seed of a mask, no party's own contribution.
 */
import type { CountRelease } from "./clientDp";
import type { AggregationComparison } from "./clientFedAgg";
import type { FedHeartDpResult, FedHeartResult } from "./clientFedHeart";

type Row = Record<string, string | number | boolean | null>;

const r6 = (x: number) => Math.round(x * 1e6) / 1e6;

/** The four-hospital matrix: one row per model, one column per hospital's test rows, and FLamby's metric. */
export function fedHeartMatrixRows(r: FedHeartResult): Row[] {
  const row = (model: string, trainedOn: string, accs: number[], mean: number): Row => {
    const out: Row = { model, trained_on: trainedOn };
    r.centres.forEach((c, j) => { out[`accuracy_on_${c}_test`] = r6(accs[j]); });
    out.mean_flamby_metric = r6(mean);
    return out;
  };
  return [
    ...r.centres.map((c, k) => row(`${c} alone`, `${r.trainRows[k]} rows of ${c}`, r.alone[k], r.mean.alone[k])),
    row(`federated, ${r.rounds} rounds of FedAvg`, `${r.trainRows.reduce((a, b) => a + b, 0)} rows, never pooled`,
      r.federated, r.mean.federated),
    row("pooled in one place", `${r.trainRows.reduce((a, b) => a + b, 0)} rows`, r.pooled, r.mean.pooled),
    row("initial model, before training", "-", r.initial, r.mean.initial),
  ];
}

/** The record-level DP table: one row per target epsilon; this run's noise, so each run's file differs. */
export function fedHeartDpRows(dp: FedHeartDpResult): Row[] {
  return dp.rows.map((x) => ({
    epsilon_target: Number.isFinite(x.epsilon) ? x.epsilon : "none (clipping only)",
    epsilon_achieved: Number.isFinite(x.epsilonAchieved) ? r6(x.epsilonAchieved) : null,
    epsilon_if_one_hospital_adds_none: Number.isFinite(x.epsilonOneDishonest) ? r6(x.epsilonOneDishonest) : null,
    delta: dp.delta,
    rho: Number.isFinite(x.rho) ? r6(x.rho) : null,
    variance_per_hospital: x.variancePerHospital,
    noise_sd_total_gradient_units: r6(x.sdTotal),
    accuracy_mean_flamby_metric: r6(x.mean),
    steps: dp.rounds, clip: dp.clip, noise_source: dp.source,
  }));
}

/** The encrypted-aggregation comparison, one row per design. */
export function aggregationRows(res: AggregationComparison): Row[] {
  return res.designs.map((d) => ({
    design: d.label, scheme: d.scheme, who_can_read_one_update: d.whoDecrypts,
    statistical_security_bits: d.statSecBits,
    max_distance_from_exact_fedavg: d.maxDiffFromExact,
    accuracy_mean_flamby_metric: r6(d.mean),
    same_as_masks: d.identicalToMasks,
    dp_epsilon_achieved: d.dp ? r6(d.dp.epsilonAchieved) : null,
    dp_steps_equal_to_masks: d.dp ? `${d.dp.stepsEqual} of ${d.dp.steps}` : null,
    bytes_per_hospital_per_round: d.bytesPerHospitalPerRound,
    key_setup_ms: r6(d.keySetupMs), ms_per_round: r6(d.msPerRound),
    seed: res.seed, rounds: res.rounds, hospitals: res.hospitals,
  }));
}

/** What the genomics card shows. Each cohort's contribution stays out: the secure sum exists to keep it. */
export interface GenomicsDpShown {
  released: number; exact: number; epsilon: number; epsilonAchieved: number; epsilonOneDishonest: number;
  delta: number; sdTotal: number; variancePerCohort: number; scale: number; variant: string; individuals: number;
  cohorts: number;
}

export function genomicsDpRows(dp: GenomicsDpShown): Row[] {
  return [{
    variant: dp.variant, released_count: r6(dp.released), exact_count_not_for_publication: dp.exact,
    epsilon_asked: dp.epsilon, epsilon_achieved: r6(dp.epsilonAchieved),
    epsilon_if_one_cohort_adds_none: r6(dp.epsilonOneDishonest), delta: dp.delta,
    noise_sd_alleles: r6(dp.sdTotal), variance_per_cohort_fixed_point: dp.variancePerCohort, fixed_point_scale: dp.scale,
    individuals: dp.individuals, cohorts: dp.cohorts,
  }];
}

export function financeDpRows(dp: CountRelease & { exact: number; sizeA: number; sizeB: number }): Row[] {
  return [{
    published_count: dp.released, exact_count_held_by_the_banks: dp.exact, epsilon_asked: dp.epsilon,
    epsilon_achieved: r6(dp.epsilonAchieved), delta: dp.delta, variance: dp.variance, noise_sd_accounts: r6(dp.sd),
    list_size_bank_a: dp.sizeA, list_size_bank_b: dp.sizeB,
  }];
}

/** Every published model's re-scoring in this tab (clientCompute's `verifications`). */
export function verificationRows(list: readonly import("./clientCompute").VerificationSummary[]): Row[] {
  return list.map((v) => ({
    model: v.modelKey, selected: v.selected, outcome: v.outcome, metric: v.metric ?? null,
    score_this_tab: v.score != null ? r6(v.score) : null, recorded_with_the_sample: v.expected != null ? r6(v.expected) : null,
    rows: v.n ?? null, split: v.split ?? null, decision_threshold: v.decisionThreshold ?? null,
    rows_as_recorded: v.rowsAsRecorded ?? null, near_ties: v.nearTies ?? null, rows_different: v.mismatches ?? null,
    max_margin_difference: v.maxMarginDifference ?? null, tolerance: v.tolerance ?? null, backend: v.backend ?? null,
    note: v.error ?? v.perRowRefused ?? v.noSampleReason ?? null,
  }));
}
