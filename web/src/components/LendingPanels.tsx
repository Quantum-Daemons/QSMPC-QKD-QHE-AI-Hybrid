// The lending page's results (Round 86): the simulated market, the private check and where the work ran.
//
// Every number comes from the run. The self-tests against the clear margin exist only because both parties are in this tab, and the card says so; the limits are part of the result.
import { useId, useMemo } from "react";
import {
  CartesianGrid, ComposedChart, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { ChartReadout, HINT_KEYBOARD, HINT_POINTER, ReadoutTooltip } from "./ChartReadout";
import Details from "./Details";
import KeyPoints from "./KeyPoints";
import StageBadge from "./StageBadge";
import { MarksHidden } from "./VerificationChart";
import type { LendingMarketReport, LendingPrivateReport, LendingSweepReport } from "../lib/lending/report";

const tick = { fill: "var(--muted)", fontSize: 11 };
const grid = <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />;
const axisLabel = (value: string) => ({ value, position: "insideBottom" as const, offset: -8, fill: "var(--muted)", fontSize: 11 });
const yLabel = (value: string, side: "left" | "right" = "left") => ({ value, angle: side === "left" ? -90 : 90, position: side === "left" ? "insideLeft" as const : "insideRight" as const, fill: "var(--muted)", fontSize: 11 });
const money = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 0 });
const mb = (bytes: number) => `${(bytes / 1e6).toFixed(2)} MB`;
const ms = (v: number) => `${v.toFixed(0)} ms`;
const short = (v: number) => (Math.abs(v) >= 1e6 ? `${+(v / 1e6).toFixed(1)}M` : Math.abs(v) >= 1e3 ? `${+(v / 1e3).toFixed(0)}k` : String(v));

const COMMON_LIMITS = [
  <><b>A bounded mask.</b> The mask is a real number of limited width, so it hides the margin statistically, not perfectly. Gazelle-style masking over the plaintext ring is perfect and costs about two million AND gates per ciphertext in this circuit library.</>,
  <><b>What is post-quantum.</b> ML-KEM (with X25519 alongside as a hedge), ML-DSA and CKKS rest on lattice assumptions. The garbled circuit's oblivious transfers rest on discrete logs: in a deployment the sealed session would keep them from outsiders, but a counterparty that records them could open them with a future quantum computer. In this tab they stay in memory.</>,
  <><b>A simulated market.</b> There is no chain, no oracle and no real price here: the prices are generated, the parameters illustrative, and the mechanics re-derived from Aave V3's public contracts, with none of their code. Aave V4's target health factor and auction-style bonus are not modelled.</>,
];

const KEYHOLDER_LIMITS = [
  <><b>Why a separate key holder.</b> Liquidations are carried out by third parties, without the borrower taking part, so a check cannot wait on the borrower. Carol holds the key instead, and Bob is needed only for his one submission. Confidential-DeFi systems use a threshold network for this: Zama's KMS runs 13 MPC nodes and, in its documentation's example, needs 9 of them. node-seal has no threshold CKKS, so here Carol is one party standing in for such a committee.</>,
  <><b>Two parties must not collude.</b> The pool holds Bob's ciphertext and Carol the key, so together they see everything. Carol must never hold his ciphertext unmasked: here only the pool keeps it, and a deployment would enforce that with access control.</>,
  <><b>Semi-honest.</b> Nothing proves that Bob encrypted his real balances, which takes commitments and zero-knowledge proofs.</>,
  ...COMMON_LIMITS,
  <><b>What leaks by design.</b> The bit is published and signed, and a liquidation is settled in the open. In production so far, confidential lending hides deposits (Zama, Morpho and Steakhouse, June 2026), and this project found no primary source for a confidential liquidation running in production.</>,
];

const BILATERAL_LIMITS = [
  <><b>Why the thresholds are the pool's secret here.</b> On a public chain they are public, and then Bob can compute his margin himself: the mask and the circuit would hide nothing from anyone. This check suits a lender whose thresholds are its own, and a Bob who should learn no more than the answer.</>,
  <><b>Bob has to answer.</b> He holds the key, so every check waits on him, and he is the party who loses by a liquidation. He could refuse or go offline, so a deployment would need a rule that no answer counts as a default.</>,
  <><b>Semi-honest.</b> Nothing proves that Bob encrypted his real balances, which takes commitments and zero-knowledge proofs, and a malicious Bob could probe the thresholds with crafted ciphertexts, as MUSE (USENIX Security 2021) does to model weights.</>,
  ...COMMON_LIMITS,
  <><b>What leaks by design.</b> The bit reaches both parties, repeated bits narrow the thresholds down, and a liquidation is settled in the open.</>,
];

export default function LendingPanels({ market, sweep, check, ran, state, custody }: { market?: LendingMarketReport; sweep?: LendingSweepReport; check?: LendingPrivateReport; ran: boolean; state: string; custody: string }) {
  // The run's own arrangement once it has one, and the control's before it does.
  const held = check?.custody ?? market?.cfg.custody ?? custody;
  return (
    <>
      {market ? <Market report={market} check={check} /> : ran && state !== "running" ? (
        <div className="card"><h3>The market, this run</h3><p className="small" style={{ color: "var(--danger)" }}>The market stage produced nothing: see the log above.</p></div>
      ) : null}
      {check ? <PrivateCheck report={check} /> : market && state !== "running" ? (
        <div className="card"><h3>Private liquidation check</h3><p className="small" style={{ color: "var(--danger)" }}>The private check did not finish: see the log above.</p></div>
      ) : null}
      {sweep ? <Sweep report={sweep} /> : null}
      {market ? <WhereItRan market={market} sweep={sweep} check={check} /> : null}
      <div className="note" style={{ marginTop: 14, fontSize: 12 }}>
        <KeyPoints title="What this page does and does not show" items={held === "borrower" ? BILATERAL_LIMITS : KEYHOLDER_LIMITS} />
      </div>
    </>
  );
}

function Market({ report, check }: { report: LendingMarketReport; check?: LendingPrivateReport }) {
  const { run, cfg } = report;
  const rows = useMemo(() => {
    let cumulative = 0;
    return run.utilisation.map((u, hour) => {
      cumulative += run.liquidatedPerHour[hour];
      return {
        hour, asset1: (100 * run.prices[0][hour]) / run.prices[0][0], asset2: (100 * run.prices[1][hour]) / run.prices[1][0],
        health: Number.isFinite(run.bobHealth[hour]) ? run.bobHealth[hour] : null,
        utilisation: 100 * u, rate: 100 * run.borrowRate[hour], liquidated: cumulative,
      };
    });
  }, [run]);
  const safe = check?.outcomes.filter((o) => !o.flag && Number.isFinite(o.healthFactor)).map((o) => ({ hour: o.hour, safe: o.healthFactor })) ?? [];
  const flagged = check?.outcomes.filter((o) => o.flag).map((o) => ({ hour: o.hour, flagged: o.healthFactor })) ?? [];
  const repaid = run.events.reduce((s, e) => s + e.repaid, 0);
  const bobEvents = run.events.filter((e) => e.borrower === 0).length;
  const shock = cfg.shock_fraction > 0 ? <ReferenceLine x={cfg.shock_hour} stroke="var(--danger)" strokeDasharray="4 3" /> : null;
  const xAxis = <XAxis dataKey="hour" type="number" domain={[0, run.hours]} tick={tick} label={axisLabel("hour")} />;
  return (
    <div className="card" id="lending-market">
      <h3>The market, this run <StageBadge kind="simulated" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        {run.events.length.toLocaleString("en-US")} liquidations over {run.hours} hours among {run.borrowers} borrowers, {bobEvents} of them Bob&apos;s, repaying {money(repaid)} stable units and leaving {money(run.badDebt)} of bad debt.
        Utilisation peaked at {(100 * Math.max(...run.utilisation)).toFixed(1)}% and the borrow rate at {(100 * Math.max(...run.borrowRate)).toFixed(2)}% a year.
        The pool&apos;s books balanced to within {run.maxAccountingGap.toExponential(1)} stable units at every hour.
      </p>
      <div className="grid cols-2">
        <figure className="chart-figure">
          <figcaption className="small muted"><b>Generated prices, rebased to 100.</b> The dashed line is the stated fall.</figcaption>
          <ChartReadout hint={HINT_KEYBOARD}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rows} accessibilityLayer title="Generated prices of the two collateral assets over the simulated month, rebased to 100" margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
                {grid}{xAxis}
                <YAxis tick={tick} domain={["auto", "auto"]} />
                <Tooltip content={<ReadoutTooltip labelName="hour" />} formatter={(v: number) => v.toFixed(1)} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                {shock}
                <Line type="linear" dataKey="asset1" name="asset 1" stroke="var(--accent)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line type="linear" dataKey="asset2" name="asset 2" stroke="var(--accent-2)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartReadout>
        </figure>
        <figure className="chart-figure">
          <figcaption className="small muted"><b>Bob&apos;s health factor, before each hour&apos;s liquidation.</b> Marks are the private checks, drawn at the clear value for reference: a cross is a check the circuit answered &quot;can be liquidated&quot;. The line ends where his position closes.</figcaption>
          <ChartReadout hint={HINT_KEYBOARD}>
            <MarksHidden>
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart accessibilityLayer title="Bob's health factor over the simulated month, with the private checks marked" margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
                {grid}{xAxis}
                <YAxis tick={tick} domain={[0, "auto"]} />
                <Tooltip content={<ReadoutTooltip labelName="hour" />} formatter={(v: number) => v.toFixed(3)} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                <ReferenceLine y={1} stroke="var(--danger)" strokeDasharray="4 3" label={{ value: "liquidation below 1", fill: "var(--danger)", fontSize: 10, position: "insideBottomRight" }} />
                <Line data={rows} type="linear" dataKey="health" name="health factor" stroke="var(--accent)" dot={false} strokeWidth={1.5} isAnimationActive={false} connectNulls={false} />
                <Scatter data={safe} dataKey="safe" name="check: safe (circle)" fill="var(--accent-2)" shape="circle" isAnimationActive={false} />
                <Scatter data={flagged} dataKey="flagged" name="check: can be liquidated (cross)" fill="var(--danger)" shape="cross" isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
            </MarksHidden>
          </ChartReadout>
        </figure>
        <figure className="chart-figure">
          <figcaption className="small muted"><b>The pool.</b> Above {(cfg.optimal_utilisation * 100).toFixed(0)}% utilisation the second slope of the rate applies, and borrowers start repaying.</figcaption>
          <ChartReadout hint={HINT_KEYBOARD}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rows} accessibilityLayer title="Pool utilisation and borrow rate over the simulated month" margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
                {grid}{xAxis}
                <YAxis yAxisId="u" tick={tick} domain={[0, 100]} label={yLabel("utilisation (%)")} />
                <YAxis yAxisId="r" orientation="right" tick={tick} domain={[0, "auto"]} label={yLabel("borrow rate (% a year)", "right")} />
                <Tooltip content={<ReadoutTooltip labelName="hour" />} formatter={(v: number) => v.toFixed(2)} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
                <Line yAxisId="u" type="linear" dataKey="utilisation" name="utilisation" stroke="var(--accent)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line yAxisId="r" type="linear" dataKey="rate" name="borrow rate" stroke="var(--warn)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartReadout>
        </figure>
        <figure className="chart-figure">
          <figcaption className="small muted"><b>Debt repaid by liquidation, cumulative.</b> In stable units, over every borrower.</figcaption>
          <ChartReadout hint={HINT_KEYBOARD}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rows} accessibilityLayer title="Cumulative debt repaid by liquidation over the simulated month" margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
                {grid}{xAxis}
                <YAxis tick={tick} tickFormatter={short} domain={[0, "auto"]} />
                <Tooltip content={<ReadoutTooltip labelName="hour" />} formatter={(v: number) => money(v)} />
                {shock}
                <Line type="stepAfter" dataKey="liquidated" name="repaid by liquidation" stroke="var(--danger)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartReadout>
        </figure>
      </div>
      <Details summary={`Liquidations, one row each (${run.events.length})`}>
        <div style={{ overflowX: "auto" }}>
          <table className="table small">
            <thead><tr><th>Hour</th><th>Borrower</th><th>Health factor before</th><th>Debt repaid</th><th>Collateral value seized</th><th>Bad debt</th></tr></thead>
            <tbody>
              {run.events.map((e, i) => (
                <tr key={i}><td>{e.hour}</td><td>{e.borrower === 0 ? "Bob" : `#${e.borrower}`}</td><td>{e.hfBefore.toFixed(3)}</td><td>{money(e.repaid)}</td><td>{money(e.seizedValue)}</td><td>{money(e.badDebt)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Details>
    </div>
  );
}

function PrivateCheck({ report: r }: { report: LendingPrivateReport }) {
  const captionId = useId();
  const n = r.outcomes.length;
  const safe = r.outcomes.filter((o) => !o.flag).map((o) => ({ clear: o.clearMargin, shares: o.reconstructedMargin }));
  const flagged = r.outcomes.filter((o) => o.flag).map((o) => ({ clear: o.clearMargin, shares: o.reconstructedMargin }));
  const all = r.outcomes.map((o) => o.clearMargin);
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const pad = (hi - lo) * 0.05 || 1;
  const bilateral = r.custody === "borrower";
  const held = bilateral ? "Bob" : "Carol";
  const costs: [string, string][] = [
    ["Per check", `${mb(r.ckksBytes / n)} of sealed CKKS traffic${bilateral ? `, ${r.signatureBytes.toLocaleString("en-US")} B of it Bob's ${r.signature} signature` : ""}; ${(r.andGates / n).toLocaleString("en-US")} AND gates and ${mb(r.gcBytes / n)} of garbled-circuit traffic in ${Math.round(r.gcMessages / n).toLocaleString("en-US")} messages${bilateral ? "" : `; ${r.signatureBytes.toLocaleString("en-US")} B for Carol's ${r.signature} signature on the bit`}`],
    ["Keys", bilateral
      ? `${mb(r.keyBytes)} of public and rotation keys, once per key pair; ${r.pool.workers === 1 ? "one key pair" : `${r.pool.workers} key pairs here, one per worker`}, ${ms(r.keygenMs)} to set up the slowest`
      : `${mb(r.keyBytes)} of Carol's public and rotation keys, sent once. Her one key pair is loaded on ${r.pool.workers === 1 ? "the worker" : `each of the ${r.pool.workers} workers`}, ${ms(r.keygenMs)} for the slowest, which only a tab holding every party can do`],
    ["Time, summed over the checks", `CKKS ${ms(r.ckksMs)}, the circuits ${ms(r.gcMs)}, ML-DSA and sealing ${ms(r.pqcMs)}`],
    ["Wall-clock", `${ms(r.pool.wallMs)} in ${r.pool.ranIn}`],
    ["The mask", `uniform on [-2^${r.maskBits}, 2^${r.maskBits}), on every slot; this run's bound on the statistical distance from any margin of no larger magnitude, summed over all the slots it masks, is 2^${r.statisticalDistanceLog2.toFixed(1)}`],
    ["Noise floor", `the shares and the clear margin differ by at most ${(r.maxAbsError * 1000).toFixed(3)} stable units. A margin closer to zero than that can come out either way`],
  ];
  return (
    <div className="card" id="lending-private-check" style={{ marginTop: 14 }}>
      <h3>Private liquidation check <StageBadge kind="notional" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        {bilateral ? (
          <>At each checked hour Bob encrypted his two collateral amounts and his scaled debt under his own CKKS key, signed the ciphertext and sent it over the session sealed under the PQC key.
          The pool applied its weights, each asset&apos;s public price times its private liquidation threshold and the public interest index, and added a mask it kept.
          Bob decrypted the masked margin, and the two of them put their shares into a garbled circuit, which told both whether the margin is negative: a health factor below 1.</>
        ) : (
          <>Bob encrypted his two collateral amounts and his scaled debt once, under Carol&apos;s public key, signed the ciphertext and sent it to the pool over their sealed session. After that he took no part.
          At each checked hour the pool applied what had been settled in the open, multiplied by the public weights, each asset&apos;s price times its liquidation threshold and the interest index, and added a mask it kept.
          Carol decrypted only the masked margin, the pool and Carol put their shares into a garbled circuit, which told both whether the margin is negative, a health factor below 1, and Carol signed that bit.</>
        )}
      </p>
      <p style={{ margin: "6px 0 10px" }}>
        {r.flagged} of {n} checks: can be liquidated.{" "}
        <span className={`badge-state ${r.allAgree ? "done" : "error"}`}>{r.allAgree ? `all ${n} match the check in the clear` : "DISAGREES with the check in the clear"}</span>{" "}
        {r.bitSignaturesVerified != null && (
          <span className={`badge-state ${r.bitSignaturesVerified ? "done" : "error"}`}>{r.bitSignaturesVerified ? "every bit's signature verifies" : "A SIGNATURE ON A BIT DOES NOT VERIFY"}</span>
        )}
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>Hour</th><th>Health factor (clear)</th><th>Margin (clear)</th><th>Margin from both shares</th><th>Difference</th><th>The circuit&apos;s answer</th></tr></thead>
          <tbody>
            {r.outcomes.map((o) => (
              <tr key={o.hour}>
                <td>{o.hour}</td><td>{o.healthFactor.toFixed(4)}</td><td>{(o.clearMargin * 1000).toFixed(2)}</td><td>{(o.reconstructedMargin * 1000).toFixed(2)}</td><td>{(o.absError * 1000).toFixed(3)}</td>
                <td>{o.flag ? "can be liquidated" : "safe"}{o.agrees ? "" : " (differs from the clear answer)"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "6px 0 10px" }}>
        In stable units. The clear columns and the sum of the shares are a self-test, possible only because both parties are in this tab. In a deployment neither party would see them: {held} sees the masked margin, about 2^{r.maskBits - 1} in size at every slot, and the pool sees ciphertext; each receives the one bit.
      </p>
      <figure className="chart-figure">
        <figcaption id={captionId} className="small muted"><b>The margin from both shares against the margin in the clear,</b> in thousands of stable units, one point per check. A point on the diagonal is a check that came out the same; a cross is one the circuit answered &quot;can be liquidated&quot;.</figcaption>
        <ChartReadout hint={HINT_POINTER}>
          <MarksHidden>
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart role="img" title="Margin reconstructed from both shares against the margin in the clear, per check" aria-describedby={captionId} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
              {grid}
              <XAxis type="number" dataKey="clear" name="clear margin" domain={[lo - pad, hi + pad]} tick={tick} tickFormatter={(v: number) => v.toFixed(0)} label={axisLabel("margin in the clear")} />
              <YAxis type="number" dataKey="shares" name="from both shares" domain={[lo - pad, hi + pad]} tick={tick} tickFormatter={(v: number) => v.toFixed(0)} label={yLabel("margin from both shares")} />
              <Tooltip content={<ReadoutTooltip />} formatter={(v: number) => v.toFixed(5)} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine segment={[{ x: lo - pad, y: lo - pad }, { x: hi + pad, y: hi + pad }]} stroke="var(--muted)" strokeDasharray="4 4" />
              <Scatter name="safe (circle)" data={safe} fill="var(--accent-2)" shape="circle" isAnimationActive={false} />
              {flagged.length > 0 && <Scatter name="can be liquidated (cross)" data={flagged} fill="var(--danger)" shape="cross" isAnimationActive={false} />}
            </ScatterChart>
          </ResponsiveContainer>
          </MarksHidden>
        </ChartReadout>
      </figure>
      <table className="table small" style={{ marginTop: 10 }}>
        <tbody>{costs.map(([k, v]) => <tr key={k}><th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{k}</th><td>{v}</td></tr>)}</tbody>
      </table>
      <h4 style={{ margin: "12px 0 4px" }}>Who sent what</h4>
      <div style={{ overflowX: "auto" }}>
        <table className="table small">
          <thead><tr><th>From, to</th><th>What</th><th>When</th><th>Size</th></tr></thead>
          <tbody>{whoSentWhat(r).map(([route, what, when, size]) => <tr key={route + what}><td>{route}</td><td>{what}</td><td>{when}</td><td>{size}</td></tr>)}</tbody>
        </table>
      </div>
    </div>
  );
}

/** The run's own messages, grouped: every row is read from what the checks reported. */
function whoSentWhat(r: LendingPrivateReport): [string, string, string, string][] {
  const n = r.outcomes.length;
  const rows: [string, string, string, string][] = [];
  if (r.setup) rows.push(["Bob to the pool", "his ciphertext, signed and sealed", "once, at the start", mb(r.setup.submissionBytes)]);
  const first = r.outcomes[0];
  for (const m of first?.messages ?? []) {
    const total = r.outcomes.reduce((s, o) => s + (o.messages.find((x) => x.what === m.what)?.bytes ?? 0), 0);
    rows.push([`${m.from} to ${m.to}`, m.what, `each of the ${n} checks`, total / n >= 1e5 ? mb(total / n) : `${Math.round(total / n).toLocaleString("en-US")} B`]);
  }
  rows.push([r.custody === "borrower" ? "the pool and Bob" : "the pool and Carol", "the garbled circuit's messages", `each of the ${n} checks`, mb(r.gcBytes / n)]);
  return rows;
}

function Sweep({ report }: { report: LendingSweepReport }) {
  const { rows, seeds } = report;
  const data = rows.map((r) => ({ ...r, label: `${r.multiplier}x`, band: [r.repaidPct - r.repaidP10, r.repaidP90 - r.repaidPct] }));
  return (
    <div className="card" id="lending-sweep" style={{ marginTop: 14 }}>
      <h3>Volatility against liquidations <StageBadge kind="simulated" /></h3>
      <figure className="chart-figure">
        <figcaption className="small muted">
          <b>{seeds} generated price paths at each multiple of the configured volatility,</b> with no scripted fall, so that only volatility moves. Line: the mean share of opening debt repaid by liquidation, which can pass 100% because borrowers borrow again as their collateral rises. Bars: its 10th and 90th percentiles over the paths. This model&apos;s own behaviour at illustrative parameters, and a statement about no live market.
        </figcaption>
        <ChartReadout hint={HINT_KEYBOARD}>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={data} accessibilityLayer title="Share of opening debt liquidated and share of paths with bad debt, against the volatility multiple" margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
              {grid}
              <XAxis dataKey="label" tick={tick} label={axisLabel("volatility, as a multiple of the configured one")} />
              <YAxis tick={tick} domain={[0, "auto"]} label={yLabel("per cent")} />
              <Tooltip content={<ReadoutTooltip labelName="volatility" />} formatter={(v: number) => (Array.isArray(v) ? "" : `${v.toFixed(1)}%`)} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="repaidPct" name="opening debt liquidated" stroke="var(--danger)" strokeWidth={2} isAnimationActive={false}>
                <ErrorBar dataKey="band" width={4} strokeWidth={1} stroke="var(--danger)" direction="y" />
              </Line>
              <Line type="monotone" dataKey="pathsWithBadDebtPct" name="paths ending with bad debt" stroke="var(--warn)" strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartReadout>
      </figure>
    </div>
  );
}

function WhereItRan({ market, sweep, check }: { market: LendingMarketReport; sweep?: LendingSweepReport; check?: LendingPrivateReport }) {
  const p = sweep?.pool, c = check?.pool;
  const worker = (name: string, q: NonNullable<typeof p>) => `${name} in ${q.ranIn}: ${q.speedUp.toFixed(1)} cores' worth of work at once, ${(100 * q.utilisation).toFixed(0)}% busy, of ${q.logicalCores} logical cores`;
  const rows: [string, string][] = [
    ["Web Workers", [c && worker("the private checks", c), p && worker("the sweep", p)].filter(Boolean).join("; ") + ". The pool takes at most half the logical cores"],
    ["WebAssembly, from C++", check ? "SEAL for CKKS (node-seal 7.0.0): one instance per worker. Its wasm build has one thread, so each check runs on one core and the speed-up comes from the pool" : "the private check did not run"],
    ["WebAssembly, from Rust", check ? "Polytune for the garbled circuit: one instance per worker, with both parties on that worker's thread" : "the private check did not run"],
    ["WebGPU", "not used. Neither SEAL nor Polytune has a GPU path, and the market is a few milliseconds of arithmetic"],
    ["The page's thread", `the market (${market.simulateMs.toFixed(1)} ms), the handshake, the signature on the record and the charts`],
  ];
  return (
    <div className="card" id="lending-resources" style={{ marginTop: 14 }}>
      <h3>Where this ran</h3>
      <p className="muted small" style={{ marginTop: 4 }}>All of it in this browser tab. Nothing on this page is sent to a server.</p>
      <table className="table small">
        <tbody>{rows.map(([k, v]) => <tr key={k}><th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{k}</th><td>{v}</td></tr>)}</tbody>
      </table>
    </div>
  );
}
