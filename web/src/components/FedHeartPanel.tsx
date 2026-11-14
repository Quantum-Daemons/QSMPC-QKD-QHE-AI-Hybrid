// Four real hospitals, each alone, federated and pooled (Round 86, the owner's choice of 2026-11-13).
//
// Every number comes from this run (clientFedHeart.ts, twinned in Python by qsh.ai.fed_heart). The comparison is the one
// the published cases make - a hospital alone, the federation, and the pooled model no hospital could build without
// sharing its rows - and the caption says what the federation is not.
//
// THE TABLE IS A MATRIX, models by test sets. The first version put each hospital's model on its own test rows in an
// "Alone" column beside the federated model and averaged that column under "FLamby's metric". It is not FLamby's metric
// (which averages ONE model over the four hospitals' test rows), and on its own test rows a hospital's own model scores
// above the federated one at most hospitals here, so the table showed the federation losing in every row while the
// caption said it won. Both are true, of different comparisons; the matrix shows both, and the caption counts each from
// this run instead of asserting either.
import { useState } from "react";
import StageBadge from "./StageBadge";
import { type FedHeartDpResult, fedHeartComparisons, type FedHeartResult } from "../lib/clientFedHeart";
import { composedEpsilon, formatDelta } from "../lib/clientDp";
import { type AggregationComparison, DP_ROW_EPSILON } from "../lib/clientFedAgg";
import { compareAggregationsInWorker } from "../lib/clientFedAggRun";
import { aggregationRows, fedHeartDpRows, fedHeartMatrixRows } from "../lib/cardRows";
import CardExport from "./CardExport";

export interface FedHeartSitesMeta {
  corpus: string; provenance: string; licence: string; licence_url: string; attribution: string; modifications: string;
  label: string; split: string; feature_names: string[];
}

const p = (x: number) => x.toFixed(3);

export default function FedHeartPanel({ result, sites, dp }: { result: FedHeartResult; sites: FedHeartSitesMeta | null;
                                        dp?: FedHeartDpResult | null }) {
  const r = result;
  const K = r.centres.length;
  const c = fedHeartComparisons(r);
  const trainTotal = r.trainRows.reduce((a, b) => a + b, 0);
  const row = (label: string, accs: number[], mean: number, diagonal?: number) => (
    <tr key={label}>
      <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{label}</th>
      {accs.map((a, j) => (
        <td key={j} style={j === diagonal ? { fontWeight: 700, textDecoration: "underline" } : undefined}>{p(a)}</td>
      ))}
      <td><b>{p(mean)}</b></td>
    </tr>
  );
  return (
    <div className="card" style={{ marginTop: 14 }} id="fed-heart">
      <h3>Federated learning across four real hospitals <StageBadge kind="notional" /></h3>
      <p style={{ marginTop: 4 }}>
        FLamby&apos;s Fed-Heart-Disease benchmark on the UCI Heart Disease records of four hospitals: logistic regression on
        {" "}{sites?.feature_names.length ?? 13} standardised features, each hospital&apos;s rows staying with it. Each hospital
        alone and all four pooled train with Adam for 50 epochs; the federation runs {r.rounds} rounds of FedAvg, each hospital
        weighted by its training rows, through the page&apos;s pairwise masks. Each cell is one model&apos;s accuracy at 0.5
        on one hospital&apos;s test rows; the last column is FLamby&apos;s metric, that model&apos;s mean over the four.
      </p>
      <table className="table">
        <caption className="small" style={{ captionSide: "top", textAlign: "left" }}>
          Rows: the model and where it trained. Columns: whose test rows it is scored on. Underlined: a hospital&apos;s own
          model on its own test rows.
        </caption>
        <thead>
          <tr>
            <th scope="col" style={{ textAlign: "left" }}>Model (training rows)</th>
            {r.centres.map((name, j) => <th scope="col" key={name}>{name} test ({r.testRows[j]})</th>)}
            <th scope="col">Mean (FLamby&apos;s metric)</th>
          </tr>
        </thead>
        <tbody>
          {r.centres.map((name, k) => row(`${name} alone (${r.trainRows[k]})`, r.alone[k], r.mean.alone[k], k))}
          {row(`Federated, ${r.rounds} rounds of FedAvg (${trainTotal}, never pooled)`, r.federated, r.mean.federated)}
          {row(`Pooled in one place (${trainTotal})`, r.pooled, r.mean.pooled)}
        </tbody>
      </table>
      <div style={{ marginTop: 6 }}>
        <CardExport name="medical_fl-four-hospitals" label="the four-hospital matrix" rows={fedHeartMatrixRows(r)}
          json={r} />
      </div>
      <p className="small" style={{ marginTop: 8 }}>
        By FLamby&apos;s metric the federated model is above {c.aboveAlone} of the {K} hospitals&apos; models alone and
        {c.abovePooled ? " above" : " below"} the pooled one in this run. On its own test rows, though, a hospital&apos;s own
        model scores above the federated one at {c.ownBeatsFederated} of the {K} hospitals (the underlined cells): what
        the federation adds here is on the other hospitals&apos; patients, where a model trained at one hospital falls as low
        as {p(c.offDiagonalMin)}. Before training the shared initial model scores {p(r.mean.initial)}; the federated model
        after each round: {r.roundMean.map(p).join(", ")}.
      </p>
      <p className="muted small" style={{ marginTop: 8 }}>
        A review of 160 articles (Diniz et al., npj Digital Medicine, 2026) found decentralised learning, federated in 557
        of its 710 models, above local training in 83 percent of accuracy comparisons, and centralised training above
        decentralised in 78 percent of them. FLamby&apos;s published results on these four hospitals have the same order by
        its metric: federated above three of the four hospitals alone, and below pooled. All 30 of the Swiss
        hospital&apos;s training rows have disease and 15 of its 16 test rows do, so its own model scores 15 of 16 by
        calling every patient diseased, and its column moves the most from run to run. {r.aggregation}; all four hospitals
        run in this tab, which holds every private key, so the separation is notional: the masks hide each
        hospital&apos;s update from an aggregator that holds only the masked shares, and nothing bounds what the model
        itself reveals. A research benchmark, not a clinical model.
      </p>
      {sites && (
        <p className="muted small" style={{ marginTop: 6 }}>
          Data: {sites.attribution} Licence: <a href={sites.licence_url}>{sites.licence}</a>. Prepared as FLamby prepares
          it: {sites.modifications}
        </p>
      )}
      {dp && <DpTable dp={dp} />}
      <EncryptedAggregation seed={r.seed} dp={dp ?? null} />
    </div>
  );
}

const kb = (n: number) => (n < 1000 ? `${n} B` : `${(n / 1000).toFixed(0)} kB`);

/**
 * The same federation aggregated five ways, and the DP arm once more under threshold BGV (clientFedAgg.ts), on
 * request: the Lattigo module is about 3 MB compressed and the six runs take seconds, so it runs in a Web Worker when
 * the button is pressed, never on its own.
 */
function EncryptedAggregation({ seed, dp }: { seed: number; dp: FedHeartDpResult | null }) {
  const [state, setState] = useState<{ status: "idle" | "running" | "done" | "error"; progress?: string;
                                       result?: AggregationComparison; error?: string }>({ status: "idle" });
  const run = () => {
    setState({ status: "running", progress: "loading the WebAssembly module" });
    compareAggregationsInWorker(seed, (kind, done, of) =>
      setState((s) => ({ ...s, progress: `${done} of ${of} designs done; now ${kind}` })))
      .then((result) => setState({ status: "done", result }))
      .catch((e) => setState({ status: "error", error: e instanceof Error ? e.message : String(e) }));
  };
  const res = state.result;
  return (
    <div style={{ marginTop: 14 }} id="fed-heart-aggregation">
      <h4>The same federation, aggregated five ways, and once with differential privacy</h4>
      <p className="small" style={{ marginTop: 4 }}>
        Each design reruns this federation from the same seed and changes only how each round&apos;s FedAvg is
        aggregated: the pairwise masks above; CKKS with one key every hospital holds, as NVIDIA FLARE does it; and
        threshold CKKS and BGV on Lattigo, where no party holds the key and all four hospitals&apos; shares decrypt each
        round&apos;s sum once. The sixth row is the differential-privacy arm above at epsilon {DP_ROW_EPSILON}, run once
        more: each step&apos;s four noisy vectors are summed under threshold BGV, so the noise is inside the encrypted
        sum and the model trains on what the four shares decrypt, and under the masks beside it, to compare the two sums
        step by step. It runs in a Web Worker and loads a 12.6 MB WebAssembly module (Lattigo v6.2.0, Apache-2.0;
        {" "}<a href="/licenses/qsh-fedthresh.txt" target="_blank" rel="noopener noreferrer">its notices</a>).
      </p>
      <button className="btn" disabled={state.status === "running"} onClick={run}>
        {state.status === "running" ? `Running: ${state.progress}` : "Compare encrypted aggregation"}
      </button>
      {state.status === "error" && <p className="small" role="alert">The comparison failed: {state.error}</p>}
      {res && (
        <>
          <table className="table" style={{ marginTop: 10 }}>
            <caption className="small" style={{ captionSide: "top", textAlign: "left" }}>
              {res.rounds} rounds, {res.hospitals} hospitals, seed {res.seed}. Distance: the largest gap, over every round
              and coordinate, between a design&apos;s aggregate and the exact FedAvg of the same round&apos;s updates. The
              DP row trains differently (full-batch steps with noise) and is compared with the masks&apos; sums of its own
              noisy vectors instead.
            </caption>
            <thead>
              <tr>
                <th scope="col" style={{ textAlign: "left" }}>Design</th>
                <th scope="col" style={{ textAlign: "left" }}>Who can read one hospital&apos;s update</th>
                <th scope="col">Statistical security of the decryption</th>
                <th scope="col">Distance from exact FedAvg</th>
                <th scope="col">Accuracy (FLamby&apos;s metric)</th>
                <th scope="col">Same model as the masks</th>
                <th scope="col">Sent per hospital per round</th>
                <th scope="col">Time per round</th>
              </tr>
            </thead>
            <tbody>
              {res.designs.map((d) => (
                <tr key={d.kind}>
                  <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{d.label}<br />
                    <span className="muted small">{d.scheme}</span></th>
                  <td style={{ textAlign: "left" }}>{d.whoDecrypts}</td>
                  <td>{d.statSecBits == null ? "no decryption share"
                    : `${d.statSecBits} bits a decryption${d.statSecBitsPerKey != null && d.flooding
                      ? `; ${d.statSecBitsPerKey.toFixed(1)} over this run's ${d.flooding.decryptions} under one key` : ""}`}</td>
                  <td>{d.maxDiffFromExact == null ? "not FedAvg" : d.maxDiffFromExact.toExponential(1)}</td>
                  <td>{d.mean.toFixed(4)}</td>
                  <td>{d.dp ? `${d.dp.stepsEqual} of ${d.dp.steps} steps' sums equal the masks', to the bit`
                    : d.identicalToMasks == null ? "-" : d.identicalToMasks ? "yes, to the bit" : "no"}</td>
                  <td>{kb(d.bytesPerHospitalPerRound)}</td>
                  <td>{d.msPerRound < 10 ? d.msPerRound.toFixed(1) : d.msPerRound.toFixed(0)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 6 }}>
            <CardExport name="medical_fl-encrypted-aggregation" label="the encrypted-aggregation comparison"
              rows={aggregationRows(res)} json={res} />
          </div>
          <p className="muted small" style={{ marginTop: 8 }}>
            Each hospital floods its decryption share with noise drawn here, uniform on [-F, F] in every coefficient,
            the form the smudging lemma states (Asharov et al., ePrint 2011/613, Lemma 2.1), with F sized from the
            analytic worst case of the summed ciphertexts&apos; noise (Mouchet et al., ePrint 2020/304, eq. 8) rather
            than a measurement, as shared decryptions need (Guo et al., USENIX Security 2024). The bits are the
            statistical distance between a share and its simulation over all of its coefficients; shares under one key
            add up, so a run&apos;s rounds cost log2 of their number. The 16-bit row is a demonstration; a deployment
            chooses between the 40- and 128-bit rows. Each hospital makes one decryption share per sum and never a
            second for the same ciphertext, which <a href="https://github.com/tuneinsight/lattigo/blob/main/SECURITY.md">Lattigo&apos;s security notes</a> say must be considered insecure.
            Lattigo&apos;s own noise in a share stays at its fresh level, so <a href="https://github.com/tuneinsight/lattigo/issues/596">tuneinsight/lattigo#596</a> does not reach these settings. With one shared key, an aggregator that colludes with any one hospital
            reads every update. The masks and BGV differ from the exact FedAvg by their 2^20 fixed point, and BGV encrypts
            the masks&apos; own integers, so its sum is theirs; CKKS differs by its encoding and the smudging. All parties
            run in this tab, so every separation is notional, and no design here survives a dropout or authenticates a
            party.
          </p>
          {res.designs.filter((d) => d.dp).map((d) => (
            <p className="muted small" style={{ marginTop: 6 }} key="dp-row">
              The DP row draws fresh noise, so its accuracy differs from the epsilon {DP_ROW_EPSILON} row of the table
              above by the noise alone; its accountant gives epsilon {d.dp!.epsilonAchieved.toFixed(2)} at delta
              {" "}{formatDelta(d.dp!.delta)}. It is one more release from the same patients{dp
                ? `: with that table's ${dp.rows.filter((x) => Number.isFinite(x.epsilon)).length} noised rows, published together, epsilon ${
                  composedEpsilon([...dp.rows.filter((x) => Number.isFinite(x.epsilon)).map((x) => x.rho), d.dp!.rho], dp.delta).toFixed(2)}`
                : ""}. Because BGV is exact, the decrypted sum is the masks&apos; integer sum, so the encryption costs this
              row no accuracy; what it changes is who could read one hospital&apos;s noisy vector, and that
              hospital&apos;s share of the noise alone is too little to protect its patients.
            </p>
          ))}
        </>
      )}
    </div>
  );
}

/**
 * Record-level differential privacy beside the federation (fedHeartDp in clientFedHeart.ts): the epsilon, what it would
 * be with one hospital adding no noise, the noise, and the accuracy. The noise comes from the tab's CSPRNG, so a reload
 * shows other figures; the caption says so rather than presenting one draw as the result.
 */
function DpTable({ dp }: { dp: FedHeartDpResult }) {
  return (
    <div style={{ marginTop: 14 }} id="fed-heart-dp">
      <h4>With record-level differential privacy</h4>
      <p className="small" style={{ marginTop: 4 }}>
        The same four hospitals and initial model, trained by {dp.rounds} full-batch gradient steps instead of
        FedAvg&apos;s local steps, which have no ready per-patient bound: each hospital clips every patient&apos;s
        gradient to norm {dp.clip}, adds its share of exact discrete Gaussian noise, and the pairwise masks reveal only the
        noisy sum (the distributed discrete Gaussian of Kairouz, Liu and Steinke, ICML 2021). The unit is one patient,
        added or removed; delta is {formatDelta(dp.delta)}, below 1/n^2 for these 486 patients, as NIST SP 800-226 advises.
      </p>
      <table className="table">
        <caption className="small" style={{ captionSide: "top", textAlign: "left" }}>
          This run&apos;s figures: the noise is drawn from this tab&apos;s CSPRNG, so each run differs.
        </caption>
        <thead>
          <tr>
            <th scope="col" style={{ textAlign: "left" }}>Privacy</th>
            <th scope="col">Epsilon if one hospital adds no noise</th>
            <th scope="col">Noise standard deviation, all four</th>
            <th scope="col">Accuracy (FLamby&apos;s metric)</th>
          </tr>
        </thead>
        <tbody>
          {dp.rows.map((x) => (
            <tr key={String(x.epsilon)}>
              <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>
                {Number.isFinite(x.epsilon) ? `epsilon ${x.epsilon}` : "no noise (clipping only)"}</th>
              <td>{Number.isFinite(x.epsilonOneDishonest) ? x.epsilonOneDishonest.toFixed(2) : "-"}</td>
              <td>{x.sdTotal === 0 ? "0" : x.sdTotal.toFixed(1)}</td>
              <td>{x.mean.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 6 }}>
        <CardExport name="medical_fl-differential-privacy" label="the differential-privacy table" rows={fedHeartDpRows(dp)}
          json={dp} />
      </div>
      <p className="muted small" style={{ marginTop: 8 }}>
        The no-noise row is the comparison that isolates the noise: it shows what clipping and full-batch steps cost
        alone. The noise is in gradient units against a sum of up to 486 clipped gradients a step. Each epsilon counts the
        noise of the hospitals that add theirs; the second column is the guarantee if one of the four adds none. The
        sampler is exact (Canonne, Kamath and Steinke, 2020), because floating-point noise leaks through its low bits, and
        concentrated DP becomes epsilon by their Renyi conversion, searched over the order.
        Each noised row is its own release from the same patients, and concentrated DP adds across releases (Bun and
        Steinke, 2016): published together, the {dp.rows.filter((x) => Number.isFinite(x.epsilon)).length} rows are
        epsilon {dp.epsilonAllRows.toFixed(2)} at the same delta, not the largest of them.
      </p>
    </div>
  );
}
