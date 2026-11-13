// The genomics page's pooled allele count, released with differential privacy (Round 86, the owner's choice of
// 2026-11-13). Every figure is this run's (clientDp.genomicsDp, through the secure sum in clientCompute.ts). On
// request, the same noisy contributions are summed once more under threshold BGV (clientFedAgg.sumUnderThresholdBgv,
// in a Web Worker), which must give the masked sum's integer exactly.
import { useState } from "react";
import { formatDelta, type GenomicsDp } from "../lib/clientDp";
import type { ThresholdSum } from "../lib/clientFedAgg";
import { sumUnderThresholdBgvInWorker } from "../lib/clientFedAggRun";
import { genomicsDpRows } from "../lib/cardRows";
import CardExport from "./CardExport";

export interface GenomicsDpArtifact extends GenomicsDp {
  released: number; exact: number; variant: string; individuals: number; cohorts: number;
}

const f2 = (x: number) => x.toFixed(2);

export default function GenomicsDpCard({ dp }: { dp: GenomicsDpArtifact }) {
  return (
    <div className="card" style={{ marginTop: 14 }} id="genomics-dp">
      <h3>The pooled count, released with differential privacy</h3>
      <p style={{ marginTop: 4 }}>
        Each of the {dp.cohorts} cohorts adds its share of exact discrete Gaussian noise to its count of the effect allele
        at {dp.variant} before the secure sum, so the sum the cohorts recover is already noisy (the distributed discrete
        Gaussian of Kairouz, Liu and Steinke, ICML 2021). The unit is one person, who moves the count by at most 2.
      </p>
      <table className="table">
        <tbody>
          <tr><th scope="row" style={{ textAlign: "left" }}>Released count</th><td><b>{f2(dp.released)}</b></td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>Epsilon (the selector above), delta</th>
            <td>{f2(dp.epsilonAchieved)} (asked {dp.epsilon}), {formatDelta(dp.delta)}</td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>Noise standard deviation, all cohorts</th>
            <td>{f2(dp.sdTotal)} alleles</td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>Epsilon if one cohort adds no noise</th>
            <td>{f2(dp.epsilonOneDishonest)}</td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>The exact count, which a deployment would not publish</th>
            <td>{dp.exact} of at most {2 * dp.individuals}</td></tr>
        </tbody>
      </table>
      <div style={{ marginTop: 6 }}>
        <CardExport name="genomics-differential-privacy" label="the released count" rows={genomicsDpRows(dp)} />
      </div>
      <p className="muted small" style={{ marginTop: 8 }}>
        With {dp.individuals} people the noise a strong guarantee needs is large next to the count: DP is built for
        releases about many people, and a study NIST SP 800-226 cites calls epsilon below 1 reasonable. Delta is 1e-5, the
        largest value NIST does not call suspicious; 1/n^2 would be larger than that for so few people, and NIST says that rule does not
        suit small n. The exact count is shown only to measure the noise. The person&apos;s own score below is not noised:
        it goes back to that person, and DP protects people inside a released dataset. The noise is drawn from this
        tab&apos;s CSPRNG with an exact sampler (Canonne, Kamath and Steinke, 2020), and every 1000 Genomes individual
        here is public. Each run draws fresh noise, so each run is its own release: k runs of the same count together
        are k times the rho of one.
      </p>
      <ThresholdBgv dp={dp} />
    </div>
  );
}

/** The same contributions under threshold BGV: no cohort holds the key, and all of their shares decrypt the sum once. */
function ThresholdBgv({ dp }: { dp: GenomicsDpArtifact }) {
  const [state, setState] = useState<{ status: "idle" | "running" | "done" | "error"; result?: ThresholdSum; error?: string }>(
    { status: "idle" });
  const run = () => {
    setState({ status: "running" });
    sumUnderThresholdBgvInWorker(dp.contributions, `qsh/fedthresh/crs/v1/genomics/${dp.variant}`)
      .then((result) => setState({ status: "done", result }))
      .catch((e) => setState({ status: "error", error: e instanceof Error ? e.message : String(e) }));
  };
  const r = state.result;
  const viaMasks = Math.round(dp.released * dp.scale);
  return (
    <div style={{ marginTop: 12 }} id="genomics-dp-bgv">
      <h4>The same noisy contributions, summed under threshold BGV</h4>
      <p className="small" style={{ marginTop: 4 }}>
        Each cohort encrypts its contribution - its count in the fixed point 1/16 plus its share of the noise - under the
        {" "}{dp.cohorts} cohorts&apos; collective key, the ciphertexts are added, and all {dp.cohorts} cohorts&apos;
        decryption shares open the sum once (Lattigo v6.2.0, threshold BGV, 128-bit smudging, in a Web Worker; the 12.6 MB
        module loads only when asked; <a href="/licenses/qsh-fedthresh.txt" target="_blank" rel="noopener noreferrer">its
        notices</a>). BGV is exact, so it must give the masked secure sum&apos;s integer.
      </p>
      <button className="btn" disabled={state.status === "running"} onClick={run}>
        {state.status === "running" ? "Running: loading the WebAssembly module and summing" : "Sum under threshold BGV"}
      </button>
      {state.status === "error" && <p className="small" role="alert">The threshold sum failed: {state.error}</p>}
      {r && (
        <table className="table" style={{ marginTop: 10 }}>
          <tbody>
            <tr><th scope="row" style={{ textAlign: "left" }}>Released count, decrypted from the BGV sum</th>
              <td><b>{(r.sum / dp.scale).toFixed(2)}</b></td></tr>
            <tr><th scope="row" style={{ textAlign: "left" }}>The masked secure sum&apos;s integer, from the same contributions</th>
              <td>{viaMasks} against {r.sum}: {r.sum === viaMasks ? "the same, to the bit" : "DIFFERENT"}</td></tr>
            <tr><th scope="row" style={{ textAlign: "left" }}>Statistical security of the decryption shares</th>
              <td>{r.statSecBits} bits over each whole share (one decryption under a fresh key); flooding uniform on
                [-(2^{r.floodBits} - 1), 2^{r.floodBits} - 1] for a worst-case noise bound of 2^{r.noiseBoundLog2.toFixed(2)}</td></tr>
            <tr><th scope="row" style={{ textAlign: "left" }}>Sent per cohort</th>
              <td>{(r.ciphertextBytes / 1000).toFixed(0)} kB ciphertext, {(r.decryptShareBytes / 1000).toFixed(0)} kB decryption share</td></tr>
            <tr><th scope="row" style={{ textAlign: "left" }}>Time</th>
              <td>key generation {r.keyGenMs.toFixed(0)} ms, the sum {r.roundMs.toFixed(0)} ms</td></tr>
          </tbody>
        </table>
      )}
      <p className="muted small" style={{ marginTop: 8 }}>
        What this changes is who could read one cohort&apos;s contribution: under the masks nobody can until all are
        added, and under BGV nobody can without every cohort&apos;s decryption share. It costs each cohort a ciphertext and
        a decryption share of hundreds of kilobytes (the table gives this run&apos;s), where its two additive shares are 8
        bytes. It is the same release, not a second one: the contributions and so their noise
        are the masked run&apos;s. A fresh key is made for each press, so no cohort makes a second decryption share for a
        ciphertext, which Lattigo&apos;s security notes say must be considered insecure. Every cohort runs in this tab.
      </p>
    </div>
  );
}
