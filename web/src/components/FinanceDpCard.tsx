// The finance page's count of shared accounts, as it would be published (Round 86, the owner's choice of 2026-11-13).
// Every figure is this run's (clientDp.dpCount, applied in clientCompute.ts after the two banks' PSI).
import type { CountRelease } from "../lib/clientDp";

export interface FinanceDpArtifact extends CountRelease { exact: number; sizeA: number; sizeB: number }

export default function FinanceDpCard({ dp }: { dp: FinanceDpArtifact }) {
  return (
    <div className="card" style={{ marginTop: 14 }} id="finance-dp">
      <h3>The count of shared accounts, as it would be published</h3>
      <p style={{ marginTop: 4 }}>
        The two banks learn which accounts they share: that is what the PSI is for, so noise between them would protect
        nothing. What differential privacy can protect is the count once it leaves them, for instance in a
        supervisor&apos;s statistics, where it should not reveal whether any one account is among the flagged. One party
        adds exact discrete Gaussian noise to the count once; the unit is one account, which moves it by at most 1.
      </p>
      <table className="table">
        <tbody>
          <tr><th scope="row" style={{ textAlign: "left" }}>Published count</th><td><b>{dp.released}</b></td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>Epsilon, delta</th>
            <td>{dp.epsilonAchieved.toFixed(2)} (asked {dp.epsilon}), {dp.delta}</td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>Noise standard deviation</th><td>{dp.sd.toFixed(2)} accounts</td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>The exact count, which the banks hold and would not publish</th>
            <td>{dp.exact}, from lists of {dp.sizeA} and {dp.sizeB}</td></tr>
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>
        Delta is 1e-8, below 1/n^2 for the at most 4,600 accounts the two lists hold together (NIST SP 800-226). A noisy
        count below zero is published as zero, which is post-processing and costs no privacy. Each run draws fresh noise
        and is its own release. The exact count is shown only to measure the noise, and the account IDs are stand-ins
        drawn from the run&apos;s seed. The PSI among three to five banks, when ticked, reports its own intersection to
        those banks, without noise.
      </p>
    </div>
  );
}
