// The genomics page's pooled allele count, released with differential privacy (Round 86, the owner's choice of
// 2026-11-13). Every figure is this run's (clientDp.genomicsDp, through the secure sum in clientCompute.ts).
import type { GenomicsDp } from "../lib/clientDp";

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
            <td>{f2(dp.epsilonAchieved)} (asked {dp.epsilon}), {dp.delta}</td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>Noise standard deviation, all cohorts</th>
            <td>{f2(dp.sdTotal)} alleles</td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>Epsilon if one cohort adds no noise</th>
            <td>{f2(dp.epsilonOneDishonest)}</td></tr>
          <tr><th scope="row" style={{ textAlign: "left" }}>The exact count, which a deployment would not publish</th>
            <td>{dp.exact} of at most {2 * dp.individuals}</td></tr>
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>
        With {dp.individuals} people the noise a strong guarantee needs is large next to the count: DP is built for
        releases about many people, and a study NIST SP 800-226 cites calls epsilon below 1 reasonable. Delta is 1e-5, the
        largest value NIST does not call suspicious; 1/n^2 would be larger than that for so few people, and NIST says that rule does not
        suit small n. The exact count is shown only to measure the noise. The person&apos;s own score below is not noised:
        it goes back to that person, and DP protects people inside a released dataset. The noise is drawn from this
        tab&apos;s CSPRNG with an exact sampler (Canonne, Kamath and Steinke, 2020), and every 1000 Genomes individual
        here is public.
      </p>
    </div>
  );
}
