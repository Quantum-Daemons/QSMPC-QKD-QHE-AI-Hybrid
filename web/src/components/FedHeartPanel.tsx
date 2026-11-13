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
import StageBadge from "./StageBadge";
import { fedHeartComparisons, type FedHeartResult } from "../lib/clientFedHeart";

export interface FedHeartSitesMeta {
  corpus: string; provenance: string; licence: string; licence_url: string; attribution: string; modifications: string;
  label: string; split: string; feature_names: string[];
}

const p = (x: number) => x.toFixed(3);

export default function FedHeartPanel({ result, sites }: { result: FedHeartResult; sites: FedHeartSitesMeta | null }) {
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
    </div>
  );
}
