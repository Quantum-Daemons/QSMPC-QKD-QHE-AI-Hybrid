// A CKKS pass's level and scale ladder (Round 86, the owner's choice of 2026-11-22: FHE budget views), read from the
// ciphertexts the pass made in this tab (clientFHESquare.ts records one rung after each step). Nothing here needs the
// secret key: a ciphertext's place in the modulus chain and its scale travel with it.
import type { LadderRung } from "../lib/clientFHESquare";

export default function CkksLadder({ ladder, setName }: { ladder: LadderRung[]; setName: string }) {
  if (ladder.length === 0) return null;
  const top = ladder[0], last = ladder[ladder.length - 1];
  const rescales = top.primes - last.primes;
  return (
    <div style={{ marginTop: 12 }} id="ckks-ladder">
      <h4>The level and scale ladder of one image&apos;s pass, under {setName}</h4>
      <table className="table">
        <thead>
          <tr>
            <th scope="col" style={{ textAlign: "left" }}>After</th>
            <th scope="col">Primes left</th>
            <th scope="col">Modulus bits</th>
            <th scope="col">Scale, bits</th>
          </tr>
        </thead>
        <tbody>
          {ladder.map((r, i) => (
            <tr key={i}>
              <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{r.step}</th>
              <td>{r.primes}</td>
              <td>{r.modulusBits}</td>
              <td>{r.scaleBits.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>
        Read from the first image&apos;s ciphertexts, which this tab made. A fresh ciphertext holds every prime of the
        chain but the special one kept for keys: {top.primes} here, {top.modulusBits} bits. A product doubles the scale and
        a rescale divides it back by dropping one prime, so the pass used {rescales} of the {top.primes - 1} rescales the
        chain allows and ends on {last.primes} {last.primes === 1 ? "prime" : "primes"}: with one prime left, no further
        multiplication fits without bootstrapping, which this page does not run. The scale is the fixed-point precision:
        it stays near 2^{Math.round(top.scaleBits)} because every rescale undoes one product.
      </p>
    </div>
  );
}
