// A BFV noise-budget meter with a "one more multiply" button (Round 86, the owner's choice of 2026-11-22: FHE budget
// views). The page's encrypted paths are CKKS, whose limit is the level ladder; BFV's is a noise budget, which falls
// with each multiplication until decryption fails. The arithmetic is lib/bfvMeter.ts; this panel shows it.
import { useEffect, useRef, useState } from "react";
import { BFV_DEMO, BFV_DEMO_COEFF_BITS, type BfvMeter, type BudgetStep, makeBfvMeter } from "../lib/bfvMeter";
import { loadSeal } from "../lib/clientFHESquare";

export default function BfvBudgetMeter() {
  const [steps, setSteps] = useState<BudgetStep[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const meter = useRef<BfvMeter | null>(null);

  useEffect(() => () => { meter.current?.dispose(); }, []);

  const start = async () => {
    setBusy("Loading SEAL and making keys");
    setFailed(null);
    try {
      const seal = await loadSeal();
      meter.current?.dispose();
      meter.current = makeBfvMeter(seal);
      setSteps([meter.current.read()]);
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const multiply = () => {
    if (!meter.current) return;
    const next = meter.current.square();
    setSteps((s) => [...s, next]);
  };

  const last = steps[steps.length - 1];
  return (
    <div style={{ marginTop: 14 }} id="bfv-budget">
      <h4>A noise budget, spent one multiplication at a time (BFV, for comparison)</h4>
      <p className="small" style={{ marginTop: 4 }}>
        The encrypted paths on this page are CKKS, whose limit is the ladder above. BFV, an exact scheme, has a noise budget
        instead: each multiplication spends part of it, and once it is gone decryption returns the wrong number. This
        encrypts {String(BFV_DEMO.start)} in slot 0 under N = {BFV_DEMO.N}, a {BFV_DEMO_COEFF_BITS.reduce((a, b) => a + b, 0)}-bit
        coefficient modulus ({BFV_DEMO_COEFF_BITS.join(" + ")}, the HomomorphicEncryption.org standard&apos;s 128-bit
        post-quantum bound at this N) and a{" "}
        {BFV_DEMO.plainBits}-bit plaintext prime t, then squares it on each press. The budget is computed with the secret
        key, so it is read only from this ciphertext, which this tab made: computed for a ciphertext someone else chose, it
        would tell them about the key.
      </p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button className="btn" onClick={start} disabled={busy != null}>
          {busy ?? (steps.length ? "Start again" : "Encrypt and read the budget")}</button>
        <button className="btn" onClick={multiply} disabled={busy != null || !last || last.budget === 0}>One more multiply</button>
      </div>
      {failed ? <p className="small" style={{ color: "var(--danger)" }}>It did not run: {failed}</p> : null}
      {steps.length ? (
        <table className="table" style={{ marginTop: 8 }}>
          <thead>
            <tr>
              <th scope="col" style={{ textAlign: "left" }}>Squarings</th>
              <th scope="col">Noise budget, bits</th>
              <th scope="col">Decrypted</th>
              <th scope="col">Expected, mod t</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s) => (
              <tr key={s.squarings}>
                <th scope="row" style={{ textAlign: "left", fontWeight: 500 }}>{s.squarings}</th>
                <td>{s.budget}</td>
                <td>{String(s.decrypted)}</td>
                <td>{String(s.expected)}{s.decrypted === s.expected ? "" : " (wrong)"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {last && last.budget === 0 ? (
        <p className="small" style={{ marginTop: 6 }}>
          The budget is spent{last.decrypted === last.expected ? ", though this decryption still happened to come out right"
            : ", and the decryption is wrong"}: a deployment sizes its parameters so that its circuit never gets here, or
          refreshes the ciphertext.
        </p>
      ) : null}
    </div>
  );
}
