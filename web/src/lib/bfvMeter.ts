/**
 * The BFV noise-budget meter's arithmetic (Round 86, the owner's choice of 2026-11-22: FHE budget views), apart from
 * its panel (BfvBudgetMeter.tsx) so that tests/test_r86_fhe_budget_views.py can run it on node-seal in node.
 *
 * One ciphertext, made here: `start` in slot 0 under N = 8192, a 202-bit coefficient modulus and a 20-bit batching
 * prime t. Each step squares it and relinearises. 202 bits is the HomomorphicEncryption.org standard's 128-bit
 * post-quantum bound for a ternary secret at N = 8192 (Table 2, BKZ.qsieve), the bound the CBOM holds every HE set to
 * (`tests/test_r86_cbom_he_parameters.py`). SEAL's default for this N, 218 bits, meets only the classical bound; the
 * meter used it until 2026-11-23. The budget is read with the secret key, so it is read
 * only from this ciphertext: computed for a ciphertext someone else chose, it would tell them about the key.
 */
export const BFV_DEMO = { N: 8192, plainBits: 20, start: 3n };

/** The coefficient modulus's prime sizes, 40 + 40 + 40 + 41 + 41 = 202 bits; the last is the special prime. */
export const BFV_DEMO_COEFF_BITS = [40, 40, 40, 41, 41] as const;

/** The value slot 0 should hold after `k` squarings of `start`, modulo t: start^(2^k) mod t. */
export function expectedAfter(start: bigint, k: number, t: bigint): bigint {
  let v = start % t;
  for (let i = 0; i < k; i++) v = (v * v) % t;
  return v;
}

export interface BudgetStep { squarings: number; budget: number; decrypted: bigint; expected: bigint }

export interface BfvMeter {
  /** Square the ciphertext once, relinearise, and read the new step. */
  square(): BudgetStep;
  /** The step as it stands. */
  read(): BudgetStep;
  /** Release every SEAL object this meter made. */
  dispose(): void;
  t: bigint;
}

export function makeBfvMeter(seal: any): BfvMeter {
  const owned: { delete(): void }[] = [];
  const keep = <T extends { delete(): void }>(o: T): T => { owned.push(o); return o; };
  const parms = keep(new seal.EncryptionParameters(seal.SchemeType.bfv));
  parms.setPolyModulusDegree(BFV_DEMO.N);
  parms.setCoeffModulus(keep(seal.CoeffModulus.Create(BFV_DEMO.N, Int32Array.from(BFV_DEMO_COEFF_BITS))));
  const plain = keep(seal.PlainModulus.Batching(BFV_DEMO.N, BFV_DEMO.plainBits));
  parms.setPlainModulus(plain);
  const context = keep(new seal.SEALContext(parms, true, seal.SecLevelType.tc128));
  if (!context.parametersSet()) throw new Error("SEAL rejected the demonstration's BFV parameters");
  const keys = keep(new seal.KeyGenerator(context));
  const relin = keep(keys.createRelinKeys());
  const encoder = keep(new seal.BatchEncoder(context));
  const encryptor = keep(new seal.Encryptor(context, keep(keys.createPublicKey())));
  const decryptor = keep(new seal.Decryptor(context, keep(keys.secretKey())));
  const evaluator = keep(new seal.Evaluator(context));
  const t = BigInt(plain.value());
  const values = new BigUint64Array(encoder.slotCount());
  values[0] = BFV_DEMO.start;
  const pt = keep(new seal.Plaintext());
  encoder.encode(values, pt);
  let ct = keep(new seal.Ciphertext());
  encryptor.encrypt(pt, ct);
  let squarings = 0;
  const read = (): BudgetStep => {
    const out = keep(new seal.Plaintext());
    decryptor.decrypt(ct, out);
    return { squarings, budget: decryptor.invariantNoiseBudget(ct), decrypted: BigInt(encoder.decodeBigUint64(out)[0]),
             expected: expectedAfter(BFV_DEMO.start, squarings, t) };
  };
  return {
    t,
    read,
    square() {
      const next = keep(new seal.Ciphertext());
      evaluator.square(ct, next);
      evaluator.relinearizeInplace(next, relin);
      ct = next;
      squarings += 1;
      return read();
    },
    dispose() {
      for (const o of owned.splice(0)) o.delete();
    },
  };
}
