// What NIST IR 8547 says of each cryptographic asset in the CBOM, on a stated date (Round 86).
//
// The owner's choice of 2026-11-23. The CBOM panel on the Validations page reads each asset of
// web/public/cbom/qsh.cdx.json against NIST IR 8547, "Transition to Post-Quantum Cryptography Standards"
// (final, October 2026, doi:10.6028/NIST.IR.8547), read on the final PDF on 2026-11-23: Tables 3 and 5
// (post-quantum signatures and key establishment), Table 4 (quantum-vulnerable key establishment, with its
// dates), Tables 6 and 8 (block ciphers; hash functions and XOFs), and the text of sections 3.2.1 (hybrid key
// establishment), 4.1.2 and 4.1.3. Each row names the table or section it rests on.
//
// It is a classification by IR 8547's tables, not a certification, and it says nothing about this code being
// correct or validated: nothing here is FIPS 140-3 validated. An asset IR 8547 does not name is "outside its
// tables", with what it is; the homomorphic-encryption and MPC schemes are all there.
//
// The hybrid is read by both of its parts, as the owner chose: ML-KEM (Table 5), X25519 (quantum-vulnerable,
// with the dates section 4.1.2 gives all quantum-vulnerable cryptography), and the combination, which section
// 3.2.1 allows even when one input is disallowed, which holds while either part holds and, against a quantum
// attacker, while ML-KEM does. It is not validated.
//
// The Python twin is src/qsh/pqc/ir8547.py; tests/test_r86_pqc_readiness.py holds the two to the same rows and
// the same standing on 2026-11-23, 2030-12-31, 2031-01-01, 2035-12-31 and 2036-01-01.

export type Ir8547Class = "post-quantum" | "quantum-vulnerable" | "symmetric" | "hybrid" | "outside";

export interface Ir8547Rule {
  /** The asset's bom-ref in the CBOM. */
  ref: string;
  cls: Ir8547Class;
  /** The table or section of IR 8547 the row rests on; "none" for an asset outside its tables. */
  basis: string;
  /** The security category the table gives; for a hash or XOF, its collision resistance's. Null where none is given. */
  category: number | null;
  /** For a hash or XOF, its preimage resistance's category (Table 8). */
  preimageCategory: number | null;
  /** Classical security strength in bits, as the table states it; null where it states none. */
  classicalBits: number | null;
  /** IR 8547 names the algorithm, or a family it belongs to. */
  named: boolean;
  /** "Deprecated after <year>" and "Disallowed after <year>", where IR 8547 gives them. */
  deprecatedAfter: number | null;
  disallowedAfter: number | null;
  /** A large quantum computer breaks it; stated for the rows outside IR 8547's tables too. */
  quantumVulnerable: boolean;
  /** What IR 8547 says of it, or, outside its tables, what it is. */
  reading: string;
}

export const IR8547 = {
  title: "NIST IR 8547, Transition to Post-Quantum Cryptography Standards (final, October 2026)",
  doi: "10.6028/NIST.IR.8547",
  url: "https://doi.org/10.6028/NIST.IR.8547",
  read: "2026-11-23",
} as const;

const HE = "each parameter set the CBOM names for it is inside the HE Standard's (2018) post-quantum limit for 128-bit security, which the build checks.";

function rule(ref: string, cls: Ir8547Class, basis: string, reading: string, o: Partial<Ir8547Rule> = {}): Ir8547Rule {
  return {
    ref: `crypto/${ref}`, cls, basis, category: null, preimageCategory: null, classicalBits: null, named: basis !== "none",
    deprecatedAfter: null, disallowedAfter: null, quantumVulnerable: cls === "quantum-vulnerable", reading, ...o,
  };
}

/** One row per asset, in the CBOM's order. */
export const RULES: readonly Ir8547Rule[] = [
  rule("ml-kem-768", "post-quantum", "Table 5",
    "ML-KEM-768 [FIPS 203]: 192 bits of classical security, security category 3. Section 4.1.2: NIST recommends it as the default parameter set.",
    { category: 3, classicalBits: 192 }),
  rule("ml-kem-1024", "post-quantum", "Table 5",
    "ML-KEM-1024 [FIPS 203]: 256 bits of classical security, security category 5.", { category: 5, classicalBits: 256 }),
  rule("x25519", "quantum-vulnerable", "Section 4.1.2",
    "IR 8547 does not name X25519. Its Table 4 dates elliptic-curve DH under SP 800-56A, and section 4.1.2 says NIST will deprecate all quantum-vulnerable cryptography after 2030 and disallow it after 2035.",
    { named: false, deprecatedAfter: 2030, disallowedAfter: 2035 }),
  rule("ecdh-p256", "quantum-vulnerable", "Table 4",
    "Elliptic-curve DH [SP 800-56A] at 112 bits of classical security or more, listed as currently approved: deprecated after 2030, disallowed after 2035.",
    { deprecatedAfter: 2030, disallowedAfter: 2035 }),
  rule("hkdf-sha256", "symmetric", "Section 4.1.3",
    "A key-derivation function. Section 4.1.3 counts KDFs [SP 800-56C, SP 800-108] among NIST's symmetric standards and believes every approved symmetric primitive of 128 bits of classical security or more meets security category 1 or above. No table rates HKDF."),
  rule("hybrid-combiner", "hybrid", "Sections 3.2.1 and 4.1.2",
    "Section 3.2.1 allows a composite of a shared secret from an approved scheme (here ML-KEM's) with one from another scheme (here X25519's), even if the other is disallowed; the desired property is that keys stay secure while either part is. Against a quantum attacker this one rests on ML-KEM alone. In the form of SP 800-227 section 4.6.2's two-step combiner, over the section 4.6.3 transcript; not validated."),
  rule("aes-256-gcm", "symmetric", "Table 6",
    "AES-256 [FIPS 197]: 256 bits of classical security, security category 5. GCM [SP 800-38D] is its mode, which no table rates.",
    { category: 5, classicalBits: 256 }),
  rule("chacha20-poly1305", "outside", "none",
    "IR 8547 does not name ChaCha20-Poly1305 (RFC 8439); its AEAD table lists only Ascon-AEAD128. A symmetric cipher with a 256-bit key."),
  rule("ml-dsa-65", "post-quantum", "Table 3",
    "ML-DSA-65 [FIPS 204]: 192 bits of classical security, security category 3.", { category: 3, classicalBits: 192 }),
  rule("ml-dsa-87", "post-quantum", "Table 3",
    "ML-DSA-87 [FIPS 204]: 256 bits of classical security, security category 5.", { category: 5, classicalBits: 256 }),
  rule("sha-256", "symmetric", "Table 8",
    "SHA-256 [FIPS 180]: collision resistance 128 bits, security category 2; preimage resistance 256 bits, category 5.",
    { category: 2, preimageCategory: 5, classicalBits: 128 }),
  rule("sha-384", "symmetric", "Table 8",
    "SHA-384 [FIPS 180]: collision resistance 192 bits, security category 4; preimage resistance 384 bits, category 5.",
    { category: 4, preimageCategory: 5, classicalBits: 192 }),
  rule("sha-512", "symmetric", "Table 8",
    "SHA-512 [FIPS 180]: collision resistance 256 bits, security category 5; preimage resistance 512 bits, category 5.",
    { category: 5, preimageCategory: 5, classicalBits: 256 }),
  rule("shake256", "symmetric", "Table 8",
    "SHAKE256 [FIPS 202], assuming a large enough output: collision and preimage resistance 256 bits, security category 5.",
    { category: 5, preimageCategory: 5, classicalBits: 256 }),
  rule("hmac-sha256", "symmetric", "Section 4.1.3",
    "IR 8547 does not name HMAC [FIPS 198-1]. Section 4.1.3 believes every NIST-approved symmetric primitive of 128 bits of classical security or more meets security category 1 or above; its hash, SHA-256, is in Table 8.",
    { named: false }),
  rule("blake2s-256", "outside", "none", "IR 8547 does not name BLAKE2s (RFC 7693), a hash function with a 256-bit output."),
  rule("oprf-ristretto255", "outside", "none",
    "IR 8547 does not classify oblivious PRFs. This one (RFC 9497) rests on discrete logarithms in ristretto255, which a large quantum computer breaks.",
    { quantumVulnerable: true }),
  rule("ckks", "outside", "none", `IR 8547 does not classify homomorphic encryption. CKKS rests on ring LWE; ${HE}`),
  rule("ckks-threshold", "outside", "none",
    `IR 8547 does not classify homomorphic encryption or threshold decryption. Threshold CKKS rests on ring LWE; ${HE}`),
  rule("bgv-threshold", "outside", "none",
    `IR 8547 does not classify homomorphic encryption or threshold decryption. Threshold BGV rests on ring LWE; ${HE}`),
  rule("bgv-threshold-committee", "outside", "none",
    `IR 8547 does not classify homomorphic encryption or threshold decryption. The committee's threshold BGV rests on ring LWE, with Shamir shares of its key; ${HE}`),
  rule("wrk17b", "outside", "none",
    "IR 8547 does not classify secure computation. WRK17b's oblivious transfers rest on a discrete-log assumption, which a large quantum computer breaks.",
    { quantumVulnerable: true }),
  rule("shamir", "outside", "none",
    "IR 8547 does not classify secret sharing. Shamir's sharing itself rests on no computational assumption, so a quantum computer does not change it; the channels between the parties are a separate matter."),
  rule("csprng", "outside", "none",
    "IR 8547 does not name a browser's generator. This is Web Crypto's getRandomValues, whose generator the browser chooses; section 4.1.3 counts DRBGs [SP 800-90A] among NIST's symmetric standards, but this page cannot tell which generator a browser uses."),
];

export type Standing =
  | "post-quantum" | "symmetric" | "approved-quantum-vulnerable" | "quantum-vulnerable" | "deprecated" | "disallowed"
  | "allowed-composite" | "not-classified";

export interface StandingOn {
  standing: Standing;
  label: string;
}

/** The calendar year of an ISO date, "YYYY-MM-DD". IR 8547's dates are "after <year>", so the year decides. */
export function yearOf(isoDate: string): number {
  const m = /^(\d{4})-\d{2}-\d{2}$/.exec(isoDate);
  if (!m) throw new Error(`not an ISO date: ${isoDate}`);
  return Number(m[1]);
}

/** What IR 8547 says of the asset on the date: "deprecated after 2030" starts on 2031-01-01. */
export function standingOn(r: Ir8547Rule, isoDate: string): StandingOn {
  const year = yearOf(isoDate);
  if (r.deprecatedAfter !== null) {
    const via = r.named ? "" : " (section 4.1.2)";
    if (r.disallowedAfter !== null && year > r.disallowedAfter) return { standing: "disallowed", label: `Disallowed${via}` };
    if (year > r.deprecatedAfter) return { standing: "deprecated", label: `Deprecated${via}` };
    return r.named
      ? { standing: "approved-quantum-vulnerable", label: `Listed as approved; deprecated after ${r.deprecatedAfter}` }
      : { standing: "quantum-vulnerable", label: `Not named; deprecated after ${r.deprecatedAfter} (section 4.1.2)` };
  }
  switch (r.cls) {
    case "post-quantum":
      return { standing: "post-quantum", label: "Post-quantum; no transition date" };
    case "symmetric": {
      const cat = r.category === null ? "Category 1 or above" : r.preimageCategory === null ? `Category ${r.category}`
        : `Category ${r.category} (collision), ${r.preimageCategory} (preimage)`;
      return { standing: "symmetric", label: `${cat}; no transition date` };
    }
    case "hybrid":
      return { standing: "allowed-composite", label: "Allowed as a composite (section 3.2.1); not validated" };
    default:
      return { standing: "not-classified", label: r.quantumVulnerable ? "Not classified; quantum-vulnerable" : "Not classified" };
  }
}

export function ruleFor(ref: string): Ir8547Rule | undefined {
  return RULES.find((r) => r.ref === ref);
}

/** How many assets have each standing on the date. */
export function standingCounts(isoDate: string, refs: readonly string[] = RULES.map((r) => r.ref)): Record<Standing, number> {
  const out = { "post-quantum": 0, symmetric: 0, "approved-quantum-vulnerable": 0, "quantum-vulnerable": 0, deprecated: 0, disallowed: 0,
    "allowed-composite": 0, "not-classified": 0 } as Record<Standing, number>;
  for (const ref of refs) {
    const r = ruleFor(ref);
    if (r) out[standingOn(r, isoDate).standing] += 1;
  }
  return out;
}

/** The viewer's calendar date, as the date input holds it. */
export function localIsoDate(d: Date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}
