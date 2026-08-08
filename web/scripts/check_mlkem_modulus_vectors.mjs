// FIPS 203 section 7.2 modulus-check conformance for the BROWSER PQC library.
//
// Run:  node scripts/check_mlkem_modulus_vectors.mjs
//
// Why this exists. FIPS 203 section 7.2 makes the encapsulation-key modulus check normative -
// "ML-KEM.Encaps shall not be run with an encapsulation key that has not been checked as above" -
// and a library that skips it will happily encapsulate to a malformed key and return a shared
// secret. That is silent: nothing in a normal run reveals it.
//
// It was not hypothetical here. The version this project pinned, @noble/post-quantum 0.2.1,
// accepted ALL 780 malformed ML-KEM-768 keys in the C2SP/CCTV modulus set - zero rejected. Its
// polynomial coder used the compression-based path for d=12, where FIPS 203 specifies plain
// ByteEncode12/ByteDecode12, so the round-trip check that detects an out-of-range coefficient was
// a no-op. 0.6.1 rejects all 780.
//
// Vectors: C2SP/CCTV ML-KEM `modulus/`, CC0-1.0. Every value in [q, 2^12-1] at every position in
// the key, tested individually. They are fetched at run time rather than vendored so this checks
// the CURRENT published set; pin the URL to a tag if you need reproducibility across time.
//
// NOTE ON SCOPE, so this is not over-read: it tests one normative input check in one library. It
// is not a proof of correctness, and noble is self-audited with no side-channel protection.
import { ml_kem512, ml_kem768, ml_kem1024 } from "@noble/post-quantum/ml-kem.js";
import { gunzipSync } from "node:zlib";

const SETS = [
  ["ML-KEM-512", ml_kem512],
  ["ML-KEM-768", ml_kem768],
  ["ML-KEM-1024", ml_kem1024],
];
const BASE = "https://raw.githubusercontent.com/C2SP/CCTV/main/ML-KEM/modulus";

let failures = 0;
for (const [name, kem] of SETS) {
  const res = await fetch(`${BASE}/${name}.txt.gz`);
  if (!res.ok) {
    console.error(`  ${name}: could not fetch vectors (HTTP ${res.status})`);
    failures++;
    continue;
  }
  const text = gunzipSync(Buffer.from(await res.arrayBuffer())).toString("utf8");
  const lines = text.trim().split("\n").filter(Boolean);

  let accepted = 0;
  for (const line of lines) {
    const ek = Uint8Array.from(Buffer.from(line.trim(), "hex"));
    try {
      kem.encapsulate(ek);
      accepted++;              // a malformed key was accepted: section 7.2 violation
    } catch {
      /* correctly rejected */
    }
  }
  const ok = accepted === 0;
  if (!ok) failures++;
  console.log(
    `  ${name.padEnd(12)} ${lines.length} malformed keys, ` +
    `${lines.length - accepted} rejected, ${accepted} accepted  ${ok ? "OK" : "FAIL"}`);
}

if (failures > 0) {
  console.error(`\nFIPS 203 section 7.2: ${failures} parameter set(s) accepted malformed keys.`);
  process.exit(1);
}
console.log("\nFIPS 203 section 7.2: all malformed encapsulation keys rejected.");
