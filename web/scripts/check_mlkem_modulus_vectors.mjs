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
// the key, tested individually.
//
// WHERE THE VECTORS COME FROM, and why that changed. They used to be fetched at run time so this
// always tested the CURRENT published set. That benefit is kept as `--live`, but the default is now
// the vendored snapshot in `vectors/`, because a check that needs the network cannot gate CI - CI is
// offline by design - and the only other way to gate it would be to skip on fetch failure. This
// script is one such skip away from being a conformance check that always passes, which is worse
// than not having one: it found @noble/post-quantum 0.2.1 accepting all 2,595 malformed
// encapsulation keys, and a version that skipped quietly would have called that build clean. So
// there is deliberately NO network fallback in either direction: `--live` fails if the network
// fails, and the default fails if a vector file is missing.
//
// NOTE ON SCOPE, so this is not over-read: it tests one normative input check in one library. It
// is not a proof of correctness, and noble is self-audited with no side-channel protection.
import { ml_kem512, ml_kem768, ml_kem1024 } from "@noble/post-quantum/ml-kem.js";
import { ml_dsa65, ml_dsa87 } from "@noble/post-quantum/ml-dsa.js";
import { gunzipSync } from "node:zlib";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SETS = [
  ["ML-KEM-512", ml_kem512],
  ["ML-KEM-768", ml_kem768],
  ["ML-KEM-1024", ml_kem1024],
];
const BASE = "https://raw.githubusercontent.com/C2SP/CCTV/main/ML-KEM/modulus";
const VECTORS = join(dirname(fileURLToPath(import.meta.url)), "vectors");
const LIVE = process.argv.includes("--live");

/** The gzipped vector blob for one parameter set, from disk or from upstream.
 *
 * Fails loudly on either path and never substitutes the other. An unreadable snapshot means the
 * check cannot run, which is a different thing from the check passing, and the message has to say
 * which - this is read from a CI log by someone who did not write it.
 */
async function loadVectors(name) {
  if (!LIVE) {
    const path = join(VECTORS, `${name}.txt.gz`);
    try {
      return readFileSync(path);
    } catch (err) {
      console.error(`\n  ${name}: cannot read the vendored vectors at ${path} (${err.code ?? err.message}).`);
      console.error("  This is NOT a pass. The conformance check did not run. Either restore the");
      console.error("  snapshot (see scripts/vectors/PROVENANCE.txt for the pinned upstream commit)");
      console.error("  or re-fetch the current published set with --live.");
      process.exit(1);
    }
  }
  const res = await fetch(`${BASE}/${name}.txt.gz`);
  if (!res.ok) {
    console.error(`\n  ${name}: HTTP ${res.status} from ${BASE}. --live requires the network;`);
    console.error("  drop --live to check against the vendored snapshot instead.");
    process.exit(1);
  }
  return Buffer.from(await res.arrayBuffer());
}

console.log(LIVE
  ? `  vectors: LIVE from ${BASE}`
  : `  vectors: vendored snapshot in scripts/vectors (see PROVENANCE.txt; --live to re-fetch)`);

let failures = 0;
for (const [name, kem] of SETS) {
  const text = gunzipSync(await loadVectors(name)).toString("utf8");
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

// ---------------------------------------------------------------------------
// ML-DSA round trip, because a library upgrade silently broke signing once.
//
// noble 0.6.x takes sign(message, secretKey) and verify(signature, message, publicKey); 0.2.x took
// (secretKey, message) and (publicKey, message, signature). BOTH arguments are Uint8Array, so
// TypeScript and the production build stayed green while every signature threw at runtime. Only
// clicking the button in a browser surfaced it. This asserts the round trip AND that a tampered
// message is rejected - the second half matters, because a verify() that always returns true would
// pass the first.
for (const [name, dsa, skLen, pkLen, sigLen] of [
  ["ML-DSA-65", ml_dsa65, 4032, 1952, 3309],
  ["ML-DSA-87", ml_dsa87, 4896, 2592, 4627],
]) {
  const { secretKey, publicKey } = dsa.keygen(new Uint8Array(32).fill(7));
  const msg = new TextEncoder().encode("qsh conformance probe");
  const sig = dsa.sign(msg, secretKey);
  const good = dsa.verify(sig, msg, publicKey);
  const bad = dsa.verify(sig, new TextEncoder().encode("qsh conformance probe!"), publicKey);
  const sizesOk = secretKey.length === skLen && publicKey.length === pkLen && sig.length === sigLen;
  const ok = good && !bad && sizesOk;
  if (!ok) failures++;
  console.log(`  ${name.padEnd(12)} sk ${secretKey.length} pk ${publicKey.length} sig ${sig.length}, ` +
    `verify ${good}, tampered-rejected ${!bad}  ${ok ? "OK" : "FAIL"}`);
}

if (failures > 0) {
  console.error(`\nPQC conformance: ${failures} check(s) FAILED.`);
  process.exit(1);
}
console.log("\nPQC conformance OK: FIPS 203 section 7.2 rejections and FIPS 204 sizes/round-trip.");
