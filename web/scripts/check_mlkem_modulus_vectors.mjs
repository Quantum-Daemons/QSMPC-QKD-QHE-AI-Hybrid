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
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Expected vector count per set, and the SHA-256 of the vendored gzip. BOTH are asserted, and the
// reason is a hole this script shipped with for a round.
//
// The loop below asserts `accepted === 0` and, until Round 32, asserted nothing about how many keys it
// had tested. An empty or truncated gzip therefore gave `0 malformed keys, 0 rejected, 0 accepted OK`
// and **exit 0** - measured, not hypothetical. Vendoring the vectors is what introduced that: a fetch
// either returns the file or fails, but a file on disk can be present and empty.
//
// This is the project's own SP 800-90B lesson - a health test that reports PASS because it was handed
// too little data to fail - reappearing inside the check that had just been promoted to a CI gate. The
// three re-validations recorded when it was gated (clean, missing file, smuggled well-formed key) did
// not include it, because they were the failure modes of the OLD design.
//
// The counts are the same 775 / 780 / 1040 that every document quotes, and the hashes are the ones
// already written in vectors/PROVENANCE.txt - recorded there and, until now, never checked.
// The trailing number is the FIPS 203 Table 3 ciphertext length for that parameter set - 768, 1088,
// 1568 bytes. Asserted on the positive round trip below, which this file did not previously do.
const SETS = [
  ["ML-KEM-512", ml_kem512, 775, "4292314fd44bb211a27ef38273dcac32f602601c2f65b15a69773221227bf298", 768],
  ["ML-KEM-768", ml_kem768, 780, "22308df988866e2731c9b79bfa0ce754ea962a77b198436d56bde6481a526af7", 1088],
  ["ML-KEM-1024", ml_kem1024, 1040, "5792e788280ecdf5be7bfd2d8b66e18a0f9b9c80f4beb1d801fd27d20e882bf7", 1568],
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
async function loadVectors(name, expectedSha) {
  if (!LIVE) {
    const path = join(VECTORS, `${name}.txt.gz`);
    let buf;
    try {
      buf = readFileSync(path);
    } catch (err) {
      console.error(`\n  ${name}: cannot read the vendored vectors at ${path} (${err.code ?? err.message}).`);
      console.error("  This is NOT a pass. The conformance check did not run. Either restore the");
      console.error("  snapshot (see scripts/vectors/PROVENANCE.txt for the pinned upstream commit)");
      console.error("  or re-fetch the current published set with --live.");
      process.exit(1);
    }
    // The hashes were written into PROVENANCE.txt when the snapshot was vendored and never checked.
    // A recorded hash that nothing verifies documents an intention, not a property.
    const actual = createHash("sha256").update(buf).digest("hex");
    if (actual !== expectedSha) {
      console.error(`\n  ${name}: vendored snapshot does not match its recorded SHA-256.`);
      console.error(`    expected ${expectedSha}`);
      console.error(`    actual   ${actual}`);
      console.error("  This is NOT a pass. The file on disk is not the file that was vetted, so the");
      console.error("  result of testing against it means nothing. Restore it from the pinned upstream");
      console.error("  commit in scripts/vectors/PROVENANCE.txt, or re-vendor and update both.");
      process.exit(1);
    }
    return buf;
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
for (const [name, kem, expectedCount, expectedSha, expectedCtLen] of SETS) {
  const text = gunzipSync(await loadVectors(name, expectedSha)).toString("utf8");
  const lines = text.trim().split("\n").filter(Boolean);

  // ASSERT WHAT WAS TESTED, not only the outcome. Without this, zero vectors is indistinguishable
  // from zero acceptances, and the second is a pass while the first is a check that did not run.
  //
  // On the vendored path the SHA-256 above fires first and catches empty, truncated and tampered
  // files alike - measured, so this count check is the second line rather than the first. It still
  // earns its place in two cases the hash cannot cover: a re-vendor that updates the hash to match a
  // file that is genuinely short, and `--live`, where there is no hash to check at all.
  //
  // `--live` is exempt from failing because upstream is allowed to add vectors; a count CHANGE there
  // is information, and the message says so rather than turning a CI-green run red for it.
  if (lines.length !== expectedCount) {
    if (LIVE) {
      console.log(`  ${name.padEnd(12)} upstream now has ${lines.length} vectors, snapshot expects `
        + `${expectedCount} - upstream changed; re-vendor and update SETS + PROVENANCE.txt`);
    } else {
      console.error(`\n  ${name}: expected ${expectedCount} vectors, read ${lines.length}.`);
      console.error("  This is NOT a pass. A truncated or empty snapshot tests nothing, and testing");
      console.error("  nothing reports zero acceptances - which is what a clean run also reports.");
      failures++;
      continue;
    }
  }

  // A POSITIVE ROUND TRIP FIRST, because "everything was rejected" is what a WORKING library and a
  // TOTALLY BROKEN one both report.
  //
  // This half of the file read one outcome - did `encapsulate` throw - and never encapsulated to a
  // WELL-FORMED key. Seeded by making `encapsulate` throw unconditionally, it printed
  // "780 malformed keys, 780 rejected, 0 accepted  OK" and exited 0. A library in which ML-KEM does
  // not work at all passed the PQC conformance gate.
  //
  // The ML-DSA block below already had exactly this discipline, and its own comment says why: a noble
  // 0.2.x -> 0.6.x argument-order change broke every signature while TypeScript and the production
  // build stayed green. `web/package.json` pins `^0.6.1`, so the same class of break is in range for
  // ML-KEM. The lesson reached one of the two primitives in this file - the one-of-several-twins
  // shape this project has now recorded eleven times.
  //
  // Both halves seeded: `encapsulate` throwing gives "round-trip false ... FAIL" exit 1, and a
  // `decapsulate` returning the wrong 32 bytes gives "ct 1088 ss 32 round-trip false ... FAIL", so
  // the length checks alone cannot carry it.
  let roundTripOk = false;
  let ctLen = 0, ssLen = 0;
  try {
    const { publicKey, secretKey } = kem.keygen(new Uint8Array(64).fill(9));
    const { cipherText, sharedSecret } = kem.encapsulate(publicKey);
    const back = kem.decapsulate(cipherText, secretKey);
    ctLen = cipherText.length;
    ssLen = sharedSecret.length;
    roundTripOk = ctLen === expectedCtLen && ssLen === 32
      && Buffer.compare(Buffer.from(sharedSecret), Buffer.from(back)) === 0;
  } catch (e) {
    console.error(`  ${name}: encapsulate/decapsulate round trip threw - ${e.message}`);
  }
  if (!roundTripOk) failures++;

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
  // The round trip prints on the SAME LINE as the rejection count, so "N rejected, 0 accepted"
  // cannot appear without evidence that the library can encapsulate at all.
  console.log(
    `  ${name.padEnd(12)} ct ${ctLen} ss ${ssLen} round-trip ${roundTripOk}, ` +
    `${lines.length} malformed keys, ${lines.length - accepted} rejected, ${accepted} accepted  ` +
    `${ok && roundTripOk ? "OK" : "FAIL"}`);
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
