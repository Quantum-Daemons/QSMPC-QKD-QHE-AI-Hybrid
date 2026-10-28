// In-browser post-quantum crypto demo (pure JS, no server compute) using
// @noble/post-quantum: ML-KEM-768 key encapsulation + ML-DSA-65 signatures, and
// @noble/curves for the classical X25519 half of the hybrid.
//
// The X25519 half used to be a LABEL ONLY. Every page said "ML-KEM-768 + X25519" while this file
// ran ml_kem768 alone, and the reported 1088-byte ciphertext was ML-KEM-768's on its own — a real
// hybrid carries X25519's 32-byte ephemeral public key too. Hybrid PQC+classical is the project's
// headline positioning and the thing BSI TR-02102-1 actually recommends, so the fix is to perform
// it rather than to soften the wording. Unlike QKD or WASM-FHE, nothing about a browser prevented
// this; it was simply never wired up.
import { ml_kem768 } from "@noble/post-quantum/ml-kem.js";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";
import { x25519 } from "@noble/curves/ed25519.js";
import { hybridCombine, hybridCombinerInfo } from "./hybridCombiner";

const hex = (u: Uint8Array, n = 16) =>
  Array.from(u.slice(0, n)).map((b) => b.toString(16).padStart(2, "0")).join("");

export interface KemDemo {
  algorithm: string;
  publicKeyBytes: number;
  cipherTextBytes: number;
  sharedSecretBytes: number;
  agreement: boolean;
  sharedSecretPrefix: string;
  elapsedMs: number;
  /** Bytes contributed by each half, so the reported total is checkable rather than asserted. */
  pqcCipherTextBytes: number;
  classicalPublicKeyBytes: number;
  classical: string;
  combiner: string;
}

export function mlkemDemo(): KemDemo {
  const t0 = performance.now();

  // --- post-quantum half: ML-KEM-768 (FIPS 203) --------------------------------------------
  const bob = ml_kem768.keygen();
  const { cipherText, sharedSecret: pqSecret } = ml_kem768.encapsulate(bob.publicKey);
  const pqRecovered = ml_kem768.decapsulate(cipherText, bob.secretKey);

  // --- classical half: X25519 ephemeral-static ECDH ------------------------------------------
  const bobX = x25519.utils.randomSecretKey();
  const bobXPub = x25519.getPublicKey(bobX);
  const aliceX = x25519.utils.randomSecretKey();
  const aliceXPub = x25519.getPublicKey(aliceX);
  const xSecretA = x25519.getSharedSecret(aliceX, bobXPub);
  const xSecretB = x25519.getSharedSecret(bobX, aliceXPub);

  // --- combiner ------------------------------------------------------------------------------
  // HKDF-SHA256 over (pq_ss || x25519_ss) with the SAME domain-separation string and transcript
  // shape as the backend's src/qsh/pqc/hybrid_kem.py::_combine. The demo is supposed to mirror the
  // full deployment, so inventing a second, different KDF here would just be a new kind of drift.
  // Concatenate-then-KDF is what makes a hybrid sound: the result holds as long as EITHER half does.
  // Binding the transcript stops an attacker swapping one half for a value of their choosing.
  // SP 800-227 s4.6.3: transcript is c1 || c2 || ek1 || ek2. It was ek1 || ek2 || c2 until
  // Round 38 - the ML-KEM ciphertext was missing, which is the input that delivers the
  // IND-CCA-preserving property; the standard says the encapsulation keys are NOT needed for
  // ONE combiner, shared with `clientCompute`'s handshake and byte-compatible with
  // `src/qsh/pqc/hybrid_kem.py`. The separator, the transcript order and the output-length tag all
  // live in `hybridCombiner.ts`; this file no longer carries a copy of any of them.
  //
  // It used to. Three browser sites each built the info string themselves, and they diverged from
  // Python (the length tag was added there alone) AND from each other (this one hardcoded
  // ML-KEM-768 while the handshake derived its label from the run's selection). Editing three
  // literals in step is the method that failed twice in Round 38; `check:combiner` now asserts the
  // literal exists in exactly one module, so a site left behind fails a gate.
  //
  // This path is fixed at ML-KEM-768 - `ml_kem768.keygen()` above - so the name is passed rather
  // than derived, next to the keygen that justifies it.
  const combine = (pq: Uint8Array, cl: Uint8Array) =>
    hybridCombine(pq, cl,
      { c1: cipherText, c2: aliceXPub, ek1: bob.publicKey, ek2: bobXPub }, "ML-KEM-768", 32);

  const sessionA = combine(pqSecret, xSecretA);
  const sessionB = combine(pqRecovered, xSecretB);

  // Agreement is now asserted over the COMBINED key, so it fails if either half disagrees.
  const agreement =
    sessionA.length === sessionB.length && sessionA.every((b: number, i: number) => b === sessionB[i]);

  return {
    algorithm: "ML-KEM-768 (FIPS 203) + X25519, HKDF-SHA256 combiner",
    classical: "X25519 (RFC 7748)",
    // DERIVED from the combiner, not asserted about it. This read "same as the backend" while
    // the backend had a length tag this side did not, so the sentence was false on the very
    // page that displays it. Printing the actual bytes makes it an observation.
    combiner: `HKDF-SHA256(pq_ss || x25519_ss), info=${new TextDecoder().decode(hybridCombinerInfo("ML-KEM-768", 32))} then transcript c1||c2||ek1||ek2: an SP 800-227 s4.6.2 two-step combiner over the s4.6.3 transcript`,
    publicKeyBytes: bob.publicKey.length + bobXPub.length,
    pqcCipherTextBytes: cipherText.length,
    classicalPublicKeyBytes: aliceXPub.length,
    cipherTextBytes: cipherText.length + aliceXPub.length,
    sharedSecretBytes: sessionA.length,
    agreement,
    sharedSecretPrefix: hex(sessionA),
    elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
  };
}

export interface SigDemo {
  algorithm: string;
  publicKeyBytes: number;
  signatureBytes: number;
  validSignature: boolean;
  rejectsTampered: boolean;
  elapsedMs: number;
}

export function mldsaDemo(message = "QSH orchestration message"): SigDemo {
  const t0 = performance.now();
  const enc = new TextEncoder();
  const seed = crypto.getRandomValues(new Uint8Array(32));
  const { publicKey, secretKey } = ml_dsa65.keygen(seed);
  const msg = enc.encode(message);
  // @noble/post-quantum 0.6.x argument order: sign(message, secretKey) and
  // verify(signature, message, publicKey). 0.2.x took (secretKey, message) and
  // (publicKey, message, signature). Both arguments are Uint8Array, so tsc and the build stayed
  // green and the only symptom was a RangeError at runtime naming the expected key length.
  const sig = ml_dsa65.sign(msg, secretKey);
  const valid = ml_dsa65.verify(sig, msg, publicKey);
  const tampered = ml_dsa65.verify(sig, enc.encode(message + "!"), publicKey);
  return {
    algorithm: "ML-DSA-65 (FIPS 204)",
    publicKeyBytes: publicKey.length,
    signatureBytes: sig.length,
    validSignature: valid,
    rejectsTampered: !tampered,
    elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
  };
}
