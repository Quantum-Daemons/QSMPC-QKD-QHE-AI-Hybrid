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
import { hkdf } from "@noble/hashes/hkdf.js";
import { sha256 } from "@noble/hashes/sha2.js";

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
  // it. See src/qsh/pqc/hybrid_kem.py::encapsulate for the full note. v2 because the derived
  // key changes and deriving a different key under an unchanged label is what domain
  // separation exists to prevent.
  const INFO = new TextEncoder().encode("qsh/hybrid-kem/v2 ML-KEM-768+X25519|");
  const combine = (pq: Uint8Array, cl: Uint8Array) => {
    const transcript = new Uint8Array(
      cipherText.length + aliceXPub.length + bob.publicKey.length + bobXPub.length);
    transcript.set(cipherText, 0);                                            // c1, ML-KEM ct
    transcript.set(aliceXPub, cipherText.length);                             // c2, X25519 ephemeral
    transcript.set(bob.publicKey, cipherText.length + aliceXPub.length);      // ek1
    transcript.set(bobXPub, cipherText.length + aliceXPub.length + bob.publicKey.length);  // ek2
    const info = new Uint8Array(INFO.length + transcript.length);
    info.set(INFO, 0); info.set(transcript, INFO.length);
    const ikm = new Uint8Array(pq.length + cl.length);
    ikm.set(pq, 0); ikm.set(cl, pq.length);
    return hkdf(sha256, ikm, undefined, info, 32);
  };
  const sessionA = combine(pqSecret, xSecretA);
  const sessionB = combine(pqRecovered, xSecretB);

  // Agreement is now asserted over the COMBINED key, so it fails if either half disagrees.
  const agreement =
    sessionA.length === sessionB.length && sessionA.every((b: number, i: number) => b === sessionB[i]);

  return {
    algorithm: "ML-KEM-768 (FIPS 203) + X25519, HKDF-SHA256 combiner",
    classical: "X25519 (RFC 7748)",
    combiner: "HKDF-SHA256(pq_ss || x25519_ss), info=qsh/hybrid-kem/v2, transcript c1||c2||ek1||ek2 per SP 800-227 s4.6.3 — same as the backend",
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
